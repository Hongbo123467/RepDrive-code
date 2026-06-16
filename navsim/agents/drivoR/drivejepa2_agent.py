from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from .drivejepa2_model import DriveJEPA2Model           # DriveJEPA2 版本模型
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from .drivor_features import DrivoRTargetBuilder       # 复用 DrivoR 的 target builder
from .drivejepa2_features import DriveJEPA2FeatureBuilder  # DriveJEPA2 版本数据处理
import sys
from omegaconf import OmegaConf
import math


class LitProgressBar(ProgressBar):
    """自定义进度条，每 100 个 batch 打印一次。"""

    def __init__(self):
        super().__init__()
        self.enable = True

    def disable(self):
        self.enable = False

    @staticmethod
    def _format_metric_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                return str(value)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return str(value)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx % 100 == 0:
            print(
                f"Epoch {trainer.current_epoch} - train {batch_idx} / "
                f"{self.total_train_batches} - {self.get_metrics(trainer, pl_module)}"
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx % 100 == 0:
            print(
                f"Epoch {trainer.current_epoch} - val {batch_idx} / "
                f"{self.total_train_batches} - {self.get_metrics(trainer, pl_module)}"
            )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics, val_metrics, other_metrics = {}, {}, {}
        for k, v in metrics.items():
            if "train/" in k:
                train_metrics[k] = v
            elif "val/" in k:
                val_metrics[k] = v
            else:
                other_metrics[k] = v
        print(f"\n###########  Epoch {trainer.current_epoch} ##########")
        for k, v in train_metrics.items():
            print(f"{k},{self._format_metric_value(v)}")
        for k, v in val_metrics.items():
            print(f"{k},{self._format_metric_value(v)}")
        for k, v in other_metrics.items():
            print(f"{k},{self._format_metric_value(v)}")
        print("###########\n")


class DriveJEPA2Agent(AbstractAgent):
    """
    DriveJEPA2 Agent：以 V-JEPA2 为图像骨干的 DrivoR Agent 变体。
    除图像特征构建（DriveJEPA2FeatureBuilder）和模型（DriveJEPA2Model）外，
    训练/评估逻辑与 DrivoRAgent 完全一致。
    """

    def __init__(
        self,
        config,
        lr_args: dict,
        checkpoint_path: str = None,
        loss: nn.Module = None,
        progress_bar: bool = True,
        scheduler_args: dict = None,
        batch_size: int = 64,
        num_gpus: int = 1,
    ):
        """
        @param {OmegaConf} config - 来自 drivoR.yaml 的配置
        @param {dict} lr_args - 优化器参数（name, base_lr, base_batch_size）
        @param {str} checkpoint_path - 预训练 checkpoint 路径，空字符串表示从头训练
        @param {nn.Module} loss - 损失函数模块
        @param {bool} progress_bar - 是否使用默认进度条
        @param {dict} scheduler_args - 学习率调度器参数
        @param {int} batch_size - 单 GPU 的 batch size
        @param {int} num_gpus - 参与训练的 GPU 数量
        """
        super().__init__()
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        cache_data = False

        if not cache_data:
            # [核心改动] 使用 DriveJEPA2Model（V-JEPA2 骨干）
            self._drivor_model = DriveJEPA2Model(config)

        if not cache_data and self._checkpoint_path == "":
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            self.ray = True
            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                self.worker = RayDistributedNoTorch(threads_per_node=8)
                self.worker_map = worker_map

            from .score_module.compute_navsim_score import get_scores

            metric_cache = MetricCacheLoader(
                Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache")
            )
            try:
                metric_cache_synthetic_0 = MetricCacheLoader(
                    Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-0")
                )
                metric_cache_synthetic_1 = MetricCacheLoader(
                    Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-1")
                )
                metric_cache_synthetic_2 = MetricCacheLoader(
                    Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-2")
                )
                metric_cache_synthetic_3 = MetricCacheLoader(
                    Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-3")
                )
                metric_cache_synthetic_4 = MetricCacheLoader(
                    Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-4")
                )

                self.train_metric_cache_paths_synthetic = metric_cache_synthetic_0.metric_cache_paths
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_1.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_2.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_3.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_4.metric_cache_paths)
                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
            except Exception:
                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic = None

            self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            self.test_metric_cache_paths = metric_cache.metric_cache_paths
            self.get_scores = get_scores
            self.loss = loss

    def name(self) -> str:
        """@returns {str} - Agent 名称"""
        return self.__class__.__name__

    def initialize(self) -> None:
        """从 checkpoint 加载权重（推理 / 继续训练时调用）"""
        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(
                    self._checkpoint_path, map_location=torch.device("cpu")
                )["state_dict"]
            # 兼容 drivor_model / drivejepa_model / drivejepa2_model checkpoint 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("agent._drivor_model", "_drivor_model")
                new_state_dict[k] = v
            self.load_state_dict(new_state_dict)

    def get_sensor_config(self) -> SensorConfig:
        """
        返回传感器配置，与 DrivoRAgent 完全一致。
        @returns {SensorConfig}
        """
        return SensorConfig(
            cam_f0=OmegaConf.to_object(self._config["cam_f0"]),
            cam_l0=OmegaConf.to_object(self._config["cam_l0"]),
            cam_l1=OmegaConf.to_object(self._config["cam_l1"]),
            cam_l2=OmegaConf.to_object(self._config["cam_l2"]),
            cam_r0=OmegaConf.to_object(self._config["cam_r0"]),
            cam_r1=OmegaConf.to_object(self._config["cam_r1"]),
            cam_r2=OmegaConf.to_object(self._config["cam_r2"]),
            cam_b0=OmegaConf.to_object(self._config["cam_b0"]),
            lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
        )

    def get_target_builders(self):
        """复用 DrivoR 的 target builder（轨迹 GT 不变）"""
        return [DrivoRTargetBuilder(config=self._config)]

    def get_feature_builders(self):
        """
        [核心改动] 使用 DriveJEPA2FeatureBuilder，适配 V-JEPA2 的图像分辨率预处理。
        """
        return [DriveJEPA2FeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """@returns {Dict} - 模型输出（trajectory, pdm_score 等）"""
        return self._drivor_model(features)

    def compute_score(self, targets, proposals, test=True):
        """计算 NAVSIM 评分，与 DrivoRAgent 逻辑完全一致。"""
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
            metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
        else:
            metric_cache_paths = self.test_metric_cache_paths
            metric_cache_paths_synthetic = self.test_metric_cache_paths_synthetic

        target_trajectory = targets["trajectory"]
        proposals = proposals.detach()

        data_points = [
            {
                "token": metric_cache_paths[token] if token in metric_cache_paths
                         else metric_cache_paths_synthetic[token],
                "poses": poses,
                "test": test,
            }
            for token, poses in zip(targets["token"], proposals.cpu().numpy())
        ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)
        final_scores = target_scores[:, :, -1]
        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]
            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)
            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)
            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)
            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
    ) -> Dict:
        """调用 loss 模块计算训练损失。"""
        return self.loss(targets, pred, self._config, self.compute_score)

    def get_optimizers(self):
        """配置优化器和学习率调度器，与 DrivoRAgent 完全一致。"""
        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] == "Adam":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.Adam(self._drivor_model.parameters(), lr=lr)
        elif self._lr_args["name"] == "AdamW":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.AdamW(self._drivor_model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"不支持的优化器: {self._lr_args['name']}")

        if self.scheduler_args is not None:
            T_max = int(
                math.ceil(self.scheduler_args.dataset_size / global_batchsize)
                * self.scheduler_args.num_epochs
            )
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, total_iters=T_max_ramp
            )
            T_max_cosine = T_max - T_max_ramp
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max_cosine, eta_min=0.0, last_epoch=-1
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_ramp, scheduler_cosine],
                milestones=[T_max_ramp],
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return [optimizer]

    def get_training_callbacks(self):
        """返回训练回调（checkpoint 保存、LR 监控）。"""
        checkpoint_cb_best = ModelCheckpoint(
            save_top_k=1,
            monitor="val/score_epoch",
            filename="best-{epoch}-{step}",
            mode="max",
            save_on_train_epoch_end=False,
        )
        checkpoint_cb = ModelCheckpoint(save_last=True)
        lr_monitor = LearningRateMonitor(
            logging_interval="step", log_momentum=False, log_weight_decay=False
        )
        if self.progress_bar:
            return [checkpoint_cb_best, checkpoint_cb, lr_monitor]
        else:
            progress_bar = LitProgressBar()
            return [checkpoint_cb_best, checkpoint_cb, progress_bar, lr_monitor]
