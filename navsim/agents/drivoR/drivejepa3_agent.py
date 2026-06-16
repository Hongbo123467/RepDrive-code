from pathlib import Path
from typing import Any, Dict
import math
import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from .drivejepa2_agent import LitProgressBar
from .drivejepa2_features import DriveJEPA2FeatureBuilder
from .drivejepa3_model import DriveJEPA3Model
from .drivor_features import DrivoRTargetBuilder


class DriveJEPA3Agent(AbstractAgent):
    """DriveJEPA3 Agent with post-scorer PAD/VeteranAD-style decoder."""

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
        super().__init__()
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.loss = loss
        self.b2d = config.b2d
        self.ray = True
        self._score_resources_ready = False

        self._drivor_model = DriveJEPA3Model(config)

    def _ensure_score_resources(self):
        if self._score_resources_ready:
            return

        if self.ray:
            from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
            from nuplan.planning.utils.multithreading.worker_utils import worker_map

            self.worker = RayDistributedNoTorch(threads_per_node=8)
            self.worker_map = worker_map

        from .score_module.compute_navsim_score import get_scores

        metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
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
        self._score_resources_ready = True

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(
                    self._checkpoint_path,
                    map_location=torch.device("cpu"),
                )["state_dict"]
            new_state_dict = {}
            for key, value in state_dict.items():
                key = key.replace("agent._drivor_model", "_drivor_model")
                new_state_dict[key] = value
            self.load_state_dict(new_state_dict, strict=False)

    def get_sensor_config(self) -> SensorConfig:
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
        return [DrivoRTargetBuilder(config=self._config)]

    def get_feature_builders(self):
        return [DriveJEPA2FeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._drivor_model(features)

    def compute_score(self, targets, proposals, test=True):
        self._ensure_score_resources()

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
                "token": metric_cache_paths[token] if token in metric_cache_paths else metric_cache_paths_synthetic[token],
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
        return self.loss(targets, pred, self._config, self.compute_score)

    def get_optimizers(self):
        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] == "Adam":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.Adam(self._drivor_model.parameters(), lr=lr)
        elif self._lr_args["name"] == "AdamW":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.AdamW(self._drivor_model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self._lr_args['name']}")

        if self.scheduler_args is None:
            return [optimizer]

        t_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) * self.scheduler_args.num_epochs)
        t_max_ramp = int(t_max * 0.1)
        scheduler_ramp = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            total_iters=t_max_ramp,
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max - t_max_ramp,
            eta_min=0.0,
            last_epoch=-1,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_ramp, scheduler_cosine],
            milestones=[t_max_ramp],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def get_training_callbacks(self):
        checkpoint_cb_best = ModelCheckpoint(
            save_top_k=1,
            monitor="val/score_epoch",
            filename="best-{epoch}-{step}",
            mode="max",
            save_on_train_epoch_end=False,
        )
        checkpoint_cb = ModelCheckpoint(save_last=True)
        lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=False, log_weight_decay=False)
        if self.progress_bar:
            return [checkpoint_cb_best, checkpoint_cb, lr_monitor]
        progress_bar = LitProgressBar()
        return [checkpoint_cb_best, checkpoint_cb, progress_bar, lr_monitor]
