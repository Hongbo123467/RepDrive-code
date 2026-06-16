from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.vjepa2_lora import ImgEncoderVJEPA2   # 使用 V-JEPA2 编码器
from .layers.image_encoder.vjepa2_1_lora import ImgEncoderVJEPA21
from .layers.utils.mlp import MLP
from .drivejepa2_bevformer.fpn_adapter import DriveJEPA2FPNAdapter
from .drivejepa2_bevformer.lss_bev import (
    DriveJEPA2BEVSemanticHead,
    DriveJEPA2LSSProjector,
)
from navsim.agents.drivoR.utils import pylogger
# from .mmdit.mmdit_cross_attn import MMDiT as DiT

log = pylogger.get_pylogger(__name__)


def _cfg_get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


class DriveJEPA2AgentHead(nn.Module):
    """VeteranAD-style BEV agent detection head."""

    def __init__(self, num_agents: int, d_model: int, d_ffn: int):
        super().__init__()
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, 5),
        )
        self.label_head = nn.Linear(d_model, 1)

    def forward(self, agent_queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        agent_states = self.state_head(agent_queries)
        agent_states[..., 0:2] = agent_states[..., 0:2].tanh() * 32.0
        agent_states[..., 2] = agent_states[..., 2].tanh() * np.pi
        agent_labels = self.label_head(agent_queries).squeeze(-1)
        return {"agent_states": agent_states, "agent_labels": agent_labels}


class DriveJEPA2BEVAgentDecoder(nn.Module):
    """Decode BEV feature into semantic map and agent box/class predictions."""

    def __init__(self, config, in_channels: int = 64):
        super().__init__()
        self.d_model = int(_cfg_get(config, "tf_d_model", 256))
        self.bev_downscale_size = int(_cfg_get(config, "bev_downscale_size", 8))
        self.num_bounding_boxes = int(_cfg_get(config, "num_bounding_boxes", 30))

        self.segmentation_head = DriveJEPA2BEVSemanticHead(
            in_channels=in_channels,
            num_classes=int(_cfg_get(config, "num_bev_classes", 7)),
        )
        self.bev_downscale = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.bev_downscale_size, self.bev_downscale_size)),
            nn.Conv2d(in_channels, self.d_model, kernel_size=1),
        )
        self.keyval_embedding = nn.Embedding(self.bev_downscale_size**2 + 1, self.d_model)
        self.status_encoding = nn.Linear(8, self.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(_cfg_get(config, "tf_num_head", 8)),
            dim_feedforward=int(_cfg_get(config, "tf_d_ffn", 1024)),
            dropout=float(_cfg_get(config, "tf_dropout", 0.0)),
            batch_first=True,
        )
        self.query_embedding = nn.Embedding(self.num_bounding_boxes, self.d_model)
        self.tf_decoder = nn.TransformerDecoder(
            decoder_layer,
            int(_cfg_get(config, "tf_num_layers", 3)),
        )
        self.agent_head = DriveJEPA2AgentHead(
            num_agents=self.num_bounding_boxes,
            d_model=self.d_model,
            d_ffn=int(_cfg_get(config, "tf_d_ffn", 1024)),
        )

    def forward(self, bev_feature: torch.Tensor, status_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        bev_semantic_map = self.segmentation_head(bev_feature)
        bev_tokens = self.bev_downscale(bev_feature).flatten(2).transpose(1, 2)
        status_token = self.status_encoding(status_feature).unsqueeze(1)
        keyval = torch.cat([bev_tokens, status_token], dim=1)
        keyval = keyval + self.keyval_embedding.weight[None, : keyval.shape[1]].to(keyval.dtype)

        query = self.query_embedding.weight[None].expand(bev_feature.shape[0], -1, -1)
        agent_queries = self.tf_decoder(query, keyval)
        output = {"bev_semantic_map": bev_semantic_map}
        output.update(self.agent_head(agent_queries))
        return output


class DriveJEPA2Model(nn.Module):
    """
    DriveJEPA2 模型：以 V-JEPA2 为图像骨干的 DrivoR 变体。
    除图像编码器替换为 ImgEncoderVJEPA2 外，其余结构（轨迹解码器、评分模块等）与 DrivoRModel 完全一致。
    """

    def __init__(self, config):
        """
        @param {OmegaConf|dict} config - 模型配置，字段与 drivoR.yaml 保持一致
        """
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = self._config.tf_d_model

        # --------------------------------------------------
        # 相机数量统计
        # --------------------------------------------------
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        # --------------------------------------------------
        # LiDAR 数量统计（本版本暂不启用 LiDAR，保留接口）
        # --------------------------------------------------
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # --------------------------------------------------
        # [核心改动] 图像骨干：使用 V-JEPA2 LoRA 编码器
        # 接收 DriveJEPAFeatureBuilder 输出的双帧前视图像
        # --------------------------------------------------
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            _vjepa_version = str(config_image_backbone.get("vjepa_version", "2")).lower()
            if _vjepa_version in ("2.1", "v2.1", "vjepa2.1", "vjepa2_1"):
                self.image_backbone = ImgEncoderVJEPA21(config_image_backbone)
            else:
                self.image_backbone = ImgEncoderVJEPA2(config_image_backbone)

            # scene_embeds 仅在 scene_lora 模式下创建：
            # 可学习的查询向量，供 V-JEPA encoder 注入到 attention 序列
            # frozen / lora 模式下不需要此参数，设为 None
            _backbone_mode = config_image_backbone.get("backbone_mode", "scene_lora")
            if _backbone_mode == "scene_lora":
                self.scene_embeds = nn.Parameter(
                    torch.randn(
                        1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features
                    ) * 1e-6,
                    requires_grad=True,
                )
            else:
                # frozen / lora 模式无可学习 scene_embeds
                self.scene_embeds = None

            self.image_fpn = DriveJEPA2FPNAdapter(
                in_chs=(config.tf_d_model, config.tf_d_model, config.tf_d_model, config.tf_d_model),
                mid_ch=config.tf_d_model,
                out_ch=config.tf_d_model,
            )
            self.lss_bev_projector = DriveJEPA2LSSProjector(
                config=config,
                in_channels=config.tf_d_model,
                out_channels=getattr(config, "MODEL_ENCODER_OUT_CHANNELS", 64),
            )
            self.bev_agent_decoder = DriveJEPA2BEVAgentDecoder(
                config=config,
                in_channels=getattr(config, "MODEL_ENCODER_OUT_CHANNELS", 64),
            )

            # ImageNet 归一化（与 Drive-JEPA 的 make_transform 完全对齐）
            # 放在 model 层而非 encoder 层，确保与原始实现一致
            self.transform = transforms.Compose([
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            ])

        # LiDAR 分支（如果启用，使用 DINOv2 LoRA 保持原有逻辑）
        if self.num_lidar > 0:
            from .layers.image_encoder.dinov2_lora import ImgEncoder
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(
                torch.randn(
                    1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features
                ) * 1e-6,
                requires_grad=True,
            )

        # --------------------------------------------------
        # Ego 状态编码
        # --------------------------------------------------
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11 * 4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # --------------------------------------------------
        # 轨迹 embedding 与解码
        # --------------------------------------------------
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num * self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.state_size

        self.trajectory_decoder = TransformerDecoder(
            proj_drop=0.1, drop_path=0.2, config=config
        )

        self.scorer_attention = TransformerDecoderScorer(
            num_layers=config.scorer_ref_num,
            d_model=config.tf_d_model,
            proj_drop=0.1,
            drop_path=0.2,
            config=config,
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(self.poses_num * 3, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        ref_num = config.ref_num
        self.traj_head = nn.ModuleList(
            [MLP(config.tf_d_model, config.tf_d_ffn, traj_head_output_size) for _ in range(ref_num + 1)]
        )

        # --------------------------------------------------
        # 评分模块
        # --------------------------------------------------
        self.scorer = Scorer(config)
        self.b2d = config.b2d

    @staticmethod
    def _status_feature_from_ego(ego_status: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                ego_status[:, 7:11],
                ego_status[:, 3:5],
                ego_status[:, 5:7],
            ],
            dim=-1,
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        模型前向传播。

        @param {Dict[str, Tensor]} features - 包含以下字段的特征字典：
            - "image" 或 "camera_feature"：形状 [B, N_cam, C, H, W]
            - "ego_status"：形状 [B, 11] 或 [B, 4, 11]
            - "lidar_feature"（可选）：形状 [B, N_lidar, C, H, W]
        @returns {Dict[str, Tensor]} - 输出字典，包含 trajectory、pdm_score 等
        """
        # --- Ego 状态 ---
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
            ego_status_current: torch.Tensor = features["ego_status"][:, -1]
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]
            ego_status_current = ego_status

        ego_token = self.hist_encoding(ego_status)[:, None]      # [B, 1, d_model]
        traj_tokens = ego_token + self.init_feature.weight[None]  # [B, proposal_num, d_model]

        batch_size = ego_status.shape[0]
        scene_features = []
        bev_decoder_output = None

        # --- 相机特征（V-JEPA2 双帧前视图像 + scene_tokens 注入）---
        if self.num_cams > 0:
            # DriveJEPA2FeatureBuilder 输出 camera_feature_1（当前帧）和 camera_feature_2（上一帧）
            if "camera_feature_1" not in features or "camera_feature_2" not in features:
                raise ValueError("features 中未找到 camera_feature_1 / camera_feature_2，"
                                 "请检查 DriveJEPAFeatureBuilder 是否正确配置")

            cam_f1 = features["camera_feature_1"].float()
            cam_f2 = features["camera_feature_2"].float()
            if cam_f1.dim() != 5 or cam_f2.dim() != 5:
                raise ValueError(
                    "DriveJEPA2 LSS path expects camera_feature_1/2 as [B, N_cam, C, H, W], "
                    f"got {tuple(cam_f1.shape)} and {tuple(cam_f2.shape)}"
                )
            batch_size, num_cam, channels, height, width = cam_f1.shape

            # ImageNet 归一化（与 Drive-JEPA model.forward 完全对齐，放在 model 层而非 encoder 层）
            cam_f1 = self.transform(cam_f1.reshape(batch_size * num_cam, channels, height, width))
            cam_f2 = self.transform(cam_f2.reshape(batch_size * num_cam, channels, height, width))

            # scene_embeds_input: scene_lora 模式下扩展并传入；
            # frozen / lora 模式下为 None，编码器内部不注入 scene_tokens
            if self.scene_embeds is not None:
                if num_cam > self.scene_embeds.shape[1]:
                    raise ValueError(
                        "DriveJEPA2 got more input cameras than initialized scene tokens: "
                        f"num_cam={num_cam}, scene_embeds={tuple(self.scene_embeds.shape)}"
                    )
                scene_embeds_input = self.scene_embeds[:, :num_cam].repeat(
                    batch_size, 1, 1, 1
                ).reshape(
                    batch_size * num_cam,
                    self._config.num_scene_tokens,
                    self.image_backbone.num_features,
                )
            else:
                scene_embeds_input = None

            if "intrinsics" not in features or "extrinsics" not in features:
                raise ValueError(
                    "features 中未找到 intrinsics / extrinsics，"
                    "请检查 DriveJEPA2FeatureBuilder 的 LSS metadata 输出"
                )
            if not hasattr(self.image_backbone, "forward_scene_and_multiscale"):
                raise ValueError("drivejepa2 BEV path requires V-JEPA2.1 forward_scene_and_multiscale")

            image_scene_tokens, multi_feats = self.image_backbone.forward_scene_and_multiscale(
                camera_feature_1=cam_f1,
                camera_feature_2=cam_f2,
                scene_tokens=scene_embeds_input,
            )   # [B*N_cam, num_scene_tokens, tf_d_model], list[[B*N_cam, 256, 16, 32]]
            image_bev_feat = self.image_fpn(multi_feats)
            _, feat_channels, feat_h, feat_w = image_bev_feat.shape
            image_bev_feat = image_bev_feat.view(batch_size, num_cam, feat_channels, feat_h, feat_w)

            intrinsics = features["intrinsics"].to(device=image_bev_feat.device, dtype=torch.float32)
            extrinsics = features["extrinsics"].to(device=image_bev_feat.device, dtype=torch.float32)
            future_egomotion = features.get("future_egomotion")
            if future_egomotion is None:
                future_egomotion = torch.zeros(batch_size, 1, 6, device=image_bev_feat.device)
            else:
                future_egomotion = future_egomotion.to(device=image_bev_feat.device, dtype=torch.float32)

            bev_feature = self.lss_bev_projector(
                image_bev_feat,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                future_egomotion=future_egomotion,
            )
            bev_decoder_output = self.bev_agent_decoder(
                bev_feature,
                status_feature=self._status_feature_from_ego(ego_status_current),
            )
            image_scene_tokens = image_scene_tokens.view(
                batch_size,
                num_cam * self._config.num_scene_tokens,
                self._config.tf_d_model,
            )

            log.debug(f"V-JEPA2 图像特征 - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # --- LiDAR 特征（可选）---
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"LiDAR 特征 - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"场景特征合并 - {scene_features.shape}")

        #前向得到action tokens,并且直接在MMDiT中进行交互


        # --- 轨迹初始化 ---
        proposals = self.traj_head[0](traj_tokens).reshape(
            traj_tokens.shape[0], -1, self.poses_num, self.state_size
        )
        proposal_list = [proposals]

        # --- 迭代优化 ---
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i + 1](tokens).reshape(
                tokens.shape[0], -1, self.poses_num, self.state_size
            )
            proposal_list.append(proposals)

        traj_tokens = token_list[-1]
        proposals = proposal_list[-1]

        output = {}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        # --- 评分 ---
        B, N, _, _ = proposals.shape
        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())
        tr_out = self.scorer_attention(embedded_traj, scene_features)
        tr_out = tr_out + ego_token

        (
            pred_logit, pred_logit2, pred_agents_states,
            pred_area_logit, bev_semantic_map, agent_states, agent_labels,
        ) = self.scorer(proposals, tr_out)
        if bev_decoder_output is not None:
            bev_semantic_map = bev_decoder_output["bev_semantic_map"]
            agent_states = bev_decoder_output["agent_states"]
            agent_labels = bev_decoder_output["agent_labels"]

        output["pred_logit"] = pred_logit
        output["pred_logit2"] = pred_logit2
        output["pred_agents_states"] = pred_agents_states
        output["pred_area_logit"] = pred_area_logit
        output["bev_semantic_map"] = bev_semantic_map
        output["agent_states"] = agent_states
        output["agent_labels"] = agent_labels

        # --- 最终轨迹选择 ---
        pdm_score = (
            self._config.noc * pred_logit["no_at_fault_collisions"].sigmoid().log()
            + self._config.dac * pred_logit["drivable_area_compliance"].sigmoid().log()
            + self._config.ddc * pred_logit["driving_direction_compliance"].sigmoid().log()
            + (
                self._config.ttc * pred_logit["time_to_collision_within_bound"].sigmoid()
                + self._config.ep * pred_logit["ego_progress"].sigmoid()
                + self._config.comfort * pred_logit["comfort"].sigmoid()
            ).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output
