from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms

from .drivejepa2_bevformer.fpn_adapter import DriveJEPA2FPNAdapter
from .drivejepa2_bevformer.lss_bev import DriveJEPA2LSSProjector
from .drivejepa3_decoder import DriveJEPA3Decoder
from .layers.image_encoder.vjepa2_1_lora import ImgEncoderVJEPA21
from .layers.image_encoder.vjepa2_lora import ImgEncoderVJEPA2
from .layers.utils.mlp import MLP
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from navsim.agents.drivoR.utils import pylogger


log = pylogger.get_pylogger(__name__)


def _cfg_get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


class DriveJEPA3Model(nn.Module):
    """DriveJEPA2 backbone with PAD/VeteranAD-style post-scorer decoder."""

    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = self._config.tf_d_model
        stage2_bridge_config = _cfg_get(config, "stage2_bridge", {})
        self.freeze_image_backbone = bool(
            _cfg_get(stage2_bridge_config, "freeze_image_backbone", False)
            or _cfg_get(stage2_bridge_config, "freeze_backbone", False)
        )

        self.num_cams = 0
        for camera_name in ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"):
            if len(self._config[camera_name]) > 0:
                self.num_cams += 1

        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            vjepa_version = str(config_image_backbone.get("vjepa_version", "2")).lower()
            if vjepa_version in ("2.1", "v2.1", "vjepa2.1", "vjepa2_1"):
                self.image_backbone = ImgEncoderVJEPA21(config_image_backbone)
            else:
                self.image_backbone = ImgEncoderVJEPA2(config_image_backbone)

            backbone_mode = config_image_backbone.get("backbone_mode", "scene_lora")
            if backbone_mode == "scene_lora":
                self.scene_embeds = nn.Parameter(
                    torch.randn(
                        1,
                        self.num_cams,
                        self._config.num_scene_tokens,
                        self.image_backbone.num_features,
                    )
                    * 1e-6,
                    requires_grad=True,
                )
            else:
                self.scene_embeds = None

            if self.freeze_image_backbone:
                self.image_backbone.requires_grad_(False)
                if self.scene_embeds is not None:
                    self.scene_embeds.requires_grad_(False)

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
            self.post_scorer_decoder = DriveJEPA3Decoder(
                config=config,
                in_channels=getattr(config, "MODEL_ENCODER_OUT_CHANNELS", 64),
            )
            self.transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    )
                ]
            )

        if self.num_lidar > 0:
            from .layers.image_encoder.dinov2_lora import ImgEncoder

            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(
                torch.randn(
                    1,
                    self.num_lidar,
                    self._config.num_scene_tokens,
                    self.image_backbone.num_features,
                )
                * 1e-6,
                requires_grad=True,
            )

        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11 * 4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num * self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.state_size

        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)
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
        self.traj_head = nn.ModuleList(
            [MLP(config.tf_d_model, config.tf_d_ffn, traj_head_output_size) for _ in range(config.ref_num + 1)]
        )
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
        if self._config.full_history_status:
            ego_status = features["ego_status"].flatten(-2)
            ego_status_current = features["ego_status"][:, -1]
        else:
            ego_status = features["ego_status"][:, -1]
            ego_status_current = ego_status

        ego_token = self.hist_encoding(ego_status)[:, None]
        traj_tokens = ego_token + self.init_feature.weight[None]

        batch_size = ego_status.shape[0]
        scene_features = []
        bev_feature = None

        if self.num_cams > 0:
            if "camera_feature_1" not in features or "camera_feature_2" not in features:
                raise ValueError("features must contain camera_feature_1 and camera_feature_2")

            cam_f1 = features["camera_feature_1"].float()
            cam_f2 = features["camera_feature_2"].float()
            if cam_f1.dim() != 5 or cam_f2.dim() != 5:
                raise ValueError(
                    "DriveJEPA3 expects camera_feature_1/2 as [B, N_cam, C, H, W], "
                    f"got {tuple(cam_f1.shape)} and {tuple(cam_f2.shape)}"
                )
            batch_size, num_cam, channels, height, width = cam_f1.shape
            cam_f1 = self.transform(cam_f1.reshape(batch_size * num_cam, channels, height, width))
            cam_f2 = self.transform(cam_f2.reshape(batch_size * num_cam, channels, height, width))

            if self.scene_embeds is not None:
                if num_cam > self.scene_embeds.shape[1]:
                    raise ValueError(
                        "DriveJEPA3 got more input cameras than initialized scene tokens: "
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
                raise ValueError("features must contain intrinsics and extrinsics for DriveJEPA3 LSS")
            if not hasattr(self.image_backbone, "forward_scene_and_multiscale"):
                raise ValueError("DriveJEPA3 requires V-JEPA2.1 forward_scene_and_multiscale")

            if self.freeze_image_backbone:
                self.image_backbone.eval()
                with torch.no_grad():
                    image_scene_tokens, multi_feats = self.image_backbone.forward_scene_and_multiscale(
                        camera_feature_1=cam_f1,
                        camera_feature_2=cam_f2,
                        scene_tokens=scene_embeds_input,
                    )
            else:
                image_scene_tokens, multi_feats = self.image_backbone.forward_scene_and_multiscale(
                    camera_feature_1=cam_f1,
                    camera_feature_2=cam_f2,
                    scene_tokens=scene_embeds_input,
                )
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

            image_scene_tokens = image_scene_tokens.view(
                batch_size,
                num_cam * self._config.num_scene_tokens,
                self._config.tf_d_model,
            )
            scene_features.append(image_scene_tokens)
            log.debug(f"V-JEPA3 image scene tokens - {image_scene_tokens.shape}")

        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)

        proposals = self.traj_head[0](traj_tokens).reshape(
            traj_tokens.shape[0], -1, self.poses_num, self.state_size
        )
        proposal_list = [proposals]

        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i + 1](tokens).reshape(
                tokens.shape[0], -1, self.poses_num, self.state_size
            )
            proposal_list.append(proposals)

        raw_proposals = proposal_list[-1]

        output = {
            "raw_proposals": raw_proposals,
            "raw_proposal_list": proposal_list,
        }

        batch, num_proposals, _, _ = raw_proposals.shape
        embedded_traj = self.pos_embed(raw_proposals.reshape(batch, num_proposals, -1).detach())
        tr_out = self.scorer_attention(embedded_traj, scene_features)
        tr_out = tr_out + ego_token

        if bev_feature is None:
            refined_proposals = raw_proposals
            refined_proposal_list = [refined_proposals]
            bev_semantic_map = None
            agent_states = None
            agent_labels = None
        else:
            decoder_output = self.post_scorer_decoder(
                bev_feature=bev_feature,
                status_feature=self._status_feature_from_ego(ego_status_current),
                proposals=raw_proposals,
                traj_feature_bev=tr_out,
            )
            output["poses_reg_list"] = decoder_output["poses_reg_list"]
            refined_proposals = decoder_output["poses_reg_list"][-1]
            refined_proposal_list = list(decoder_output["poses_reg_list"])
            bev_semantic_map = decoder_output["bev_semantic_map"]
            agent_states = decoder_output["agent_states"]
            agent_labels = decoder_output["agent_labels"]

        output["refined_proposals"] = refined_proposals
        output["proposals"] = refined_proposals
        output["proposal_list"] = refined_proposal_list

        embedded_refined_traj = self.pos_embed(refined_proposals.reshape(batch, num_proposals, -1).detach())
        refined_tr_out = self.scorer_attention(embedded_refined_traj, scene_features)
        refined_tr_out = refined_tr_out + ego_token

        (
            pred_logit,
            pred_logit2,
            pred_agents_states,
            pred_area_logit,
            scorer_bev_semantic_map,
            scorer_agent_states,
            scorer_agent_labels,
        ) = self.scorer(refined_proposals, refined_tr_out)

        output["pred_logit"] = pred_logit
        output["pred_logit2"] = pred_logit2
        output["pred_agents_states"] = pred_agents_states
        output["pred_area_logit"] = pred_area_logit
        output["bev_semantic_map"] = bev_semantic_map if bev_feature is not None else scorer_bev_semantic_map
        output["agent_states"] = agent_states if bev_feature is not None else scorer_agent_states
        output["agent_labels"] = agent_labels if bev_feature is not None else scorer_agent_labels

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
        batch_index = torch.arange(batch_size, device=refined_proposals.device)
        trajectory = refined_proposals[batch_index, token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score
        return output
