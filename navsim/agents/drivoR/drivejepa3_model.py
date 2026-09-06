from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms

from .drivejepa2_bevformer.fpn_adapter import DriveJEPA2FPNAdapter
from .drivejepa2_bevformer.lss_bev import DriveJEPA2LSSProjector
from .drivejepa3_decoder import (
    DriveJEPA3Decoder,
    DriveJEPA3PlanningAwareDecoder,
)
from .layers.image_encoder.vjepa2_1_lora import ImgEncoderVJEPA21
from .layers.image_encoder.vjepa2_lora import ImgEncoderVJEPA2
from .layers.utils.mlp import MLP
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from navsim.agents.drivoR.utils import pylogger


log = pylogger.get_pylogger(__name__)


def _cfg_get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


class InteractionResidualScorer(nn.Module):
    """Bounded logit correction from the final planning-interaction query."""

    score_names = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "comfort",
    )

    def __init__(self, d_model: int, hidden_dim: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.score_names)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.net(query))


class DriveJEPA3Model(nn.Module):
    """DriveJEPA2 backbone with PAD/VeteranAD-style post-scorer decoder."""

    # Decoder version history:
    # V1 (legacy): waypoint-level GRU autoregression. Kept for reproduction.
    # decoder_cls = DriveJEPA3Decoder
    # V2 (active): complete-trajectory hierarchical Scene -> BEV -> Agent refinement.
    decoder_cls = DriveJEPA3PlanningAwareDecoder

    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = self._config.tf_d_model
        stage2_bridge_config = _cfg_get(config, "stage2_bridge", {})
        self.freeze_stage1 = bool(_cfg_get(stage2_bridge_config, "freeze_stage1", True))
        self.freeze_stage1_except_scorer = bool(
            _cfg_get(stage2_bridge_config, "freeze_stage1_except_scorer", False)
            and not self.freeze_stage1
        )
        self.freeze_proposal_generator = bool(
            _cfg_get(stage2_bridge_config, "freeze_proposal_generator", False)
            or self.freeze_stage1_except_scorer
            or self.freeze_stage1
        )
        self.detach_initial_query = bool(
            _cfg_get(stage2_bridge_config, "detach_initial_query", False)
            or self.freeze_stage1_except_scorer
            or self.freeze_stage1
        )
        self.freeze_image_backbone = bool(
            _cfg_get(stage2_bridge_config, "freeze_image_backbone", False)
            or _cfg_get(stage2_bridge_config, "freeze_backbone", False)
            or self.freeze_stage1_except_scorer
            or self.freeze_stage1
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
            self.post_scorer_decoder = self.decoder_cls(
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
        self.interaction_residual_scorer = InteractionResidualScorer(
            d_model=config.tf_d_model,
            hidden_dim=int(_cfg_get(config, "interaction_scorer_hidden_dim", config.tf_d_model)),
            scale=float(_cfg_get(config, "interaction_score_residual_scale", 0.0)),
        )
        self.b2d = config.b2d

        self._proposal_generator_modules = [
            self.hist_encoding,
            self.init_feature,
            self.trajectory_decoder,
            self.traj_head,
        ]
        self._scorer_modules = [
            self.scorer_attention,
            self.pos_embed,
            self.scorer,
        ]
        self._stage1_modules = [
            self.image_backbone,
            self.image_fpn,
            self.lss_bev_projector,
            self.hist_encoding,
            self.init_feature,
            self.trajectory_decoder,
            self.scorer_attention,
            self.pos_embed,
            self.traj_head,
            self.scorer,
            self.post_scorer_decoder.segmentation_head,
            self.post_scorer_decoder.bev_downscale,
            self.post_scorer_decoder.keyval_embedding,
            self.post_scorer_decoder.status_encoding,
            self.post_scorer_decoder.trajectory_query,
            self.post_scorer_decoder.query_embedding,
            self.post_scorer_decoder.tf_decoder,
            self.post_scorer_decoder.agent_head,
        ]
        if self.num_lidar > 0:
            self._stage1_modules.append(self.lidar_backbone)
        if self.freeze_stage1:
            for module in self._stage1_modules:
                module.requires_grad_(False)
            if self.scene_embeds is not None:
                self.scene_embeds.requires_grad_(False)
            if self.num_lidar > 0:
                self.lidar_scene_embeds.requires_grad_(False)
        elif self.freeze_stage1_except_scorer:
            for module in self._stage1_modules:
                module.requires_grad_(False)
            for module in self._scorer_modules:
                module.requires_grad_(True)
            if self.scene_embeds is not None:
                self.scene_embeds.requires_grad_(False)
            if self.num_lidar > 0:
                self.lidar_scene_embeds.requires_grad_(False)
        elif self.freeze_proposal_generator:
            for module in self._proposal_generator_modules:
                module.requires_grad_(False)

        self.scorer_trainable = any(
            parameter.requires_grad
            for module in self._scorer_modules
            for parameter in module.parameters()
        )

        log.info(
            "Stage2 bridge: freeze_stage1=%s, freeze_stage1_except_scorer=%s, freeze_image_backbone=%s, "
            "freeze_proposal_generator=%s, detach_initial_query=%s",
            self.freeze_stage1,
            self.freeze_stage1_except_scorer,
            self.freeze_image_backbone,
            self.freeze_proposal_generator,
            self.detach_initial_query,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_stage1 or self.freeze_stage1_except_scorer:
            for module in self._stage1_modules:
                module.eval()
            if self.freeze_stage1_except_scorer:
                for module in self._scorer_modules:
                    module.train(mode)
        elif self.freeze_proposal_generator:
            for module in self._proposal_generator_modules:
                module.eval()
        if self.freeze_image_backbone:
            self.image_backbone.eval()
        return self

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

    def _encode_trajectory_query(
        self,
        proposals: torch.Tensor,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
        detach_output: bool,
    ) -> torch.Tensor:
        """Encode trajectory coordinates without poisoning AMP's weight cache."""
        batch, num_proposals = proposals.shape[:2]
        embedded_traj = self.pos_embed(
            proposals.reshape(batch, num_proposals, -1).detach()
        )
        trajectory_query = self.scorer_attention(embedded_traj, scene_features)
        trajectory_query = trajectory_query + ego_token
        return trajectory_query.detach() if detach_output else trajectory_query

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._config.full_history_status:
            ego_status = features["ego_status"].flatten(-2)
            ego_status_current = features["ego_status"][:, -1]
        else:
            ego_status = features["ego_status"][:, -1]
            ego_status_current = ego_status

        if self.freeze_proposal_generator:
            with torch.no_grad():
                ego_token = self.hist_encoding(ego_status)[:, None]
                traj_tokens = ego_token + self.init_feature.weight[None]
        else:
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

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.freeze_proposal_generator
        ):
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
        # Keep this call grad-enabled and detach only its output. Under mixed
        # precision, invoking these shared modules under no_grad first causes
        # autocast to cache detached FP16 weights that are then reused by the
        # trainable final scorer call in the same forward pass.
        tr_out = self._encode_trajectory_query(
            proposals=raw_proposals,
            scene_features=scene_features,
            ego_token=ego_token,
            detach_output=self.detach_initial_query,
        )

        refiner_proposals = (
            raw_proposals.detach()
            if self.freeze_proposal_generator
            else raw_proposals
        )

        if bev_feature is None:
            refined_proposals = refiner_proposals
            refined_proposal_list = [refined_proposals]
            refiner_query_state = tr_out
            bev_semantic_map = None
            agent_states = None
            agent_labels = None
        else:
            decoder_output = self.post_scorer_decoder(
                bev_feature=bev_feature,
                status_feature=self._status_feature_from_ego(ego_status_current),
                proposals=refiner_proposals,
                traj_feature_bev=tr_out,
                scene_features=scene_features,
            )
            output["poses_reg_list"] = decoder_output["poses_reg_list"]
            refined_proposals = decoder_output["poses_reg_list"][-1]
            refined_proposal_list = list(decoder_output["poses_reg_list"])
            refiner_query_state = decoder_output["refiner_query_list"][-1]
            output["refiner_query_list"] = decoder_output["refiner_query_list"]
            bev_semantic_map = decoder_output["bev_semantic_map"]
            agent_states = decoder_output["agent_states"]
            agent_labels = decoder_output["agent_labels"]

        output["refined_proposals"] = refined_proposals
        output["proposals"] = refined_proposals
        output["proposal_list"] = refined_proposal_list

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.scorer_trainable):
            refined_tr_out = self._encode_trajectory_query(
                proposals=refined_proposals,
                scene_features=scene_features,
                ego_token=ego_token,
                detach_output=False,
            )

            (
                pred_logit,
                pred_logit2,
                pred_agents_states,
                pred_area_logit,
                scorer_bev_semantic_map,
                scorer_agent_states,
                scorer_agent_labels,
            ) = self.scorer(refined_proposals, refined_tr_out)

            # V8 scorer-alignment ablation (2026-08-02): match the legacy
            # coordinate scorer path exactly. Keep the module in the model so
            # V7 checkpoints remain load-compatible, but do not use its query
            # residual to alter any of the six PDM logits.
            # interaction_score_residual = self.interaction_residual_scorer(
            #     refiner_query_state.detach()
            # )
            # for score_idx, score_name in enumerate(
            #     self.interaction_residual_scorer.score_names
            # ):
            #     pred_logit[score_name] = (
            #         pred_logit[score_name]
            #         + interaction_score_residual[..., score_idx]
            #     )
            interaction_score_residual = None

        output["pred_logit"] = pred_logit
        output["pred_logit2"] = pred_logit2
        output["pred_agents_states"] = pred_agents_states
        output["pred_area_logit"] = pred_area_logit
        output["interaction_score_residual"] = interaction_score_residual
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
