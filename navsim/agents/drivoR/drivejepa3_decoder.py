from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.drivoR.drivejepa2_bevformer.lss_bev import DriveJEPA2BEVSemanticHead


def _cfg_get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


class DriveJEPA3AgentHead(nn.Module):
    def __init__(self, num_agents: int, d_model: int, d_ffn: int):
        super().__init__()
        self.num_agents = num_agents
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


def _nerf_positional_encoding(points: torch.Tensor, num_frequencies: int = 6) -> torch.Tensor:
    frequencies = 2.0 ** torch.arange(
        num_frequencies,
        device=points.device,
        dtype=points.dtype,
    )
    encoded = []
    for frequency in frequencies:
        encoded.extend((torch.sin(points * frequency), torch.cos(points * frequency)))
    return torch.cat(encoded, dim=-1)


class PositionConditionedNorm(nn.Module):
    def __init__(self, condition_dim: int, d_model: int):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, query: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = self.condition(condition)
        return self.gamma(condition) * self.norm(query) + self.beta(condition)


class PositionGuidedBEVAttention(nn.Module):
    """Sample BEV features at the current trajectory point."""

    def __init__(self, config, in_channels: int):
        super().__init__()
        self.d_model = int(_cfg_get(config, "tf_d_model", 256))
        x_bound = list(_cfg_get(config, "LIFT_X_BOUND", [0.0, 32.0, 0.25]))
        y_bound = list(_cfg_get(config, "LIFT_Y_BOUND", [-32.0, 32.0, 0.25]))
        self.x_min, self.x_step = float(x_bound[0]), float(x_bound[2])
        self.y_min, self.y_step = float(y_bound[0]), float(y_bound[2])
        self.value_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output_proj = nn.Linear(in_channels, self.d_model)
        self.dropout = nn.Dropout(float(_cfg_get(config, "decoder_dropout", 0.1)))
        self.norm = nn.LayerNorm(self.d_model)

    def _sampling_grid(self, points: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # LSS stores forward x on the BEV height axis and lateral y on the width axis.
        row = (points[..., 0] - self.x_min) / self.x_step
        col = (points[..., 1] - self.y_min) / self.y_step
        grid_x = 2.0 * (col + 0.5) / width - 1.0
        grid_y = 2.0 * (row + 0.5) / height - 1.0
        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        points: torch.Tensor,
        projected_bev: torch.Tensor,
    ) -> torch.Tensor:
        grid = self._sampling_grid(
            points,
            projected_bev.shape[-2],
            projected_bev.shape[-1],
        ).unsqueeze(2)
        sampled = F.grid_sample(
            projected_bev,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return self.norm(query + self.dropout(self.output_proj(sampled)))


class RelativeAgentAttention(nn.Module):
    """Fuse detected agents using trajectory-agent relative positions as edge features."""

    def __init__(self, config):
        super().__init__()
        d_model = int(_cfg_get(config, "tf_d_model", 256))
        self.query_memory = nn.Linear(d_model, d_model)
        self.agent_memory = nn.Linear(d_model, d_model)
        self.relative_encoding = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            int(_cfg_get(config, "tf_num_head", 8)),
            dropout=float(_cfg_get(config, "tf_dropout", 0.0)),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(_cfg_get(config, "tf_d_ffn", 1024))),
            nn.GELU(),
            nn.Linear(int(_cfg_get(config, "tf_d_ffn", 1024)), d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        points: torch.Tensor,
        agent_queries: torch.Tensor,
        agent_states: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_proposals, dim = query.shape
        num_agents = agent_queries.shape[1]
        relative = points.unsqueeze(2) - agent_states[:, None, :, :2]
        memory = self.query_memory(query).unsqueeze(2)
        memory = memory + self.agent_memory(agent_queries).unsqueeze(1)
        memory = memory + self.relative_encoding(relative)
        memory = F.gelu(self.memory_norm(memory))
        attended, _ = self.attention(
            query.reshape(batch * num_proposals, 1, dim),
            memory.reshape(batch * num_proposals, num_agents, dim),
            memory.reshape(batch * num_proposals, num_agents, dim),
            need_weights=False,
        )
        query = self.norm1(query + attended.reshape(batch, num_proposals, dim))
        return self.norm2(query + self.ffn(query))


class DriveJEPA3AutoregressiveLayer(nn.Module):
    def __init__(self, config, in_channels: int):
        super().__init__()
        self.num_poses = int(_cfg_get(config, "num_poses", 8))
        self.d_model = int(_cfg_get(config, "tf_d_model", 256))
        self.residual_scale = float(_cfg_get(config, "decoder_residual_scale", 1.0))
        self.step_embedding = nn.Embedding(self.num_poses, self.d_model)
        self.position_norm = PositionConditionedNorm(24, self.d_model)
        self.previous_pose_encoding = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
        )
        self.recurrent = nn.GRUCell(self.d_model, self.d_model)
        self.bev_attention = PositionGuidedBEVAttention(config, in_channels)
        self.agent_attention = RelativeAgentAttention(config)
        self.ego_attention = nn.MultiheadAttention(
            self.d_model,
            int(_cfg_get(config, "tf_num_head", 8)),
            dropout=float(_cfg_get(config, "tf_dropout", 0.0)),
            batch_first=True,
        )
        self.ego_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, int(_cfg_get(config, "tf_d_ffn", 1024))),
            nn.GELU(),
            nn.Linear(int(_cfg_get(config, "tf_d_ffn", 1024)), self.d_model),
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)
        self.delta_head = nn.Sequential(
            nn.Linear(self.d_model, int(_cfg_get(config, "tf_d_ffn", 1024))),
            nn.ReLU(inplace=True),
            nn.Linear(int(_cfg_get(config, "tf_d_ffn", 1024)), 3),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        anchors: torch.Tensor,
        bev_feature: torch.Tensor,
        agent_queries: torch.Tensor,
        agent_states: torch.Tensor,
        ego_query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, num_proposals, _ = query.shape
        hidden = query
        previous_pose = anchors.new_zeros(batch, num_proposals, 3)
        refined_poses = []
        projected_bev = self.bev_attention.value_proj(bev_feature)

        for step_idx in range(self.num_poses):
            anchor = anchors[:, :, step_idx]
            step_query = hidden + self.step_embedding.weight[step_idx].view(1, 1, -1)
            if step_idx > 0:
                step_query = step_query + self.previous_pose_encoding(previous_pose)
            step_query = self.position_norm(
                step_query,
                _nerf_positional_encoding(anchor[..., :2]),
            )
            step_query = self.bev_attention(step_query, anchor[..., :2], projected_bev)
            step_query = self.agent_attention(
                step_query,
                anchor[..., :2],
                agent_queries,
                agent_states,
            )
            ego_feature, _ = self.ego_attention(step_query, ego_query, ego_query, need_weights=False)
            step_query = self.ego_norm(step_query + ego_feature)
            step_query = self.ffn_norm(step_query + self.ffn(step_query))

            delta = self.delta_head(step_query)
            pose = anchor + self.residual_scale * delta
            refined_poses.append(pose)
            previous_pose = pose
            hidden = self.recurrent(
                self.previous_pose_encoding(pose).reshape(batch * num_proposals, self.d_model),
                step_query.reshape(batch * num_proposals, self.d_model),
            ).reshape(batch, num_proposals, self.d_model)

        return torch.stack(refined_poses, dim=2), hidden


class DriveJEPA3ProposalRefiner(nn.Module):
    def __init__(self, config, in_channels: int):
        super().__init__()
        self.num_poses = int(_cfg_get(config, "num_poses", 8))
        self.state_size = 3
        self.d_model = int(_cfg_get(config, "tf_d_model", 256))
        self.ref_num = int(_cfg_get(config, "decoder_ref_num", 2))
        self.detach_between_rounds = bool(
            _cfg_get(config, "decoder_detach_between_rounds", True)
        )

        self.proposal_pos_embed = nn.Sequential(
            nn.Linear(self.num_poses * self.state_size, int(_cfg_get(config, "tf_d_ffn", 1024))),
            nn.ReLU(inplace=True),
            nn.Linear(int(_cfg_get(config, "tf_d_ffn", 1024)), self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.refiners = nn.ModuleList(
            [DriveJEPA3AutoregressiveLayer(config, in_channels) for _ in range(self.ref_num)]
        )

    def _proposal_feature(
        self,
        proposals: torch.Tensor,
        traj_feature_bev: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_proposals, num_poses, _ = proposals.shape
        if traj_feature_bev.dim() != 3:
            raise ValueError(f"traj_feature_bev must be 3D, got {tuple(traj_feature_bev.shape)}")
        if traj_feature_bev.shape[1] == num_proposals:
            return traj_feature_bev
        if traj_feature_bev.shape[1] == num_proposals * num_poses:
            return traj_feature_bev.reshape(batch, num_proposals, num_poses, -1).amax(-2)
        raise ValueError(
            "traj_feature_bev must be [B, N, D] or [B, N*T, D], "
            f"got {tuple(traj_feature_bev.shape)} for proposals {tuple(proposals.shape)}"
        )

    def forward(
        self,
        proposals: torch.Tensor,
        traj_feature_bev: torch.Tensor,
        bev_feature: torch.Tensor,
        agent_queries: torch.Tensor,
        agent_states: torch.Tensor,
        ego_query: torch.Tensor,
    ) -> List[torch.Tensor]:
        current = proposals
        proposal_feature = self._proposal_feature(proposals, traj_feature_bev)

        poses_reg_list = []
        for round_idx, refiner in enumerate(self.refiners):
            anchors = current.detach() if round_idx > 0 and self.detach_between_rounds else current
            query = proposal_feature + self.proposal_pos_embed(anchors.flatten(-2)) + ego_query
            current, _ = refiner(
                query=query,
                anchors=anchors,
                bev_feature=bev_feature,
                agent_queries=agent_queries,
                agent_states=agent_states,
                ego_query=ego_query,
            )
            poses_reg_list.append(current)
        return poses_reg_list


class DriveJEPA3Decoder(nn.Module):
    """Legacy v1 waypoint-autoregressive decoder.

    Kept intact for checkpoint reproduction and ablation. The active decoder is
    selected explicitly in drivejepa3_model.py.
    """

    def __init__(self, config, in_channels: int = 64):
        super().__init__()
        self.config = config
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
        self.query_splits: Tuple[int, int] = (1, self.num_bounding_boxes)
        self.trajectory_query = nn.Embedding(1, self.d_model)
        self.query_embedding = nn.Embedding(self.num_bounding_boxes, self.d_model)
        self.tf_decoder = nn.TransformerDecoder(
            decoder_layer,
            int(_cfg_get(config, "tf_num_layers", 3)),
        )
        self.agent_head = DriveJEPA3AgentHead(
            num_agents=self.num_bounding_boxes,
            d_model=self.d_model,
            d_ffn=int(_cfg_get(config, "tf_d_ffn", 1024)),
        )
        self.proposal_refiner = DriveJEPA3ProposalRefiner(config, in_channels)

    def forward(
        self,
        bev_feature: torch.Tensor,
        status_feature: torch.Tensor,
        proposals: torch.Tensor,
        traj_feature_bev: torch.Tensor,
        scene_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if bev_feature.dim() == 5:
            batch, seq_len, channels, height, width = bev_feature.shape
            if seq_len != 1:
                raise ValueError(f"DriveJEPA3 decoder expects S=1, got {seq_len}")
            bev_feature = bev_feature[:, 0]
        elif bev_feature.dim() == 4:
            batch, channels, height, width = bev_feature.shape
        else:
            raise ValueError(f"bev_feature must be [B, C, H, W] or [B, 1, C, H, W], got {tuple(bev_feature.shape)}")

        bev_semantic_map = self.segmentation_head(bev_feature)
        bev_tokens = self.bev_downscale(bev_feature).flatten(2).transpose(1, 2)
        status_token = self.status_encoding(status_feature).unsqueeze(1)
        keyval = torch.cat([bev_tokens, status_token], dim=1)
        keyval = keyval + self.keyval_embedding.weight[None, : keyval.shape[1]].to(keyval.dtype)

        query = torch.cat(
            (self.trajectory_query.weight, self.query_embedding.weight),
            dim=0,
        )[None].expand(batch, -1, -1)
        query_out = self.tf_decoder(query, keyval)
        trajectory_query, agents_query = query_out.split(self.query_splits, dim=1)

        output = {"bev_semantic_map": bev_semantic_map}
        output.update(self.agent_head(agents_query))
        output["poses_reg_list"] = self.proposal_refiner(
            proposals=proposals,
            traj_feature_bev=traj_feature_bev,
            bev_feature=bev_feature,
            agent_queries=agents_query,
            agent_states=output["agent_states"],
            ego_query=trajectory_query,
        )
        return output


class PredictiveSceneInteraction(nn.Module):
    """Global proposal-to-scene-token interaction."""

    def __init__(self, config):
        super().__init__()
        d_model = int(_cfg_get(config, "tf_d_model", 256))
        attention_dropout = float(
            _cfg_get(config, "decoder_attention_dropout", 0.0)
        )
        residual_dropout = float(
            _cfg_get(config, "decoder_scene_residual_dropout", 0.0)
        )
        self.attention = nn.MultiheadAttention(
            d_model,
            int(_cfg_get(config, "tf_num_head", 8)),
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(residual_dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, scene_features: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            query,
            scene_features,
            scene_features,
            need_weights=False,
        )
        return self.norm(query + self.dropout(attended))


class WholeTrajectoryBEVInteraction(nn.Module):
    """Cross-attend to LSS features sampled along every proposal waypoint."""

    def __init__(self, config, in_channels: int):
        super().__init__()
        self.d_model = int(_cfg_get(config, "tf_d_model", 256))
        attention_dropout = float(
            _cfg_get(config, "decoder_attention_dropout", 0.0)
        )
        residual_dropout = float(
            _cfg_get(config, "decoder_bev_residual_dropout", 0.1)
        )
        x_bound = list(_cfg_get(config, "LIFT_X_BOUND", [0.0, 32.0, 0.25]))
        y_bound = list(_cfg_get(config, "LIFT_Y_BOUND", [-32.0, 32.0, 0.25]))
        self.x_min, self.x_step = float(x_bound[0]), float(x_bound[2])
        self.y_min, self.y_step = float(y_bound[0]), float(y_bound[2])
        self.value_proj = nn.Sequential(
            nn.Conv2d(in_channels, self.d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.MultiheadAttention(
            self.d_model,
            int(_cfg_get(config, "tf_num_head", 8)),
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(residual_dropout)
        self.norm = nn.LayerNorm(self.d_model)

    def _sampling_grid(self, points: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # LSS layout: forward x is the height axis; lateral y is the width axis.
        row = (points[..., 0] - self.x_min) / self.x_step
        col = (points[..., 1] - self.y_min) / self.y_step
        grid_x = 2.0 * (col + 0.5) / width - 1.0
        grid_y = 2.0 * (row + 0.5) / height - 1.0
        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        trajectory: torch.Tensor,
        bev_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_proposals, _ = query.shape
        projected_bev = self.value_proj(bev_feature)
        grid = self._sampling_grid(
            trajectory[..., :2],
            projected_bev.shape[-2],
            projected_bev.shape[-1],
        )
        sampled = F.grid_sample(
            projected_bev,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        waypoint_memory = sampled.permute(0, 2, 3, 1).reshape(
            batch * num_proposals,
            trajectory.shape[2],
            self.d_model,
        )
        flat_query = query.reshape(batch * num_proposals, 1, self.d_model)
        attended, _ = self.attention(
            flat_query,
            waypoint_memory,
            waypoint_memory,
            need_weights=False,
        )
        attended = attended.reshape(batch, num_proposals, self.d_model)
        return self.norm(query + self.dropout(attended))


class DistanceAwareAgentInteraction(nn.Module):
    """Attend to independent agent queries using nearest-waypoint geometry."""

    def __init__(self, config):
        super().__init__()
        self.d_model = int(_cfg_get(config, "tf_d_model", 256))
        attention_dropout = float(
            _cfg_get(config, "decoder_attention_dropout", 0.0)
        )
        residual_dropout = float(
            _cfg_get(config, "decoder_agent_residual_dropout", 0.0)
        )
        self.relative_encoding = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
        )
        self.attention = nn.MultiheadAttention(
            self.d_model,
            int(_cfg_get(config, "tf_num_head", 8)),
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(residual_dropout)
        self.norm = nn.LayerNorm(self.d_model)

    @staticmethod
    def _nearest_relative_geometry(
        trajectory: torch.Tensor,
        agent_states: torch.Tensor,
    ) -> torch.Tensor:
        trajectory_xy = trajectory[..., :2]
        agent_xy = agent_states[..., :2]
        relative_xy = agent_xy[:, None, None] - trajectory_xy[:, :, :, None]
        distance_sq = relative_xy.square().sum(dim=-1)
        nearest_step = distance_sq.argmin(dim=2, keepdim=True)
        gather_xy = nearest_step[..., None].expand(-1, -1, -1, -1, 2)
        nearest_xy = relative_xy.gather(2, gather_xy).squeeze(2)
        nearest_distance = distance_sq.gather(2, nearest_step).squeeze(2).clamp_min(0).sqrt()

        trajectory_heading = trajectory[..., 2]
        agent_heading = agent_states[..., 2]
        relative_heading = agent_heading[:, None, None] - trajectory_heading[:, :, :, None]
        relative_heading = torch.atan2(
            torch.sin(relative_heading),
            torch.cos(relative_heading),
        )
        nearest_heading = relative_heading.gather(2, nearest_step).squeeze(2)
        return torch.cat(
            (
                nearest_xy,
                nearest_distance.unsqueeze(-1),
                nearest_heading.unsqueeze(-1),
            ),
            dim=-1,
        )

    def forward(
        self,
        query: torch.Tensor,
        trajectory: torch.Tensor,
        agent_queries: torch.Tensor,
        agent_states: torch.Tensor,
        agent_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_proposals, _ = query.shape
        relative_geometry = self._nearest_relative_geometry(trajectory, agent_states)
        agent_memory = agent_queries[:, None] + self.relative_encoding(relative_geometry)
        confidence = agent_logits.sigmoid()[:, None, :, None]
        agent_memory = agent_memory * confidence

        flat_query = query.reshape(batch * num_proposals, 1, self.d_model)
        flat_memory = agent_memory.reshape(
            batch * num_proposals,
            agent_queries.shape[1],
            self.d_model,
        )
        attended, _ = self.attention(
            flat_query,
            flat_memory,
            flat_memory,
            need_weights=False,
        )
        attended = attended.reshape(batch, num_proposals, self.d_model)
        return self.norm(query + self.dropout(attended))


class DriveJEPA3PlanningAwareRefinementLayer(nn.Module):
    """V2: scene -> BEV -> agent interaction with whole-trajectory residuals."""

    def __init__(self, config, in_channels: int):
        super().__init__()
        self.num_poses = int(_cfg_get(config, "num_poses", 8))
        self.d_model = int(_cfg_get(config, "tf_d_model", 256))
        self.residual_scale = float(_cfg_get(config, "decoder_residual_scale", 1.0))
        self.scene_interaction = PredictiveSceneInteraction(config)
        self.bev_interaction = WholeTrajectoryBEVInteraction(config, in_channels)
        self.agent_interaction = DistanceAwareAgentInteraction(config)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, int(_cfg_get(config, "tf_d_ffn", 1024))),
            nn.GELU(),
            nn.Dropout(float(_cfg_get(config, "decoder_ffn_dropout", 0.0))),
            nn.Linear(int(_cfg_get(config, "tf_d_ffn", 1024)), self.d_model),
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)
        self.delta_head = nn.Sequential(
            nn.Linear(self.d_model, int(_cfg_get(config, "tf_d_ffn", 1024))),
            nn.ReLU(inplace=True),
            nn.Linear(int(_cfg_get(config, "tf_d_ffn", 1024)), self.num_poses * 3),
        )

        # Exact zero initialization preserves the Stage1 trajectory at startup.
        init_std = float(_cfg_get(config, "decoder_residual_init_std", 1e-3))
        if init_std == 0.0:
            nn.init.zeros_(self.delta_head[-1].weight)
        else:
            nn.init.normal_(self.delta_head[-1].weight, std=init_std)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        anchors: torch.Tensor,
        scene_features: torch.Tensor,
        bev_feature: torch.Tensor,
        agent_queries: torch.Tensor,
        agent_states: torch.Tensor,
        agent_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.scene_interaction(query, scene_features)
        query = self.bev_interaction(query, anchors, bev_feature)
        query = self.agent_interaction(
            query,
            anchors,
            agent_queries,
            agent_states,
            agent_logits,
        )
        query = self.ffn_norm(query + self.ffn(query))

        delta = self.delta_head(query).reshape(
            anchors.shape[0],
            anchors.shape[1],
            self.num_poses,
            3,
        )
        heading_delta = torch.tanh(delta[..., 2:3]) * np.pi
        delta = torch.cat((delta[..., :2], heading_delta), dim=-1)
        refined = anchors + self.residual_scale * delta
        return refined, query


class DriveJEPA3PlanningAwareProposalRefiner(nn.Module):
    """V2 progressive complete-trajectory refinement."""

    def __init__(self, config, in_channels: int):
        super().__init__()
        self.num_poses = int(_cfg_get(config, "num_poses", 8))
        self.d_model = int(_cfg_get(config, "tf_d_model", 256))
        self.ref_num = int(_cfg_get(config, "decoder_ref_num", 2))
        self.detach_between_rounds = bool(
            _cfg_get(config, "decoder_detach_between_rounds", True)
        )
        self.detach_query_between_rounds = bool(
            _cfg_get(
                config,
                "decoder_detach_query_between_rounds",
                self.detach_between_rounds,
            )
        )
        self.reset_query_each_round = bool(
            _cfg_get(config, "decoder_reset_query_each_round", False)
        )
        self.proposal_pos_embed = nn.Sequential(
            nn.Linear(self.num_poses * 3, int(_cfg_get(config, "tf_d_ffn", 1024))),
            nn.ReLU(inplace=True),
            nn.Linear(int(_cfg_get(config, "tf_d_ffn", 1024)), self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.refiners = nn.ModuleList(
            [
                DriveJEPA3PlanningAwareRefinementLayer(config, in_channels)
                for _ in range(self.ref_num)
            ]
        )

    @staticmethod
    def _proposal_feature(
        proposals: torch.Tensor,
        traj_feature_bev: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_proposals, num_poses, _ = proposals.shape
        if traj_feature_bev.dim() != 3:
            raise ValueError(f"traj_feature_bev must be 3D, got {tuple(traj_feature_bev.shape)}")
        if traj_feature_bev.shape[1] == num_proposals:
            return traj_feature_bev
        if traj_feature_bev.shape[1] == num_proposals * num_poses:
            return traj_feature_bev.reshape(
                batch,
                num_proposals,
                num_poses,
                -1,
            ).amax(dim=2)
        raise ValueError(
            "traj_feature_bev must be [B, N, D] or [B, N*T, D], "
            f"got {tuple(traj_feature_bev.shape)} for proposals {tuple(proposals.shape)}"
        )

    def forward(
        self,
        proposals: torch.Tensor,
        traj_feature_bev: torch.Tensor,
        scene_features: torch.Tensor,
        bev_feature: torch.Tensor,
        agent_queries: torch.Tensor,
        agent_states: torch.Tensor,
        agent_logits: torch.Tensor,
        ego_query: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        current = proposals
        initial_query_state = self._proposal_feature(proposals, traj_feature_bev)
        query_state = initial_query_state
        poses_reg_list = []
        query_state_list = []
        for round_idx, refiner in enumerate(self.refiners):
            anchors = current.detach() if round_idx > 0 and self.detach_between_rounds else current
            if round_idx > 0 and self.reset_query_each_round:
                # VeteranAD-style refinement carries only the updated geometry.
                round_query_state = initial_query_state
            elif round_idx > 0 and self.detach_query_between_rounds:
                round_query_state = query_state.detach()
            else:
                round_query_state = query_state
            query = round_query_state + self.proposal_pos_embed(anchors.flatten(-2)) + ego_query
            current, query_state = refiner(
                query=query,
                anchors=anchors,
                scene_features=scene_features,
                bev_feature=bev_feature,
                agent_queries=agent_queries,
                agent_states=agent_states,
                agent_logits=agent_logits,
            )
            poses_reg_list.append(current)
            query_state_list.append(query_state)
        return poses_reg_list, query_state_list


class DriveJEPA3PlanningAwareDecoder(DriveJEPA3Decoder):
    """V2 hierarchical planning-aware decoder.

    The legacy BEV/agent-query decoder is reused; only proposal refinement is
    replaced with complete-trajectory Scene/BEV/Agent interaction.
    """

    def __init__(self, config, in_channels: int = 64):
        super().__init__(config, in_channels)
        self.proposal_refiner = DriveJEPA3PlanningAwareProposalRefiner(
            config,
            in_channels,
        )

    def forward(
        self,
        bev_feature: torch.Tensor,
        status_feature: torch.Tensor,
        proposals: torch.Tensor,
        traj_feature_bev: torch.Tensor,
        scene_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if scene_features is None:
            raise ValueError("Planning-aware decoder requires predictive scene_features")
        if bev_feature.dim() == 5:
            batch, seq_len, _, _, _ = bev_feature.shape
            if seq_len != 1:
                raise ValueError(f"DriveJEPA3 decoder expects S=1, got {seq_len}")
            bev_feature = bev_feature[:, 0]
        elif bev_feature.dim() == 4:
            batch = bev_feature.shape[0]
        else:
            raise ValueError(
                "bev_feature must be [B, C, H, W] or [B, 1, C, H, W], "
                f"got {tuple(bev_feature.shape)}"
            )

        bev_semantic_map = self.segmentation_head(bev_feature)
        bev_tokens = self.bev_downscale(bev_feature).flatten(2).transpose(1, 2)
        status_token = self.status_encoding(status_feature).unsqueeze(1)
        keyval = torch.cat((bev_tokens, status_token), dim=1)
        keyval = keyval + self.keyval_embedding.weight[None, : keyval.shape[1]].to(
            keyval.dtype
        )

        query = torch.cat(
            (self.trajectory_query.weight, self.query_embedding.weight),
            dim=0,
        )[None].expand(batch, -1, -1)
        query_out = self.tf_decoder(query, keyval)
        trajectory_query, agents_query = query_out.split(self.query_splits, dim=1)

        output = {"bev_semantic_map": bev_semantic_map}
        output.update(self.agent_head(agents_query))
        poses_reg_list, refiner_query_list = self.proposal_refiner(
            proposals=proposals,
            traj_feature_bev=traj_feature_bev,
            scene_features=scene_features,
            bev_feature=bev_feature,
            agent_queries=agents_query,
            agent_states=output["agent_states"],
            agent_logits=output["agent_labels"],
            ego_query=trajectory_query,
        )
        output["poses_reg_list"] = poses_reg_list
        output["refiner_query_list"] = refiner_query_list
        return output
