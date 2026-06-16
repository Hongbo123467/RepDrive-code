from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.drivoR.drivejepa2_bevformer.lss_bev import DriveJEPA2BEVSemanticHead


def _cfg_get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def _linear_relu_ln(embed_dims: int, in_loops: int, out_loops: int, input_dims: int = None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


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


class DriveJEPA3ProposalRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_poses = int(_cfg_get(config, "num_poses", 8))
        self.state_size = 3
        self.d_model = int(_cfg_get(config, "tf_d_model", 256))
        self.ref_num = int(_cfg_get(config, "decoder_ref_num", 2))
        self.residual_scale = float(_cfg_get(config, "decoder_residual_scale", 1.0))

        self.proposal_pos_embed = nn.Sequential(
            nn.Linear(self.num_poses * self.state_size, int(_cfg_get(config, "tf_d_ffn", 1024))),
            nn.ReLU(inplace=True),
            nn.Linear(int(_cfg_get(config, "tf_d_ffn", 1024)), self.d_model),
            nn.LayerNorm(self.d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(_cfg_get(config, "tf_num_head", 8)),
            dim_feedforward=int(_cfg_get(config, "tf_d_ffn", 1024)),
            dropout=float(_cfg_get(config, "tf_dropout", 0.0)),
            batch_first=True,
        )
        self.refiners = nn.ModuleList(
            [nn.TransformerDecoder(decoder_layer, 1) for _ in range(self.ref_num)]
        )
        self.delta_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d_model, int(_cfg_get(config, "tf_d_ffn", 1024))),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(_cfg_get(config, "tf_d_ffn", 1024)), self.num_poses * self.state_size),
                )
                for _ in range(self.ref_num)
            ]
        )
        for head in self.delta_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

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
        memory: torch.Tensor,
        ego_query: torch.Tensor,
    ) -> List[torch.Tensor]:
        current = proposals
        proposal_feature = self._proposal_feature(proposals, traj_feature_bev)
        query = proposal_feature + self.proposal_pos_embed(proposals.flatten(-2)) + ego_query

        poses_reg_list = []
        for refiner, delta_head in zip(self.refiners, self.delta_heads):
            query = refiner(query, memory)
            delta = delta_head(query).reshape_as(current)
            current = current + self.residual_scale * delta
            poses_reg_list.append(current)
        return poses_reg_list


class DriveJEPA3Decoder(nn.Module):
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
        self.query_embedding = nn.Embedding(sum(self.query_splits), self.d_model)
        self.tf_decoder = nn.TransformerDecoder(
            decoder_layer,
            int(_cfg_get(config, "tf_num_layers", 3)),
        )
        self.agent_head = DriveJEPA3AgentHead(
            num_agents=self.num_bounding_boxes,
            d_model=self.d_model,
            d_ffn=int(_cfg_get(config, "tf_d_ffn", 1024)),
        )
        self.proposal_refiner = DriveJEPA3ProposalRefiner(config)

    def forward(
        self,
        bev_feature: torch.Tensor,
        status_feature: torch.Tensor,
        proposals: torch.Tensor,
        traj_feature_bev: torch.Tensor,
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

        query = self.query_embedding.weight[None].expand(batch, -1, -1)
        query_out = self.tf_decoder(query, keyval)
        trajectory_query, agents_query = query_out.split(self.query_splits, dim=1)

        output = {"bev_semantic_map": bev_semantic_map}
        output.update(self.agent_head(agents_query))
        output["poses_reg_list"] = self.proposal_refiner(
            proposals=proposals,
            traj_feature_bev=traj_feature_bev,
            memory=keyval,
            ego_query=trajectory_query,
        )
        return output
