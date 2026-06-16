import torch
import torch.nn as nn


class JEPABEVRefiner(nn.Module):
    """
    Build per-CAV local BEV features from V-JEPA2 spatial image features.

    The module keeps proposal_query fixed to False: BEV queries are learned
    dense grid queries, not planning proposal queries.
    """

    def __init__(self, args):
        super().__init__()
        self.in_channels = args.get("in_channels", args.get("input_dim", 128))
        self.bev_channels = args.get("bev_channels", args.get("output_dim", 128))
        self.bev_h = args.get("bev_h", 32)
        self.bev_w = args.get("bev_w", 32)
        self.num_layers = args.get("num_layers", 2)
        self.num_heads = args.get("num_heads", 4)
        self.dropout = args.get("dropout", 0.1)
        self.proposal_query = False

        self.input_proj = nn.Conv2d(self.in_channels, self.bev_channels, 1)
        self.camera_embed = nn.Embedding(args.get("num_cams", 4), self.bev_channels)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(16, self.bev_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.bev_channels, self.bev_channels),
        )

        self.bev_queries = nn.Parameter(
            torch.randn(self.bev_h * self.bev_w, self.bev_channels) * 0.02
        )
        self.bev_pos = nn.Parameter(
            torch.randn(self.bev_h * self.bev_w, self.bev_channels) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                _BEVCrossAttentionLayer(
                    self.bev_channels,
                    self.num_heads,
                    self.dropout,
                    args.get("ffn_dim", self.bev_channels * 4),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.bev_channels)

    def forward(self, image_features, intrinsic=None, extrinsic=None):
        """
        Parameters
        ----------
        image_features : torch.Tensor
            Shape (N, M, C, H, W), where N is sum(record_len).
        intrinsic : torch.Tensor, optional
            Shape (N,T,M,3,3) or (N,M,3,3).
        extrinsic : torch.Tensor, optional
            Shape (N,T,M,4,4) or (N,M,4,4), interpreted as camera-to-ego.

        Returns
        -------
        torch.Tensor
            Local BEV features shaped (N, C, bev_h, bev_w).
        """
        if image_features.dim() != 5:
            raise ValueError(
                "JEPABEVRefiner expects image features shaped (N,M,C,H,W), got "
                f"{tuple(image_features.shape)}"
            )

        n_agents, num_cams, channels, height, width = image_features.shape
        x = image_features.reshape(n_agents * num_cams, channels, height, width)
        x = self.input_proj(x)
        x = x.reshape(n_agents, num_cams, self.bev_channels, height, width)
        x = x.flatten(3).permute(0, 1, 3, 2).contiguous()

        cam_ids = torch.arange(num_cams, device=image_features.device)
        cam_embed = self.camera_embed(cam_ids).view(1, num_cams, 1, self.bev_channels)
        x = x + cam_embed

        if intrinsic is not None and extrinsic is not None:
            lidar2img = self._build_lidar2img(intrinsic, extrinsic, num_cams)
            geom_embed = self.geometry_mlp(lidar2img.reshape(n_agents, num_cams, 16))
            x = x + geom_embed.unsqueeze(2).to(dtype=x.dtype)

        memory = x.reshape(n_agents, num_cams * height * width, self.bev_channels)
        query = self.bev_queries.unsqueeze(0).expand(n_agents, -1, -1)
        query = query + self.bev_pos.unsqueeze(0)

        for layer in self.layers:
            query = layer(query, memory)

        query = self.out_norm(query)
        query = query.transpose(1, 2).reshape(
            n_agents, self.bev_channels, self.bev_h, self.bev_w
        )
        return query

    @staticmethod
    def _select_current_timestamp(tensor):
        if tensor.dim() == 5:
            return tensor[:, 0]
        return tensor

    def _build_lidar2img(self, intrinsic, extrinsic, num_cams):
        intrinsic = self._select_current_timestamp(intrinsic)
        extrinsic = self._select_current_timestamp(extrinsic)
        if intrinsic.shape[1] != num_cams or extrinsic.shape[1] != num_cams:
            raise ValueError(
                "Camera calibration shape does not match feature camera count: "
                f"intrinsic={tuple(intrinsic.shape)}, "
                f"extrinsic={tuple(extrinsic.shape)}, num_cams={num_cams}"
            )

        k4 = torch.eye(4, device=intrinsic.device, dtype=intrinsic.dtype)
        k4 = k4.view(1, 1, 4, 4).repeat(intrinsic.shape[0], num_cams, 1, 1)
        k4[:, :, :3, :3] = intrinsic

        ego_to_cam = torch.linalg.inv(extrinsic)
        return torch.matmul(k4, ego_to_cam).to(dtype=intrinsic.dtype)


class _BEVCrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout, ffn_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory):
        attn_out, _ = self.cross_attn(query, memory, memory, need_weights=False)
        query = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_out))
        return query
