from typing import Sequence, Tuple

import torch
import torch.nn as nn


def _cfg_get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
        dtype=torch.long,
    )
    return bev_resolution, bev_start_position, bev_dimension


def euler2mat(angle: torch.Tensor):
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack(
        [cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1
    ).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack(
        [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1
    ).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack(
        [ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1
    ).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat.view(*shape[:-1], 3, 3)


def pose_vec2mat(vec: torch.Tensor):
    translation = vec[..., :3].unsqueeze(-1)
    rot_mat = euler2mat(vec[..., 3:].contiguous())
    transform_mat = torch.cat([rot_mat, translation], dim=-1)
    transform_mat = torch.nn.functional.pad(transform_mat, [0, 0, 0, 1], value=0)
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        return self.upsample_layer(x) + x_skip


class DriveJEPA2LSSProjector(nn.Module):
    """PAD/VeteranAD-style lift-splat projector for per-camera image features."""

    def __init__(self, config, in_channels: int, out_channels: int = 64):
        super().__init__()
        image_h, image_w = _cfg_get(config, "IMAGE_FINAL_DIM", [256, 512])
        downsample = int(_cfg_get(config, "MODEL_ENCODER_DOWNSAMPLE", 16))
        self.image_final_dim: Tuple[int, int] = (int(image_h), int(image_w))
        self.downsample = downsample
        self.out_channels = out_channels

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            _cfg_get(config, "LIFT_X_BOUND", [0.0, 32.0, 0.25]),
            _cfg_get(config, "LIFT_Y_BOUND", [-32.0, 32.0, 0.25]),
            _cfg_get(config, "LIFT_Z_BOUND", [-10.0, 10.0, 20.0]),
        )
        self.register_buffer("bev_resolution", bev_resolution.float(), persistent=False)
        self.register_buffer("bev_start_position", bev_start_position.float(), persistent=False)
        self.register_buffer("bev_dimension", bev_dimension.long(), persistent=False)
        self.register_buffer("frustum", self.create_frustum(config), persistent=False)
        self.depth_channels = self.frustum.shape[0]

        self.feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def create_frustum(self, config):
        image_h, image_w = self.image_final_dim
        feat_h = image_h // self.downsample
        feat_w = image_w // self.downsample

        depth_grid = torch.arange(*_cfg_get(config, "LIFT_D_BOUND", [2.0, 50.0, 1.0]), dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, feat_h, feat_w)
        n_depth = depth_grid.shape[0]

        x_grid = torch.linspace(0, image_w - 1, feat_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, feat_w).expand(n_depth, feat_h, feat_w)
        y_grid = torch.linspace(0, image_h - 1, feat_h, dtype=torch.float)
        y_grid = y_grid.view(1, feat_h, 1).expand(n_depth, feat_h, feat_w)
        return torch.stack((x_grid, y_grid, depth_grid), -1)

    def get_geometry(self, intrinsics, extrinsics):
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        batch, num_cam, _ = translation.shape
        points = self.frustum.to(device=intrinsics.device, dtype=intrinsics.dtype)
        points = points.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), dim=5
        )
        combined = rotation.matmul(torch.inverse(intrinsics))
        points = combined.view(batch, num_cam, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points = points + translation.view(batch, num_cam, 1, 1, 1, 3)
        return points

    def projection_to_birds_eye_view(self, x, geometry, future_egomotion):
        batch, seq_len, num_cam, depth, height, width, channels = x.shape
        out_h = int(self.bev_dimension[0].item())
        out_w = int(self.bev_dimension[1].item())
        out_z = int(self.bev_dimension[2].item())
        output = x.new_zeros((batch, seq_len, channels, out_h, out_w))

        future_egomotion_mat = pose_vec2mat(future_egomotion.to(dtype=x.dtype))
        rotation, translation = future_egomotion_mat[..., :3, :3], future_egomotion_mat[..., :3, 3]
        n_points = num_cam * depth * height * width
        flat_voxels = out_h * out_w * out_z

        for b_idx in range(batch):
            flow_b = x[b_idx]
            flow_geo = geometry[b_idx].clone()

            for t_idx in range(seq_len):
                if t_idx != seq_len - 1:
                    rot = rotation[b_idx, t_idx].view(1, 1, 1, 1, 1, 3, 3)
                    trans = translation[b_idx, t_idx].view(1, 1, 1, 1, 1, 3)
                    flow_geo[: t_idx + 1] = rot.matmul(flow_geo[: t_idx + 1].unsqueeze(-1)).squeeze(-1) + trans

            bev_feature = x.new_zeros((out_z, out_h, out_w, channels))
            for t_idx in range(seq_len):
                x_b = flow_b[t_idx].reshape(n_points, channels)
                geometry_b = (
                    (flow_geo[t_idx] - (self.bev_start_position - self.bev_resolution / 2.0))
                    / self.bev_resolution
                )
                geometry_b = geometry_b.view(n_points, 3).long()
                mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < out_h)
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < out_w)
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < out_z)
                )
                x_b = x_b[mask]
                geometry_b = geometry_b[mask]
                ranks = geometry_b[:, 0] * (out_w * out_z) + geometry_b[:, 1] * out_z + geometry_b[:, 2]

                tmp = x.new_zeros((flat_voxels, channels))
                tmp.index_add_(0, ranks, x_b)
                tmp = tmp.view(out_h, out_w, out_z, channels).permute(2, 0, 1, 3)
                bev_feature = tmp
                output[b_idx, t_idx] = bev_feature.permute(0, 3, 1, 2).squeeze(0)

        return output

    def forward(self, img_feats, intrinsics, extrinsics, future_egomotion):
        if img_feats.dim() != 5:
            raise ValueError(f"img_feats must be [B, N, C, H, W], got {tuple(img_feats.shape)}")
        batch, num_cam, channels, feat_h, feat_w = img_feats.shape
        image_h, image_w = self.image_final_dim
        expected_hw = (image_h // self.downsample, image_w // self.downsample)
        if (feat_h, feat_w) != expected_hw:
            raise ValueError(f"LSS feature map must be {expected_hw}, got {(feat_h, feat_w)}")

        feats = img_feats.reshape(batch * num_cam, channels, feat_h, feat_w)
        feats = self.feature_proj(feats)
        feats = feats.view(batch, num_cam, self.out_channels, feat_h, feat_w)
        feats = feats.unsqueeze(2).repeat(1, 1, self.depth_channels, 1, 1, 1)
        feats = feats.permute(0, 1, 2, 4, 5, 3)

        geometry = self.get_geometry(intrinsics, extrinsics).to(dtype=feats.dtype)
        feats = feats.unsqueeze(1)
        geometry = geometry.unsqueeze(1)
        bev = self.projection_to_birds_eye_view(feats, geometry, future_egomotion)
        return bev[:, -1]


class DriveJEPA2BEVSemanticHead(nn.Module):
    def __init__(self, in_channels: int = 64, num_classes: int = 7):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, in_channels, scale_factor=2)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        skip_1 = x
        x = self.relu(self.bn1(self.first_conv(x)))
        x = self.layer1(x)
        skip_2 = x
        x = self.layer2(x)
        skip_3 = x
        x = self.layer3(x)
        x = self.up3_skip(x, skip_3)
        x = self.up2_skip(x, skip_2)
        x = self.up1_skip(x, skip_1)
        return self.segmentation_head(x)


class SceneBEVGatedCrossAttention(nn.Module):
    def __init__(self, config, bev_channels: int = 64):
        super().__init__()
        d_model = int(_cfg_get(config, "tf_d_model", 256))
        nhead = int(_cfg_get(config, "tf_num_head", 8))
        dropout = float(_cfg_get(config, "tf_dropout", 0.0))
        token_h, token_w = _cfg_get(config, "bev_token_hw", [8, 16])
        self.num_cams = int(_cfg_get(config, "num_lss_cams", 4))
        self.bev_pool = nn.AdaptiveAvgPool2d((int(token_h), int(token_w)))
        self.bev_proj = nn.Conv2d(bev_channels, d_model, kernel_size=1)
        self.camera_embed = nn.Parameter(torch.randn(self.num_cams, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm1 = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(_cfg_get(config, "tf_d_ffn", 1024)),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.scene_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, scene_tokens: torch.Tensor, bev_feature: torch.Tensor) -> torch.Tensor:
        if scene_tokens.dim() != 4:
            raise ValueError(f"scene_tokens must be [B, N, S, D], got {tuple(scene_tokens.shape)}")
        batch, num_cam, num_scene, dim = scene_tokens.shape
        cam_embed = self.camera_embed[:num_cam].view(1, num_cam, 1, dim).to(scene_tokens.dtype)
        scene = (scene_tokens + cam_embed).reshape(batch, num_cam * num_scene, dim)

        bev = self.bev_proj(self.bev_pool(bev_feature)).flatten(2).transpose(1, 2)
        scene_bev, _ = self.attn(scene, bev, bev, need_weights=False)
        gate = self.gate(torch.cat([scene, scene_bev], dim=-1))
        scene = self.norm1(scene + gate * scene_bev)
        return self.scene_encoder(scene)
