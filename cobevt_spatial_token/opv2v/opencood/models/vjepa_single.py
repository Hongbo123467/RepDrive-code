"""
Single-CAV V-JEPA + robustbev-style geometric BEV projector.

This model keeps the single-ego data handling from FaxFusedTransformerSingle
but replaces FAX cross-view attention with deterministic OPV2V/CARLA
camera-to-BEV voxel unprojection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opencood.models.fax_fused_transformer_single import FaxFusedTransformerSingle
from opencood.models.sub_modules.jepa_spatial_encoder import JEPASpatialEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead


class VJEPABEVProjector(nn.Module):
    """
    Project per-camera V-JEPA image features into an ego-aligned BEV grid.

    OPV2V/CARLA camera coordinates are UE-style (x forward, y right, z up).
    Projection converts them to pinhole coordinates (y, -z, x), using x as
    depth. The BEV grid is label-aligned: rows go from +x to -x and columns
    go from -y to +y, matching ``col=c+ppm*y,row=c-ppm*x``.
    """

    def __init__(self, config, in_channels):
        super().__init__()
        self.raw_image_size = tuple(config.get("raw_image_size", [800, 600]))
        self.bev_shape = tuple(config.get("bev_shape", [32, 4, 32]))
        self.bev_bounds = tuple(config.get("bev_bounds", [-50, 50, -3, 1, -50, 50]))
        self.depth_eps = float(config.get("depth_eps", 1e-4))
        self.valid_debug = bool(config.get("valid_debug", False))

        if len(self.raw_image_size) != 2:
            raise ValueError("projector.raw_image_size must be [width, height]")
        if len(self.bev_shape) != 3:
            raise ValueError("projector.bev_shape must be [x_bins, z_bins, y_bins]")
        if len(self.bev_bounds) != 6:
            raise ValueError(
                "projector.bev_bounds must be [x_min, x_max, z_min, z_max, y_min, y_max]"
            )

        x_bins, z_bins, y_bins = self.bev_shape
        out_channels = int(config.get("out_channels", 128))
        self.height_compressor = nn.Sequential(
            nn.Conv2d(in_channels * z_bins, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.register_buffer("ego_grid", self._build_ego_grid(), persistent=False)

    def forward(self, image_features, intrinsic, extrinsic):
        """
        Parameters
        ----------
        image_features : torch.Tensor
            (B, M, C, Hf, Wf) current-frame camera features.
        intrinsic : torch.Tensor
            (B, M, 3, 3) or (B, 1, M, 3, 3), in raw OPV2V image pixels.
        extrinsic : torch.Tensor
            (B, M, 4, 4) or (B, 1, M, 4, 4), camera-to-ego transforms.

        Returns
        -------
        torch.Tensor
            Ego BEV bottleneck shaped (B, 128, x_bins, y_bins).
        """
        if image_features.dim() != 5:
            raise ValueError(
                "VJEPABEVProjector expects image_features shaped [B,M,C,H,W], "
                f"got {tuple(image_features.shape)}"
            )
        intrinsic = self._select_current_calibration(intrinsic)
        extrinsic = self._select_current_calibration(extrinsic)

        batch_size, num_cams, channels, feat_h, feat_w = image_features.shape
        if intrinsic.shape[:2] != (batch_size, num_cams):
            raise ValueError(
                "Intrinsic shape does not match image features: "
                f"intrinsic={tuple(intrinsic.shape)}, features={tuple(image_features.shape)}"
            )
        if extrinsic.shape[:2] != (batch_size, num_cams):
            raise ValueError(
                "Extrinsic shape does not match image features: "
                f"extrinsic={tuple(extrinsic.shape)}, features={tuple(image_features.shape)}"
            )

        grid = self.ego_grid.to(device=image_features.device, dtype=image_features.dtype)
        uv, depth = self._project_grid(grid, intrinsic, extrinsic, feat_h, feat_w)
        valid = (
            (depth > self.depth_eps)
            & (uv[..., 0] >= -0.5)
            & (uv[..., 0] < feat_w - 0.5)
            & (uv[..., 1] >= -0.5)
            & (uv[..., 1] < feat_h - 0.5)
        )

        sample_grid = self._normalize_grid(uv, feat_h, feat_w)
        flat_features = rearrange(image_features, "b m c h w -> (b m) c h w")
        flat_grid = rearrange(sample_grid, "b m n xy -> (b m) n 1 xy")
        sampled = F.grid_sample(
            flat_features,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1)
        sampled = rearrange(
            sampled,
            "(b m) c n -> b m c n",
            b=batch_size,
            m=num_cams,
        )
        valid = valid.unsqueeze(2).to(dtype=sampled.dtype)
        fused = (sampled * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        x_bins, z_bins, y_bins = self.bev_shape
        feat_mem = fused.reshape(batch_size, channels, x_bins, z_bins, y_bins)
        feat_bev = feat_mem.permute(0, 1, 3, 2, 4).reshape(
            batch_size, channels * z_bins, x_bins, y_bins
        )
        feat_bev = self.height_compressor(feat_bev)

        if self.valid_debug and (not torch.jit.is_scripting()):
            ratio = valid.squeeze(2).any(dim=1).float().mean().detach().item()
            print(f"[VJEPABEVProjector] any-camera valid voxel ratio: {ratio:.4f}")

        return feat_bev

    def _build_ego_grid(self):
        x_min, x_max, z_min, z_max, y_min, y_max = self.bev_bounds
        x_bins, z_bins, y_bins = self.bev_shape

        # Row 0 should correspond to the forward/top side of the BEV label.
        x = torch.linspace(float(x_max), float(x_min), int(x_bins))
        z = torch.linspace(float(z_min), float(z_max), int(z_bins))
        y = torch.linspace(float(y_min), float(y_max), int(y_bins))
        xx, zz, yy = torch.meshgrid(x, z, y, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    def _project_grid(self, grid, intrinsic, extrinsic, feat_h, feat_w):
        batch_size, num_cams = intrinsic.shape[:2]
        num_points = grid.shape[0]

        grid_h = torch.cat(
            [grid, torch.ones(num_points, 1, device=grid.device, dtype=grid.dtype)],
            dim=1,
        )
        grid_h = grid_h.view(1, 1, num_points, 4, 1)

        ego_to_cam = torch.linalg.inv(extrinsic)
        cam_ue = torch.matmul(ego_to_cam.unsqueeze(2), grid_h).squeeze(-1)[..., :3]

        pinhole = torch.stack(
            [cam_ue[..., 1], -cam_ue[..., 2], cam_ue[..., 0]],
            dim=-1,
        )
        depth = pinhole[..., 2]

        k_feat = self._scale_intrinsic(intrinsic, feat_h, feat_w)
        uvw = torch.matmul(k_feat.unsqueeze(2), pinhole.unsqueeze(-1)).squeeze(-1)
        uv = uvw[..., :2] / uvw[..., 2:3].clamp_min(self.depth_eps)
        return uv.view(batch_size, num_cams, num_points, 2), depth.view(
            batch_size, num_cams, num_points
        )

    def _scale_intrinsic(self, intrinsic, feat_h, feat_w):
        raw_w, raw_h = self.raw_image_size
        scaled = intrinsic.clone()
        scaled[..., 0, :] = scaled[..., 0, :] * (float(feat_w) / float(raw_w))
        scaled[..., 1, :] = scaled[..., 1, :] * (float(feat_h) / float(raw_h))
        return scaled

    @staticmethod
    def _normalize_grid(uv, feat_h, feat_w):
        x = (uv[..., 0] + 0.5) * (2.0 / float(feat_w)) - 1.0
        y = (uv[..., 1] + 0.5) * (2.0 / float(feat_h)) - 1.0
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _select_current_calibration(tensor):
        if tensor is None:
            raise ValueError("VJEPABEVProjector requires camera calibration")
        if tensor.dim() == 5:
            return tensor[:, 0]
        if tensor.dim() == 4:
            return tensor
        raise ValueError(
            "Expected calibration shaped [B,1,M,...] or [B,M,...], got "
            f"{tuple(tensor.shape)}"
        )


class VjepaSingle(FaxFusedTransformerSingle):
    """
    Single-vehicle segmentation model:
        V-JEPA2.1 spatial features -> voxel unprojection -> NaiveDecoder -> BevSegHead.
    """

    def __init__(self, config):
        nn.Module.__init__(self)
        config = self._unwrap_model_config(config)
        if "jepa_encoder" not in config:
            raise ValueError(
                "VjepaSingle expects a 'jepa_encoder' config block. "
                f"Got config keys: {sorted(config.keys())}"
            )

        self.jepa_encoder = JEPASpatialEncoder(config["jepa_encoder"])
        self.projector = VJEPABEVProjector(
            config.get("projector", {}),
            in_channels=self.jepa_encoder.output_dim,
        )
        self.decoder = NaiveDecoder(config["decoder"])
        self.target = config["target"]
        self.seg_head = BevSegHead(
            self.target, config["seg_head_dim"], config["output_class"]
        )

    def forward(self, batch_dict):
        encoder_inputs, _, intrinsic, extrinsic = self._prepare_ego_batch(batch_dict)

        image_features = self.jepa_encoder(encoder_inputs)
        x = self.projector(image_features, intrinsic, extrinsic).unsqueeze(1)
        x = self.decoder(x)
        x = rearrange(x, "b l c h w -> (b l) c h w")
        batch_size = x.shape[0]
        return self.seg_head(x, batch_size, 1)