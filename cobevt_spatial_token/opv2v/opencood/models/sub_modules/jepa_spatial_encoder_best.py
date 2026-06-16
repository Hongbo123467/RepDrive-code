"""
jepa_spatial_encoder_best.py

Drop-in replacement for:
    opencood/models/sub_modules/jepa_spatial_encoder.py

Purpose:
    V-JEPA2/2.1 spatial-token encoder for geometry-aware BEV pipelines.

Main changes over the previous version:
    1. Explicitly supports V-JEPA2.1 multiscale token extraction.
    2. Passes LoRA advanced configuration fields to ImgEncoderVJEPA2/2.1.
    3. Uses LayerNorm-based token necks before projecting hierarchical tokens
       to fpn_feature_dim, which is more stable for foundation-model tokens.
    4. Supports current-frame geometry usage by accepting 5D images
       [N, M, H, W, C] and internally duplicating the current frame.
    5. Adds clearer shape checks and a configurable previous-frame index.

Recommended config keys:
    jepa_encoder:
      vjepa_version: '2.1'
      backbone_mode: 'lora'
      lora_rank: 8
      fpn_feature_dim: 256
      fpn_layer_indices: [5, 11, 17, 23]
      fpn_neck_norm: true
      fpn_neck_act: true
      prev_frame_index: 1
      use_grid_mask: false

If your ImgEncoderVJEPA21 supports advanced LoRA settings, place them either at
jepa_encoder level or inside jepa_encoder.image_backbone.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.vjepa2_lora import ImgEncoderVJEPA2
from opencood.models.sub_modules.vjepa2_1_lora import ImgEncoderVJEPA21


class DotDict(dict):
    """Dict with attribute access for V-JEPA2-style configs."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _make_token_neck(
    in_dim: int,
    out_dim: int,
    use_norm: bool = True,
    use_act: bool = True,
) -> nn.Module:
    """Project token features [B, N, C] to [B, N, out_dim]."""
    layers: List[nn.Module] = []
    if use_norm:
        layers.append(nn.LayerNorm(in_dim))
    layers.append(nn.Linear(in_dim, out_dim))
    if use_act:
        layers.append(nn.GELU())
    if use_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class JEPASpatialEncoder(nn.Module):
    """
    Adapter around ImgEncoderVJEPA2 / ImgEncoderVJEPA21 that returns V-JEPA
    spatial tokens as image-like feature maps.

    Input:
        - Temporal: (N, T, M, H, W, C)
        - Current-frame only: (N, M, H, W, C)

    Output:
        forward():          (N, M, output_dim, Hs, Ws)
        forward_multiscale(): list of (N, M, fpn_feature_dim, Hs, Ws)
    """

    _PASSTHROUGH_KEYS = (
        # original/common keys
        "vjepa2_config_path",
        "model_weights",
        "image_architecture",
        "vjepa2_resolution",
        "img_as_video_nframes",
        "backbone_mode",
        "lora_rank",
        "use_grid_mask",
        "use_feature_pooling",
        "focus_front_cam",
        "compress_fc",
        "vjepa2_n_tokens",
        # advanced LoRA/adaptation keys; harmless if ignored by ImgEncoder
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "lora_train_blocks",
        "lora_train_layers",
        "lora_layers",
        "train_blocks",
        "finetune_blocks",
        "freeze_patch_embed",
        "freeze_pos_embed",
    )

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.spatial_hw = tuple(args.get("spatial_hw", [16, 32]))
        self.vjepa2_resolution = tuple(args.get("vjepa2_resolution", [256, 512]))
        self.num_scene_tokens = int(args.get("num_scene_tokens", 16))
        self.output_dim = int(args.get("tf_d_model", args.get("output_dim", 128)))
        self.fpn_feature_dim = int(args.get("fpn_feature_dim", self.output_dim))
        self.fpn_layer_indices = list(args.get("fpn_layer_indices", [5, 11, 17, 23]))
        if len(self.fpn_layer_indices) == 0:
            raise ValueError("fpn_layer_indices must not be empty for forward_multiscale")

        self.input_is_normalized = bool(args.get("input_is_normalized", True))
        self.prev_frame_index = int(args.get("prev_frame_index", 1))
        self.fpn_neck_norm = bool(args.get("fpn_neck_norm", True))
        self.fpn_neck_act = bool(args.get("fpn_neck_act", True))

        backbone_cfg = DotDict(args.get("image_backbone", {}))
        for key in self._PASSTHROUGH_KEYS:
            if key in args and key not in backbone_cfg:
                backbone_cfg[key] = args[key]

        backbone_cfg["image_size"] = args.get(
            "image_size", args.get("vjepa2_resolution", [256, 512])
        )
        backbone_cfg["num_scene_tokens"] = self.num_scene_tokens
        backbone_cfg["tf_d_model"] = self.output_dim
        backbone_cfg.setdefault("img_as_video_nframes", 2)
        backbone_cfg.setdefault("backbone_mode", "frozen")
        backbone_cfg.setdefault("lora_rank", 8)

        self.vjepa_version = str(args.get("vjepa_version", "2")).lower()
        if self.vjepa_version in ("2.1", "v2.1", "vjepa2.1", "vjepa2_1"):
            self.image_backbone = ImgEncoderVJEPA21(backbone_cfg)
        else:
            self.image_backbone = ImgEncoderVJEPA2(backbone_cfg)

        self.backbone_mode = self.image_backbone.backbone_mode
        self.use_scene_tokens = self.backbone_mode == "scene_lora"

        self.fpn_necks = nn.ModuleList(
            [
                _make_token_neck(
                    self.image_backbone.num_features,
                    self.fpn_feature_dim,
                    use_norm=self.fpn_neck_norm,
                    use_act=self.fpn_neck_act,
                )
                for _ in self.fpn_layer_indices
            ]
        )

        if self.use_scene_tokens:
            self.scene_embeds = nn.Parameter(
                torch.randn(1, self.num_scene_tokens, self.image_backbone.num_features)
                * 1e-6,
                requires_grad=True,
            )
        else:
            self.scene_embeds = None

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """Return a single spatial map shaped (N, M, output_dim, Hs, Ws)."""
        cur_frame, prev_frame, shape_meta = self._prepare_current_previous_frames(input_images)
        n_agents, num_cams, _, _, _ = shape_meta

        spatial_tokens = self._encode_raw_spatial_tokens(cur_frame, prev_frame)
        token_h, token_w = self.spatial_hw
        self._check_token_count(spatial_tokens, token_h, token_w)

        # image_backbone.neck is kept for compatibility with the original encoder.
        spatial_features = self.image_backbone.neck(spatial_tokens)
        spatial_features = spatial_features.transpose(1, 2).reshape(
            n_agents, num_cams, self.output_dim, token_h, token_w
        )
        return spatial_features

    def forward_multiscale(self, input_images: torch.Tensor) -> List[torch.Tensor]:
        """
        Return V-JEPA hierarchical layers as same-scale spatial feature maps.

        Returns:
            list of tensors shaped (N, M, fpn_feature_dim, Hs, Ws).
        """
        if self.vjepa_version not in ("2.1", "v2.1", "vjepa2.1", "vjepa2_1"):
            raise ValueError(
                "forward_multiscale currently expects V-JEPA 2.1. "
                "Set jepa_encoder.vjepa_version: '2.1' in the YAML."
            )

        cur_frame, prev_frame, shape_meta = self._prepare_current_previous_frames(input_images)
        n_agents, num_cams, _, _, _ = shape_meta

        layer_tokens = self._encode_raw_multiscale_tokens(cur_frame, prev_frame)
        token_h, token_w = self.spatial_hw
        if len(layer_tokens) != len(self.fpn_necks):
            raise ValueError(
                f"Expected {len(self.fpn_necks)} token layers, got {len(layer_tokens)}. "
                "Check fpn_layer_indices and V-JEPA model out_layers behavior."
            )

        spatial_features: List[torch.Tensor] = []
        for tokens, neck in zip(layer_tokens, self.fpn_necks):
            self._check_token_count(tokens, token_h, token_w)
            feature = neck(tokens)
            feature = feature.transpose(1, 2).reshape(
                n_agents, num_cams, self.fpn_feature_dim, token_h, token_w
            )
            spatial_features.append(feature)
        return spatial_features

    def forward_bevformer(self, input_images, intrinsic=None, extrinsic=None):
        """
        Return V-JEPA spatial tokens in RAP BEVFormer format.
        """
        spatial_features = self(input_images)
        n_agents, num_cams, channels, height, width = spatial_features.shape

        feat_flatten = spatial_features.flatten(3).permute(1, 3, 0, 2).contiguous()
        spatial_shapes = torch.as_tensor(
            [[height, width]], dtype=torch.long, device=spatial_features.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        img_metas = {
            "img_shape": self._build_img_shape(input_images, n_agents, num_cams),
        }
        if intrinsic is not None and extrinsic is not None:
            img_metas["lidar2img"] = self._build_lidar2img(intrinsic, extrinsic, num_cams)
        else:
            img_metas["lidar2img"] = self._identity_lidar2img(
                input_images, n_agents, num_cams
            )

        return feat_flatten, spatial_shapes, level_start_index, {"img_metas": img_metas}

    def _prepare_current_previous_frames(self, input_images: torch.Tensor):
        """
        Convert input to current/previous BCHW tensors.

        input_images:
            - [N, M, H, W, C]: current-only; previous is duplicated current.
            - [N, T, M, H, W, C]: current at T=0; previous selected by prev_frame_index.
        """
        if input_images.dim() == 5:
            input_images = input_images.unsqueeze(1)
        if input_images.dim() != 6:
            raise ValueError(
                "JEPASpatialEncoder expects inputs with shape "
                "(N,T,M,H,W,C) or (N,M,H,W,C), got "
                f"{tuple(input_images.shape)}"
            )

        n_agents, time_len, num_cams, height, width, channels = input_images.shape
        if channels != 3:
            raise ValueError(f"Expected RGB channel-last images, got C={channels}")

        cur = input_images[:, 0]
        prev_idx = self.prev_frame_index if time_len > self.prev_frame_index else 0
        prev = input_images[:, prev_idx]

        cur = cur.permute(0, 1, 4, 2, 3).reshape(n_agents * num_cams, channels, height, width)
        prev = prev.permute(0, 1, 4, 2, 3).reshape(n_agents * num_cams, channels, height, width)
        return cur, prev, (n_agents, num_cams, channels, height, width)

    def _make_img_video(self, cur_frame: torch.Tensor, prev_frame: torch.Tensor) -> torch.Tensor:
        # Drive-JEPA order: current frame first, previous frame second.
        img_video = torch.stack([cur_frame, prev_frame], dim=2)  # [B, C, 2, H, W]
        if tuple(img_video.shape[-2:]) != self.vjepa2_resolution:
            batch, channels, frames, height, width = img_video.shape
            img_flat = img_video.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, channels, height, width
            )
            img_flat = F.interpolate(
                img_flat,
                size=self.vjepa2_resolution,
                mode="bilinear",
                align_corners=False,
            )
            img_video = img_flat.reshape(
                batch, frames, channels, *self.vjepa2_resolution
            ).permute(0, 2, 1, 3, 4)

        if self.image_backbone.use_grid_mask and self.training:
            batch, channels, frames, height, width = img_video.shape
            img_flat = img_video.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, channels, height, width
            ).contiguous()
            img_flat = self.image_backbone.grid_mask(img_flat)
            img_video = img_flat.reshape(batch, frames, channels, height, width).permute(
                0, 2, 1, 3, 4
            )
        return img_video

    def _encode_raw_spatial_tokens(self, cur_frame: torch.Tensor, prev_frame: torch.Tensor):
        img_video = self._make_img_video(cur_frame, prev_frame)
        scene_tokens, scene_token_count = self._get_scene_tokens(img_video.shape[0])

        if self.backbone_mode == "frozen":
            self.image_backbone.model.eval()
            with torch.no_grad():
                tokens = self.image_backbone.model(img_video, scene_tokens=None)
        elif self.backbone_mode == "lora":
            tokens = self.image_backbone.model(img_video, scene_tokens=None)
        else:
            tokens = self.image_backbone.model(img_video, scene_tokens=scene_tokens)
        return self._strip_scene_tokens(tokens, scene_token_count)

    def _encode_raw_multiscale_tokens(self, cur_frame: torch.Tensor, prev_frame: torch.Tensor):
        img_video = self._make_img_video(cur_frame, prev_frame)
        scene_tokens, scene_token_count = self._get_scene_tokens(img_video.shape[0])

        vit_model = self._underlying_vit_model()
        old_out_layers = getattr(vit_model, "out_layers", None)
        vit_model.out_layers = self.fpn_layer_indices
        try:
            if self.backbone_mode == "frozen":
                self.image_backbone.model.eval()
                with torch.no_grad():
                    tokens = self.image_backbone.model(img_video, scene_tokens=None)
            elif self.backbone_mode == "lora":
                tokens = self.image_backbone.model(img_video, scene_tokens=None)
            else:
                tokens = self.image_backbone.model(img_video, scene_tokens=scene_tokens)
        finally:
            vit_model.out_layers = old_out_layers

        if not isinstance(tokens, (list, tuple)):
            raise ValueError(
                "Expected V-JEPA 2.1 to return a list of hierarchical layers, "
                f"got {type(tokens)}. Check fpn_layer_indices / out_layers support."
            )
        return [self._strip_scene_tokens(t, scene_token_count) for t in tokens]

    def _get_scene_tokens(self, batch_size: int):
        if self.scene_embeds is not None:
            scene_tokens = self.scene_embeds.expand(batch_size, -1, -1)
            return scene_tokens, scene_tokens.shape[1]
        return None, 0

    def _strip_scene_tokens(self, tokens: torch.Tensor, scene_token_count: int) -> torch.Tensor:
        if scene_token_count == 0:
            return tokens
        if self.vjepa_version in ("2.1", "v2.1", "vjepa2.1", "vjepa2_1"):
            return tokens[:, :-scene_token_count]
        return tokens[:, scene_token_count:]

    def _underlying_vit_model(self):
        model = self.image_backbone.model
        return getattr(model, "lora_vit", model)

    @staticmethod
    def _check_token_count(tokens: torch.Tensor, token_h: int, token_w: int):
        expected_tokens = token_h * token_w
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                "V-JEPA spatial token count does not match spatial_hw: "
                f"tokens={tokens.shape[1]}, spatial_hw=({token_h}, {token_w}). "
                "Check vjepa2_resolution, patch size, and spatial_hw in YAML."
            )

    @staticmethod
    def _select_current_timestamp(tensor):
        if tensor.dim() == 5:
            return tensor[:, 0]
        return tensor

    @staticmethod
    def _build_lidar2img(intrinsic, extrinsic, num_cams):
        intrinsic = JEPASpatialEncoder._select_current_timestamp(intrinsic)
        extrinsic = JEPASpatialEncoder._select_current_timestamp(extrinsic)
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

    @staticmethod
    def _build_img_shape(input_images, n_agents, num_cams):
        height, width = input_images.shape[-3], input_images.shape[-2]
        return torch.as_tensor(
            [[[height, width, 3]] * num_cams] * n_agents,
            dtype=torch.float32,
            device=input_images.device,
        )

    @staticmethod
    def _identity_lidar2img(input_images, n_agents, num_cams):
        eye = torch.eye(4, device=input_images.device, dtype=input_images.dtype)
        return eye.view(1, 1, 4, 4).repeat(n_agents, num_cams, 1, 1)
