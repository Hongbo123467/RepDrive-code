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


class JEPASpatialEncoder(nn.Module):
    """
    Adapter around ImgEncoderVJEPA2 that returns V-JEPA2 spatial tokens as
    image-like feature maps instead of scene-token summaries.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.spatial_hw = tuple(args.get("spatial_hw", [16, 32]))
        self.vjepa2_resolution = tuple(args.get("vjepa2_resolution", [256, 512]))
        self.num_scene_tokens = args.get("num_scene_tokens", 16)
        self.output_dim = args.get("tf_d_model", args.get("output_dim", 128))
        self.fpn_feature_dim = args.get("fpn_feature_dim", self.output_dim)
        self.fpn_layer_indices = list(args.get("fpn_layer_indices", [5, 11, 17, 23]))
        self.input_is_normalized = args.get("input_is_normalized", True)

        backbone_cfg = DotDict(args.get("image_backbone", {}))
        for key in (
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
        ):
            if key in args and key not in backbone_cfg:
                backbone_cfg[key] = args[key]

        backbone_cfg["image_size"] = args.get("image_size", args.get("vjepa2_resolution", [256, 512]))
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
                nn.Linear(self.image_backbone.num_features, self.fpn_feature_dim)
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

    def forward(self, input_images):
        """
        Parameters
        ----------
        input_images : torch.Tensor
            Temporal CoBEVT images, expected as (N, T, M, H, W, C), with the
            current frame at T=0 and history frames after it. A legacy
            non-temporal (N, M, H, W, C) tensor is accepted by duplicating the
            current frame.

        Returns
        -------
        torch.Tensor
            Spatial feature maps shaped (N, M, C, Hs, Ws).
        """
        if input_images.dim() == 5:
            input_images = input_images.unsqueeze(1).repeat(1, 2, 1, 1, 1, 1)
        if input_images.dim() != 6:
            raise ValueError(
                "JEPASpatialEncoder expects inputs with shape "
                "(N,T,M,H,W,C) or (N,M,H,W,C), got "
                f"{tuple(input_images.shape)}"
            )

        n_agents, time_len, num_cams, height, width, channels = input_images.shape
        if channels != 3:
            raise ValueError(f"Expected RGB channel-last images, got C={channels}")
        if time_len < 2:
            cur_frame = input_images[:, 0]
            prev_frame = input_images[:, 0]
        else:
            cur_frame = input_images[:, 0]
            prev_frame = input_images[:, 1]

        prev_frame = prev_frame.permute(0, 1, 4, 2, 3).reshape(
            n_agents * num_cams, channels, height, width
        )
        cur_frame = cur_frame.permute(0, 1, 4, 2, 3).reshape(
            n_agents * num_cams, channels, height, width
        )

        spatial_tokens = self._encode_raw_spatial_tokens(cur_frame, prev_frame)
        token_h, token_w = self.spatial_hw
        expected_tokens = token_h * token_w
        if spatial_tokens.shape[1] != expected_tokens:
            raise ValueError(
                "V-JEPA2 spatial token count does not match spatial_hw: "
                f"tokens={spatial_tokens.shape[1]}, spatial_hw={self.spatial_hw}"
            )

        spatial_features = self.image_backbone.neck(spatial_tokens)
        spatial_features = spatial_features.transpose(1, 2).reshape(
            n_agents, num_cams, self.output_dim, token_h, token_w
        )
        return spatial_features

    def forward_multiscale(self, input_images):
        """
        Return V-JEPA hierarchical layers as same-scale spatial feature maps.

        Returns a list of tensors shaped (N, M, fpn_feature_dim, Hs, Ws).
        """
        if input_images.dim() == 5:
            input_images = input_images.unsqueeze(1).repeat(1, 2, 1, 1, 1, 1)
        if input_images.dim() != 6:
            raise ValueError(
                "JEPASpatialEncoder.forward_multiscale expects inputs with shape "
                "(N,T,M,H,W,C) or (N,M,H,W,C), got "
                f"{tuple(input_images.shape)}"
            )

        n_agents, time_len, num_cams, height, width, channels = input_images.shape
        if channels != 3:
            raise ValueError(f"Expected RGB channel-last images, got C={channels}")
        if time_len < 2:
            cur_frame = input_images[:, 0]
            prev_frame = input_images[:, 0]
        else:
            cur_frame = input_images[:, 0]
            prev_frame = input_images[:, 1]

        prev_frame = prev_frame.permute(0, 1, 4, 2, 3).reshape(
            n_agents * num_cams, channels, height, width
        )
        cur_frame = cur_frame.permute(0, 1, 4, 2, 3).reshape(
            n_agents * num_cams, channels, height, width
        )

        layer_tokens = self._encode_raw_multiscale_tokens(cur_frame, prev_frame)
        token_h, token_w = self.spatial_hw
        expected_tokens = token_h * token_w
        spatial_features = []
        for tokens, neck in zip(layer_tokens, self.fpn_necks):
            if tokens.shape[1] != expected_tokens:
                raise ValueError(
                    "V-JEPA2 spatial token count does not match spatial_hw: "
                    f"tokens={tokens.shape[1]}, spatial_hw={self.spatial_hw}"
                )
            feature = neck(tokens)
            feature = feature.transpose(1, 2).reshape(
                n_agents, num_cams, self.fpn_feature_dim, token_h, token_w
            )
            spatial_features.append(feature)
        return spatial_features

    def forward_bevformer(self, input_images, intrinsic=None, extrinsic=None):
        """
        Return V-JEPA spatial tokens in RAP BEVFormer format.

        The feature tuple matches RAP's ImgEncoder output:
            feat_flatten: (num_cam, Hs*Ws, N, output_dim)
            spatial_shapes: (1, 2)
            level_start_index: (1,)
            kwargs: {"img_metas": {"lidar2img", "img_shape"}}
        """
        spatial_features = self(input_images)
        n_agents, num_cams, channels, height, width = spatial_features.shape

        feat_flatten = spatial_features.flatten(3).permute(1, 3, 0, 2).contiguous()
        spatial_shapes = torch.as_tensor(
            [[height, width]], dtype=torch.long, device=spatial_features.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        img_metas = {
            "img_shape": self._build_img_shape(input_images, n_agents, num_cams),
        }
        if intrinsic is not None and extrinsic is not None:
            img_metas["lidar2img"] = self._build_lidar2img(
                intrinsic, extrinsic, num_cams
            )
        else:
            img_metas["lidar2img"] = self._identity_lidar2img(
                input_images, n_agents, num_cams
            )

        return feat_flatten, spatial_shapes, level_start_index, {"img_metas": img_metas}

    def _encode_raw_spatial_tokens(self, cur_frame, prev_frame):
        # Drive-JEPA order: current frame first, previous frame second.
        img_video = torch.stack([cur_frame, prev_frame], dim=2)
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

        if self.scene_embeds is not None:
            scene_tokens = self.scene_embeds.expand(img_video.shape[0], -1, -1)
            scene_token_count = scene_tokens.shape[1]
        else:
            scene_tokens = None
            scene_token_count = 0

        if self.backbone_mode == "frozen":
            self.image_backbone.model.eval()
            with torch.no_grad():
                tokens = self.image_backbone.model(img_video, scene_tokens=None)
        elif self.backbone_mode == "lora":
            tokens = self.image_backbone.model(img_video, scene_tokens=None)
        else:
            tokens = self.image_backbone.model(img_video, scene_tokens=scene_tokens)

        return self._strip_scene_tokens(tokens, scene_token_count)

    def _encode_raw_multiscale_tokens(self, cur_frame, prev_frame):
        if not self.vjepa_version in ("2.1", "v2.1", "vjepa2.1", "vjepa2_1"):
            raise ValueError("forward_multiscale currently expects V-JEPA 2.1")

        img_video = torch.stack([cur_frame, prev_frame], dim=2)
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

        if self.scene_embeds is not None:
            scene_tokens = self.scene_embeds.expand(img_video.shape[0], -1, -1)
            scene_token_count = scene_tokens.shape[1]
        else:
            scene_tokens = None
            scene_token_count = 0

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
                f"got {type(tokens)}"
            )
        return [self._strip_scene_tokens(t, scene_token_count) for t in tokens]

    def _strip_scene_tokens(self, tokens, scene_token_count):
        if scene_token_count == 0:
            return tokens
        if self.vjepa_version in ("2.1", "v2.1", "vjepa2.1", "vjepa2_1"):
            return tokens[:, :-scene_token_count]
        return tokens[:, scene_token_count:]

    def _underlying_vit_model(self):
        model = self.image_backbone.model
        return getattr(model, "lora_vit", model)

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
