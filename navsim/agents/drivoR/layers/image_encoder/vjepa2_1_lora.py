# V-JEPA 2.1 LoRA image encoder.
#
# Difference from vjepa2_lora.py:
#   - uses the V-JEPA 2.1 model implementation under app/vjepa_2_1
#   - appends scene tokens to the end of the patch sequence
#   - treats scene tokens as RoPE registers, matching V-JEPA 2.1's native
#     n_registers handling instead of the older prefix action-token path

import math
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn.parameter import Parameter

from .grid_mask import GridMask
from .vjepa2_lora import _LoRA_qkv_vjepa2
from navsim.agents.drivoR.utils import pylogger

log = pylogger.get_pylogger(__name__)


def _config_get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def _add_vjepa21_repo_to_path(config):
    repo_path = _config_get(config, "vjepa2_repo_path", None)
    if repo_path is None:
        repo_path = os.environ.get(
            "VJEPA2_1_REPO_PATH",
            "/home/dataset-assist-0/yinhongbo/code/robotics/vjepa2-main",
        )
    repo_path = Path(repo_path).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"V-JEPA 2.1 repo path not found: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path


def _set_rope_register_count(model: nn.Module, n_registers: int) -> None:
    for blk in getattr(model, "blocks", []):
        attn = getattr(blk, "attn", None)
        if hasattr(attn, "n_registers"):
            attn.n_registers = n_registers


class VisionTransformerV21WithSceneRegisters(nn.Module):
    """
    Mixin target for V-JEPA 2.1 VisionTransformer.

    The original V-JEPA 2.1 RoPE implementation excludes the final
    ``n_registers`` tokens from RoPE. We append external scene tokens at the
    end of the token sequence and set ``n_registers`` dynamically to their
    count, so patch tokens keep the native 3D RoPE coordinates and scene tokens
    remain unpositioned global queries.
    """

    def forward_features(self, x: Tensor, scene_tokens: Tensor = None, masks=None, training: bool = False) -> Tensor:
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        elif x.ndim == 5:
            _, _, T_frames, H, W = x.shape
            if self.check_temporal_dim(x.shape):
                T = T_frames
            else:
                T = T_frames // self.tubelet_size
        else:
            raise ValueError(f"Unsupported input ndim for V-JEPA 2.1: {x.ndim}")

        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)

        if self.check_temporal_dim(x.shape):
            if self.patch_embed_img is None:
                raise RuntimeError("img_temporal_dim_size is set but patch_embed_img is missing")
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x = x + self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x = x + self.video_mod_embed.repeat(x.shape[0], 1, 1)

        if not self.use_rope:
            x = x + pos_embed

        if masks is not None:
            # Keep mask semantics identical to upstream: masks index patch tokens.
            from src.masks.utils import apply_masks

            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        n_scene = scene_tokens.shape[1] if scene_tokens is not None else 0
        if n_scene > 0:
            if scene_tokens.shape[0] != x.shape[0] or scene_tokens.shape[2] != x.shape[2]:
                raise ValueError(
                    "scene_tokens must have shape [B, S, embed_dim], "
                    f"got {tuple(scene_tokens.shape)} for patch tokens {tuple(x.shape)}"
                )
            x = torch.cat([x, scene_tokens], dim=1)

        _set_rope_register_count(self, n_scene)

        outs = []
        hier = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x, attn = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    use_reentrant=False,
                    return_attn=self.attn_out,
                    mode=mode,
                )
            else:
                x, attn = blk(
                    x,
                    mask=masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    return_attn=self.attn_out,
                    mode=mode,
                )

            if self.out_layers is not None and i in self.out_layers:
                out_idx = self.hierarchical_layers.index(i)
                outs.append(self.norms_block[out_idx](x))

            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if self.out_layers is not None:
            return outs

        if training or self.return_hierarchical:
            return torch.cat(hier, dim=2)

        return self.norms_block[-1](x)

    def forward(self, x: Tensor, scene_tokens: Tensor = None, masks=None, training: bool = False) -> Tensor:
        return self.forward_features(x, scene_tokens=scene_tokens, masks=masks, training=training)


class LoRA_ViT_VJEPA21(nn.Module):
    """LoRA wrapper for V-JEPA 2.1 ViT qkv layers."""

    def __init__(self, vit_model: nn.Module, r: int, lora_layer: list = None):
        super().__init__()
        for param in vit_model.parameters():
            param.requires_grad = False

        self.lora_vit = vit_model
        self.w_As = []
        self.w_Bs = []
        if r == 0:
            return

        self.lora_layer = lora_layer or list(range(len(vit_model.blocks)))
        for layer_i, blk in enumerate(vit_model.blocks):
            if layer_i not in self.lora_layer:
                continue

            qkv_linear = blk.attn.qkv
            dim = qkv_linear.in_features
            w_a_q = nn.Linear(dim, r, bias=False)
            w_b_q = nn.Linear(r, dim, bias=False)
            w_a_v = nn.Linear(dim, r, bias=False)
            w_b_v = nn.Linear(r, dim, bias=False)

            self.w_As.extend([w_a_q, w_a_v])
            self.w_Bs.extend([w_b_q, w_b_v])
            blk.attn.qkv = _LoRA_qkv_vjepa2(qkv_linear, w_a_q, w_b_q, w_a_v, w_b_v)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(len(self.w_As))}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(len(self.w_Bs))}
        save_file({**a_tensors, **b_tensors}, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")
        with safe_open(filename, framework="pt") as f:
            for i, w_A in enumerate(self.w_As):
                w_A.weight = Parameter(f.get_tensor(f"w_a_{i:03d}"))
            for i, w_B in enumerate(self.w_Bs):
                w_B.weight = Parameter(f.get_tensor(f"w_b_{i:03d}"))

    def forward(self, x: Tensor, scene_tokens: Tensor = None) -> Tensor:
        return self.lora_vit(x, scene_tokens=scene_tokens)


def _load_vjepa21_model(config):
    _add_vjepa21_repo_to_path(config)
    import app.vjepa_2_1.models.vision_transformer as vit

    vjepa2_config_path = _config_get(config, "vjepa2_config_path")
    pretrain_pt_path = _config_get(config, "model_weights")
    image_architecture = _config_get(config, "image_architecture", "vit_large")
    img_as_video_nframes = _config_get(config, "img_as_video_nframes", 2)
    resolution = tuple(_config_get(config, "vjepa2_resolution", [256, 512]))

    with open(vjepa2_config_path, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    if "model_kwargs" in params:
        model_kwargs = params["model_kwargs"]["pretrain_kwargs"]
        enc_kwargs = dict(model_kwargs["encoder"])
        wrapper_kwargs = dict(params["model_kwargs"].get("wrapper_kwargs", {}))
        img_as_video_nframes = _config_get(config, "img_as_video_nframes", wrapper_kwargs.get("img_as_video_nframes", img_as_video_nframes))
    else:
        enc_kwargs = dict(params.get("model", {}))
        data_kwargs = params.get("data", {})
        enc_kwargs.setdefault("patch_size", data_kwargs.get("patch_size", 16))
        enc_kwargs.setdefault("tubelet_size", data_kwargs.get("tubelet_size", 2))

    enc_kwargs["model_name"] = image_architecture
    enc_kwargs.setdefault("checkpoint_key", "ema_encoder")
    for key, default in (
        ("use_rope", True),
        ("img_temporal_dim_size", 1),
        ("interpolate_rope", True),
        ("modality_embedding", True),
    ):
        value = _config_get(config, key, None)
        enc_kwargs[key] = default if value is None else value
    enc_kwargs["use_activation_checkpointing"] = _config_get(
        config,
        "use_activation_checkpointing",
        enc_kwargs.get("use_activation_checkpointing", False),
    )

    enc_ckp_key = _config_get(config, "checkpoint_key", enc_kwargs.get("checkpoint_key", "ema_encoder"))
    enc_kwargs.pop("checkpoint_key", None)
    enc_kwargs.pop("model_name", None)

    log.info(f"Loading V-JEPA 2.1 weights: {pretrain_pt_path}")
    checkpoint = torch.load(pretrain_pt_path, map_location="cpu")

    model = vit.__dict__[image_architecture](
        img_size=resolution,
        num_frames=img_as_video_nframes,
        **enc_kwargs,
    )
    model.__class__ = type(
        "VisionTransformerV21WithSceneRegisters",
        (VisionTransformerV21WithSceneRegisters, model.__class__),
        {},
    )
    log.info("V-JEPA 2.1 model uses appended scene-register tokens")

    pretrained_dict = checkpoint[enc_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}

    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            log.warning(f'V-JEPA 2.1 pretrained key missing: "{k}"')
        elif pretrained_dict[k].shape != v.shape:
            log.warning(f'V-JEPA 2.1 key shape mismatch, skipped: "{k}"')
            pretrained_dict[k] = v

    msg = model.load_state_dict(pretrained_dict, strict=False)
    log.info(f"V-JEPA 2.1 weights loaded: {msg}")
    del checkpoint
    return model, img_as_video_nframes


class ImgEncoderVJEPA21(nn.Module):
    """
    V-JEPA 2.1 image encoder with frozen/lora/scene_lora modes.

    In scene_lora mode, scene tokens are appended after patch tokens and treated
    as V-JEPA 2.1 RoPE registers. The returned scene representation is the
    final S tokens projected to ``tf_d_model``.
    """

    def __init__(self, config):
        super().__init__()
        self.fpn_layer_indices = list(_config_get(config, "fpn_layer_indices", [5, 11, 17, 23]))
        self.spatial_hw = tuple(_config_get(config, "spatial_hw", [16, 32]))

        if _config_get(config, "backbone_mode", None) is not None:
            self.backbone_mode = _config_get(config, "backbone_mode")
        elif _config_get(config, "use_lora", False):
            self.backbone_mode = "scene_lora"
        elif _config_get(config, "finetune", False):
            self.backbone_mode = "lora"
        else:
            self.backbone_mode = "frozen"

        if self.backbone_mode not in ("frozen", "lora", "scene_lora"):
            raise ValueError("backbone_mode must be one of: frozen, lora, scene_lora")

        self.use_scene_tokens = self.backbone_mode == "scene_lora"
        vit_model, self.img_as_video_nframes = _load_vjepa21_model(config)

        self.num_features = vit_model.embed_dim
        self.num_prefix_tokens = config.num_scene_tokens
        self.neck = nn.Linear(self.num_features, config.tf_d_model)

        if self.backbone_mode == "frozen":
            for param in vit_model.parameters():
                param.requires_grad = False
            vit_model.eval()
            self.model = vit_model
            log.info("V-JEPA 2.1 frozen mode")
        elif self.backbone_mode == "lora":
            self.model = LoRA_ViT_VJEPA21(vit_model, r=config.lora_rank)
            log.info(f"V-JEPA 2.1 LoRA mode: rank={config.lora_rank}")
        else:
            self.model = LoRA_ViT_VJEPA21(vit_model, r=config.lora_rank)
            log.info(f"V-JEPA 2.1 scene_lora mode: rank={config.lora_rank}")

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = _config_get(config, "use_grid_mask", True)
        self.use_feature_pooling = _config_get(config, "use_feature_pooling", False)
        if self.use_feature_pooling:
            self.pool_proj = nn.Sequential(nn.AdaptiveAvgPool1d(self.num_prefix_tokens))

    def _make_img_video(self, camera_feature_1: Tensor, camera_feature_2: Tensor) -> Tensor:
        img_video = torch.stack([camera_feature_2, camera_feature_1], dim=2)

        if self.use_grid_mask and self.training:
            B, C, T, H, W = img_video.shape
            img_flat = img_video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            img_flat = self.grid_mask(img_flat)
            img_video = img_flat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return img_video

    def _base_vit(self) -> nn.Module:
        return self.model.lora_vit if hasattr(self.model, "lora_vit") else self.model

    def _encode_tokens(self, img_video: Tensor, scene_tokens: Tensor = None):
        if self.backbone_mode == "frozen":
            with torch.no_grad():
                return self.model(img_video, scene_tokens=None)
        if self.backbone_mode == "lora":
            return self.model(img_video, scene_tokens=None)
        if scene_tokens is None:
            raise ValueError("scene_lora mode requires scene_tokens")
        return self.model(img_video, scene_tokens=scene_tokens)

    def forward(
        self,
        camera_feature_1: Tensor,
        camera_feature_2: Tensor,
        scene_tokens: Tensor = None,
    ) -> Tensor:
        img_video = self._make_img_video(camera_feature_1, camera_feature_2)
        tokens = self._encode_tokens(img_video, scene_tokens=scene_tokens)

        if self.use_scene_tokens:
            S = scene_tokens.shape[1]
            scene_out = tokens[:, -S:]
        elif self.use_feature_pooling:
            scene_out = self.pool_proj(tokens.transpose(1, 2)).transpose(1, 2)
        else:
            scene_out = tokens[:, : self.num_prefix_tokens]

        return self.neck(scene_out)

    def forward_multiscale(
        self,
        camera_feature_1: Tensor,
        camera_feature_2: Tensor,
        scene_tokens: Tensor = None,
    ):
        """Return projected spatial patch features from configured V-JEPA blocks."""
        _, features = self.forward_scene_and_multiscale(
            camera_feature_1=camera_feature_1,
            camera_feature_2=camera_feature_2,
            scene_tokens=scene_tokens,
        )
        return features

    def forward_scene_and_multiscale(
        self,
        camera_feature_1: Tensor,
        camera_feature_2: Tensor,
        scene_tokens: Tensor = None,
    ):
        """Return final scene tokens and projected spatial features in one ViT pass."""
        img_video = self._make_img_video(camera_feature_1, camera_feature_2)
        vit = self._base_vit()
        old_out_layers = getattr(vit, "out_layers", None)
        vit.out_layers = self.fpn_layer_indices
        try:
            tokens_per_layer = self._encode_tokens(img_video, scene_tokens=scene_tokens)
        finally:
            vit.out_layers = old_out_layers

        if not isinstance(tokens_per_layer, (list, tuple)):
            raise RuntimeError("V-JEPA multiscale path expected a list of layer outputs")

        last_tokens = tokens_per_layer[-1]
        if self.use_scene_tokens:
            S = scene_tokens.shape[1]
            scene_out = last_tokens[:, -S:]
        elif self.use_feature_pooling:
            scene_out = self.pool_proj(last_tokens.transpose(1, 2)).transpose(1, 2)
        else:
            scene_out = last_tokens[:, : self.num_prefix_tokens]
        scene_out = self.neck(scene_out)

        spatial_h, spatial_w = self.spatial_hw
        expected_tokens = spatial_h * spatial_w
        features = []
        for tokens in tokens_per_layer:
            if self.use_scene_tokens and scene_tokens is not None:
                tokens = tokens[:, :-scene_tokens.shape[1]]
            if tokens.shape[1] != expected_tokens:
                raise ValueError(
                    f"Expected {expected_tokens} patch tokens for spatial_hw={self.spatial_hw}, "
                    f"got {tokens.shape[1]}"
                )
            tokens = self.neck(tokens)
            feat = tokens.transpose(1, 2).reshape(
                tokens.shape[0], tokens.shape[2], spatial_h, spatial_w
            )
            features.append(feat.contiguous())

        return scene_out, features
