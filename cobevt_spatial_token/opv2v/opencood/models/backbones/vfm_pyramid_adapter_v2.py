"""
vfm_pyramid_adapter_v2.py

A V-JEPA2-friendly image-feature adapter for cooperative BEV perception.

Why this version:
1) Avoids BatchNorm on frozen/low-batch VFM features. Uses GroupNorm.
2) Avoids early all-layer collapse. Builds P3/P4/P5 from shallow/middle/deep
   V-JEPA feature layers separately, with only light cross-layer residual mixing.
3) Produces ResNet-like FAX features:
      P3: [B, C3, 2H, 2W]
      P4: [B, C4,  H,  W]
      P5: [B, C5, H/2,W/2]
   where H,W are V-JEPA token-grid dimensions.

Place this file at:
    opencood/models/backbones/vfm_pyramid_adapter_v2.py
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    """Return a GroupNorm group number that divides num_channels."""
    for g in reversed(range(1, min(max_groups, num_channels) + 1)):
        if num_channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.GELU(),
        )


class DWConvGNAct(nn.Sequential):
    def __init__(self, ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(ch, ch, kernel_size=k, stride=s, padding=p, groups=ch, bias=False),
            nn.GroupNorm(_gn_groups(ch), ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(ch), ch),
            nn.GELU(),
        )


class ASPPGN(nn.Module):
    """Lightweight ASPP with GroupNorm for local geometry/context refinement."""

    def __init__(self, ch: int, rates: Sequence[int] = (1, 2, 4), out_ch: int | None = None):
        super().__init__()
        out_ch = out_ch or ch
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, ch, 3, padding=r, dilation=r, groups=ch, bias=False),
                    nn.GroupNorm(_gn_groups(ch), ch),
                    nn.GELU(),
                    nn.Conv2d(ch, ch, 1, bias=False),
                    nn.GroupNorm(_gn_groups(ch), ch),
                    nn.GELU(),
                )
                for r in rates
            ]
        )
        self.proj = ConvGNAct(ch * len(rates), out_ch, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([branch(x) for branch in self.branches], dim=1))


class UpProject(nn.Module):
    """Token grid H,W -> 2H,2W feature."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(in_ch, out_ch, k=1),
            DWConvGNAct(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SameProject(nn.Module):
    """Token grid H,W -> H,W feature."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_ch, out_ch, k=1),
            DWConvGNAct(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownProject(nn.Module):
    """Token grid H,W -> H/2,W/2 feature."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_ch, out_ch, k=3, s=2, p=1),
            DWConvGNAct(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VJEPAGeometryPyramidAdapter(nn.Module):
    """
    Convert V-JEPA2 spatial feature layers to ResNet-like FAX pyramid features.

    Input:
        feats: list of tensors [B, C_l, H_l, W_l]. They can be same-scale
               V-JEPA features or truly multi-scale features.

    Output:
        [P3, P4, P5]
        P3: [B, out_channels[0], 2H, 2W]
        P4: [B, out_channels[1],  H,  W]
        P5: [B, out_channels[2], H/2,W/2]

    Design note:
        We do not collapse all layers into a single fused map before creating
        P3/P4/P5. Instead, shallow/middle/deep layers seed P3/P4/P5 separately.
        This better matches V-JEPA/ViT features, where layers differ semantically
        but often share token resolution.
    """

    def __init__(
        self,
        in_chs: Sequence[int] = (256, 256, 256, 256),
        mid_ch: int = 256,
        out_channels: Sequence[int] = (128, 256, 512),
        aspp_rates: Sequence[int] = (1, 2, 4),
        cross_layer_residual: bool = True,
        residual_init: float = 0.10,
    ):
        super().__init__()
        if len(out_channels) != 3:
            raise ValueError(f"out_channels must contain 3 values, got {out_channels}")
        self.in_chs = tuple(in_chs)
        self.mid_ch = mid_ch
        self.out_channels = tuple(out_channels)
        self.cross_layer_residual = cross_layer_residual

        self.lateral = nn.ModuleList([ConvGNAct(c, mid_ch, k=1) for c in self.in_chs])
        self.local_refine = nn.ModuleList([DWConvGNAct(mid_ch) for _ in self.in_chs])
        self.context = nn.ModuleList([ASPPGN(mid_ch, rates=aspp_rates, out_ch=mid_ch) for _ in range(3)])

        self.p3_proj = UpProject(mid_ch, self.out_channels[0])
        self.p4_proj = SameProject(mid_ch, self.out_channels[1])
        self.p5_proj = DownProject(mid_ch, self.out_channels[2])

        if cross_layer_residual:
            self.res_scale = nn.Parameter(torch.tensor(float(residual_init)))
        else:
            self.register_parameter("res_scale", None)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _layer_indices(num_layers: int) -> Tuple[int, int, int]:
        if num_layers <= 0:
            raise ValueError("At least one V-JEPA feature layer is required.")
        if num_layers == 1:
            return 0, 0, 0
        if num_layers == 2:
            return 0, 0, 1
        return 0, num_layers // 2, num_layers - 1

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) != len(self.lateral):
            raise ValueError(
                f"Expected {len(self.lateral)} V-JEPA feature maps, got {len(feats)}. "
                "Set vfm_fpn.in_chs to match jepa_encoder.fpn_layer_indices."
            )
        for i, feat in enumerate(feats):
            if feat.dim() != 4:
                raise ValueError(f"Feature {i} must be [B,C,H,W], got {tuple(feat.shape)}")

        xs = [refine(lat(feat)) for feat, lat, refine in zip(feats, self.lateral, self.local_refine)]
        p3_i, p4_i, p5_i = self._layer_indices(len(xs))

        p3_base = xs[p3_i]
        p4_base = xs[p4_i]
        p5_base = xs[p5_i]

        if self.cross_layer_residual:
            # Light adjacent-layer residual, not early all-layer collapse.
            a = self.res_scale.tanh()
            p3_base = p3_base + a * self._resize_like(xs[p4_i], p3_base)
            p4_base = p4_base + 0.5 * a * (
                self._resize_like(xs[p3_i], p4_base) + self._resize_like(xs[p5_i], p4_base)
            )
            p5_base = p5_base + a * self._resize_like(xs[p4_i], p5_base)

        p3_base = self.context[0](p3_base)
        p4_base = self.context[1](p4_base)
        p5_base = self.context[2](p5_base)

        return [self.p3_proj(p3_base), self.p4_proj(p4_base), self.p5_proj(p5_base)]
