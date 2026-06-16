"""
vfm_pyramid_adapter_best.py

V-JEPA2/2.1-friendly geometry pyramid adapter for camera-to-BEV FAX modules.

Place at:
    opencood/models/backbones/vfm_pyramid_adapter_best.py

Purpose:
    Convert same-resolution V-JEPA spatial feature layers, e.g.
        [B, C, 32, 32] x 4
    into FAX-compatible image feature pyramid:
        P3: [B, 128, 64, 64]
        P4: [B, 256, 32, 32]
        P5: [B, 512, 16, 16]

Why this version is usually stronger than a naive pseudo-FPN:
    1. It does not collapse all V-JEPA layers before creating the pyramid.
    2. It uses layer-specific shallow/mid/deep sources for P3/P4/P5.
    3. It uses GroupNorm instead of BatchNorm, which is more stable for frozen/LoRA VFM features.
    4. It applies only light cross-layer residual mixing to preserve layer roles.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(channels: int, max_groups: int = 32) -> int:
    for g in reversed(range(1, min(max_groups, channels) + 1)):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.GELU(),
        )


class DWConvGNAct(nn.Sequential):
    def __init__(self, ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(ch, ch, k, s, p, groups=ch, bias=False),
            nn.GroupNorm(_gn_groups(ch), ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.GroupNorm(_gn_groups(ch), ch),
            nn.GELU(),
        )


class ResidualGNBlock(nn.Module):
    def __init__(self, ch: int, expansion: int = 2):
        super().__init__()
        hidden = ch * expansion
        self.net = nn.Sequential(
            ConvGNAct(ch, hidden, k=1),
            ConvGNAct(hidden, hidden, k=3, p=1),
            nn.Conv2d(hidden, ch, 1, bias=False),
            nn.GroupNorm(_gn_groups(ch), ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ASPPGN(nn.Module):
    def __init__(self, ch: int, rates: Sequence[int] = (1, 2, 4), out_ch: int | None = None):
        super().__init__()
        out_ch = out_ch or ch
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, 3, padding=r, dilation=r, groups=ch, bias=False),
                    nn.GroupNorm(_gn_groups(ch), ch),
                    nn.GELU(),
                    nn.Conv2d(ch, ch, 1, bias=False),
                    nn.GroupNorm(_gn_groups(ch), ch),
                    nn.GELU(),
                )
            )
        self.proj = ConvGNAct(ch * len(rates), out_ch, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))


class UpProject(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(in_ch, out_ch, k=1),
            DWConvGNAct(out_ch),
            ResidualGNBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SameProject(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, out_ch, k=1),
            DWConvGNAct(out_ch),
            ResidualGNBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownProject(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, out_ch, k=3, s=2, p=1),
            DWConvGNAct(out_ch),
            ResidualGNBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VJEPAGeometryPyramidAdapterBest(nn.Module):
    def __init__(
        self,
        in_chs: Sequence[int] = (256, 256, 256, 256),
        mid_ch: int = 256,
        out_channels: Sequence[int] = (128, 256, 512),
        aspp_rates: Sequence[int] = (1, 2, 4),
        cross_layer_residual: bool = True,
        residual_init: float = 0.10,
        use_context: bool = True,
    ):
        super().__init__()
        if len(out_channels) != 3:
            raise ValueError("out_channels must contain three values for P3/P4/P5")
        self.in_chs = tuple(in_chs)
        self.mid_ch = int(mid_ch)
        self.out_channels = tuple(out_channels)
        self.cross_layer_residual = bool(cross_layer_residual)
        self.use_context = bool(use_context)

        self.lateral = nn.ModuleList([ConvGNAct(c, mid_ch, k=1) for c in self.in_chs])
        self.refine = nn.ModuleList([nn.Sequential(DWConvGNAct(mid_ch), ResidualGNBlock(mid_ch)) for _ in self.in_chs])

        if use_context:
            self.context = nn.ModuleList([ASPPGN(mid_ch, rates=aspp_rates, out_ch=mid_ch) for _ in range(3)])
        else:
            self.context = nn.ModuleList([nn.Identity() for _ in range(3)])

        self.p3_proj = UpProject(mid_ch, self.out_channels[0])
        self.p4_proj = SameProject(mid_ch, self.out_channels[1])
        self.p5_proj = DownProject(mid_ch, self.out_channels[2])

        if self.cross_layer_residual:
            self.res_scale = nn.Parameter(torch.tensor(float(residual_init)))
        else:
            self.register_parameter("res_scale", None)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _pick_indices(num_layers: int) -> Tuple[int, int, int]:
        if num_layers <= 0:
            raise ValueError("At least one feature layer is required")
        if num_layers == 1:
            return 0, 0, 0
        if num_layers == 2:
            return 0, 0, 1
        return 0, num_layers // 2, num_layers - 1

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) != len(self.lateral):
            raise ValueError(
                f"Expected {len(self.lateral)} V-JEPA feature maps, got {len(feats)}. "
                "Check jepa_encoder.fpn_layer_indices and vfm_fpn.in_chs."
            )
        xs = []
        for i, (feat, lat, ref) in enumerate(zip(feats, self.lateral, self.refine)):
            if feat.dim() != 4:
                raise ValueError(f"Feature {i} should be [B,C,H,W], got {tuple(feat.shape)}")
            xs.append(ref(lat(feat)))

        p3_i, p4_i, p5_i = self._pick_indices(len(xs))
        p3 = xs[p3_i]
        p4 = xs[p4_i]
        p5 = xs[p5_i]

        if self.cross_layer_residual:
            a = self.res_scale.tanh()
            p3 = p3 + a * self._resize_like(xs[p4_i], p3)
            p4 = p4 + 0.5 * a * (self._resize_like(xs[p3_i], p4) + self._resize_like(xs[p5_i], p4))
            p5 = p5 + a * self._resize_like(xs[p4_i], p5)

        p3 = self.context[0](p3)
        p4 = self.context[1](p4)
        p5 = self.context[2](p5)
        return [self.p3_proj(p3), self.p4_proj(p4), self.p5_proj(p5)]


# Backward-friendly alias
VJEPAGeometryPyramidAdapter = VJEPAGeometryPyramidAdapterBest
