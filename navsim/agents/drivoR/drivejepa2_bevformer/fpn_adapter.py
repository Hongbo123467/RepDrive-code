import torch
import torch.nn as nn


class Conv1x1BNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class DWConvBNAct(nn.Sequential):
    def __init__(self, ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(ch, ch, k, s, p, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )


class ASPP(nn.Module):
    def __init__(self, ch, rates=(1, 2, 4, 8), out_ch=None):
        super().__init__()
        out_ch = out_ch or ch
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, ch, 3, padding=r, dilation=r, groups=ch, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.GELU(),
                    nn.Conv2d(ch, ch, 1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.GELU(),
                )
                for r in rates
            ]
        )
        self.proj = Conv1x1BNAct(ch * len(rates), out_ch)

    def forward(self, x):
        return self.proj(torch.cat([branch(x) for branch in self.branches], dim=1))


class DepthAttnFuse(nn.Module):
    def __init__(self, in_chs, mid_ch=256):
        super().__init__()
        self.proj = nn.ModuleList([Conv1x1BNAct(c, mid_ch) for c in in_chs])
        self.score = nn.ModuleList([nn.Conv2d(mid_ch, 1, 1) for _ in in_chs])
        self.post = DWConvBNAct(mid_ch)

    def forward(self, feats):
        if len(feats) != len(self.proj):
            raise ValueError(f"Expected {len(self.proj)} feature maps, got {len(feats)}")
        xs = [proj(feat) for proj, feat in zip(self.proj, feats)]
        scores = torch.stack(
            [torch.sigmoid(head(x)) for head, x in zip(self.score, xs)], dim=1
        )
        weights = torch.softmax(scores, dim=1)
        fused = (weights * torch.stack(xs, dim=1)).sum(dim=1)
        return self.post(fused)


class DriveJEPA2FPNAdapter(nn.Module):
    """Fuse four V-JEPA block features into one BEVFormer image level."""

    def __init__(self, in_chs=(256, 256, 256, 256), mid_ch=256, out_ch=256):
        super().__init__()
        self.fuse = DepthAttnFuse(in_chs, mid_ch)
        self.context = ASPP(mid_ch, rates=(1, 2, 4, 8), out_ch=out_ch)

    def forward(self, feats):
        return self.context(self.fuse(feats))


class DriveJEPA2BEVImageAdapter(nn.Module):
    """Format fused image features as Drive-JEPA BEVFormer image_feature tuple."""

    def __init__(self, embed_dims=256, num_cams=1, num_levels=1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.level_embeds = nn.Parameter(torch.randn(num_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.randn(num_cams, embed_dims))

    def forward(self, img_feat, img_metas):
        if img_feat.dim() != 4:
            raise ValueError(f"img_feat must be [B, C, H, W], got {tuple(img_feat.shape)}")
        batch_size, channels, height, width = img_feat.shape
        if channels != self.embed_dims:
            raise ValueError(f"Expected {self.embed_dims} channels, got {channels}")

        feat = img_feat[:, None]  # [B, 1, C, H, W]
        _, num_cam, _, _, _ = feat.shape
        if num_cam != self.num_cams:
            raise ValueError(f"Expected {self.num_cams} cameras, got {num_cam}")

        feat = feat.flatten(3).permute(1, 0, 3, 2)  # [num_cam, B, HW, C]
        feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
        feat = feat + self.level_embeds[None, None, 0:1, :].to(feat.dtype)
        feat = feat.permute(0, 2, 1, 3).contiguous()  # [num_cam, HW, B, C]

        spatial_shapes = torch.as_tensor(
            [[height, width]], dtype=torch.long, device=img_feat.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        return feat, spatial_shapes, level_start_index, {"img_metas": img_metas}

