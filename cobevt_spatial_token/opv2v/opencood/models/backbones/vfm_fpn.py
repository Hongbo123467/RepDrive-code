import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1x1BNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

class DWConvBNAct(nn.Sequential):
    def __init__(self, ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(ch, ch, k, s, p, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU()
        )

class ASPP(nn.Module):
    def __init__(self, ch, rates=(1, 2, 4, 8), out_ch=None):
        super().__init__()
        out_ch = out_ch or ch
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=r, dilation=r, groups=ch, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
            ) for r in rates
        ])
        self.proj = Conv1x1BNAct(ch * len(rates), out_ch)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.proj(x)

class DepthAttnFuse(nn.Module):
    """对同分辨率的多层特征做【空间可变】深度注意力融合"""
    def __init__(self, in_chs, mid_ch=256):
        super().__init__()
        self.proj = nn.ModuleList([Conv1x1BNAct(c, mid_ch) for c in in_chs])#改变通道数，不改变分辨率,都变为256
        self.score = nn.ModuleList([nn.Conv2d(mid_ch, 1, 1) for _ in in_chs])
        self.post = DWConvBNAct(mid_ch)

    def forward(self, feats):
        # feats: list of [F1, F2, F3, F4], each (B,Ci,H,W), 同分辨率(1/16)
        xs = [p(f) for p, f in zip(self.proj, feats)]      # -> (B,mid,H,W)
        S = torch.stack([torch.sigmoid(h(x)) for h, x in zip(self.score, xs)], dim=1)  # (B,4,1,H,W)
        A = torch.softmax(S, dim=1)                        # 归一化到层维度
        X = torch.stack(xs, dim=1)                         # (B,4,mid,H,W)
        fused = (A * X).sum(dim=1)                         # (B,mid,H,W)
        return self.post(fused)                            # 平滑一下

class SSBiFPNDecoder(nn.Module):
    def __init__(self, in_chs=(384, 384, 768, 1024), mid_ch=256, out_ch=1):
        super().__init__()
        self.fuse = DepthAttnFuse(in_chs, mid_ch)          # 同尺度多层融合
        self.context = ASPP(mid_ch, rates=(1,2,4,8), out_ch=mid_ch)  # 感受野金字塔

        # 细节回注（1/4 分支，可换成 |A-B|、梯度等作为输入）
        self.detail_in = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),   # 1/2
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),   # 1/4
        )
        self.detail_gate = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 1), nn.Sigmoid())

        # 上采样头：1/16 -> 1/8 -> 1/4 -> 1/1
        up = (lambda: nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.up1 = up(); self.up2 = up(); self.up3 = up()
        self.refine1 = DWConvBNAct(mid_ch)
        self.refine2 = DWConvBNAct(mid_ch)
        self.refine3 = DWConvBNAct(mid_ch)

        self.head = nn.Sequential(
            DWConvBNAct(mid_ch),
            nn.Conv2d(mid_ch, out_ch, 1)
        )

    def forward(self, feats, img=None):
        """
        feats: [F1,F2,F3,F4]  # 全是 1/16 的特征图
        img: 原图 (B,3,H,W) （可选；变化检测可传 concat(A,B,|A-B|) 的卷积结果）
        """
        x = self.fuse(feats)           # (B,mid,H/16,W/16) 4层变为1层，也可以直接一层
        x = self.context(x)            # 提升全局/多尺度上下文

        x = self.up1(x); x = self.refine1(x)   # -> 1/8
        x = self.up2(x); x = self.refine2(x)   # -> 1/4

        if img is not None:
            d = self.detail_in(img)                         # 1/4
            g = self.detail_gate(F.adaptive_avg_pool2d(x, 1))
            x = x + g * d                                   # 门控细节回注

        x = self.refine3(x)           # 1/4
        x = self.up3(x)               # 1/2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 1/1
        return self.head(x)


class UpChangeBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            DWConvBNAct(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class DownChangeBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            DWConvBNAct(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class SSBiFPNPyramidAdapter(nn.Module):
    """
    Adapt same-scale V-JEPA hierarchical features into ResNet-like FAX features.

    Input:
        four tensors shaped [B, 256, 32, 32]
    Output:
        [B, 128, 64, 64], [B, 256, 32, 32], [B, 512, 16, 16]
    """

    def __init__(
        self,
        in_chs=(256, 256, 256, 256),
        mid_ch=256,
        out_channels=(128, 256, 512),
    ):
        super().__init__()
        p3_ch, p4_ch, p5_ch = out_channels
        self.fuse = DepthAttnFuse(in_chs, mid_ch)
        self.context = ASPP(mid_ch, rates=(1, 2, 4, 8), out_ch=mid_ch)
        self.p3_block = UpChangeBlock(mid_ch, p3_ch)
        self.p4_block = nn.Sequential(
            Conv1x1BNAct(mid_ch, p4_ch),
            DWConvBNAct(p4_ch),
        )
        self.p5_block = DownChangeBlock(mid_ch, p5_ch)

    def forward(self, feats):
        if len(feats) != len(self.fuse.proj):
            raise ValueError(
                f"Expected {len(self.fuse.proj)} V-JEPA feature maps, got {len(feats)}"
            )
        x = self.fuse(feats)
        x = self.context(x)
        p3 = self.p3_block(x)
        p4 = self.p4_block(x)
        p5 = self.p5_block(x)
        return [p3, p4, p5]
