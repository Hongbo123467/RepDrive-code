# V-JEPA2 LoRA 图像编码器
# 图像输入格式与 Drive-JEPA 对齐：双帧前视摄像头 [B, C, H, W] -> [B, C, 2, H, W] (视频格式)
# 参考：Drive-JEPA/navsim_v1/.../bevformer/image_encoder.py
#
# scene_tokens 实现策略（与 DINOv2 的 timm_ViT._pos_embed 对称）：
#   - 在 patch_embed + pos_embed 之后、attention blocks 之前，cat 到序列头部
#   - 通过子类化 VisionTransformer 实现，与 DINOv2 的 monkey-patch (__class__) 手法对应
#   - 取 forward 输出的前 num_scene_tokens 个 token 作为场景特征

import math
import logging
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file

from .grid_mask import GridMask

OPV2V_ROOT = Path(__file__).resolve().parents[3]
if str(OPV2V_ROOT) not in sys.path:
    sys.path.insert(0, str(OPV2V_ROOT))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA QKV 层（适配 V-JEPA2 的 nn.Linear(dim, 3*dim) qkv 结构）
# ---------------------------------------------------------------------------

class _LoRA_qkv_vjepa2(nn.Module):
    """
    V-JEPA2 的 Block.attn.qkv 结构为：
        nn.Linear(dim, dim * 3)
    forward 中通过 .unflatten(-1, (3, num_heads, -1)).permute(...) 拆分 q/k/v

    此类在原始 qkv 输出上叠加 LoRA 的低秩矩阵分支（仅对 Q 和 V）。
    与 DINOv2 的 _LoRA_qkv_timm 结构完全对称。
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        """
        @param {nn.Module} qkv - 原始的 qkv 线性层
        @param {nn.Module} linear_a_q - Q 分支的 LoRA 下投影 (dim -> r)
        @param {nn.Module} linear_b_q - Q 分支的 LoRA 上投影 (r -> dim)
        @param {nn.Module} linear_a_v - V 分支的 LoRA 下投影 (dim -> r)
        @param {nn.Module} linear_b_v - V 分支的 LoRA 上投影 (r -> dim)
        """
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x: Tensor) -> Tensor:
        """
        @param {Tensor} x - 输入特征，形状 [B, N, dim]
        @returns {Tensor} - qkv 输出，形状 [B, N, 3*dim]
        """
        qkv = self.qkv(x)                            # [B, N, 3*dim]
        new_q = self.linear_b_q(self.linear_a_q(x))  # [B, N, dim]
        new_v = self.linear_b_v(self.linear_a_v(x))  # [B, N, dim]
        # Q 位于前 dim 维，V 位于最后 dim 维
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


# ---------------------------------------------------------------------------
# LoRA 包装器（将 LoRA 注入 V-JEPA2 ViT 的各 Block）
# ---------------------------------------------------------------------------

class LoRA_ViT_VJEPA2(nn.Module):
    """
    对 V-JEPA2 VisionTransformerWithSceneTokens 的 LoRA 包装器。
    冻结原始参数，仅在每个 Block 的 attn.qkv 上插入低秩适配矩阵（Q 和 V 方向）。
    forward 透传 scene_tokens 参数给底层 ViT。
    """

    def __init__(self, vit_model: nn.Module, r: int, lora_layer: list = None):
        """
        @param {nn.Module} vit_model - 已加载预训练权重的 VisionTransformerWithSceneTokens
        @param {int} r - LoRA 秩，r=0 时退化为纯冻结模型
        @param {list} lora_layer - 哪些 block 层启用 LoRA，默认全部启用
        """
        super().__init__()

        if r == 0:
            # 纯冻结，不注入 LoRA
            for param in vit_model.parameters():
                param.requires_grad = False
            self.lora_vit = vit_model
            self.w_As = []
            self.w_Bs = []
            return

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        self.w_As = []  # LoRA A 矩阵（下投影）列表
        self.w_Bs = []  # LoRA B 矩阵（上投影）列表

        # 冻结所有原始参数
        for param in vit_model.parameters():
            param.requires_grad = False

        # 遍历每个 Block，注入 LoRA
        for t_layer_i, blk in enumerate(vit_model.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features

            # Q 分支
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            # V 分支
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            blk.attn.qkv = _LoRA_qkv_vjepa2(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()
        self.lora_vit = vit_model

    def reset_parameters(self) -> None:
        """初始化 LoRA 参数：A 使用 kaiming_uniform，B 初始化为零（保证初始无变化）"""
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def save_lora_parameters(self, filename: str) -> None:
        """
        仅保存 LoRA 参数（A/B 矩阵），以 safetensors 格式存储。
        @param {str} filename - 保存路径，必须以 .safetensors 结尾
        """
        assert filename.endswith(".safetensors")
        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        save_file({**a_tensors, **b_tensors}, filename)

    def load_lora_parameters(self, filename: str) -> None:
        """
        从 safetensors 文件加载 LoRA 参数。
        @param {str} filename - 加载路径，必须以 .safetensors 结尾
        """
        assert filename.endswith(".safetensors")
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                w_A_linear.weight = Parameter(f.get_tensor(f"w_a_{i:03d}"))
            for i, w_B_linear in enumerate(self.w_Bs):
                w_B_linear.weight = Parameter(f.get_tensor(f"w_b_{i:03d}"))

    def forward(self, x: Tensor, scene_tokens: Tensor = None) -> Tensor:
        """
        调用 VisionTransformerWithSceneTokens 的前向，注入 scene_tokens。
        @param {Tensor} x - 视频张量，形状 [B, C, T, H, W]
        @param {Tensor} scene_tokens - 场景查询向量，形状 [B, S, embed_dim]，可为 None
        @returns {Tensor} - token 特征，形状 [B, S+N_patches, embed_dim]
        """
        return self.lora_vit(x, scene_tokens=scene_tokens)


# ---------------------------------------------------------------------------
# 主编码器：ImgEncoderVJEPA2
# 接口与 dinov2_lora.py 中的 ImgEncoder 完全一致
# ---------------------------------------------------------------------------

def _load_vjepa2_model(config):
    """
    根据 config 加载并初始化 V-JEPA2 ViT 模型。
    使用 VisionTransformerWithSceneTokens 子类（通过 __class__ 替换，
    与 DINOv2 的 `model.__class__ = timm_ViT` 策略完全对称）。

    @param {dict} config - 包含 vjepa2_config_path, model_weights 等字段的配置
    @returns {Tuple[nn.Module, int]} - (模型, img_as_video_nframes)
    """
    import vjepa2.src.models.vision_transformer as vit
    from vjepa2.src.models.vision_transformer import VisionTransformerWithSceneTokens

    vjepa2_config_path = config.vjepa2_config_path
    pretrain_pt_path = config.model_weights
    image_architecture = config.get("image_architecture", "vit_large")
    img_as_video_nframes = config.get("img_as_video_nframes", 2)

    with open(vjepa2_config_path, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    resolution = tuple(config.get("vjepa2_resolution", [256, 512]))  # (H, W)
    model_kwargs = params["model_kwargs"]
    wrapper_kwargs = model_kwargs["wrapper_kwargs"]
    wrapper_kwargs["img_as_video_nframes"] = img_as_video_nframes
    model_kwargs = model_kwargs["pretrain_kwargs"]
    model_kwargs["encoder"]["model_name"] = image_architecture

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key", "target_encoder")

    log.info(f"正在加载 V-JEPA2 预训练权重: {pretrain_pt_path}")
    checkpoint = torch.load(pretrain_pt_path, map_location="cpu")

    # 使用原始工厂函数创建模型（确保结构兼容权重）
    model = vit.__dict__[image_architecture](
        input_size=resolution,
        num_frames=img_as_video_nframes,
        **enc_kwargs,
    )

    # ---- [核心] monkey-patch：替换为支持 scene_tokens 的子类 ----
    # 与 DINOv2: `self.model.__class__ = timm_ViT` 策略完全对称
    model.__class__ = VisionTransformerWithSceneTokens
    log.info("V-JEPA2 模型已升级为 VisionTransformerWithSceneTokens")

    # 加载预训练权重
    pretrained_dict = checkpoint[enc_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}

    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            log.warning(f'预训练权重中缺少 key: "{k}"')
        elif pretrained_dict[k].shape != v.shape:
            log.warning(f'key "{k}" 形状不匹配，跳过')
            pretrained_dict[k] = v

    msg = model.load_state_dict(pretrained_dict, strict=False)
    log.info(f"V-JEPA2 权重加载完成: {msg}")
    del checkpoint

    return model, img_as_video_nframes


class ImgEncoderVJEPA2(nn.Module):
    """
    基于 V-JEPA2 ViT 的图像编码器，支持三种骨干训练模式。

    backbone_mode 选项：
      - ``frozen``     : 冻结全部 V-JEPA2 参数，不注入 scene_tokens，
                         直接截取前 num_scene_tokens 个 patch token 投影输出。
      - ``lora``       : 冻结主干，仅训练 LoRA 低秩矩阵（Q/V），
                         不注入 scene_tokens。
      - ``scene_lora`` : 冻结主干，训练 LoRA 矩阵 + 外部传入的 scene_embeds，
                         将 scene_tokens cat 到 patch 序列头部参与 attention，
                         取对应输出作为场景特征（与原始行为完全一致）。

    输入（在 forward 中）：
      - camera_feature_1：上一帧  [B, C, H, W]，值域 [0,1]（ToTensor 输出）
      - camera_feature_2：当前帧  [B, C, H, W]，值域 [0,1]
      - scene_tokens：可学习查询向量 [B, S, embed_dim]，
                      frozen/lora 模式下传入 None 即可
    输出：[B, num_scene_tokens, tf_d_model]
    """

    def __init__(self, config):
        """
        @param {dict|OmegaConf} config - 图像骨干网络配置，字段说明见 drivejepa.yaml
        """
        super().__init__()

        # 加载 V-JEPA2 模型（已 monkey-patch 为 VisionTransformerWithSceneTokens）
        vit_model, self.img_as_video_nframes = _load_vjepa2_model(config)

        # V-JEPA2 ViT-L 的 embed_dim = 1024
        self.num_features = vit_model.embed_dim
        self.num_prefix_tokens = config.num_scene_tokens

        # 投影层：将 embed_dim 维特征映射到模型统一维度（三种模式共用）
        self.neck = nn.Linear(self.num_features, config.tf_d_model)

        # --- 骨干模式：读取 backbone_mode，向后兼容旧配置 ---
        # 若配置中无 backbone_mode，则根据旧字段 use_lora/finetune 自动推导
        if hasattr(config, 'backbone_mode') and config.backbone_mode is not None:
            self.backbone_mode = config.backbone_mode
        elif config.get('use_lora', False):
            self.backbone_mode = 'scene_lora'   # 旧配置 use_lora=True → scene_lora
        elif config.get('finetune', False):
            self.backbone_mode = 'lora'          # 旧配置 finetune=True → 视为 lora
        else:
            self.backbone_mode = 'frozen'

        if self.backbone_mode not in ('frozen', 'lora', 'scene_lora'):
            raise ValueError(
                f"未知的 backbone_mode='{self.backbone_mode}'，"
                "可选值：'frozen' | 'lora' | 'scene_lora'"
            )

        # scene_tokens 仅在 scene_lora 模式下启用
        self.use_scene_tokens = (self.backbone_mode == 'scene_lora')

        # --- 按模式初始化骨干 ---
        if self.backbone_mode == 'frozen':
            # Mode A: 全部冻结，eval 模式，不注入 scene_tokens
            for param in vit_model.parameters():
                param.requires_grad = False
            vit_model.eval()
            self.model = vit_model
            log.info("V-JEPA2 冻结模式（frozen）：全部参数冻结，不注入 scene_tokens")

        elif self.backbone_mode == 'lora':
            # Mode B: LoRA 微调，不注入 scene_tokens
            self.model = LoRA_ViT_VJEPA2(vit_model, r=config.lora_rank)
            log.info(f"V-JEPA2 LoRA 微调模式（lora）：rank={config.lora_rank}，不注入 scene_tokens")

        else:  # scene_lora
            # Mode C: LoRA 微调 + scene_tokens 注入
            self.model = LoRA_ViT_VJEPA2(vit_model, r=config.lora_rank)
            log.info(f"V-JEPA2 scene_tokens LoRA 模式（scene_lora）：rank={config.lora_rank}，启用 scene_tokens")

        # GridMask 数据增强
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = config.get("use_grid_mask", True)

        # Feature pooling（可选）
        self.use_feature_pooling = config.get("use_feature_pooling", False)
        if self.use_feature_pooling:
            self.pool_proj = nn.Sequential(
                nn.AdaptiveAvgPool1d(self.num_prefix_tokens)
            )

        # compress_fc（可选）
        self.focus_front_cam = config.get("focus_front_cam", False)
        self.compress_fc = config.get("compress_fc", False)
        if self.compress_fc:
            # 根据 V-JEPA2 分辨率(256,512) patch=16 nframes=2 tubelet=2 计算：
            # T_tokens=1, H_tokens=16, W_tokens=32 → N_tokens=512
            vjepa2_n_tokens = config.get("vjepa2_n_tokens", 512)
            self.compress_fc_layer = nn.Linear(vjepa2_n_tokens, self.num_prefix_tokens)

    def forward(
        self,
        camera_feature_1: Tensor,
        camera_feature_2: Tensor,
        scene_tokens: Tensor = None,
    ) -> Tensor:
        """
        编码双帧前视摄像头图像，按 backbone_mode 决定是否注入 scene_tokens。

        - frozen / lora 模式：scene_tokens 传入 None，骨干输出后截取前 num_prefix_tokens 个 patch token。
        - scene_lora 模式  ：scene_tokens 注入到 attention 前缀，取对应输出（与原始行为完全一致）。

        @param {Tensor} camera_feature_1 - 上一帧，形状 [B, C, H, W]，值域 [0,1]
        @param {Tensor} camera_feature_2 - 当前帧，形状 [B, C, H, W]，值域 [0,1]
        @param {Tensor} scene_tokens - 可学习场景查询向量，形状 [B, S, embed_dim]；
                                       frozen/lora 模式下传入 None
        @returns {Tensor} - 场景特征，形状 [B, num_prefix_tokens, tf_d_model]
        """
        # 拼接为视频格式：[B, C, T=2, H, W]
        # 当前帧在前（t=0），上一帧在后（t=1）
        img_video = torch.stack([camera_feature_2, camera_feature_1], dim=2)  # [B, C, 2, H, W]

        # GridMask 增强（训练阶段，对每帧分别施加）
        if self.use_grid_mask and self.training:
            B_, C_, T_, H_, W_ = img_video.shape
            img_flat = img_video.permute(0, 2, 1, 3, 4).reshape(B_ * T_, C_, H_, W_)
            img_flat = self.grid_mask(img_flat)
            img_video = img_flat.reshape(B_, T_, C_, H_, W_).permute(0, 2, 1, 3, 4)

        # --- 骨干前向（按 backbone_mode 分三路）---
        if self.backbone_mode == 'frozen':
            # Mode A: 全程 no_grad，不注入 scene_tokens
            with torch.no_grad():
                tokens = self.model(img_video, scene_tokens=None)
        elif self.backbone_mode == 'lora':
            # Mode B: LoRA 可训练，不注入 scene_tokens
            tokens = self.model(img_video, scene_tokens=None)
        else:  # scene_lora
            # Mode C: LoRA 可训练 + 注入 scene_tokens 前缀
            tokens = self.model(img_video, scene_tokens=scene_tokens)

        # --- 输出 token 提取 ---
        if self.use_scene_tokens and scene_tokens is not None:
            # scene_lora 模式：取前 S 个 token（scene_tokens 的 attention 输出）
            S = scene_tokens.shape[1]
            scene_out = tokens[:, :S]                               # [B, S, embed_dim]
        elif self.use_feature_pooling:
            # 特征池化（frozen/lora 模式的可选后处理）
            scene_out = self.pool_proj(tokens.transpose(1, 2)).transpose(1, 2)  # [B, S, D]
        else:
            # frozen/lora 模式：截取前 num_prefix_tokens 个 patch token
            scene_out = tokens[:, :self.num_prefix_tokens]          # [B, num_prefix_tokens, D]

        # 投影到模型统一维度
        scene_out = self.neck(scene_out)   # [B, num_prefix_tokens, tf_d_model]

        return scene_out
