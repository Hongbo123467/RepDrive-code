"""
Cooperative V-JEPA BEV segmentation model.

Single-CAV branch:
    V-JEPA spatial features -> robustbev-style geometric BEV projector.

Cooperative branch:
    STTF -> SwapFusion -> decoder -> BEV segmentation.
"""

import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.sub_modules.jepa_spatial_encoder import JEPASpatialEncoder
from opencood.models.vjepa_single import VJEPABEVProjector
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.disconet_fuse import DiscoNetFusion
from opencood.models.fusion_modules.f_cooper_fuse import SpatialFusionMask
from opencood.models.fusion_modules.self_attn import AttFusion
from opencood.models.fusion_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.fusion_modules.v2v_fuse import V2VNetFusion
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import (
    get_transformation_matrix,
    warp_affine,
    get_roi_and_cav_mask,
    get_discretized_transformation_matrix,
)


class STTF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.discrete_ratio = args["resolution"]
        self.downsample_rate = args["downsample_rate"]

    def forward(self, x, spatial_correction_matrix):
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio, self.downsample_rate
        )

        x = rearrange(x, "b l c h w -> b l c w h")
        x = torch.flip(x, dims=(4,))
        batch, cavs, channels, height, width = x.shape

        transform = get_transformation_matrix(
            dist_correction_matrix.reshape(-1, 2, 3), (height, width)
        )
        cav_features = warp_affine(
            x.reshape(-1, channels, height, width), transform, (height, width)
        )
        cav_features = cav_features.reshape(batch, cavs, channels, height, width)

        cav_features = torch.flip(cav_features, dims=(4,))
        return rearrange(cav_features, "b l c w h -> b l h w c")


class CorpBEVTJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_cav = config["max_cav"]
        self.debug_shapes = config.get("debug_shapes", False)

        self.jepa_encoder = JEPASpatialEncoder(config["jepa_encoder"])
        self.freeze_jepa_encoder = config.get("freeze_jepa_encoder", False)
        if self.freeze_jepa_encoder:
            for parameter in self.jepa_encoder.parameters():
                parameter.requires_grad = False

        projector_config = config.get("projector", {})
        self.projector = VJEPABEVProjector(
            projector_config,
            in_channels=self.jepa_encoder.output_dim,
        )
        bev_channels = int(projector_config.get("out_channels", 128))

        compression_ratio = config.get("compression", 0)
        if compression_ratio > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(bev_channels, compression_ratio)
        else:
            self.compression = False

        self.downsample_rate = config["sttf"]["downsample_rate"]
        self.discrete_ratio = config["sttf"]["resolution"]
        self.use_roi_mask = config["sttf"]["use_roi_mask"]
        self.sttf = STTF(config["sttf"])

        self.fusion_method = config.get("fusion_method", "swap")
        if self.fusion_method == "swap":
            fusion_config = config.get("fax_fusion", config.get("swap_fusion"))
            self.fusion_net = SwapFusionEncoder(fusion_config)
        elif self.fusion_method == "attn":
            self.fusion_net = AttFusion(bev_channels)
        elif self.fusion_method == "disconet":
            self.fusion_net = DiscoNetFusion(config["disconet_fusion"])
        elif self.fusion_method == "v2vnet":
            self.fusion_net = V2VNetFusion(config["v2vnet_fusion"])
        elif self.fusion_method == "fcooper":
            self.fusion_net = SpatialFusionMask()
        elif self.fusion_method in ("max", "mean"):
            self.fusion_net = None
        else:
            raise ValueError(f"Unexpected fusion_method: {self.fusion_method}")

        self.fusion_debug_mode = config.get("fusion_debug_mode", "normal")
        if self.fusion_debug_mode not in (
            "normal",
            "ego_only_before_sttf",
            "ego_only_after_sttf",
        ):
            raise ValueError(f"Unexpected fusion_debug_mode: {self.fusion_debug_mode}")

        self.decoder = NaiveDecoder(config["decoder"])
        self.target = config["target"]
        self.seg_head = BevSegHead(self.target, config["seg_head_dim"], config["output_class"])

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_jepa_encoder:
            self.jepa_encoder.eval()
        return self

    def forward(self, batch_dict):
        record_len = batch_dict["record_len"]
        transformation_matrix = batch_dict["transformation_matrix"]

        inputs_all = self._flatten_temporal_inputs(batch_dict["inputs"], record_len)
        intrinsic_all = self._flatten_temporal_calibration(batch_dict.get("intrinsic"), record_len)
        extrinsic_all = self._flatten_temporal_calibration(batch_dict.get("extrinsic"), record_len)

        current_intrinsic = self._select_current_calibration(intrinsic_all)
        current_extrinsic = self._select_current_calibration(extrinsic_all)

        image_features = self.jepa_encoder(inputs_all)
        x = self.projector(image_features, current_intrinsic, current_extrinsic)
        if self.debug_shapes:
            print("[CorpBEVTJEPA] JEPA spatial:", tuple(image_features.shape))
            print("[CorpBEVTJEPA] projected BEV:", tuple(x.shape))

        if self.compression:
            x = self.naive_compressor(x)

        if self.fusion_debug_mode == "ego_only_before_sttf":
            x = self._select_ego_features(x, record_len)
            return self._decode_bev(x)

        if self.fusion_method in ("disconet", "v2vnet"):
            x = self.fusion_net(x, record_len, batch_dict["pairwise_t_matrix"], None)
            x = rearrange(x, "b h w c -> b c h w")
            return self._decode_bev(x)

        x, mask = regroup(x, record_len, self.max_cav)  # [B,L,C,H,W]
        x = self.sttf(x, transformation_matrix)          # [B,L,H,W,C]

        if self.fusion_debug_mode == "ego_only_after_sttf":
            x = rearrange(x[:, 0], "b h w c -> b c h w")
            return self._decode_bev(x)

        if self.use_roi_mask:
            com_mask = get_roi_and_cav_mask(
                x.shape, mask, transformation_matrix, self.discrete_ratio, self.downsample_rate
            )
        else:
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        x = rearrange(x, "b l h w c -> b l c h w")
        x = self._fuse_aligned_features(x, com_mask, record_len)
        return self._decode_bev(x)

    def _fuse_aligned_features(self, x, com_mask, record_len):
        if self.fusion_method == "swap":
            return self.fusion_net(x, com_mask)

        if self.fusion_method == "attn":
            valid_features = []
            for batch_idx, length in enumerate(record_len.detach().cpu().tolist()):
                valid_features.append(x[batch_idx, :int(length)])
            return self.fusion_net(torch.cat(valid_features, dim=0), record_len)

        if self.fusion_method == "fcooper":
            return rearrange(self.fusion_net(rearrange(x, "b l c h w -> b l h w c")),
                             "b h w c -> b c h w")

        agent_mask = rearrange(com_mask, "b h w e l -> b l e h w").to(dtype=x.dtype)
        if self.fusion_method == "mean":
            denominator = agent_mask.sum(dim=1).clamp_min(1.0)
            return (x * agent_mask).sum(dim=1) / denominator

        if self.fusion_method == "max":
            masked_x = x.masked_fill(agent_mask == 0, torch.finfo(x.dtype).min)
            return masked_x.max(dim=1).values

        raise ValueError(f"Fusion method {self.fusion_method} does not use aligned features")

    def _decode_bev(self, x):
        x = x.unsqueeze(1)
        x = self.decoder(x)
        x = rearrange(x, "b l c h w -> (b l) c h w")
        batch_size = x.shape[0]
        return self.seg_head(x, batch_size, 1)

    @staticmethod
    def _select_ego_features(x, record_len):
        ego_features = []
        start = 0
        for length in record_len.detach().cpu().tolist():
            ego_features.append(x[start])
            start += int(length)
        return torch.stack(ego_features, dim=0)

    @staticmethod
    def _flatten_temporal_inputs(inputs, record_len):
        num_agents = int(record_len.sum().item())
        if inputs.dim() == 5:
            if inputs.shape[0] != num_agents:
                raise ValueError(f"Expected first dim=sum(record_len)={num_agents}, got {inputs.shape[0]}")
            return inputs
        if inputs.dim() == 6:
            if inputs.shape[0] == num_agents:
                return inputs
            return inputs.reshape(-1, *inputs.shape[2:])
        if inputs.dim() == 7:
            return inputs.reshape(-1, *inputs.shape[2:])
        raise ValueError(
            "Expected inputs [N,M,H,W,C], [N,T,M,H,W,C], [B,L,M,H,W,C], or [B,L,T,M,H,W,C], "
            f"got {tuple(inputs.shape)}"
        )

    @staticmethod
    def _flatten_temporal_calibration(calibration, record_len):
        if calibration is None:
            return None
        num_agents = int(record_len.sum().item())
        if calibration.dim() in (4, 5):
            if calibration.shape[0] == num_agents:
                return calibration
            if calibration.dim() == 5:
                return calibration.reshape(-1, *calibration.shape[2:])
            return calibration
        if calibration.dim() == 6:
            return calibration.reshape(-1, *calibration.shape[2:])
        raise ValueError(f"Unexpected calibration shape: {tuple(calibration.shape)}")

    @staticmethod
    def _select_current_calibration(calibration):
        if calibration is None:
            raise ValueError("V-JEPA BEV projector requires camera calibration")
        if calibration.dim() == 4:
            return calibration
        if calibration.dim() == 5:
            return calibration[:, 0]
        raise ValueError(f"Expected [N,M,*,*] or [N,T,M,*,*], got {tuple(calibration.shape)}")
