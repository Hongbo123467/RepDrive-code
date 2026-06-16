"""Cooperative V-JEPA 2.1 + CrossViewModule BEV segmentation model."""

import copy

import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.fusion_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.cvt_modules import CrossViewModule
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.jepa_spatial_encoder import JEPASpatialEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.torch_transformation_utils import (
    get_discretized_transformation_matrix,
    get_roi_and_cav_mask,
    get_transformation_matrix,
    warp_affine,
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


class VjepaCvtFuse(nn.Module):
    """V-JEPA spatial features -> CVM -> STTF -> SwapFusion -> BEV head."""

    def __init__(self, config):
        super().__init__()
        self.max_cav = config["max_cav"]
        self.debug_shapes = config.get("debug_shapes", False)

        self.jepa_encoder = JEPASpatialEncoder(config["jepa_encoder"])

        num_cams = int(config["jepa_encoder"].get("num_cams", 4))
        feat_h, feat_w = self.jepa_encoder.spatial_hw
        cvm_config = copy.deepcopy(config["cvm"])
        cvm_config["backbone_output_shape"] = [
            (1, 1, num_cams, self.jepa_encoder.output_dim, feat_h, feat_w)
        ]
        if len(cvm_config["middle"]) != 1:
            raise ValueError("VjepaCvtFuse currently expects one CVM feature level")
        self.cvm = CrossViewModule(cvm_config)

        self.downsample_rate = config["sttf"]["downsample_rate"]
        self.discrete_ratio = config["sttf"]["resolution"]
        self.use_roi_mask = config["sttf"]["use_roi_mask"]
        self.sttf = STTF(config["sttf"])
        self.fusion_net = SwapFusionEncoder(config["swap_fusion"])

        self.decoder = NaiveDecoder(config["decoder"])
        self.target = config["target"]
        self.seg_head = BevSegHead(
            self.target, config["seg_head_dim"], config["output_class"]
        )

    def forward(self, batch_dict):
        record_len = batch_dict["record_len"]
        transformation_matrix = batch_dict["transformation_matrix"]

        encoder_inputs = self._ensure_temporal_agent_inputs(batch_dict["inputs"])
        current_inputs = self._select_current_image_inputs(encoder_inputs).unsqueeze(1)
        intrinsic = self._select_current_calibration(batch_dict["intrinsic"]).unsqueeze(1)
        extrinsic = self._select_current_calibration(batch_dict["extrinsic"]).unsqueeze(1)

        image_features = self.jepa_encoder(encoder_inputs)
        cvm_batch = {
            "inputs": current_inputs,
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "features": [image_features.unsqueeze(1)],
        }
        x = self.cvm(cvm_batch).squeeze(1)

        if self.debug_shapes:
            print("[VjepaCvtFuse] JEPA spatial:", tuple(image_features.shape))
            print("[VjepaCvtFuse] CVM BEV:", tuple(x.shape))

        x, mask = regroup(x, record_len, self.max_cav)
        x = self.sttf(x, transformation_matrix)

        if self.use_roi_mask:
            com_mask = get_roi_and_cav_mask(
                x.shape, mask, transformation_matrix, self.discrete_ratio, self.downsample_rate
            )
        else:
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        x = rearrange(x, "b l h w c -> b l c h w")
        x = self.fusion_net(x, com_mask)
        x = x.unsqueeze(1)

        x = self.decoder(x)
        x = rearrange(x, "b l c h w -> (b l) c h w")
        return self.seg_head(x, x.shape[0], 1)

    @staticmethod
    def _ensure_temporal_agent_inputs(inputs):
        if inputs.dim() == 5:
            return inputs.unsqueeze(1)
        if inputs.dim() == 6:
            return inputs
        if inputs.dim() == 7:
            return inputs.reshape(-1, *inputs.shape[2:])
        raise ValueError(
            "Expected inputs [N,M,H,W,C], [N,T,M,H,W,C], [B,L,T,M,H,W,C], "
            f"got {tuple(inputs.shape)}"
        )

    @staticmethod
    def _select_current_image_inputs(inputs):
        return inputs[:, 0]

    @staticmethod
    def _select_current_calibration(tensor):
        if tensor is None:
            raise ValueError("VjepaCvtFuse requires camera calibration")
        if tensor.dim() == 4:
            return tensor
        if tensor.dim() == 5:
            return tensor[:, 0]
        if tensor.dim() == 6:
            return tensor.reshape(-1, *tensor.shape[2:])[:, 0]
        raise ValueError(f"Unexpected calibration shape: {tuple(tensor.shape)}")
