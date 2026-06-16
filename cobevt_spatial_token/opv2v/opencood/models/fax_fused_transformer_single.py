"""
Single-CAV V-JEPA + FAX segmentation model.

This keeps the FAX image-to-BEV path from fax_fused_transformer.py and only
replaces the image backbone with V-JEPA spatial features. It predicts the ego
vehicle BEV segmentation for each batch item.
"""

import copy
import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.sub_modules.jepa_spatial_encoder import JEPASpatialEncoder
from opencood.models.backbones.vfm_fpn import SSBiFPNPyramidAdapter
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead


class FaxFusedTransformerSingle(nn.Module):
    """
    Single-vehicle segmentation variant:
        V-JEPA spatial tokens -> FAXModule -> NaiveDecoder -> BevSegHead.

    OpenCOOD's intermediate-fusion dataloader concatenates CAVs from every
    scene. The ego CAV is the first CAV in each scene group, so this model uses
    ``record_len`` to gather those ego rows before running the backbone. Temporal
    image inputs are ordered as current frame first, followed by history frames.
    """

    def __init__(self, config):
        super(FaxFusedTransformerSingle, self).__init__()
        config = self._unwrap_model_config(config)
        if "jepa_encoder" not in config:
            raise ValueError(
                "FaxFusedTransformerSingle expects a 'jepa_encoder' config block. "
                f"Got config keys: {sorted(config.keys())}"
            )

        self.jepa_encoder = JEPASpatialEncoder(config["jepa_encoder"])
        num_cams = config["jepa_encoder"].get("num_cams", 4)
        fpn_channels = config.get("vfm_fpn", {})
        fpn_in_ch = self.jepa_encoder.fpn_feature_dim
        self.vfm_fpn = SSBiFPNPyramidAdapter(
            in_chs=tuple(
                fpn_channels.get(
                    "in_chs", [fpn_in_ch] * len(self.jepa_encoder.fpn_layer_indices)
                )
            ),
            mid_ch=fpn_channels.get("mid_ch", 256),
            out_channels=tuple(fpn_channels.get("out_channels", [128, 256, 512])),
        )

        fax_config = copy.deepcopy(config["fax"])
        fax_config["backbone_output_shape"] = [
            (1, 1, num_cams, 128, 64, 64),
            (1, 1, num_cams, 256, 32, 32),
            (1, 1, num_cams, 512, 16, 16),
        ]
        self.fax = FAXModule(fax_config)

        self.decoder = NaiveDecoder(config["decoder"])
        self.target = config["target"]
        self.seg_head = BevSegHead(
            self.target, config["seg_head_dim"], config["output_class"]
        )

    @staticmethod
    def _unwrap_model_config(config):
        if "jepa_encoder" in config:
            return config
        if "args" in config and isinstance(config["args"], dict):
            return config["args"]
        if (
            "model" in config
            and isinstance(config["model"], dict)
            and isinstance(config["model"].get("args"), dict)
        ):
            return config["model"]["args"]
        return config

    def forward(self, batch_dict):
        encoder_inputs, current_inputs, intrinsic, extrinsic = self._prepare_ego_batch(
            batch_dict
        )

        multiscale_features = self.jepa_encoder.forward_multiscale(encoder_inputs)
        image_features = self._build_fax_features(multiscale_features)
        fax_batch = dict(batch_dict)
        fax_batch.update(
            {
                "inputs": current_inputs,
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "features": image_features,
            }
        )
        self._assert_fax_batch_is_current_frame(fax_batch)

        x = self.fax(fax_batch)
        x = self.decoder(x)
        x = rearrange(x, "b l c h w -> (b l) c h w")
        batch_size = x.shape[0]
        return self.seg_head(x, batch_size, 1)

    def _build_fax_features(self, multiscale_features):
        batch_size, num_cams = multiscale_features[0].shape[:2]
        flat_features = [
            rearrange(feature, "b m c h w -> (b m) c h w")
            for feature in multiscale_features
        ]
        pyramid_features = self.vfm_fpn(flat_features)
        return [
            rearrange(
                feature,
                "(b m) c h w -> b 1 m c h w",
                b=batch_size,
                m=num_cams,
            )
            for feature in pyramid_features
        ]

    @staticmethod
    def _ego_indices(record_len, device):
        starts = torch.cumsum(
            torch.cat([record_len.new_zeros(1), record_len[:-1]]), dim=0
        )
        return starts.to(device=device, dtype=torch.long)

    def _prepare_ego_batch(self, batch_dict):
        inputs = batch_dict["inputs"]
        intrinsic = batch_dict.get("intrinsic")
        extrinsic = batch_dict.get("extrinsic")

        if "record_len" in batch_dict:
            record_len = batch_dict["record_len"]
            ego_indices = self._ego_indices(record_len, inputs.device)
            inputs = self._select_agent_rows(inputs, ego_indices, record_len)
            intrinsic = self._select_agent_rows(intrinsic, ego_indices, record_len)
            extrinsic = self._select_agent_rows(extrinsic, ego_indices, record_len)
        else:
            inputs = self._dense_inputs_to_agent_rows(inputs)

        encoder_inputs = self._ensure_temporal_agent_inputs(inputs)
        current_inputs = self._select_current_image_inputs(encoder_inputs).unsqueeze(1)
        intrinsic = self._select_current_calibration(intrinsic).unsqueeze(1)
        extrinsic = self._select_current_calibration(extrinsic).unsqueeze(1)

        return encoder_inputs, current_inputs, intrinsic, extrinsic

    @staticmethod
    def _select_agent_rows(tensor, ego_indices, record_len):
        if tensor is None:
            return None
        num_agents = int(record_len.sum().item())
        if tensor.shape[0] == num_agents:
            return tensor.index_select(0, ego_indices.to(tensor.device))
        batch_size = record_len.shape[0]
        max_cav = int(record_len.max().item())
        if tensor.dim() >= 2 and tensor.shape[0] == batch_size:
            if tensor.shape[1] == max_cav:
                return tensor[:, 0]
            if tensor.dim() >= 3 and tensor.shape[2] == max_cav:
                return tensor[:, :, 0]
            return tensor[:, 0]
        return tensor

    @staticmethod
    def _dense_inputs_to_agent_rows(inputs):
        if inputs.dim() in (5, 6):
            return inputs
        if inputs.dim() == 7:
            # Support both [B,L,T,M,H,W,C] and [B,T,L,M,H,W,C].
            return inputs[:, 0] if inputs.shape[1] > inputs.shape[2] else inputs[:, :, 0]
        raise ValueError(
            "FaxFusedTransformerSingle expects inputs shaped (N,T,M,H,W,C), "
            "(B,L,T,M,H,W,C), (B,T,L,M,H,W,C), or (B,M,H,W,C); got "
            f"{tuple(inputs.shape)}"
        )

    @staticmethod
    def _ensure_temporal_agent_inputs(inputs):
        if inputs.dim() == 5:
            return inputs.unsqueeze(1)
        if inputs.dim() == 6:
            return inputs
        raise ValueError(
            "Expected ego inputs shaped (B,T,M,H,W,C) or (B,M,H,W,C), got "
            f"{tuple(inputs.shape)}"
        )

    @staticmethod
    def _select_current_image_inputs(inputs):
        return inputs[:, 0]

    @staticmethod
    def _select_current_calibration(tensor):
        if tensor is None:
            raise ValueError("FaxFusedTransformerSingle requires camera calibration")
        if tensor.dim() in (5, 6):
            return tensor[:, 0]
        return tensor

    @staticmethod
    def _assert_fax_batch_is_current_frame(batch):
        if batch["inputs"].dim() != 6:
            raise ValueError(
                "FAX expects current-frame inputs with shape [B,1,M,H,W,C], "
                f"got {tuple(batch['inputs'].shape)}"
            )
        if batch["intrinsic"].dim() != 5:
            raise ValueError(
                "FAX expects current-frame intrinsic with shape [B,1,M,3,3], "
                f"got {tuple(batch['intrinsic'].shape)}"
            )
        if batch["extrinsic"].dim() != 5:
            raise ValueError(
                "FAX expects current-frame extrinsic with shape [B,1,M,4,4], "
                f"got {tuple(batch['extrinsic'].shape)}"
            )
        for feature in batch["features"]:
            if feature.dim() != 6:
                raise ValueError(
                    "FAX expects feature maps with shape [B,1,M,C,H,W], "
                    f"got {tuple(feature.shape)}"
                )
