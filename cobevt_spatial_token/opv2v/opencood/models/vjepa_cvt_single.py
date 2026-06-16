"""Single-CAV V-JEPA 2.1 + CrossViewModule BEV segmentation model."""

import copy

import torch.nn as nn
from einops import rearrange

from opencood.models.fax_fused_transformer_single import FaxFusedTransformerSingle
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.cvt_modules import CrossViewModule
from opencood.models.sub_modules.jepa_spatial_encoder import JEPASpatialEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder


class VjepaCvtSingle(FaxFusedTransformerSingle):
    """V-JEPA spatial features -> learned cross-view attention -> BEV."""

    def __init__(self, config):
        nn.Module.__init__(self)
        config = self._unwrap_model_config(config)

        self.jepa_encoder = JEPASpatialEncoder(config["jepa_encoder"])

        num_cams = int(config["jepa_encoder"].get("num_cams", 4))
        feat_h, feat_w = self.jepa_encoder.spatial_hw
        cvm_config = copy.deepcopy(config["cvm"])
        cvm_config["backbone_output_shape"] = [
            (1, 1, num_cams, self.jepa_encoder.output_dim, feat_h, feat_w)
        ]
        if len(cvm_config["middle"]) != 1:
            raise ValueError("VjepaCvtSingle currently expects one CVM feature level")
        self.cvm = CrossViewModule(cvm_config)

        self.decoder = NaiveDecoder(config["decoder"])
        self.target = config["target"]
        self.seg_head = BevSegHead(
            self.target, config["seg_head_dim"], config["output_class"]
        )

    def forward(self, batch_dict):
        encoder_inputs, current_inputs, intrinsic, extrinsic = self._prepare_ego_batch(
            batch_dict
        )

        image_features = self.jepa_encoder(encoder_inputs)
        cvm_batch = {
            "inputs": current_inputs,
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "features": [image_features.unsqueeze(1)],
        }
        x = self.cvm(cvm_batch)
        x = self.decoder(x)
        x = rearrange(x, "b l c h w -> (b l) c h w")
        return self.seg_head(x, x.shape[0], 1)
