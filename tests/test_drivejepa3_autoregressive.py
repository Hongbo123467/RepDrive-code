from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navsim.agents.drivoR.drivejepa3_decoder import DriveJEPA3Decoder


def make_config():
    return OmegaConf.create(
        {
            "num_poses": 8,
            "tf_d_model": 64,
            "tf_d_ffn": 128,
            "tf_num_head": 4,
            "tf_num_layers": 2,
            "tf_dropout": 0.0,
            "decoder_dropout": 0.0,
            "decoder_ref_num": 2,
            "decoder_residual_scale": 1.0,
            "decoder_detach_between_rounds": True,
            "bev_downscale_size": 8,
            "num_bounding_boxes": 6,
            "num_bev_classes": 7,
            "LIFT_X_BOUND": [0.0, 32.0, 0.25],
            "LIFT_Y_BOUND": [-32.0, 32.0, 0.25],
        }
    )


def test_zero_initialized_refiner_is_identity():
    torch.manual_seed(0)
    batch, proposals, poses, dim = 2, 5, 8, 64
    decoder = DriveJEPA3Decoder(make_config(), in_channels=16).eval()
    anchors = torch.randn(batch, proposals, poses, 3)
    anchors[..., 0] = anchors[..., 0].sigmoid() * 30.0
    anchors[..., 1] = anchors[..., 1].tanh() * 30.0

    output = decoder(
        bev_feature=torch.randn(batch, 16, 128, 256),
        status_feature=torch.randn(batch, 8),
        proposals=anchors,
        traj_feature_bev=torch.randn(batch, proposals, dim),
    )

    assert len(output["poses_reg_list"]) == 2
    for refined in output["poses_reg_list"]:
        torch.testing.assert_close(refined, anchors, rtol=0.0, atol=0.0)


def test_residual_head_receives_gradient_at_identity_initialization():
    torch.manual_seed(1)
    batch, proposals, poses, dim = 1, 4, 8, 64
    decoder = DriveJEPA3Decoder(make_config(), in_channels=16).train()
    anchors = torch.randn(batch, proposals, poses, 3)
    anchors[..., 0] = anchors[..., 0].sigmoid() * 30.0
    anchors[..., 1] = anchors[..., 1].tanh() * 30.0

    output = decoder(
        bev_feature=torch.randn(batch, 16, 128, 256),
        status_feature=torch.randn(batch, 8),
        proposals=anchors,
        traj_feature_bev=torch.randn(batch, proposals, dim),
    )
    target = anchors + 0.1
    loss = sum(
        torch.nn.functional.l1_loss(refined, target)
        for refined in output["poses_reg_list"]
    )
    loss.backward()

    for refiner in decoder.proposal_refiner.refiners:
        final_layer = refiner.delta_head[-1]
        assert final_layer.weight.grad is not None
        assert final_layer.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    test_zero_initialized_refiner_is_identity()
    test_residual_head_receives_gradient_at_identity_initialization()
    print("DriveJEPA3 autoregressive decoder tests passed")
