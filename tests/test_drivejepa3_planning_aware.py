from pathlib import Path
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navsim.agents.drivoR.drivejepa3_decoder import (
    DistanceAwareAgentInteraction,
    DriveJEPA3PlanningAwareDecoder,
    DriveJEPA3PlanningAwareRefinementLayer,
)
from navsim.agents.drivoR.drivejepa3_model import (
    DriveJEPA3Model,
    InteractionResidualScorer,
)
from navsim.agents.drivoR.transformer_decoder import TransformerDecoderScorer


def make_config():
    return OmegaConf.create(
        {
            "num_poses": 8,
            "tf_d_model": 32,
            "tf_d_ffn": 64,
            "tf_num_head": 4,
            "tf_num_layers": 1,
            "tf_dropout": 0.0,
            "decoder_dropout": 0.0,
            "decoder_ref_num": 2,
            "decoder_residual_scale": 1.0,
            "decoder_residual_init_std": 1e-3,
            "decoder_detach_between_rounds": False,
            "bev_downscale_size": 4,
            "num_bounding_boxes": 6,
            "num_bev_classes": 7,
            "LIFT_X_BOUND": [0.0, 32.0, 0.25],
            "LIFT_Y_BOUND": [-32.0, 32.0, 0.25],
        }
    )


def test_complete_trajectory_refinement_and_gradients():
    torch.manual_seed(7)
    batch, proposals, poses, dim = 2, 5, 8, 32
    decoder = DriveJEPA3PlanningAwareDecoder(make_config(), in_channels=8).train()
    anchors = torch.randn(batch, proposals, poses, 3)
    anchors[..., 0] = anchors[..., 0].sigmoid() * 30.0
    anchors[..., 1] = anchors[..., 1].tanh() * 30.0

    output = decoder(
        bev_feature=torch.randn(batch, 8, 16, 32),
        status_feature=torch.randn(batch, 8),
        proposals=anchors,
        traj_feature_bev=torch.randn(batch, proposals, dim),
        scene_features=torch.randn(batch, 12, dim),
    )

    assert len(output["poses_reg_list"]) == 2
    assert len(output["refiner_query_list"]) == 2
    assert all(poses_i.shape == anchors.shape for poses_i in output["poses_reg_list"])
    assert all(
        query_i.shape == (batch, proposals, dim)
        for query_i in output["refiner_query_list"]
    )
    assert not torch.equal(output["poses_reg_list"][0], anchors)
    assert not torch.equal(output["poses_reg_list"][1], output["poses_reg_list"][0])

    target = anchors + 0.2
    loss = torch.nn.functional.smooth_l1_loss(output["poses_reg_list"][-1], target)
    loss.backward()

    for refiner in decoder.proposal_refiner.refiners:
        assert isinstance(refiner, DriveJEPA3PlanningAwareRefinementLayer)
        assert not any(isinstance(module, nn.GRUCell) for module in refiner.modules())
        parameters = (
            refiner.scene_interaction.attention.in_proj_weight,
            refiner.bev_interaction.value_proj[0].weight,
            refiner.agent_interaction.relative_encoding[0].weight,
            refiner.delta_head[-1].weight,
        )
        for parameter in parameters:
            assert parameter.grad is not None
            assert parameter.grad.abs().sum() > 0


def test_agent_geometry_uses_euclidean_distance():
    trajectory = torch.tensor(
        [[[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]]
    )
    agent_states = torch.tensor([[[3.0, 4.0, 0.0, 1.0, 1.0]]])

    geometry = DistanceAwareAgentInteraction._nearest_relative_geometry(
        trajectory,
        agent_states,
    )

    torch.testing.assert_close(geometry[0, 0, 0, :2], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(geometry[0, 0, 0, 2], torch.tensor(5.0))


def test_interaction_residual_scorer_starts_zero_and_is_bounded():
    torch.manual_seed(11)
    scorer = InteractionResidualScorer(d_model=32, hidden_dim=16, scale=0.5)
    query = torch.randn(2, 5, 32)

    initial = scorer(query)
    torch.testing.assert_close(initial, torch.zeros_like(initial))

    optimizer = torch.optim.SGD(scorer.parameters(), lr=0.1)
    loss = (scorer(query) - 0.25).square().mean()
    loss.backward()
    optimizer.step()

    updated = scorer(query)
    assert updated.abs().max() <= 0.5
    assert updated.abs().sum() > 0


def test_detached_initial_query_preserves_amp_scorer_gradients():
    torch.manual_seed(13)
    config = make_config()
    model = DriveJEPA3Model.__new__(DriveJEPA3Model)
    nn.Module.__init__(model)
    model.pos_embed = nn.Sequential(
        nn.Linear(config.num_poses * 3, config.tf_d_ffn),
        nn.ReLU(),
        nn.Linear(config.tf_d_ffn, config.tf_d_model),
    )
    model.scorer_attention = TransformerDecoderScorer(
        num_layers=2,
        d_model=config.tf_d_model,
        proj_drop=0.0,
        drop_path=0.0,
        config=config,
    )

    proposals = torch.randn(2, 5, config.num_poses, 3)
    scene_features = torch.randn(2, 7, config.tf_d_model)
    ego_token = torch.randn(2, 1, config.tf_d_model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        initial_query = model._encode_trajectory_query(
            proposals,
            scene_features,
            ego_token,
            detach_output=True,
        )
        final_query = model._encode_trajectory_query(
            proposals,
            scene_features,
            ego_token,
            detach_output=False,
        )
        loss = final_query.float().square().mean()

    assert not initial_query.requires_grad
    assert final_query.requires_grad
    loss.backward()
    for module in (model.pos_embed, model.scorer_attention):
        for parameter in module.parameters():
            assert parameter.grad is not None


if __name__ == "__main__":
    test_complete_trajectory_refinement_and_gradients()
    test_agent_geometry_uses_euclidean_distance()
    test_interaction_residual_scorer_starts_zero_and_is_bounded()
    test_detached_initial_query_preserves_amp_scorer_gradients()
    print("DriveJEPA3 planning-aware decoder tests passed")
