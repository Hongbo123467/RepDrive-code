#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


GROUP_PREFIXES = {
    "image_backbone": ("image_backbone.", "scene_embeds"),
    "image_fpn_lss": ("image_fpn.", "lss_bev_projector."),
    "proposal_generator": (
        "hist_encoding.",
        "init_feature.",
        "trajectory_decoder.",
        "traj_head.",
    ),
    "scorer_path": ("pos_embed.", "scorer_attention.", "scorer."),
    "decoder_shared": (
        "post_scorer_decoder.segmentation_head.",
        "post_scorer_decoder.bev_downscale.",
        "post_scorer_decoder.keyval_embedding.",
        "post_scorer_decoder.status_encoding.",
        "post_scorer_decoder.query_embedding.",
        "post_scorer_decoder.tf_decoder.",
        "post_scorer_decoder.agent_head.",
    ),
    "interaction_refiner": ("post_scorer_decoder.proposal_refiner.",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit parameter updates between DriveJEPA3 checkpoints."
    )
    parser.add_argument("stage1_checkpoint", type=Path)
    parser.add_argument("trained_checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def canonical_key(key: str) -> str:
    for prefix in ("state_dict.", "agent._drivor_model.", "_drivor_model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key.replace(
        "bev_agent_decoder.",
        "post_scorer_decoder.",
    )


def load_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    raw_state = checkpoint.get("state_dict", checkpoint)
    state = {
        canonical_key(key): value
        for key, value in raw_state.items()
        if torch.is_tensor(value)
    }
    metadata = {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }
    return state, metadata


def belongs_to_group(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key == prefix or key.startswith(prefix) for prefix in prefixes)


def summarize_group(
    initial: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    trained_keys = sorted(
        key for key in trained if belongs_to_group(key, prefixes)
    )
    shared_keys = [
        key
        for key in trained_keys
        if key in initial and initial[key].shape == trained[key].shape
    ]
    new_keys = [key for key in trained_keys if key not in shared_keys]

    diff_sq = 0.0
    initial_sq = 0.0
    trained_sq = 0.0
    max_abs_delta = 0.0
    numel = 0
    changed_tensors = 0
    for key in shared_keys:
        before = initial[key].detach().float()
        after = trained[key].detach().float()
        delta = after - before
        diff_sq += delta.square().sum().item()
        initial_sq += before.square().sum().item()
        trained_sq += after.square().sum().item()
        max_abs_delta = max(max_abs_delta, delta.abs().max().item())
        numel += delta.numel()
        changed_tensors += int(torch.count_nonzero(delta).item() > 0)

    new_sq = 0.0
    new_numel = 0
    for key in new_keys:
        value = trained[key].detach().float()
        new_sq += value.square().sum().item()
        new_numel += value.numel()

    l2_delta = math.sqrt(diff_sq)
    initial_l2 = math.sqrt(initial_sq)
    return {
        "shared_tensors": len(shared_keys),
        "changed_tensors": changed_tensors,
        "shared_numel": numel,
        "l2_delta": l2_delta,
        "relative_l2_delta": l2_delta / max(initial_l2, 1e-12),
        "max_abs_delta": max_abs_delta,
        "initial_l2": initial_l2,
        "trained_l2": math.sqrt(trained_sq),
        "new_tensors": len(new_keys),
        "new_numel": new_numel,
        "new_parameter_l2": math.sqrt(new_sq),
    }


def main() -> None:
    args = parse_args()
    initial, initial_metadata = load_state_dict(args.stage1_checkpoint)
    trained, trained_metadata = load_state_dict(args.trained_checkpoint)

    groups = {
        name: summarize_group(initial, trained, prefixes)
        for name, prefixes in GROUP_PREFIXES.items()
    }
    zero_init_delta_biases = {
        key: value.detach().float().norm().item()
        for key, value in trained.items()
        if key.startswith("post_scorer_decoder.proposal_refiner.refiners.")
        and key.endswith("delta_head.2.bias")
    }
    image_unchanged = (
        groups["image_backbone"]["l2_delta"] == 0.0
        and groups["image_backbone"]["changed_tensors"] == 0
    )
    proposal_generator_unchanged = (
        groups["proposal_generator"]["l2_delta"] == 0.0
        and groups["proposal_generator"]["changed_tensors"] == 0
    )
    scorer_updated = (
        groups["scorer_path"]["l2_delta"] > 0.0
        and groups["scorer_path"]["changed_tensors"] > 0
    )
    interaction_present = (
        groups["interaction_refiner"]["new_parameter_l2"] > 0.0
        and groups["interaction_refiner"]["new_tensors"] > 0
    )
    interaction_updated = bool(zero_init_delta_biases) and all(
        norm > 0.0 for norm in zero_init_delta_biases.values()
    )
    result = {
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "trained_checkpoint": str(args.trained_checkpoint),
        "stage1_metadata": initial_metadata,
        "trained_metadata": trained_metadata,
        "checks": {
            "image_backbone_unchanged": image_unchanged,
            "proposal_generator_unchanged": proposal_generator_unchanged,
            "scorer_updated": scorer_updated,
            "interaction_refiner_present": interaction_present,
            "interaction_zero_init_biases_updated": interaction_updated,
        },
        "status": (
            "pass"
            if (
                image_unchanged
                and proposal_generator_unchanged
                and scorer_updated
                and interaction_present
                and interaction_updated
            )
            else "fail"
        ),
        "zero_init_delta_bias_l2": zero_init_delta_biases,
        "groups": groups,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
