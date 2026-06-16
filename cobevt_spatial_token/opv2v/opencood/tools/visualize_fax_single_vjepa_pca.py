"""
PCA visualization for FaxFusedTransformerSingle's frozen V-JEPA 2.1 features.

This script uses the real OPV2V camera dataloader and the
FaxFusedTransformerSingle model path:

    batch -> model._prepare_ego_batch() -> model.jepa_encoder raw patch tokens

It is meant as a quick sanity check for whether the model can extract image
features before those features are handed to FAXModule.

Example:
    cd /home/dataset-local/yinhongbo/code/CoBEVT/opv2v
    python opencood/tools/visualize_fax_single_vjepa_pca.py \
        --hypes_yaml opencood/hypes_yaml/opcamera/fax_jepa.yaml \
        --output_dir ./fax_single_vjepa_pca \
        --num_batches 1
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

OPV2V_ROOT = Path(__file__).resolve().parents[2]
if str(OPV2V_ROOT) not in sys.path:
    sys.path.insert(0, str(OPV2V_ROOT))

try:
    import open3d  # noqa: F401
except ModuleNotFoundError:
    import types

    open3d_stub = types.ModuleType("open3d")
    open3d_stub.io = types.SimpleNamespace()
    sys.modules["open3d"] = open3d_stub

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class TorchPCA:
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.mean: Optional[torch.Tensor] = None
        self.components: Optional[torch.Tensor] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, x: torch.Tensor) -> "TorchPCA":
        x = x.float()
        self.mean = x.mean(dim=0, keepdim=True)
        centered = x - self.mean
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        self.components = vh[: self.n_components].contiguous()

        variances = (s ** 2) / max(x.shape[0] - 1, 1)
        total = variances.sum().clamp_min(1e-12)
        self.explained_variance_ratio_ = (
            variances[: self.n_components] / total
        ).cpu().numpy()
        return self

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA has not been fitted.")
        return (x.float() - self.mean) @ self.components.T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FaxFusedTransformerSingle V-JEPA2.1 patch tokens."
    )
    parser.add_argument(
        "--hypes_yaml",
        type=str,
        default="/home/dataset-local/yinhongbo/code/CoBEVT/opv2v/opencood/hypes_yaml/opcamera/fax_jepa.yaml",
    )
    parser.add_argument("--output_dir", type=str, default="./fax_single_vjepa_pca")
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=["train", "validate"], default="validate")
    parser.add_argument("--max_visualizations", type=int, default=8)
    parser.add_argument("--camera_ids", type=int, nargs="*", default=None)
    parser.add_argument("--upsample_mode", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--fit_per_image", action="store_true")
    parser.add_argument(
        "--keep_yaml_backbone_mode",
        action="store_true",
        help="Do not override jepa_encoder to frozen. By default the script uses frozen V-JEPA2.1.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint path or model directory. Not needed for frozen pretrained feature extraction.",
    )
    return parser.parse_args()


def force_frozen_vjepa21(hypes: Dict) -> Dict:
    hypes = copy.deepcopy(hypes)
    jepa_cfg = hypes["model"]["args"]["jepa_encoder"]
    jepa_cfg["vjepa_version"] = "2.1"
    jepa_cfg["backbone_mode"] = "frozen"
    jepa_cfg["lora_rank"] = 0
    jepa_cfg["use_grid_mask"] = False
    return hypes


def build_loader(hypes: Dict, args: argparse.Namespace) -> DataLoader:
    train = args.split == "train"
    dataset = build_dataset(hypes, visualize=False, train=train, validate=not train)
    batch_size = 1 if not train else hypes["train_params"].get("batch_size", 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )


def load_optional_checkpoint(model: torch.nn.Module, checkpoint: Optional[str]) -> None:
    if checkpoint is None:
        return
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        _, model = train_utils.load_saved_model(str(checkpoint_path), model)
        return
    state = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=False)


def unnormalize_image(image_chw: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(image_chw.device, image_chw.dtype)
    std = STD.to(image_chw.device, image_chw.dtype)
    return torch.clamp(image_chw * std + mean, 0.0, 1.0)


@torch.no_grad()
def extract_raw_tokens(
    model: torch.nn.Module,
    batch_ego: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder_inputs, current_inputs, intrinsic, extrinsic = model._prepare_ego_batch(batch_ego)

    if intrinsic.shape[1] != 1 or extrinsic.shape[1] != 1:
        raise RuntimeError(
            "Expected current-frame calibration for FAX, got "
            f"intrinsic={tuple(intrinsic.shape)}, extrinsic={tuple(extrinsic.shape)}"
        )

    batch, time_len, num_cams, height, width, channels = encoder_inputs.shape
    if time_len < 2:
        cur_frame = encoder_inputs[:, 0]
        prev_frame = encoder_inputs[:, 0]
    else:
        cur_frame = encoder_inputs[:, 0]
        prev_frame = encoder_inputs[:, 1]

    cur_flat = cur_frame.permute(0, 1, 4, 2, 3).reshape(
        batch * num_cams, channels, height, width
    )
    prev_flat = prev_frame.permute(0, 1, 4, 2, 3).reshape(
        batch * num_cams, channels, height, width
    )
    tokens = model.jepa_encoder._encode_raw_spatial_tokens(cur_flat, prev_flat)

    print(
        "Prepared from fax_fused_transformer_single: "
        f"encoder_inputs={tuple(encoder_inputs.shape)}, "
        f"current_inputs={tuple(current_inputs.shape)}, "
        f"intrinsic={tuple(intrinsic.shape)}, extrinsic={tuple(extrinsic.shape)}"
    )
    return tokens.detach().cpu(), current_inputs.detach().cpu()


def fit_pca(tokens: torch.Tensor) -> Tuple[TorchPCA, np.ndarray, np.ndarray]:
    flat = tokens.reshape(-1, tokens.shape[-1])
    pca = TorchPCA(n_components=3)
    reduced = pca.fit_transform(flat).numpy()
    pca_min = reduced.min(axis=0)
    pca_max = reduced.max(axis=0)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"PCA cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return pca, pca_min, pca_max


def pca_tokens_to_image(
    tokens_1n: torch.Tensor,
    pca: TorchPCA,
    pca_min: np.ndarray,
    pca_max: np.ndarray,
    grid_hw: Tuple[int, int],
    output_hw: Tuple[int, int],
    upsample_mode: str,
) -> torch.Tensor:
    reduced = pca.transform(tokens_1n.squeeze(0)).numpy()
    denom = np.maximum(pca_max - pca_min, 1e-6)
    rgb = np.clip((reduced - pca_min) / denom, 0.0, 1.0)

    grid_h, grid_w = grid_hw
    if rgb.shape[0] != grid_h * grid_w:
        raise ValueError(
            f"Token count {rgb.shape[0]} does not match grid {grid_h}x{grid_w}."
        )
    rgb_grid = torch.from_numpy(rgb).float().view(1, grid_h, grid_w, 3).permute(0, 3, 1, 2)
    kwargs = {"size": output_hw, "mode": upsample_mode}
    if upsample_mode == "bilinear":
        kwargs["align_corners"] = False
    return F.interpolate(rgb_grid, **kwargs).squeeze(0)


def save_visualization(
    pca_image: torch.Tensor,
    original_chw: torch.Tensor,
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(pca_image.permute(1, 2, 0).numpy())
    axes[0].set_title("FaxSingle V-JEPA2.1 PCA")
    axes[0].axis("off")
    axes[1].imshow(original_chw.permute(1, 2, 0).numpy())
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    if not args.keep_yaml_backbone_mode:
        hypes = force_frozen_vjepa21(hypes)

    device = torch.device(args.device)
    loader = build_loader(hypes, args)

    print("Creating FaxFusedTransformerSingle model")
    print(f"  hypes_yaml: {args.hypes_yaml}")
    print(f"  backbone_mode: {hypes['model']['args']['jepa_encoder'].get('backbone_mode')}")
    model = train_utils.create_model(hypes).to(device)
    load_optional_checkpoint(model, args.checkpoint)
    model.eval()

    output_dir = Path(args.output_dir)
    all_tokens: List[torch.Tensor] = []
    all_current_inputs: List[torch.Tensor] = []

    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        batch_ego = train_utils.to_device(batch_data["ego"], device)
        tokens, current_inputs = extract_raw_tokens(model, batch_ego)
        print(f"Batch {batch_idx}: raw patch tokens={tuple(tokens.shape)}")
        all_tokens.append(tokens)
        all_current_inputs.append(current_inputs)

    if not all_tokens:
        raise RuntimeError("No batches were processed.")

    tokens_all = torch.cat(all_tokens, dim=0)
    current_all = torch.cat(all_current_inputs, dim=0)
    token_h, token_w = tuple(hypes["model"]["args"]["jepa_encoder"]["spatial_hw"])
    output_h, output_w = current_all.shape[-3], current_all.shape[-2]
    print(
        f"Collected tokens={tuple(tokens_all.shape)}, "
        f"patch_grid={token_h}x{token_w}, current_images={tuple(current_all.shape)}"
    )

    global_pca = global_min = global_max = None
    if not args.fit_per_image:
        global_pca, global_min, global_max = fit_pca(tokens_all)

    num_cams = current_all.shape[2]
    camera_ids = args.camera_ids if args.camera_ids is not None else list(range(num_cams))
    saved = 0

    for batch_id in range(current_all.shape[0]):
        for cam_id in camera_ids:
            if saved >= args.max_visualizations:
                print(f"Reached max_visualizations={args.max_visualizations}")
                return
            token_index = batch_id * num_cams + cam_id
            tokens_1n = tokens_all[token_index : token_index + 1]

            if args.fit_per_image:
                pca, pca_min, pca_max = fit_pca(tokens_1n)
            else:
                pca, pca_min, pca_max = global_pca, global_min, global_max

            pca_image = pca_tokens_to_image(
                tokens_1n,
                pca,
                pca_min,
                pca_max,
                grid_hw=(token_h, token_w),
                output_hw=(output_h, output_w),
                upsample_mode=args.upsample_mode,
            )

            original = current_all[batch_id, 0, cam_id].permute(2, 0, 1)
            original = unnormalize_image(original)
            save_path = output_dir / f"batch{batch_id:03d}_cam{cam_id}_pca.png"
            save_visualization(
                pca_image,
                original,
                save_path,
                title=f"batch {batch_id}, camera {cam_id}",
            )
            print(f"Saved: {save_path}")
            saved += 1


if __name__ == "__main__":
    main()
