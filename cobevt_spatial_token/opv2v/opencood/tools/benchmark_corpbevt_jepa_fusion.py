import argparse
import statistics

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils.seg_utils import cal_iou_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--fusion_debug_mode",
        choices=["normal", "ego_only_before_sttf", "ego_only_after_sttf"],
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hypes = yaml_utils.load_yaml(None, args)
    model_args = hypes["model"]["args"]
    fusion_method = model_args.get("fusion_method", "swap")
    debug_mode = model_args.get("fusion_debug_mode", "normal")
    print(f"fusion_method={fusion_method} fusion_debug_mode={debug_mode}")

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch,
        shuffle=False,
    )

    device = torch.device("cuda")
    model = train_utils.create_model(hypes).to(device)
    epoch, model = train_utils.load_saved_model(args.model_dir, model)
    if args.fusion_debug_mode is not None:
        model.fusion_debug_mode = args.fusion_debug_mode
        debug_mode = args.fusion_debug_mode
    model.eval()

    dynamic_ious = []
    with torch.no_grad():
        for index, batch_data in enumerate(loader):
            if index >= args.num_samples:
                break
            batch_data = train_utils.to_device(batch_data, device)
            output = dataset.post_process(batch_data["ego"], model(batch_data["ego"]))
            dynamic_iou, _ = cal_iou_training(batch_data, output)
            dynamic_ious.append(float(dynamic_iou[1]))
            if len(dynamic_ious) % 25 == 0:
                print(f"samples={len(dynamic_ious)} dynamic_iou={statistics.mean(dynamic_ious):.9f}")

    print(
        f"RESULT fusion_method={fusion_method} fusion_debug_mode={debug_mode} epoch={epoch} "
        f"samples={len(dynamic_ious)} dynamic_iou={statistics.mean(dynamic_ious):.9f}"
    )


if __name__ == "__main__":
    main()
