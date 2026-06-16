# CoBEVT Spatial-Token Cooperative Perception Overlay

This directory contains the CoBEVT/OPV2V cooperative perception experiment code
that reuses RepDrive/DriveJEPA-style V-JEPA spatial tokens.

It is an overlay snapshot, not a standalone Python package. The files keep their
original relative paths under `opv2v/opencood/...` so they can be copied back on
top of a CoBEVT checkout.

## Main Path

```text
camera frames
  -> V-JEPA 2.1 spatial-token encoder
  -> robustbev-style camera-to-BEV projection
  -> STTF ego-frame alignment
  -> multi-CAV BEV fusion
  -> BEV segmentation head
```

Core files:

| File | Purpose |
| --- | --- |
| `opv2v/opencood/models/corpbevt_jepa.py` | Cooperative V-JEPA spatial-token BEV segmentation model. |
| `opv2v/opencood/models/sub_modules/jepa_spatial_encoder.py` | Adapter that exposes V-JEPA spatial tokens as camera feature maps. |
| `opv2v/opencood/models/vjepa_single.py` | Single-CAV V-JEPA BEV projector and baseline. |
| `opv2v/opencood/data_utils/datasets/basedataset.py` | Current-frame-first temporal queue support. |
| `opv2v/opencood/data_utils/datasets/camera_only/intermediate_fusion_dataset.py` | Multi-CAV camera temporal collation and current-frame calibration handling. |
| `opv2v/opencood/hypes_yaml/opcamera/corpbevt_jepa.yaml` | Main cooperative spatial-token training config. |
| `opv2v/opencood/hypes_yaml/opcamera/corpbevt_jepa_best.yaml` | Tuned/ablation config variant. |

Additional single-CAV and V-JEPA/FAX ablation paths are included in
`fax_jepa.yaml`, `vjepa_single.yaml`, `vjepa_cvt_*.yaml`, and the matching model
files.

## Use In CoBEVT

From a CoBEVT checkout, copy this overlay into the repository root:

```bash
rsync -av cobevt_spatial_token/opv2v/ /path/to/CoBEVT/opv2v/
```

Then adjust the local paths in the YAML files, especially:

```text
root_dir
validate_dir
model.args.jepa_encoder.vjepa2_repo_path
model.args.jepa_encoder.vjepa2_config_path
model.args.jepa_encoder.model_weights
```

Example training command from the CoBEVT `opv2v` directory:

```bash
python opencood/tools/train_camera.py \
  --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt_jepa.yaml
```

## Notes

- Logs, checkpoints, datasets, `TempCoBEV-main`, zip files, and generated C
  extension output were intentionally excluded.
- The copied YAML files still preserve the original machine-local experiment
  paths for traceability; update them before running elsewhere.
- V-JEPA source/config dependencies are already present elsewhere in this
  RepDrive upload repository under `vjepa2-main/`.
