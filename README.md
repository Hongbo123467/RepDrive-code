# DriveJEPA3

DriveJEPA3 is a two-stage camera-only planning model built on the DrivoR/NAVSIM codebase. The first stage generates multi-modal trajectory proposals from multi-camera V-JEPA scene tokens. The second stage builds a BEV feature with LSS-style projection and learns residual corrections for the proposals.

The intended model behavior is:

```text
4 cameras x 2 frames
  -> V-JEPA 2.1 scene_lora encoder
  -> scene tokens + ego status
  -> raw trajectory proposals [B, 64, 8, 3]
  -> V-JEPA spatial features -> FPN -> LSS BEV feature
  -> BEV-conditioned residual proposal refiner
  -> refined trajectory proposals [B, 64, 8, 3]
  -> DrivoR scorer on refined proposals
  -> final selected trajectory [B, 8, 3]
```

## Main Idea

DriveJEPA3 uses a conservative two-stage design:

1. **Stage 1: proposal generation**
   - V-JEPA 2.1 with `scene_lora` produces per-camera scene tokens.
   - The proposal decoder generates 64 candidate trajectories.
   - These raw proposals are treated as the learned trajectory prior.

2. **Stage 2: residual refinement**
   - V-JEPA multi-scale spatial features are fused by an FPN adapter.
   - The fused image features are projected into BEV by an LSS-style projector.
   - A PAD/VeteranAD-style decoder predicts BEV semantic / agent outputs and residual trajectory corrections.
   - The refined proposals are scored by the DrivoR scorer.

The residual form is:

```text
Y_refined = Y_raw + DeltaY
```

The residual heads are zero-initialized, so the stage2 model starts from approximately the stage1 behavior:

```text
Y_refined ~= Y_raw
```

## Core Files

These files define the current DriveJEPA3 model path.

| File | Purpose |
| --- | --- |
| `navsim/agents/drivoR/drivejepa3_agent.py` | NAVSIM agent wrapper, checkpoint loading, feature/target builders, scorer resource setup, optimizer setup. |
| `navsim/agents/drivoR/drivejepa3_model.py` | Main DriveJEPA3 network: V-JEPA encoder, raw proposal generation, BEV projection, residual refinement, scorer on refined proposals. |
| `navsim/agents/drivoR/drivejepa3_decoder.py` | BEV decoder, BEV semantic head, agent detection head, and residual proposal refiner. |
| `navsim/agents/drivoR/layers/losses/drivejepa3_loss.py` | DrivoR-style loss using refined proposals for trajectory and scorer supervision. |
| `navsim/planning/script/config/common/agent/drivejepa3.yaml` | Hydra configuration for DriveJEPA3 stage2 training. |
| `train_jepa3_stage2.sh` | Stage2 training script using a pretrained stage1 checkpoint. |

## Stage1 / Shared Support Files

DriveJEPA3 reuses the DriveJEPA2 feature path and stage1 proposal stack. These files should be included when uploading the model code.

| File | Purpose |
| --- | --- |
| `navsim/agents/drivoR/drivejepa2_agent.py` | DriveJEPA2 agent wrapper used for stage1 training. |
| `navsim/agents/drivoR/drivejepa2_model.py` | Stage1 V-JEPA proposal model with 4-camera scene embeddings. |
| `navsim/agents/drivoR/drivejepa2_features.py` | Four-camera, two-frame feature builder. Produces `camera_feature_1`, `camera_feature_2`, intrinsics, extrinsics, and ego status. |
| `navsim/planning/script/config/common/agent/drivejepa2.yaml` | Hydra config for DriveJEPA2 stage1 training. |
| `train_jepa_v2.1_fpn.sh` | Stage1 training script for the current 4-camera DriveJEPA2 setup. |

## Shared Model Components

| File | Purpose |
| --- | --- |
| `navsim/agents/drivoR/drivejepa2_bevformer/fpn_adapter.py` | Fuses V-JEPA multi-layer spatial features into one image feature map. |
| `navsim/agents/drivoR/drivejepa2_bevformer/lss_bev.py` | LSS-style camera-to-BEV projection and BEV semantic head. |
| `navsim/agents/drivoR/layers/image_encoder/vjepa2_1_lora.py` | V-JEPA 2.1 LoRA / scene-register-token image encoder. |
| `navsim/agents/drivoR/layers/image_encoder/vjepa2_lora.py` | V-JEPA 2.0 LoRA fallback and shared LoRA utilities. |
| `navsim/agents/drivoR/layers/image_encoder/grid_mask.py` | Grid mask augmentation used by the V-JEPA encoders. |
| `navsim/agents/drivoR/transformer_decoder.py` | Proposal decoder and scorer attention blocks. |
| `navsim/agents/drivoR/layers/utils/mlp.py` | MLP utility used by trajectory heads and scorer components. |
| `navsim/agents/drivoR/score_module/scorer.py` | DrivoR PDM-style scorer heads for trajectory candidates. |
| `navsim/agents/drivoR/score_module/compute_navsim_score.py` | NAVSIM metric-cache-based score computation used for scorer supervision. |
| `navsim/agents/drivoR/layers/losses/drivor_loss.py` | Base DrivoR loss utilities, including scorer loss and agent Hungarian loss. |
| `navsim/agents/drivoR/drivor_features.py` | Target builder for trajectories, BEV semantic targets, and agent targets. |

## Configuration Summary

Current DriveJEPA3 settings are defined in:

```text
navsim/planning/script/config/common/agent/drivejepa3.yaml
```

Important settings:

```yaml
config:
  cam_f0: [2,3]
  cam_l0: [2,3]
  cam_r0: [2,3]
  cam_b0: [2,3]
  num_scene_tokens: 16
  proposal_num: 64
  num_poses: 8
  tf_d_model: 256
  tf_d_ffn: 1024

  image_backbone:
    vjepa_version: "2.1"
    image_architecture: vit_large
    backbone_mode: "scene_lora"
    lora_rank: 32
    fpn_layer_indices: [5, 11, 17, 23]
    spatial_hw: [16, 32]

  stage2_bridge:
    freeze_image_backbone: true

  decoder_ref_num: 2
  decoder_residual_scale: 1.0
```

`backbone_mode` remains `scene_lora`, but `stage2_bridge.freeze_image_backbone: true` freezes the V-JEPA image backbone and scene embeddings during stage2 training. This preserves stage1 proposal behavior while training the BEV/refinement path.

## Training

### Stage1: DriveJEPA2 proposal model

Train the stage1 4-camera proposal generator:

```bash
cd /home/dataset-assist-0/yinhongbo/code/robotics
bash train_jepa_v2.1_fpn.sh
```

The resulting checkpoint should have four-camera scene embeddings:

```text
_drivor_model.scene_embeds: [1, 4, 16, 1024]
```

This matters because DriveJEPA3 expects one scene-token bank per camera. A checkpoint with `[1, 1, 16, 1024]` was produced by the older shared-scene-token version and will not load into the current DriveJEPA3 without conversion.

### Stage2: DriveJEPA3 residual refiner

Update `STAGE1_CKPT` in:

```text
train_jepa3_stage2.sh
```

Then run:

```bash
cd /home/dataset-assist-0/yinhongbo/code/robotics
bash train_jepa3_stage2.sh
```

The script uses:

```text
AGENT=drivejepa3
EXPERIMENT=training_drivejepa4_stage2_ar_refiner
```

The experiment name is historical. The agent itself is DriveJEPA3.

## Losses

DriveJEPA3 uses `DriveJEPA3Loss`, which inherits the DrivoR loss style but applies trajectory and scorer supervision to refined proposals.

The relevant fields from model output are:

```text
raw_proposals       stage1 proposals before residual refinement
raw_proposal_list   intermediate stage1 proposal list
refined_proposals   final residual-refined proposals
proposal_list       residual refinement steps used for trajectory loss
pdm_score           scorer output over refined proposals
trajectory          final selected refined trajectory
bev_semantic_map    BEV semantic prediction
agent_states        agent box/state prediction
agent_labels        agent objectness logits
```

The high-level loss is:

```text
loss =
  trajectory_weight * trajectory_loss(refined proposals)
+ final_score_weight * scorer_loss(refined proposals)
+ agent_class_weight * agent_class_loss
+ agent_box_weight * agent_box_loss
+ bev_semantic_weight * bev_semantic_loss
```

The current design intentionally avoids applying the trajectory loss to both raw and refined proposals. The trajectory loss should train the residual refinement path.

## Important Implementation Notes

- `drivejepa2_features.py` uses camera order:

  ```text
  cam_f0, cam_l0, cam_r0, cam_b0
  ```

- The feature builder reads the current frame and previous frame:

  ```text
  camera_feature_1: current frame  [N_cam, C, H, W]
  camera_feature_2: previous frame [N_cam, C, H, W]
  ```

- Images are cropped with `image[28:-28]` and resized to `(512, 256)` in `(W, H)` order.
- ImageNet normalization is applied inside the model forward pass.
- V-JEPA 2.1 uses appended scene-register tokens and returns both scene tokens and multi-scale spatial features.
- The residual proposal heads in `DriveJEPA3ProposalRefiner` are zero-initialized so stage2 initially behaves close to stage1.
- The scorer is applied to `refined_proposals`, not `raw_proposals`.

## Files to Upload to GitHub

Minimum DriveJEPA3 model code:

```text
navsim/agents/drivoR/drivejepa3_agent.py
navsim/agents/drivoR/drivejepa3_model.py
navsim/agents/drivoR/drivejepa3_decoder.py
navsim/agents/drivoR/layers/losses/drivejepa3_loss.py
navsim/planning/script/config/common/agent/drivejepa3.yaml
train_jepa3_stage2.sh
README_DRIVEJEPA3.md
```

Required shared/stage1 code:

```text
navsim/agents/drivoR/drivejepa2_agent.py
navsim/agents/drivoR/drivejepa2_model.py
navsim/agents/drivoR/drivejepa2_features.py
navsim/planning/script/config/common/agent/drivejepa2.yaml
train_jepa_v2.1_fpn.sh
navsim/agents/drivoR/drivejepa2_bevformer/fpn_adapter.py
navsim/agents/drivoR/drivejepa2_bevformer/lss_bev.py
navsim/agents/drivoR/layers/image_encoder/vjepa2_1_lora.py
navsim/agents/drivoR/layers/image_encoder/vjepa2_lora.py
navsim/agents/drivoR/layers/image_encoder/grid_mask.py
navsim/agents/drivoR/transformer_decoder.py
navsim/agents/drivoR/layers/utils/mlp.py
navsim/agents/drivoR/score_module/scorer.py
navsim/agents/drivoR/score_module/compute_navsim_score.py
navsim/agents/drivoR/layers/losses/drivor_loss.py
navsim/agents/drivoR/drivor_features.py
```

Recommended `git add` command:

```bash
git add \
  README_DRIVEJEPA3.md \
  train_jepa3_stage2.sh \
  train_jepa_v2.1_fpn.sh \
  navsim/planning/script/config/common/agent/drivejepa3.yaml \
  navsim/planning/script/config/common/agent/drivejepa2.yaml \
  navsim/agents/drivoR/drivejepa3_agent.py \
  navsim/agents/drivoR/drivejepa3_model.py \
  navsim/agents/drivoR/drivejepa3_decoder.py \
  navsim/agents/drivoR/layers/losses/drivejepa3_loss.py \
  navsim/agents/drivoR/drivejepa2_agent.py \
  navsim/agents/drivoR/drivejepa2_model.py \
  navsim/agents/drivoR/drivejepa2_features.py \
  navsim/agents/drivoR/drivejepa2_bevformer/fpn_adapter.py \
  navsim/agents/drivoR/drivejepa2_bevformer/lss_bev.py \
  navsim/agents/drivoR/layers/image_encoder/vjepa2_1_lora.py \
  navsim/agents/drivoR/layers/image_encoder/vjepa2_lora.py \
  navsim/agents/drivoR/layers/image_encoder/grid_mask.py \
  navsim/agents/drivoR/transformer_decoder.py \
  navsim/agents/drivoR/layers/utils/mlp.py \
  navsim/agents/drivoR/score_module/scorer.py \
  navsim/agents/drivoR/score_module/compute_navsim_score.py \
  navsim/agents/drivoR/layers/losses/drivor_loss.py \
  navsim/agents/drivoR/drivor_features.py
```

Do not upload generated artifacts unless needed for a release:

```text
__pycache__/
*.pyc
*.log
ke/
exp/
lightning_logs/
checkpoints/
*.ckpt
*.pth
datazoo/
modelzoo/
vjepa2-main/   # prefer documenting this as an external dependency
```

## External Dependencies

The code assumes the DrivoR/NAVSIM environment plus:

- PyTorch
- PyTorch Lightning
- Hydra / OmegaConf
- nuPlan / NAVSIM dependencies
- Ray for metric-cache score computation
- V-JEPA 2.1 source tree, referenced by:

  ```text
  config.image_backbone.vjepa2_repo_path
  VJEPA2_1_REPO_PATH
  ```

- V-JEPA 2.1 checkpoint, referenced by:

  ```text
  config.image_backbone.model_weights
  ```

Paths in the local training scripts are machine-specific and should be edited before running on a new host.

## Quick Sanity Checks

Compile the main files:

```bash
python -m py_compile \
  navsim/agents/drivoR/drivejepa3_agent.py \
  navsim/agents/drivoR/drivejepa3_model.py \
  navsim/agents/drivoR/drivejepa3_decoder.py \
  navsim/agents/drivoR/layers/losses/drivejepa3_loss.py \
  navsim/agents/drivoR/drivejepa2_model.py \
  navsim/agents/drivoR/drivejepa2_features.py
```

Check the stage1 checkpoint scene-token shape before stage2 training:

```python
import torch

ckpt = torch.load("PATH/TO/STAGE1.ckpt", map_location="cpu")
state = ckpt["state_dict"]
for key, value in state.items():
    if key.endswith("scene_embeds"):
        print(key, value.shape)
```

Expected:

```text
[1, 4, 16, 1024]
```

