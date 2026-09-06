#!/bin/bash

# Activate drivoR environment
source /home/dataset-assist-0/yinhongbo/miniconda3/etc/profile.d/conda.sh
conda activate drivoR

# -----------------
# Environment
# -----------------
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/dataset-assist-0/yinhongbo/datazoo/navsim/maps"
export NAVSIM_EXP_ROOT="/home/dataset-assist-0/yinhongbo/code/robotics"
export NAVSIM_DEVKIT_ROOT="/home/dataset-assist-0/yinhongbo/code/robotics"
export VJEPA2_1_REPO_PATH="$NAVSIM_DEVKIT_ROOT/vjepa2-main"
export OPENSCENE_DATA_ROOT="/home/dataset-assist-0/yinhongbo/datazoo/navsim"
export TMPDIR="/home/dataset-local/yinhongbo/tmp"
export RAY_TMPDIR="/home/dataset-local/yinhongbo/ray_tmp"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

unset RAY_ADDRESS
unset ip_head
unset redis_password
unset num_nodes
unset RAY_HEAD_SERVICE_HOST
unset RAY_HEAD_SERVICE_PORT

set -euo pipefail

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

EXPERIMENT=training_drivejepa3_stage2_interaction_v8_amp_scorer_fix
AGENT=drivejepa3
STAGE1_CKPT=/home/dataset-assist-0/yinhongbo/code/robotics/ke/training_drivejepa2_4cam_bev_agent_scene_fix/06.09_16.17/lightning_logs/version_0/checkpoints/best-epoch=18-step=7657.ckpt
STAGE1_CKPT_HYDRA=${STAGE1_CKPT//=/\\=}
RUN_UID=$(date +%m.%d_%H.%M)
OUTPUT_DIR=/home/dataset-local/yinhongbo/drivejepa3_stage2_interaction_v8_amp_scorer_fix/${RUN_UID}
TRAIN_LOG=${OUTPUT_DIR}/run_training_full.log
mkdir -p "$OUTPUT_DIR"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    output_dir="$OUTPUT_DIR" \
    train_ckpt_path=null \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=10 \
    trainer.params.gradient_clip_val=1.0 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.num_workers=8 \
    dataloader.params.batch_size=64 \
    agent.checkpoint_path="$STAGE1_CKPT_HYDRA" \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.stage2_bridge.freeze_stage1=false \
    agent.config.stage2_bridge.freeze_stage1_except_scorer=true \
    agent.config.stage2_bridge.freeze_image_backbone=true \
    agent.config.stage2_bridge.freeze_proposal_generator=true \
    agent.config.stage2_bridge.detach_initial_query=true \
    agent.config.stage2_bridge.scorer_lr_multiplier=1.0 \
    agent.config.stage2_bridge.scorer_weight_decay=0.0 \
    agent.config.decoder_detach_between_rounds=true \
    agent.config.decoder_reset_query_each_round=true \
    agent.config.decoder_residual_init_std=0.0 \
    agent.config.decoder_attention_dropout=0.0 \
    agent.config.decoder_scene_residual_dropout=0.0 \
    agent.config.decoder_bev_residual_dropout=0.1 \
    agent.config.decoder_agent_residual_dropout=0.0 \
    agent.config.decoder_ffn_dropout=0.0 \
    agent.config.decoder_dropout=0.0 \
    agent.config.interaction_score_residual_scale=0.0 \
    agent.config.decoder_intermediate_loss_weight=0.1 \
    agent.config.score_threads_per_rank=16 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    seed=2 > "$TRAIN_LOG" 2>&1
