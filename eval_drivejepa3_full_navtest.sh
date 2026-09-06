#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CHECKPOINT [EXPERIMENT_NAME]" >&2
    exit 2
fi

source /home/dataset-assist-0/yinhongbo/miniconda3/etc/profile.d/conda.sh
conda activate drivoR

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/dataset-assist-0/dataset/navsimv1_full/maps"
export NAVSIM_EXP_ROOT="/home/dataset-assist-0/yinhongbo/code/robotics"
export NAVSIM_DEVKIT_ROOT="/home/dataset-assist-0/yinhongbo/code/robotics"
export OPENSCENE_DATA_ROOT="/home/dataset-assist-0/dataset/navsimv1_full/"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

CHECKPOINT=$1
EXPERIMENT=${2:-drivejepa3_interaction_v3_full_navtest_$(date +%m%d_%H%M)}
EXTRA_OVERRIDES=("${@:3}")
NOC_WEIGHT=${NOC_WEIGHT:-1}
DAC_WEIGHT=${DAC_WEIGHT:-1}
DDC_WEIGHT=${DDC_WEIGHT:-0.0}
TTC_WEIGHT=${TTC_WEIGHT:-5}
EP_WEIGHT=${EP_WEIGHT:-5}
COMFORT_WEIGHT=${COMFORT_WEIGHT:-2}
METRIC_CACHE_PATH="/home/dataset-local/yinhongbo/code/robotics/test/DrivoR-main/metric_cache"
LOG_DIR="/home/dataset-local/yinhongbo/drivejepa3_eval_logs"
LOG_PATH="${LOG_DIR}/${EXPERIMENT}.log"
mkdir -p "$LOG_DIR"

python -u "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    train_test_split=navtest \
    agent=drivejepa3 \
    "agent.checkpoint_path='$CHECKPOINT'" \
    experiment_name="$EXPERIMENT" \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.noc="$NOC_WEIGHT" \
    agent.config.dac="$DAC_WEIGHT" \
    agent.config.ddc="$DDC_WEIGHT" \
    agent.config.ttc="$TTC_WEIGHT" \
    agent.config.ep="$EP_WEIGHT" \
    agent.config.comfort="$COMFORT_WEIGHT" \
    metric_cache_path="$METRIC_CACHE_PATH" \
    "${EXTRA_OVERRIDES[@]}" > "$LOG_PATH" 2>&1

echo "$LOG_PATH"
