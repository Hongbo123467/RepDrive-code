#!/bin/bash

# 激活 drivoR 虚拟环境
source /home/dataset-assist-0/yinhongbo/miniconda3/etc/profile.d/conda.sh
conda activate drivoR

# 如果您的服务器使用了 module 系统，请取消注释以下行加载相应的环境依赖
# module load Ninja/1.11.1-GCCcore-12.2.0
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1
# module load GCC/12.2.0

# -----------------
# 环境变量配置
# -----------------
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# 请根据实际存放位置修改以下路径，这里我帮您填入了当前项目相关的参考路径
export NUPLAN_MAPS_ROOT="/home/dataset-assist-0/yinhongbo/datazoo/navsim/maps"
export NAVSIM_EXP_ROOT="/home/dataset-assist-0/yinhongbo/code/robotics"
export NAVSIM_DEVKIT_ROOT="/home/dataset-assist-0/yinhongbo/code/robotics"
export VJEPA2_1_REPO_PATH="$NAVSIM_DEVKIT_ROOT/vjepa2-main"
export OPENSCENE_DATA_ROOT="/home/dataset-assist-0/yinhongbo/datazoo/navsim"
export TMPDIR="/home/dataset-local/yinhongbo/tmp"
export RAY_TMPDIR="/home/dataset-local/yinhongbo/ray_tmp"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

# -----------------
# 清除 Ray 相关环境变量（防止连接到旧集群导致 GCS 启动失败）
# -----------------
unset RAY_ADDRESS
unset ip_head
unset redis_password
unset num_nodes
unset RAY_HEAD_SERVICE_HOST
unset RAY_HEAD_SERVICE_PORT

# -----------------
# 训练参数
# -----------------
set -euo pipefail

export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

EXPERIMENT=training_drivejepa2_4cam_bev_agent_scene_fix
AGENT=drivejepa2

# 执行训练脚本 (NAVSIM-v1)
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py  \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_ckpt_path=null \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=25 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.num_workers=8 \
    dataloader.params.batch_size=32 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=8 \
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
    agent.loss.prev_weight=0.0 \
    agent.loss.agent_class_weight=1.0 \
    agent.loss.agent_box_weight=1.0 \
    agent.config.long_trajectory_additional_poses=2 \
    seed=2 > train_drivejepa2_4cam_bev_agent_scene_fix.log 2>&1
