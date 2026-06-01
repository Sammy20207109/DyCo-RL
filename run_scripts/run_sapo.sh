#!/bin/bash
# export PET_NNODES=1
# export PET_NODE_RANK=0
# export PET_MASTER_ADDR=127.0.0.1
# export PET_MASTER_PORT=29500
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"

# ============ 从PET环境变量读取配置 ============
export MASTER_ADDR="${PET_MASTER_ADDR}"
export MASTER_PORT="${PET_MASTER_PORT}"
export NNODES="${PET_NNODES}"
export NODE_RANK="${PET_NODE_RANK}"
export GPUS_PER_NODE="8"

echo "================================================"
echo "PET 环境变量："
echo "  PET_NNODES: ${PET_NNODES}"
echo "  PET_NODE_RANK: ${PET_NODE_RANK}"
echo "  PET_MASTER_ADDR: ${PET_MASTER_ADDR}"
echo "  PET_MASTER_PORT: ${PET_MASTER_PORT}"
echo "================================================"

if [ -z "$PET_MASTER_ADDR" ] || [ -z "$PET_MASTER_PORT" ] || [ -z "$PET_NNODES" ] || [ -z "$PET_NODE_RANK" ]; then
    echo "❌ 错误：PET 环境变量未设置！"
    exit 1
fi

# ============ 网络配置 ============
NETIF=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' || echo "eth0")
echo "使用网卡: ${NETIF}"

export NCCL_SOCKET_IFNAME=${NETIF}
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

export TORCH_DISTRIBUTED_BACKEND=nccl
export ACCELERATE_TORCH_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export GLOO_SOCKET_IFNAME=${NETIF}
export GLOO_DEVICE_TRANSPORT=TCP

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export RANK=$((NODE_RANK * GPUS_PER_NODE))

# ============ 离线模式 ============
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export DO_NOT_TRACK=1

# ============ 数据和模型路径 ============

data_paths="/volume/med-train/users/shuyan/lhg/dataset/ThinkLite-VL-hard-11k/annotations.jsonl"

image_folders="/volume/med-train/users/shuyan/lhg/dataset/"
model_path="/volume/med-train/users/shuyan/lhg/VLM-R1/Qwen2.5-VL-3B-Instruct"
is_reward_customized_from_vlm_module=True

export EXP_NAME="DyCo-RL_SAPO_Qwen2.5-VL-3B"
echo "exp_name:$EXP_NAME"
TASK_TYPE="general"

# ============ 日志配置 ============
cd ${REPO_HOME}/src/open-r1-multimodal/src
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/node${NODE_RANK}_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
export DEBUG_MODE="true"

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
#MAX_STEPS=600 # TODO: change this to your own max steps

export WANDB_MODE=offline
#export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=3,4
export PYTHONPATH="${REPO_HOME}/src/open-r1-multimodal/src:$PYTHONPATH"

PER_DEVICE_BS=4
GRAD_ACCUM=2
NUM_GENERATIONS=4  

GLOBAL_BS=$((NNODES * GPUS_PER_NODE * PER_DEVICE_BS * GRAD_ACCUM))
echo "================================================"
echo "Batch Size 配置："
echo "  Per device batch size: ${PER_DEVICE_BS}"
echo "  Gradient accumulation: ${GRAD_ACCUM}"
echo "  Num generations: ${NUM_GENERATIONS}"
echo "  Global batch size: ${GLOBAL_BS}"
echo "  检查: ${GLOBAL_BS} % ${NUM_GENERATIONS} = $((GLOBAL_BS % NUM_GENERATIONS))"
if [ $((GLOBAL_BS % NUM_GENERATIONS)) -eq 0 ]; then
    echo "  ✅ 配置有效"
else
    echo "  ❌ 配置无效，全局batch size必须能被num_generations整除"
    exit 1
fi
echo "================================================"

echo "日志文件: ${LOG_PATH}"
echo "等待3秒后启动训练..."
sleep 3

# ============ 启动训练 ============
torchrun \
    --nproc-per-node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node-rank=${NODE_RANK} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    open_r1/grpo_jsonl.py \
        --use_vllm False \
        --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
        --resume_from_checkpoint True \
        --model_name_or_path $model_path \
        --data_file_paths $data_paths \
        --image_folders $image_folders \
        --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
        --task_type $TASK_TYPE \
        --max_anyres_num 6 \
        --per_device_train_batch_size ${PER_DEVICE_BS} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --gradient_checkpointing true \
        --logging_steps 1 \
        --num_train_epochs 1 \
        --bf16 \
        --attn_implementation flash_attention_2 \
        --run_name ${EXP_NAME} \
        --data_seed 42 \
        --save_steps 1000 \
        --num_generations ${NUM_GENERATIONS} \
        --max_completion_length 2048 \
        --reward_funcs accuracy format \
        --beta 0.001 \
        --loss_type sapo \
        --sapo_temperature_pos 1.0 \
        --sapo_temperature_neg 1.05 \
        --importance_sampling_level token \
        --mask_truncated_completions True \
        --report_to wandb \
        --dataset-name this_is_not_used \
        --deepspeed /volume/med-train/users/shuyan/lhg/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json\
        --max_pixels 802816 \
    2>&1 | tee ${LOG_PATH}

echo "✅ 节点 ${NODE_RANK} 训练完成"




