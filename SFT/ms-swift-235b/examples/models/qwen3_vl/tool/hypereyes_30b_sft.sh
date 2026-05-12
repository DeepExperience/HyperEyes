
#!/bin/bash
set -euo pipefail
set -x

########################
# 自动分布式环境设置
# NNODES=$WORLD_SIZE RANK=$RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
########################
# 单机模式检测，如果没有 MASTER_ADDR，则默认为单机
if [ -z "${MASTER_ADDR:-}" ]; then
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=$(shuf -n 1 -i 20000-65000)
    export NNODES=1
    export NODE_RANK=0
    export NPROC_PER_NODE=8
    export WORLD_SIZE=8
else
    # 多机模式，平台需要提供 MASTER_ADDR, RANK, WORLD_SIZE
    export MASTER_ADDR=$MASTER_ADDR \
    export MASTER_PORT=$MASTER_PORT \
    export NNODES=${WORLD_SIZE:-1}       # 平台的WORLD_SIZE是节点数
    export NODE_RANK=${RANK:-0}          # 节点排名
    export RANK=${RANK:-0}
    export NPROC_PER_NODE=8  # 默认每个节点8张卡

fi

# 动态调整批处理大小（可选，根据节点数量调整）
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
# 全局批处理大小应该根据节点数量进行调整
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((8 * NNODES))} 

# 其他配置
export TF_CPP_MIN_LOG_LEVEL=3

echo "=== 分布式配置信息 ==="
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "TOTAL_PROCESSES: $((NNODES * NPROC_PER_NODE))"
echo "MICRO_BATCH_SIZE: ${MICRO_BATCH_SIZE}"
echo "GLOBAL_BATCH_SIZE: ${GLOBAL_BATCH_SIZE}"
echo "========================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO


export WANDB_NAME=HyperEyes_30k_LoraV1.6.3_spV3_wo_tail_30B
export https_proxy=10.7.4.2:3128
export WANDB_API_KEY="xxxx"
export WANDB_PROJECT=default
export WANDB_MODE=online


export IMAGE_MAX_TOKEN_NUM=1200

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export MEGATRON_LM_PATH='/mnt/tidalfs-bdsz01/usr/xx/Megatron-LM'


$(which megatron) sft \
    --model /mnt/tidalfs-bdsz01/dataset/llm_ckpt/video_agent/models/Qwen3-VL-30B-A3B-Instruct  \
    --load_safetensors true \
    --save_safetensors true \
    --dataset \
        /mnt/tidalfs-bdsz01/dataset/llm_dataset/video_agent_data/video_agent_sft_dataset_1219/a_HyperEyes_dataset/loraV1.6.3_spV3_parallel_wo_tail/*/*.jsonl \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --moe_permute_fusion true \
    --tensor_model_parallel_size 2 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 2 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap false \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save  /mnt/tidalfs-bdsz01/dataset/llm_ckpt/video_agent/HyperEyes/LoraV1.6.3_spV3_wo_tail/30B \
    --eval_interval 1000 \
    --save_interval 1000 \
    --max_length 32000 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --model_author xx \
    --model_name HyperEyes_38k_LoraV1.6.3_spV3_wo_tail_30B \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_NAME
