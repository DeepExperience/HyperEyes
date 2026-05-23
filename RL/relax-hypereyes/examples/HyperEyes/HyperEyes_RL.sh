
set -ex
set -o pipefail

# 初始化clearML
cat > ~/clearml.conf << EOF
api {
  web_server: xxx
  api_server: xxxx
  files_server: xxxxx
  credentials {
    "access_key" = "xxx"
    "secret_key" = "xxxx"
  }
  http {
     retries {
        total: 5
        connect: 5
        read: 5
        redirect: 5
        status: 5
        backoff_factor: 1.0
        backoff_max: 120.0
      }
  }
}
EOF

clearml-init

###############################################################################
#                                 ENVIRONMENT                                 #
###############################################################################

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}/../.."
RELAX_ROOT="${WORKSPACE_ROOT}/Relax"

# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    export RELAX="${RELAX_ROOT}"
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi

export PYTHONPATH=${WORKSPACE_ROOT}:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"

NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${WORLD_SIZE:-1}"
TOTAL_GPUS=$((NUM_GPUS * NNODES))

export GEMINI_API_KEY="xxx"
export GEMINI_FLASH_URL="xxx"

export MAX_TOOL_CALLS_NUM=${MAX_TOOL_CALLS_NUM:-"8"}
export MAX_ITERATIONS=${MAX_ITERATIONS:-"9"}
export RAG_MAX_CONCURRENCY=64

###############################################################################
#                                    DIRS                                     #
###############################################################################

export QS_USER=${QS_USER:-"xxx"}

PROJECT_NAME="xxx"

# EXP_NAME 支持外部传入，保持固定名字才能自动续训
# 首次运行不传则自动生成带 timestamp 的名字；中断后重跑传同一个名字即可续训
EXP_NAME="xxxxx"

export MODEL_DIR=xxxx
export SAVE_DIR=xxxx
DATSET_DIR="xxx" # epoch1 -> epoch2: updated data(with more efficient toolcall reference by RL-epoch1-ckpt)

mkdir -p ${SAVE_DIR}

# 动态 Baseline Cache 路径
export RL_BASELINE_CACHE_DIR="${SAVE_DIR}/${EXP_NAME}/rollout_result/train"
mkdir -p ${RL_BASELINE_CACHE_DIR}

# 把 agent 特有的环境变量追加到 RUNTIME_ENV_JSON 中
if [ -n "${RUNTIME_ENV_JSON:-}" ]; then
    RUNTIME_ENV_JSON=$(python3 -c "
import json, sys
cfg = json.loads(sys.argv[1])
cfg.setdefault('env_vars', {}).update({
    'PYTHONPATH': '${PYTHONPATH}',
    'GEMINI_API_KEY': '${GEMINI_API_KEY}',
    'GEMINI_FLASH_URL': '${GEMINI_FLASH_URL}',
    'MAX_TOOL_CALLS_NUM': '${MAX_TOOL_CALLS_NUM}',
    'MAX_ITERATIONS': '${MAX_ITERATIONS}',
    'RAG_MAX_CONCURRENCY': '${RAG_MAX_CONCURRENCY}',
    'RL_BASELINE_CACHE_DIR': '${RL_BASELINE_CACHE_DIR}'
})
print(json.dumps(cfg))
" "$RUNTIME_ENV_JSON")
    export RUNTIME_ENV_JSON
fi

# 打印实验名，方便记录（中断后续训需要传入此名字）
echo "================================================================"
echo "  EXP_NAME = ${EXP_NAME}"
echo "  SAVE/LOAD = ${SAVE_DIR}/${EXP_NAME}"
# echo "  如需续训，请运行:"
# echo "    EXP_NAME=\"${EXP_NAME}\" bash examples/HyperEyes/0425_resume_HyperEyes_OPD_update.sh"
echo "================================================================"

###############################################################################
#                                  MODEL CONFIG                               #
###############################################################################

source "${MODEL_CONFIG_DIR}/qwen3-vl-30B-A3B.sh"

CKPT_PATH="${CKPT_PATH:-"xxx"}"

# --load 与 --save 指向同一目录，实现自动断点续训：
#   目录为空或不存在 → 从 --hf-checkpoint 加载（首次）
#   目录有 Megatron checkpoint → 从断点恢复（续训）
# checkpoint 里没有 optimizer 状态（--no-save-optim），必须保留此参数

CKPT_ARGS=(
    --hf-checkpoint ${CKPT_PATH}
    --ref-load ${CKPT_PATH}
    
    --load ${SAVE_DIR}/${EXP_NAME}
    --save ${SAVE_DIR}/${EXP_NAME}
    --megatron-to-hf-mode bridge
    --save-interval 20
    --max-actor-ckpt-to-keep 8
    --no-load-optim
    --no-load-rng
    --no-save-optim
    --no-save-rng
)

###############################################################################
#                                  DATASETS                                   #
###############################################################################

TRAIN_GLOB_PATTERNS=(
    "${DATSET_DIR}/*.parquet"
)

TEST_FILE="xxxx.parquet"

_build_prompt_set() {
    local result=""
    for pattern in "${TRAIN_GLOB_PATTERNS[@]}"; do
        for f in $pattern; do
            [ -f "$f" ] || continue
            [ -n "$result" ] && result="${result},"
            result="${result}${f}"
        done
    done
    echo "[${result}]"
}
PROMPT_SET=$(_build_prompt_set)

###############################################################################
#                               ROLLOUT CONFIG                                #
###############################################################################

RESPONSE_LENGTH=${RESPONSE_LENGTH:-1524}
NUM_ROLLOUT="${NUM_ROLLOUT:=400}"

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key reward_model
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num 1200
    --reward-key score
    --metadata-key extra_info
    --apply-chat-template
    --custom-generate-function-path examples.deepeyes.rollout.generate
    --custom-rm-path examples.HyperEyes.HyperEyes_reward_V1_2_w_dynamic_w_OPD_v0.reward_func
    --custom-config-path ${WORKSPACE_ROOT}/examples/HyperEyes/agent_config.yaml
    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size 32
    --micro-batch-size 1
    --n-samples-per-prompt 8
    --rollout-max-response-len ${RESPONSE_LENGTH}
    --rollout-max-prompt-len 15000
    --rollout-temperature 0.8
    --rollout-top-p 0.95
    --global-batch-size 256
    --use-fault-tolerance
    --rollout-shuffle
    --group-rm
    --balance-data
)

###############################################################################
#                                EVAL CONFIG                                  #
###############################################################################

EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data HyperEyes_test ${TEST_FILE}
    --n-samples-per-eval-prompt 1
    --eval-max-response-len ${RESPONSE_LENGTH}
    --eval-top-p 0.7
    --skip-eval-before-train
)

###############################################################################
#                              ALGORITHM CONFIG                               #
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --eps-clip-c 3
    --use-tis
)

###############################################################################
#                              OPTIMIZER CONFIG                               #
###############################################################################

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

###############################################################################
#                               SGLANG CONFIG                                 #
###############################################################################

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 4
    --sglang-mem-fraction-static 0.5
    --sglang-max-running-requests 64
    --sglang-enable-chunked-prefill
    --sglang-max-num-batched-tokens 132144
    --sglang-limit-images 50
)

###############################################################################
#                               LOGGING CONFIG                                #
###############################################################################

LOG_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
)

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################

PP_SIZE=$((TOTAL_GPUS / 8))
[ "$PP_SIZE" -lt 1 ] && PP_SIZE=1

MEGATRON_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size ${PP_SIZE}
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --max-tokens-per-gpu 9216
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-dynamic-batch-size
    --checkpoint-mem-efficient
    --no-rope-fusion
    # --freeze-vision-model
    --no-pin-cpu-grads
    --no-pin-cpu-params
)

###############################################################################
#                                OPD CONFIG                                   #
###############################################################################

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.05
   --rm-url xxx
   --opd-log-prob-top-k 20
   --opd-teacher-timeout-s 500
   --opd-teacher-connector-limit 32     # ← 默认 256 → 32, 减少同时打 teacher 的请求
)

###############################################################################
#                                 LAUNCH JOB                                  #
###############################################################################

mkdir -p logs

ray job submit --address=${RAY_ADDRESS:-"http://127.0.0.1:8265"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${RELAX_ROOT}/relax/entrypoints/train.py" \
    --resource "{\"actor\": [1, ${TOTAL_GPUS}], \"rollout\": [1, ${TOTAL_GPUS}]}" \
    --max-staleness 0 \
    --num-data-storage-units 2 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    2>&1 | tee logs/${EXP_NAME}.log
