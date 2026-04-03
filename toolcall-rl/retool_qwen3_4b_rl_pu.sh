#!/bin/bash

# Adapted for puyuan's 4-GPU environment
# Based on retool_qwen3_4b_rl.sh + run_qwen3_4b_openclaw_combine_pu.sh

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# keep stdout/stderr unbuffered in ray jobs
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ============================================================================
# GPU Layout: 4 GPUs total
#   Actor:   2 GPUs (TP=2)
#   Rollout: 1 GPU  (sglang engine)
#   PRM:     1 GPU  (optional, comment out PRM section to disable)
# ============================================================================
NUM_GPUS=4
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
PRM_GPUS=${PRM_GPUS:-1}

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${PRM_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

# Increase Ray heartbeat/health-check timeouts to reduce false node failures under heavy init.
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MEGATRON_LM_PATH="${SCRIPT_DIR}/../Megatron-LM"

source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# ============================================================================
# Paths: model, data, checkpoint (adapted from pu env)
# ============================================================================
# NOTE: Using base Qwen3-4B since retool-sft checkpoint is not available locally.
#       For best results, first run retool_qwen3_4b_sft.sh to produce the SFT model,
#       then update HF_CKPT/REF_LOAD below.
HF_CKPT=${HF_CKPT:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B/}
REF_LOAD=${REF_LOAD:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B_torch_dist/}
SAVE_CKPT=${SAVE_CKPT:-/mnt/shared-storage-user/puyuan/code/OpenClaw-RL/toolcall-rl/ckpt/qwen3-4b-retool-rl/}
RESUME_LOAD=${RESUME_LOAD:-${SAVE_CKPT}}

# PRM uses same model (self-PRM)
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}

# Data paths (already downloaded in slime/)
PROMPT_DATA="/mnt/shared-storage-user/puyuan/code/slime/dapo-math-17k/dapo-math-17k.jsonl"
EVAL_DATA="/mnt/shared-storage-user/puyuan/code/slime/aime-2024/aime-2024.jsonl"

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_LOAD}
   --load ${RESUME_LOAD}
   --save ${SAVE_CKPT}
   --save-interval 20
   --rotary-base 1000000  # Qwen3-4B base uses 1000000 (retool-sft/Thinking uses 5000000)
)

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 16       # 32->16 for 4-GPU
   --n-samples-per-prompt 4      # 8->4 for 4-GPU memory
   --rollout-max-response-len 8192
   --rollout-max-context-len 16384
   --rollout-temperature 1

   --num-steps-per-rollout 2
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${EVAL_DATA}
   --n-samples-per-eval-prompt 8  # 16->8 for 4-GPU
   --eval-max-response-len 16384
   --eval-max-context-len 32768
   --eval-top-p 1
   --eval-reward-key acc
)

PERF_ARGS=(
   --tensor-model-parallel-size 2  # 4->2 for 4-GPU
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type k3
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1  # 2->1 for 4-GPU (only 1 rollout GPU)
   --sglang-mem-fraction-static 0.6
)

PRM_ARGS=(
   --prm-enable
   --prm-num-gpus "${PRM_GPUS}"
   --prm-num-gpus-per-engine 1
   --prm-model-path "${PRM_MODEL_PATH}"
   --prm-m "${PRM_M:-1}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-8192}"
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============================================================================
# W&B (optional)
# ============================================================================
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-slime_retool}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group qwen3-4B-rl_retool
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

# ============================================================================
# Launch
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_GPUS} \
   --rollout-num-gpus ${ROLLOUT_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]}
