#!/bin/bash

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-$((NUM_NODES * ROLLOUT_GPUS_PER_NODE))}
SGLANG_GPUS_PER_ENGINE=${SGLANG_GPUS_PER_ENGINE:-8}

if (( ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE > NUM_GPUS_PER_NODE )); then
    echo "ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE must be <= NUM_GPUS_PER_NODE"
    echo "ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE}, ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE}, NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="${ROOT_DIR}/slime"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${ROOT_DIR}/Megatron-LM"}
if [[ ! -d "${MEGATRON_LM_PATH}" ]]; then
    echo "MEGATRON_LM_PATH does not exist: ${MEGATRON_LM_PATH}"
    exit 1
fi

# Do not mix the bridge checkout's bundled Megatron with this script.
# For text-only retool we intentionally use the repo's Megatron-LM tree.
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"

HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/Qwen35-27B}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-/data_storage/wyj/OpenClaw-RL/ckpt/qwen35-27b-retool-rl/}
RESUME_LOAD=${RESUME_LOAD:-${SAVE_CKPT}}
PROMPT_DATA=${PROMPT_DATA:-/data_storage/wyj/OpenClaw-RL1/data/dapo-math-17k/dapo-math-17k.jsonl}
EVAL_DATA=${EVAL_DATA:-/data_storage/wyj/OpenClaw-RL1/data/aime-2024/aime-2024.jsonl}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-16384}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-16384}
EVAL_MAX_CONTEXT_LEN=${EVAL_MAX_CONTEXT_LEN:-16384}

export SGLANG_LANGUAGE_ONLY="${SGLANG_LANGUAGE_ONLY:-1}"
export SLIME_QWEN35_TEXT_ONLY_BRIDGE="${SLIME_QWEN35_TEXT_ONLY_BRIDGE:-1}"

CKPT_ARGS=(
   --megatron-to-hf-mode bridge
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 20
   --rotary-base 10000000
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-temperature 1
   --num-steps-per-rollout 2
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${EVAL_DATA}"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
   --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
   --eval-top-p 1
   --eval-reward-key acc
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --log-probs-chunk-size 512
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

if [[ -n "${WANDB_KEY:-}" ]]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-slime_retool}"
      --wandb-group "${WANDB_GROUP:-qwen35-27B-rl_retool-4nodes}"
      --wandb-key "${WANDB_KEY}"
   )
else
   WANDB_ARGS=()
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${SGLANG_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static 0.8
)

if [[ "${SGLANG_LANGUAGE_ONLY}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-language-only)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
_WORKER_IP_VAR="MLP_WORKER_${MLP_ROLE_INDEX}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}, MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray start --head --node-ip-address "${NODE_IP}" --num-gpus "${NUM_GPUS_PER_NODE}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
  sleep 30
  ray start --address="${MASTER_ADDR}:6379" --num-gpus "${NUM_GPUS_PER_NODE}" --node-ip-address "${NODE_IP}"
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"FLASHINFER_WORKSPACE_BASE\": \"${FLASHINFER_WORKSPACE_BASE}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"SLIME_QWEN35_TEXT_ONLY_BRIDGE\": \"${SLIME_QWEN35_TEXT_ONLY_BRIDGE}\",
    \"MEGATRON_LM_PATH\": \"${MEGATRON_LM_PATH}\",
    \"HF_CKPT\": \"${HF_CKPT}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray job submit --address="http://${MASTER_ADDR}:8265" \
     --runtime-env-json="${RUNTIME_ENV_JSON}" \
     -- python3 "${SLIME_DIR}/train_async.py" \
     --actor-num-nodes "${NUM_NODES}" \
     --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
     --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}" \
     ${MODEL_ARGS[@]} \
     ${CKPT_ARGS[@]} \
     ${ROLLOUT_ARGS[@]} \
     ${OPTIMIZER_ARGS[@]} \
     ${GRPO_ARGS[@]} \
     ${WANDB_ARGS[@]} \
     ${PERF_ARGS[@]} \
     ${EVAL_ARGS[@]} \
     ${SGLANG_ARGS[@]} \
     ${MISC_ARGS[@]} \
     ${CUSTOM_ARGS[@]}
else
  echo "Worker node ${MLP_ROLE_INDEX} joined the cluster. Waiting for job to finish..."
  while ray status > /dev/null 2>&1; do
    sleep 60
  done
  echo "Ray cluster stopped. Worker node ${MLP_ROLE_INDEX} exiting."
fi
