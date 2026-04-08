#!/bin/bash

# SWE-Bench RL training (Qwen3-4B) on 1 node (8 GPUs) with REMOTE Docker exec.
#
# Prerequisites:
#   1. swe_exec_server.py running on the ECS node (or localhost):
#        python3 server/swe_exec_server.py --port 5000 --host 0.0.0.0
#   2. SWE-Bench Docker images pulled on the ECS node
#   3. Set SWE_EXEC_SERVER_URLS in .env.swe or environment
#
# Usage:
#   bash swe-rl/scripts/run_swe_rl_4b_remote_1node.sh
#   DEBUG_MODE=1 bash swe-rl/scripts/run_swe_rl_4b_remote_1node.sh

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
# Kill python processes EXCEPT swe_exec_server to avoid breaking the exec server
pgrep -f python | while read pid; do
  if ! grep -q "swe_exec_server" /proc/$pid/cmdline 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
done
sleep 3
pkill -9 ray || true
pgrep -f python | while read pid; do
  if ! grep -q "swe_exec_server" /proc/$pid/cmdline 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
done

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SWE_RL_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SWE_RL_DIR}/../slime" &>/dev/null && pwd)"
EXPORT_ROOT=${EXPORT_ROOT:-"${SWE_RL_DIR}/../export"}
mkdir -p "${EXPORT_ROOT}/ckpt" "${EXPORT_ROOT}/swe_rollouts"
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs"}
mkdir -p "${LOG_DIR}"
RUN_LOG=${RUN_LOG:-"${LOG_DIR}/run_swe_rl_4b_remote_1node_${RUN_TIMESTAMP}.log"}
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Run log: ${RUN_LOG}"
echo "Run timestamp: ${RUN_TIMESTAMP}"

# ── Model ────────────────────────────────────────────────────────────
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SWE_RL_DIR}/../Megatron-LM"}

# ── Auto-install mini-swe-agent ──────────────────────────────────────
MINISWE_DIR="${SWE_RL_DIR}/mini-swe-agent"
MINISWE_VERSION="v1.12.0"
if ! python3 -c "import minisweagent" 2>/dev/null; then
  if [ ! -d "${MINISWE_DIR}" ]; then
    git clone --branch "${MINISWE_VERSION}" --depth 1 \
      https://github.com/SWE-agent/mini-swe-agent.git "${MINISWE_DIR}"
  fi
  pip install -e "${MINISWE_DIR}"
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MSWEA_DOCKER_EXEC_MODE=api
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

# ── GPU allocation (single node) ─────────────────────────────────────
DEBUG_MODE=${DEBUG_MODE:-0}
if [[ "${DEBUG_MODE}" == "1" ]]; then
  # Debug: auto-detect GPU count, split evenly between actor and rollout
  DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-${DETECTED_GPUS:-4}}
  HALF_GPUS=$(( NUM_GPUS_PER_NODE / 2 ))
  ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-${HALF_GPUS}}
  ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-${HALF_GPUS}}
  ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-${HALF_GPUS}}
  echo "DEBUG_MODE: detected ${DETECTED_GPUS} GPUs, actor=${ACTOR_GPUS_PER_NODE}, rollout=${ROLLOUT_GPUS_PER_NODE}"
else
  NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
  ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
  ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
  ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
fi
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-${ROLLOUT_GPUS_PER_NODE}}

if (( ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE > NUM_GPUS_PER_NODE )); then
  echo "ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE must be <= NUM_GPUS_PER_NODE"
  exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
NODE_IP=${MASTER_ADDR}
export MASTER_ADDR
echo "MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

# ── Remote SWE env pool / exec configs ───────────────────────────────
export SWE_ENV_SERVER_BIND_HOST=${SWE_ENV_SERVER_BIND_HOST:-0.0.0.0}
export SWE_ENV_SERVER_PORT=${SWE_ENV_SERVER_PORT:-18090}
export SWE_ENV_SERVER_HOST=${SWE_ENV_SERVER_HOST:-${MASTER_ADDR}}
export SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL:-http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}}
export SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS:-http://localhost:5000}
if [[ "${DEBUG_MODE}" == "1" ]]; then
  export SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE:-4}
  export SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT:-1}
else
  export SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE:-8}
  export SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT:-2}
fi

ALL_EXEC_HOSTS="$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
export NO_PROXY="localhost,127.0.0.1,${MASTER_ADDR},${ALL_EXEC_HOSTS}"
export no_proxy="${NO_PROXY}"

# ── Start pool server ────────────────────────────────────────────────
SWE_POOL_PID=""
cleanup() {
  set +e
  if [[ -n "${SWE_POOL_PID}" ]] && kill -0 "${SWE_POOL_PID}" 2>/dev/null; then
    kill "${SWE_POOL_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

SWE_POOL_LOG=${SWE_POOL_LOG:-"${LOG_DIR}/swe_env_pool_server_1node.log"}
PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${PYTHONPATH}" \
python3 -m swe_env_pool_server \
  --host "${SWE_ENV_SERVER_BIND_HOST}" \
  --port "${SWE_ENV_SERVER_PORT}" \
  --exec-server-urls "${SWE_EXEC_SERVER_URLS}" \
  --max-containers-per-node "${SWE_MAX_CONTAINERS_PER_NODE}" \
  > "${SWE_POOL_LOG}" 2>&1 &
SWE_POOL_PID=$!
echo "SWE env pool server PID=${SWE_POOL_PID}, log=${SWE_POOL_LOG}"

for i in {1..60}; do
  if curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "SWE env pool server is ready: ${SWE_ENV_SERVER_URL}"
    break
  fi
  sleep 2
done

IFS=',' read -r -a _exec_urls <<< "${SWE_EXEC_SERVER_URLS}"
for exec_url in "${_exec_urls[@]}"; do
  if ! curl -fsS --max-time 8 "${exec_url}/healthz" >/dev/null; then
    echo "ERROR: SWE exec server is not healthy: ${exec_url}/healthz"
    exit 1
  fi
done

# ── Checkpoints (raw mode, pre-converted torch_dist) ─────────────────
HF_CKPT=${HF_CKPT:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B/}
REF_LOAD=${REF_LOAD:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B_torch_dist/}
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT:-${EXPORT_ROOT}/ckpt/swe-rl-4b-remote-1node_${RUN_TIMESTAMP}}"
  --save-interval 5
)

# ── Rollout data ─────────────────────────────────────────────────────
if [[ "${DEBUG_MODE}" == "1" ]]; then
  # Debug: use train_debug4.jsonl (4 samples), minimal rollout
  DEBUG_DATA="${SWE_RL_DIR}/swe_data/swe_gym_subset/train_debug4.jsonl"
  if [[ ! -f "${DEBUG_DATA}" ]]; then
    echo "Creating debug dataset (first 4 samples)..."
    head -4 "${SWE_RL_DIR}/swe_data/swe_gym_subset/train.jsonl" > "${DEBUG_DATA}"
  fi
  PROMPT_DATA=${PROMPT_DATA:-${DEBUG_DATA}}
  NUM_ROLLOUT=4
  N_SAMPLES=1
  ROLLOUT_BATCH_SIZE=4
else
  PROMPT_DATA=${PROMPT_DATA:-${SWE_RL_DIR}/swe_data/swe_gym_subset/train.jsonl}
  NUM_ROLLOUT=500
  N_SAMPLES=4
  ROLLOUT_BATCH_SIZE=8
fi
if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "Missing prompt dataset: ${PROMPT_DATA}"
  exit 1
fi

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key text
  --metadata-key metadata
  --rollout-shuffle
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES}"
  --rollout-max-response-len 4096
  --rollout-max-context-len 32768
  --rollout-temperature 1
  --num-steps-per-rollout 1
)

# ── Performance / parallelism ────────────────────────────────────────
if [[ "${DEBUG_MODE}" == "1" ]]; then
  TP_SIZE=${ACTOR_GPUS_PER_NODE}
  MAX_TOKENS_PER_GPU=8192
else
  TP_SIZE=4
  MAX_TOKENS_PER_GPU=16384
fi
PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
  --balance-data
)

# ── GRPO ─────────────────────────────────────────────────────────────
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

# ── Optimizer ────────────────────────────────────────────────────────
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

# ── SGLang ───────────────────────────────────────────────────────────
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.6
  --sglang-router-port 30000
)

# ── Custom generate / reward (remote Docker) ─────────────────────────
CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_swe_remote.generate
  --custom-rm-path generate_with_swe_remote.reward_func
)

# ── Weights & Biases ─────────────────────────────────────────────────
if [[ "${DEBUG_MODE}" == "1" ]]; then
  WANDB_ARGS=()
else
  WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
  if [ -n "${WANDB_KEY_VALUE}" ]; then
    WANDB_ARGS=(
      --use-wandb
      --wandb-project slime_swe
      --wandb-group qwen3-4B-rl_swe_remote_1node
      --wandb-key "${WANDB_KEY_VALUE}"
    )
  else
    WANDB_ARGS=()
  fi
fi

# ── Miscellaneous ────────────────────────────────────────────────────
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# ── Environment variables ────────────────────────────────────────────
if [[ -f "${SWE_RL_DIR}/.env.swe" ]]; then
  source "${SWE_RL_DIR}/.env.swe"
fi
OPENAI_BASE_URL=${OPENAI_BASE_URL:-auto}
OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
LITELLM_MODEL_REGISTRY_PATH=${LITELLM_MODEL_REGISTRY_PATH:-"${SWE_RL_DIR}/litellm.json"}
SWE_LITELLM_MODEL_NAME=${SWE_LITELLM_MODEL_NAME:-openai/Qwen/Qwen3-4B}
SWE_SAVE_TRAJ_DIR=${SWE_SAVE_TRAJ_DIR:-${EXPORT_ROOT}/swe_rollouts/swe-rl-4b-remote-1node_${RUN_TIMESTAMP}}
mkdir -p "${SWE_SAVE_TRAJ_DIR}"
echo "SWE rollout artifacts dir: ${SWE_SAVE_TRAJ_DIR}"
echo "SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL}, SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS}"

# ── NVLink detection ─────────────────────────────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

# ── Launch Ray (single node, head only) ──────────────────────────────
ray start --head --node-ip-address "${NODE_IP}" --num-gpus "${NUM_GPUS_PER_NODE}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"OPENAI_BASE_URL\": \"${OPENAI_BASE_URL}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\",
    \"LITELLM_MODEL_REGISTRY_PATH\": \"${LITELLM_MODEL_REGISTRY_PATH}\",
    \"SWE_LITELLM_MODEL_NAME\": \"${SWE_LITELLM_MODEL_NAME}\",
    \"SWE_SAVE_TRAJ_DIR\": \"${SWE_SAVE_TRAJ_DIR}\",
    \"SWE_CONFIG_PATH\": \"${SWE_RL_DIR}/swebench.yaml\",
    \"SWE_ENV_SERVER_URL\": \"${SWE_ENV_SERVER_URL}\",
    \"SWE_MAX_CONCURRENT\": \"${SWE_MAX_CONCURRENT}\",
    \"MSWEA_DOCKER_EXEC_MODE\": \"${MSWEA_DOCKER_EXEC_MODE:-api}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${no_proxy}\"
  }
}"

# ── Submit Ray job ───────────────────────────────────────────────────
RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"swe_rl_4b_remote_1node_$(date +%Y%m%d_%H%M%S)"}

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u  /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/slime/train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
  --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${CUSTOM_ARGS[@]}

set +e
ray job logs --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
RAY_LOG_EXIT=$?
RAY_STATUS_OUTPUT=$(ray job status --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
echo "${RAY_STATUS_OUTPUT}"
set -e
if [[ "${RAY_STATUS_OUTPUT}" == *"SUCCEEDED"* ]]; then
  exit 0
fi
echo "Ray job failed (submission id: ${RAY_JOB_SUBMISSION_ID}, logs exit: ${RAY_LOG_EXIT})"
exit 1
