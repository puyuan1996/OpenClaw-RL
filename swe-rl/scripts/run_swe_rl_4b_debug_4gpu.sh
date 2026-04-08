#!/bin/bash
# SWE-RL Debug 脚本 — 4-GPU worker 上跑 4 个样本的最小化端到端测试
#
# 前提：
#   1. 宿主机上已拉取前 4 个镜像（N=4 bash pull_images_on_host.sh）
#   2. 4-GPU worker 上 swe_exec_server 已启动（bash setup_ecs_on_worker.sh）
#   3. DOCKER_HOST 已设置指向宿主机
#
# Usage:
#   cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL
#   bash swe-rl/scripts/run_swe_rl_4b_debug_4gpu.sh

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SWE_RL_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SWE_RL_DIR}/../slime" &>/dev/null && pwd)"
EXPORT_ROOT=${EXPORT_ROOT:-"${SWE_RL_DIR}/../export"}
mkdir -p "${EXPORT_ROOT}/ckpt" "${EXPORT_ROOT}/swe_rollouts"
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs"}
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/run_swe_rl_4b_debug_4gpu_${RUN_TIMESTAMP}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "=== SWE-RL Debug (4-GPU, 4 samples) ==="
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
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

# ── GPU allocation (4-GPU debug) ─────────────────────────────────────
NUM_GPUS_PER_NODE=4
ACTOR_GPUS_PER_NODE=2
ROLLOUT_GPUS_PER_NODE=2
ROLLOUT_GPUS_TOTAL=2
ROLLOUT_NUM_GPUS_PER_ENGINE=2

MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
NODE_IP=${MASTER_ADDR}
export MASTER_ADDR
echo "MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

# ── Remote SWE env pool / exec configs ───────────────────────────────
export SWE_ENV_SERVER_BIND_HOST=0.0.0.0
export SWE_ENV_SERVER_PORT=${SWE_ENV_SERVER_PORT:-18090}
export SWE_ENV_SERVER_HOST=${SWE_ENV_SERVER_HOST:-${MASTER_ADDR}}
export SWE_ENV_SERVER_URL=http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}
export SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS:-http://localhost:5000}
# Debug: 少量容器，低并发，保护宿主机资源
export SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE:-4}
export SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT:-1}

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

SWE_POOL_LOG="${LOG_DIR}/swe_env_pool_server_debug_4gpu.log"
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

# ── Checkpoints (raw mode) ───────────────────────────────────────────
HF_CKPT=${HF_CKPT:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B/}
REF_LOAD=${REF_LOAD:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B_torch_dist/}
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${EXPORT_ROOT}/ckpt/swe-rl-4b-debug-4gpu_${RUN_TIMESTAMP}"
  --save-interval 999
)

# ── Rollout data (4 samples only) ────────────────────────────────────
PROMPT_DATA="${SWE_RL_DIR}/swe_data/swe_gym_subset/train_debug4.jsonl"
if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "Creating debug dataset (first 4 samples)..."
  head -4 "${SWE_RL_DIR}/swe_data/swe_gym_subset/train.jsonl" > "${PROMPT_DATA}"
fi

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key text
  --metadata-key metadata
  --rollout-shuffle
  --reward-key score
  --num-rollout 4
  --rollout-batch-size 4
  --n-samples-per-prompt 1
  --rollout-max-response-len 4096
  --rollout-max-context-len 32768
  --rollout-temperature 1
  --num-steps-per-rollout 1
)

# ── Performance / parallelism (TP=2 for 4-GPU) ──────────────────────
PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 8192
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

# ── No wandb for debug ───────────────────────────────────────────────
WANDB_ARGS=()

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
SWE_SAVE_TRAJ_DIR="${EXPORT_ROOT}/swe_rollouts/swe-rl-4b-debug-4gpu_${RUN_TIMESTAMP}"
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
RAY_JOB_SUBMISSION_ID="swe_rl_4b_debug_4gpu_$(date +%Y%m%d_%H%M%S)"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/slime/train_async.py \
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
  echo "=== Debug run SUCCEEDED ==="
  exit 0
fi
echo "Ray job failed (submission id: ${RAY_JOB_SUBMISSION_ID}, logs exit: ${RAY_LOG_EXIT})"
exit 1
