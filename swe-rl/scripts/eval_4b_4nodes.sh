#!/bin/bash

# Evaluate Qwen3-4B RL checkpoint on SWE-Bench Verified (500) + SWE-Gym train (293).
#
# Mirrors the 4-node colocate training scaffold (run_swe_rl_4b_4nodes_colocate.sh):
#   * 4 GPU nodes × 8 GPU = 32 GPU, TP=2 rollout engines (16 engines)
#   * Full 6-node ECS Docker pool
#   * generate_with_swe.{generate,reward_func} (same code as training)
#
# Usage (head node, MLP_ROLE_INDEX=0; worker nodes set MLP_ROLE_INDEX=1..3):
#   export SWE_EXEC_SERVER_URLS=http://<exec-1>:5000,http://<exec-2>:5000
#   export HF_CKPT=<REPO_ROOT>/export/hf/swe-rl-4b_iter180
#   bash swe-rl/scripts/eval_4b_4nodes.sh

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SWE_RL_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SWE_RL_DIR}/../slime" &>/dev/null && pwd)"
EXPORT_ROOT=${EXPORT_ROOT:-"${SWE_RL_DIR}/../export"}
mkdir -p "${EXPORT_ROOT}/swe_rollouts"
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs"}
mkdir -p "${LOG_DIR}"
RUN_TAG=${RUN_TAG:-4b_iter180}
RUN_LOG=${RUN_LOG:-"${LOG_DIR}/eval_${RUN_TAG}_${RUN_TIMESTAMP}.log"}
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Run log: ${RUN_LOG}"
echo "Run timestamp: ${RUN_TIMESTAMP}"

source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SWE_RL_DIR}/../Megatron-LM"}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

# ---- Cluster topology ----
NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
ROLLOUT_GPUS_TOTAL=$((NUM_NODES * NUM_GPUS_PER_NODE))

MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
_WORKER_IP_VAR="MLP_WORKER_${MLP_ROLE_INDEX}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"
export MASTER_ADDR
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}, MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

# Env overrides
if [[ -f "${SWE_RL_DIR}/.env.swe" ]]; then
  source "${SWE_RL_DIR}/.env.swe"
fi
if [[ -f "${SWE_RL_DIR}/../.env" ]]; then
  set -a; source "${SWE_RL_DIR}/../.env"; set +a
fi

# ---- Remote Docker (SWE env pool) ----
export SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS:-}
if [[ -z "${SWE_EXEC_SERVER_URLS}" ]]; then
  echo "SWE_EXEC_SERVER_URLS is required, e.g. http://<exec-1>:5000,http://<exec-2>:5000"
  exit 1
fi
export SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE:-48}
_n_exec_nodes=$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | wc -l)
export SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT:-$(( SWE_MAX_CONTAINERS_PER_NODE * _n_exec_nodes ))}
export SWE_CONTAINER_PIDS_LIMIT=${SWE_CONTAINER_PIDS_LIMIT:-1024}
export SWE_CONTAINER_MEMORY=${SWE_CONTAINER_MEMORY:-8g}
export SWE_STRICT_NO_TEST_PATCH=${SWE_STRICT_NO_TEST_PATCH:-1}
export SWE_STRICT_NO_CONFIG_PATCH=${SWE_STRICT_NO_CONFIG_PATCH:-1}
export SWE_TEST_PATCH_POLICY_SCOPE=${SWE_TEST_PATCH_POLICY_SCOPE:-eval_tests_only}
SWE_ENV_SERVER_BIND_HOST=${SWE_ENV_SERVER_BIND_HOST:-0.0.0.0}
SWE_ENV_SERVER_PORT=${SWE_ENV_SERVER_PORT:-18090}
SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL:-"http://${MASTER_ADDR}:${SWE_ENV_SERVER_PORT}"}
export SWE_ENV_SERVER_URL
CES_SSH_USER=${CES_SSH_USER:-root}
CES_SSH_KEY=${CES_SSH_KEY:-}

ALL_EXEC_HOSTS="$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
export NO_PROXY="localhost,127.0.0.1,${MASTER_ADDR},${NODE_IP},${ALL_EXEC_HOSTS}"
export no_proxy="${NO_PROXY}"
RAY_NO_PROXY="${NO_PROXY}"

export HTTP_PROXY=${HTTP_PROXY:-}
export HTTPS_PROXY=${HTTPS_PROXY:-}
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"

SWE_DOCKER_REGISTRY=${SWE_DOCKER_REGISTRY:-docker.io}
SWE_ROLLOUT_TIMEOUT=${SWE_ROLLOUT_TIMEOUT:-1800}
SWE_EVAL_TIMEOUT=${SWE_EVAL_TIMEOUT:-300}

# ---- Model (checkpoint under eval) ----
HF_CKPT=${HF_CKPT:-${EXPORT_ROOT}/hf/swe-rl-4b_iter180}
if [[ ! -d "${HF_CKPT}" ]]; then
  echo "HF_CKPT dir not found: ${HF_CKPT}"; exit 1
fi

# ---- Datasets ----
VERIFIED_DATA=${VERIFIED_DATA:-${SWE_RL_DIR}/../data/swebench_verified/verified_eval_with_script.jsonl}
SWEGYM_DATA=${SWEGYM_DATA:-${SWE_RL_DIR}/../data/swegym_293/train_with_eval_script.parquet}

# Batch size: saturate ECS concurrency, but keep moderate so repeated samples
# in the remainder are bounded.
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-${SWE_MAX_CONCURRENT}}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "${NVLINK_COUNT}" -gt 0 ] && echo 1 || echo 0)
# Colocate-incompatible setting removed; pure rollout here so we can use expandable_segments.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

SWE_POOL_PID=""

_cleanup_ecs_containers() {
  IFS=',' read -r -a _urls <<< "${SWE_EXEC_SERVER_URLS}"
  for url in "${_urls[@]}"; do
    host="$(echo "${url}" | sed -E 's#https?://([^:/]+).*#\1#')"
    echo "Cleaning Docker on ${host}..."
    ssh_key_args=()
    if [[ -n "${CES_SSH_KEY}" ]]; then
      ssh_key_args=(-i "${CES_SSH_KEY}")
    fi
    ssh "${ssh_key_args[@]}" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
      "${CES_SSH_USER}@${host}" 'docker ps -aq --filter "name=swe-" | xargs -r docker rm -f' 2>/dev/null || true
  done
}

cleanup() {
  set +e
  if [[ -n "${SWE_POOL_PID}" ]] && kill -0 "${SWE_POOL_PID}" 2>/dev/null; then
    kill "${SWE_POOL_PID}" || true
  fi
  if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
    _cleanup_ecs_containers
  fi
}
trap cleanup EXIT INT TERM

_count_dataset_rows() {
  local dpath="$1"
  if [[ "${dpath}" == *.parquet ]]; then
    python3 -c "import pandas; print(len(pandas.read_parquet('${dpath}')))" 2>/dev/null || echo 500
  elif [[ "${dpath}" == *.jsonl ]]; then
    python3 -c "
with open('${dpath}', 'r') as f:
    print(sum(1 for l in f if l.strip()))
" 2>/dev/null || echo 500
  else
    echo 500
  fi
}

_build_runtime_env() {
  local traj_dir="$1"
  echo "{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"PYTHONFAULTHANDLER\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"SWE_ENV_SERVER_URL\": \"${SWE_ENV_SERVER_URL}\",
    \"SWE_MAX_CONCURRENT\": \"${SWE_MAX_CONCURRENT}\",
    \"SWE_STRICT_NO_TEST_PATCH\": \"${SWE_STRICT_NO_TEST_PATCH}\",
    \"SWE_STRICT_NO_CONFIG_PATCH\": \"${SWE_STRICT_NO_CONFIG_PATCH}\",
    \"SWE_TEST_PATCH_POLICY_SCOPE\": \"${SWE_TEST_PATCH_POLICY_SCOPE}\",
    \"SWE_SAVE_TRAJ_DIR\": \"${traj_dir}\",
    \"SWE_ROLLOUT_TIMEOUT\": \"${SWE_ROLLOUT_TIMEOUT}\",
    \"SWE_EVAL_TIMEOUT\": \"${SWE_EVAL_TIMEOUT}\",
    \"SWE_DOCKER_REGISTRY\": \"${SWE_DOCKER_REGISTRY}\",
    \"SWE_CONTAINER_PIDS_LIMIT\": \"${SWE_CONTAINER_PIDS_LIMIT}\",
    \"SWE_CONTAINER_MEMORY\": \"${SWE_CONTAINER_MEMORY}\",
    \"HTTP_PROXY\": \"${HTTP_PROXY}\",
    \"HTTPS_PROXY\": \"${HTTPS_PROXY}\",
    \"http_proxy\": \"${http_proxy}\",
    \"https_proxy\": \"${https_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${no_proxy}\",
    \"HF_HOME\": \"${HF_HOME:-${HOME}/.cache/huggingface}\"
  }
}"
}

_submit_eval_job() {
  local job_id="$1"
  local prompt_data="$2"
  local input_key="$3"
  local metadata_key="$4"
  local traj_dir="$5"
  local dataset_rows="$6"
  local num_rollout=$(( (dataset_rows + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))
  echo "  dataset_rows=${dataset_rows}, batch_size=${ROLLOUT_BATCH_SIZE}, num_rollout=${num_rollout}"

  local runtime_env
  runtime_env=$(_build_runtime_env "${traj_dir}")

  NO_PROXY="${RAY_NO_PROXY}" no_proxy="${RAY_NO_PROXY}" \
  ray job submit --address="http://${MASTER_ADDR}:8265" \
    --submission-id "${job_id}" \
    --no-wait \
    --runtime-env-json="${runtime_env}" \
    -- python3 -u "${SLIME_DIR}/train_async.py" \
    --debug-rollout-only \
    --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}" \
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
    --actor-num-nodes "${NUM_NODES}" \
    --actor-num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint "${HF_CKPT}" \
    --prompt-data "${prompt_data}" \
    --input-key "${input_key}" \
    --metadata-key "${metadata_key}" \
    --reward-key score \
    --num-rollout "${num_rollout}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt 1 \
    --rollout-max-response-len 4096 \
    --rollout-max-context-len 32768 \
    --rollout-temperature 0.6 \
    --rollout-top-p 0.95 \
    --num-steps-per-rollout 1 \
    --advantage-estimator grpo \
    --sglang-mem-fraction-static 0.85 \
    --sglang-router-port 30000 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --max-tokens-per-gpu 32768 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --attention-backend flash \
    --custom-generate-function-path generate_with_swe.generate \
    --custom-rm-path generate_with_swe.reward_func

  set +e
  NO_PROXY="${RAY_NO_PROXY}" no_proxy="${RAY_NO_PROXY}" \
  ray job logs --address="http://${MASTER_ADDR}:8265" "${job_id}" -f --log-style=record
  local log_exit=$?
  local status_output
  status_output=$(NO_PROXY="${RAY_NO_PROXY}" no_proxy="${RAY_NO_PROXY}" \
    ray job status --address="http://${MASTER_ADDR}:8265" "${job_id}" --log-style=record 2>&1)
  echo "${status_output}"
  set -e

  if [[ "${status_output}" == *"SUCCEEDED"* ]]; then
    echo "Job ${job_id} SUCCEEDED. Artifacts: ${traj_dir}"
    return 0
  fi
  echo "Job ${job_id} FAILED (logs exit: ${log_exit})"
  return 1
}

# ==================================================================
# HEAD NODE
# ==================================================================
if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then

  # Pre-flight cleanup
  pkill -f "swe_env_pool_server" 2>/dev/null || true
  sleep 1
  if ss -ltnp 2>/dev/null | grep -q ":${SWE_ENV_SERVER_PORT} "; then
    fuser -k "${SWE_ENV_SERVER_PORT}/tcp" 2>/dev/null || true
    sleep 1
  fi
  _cleanup_ecs_containers

  # Start pool server
  SWE_POOL_LOG="${LOG_DIR}/swe_env_pool_server_${RUN_TAG}_${RUN_TIMESTAMP}.log"
  HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= \
  PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${PYTHONPATH}" \
  python3 -m swe_env_pool_server \
    --host "${SWE_ENV_SERVER_BIND_HOST}" \
    --port "${SWE_ENV_SERVER_PORT}" \
    --exec-server-urls "${SWE_EXEC_SERVER_URLS}" \
    --max-containers-per-node "${SWE_MAX_CONTAINERS_PER_NODE}" \
    > "${SWE_POOL_LOG}" 2>&1 &
  SWE_POOL_PID=$!
  echo "Pool server PID=${SWE_POOL_PID}, log=${SWE_POOL_LOG}"

  for i in {1..60}; do
    if curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
      echo "Pool server ready: ${SWE_ENV_SERVER_URL}"
      break
    fi
    sleep 2
  done

  # Start Ray head
  ray start --head --node-ip-address "${NODE_IP}" --num-gpus "${NUM_GPUS_PER_NODE}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
  sleep 15
  echo "Waiting for ${NUM_NODES} nodes to join Ray cluster..."
  for i in {1..120}; do
    TOTAL_GPUS=$(ray status 2>/dev/null | grep "GPU" | head -1 | grep -oP '\d+\.\d+/\d+\.\d+' | head -1 | cut -d/ -f2 | cut -d. -f1 || echo 0)
    if [[ "${TOTAL_GPUS}" -ge $((NUM_NODES * NUM_GPUS_PER_NODE)) ]]; then
      echo "All ${NUM_NODES} nodes joined. Total GPUs: ${TOTAL_GPUS}"
      break
    fi
    echo "  waiting... (GPUs: ${TOTAL_GPUS}/$((NUM_NODES * NUM_GPUS_PER_NODE)))"
    sleep 5
  done

  # ================================================================
  # Eval 1: SWE-Bench Verified
  # ================================================================
  echo ""
  echo "################################################################"
  echo "# [${RUN_TAG}] Part 1/2: SWE-Bench Verified"
  echo "################################################################"
  VERIFIED_TRAJ="${EXPORT_ROOT}/swe_rollouts/eval_${RUN_TAG}_verified_${RUN_TIMESTAMP}"
  mkdir -p "${VERIFIED_TRAJ}"
  VERIFIED_ROWS=$(_count_dataset_rows "${VERIFIED_DATA}")
  echo "Dataset: ${VERIFIED_DATA} (${VERIFIED_ROWS} rows)"

  _submit_eval_job \
    "eval_${RUN_TAG}_verified_$(date +%Y%m%d_%H%M%S)" \
    "${VERIFIED_DATA}" "text" "metadata" \
    "${VERIFIED_TRAJ}" "${VERIFIED_ROWS}" || true

  echo "[${RUN_TAG}] Verified eval done. Cleaning Docker..."
  _cleanup_ecs_containers

  # ================================================================
  # Eval 2: SWE-Gym 293 (training split)
  # ================================================================
  echo ""
  echo "################################################################"
  echo "# [${RUN_TAG}] Part 2/2: SWE-Gym 293 (train)"
  echo "################################################################"
  SWEGYM_TRAJ="${EXPORT_ROOT}/swe_rollouts/eval_${RUN_TAG}_swegym_${RUN_TIMESTAMP}"
  mkdir -p "${SWEGYM_TRAJ}"
  SWEGYM_ROWS=$(_count_dataset_rows "${SWEGYM_DATA}")
  echo "Dataset: ${SWEGYM_DATA} (${SWEGYM_ROWS} rows)"

  _submit_eval_job \
    "eval_${RUN_TAG}_swegym_$(date +%Y%m%d_%H%M%S)" \
    "${SWEGYM_DATA}" "prompt" "instance" \
    "${SWEGYM_TRAJ}" "${SWEGYM_ROWS}" || true

  echo "[${RUN_TAG}] SWE-Gym eval done. Cleaning Docker..."
  _cleanup_ecs_containers

  echo ""
  echo "========================================"
  echo "[${RUN_TAG}] All evaluations complete."
  echo "  HF_CKPT:  ${HF_CKPT}"
  echo "  Verified: ${VERIFIED_TRAJ}"
  echo "  SWE-Gym:  ${SWEGYM_TRAJ}"
  echo "  Log:      ${RUN_LOG}"
  echo "========================================"

# ==================================================================
# WORKER NODE
# ==================================================================
else
  sleep 30
  ray start --address="${MASTER_ADDR}:6379" --num-gpus "${NUM_GPUS_PER_NODE}" \
    --node-ip-address "${NODE_IP}"

  echo "Worker node ${MLP_ROLE_INDEX} joined cluster. Waiting for all jobs to finish..."
  while ray status >/dev/null 2>&1; do
    sleep 60
  done
  echo "Ray cluster stopped. Worker node ${MLP_ROLE_INDEX} exiting."
fi
