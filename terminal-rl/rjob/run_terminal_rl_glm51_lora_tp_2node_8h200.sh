#!/bin/bash

################################################################################
# Terminal-RL GLM-5.1 Megatron-TP LoRA RL run script - 2 nodes x 8 H200
#
# Rank 0 starts the normal terminal-rl GLM script. Other ranks join the Ray
# cluster as workers and block there.
################################################################################

set -euo pipefail

first_existing_dir() {
  local fallback="" path
  for path in "$@"; do
    if [[ -z "${fallback}" && -n "${path}" ]]; then
      fallback="${path}"
    fi
    if [[ -n "${path}" && -d "${path}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done
  printf '%s\n' "${fallback}"
}

safe_run_key() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/^_*//;s/_*$//'
}

read_fresh_timestamp() {
  local stamp_file="$1"
  local min_mtime="$2"
  local stamp_mtime
  [[ -s "${stamp_file}" ]] || return 1
  stamp_mtime="$(stat -c %Y "${stamp_file}" 2>/dev/null || printf '0')"
  [[ "${stamp_mtime}" =~ ^[0-9]+$ ]] || stamp_mtime=0
  (( stamp_mtime >= min_mtime )) || return 1
  cat "${stamp_file}"
}

update_latest_link() {
  local target="$1"
  local latest="$2"
  local backup suffix

  if [[ -L "${latest}" || ! -e "${latest}" ]]; then
    ln -sfnT "${target}" "${latest}" 2>/dev/null || true
    return
  fi

  backup="${latest}.stale_${RUN_TIMESTAMP}"
  suffix=1
  while [[ -e "${backup}" ]]; do
    backup="${latest}.stale_${RUN_TIMESTAMP}.${suffix}"
    suffix=$((suffix + 1))
  done
  if mv "${latest}" "${backup}" 2>/dev/null; then
    echo "[WARN] Moved existing non-symlink latest log path to ${backup}"
    ln -sfnT "${target}" "${latest}" 2>/dev/null || true
  else
    echo "[WARN] Could not replace non-symlink latest log path: ${latest}"
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OPENCLAW_RL_HOME="${OPENCLAW_RL_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${OPENCLAW_RL_HOME}"

MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX}}"
NODE_COUNT="${NODE_COUNT:-2}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I 2>/dev/null | awk '{print $1}' || true)}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
_WORKER_IP_VAR="MLP_WORKER_${NODE_RANK}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I 2>/dev/null | awk '{print $1}' || true)}}"
NODE_IP="${NODE_IP:-${MASTER_ADDR}}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_terminal_rl}"

if [[ -z "${RUN_TIMESTAMP:-}" ]]; then
  _timestamp_launch_epoch="$(date +%s)"
  _timestamp_min_mtime=$((_timestamp_launch_epoch - ${RJOB_TIMESTAMP_EARLY_ACCEPT_SECONDS:-30}))
  RJOB_RUN_KEY="${RJOB_RUN_KEY:-${MLP_JOB_ID:-${MLP_TASK_ID:-${MLP_JOB_NAME:-${RJOB_JOB_ID:-${RJOB_NAME:-}}}}}}"
  RJOB_RUN_KEY="${RJOB_RUN_KEY:-master_${MASTER_ADDR}_${NODE_COUNT}x${PROC_PER_NODE}}"
  if [[ -n "${RJOB_RUN_KEY}" ]]; then
    _safe_run_key="$(safe_run_key "${RJOB_RUN_KEY}")"
    _stamp_dir="${OPENCLAW_RL_HOME}/tmp_doc_rjob_state"
    _stamp_file="${_stamp_dir}/${_safe_run_key}.timestamp"
    mkdir -p "${_stamp_dir}"
    if [[ "${NODE_RANK}" == "0" ]]; then
      date +%F_%H%M%S > "${_stamp_file}.${NODE_RANK}.$$"
      mv -f "${_stamp_file}.${NODE_RANK}.$$" "${_stamp_file}"
      RUN_TIMESTAMP="$(cat "${_stamp_file}")"
    else
      _timestamp_wait_seconds="${RJOB_TIMESTAMP_WAIT_SECONDS:-300}"
      _timestamp_deadline=$((_timestamp_launch_epoch + _timestamp_wait_seconds))
      while (( $(date +%s) <= _timestamp_deadline )); do
        if RUN_TIMESTAMP="$(read_fresh_timestamp "${_stamp_file}" "${_timestamp_min_mtime}")"; then
          break
        fi
        sleep 1
      done
      if [[ -z "${RUN_TIMESTAMP:-}" ]]; then
        echo "[WARN] Timed out waiting for rank0 timestamp at ${_stamp_file}; using local timestamp"
      fi
    fi
  fi
fi
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}"
export RUN_TIMESTAMP
RJOB_NODE_LOG_DIR="${RJOB_NODE_LOG_DIR:-${OPENCLAW_RL_HOME}/tmp_doc_${RUN_TIMESTAMP}/node_logs}"
RJOB_NODE_LOG_FILE="${RJOB_NODE_LOG_FILE:-${RJOB_NODE_LOG_DIR}/node_${NODE_RANK}.log}"
export RJOB_NODE_LOG_DIR RJOB_NODE_LOG_FILE
if [[ "${RJOB_NODE_LOG_CAPTURE:-1}" == "1" ]]; then
  mkdir -p "${RJOB_NODE_LOG_DIR}"
  exec > >(tee -a "${RJOB_NODE_LOG_FILE}") 2>&1
  update_latest_link "${OPENCLAW_RL_HOME}/tmp_doc_${RUN_TIMESTAMP}" "${OPENCLAW_RL_HOME}/tmp_doc_latest"
fi

REPO_PARENT="$(cd "${OPENCLAW_RL_HOME}/.." && pwd)"
SHARED_USER_ROOT="${SHARED_USER_ROOT:-${REPO_PARENT}}"
PUYUAN_ROOT="${PUYUAN_ROOT:-/mnt/shared-storage-user/puyuan}"
LIGHTRFT_PY312_BIN="${LIGHTRFT_PY312_BIN:-$(first_existing_dir \
  "${SHARED_USER_ROOT}/conda_envs/lightrft_py312/bin" \
  "${PUYUAN_ROOT}/conda_envs/lightrft_py312/bin")}"
if [[ ! -x "${LIGHTRFT_PY312_BIN}/python" && ! -x "${LIGHTRFT_PY312_BIN}/python3" ]]; then
  echo "[ERROR] LIGHTRFT_PY312_BIN does not contain python/python3: ${LIGHTRFT_PY312_BIN}"
  echo "        Set LIGHTRFT_PY312_BIN=/path/to/lightrft_py312/bin and rerun."
  exit 1
fi
export PATH="${LIGHTRFT_PY312_BIN}:${PATH}"

export DEBUG_MODE="${DEBUG_MODE:-0}"
export NUM_GPUS="${NUM_GPUS:-${PROC_PER_NODE}}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${NODE_COUNT}}"
export COLOCATE="${COLOCATE:-1}"
export ACTOR_GPUS="${ACTOR_GPUS:-${NUM_GPUS}}"
export ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-${NUM_GPUS}}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-$(( NODE_COUNT * ROLLOUT_GPUS_PER_NODE ))}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${ROLLOUT_GPUS}}"
export RAY_WAIT_TOTAL_GPUS="${RAY_WAIT_TOTAL_GPUS:-$(( NODE_COUNT * NUM_GPUS ))}"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"
export CP_SIZE="${CP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-8}"
export ETP_SIZE="${ETP_SIZE:-1}"
export MASTER_ADDR NODE_IP RAY_TMPDIR
export WANDB_MODE="${WANDB_MODE:-offline}"
export ALGO="${ALGO:-dapo}"
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
export SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
export SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
export MAX_TURN="${MAX_TURN:-10}"

export HF_CKPT="${HF_CKPT:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d}"
export REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${OPENCLAW_RL_HOME}/terminal-rl/configs/rollout_glm51_think.yaml}"
export WORKER_URLS="${WORKER_URLS:-http://100.100.66.216:18081}"

echo "=== Terminal-RL GLM-5.1 Megatron-TP LoRA 2-node ==="
echo "NODE_RANK=${NODE_RANK}"
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}"
echo "NODE_COUNT=${NODE_COUNT}"
echo "PROC_PER_NODE=${PROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "NODE_IP=${NODE_IP}"
echo "OPENCLAW_RL_HOME=${OPENCLAW_RL_HOME}"
echo "RUN_TIMESTAMP=${RUN_TIMESTAMP}"
echo "RJOB_NODE_LOG_DIR=${RJOB_NODE_LOG_DIR}"
echo "RJOB_NODE_LOG_FILE=${RJOB_NODE_LOG_FILE}"
echo "LIGHTRFT_PY312_BIN=${LIGHTRFT_PY312_BIN}"
echo "RAY_BIN=$(command -v ray)"
echo "DEBUG_MODE=${DEBUG_MODE}"
echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES}"
echo "ACTOR_GPUS=${ACTOR_GPUS}"
echo "ROLLOUT_GPUS=${ROLLOUT_GPUS}"
echo "ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE}"
echo "ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "COLOCATE=${COLOCATE}"
echo "TP_SIZE=${TP_SIZE}"
echo "PP_SIZE=${PP_SIZE}"
echo "CP_SIZE=${CP_SIZE}"
echo "EP_SIZE=${EP_SIZE}"
echo "ETP_SIZE=${ETP_SIZE}"
echo "ALGO=${ALGO}"
echo "MAX_CKPT_KEEP=${MAX_CKPT_KEEP}"
echo "HF_CKPT=${HF_CKPT}"
echo "REF_LOAD=${REF_LOAD}"
echo "BASE_LOAD=${BASE_LOAD:-}"
echo "MEGATRON_LORA_ADAPTER_LOAD=${MEGATRON_LORA_ADAPTER_LOAD:-}"
echo "CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH}"
echo "WORKER_URLS=${WORKER_URLS}"

if [[ "${NODE_RANK}" == "0" ]]; then
  exec bash "${OPENCLAW_RL_HOME}/terminal-rl/terminal-rl_glm51_lora_tp.sh"
fi

echo "Waiting for Ray head at ${MASTER_ADDR}:6379"
for i in {1..120}; do
  if timeout 2 bash -c ":</dev/tcp/${MASTER_ADDR}/6379" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

ray stop --force || true
exec ray start \
  --address "${MASTER_ADDR}:6379" \
  --node-ip-address "${NODE_IP}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --temp-dir "${RAY_TMPDIR}" \
  --block
