#!/bin/bash

################################################################################
# Terminal-RL GLM-4.5-Air LoRA RL run script - 2 nodes x 8 H200
#
# Rank 0 starts the normal terminal-rl GLM script. Other ranks join the Ray
# cluster as workers and block there.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OPENCLAW_RL_HOME="${OPENCLAW_RL_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${OPENCLAW_RL_HOME}"

NODE_RANK="${NODE_RANK:-0}"
NODE_COUNT="${NODE_COUNT:-2}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_terminal_rl}"

export DEBUG_MODE="${DEBUG_MODE:-0}"
export NUM_GPUS="${NUM_GPUS:-${PROC_PER_NODE}}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${NODE_COUNT}}"
export ACTOR_GPUS="${ACTOR_GPUS:-$(( NUM_GPUS / 2 ))}"
export ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-$(( NUM_GPUS / 2 ))}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-$(( NODE_COUNT * ROLLOUT_GPUS_PER_NODE ))}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${ROLLOUT_GPUS_PER_NODE}}"
export RAY_WAIT_TOTAL_GPUS="${RAY_WAIT_TOTAL_GPUS:-$(( NODE_COUNT * NUM_GPUS ))}"
export MASTER_ADDR RAY_TMPDIR
export WANDB_MODE="${WANDB_MODE:-offline}"
export ALGO="${ALGO:-dapo}"
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
export SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
export SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
export MAX_TURN="${MAX_TURN:-10}"

export HF_CKPT="${HF_CKPT:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--zai-org--GLM-4.5-Air/snapshots/a24ceef6ce4f3536971efe9b778bdaa1bab18daa}"
export REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${OPENCLAW_RL_HOME}/terminal-rl/configs/rollout_glm45_air_think.yaml}"

echo "=== Terminal-RL GLM-4.5-Air LoRA 2-node ==="
echo "NODE_RANK=${NODE_RANK}"
echo "NODE_COUNT=${NODE_COUNT}"
echo "PROC_PER_NODE=${PROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "OPENCLAW_RL_HOME=${OPENCLAW_RL_HOME}"
echo "DEBUG_MODE=${DEBUG_MODE}"
echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES}"
echo "ROLLOUT_GPUS=${ROLLOUT_GPUS}"
echo "ALGO=${ALGO}"
echo "MAX_CKPT_KEEP=${MAX_CKPT_KEEP}"
echo "CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH}"

if [[ "${NODE_RANK}" == "0" ]]; then
  exec bash "${OPENCLAW_RL_HOME}/terminal-rl/terminal-rl_glm45_air_lora.sh"
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
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --temp-dir "${RAY_TMPDIR}" \
  --block
