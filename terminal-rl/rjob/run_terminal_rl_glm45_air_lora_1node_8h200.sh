#!/bin/bash

################################################################################
# Terminal-RL GLM-4.5-Air LoRA RL run script - 1 node x 8 H200
#
# Local debug:
#   NODE_RANK=0 NODE_COUNT=1 MASTER_ADDR=127.0.0.1 PROC_PER_NODE=8 \
#     bash terminal-rl/rjob/run_terminal_rl_glm45_air_lora_1node_8h200.sh
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OPENCLAW_RL_HOME="${OPENCLAW_RL_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${OPENCLAW_RL_HOME}"

NODE_RANK="${NODE_RANK:-0}"
NODE_COUNT="${NODE_COUNT:-1}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if [[ "${NODE_RANK}" != "0" ]]; then
  echo "[ERROR] 1-node run expects NODE_RANK=0, got ${NODE_RANK}"
  exit 1
fi

export NUM_GPUS="${NUM_GPUS:-${PROC_PER_NODE}}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-$(( NUM_GPUS / 2 ))}"
export RAY_WAIT_TOTAL_GPUS="${RAY_WAIT_TOTAL_GPUS:-${NUM_GPUS}}"
export MASTER_ADDR
export WANDB_MODE="${WANDB_MODE:-offline}"
export ALGO="${ALGO:-dapo}"
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
export SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
export SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
export MAX_TURN="${MAX_TURN:-10}"

export HF_CKPT="${HF_CKPT:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--zai-org--GLM-4.5-Air/snapshots/a24ceef6ce4f3536971efe9b778bdaa1bab18daa}"
export REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${OPENCLAW_RL_HOME}/terminal-rl/configs/rollout_glm45_air_think.yaml}"

echo "=== Terminal-RL GLM-4.5-Air LoRA 1-node ==="
echo "OPENCLAW_RL_HOME=${OPENCLAW_RL_HOME}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "ALGO=${ALGO}"
echo "MAX_CKPT_KEEP=${MAX_CKPT_KEEP}"
echo "CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH}"
echo "WORKER_URLS=${WORKER_URLS:-}"

exec bash "${OPENCLAW_RL_HOME}/terminal-rl/terminal-rl_glm45_air_lora.sh"
