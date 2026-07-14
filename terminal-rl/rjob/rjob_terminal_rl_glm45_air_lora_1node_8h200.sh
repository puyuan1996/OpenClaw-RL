#!/bin/bash

################################################################################
# Terminal-RL GLM-4.5-Air LoRA RL rjob submit script - 1 node x 8 H200
#
# Usage:
#   # WORKER_URLS is optional here and is passed through only if already set.
#   bash rjob_terminal_rl_glm45_air_lora_1node_8h200.sh
################################################################################

OPENCLAW_RL_HOME="${OPENCLAW_RL_HOME:-/mnt/shared-storage-user/luyudong/OpenClaw-RL}"
RUN_SCRIPT="${OPENCLAW_RL_HOME}/terminal-rl/rjob/run_terminal_rl_glm45_air_lora_1node_8h200.sh"

RJOB_ENV_ARGS=()
if [[ -n "${WORKER_URLS+x}" ]]; then
  RJOB_ENV_ARGS+=(-e WORKER_URLS="${WORKER_URLS}")
fi

rjob submit \
  --name=terminal-rl-glm45-air-lora-1node-8h200 \
  --gpu=8 \
  --memory=1000000 \
  --cpu=100 \
  --charged-group=narmodel_gpu \
  --private-machine=group \
  --host-network=false \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
  -e DISTRIBUTED_JOB=true \
  -e OPENCLAW_RL_HOME="${OPENCLAW_RL_HOME}" \
  "${RJOB_ENV_ARGS[@]}" \
  -e DATASET="${DATASET:-seta}" \
  -e ALGO="${ALGO:-dapo}" \
  -e HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}" \
  -e MAX_TURN="${MAX_TURN:-10}" \
  -e SETA_SAFETY="${SETA_SAFETY:-clawsentry}" \
  -e SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}" \
  -e CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-}" \
  -e HF_CKPT="${HF_CKPT:-}" \
  -e REF_LOAD="${REF_LOAD:-}" \
  -e SAVE_CKPT="${SAVE_CKPT:-}" \
  -e MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}" \
  -e WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-terminal_rl}" \
  -e WANDB_GROUP="${WANDB_GROUP:-glm45-air_lora}" \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=8 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  -- bash -exc "${RUN_SCRIPT}"
