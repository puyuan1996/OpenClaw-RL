#!/usr/bin/env bash
# SETA-only adaptive exploration v0629_2 launch for Qwen3-8B.
#
# Baseline:
#   terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629.sh
#
# Variant hypothesis:
#   v0629 computes a group-normalized intrinsic advantage, then multiplies it by
#   beta/max_beta again before adding it to the task advantage stream. Because
#   UCB already uses beta to collect differently exploratory samples, this
#   second beta weighting makes most intrinsic gradients too small to matter.
#
# Only algorithmic change relative to v0629:
#   EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE=none
#
# Expected validation metrics:
#   reward/exploration_post_norm_abs should rise from the v0629 O(1e-3) range
#   without changing score-space exploration_reward_score, and raw_reward/op_raw
#   should improve or remain stable at comparable truncation and fail rates.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"

RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0629_2_armw_none_dualadv_think_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-8b_seta_exploration_v0629_2}"

EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE="${EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE:-none}"

export RUN_TIMESTAMP NUM_GPUS RUN_ID RUN_NAME WANDB_GROUP
export EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE

exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629.sh" "$@"
