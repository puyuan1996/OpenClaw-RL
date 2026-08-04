#!/bin/bash

################################################################################
# Submit OpenClaw terminal-rl Qwen3-8B SETA exploration v0629_2 as a detached
# 1-node 8-GPU rjob task.
#
# Baseline rjob:
#   rjob_terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629.sh
#
# Paired run script:
#   terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629_2.sh
#
# Only algorithmic change relative to v0629:
#   EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE=none
#
# Hypothesis:
#   Keep UCB/beta as the data-collection controller, but remove the second
#   beta/max_beta attenuation from post-normalized dual-adv intrinsic gradients.
#
# Expected validation metrics:
#   - reward/exploration_post_norm_abs increases materially from v0629;
#   - reward/exploration_reward_score remains 0, preserving score-space control;
#   - raw_reward/op_raw improves or stays stable at similar truncation/fail rate.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_HOME="${OPENCLAW_HOME:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_SCRIPT="${RUN_SCRIPT:-${OPENCLAW_HOME}/terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629_2.sh}"
RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0629_2_armw_none_dualadv_think_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-8b_seta_exploration_v0629_2}"
RJOB_LOG_ROOT="${RJOB_LOG_ROOT:-${OPENCLAW_HOME}/runs/rjob_logs/terminal_rl_v0629_2}"
RJOB_NAME="${RJOB_NAME:-a57-v0629-2-armw-none}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-3}"

export OPENCLAW_HOME RUN_TIMESTAMP NUM_GPUS RUN_SCRIPT RUN_ID RUN_NAME
export WANDB_GROUP RJOB_LOG_ROOT RJOB_NAME MAX_CKPT_KEEP

exec bash "${SCRIPT_DIR}/rjob_terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629.sh" "$@"
