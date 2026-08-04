#!/bin/bash

################################################################################
# Submit OpenClaw terminal-rl Qwen3-8B SETA exploration v0629_3 as a detached
# 1-node 8-GPU rjob task.
#
# Baseline rjob:
#   rjob_terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629_2.sh
#
# Paired run script:
#   terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629_3.sh
#
# Only algorithmic changes relative to v0629_2:
#   EXPLORE_ADVANTAGE_GATE_MODE=outcome_status
#   EXPLORE_ADVANTAGE_OUTCOME_KEY=raw_score
#   EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE=1
#   EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN=1
#   EXPLORE_AGENT57_EPISODIC_TURN_MODE=bucket
#
# Hypothesis:
#   Replace the redundant trust * status_scale post-normalized intrinsic gate
#   with an outcome-aware quality gate, then restore coarse turn-phase context
#   to episodic novelty. High-raw_score truncated trajectories keep intrinsic
#   credit; low-raw_score truncated trajectories remain guarded.
#
# Expected validation metrics:
#   - reward/quality_gate_truncated and reward/truncated_outcome_score;
#   - reward/adv_intrinsic and reward/adv_penalty;
#   - raw_reward/pass_rate at comparable truncated_fraction and failed ratio.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_HOME="${OPENCLAW_HOME:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
NUM_GPUS="${NUM_GPUS:-8}"

EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN="${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN:-1}"
case "${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN,,}" in
    1|true|yes|on) EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN="1" ;;
    *) EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN="0" ;;
esac

EXPLORE_AGENT57_EPISODIC_TURN_MODE="${EXPLORE_AGENT57_EPISODIC_TURN_MODE:-bucket}"
case "${EXPLORE_AGENT57_EPISODIC_TURN_MODE,,}" in
    bucket|coarse) EXPLORE_AGENT57_EPISODIC_TURN_MODE="bucket" ;;
    phase|stage) EXPLORE_AGENT57_EPISODIC_TURN_MODE="phase" ;;
    none|off|0|false|no) EXPLORE_AGENT57_EPISODIC_TURN_MODE="none" ;;
    *) EXPLORE_AGENT57_EPISODIC_TURN_MODE="bucket" ;;
esac
if [[ "${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN}" == "0" ]]; then
    EXPLORE_AGENT57_EPISODIC_TURN_MODE="none"
fi
EPISODIC_TURN_TAG="epturn${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN}_${EXPLORE_AGENT57_EPISODIC_TURN_MODE}"

RUN_SCRIPT="${RUN_SCRIPT:-${OPENCLAW_HOME}/terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629_3.sh}"
RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0629_3_outcome_gate_${EPISODIC_TURN_TAG}_dualadv_think_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-8b_seta_exploration_v0629_3_${EPISODIC_TURN_TAG}}"
RJOB_LOG_ROOT="${RJOB_LOG_ROOT:-${OPENCLAW_HOME}/runs/rjob_logs/terminal_rl_v0629_3}"
RJOB_NAME="${RJOB_NAME:-a57-v629-3-turnb}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-3}"

export OPENCLAW_HOME RUN_TIMESTAMP NUM_GPUS RUN_SCRIPT RUN_ID RUN_NAME
export WANDB_GROUP RJOB_LOG_ROOT RJOB_NAME MAX_CKPT_KEEP
export EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN EXPLORE_AGENT57_EPISODIC_TURN_MODE

exec bash "${SCRIPT_DIR}/rjob_terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629.sh" "$@"
