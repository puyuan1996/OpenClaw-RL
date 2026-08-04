#!/usr/bin/env bash
# SETA-only adaptive exploration v0629_3 launch for Qwen3-8B.
#
# Baseline:
#   terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629_2.sh
#
# Variant hypothesis:
#   TRUNCATED trajectories are not uniformly bad for learning. A truncated
#   trajectory with high task raw_score/accuracy should keep most intrinsic
#   credit, while low-quality truncated trajectories should still be penalized.
#
# Only algorithmic changes relative to v0629_2:
#   EXPLORE_ADVANTAGE_GATE_MODE=outcome_status
#   EXPLORE_ADVANTAGE_OUTCOME_KEY=raw_score
#   EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN=1
#   EXPLORE_AGENT57_EPISODIC_TURN_MODE=bucket
#   quality_gate_i = floor(status_i) + (1 - floor(status_i)) * outcome_i
#   EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE=1
#   trunc_penalty_i = -0.01 * 1[truncated_i] * (1 - outcome_i)
#
# Expected validation metrics:
#   - reward/quality_gate_truncated rises when truncated samples have high raw_score;
#   - reward/truncated_outcome_score separates useful truncation from failed truncation;
#   - reward/adv_intrinsic stays visible without increasing low-quality truncation;
#   - reward/adv_penalty weakens only for high-outcome truncated trajectories.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}"
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

RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0629_3_outcome_gate_${EPISODIC_TURN_TAG}_dualadv_think_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-8b_seta_exploration_v0629_3_${EPISODIC_TURN_TAG}}"

EXPLORE_ADVANTAGE_GATE_MODE="${EXPLORE_ADVANTAGE_GATE_MODE:-outcome_status}"
EXPLORE_ADVANTAGE_OUTCOME_KEY="${EXPLORE_ADVANTAGE_OUTCOME_KEY:-raw_score}"
EXPLORE_ADVANTAGE_COMPLETED_FLOOR="${EXPLORE_ADVANTAGE_COMPLETED_FLOOR:-0.50}"
EXPLORE_ADVANTAGE_TRUNCATED_FLOOR="${EXPLORE_ADVANTAGE_TRUNCATED_FLOOR:-0.15}"
EXPLORE_ADVANTAGE_FAILED_FLOOR="${EXPLORE_ADVANTAGE_FAILED_FLOOR:-0.0}"
EXPLORE_ADVANTAGE_ABORTED_FLOOR="${EXPLORE_ADVANTAGE_ABORTED_FLOOR:-0.0}"
EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE="${EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE:-1}"

export RUN_TIMESTAMP NUM_GPUS RUN_ID RUN_NAME WANDB_GROUP
export EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN EXPLORE_AGENT57_EPISODIC_TURN_MODE
export EXPLORE_ADVANTAGE_GATE_MODE EXPLORE_ADVANTAGE_OUTCOME_KEY
export EXPLORE_ADVANTAGE_COMPLETED_FLOOR EXPLORE_ADVANTAGE_TRUNCATED_FLOOR
export EXPLORE_ADVANTAGE_FAILED_FLOOR EXPLORE_ADVANTAGE_ABORTED_FLOOR
export EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE

exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0629_2.sh" "$@"
