#!/usr/bin/env bash
# Submit the conservative centered-gate DiVE-PO variant.  Resource handling and
# cluster mounts stay identical to the already-used v0710 K6 rjob launcher.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OPENCLAW_HOME="${OPENCLAW_HOME:-$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"
# This validation run defaults to the 8-GPU narmodel scheduler partition.
# Both values remain overridable explicitly for later reruns.
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-narmodel_gpu}"
VARIANT_ID="${VARIANT_ID:-v0716_k6_centered_gate}"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_DIR}/run_terminal-rl_qwen3-8b_seta_dapo_dive_po_v0716_centered_gate.sh}"
RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_${VARIANT_ID}_epturn0_none_dualadv_think_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
RJOB_NAME="${RJOB_NAME:-divepo-center-v0716}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-8b_seta_exploration_${VARIANT_ID}}"

export OPENCLAW_HOME RUN_TIMESTAMP NUM_GPUS RJOB_CHARGED_GROUP VARIANT_ID RUN_SCRIPT
export RUN_ID RUN_NAME RJOB_NAME WANDB_GROUP

exec bash "${SCRIPT_DIR}/rjob_terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_0708.sh" "$@"
