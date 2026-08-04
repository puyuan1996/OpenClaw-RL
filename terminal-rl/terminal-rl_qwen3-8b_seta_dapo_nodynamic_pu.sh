#!/usr/bin/env bash
# Dataset-selectable GRPO/DAPO baseline wrapper for Qwen3-8B.
#
# This wrapper defaults DATASET=seta and delegates to the mixed nodynamic base
# script. For ALGO=dapo it explicitly disables rejection/dynamic sampling by
# default. Callers can set DATASET=swe-smith or ALGO=grpo to reuse the same
# stable training path.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

normalize_dataset() {
  case "$1" in
    swemith|swe-smith|swe_smith|SWE-Smith|SWESMITH)
      echo "swesmith"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"

DATASET="$(normalize_dataset "${DATASET:-seta}")"
ALGO="${ALGO:-dapo}"
HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3_think.yaml}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
if [[ "${DATASET}" == "swesmith" ]]; then
  N_SAMPLES="${N_SAMPLES:-4}"
else
  N_SAMPLES="${N_SAMPLES:-8}"
fi
MAX_TURN="${MAX_TURN:-10}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"

# Critical baseline knob: do not add --dynamic-sampling-filter-path.
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"

# Keep EXTRA_DAPO_ARGS empty by default. Passing dynamic-sampling flags here
# would re-enable dynamic-sampling limits without the filter, which is not useful.
EXTRA_DAPO_ARGS="${EXTRA_DAPO_ARGS:-}"

RUN_ALGO_NAME_TAG="${ALGO}"
if [[ "${ALGO}" == "dapo" && "${DAPO_DYNAMIC_SAMPLING}" == "0" ]]; then
  RUN_ALGO_NAME_TAG="dapo_nodynamic"
fi
RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_${DATASET}_${RUN_ALGO_NAME_TAG}_think_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"

export RUN_TIMESTAMP NUM_GPUS RUN_ID RUN_NAME
export DATASET ALGO HARNESS_OPTION CUSTOM_CONFIG_PATH
export ROLLOUT_BATCH_SIZE N_SAMPLES MAX_TURN MAX_CKPT_KEEP
export DAPO_DYNAMIC_SAMPLING EXTRA_DAPO_ARGS

exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh" "$@"
