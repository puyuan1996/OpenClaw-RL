#!/usr/bin/env bash
# SETA-only DAPO run for Qwen3-8B using the a3s-code harness.
#
# This wrapper keeps the PR9 command shape, pins DATASET=seta, delegates to the
# mixed nodynamic base script, and explicitly disables DAPO rejection/dynamic
# sampling. The underlying base script starts the router/Ray training path and
# forwards HARNESS_OPTION=a3s-code into the rollout config and Ray runtime env.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"

DATASET="${DATASET:-seta}"
ALGO="${ALGO:-dapo}"
HARNESS_OPTION="${HARNESS_OPTION:-a3s-code}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3_think.yaml}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_TURN="${MAX_TURN:-10}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
A3S_CODE_MAX_TOOL_ROUNDS="${A3S_CODE_MAX_TOOL_ROUNDS:-${MAX_TURN}}"

case "${HARNESS_OPTION}" in
  a3s-code|a3s_code)
    HARNESS_OPTION="a3s-code"
    ;;
  *)
    echo "[ERROR] ${0##*/} is the a3s-code SETA entrypoint; use HARNESS_OPTION=a3s-code." >&2
    exit 2
    ;;
esac

if ! [[ "${ROLLOUT_BATCH_SIZE}" =~ ^[0-9]+$ && "${N_SAMPLES}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] ROLLOUT_BATCH_SIZE and N_SAMPLES must be positive integers." >&2
  exit 2
fi
if (( ROLLOUT_BATCH_SIZE < 8 || N_SAMPLES < 8 )); then
  echo "[WARN] recommended grouped verifier RL settings are ROLLOUT_BATCH_SIZE>=8 and N_SAMPLES>=8." >&2
  echo "       got ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}, N_SAMPLES=${N_SAMPLES}" >&2
fi

# Critical baseline knob: do not add --dynamic-sampling-filter-path.
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"

# Keep EXTRA_DAPO_ARGS empty by default. Passing dynamic-sampling flags here
# would re-enable dynamic-sampling limits without the filter, which is not useful.
EXTRA_DAPO_ARGS="${EXTRA_DAPO_ARGS:-}"

RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_think_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"

export RUN_TIMESTAMP NUM_GPUS RUN_ID RUN_NAME
export DATASET ALGO HARNESS_OPTION CUSTOM_CONFIG_PATH
export ROLLOUT_BATCH_SIZE N_SAMPLES MAX_TURN MAX_CKPT_KEEP
export DAPO_DYNAMIC_SAMPLING EXTRA_DAPO_ARGS
export A3S_CODE_MAX_TOOL_ROUNDS

exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh" "$@"
