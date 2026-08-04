#!/usr/bin/env bash
# SETA-only DAPO nodynamic baseline using the Claude Code harness.
#
# This wrapper mirrors terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh and only
# pins HARNESS_OPTION=claude_code. Claude Code CLI auth, endpoint, and model
# selection are read from environment variables by the shared base script.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-2}"
ACTOR_GPUS="${ACTOR_GPUS:-1}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
TP_SIZE="${TP_SIZE:-1}"

DATASET="${DATASET:-seta}"
ALGO="${ALGO:-dapo}"
HARNESS_OPTION="${HARNESS_OPTION:-claude_code}"
CLAUDE_CODE_LLM_BACKEND="${CLAUDE_CODE_LLM_BACKEND:-sglang}"
if [[ -z "${CLAUDE_CODE_MARK_NON_TRAINABLE+x}" ]]; then
  case "${CLAUDE_CODE_LLM_BACKEND}" in
    sglang|qwen|qwen-sglang|local|local-sglang)
      CLAUDE_CODE_MARK_NON_TRAINABLE="0"
      ;;
    *)
      CLAUDE_CODE_MARK_NON_TRAINABLE="1"
      ;;
  esac
fi
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3.yaml}"

# Claude Code harness holds one env lease per sampled prompt while the CLI
# runs. A single docker-env worker commonly exposes 16 task slots, so the
# old 8*8 default could immediately saturate the worker before any trajectory
# finished. Keep the wrapper conservative by default; override these env vars
# when multiple workers are available.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-2}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
CLAUDE_CODE_TURN_TIMEOUT_SEC="${CLAUDE_CODE_TURN_TIMEOUT_SEC:-180}"
CLAUDE_CODE_QWEN_MAX_NEW_TOKENS="${CLAUDE_CODE_QWEN_MAX_NEW_TOKENS:-384}"
CLAUDE_CODE_MINIMAL_SYSTEM_PROMPT="${CLAUDE_CODE_MINIMAL_SYSTEM_PROMPT:-1}"
CLAUDE_CODE_ACCEPT_QWEN_PARTIAL_ON_TIMEOUT="${CLAUDE_CODE_ACCEPT_QWEN_PARTIAL_ON_TIMEOUT:-1}"
CLAUDE_CODE_EXECUTE_QWEN_TOOL_USES="${CLAUDE_CODE_EXECUTE_QWEN_TOOL_USES:-1}"
CLAUDE_CODE_QWEN_BRIDGE_MAX_TOOL_CALLS="${CLAUDE_CODE_QWEN_BRIDGE_MAX_TOOL_CALLS:-1}"
CLAUDE_CODE_MAX_TOOL_ROUNDS="${CLAUDE_CODE_MAX_TOOL_ROUNDS:-4}"
CLAUDE_CODE_MAX_TURNS_ARG="${CLAUDE_CODE_MAX_TURNS_ARG:-force}"
CLAUDE_CODE_EXTRA_ARGS="${CLAUDE_CODE_EXTRA_ARGS:-}"
TRAJECTORY_SAVE_INTERVAL_SETA="${TRAJECTORY_SAVE_INTERVAL_SETA:-1}"
TRAJECTORY_SAVE_FAILED_SHORT_ROLLOUTS="${TRAJECTORY_SAVE_FAILED_SHORT_ROLLOUTS:-0}"
ENV_RESET_FRESH_LEASE_RETRIES="${ENV_RESET_FRESH_LEASE_RETRIES:-2}"
ENV_HEARTBEAT_INTERVAL="${ENV_HEARTBEAT_INTERVAL:-30}"
DAPO_OVERLONG_BUFFER_LEN="${DAPO_OVERLONG_BUFFER_LEN:-128}"
MAX_TURN="${MAX_TURN:-6}"
# Claude Code harness research is rollout/integration focused. Megatron's
# precision-aware CPU-offload optimizer save path can fail with missing
# master_param on small 2-GPU runs, so keep training checkpoint saves disabled
# by default while still saving terminal trajectories/metrics.
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"

# Match the reference nodynamic baseline.
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
EXTRA_DAPO_ARGS="${EXTRA_DAPO_ARGS:-}"

RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_nothink_harness-claude_code_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"

export RUN_TIMESTAMP NUM_GPUS ACTOR_GPUS ROLLOUT_GPUS ROLLOUT_NUM_GPUS_PER_ENGINE TP_SIZE RUN_ID RUN_NAME
export DATASET ALGO HARNESS_OPTION CUSTOM_CONFIG_PATH
export CLAUDE_CODE_LLM_BACKEND CLAUDE_CODE_MARK_NON_TRAINABLE CLAUDE_CODE_TURN_TIMEOUT_SEC
export CLAUDE_CODE_QWEN_MAX_NEW_TOKENS CLAUDE_CODE_MINIMAL_SYSTEM_PROMPT CLAUDE_CODE_ACCEPT_QWEN_PARTIAL_ON_TIMEOUT
export CLAUDE_CODE_EXECUTE_QWEN_TOOL_USES CLAUDE_CODE_QWEN_BRIDGE_MAX_TOOL_CALLS
export CLAUDE_CODE_MAX_TOOL_ROUNDS CLAUDE_CODE_MAX_TURNS_ARG CLAUDE_CODE_EXTRA_ARGS
export ROLLOUT_BATCH_SIZE N_SAMPLES ROLLOUT_MAX_RESPONSE_LEN MAX_TURN MAX_CKPT_KEEP
export TRAJECTORY_SAVE_INTERVAL_SETA TRAJECTORY_SAVE_FAILED_SHORT_ROLLOUTS
export ENV_RESET_FRESH_LEASE_RETRIES ENV_HEARTBEAT_INTERVAL
export DAPO_DYNAMIC_SAMPLING DAPO_OVERLONG_BUFFER_LEN EXTRA_DAPO_ARGS

exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh" "$@"
