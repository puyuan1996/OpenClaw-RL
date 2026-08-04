#!/usr/bin/env bash
# Full SWE-bench Verified evaluation for the pre-RL Qwen3-8B checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TERMINAL_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TERMINAL_RL_DIR}/.." && pwd)"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] ${name} is required." >&2
    exit 2
  fi
}

if [[ -n "${EVAL_LIMIT:-}" ]]; then
  echo "[ERROR] This formal launcher requires all 500 SWE-bench Verified instances; unset EVAL_LIMIT." >&2
  exit 2
fi
if [[ -n "${SWEBENCH_EXPECTED_INSTANCES:-}" && "${SWEBENCH_EXPECTED_INSTANCES}" != "500" ]]; then
  echo "[ERROR] Formal SWE-bench Verified instance count is fixed at 500." >&2
  exit 2
fi
if [[ -n "${SWEBENCH_EXPECTED_DATASET_SHA256:-}" && "${SWEBENCH_EXPECTED_DATASET_SHA256}" != "4282529dbcc1b9253fa91da35b9f1768a2002b391cc90ac6a4e64575d59cfbf3" ]]; then
  echo "[ERROR] Formal SWE-bench Verified dataset SHA256 is fixed to the official converted dataset." >&2
  exit 2
fi
if [[ -n "${SWEBENCH_DEFER_GRADING:-}" && "${SWEBENCH_DEFER_GRADING}" != "1" ]]; then
  echo "[ERROR] Formal SWE-bench Verified uses prediction-only generation; SWEBENCH_DEFER_GRADING must be 1." >&2
  exit 2
fi

require_env WORKER_URLS
require_env HF_CKPT
require_env REF_LOAD
TRAIN_PYTHON="${TRAIN_PYTHON:-$(command -v python3 || true)}"
require_env TRAIN_PYTHON

export WORKER_URLS
export TRAIN_PYTHON
export EVAL_SUITE=sweverified
export EVAL_CKPT=init
export FORMAL_SWEBENCH_VERIFIED=1
export SWEBENCH_DEFER_GRADING=1
export SWEBENCH_EXPECTED_INSTANCES=500
export SWEBENCH_EXPECTED_DATASET_SHA256="4282529dbcc1b9253fa91da35b9f1768a2002b391cc90ac6a4e64575d59cfbf3"
export HF_CKPT
export REF_LOAD
export INIT_CKPT="${INIT_CKPT:-${REF_LOAD}}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${TERMINAL_RL_DIR}/configs/rollout_qwen3_think.yaml}"
export SWEBENCH_MODEL_NAME_OR_PATH="${SWEBENCH_MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"

# Qwen3 thinking-mode generation settings from the model generation_config.
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
export EVAL_TOP_K="${EVAL_TOP_K:-20}"
export EVAL_N_SAMPLES=1
export EVAL_SEED="${EVAL_SEED:-1234}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-1}"
export SGLANG_REQUEST_TIMEOUT="${SGLANG_REQUEST_TIMEOUT:-1800}"

# Two TP=2 SGLang engines keep all four H20 GPUs active while the worker runs
# up to four independent Docker tasks.
export NUM_GPUS="${NUM_GPUS:-4}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
# eval-only declares a one-rank dummy Megatron actor for argument validation;
# its TP must remain 1 and is unrelated to the two-GPU SGLang engine TP above.
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=1
export MEGATRON_TP_SIZE=1
export EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-4}"
export SWEBENCH_WORKER_MAX_CONCURRENT_BUILDS="${SWEBENCH_WORKER_MAX_CONCURRENT_BUILDS:-1}"

if [[ "${NUM_GPUS}" != "4" || "${ROLLOUT_GPUS}" != "4" || "${ROLLOUT_NUM_GPUS_PER_ENGINE}" != "2" || "${EVAL_MAX_CONCURRENCY}" != "4" ]]; then
  echo "[ERROR] Formal launcher requires NUM_GPUS=4, ROLLOUT_GPUS=4, ROLLOUT_NUM_GPUS_PER_ENGINE=2, EVAL_MAX_CONCURRENCY=4." >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec bash "${TERMINAL_RL_DIR}/terminal-rl_qwen3-8b_eval_pu.sh"
