#!/usr/bin/env bash
# Run a Harbor eval with the mode B aligned adapter (OpenClawCamelAgent).
#
# Harbor keeps owning the docker-compose lifecycle, task setup and verifier; the
# adapter only replaces the agent driver so that prompt, tool schema, iteration
# cap and sampling parameters match training. See docs/HARBOR_CAMEL_MODE_B_zh.md.
#
# Required:
#   SERVED_NAME   --served-model-name of the running SGLang server
#   MODEL_DIR     HF checkpoint dir; the adapter reads its tokenizer/chat template
#   DATASET_DIR   Terminal-Bench dataset root passed to `harbor run -p`
#
# Optional:
#   JOBS_DIR=./harbor_jobs      output root (`-o`)
#   JOB_NAME       default modeB_<SERVED_NAME>_<UTC timestamp>, with the served
#                  name reduced to [A-Za-z0-9._-]; an explicit value is used as given
#   K=3                         attempts per task (`-k`)
#   N_CONCURRENT=4              concurrent trials (`-n`)
#   TASK_ID=                    when set, restrict to one task (`-i`); use for smoke runs
#   ENVIRONMENT=docker          `-e`
#   EXTRA_ARGS=                 appended verbatim to the harbor command
#   CONDA_SH, CONDA_ENV         sourced/activated when both are set
#
# Smoke preset: TASK_ID=git-multibranch K=1 N_CONCURRENT=1
set -euo pipefail

: "${SERVED_NAME:?SERVED_NAME is required}"
: "${MODEL_DIR:?MODEL_DIR is required (HF checkpoint dir for tokenizer/chat template)}"
: "${DATASET_DIR:?DATASET_DIR is required (Terminal-Bench dataset root)}"

JOBS_DIR="${JOBS_DIR:-./harbor_jobs}"
# Only the generated default is sanitised: a served name may legitimately contain
# "/" (e.g. an org-prefixed name), which would turn the job name into a path. An
# explicitly passed JOB_NAME is the caller's choice and is used verbatim.
JOB_NAME="${JOB_NAME:-modeB_${SERVED_NAME//[^A-Za-z0-9._-]/_}_$(date -u +%Y%m%dT%H%M%SZ)}"
K="${K:-3}"
N_CONCURRENT="${N_CONCURRENT:-4}"
ENVIRONMENT="${ENVIRONMENT:-docker}"

if [[ ! -r "${MODEL_DIR}/config.json" ]]; then
  echo "[ERROR] ${MODEL_DIR}/config.json is missing or unreadable" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[ERROR] DATASET_DIR ${DATASET_DIR} is not a directory" >&2
  exit 1
fi

if [[ -n "${CONDA_SH:-}" && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

# The adapter resolves terminal-rl from its own location, so only the adapter
# directory itself has to be importable by `--agent-import-path`.
ADAPTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../adapter" && pwd)"
export PYTHONPATH="${ADAPTER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

harbor_args=(
  run
  --agent-import-path openclaw_camel_adapter:OpenClawCamelAgent
  --model "openai/${SERVED_NAME}"
  --agent-kwarg "sglang_served_name=${SERVED_NAME}"
  --agent-kwarg "hf_model_dir=${MODEL_DIR}"
  -p "${DATASET_DIR}"
  -e "${ENVIRONMENT}"
  -o "${JOBS_DIR}"
  --job-name "${JOB_NAME}"
  -k "${K}"
  -n "${N_CONCURRENT}"
  --no-delete
)
if [[ -n "${TASK_ID:-}" ]]; then
  harbor_args+=(-i "${TASK_ID}")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  harbor_args+=(${EXTRA_ARGS})
fi

echo "[INFO] job=${JOB_NAME} model=${SERVED_NAME} k=${K} n=${N_CONCURRENT} out=${JOBS_DIR}"
exec harbor "${harbor_args[@]}"
