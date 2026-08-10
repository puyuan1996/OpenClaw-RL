#!/usr/bin/env bash
# Serve an HF checkpoint over SGLang for a mode B aligned Harbor eval.
#
# Every SGLang flag below other than model/tokenizer/served-name/TP is pinned to
# the value used by the evals recorded in docs/HARBOR_CAMEL_MODE_B_zh.md. Change
# them only together with that document, otherwise later runs stop being
# comparable to the recorded ones.
#
# Required:
#   MODEL_DIR    HF checkpoint directory (used for weights AND tokenizer)
#   SERVED_NAME  --served-model-name; must match the adapter's sglang_served_name
#
# Optional (defaults are the recorded values):
#   TP_SIZE=4            tensor parallel size; the base-model runs used 8
#   HOST=127.0.0.1  PORT=30000
#   MEM_FRACTION=0.6     leaves ~40% of VRAM for the KV cache
#   CONTEXT_LENGTH=40960
#   RANDOM_SEED=1234     the only seed in play: the adapter's rollout_seed is
#                        recorded in trajectory metadata but never sent to SGLang
#   CONDA_SH, CONDA_ENV  sourced/activated when both are set
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR is required (HF checkpoint directory)}"
: "${SERVED_NAME:?SERVED_NAME is required (--served-model-name)}"

TP_SIZE="${TP_SIZE:-4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.6}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-40960}"
RANDOM_SEED="${RANDOM_SEED:-1234}"

if [[ ! -r "${MODEL_DIR}/config.json" ]]; then
  echo "[ERROR] ${MODEL_DIR}/config.json is missing or unreadable" >&2
  exit 1
fi

if [[ -n "${CONDA_SH:-}" && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

exec python -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --tokenizer-path "${MODEL_DIR}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --base-gpu-id 0 \
  --dp-size 1 \
  --mem-fraction-static "${MEM_FRACTION}" \
  --attention-backend fa3 \
  --sampling-backend flashinfer \
  --grammar-backend xgrammar \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs \
  --page-size 1 \
  --dtype auto \
  --kv-cache-dtype auto \
  --context-length "${CONTEXT_LENGTH}" \
  --random-seed "${RANDOM_SEED}" \
  --trust-remote-code
