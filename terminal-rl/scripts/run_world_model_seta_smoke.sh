#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
WM_SMOKE_PHASE="${WM_SMOKE_PHASE:-metadata}"
WM_SMOKE_ALGO="${WM_SMOKE_ALGO:-dapo}"
WM_SMOKE_OUT_ROOT="${WM_SMOKE_OUT_ROOT:-${REPO_ROOT}/runs/world_model_smoke/${STAMP}_${WM_SMOKE_ALGO}}"
WM_SMOKE_METADATA_DIR="${WM_SMOKE_METADATA_DIR:-${WM_SMOKE_OUT_ROOT}/metadata}"
WM_SMOKE_RUN_ID="${WM_SMOKE_RUN_ID:-wm_metadata_${WM_SMOKE_ALGO}_${STAMP}}"
WM_TRAIN_SCRIPT="${WM_TRAIN_SCRIPT:-terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh}"
if [[ "${WM_TRAIN_SCRIPT}" != /* ]]; then
  WM_TRAIN_SCRIPT="${REPO_ROOT}/${WM_TRAIN_SCRIPT}"
fi

mkdir -p "${WM_SMOKE_OUT_ROOT}/logs" "${WM_SMOKE_METADATA_DIR}"

cat <<EOF | tee "${WM_SMOKE_OUT_ROOT}/logs/config.txt"
[wm-smoke] repo:          ${REPO_ROOT}
[wm-smoke] phase:         ${WM_SMOKE_PHASE}
[wm-smoke] algo:          ${WM_SMOKE_ALGO}
[wm-smoke] out_root:      ${WM_SMOKE_OUT_ROOT}
[wm-smoke] metadata_dir:  ${WM_SMOKE_METADATA_DIR}
[wm-smoke] run_id:        ${WM_SMOKE_RUN_ID}
[wm-smoke] train_script:  ${WM_TRAIN_SCRIPT}
[wm-smoke] num_gpus:      ${NUM_GPUS:-auto}
[wm-smoke] actor_gpus:    ${ACTOR_GPUS:-auto}
[wm-smoke] rollout_gpus:  ${ROLLOUT_GPUS:-auto}
[wm-smoke] rollout_batch: ${ROLLOUT_BATCH_SIZE:-auto}
[wm-smoke] n_samples:     ${N_SAMPLES:-auto}
[wm-smoke] num_rollout:   ${NUM_ROLLOUT:-auto}
[wm-smoke] max_turn:      ${MAX_TURN:-auto}
EOF

if [[ "${WM_SMOKE_PHASE}" != "metadata" ]]; then
  echo "[wm-smoke] unsupported WM_SMOKE_PHASE=${WM_SMOKE_PHASE}; only metadata is implemented." >&2
  exit 2
fi
if [[ ! -f "${WM_TRAIN_SCRIPT}" ]]; then
  echo "[wm-smoke] missing WM_TRAIN_SCRIPT: ${WM_TRAIN_SCRIPT}" >&2
  exit 1
fi

WORLD_MODEL_ARGS=(
  "--skip-eval-before-train"
  "--world-model-enable"
  "--world-model-mode" "offline"
  "--world-model-loss-coef" "0"
  "--world-model-hidden-source" "none"
  "--save-debug-rollout-data" "${WM_SMOKE_METADATA_DIR}/rollout_{rollout_id}.pt"
)

EXTRA_ALGO_ARGS="${EXTRA_ALGO_ARGS:-} ${WORLD_MODEL_ARGS[*]}" \
DATASET="${DATASET:-seta}" \
ALGO="${ALGO:-${WM_SMOKE_ALGO}}" \
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}" \
RUN_ID="${RUN_ID:-${WM_SMOKE_RUN_ID}}" \
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}" \
SAVE_CKPT="${SAVE_CKPT:-}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
bash "${WM_TRAIN_SCRIPT}" \
  2>&1 | tee "${WM_SMOKE_OUT_ROOT}/logs/metadata.log"

echo "[wm-smoke] done. Outputs: ${WM_SMOKE_OUT_ROOT}"
