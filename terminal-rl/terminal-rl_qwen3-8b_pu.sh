#!/usr/bin/env bash
# Terminal-RL Qwen3-8B training on a single 4-GPU node.
#
# Adapted from terminal-rl_qwen3-8b.sh with the same local-env pattern used by
# swe-rl/scripts/run_swe_rl_8b_remote_1node_v4.sh:
#   * Hardcode Qwen3-8B / Megatron paths that live under /mnt/shared-storage-user/puyuan
#   * Use the lightrft_py312 conda env for Ray / sglang
#   * 4 GPUs: actor=2, rollout=2, TP=2, engine TP=2
#   * Structured logs at logs/<run_name>/{train.log,router.log,run_config.json}
#
# Prerequisites (remote 4-GPU worker):
#   1. Pool server(s) running on reachable host(s), default port 18081:
#        bash terminal-rl/remote/run_pool_server.sh
#   2. WORKER_URLS exported, e.g.
#        export WORKER_URLS="http://<worker-ip>:18081"
#   3. ROLLOUT_PROMPT_DATA pointing to a converted seta_env train.jsonl
#
# Usage:
#   bash terminal-rl/terminal-rl_qwen3-8b_pu.sh                    # full run
#   DEBUG_MODE=1 bash terminal-rl/terminal-rl_qwen3-8b_pu.sh       # tiny rollout
#   NUM_GPUS=4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 bash ... _pu.sh         # override

set -euo pipefail
set -x

log() { echo "[$(date +'%F %T')] $*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing cmd: $1"; exit 1; }; }

# ── Conda env (Python 3.12 with transformer_engine + sglang) ─────────
# This is THE only python on this worker that has TE installed:
#   /usr/bin/python3                 → 3.10, no TE
#   /root/miniconda3/bin/python3     → 3.13, no TE  (default `which python3`!)
#   lightrft_py312/bin/python        → 3.12, TE 2.14.1   ← required
# Megatron itself lives in this repo (Megatron-LM/) and is injected via
# PYTHONPATH on the Ray runtime env below.
# v1 (run_swe_rl_4b_remote_1node.sh) worked because the user's shell already
# had this dir on PATH; we make that explicit here so the script is self-contained.
LIGHTRFT_PY312_BIN="${LIGHTRFT_PY312_BIN:-/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin}"
export PATH="${LIGHTRFT_PY312_BIN}:${PATH}"

# ── Cleanup previous processes ───────────────────────────────────────
pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
pkill -9 -f "terminal-rl.router_server" || true
pkill -9 python || true
sleep 2

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ── GPU allocation (auto-split: half actor, half rollout) ────────────
DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS:-4}}"
HALF_GPUS=$(( NUM_GPUS / 2 ))
# Default: each gets half of available GPUs (4-GPU node → 2/2, 8-GPU → 4/4).
# Important: matching node size avoids SIGSEGV in NCCL getenv() observed when
# only a subset of GPUs are used on multi-NUMA 8-GPU nodes.
ACTOR_GPUS="${ACTOR_GPUS:-${HALF_GPUS}}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-${HALF_GPUS}}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${HALF_GPUS}}"
TP_SIZE="${TP_SIZE:-${ACTOR_GPUS}}"

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS(${ACTOR_GPUS}) + ROLLOUT_GPUS(${ROLLOUT_GPUS}) > NUM_GPUS(${NUM_GPUS})"
  exit 1
fi
log "GPU config: total=${NUM_GPUS}, actor=${ACTOR_GPUS}, rollout=${ROLLOUT_GPUS}, TP=${TP_SIZE}, engine_tp=${ROLLOUT_NUM_GPUS_PER_ENGINE}"

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export REPO_ROOT
export SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime}"
export MEGATRON_DIR="${MEGATRON_DIR:-${REPO_ROOT}/Megatron-LM}"

CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3.yaml}"

# Hardcoded Qwen3-8B (matches swe-rl v4 pattern)
HF_CKPT="${HF_CKPT:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B/}"
REF_LOAD="${REF_LOAD:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B_torch_dist/}"

EXPORT_ROOT="${EXPORT_ROOT:-/mnt/shared-storage-user/narmodel/agenticrl}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}"
DEBUG_MODE="${DEBUG_MODE:-0}"
# Checkpoint saving is OFF by default. Set MAX_CKPT_KEEP=N (N>0) to enable.
# When enabled, only the latest N checkpoints are kept; older ones are auto-deleted.
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-8}"
if [[ "${DEBUG_MODE}" == "1" ]]; then
  RUN_NAME="terminal-rl_qwen3-8b_${NUM_GPUS}gpu_debug_${RUN_TIMESTAMP}"
  # Debug mode: never save checkpoints regardless of MAX_CKPT_KEEP
  MAX_CKPT_KEEP=0
else
  RUN_NAME="terminal-rl_qwen3-8b_${NUM_GPUS}gpu_${RUN_TIMESTAMP}"
fi

# ── Unified run directory (see STORAGE.md) ───────────────────────────────
# All outputs for this run go under runs/{RUN_ID}/ with structured subdirs.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
CKPT_ROOT="${CKPT_ROOT:-${EXPORT_ROOT}/ckpt}"
RUN_ID="${RUN_ID:-${RUN_NAME}}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"

# Create directory structure via run_paths.py
MAX_CKPT_KEEP="${MAX_CKPT_KEEP}" python3 "${SCRIPT_DIR}/run_paths.py" init \
  --runs-root "${RUNS_ROOT}" \
  --ckpt-root "${CKPT_ROOT}" \
  --run-id "${RUN_ID}" > /dev/null 2>&1

# Derive all paths from RUN_DIR
RUN_LOG_DIR="${RUN_DIR}/logs"
TERMINAL_SAVE_TRAJ_DIR="${RUN_DIR}/trajectories"
WANDB_DIR="${RUN_DIR}/metrics/wandb"

# ── Rollout knobs (env-configurable, baked into per-run yaml below) ──────
# MAX_TURN: max model turns per rollout (terminal_max_iterations in generate.py).
#   Empirical guidance based on 05-21 trajectory analysis (1743 trajectories):
#     - 30.0% trajectories hit max_iteration=15 (TRUNCATED) → most tasks need fewer turns
#     - Pass cases averaged 5-9 turns; tasks taking 10+ turns rarely passed
#     - Lowering to 10 trims tail-latency rollouts ≈ 33%, saving ~3 hours / 78 rollouts at 14h
#     - For exploratory runs needing more turns, override with MAX_TURN=15 or higher.
MAX_TURN="${MAX_TURN:-10}"

# Generate a per-run yaml that overlays MAX_TURN onto the base CUSTOM_CONFIG_PATH.
# This is cleaner than mutating the base yaml — different concurrent runs can pick
# different MAX_TURN without stepping on each other.
BASE_CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH}"
RUN_CUSTOM_CONFIG_PATH="${RUN_DIR}/config/rollout_config.yaml"
mkdir -p "$(dirname "${RUN_CUSTOM_CONFIG_PATH}")"
if [[ -f "${BASE_CUSTOM_CONFIG_PATH}" ]]; then
  python3 - "$BASE_CUSTOM_CONFIG_PATH" "$RUN_CUSTOM_CONFIG_PATH" "$MAX_TURN" <<'PY'
import sys, yaml
src, dst, max_turn = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = yaml.safe_load(f) or {}
cfg["max_iteration"] = max_turn
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=True)
PY
  CUSTOM_CONFIG_PATH="${RUN_CUSTOM_CONFIG_PATH}"
  echo "[config] rollout yaml -> ${RUN_CUSTOM_CONFIG_PATH} (max_iteration=${MAX_TURN})"
else
  echo "[config] base yaml ${BASE_CUSTOM_CONFIG_PATH} not found; MAX_TURN=${MAX_TURN} will not take effect"
fi

# Symlinks for backward compatibility
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/latest" 2>/dev/null || true
ln -sfn "${RUN_DIR}" "${REPO_ROOT}/tmp_doc_latest" 2>/dev/null || true
# Keep old logs/latest symlink for tools that expect it
LOG_BASE="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_BASE}" 2>/dev/null || true
ln -sfn "${RUN_LOG_DIR}" "${LOG_BASE}/latest" 2>/dev/null || true

# Only create ckpt dir and set SAVE_CKPT when saving is enabled
if (( MAX_CKPT_KEEP > 0 )); then
  SAVE_CKPT="${SAVE_CKPT:-${CKPT_ROOT}/${RUN_ID}}"
else
  SAVE_CKPT=""
fi
RESUME_LOAD="${RESUME_LOAD:-${SAVE_CKPT}}"

# Pre-flight: refuse to start if EXPORT_ROOT has < 80GB free (only when saving).
if (( MAX_CKPT_KEEP > 0 )); then
  AVAIL_GB=$(df -BG --output=avail "${EXPORT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [[ -n "${AVAIL_GB}" && "${AVAIL_GB}" -lt 80 ]]; then
    echo "[ERROR] Free space at ${EXPORT_ROOT} is only ${AVAIL_GB}G, need >= 80G"
    echo "        Clean old ckpts or set EXPORT_ROOT to a larger disk."
    df -h "${EXPORT_ROOT}" 2>&1 | tail -2
    exit 1
  fi
fi

RUN_LOG="${RUN_LOG_DIR}/train.log"

# ── Auto-mirror logs to a stable path that Claude can Read directly ──
# Two-tier scheme:
#   tmp_doc_latest/   → always the current run's logs (symlinks)
#   tmp_doc_<ts>/     → per-run snapshot (kept for history)
# Both live under the repo root so they're on shared storage (visible from
# CPU worker too, useful if you want to grep both sides at once).
TMP_DOC_ROOT="${REPO_ROOT}/tmp_doc_${RUN_TIMESTAMP}"
TMP_DOC_LATEST="${REPO_ROOT}/tmp_doc_latest"
mkdir -p "${TMP_DOC_ROOT}"
ln -sfn "${TMP_DOC_ROOT}" "${TMP_DOC_LATEST}"

GPU_RUN_LOG="${TMP_DOC_ROOT}/gpu_run.log"      # full stdout/stderr
GPU_ERR_LOG="${TMP_DOC_ROOT}/gpu_err.log"      # filtered errors (populated on failure)
GPU_TAIL_LOG="${TMP_DOC_ROOT}/gpu_tail.log"    # last ~300 lines (populated on failure)

# Tee everything to both the run-specific file and tmp_doc copy
exec > >(tee -a "${RUN_LOG}" "${GPU_RUN_LOG}") 2>&1
echo "========================================"
echo "  Terminal-RL Run: ${RUN_NAME}"
echo "  Log dir:  ${RUN_LOG_DIR}"
echo "  Ckpt:     ${SAVE_CKPT:-<disabled>}"
echo "  HF_CKPT:  ${HF_CKPT}"
echo "  REF_LOAD: ${REF_LOAD}"
echo "  MAX_CKPT_KEEP: ${MAX_CKPT_KEEP}"
echo "========================================"

# ── Model args (source qwen3-8B.sh) ──────────────────────────────────
source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

# ── Dataset & Reward Configuration ───────────────────────────────────
# DATASET: which data to train on
#   seta    = seta_env only (capability tasks, Docker-based evaluation)
#   safety  = Agent-SafetyBench only (safety tasks, no Docker needed)
#   mixed   = both combined
#
# SETA_SAFETY: safety reward mode for seta_env data
#   none       = pure outcome reward (2*accuracy - 1), no safety signal
#   clawsentry = outcome + ClawSentry L1 safety score (weighted by SAFETY_REWARD_COEF)
#
# SAFETY_BENCH_REWARD: reward mode for Agent-SafetyBench data
#   rule       = fulfillable-based: refuse harmful=+1, comply harmful=-1
#   clawsentry = use ClawSentry safety score (same mechanism as seta)
DATASET="${DATASET:-seta}"
SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
SAFETY_BENCH_REWARD="${SAFETY_BENCH_REWARD:-rule}"

SETA_DATA="${SCRIPT_DIR}/dataset/seta_env_convert/train.jsonl"
SAFETY_DATA="${SCRIPT_DIR}/dataset/agent_safetybench_convert/train.jsonl"

case "${DATASET}" in
  seta)
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SETA_DATA}}"
    ;;
  safety)
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SAFETY_DATA}}"
    ;;
  mixed)
    MIXED_DATA="${SCRIPT_DIR}/dataset/mixed_seta_safety.jsonl"
    if [[ ! -f "${MIXED_DATA}" ]] || [[ "${SETA_DATA}" -nt "${MIXED_DATA}" ]] || [[ "${SAFETY_DATA}" -nt "${MIXED_DATA}" ]]; then
      cat "${SETA_DATA}" "${SAFETY_DATA}" > "${MIXED_DATA}"
      echo "[dataset] merged seta($(wc -l < "${SETA_DATA}")) + safety($(wc -l < "${SAFETY_DATA}")) -> ${MIXED_DATA}"
    fi
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${MIXED_DATA}}"
    ;;
  *)
    echo "[ERROR] Unknown DATASET=${DATASET}. Use: seta|safety|mixed"
    exit 1
    ;;
esac

if [[ -z "${ROLLOUT_PROMPT_DATA}" ]]; then
  echo "[ERROR] ROLLOUT_PROMPT_DATA is unset."
  exit 1
fi
if [[ ! -f "${ROLLOUT_PROMPT_DATA}" ]]; then
  echo "[ERROR] ROLLOUT_PROMPT_DATA=${ROLLOUT_PROMPT_DATA} not found"
  exit 1
fi
echo "[config] DATASET=${DATASET} SETA_SAFETY=${SETA_SAFETY} SAFETY_BENCH_REWARD=${SAFETY_BENCH_REWARD}"
echo "[config] data=${ROLLOUT_PROMPT_DATA}"

# Optional dataset blacklist (issue #3 §1.X / §2.x stuck offenders).
# Default-ON; set USE_BLACKLIST=0 to keep the raw dataset.
USE_BLACKLIST="${USE_BLACKLIST:-1}"
DATASET_BLACKLIST="${DATASET_BLACKLIST:-786,96,90,456,856,210,999,305,25,684,345,553,962,916,1264,282,324,768,46,996}"
if [[ "${USE_BLACKLIST}" == "1" && -n "${DATASET_BLACKLIST}" ]]; then
  FILTERED_DATA="${ROLLOUT_PROMPT_DATA%.jsonl}.filtered.jsonl"
  python3 - "$ROLLOUT_PROMPT_DATA" "$FILTERED_DATA" "$DATASET_BLACKLIST" <<'PY'
import json, sys
src, dst, blk = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
kept = dropped = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            fout.write(line); kept += 1; continue
        # try common task-id fields
        tid = str(obj.get("task_name") or obj.get("task_id")
                  or obj.get("metadata", {}).get("task_name")
                  or obj.get("metadata", {}).get("task_id") or "")
        if tid in blk:
            dropped += 1
        else:
            fout.write(line); kept += 1
print(f"[blacklist] kept={kept} dropped={dropped} blacklist_size={len(blk)}")
PY
  ROLLOUT_PROMPT_DATA="${FILTERED_DATA}"
  echo "[blacklist] using filtered dataset: ${ROLLOUT_PROMPT_DATA}"
fi

# ── Router / worker URLs ─────────────────────────────────────────────
WORKER_URLS="${WORKER_URLS:-}"
if [[ -z "${WORKER_URLS}" ]]; then
  echo "[ERROR] WORKER_URLS is unset. Example:"
  echo "        export WORKER_URLS=http://<worker-ip>:18081"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
NODE_IP="${MASTER_ADDR}"

export USE_REMOTE_ENV="${USE_REMOTE_ENV:-1}"
export PROVIDER_NAME="${PROVIDER_NAME:-build}"
export ENV_SERVER_BIND_HOST="${ENV_SERVER_BIND_HOST:-0.0.0.0}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18080}"
export ENV_SERVER_HOST="${ENV_SERVER_HOST:-${MASTER_ADDR}}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://${ENV_SERVER_HOST}:${ENV_SERVER_PORT}}"
export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-1}"

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-${ENV_SERVER_PORT}}"
CHECK_HOST="${CHECK_HOST:-127.0.0.1}"
CHECK_WAIT_SECS="${CHECK_WAIT_SECS:-60}"

# ── Robustness knobs (informed by issue #3 postmortem) ───────────────
# Router forward to pool_server (tuned for burst of docker-compose down/up):
#   - ROUTER_FORWARD_TIMEOUT: raise 600 → 900 (issue #3 §1.X-E: 90s → 600s still
#     tight when pool is processing 32 concurrent closes; 15min is generous).
#   - ROUTER_FORWARD_RETRIES: 1 → 3 (matches agent_runner http_utils retries).
#   - ROUTER_FORWARD_RETRY_BACKOFF: 0.2 → 1.0 (exponential-ish, gives the pool
#     a real window to finish in-flight docker operations before the retry).
export ROUTER_FORWARD_TIMEOUT="${ROUTER_FORWARD_TIMEOUT:-900}"
export ROUTER_FORWARD_RETRIES="${ROUTER_FORWARD_RETRIES:-3}"
export ROUTER_FORWARD_RETRY_BACKOFF="${ROUTER_FORWARD_RETRY_BACKOFF:-1.0}"

# ── ClawSentry safety reward (L1-only, reward-only, linear-fusion baseline) ──
# Gateway runs on the same host as router_server (CPU master). All decisions
# are reward-shaping signals; agent actions are never blocked.
# ClawSentry is enabled when SETA_SAFETY=clawsentry or SAFETY_BENCH_REWARD=clawsentry.
# SAFETY_REWARD_COEF controls the linear weight (default 0.3).
CLAWSENTRY_NEEDED="0"
if [[ "${SETA_SAFETY}" == "clawsentry" ]] || [[ "${SAFETY_BENCH_REWARD}" == "clawsentry" ]]; then
  CLAWSENTRY_NEEDED="1"
fi
export SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
export SAFETY_REWARD_SUMMARY_WEIGHT="${SAFETY_REWARD_SUMMARY_WEIGHT:-0.3}"
export SAFETY_REWARD_TIMEOUT="${SAFETY_REWARD_TIMEOUT:-2.0}"
export SAFETY_REWARD_ZERO_THRESHOLD="${SAFETY_REWARD_ZERO_THRESHOLD:-1.5}"
export CS_GATEWAY_PORT="${CS_GATEWAY_PORT:-8090}"
export CS_HTTP_HOST="${CS_HTTP_HOST:-127.0.0.1}"
export CS_HTTP_URL="http://${CS_HTTP_HOST}:${CS_GATEWAY_PORT}"
export CS_AUTH_TOKEN="${CS_AUTH_TOKEN:-}"
export CS_TRAJECTORY_DB_PATH="${CS_TRAJECTORY_DB_PATH:-/tmp/clawsentry-train.db}"
export CS_LLM_PROVIDER="${CS_LLM_PROVIDER:-}"
export CS_L3_ENABLED="${CS_L3_ENABLED:-false}"
export CS_EVOLVING_ENABLED="${CS_EVOLVING_ENABLED:-false}"

# ── Trajectory export (parallels swe-rl export/swe_rollouts) ─────────────────
# Trajectory export is now ON by default (writes to runs/{run_id}/trajectories/).
# Set TERMINAL_SAVE_TRAJ_DIR="" to disable.
export TERMINAL_SAVE_TRAJ_DIR="${TERMINAL_SAVE_TRAJ_DIR}"

# Proxy bypass: some environments inject http_proxy/HTTPS_PROXY via shell rc.
# aiohttp + requests will then try to tunnel the internal router→worker traffic
# through a proxy, causing spurious connection failures. Explicitly list all
# hosts on the rollout datapath as NO_PROXY (matches swe-rl v1/v4 pattern).
ALL_WORKER_HOSTS="$(echo "${WORKER_URLS}" | tr ',' '\n' \
  | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,${MASTER_ADDR},${ALL_WORKER_HOSTS}}"
export no_proxy="${NO_PROXY}"

# Router uses `python3` which, after the PATH export above, resolves to
# lightrft_py312/bin/python. Override ROUTER_PYTHON if you want a different env.
ROUTER_PYTHON="${ROUTER_PYTHON:-python3}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_terminal_rl}"
mkdir -p "${RAY_TMPDIR}"

# ── Args ─────────────────────────────────────────────────────────────
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --rotary-base 1000000
)
# Only add --save / --load / --save-interval when checkpointing is enabled
if [[ -n "${SAVE_CKPT}" ]]; then
  CKPT_ARGS+=(--save "${SAVE_CKPT}" --save-interval "${SAVE_INTERVAL}")
fi
if [[ -n "${RESUME_LOAD}" ]]; then
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
fi

if [[ "${DEBUG_MODE}" == "1" ]]; then
  NUM_ROLLOUT=4
  ROLLOUT_BATCH_SIZE=4
  N_SAMPLES=2
  MAX_TOKENS_PER_GPU=8192
else
  NUM_ROLLOUT="${NUM_ROLLOUT:-2000}"
  # issue #3 §1: each rollout = ROLLOUT_BATCH_SIZE * N_SAMPLES concurrent lease
  # requests against the pool. With 1 worker pool (--max-tasks 16 default), the
  # original 16*8=128 burst easily saturates docker. Drop to 8*4=32 to leave
  # room for retries and to avoid the connection-reset cascade seen in run-3.
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
  N_SAMPLES="${N_SAMPLES:-4}"
  MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
fi

ROLLOUT_ARGS=(
  --prompt-data "${ROLLOUT_PROMPT_DATA}"
  --input-key task
  --rollout-shuffle
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES}"
  --rollout-max-response-len 8192
  --rollout-max-context-len 16384
  --rollout-temperature 1
  --num-steps-per-rollout 2
  --balance-data
)

EVAL_ARGS=(
  --n-samples-per-eval-prompt 16
  --eval-max-response-len 16384
  --eval-top-p 1
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --dynamic_history
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type k3
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --clip-grad 1.0
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-terminal_rl}"
    --wandb-group   "${WANDB_GROUP:-qwen3-8b_4gpu}"
    --wandb-key     "${WANDB_KEY}"
    --wandb-dir     "${WANDB_DIR}"
  )
else
  WANDB_ARGS=()
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate.generate
  --custom-rollout-log-function-path rollout_log.rollout_log
)
# --custom-config-path is optional in slime; only attach it if the yaml exists.
if [[ -f "${CUSTOM_CONFIG_PATH}" ]]; then
  CUSTOM_ARGS+=(--custom-config-path "${CUSTOM_CONFIG_PATH}")
else
  echo "WARN: custom config not found at ${CUSTOM_CONFIG_PATH}; skipping --custom-config-path"
fi

# NOTE: safety reward params are passed via env vars (RUNTIME_ENV_JSON below),
# not CLI flags, because slime's argparse rejects unknown flags.

# ── Start router ─────────────────────────────────────────────────────
ROUTER_PID=""
CS_GATEWAY_PID=""
cleanup() {
  set +e
  if [[ -n "${ROUTER_PID}" ]] && kill -0 "${ROUTER_PID}" 2>/dev/null; then
    kill "${ROUTER_PID}" || true
  fi
  if [[ -n "${CS_GATEWAY_PID}" ]] && kill -0 "${CS_GATEWAY_PID}" 2>/dev/null; then
    kill "${CS_GATEWAY_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

ROUTER_LOG="${RUN_LOG_DIR}/router.log"
log "Starting router on ${ROUTER_HOST}:${ROUTER_PORT} -> ${WORKER_URLS} (python=${ROUTER_PYTHON})"
log "  forward_timeout=${ROUTER_FORWARD_TIMEOUT}s retries=${ROUTER_FORWARD_RETRIES} backoff=${ROUTER_FORWARD_RETRY_BACKOFF}s no_proxy=${NO_PROXY}"
(
  cd "${REPO_ROOT}"
  "${ROUTER_PYTHON}" -m terminal-rl.router_server \
    --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" --workers "${WORKER_URLS}" \
    > "${ROUTER_LOG}" 2>&1 &
  echo $! > "${RUN_LOG_DIR}/router.pid"
)
ROUTER_PID="$(cat "${RUN_LOG_DIR}/router.pid")"
log "Router PID=${ROUTER_PID}, log=${ROUTER_LOG}"

# Wait for router healthz
require_cmd curl
for ((i=1; i<=CHECK_WAIT_SECS; i++)); do
  if curl -fsS "http://${CHECK_HOST}:${ROUTER_PORT}/healthz" >/dev/null 2>&1; then
    log "router ready (attempt ${i})"
    break
  fi
  sleep 1
done
curl -fsS "http://${CHECK_HOST}:${ROUTER_PORT}/status" || true
echo

# ── Start ClawSentry gateway (L1-only, reward-only) ──────────────────
if [[ "${CLAWSENTRY_NEEDED}" == "1" ]]; then
  CS_GATEWAY_LOG="${RUN_LOG_DIR}/clawsentry_gateway.log"
  log "Starting clawsentry-gateway on ${CS_HTTP_HOST}:${CS_GATEWAY_PORT} (L1-only, reward-only)"
  if ! command -v clawsentry >/dev/null 2>&1; then
    log "WARN: 'clawsentry' CLI not found in PATH; safety reward will fail-open to 0"
  else
    (
      CS_HTTP_HOST="${CS_HTTP_HOST}" \
      CS_HTTP_PORT="${CS_GATEWAY_PORT}" \
      CS_AUTH_TOKEN="${CS_AUTH_TOKEN}" \
      CS_TRAJECTORY_DB_PATH="${CS_TRAJECTORY_DB_PATH}" \
      CS_LLM_PROVIDER="${CS_LLM_PROVIDER}" \
      CS_L3_ENABLED="${CS_L3_ENABLED}" \
      CS_EVOLVING_ENABLED="${CS_EVOLVING_ENABLED}" \
      clawsentry gateway \
        --gateway-host "${CS_HTTP_HOST}" \
        --gateway-port "${CS_GATEWAY_PORT}" \
        > "${CS_GATEWAY_LOG}" 2>&1 &
      echo $! > "${RUN_LOG_DIR}/clawsentry_gateway.pid"
    )
    CS_GATEWAY_PID="$(cat "${RUN_LOG_DIR}/clawsentry_gateway.pid" 2>/dev/null || echo '')"
    log "ClawSentry gateway PID=${CS_GATEWAY_PID}, log=${CS_GATEWAY_LOG}"

    CS_OK=0
    for ((i=1; i<=20; i++)); do
      if curl -fsS --max-time 2 --noproxy '*' "${CS_HTTP_URL}/health" >/dev/null 2>&1; then
        log "clawsentry-gateway ready (attempt ${i})"
        CS_OK=1
        break
      fi
      sleep 1
    done
    if [[ "${CS_OK}" != "1" ]]; then
      log "WARN: clawsentry-gateway not healthy at ${CS_HTTP_URL}/health; safety reward will fail-open to 0"
    fi
  fi
fi

# Pre-flight: sanity check each pool worker before launching training
# (issue #3 §1.X-E: early detection of worker transport flakes).
log "Probing worker endpoints..."
IFS=',' read -r -a _WORKERS <<< "${WORKER_URLS}"
for _w in "${_WORKERS[@]}"; do
  if curl -fsS --max-time 5 --noproxy '*' "${_w}/healthz" >/dev/null 2>&1; then
    log "  [OK] ${_w}/healthz"
  else
    log "  [WARN] ${_w}/healthz unreachable — router will retry on forward"
  fi
done

# ── NVLink detection ─────────────────────────────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT:-0}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
log "HAS_NVLINK=${HAS_NVLINK}"

# ── Dump run config ──────────────────────────────────────────────────
cat > "${RUN_DIR}/config/run_config.json" <<CFGEOF
{
  "run_name": "${RUN_NAME}",
  "timestamp": "${RUN_TIMESTAMP}",
  "debug_mode": ${DEBUG_MODE},
  "model": "Qwen3-8B",
  "hf_ckpt": "${HF_CKPT}",
  "ref_load": "${REF_LOAD}",
  "save_ckpt": "${SAVE_CKPT}",
  "num_gpus": ${NUM_GPUS},
  "actor_gpus": ${ACTOR_GPUS},
  "rollout_gpus": ${ROLLOUT_GPUS},
  "tp_size": ${TP_SIZE},
  "rollout_engine_gpus": ${ROLLOUT_NUM_GPUS_PER_ENGINE},
  "prompt_data": "${ROLLOUT_PROMPT_DATA}",
  "num_rollout": ${NUM_ROLLOUT},
  "rollout_batch_size": ${ROLLOUT_BATCH_SIZE},
  "n_samples": ${N_SAMPLES},
  "max_tokens_per_gpu": ${MAX_TOKENS_PER_GPU},
  "worker_urls": "${WORKER_URLS}",
  "env_server_url": "${ENV_SERVER_URL}",
  "safety_reward_enable": "${CLAWSENTRY_NEEDED}",
  "seta_safety": "${SETA_SAFETY}",
  "safety_bench_reward": "${SAFETY_BENCH_REWARD}",
  "safety_reward_coef": "${SAFETY_REWARD_COEF}",
  "safety_reward_summary_weight": "${SAFETY_REWARD_SUMMARY_WEIGHT}",
  "safety_reward_zero_threshold": "${SAFETY_REWARD_ZERO_THRESHOLD}",
  "clawsentry_url": "${CS_HTTP_URL}",
  "clawsentry_llm_provider": "${CS_LLM_PROVIDER}",
  "clawsentry_l3_enabled": "${CS_L3_ENABLED}",
  "clawsentry_evolving_enabled": "${CS_EVOLVING_ENABLED}",
  "log_dir": "${RUN_LOG_DIR}"
}
CFGEOF

# ── Start Ray head ───────────────────────────────────────────────────
log "ray start --head ..."
ray start --head \
  --node-ip-address "${NODE_IP}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --temp-dir "${RAY_TMPDIR}"

log "Waiting for Ray dashboard http://${MASTER_ADDR}:8265 ..."
for i in {1..40}; do
  if curl -fsS --max-time 3 "http://${MASTER_ADDR}:8265/api/version" >/dev/null 2>&1; then
    log "Ray dashboard ready (attempt $i)"
    break
  fi
  sleep 3
done

# ── Build runtime env ────────────────────────────────────────────────
# Match v1 (run_swe_rl_8b_remote_1node.sh): only code dirs in PYTHONPATH.
# Do NOT inject conda site-packages — Ray workers use the default python3
# which already has Megatron/TE/sglang installed.
RUNTIME_PYTHONPATH="${MEGATRON_DIR}:${REPO_ROOT}:${SLIME_DIR}:${SCRIPT_DIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"PYTHONFAULTHANDLER\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"USE_REMOTE_ENV\": \"${USE_REMOTE_ENV}\",
    \"ENV_SERVER_URL\": \"${ENV_SERVER_URL}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${NO_PROXY}\",
    \"CS_HTTP_URL\": \"${CS_HTTP_URL}\",
    \"CS_AUTH_TOKEN\": \"${CS_AUTH_TOKEN}\",
    \"SETA_SAFETY\": \"${SETA_SAFETY}\",
    \"SAFETY_BENCH_REWARD\": \"${SAFETY_BENCH_REWARD}\",
    \"SAFETY_REWARD_COEF\": \"${SAFETY_REWARD_COEF}\",
    \"SAFETY_REWARD_SUMMARY_WEIGHT\": \"${SAFETY_REWARD_SUMMARY_WEIGHT}\",
    \"SAFETY_REWARD_TIMEOUT\": \"${SAFETY_REWARD_TIMEOUT}\",
    \"SAFETY_REWARD_ZERO_THRESHOLD\": \"${SAFETY_REWARD_ZERO_THRESHOLD}\",
    \"TERMINAL_SAVE_TRAJ_DIR\": \"${TERMINAL_SAVE_TRAJ_DIR}\",
    \"RUN_DIR\": \"${RUN_DIR}\",
    \"DATASET\": \"${DATASET}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-offline}\"
  }
}"

RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-terminal_rl_8b_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)}"

log "Submitting Ray job ${RAY_JOB_SUBMISSION_ID}"
ray job submit --address="http://${MASTER_ADDR}:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u "${SLIME_DIR}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"

set +e
ray job logs --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
RAY_LOG_EXIT=$?
RAY_STATUS_OUTPUT=$(ray job status --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
echo "${RAY_STATUS_OUTPUT}"
set -e

# ── Checkpoint cleanup: keep only the latest MAX_CKPT_KEEP per-run ───
# (issue #3 §3: ckpt accumulation is the #1 cause of ENOSPC on shared FS.)
CKPT_DIR="${SAVE_CKPT}"
if [[ -d "${CKPT_DIR}" ]] && (( MAX_CKPT_KEEP > 0 )); then
  CKPT_DIRS=($(ls -1d "${CKPT_DIR}"/iter_* 2>/dev/null | sort -t_ -k2 -n))
  TOTAL_CKPTS=${#CKPT_DIRS[@]}
  if (( TOTAL_CKPTS > MAX_CKPT_KEEP )); then
    NUM_TO_DELETE=$(( TOTAL_CKPTS - MAX_CKPT_KEEP ))
    log "Ckpt cleanup: found ${TOTAL_CKPTS}, keeping ${MAX_CKPT_KEEP}, deleting ${NUM_TO_DELETE}"
    for (( i=0; i<NUM_TO_DELETE; i++ )); do
      log "  Removing: ${CKPT_DIRS[$i]}"
      rm -rf "${CKPT_DIRS[$i]}"
    done
  fi
fi

RAY_STATUS_LOWER=$(echo "${RAY_STATUS_OUTPUT}" | tr '[:upper:]' '[:lower:]')
if [[ "${RAY_STATUS_LOWER}" == *"succeeded"* ]]; then
  log "Ray job succeeded"
  exit 0
fi

# ── Failure auto-capture ─────────────────────────────────────────────
# Generate two condensed artifacts under tmp_doc_latest/ for easy inspection:
#   gpu_tail.log : last 300 lines (often enough to see the actual stack)
#   gpu_err.log  : grep-filtered "real" error lines (CUDA/OOM/Exception/etc.)
log "Ray job failed (logs exit: ${RAY_LOG_EXIT}). Writing condensed artifacts..."
tail -n 300 "${RUN_LOG}" > "${GPU_TAIL_LOG}" 2>/dev/null || true
grep -E "Error|Exception|Traceback|CUDA|OOM|invalid device|FAILED|CheckpointException|ENOSPC|PermissionError|Connect call failed|ConnectorError|500|502" \
     "${RUN_LOG}" 2>/dev/null \
  | grep -v "FutureWarning" \
  | grep -v "DeprecationWarning" \
  | tail -n 200 \
  > "${GPU_ERR_LOG}" 2>/dev/null || true

cat <<EOF
========================================
  Run failed. Inspect:
    full   : ${GPU_RUN_LOG}
    errors : ${GPU_ERR_LOG}
    tail   : ${GPU_TAIL_LOG}
    latest : ${TMP_DOC_LATEST}/  (symlink → ${TMP_DOC_ROOT##*/})
========================================
EOF
exit 1
