#!/usr/bin/env bash
# Terminal-RL GLM-5.1 Megatron-TP LoRA RL launcher.
#
# Sibling of terminal-rl_qwen3-8b_pu.sh. It keeps the terminal-rl data,
# reward, router, remote worker and Ray submission flow, but switches the actor
# to slime Megatron + TP/PP/EP + lightweight Megatron LoRA adapters.
#   * Use GLM-5.1 HF snapshot directly; bridge loads it into Megatron
#   * Use the lightrft_py312 conda env for Ray / sglang
#   * Adapter-only warm-start checkpoint saving by default
#   * Structured logs at logs/<run_name>/{train.log,router.log,run_config.json}
#
# Prerequisites:
#   1. Pool server running on worker 100.100.66.216:18081:
#        bash terminal-rl/remote/run_pool_server_pu_v2.sh
#
# Usage:
#   bash terminal-rl/terminal-rl_glm51_lora_tp.sh
#
# Structured reward observability:
#   TERMINAL_STRUCTURED_METRICS=1 writes per-rollout dataset reward breakdowns
#   to logs and to ${RUN_DIR}/logs/metrics.jsonl.

set -euo pipefail
set -x

log() { echo "[$(date +'%F %T')] $*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing cmd: $1"; exit 1; }; }
first_existing_dir() {
  local fallback="" path
  for path in "$@"; do
    if [[ -z "${fallback}" && -n "${path}" ]]; then
      fallback="${path}"
    fi
    if [[ -n "${path}" && -d "${path}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done
  printf '%s\n' "${fallback}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"
SHARED_USER_ROOT="${SHARED_USER_ROOT:-${REPO_PARENT}}"
PUYUAN_ROOT="${PUYUAN_ROOT:-/mnt/shared-storage-user/puyuan}"
export REPO_ROOT

# ── Direct-run defaults ──────────────────────────────────────────────
# Running this script directly keeps the single-node tinyrun defaults. The
# rjob wrappers override only the scale-related variables for scheduled runs.
DRY_RUN="${DRY_RUN:-0}"
DEBUG_MODE="${DEBUG_MODE:-1}"
NODE_COUNT="${NODE_COUNT:-1}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${NODE_COUNT}}"
NUM_GPUS="${NUM_GPUS:-8}"
COLOCATE="${COLOCATE:-1}"
ACTOR_GPUS="${ACTOR_GPUS:-${NUM_GPUS}}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-${NUM_GPUS}}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$(( ROLLOUT_GPUS_PER_NODE * ACTOR_NUM_NODES ))}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${ROLLOUT_GPUS}}"

ALGO="${ALGO:-dapo}"
DATASET="${DATASET:-seta}"
HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
SAFETY_BENCH_REWARD="${SAFETY_BENCH_REWARD:-rule}"
AGENTHARM_REWARD="${AGENTHARM_REWARD:-rule}"
MAX_TURN="${MAX_TURN:-10}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_glm51_think.yaml}"

SAVE_CKPT="${SAVE_CKPT:-}"
RESUME_LOAD="${RESUME_LOAD:-}"
# Keep full rollout artifacts for smoke-run inspection.
TRAJECTORY_SAVE_INTERVAL="${TRAJECTORY_SAVE_INTERVAL:-1}"
WORKER_URLS="${WORKER_URLS:-http://100.100.66.216:18081}"
NO_PROXY="${NO_PROXY:-}"
no_proxy="${no_proxy:-}"
WANDB_KEY="${WANDB_KEY:-}"
WANDB_MODE="${WANDB_MODE:-offline}"

ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"

CLIP_GRAD="${CLIP_GRAD:-0}"
LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
# Disable DAPO dynamic sampling for tinyrun stability; tiny batches often have
# constant-reward groups, and we want path validation rather than sample mining.
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
EXTRA_DAPO_ARGS="${EXTRA_DAPO_ARGS:-}"
EXTRA_GRPO_ARGS="${EXTRA_GRPO_ARGS:-}"
EXTRA_ALGO_ARGS="${EXTRA_ALGO_ARGS:-}"
USE_BLACKLIST="${USE_BLACKLIST:-1}"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj}"

# ── Conda env (Python 3.12 with transformer_engine + sglang) ─────────
# This is THE only python on this worker that has TE installed:
#   /usr/bin/python3                 → 3.10, no TE
#   /root/miniconda3/bin/python3     → 3.13, no TE  (default `which python3`!)
#   lightrft_py312/bin/python        → 3.12, TE 2.14.1   ← required
# Megatron itself lives in this repo (Megatron-LM/) and is injected via
# PYTHONPATH on the Ray runtime env below.
# v1 (run_swe_rl_4b_remote_1node.sh) worked because the user's shell already
# had this dir on PATH; we make that explicit here so the script is self-contained.
LIGHTRFT_PY312_BIN="${LIGHTRFT_PY312_BIN:-$(first_existing_dir \
  "${SHARED_USER_ROOT}/conda_envs/lightrft_py312/bin" \
  "${PUYUAN_ROOT}/conda_envs/lightrft_py312/bin")}"
if [[ ! -x "${LIGHTRFT_PY312_BIN}/python" && ! -x "${LIGHTRFT_PY312_BIN}/python3" ]]; then
  echo "[ERROR] LIGHTRFT_PY312_BIN does not contain python/python3: ${LIGHTRFT_PY312_BIN}"
  echo "        Set LIGHTRFT_PY312_BIN=/path/to/lightrft_py312/bin and rerun."
  exit 1
fi
export PATH="${LIGHTRFT_PY312_BIN}:${PATH}"
DRY_RUN="${DRY_RUN:-0}"

# ── Cleanup previous processes ───────────────────────────────────────
if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1: skipping process cleanup and Ray startup"
else
  pkill -9 sglang || true
  sleep 2
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 -f "terminal-rl.router_server" || true
  pkill -9 python || true
  sleep 2
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ── GPU allocation (auto-split: half actor, half rollout) ────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
else
  DETECTED_GPUS=0
fi
if [[ "${DETECTED_GPUS}" -le 0 ]]; then
  DETECTED_GPUS=8
fi
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
HALF_GPUS=$(( NUM_GPUS / 2 ))
# Default: colocate actor and rollout on all GPUs per node, matching slime
# large-model examples. Set COLOCATE=0 to use split actor/rollout GPUs.
# Important: matching node size avoids SIGSEGV in NCCL getenv() observed when
# only a subset of GPUs are used on multi-NUMA 8-GPU nodes.
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${NODE_COUNT:-1}}"
ACTOR_GPUS="${ACTOR_GPUS:-${NUM_GPUS}}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-${NUM_GPUS}}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$(( ROLLOUT_GPUS_PER_NODE * ACTOR_NUM_NODES ))}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${ROLLOUT_GPUS}}"

if [[ "${COLOCATE}" != "1" ]] && (( ACTOR_GPUS + ROLLOUT_GPUS_PER_NODE > NUM_GPUS )); then
  echo "ACTOR_GPUS(${ACTOR_GPUS}) + ROLLOUT_GPUS_PER_NODE(${ROLLOUT_GPUS_PER_NODE}) > NUM_GPUS(${NUM_GPUS})"
  exit 1
fi
log "GPU config: per_node=${NUM_GPUS}, actor_nodes=${ACTOR_NUM_NODES}, actor_per_node=${ACTOR_GPUS}, rollout_total=${ROLLOUT_GPUS}, rollout_per_node=${ROLLOUT_GPUS_PER_NODE}, engine_tp=${ROLLOUT_NUM_GPUS_PER_ENGINE}, colocate=${COLOCATE}"

# ── Paths ────────────────────────────────────────────────────────────
export SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime}"
export MEGATRON_DIR="${MEGATRON_DIR:-${REPO_ROOT}/Megatron-LM}"

CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_glm51_think.yaml}"

GLM51_HF_SNAPSHOT="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d"
HF_CKPT="${HF_CKPT:-${GLM51_HF_CKPT:-${GLM51_HF_SNAPSHOT}}}"
REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
if [[ ! -d "${HF_CKPT}" ]]; then
  echo "[ERROR] HF_CKPT not found: ${HF_CKPT}"
  echo "        Set HF_CKPT=/path/to/GLM-5.1 snapshot and rerun."
  exit 1
fi
if [[ ! -d "${REF_LOAD}" ]]; then
  echo "[ERROR] REF_LOAD not found: ${REF_LOAD}"
  echo "        For Megatron LoRA, REF_LOAD can normally point to the same HF base as HF_CKPT."
  exit 1
fi

EXPORT_ROOT="${EXPORT_ROOT:-$(first_existing_dir \
  "${SHARED_USER_ROOT}/agenticRL" \
  "${SHARED_USER_ROOT}/agenticrl" \
  "/mnt/shared-storage-user/narmodel/agenticrl")}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}"
DEBUG_MODE="${DEBUG_MODE:-0}"
# Defaults needed early so the run directory name carries the key experiment
# identity. Dataset construction and full validation still happen below.
ALGO="${ALGO:-dapo}"
case "${ALGO}" in
  grpo|dapo) ;;
  *)
    echo "[ERROR] Unknown ALGO=${ALGO}. Use: grpo|dapo"
    exit 1
    ;;
esac
export ALGO
DATASET="${DATASET:-seta}"
case "${DATASET}" in
  seta|safety|agentharm|mixed) ;;
  *)
    echo "[ERROR] Unknown DATASET=${DATASET}. Use: seta|safety|agentharm|mixed"
    exit 1
    ;;
esac
HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
case "${HARNESS_OPTION}" in
  camel-agent|camel_agent)
    HARNESS_OPTION="camel-agent"
    ;;
  a3s-code|a3s_code)
    HARNESS_OPTION="a3s-code"
    ;;
  *)
    echo "[ERROR] Unknown HARNESS_OPTION=${HARNESS_OPTION}. Use: camel-agent|a3s-code"
    exit 1
    ;;
esac
SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
SAFETY_BENCH_REWARD="${SAFETY_BENCH_REWARD:-rule}"
AGENTHARM_REWARD="${AGENTHARM_REWARD:-rule}"
SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
MAX_TURN="${MAX_TURN:-10}"
DAPO_EPS_CLIP_HIGH="${DAPO_EPS_CLIP_HIGH:-0.28}"
DAPO_CALCULATE_PER_TOKEN_LOSS="${DAPO_CALCULATE_PER_TOKEN_LOSS:-1}"
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-1}"

# Exploration defaults are defined in the main script as well as the wrapper so
# direct invocations remain stable under `set -u`, and Ray runtime_env can always
# receive explicit values. Bug fix: previously the exploration wrapper exported
# EXPLORE_* in the parent shell, but ray job submit workers only received the
# hand-built RUNTIME_ENV_JSON below, so generate.py often saw all exploration
# switches as disabled.
EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-${EXPLORE_PROFILE:-off}}"
EXPLORE_ENTROPY_COEF="${EXPLORE_ENTROPY_COEF:-0.0}"
EXPLORE_THINK_MODE="${EXPLORE_THINK_MODE:-0}"
EXPLORE_TEMP_HIGH="${EXPLORE_TEMP_HIGH:-}"
EXPLORE_INTRINSIC="${EXPLORE_INTRINSIC:-0}"
EXPLORE_INTRINSIC_ENABLED="${EXPLORE_INTRINSIC_ENABLED:-${EXPLORE_INTRINSIC}}"
EXPLORE_INTRINSIC_COEF="${EXPLORE_INTRINSIC_COEF:-0.1}"
EXPLORE_INTRINSIC_SCHEDULE="${EXPLORE_INTRINSIC_SCHEDULE:-constant}"
EXPLORE_INTRINSIC_DECAY_STEPS="${EXPLORE_INTRINSIC_DECAY_STEPS:-0}"
EXPLORE_INTRINSIC_GRANULARITY="${EXPLORE_INTRINSIC_GRANULARITY:-raw}"
EXPLORE_INTRINSIC_SCOPE="${EXPLORE_INTRINSIC_SCOPE:-process}"
EXPLORE_SAFETY_FILTER="${EXPLORE_SAFETY_FILTER:-0}"
EXPLORE_SAFETY_FILTER_ENABLED="${EXPLORE_SAFETY_FILTER_ENABLED:-${EXPLORE_SAFETY_FILTER}}"
EXPLORE_SAFETY_FILTER_COEF="${EXPLORE_SAFETY_FILTER_COEF:--0.5}"
EXPLORE_LPRND="${EXPLORE_LPRND:-0}"
EXPLORE_LPRND_ENABLED="${EXPLORE_LPRND_ENABLED:-${EXPLORE_LPRND}}"
EXPLORE_LPRND_COEF="${EXPLORE_LPRND_COEF:-0.05}"
EXPLORE_LPRND_SCHEDULE="${EXPLORE_LPRND_SCHEDULE:-constant}"
EXPLORE_LPRND_DECAY_STEPS="${EXPLORE_LPRND_DECAY_STEPS:-0}"
EXPLORE_LPRND_CLIP="${EXPLORE_LPRND_CLIP:-3.0}"
EXPLORE_LPRND_WARMUP="${EXPLORE_LPRND_WARMUP:-32}"
EXPLORE_ADVANTAGE_BONUS="${EXPLORE_ADVANTAGE_BONUS:-0}"
EXPLORE_ADVANTAGE_BONUS_ENABLED="${EXPLORE_ADVANTAGE_BONUS_ENABLED:-${EXPLORE_ADVANTAGE_BONUS}}"
EXPLORE_ADVANTAGE_BONUS_COMPONENTS="${EXPLORE_ADVANTAGE_BONUS_COMPONENTS:-explore_intrinsic_scaled}"
EXPLORE_ADVANTAGE_BONUS_COEF="${EXPLORE_ADVANTAGE_BONUS_COEF:-1.0}"
EXPLORE_ADVANTAGE_BONUS_CLIP="${EXPLORE_ADVANTAGE_BONUS_CLIP:-0.25}"
EXPLORE_CDE_ACTOR="${EXPLORE_CDE_ACTOR:-0}"
EXPLORE_CDE_ACTOR_ENABLED="${EXPLORE_CDE_ACTOR_ENABLED:-${EXPLORE_CDE_ACTOR}}"
EXPLORE_CDE_ACTOR_OMEGA="${EXPLORE_CDE_ACTOR_OMEGA:-0.05}"
EXPLORE_CDE_ACTOR_KAPPA="${EXPLORE_CDE_ACTOR_KAPPA:-2.0}"
EXPLORE_CDE_ACTOR_ALPHA="${EXPLORE_CDE_ACTOR_ALPHA:-0.1}"
EXPLORE_CDE_ACTOR_DECAY_STEPS="${EXPLORE_CDE_ACTOR_DECAY_STEPS:-0}"
EXPLORE_CDE_ACTOR_REWARD_GATE="${EXPLORE_CDE_ACTOR_REWARD_GATE:-nonzero}"
EXPLORE_RETRY_ATTEMPTS="${EXPLORE_RETRY_ATTEMPTS:-1}"
EXPLORE_RETRY_TRAJ_GAMMA="${EXPLORE_RETRY_TRAJ_GAMMA:-1.0}"

short_mode() {
  case "$1" in
    clawsentry) echo "cs" ;;
    dense_rule) echo "dense" ;;
    *) echo "$1" ;;
  esac
}

sanitize_run_part() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '-' | sed 's/-\\{1,\\}/-/g; s/^-//; s/-$//'
}

build_dataset_tag() {
  case "${DATASET}" in
    seta)
      echo "seta-$(short_mode "${SETA_SAFETY}")-c${SAFETY_REWARD_COEF}"
      ;;
    safety)
      echo "asb-$(short_mode "${SAFETY_BENCH_REWARD}")"
      ;;
    agentharm)
      echo "agentharm-$(short_mode "${AGENTHARM_REWARD}")"
      ;;
    mixed)
      local seta_ratio safety_ratio agentharm_ratio
      if [[ -n "${MIX_AGENTHARM_RATIO:-}" ]]; then
        seta_ratio="${MIX_SETA_RATIO:-0}"
        safety_ratio="${MIX_SAFETY_RATIO:-0}"
        agentharm_ratio="${MIX_AGENTHARM_RATIO:-0}"
      else
        seta_ratio="${MIX_SETA_RATIO:-1}"
        safety_ratio="${MIX_SAFETY_RATIO:-1}"
        agentharm_ratio="0"
      fi
      echo "mixed-s${seta_ratio}_asb${safety_ratio}_ah${agentharm_ratio}-rw$(short_mode "${SETA_SAFETY}")_$(short_mode "${SAFETY_BENCH_REWARD}")_$(short_mode "${AGENTHARM_REWARD}")-c${SAFETY_REWARD_COEF}"
      ;;
    *)
      echo "${DATASET}"
      ;;
  esac
}

build_algo_tag() {
  case "${ALGO}" in
    dapo)
      echo "dapo-ch${DAPO_EPS_CLIP_HIGH}-tok${DAPO_CALCULATE_PER_TOKEN_LOSS}-dyn${DAPO_DYNAMIC_SAMPLING}"
      ;;
    *)
      echo "${ALGO}"
      ;;
  esac
}

RUN_DATASET_TAG="$(sanitize_run_part "$(build_dataset_tag)")"
RUN_ALGO_TAG="$(sanitize_run_part "$(build_algo_tag)")"
# Tinyrun validation does not save checkpoints by default. Keep the checkpoint
# path logic below for future non-debug runs, but DEBUG_MODE forces it off.
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-8}"
if [[ "${DEBUG_MODE}" == "1" ]]; then
  RUN_NAME="${RUN_NAME:-terminal-rl_glm51_lora_tp_${ACTOR_NUM_NODES}node_${NUM_GPUS}gpu_debug_${RUN_DATASET_TAG}_${RUN_ALGO_TAG}_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
  # Debug mode: never save checkpoints regardless of MAX_CKPT_KEEP
  MAX_CKPT_KEEP=0
else
  RUN_NAME="${RUN_NAME:-terminal-rl_glm51_lora_tp_${ACTOR_NUM_NODES}node_${NUM_GPUS}gpu_${RUN_DATASET_TAG}_${RUN_ALGO_TAG}_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
fi

# ── Unified run directory (see STORAGE.md) ───────────────────────────────
# All outputs for this run go under runs/{RUN_ID}/ with structured subdirs.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
CKPT_ROOT="${CKPT_ROOT:-${EXPORT_ROOT}/ckpt}"
RUN_ID="${RUN_ID:-${RUN_NAME}}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
export CKPT_ROOT

# Create directory structure via run_paths.py. Dry-run should not touch the
# checkpoint root even when MAX_CKPT_KEEP defaults to a positive value.
RUN_PATHS_MAX_CKPT_KEEP="${MAX_CKPT_KEEP}"
if [[ "${DRY_RUN}" == "1" ]]; then
  RUN_PATHS_MAX_CKPT_KEEP=0
fi
MAX_CKPT_KEEP="${RUN_PATHS_MAX_CKPT_KEEP}" python3 "${SCRIPT_DIR}/run_paths.py" init \
  --runs-root "${RUNS_ROOT}" \
  --ckpt-root "${CKPT_ROOT}" \
  --run-id "${RUN_ID}" > /dev/null 2>&1

# Derive all paths from RUN_DIR
RUN_LOG_DIR="${RUN_DIR}/logs"
TERMINAL_SAVE_TRAJ_DIR="${RUN_DIR}/trajectories"
WANDB_DIR="${RUN_DIR}/metrics/wandb"
TERMINAL_STRUCTURED_METRICS="${TERMINAL_STRUCTURED_METRICS:-1}"
TERMINAL_METRICS_JSONL="${TERMINAL_METRICS_JSONL:-${RUN_LOG_DIR}/metrics.jsonl}"
export TERMINAL_STRUCTURED_METRICS TERMINAL_METRICS_JSONL
TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"
A3S_CODE_REPO_ROOT="${A3S_CODE_REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/a3s-lab/Code}"
A3S_CODE_CONFIG_PATH="${A3S_CODE_CONFIG_PATH:-${REPO_ROOT}/a3s-code-adapter/generated_configs/a3s-code-shared.hcl}"
A3S_CODE_PY_TAG="$("${TRAIN_PYTHON}" - <<'PY'
import platform
import sys

machine = platform.machine().lower().replace("amd64", "x86_64")
print(f"cp{sys.version_info.major}{sys.version_info.minor}-{machine}")
PY
)"
A3S_CODE_CACHE_DIR="${A3S_CODE_CACHE_DIR:-/mnt/shared-storage-user/puyuan/.cache/a3s-code-${A3S_CODE_PY_TAG}}"
A3S_CODE_WORKSPACE_ROOT="${A3S_CODE_WORKSPACE_ROOT:-${RUN_DIR}/a3s_code_workspaces}"
A3S_CODE_EXTRA_SITE_PACKAGES="${A3S_CODE_EXTRA_SITE_PACKAGES:-}"
A3S_CODE_TURN_TIMEOUT_SEC="${A3S_CODE_TURN_TIMEOUT_SEC:-900}"
A3S_CODE_TOOL_TIMEOUT_MS="${A3S_CODE_TOOL_TIMEOUT_MS:-7200000}"
A3S_CODE_MAX_TOOL_ROUNDS="${A3S_CODE_MAX_TOOL_ROUNDS:-10}"
A3S_CODE_MAX_PARSE_RETRIES="${A3S_CODE_MAX_PARSE_RETRIES:-4}"
A3S_CODE_OUTPUT_TOKENS="${A3S_CODE_OUTPUT_TOKENS:-8192}"
A3S_CODE_PLANNING_MODE="${A3S_CODE_PLANNING_MODE:-disabled}"
A3S_CODE_THINKING_BUDGET="${A3S_CODE_THINKING_BUDGET:-}"
A3S_CODE_PIP_PACKAGE="${A3S_CODE_PIP_PACKAGE:-a3s-code==3.3.0}"
ENV_EVALUATE_MAX_RETRIES="${ENV_EVALUATE_MAX_RETRIES:-3}"

a3s_code_import_check() {
  mkdir -p "${RUN_LOG_DIR}"
  A3S_CODE_CACHE_DIR="${A3S_CODE_CACHE_DIR}" \
    "${TRAIN_PYTHON}" -c "import a3s_code" > "${RUN_LOG_DIR}/a3s_code_import_check.log" 2>&1
}

a3s_code_print_cache_hint() {
  echo "[ERROR] HARNESS_OPTION=a3s-code but a3s_code import failed."
  echo "[ERROR] a3s-code==3.3.0 downloads its native wheel on first import."
  echo "[ERROR] Offline GPU workers need a prewarmed shared native cache."
  echo "[ERROR] On an online CPU/shared-storage node, run:"
  echo "  A3S_CODE_CACHE_DIR=${A3S_CODE_CACHE_DIR} ${TRAIN_PYTHON} -m pip install ${A3S_CODE_PIP_PACKAGE}"
  echo "  A3S_CODE_CACHE_DIR=${A3S_CODE_CACHE_DIR} ${TRAIN_PYTHON} -c 'import a3s_code'"
  echo "[ERROR] Then retry on the GPU worker with the same A3S_CODE_CACHE_DIR."
  echo "[ERROR] Expected native cache under: ${A3S_CODE_CACHE_DIR}/3.3.0/_native.*"
  echo "[ERROR] Import log: ${RUN_LOG_DIR}/a3s_code_import_check.log"
}

# ── Rollout knobs (env-configurable, baked into per-run yaml below) ──────
# MAX_TURN: max model turns per rollout (terminal_max_iterations in generate.py).
#   Empirical guidance based on 05-21 trajectory analysis (1743 trajectories):
#     - 30.0% trajectories hit max_iteration=15 (TRUNCATED) → most tasks need fewer turns
#     - Pass cases averaged 5-9 turns; tasks taking 10+ turns rarely passed
#     - Lowering to 10 trims tail-latency rollouts ≈ 33%, saving ~3 hours / 78 rollouts at 14h
#     - For exploratory runs needing more turns, override with MAX_TURN=15 or higher.
MAX_TURN="${MAX_TURN:-10}"
# TRAJECTORY_SAVE_INTERVAL controls full trajectory artifact storage.
#   unset / config value 1: save every rollout step (backward compatible)
#   N>1: save only when train_step % N == 0
#   0: disable trajectory artifact writes even when TERMINAL_SAVE_TRAJ_DIR is set
TRAJECTORY_SAVE_INTERVAL="${TRAJECTORY_SAVE_INTERVAL:-}"

# Generate a per-run yaml that overlays MAX_TURN onto the base CUSTOM_CONFIG_PATH.
# This is cleaner than mutating the base yaml — different concurrent runs can pick
# different MAX_TURN without stepping on each other.
BASE_CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH}"
RUN_CUSTOM_CONFIG_PATH="${RUN_DIR}/config/rollout_config.yaml"
mkdir -p "$(dirname "${RUN_CUSTOM_CONFIG_PATH}")"
if [[ -f "${BASE_CUSTOM_CONFIG_PATH}" ]]; then
  python3 - "$BASE_CUSTOM_CONFIG_PATH" "$RUN_CUSTOM_CONFIG_PATH" "$MAX_TURN" "$TRAJECTORY_SAVE_INTERVAL" "$HARNESS_OPTION" <<'PY'
import sys, yaml
src, dst, max_turn, traj_interval, harness_option = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].strip(), sys.argv[5].strip()
with open(src) as f:
    cfg = yaml.safe_load(f) or {}
cfg["max_iteration"] = max_turn
cfg["harness_option"] = harness_option
if traj_interval:
    cfg["trajectory_save_interval"] = int(traj_interval)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=True)
PY
  CUSTOM_CONFIG_PATH="${RUN_CUSTOM_CONFIG_PATH}"
  if [[ -n "${TRAJECTORY_SAVE_INTERVAL}" ]]; then
    echo "[config] rollout yaml -> ${RUN_CUSTOM_CONFIG_PATH} (max_iteration=${MAX_TURN}, harness_option=${HARNESS_OPTION}, trajectory_save_interval=${TRAJECTORY_SAVE_INTERVAL})"
  else
    echo "[config] rollout yaml -> ${RUN_CUSTOM_CONFIG_PATH} (max_iteration=${MAX_TURN}, harness_option=${HARNESS_OPTION})"
  fi
else
  echo "[config] base yaml ${BASE_CUSTOM_CONFIG_PATH} not found; MAX_TURN=${MAX_TURN} will not take effect"
fi

if [[ "${HARNESS_OPTION}" == "a3s-code" && "${DRY_RUN}" != "1" ]]; then
  if ! a3s_code_import_check; then
    A3S_SDK_DIR="${A3S_CODE_REPO_ROOT}/sdk/python"
    if [[ -d "${A3S_SDK_DIR}" ]]; then
      log "Installing a3s_code SDK with ${TRAIN_PYTHON} from ${A3S_SDK_DIR}"
      (
        cd "${A3S_SDK_DIR}"
        "${TRAIN_PYTHON}" -m pip install -q maturin
        "${TRAIN_PYTHON}" -m maturin develop --release
      )
    elif "${TRAIN_PYTHON}" -m pip show a3s-code >/dev/null 2>&1; then
      a3s_code_print_cache_hint
      exit 1
    else
      log "Installing ${A3S_CODE_PIP_PACKAGE} with ${TRAIN_PYTHON}"
      "${TRAIN_PYTHON}" -m pip install "${A3S_CODE_PIP_PACKAGE}"
    fi
    if ! a3s_code_import_check; then
      a3s_code_print_cache_hint
      exit 1
    fi
  fi
fi

# Symlinks for backward compatibility. Dry-run avoids touching stable repo links.
if [[ "${DRY_RUN}" != "1" ]]; then
  ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/latest" 2>/dev/null || true
  ln -sfn "${RUN_DIR}" "${REPO_ROOT}/tmp_doc_latest" 2>/dev/null || true
  # Keep old logs/latest symlink for tools that expect it
  LOG_BASE="${SCRIPT_DIR}/logs"
  mkdir -p "${LOG_BASE}" 2>/dev/null || true
  ln -sfn "${RUN_LOG_DIR}" "${LOG_BASE}/latest" 2>/dev/null || true
fi

# Only create ckpt dir and set SAVE_CKPT when saving is enabled
if (( MAX_CKPT_KEEP > 0 )); then
  SAVE_CKPT="${SAVE_CKPT:-${CKPT_ROOT}/${RUN_ID}}"
else
  SAVE_CKPT=""
fi
BASE_LOAD="${BASE_LOAD:-${HF_CKPT}}"
# Adapter checkpoints are warm-start inputs only: they restore LoRA weights after
# BASE_LOAD, but do not restore optimizer, scheduler, RNG, or rollout position.
# RESUME_LOAD is kept as a compatibility alias for old wrapper invocations.
MEGATRON_LORA_ADAPTER_LOAD="${MEGATRON_LORA_ADAPTER_LOAD:-${RESUME_LOAD:-}}"
export BASE_LOAD MEGATRON_LORA_ADAPTER_LOAD

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
if [[ "${DRY_RUN}" == "1" ]]; then
  TMP_DOC_ROOT="${RUN_DIR}/tmp_doc_${RUN_TIMESTAMP}"
  TMP_DOC_LATEST="${TMP_DOC_ROOT}"
  mkdir -p "${TMP_DOC_ROOT}"
else
  TMP_DOC_ROOT="${REPO_ROOT}/tmp_doc_${RUN_TIMESTAMP}"
  TMP_DOC_LATEST="${REPO_ROOT}/tmp_doc_latest"
  mkdir -p "${TMP_DOC_ROOT}"
  ln -sfn "${TMP_DOC_ROOT}" "${TMP_DOC_LATEST}"
fi

GPU_RUN_LOG="${TMP_DOC_ROOT}/gpu_run.log"      # full stdout/stderr
GPU_ERR_LOG="${TMP_DOC_ROOT}/gpu_err.log"      # filtered errors (populated on failure)
GPU_TAIL_LOG="${TMP_DOC_ROOT}/gpu_tail.log"    # last ~300 lines (populated on failure)

# Tee everything to both the run-specific file and tmp_doc copy. Keep dry-run
# stdout direct so command validation remains visible even in minimal shells.
if [[ "${DRY_RUN}" != "1" ]]; then
  TEE_LOGS=("${RUN_LOG}" "${GPU_RUN_LOG}")
  if [[ -n "${RJOB_NODE_LOG_FILE:-}" ]]; then
    mkdir -p "$(dirname "${RJOB_NODE_LOG_FILE}")"
    TEE_LOGS+=("${RJOB_NODE_LOG_FILE}")
  fi
  exec > >(tee -a "${TEE_LOGS[@]}") 2>&1
fi
echo "========================================"
echo "  Terminal-RL Run: ${RUN_NAME}"
echo "  Log dir:  ${RUN_LOG_DIR}"
if [[ -n "${RJOB_NODE_LOG_DIR:-}" ]]; then
  echo "  Node logs: ${RJOB_NODE_LOG_DIR}"
fi
echo "  Metrics:  ${TERMINAL_METRICS_JSONL} (structured=${TERMINAL_STRUCTURED_METRICS})"
echo "  Harness:  ${HARNESS_OPTION}"
echo "  Ckpt:     ${SAVE_CKPT:-<disabled>}"
echo "  Python:   ${LIGHTRFT_PY312_BIN}"
echo "  HF_CKPT:  ${HF_CKPT}"
echo "  REF_LOAD: ${REF_LOAD}"
echo "  Export:   ${EXPORT_ROOT}"
echo "  MAX_CKPT_KEEP: ${MAX_CKPT_KEEP}"
echo "========================================"

# Megatron LoRA loads the GLM HF checkpoint through Megatron Bridge.
# Do not source Megatron MODEL_ARGS here.

# ── Dataset & Reward Configuration ───────────────────────────────────
# DATASET: which data to train on
#   seta    = seta_env only (capability tasks, Docker-based evaluation)
#   safety  = Agent-SafetyBench only (safety tasks, no Docker needed)
#   agentharm = inspect_evals/agentharm only (safety tool tasks, no Docker needed)
#   mixed   = configurable mix of seta / safety / agentharm
#
# SETA_SAFETY: safety reward mode for seta_env data
#   none       = pure outcome reward (2*accuracy - 1), no safety signal
#   clawsentry = outcome + ClawSentry L1 safety score (weighted by SAFETY_REWARD_COEF)
#
# SAFETY_BENCH_REWARD: reward mode for Agent-SafetyBench data
#   rule       = fulfillable-based: refuse harmful=+1, comply harmful=-1
#   dense_rule = rule-compatible dense shaping for GRPO
#   clawsentry = use ClawSentry safety score (same mechanism as seta)
#
# AGENTHARM_REWARD: reward mode for AgentHarm data
#   rule       = target-tool/refusal based safety reward
#   dense_rule = rule-compatible dense shaping for GRPO
#   clawsentry = use ClawSentry safety score (same mechanism as seta)
#
# ALGO:
#   grpo = baseline path
#   dapo = default GLM LoRA setting, verl DAPO recipe knobs on top of GRPO estimator:
#          clip-higher, token-level loss, dynamic sampling, overlong shaping
ALGO="${ALGO:-dapo}"
case "${ALGO}" in
  grpo|dapo) ;;
  *)
    echo "[ERROR] Unknown ALGO=${ALGO}. Use: grpo|dapo"
    exit 1
    ;;
esac
export ALGO

DATASET="${DATASET:-seta}"
SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
SAFETY_BENCH_REWARD="${SAFETY_BENCH_REWARD:-rule}"
AGENT_SAFETYBENCH_REMOTE_ENV="${AGENT_SAFETYBENCH_REMOTE_ENV:-0}"
AGENTHARM_REMOTE_ENV="${AGENTHARM_REMOTE_ENV:-0}"
AGENT_SAFETYBENCH_ROOT="${AGENT_SAFETYBENCH_ROOT:-$(first_existing_dir \
  "${SHARED_USER_ROOT}/code/Agent-SafetyBench" \
  "${SHARED_USER_ROOT}/Agent-SafetyBench" \
  "${REPO_PARENT}/Agent-SafetyBench" \
  "${PUYUAN_ROOT}/code/Agent-SafetyBench")}"

SETA_DATA="${SCRIPT_DIR}/dataset/seta_env_convert/train.jsonl"
SAFETY_DATA="${SCRIPT_DIR}/dataset/agent_safetybench_convert/train.jsonl"
AGENTHARM_RAW_DIR="${SCRIPT_DIR}/dataset/agentharm"
AGENTHARM_DATA="${SCRIPT_DIR}/dataset/agentharm_convert/train.jsonl"
AGENTHARM_ROOT="${AGENTHARM_ROOT:-${AGENTHARM_RAW_DIR}}"

ensure_agentharm_dataset() {
  if [[ ! -d "${AGENTHARM_RAW_DIR}" ]]; then
    echo "[ERROR] AgentHarm raw data dir not found: ${AGENTHARM_RAW_DIR}"
    exit 1
  fi
  python3 "${SCRIPT_DIR}/data_utils/convert_agentharm_to_dataset.py" \
    --input-dir "${AGENTHARM_RAW_DIR}" \
    --output-dir "${SCRIPT_DIR}/dataset/agentharm_convert"
}

INCLUDES_SETA="0"
INCLUDES_SAFETY="0"
INCLUDES_AGENTHARM="0"

case "${DATASET}" in
  seta)
    INCLUDES_SETA="1"
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SETA_DATA}}"
    ;;
  safety)
    INCLUDES_SAFETY="1"
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SAFETY_DATA}}"
    ;;
  agentharm)
    INCLUDES_AGENTHARM="1"
    ensure_agentharm_dataset
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${AGENTHARM_DATA}}"
    ;;
  mixed)
    if [[ -n "${MIX_AGENTHARM_RATIO:-}" ]]; then
      ensure_agentharm_dataset
      MIXED_DATA="${SCRIPT_DIR}/dataset/mixed_sources.jsonl"
      MIX_ARGS=(
        --output "${MIXED_DATA}"
        --seed "${MIX_SEED:-42}"
      )
      MIX_LABELS=()
      add_mix_source() {
        local path="$1"
        local ratio="$2"
        local label="$3"
        if [[ -z "${ratio}" || "${ratio}" == "0" ]]; then
          return
        fi
        if [[ ! -f "${path}" ]]; then
          echo "[ERROR] mixed source not found: ${path}"
          exit 1
        fi
        MIX_ARGS+=(--source "${path}:${ratio}")
        MIX_LABELS+=("${label}(${ratio})")
      }
      add_mix_source "${SETA_DATA}" "${MIX_SETA_RATIO:-}" "seta"
      add_mix_source "${SAFETY_DATA}" "${MIX_SAFETY_RATIO:-}" "safety"
      add_mix_source "${AGENTHARM_DATA}" "${MIX_AGENTHARM_RATIO:-}" "agentharm"
      if [[ "${#MIX_LABELS[@]}" -eq 0 ]]; then
        echo "[ERROR] No mixed sources selected. Set MIX_SETA_RATIO, MIX_SAFETY_RATIO, or MIX_AGENTHARM_RATIO to a positive value."
        exit 1
      fi
      [[ -n "${MIX_SETA_RATIO:-}" && "${MIX_SETA_RATIO}" != "0" ]] && INCLUDES_SETA="1"
      [[ -n "${MIX_SAFETY_RATIO:-}" && "${MIX_SAFETY_RATIO}" != "0" ]] && INCLUDES_SAFETY="1"
      [[ -n "${MIX_AGENTHARM_RATIO:-}" && "${MIX_AGENTHARM_RATIO}" != "0" ]] && INCLUDES_AGENTHARM="1"
      if [[ -n "${MIX_TOTAL:-}" ]]; then
        MIX_ARGS+=(--total "${MIX_TOTAL}")
      fi
      if [[ -n "${MIX_OVERSAMPLE:-}" ]]; then
        MIX_ARGS+=(--oversample)
      fi
      python "${SCRIPT_DIR}/data_utils/mix_jsonl_datasets.py" "${MIX_ARGS[@]}"
      echo "[dataset] mixed sources: ${MIX_LABELS[*]} -> ${MIXED_DATA}"
    else
      INCLUDES_SETA="1"
      INCLUDES_SAFETY="1"
      MIXED_DATA="${SCRIPT_DIR}/dataset/mixed_seta_safety.jsonl"
      if [[ ! -f "${MIXED_DATA}" ]] || [[ "${SETA_DATA}" -nt "${MIXED_DATA}" ]] || [[ "${SAFETY_DATA}" -nt "${MIXED_DATA}" ]]; then
        if [[ -n "${MIX_SETA_RATIO:-}" ]] || [[ -n "${MIX_SAFETY_RATIO:-}" ]]; then
          MIX_ARGS=(
            --source "${SETA_DATA}:${MIX_SETA_RATIO:-1}"
            --source "${SAFETY_DATA}:${MIX_SAFETY_RATIO:-1}"
            --output "${MIXED_DATA}"
            --seed "${MIX_SEED:-42}"
          )
          if [[ -n "${MIX_TOTAL:-}" ]]; then
            MIX_ARGS+=(--total "${MIX_TOTAL}")
          fi
          if [[ -n "${MIX_OVERSAMPLE:-}" ]]; then
            MIX_ARGS+=(--oversample)
          fi
          python "${SCRIPT_DIR}/data_utils/mix_jsonl_datasets.py" "${MIX_ARGS[@]}"
        else
          cat "${SETA_DATA}" "${SAFETY_DATA}" > "${MIXED_DATA}"
          echo "[dataset] merged seta($(wc -l < "${SETA_DATA}")) + safety($(wc -l < "${SAFETY_DATA}")) -> ${MIXED_DATA}"
        fi
      fi
    fi
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${MIXED_DATA}}"
    ;;
  *)
    echo "[ERROR] Unknown DATASET=${DATASET}. Use: seta|safety|agentharm|mixed"
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
if [[ "${DATASET}" != "seta" && ! -d "${AGENT_SAFETYBENCH_ROOT}" ]]; then
  echo "[ERROR] AGENT_SAFETYBENCH_ROOT not found: ${AGENT_SAFETYBENCH_ROOT}"
  echo "        Set AGENT_SAFETYBENCH_ROOT=/path/to/Agent-SafetyBench and rerun."
  exit 1
fi
echo "[config] DATASET=${DATASET} SETA_SAFETY=${SETA_SAFETY} SAFETY_BENCH_REWARD=${SAFETY_BENCH_REWARD}"
echo "[config] data=${ROLLOUT_PROMPT_DATA}"
echo "[config] agent_safetybench_root=${AGENT_SAFETYBENCH_ROOT}"

NEEDS_ENV_ROUTER="0"
if [[ "${INCLUDES_SETA}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
if [[ "${INCLUDES_SAFETY}" == "1" && "${AGENT_SAFETYBENCH_REMOTE_ENV}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
if [[ "${INCLUDES_AGENTHARM}" == "1" && "${AGENTHARM_REMOTE_ENV}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
echo "[config] needs_env_router=${NEEDS_ENV_ROUTER} AGENT_SAFETYBENCH_REMOTE_ENV=${AGENT_SAFETYBENCH_REMOTE_ENV} AGENTHARM_REMOTE_ENV=${AGENTHARM_REMOTE_ENV}"

# Optional dataset blacklist (issue #3 §1.X / §2.x stuck offenders).
# Default-ON; set USE_BLACKLIST=0 to keep the raw dataset.
USE_BLACKLIST="${USE_BLACKLIST:-1}"
DATASET_BLACKLIST="${DATASET_BLACKLIST:-786,96,90,456,856,210,999,305,25,684,345,553,962,916,1264,282,324,768,46,996}"
if [[ "${USE_BLACKLIST}" == "1" && -n "${DATASET_BLACKLIST}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    FILTERED_DATA="${RUN_DIR}/config/$(basename "${ROLLOUT_PROMPT_DATA%.jsonl}").filtered.jsonl"
  else
    FILTERED_DATA="${ROLLOUT_PROMPT_DATA%.jsonl}.filtered.jsonl"
  fi
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
if [[ "${DRY_RUN}" != "1" && "${NEEDS_ENV_ROUTER}" == "1" && -z "${WORKER_URLS}" ]]; then
  echo "[ERROR] WORKER_URLS is unset. Example:"
  echo "        export WORKER_URLS=http://<worker-ip>:18081"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_ADDR
NODE_IP="${MASTER_ADDR}"

export USE_REMOTE_ENV="${USE_REMOTE_ENV:-${NEEDS_ENV_ROUTER}}"
export PROVIDER_NAME="${PROVIDER_NAME:-build}"
export ENV_SERVER_BIND_HOST="${ENV_SERVER_BIND_HOST:-0.0.0.0}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18080}"
export ENV_SERVER_HOST="${ENV_SERVER_HOST:-${MASTER_ADDR}}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://${ENV_SERVER_HOST}:${ENV_SERVER_PORT}}"
export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-${NEEDS_ENV_ROUTER}}"
export AGENT_SAFETYBENCH_REMOTE_ENV
export AGENTHARM_REMOTE_ENV
export AGENTHARM_ROOT
export AGENTHARM_REWARD

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
# ClawSentry is enabled only for the active dataset family.
# SAFETY_REWARD_COEF controls the linear weight (default 0.3).
CLAWSENTRY_NEEDED="0"
if [[ "${INCLUDES_SETA}" == "1" && "${SETA_SAFETY}" == "clawsentry" ]]; then
  CLAWSENTRY_NEEDED="1"
fi
if [[ "${INCLUDES_SAFETY}" == "1" && "${SAFETY_BENCH_REWARD}" == "clawsentry" ]]; then
  CLAWSENTRY_NEEDED="1"
fi
if [[ "${INCLUDES_AGENTHARM}" == "1" && "${AGENTHARM_REWARD}" == "clawsentry" ]]; then
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
export TRAJECTORY_SAVE_INTERVAL="${TRAJECTORY_SAVE_INTERVAL}"

# Proxy bypass: some environments inject http_proxy/HTTPS_PROXY via shell rc.
# aiohttp + requests will then try to tunnel the internal router→worker traffic
# through a proxy, causing spurious connection failures. Explicitly list all
# hosts on the rollout datapath as NO_PROXY (matches swe-rl v1/v4 pattern).
ALL_WORKER_HOSTS=""
if [[ -n "${WORKER_URLS}" ]]; then
  ALL_WORKER_HOSTS="$(echo "${WORKER_URLS}" | tr ',' '\n' \
    | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
fi
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,${MASTER_ADDR}${ALL_WORKER_HOSTS:+,${ALL_WORKER_HOSTS}}}"
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
  --load "${BASE_LOAD}"
)
# Only add --save / --load / --save-interval when checkpointing is enabled
if [[ -n "${SAVE_CKPT}" ]]; then
  CKPT_ARGS+=(--save "${SAVE_CKPT}" --save-interval "${SAVE_INTERVAL}")
fi
if [[ -n "${MEGATRON_LORA_ADAPTER_LOAD}" ]]; then
  CKPT_ARGS+=(--megatron-lora-adapter-load "${MEGATRON_LORA_ADAPTER_LOAD}")
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
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"

if (( NUM_STEPS_PER_ROLLOUT <= 0 )); then
  echo "[ERROR] NUM_STEPS_PER_ROLLOUT(${NUM_STEPS_PER_ROLLOUT}) must be > 0"
  exit 1
fi
ROLLOUT_TOTAL_SAMPLES=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES ))
EXPECTED_GLOBAL_BATCH_SIZE=$(( ROLLOUT_TOTAL_SAMPLES / NUM_STEPS_PER_ROLLOUT ))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${EXPECTED_GLOBAL_BATCH_SIZE}}"
TP_SIZE="${TP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
ETP_SIZE="${ETP_SIZE:-1}"
ACTOR_WORLD_SIZE=$(( ACTOR_NUM_NODES * ACTOR_GPUS ))
MODEL_PARALLEL_SIZE=$(( TP_SIZE * PP_SIZE * CP_SIZE ))
if (( MODEL_PARALLEL_SIZE <= 0 || ACTOR_WORLD_SIZE % MODEL_PARALLEL_SIZE != 0 )); then
  echo "[ERROR] actor world size ${ACTOR_WORLD_SIZE} must be divisible by TP_SIZE(${TP_SIZE}) * PP_SIZE(${PP_SIZE}) * CP_SIZE(${CP_SIZE}) = ${MODEL_PARALLEL_SIZE}"
  exit 1
fi
ACTOR_DP_SIZE=$(( ACTOR_WORLD_SIZE / MODEL_PARALLEL_SIZE ))

if (( GLOBAL_BATCH_SIZE != EXPECTED_GLOBAL_BATCH_SIZE )); then
  echo "[ERROR] GLOBAL_BATCH_SIZE(${GLOBAL_BATCH_SIZE}) must equal ROLLOUT_BATCH_SIZE(${ROLLOUT_BATCH_SIZE}) * N_SAMPLES(${N_SAMPLES}) // NUM_STEPS_PER_ROLLOUT(${NUM_STEPS_PER_ROLLOUT}) = ${EXPECTED_GLOBAL_BATCH_SIZE}"
  echo "[ERROR] For 1-node 4-actor-GPU tiny validation, use for example: ROLLOUT_BATCH_SIZE=4 N_SAMPLES=2 NUM_STEPS_PER_ROLLOUT=2"
  exit 1
fi
if (( GLOBAL_BATCH_SIZE <= 0 )); then
  echo "[ERROR] GLOBAL_BATCH_SIZE computed as ${GLOBAL_BATCH_SIZE}. Increase ROLLOUT_BATCH_SIZE * N_SAMPLES or reduce NUM_STEPS_PER_ROLLOUT."
  echo "[ERROR] For 1-node 4-actor-GPU tiny validation, use for example: ROLLOUT_BATCH_SIZE=4 N_SAMPLES=2 NUM_STEPS_PER_ROLLOUT=2"
  exit 1
fi
if (( GLOBAL_BATCH_SIZE % ACTOR_DP_SIZE != 0 )); then
  echo "[ERROR] GLOBAL_BATCH_SIZE(${GLOBAL_BATCH_SIZE}) must be divisible by Megatron actor dp size ${ACTOR_DP_SIZE}."
  exit 1
fi
log "Batch config: rollout_batch_size=${ROLLOUT_BATCH_SIZE}, n_samples=${N_SAMPLES}, num_steps_per_rollout=${NUM_STEPS_PER_ROLLOUT}, global_batch_size=${GLOBAL_BATCH_SIZE}, actor_dp_size=${ACTOR_DP_SIZE}, tp=${TP_SIZE}, pp=${PP_SIZE}, cp=${CP_SIZE}, ep=${EP_SIZE}, etp=${ETP_SIZE}"

ROLLOUT_ARGS=(
  --prompt-data "${ROLLOUT_PROMPT_DATA}"
  --input-key task
  --rollout-shuffle
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

EVAL_ARGS=(
  --n-samples-per-eval-prompt 16
  --eval-max-response-len 16384
  --eval-top-p 1
)

TRAIN_ENV_VARS_JSON="{\"PYTORCH_CUDA_ALLOC_CONF\":\"${PYTORCH_CUDA_ALLOC_CONF}\",\"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\":\"${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}\",\"TORCH_NCCL_ENABLE_MONITORING\":\"${TORCH_NCCL_ENABLE_MONITORING}\",\"NCCL_CUMEM_ENABLE\":\"${NCCL_CUMEM_ENABLE}\",\"TORCH_NCCL_AVOID_RECORD_STREAMS\":\"${TORCH_NCCL_AVOID_RECORD_STREAMS}\"}"

MODEL_ARGS=(
  --disable-bias-linear
  --qk-layernorm
  --group-query-attention
  --multi-latent-attention
  --experimental-attention-variant dsa
  --num-attention-heads 64
  --num-query-groups 64
  --num-layers 78
  --hidden-size 6144
  --ffn-hidden-size 12288
  --max-position-embeddings 202752
  --normalization RMSNorm
  --position-embedding-type rope
  --rotary-percent 1.0
  --rotary-base 1000000
  --swiglu
  --untie-embeddings-and-output-weights
  --vocab-size 154880
  --q-lora-rank 2048
  --kv-lora-rank 512
  --qk-head-dim 256
  --qk-pos-emb-head-dim 64
  --v-head-dim 256
  --dsa-indexer-n-heads 32
  --dsa-indexer-head-dim 128
  --dsa-indexer-topk 2048
  --moe-ffn-hidden-size 2048
  --moe-shared-expert-intermediate-size 2048
  --moe-router-pre-softmax
  --moe-router-score-function sigmoid
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 0
  --moe-router-load-balancing-type none
  --moe-router-topk 8
  --moe-router-topk-scaling-factor 2.5
  --moe-layer-freq "[0]*3+[1]*75"
  --num-experts 256
  --moe-grouped-gemm
  --moe-router-dtype fp32
  --moe-permute-fusion
  --moe-aux-loss-coeff 0
)

TRAIN_BACKEND_ARGS=(
  --train-backend megatron
  --megatron-to-hf-mode bridge
  --model-name glm_moe_dsa
  --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-536870912}"
  --gradient-checkpointing
  --train-env-vars "${TRAIN_ENV_VARS_JSON}"
)

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj}"

LORA_ARGS=(
  --use-megatron-lora
  --megatron-lora-save-adapter-only
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
)
if [[ "${MEGATRON_LORA_INCLUDE_EXPERTS:-0}" == "1" ]]; then
  LORA_ARGS+=(--megatron-lora-include-experts)
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size "${ETP_SIZE}"
  --sequence-parallel
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --dynamic_history
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type k3
)

DAPO_EPS_CLIP_LOW="${DAPO_EPS_CLIP_LOW:-0.2}"
DAPO_EPS_CLIP_HIGH="${DAPO_EPS_CLIP_HIGH:-0.28}"
DAPO_USE_KL_LOSS="${DAPO_USE_KL_LOSS:-0}"
DAPO_KL_LOSS_COEF="${DAPO_KL_LOSS_COEF:-0.0}"
DAPO_KL_LOSS_TYPE="${DAPO_KL_LOSS_TYPE:-k3}"
DAPO_CALCULATE_PER_TOKEN_LOSS="${DAPO_CALCULATE_PER_TOKEN_LOSS:-1}"
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-1}"
DAPO_DYNAMIC_FILTER_PATH="${DAPO_DYNAMIC_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
DAPO_OVER_SAMPLING_BATCH_SIZE="${DAPO_OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
DAPO_GRPO_STD_NORMALIZATION="${DAPO_GRPO_STD_NORMALIZATION:-1}"
DAPO_REWARDS_NORMALIZATION="${DAPO_REWARDS_NORMALIZATION:-0}"
DAPO_OVERLONG_BUFFER_ENABLE="${DAPO_OVERLONG_BUFFER_ENABLE:-1}"
DAPO_OVERLONG_BUFFER_LEN="${DAPO_OVERLONG_BUFFER_LEN:-4096}"
DAPO_OVERLONG_PENALTY_FACTOR="${DAPO_OVERLONG_PENALTY_FACTOR:-1.0}"

DAPO_ARGS=(
  --advantage-estimator grpo
  --dynamic_history
  --eps-clip "${DAPO_EPS_CLIP_LOW}"
  --eps-clip-high "${DAPO_EPS_CLIP_HIGH}"
)

if [[ "${DAPO_CALCULATE_PER_TOKEN_LOSS}" == "1" ]]; then
  DAPO_ARGS+=(--calculate-per-token-loss)
fi
if [[ "${DAPO_USE_KL_LOSS}" == "1" ]]; then
  DAPO_ARGS+=(--use-kl-loss --kl-loss-coef "${DAPO_KL_LOSS_COEF}" --kl-loss-type "${DAPO_KL_LOSS_TYPE}")
fi
if [[ "${DAPO_DYNAMIC_SAMPLING}" == "1" ]]; then
  if (( DAPO_OVER_SAMPLING_BATCH_SIZE < ROLLOUT_BATCH_SIZE )); then
    echo "[ERROR] DAPO_OVER_SAMPLING_BATCH_SIZE(${DAPO_OVER_SAMPLING_BATCH_SIZE}) must be >= ROLLOUT_BATCH_SIZE(${ROLLOUT_BATCH_SIZE})"
    exit 1
  fi
  DAPO_ARGS+=(
    --dynamic-sampling-filter-path "${DAPO_DYNAMIC_FILTER_PATH}"
    --over-sampling-batch-size "${DAPO_OVER_SAMPLING_BATCH_SIZE}"
  )
fi
if [[ "${DAPO_GRPO_STD_NORMALIZATION}" == "0" ]]; then
  DAPO_ARGS+=(--disable-grpo-std-normalization)
fi
if [[ "${DAPO_REWARDS_NORMALIZATION}" == "0" ]]; then
  DAPO_ARGS+=(--disable-rewards-normalization)
fi

case "${ALGO}" in
  grpo)
    ALGO_ARGS=("${GRPO_ARGS[@]}")
    ALGO_EXTRA_ARGS="${EXTRA_GRPO_ARGS:-} ${EXTRA_ALGO_ARGS:-}"
    ;;
  dapo)
    ALGO_ARGS=("${DAPO_ARGS[@]}")
    ALGO_EXTRA_ARGS="${EXTRA_DAPO_ARGS:-} ${EXTRA_ALGO_ARGS:-}"
    ;;
esac
ALGO_EXTRA_ARGS_ARRAY=()
if [[ -n "${ALGO_EXTRA_ARGS// }" ]]; then
  # Preserve the existing unquoted EXTRA_GRPO_ARGS behavior for compatibility.
  ALGO_EXTRA_ARGS_ARRAY=(${ALGO_EXTRA_ARGS})
fi
log "Algorithm config: ALGO=${ALGO} args=${ALGO_ARGS[*]} extra=${ALGO_EXTRA_ARGS_ARRAY[*]:-<none>}"
log "Exploration config: profile=${EXPLORATION_PROFILE} entropy=${EXPLORE_ENTROPY_COEF} intrinsic=${EXPLORE_INTRINSIC_ENABLED}/${EXPLORE_INTRINSIC} coef=${EXPLORE_INTRINSIC_COEF} schedule=${EXPLORE_INTRINSIC_SCHEDULE}/${EXPLORE_INTRINSIC_DECAY_STEPS} granularity=${EXPLORE_INTRINSIC_GRANULARITY} scope=${EXPLORE_INTRINSIC_SCOPE} safety_filter=${EXPLORE_SAFETY_FILTER_ENABLED}/${EXPLORE_SAFETY_FILTER} lprnd=${EXPLORE_LPRND_ENABLED}/${EXPLORE_LPRND} coef=${EXPLORE_LPRND_COEF} schedule=${EXPLORE_LPRND_SCHEDULE}/${EXPLORE_LPRND_DECAY_STEPS} cde_actor=${EXPLORE_CDE_ACTOR_ENABLED}/${EXPLORE_CDE_ACTOR} omega=${EXPLORE_CDE_ACTOR_OMEGA} alpha=${EXPLORE_CDE_ACTOR_ALPHA} kappa=${EXPLORE_CDE_ACTOR_KAPPA} gate=${EXPLORE_CDE_ACTOR_REWARD_GATE} decay_steps=${EXPLORE_CDE_ACTOR_DECAY_STEPS} post_norm_bonus=${EXPLORE_ADVANTAGE_BONUS_ENABLED}/${EXPLORE_ADVANTAGE_BONUS} components=${EXPLORE_ADVANTAGE_BONUS_COMPONENTS} coef=${EXPLORE_ADVANTAGE_BONUS_COEF} clip=${EXPLORE_ADVANTAGE_BONUS_CLIP}"
if [[ "${ALGO}" == "dapo" ]]; then
  log "DAPO knobs: clip_low=${DAPO_EPS_CLIP_LOW} clip_high=${DAPO_EPS_CLIP_HIGH} token_loss=${DAPO_CALCULATE_PER_TOKEN_LOSS} dynamic_sampling=${DAPO_DYNAMIC_SAMPLING} rewards_norm=${DAPO_REWARDS_NORMALIZATION} overlong=${DAPO_OVERLONG_BUFFER_ENABLE}/${DAPO_OVERLONG_BUFFER_LEN}/${DAPO_OVERLONG_PENALTY_FACTOR}"
fi

CLIP_GRAD="${CLIP_GRAD:-0}"

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.98}"
  --clip-grad "${CLIP_GRAD}"
)

if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(
	    --use-wandb
	    --wandb-project "${WANDB_PROJECT:-terminal_rl}"
	    --wandb-group   "${WANDB_GROUP:-glm51_lora_tp}"
	    --wandb-key     "${WANDB_KEY}"
	    --wandb-dir     "${WANDB_DIR}"
	  )
else
  WANDB_ARGS=()
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.6}"
)

MISC_ARGS=()

CUSTOM_ARGS=(
  --custom-generate-function-path generate.generate
  --custom-rollout-log-function-path rollout_log.rollout_log
  --custom-eval-rollout-log-function-path rollout_log.eval_rollout_log
)
if [[ "${EXPLORE_ADVANTAGE_BONUS_ENABLED}" == "1" ]]; then
  CUSTOM_ARGS+=(--custom-reward-post-process-path reward_postprocess.post_process_rewards)
fi
# --custom-config-path is optional in slime; only attach it if the yaml exists.
if [[ -f "${CUSTOM_CONFIG_PATH}" ]]; then
  CUSTOM_ARGS+=(--custom-config-path "${CUSTOM_CONFIG_PATH}")
else
  echo "WARN: custom config not found at ${CUSTOM_CONFIG_PATH}; skipping --custom-config-path"
fi

TRAIN_ARGS=(
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  --num-gpus-per-node "${NUM_GPUS}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${ALGO_ARGS[@]}"
  "${ALGO_EXTRA_ARGS_ARRAY[@]}"
  "${WANDB_ARGS[@]}"
  "${TRAIN_BACKEND_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${CUSTOM_ARGS[@]}"
  "${LORA_ARGS[@]}"
)
if [[ "${COLOCATE}" == "1" ]]; then
  TRAIN_ARGS+=(--colocate)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1: final train_async command only; router/Ray/training will not start"
  printf '[dry-run] '
  printf '%q ' "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py" "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
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
require_cmd curl
if [[ "${NEEDS_ENV_ROUTER}" == "1" ]]; then
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
  for ((i=1; i<=CHECK_WAIT_SECS; i++)); do
    if curl -fsS "http://${CHECK_HOST}:${ROUTER_PORT}/healthz" >/dev/null 2>&1; then
      log "router ready (attempt ${i})"
      break
    fi
    sleep 1
  done
  curl -fsS "http://${CHECK_HOST}:${ROUTER_PORT}/status" || true
  echo
else
  log "Skipping terminal env router; Agent-SafetyBench uses local env backend"
fi

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
if [[ "${NEEDS_ENV_ROUTER}" == "1" ]]; then
  log "Probing worker endpoints..."
  IFS=',' read -r -a _WORKERS <<< "${WORKER_URLS}"
  for _w in "${_WORKERS[@]}"; do
    if curl -fsS --max-time 5 --noproxy '*' "${_w}/healthz" >/dev/null 2>&1; then
      log "  [OK] ${_w}/healthz"
    else
      log "  [WARN] ${_w}/healthz unreachable — router will retry on forward"
    fi
  done
fi

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
  "run_dataset_tag": "${RUN_DATASET_TAG}",
  "run_algo_tag": "${RUN_ALGO_TAG}",
  "debug_mode": ${DEBUG_MODE},
  "dry_run": "${DRY_RUN}",
  "algo": "${ALGO}",
  "harness_option": "${HARNESS_OPTION}",
  "model": "GLM-5.1",
  "train_backend": "megatron",
  "clip_grad": "${CLIP_GRAD}",
  "lora_enabled": true,
  "lora_rank": "${LORA_RANK}",
  "lora_alpha": "${LORA_ALPHA}",
  "lora_dropout": "${LORA_DROPOUT}",
  "lora_target_modules": "${LORA_TARGET_MODULES}",
  "hf_ckpt": "${HF_CKPT}",
  "ref_load": "${REF_LOAD}",
  "base_load": "${BASE_LOAD}",
  "megatron_lora_adapter_load": "${MEGATRON_LORA_ADAPTER_LOAD}",
  "megatron_lora_adapter_mode": "warm_start",
  "save_ckpt": "${SAVE_CKPT}",
  "num_gpus": ${NUM_GPUS},
  "actor_num_nodes": ${ACTOR_NUM_NODES},
  "actor_gpus": ${ACTOR_GPUS},
  "tp_size": ${TP_SIZE},
  "pp_size": ${PP_SIZE},
  "cp_size": ${CP_SIZE},
  "ep_size": ${EP_SIZE},
  "etp_size": ${ETP_SIZE},
  "colocate": "${COLOCATE}",
  "rollout_gpus": ${ROLLOUT_GPUS},
  "rollout_gpus_per_node": ${ROLLOUT_GPUS_PER_NODE},
  "rollout_engine_gpus": ${ROLLOUT_NUM_GPUS_PER_ENGINE},
  "dataset": "${DATASET}",
  "includes_seta": "${INCLUDES_SETA}",
  "includes_safety": "${INCLUDES_SAFETY}",
  "includes_agentharm": "${INCLUDES_AGENTHARM}",
  "prompt_data": "${ROLLOUT_PROMPT_DATA}",
  "num_rollout": ${NUM_ROLLOUT},
  "rollout_batch_size": ${ROLLOUT_BATCH_SIZE},
  "n_samples": ${N_SAMPLES},
  "num_steps_per_rollout": ${NUM_STEPS_PER_ROLLOUT},
  "global_batch_size": ${GLOBAL_BATCH_SIZE},
  "actor_dp_size": ${ACTOR_DP_SIZE},
  "rollout_max_response_len": ${ROLLOUT_MAX_RESPONSE_LEN},
  "rollout_max_context_len": ${ROLLOUT_MAX_CONTEXT_LEN},
  "max_tokens_per_gpu": ${MAX_TOKENS_PER_GPU},
  "worker_urls": "${WORKER_URLS}",
  "env_server_url": "${ENV_SERVER_URL}",
  "needs_env_router": "${NEEDS_ENV_ROUTER}",
  "agent_safetybench_remote_env": "${AGENT_SAFETYBENCH_REMOTE_ENV}",
  "agentharm_remote_env": "${AGENTHARM_REMOTE_ENV}",
  "safety_reward_enable": "${CLAWSENTRY_NEEDED}",
  "seta_safety": "${SETA_SAFETY}",
  "safety_bench_reward": "${SAFETY_BENCH_REWARD}",
  "agentharm_reward": "${AGENTHARM_REWARD}",
  "agentharm_root": "${AGENTHARM_ROOT}",
  "dapo_eps_clip_low": "${DAPO_EPS_CLIP_LOW}",
  "dapo_eps_clip_high": "${DAPO_EPS_CLIP_HIGH}",
  "dapo_calculate_per_token_loss": "${DAPO_CALCULATE_PER_TOKEN_LOSS}",
  "dapo_dynamic_sampling": "${DAPO_DYNAMIC_SAMPLING}",
  "dapo_dynamic_filter_path": "${DAPO_DYNAMIC_FILTER_PATH}",
  "dapo_over_sampling_batch_size": "${DAPO_OVER_SAMPLING_BATCH_SIZE}",
  "dapo_grpo_std_normalization": "${DAPO_GRPO_STD_NORMALIZATION}",
  "dapo_rewards_normalization": "${DAPO_REWARDS_NORMALIZATION}",
  "dapo_use_kl_loss": "${DAPO_USE_KL_LOSS}",
  "dapo_kl_loss_coef": "${DAPO_KL_LOSS_COEF}",
  "dapo_overlong_buffer_enable": "${DAPO_OVERLONG_BUFFER_ENABLE}",
  "dapo_overlong_buffer_len": "${DAPO_OVERLONG_BUFFER_LEN}",
  "dapo_overlong_penalty_factor": "${DAPO_OVERLONG_PENALTY_FACTOR}",
  "safety_reward_coef": "${SAFETY_REWARD_COEF}",
  "safety_reward_summary_weight": "${SAFETY_REWARD_SUMMARY_WEIGHT}",
  "safety_reward_zero_threshold": "${SAFETY_REWARD_ZERO_THRESHOLD}",
  "trajectory_save_interval_env": "${TRAJECTORY_SAVE_INTERVAL}",
  "exploration_profile": "${EXPLORATION_PROFILE}",
  "explore_entropy_coef": "${EXPLORE_ENTROPY_COEF}",
  "explore_think_mode": "${EXPLORE_THINK_MODE}",
  "explore_temp_high": "${EXPLORE_TEMP_HIGH}",
  "explore_intrinsic": "${EXPLORE_INTRINSIC}",
  "explore_intrinsic_enabled": "${EXPLORE_INTRINSIC_ENABLED}",
  "explore_intrinsic_coef": "${EXPLORE_INTRINSIC_COEF}",
  "explore_intrinsic_schedule": "${EXPLORE_INTRINSIC_SCHEDULE}",
  "explore_intrinsic_decay_steps": "${EXPLORE_INTRINSIC_DECAY_STEPS}",
  "explore_intrinsic_granularity": "${EXPLORE_INTRINSIC_GRANULARITY}",
  "explore_intrinsic_scope": "${EXPLORE_INTRINSIC_SCOPE}",
  "explore_safety_filter": "${EXPLORE_SAFETY_FILTER}",
  "explore_safety_filter_enabled": "${EXPLORE_SAFETY_FILTER_ENABLED}",
  "explore_safety_filter_coef": "${EXPLORE_SAFETY_FILTER_COEF}",
  "explore_lprnd": "${EXPLORE_LPRND}",
  "explore_lprnd_enabled": "${EXPLORE_LPRND_ENABLED}",
  "explore_lprnd_coef": "${EXPLORE_LPRND_COEF}",
  "explore_lprnd_schedule": "${EXPLORE_LPRND_SCHEDULE}",
  "explore_lprnd_decay_steps": "${EXPLORE_LPRND_DECAY_STEPS}",
  "explore_lprnd_clip": "${EXPLORE_LPRND_CLIP}",
  "explore_lprnd_warmup": "${EXPLORE_LPRND_WARMUP}",
  "explore_advantage_bonus": "${EXPLORE_ADVANTAGE_BONUS}",
  "explore_advantage_bonus_enabled": "${EXPLORE_ADVANTAGE_BONUS_ENABLED}",
  "explore_advantage_bonus_components": "${EXPLORE_ADVANTAGE_BONUS_COMPONENTS}",
  "explore_advantage_bonus_coef": "${EXPLORE_ADVANTAGE_BONUS_COEF}",
  "explore_advantage_bonus_clip": "${EXPLORE_ADVANTAGE_BONUS_CLIP}",
  "explore_cde_actor": "${EXPLORE_CDE_ACTOR}",
  "explore_cde_actor_enabled": "${EXPLORE_CDE_ACTOR_ENABLED}",
  "explore_cde_actor_omega": "${EXPLORE_CDE_ACTOR_OMEGA}",
  "explore_cde_actor_kappa": "${EXPLORE_CDE_ACTOR_KAPPA}",
  "explore_cde_actor_alpha": "${EXPLORE_CDE_ACTOR_ALPHA}",
  "explore_cde_actor_reward_gate": "${EXPLORE_CDE_ACTOR_REWARD_GATE}",
  "explore_cde_actor_decay_steps": "${EXPLORE_CDE_ACTOR_DECAY_STEPS}",
  "explore_retry_attempts": "${EXPLORE_RETRY_ATTEMPTS}",
  "explore_retry_traj_gamma": "${EXPLORE_RETRY_TRAJ_GAMMA}",
  "clawsentry_url": "${CS_HTTP_URL}",
  "clawsentry_llm_provider": "${CS_LLM_PROVIDER}",
  "clawsentry_l3_enabled": "${CS_L3_ENABLED}",
  "clawsentry_evolving_enabled": "${CS_EVOLVING_ENABLED}",
  "terminal_structured_metrics": "${TERMINAL_STRUCTURED_METRICS}",
  "terminal_metrics_jsonl": "${TERMINAL_METRICS_JSONL}",
  "train_python": "${TRAIN_PYTHON}",
  "a3s_code_repo_root": "${A3S_CODE_REPO_ROOT}",
  "a3s_code_config_path": "${A3S_CODE_CONFIG_PATH}",
  "a3s_code_cache_dir": "${A3S_CODE_CACHE_DIR}",
  "a3s_code_workspace_root": "${A3S_CODE_WORKSPACE_ROOT}",
  "a3s_code_max_tool_rounds": "${A3S_CODE_MAX_TOOL_ROUNDS}",
  "a3s_code_tool_timeout_ms": "${A3S_CODE_TOOL_TIMEOUT_MS}",
  "a3s_code_turn_timeout_sec": "${A3S_CODE_TURN_TIMEOUT_SEC}",
  "a3s_code_output_tokens": "${A3S_CODE_OUTPUT_TOKENS}",
  "a3s_code_planning_mode": "${A3S_CODE_PLANNING_MODE}",
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

if [[ -n "${RAY_WAIT_TOTAL_GPUS:-}" ]]; then
  log "Waiting for Ray cluster GPUs >= ${RAY_WAIT_TOTAL_GPUS}"
  "${TRAIN_PYTHON}" - "${RAY_WAIT_TOTAL_GPUS}" <<'PY'
import os
import sys
import time

import ray

target = float(sys.argv[1])
address = f"{os.environ.get('MASTER_ADDR', '127.0.0.1')}:6379"
ray.init(address=address, log_to_driver=False)
try:
    for attempt in range(1, 121):
        got = float(ray.cluster_resources().get("GPU", 0))
        print(f"[ray-wait] GPUs {got:g}/{target:g} (attempt {attempt})", flush=True)
        if got >= target:
            break
        time.sleep(5)
    else:
        raise SystemExit(f"Ray cluster did not reach {target:g} GPUs")
finally:
    ray.shutdown()
PY
fi

# ── Build runtime env ────────────────────────────────────────────────
# Match v1 (run_swe_rl_8b_remote_1node.sh): only code dirs in PYTHONPATH.
# Do NOT inject conda site-packages — Ray workers use the default python3
# which already has Megatron/TE/sglang installed.
RUNTIME_PYTHONPATH="${MEGATRON_DIR}:${REPO_ROOT}:${SLIME_DIR}:${SCRIPT_DIR}"
if [[ "${HARNESS_OPTION}" == "a3s-code" ]]; then
  RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH}:${REPO_ROOT}/a3s-code-adapter"
fi

A3S_RUNTIME_ENV_JSON=""
if [[ "${HARNESS_OPTION}" == "a3s-code" ]]; then
  A3S_RUNTIME_ENV_JSON=",
    \"A3S_CODE_REPO_ROOT\": \"${A3S_CODE_REPO_ROOT}\",
    \"A3S_CODE_CONFIG_PATH\": \"${A3S_CODE_CONFIG_PATH}\",
    \"A3S_CODE_CACHE_DIR\": \"${A3S_CODE_CACHE_DIR}\",
    \"A3S_CODE_WORKSPACE_ROOT\": \"${A3S_CODE_WORKSPACE_ROOT}\",
    \"A3S_CODE_EXTRA_SITE_PACKAGES\": \"${A3S_CODE_EXTRA_SITE_PACKAGES}\",
    \"A3S_CODE_TURN_TIMEOUT_SEC\": \"${A3S_CODE_TURN_TIMEOUT_SEC}\",
    \"A3S_CODE_TOOL_TIMEOUT_MS\": \"${A3S_CODE_TOOL_TIMEOUT_MS}\",
    \"A3S_CODE_MAX_TOOL_ROUNDS\": \"${A3S_CODE_MAX_TOOL_ROUNDS}\",
    \"A3S_CODE_MAX_PARSE_RETRIES\": \"${A3S_CODE_MAX_PARSE_RETRIES}\",
    \"A3S_CODE_OUTPUT_TOKENS\": \"${A3S_CODE_OUTPUT_TOKENS}\",
    \"A3S_CODE_PLANNING_MODE\": \"${A3S_CODE_PLANNING_MODE}\",
    \"A3S_CODE_THINKING_BUDGET\": \"${A3S_CODE_THINKING_BUDGET}\",
    \"ENV_EVALUATE_MAX_RETRIES\": \"${ENV_EVALUATE_MAX_RETRIES}\""
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"PYTHONFAULTHANDLER\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_CUMEM_ENABLE\": \"${NCCL_CUMEM_ENABLE}\",
    \"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\": \"${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}\",
    \"TORCH_NCCL_ENABLE_MONITORING\": \"${TORCH_NCCL_ENABLE_MONITORING}\",
    \"TORCH_NCCL_AVOID_RECORD_STREAMS\": \"${TORCH_NCCL_AVOID_RECORD_STREAMS}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"USE_REMOTE_ENV\": \"${USE_REMOTE_ENV}\",
    \"ENV_SERVER_URL\": \"${ENV_SERVER_URL}\",
    \"AGENT_SAFETYBENCH_REMOTE_ENV\": \"${AGENT_SAFETYBENCH_REMOTE_ENV}\",
    \"AGENTHARM_REMOTE_ENV\": \"${AGENTHARM_REMOTE_ENV}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${NO_PROXY}\",
    \"CS_HTTP_URL\": \"${CS_HTTP_URL}\",
    \"CS_AUTH_TOKEN\": \"${CS_AUTH_TOKEN}\",
    \"SETA_SAFETY\": \"${SETA_SAFETY}\",
    \"SAFETY_BENCH_REWARD\": \"${SAFETY_BENCH_REWARD}\",
    \"AGENT_SAFETYBENCH_ROOT\": \"${AGENT_SAFETYBENCH_ROOT}\",
    \"AGENTHARM_REWARD\": \"${AGENTHARM_REWARD}\",
    \"AGENTHARM_ROOT\": \"${AGENTHARM_ROOT}\",
    \"SAFETY_REWARD_COEF\": \"${SAFETY_REWARD_COEF}\",
    \"SAFETY_REWARD_SUMMARY_WEIGHT\": \"${SAFETY_REWARD_SUMMARY_WEIGHT}\",
    \"SAFETY_REWARD_TIMEOUT\": \"${SAFETY_REWARD_TIMEOUT}\",
    \"SAFETY_REWARD_ZERO_THRESHOLD\": \"${SAFETY_REWARD_ZERO_THRESHOLD}\",
    \"TERMINAL_SAVE_TRAJ_DIR\": \"${TERMINAL_SAVE_TRAJ_DIR}\",
    \"TRAJECTORY_SAVE_INTERVAL\": \"${TRAJECTORY_SAVE_INTERVAL}\",
    \"RUN_DIR\": \"${RUN_DIR}\",
    \"RUN_ID\": \"${RUN_ID}\",
    \"RUN_NAME\": \"${RUN_NAME}\",
    \"RUN_LOG_DIR\": \"${RUN_LOG_DIR}\",
    \"TERMINAL_STRUCTURED_METRICS\": \"${TERMINAL_STRUCTURED_METRICS}\",
    \"TERMINAL_METRICS_JSONL\": \"${TERMINAL_METRICS_JSONL}\",
    \"HARNESS_OPTION\": \"${HARNESS_OPTION}\",
    \"DATASET\": \"${DATASET}\",
    \"ALGO\": \"${ALGO}\",
    \"DAPO_OVERLONG_BUFFER_ENABLE\": \"${DAPO_OVERLONG_BUFFER_ENABLE}\",
    \"DAPO_OVERLONG_BUFFER_LEN\": \"${DAPO_OVERLONG_BUFFER_LEN}\",
    \"DAPO_OVERLONG_PENALTY_FACTOR\": \"${DAPO_OVERLONG_PENALTY_FACTOR}\",
    \"DAPO_MAX_RESPONSE_LEN\": \"${ROLLOUT_MAX_RESPONSE_LEN}\",
    \"EXPLORATION_PROFILE\": \"${EXPLORATION_PROFILE}\",
    \"EXPLORE_ENTROPY_COEF\": \"${EXPLORE_ENTROPY_COEF}\",
    \"EXPLORE_THINK_MODE\": \"${EXPLORE_THINK_MODE}\",
    \"EXPLORE_TEMP_HIGH\": \"${EXPLORE_TEMP_HIGH}\",
    \"EXPLORE_INTRINSIC\": \"${EXPLORE_INTRINSIC}\",
    \"EXPLORE_INTRINSIC_ENABLED\": \"${EXPLORE_INTRINSIC_ENABLED}\",
    \"EXPLORE_INTRINSIC_COEF\": \"${EXPLORE_INTRINSIC_COEF}\",
    \"EXPLORE_INTRINSIC_SCHEDULE\": \"${EXPLORE_INTRINSIC_SCHEDULE}\",
    \"EXPLORE_INTRINSIC_DECAY_STEPS\": \"${EXPLORE_INTRINSIC_DECAY_STEPS}\",
    \"EXPLORE_INTRINSIC_GRANULARITY\": \"${EXPLORE_INTRINSIC_GRANULARITY}\",
    \"EXPLORE_INTRINSIC_SCOPE\": \"${EXPLORE_INTRINSIC_SCOPE}\",
    \"EXPLORE_SAFETY_FILTER\": \"${EXPLORE_SAFETY_FILTER}\",
    \"EXPLORE_SAFETY_FILTER_ENABLED\": \"${EXPLORE_SAFETY_FILTER_ENABLED}\",
    \"EXPLORE_SAFETY_FILTER_COEF\": \"${EXPLORE_SAFETY_FILTER_COEF}\",
    \"EXPLORE_LPRND\": \"${EXPLORE_LPRND}\",
    \"EXPLORE_LPRND_ENABLED\": \"${EXPLORE_LPRND_ENABLED}\",
    \"EXPLORE_LPRND_COEF\": \"${EXPLORE_LPRND_COEF}\",
    \"EXPLORE_LPRND_SCHEDULE\": \"${EXPLORE_LPRND_SCHEDULE}\",
    \"EXPLORE_LPRND_DECAY_STEPS\": \"${EXPLORE_LPRND_DECAY_STEPS}\",
    \"EXPLORE_LPRND_CLIP\": \"${EXPLORE_LPRND_CLIP}\",
    \"EXPLORE_LPRND_WARMUP\": \"${EXPLORE_LPRND_WARMUP}\",
    \"EXPLORE_ADVANTAGE_BONUS\": \"${EXPLORE_ADVANTAGE_BONUS}\",
    \"EXPLORE_ADVANTAGE_BONUS_ENABLED\": \"${EXPLORE_ADVANTAGE_BONUS_ENABLED}\",
    \"EXPLORE_ADVANTAGE_BONUS_COMPONENTS\": \"${EXPLORE_ADVANTAGE_BONUS_COMPONENTS}\",
    \"EXPLORE_ADVANTAGE_BONUS_COEF\": \"${EXPLORE_ADVANTAGE_BONUS_COEF}\",
    \"EXPLORE_ADVANTAGE_BONUS_CLIP\": \"${EXPLORE_ADVANTAGE_BONUS_CLIP}\",
    \"EXPLORE_CDE_ACTOR\": \"${EXPLORE_CDE_ACTOR}\",
    \"EXPLORE_CDE_ACTOR_ENABLED\": \"${EXPLORE_CDE_ACTOR_ENABLED}\",
    \"EXPLORE_CDE_ACTOR_OMEGA\": \"${EXPLORE_CDE_ACTOR_OMEGA}\",
    \"EXPLORE_CDE_ACTOR_KAPPA\": \"${EXPLORE_CDE_ACTOR_KAPPA}\",
    \"EXPLORE_CDE_ACTOR_ALPHA\": \"${EXPLORE_CDE_ACTOR_ALPHA}\",
    \"EXPLORE_CDE_ACTOR_REWARD_GATE\": \"${EXPLORE_CDE_ACTOR_REWARD_GATE}\",
    \"EXPLORE_CDE_ACTOR_DECAY_STEPS\": \"${EXPLORE_CDE_ACTOR_DECAY_STEPS}\",
    \"EXPLORE_RETRY_ATTEMPTS\": \"${EXPLORE_RETRY_ATTEMPTS}\",
    \"EXPLORE_RETRY_TRAJ_GAMMA\": \"${EXPLORE_RETRY_TRAJ_GAMMA}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-offline}\"
    ${A3S_RUNTIME_ENV_JSON}
  }
}"

RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-terminal_rl_glm51_lora_tp_${ACTOR_NUM_NODES}node_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)}"

log "Submitting Ray job ${RAY_JOB_SUBMISSION_ID}"
ray job submit --address="http://${MASTER_ADDR}:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py" \
  "${TRAIN_ARGS[@]}"

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
