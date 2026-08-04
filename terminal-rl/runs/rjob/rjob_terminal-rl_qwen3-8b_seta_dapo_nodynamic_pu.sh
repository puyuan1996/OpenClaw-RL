#!/bin/bash

################################################################################
# Submit OpenClaw terminal-rl Qwen3-8B SETA+DAPO nodynamic baseline as a
# detached 1-node 8-GPU rjob task.
#
# Paired run script:
#   terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh
#
# Baseline semantics for exploration comparisons:
#   - DATASET=seta, ALGO=dapo
#   - DAPO_DYNAMIC_SAMPLING=0
#   - WORKER_URLS_FILE=terminal-rl/worker_urls_2.txt
#   - no Agent57/SimHash/lifelong/UCB exploration bonus
#   - no post-normalization intrinsic advantage bonus or truncation penalty
#
# This script intentionally keeps the rjob resource shape aligned with the
# recent SETA exploration wrappers, while making the algorithmic baseline
# knobs explicit and overrideable via environment variables.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_HOME="${OPENCLAW_HOME:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-${OPENCLAW_HOME}/terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TURN="${MAX_TURN:-10}"
RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_baseline_think_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${OPENCLAW_HOME}/runs}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"

# Baseline training knobs. Keep these explicit so this rjob remains a clean
# reference when launched from a shell that recently ran exploration variants.
DATASET="${DATASET:-seta}"
ALGO="${ALGO:-dapo}"
HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${OPENCLAW_HOME}/terminal-rl/configs/rollout_qwen3_think.yaml}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
EXTRA_DAPO_ARGS="${EXTRA_DAPO_ARGS:-}"
WORKER_URLS_FILE="${WORKER_URLS_FILE:-${OPENCLAW_HOME}/terminal-rl/worker_urls_2.txt}"
WORKER_URLS_RELOAD_INTERVAL="${WORKER_URLS_RELOAD_INTERVAL:-120}"

EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-off}"
EXPLORE_INTRINSIC="${EXPLORE_INTRINSIC:-0}"
EXPLORE_INTRINSIC_ENABLED="${EXPLORE_INTRINSIC_ENABLED:-0}"
EXPLORE_LPRND="${EXPLORE_LPRND:-0}"
EXPLORE_LPRND_ENABLED="${EXPLORE_LPRND_ENABLED:-0}"
EXPLORE_AGENT57_LITE="${EXPLORE_AGENT57_LITE:-0}"
EXPLORE_AGENT57_LITE_ENABLED="${EXPLORE_AGENT57_LITE_ENABLED:-0}"
EXPLORE_AGENT57_LIFELONG="${EXPLORE_AGENT57_LIFELONG:-0}"
EXPLORE_AGENT57_LIFELONG_ENABLED="${EXPLORE_AGENT57_LIFELONG_ENABLED:-0}"
EXPLORE_ADVANTAGE_BONUS="${EXPLORE_ADVANTAGE_BONUS:-0}"
EXPLORE_ADVANTAGE_BONUS_ENABLED="${EXPLORE_ADVANTAGE_BONUS_ENABLED:-0}"
EXPLORE_TRUNCATION_PENALTY="${EXPLORE_TRUNCATION_PENALTY:-0}"
EXPLORE_ADVANTAGE_TRUNCATION_PENALTY="${EXPLORE_ADVANTAGE_TRUNCATION_PENALTY:-0}"

WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_DIR="${WANDB_DIR:-${RUNS_ROOT}/${RUN_ID}/metrics/wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-terminal_rl}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-8b_seta_dapo_nodynamic_baseline}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [[ -n "${WANDB_KEY}" && -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="${WANDB_KEY}"
fi

RJOB_LOG_ROOT="${RJOB_LOG_ROOT:-${OPENCLAW_HOME}/runs/rjob_logs/terminal_rl_seta_dapo_baseline}"
RJOB_NAME="${RJOB_NAME:-seta-dapo-base}"
RJOB_MEMORY="${RJOB_MEMORY:-1500000}"
RJOB_CPU="${RJOB_CPU:-150}"
RJOB_GPU="${RJOB_GPU:-8}"
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-narmodel_gpu}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"
RJOB_IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
RJOB_REPLICA="${RJOB_REPLICA:-1}"
RJOB_CUSTOM_RESOURCES="${RJOB_CUSTOM_RESOURCES:-brainpp.cn/fuse=1}"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "Run script not found: ${RUN_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${WORKER_URLS_FILE}" ]]; then
    echo "Worker URL file not found: ${WORKER_URLS_FILE}" >&2
    exit 1
fi

case "${RJOB_PRIVATE_MACHINE,,}" in
    yes|true|1)
        RJOB_PRIVATE_MACHINE="group"
        ;;
    false|0)
        RJOB_PRIVATE_MACHINE="no"
        ;;
    group|no|project|tenant)
        RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE,,}"
        ;;
    *)
        echo "Invalid RJOB_PRIVATE_MACHINE=${RJOB_PRIVATE_MACHINE}; expected group, no, project, tenant, or legacy yes/true/1/false/0." >&2
        exit 1
        ;;
esac

if [[ ! "${RJOB_NAME}" =~ ^[a-zA-Z0-9]([-a-zA-Z0-9]*[a-zA-Z0-9])?$ ]]; then
    echo "Invalid RJOB_NAME=${RJOB_NAME}; use only letters, numbers, and '-' and start/end with alnum." >&2
    exit 1
fi
if (( ${#RJOB_NAME} > 31 )); then
    echo "Invalid RJOB_NAME=${RJOB_NAME}; keep it <=31 chars because rjob builds task labels as <job>-<task> with a 63-char limit." >&2
    exit 1
fi

WANDB_KEY_FORWARDED=0
wandb_key_env_args=()
if [[ -n "${WANDB_KEY}" && ( "${WANDB_MODE}" != "offline" || "${WANDB_FORWARD_KEY:-0}" == "1" ) ]]; then
    wandb_key_env_args=(
        -e WANDB_KEY="${WANDB_KEY}"
        -e WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY}}"
    )
    WANDB_KEY_FORWARDED=1
fi

baseline_env_args=(
    -e DATASET="${DATASET}"
    -e ALGO="${ALGO}"
    -e HARNESS_OPTION="${HARNESS_OPTION}"
    -e CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH}"
    -e ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}"
    -e N_SAMPLES="${N_SAMPLES}"
    -e MAX_TURN="${MAX_TURN}"
    -e DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING}"
    -e EXTRA_DAPO_ARGS="${EXTRA_DAPO_ARGS}"
    -e WORKER_URLS_FILE="${WORKER_URLS_FILE}"
    -e WORKER_URLS_RELOAD_INTERVAL="${WORKER_URLS_RELOAD_INTERVAL}"
    -e EXPLORATION_PROFILE="${EXPLORATION_PROFILE}"
    -e EXPLORE_INTRINSIC="${EXPLORE_INTRINSIC}"
    -e EXPLORE_INTRINSIC_ENABLED="${EXPLORE_INTRINSIC_ENABLED}"
    -e EXPLORE_LPRND="${EXPLORE_LPRND}"
    -e EXPLORE_LPRND_ENABLED="${EXPLORE_LPRND_ENABLED}"
    -e EXPLORE_AGENT57_LITE="${EXPLORE_AGENT57_LITE}"
    -e EXPLORE_AGENT57_LITE_ENABLED="${EXPLORE_AGENT57_LITE_ENABLED}"
    -e EXPLORE_AGENT57_LIFELONG="${EXPLORE_AGENT57_LIFELONG}"
    -e EXPLORE_AGENT57_LIFELONG_ENABLED="${EXPLORE_AGENT57_LIFELONG_ENABLED}"
    -e EXPLORE_ADVANTAGE_BONUS="${EXPLORE_ADVANTAGE_BONUS}"
    -e EXPLORE_ADVANTAGE_BONUS_ENABLED="${EXPLORE_ADVANTAGE_BONUS_ENABLED}"
    -e EXPLORE_TRUNCATION_PENALTY="${EXPLORE_TRUNCATION_PENALTY}"
    -e EXPLORE_ADVANTAGE_TRUNCATION_PENALTY="${EXPLORE_ADVANTAGE_TRUNCATION_PENALTY}"
)

submit_cmd=(
    rjob submit
    --name="${RJOB_NAME}"
    --gpu="${RJOB_GPU}"
    --memory="${RJOB_MEMORY}"
    --cpu="${RJOB_CPU}"
    --charged-group="${RJOB_CHARGED_GROUP}"
    --private-machine="${RJOB_PRIVATE_MACHINE}"
    -P "${RJOB_REPLICA}"
    --image="${RJOB_IMAGE}"
    --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan
    --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public
    --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel
    -e INSIDE_RJOB=1
    -e SUBMIT_RJOB=0
    -e OPENCLAW_HOME="${OPENCLAW_HOME}"
    -e RUN_SCRIPT="${RUN_SCRIPT}"
    -e RUN_TIMESTAMP="${RUN_TIMESTAMP}"
    -e RUN_ID="${RUN_ID}"
    -e RUN_NAME="${RUN_NAME}"
    -e NUM_GPUS="${NUM_GPUS}"
    -e RUNS_ROOT="${RUNS_ROOT}"
    -e MAX_CKPT_KEEP="${MAX_CKPT_KEEP}"
    -e SAVE_INTERVAL="${SAVE_INTERVAL}"
    -e RJOB_LOG_ROOT="${RJOB_LOG_ROOT}"
    -e RJOB_MEMORY="${RJOB_MEMORY}"
    -e RJOB_CPU="${RJOB_CPU}"
    -e RJOB_GPU="${RJOB_GPU}"
    -e RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP}"
    -e RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE}"
    -e RJOB_CUSTOM_RESOURCES="${RJOB_CUSTOM_RESOURCES}"
    -e RJOB_IMAGE="${RJOB_IMAGE}"
    -e WANDB_ENABLE="${WANDB_ENABLE}"
    -e WANDB_MODE="${WANDB_MODE}"
    -e WANDB_DIR="${WANDB_DIR}"
    -e WANDB_PROJECT="${WANDB_PROJECT}"
    -e WANDB_GROUP="${WANDB_GROUP}"
    "${baseline_env_args[@]}"
    "${wandb_key_env_args[@]}"
    --custom-resources "${RJOB_CUSTOM_RESOURCES}"
    -- bash -exc "cd \"${OPENCLAW_HOME}\" && exec bash \"${RUN_SCRIPT}\""
)

echo "Submitting OpenClaw terminal-rl SETA+DAPO nodynamic baseline rjob:"
echo "  name:       ${RJOB_NAME}"
echo "  run_id:     ${RUN_ID}"
echo "  group:      ${RJOB_CHARGED_GROUP}"
echo "  gpu:        ${RJOB_GPU}"
echo "  ckpt_keep:  ${MAX_CKPT_KEEP}"
echo "  save_intvl: ${SAVE_INTERVAL}"
echo "  script:     ${RUN_SCRIPT}"
echo "  logs:       ${RJOB_LOG_ROOT}/${RUN_ID}"
echo "  wandb:      ${WANDB_MODE} -> ${WANDB_DIR}"
echo "  wandb_key:  forwarded=${WANDB_KEY_FORWARDED}"
echo "  workers:    ${WORKER_URLS_FILE} reload=${WORKER_URLS_RELOAD_INTERVAL}s"
echo "  baseline:   dataset=${DATASET} algo=${ALGO} dyn=${DAPO_DYNAMIC_SAMPLING} explore=${EXPLORATION_PROFILE} agent57=${EXPLORE_AGENT57_LITE_ENABLED} adv_bonus=${EXPLORE_ADVANTAGE_BONUS_ENABLED}"

if [[ "${RJOB_DRY_RUN:-0}" == "1" ]]; then
    printf 'Dry-run command:'
    printf ' %q' "${submit_cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${submit_cmd[@]}"
