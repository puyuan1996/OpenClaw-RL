#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

DEFAULT_A3S_CODE_PYTHON_BIN="python3"
if [[ -n "${CONDA_ENV:-}" && -x "${CONDA_ENV}/bin/python3" ]]; then
  DEFAULT_A3S_CODE_PYTHON_BIN="${CONDA_ENV}/bin/python3"
  export CONDA_PREFIX="${CONDA_ENV}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python3" ]]; then
  DEFAULT_A3S_CODE_PYTHON_BIN="${CONDA_PREFIX}/bin/python3"
fi

export RL_BASE_URL="${RL_BASE_URL:-http://127.0.0.1:30000}"
export KIMI_BASE_URL="${KIMI_BASE_URL:-http://s-20260204175507-cqflp.ailab-pj.pjh-service.org.cn}"
export SIMULATED_USER_BASE_URL="${SIMULATED_USER_BASE_URL:-${KIMI_BASE_URL}}"
export SIMULATED_USER_MODEL_URL="${SIMULATED_USER_MODEL_URL:-${SIMULATED_USER_BASE_URL}/v1/chat/completions}"
export SIMULATED_USER_MODEL_NAME="${SIMULATED_USER_MODEL_NAME:-kimi-k2.5}"
export SIMULATED_USER_API_KEY="${SIMULATED_USER_API_KEY:-}"
export SIMULATED_USER_MODEL_URLS="${SIMULATED_USER_MODEL_URLS:-${SIMULATED_USER_MODEL_URL}}"
export SIMULATED_USER_MODEL_NAMES="${SIMULATED_USER_MODEL_NAMES:-${SIMULATED_USER_MODEL_NAME}}"
export SIMULATED_USER_API_KEYS="${SIMULATED_USER_API_KEYS:-${SIMULATED_USER_API_KEY}}"
export A3S_MODEL_NAME="${A3S_MODEL_NAME:-${SERVED_MODEL_NAME:-qwen3.5-4b}}"
export A3S_API_KEY="${A3S_API_KEY:-${SGLANG_API_KEY:-apiKey}}"
export A3S_CODE_PYTHON_BIN="${A3S_CODE_PYTHON_BIN:-${DEFAULT_A3S_CODE_PYTHON_BIN}}"
if [[ -n "${A3S_CODE_REPO_ROOT:-}" ]]; then
  export A3S_CODE_REPO_ROOT
else
  unset A3S_CODE_REPO_ROOT
fi
export A3S_CODE_EXTRA_SITE_PACKAGES="${A3S_CODE_EXTRA_SITE_PACKAGES:-}"
export A3S_CODE_REQUIRED_VERSION="${A3S_CODE_REQUIRED_VERSION:-latest}"
export A3S_CODE_TRAFFIC_CONCURRENCY="${A3S_CODE_TRAFFIC_CONCURRENCY:-1}"
_a3s_code_traffic_session_limit_was_set=0
if [[ -n "${A3S_CODE_TRAFFIC_SESSION_LIMIT:-}" ]]; then
  _a3s_code_traffic_session_limit_was_set=1
fi
export A3S_CODE_MAX_MAIN_TURNS="${A3S_CODE_MAX_MAIN_TURNS:-4}"
export A3S_CODE_MAX_TOOL_ROUNDS="${A3S_CODE_MAX_TOOL_ROUNDS:-16}"
export A3S_CODE_TOOL_TIMEOUT_MS="${A3S_CODE_TOOL_TIMEOUT_MS:-300000}"
export A3S_CODE_MAX_PARSE_RETRIES="${A3S_CODE_MAX_PARSE_RETRIES:-4}"
export A3S_CODE_CIRCUIT_BREAKER_THRESHOLD="${A3S_CODE_CIRCUIT_BREAKER_THRESHOLD:-5}"
export A3S_CODE_BUILTIN_SKILLS="${A3S_CODE_BUILTIN_SKILLS:-1}"
export A3S_CODE_PLANNING="${A3S_CODE_PLANNING:-1}"
export A3S_CODE_PLANNING_MODE="${A3S_CODE_PLANNING_MODE:-}"
export A3S_CODE_SIMULATED_USER_TIMEOUT_SEC="${A3S_CODE_SIMULATED_USER_TIMEOUT_SEC:-45}"
export A3S_CODE_SIMULATED_USER_BACKEND_COOLDOWN_SEC="${A3S_CODE_SIMULATED_USER_BACKEND_COOLDOWN_SEC:-60}"
export A3S_CODE_SIMULATED_USER_MAX_ATTEMPTS="${A3S_CODE_SIMULATED_USER_MAX_ATTEMPTS:-0}"
export A3S_CODE_REQUEST_TIMEOUT_SEC="${A3S_CODE_REQUEST_TIMEOUT_SEC:-900}"
export A3S_CODE_TURN_TIMEOUT_SEC="${A3S_CODE_TURN_TIMEOUT_SEC:-600}"
export A3S_CODE_SESSION_DELAY_SEC="${A3S_CODE_SESSION_DELAY_SEC:-1}"
export A3S_CODE_KEEP_WORKSPACES="${A3S_CODE_KEEP_WORKSPACES:-0}"
export A3S_CODE_KEEP_WORKSPACES_ON_ERROR="${A3S_CODE_KEEP_WORKSPACES_ON_ERROR:-0}"
export A3S_CODE_KEEP_CONFIGS="${A3S_CODE_KEEP_CONFIGS:-0}"
export A3S_CODE_AGENT_CONFIG_MODE="${A3S_CODE_AGENT_CONFIG_MODE:-shared}"
export A3S_CODE_SESSION_ID_HEADER_NAME="${A3S_CODE_SESSION_ID_HEADER_NAME:-X-Session-Id}"
export A3S_CODE_WORKSPACE_COPY_MODE="${A3S_CODE_WORKSPACE_COPY_MODE:-reflink_auto}"
export A3S_CODE_INCLUDE_SEED_TAGS="${A3S_CODE_INCLUDE_SEED_TAGS:-}"
export A3S_CODE_INCLUDE_SEED_IDS="${A3S_CODE_INCLUDE_SEED_IDS:-}"
export A3S_CODE_SEED_DATA_FILE="${A3S_CODE_SEED_DATA_FILE:-${SCRIPT_DIR}/seed_data/code_task_seeds.json}"
export A3S_CODE_TASK_TEMPLATE_ROOT="${A3S_CODE_TASK_TEMPLATE_ROOT:-${SCRIPT_DIR}/task_templates}"
export A3S_CODE_ENABLE_TASK_VERIFIER_REWARD="${A3S_CODE_ENABLE_TASK_VERIFIER_REWARD:-1}"
export A3S_CODE_VERIFIER_FALLBACK_TO_TEST_COMMAND="${A3S_CODE_VERIFIER_FALLBACK_TO_TEST_COMMAND:-1}"
export A3S_CODE_TASK_VERIFIER_TIMEOUT_SEC="${A3S_CODE_TASK_VERIFIER_TIMEOUT_SEC:-180}"
export CODE_RL_MATCHED_CONTEXT_TOKENS="${CODE_RL_MATCHED_CONTEXT_TOKENS:-16384}"
export A3S_CODE_CONTEXT_HEADROOM_TOKENS="${A3S_CODE_CONTEXT_HEADROOM_TOKENS:-2048}"
default_a3s_context_tokens=$((CODE_RL_MATCHED_CONTEXT_TOKENS - A3S_CODE_CONTEXT_HEADROOM_TOKENS))
if (( default_a3s_context_tokens < 1024 )); then
  default_a3s_context_tokens=1024
fi
export A3S_CODE_MODEL_CONTEXT_TOKENS="${A3S_CODE_MODEL_CONTEXT_TOKENS:-${default_a3s_context_tokens}}"
export A3S_CODE_MODEL_OUTPUT_TOKENS="${A3S_CODE_MODEL_OUTPUT_TOKENS:-4096}"
export A3S_CODE_THINKING_BUDGET="${A3S_CODE_THINKING_BUDGET:-24000}"
export A3S_CODE_AUTO_COMPACT="${A3S_CODE_AUTO_COMPACT:-1}"
export A3S_CODE_AUTO_COMPACT_THRESHOLD="${A3S_CODE_AUTO_COMPACT_THRESHOLD:-0.7}"
export A3S_CODE_CONTINUATION_ENABLED="${A3S_CODE_CONTINUATION_ENABLED:-0}"
export A3S_CODE_MAX_CONTINUATION_TURNS="${A3S_CODE_MAX_CONTINUATION_TURNS:-5}"
export A3S_CODE_GIT_USER_NAME="${A3S_CODE_GIT_USER_NAME:-A3S Code RL}"
export A3S_CODE_GIT_USER_EMAIL="${A3S_CODE_GIT_USER_EMAIL:-a3s-code-adapter@example.com}"

_a3s_code_require_uint() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got '${value}'" >&2
    exit 1
  fi
}

_a3s_code_rollout_batch_size="${ROLLOUT_BATCH_SIZE:-0}"
_a3s_code_samples_per_prompt="${N_SAMPLES_PER_PROMPT:-1}"
_a3s_code_num_rollout="${NUM_ROLLOUT:-0}"
_a3s_code_start_rollout_id="${START_ROLLOUT_ID:-0}"
for _a3s_code_pair in \
  "ROLLOUT_BATCH_SIZE:${_a3s_code_rollout_batch_size}" \
  "N_SAMPLES_PER_PROMPT:${_a3s_code_samples_per_prompt}" \
  "NUM_ROLLOUT:${_a3s_code_num_rollout}" \
  "START_ROLLOUT_ID:${_a3s_code_start_rollout_id}"
do
  _a3s_code_require_uint "${_a3s_code_pair%%:*}" "${_a3s_code_pair#*:}"
done

_a3s_code_min_sessions_per_rollout=0
_a3s_code_required_sessions=0
if (( _a3s_code_rollout_batch_size > 0 && _a3s_code_samples_per_prompt > 0 )); then
  _a3s_code_min_sessions_per_rollout=$(( _a3s_code_rollout_batch_size * _a3s_code_samples_per_prompt ))
  if (( _a3s_code_num_rollout > _a3s_code_start_rollout_id )); then
    _a3s_code_required_sessions=$(( (_a3s_code_num_rollout - _a3s_code_start_rollout_id) * _a3s_code_min_sessions_per_rollout ))
  else
    _a3s_code_required_sessions="${_a3s_code_min_sessions_per_rollout}"
  fi
fi

if (( _a3s_code_traffic_session_limit_was_set == 0 )); then
  export A3S_CODE_TRAFFIC_SESSION_LIMIT="${_a3s_code_required_sessions:-0}"
else
  _a3s_code_require_uint "A3S_CODE_TRAFFIC_SESSION_LIMIT" "${A3S_CODE_TRAFFIC_SESSION_LIMIT}"
  if (( A3S_CODE_TRAFFIC_SESSION_LIMIT > 0 && _a3s_code_required_sessions > 0 && A3S_CODE_TRAFFIC_SESSION_LIMIT < _a3s_code_required_sessions )); then
    echo "A3S_CODE_TRAFFIC_SESSION_LIMIT=${A3S_CODE_TRAFFIC_SESSION_LIMIT} is too small for this training budget." >&2
    echo "required_sessions=${_a3s_code_required_sessions} = (NUM_ROLLOUT=${_a3s_code_num_rollout} - START_ROLLOUT_ID=${_a3s_code_start_rollout_id}) * ROLLOUT_BATCH_SIZE=${_a3s_code_rollout_batch_size} * N_SAMPLES_PER_PROMPT=${_a3s_code_samples_per_prompt}" >&2
    exit 1
  fi
fi
export A3S_CODE_REQUIRED_TRAFFIC_SESSIONS="${_a3s_code_required_sessions}"
export A3S_CODE_TRAFFIC_BACKPRESSURE="${A3S_CODE_TRAFFIC_BACKPRESSURE:-1}"
export A3S_CODE_TRAFFIC_RAW_TRAIN_BATCH_SIZE="${A3S_CODE_TRAFFIC_RAW_TRAIN_BATCH_SIZE:-${_a3s_code_min_sessions_per_rollout}}"
export A3S_CODE_TRAFFIC_MAX_BATCH_OVERSHOOT="${A3S_CODE_TRAFFIC_MAX_BATCH_OVERSHOOT:-0}"
export A3S_CODE_TRAFFIC_BACKPRESSURE_INTERVAL_SEC="${A3S_CODE_TRAFFIC_BACKPRESSURE_INTERVAL_SEC:-5}"
export A3S_CODE_TRAFFIC_BACKPRESSURE_TIMEOUT_SEC="${A3S_CODE_TRAFFIC_BACKPRESSURE_TIMEOUT_SEC:-0}"

RUNTIME_ROOT="${A3S_CODE_RUN_ROOT:-${RUN_ROOT:-}}"
if [[ -z "${RUNTIME_ROOT}" ]]; then
  RUNTIME_ROOT="${ARTIFACT_ROOT:-${SCRIPT_DIR}}/traffic_runs/${A3S_CODE_RUN_ID:-manual_$(date +%Y%m%d_%H%M%S)_$$}"
fi
if [[ -n "${RUNTIME_ROOT}" ]]; then
  export A3S_CODE_WORKSPACE_ROOT="${A3S_CODE_WORKSPACE_ROOT:-${RUNTIME_ROOT}/a3s_workspaces}"
  export A3S_CODE_CONFIG_ROOT="${A3S_CODE_CONFIG_ROOT:-${RUNTIME_ROOT}/a3s_configs}"
  export A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT="${A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT:-${RUNTIME_ROOT}/a3s_workspace_template_cache}"
  export A3S_CODE_RESULTS_DIR="${A3S_CODE_RESULTS_DIR:-${RUNTIME_ROOT}/a3s_results}"
  export A3S_CODE_TRAFFIC_RECORD_FILE="${A3S_CODE_TRAFFIC_RECORD_FILE:-${A3S_CODE_RESULTS_DIR}/a3s_code_agent_traffic.jsonl}"
else
  export A3S_CODE_WORKSPACE_ROOT="${A3S_CODE_WORKSPACE_ROOT:-${SCRIPT_DIR}/generated_workspaces}"
  export A3S_CODE_CONFIG_ROOT="${A3S_CODE_CONFIG_ROOT:-${SCRIPT_DIR}/generated_configs}"
  export A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT="${A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT:-${SCRIPT_DIR}/workspace_template_cache}"
fi
export A3S_CODE_SIMULATED_USER_BACKENDS_FILE="${A3S_CODE_SIMULATED_USER_BACKENDS_FILE:-${SCRIPT_DIR}/simulated_user_backends.json}"
export A3S_CODE_SIMULATED_USER_PROBE_TIMEOUT_SEC="${A3S_CODE_SIMULATED_USER_PROBE_TIMEOUT_SEC:-20}"
export A3S_CODE_REFRESH_SIMULATED_USER_BACKENDS_ON_START="${A3S_CODE_REFRESH_SIMULATED_USER_BACKENDS_ON_START:-1}"
export A3S_CODE_GIT_CONFIG_GLOBAL="${A3S_CODE_GIT_CONFIG_GLOBAL:-${A3S_CODE_CONFIG_ROOT}/git-global-$$.cfg}"
export A3S_CODE_CLEANUP_ON_START="${A3S_CODE_CLEANUP_ON_START:-0}"
export A3S_CODE_GENERATED_TTL_HOURS="${A3S_CODE_GENERATED_TTL_HOURS:-12}"

# Keep local RL requests off any inherited outbound proxy.
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

mkdir -p "${A3S_CODE_CONFIG_ROOT}"
: > "${A3S_CODE_GIT_CONFIG_GLOBAL}"
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-${A3S_CODE_GIT_CONFIG_GLOBAL}}"

if [[ "${A3S_CODE_CLEANUP_ON_START}" == "1" && "${A3S_CODE_GENERATED_TTL_HOURS}" != "0" ]]; then
  ttl_minutes="$((A3S_CODE_GENERATED_TTL_HOURS * 60))"
  mkdir -p "${A3S_CODE_WORKSPACE_ROOT}" "${A3S_CODE_CONFIG_ROOT}"
  find "${A3S_CODE_WORKSPACE_ROOT}" -mindepth 1 -maxdepth 1 -mmin "+${ttl_minutes}" -exec rm -rf {} + 2>/dev/null || true
  find "${A3S_CODE_CONFIG_ROOT}" -mindepth 1 -maxdepth 1 -mmin "+${ttl_minutes}" -delete 2>/dev/null || true
fi

if [[ "${A3S_CODE_REFRESH_SIMULATED_USER_BACKENDS_ON_START}" == "1" ]]; then
  bash "${SCRIPT_DIR}/refresh_simulated_user_backends.sh"
fi

"${A3S_CODE_PYTHON_BIN}" -u "${SCRIPT_DIR}/a3s_code_agent_traffic_driver.py"
