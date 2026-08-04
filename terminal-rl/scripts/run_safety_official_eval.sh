#!/usr/bin/env bash
# Run official AgentHarm / AgentSafetyBench split evaluation for Terminal-RL runs.
#
# Responsibilities:
#   1. For runs that contain AgentSafetyBench, export ShieldAgent inputs.
#   2. Run the official Agent-SafetyBench ShieldAgent judge.
#   3. Summarize AgentHarm trajectory metrics and ASB ShieldAgent labels.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ASB_ROOT="${ASB_ROOT:-${AGENT_SAFETYBENCH_ROOT:-}}"
if [[ -z "${ASB_ROOT}" ]]; then
  for candidate in \
    "${REPO_ROOT}/../Agent-SafetyBench" \
    "${REPO_ROOT}/external/Agent-SafetyBench"; do
    if [[ -d "${candidate}/score" ]]; then
      ASB_ROOT="${candidate}"
      break
    fi
  done
fi
ASB_ROOT="${ASB_ROOT:-${REPO_ROOT}/../Agent-SafetyBench}"

RUN_ASB_SHIELD="${RUN_ASB_SHIELD:-1}"
ALLOW_PARTIAL_ASB_SHIELD="${ALLOW_PARTIAL_ASB_SHIELD:-0}"
ASB_SHIELD_DRY_RUN="${ASB_SHIELD_DRY_RUN:-0}"
SUMMARY_OUT="${SUMMARY_OUT:-${REPO_ROOT}/runs/official_safety_eval/summary_$(date +%Y%m%d_%H%M%S).md}"

BATCH_SIZE="${BATCH_SIZE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SHIELD_PRECHECK="${SHIELD_PRECHECK:-0}"
SHIELD_PRECHECK_TIMEOUT="${SHIELD_PRECHECK_TIMEOUT:-600}"
FORCE_ASB_EXPORT="${FORCE_ASB_EXPORT:-1}"
REUSE_ASB_SHIELD_RESULTS="${REUSE_ASB_SHIELD_RESULTS:-0}"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash terminal-rl/scripts/run_safety_official_eval.sh <run_dir>...
  bash terminal-rl/scripts/run_safety_official_eval.sh model_a=<run_dir> model_b=<run_dir>

Options:
  RUN_ASB_SHIELD=1             run ShieldAgent for runs containing AgentSafetyBench
  RUN_ASB_SHIELD=0             reuse existing ASB shield_results/<target_name>
  ASB_SHIELD_DRY_RUN=1         export ASB inputs and validate paths only
  BATCH_SIZE=4                 ShieldAgent batch size
  CUDA_VISIBLE_DEVICES=0       GPU used by ShieldAgent
  ALLOW_PARTIAL_ASB_SHIELD=0   fail if ASB ShieldAgent rows are incomplete
  SUMMARY_OUT=<path>           markdown output path

Example:
  BATCH_SIZE=4 CUDA_VISIBLE_DEVICES=0 \
  bash terminal-rl/scripts/run_safety_official_eval.sh \
    init=runs/eval/eval_qwen3-8b_init_mock_2026-06-09_022431
EOF
}

if [[ "$#" -lt 1 ]]; then
  usage
  exit 2
fi

sanitize_name() {
  local raw="$1"
  raw="${raw%/}"
  raw="${raw##*/}"
  printf '%s' "${raw}" | tr -c '[:alnum:]_.-' '_'
}

run_key() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).resolve(strict=False))
PY
}

run_maybe_timeout() {
  local seconds="$1"
  shift
  if [[ "${seconds}" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    timeout "${seconds}" "$@"
  else
    "$@"
  fi
}

has_asb_examples() {
  local run_dir="$1"
  local metrics="${run_dir}/logs/metrics.jsonl"

  if [[ -f "${metrics}" ]] && grep -q 'agent_safetybench' "${metrics}"; then
    return 0
  fi

  local probe_status
  set +e
  "${PYTHON_BIN}" - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
for meta_path in sorted((run_dir / "trajectories").glob("*/meta.json")):
    try:
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        continue
    if (meta.get("dataset_slug") or meta.get("data_source")) == "agent_safetybench":
        raise SystemExit(0)
raise SystemExit(1)
PY
  probe_status=$?
  set -e
  if [[ "${probe_status}" == "0" ]]; then
    return 0
  fi
  if [[ "${probe_status}" == "1" ]]; then
    return 1
  fi
  echo "[ERROR] failed to inspect AgentSafetyBench examples in ${run_dir}" >&2
  return "${probe_status}"
}

check_python_deps() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util

required = {
    "torch": "torch",
    "transformers": "transformers",
    "tqdm": "tqdm",
    "tabulate": "tabulate",
    "sklearn": "scikit-learn",
}
missing = [pkg for module, pkg in required.items() if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(
        f"[ERROR] PYTHON_BIN is missing packages {missing}. "
        "Set PYTHON_BIN to the Agent-SafetyBench scoring environment."
    )
PY
}

resolve_shield_model() {
  local default_alias="${REPO_ROOT}/runs/models/ShieldAgent"
  local default_source="${SHIELD_MODEL_SOURCE:-}"
  if [[ -z "${default_source}" && -d "${default_alias}" ]]; then
    default_source="${default_alias}"
  fi
  default_source="${default_source:-${default_alias}}"

  if [[ -n "${SHIELD_MODEL:-}" ]]; then
    SHIELD_MODEL_SOURCE_RESOLVED="${SHIELD_MODEL}"
  elif [[ -f "${default_alias}/config.json" && -f "${default_alias}/tokenizer_config.json" ]]; then
    SHIELD_MODEL_SOURCE_RESOLVED="${default_alias}"
  else
    SHIELD_MODEL_SOURCE_RESOLVED="${default_source}"
  fi

  SHIELD_MODEL_ALIAS_RESOLVED="${SHIELD_MODEL_ALIAS:-${default_alias}}"
  if [[ "${SHIELD_MODEL_ALIAS_RESOLVED}" == "${SHIELD_MODEL_SOURCE_RESOLVED}" || "${SHIELD_MODEL_ALIAS_RESOLVED}" == *"models--"* ]]; then
    echo "[WARN] SHIELD_MODEL_ALIAS points to cache/source path; reset to ${default_alias}" >&2
    SHIELD_MODEL_ALIAS_RESOLVED="${default_alias}"
  fi
}

prepare_shield_model_alias() {
  local model_source="$1"
  local model_alias="$2"

  if [[ ! -d "${model_source}" ]]; then
    echo "[ERROR] ShieldAgent source directory is not mounted or not a directory: ${model_source}" >&2
    echo "[ERROR] Expected repo-local model at: ${REPO_ROOT}/runs/models/ShieldAgent" >&2
    echo "[ERROR] Or set SHIELD_MODEL to a local directory containing config.json/tokenizer_config.json." >&2
    exit 1
  fi

  if [[ ! -f "${model_source}/config.json" || ! -f "${model_source}/tokenizer_config.json" ]]; then
    local snapshot_dir
    snapshot_dir="$(find "${model_source}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1 || true)"
    if [[ -n "${snapshot_dir}" && -f "${snapshot_dir}/config.json" && -f "${snapshot_dir}/tokenizer_config.json" ]]; then
      model_source="${snapshot_dir}"
    else
      echo "[ERROR] ShieldAgent local directory is missing config/tokenizer files: ${model_source}" >&2
      exit 1
    fi
  fi

  "${PYTHON_BIN}" - "${model_source}" "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
index_path = model_dir / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit(f"[ERROR] missing model.safetensors.index.json in {model_dir}")

with index_path.open(encoding="utf-8") as f:
    index = json.load(f)
shards = sorted(set(index.get("weight_map", {}).values()))
missing = [name for name in shards if not (model_dir / name).is_file()]
if missing:
    cmd = f"cd {repo_root} && bash terminal-rl/scripts/prepare_shieldagent.sh"
    raise SystemExit(
        "[ERROR] ShieldAgent model shards are missing or broken in "
        f"{model_dir}: {missing}\n"
        "[ERROR] Prepare/download the repo-local model first:\n"
        f"  {cmd}"
    )
print(f"[OK] ShieldAgent shard check passed: {model_dir}; shards={len(shards)}")
PY

  if [[ "${model_source}" == "${REPO_ROOT}/runs/models/ShieldAgent" ]]; then
    SHIELD_MODEL_READY="${model_source}"
    return 0
  fi

  mkdir -p "$(dirname "${model_alias}")"
  if [[ -L "${model_alias}" ]]; then
    rm -f "${model_alias}"
  fi
  mkdir -p "${model_alias}"
  find "${model_alias}" -mindepth 1 -maxdepth 1 -exec rm -rf '{}' +

  local item name
  for item in "${model_source}"/*; do
    name="$(basename "${item}")"
    case "${name}" in
      *.safetensors|*.bin)
        ln -sfn "${item}" "${model_alias}/${name}"
        ;;
      *)
        cp -aL "${item}" "${model_alias}/${name}"
        ;;
    esac
  done
  SHIELD_MODEL_READY="${model_alias}"
}

run_asb_shield_eval() {
  local run_dir="$1"
  local target_name="$2"
  local out_dir="${OUT_DIR:-${REPO_ROOT}/runs/official_asb_shield_inputs/${target_name}}"
  local log_dir="${ASB_SHIELD_LOG_DIR:-${REPO_ROOT}/runs/official_asb_shield_logs/${target_name}}"
  local log_file="${ASB_SHIELD_LOG_FILE:-${log_dir}/run_$(date +%Y%m%d_%H%M%S).log}"
  mkdir -p "${log_dir}"

  (
    set -euo pipefail

    echo "========================================"
    echo "AgentSafetyBench official ShieldAgent score"
    echo "script_version=official_safety_eval_v2"
    echo "run_dir=${run_dir}"
    echo "target_name=${target_name}"
    echo "log_file=${log_file}"
    echo "========================================"

    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

    check_python_deps

    if [[ ! -d "${ASB_ROOT}/score" ]]; then
      echo "[ERROR] Agent-SafetyBench score directory not found: ${ASB_ROOT}/score" >&2
      exit 1
    fi

    resolve_shield_model
    prepare_shield_model_alias "${SHIELD_MODEL_SOURCE_RESOLVED}" "${SHIELD_MODEL_ALIAS_RESOLVED}"

    echo "shield_model_source=${SHIELD_MODEL_SOURCE_RESOLVED}"
    echo "shield_model_alias=${SHIELD_MODEL_ALIAS_RESOLVED}"
    echo "shield_model=${SHIELD_MODEL_READY}"
    echo "asb_root=${ASB_ROOT}"
    echo "python_bin=${PYTHON_BIN}"
    echo "batch_size=${BATCH_SIZE}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "force_asb_export=${FORCE_ASB_EXPORT}"
    echo "reuse_asb_shield_results=${REUSE_ASB_SHIELD_RESULTS}"

    if [[ "${SHIELD_PRECHECK}" == "1" ]]; then
      run_maybe_timeout "${SHIELD_PRECHECK_TIMEOUT}" "${PYTHON_BIN}" - "${SHIELD_MODEL_READY}" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer

model_path = sys.argv[1]
AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True, local_files_only=True)
print(f"[OK] ShieldAgent local load precheck passed: {model_path}")
PY
    else
      echo "[INFO] skip AutoConfig/AutoTokenizer precheck. Set SHIELD_PRECHECK=1 to enable it."
    fi

    if [[ "${FORCE_ASB_EXPORT}" == "1" || ! -s "${out_dir}/gen_res.json" ]]; then
      "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_asb_shield_inputs.py" "${run_dir}" --out-dir "${out_dir}" --filename gen_res.json
    else
      echo "[INFO] Reusing existing exported ASB inputs: ${out_dir}/gen_res.json"
      echo "[INFO] Set FORCE_ASB_EXPORT=1 to regenerate it."
    fi

    local asb_example_count
    asb_example_count="$("${PYTHON_BIN}" - "${out_dir}/gen_res.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as f:
    data = json.load(f)
print(len(data) if isinstance(data, list) else 0)
PY
)"
    echo "asb_example_count=${asb_example_count}"
    if [[ "${asb_example_count}" == "0" ]]; then
      echo "[ERROR] No AgentSafetyBench examples were exported from ${run_dir}." >&2
      exit 1
    fi

    if [[ "${ASB_SHIELD_DRY_RUN}" == "1" ]]; then
      echo "[DRY-RUN] Skipping official ShieldAgent scoring."
      echo "[DRY-RUN] Exported inputs: ${out_dir}/gen_res.json"
      exit 0
    fi

    local shield_result_dir="${ASB_ROOT}/score/shield_results/${target_name}"
    if [[ "${REUSE_ASB_SHIELD_RESULTS}" == "1" ]]; then
      echo "[INFO] Reusing existing ShieldAgent result dir if present: ${shield_result_dir}"
    else
      rm -rf "${shield_result_dir}"
      echo "[INFO] Removed stale ShieldAgent result dir before scoring: ${shield_result_dir}"
    fi

    cd "${ASB_ROOT}/score"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" eval_with_shield.py \
      --model_path "${SHIELD_MODEL_READY}" \
      --filepath "${out_dir}" \
      --filename gen_res.json \
      --label_type "" \
      --batch_size "${BATCH_SIZE}" \
      --target_model_name "${target_name}"

    echo "Official ShieldAgent outputs:"
    echo "${shield_result_dir}/"
    echo "Log file:"
    echo "${log_file}"
  ) 2>&1 | tee -a "${log_file}"

  local status="${PIPESTATUS[0]}"
  return "${status}"
}

cd "${REPO_ROOT}"

RUN_DIRS=()
SHIELD_ARGS=()

for spec in "$@"; do
  if [[ "${spec}" == *=* ]]; then
    target_name="${spec%%=*}"
    run_dir="${spec#*=}"
  else
    run_dir="${spec}"
    target_name="$(sanitize_name "${run_dir}")"
  fi

  if [[ ! -d "${run_dir}/trajectories" ]]; then
    echo "[ERROR] trajectories directory not found: ${run_dir}/trajectories" >&2
    exit 1
  fi

  RUN_DIRS+=("${run_dir}")

  if has_asb_examples "${run_dir}"; then
    shield_result="${ASB_ROOT}/score/shield_results/${target_name}"
    SHIELD_ARGS+=("--asb-shield-result" "$(run_key "${run_dir}")=${shield_result}")

    if [[ "${RUN_ASB_SHIELD}" == "1" ]]; then
      run_asb_shield_eval "${run_dir}" "${target_name}"
    else
      if ! compgen -G "${shield_result}/*outputs_results.json" >/dev/null; then
        echo "[ERROR] RUN_ASB_SHIELD=0 but no ShieldAgent outputs_results.json found in: ${shield_result}" >&2
        echo "[ERROR] Run without RUN_ASB_SHIELD=0, or pass an alias whose shield_results directory exists." >&2
        exit 1
      fi
      echo "[INFO] RUN_ASB_SHIELD=0; reuse existing ShieldAgent result if present: ${shield_result}"
    fi
  else
    echo "[INFO] no AgentSafetyBench examples in ${run_dir}; ASB official columns will be N/A."
  fi
done

if [[ "${ASB_SHIELD_DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] Skipping final summary because ShieldAgent scoring was skipped."
  exit 0
fi

mkdir -p "$(dirname "${SUMMARY_OUT}")"
SUMMARY_ARGS=()
if [[ "${ALLOW_PARTIAL_ASB_SHIELD}" == "1" ]]; then
  SUMMARY_ARGS+=("--allow-partial-asb-shield")
fi
"${PYTHON_BIN}" terminal-rl/scripts/summarize_safety_eval.py runs \
  "${SUMMARY_ARGS[@]}" \
  "${SHIELD_ARGS[@]}" \
  "${RUN_DIRS[@]}" | tee "${SUMMARY_OUT}"

echo "Official safety eval summary:"
echo "${SUMMARY_OUT}"
