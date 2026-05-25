#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export A3S_CODE_MODEL_VARIANT="${A3S_CODE_MODEL_VARIANT:-qwen3-4b}"

exec bash "${SCRIPT_DIR}/run_a3s_code_rl.sh" "$@"
