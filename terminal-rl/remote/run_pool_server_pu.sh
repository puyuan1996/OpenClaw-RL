#!/usr/bin/env bash
# Wrapper around run_pool_server.sh that auto-mirrors logs to tmp_doc_latest/
# for easy inspection from any machine on the shared storage.
#
# Usage (on CPU/docker worker):
#   bash terminal-rl/remote/run_pool_server_pu.sh
#
# Logs written:
#   tmp_doc_latest/cpu_pool.log   — full stdout/stderr
#   tmp_doc_latest/cpu_err.log    — live-filtered errors (updated every 30s)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TERMINAL_RL="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

TMP_DOC_LATEST="${REPO_ROOT}/tmp_doc_latest"
mkdir -p "${TMP_DOC_LATEST}"

CPU_POOL_LOG="${TMP_DOC_LATEST}/cpu_pool.log"
CPU_ERR_LOG="${TMP_DOC_LATEST}/cpu_err.log"

echo "[$(date +'%F %T')] pool_server_pu wrapper starting" | tee "${CPU_POOL_LOG}"
echo "  full log: ${CPU_POOL_LOG}"
echo "  err log:  ${CPU_ERR_LOG}"
echo "  ======================================================="
echo "  Press Ctrl-C to stop. Live logs below:"
echo "  ======================================================="
echo

# Force unbuffered output so uvicorn access logs show immediately on the
# foreground terminal (instead of being held in stdio buffers).
export PYTHONUNBUFFERED=1

# Background error filter: every 30s, extract error lines from the full log.
(
  while true; do
    sleep 30
    grep -E "Error|Exception|Traceback|500|502|PermissionError|docker|FAILED|Connection" \
         "${CPU_POOL_LOG}" 2>/dev/null \
      | grep -v "DeprecationWarning" \
      | tail -n 200 \
      > "${CPU_ERR_LOG}" 2>/dev/null || true
  done
) &
ERR_FILTER_PID=$!

cleanup() {
  kill "${ERR_FILTER_PID}" 2>/dev/null || true
  # Final snapshot of errors
  grep -E "Error|Exception|Traceback|500|502|PermissionError|docker|FAILED|Connection" \
       "${CPU_POOL_LOG}" 2>/dev/null \
    | grep -v "DeprecationWarning" \
    | tail -n 200 \
    > "${CPU_ERR_LOG}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Run the real pool_server, tee-ing output to the shared log.
# stdbuf forces line-buffered stdout/stderr so logs appear in real time on the
# foreground terminal AND in the shared log file simultaneously.
stdbuf -oL -eL bash "${SCRIPT_DIR}/run_pool_server.sh" 2>&1 | tee -a "${CPU_POOL_LOG}"
