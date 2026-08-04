#!/usr/bin/env bash
# Repair SQLAlchemy in the exact Python environment used by pool_server.
#
# Usage on a CPU/docker worker:
#   bash terminal-rl/remote/fix_cpu_worker_sqlalchemy_env.sh
#   WORKER_MAX_TASKS=32 WORKER_MAX_RUNS_PER_TASK=4 WORKER_MAX_CONCURRENT_CLOSES=16 \
#     bash terminal-rl/remote/run_pool_server_pu_v2.sh

set -euo pipefail

REPO_ROOT="${1:-/mnt/shared-storage-user/puyuan/code/OpenClaw-RL}"
VENV_PY="${VENV_PY:-${REPO_ROOT}/.venv/bin/python}"
SQLALCHEMY_VERSION="${SQLALCHEMY_VERSION:-2.0.50}"
CLEAN_SQLALCHEMY="${CLEAN_SQLALCHEMY:-1}"

log() { echo "[$(date +'%F %T')] $*"; }

cd "${REPO_ROOT}"

log "=== CPU worker SQLAlchemy repair ==="
log "repo_root=${REPO_ROOT}"
log "venv_python=${VENV_PY}"

if [[ ! -x "${VENV_PY}" ]]; then
  log "ERROR: ${VENV_PY} is not executable."
  log "This usually means the shared .venv is broken on this worker."
  log "Recreate it with: uv venv .venv --python 3.12"
  exit 1
fi

log "Shell python: $(command -v python || true)"
python -V 2>/dev/null || true

log "Pool python identity:"
"${VENV_PY}" - <<'PY'
import site
import sys
import sysconfig

print("  executable:", sys.executable)
print("  version:", sys.version.replace("\n", " "))
print("  purelib:", sysconfig.get_paths().get("purelib"))
print("  sitepackages:", site.getsitepackages())
PY

SITE_PACKAGES="$("${VENV_PY}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

log "site_packages=${SITE_PACKAGES}"
mkdir -p "${SITE_PACKAGES}"

if [[ "${CLEAN_SQLALCHEMY}" == "1" ]]; then
  BACKUP_DIR="/tmp/openclaw_sqlalchemy_backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${BACKUP_DIR}"
  shopt -s nullglob
  moved=0
  for path in \
    "${SITE_PACKAGES}/sqlalchemy" \
    "${SITE_PACKAGES}/SQLAlchemy-"*.dist-info \
    "${SITE_PACKAGES}/sqlalchemy-"*.dist-info
  do
    if [[ -e "${path}" ]]; then
      log "Moving stale package aside: ${path}"
      mv "${path}" "${BACKUP_DIR}/"
      moved=1
    fi
  done
  shopt -u nullglob
  if [[ "${moved}" == "1" ]]; then
    log "Backed up old SQLAlchemy files to ${BACKUP_DIR}"
  else
    rmdir "${BACKUP_DIR}" 2>/dev/null || true
  fi
fi

log "Ensuring pip exists in pool Python..."
"${VENV_PY}" -m ensurepip --upgrade >/dev/null 2>&1 || true

log "Installing SQLAlchemy into pool Python..."
"${VENV_PY}" -m pip install --force-reinstall --no-cache-dir \
  "SQLAlchemy==${SQLALCHEMY_VERSION}" \
  "greenlet>=1" \
  "typing_extensions>=4.6.0"

log "Verifying imports with pool Python..."
"${VENV_PY}" - <<'PY'
import importlib.util
import sqlalchemy
import sys

print("  executable:", sys.executable)
print("  sqlalchemy_version:", sqlalchemy.__version__)
print("  sqlalchemy_file:", sqlalchemy.__file__)
print("  postgresql_spec:", importlib.util.find_spec("sqlalchemy.dialects.postgresql"))

from sqlalchemy.dialects.postgresql import JSONB
print("  JSONB OK:", JSONB)

from terminal_bench.handlers.trial_handler import TrialHandler
print("  terminal_bench TrialHandler OK:", TrialHandler)
PY

log "Repair complete."
log "Restart pool_server with:"
log "  WORKER_MAX_TASKS=32 WORKER_MAX_RUNS_PER_TASK=4 WORKER_MAX_CONCURRENT_CLOSES=16 bash terminal-rl/remote/run_pool_server_pu_v2.sh"
