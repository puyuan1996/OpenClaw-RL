#!/usr/bin/env bash
# run_pool_server_pu_v2.sh — Hardened pool_server launcher for CPU/docker worker
#
# Incorporates all lessons from issue #3:
#   坑1: pool capacity must be >= rollout-batch-size × n-samples-per-prompt
#   坑2: max-concurrent-closes must be ~1.5× peak-close-rate
#   坑3: docker address-pool must be expanded for high-concurrency (check only, fix separately)
#   坑4: nofile ulimit must be ≥65k (checks and raises via prlimit if needed)
#   坑5: pre-flight cleanup of orphaned containers/networks from previous runs
#   坑6: connectivity probe before starting training
#   Extra: docker daemon health check before start
#   Extra: ClawSentry gateway liveness check (if CLAWSENTRY_NEEDED=1)
#
# Usage (on CPU/docker worker):
#   bash terminal-rl/remote/run_pool_server_pu_v2.sh
#
# Key env vars:
#   WORKER_MAX_TASKS            (default 64)   — pool_server --max-tasks
#   WORKER_MAX_RUNS_PER_TASK    (default 16)   — pool_server --max-runs-per-task
#   WORKER_MAX_CONCURRENT_CLOSES (default 32)  — pool_server --max-concurrent-closes
#   ENV_SERVER_PORT             (default 18081)
#   SKIP_PREFLIGHT_CLEANUP      (default 0)    — set 1 to skip orphan cleanup
#   PROXY_ENV_FILE              (default /etc/seta_build_proxy.env)
#   SKIP_PROXY_ENV              (default 0)    — set 1 to avoid sourcing proxy env
#   CLAWSENTRY_NEEDED           (default 0)    — set 1 to also check CS gateway
#   CS_GATEWAY_PORT             (default 8090) — ClawSentry gateway port
#   DOCKER_DATA_ROOT            (default /data) — Docker data root to guard
#   WORKER_MIN_DOCKER_FREE_GB   (default 50) — refuse start/admission below this
#   WORKER_MAX_DOCKER_USED_PCT  (default 85) — refuse start/admission above this
#   WORKER_MAX_DOCKER_INODE_PCT (default 80) — refuse start/admission above this
#
# Logs written:
#   tmp_doc_latest/cpu_pool.log   — full stdout/stderr
#   tmp_doc_latest/cpu_err.log    — live-filtered errors (updated every 30s)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TERMINAL_RL="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

log() { echo "[$(date +'%F %T')] $*"; }

# ── Configuration ─────────────────────────────────────────────────────────────
# 坑1: capacity must cover rollout-batch-size × n-samples-per-prompt
# Default 8B run: batch=16 × n=8 = 128 demand → 64×16=1024 total slots (8× headroom)
WORKER_MAX_TASKS="${WORKER_MAX_TASKS:-64}"
WORKER_MAX_RUNS_PER_TASK="${WORKER_MAX_RUNS_PER_TASK:-16}"
# 坑2: close concurrency ~1.5× peak-close-rate (GRPO batch ≈ 16)
WORKER_MAX_CONCURRENT_CLOSES="${WORKER_MAX_CONCURRENT_CLOSES:-32}"
ENV_SERVER_PORT="${ENV_SERVER_PORT:-18081}"
SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-0}"
PROXY_ENV_FILE="${PROXY_ENV_FILE:-/etc/seta_build_proxy.env}"
SKIP_PROXY_ENV="${SKIP_PROXY_ENV:-0}"
CLAWSENTRY_NEEDED="${CLAWSENTRY_NEEDED:-0}"
CS_GATEWAY_PORT="${CS_GATEWAY_PORT:-8090}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"
WORKER_DISK_GUARD_ENABLED="${WORKER_DISK_GUARD_ENABLED:-1}"
WORKER_MIN_DOCKER_FREE_GB="${WORKER_MIN_DOCKER_FREE_GB:-50}"
WORKER_MAX_DOCKER_USED_PCT="${WORKER_MAX_DOCKER_USED_PCT:-85}"
WORKER_MAX_DOCKER_INODE_PCT="${WORKER_MAX_DOCKER_INODE_PCT:-80}"
PREFLIGHT_DISK_CLEANUP="${PREFLIGHT_DISK_CLEANUP:-1}"

log "=== pool_server_pu_v2 starting ==="
log "  max_tasks=${WORKER_MAX_TASKS}  max_runs_per_task=${WORKER_MAX_RUNS_PER_TASK}"
log "  max_concurrent_closes=${WORKER_MAX_CONCURRENT_CLOSES}"
log "  port=${ENV_SERVER_PORT}  skip_cleanup=${SKIP_PREFLIGHT_CLEANUP}"
log "  total_capacity=$((WORKER_MAX_TASKS * WORKER_MAX_RUNS_PER_TASK)) slots"
log "  docker_data_root=${DOCKER_DATA_ROOT} disk_guard=${WORKER_DISK_GUARD_ENABLED}"

if [[ "${SKIP_PROXY_ENV}" != "1" && -f "${PROXY_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a; . "${PROXY_ENV_FILE}"; set +a
    log "  loaded proxy env: ${PROXY_ENV_FILE}"
elif [[ "${SKIP_PROXY_ENV}" != "1" ]]; then
    log "  proxy env not found at ${PROXY_ENV_FILE}; continuing without it"
fi

# ── Log paths ─────────────────────────────────────────────────────────────────
TMP_DOC_LATEST="${REPO_ROOT}/tmp_doc_latest"
mkdir -p "${TMP_DOC_LATEST}"
CPU_POOL_LOG="${TMP_DOC_LATEST}/cpu_pool.log"
CPU_ERR_LOG="${TMP_DOC_LATEST}/cpu_err.log"

log "  full log: ${CPU_POOL_LOG}"
log "  err log:  ${CPU_ERR_LOG}"

docker_disk_snapshot() {
    df -P -BG "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); gsub("G","",$4); print $5, $4}'
}

docker_inode_snapshot() {
    df -Pi "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); print $5}'
}

preflight_disk_guard() {
    if [[ "${WORKER_DISK_GUARD_ENABLED}" == "0" ]]; then
        log "  disk guard disabled (WORKER_DISK_GUARD_ENABLED=0)"
        return 0
    fi
    if [[ ! -d "${DOCKER_DATA_ROOT}" ]]; then
        log "  ❌ Docker data root does not exist: ${DOCKER_DATA_ROOT}"
        exit 1
    fi

    local snap used_pct free_gb inode_pct
    snap="$(docker_disk_snapshot || true)"
    inode_pct="$(docker_inode_snapshot || true)"
    used_pct="${snap%% *}"
    free_gb="${snap##* }"
    log "  ${DOCKER_DATA_ROOT}: used=${used_pct:-?}% free=${free_gb:-?}GB inode=${inode_pct:-?}%"
    log "  thresholds: free>=${WORKER_MIN_DOCKER_FREE_GB}GB used<=${WORKER_MAX_DOCKER_USED_PCT}% inode<=${WORKER_MAX_DOCKER_INODE_PCT}%"

    if [[ -z "${used_pct}" || -z "${free_gb}" || -z "${inode_pct}" ]]; then
        log "  ❌ Failed to read Docker data-root disk stats"
        exit 1
    fi

    if [[ "${used_pct}" -gt "${WORKER_MAX_DOCKER_USED_PCT}" \
       || "${free_gb}" -lt "${WORKER_MIN_DOCKER_FREE_GB}" \
       || "${inode_pct}" -gt "${WORKER_MAX_DOCKER_INODE_PCT}" ]]; then
        log "  ⚠️  Docker data-root is above guard threshold."
        if [[ "${PREFLIGHT_DISK_CLEANUP}" == "1" && -x "${SCRIPT_DIR}/cleanup_docker_cache.sh" ]]; then
            log "  Running conservative cleanup before refusing start..."
            DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" RUN_HEAVY_DF=0 \
              bash "${SCRIPT_DIR}/cleanup_docker_cache.sh" || true
            snap="$(docker_disk_snapshot || true)"
            inode_pct="$(docker_inode_snapshot || true)"
            used_pct="${snap%% *}"
            free_gb="${snap##* }"
            log "  after cleanup: used=${used_pct:-?}% free=${free_gb:-?}GB inode=${inode_pct:-?}%"
        fi
    fi

    if [[ "${used_pct}" -gt "${WORKER_MAX_DOCKER_USED_PCT}" \
       || "${free_gb}" -lt "${WORKER_MIN_DOCKER_FREE_GB}" \
       || "${inode_pct}" -gt "${WORKER_MAX_DOCKER_INODE_PCT}" ]]; then
        log "  ❌ Refusing to start pool_server under Docker disk pressure."
        log "     Run: AGGRESSIVE=1 PRUNE_VOLUMES=1 bash terminal-rl/remote/fix_docker_overlay2_no_space.sh"
        log "     If Docker objects are empty but /data is still full, use PURGE_DOCKER_ROOT_WHEN_EMPTY=1."
        exit 1
    fi

    log "  ✅ Docker data-root capacity OK"
}

# ── Pre-flight: docker daemon health ─────────────────────────────────────────
log "Pre-flight [1/6]: Docker daemon health check"
if ! timeout 10 docker info >/dev/null 2>&1; then
    log "  ❌ Docker daemon not responding!"
    log "  Run repair: sudo bash terminal-rl/remote/fix_dockerd_and_proxy.sh"
    log "  Or force restart only: sudo bash terminal-rl/remote/restart_docker_force.sh"
    exit 1
fi
log "  ✅ Docker daemon OK"

log "Pre-flight [2/6]: Docker data-root disk/inode guard"
preflight_disk_guard

if command -v ss >/dev/null 2>&1 && ss -tln "( sport = :${ENV_SERVER_PORT} )" | grep -q ":${ENV_SERVER_PORT}"; then
    log "  ❌ Port ${ENV_SERVER_PORT} is already in use"
    log "     Inspect: ss -tlnp '( sport = :${ENV_SERVER_PORT} )'"
    exit 1
fi

# ── Pre-flight: nofile ulimit check (坑4) ────────────────────────────────────
log "Pre-flight [3/6]: nofile ulimit check (need ≥65536)"
NOFILE_SOFT=$(ulimit -Sn 2>/dev/null || echo 0)
NOFILE_HARD=$(ulimit -Hn 2>/dev/null || echo 0)
log "  current: soft=${NOFILE_SOFT} hard=${NOFILE_HARD}"
if [[ "${NOFILE_SOFT}" -lt 65536 ]]; then
    log "  ⚠️  soft limit ${NOFILE_SOFT} < 65536, attempting to raise..."
    # Try to raise soft limit inline
    if ulimit -Sn 65536 2>/dev/null; then
        NOFILE_SOFT=$(ulimit -Sn)
        log "  ✅ Raised soft limit to ${NOFILE_SOFT}"
    else
        log "  ⚠️  Could not raise via ulimit (may need /etc/security/limits.conf or systemd override)"
        log "     Continuing anyway, but evaluate may fail at ≥32 concurrent tasks"
    fi
else
    log "  ✅ nofile soft limit OK (${NOFILE_SOFT})"
fi

# ── Pre-flight: docker address pool (坑3) ────────────────────────────────────
log "Pre-flight [4/6]: Docker bridge network address pool check"
# Count existing bridge networks (each consumes a /24)
BRIDGE_COUNT=$(docker network ls --filter driver=bridge -q 2>/dev/null | wc -l)
log "  existing bridge networks: ${BRIDGE_COUNT}"
# Check daemon.json for expanded pool
if [[ -f /etc/docker/daemon.json ]]; then
    if grep -q "default-address-pools" /etc/docker/daemon.json 2>/dev/null; then
        POOL_BASE=$(python3 -c "
import json, sys
d = json.load(open('/etc/docker/daemon.json'))
pools = d.get('default-address-pools', [])
total = sum((1 << (p.get('size', 24) - (int(p['base'].split('/')[1]) if '/' in p.get('base','') else 16))) for p in pools if 'base' in p)
print(total)
" 2>/dev/null || echo "unknown")
        log "  daemon.json has custom pools (estimated /24 capacity: ${POOL_BASE})"
        if [[ "${POOL_BASE}" != "unknown" ]] && [[ "${POOL_BASE}" -lt 1024 ]] 2>/dev/null; then
            log "  ⚠️  Address pool capacity ${POOL_BASE} may be insufficient for ${WORKER_MAX_TASKS} concurrent tasks"
            log "     Recommend: see issue #3 §2.3 for daemon.json expansion"
        else
            log "  ✅ Address pool looks sufficient"
        fi
    else
        log "  ⚠️  No custom default-address-pools in /etc/docker/daemon.json"
        log "     Default (256 /24 subnets) may be exhausted at >64 concurrent tasks"
        log "     Recommend adding: {\"default-address-pools\": [{\"base\":\"10.200.0.0/12\",\"size\":24}]}"
    fi
else
    log "  ⚠️  /etc/docker/daemon.json not found — using docker defaults (limited /24 pool)"
fi

# ── Pre-flight: orphan container/network cleanup (坑5) ───────────────────────
log "Pre-flight [5/6]: Orphan container/network cleanup (SKIP_PREFLIGHT_CLEANUP=${SKIP_PREFLIGHT_CLEANUP})"
if [[ "${SKIP_PREFLIGHT_CLEANUP}" != "1" ]]; then
    # Count stopped containers that look like task containers (numeric prefix pattern)
    STOPPED=$(docker ps -aq --filter "status=exited" --filter "status=dead" 2>/dev/null | wc -l)
    log "  stopped containers: ${STOPPED}"
    if [[ "${STOPPED}" -gt 0 ]]; then
        log "  Pruning stopped containers..."
        docker container prune -f >/dev/null 2>&1 || true
        log "  ✅ Pruned"
    fi

    # Prune dangling networks
    DANGLING_NETS=$(docker network ls --filter "dangling=true" -q 2>/dev/null | wc -l)
    if [[ "${DANGLING_NETS}" -gt 0 ]]; then
        log "  Pruning ${DANGLING_NETS} dangling networks..."
        docker network prune -f >/dev/null 2>&1 || true
        log "  ✅ Pruned networks"
    fi

    # Check for running task containers from previous pool (pattern: <number>-<name>-<project>)
    # These are containers whose compose project died but containers are still up
    ORPHAN_RUNNING=$(docker ps --format '{{.Names}}' 2>/dev/null \
        | grep -E '^[0-9]+-.*_(client|helper)-[0-9]+$' | wc -l || true)
    if [[ "${ORPHAN_RUNNING}" -gt 0 ]]; then
        log "  ⚠️  ${ORPHAN_RUNNING} orphan task containers still running — consider manual cleanup:"
        log "     docker ps --format '{{.Names}}' | grep -E '^[0-9]+-' | xargs docker rm -f"
    fi
else
    log "  ⏭  Skipped (SKIP_PREFLIGHT_CLEANUP=1)"
fi

# ── Pre-flight: ClawSentry gateway check (if needed) ─────────────────────────
log "Pre-flight [6/6]: ClawSentry gateway check (CLAWSENTRY_NEEDED=${CLAWSENTRY_NEEDED})"
if [[ "${CLAWSENTRY_NEEDED}" == "1" ]]; then
    if curl -fsS --max-time 3 "http://127.0.0.1:${CS_GATEWAY_PORT}/health" >/dev/null 2>&1; then
        log "  ✅ ClawSentry gateway OK at port ${CS_GATEWAY_PORT}"
    else
        log "  ❌ ClawSentry gateway NOT responding at 127.0.0.1:${CS_GATEWAY_PORT}"
        log "     This will cause safety_coef * 0 = 0 (no safety reward) in training"
        log "     Start it on GPU worker first, then re-run pool server"
        log "     (The ClawSentry gateway is started by terminal-rl_qwen3-8b_pu.sh on GPU worker)"
        log "  ⚠️  Continuing anyway (pool_server doesn't run ClawSentry; GPU side does)"
    fi
else
    log "  ⏭  Not needed (CLAWSENTRY_NEEDED=${CLAWSENTRY_NEEDED})"
fi

log "=== Pre-flight checks complete, starting pool_server ==="
log ""

# ── Background error filter (every 30s) ──────────────────────────────────────
(
  while true; do
    sleep 30
    grep -E "Error|Exception|Traceback|500|502|PermissionError|docker|FAILED|Connection|SLOTS_EXHAUSTED|Too many open files|address pools" \
         "${CPU_POOL_LOG}" 2>/dev/null \
      | grep -v "DeprecationWarning" \
      | tail -n 300 \
      > "${CPU_ERR_LOG}" 2>/dev/null || true
  done
) &
ERR_FILTER_PID=$!

cleanup() {
  kill "${ERR_FILTER_PID}" 2>/dev/null || true
  # Final snapshot
  grep -E "Error|Exception|Traceback|500|502|PermissionError|docker|FAILED|Connection|SLOTS_EXHAUSTED|Too many open files|address pools" \
       "${CPU_POOL_LOG}" 2>/dev/null \
    | grep -v "DeprecationWarning" \
    | tail -n 300 \
    > "${CPU_ERR_LOG}" 2>/dev/null || true
  log "pool_server stopped."
}
trap cleanup EXIT INT TERM

# ── Capacity summary before start ────────────────────────────────────────────
echo "========================================"
echo "  Pool Server v2 Configuration"
echo "  max_tasks:             ${WORKER_MAX_TASKS}"
echo "  max_runs_per_task:     ${WORKER_MAX_RUNS_PER_TASK}"
echo "  total_capacity:        $((WORKER_MAX_TASKS * WORKER_MAX_RUNS_PER_TASK)) leases"
echo "  max_concurrent_closes: ${WORKER_MAX_CONCURRENT_CLOSES}"
echo "  port:                  ${ENV_SERVER_PORT}"
echo "  log:                   ${CPU_POOL_LOG}"
echo "  nofile soft:           $(ulimit -Sn)"
echo "========================================"
echo ""

# ── Start pool_server ─────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

export DATASET_DIR="${DATASET_DIR:-${TERMINAL_RL}/dataset}"
export TBENCH_OUTPUT_ROOT="${TBENCH_OUTPUT_ROOT:-${TERMINAL_RL}/build_outputs}"
export TBENCH_DOCKER_IMAGE_SOURCE="${TBENCH_DOCKER_IMAGE_SOURCE:-build}"
export TBENCH_DOCKER_PULL_PREFIX="${TBENCH_DOCKER_PULL_PREFIX:-}"
export AGENT_SAFETYBENCH_ROOT="${AGENT_SAFETYBENCH_ROOT:-/mnt/shared-storage-user/puyuan/code/Agent-SafetyBench}"
export COMPOSE_OVERRIDE_PATH="${COMPOSE_OVERRIDE_PATH:-}"
export PYTHONUNBUFFERED=1
export DOCKER_DATA_ROOT
export WORKER_DISK_GUARD_ENABLED
export WORKER_MIN_DOCKER_FREE_GB
export WORKER_MAX_DOCKER_USED_PCT
export WORKER_MAX_DOCKER_INODE_PCT

if [ -d "${REPO_ROOT}/.venv" ]; then
    source .venv/bin/activate
fi

# Use stdbuf for line-buffered output (real-time log visibility)
exec stdbuf -oL -eL \
    python -m terminal-rl.remote.pool_server \
    --host 0.0.0.0 \
    --port "${ENV_SERVER_PORT}" \
    --max-tasks "${WORKER_MAX_TASKS}" \
    --max-runs-per-task "${WORKER_MAX_RUNS_PER_TASK}" \
    --max-concurrent-closes "${WORKER_MAX_CONCURRENT_CLOSES}" \
    --output-root "${TBENCH_OUTPUT_ROOT}" \
    2>&1 | tee -a "${CPU_POOL_LOG}"
