#!/usr/bin/env bash
# restart_docker_force.sh — 强制重启 docker（绕过卡住的 systemctl）
# 用法:
#   sudo env DOCKER_DATA_ROOT=/data bash terminal-rl/remote/restart_docker_force.sh
#
# Env:
#   DOCKER_DATA_ROOT  Docker data root. DOCKER_ROOT is accepted as legacy alias.
#   PROXY_ENV_FILE    Proxy env file sourced before starting dockerd. Default: /etc/seta_build_proxy.env

set -uo pipefail

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "[ERROR] $*"; exit 1; }

if [ "$(id -u)" -ne 0 ]; then
    die "Must run as root. Use: sudo env DOCKER_DATA_ROOT=/data bash terminal-rl/remote/restart_docker_force.sh"
fi

DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"
PROXY_ENV_FILE="${PROXY_ENV_FILE:-/etc/seta_build_proxy.env}"

log "Force restarting docker on $(hostname)"
log "  DOCKER_DATA_ROOT=${DATA_ROOT}"
log "  PROXY_ENV_FILE=${PROXY_ENV_FILE}"

# ── Step 1: 直接 pkill -9，不走 systemctl ─────────────────────────────
log "Step 1: Kill dockerd (skip systemctl, use kill -9 directly)"

# 1a. 找 dockerd PID
DOCKERD_PIDS=$(pgrep -x dockerd 2>/dev/null || true)
if [ -n "${DOCKERD_PIDS}" ]; then
    log "  Found dockerd PIDs: ${DOCKERD_PIDS}"
    for pid in ${DOCKERD_PIDS}; do
        log "    kill -9 ${pid}"
        kill -9 "${pid}" 2>/dev/null || true
    done
    sleep 2
    if pgrep -x dockerd >/dev/null 2>&1; then
        log "  ⚠️ dockerd still alive after SIGKILL (may be in D state, kernel will reap eventually)"
    else
        log "  ✓ dockerd killed"
    fi
else
    log "  No dockerd running"
fi

# 1b. 杀残留 shim（直接 pkill，不要 -f 全匹配）
SHIM_PIDS=$(pgrep containerd-shim 2>/dev/null || true)
if [ -n "${SHIM_PIDS}" ]; then
    SHIM_COUNT=$(echo "${SHIM_PIDS}" | wc -l)
    log "  Killing ${SHIM_COUNT} containerd-shim processes..."
    echo "${SHIM_PIDS}" | xargs -r kill -9 2>/dev/null || true
    sleep 1
fi

# 1c. 阻止 systemd 自动重启 dockerd（避免 race）
log "  Disabling systemd auto-restart..."
timeout 5 systemctl reset-failed docker.service docker.socket 2>/dev/null || log "  (reset-failed timed out, ignoring)"
timeout 5 systemctl stop docker.socket 2>/dev/null || log "  (stop docker.socket timed out, ignoring)"

# ── Step 2: 清理残留文件 ──────────────────────────────────────────────
log "Step 2: Clean stale files"
rm -f /var/run/docker.pid /var/run/docker.sock
log "  Removed docker.pid and docker.sock"

# ── Step 3: 删除旧容器状态 ────────────────────────────────────────────
log "Step 3: Remove stale container state"
if [ -d "${DATA_ROOT}/containers" ]; then
    N=$(ls "${DATA_ROOT}/containers" 2>/dev/null | wc -l)
    log "  Found ${N} stale containers, removing..."
    rm -rf "${DATA_ROOT}/containers"/* 2>/dev/null || true
fi
rm -f "${DATA_ROOT}/network/files/local-kv.db" 2>/dev/null
log "  Done"

# ── Step 4: 确保 containerd 在跑 ──────────────────────────────────────
log "Step 4: Ensure containerd"
if pgrep -x containerd >/dev/null; then
    log "  containerd PID: $(pgrep -x containerd)"
else
    log "  containerd not running, starting via systemd..."
    timeout 10 systemctl start containerd 2>/dev/null || log "  (start containerd timed out)"
    sleep 3
fi

# ── Step 5: 启动 dockerd（直接 nohup 后台跑，不走 systemd）────────────
log "Step 5: Start dockerd (direct nohup, NOT via systemd)"
if [ -f "${PROXY_ENV_FILE}" ]; then
    # shellcheck disable=SC1090
    set -a; . "${PROXY_ENV_FILE}"; set +a
    log "  Loaded proxy env from ${PROXY_ENV_FILE}"
fi
LOG_FILE="/tmp/dockerd_start_$(date +%H%M%S).log"
nohup dockerd --containerd=/run/containerd/containerd.sock > "${LOG_FILE}" 2>&1 &
DOCKERD_PID=$!
disown 2>/dev/null || true
log "  dockerd PID: ${DOCKERD_PID}, log: ${LOG_FILE}"

# ── Step 6: 等待 API ready ────────────────────────────────────────────
log "Step 6: Waiting for docker API (up to 5 min)..."
READY=0
for i in $(seq 1 60); do
    if timeout 5 docker info >/dev/null 2>&1; then
        log "  ✅ Docker ready at attempt ${i} (~$((i*5))s)"
        READY=1
        break
    fi
    if ! kill -0 ${DOCKERD_PID} 2>/dev/null; then
        log "  ❌ dockerd died! Last log:"
        tail -20 "${LOG_FILE}"
        exit 1
    fi
    if [ $((i % 6)) -eq 0 ]; then
        log "    still waiting... ($((i*5))s elapsed)"
        tail -1 "${LOG_FILE}" 2>/dev/null | sed 's/^/      /'
    fi
    sleep 5
done

if [ "${READY}" != "1" ]; then
    log "  ❌ Timeout. dockerd is still running but API not ready."
    log "  Check: tail -f ${LOG_FILE}"
    exit 1
fi

# ── Done ──────────────────────────────────────────────────────────────
log "✅ Done. Docker is ready."
docker info 2>&1 | grep -E "Containers:|Running:|Stopped:|Images:|Server Version"
echo
df -h "${DATA_ROOT}" 2>&1 | grep -F "${DATA_ROOT}" || true
echo
log "Next: bash terminal-rl/remote/run_pool_server_pu_v2.sh"
