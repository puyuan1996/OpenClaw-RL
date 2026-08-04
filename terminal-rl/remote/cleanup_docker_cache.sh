#!/usr/bin/env bash
# cleanup_docker_cache.sh — 清理 docker build cache + 停止的容器 + 悬空 image
# 用法: bash terminal-rl/remote/cleanup_docker_cache.sh
# 安全: 不会删除正在运行的容器，不会删除有 tag 的 image

set -uo pipefail
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"
DOCKER_CMD_TIMEOUT="${DOCKER_CMD_TIMEOUT:-30}"
DOCKER_PRUNE_TIMEOUT="${DOCKER_PRUNE_TIMEOUT:-120}"
DOCKER_NETWORK_LIFECYCLE_LOCK="${DOCKER_NETWORK_LIFECYCLE_LOCK:-/tmp/openclaw_docker_network_lifecycle.lock}"
RUN_HEAVY_DF="${RUN_HEAVY_DF:-0}"
echo "[$(date '+%F %T')] Docker cleanup starting on $(hostname)"
echo "Docker data root: ${DOCKER_DATA_ROOT}"
echo

if ! timeout 10 docker info >/dev/null 2>&1; then
    echo "[ERROR] docker daemon is not responding. Run:"
    echo "        sudo bash terminal-rl/remote/fix_dockerd_and_proxy.sh"
    exit 1
fi

# ── 1. 停止的容器清理 ──────────────────────────────────────────────────
echo "=== Step 1: Remove stopped containers ==="
STOPPED=$(timeout "${DOCKER_CMD_TIMEOUT}" docker ps -aq --filter "status=exited" --filter "status=dead" --filter "status=created" 2>/dev/null | wc -l)
echo "  Stopped containers: ${STOPPED}"
if [ "${STOPPED}" -gt 0 ]; then
    timeout "${DOCKER_PRUNE_TIMEOUT}" docker container prune -f
    echo "  Done."
else
    echo "  Nothing to remove."
fi
echo

# ── 2. Build cache 清理 ────────────────────────────────────────────────
echo "=== Step 2: Clear build cache ==="
if timeout "${DOCKER_CMD_TIMEOUT}" docker buildx version >/dev/null 2>&1; then
    echo "  Using: docker buildx prune -af"
    timeout "${DOCKER_PRUNE_TIMEOUT}" docker buildx prune -af 2>&1 | tail -5
else
    echo "  buildx not available, using: docker image prune (dangling layers)"
    timeout "${DOCKER_PRUNE_TIMEOUT}" docker image prune -f 2>&1 | tail -5
    echo "  Also removing all untagged images..."
    timeout "${DOCKER_CMD_TIMEOUT}" docker images --filter "dangling=true" -q 2>/dev/null \
      | xargs -r timeout "${DOCKER_PRUNE_TIMEOUT}" docker rmi -f 2>&1 | tail -5
fi
echo

# ── 3. 悬空 volume 清理 ────────────────────────────────────────────────
echo "=== Step 3: Remove dangling volumes ==="
timeout "${DOCKER_PRUNE_TIMEOUT}" docker volume prune -f 2>&1 | tail -3
echo

# ── 4. 未使用 network 清理 ─────────────────────────────────────────────
echo "=== Step 4: Remove unused networks ==="
if command -v flock >/dev/null 2>&1; then
    timeout "${DOCKER_PRUNE_TIMEOUT}" flock -w "${DOCKER_PRUNE_TIMEOUT}" \
        "${DOCKER_NETWORK_LIFECYCLE_LOCK}" \
        docker network prune -f 2>&1 | tail -3
else
    echo "  WARN: flock is unavailable; skipping unsafe docker network prune"
fi
echo

# ── 5. 最终状态 ───────────────────────────────────────────────────────
echo "=== Final state ==="
if [ "${RUN_HEAVY_DF}" = "1" ]; then
    timeout "${DOCKER_CMD_TIMEOUT}" docker system df 2>&1 || echo "docker system df timed out/failed; skipped"
else
    echo "Skipping docker system df. Set RUN_HEAVY_DF=1 to enable it."
fi
echo
timeout "${DOCKER_CMD_TIMEOUT}" docker info 2>&1 | grep -E "Containers:|Running:|Stopped:|Images:" || true
echo
df -h "${DOCKER_DATA_ROOT}" 2>&1 || true
echo
echo "[$(date '+%F %T')] Cleanup done."
