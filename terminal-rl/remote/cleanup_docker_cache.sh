#!/usr/bin/env bash
# cleanup_docker_cache.sh — 清理 docker build cache + 停止的容器 + 悬空 image
# 用法: bash terminal-rl/remote/cleanup_docker_cache.sh
# 安全: 不会删除正在运行的容器，不会删除有 tag 的 image

set -uo pipefail
echo "[$(date '+%F %T')] Docker cleanup starting on $(hostname)"
echo

# ── 1. 停止的容器清理 ──────────────────────────────────────────────────
echo "=== Step 1: Remove stopped containers ==="
STOPPED=$(docker ps -aq --filter "status=exited" --filter "status=dead" --filter "status=created" 2>/dev/null | wc -l)
echo "  Stopped containers: ${STOPPED}"
if [ "${STOPPED}" -gt 0 ]; then
    docker container prune -f
    echo "  Done."
else
    echo "  Nothing to remove."
fi
echo

# ── 2. Build cache 清理 ────────────────────────────────────────────────
echo "=== Step 2: Clear build cache ==="
if docker buildx version >/dev/null 2>&1; then
    echo "  Using: docker buildx prune -af"
    docker buildx prune -af 2>&1 | tail -5
else
    echo "  buildx not available, using: docker image prune (dangling layers)"
    docker image prune -f 2>&1 | tail -5
    echo "  Also removing all untagged images..."
    docker images --filter "dangling=true" -q 2>/dev/null | xargs -r docker rmi -f 2>&1 | tail -5
fi
echo

# ── 3. 悬空 volume 清理 ────────────────────────────────────────────────
echo "=== Step 3: Remove dangling volumes ==="
docker volume prune -f 2>&1 | tail -3
echo

# ── 4. 未使用 network 清理 ─────────────────────────────────────────────
echo "=== Step 4: Remove unused networks ==="
docker network prune -f 2>&1 | tail -3
echo

# ── 5. 最终状态 ───────────────────────────────────────────────────────
echo "=== Final state ==="
docker system df 2>&1
echo
docker info 2>&1 | grep -E "Containers:|Running:|Stopped:|Images:"
echo
df -h /data 2>&1
echo
echo "[$(date '+%F %T')] Cleanup done."
