#!/usr/bin/env bash
# diag_docker_failures_lite.sh — 轻量版诊断脚本
# 与训练并行跑安全（不做 build 探针，不调 du 长扫描）。
# 只做 3 件事：
#   1) docker daemon 状态快照 (5s 内)
#   2) GPU 训练日志里 exit-17 task 黑名单提取
#   3) pool_server /healthz + /status 当前 lease

set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
LOG_DIR="${REPO_ROOT}/tmp_doc_latest"
mkdir -p "${LOG_DIR}"

GPU_LOG="${LOG_DIR}/gpu_run.log"
SUMMARY="${LOG_DIR}/cpu_diag_summary.txt"
POOL_HOST="${POOL_HOST:-127.0.0.1}"
POOL_PORT="${POOL_PORT:-18081}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"

echo "[$(date '+%F %T')] diag_lite starting on $(hostname)" | tee "${SUMMARY}"

# ───────────────────────────────────────────────────────────
# 1. 简版 docker state（不调 du）
# ───────────────────────────────────────────────────────────
{
  echo "=== docker info (5s timeout) ==="
  timeout 5 docker info 2>&1 | grep -E "Containers:|Running:|Paused:|Stopped:|Images:|Server Version" | head -10
  echo
  echo "=== ${DOCKER_DATA_ROOT} ==="
  df -h "${DOCKER_DATA_ROOT}" 2>&1
  echo
  echo "=== currently running container count ==="
  timeout 5 docker ps -q 2>&1 | wc -l
  echo
  echo "=== top 5 oldest running containers (potential stuck) ==="
  timeout 10 docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}' 2>&1 | head -10
} > "${LOG_DIR}/cpu_diag_dockerd_state.txt"
echo "[1/3] docker state -> cpu_diag_dockerd_state.txt" | tee -a "${SUMMARY}"

# ───────────────────────────────────────────────────────────
# 2. exit-17 黑名单（轻量 grep + sort + uniq）
# ───────────────────────────────────────────────────────────
{
  echo "=== exit-17 task ranking (from gpu_run.log) ==="
  if [ -f "${GPU_LOG}" ]; then
    echo "source: ${GPU_LOG} ($(wc -l < "${GPU_LOG}") lines, $(du -h "${GPU_LOG}" | awk '{print $1}'))"
    echo "total exit-17 hits: $(grep -c 'exit status 17' "${GPU_LOG}" 2>/dev/null || echo 0)"
    echo "total /reset 500: $(grep -c '500 Internal Server Error.*reset' "${GPU_LOG}" 2>/dev/null || echo 0)"
    echo
    echo "rank  count  task_id"
    echo "----  -----  -------"
    grep 'exit status 17' "${GPU_LOG}" 2>/dev/null \
      | grep -oE 'seta_env/[0-9]+' \
      | sort | uniq -c | sort -rn \
      | head -30 \
      | awk '{printf "%-4d  %-5d  %s\n", NR, $1, $2}'
    echo
    echo "=== unique task_id list (for blacklist) ==="
    grep 'exit status 17' "${GPU_LOG}" 2>/dev/null \
      | grep -oE 'seta_env/[0-9]+' \
      | sort -u
    echo
    echo "=== one-liner blacklist (paste into env: SETA_BLACKLIST_TASKS=...) ==="
    grep 'exit status 17' "${GPU_LOG}" 2>/dev/null \
      | grep -oE 'seta_env/[0-9]+' \
      | grep -oE '[0-9]+$' \
      | sort -un \
      | tr '\n' ',' \
      | sed 's/,$//'
    echo
  else
    echo "WARN: ${GPU_LOG} not found"
  fi
} > "${LOG_DIR}/cpu_diag_exit17_tasks.txt"
echo "[2/3] exit-17 ranking -> cpu_diag_exit17_tasks.txt" | tee -a "${SUMMARY}"

# ───────────────────────────────────────────────────────────
# 3. pool_server /healthz + /status
# ───────────────────────────────────────────────────────────
{
  echo "=== pool_server processes ==="
  pgrep -a -f "remote.pool_server" 2>&1
  echo
  echo "=== /healthz ==="
  curl -fsS --max-time 5 "http://${POOL_HOST}:${POOL_PORT}/healthz" 2>&1
  echo
  echo "=== /status (active leases) ==="
  curl -fsS --max-time 5 "http://${POOL_HOST}:${POOL_PORT}/status" 2>&1 | python3 -m json.tool 2>&1 | head -50
} > "${LOG_DIR}/cpu_diag_pool_health.txt"
echo "[3/3] pool_server -> cpu_diag_pool_health.txt" | tee -a "${SUMMARY}"

# ───────────────────────────────────────────────────────────
# Summary
# ───────────────────────────────────────────────────────────
{
  echo
  echo "===================== SUMMARY ====================="
  echo "Generated: $(date '+%F %T')"
  echo
  echo "--- exit-17 top-10 (blacklist candidates) ---"
  grep -A 30 "rank  count  task_id" "${LOG_DIR}/cpu_diag_exit17_tasks.txt" \
    | head -12
  echo
  echo "--- one-liner blacklist ---"
  grep -A 1 "one-liner blacklist" "${LOG_DIR}/cpu_diag_exit17_tasks.txt" \
    | tail -1
  echo
  echo "--- docker state ---"
  grep -E "Containers:|Running:|${DOCKER_DATA_ROOT}" "${LOG_DIR}/cpu_diag_dockerd_state.txt" | head -5
  echo
  echo "--- pool_server ---"
  grep -E '"ok"|18081' "${LOG_DIR}/cpu_diag_pool_health.txt" | head -3
  echo "==================================================="
} >> "${SUMMARY}"

echo
echo "DONE. Summary: ${SUMMARY}"
