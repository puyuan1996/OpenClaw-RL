#!/usr/bin/env bash
# diag_docker_failures.sh — 在 CPU worker (share-machine) 上诊断 docker compose build
# 失败的根因 + 提取 task 黑名单。所有产物写到 tmp_doc_latest/cpu_diag_*，方便跨机分析。
#
# 用法（在 CPU worker 上）：
#   bash terminal-rl/remote/diag_docker_failures.sh
#
# 输出文件（全部在 ${REPO_ROOT}/tmp_doc_latest/）：
#   cpu_diag_summary.txt        — 一页总览
#   cpu_diag_dockerd_state.txt  — daemon/socket/磁盘/proxy
#   cpu_diag_exit17_tasks.txt   — exit-17 高频 task_id 排行（黑名单候选）
#   cpu_diag_build_probe.txt    — 对 top-3 高频 task 真实跑一次 build 看错误
#   cpu_diag_pool_health.txt    — pool_server 状态 + 当前 active leases
#   cpu_diag_journal_dockerd.txt — 最近 200 行 dockerd journal

set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
LOG_DIR="${REPO_ROOT}/tmp_doc_latest"
mkdir -p "${LOG_DIR}"

GPU_LOG="${LOG_DIR}/gpu_run.log"
SUMMARY="${LOG_DIR}/cpu_diag_summary.txt"

echo "[$(date '+%F %T')] diag_docker_failures starting on $(hostname)" | tee "${SUMMARY}"
echo "log dir: ${LOG_DIR}" | tee -a "${SUMMARY}"
echo

# ───────────────────────────────────────────────────────────
# 1. dockerd / socket / 磁盘 / proxy 配置
# ───────────────────────────────────────────────────────────
{
  echo "=== systemctl status docker (head 25) ==="
  systemctl status docker.service --no-pager -l 2>&1 | head -25
  echo
  echo "=== docker info (5s timeout) ==="
  timeout 5 docker info 2>&1 | head -40
  echo
  echo "=== docker system df ==="
  timeout 5 docker system df 2>&1
  echo
  echo "=== /data disk (real data-root) ==="
  df -h /data 2>&1
  echo
  echo "=== /data top usage ==="
  du -sh /data/overlay2 2>/dev/null
  echo
  echo "=== inode usage ==="
  df -i /data 2>&1
  echo
  echo "=== /var/run/docker.sock ==="
  ls -la /var/run/docker.sock /run/docker.sock 2>&1
  echo
  echo "=== drop-in proxy.conf ==="
  cat /etc/systemd/system/docker.service.d/proxy.conf 2>&1
  echo
  echo "=== /etc/docker/daemon.json ==="
  cat /etc/docker/daemon.json 2>&1
  echo
  echo "=== dockerd process ==="
  pgrep -a dockerd 2>&1
  echo
  echo "=== orphan containerd-shim count ==="
  pgrep -c containerd-shim-runc-v2 2>&1
} > "${LOG_DIR}/cpu_diag_dockerd_state.txt" 2>&1

echo "[1/5] dockerd state -> cpu_diag_dockerd_state.txt" | tee -a "${SUMMARY}"

# ───────────────────────────────────────────────────────────
# 2. dockerd journal 最近 200 行（看是否有重新触发的失败）
# ───────────────────────────────────────────────────────────
journalctl -u docker.service -n 200 --no-pager 2>&1 \
  > "${LOG_DIR}/cpu_diag_journal_dockerd.txt"
echo "[2/5] dockerd journal -> cpu_diag_journal_dockerd.txt" | tee -a "${SUMMARY}"

# ───────────────────────────────────────────────────────────
# 3. exit-17 task 黑名单提取（来自 GPU 训练日志）
# ───────────────────────────────────────────────────────────
{
  echo "=== exit-17 task_id 排行 (来自 gpu_run.log) ==="
  if [ -f "${GPU_LOG}" ]; then
    echo "source: ${GPU_LOG}"
    echo "total exit-17 hits: $(grep -c 'exit status 17' "${GPU_LOG}")"
    echo
    echo "rank  count  task_id"
    echo "----  -----  -------"
    grep 'exit status 17' "${GPU_LOG}" \
      | grep -oE 'seta_env/[0-9]+' \
      | sort | uniq -c | sort -rn \
      | head -30 \
      | awk '{printf "%-4d  %-5d  %s\n", NR, $1, $2}'
    echo
    echo "=== 500 reset 失败总数 ==="
    grep -c '500 Internal Server Error.*reset' "${GPU_LOG}"
    echo
    echo "=== 受影响 task_id 完整列表（去重） ==="
    grep 'exit status 17' "${GPU_LOG}" \
      | grep -oE 'seta_env/[0-9]+' \
      | sort -u
  else
    echo "WARN: ${GPU_LOG} 不存在（可能 GPU 端 symlink 漂移）"
    echo "尝试找最新的 gpu_run.log:"
    find "${REPO_ROOT}" -maxdepth 3 -name 'gpu_run.log' -mtime -1 -ls 2>/dev/null | head -5
  fi
} > "${LOG_DIR}/cpu_diag_exit17_tasks.txt" 2>&1

echo "[3/5] exit-17 task ranking -> cpu_diag_exit17_tasks.txt" | tee -a "${SUMMARY}"

# ───────────────────────────────────────────────────────────
# 4. 对 top-3 高频 task 真实跑一次 build 看实际错误
# ───────────────────────────────────────────────────────────
{
  echo "=== 对 top-3 exit-17 task 跑实际 build 拿真错误 ==="
  if [ ! -f "${GPU_LOG}" ]; then
    echo "skip: 没找到 gpu_run.log"
  else
    TOP_TASKS=$(grep 'exit status 17' "${GPU_LOG}" \
      | grep -oE 'seta_env/[0-9]+' \
      | sort | uniq -c | sort -rn \
      | head -3 | awk '{print $2}')

    if [ -z "${TOP_TASKS}" ]; then
      echo "no exit-17 tasks found in gpu_run.log"
    else
      for tk in ${TOP_TASKS}; do
        TASK_DIR="${REPO_ROOT}/terminal-rl/dataset/${tk}"
        echo
        echo "######## task=${tk} ########"
        if [ ! -f "${TASK_DIR}/docker-compose.yaml" ]; then
          echo "ERR: ${TASK_DIR}/docker-compose.yaml 不存在, skip"
          continue
        fi
        echo "--- docker-compose.yaml head ---"
        head -30 "${TASK_DIR}/docker-compose.yaml"
        echo "--- Dockerfile head (if any) ---"
        find "${TASK_DIR}" -maxdepth 2 -name 'Dockerfile*' -exec head -20 {} \; 2>/dev/null | head -40
        echo "--- timeout 180 docker compose build ---"
        cd "${TASK_DIR}" || continue
        timeout 180 docker compose -p "diag_$(echo "${tk}" | tr '/' '_')" \
          -f docker-compose.yaml build 2>&1 | tail -50
        echo "--- exit: $? ---"
        # 清理探针
        timeout 30 docker compose -p "diag_$(echo "${tk}" | tr '/' '_')" \
          -f docker-compose.yaml down --rmi local --remove-orphans 2>&1 | tail -5
      done
    fi
  fi
} > "${LOG_DIR}/cpu_diag_build_probe.txt" 2>&1

echo "[4/5] build probe (top-3 tasks) -> cpu_diag_build_probe.txt" | tee -a "${SUMMARY}"

# ───────────────────────────────────────────────────────────
# 5. pool_server 健康 + 当前 lease 状态
# ───────────────────────────────────────────────────────────
{
  echo "=== pool_server processes ==="
  pgrep -a -f "remote.pool_server" 2>&1
  echo
  echo "=== port 18081 listener ==="
  ss -tlnp 2>&1 | grep 18081 || netstat -tlnp 2>&1 | grep 18081
  echo
  echo "=== /healthz ==="
  curl -fsS --max-time 5 http://127.0.0.1:18081/healthz 2>&1
  echo
  echo "=== /status (active leases) ==="
  curl -fsS --max-time 5 http://127.0.0.1:18081/status 2>&1 | python3 -m json.tool 2>&1 | head -50
  echo
  echo "=== last 30 lines of cpu_pool.log ==="
  tail -30 "${LOG_DIR}/cpu_pool.log" 2>&1
} > "${LOG_DIR}/cpu_diag_pool_health.txt" 2>&1

echo "[5/5] pool_server health -> cpu_diag_pool_health.txt" | tee -a "${SUMMARY}"

# ───────────────────────────────────────────────────────────
# 总览
# ───────────────────────────────────────────────────────────
{
  echo
  echo "===================== SUMMARY ====================="
  echo "Generated at: $(date '+%F %T')"
  echo
  echo "--- exit-17 top-10 tasks (blacklist candidates) ---"
  grep -A 50 "rank  count  task_id" "${LOG_DIR}/cpu_diag_exit17_tasks.txt" \
    | head -12
  echo
  echo "--- dockerd state (one-liner) ---"
  grep -E "Active:|Containers:|Images:|exit:" "${LOG_DIR}/cpu_diag_dockerd_state.txt" | head -5
  echo
  echo "--- /data disk ---"
  grep -E "/data" "${LOG_DIR}/cpu_diag_dockerd_state.txt" | head -3
  echo
  echo "--- pool_server ---"
  grep -E '"ok"|18081' "${LOG_DIR}/cpu_diag_pool_health.txt" | head -3
  echo
  echo "==================================================="
} >> "${SUMMARY}"

echo
echo "DONE. Summary: ${SUMMARY}"
echo "All artifacts under: ${LOG_DIR}/cpu_diag_*"
