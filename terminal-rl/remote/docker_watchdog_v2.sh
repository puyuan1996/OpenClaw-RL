#!/usr/bin/env bash
# docker_watchdog_v2.sh — 加固版守护进程（针对 DinD/DooD + agentic-RL 场景）
#
# 设计目标：在 14 h 长跑中拦截 issue #3 §3/§4/§5 描述的环境层崩溃
#   §3 docker bridge address-pool 耗尽
#   §4 nofile=1024 触发 "Too many open files"
#   §5 长跑后 dockerd 状态污染（孤儿容器/网络残留）
#
# 修复了 v2 早期版本的 9 个问题（详见 ../runs/.../analysis/REPORT.md → docker_watchdog 修复方案）：
#   P0-1: 删除 restart_docker 中的 systemctl restart 调用（D state 时会无限挂起）
#   P0-2: pkill 前先 stop docker.socket，阻断 systemd auto-restart race
#   P0-3: emergency_pressure_relief 加 60 s 冷却 + foreground+timeout，避免并发拖死 dockerd
#   P1-1: 新增 pool_server /healthz + /status + bridge 网络数 监控（这次崩溃的真正信号）
#   P1-2: cgroup v2 detection（统一 hierarchy 路径解析）
#   P1-3: 检测 host pid namespace；不在 host ns 时不擦容器 state
#   P1-4: 周期性深度探活（network create/rm 模拟 pool 真实路径）
#   P1-5: 日志 rotate 改为 truncate-in-place，nohup fd 不丢
#   P1-6: enforce_container_limit 排除 pool_server 容器（杀候选改用 task 容器 pattern）
#   P1-7: 启动时打印 namespace 信息便于诊断
#
# 用法（推荐 systemd 起，见 docker-watchdog.service）：
#   systemctl enable --now docker-watchdog
# 或临时：
#   nohup bash docker_watchdog_v2.sh > /tmp/docker_watchdog.log 2>&1 &

set -uo pipefail

# ── 可调参数 ──────────────────────────────────────────────────────────
MAX_RUNNING_CONTAINERS="${MAX_RUNNING_CONTAINERS:-80}"
HARD_KILL_THRESHOLD="${HARD_KILL_THRESHOLD:-120}"
CLEANUP_INTERVAL="${CLEANUP_INTERVAL:-60}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
CGROUP_MONITOR_INTERVAL="${CGROUP_MONITOR_INTERVAL:-15}"
POOL_CHECK_INTERVAL="${POOL_CHECK_INTERVAL:-30}"
DEEP_PROBE_INTERVAL="${DEEP_PROBE_INTERVAL:-300}"
PIDS_WARN_PCT="${PIDS_WARN_PCT:-75}"
PIDS_EMERGENCY_PCT="${PIDS_EMERGENCY_PCT:-90}"
MEM_WARN_PCT="${MEM_WARN_PCT:-80}"
MEM_EMERGENCY_PCT="${MEM_EMERGENCY_PCT:-92}"
MAX_CONSECUTIVE_HEALTH_FAILS="${MAX_CONSECUTIVE_HEALTH_FAILS:-3}"
LOG_FILE="${LOG_FILE:-/tmp/docker_watchdog.log}"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-209715200}"            # 200 MiB
DOCKER_SOCK="${DOCKER_SOCK:-/var/run/docker.sock}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/data}"

POOL_HOST="${POOL_HOST:-127.0.0.1}"
POOL_PORT="${POOL_PORT:-18081}"
POOL_PENDING_CLOSES_WARN="${POOL_PENDING_CLOSES_WARN:-50}"
BRIDGE_NETS_WARN="${BRIDGE_NETS_WARN:-200}"
EMERGENCY_COOLDOWN_S="${EMERGENCY_COOLDOWN_S:-60}"
POOL_SERVER_NAME_REGEX="${POOL_SERVER_NAME_REGEX:-openclaw_pool_server}"
TASK_CONTAINER_REGEX="${TASK_CONTAINER_REGEX:-^[0-9]+-.*[-_](client|helper)([-_][0-9]+)?$}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-600}"  # "I'm alive" line every 10 min

LOG_PREFIX="[docker-watchdog]"

# ── 自身防 OOM ────────────────────────────────────────────────────────
echo -900 > /proc/self/oom_score_adj 2>/dev/null || true

# ── 状态 ──────────────────────────────────────────────────────────────
LAST_EMERGENCY_TS=0
LAST_DEEP_PROBE_TS=0

# ── namespace 检测 ────────────────────────────────────────────────────
HOST_PID_NS=0
detect_pid_namespace() {
    # 与 PID 1 共享 mnt namespace 的进程基本就是 host
    local self_pid_ns host_pid_ns
    self_pid_ns=$(readlink /proc/self/ns/pid 2>/dev/null || echo "?")
    host_pid_ns=$(readlink /proc/1/ns/pid 2>/dev/null || echo "?")
    if [ -n "$self_pid_ns" ] && [ "$self_pid_ns" = "$host_pid_ns" ] && [ "$self_pid_ns" != "?" ]; then
        HOST_PID_NS=1
    else
        HOST_PID_NS=0
    fi
}

# ── 工具函数 ──────────────────────────────────────────────────────────
# 用 truncate-in-place 而不是 mv，否则 nohup 重定向的 fd 会丢
rotate_log_if_big() {
    [ -f "${LOG_FILE}" ] || return 0
    local sz
    sz=$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo 0)
    [ "${sz}" -gt "${LOG_MAX_BYTES}" ] || return 0
    local tail_bytes=52428800   # 保留尾部 50 MB
    local tmp
    tmp=$(tail -c "$tail_bytes" "${LOG_FILE}" 2>/dev/null)
    : > "${LOG_FILE}"
    printf '%s\n' "$tmp" >> "${LOG_FILE}" 2>/dev/null || true
}

log() {
    echo "$(date '+%F %T') ${LOG_PREFIX} $*"
    rotate_log_if_big
}

docker_alive() {
    timeout 3 curl -fsS --max-time 2 \
        --unix-socket "${DOCKER_SOCK}" \
        http://./_ping >/dev/null 2>&1
}

# 深度探活：模拟 pool_server 真实 reset 路径——能创建+删 bridge 网络
docker_deep_alive() {
    local netname="wd_probe_$(date +%s)_$$"
    if ! timeout 10 docker network create --driver bridge "$netname" >/dev/null 2>&1; then
        return 1
    fi
    timeout 5 docker network rm "$netname" >/dev/null 2>&1 || true
    return 0
}

# ── 紧急泄压（带冷却 + foreground + timeout）─────────────────────────
emergency_pressure_relief() {
    local reason="$1"
    local now
    now=$(date +%s)
    if [ $((now - LAST_EMERGENCY_TS)) -lt "${EMERGENCY_COOLDOWN_S}" ]; then
        log "EMERGENCY suppressed (cooldown ${EMERGENCY_COOLDOWN_S}s active): ${reason}"
        return
    fi
    LAST_EMERGENCY_TS="$now"
    log "EMERGENCY: ${reason} — kill task containers + prune"

    # 杀最旧的 30 个 task 容器（按 pattern，绝不动 pool_server）
    local victims
    victims=$(docker ps --format '{{.ID}} {{.Names}}' 2>/dev/null \
        | grep -vE "${POOL_SERVER_NAME_REGEX}" \
        | grep -E "${TASK_CONTAINER_REGEX}" \
        | tail -n 30 \
        | awk '{print $1}')
    if [ -n "${victims}" ]; then
        local n
        n=$(echo "${victims}" | wc -l)
        echo "${victims}" | xargs -r -n 10 timeout 30 docker kill >/dev/null 2>&1 || true
        log "EMERGENCY: killed ~${n} task containers"
    else
        log "EMERGENCY: no task containers matched pattern (regex=${TASK_CONTAINER_REGEX})"
    fi

    # 清理 stopped + dangling network（foreground，防并发拖死 dockerd）
    timeout 30 docker container prune -f >/dev/null 2>&1 || true
    timeout 30 docker network prune -f >/dev/null 2>&1 || true
}

# ── cgroup 检测（v1 + v2）────────────────────────────────────────────
CGROUP_VERSION=""
CGROUP_PIDS_DIR=""
CGROUP_MEM_DIR=""
CGROUP_PIDS_MAX_VAL=""
CGROUP_MEM_MAX_VAL=""
CGROUP_PIDS_CUR_FILE=""
CGROUP_MEM_CUR_FILE=""

# 沿 cgroup 路径从深到浅扫描，找到"最严有限限制"所在的目录
# 参数: $1=控制器挂载点  $2=cgroup 相对路径  $3=current 文件名  $4=max 文件名
find_tightest_limit() {
    local mount="$1" rel="$2" cur_name="$3" max_name="$4"
    local best_dir="" best_val=""
    local path="$rel"
    while [ -n "$path" ] && [ "$path" != "/" ]; do
        local full="${mount}${path}"
        if [ -f "${full}/${max_name}" ]; then
            local v
            v=$(cat "${full}/${max_name}" 2>/dev/null)
            if [ -n "$v" ] && [ "$v" != "max" ] \
               && [ "$v" -gt 0 ] 2>/dev/null \
               && [ "$v" -lt 9000000000000000000 ] 2>/dev/null; then
                if [ -z "$best_val" ] || [ "$v" -lt "$best_val" ] 2>/dev/null; then
                    best_val="$v"
                    best_dir="$full"
                fi
            fi
        fi
        path="${path%/*}"
    done
    [ -n "$best_dir" ] && echo "${best_dir}|${best_val}"
}

detect_cgroup_v2() {
    [ -f /sys/fs/cgroup/cgroup.controllers ] || return 1
    local rel
    rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
    [ -n "$rel" ] || return 1
    # v2: pids.max / memory.max 在同一 unified hierarchy
    local r
    r=$(find_tightest_limit /sys/fs/cgroup "$rel" pids.current pids.max)
    if [ -n "$r" ]; then
        CGROUP_PIDS_DIR="${r%|*}"
        CGROUP_PIDS_MAX_VAL="${r#*|}"
        CGROUP_PIDS_CUR_FILE="${CGROUP_PIDS_DIR}/pids.current"
    fi
    r=$(find_tightest_limit /sys/fs/cgroup "$rel" memory.current memory.max)
    if [ -n "$r" ]; then
        CGROUP_MEM_DIR="${r%|*}"
        CGROUP_MEM_MAX_VAL="${r#*|}"
        CGROUP_MEM_CUR_FILE="${CGROUP_MEM_DIR}/memory.current"
    fi
    if [ -n "$CGROUP_PIDS_DIR" ] || [ -n "$CGROUP_MEM_DIR" ]; then
        CGROUP_VERSION="v2"
        return 0
    fi
    return 1
}

detect_cgroup_v1() {
    local pids_rel mem_rel
    pids_rel=$(awk -F: '$2 ~ /(^|,)pids(,|$)/  {print $3}'   /proc/self/cgroup 2>/dev/null | head -1)
    mem_rel=$(awk  -F: '$2 ~ /(^|,)memory(,|$)/{print $3}'   /proc/self/cgroup 2>/dev/null | head -1)
    if [ -d /sys/fs/cgroup/pids ] && [ -n "$pids_rel" ]; then
        local r
        r=$(find_tightest_limit /sys/fs/cgroup/pids "$pids_rel" pids.current pids.max)
        if [ -n "$r" ]; then
            CGROUP_PIDS_DIR="${r%|*}"
            CGROUP_PIDS_MAX_VAL="${r#*|}"
            CGROUP_PIDS_CUR_FILE="${CGROUP_PIDS_DIR}/pids.current"
        fi
    fi
    if [ -d /sys/fs/cgroup/memory ] && [ -n "$mem_rel" ]; then
        local r
        r=$(find_tightest_limit /sys/fs/cgroup/memory "$mem_rel" memory.usage_in_bytes memory.limit_in_bytes)
        if [ -n "$r" ]; then
            CGROUP_MEM_DIR="${r%|*}"
            CGROUP_MEM_MAX_VAL="${r#*|}"
            CGROUP_MEM_CUR_FILE="${CGROUP_MEM_DIR}/memory.usage_in_bytes"
        fi
    fi
    if [ -n "$CGROUP_PIDS_DIR" ] || [ -n "$CGROUP_MEM_DIR" ]; then
        CGROUP_VERSION="v1"
        return 0
    fi
    return 1
}

detect_cgroup() {
    detect_cgroup_v2 || detect_cgroup_v1 || return 1
}

monitor_pod_cgroup() {
    [ -z "$CGROUP_VERSION" ] && return 0

    if [ -n "$CGROUP_PIDS_DIR" ] && [ -f "$CGROUP_PIDS_CUR_FILE" ]; then
        local cur
        read -r cur < "$CGROUP_PIDS_CUR_FILE" 2>/dev/null
        if [ -n "$cur" ] && [ "$cur" -ge 0 ] 2>/dev/null; then
            local pct=$(( cur * 100 / CGROUP_PIDS_MAX_VAL ))
            if [ "$pct" -ge "$PIDS_EMERGENCY_PCT" ]; then
                emergency_pressure_relief "PIDs ${cur}/${CGROUP_PIDS_MAX_VAL} (${pct}%)"
            elif [ "$pct" -ge "$PIDS_WARN_PCT" ]; then
                log "WARN: PIDs ${cur}/${CGROUP_PIDS_MAX_VAL} (${pct}%) — aggressive cleanup"
                timeout 20 docker container prune -f --filter "until=30s" >/dev/null 2>&1 || true
            fi
        fi
    fi

    if [ -n "$CGROUP_MEM_DIR" ] && [ -f "$CGROUP_MEM_CUR_FILE" ]; then
        local cur
        read -r cur < "$CGROUP_MEM_CUR_FILE" 2>/dev/null
        if [ -n "$cur" ] && [ "$cur" -ge 0 ] 2>/dev/null; then
            local pct=$(( cur * 100 / CGROUP_MEM_MAX_VAL ))
            if [ "$pct" -ge "$MEM_EMERGENCY_PCT" ]; then
                emergency_pressure_relief "Memory ${cur}/${CGROUP_MEM_MAX_VAL} (${pct}%)"
            elif [ "$pct" -ge "$MEM_WARN_PCT" ]; then
                log "WARN: Memory ${cur}/${CGROUP_MEM_MAX_VAL} (${pct}%) — pruning"
                timeout 20 docker container prune -f --filter "until=30s" >/dev/null 2>&1 || true
            fi
        fi
    fi
}

# ── pool_server 监控（核心修复：捕捉 dockerd OK 但 /reset 500 的故障形态）──
# 副作用：每次成功调用会更新 LAST_POOL_ACTIVE / LAST_POOL_PENDING / LAST_BRIDGE_NETS
# 供 heartbeat 复用，不重新发起 HTTP/docker 调用。
LAST_POOL_ACTIVE="?"
LAST_POOL_PENDING="?"
LAST_BRIDGE_NETS="?"
check_pool_server() {
    timeout 3 curl -fsS --noproxy '*' "http://${POOL_HOST}:${POOL_PORT}/healthz" >/dev/null 2>&1 || {
        log "WARN: pool_server /healthz unreachable"
        LAST_POOL_ACTIVE="down"
        LAST_POOL_PENDING="down"
        return 1
    }
    local body pending=0 active=0
    body=$(timeout 3 curl -fsS --noproxy '*' "http://${POOL_HOST}:${POOL_PORT}/status" 2>/dev/null)
    if [ -n "$body" ]; then
        # 合并到 1 个 python 调用减少 fork 开销
        local parsed
        parsed=$(echo "$body" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    p = d.get("pool", d)
    print(p.get("pending_closes", 0), p.get("active_tasks", p.get("total_active_runs", 0)))
except Exception:
    print("0 0")
' 2>/dev/null)
        pending="${parsed% *}"
        active="${parsed#* }"
        pending="${pending:-0}"
        active="${active:-0}"
        LAST_POOL_PENDING="$pending"
        LAST_POOL_ACTIVE="$active"
        if [ "$pending" -gt "$POOL_PENDING_CLOSES_WARN" ] 2>/dev/null; then
            log "WARN: pool_server pending_closes=${pending} (active=${active}); pruning networks"
            timeout 30 docker network prune -f >/dev/null 2>&1 || true
        fi
    fi

    local nets
    nets=$(docker network ls --filter driver=bridge -q 2>/dev/null | wc -l)
    LAST_BRIDGE_NETS="$nets"
    if [ "$nets" -gt "$BRIDGE_NETS_WARN" ] 2>/dev/null; then
        log "WARN: ${nets} bridge networks, address-pool risk; pruning"
        timeout 30 docker network prune -f >/dev/null 2>&1 || true
    fi
    return 0
}

# ── stopped 容器清理 ─────────────────────────────────────────────────
cleanup_stopped() {
    local stopped
    stopped=$(docker ps -aq --filter "status=exited" --filter "status=dead" 2>/dev/null | wc -l)
    if [ "${stopped}" -gt 5 ]; then
        log "Cleaning ${stopped} stopped containers..."
        timeout 30 docker container prune -f --filter "until=2m" >/dev/null 2>&1 || true
    fi
    local dn
    dn=$(docker network ls --filter "dangling=true" -q 2>/dev/null | wc -l)
    if [ "${dn}" -gt 0 ]; then
        timeout 20 docker network prune -f >/dev/null 2>&1 || true
    fi
}

# ── 运行容器数上限（双闸门，排除 pool_server）─────────────────────────
# 副作用：更新 LAST_RUNNING_TASKS 供 heartbeat 复用
LAST_RUNNING_TASKS="?"
enforce_container_limit() {
    local running victims
    # 只统计 task 容器（带数字前缀 + client/helper 后缀），不算 pool_server 等基础容器
    running=$(docker ps --format '{{.Names}}' 2>/dev/null \
        | grep -cE "${TASK_CONTAINER_REGEX}" || true)
    LAST_RUNNING_TASKS="$running"

    if [ "${running}" -gt "${HARD_KILL_THRESHOLD}" ]; then
        local excess=$((running - MAX_RUNNING_CONTAINERS))
        log "HARD LIMIT: ${running} task containers > ${HARD_KILL_THRESHOLD}, killing ${excess} oldest"
        victims=$(docker ps --format '{{.ID}} {{.Names}}' 2>/dev/null \
            | grep -vE "${POOL_SERVER_NAME_REGEX}" \
            | grep -E "${TASK_CONTAINER_REGEX}" \
            | tail -n "${excess}" \
            | awk '{print $1}')
        [ -n "$victims" ] && echo "$victims" | xargs -r -n 10 timeout 30 docker kill >/dev/null 2>&1 || true
        return
    fi

    if [ "${running}" -gt "${MAX_RUNNING_CONTAINERS}" ]; then
        local excess=$((running - MAX_RUNNING_CONTAINERS))
        log "Soft limit: ${running} task containers > ${MAX_RUNNING_CONTAINERS}, killing ${excess} oldest"
        victims=$(docker ps --format '{{.ID}} {{.Names}}' 2>/dev/null \
            | grep -vE "${POOL_SERVER_NAME_REGEX}" \
            | grep -E "${TASK_CONTAINER_REGEX}" \
            | tail -n "${excess}" \
            | awk '{print $1}')
        [ -n "$victims" ] && echo "$victims" | xargs -r -n 10 timeout 30 docker kill >/dev/null 2>&1 || true
    fi
}

# ── dockerd 重启（绕过 systemctl restart，沿用 restart_docker_force.sh 模式）──
restart_docker() {
    log "Docker daemon is DOWN. Attempting forced restart (no systemctl restart)..."

    # 1) 阻断 systemd auto-restart：reset-failed + stop docker.socket
    timeout 5 systemctl reset-failed docker.service docker.socket 2>/dev/null || true
    timeout 5 systemctl stop docker.socket 2>/dev/null || true

    # 2) pkill -9
    pkill -9 -x dockerd 2>/dev/null || true
    sleep 2
    if pgrep -x dockerd >/dev/null 2>&1; then
        log "WARN: dockerd still alive after SIGKILL (D state?), aborting state cleanup"
        return 1
    fi
    rm -f /var/run/docker.pid "${DOCKER_SOCK}"

    # 3) 谨慎清理 container state — 仅在 host pid namespace 下且没有 shim 残留时才清
    if [ "$HOST_PID_NS" = "1" ]; then
        if ! pgrep -f containerd-shim >/dev/null 2>&1; then
            if [ -d "${DOCKER_DATA_ROOT}/containers" ]; then
                local n
                n=$(find "${DOCKER_DATA_ROOT}/containers" -maxdepth 1 -mindepth 1 2>/dev/null | wc -l)
                if [ "${n}" -gt 0 ]; then
                    log "Clearing ${n} stale container states (host ns + no shim alive)"
                    rm -rf "${DOCKER_DATA_ROOT}/containers"/* 2>/dev/null || true
                fi
            fi
            rm -f "${DOCKER_DATA_ROOT}/network/files/local-kv.db" 2>/dev/null || true
        else
            log "containerd-shim still alive, preserving container state"
        fi
    else
        log "Not in host PID namespace; cannot reliably enumerate shim — skipping state cleanup"
    fi

    # 4) 启动 dockerd（直接 nohup，不走 systemd）
    nohup dockerd --containerd=/run/containerd/containerd.sock \
        > /tmp/dockerd_watchdog_restart.log 2>&1 &
    local pid=$!
    log "Started dockerd PID=${pid}, waiting for API..."

    local i
    for i in $(seq 1 60); do
        if docker_alive; then
            log "Docker API ready after ${i} attempts (~$((i*5))s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log "ERROR: dockerd died during startup; tail log:"
            tail -20 /tmp/dockerd_watchdog_restart.log 2>/dev/null | sed 's/^/  /' || true
            return 1
        fi
        sleep 5
    done
    log "ERROR: Docker failed to start after 5 min. See /tmp/dockerd_watchdog_restart.log"
    return 1
}

# ── Main ─────────────────────────────────────────────────────────────
log "========================================"
log "Starting docker_watchdog_v2 PID=$$"
log "  MAX_RUNNING=${MAX_RUNNING_CONTAINERS}  HARD_KILL=${HARD_KILL_THRESHOLD}"
log "  health every ${HEALTH_CHECK_INTERVAL}s; cgroup every ${CGROUP_MONITOR_INTERVAL}s"
log "  pool every ${POOL_CHECK_INTERVAL}s; deep probe every ${DEEP_PROBE_INTERVAL}s"
log "  heartbeat every ${HEARTBEAT_INTERVAL}s"
log "  PIDs warn=${PIDS_WARN_PCT}% emerg=${PIDS_EMERGENCY_PCT}%"
log "  Mem  warn=${MEM_WARN_PCT}% emerg=${MEM_EMERGENCY_PCT}%"
log "  pool=${POOL_HOST}:${POOL_PORT}  pool_server_regex=${POOL_SERVER_NAME_REGEX}"
log "  task_container_regex=${TASK_CONTAINER_REGEX}"
log "  log_file=${LOG_FILE}  log_max=${LOG_MAX_BYTES}"

detect_pid_namespace
log "  pid_namespace: $([ "$HOST_PID_NS" = "1" ] && echo host || echo containerized)"

if detect_cgroup; then
    log "  cgroup: ${CGROUP_VERSION}"
    [ -n "$CGROUP_PIDS_DIR" ] && log "    pids: ${CGROUP_PIDS_DIR}  (max=${CGROUP_PIDS_MAX_VAL})"
    [ -n "$CGROUP_MEM_DIR"  ] && log "    mem : ${CGROUP_MEM_DIR}  (max=${CGROUP_MEM_MAX_VAL})"
else
    log "  cgroup: <NOT DETECTED — pressure monitoring DISABLED>"
fi
log "========================================"

LAST_CLEANUP=0
LAST_CGROUP_CHECK=0
LAST_POOL_CHECK=0
LAST_HEARTBEAT_TS=0
HEALTH_FAILS=0

while true; do
    NOW=$(date +%s)

    # 1) 浅探活
    if docker_alive; then
        HEALTH_FAILS=0
    else
        HEALTH_FAILS=$((HEALTH_FAILS + 1))
        log "Health check failed (${HEALTH_FAILS}/${MAX_CONSECUTIVE_HEALTH_FAILS})"
        if [ "${HEALTH_FAILS}" -ge "${MAX_CONSECUTIVE_HEALTH_FAILS}" ]; then
            restart_docker
            HEALTH_FAILS=0
            sleep 10
            continue
        fi
        sleep "${HEALTH_CHECK_INTERVAL}"
        continue
    fi

    # 2) 深度探活（5 min 一次）—— 抓 address-pool 耗尽这种 ping OK 但 reset 500 的形态
    if [ $((NOW - LAST_DEEP_PROBE_TS)) -ge "${DEEP_PROBE_INTERVAL}" ]; then
        if docker_deep_alive; then
            LAST_DEEP_PROBE_TS="$NOW"
        else
            log "WARN: deep probe failed (network create/rm) — likely address-pool exhausted"
            timeout 30 docker network prune -f >/dev/null 2>&1 || true
            LAST_DEEP_PROBE_TS="$NOW"
        fi
    fi

    # 3) cgroup 监控
    if [ $((NOW - LAST_CGROUP_CHECK)) -ge "${CGROUP_MONITOR_INTERVAL}" ]; then
        monitor_pod_cgroup
        LAST_CGROUP_CHECK="$NOW"
    fi

    # 4) pool_server 监控
    if [ $((NOW - LAST_POOL_CHECK)) -ge "${POOL_CHECK_INTERVAL}" ]; then
        check_pool_server || true
        LAST_POOL_CHECK="$NOW"
    fi

    # 5) 容器清理 + 上限
    if [ $((NOW - LAST_CLEANUP)) -ge "${CLEANUP_INTERVAL}" ]; then
        cleanup_stopped
        enforce_container_limit
        LAST_CLEANUP="$NOW"
    fi

    # 6) 低频心跳（默认 10 min）—— 复用上面已采集的指标，不发起新的 docker / curl
    if [ $((NOW - LAST_HEARTBEAT_TS)) -ge "${HEARTBEAT_INTERVAL}" ]; then
        log "OK: dockerd alive | pool active=${LAST_POOL_ACTIVE} pending_closes=${LAST_POOL_PENDING} | bridges=${LAST_BRIDGE_NETS} | task_containers=${LAST_RUNNING_TASKS}"
        LAST_HEARTBEAT_TS="$NOW"
    fi

    sleep "${HEALTH_CHECK_INTERVAL}"
done
