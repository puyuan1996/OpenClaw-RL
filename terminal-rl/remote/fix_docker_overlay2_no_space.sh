#!/usr/bin/env bash
# Recover cpu-worker Docker failures like:
#   mkdir /data/overlay2/<id>: no space left on device
#
# Default mode is conservative: remove stopped containers, unused networks, and
# old BuildKit cache. Set AGGRESSIVE=1 to prune unused images too. Set
# PRUNE_VOLUMES=1 only if task volumes are disposable.

set -Eeuo pipefail

DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/data}"
POOL_PORT="${POOL_PORT:-18081}"
BUILDER_CACHE_UNTIL="${BUILDER_CACHE_UNTIL:-24h}"
CONTAINER_UNTIL="${CONTAINER_UNTIL:-1h}"
AGGRESSIVE="${AGGRESSIVE:-0}"
PRUNE_VOLUMES="${PRUNE_VOLUMES:-0}"
RESTART_DOCKER="${RESTART_DOCKER:-0}"
RESTART_POOL="${RESTART_POOL:-0}"
DOCKER_CMD_TIMEOUT="${DOCKER_CMD_TIMEOUT:-30}"
DOCKER_PRUNE_TIMEOUT="${DOCKER_PRUNE_TIMEOUT:-120}"
RUN_HEAVY_DF="${RUN_HEAVY_DF:-0}"
RUN_PROBE="${RUN_PROBE:-1}"
PROBE_IMAGE="${PROBE_IMAGE:-}"
RUN_DEEP_DIAG="${RUN_DEEP_DIAG:-1}"
DU_TIMEOUT="${DU_TIMEOUT:-180}"
DRY_RUN="${DRY_RUN:-0}"
KILL_TASK_CONTAINERS="${KILL_TASK_CONTAINERS:-0}"
REMOVE_ALL_CONTAINERS="${REMOVE_ALL_CONTAINERS:-0}"
TASK_CONTAINER_REGEX="${TASK_CONTAINER_REGEX:-^[0-9]+-.*[-_](client|helper)([-_][0-9]+)?$}"
PURGE_DOCKER_ROOT_WHEN_EMPTY="${PURGE_DOCKER_ROOT_WHEN_EMPTY:-0}"
PURGE_DOCKER_ROOT_BACKUP="${PURGE_DOCKER_ROOT_BACKUP:-0}"
PURGE_DELETE_VOLUMES="${PURGE_DELETE_VOLUMES:-0}"
DOCKER_PURGE_DIRS="${DOCKER_PURGE_DIRS:-overlay2 containers image buildkit volumes network tmp runtimes plugins engine-id swarm}"
TRUNCATE_CONTAINER_LOGS="${TRUNCATE_CONTAINER_LOGS:-0}"
LOG_TRUNCATE_THRESHOLD_MB="${LOG_TRUNCATE_THRESHOLD_MB:-1024}"
LOCK_FILE="${LOCK_FILE:-/tmp/fix_docker_overlay2_no_space.lock}"
POOL_RUN_SCRIPT="${POOL_RUN_SCRIPT:-terminal-rl/remote/run_pool_server_pu_v2.sh}"
REPO_DIR="${REPO_DIR:-/mnt/shared-storage-user/puyuan/code/OpenClaw-RL}"
PRUNE_ERRORS=0

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

trap 'rc=$?; log "ERROR at line ${LINENO}: ${BASH_COMMAND} (exit=${rc})"' ERR

acquire_lock() {
    if command -v flock >/dev/null 2>&1; then
        exec 9>"${LOCK_FILE}"
        if ! flock -n 9; then
            log "Another cleanup instance is running (lock=${LOCK_FILE}). Exiting."
            exit 0
        fi
        return 0
    fi
    log "WARN: flock not found; proceeding without inter-process lock."
}

run() {
    log "+ $*"
    if [ "${DRY_RUN}" = "1" ]; then
        log "DRY_RUN: skipped command"
        return 0
    fi
    "$@"
}

need_root_for_restart() {
    if [ "$(id -u)" -ne 0 ]; then
        log "ERROR: RESTART_DOCKER=1 requires root. Re-run with sudo."
        exit 1
    fi
}

docker_ok() {
    timeout 10 docker info >/dev/null 2>&1
}

docker_t() {
    timeout "${DOCKER_CMD_TIMEOUT}" docker "$@"
}

docker_prune_t() {
    timeout "${DOCKER_PRUNE_TIMEOUT}" docker "$@"
}

record_prune_error() {
    PRUNE_ERRORS=$((PRUNE_ERRORS + 1))
    log "WARN: cleanup step failed: $*"
}

docker_pressure_stats() {
    local block inode
    block="$(df -P "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); print $5}')"
    inode="$(df -Pi "${DOCKER_DATA_ROOT}" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); print $5}')"
    echo "${block:-?} ${inode:-?}"
}

print_space() {
    log "Filesystem usage:"
    df -h "${DOCKER_DATA_ROOT}" / 2>/dev/null || true
    log "Inode usage:"
    df -ih "${DOCKER_DATA_ROOT}" / 2>/dev/null || true
    local stats
    stats="$(docker_pressure_stats)"
    log "Pressure summary: block=$(echo "$stats" | awk '{print $1}')% inode=$(echo "$stats" | awk '{print $2}')%"
}

print_docker_summary() {
    log "Docker root/storage:"
    timeout "${DOCKER_CMD_TIMEOUT}" docker info 2>/dev/null \
        | grep -E 'Docker Root Dir|Storage Driver|Containers:|Images:' || true
    if [ "${RUN_HEAVY_DF}" = "1" ]; then
        log "Docker system df (timeout=${DOCKER_CMD_TIMEOUT}s):"
        if ! timeout "${DOCKER_CMD_TIMEOUT}" docker system df 2>/tmp/docker_system_df.err; then
            log "WARN: docker system df timed out or failed; skipping heavy docker accounting."
            tail -20 /tmp/docker_system_df.err 2>/dev/null || true
        fi
    else
        log "Skipping docker system df. Set RUN_HEAVY_DF=1 to enable it."
    fi
    log "Object counts:"
    printf 'containers=%s images=%s volumes=%s networks=%s\n' \
        "$(docker_t ps -aq 2>/dev/null | wc -l || echo timeout)" \
        "$(docker_t images -q 2>/dev/null | wc -l || echo timeout)" \
        "$(docker_t volume ls -q 2>/dev/null | wc -l || echo timeout)" \
        "$(docker_t network ls -q 2>/dev/null | wc -l || echo timeout)"
}

print_deep_diagnostics() {
    [ "${RUN_DEEP_DIAG}" = "1" ] || return 0

    log "Deep diagnostics: top-level ${DOCKER_DATA_ROOT} usage (timeout=${DU_TIMEOUT}s)."
    timeout "${DU_TIMEOUT}" du -xhd1 "${DOCKER_DATA_ROOT}" 2>/tmp/docker_du_diag.err \
        | sort -h \
        | tail -30 || {
            log "WARN: du top-level scan timed out or failed."
            tail -20 /tmp/docker_du_diag.err 2>/dev/null || true
        }

    if [ -d "${DOCKER_DATA_ROOT}/overlay2" ]; then
        log "Deep diagnostics: largest overlay2 directories (timeout=${DU_TIMEOUT}s)."
        timeout "${DU_TIMEOUT}" sh -c '
            root="$1"
            find "$root/overlay2" -mindepth 1 -maxdepth 1 -type d ! -name l -print0 2>/dev/null \
              | xargs -0 -r du -xsh 2>/dev/null \
              | sort -h \
              | tail -20
        ' sh "${DOCKER_DATA_ROOT}" || log "WARN: overlay2 size scan timed out or failed."

        log "Deep diagnostics: overlay2 directory count."
        timeout "${DOCKER_CMD_TIMEOUT}" find "${DOCKER_DATA_ROOT}/overlay2" -mindepth 1 -maxdepth 1 -type d ! -name l 2>/dev/null \
            | wc -l || true
    fi

    if [ -d "${DOCKER_DATA_ROOT}/containers" ]; then
        log "Deep diagnostics: largest container JSON logs."
        timeout "${DU_TIMEOUT}" sh -c '
            root="$1"
            find "$root/containers" -name "*-json.log" -type f -printf "%s %p\n" 2>/dev/null \
              | sort -n \
              | tail -20 \
              | awk "{printf \"%.1fG %s\\n\", \$1/1024/1024/1024, \$2}"
        ' sh "${DOCKER_DATA_ROOT}" || true
    fi

    log "Deep diagnostics: running containers with writable layer sizes."
    timeout "${DOCKER_CMD_TIMEOUT}" docker ps -s --no-trunc 2>/dev/null || true
}

remove_active_containers_if_requested() {
    if [ "${REMOVE_ALL_CONTAINERS}" = "1" ]; then
        log "REMOVE_ALL_CONTAINERS=1: removing all Docker containers."
        local ids
        ids="$(timeout "${DOCKER_CMD_TIMEOUT}" docker ps -aq 2>/dev/null || true)"
        if [ -n "${ids}" ]; then
            log "Containers to remove: $(echo "${ids}" | wc -l)"
            if [ "${DRY_RUN}" = "1" ]; then
                echo "${ids}" | sed 's/^/DRY_RUN would remove container: /'
            else
                echo "${ids}" | xargs -r timeout "${DOCKER_PRUNE_TIMEOUT}" docker rm -f \
                    || record_prune_error "docker rm -f all containers"
            fi
        fi
        return 0
    fi

    [ "${KILL_TASK_CONTAINERS}" = "1" ] || return 0

    log "KILL_TASK_CONTAINERS=1: removing task containers matching ${TASK_CONTAINER_REGEX}."
    local ids
    ids="$(timeout "${DOCKER_CMD_TIMEOUT}" docker ps -a --format '{{.ID}}\t{{.Names}}' 2>/dev/null \
        | awk -F '\t' -v re="${TASK_CONTAINER_REGEX}" '$2 ~ re {print $1}' \
        || true)"
    if [ -n "${ids}" ]; then
        log "Task containers to remove: $(echo "${ids}" | wc -l)"
        if [ "${DRY_RUN}" = "1" ]; then
            echo "${ids}" | sed 's/^/DRY_RUN would remove task container: /'
        else
            echo "${ids}" | xargs -r timeout "${DOCKER_PRUNE_TIMEOUT}" docker rm -f \
                || record_prune_error "docker rm -f task containers"
        fi
    fi
}

remove_volumes_if_requested() {
    [ "${PURGE_DELETE_VOLUMES}" = "1" ] || return 0
    log "PURGE_DELETE_VOLUMES=1: removing all Docker volumes before offline purge."
    local ids
    ids="$(timeout "${DOCKER_CMD_TIMEOUT}" docker volume ls -q 2>/dev/null || true)"
    if [ -z "${ids}" ]; then
        return 0
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        echo "${ids}" | sed 's/^/DRY_RUN would remove volume: /'
    else
        echo "${ids}" | xargs -r timeout "${DOCKER_PRUNE_TIMEOUT}" docker volume rm -f \
            || record_prune_error "docker volume rm -f all volumes"
    fi
}

docker_object_count() {
    local kind="$1" out
    case "$kind" in
        containers) out="$(docker_t ps -aq 2>/dev/null | wc -l)" || return 1 ;;
        images) out="$(docker_t images -q 2>/dev/null | wc -l)" || return 1 ;;
        volumes) out="$(docker_t volume ls -q 2>/dev/null | wc -l)" || return 1 ;;
        *) return 1 ;;
    esac
    echo "${out:-0}"
}

truncate_container_logs_if_requested() {
    [ "${TRUNCATE_CONTAINER_LOGS}" = "1" ] || return 0
    [ -d "${DOCKER_DATA_ROOT}/containers" ] || return 0
    log "TRUNCATE_CONTAINER_LOGS=1: truncating *-json.log files larger than ${LOG_TRUNCATE_THRESHOLD_MB}MB."
    if [ "${DRY_RUN}" = "1" ]; then
        find "${DOCKER_DATA_ROOT}/containers" -name '*-json.log' -type f \
            -size +"${LOG_TRUNCATE_THRESHOLD_MB}"M -print 2>/dev/null \
            | sed 's/^/DRY_RUN would truncate log: /'
        return 0
    fi
    find "${DOCKER_DATA_ROOT}/containers" -name '*-json.log' -type f \
        -size +"${LOG_TRUNCATE_THRESHOLD_MB}"M -print -exec truncate -s 0 {} \; 2>/dev/null \
        || record_prune_error "truncate container json logs"
}

warn_if_log_rotation_missing() {
    local daemon_json="/etc/docker/daemon.json"
    if [ ! -f "${daemon_json}" ]; then
        log "WARN: ${daemon_json} not found; Docker log rotation may be unset."
        return 0
    fi
    if ! grep -q '"max-size"' "${daemon_json}" 2>/dev/null; then
        log "WARN: Docker daemon.json appears to lack log-opts max-size; container logs can grow without bound."
    fi
}

stop_docker_for_offline_cleanup() {
    need_root_for_restart
    log "Stopping docker-watchdog/docker for offline Docker root cleanup."
    if [ "${DRY_RUN}" = "1" ]; then
        log "DRY_RUN: would stop docker-watchdog/docker and unmount overlay2 mounts."
        return 0
    fi
    timeout 15 systemctl stop docker-watchdog 2>/dev/null || true
    timeout 15 systemctl stop docker.socket 2>/dev/null || true
    timeout 30 systemctl stop docker 2>/dev/null || true
    pkill -9 -x dockerd 2>/dev/null || true
    pkill -9 -f containerd-shim 2>/dev/null || true
    local i
    for i in $(seq 1 40); do
        if ! pgrep -x dockerd >/dev/null 2>&1 && ! pgrep -f containerd-shim >/dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
    if pgrep -x dockerd >/dev/null 2>&1 || pgrep -f containerd-shim >/dev/null 2>&1; then
        log "WARN: dockerd/containerd-shim still visible after stop attempts; offline deletion may be blocked."
    fi
    mount | awk -v root="${DOCKER_DATA_ROOT}/overlay2" '$0 ~ /overlay/ && index($3, root) == 1 {print $3}' \
        | sort -r \
        | xargs -r umount -l 2>/dev/null || true
    rm -f /var/run/docker.pid /var/run/docker.sock 2>/dev/null || true
}

purge_docker_root_when_empty_if_requested() {
    [ "${PURGE_DOCKER_ROOT_WHEN_EMPTY}" = "1" ] || return 0
    need_root_for_restart

    local docker_root
    docker_root="$(timeout "${DOCKER_CMD_TIMEOUT}" docker info 2>/dev/null | awk -F': ' '/Docker Root Dir/{print $2; exit}')"
    if [ -z "${docker_root}" ]; then
        log "ERROR: cannot resolve Docker Root Dir; refusing offline purge."
        exit 4
    fi
    if [ "${docker_root}" != "${DOCKER_DATA_ROOT}" ]; then
        log "ERROR: Docker Root Dir (${docker_root}) != DOCKER_DATA_ROOT (${DOCKER_DATA_ROOT}); refusing offline purge."
        exit 4
    fi
    if [ "${DOCKER_DATA_ROOT}" = "/" ] || [ "${DOCKER_DATA_ROOT}" = "/var" ]; then
        log "ERROR: unsafe DOCKER_DATA_ROOT=${DOCKER_DATA_ROOT}; refusing offline purge."
        exit 4
    fi

    remove_volumes_if_requested

    local n_containers n_images n_volumes
    if ! n_containers="$(docker_object_count containers)"; then n_containers=999999; fi
    if ! n_images="$(docker_object_count images)"; then n_images=999999; fi
    if ! n_volumes="$(docker_object_count volumes)"; then n_volumes=999999; fi
    log "Offline purge precheck: containers=${n_containers} images=${n_images} volumes=${n_volumes}"
    if [ "${n_containers}" -ne 0 ] || [ "${n_images}" -ne 0 ]; then
        log "ERROR: Docker still has containers/images. Re-run with REMOVE_ALL_CONTAINERS=1 AGGRESSIVE=1 first."
        exit 4
    fi
    if [ "${n_volumes}" -ne 0 ] && [ "${PURGE_DELETE_VOLUMES}" != "1" ]; then
        log "ERROR: Docker still has volumes. Re-run with PURGE_DELETE_VOLUMES=1 if volume data is disposable."
        exit 4
    fi

    stop_docker_for_offline_cleanup

    local ts target d
    ts="$(date '+%Y%m%d_%H%M%S')"
    if [ "${PURGE_DOCKER_ROOT_BACKUP}" = "1" ]; then
        target="${DOCKER_DATA_ROOT}.orphan.${ts}"
        log "PURGE_DOCKER_ROOT_BACKUP=1: moving Docker root to ${target}"
        run mv "${DOCKER_DATA_ROOT}" "${target}"
        run mkdir -p "${DOCKER_DATA_ROOT}"
    else
        log "Deleting Docker root subdirs: ${DOCKER_PURGE_DIRS}"
        for d in ${DOCKER_PURGE_DIRS}; do
            if [ -e "${DOCKER_DATA_ROOT}/${d}" ]; then
                log "Deleting ${DOCKER_DATA_ROOT}/${d}"
                if [ "${DRY_RUN}" = "1" ]; then
                    log "DRY_RUN: would delete ${DOCKER_DATA_ROOT}/${d}"
                else
                    rm -rf --one-file-system "${DOCKER_DATA_ROOT:?}/${d}"
                fi
            fi
        done
    fi

    log "Starting Docker after offline cleanup."
    [ "${DRY_RUN}" = "1" ] && return 0
    if [ -x "${REPO_DIR}/terminal-rl/remote/restart_docker_force.sh" ]; then
        env DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" bash "${REPO_DIR}/terminal-rl/remote/restart_docker_force.sh" || true
    else
        systemctl start docker || true
    fi
}

stop_pool_server_if_local() {
    local pids
    pids="$(ss -tlnp 2>/dev/null | awk -v port=":${POOL_PORT}" '$0 ~ port {print $0}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u || true)"
    if [ -z "${pids}" ]; then
        log "No local pool_server listener found on port ${POOL_PORT}."
        return 0
    fi

    log "Stopping local service on port ${POOL_PORT}: pid(s) ${pids}"
    echo "${pids}" | xargs -r kill
    sleep 3
    echo "${pids}" | xargs -r kill -9 2>/dev/null || true
}

restart_docker_if_requested() {
    [ "${RESTART_DOCKER}" = "1" ] || return 0
    need_root_for_restart

    if [ -x "${REPO_DIR}/terminal-rl/remote/restart_docker_force.sh" ]; then
        log "Restarting Docker via repo force-restart helper."
        run env DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" bash "${REPO_DIR}/terminal-rl/remote/restart_docker_force.sh"
    else
        log "Restarting Docker via systemctl."
        run systemctl restart docker
    fi
}

restart_pool_if_requested() {
    [ "${RESTART_POOL}" = "1" ] || return 0
    if [ ! -f "${REPO_DIR}/${POOL_RUN_SCRIPT}" ]; then
        log "WARN: pool run script not found: ${REPO_DIR}/${POOL_RUN_SCRIPT}"
        return 0
    fi

    log "Restarting pool_server in background."
    cd "${REPO_DIR}"
    nohup bash "${POOL_RUN_SCRIPT}" > /tmp/cpu_pool.log 2>&1 &
    echo "$!" > /tmp/cpu_pool.pid
    sleep 2
    curl --noproxy '*' --max-time 5 "http://127.0.0.1:${POOL_PORT}/healthz" || true
    echo
}

main() {
    acquire_lock
    if [ "${PURGE_DOCKER_ROOT_WHEN_EMPTY}" = "1" ]; then
        REMOVE_ALL_CONTAINERS=1
        AGGRESSIVE=1
        PRUNE_VOLUMES=1
        PURGE_DELETE_VOLUMES=1
        RUN_PROBE=0
        log "PURGE mode: forced REMOVE_ALL_CONTAINERS=1 AGGRESSIVE=1 PRUNE_VOLUMES=1 PURGE_DELETE_VOLUMES=1 RUN_PROBE=0"
    fi
    log "Docker overlay2 no-space recovery starting."
    log "DOCKER_DATA_ROOT=${DOCKER_DATA_ROOT} AGGRESSIVE=${AGGRESSIVE} PRUNE_VOLUMES=${PRUNE_VOLUMES} RESTART_DOCKER=${RESTART_DOCKER}"
    log "DOCKER_CMD_TIMEOUT=${DOCKER_CMD_TIMEOUT} DOCKER_PRUNE_TIMEOUT=${DOCKER_PRUNE_TIMEOUT} RUN_HEAVY_DF=${RUN_HEAVY_DF} RUN_DEEP_DIAG=${RUN_DEEP_DIAG}"
    log "KILL_TASK_CONTAINERS=${KILL_TASK_CONTAINERS} REMOVE_ALL_CONTAINERS=${REMOVE_ALL_CONTAINERS} PURGE_DOCKER_ROOT_WHEN_EMPTY=${PURGE_DOCKER_ROOT_WHEN_EMPTY} DRY_RUN=${DRY_RUN}"

    print_space
    warn_if_log_rotation_missing

    if ! docker_ok; then
        log "Docker daemon is not responding before cleanup."
        restart_docker_if_requested
    fi

    if ! docker_ok; then
        log "ERROR: Docker daemon is still not responding. Try: sudo RESTART_DOCKER=1 $0"
        exit 2
    fi

    print_docker_summary
    print_deep_diagnostics
    stop_pool_server_if_local
    remove_active_containers_if_requested
    truncate_container_logs_if_requested

    log "Cleaning stopped/dead containers older than ${CONTAINER_UNTIL}."
    docker_prune_t container prune -f --filter "until=${CONTAINER_UNTIL}" \
        || record_prune_error "docker container prune"

    log "Cleaning unused Docker networks."
    docker_prune_t network prune -f || record_prune_error "docker network prune"

    log "Cleaning BuildKit/build cache older than ${BUILDER_CACHE_UNTIL}."
    docker_prune_t builder prune -af --filter "until=${BUILDER_CACHE_UNTIL}" \
        || record_prune_error "docker builder prune"

    log "Cleaning dangling images."
    docker_prune_t image prune -f || record_prune_error "docker image prune"

    if [ "${AGGRESSIVE}" = "1" ]; then
        log "AGGRESSIVE=1: cleaning all unused images. This may force later rebuild/pull."
        docker_prune_t image prune -af || record_prune_error "docker image prune -af"
    fi

    if [ "${PRUNE_VOLUMES}" = "1" ]; then
        log "PRUNE_VOLUMES=1: cleaning unused volumes."
        docker_prune_t volume prune -f || record_prune_error "docker volume prune"
    fi

    purge_docker_root_when_empty_if_requested

    restart_docker_if_requested

    log "Post-cleanup diagnostics."
    print_space
    print_docker_summary
    print_deep_diagnostics

    if ! docker_ok; then
        log "ERROR: Docker daemon is not healthy after cleanup."
        exit 3
    fi

    if [ "${RUN_PROBE}" = "1" ]; then
        if [ -z "${PROBE_IMAGE}" ]; then
            PROBE_IMAGE="$(docker_t images -q 2>/dev/null | head -1 || true)"
        fi
        if [ -z "${PROBE_IMAGE}" ]; then
            log "WARN: no local image found for docker create probe; skipping pull-based probe."
        elif ! timeout "${DOCKER_CMD_TIMEOUT}" docker create --name "overlay2_probe_$$" "${PROBE_IMAGE}" true >/tmp/docker_overlay2_probe.log 2>&1; then
            log "WARN: docker create probe failed. Last output:"
            tail -80 /tmp/docker_overlay2_probe.log || true
        else
            timeout "${DOCKER_CMD_TIMEOUT}" docker rm -f "overlay2_probe_$$" >/dev/null 2>&1 || true
            log "Docker create probe passed."
        fi
    fi

    restart_pool_if_requested

    if [ "${PRUNE_ERRORS}" -gt 0 ]; then
        log "WARN: ${PRUNE_ERRORS} cleanup step(s) failed; review logs above."
    fi
    log "Done."
}

main "$@"
