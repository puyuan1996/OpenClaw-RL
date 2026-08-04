#!/usr/bin/env bash
# One-shot setup for a NEW CPU worker to run terminal-rl pool_server.
#
# Requirements for the new machine:
#   - Linux (Ubuntu 20.04+)
#   - Docker installed (or this script will try to install it)
#   - Network access (direct or via proxy)
#   - At least 200GB free disk on the partition where docker stores data
#   - Can reach GPU worker on port 18081 (pool_server listen port)
#
# Usage:
#   sudo env DOCKER_DATA_ROOT=/data bash terminal-rl/remote/setup_new_worker.sh
#
# Environment variables (optional):
#   PROXY_URL          - HTTP proxy for dockerd/pip/builds. Auto-detected on pjlab.
#   AUTO_PJLAB_PROXY   - 1 to default to PJLab proxy when PROXY_URL is unset. Default: 1.
#   DEFAULT_PROXY_URL  - Proxy used by AUTO_PJLAB_PROXY.
#   PROXY_CHECK_URL    - URL used to verify auto/default proxy. Default: https://pypi.org/simple/pip/
#   PROXY_CHECK_TIMEOUT - Proxy verification timeout. Default: 5.
#   NO_PROXY_LIST      - no_proxy list for internal network bypass.
#   DOCKER_DATA_ROOT   - Docker data root. DOCKER_ROOT is accepted as legacy alias.
#   DOCKER_STORAGE_DRIVER - Docker storage driver. Default: auto.
#   SHARED_CONDA_POOL_SERVER_VENV - Preferred ready-made pool_server env. Default: ../conda_envs/openclaw-worker-py312.
#   POOL_SERVER_VENV  - Python env for pool_server. Default: SHARED_CONDA_POOL_SERVER_VENV if valid, else <repo>/.venv.
#   POOL_SERVER_MIN_PYTHON - Minimum Python version for reusing a venv. Default: 3.12.
#   POOL_SERVER_CREATE_PYTHON - Python version used when creating a venv. Default: 3.12.
#   POOL_SERVER_VENV_BACKEND - auto|uv|python. Default: python.
#   POOL_SERVER_INSTALL_BACKEND - auto|uv|pip. Default: pip.
#   ASSUME_YES         - 1 to continue through low-disk warning non-interactively.
#   RUN_PROXY_FIX      - 1 to run fix_dockerd_and_proxy.sh at the end. Default: 1.
#   INSTALL_WATCHDOG   - 1 to install and start docker-watchdog.service. Default: 1.
#   SKIP_VERIFY        - 1 to skip docker build verification. Default: 0.
#   DOCKER_INFO_TIMEOUT - Timeout for docker info probes. Default: 10.
#   DOCKER_PULL_TIMEOUT - Timeout for docker pulls. Default: 1800.
#   UV_HTTP_TIMEOUT    - Timeout for uv downloads. Default: 300.
#   PIP_TIMEOUT        - Timeout for pip downloads. Default: 300.
#   PIP_RETRIES        - Retry count for pip downloads. Default: 10.
#   PIP_CACHE_DIR      - pip cache directory. Default: /tmp/pip-cache-openclaw.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "[ERROR] $*"; exit 1; }
docker_info_ok() { timeout "${DOCKER_INFO_TIMEOUT}" docker info >/dev/null 2>&1; }
docker_compose_version() { timeout 10 docker compose version 2>/dev/null; }
systemd_available() {
  command -v systemctl >/dev/null 2>&1 &&
    [ -d /run/systemd/system ] &&
    timeout 5 systemctl list-units --no-pager >/dev/null 2>&1
}
run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}
force_restart_docker() {
  local use_systemd_start="${USE_SYSTEMD_START:-1}"
  local force_restart_containerd="${FORCE_RESTART_CONTAINERD:-1}"
  if ! systemd_available; then
    use_systemd_start="0"
  fi
  if [ "$(id -u)" -eq 0 ]; then
    DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" USE_SYSTEMD_START="${use_systemd_start}" FORCE_RESTART_CONTAINERD="${force_restart_containerd}" \
      bash "${SCRIPT_DIR}/restart_docker_force.sh"
  else
    sudo env DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" USE_SYSTEMD_START="${use_systemd_start}" FORCE_RESTART_CONTAINERD="${force_restart_containerd}" \
      bash "${SCRIPT_DIR}/restart_docker_force.sh"
  fi
}

echo "============================================================"
echo " Terminal-RL Pool Server - New Worker Setup"
echo " $(date)"
echo "============================================================"
echo ""

# ── 0. Detect proxy ─────────────────────────────────────────────────
AUTO_PJLAB_PROXY="${AUTO_PJLAB_PROXY:-1}"
DEFAULT_PROXY_URL="${DEFAULT_PROXY_URL:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}"
PROXY_CHECK_URL="${PROXY_CHECK_URL:-https://pypi.org/simple/pip/}"
PROXY_CHECK_TIMEOUT="${PROXY_CHECK_TIMEOUT:-5}"
PROXY_URL="${PROXY_URL:-${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}}"
PROXY_SOURCE=""
[ -n "$PROXY_URL" ] && PROXY_SOURCE="env"
NO_PROXY_LIST="${NO_PROXY_LIST:-${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc}}}"

proxy_url_ok() {
  local proxy="$1"
  [ -n "$proxy" ] || return 1
  command -v curl >/dev/null 2>&1 || return 1
  curl -fsS --max-time "${PROXY_CHECK_TIMEOUT}" -x "$proxy" "${PROXY_CHECK_URL}" >/dev/null 2>&1
}

if [ -z "$PROXY_URL" ] && [ "${AUTO_PJLAB_PROXY}" = "1" ]; then
  # Try the pjlab proxy setup script first; fall back to the known cluster proxy.
  if command -v curl >/dev/null 2>&1 &&
     curl -fsS --max-time 3 "http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh" >/dev/null 2>&1; then
    PROXY_URL="$DEFAULT_PROXY_URL"
    PROXY_SOURCE="auto"
    echo "[auto] Detected pjlab proxy: $PROXY_URL"
  else
    PROXY_URL="$DEFAULT_PROXY_URL"
    PROXY_SOURCE="default"
    echo "[auto] Using default pjlab proxy: $PROXY_URL"
  fi
fi
if [ -n "$PROXY_URL" ] && [ "$PROXY_SOURCE" != "env" ]; then
  if proxy_url_ok "$PROXY_URL"; then
    echo "  Proxy check OK: ${PROXY_CHECK_URL}"
  else
    echo "  [WARN] Auto/default proxy is not reachable: ${PROXY_URL}"
    echo "         Falling back to bare network environment. Set PROXY_URL explicitly to force a proxy."
    PROXY_URL=""
    PROXY_SOURCE=""
  fi
fi
if [ -n "$PROXY_URL" ]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
  export no_proxy="$NO_PROXY_LIST"
  export NO_PROXY="$no_proxy"
  echo "  Proxy URL: $PROXY_URL"
  echo "  No proxy: $NO_PROXY_LIST"
fi
ASSUME_YES="${ASSUME_YES:-0}"
RUN_PROXY_FIX="${RUN_PROXY_FIX:-1}"
INSTALL_WATCHDOG="${INSTALL_WATCHDOG:-1}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
DOCKER_INFO_TIMEOUT="${DOCKER_INFO_TIMEOUT:-10}"
DOCKER_PULL_TIMEOUT="${DOCKER_PULL_TIMEOUT:-1800}"
DOCKER_STORAGE_DRIVER="${DOCKER_STORAGE_DRIVER:-}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
PIP_TIMEOUT="${PIP_TIMEOUT:-300}"
PIP_RETRIES="${PIP_RETRIES:-10}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-openclaw}"
export UV_HTTP_TIMEOUT
export PIP_CACHE_DIR
export PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT"

# ── 1. Check disk space ─────────────────────────────────────────────
echo "=== 1. Disk Space Check ==="
if [ -n "${DOCKER_DATA_ROOT:-}" ]; then
  DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}"
elif [ -n "${DOCKER_ROOT:-}" ]; then
  DOCKER_DATA_ROOT="${DOCKER_ROOT}"
else
  DETECTED_DOCKER_ROOT=""
  if command -v docker &>/dev/null; then
    DETECTED_DOCKER_ROOT="$(timeout "${DOCKER_INFO_TIMEOUT}" docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
  fi
  if [ -n "${DETECTED_DOCKER_ROOT}" ]; then
    DOCKER_DATA_ROOT="${DETECTED_DOCKER_ROOT}"
    echo "  [auto] Detected existing Docker root: ${DOCKER_DATA_ROOT}"
  elif [ -f /etc/docker/daemon.json ] && command -v python3 &>/dev/null; then
    DETECTED_DOCKER_ROOT="$(python3 - <<'PY' 2>/dev/null || true
import json
with open("/etc/docker/daemon.json") as f:
    print(json.load(f).get("data-root", ""))
PY
)"
    if [ -n "${DETECTED_DOCKER_ROOT}" ]; then
      DOCKER_DATA_ROOT="${DETECTED_DOCKER_ROOT}"
      echo "  [auto] Detected Docker root from daemon.json: ${DOCKER_DATA_ROOT}"
    else
      DOCKER_DATA_ROOT="/var/lib/docker"
    fi
  else
    DOCKER_DATA_ROOT="/var/lib/docker"
  fi
fi
DOCKER_ROOT="${DOCKER_DATA_ROOT}"
run_sudo mkdir -p "$DOCKER_DATA_ROOT"
DOCKER_PARTITION=$(df "$DOCKER_DATA_ROOT" --output=target 2>/dev/null | tail -1)
AVAIL_GB=$(df -BG --output=avail "$DOCKER_DATA_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
DOCKER_ROOT_FSTYPE="$(findmnt -n -T "$DOCKER_DATA_ROOT" -o FSTYPE 2>/dev/null || true)"

if [ -z "$DOCKER_STORAGE_DRIVER" ]; then
  case "$DOCKER_ROOT_FSTYPE" in
    ext2|ext3|ext4|xfs|btrfs|zfs)
      DOCKER_STORAGE_DRIVER="overlay2"
      ;;
    *)
      if [ -e /dev/fuse ]; then
        DOCKER_STORAGE_DRIVER="fuse-overlayfs"
      else
        DOCKER_STORAGE_DRIVER="vfs"
      fi
      echo "  [auto] Selected Docker storage driver ${DOCKER_STORAGE_DRIVER} for filesystem ${DOCKER_ROOT_FSTYPE:-unknown}"
      ;;
  esac
fi

echo "  Docker data root: $DOCKER_DATA_ROOT"
echo "  Docker storage driver: $DOCKER_STORAGE_DRIVER"
echo "  Docker root filesystem: ${DOCKER_ROOT_FSTYPE:-unknown}"
echo "  Partition: $DOCKER_PARTITION"
echo "  Available: ${AVAIL_GB}GB"
echo ""

if [ "${AVAIL_GB:-0}" -lt 150 ]; then
  echo "  [WARN] Less than 150GB available. Full training needs ~100-200GB for docker images."
  echo "         Consider setting DOCKER_DATA_ROOT to a larger partition."
  echo ""
  echo "  Available mount points with >150GB:"
  df -BG --output=target,avail 2>/dev/null | awk 'NR>1 && $2+0 > 150 {print "    "$1" ("$2" free)"}'
  echo ""
  if [ "${ASSUME_YES}" != "1" ]; then
    read -p "  Continue anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      die "Aborted. Set DOCKER_DATA_ROOT=/path/to/large/disk and re-run."
    fi
  else
    echo "  ASSUME_YES=1, continuing despite low disk."
  fi
fi

# ── 2. Install Docker (if missing) ──────────────────────────────────
echo "=== 2. Docker Installation ==="
if command -v docker &>/dev/null; then
  echo "  [OK] Docker already installed: $(docker --version)"
else
  echo "  Installing docker..."
  run_sudo apt-get update
  run_sudo apt-get install -y docker.io
  if systemd_available; then
    run_sudo systemctl enable docker
    run_sudo systemctl start docker
  else
    echo "  [WARN] systemd is not available; starting Docker with direct fallback"
    force_restart_docker
  fi
  echo "  [OK] Docker installed"
fi

# Probe Docker daemon with a hard timeout. On broken workers `docker info`
# can block forever on a stale socket; setup must continue to the repair path.
if docker_info_ok; then
  echo "  [OK] Docker daemon API responding"
else
  echo "  [WARN] Docker daemon API did not respond within ${DOCKER_INFO_TIMEOUT}s."
  echo "         This usually means dockerd is stopped, wedged, or has a stale socket."
  echo "         Continuing to daemon config; Step 4 will restart/repair dockerd."
  if [ "$(id -u)" -ne 0 ]; then
    run_sudo usermod -aG docker "${SUDO_USER:-$USER}" 2>/dev/null || true
    echo "  [NOTE] Added ${SUDO_USER:-$USER} to docker group. You may need to re-login or run: newgrp docker"
  fi
fi

# ── 3. Install Docker Compose V2 ────────────────────────────────────
echo "=== 3. Docker Compose V2 ==="
if COMPOSE_VERSION="$(docker_compose_version)"; then
  echo "  [OK] ${COMPOSE_VERSION}"
else
  echo "  Installing docker compose V2 plugin..."
  COMPOSE_PLUGIN_DIR="/usr/local/lib/docker/cli-plugins"
  run_sudo mkdir -p "$COMPOSE_PLUGIN_DIR"
  COMPOSE_URL="https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64"
  if [ -n "$PROXY_URL" ]; then
    curl -SL --proxy "$PROXY_URL" "$COMPOSE_URL" -o /tmp/docker-compose
  else
    curl -SL "$COMPOSE_URL" -o /tmp/docker-compose
  fi
  run_sudo install -m 0755 /tmp/docker-compose "$COMPOSE_PLUGIN_DIR/docker-compose"
  rm -f /tmp/docker-compose
  echo "  [OK] $(docker_compose_version)"
fi

# ── 4. Configure Docker daemon (proxy + data-root + address pools) ──
echo "=== 4. Docker Daemon Configuration ==="

# 4a. Proxy for pulling images
if [ -n "$PROXY_URL" ]; then
  run_sudo mkdir -p /etc/systemd/system/docker.service.d
  run_sudo tee /etc/seta_build_proxy.env > /dev/null <<EOF
HTTP_PROXY=${PROXY_URL}
HTTPS_PROXY=${PROXY_URL}
http_proxy=${PROXY_URL}
https_proxy=${PROXY_URL}
NO_PROXY=${NO_PROXY_LIST}
no_proxy=${NO_PROXY_LIST}
EOF
  run_sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<EOF
[Service]
Environment="HTTP_PROXY=${PROXY_URL}"
Environment="HTTPS_PROXY=${PROXY_URL}"
Environment="NO_PROXY=${NO_PROXY_LIST}"
EOF
  echo "  [OK] Docker daemon proxy configured"
fi

if [ "${DOCKER_STORAGE_DRIVER}" = "fuse-overlayfs" ] && ! command -v fuse-overlayfs &>/dev/null; then
  echo "  Installing fuse-overlayfs for Docker storage driver..."
  run_sudo apt-get update
  run_sudo apt-get install -y fuse-overlayfs
fi
if [ "${DOCKER_STORAGE_DRIVER}" = "fuse-overlayfs" ] && [ ! -e /dev/fuse ]; then
  die "DOCKER_STORAGE_DRIVER=fuse-overlayfs requires /dev/fuse. Use a local ext4/xfs Docker root or set DOCKER_STORAGE_DRIVER=vfs."
fi

# 4b. daemon.json (data-root + address pools)
DAEMON_JSON="/etc/docker/daemon.json"
if [ -f "$DAEMON_JSON" ]; then
  DAEMON_BACKUP="${DAEMON_JSON}.bak.$(date +%Y%m%d_%H%M%S)"
  run_sudo cp -a "$DAEMON_JSON" "$DAEMON_BACKUP"
  echo "  [backup] existing daemon.json -> ${DAEMON_BACKUP}"
fi
run_sudo mkdir -p "$DOCKER_DATA_ROOT"
run_sudo tee "$DAEMON_JSON" > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com",
    "https://mirror.ccs.tencentyun.com"
  ],
  "insecure-registries": [
    "registry.h.pjlab.org.cn"
  ],
  "data-root": "${DOCKER_DATA_ROOT}",
  "storage-driver": "${DOCKER_STORAGE_DRIVER}",
  "live-restore": true,
  "max-concurrent-downloads": 6,
  "max-concurrent-uploads": 6,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  },
  "default-address-pools": [
    {"base": "10.200.0.0/12", "size": 24}
  ],
  "default-ulimits": {
    "nproc": {
      "Name": "nproc",
      "Hard": 4096,
      "Soft": 2048
    },
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    },
    "core": {
      "Name": "core",
      "Hard": 0,
      "Soft": 0
    }
  },
  "default-shm-size": "64M"
}
EOF
echo "  [OK] Docker daemon.json configured (data-root=${DOCKER_DATA_ROOT}, storage-driver=${DOCKER_STORAGE_DRIVER}, address-pool=10.200.0.0/12)"

# 4c. Restart docker
echo "  Restarting docker daemon..."
if systemd_available; then
  run_sudo systemctl daemon-reload
  if [ "$(id -u)" -eq 0 ]; then
    RESTART_CMD=(timeout 90 systemctl restart docker)
  else
    RESTART_CMD=(timeout 90 sudo systemctl restart docker)
  fi
  if ! "${RESTART_CMD[@]}"; then
    echo "  [WARN] systemctl restart docker failed or timed out; using force restart fallback"
    force_restart_docker
  fi
else
  echo "  [WARN] systemd is not available; using force restart fallback"
  force_restart_docker
fi
sleep 3
if ! docker_info_ok; then
  echo "  [WARN] Docker API still not responding after systemd restart; using force restart fallback"
  force_restart_docker
fi
docker_info_ok || { echo "  [FAIL] Docker not responding after restart/repair"; exit 1; }
echo "  [OK] Docker daemon running"
echo ""

# ── 5. Pre-pull base images ─────────────────────────────────────────
echo "=== 5. Pre-pull Base Images ==="
BASE_IMAGES=(
  "ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624"
  "ghcr.io/laude-institute/t-bench/python-3-13:20250620"
  "ubuntu:22.04"
)
for img in "${BASE_IMAGES[@]}"; do
  if timeout 30 docker image inspect "$img" >/dev/null 2>&1; then
    echo "  [cached] $img"
  else
    echo "  [pulling] $img ..."
    PULL_LOG="$(mktemp)"
    if timeout "${DOCKER_PULL_TIMEOUT}" docker pull "$img" > "${PULL_LOG}" 2>&1; then
      grep -E "Pull complete|Digest|Status" "${PULL_LOG}" | tail -3 || tail -3 "${PULL_LOG}"
    else
      echo "  [FAIL] docker pull timed out or failed for ${img}; last log lines:"
      tail -20 "${PULL_LOG}"
      rm -f "${PULL_LOG}"
      exit 1
    fi
    rm -f "${PULL_LOG}"
  fi
done
echo ""

# ── 6. Python environment ───────────────────────────────────────────
echo "=== 6. Python Environment ==="
cd "$REPO_ROOT"
DEFAULT_POOL_SERVER_VENV="${REPO_ROOT}/.venv"
SHARED_CONDA_POOL_SERVER_VENV="${SHARED_CONDA_POOL_SERVER_VENV:-$(cd "${REPO_ROOT}/.." && pwd)/conda_envs/openclaw-worker-py312}"
POOL_SERVER_VENV_EXPLICIT="${POOL_SERVER_VENV:-}"
POOL_SERVER_VENV="${POOL_SERVER_VENV:-${DEFAULT_POOL_SERVER_VENV}}"
POOL_SERVER_MIN_PYTHON="${POOL_SERVER_MIN_PYTHON:-3.12}"
POOL_SERVER_CREATE_PYTHON="${POOL_SERVER_CREATE_PYTHON:-3.12}"
VENV_PYTHON="${POOL_SERVER_VENV}/bin/python"
REQUIRED_PY_MODULES=(terminal_bench fastapi uvicorn camel)
PIP_DEPENDENCIES=(
  "fastapi"
  "uvicorn"
  "camel-ai"
  "git+https://github.com/laude-institute/terminal-bench.git"
)

set_pool_server_venv() {
  POOL_SERVER_VENV="$1"
  VENV_PYTHON="${POOL_SERVER_VENV}/bin/python"
}

venv_python_ok() {
  venv_python_entry_ok && venv_python_home_ok && "$VENV_PYTHON" - "$POOL_SERVER_MIN_PYTHON" <<'PY'
import sys
min_version = tuple(int(part) for part in sys.argv[1].split("."))
raise SystemExit(0 if sys.version_info[:len(min_version)] >= min_version else 1)
PY
}

venv_python_entry_ok() {
  local target resolved
  [ -x "$VENV_PYTHON" ] || return 1
  if [ ! -L "$VENV_PYTHON" ]; then
    return 0
  fi
  target="$(readlink "$VENV_PYTHON" 2>/dev/null || true)"
  case "$target" in
    /*)
      return 1
      ;;
  esac
  resolved="$(readlink -f "$VENV_PYTHON" 2>/dev/null || true)"
  case "$resolved" in
    "${POOL_SERVER_VENV}/bin/"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

venv_python_home_ok() {
  local cfg home
  cfg="${POOL_SERVER_VENV}/pyvenv.cfg"
  [ -f "$cfg" ] || return 0
  home="$(awk -F= '$1 ~ /^[[:space:]]*home[[:space:]]*$/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$cfg")"
  case "$home" in
    */uv/python/*)
      [ "$home" = "${POOL_SERVER_VENV}/python/bin" ]
      ;;
    *)
      return 0
      ;;
  esac
}

venv_python_version() {
  if [ -L "$VENV_PYTHON" ]; then
    echo "symlink->$(readlink "$VENV_PYTHON" 2>/dev/null || echo unknown)"
  elif [ -x "$VENV_PYTHON" ]; then
    "$VENV_PYTHON" - <<'PY' 2>/dev/null || true
import sys
print(".".join(str(part) for part in sys.version_info[:3]))
PY
  else
    echo "missing"
  fi
}

venv_deps_ok() {
  "$VENV_PYTHON" - "$@" <<'PY'
import importlib.util
import sys
missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print("missing Python deps: " + ", ".join(missing))
    raise SystemExit(1)
PY
}

ensure_venv_pip() {
  if "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  echo "  [WARN] pip is missing in ${POOL_SERVER_VENV}; bootstrapping with ensurepip..."
  if "$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 &&
     "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "  [OK] pip bootstrapped"
    return 0
  fi
  return 1
}

activate_python_env() {
  if [ -f "${POOL_SERVER_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${POOL_SERVER_VENV}/bin/activate"
  else
    PATH="${POOL_SERVER_VENV}/bin:${PATH}"
    export PATH
  fi
}

remove_existing_venv() {
  case "$POOL_SERVER_VENV" in
    "${REPO_ROOT}/.venv"|${REPO_ROOT}/.venv-*)
      echo "  Removing existing invalid venv: $POOL_SERVER_VENV"
      rm -rf -- "$POOL_SERVER_VENV"
      ;;
    *)
      die "Refusing to remove non-standard venv path: ${POOL_SERVER_VENV}"
      ;;
  esac
}

copy_venv_python_binaries() {
  local exe path target tmp
  for exe in python python3 "python${POOL_SERVER_CREATE_PYTHON}"; do
    path="${POOL_SERVER_VENV}/bin/${exe}"
    if [ -L "$path" ]; then
      target="$(readlink -f "$path" 2>/dev/null || true)"
      if [ -z "$target" ] || [ ! -x "$target" ]; then
        die "Cannot replace symlink ${path}; target is missing: ${target:-unknown}"
      fi
      tmp="${path}.copy"
      cp -f "$target" "$tmp"
      chmod 0755 "$tmp"
      rm -f "$path"
      mv "$tmp" "$path"
    fi
  done
}

vendor_venv_python_home() {
  local cfg home base_dir vendored_home tmp
  cfg="${POOL_SERVER_VENV}/pyvenv.cfg"
  [ -f "$cfg" ] || return 0
  home="$(awk -F= '$1 ~ /^[[:space:]]*home[[:space:]]*$/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$cfg")"
  [ -n "$home" ] || return 0
  base_dir="${home%/bin}"
  case "$base_dir" in
    */uv/python/cpython-*|*/uv/python/pypy-*)
      ;;
    *)
      return 0
      ;;
  esac
  vendored_home="${POOL_SERVER_VENV}/python/bin"
  if [ "$home" != "$vendored_home" ]; then
    rm -rf -- "${POOL_SERVER_VENV}/python"
    cp -aL "$base_dir" "${POOL_SERVER_VENV}/python"
    tmp="${cfg}.tmp"
    awk -v new_home="$vendored_home" '
      /^[[:space:]]*home[[:space:]]*=/ { print "home = " new_home; next }
      { print }
    ' "$cfg" > "$tmp"
    mv "$tmp" "$cfg"
  fi
}

remove_venv_symlink_shims() {
  if [ -L "${POOL_SERVER_VENV}/lib64" ]; then
    rm -f -- "${POOL_SERVER_VENV}/lib64"
  fi
}

create_pool_server_venv() {
  local uv_venv_args backend
  echo "  Creating venv: $POOL_SERVER_VENV"
  backend="${POOL_SERVER_VENV_BACKEND:-python}"
  if [ "$backend" != "auto" ] && [ "$backend" != "uv" ] && [ "$backend" != "python" ]; then
    die "Unsupported POOL_SERVER_VENV_BACKEND=${backend}; expected auto|uv|python"
  fi
  if [ "$backend" != "python" ] && command -v uv &>/dev/null; then
    uv_venv_args=(venv --python "$POOL_SERVER_CREATE_PYTHON" --seed --link-mode copy)
    if uv venv --help 2>&1 | grep -q -- "--relocatable"; then
      uv_venv_args+=(--relocatable)
    fi
    if UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-openclaw}" UV_LINK_MODE=copy \
      uv "${uv_venv_args[@]}" "$POOL_SERVER_VENV"; then
      :
    elif [ "$backend" = "uv" ]; then
      return 1
    else
      echo "  [WARN] uv venv failed; falling back to python -m venv"
      if [ -d "$POOL_SERVER_VENV" ]; then
        remove_existing_venv
      fi
      create_pool_server_python_venv
    fi
  else
    if [ "$backend" = "uv" ]; then
      die "POOL_SERVER_VENV_BACKEND=uv requested but uv is not available"
    fi
    create_pool_server_python_venv
  fi
  vendor_venv_python_home
  copy_venv_python_binaries
  remove_venv_symlink_shims
}

create_pool_server_python_venv() {
  if command -v "python${POOL_SERVER_CREATE_PYTHON}" &>/dev/null; then
    "python${POOL_SERVER_CREATE_PYTHON}" -m venv --copies "$POOL_SERVER_VENV"
  else
    python3 -m venv --copies "$POOL_SERVER_VENV"
  fi
}

select_pool_server_env() {
  if [ -n "$POOL_SERVER_VENV_EXPLICIT" ]; then
    set_pool_server_venv "$POOL_SERVER_VENV_EXPLICIT"
    echo "  Pool server env explicitly set: ${POOL_SERVER_VENV}"
    return 0
  fi

  if [ -x "${SHARED_CONDA_POOL_SERVER_VENV}/bin/python" ]; then
    set_pool_server_venv "$SHARED_CONDA_POOL_SERVER_VENV"
    echo "  Checking shared conda env: ${SHARED_CONDA_POOL_SERVER_VENV}"
    if venv_python_ok && venv_deps_ok "${REQUIRED_PY_MODULES[@]}" >/dev/null 2>&1; then
      echo "  [OK] Using shared conda pool_server env: ${POOL_SERVER_VENV}"
      return 0
    fi
    echo "  [WARN] Shared conda env is present but incomplete or incompatible; falling back to install env."
  else
    echo "  Shared conda env not found: ${SHARED_CONDA_POOL_SERVER_VENV}"
  fi

  set_pool_server_venv "$DEFAULT_POOL_SERVER_VENV"
}

select_pool_server_env

if [ -d "$POOL_SERVER_VENV" ] && venv_python_ok; then
  echo "  [OK] Reusing existing venv: $POOL_SERVER_VENV"
else
  if [ -d "$POOL_SERVER_VENV" ]; then
    echo "  [WARN] Existing venv Python $(venv_python_version) does not satisfy >=${POOL_SERVER_MIN_PYTHON}"
    remove_existing_venv
  fi
  create_pool_server_venv
fi
venv_python_ok || die "Failed to create a usable Python >=${POOL_SERVER_MIN_PYTHON} venv: ${POOL_SERVER_VENV}"
activate_python_env
if ! ensure_venv_pip; then
  echo "  [WARN] Could not bootstrap pip in existing venv; recreating ${POOL_SERVER_VENV}"
  remove_existing_venv
  create_pool_server_venv
  venv_python_ok || die "Failed to create a usable Python >=${POOL_SERVER_MIN_PYTHON} venv: ${POOL_SERVER_VENV}"
  activate_python_env
  ensure_venv_pip || die "pip is missing in ${POOL_SERVER_VENV}; install python3-venv/ensurepip or use a Python build with pip support."
fi
PIP_INSTALL_ARGS=(--timeout "$PIP_TIMEOUT" --retries "$PIP_RETRIES" --progress-bar off)
if [ -n "${PROXY_URL:-}" ]; then
  PIP_INSTALL_ARGS+=(--proxy "$PROXY_URL")
  export PIP_PROXY="$PROXY_URL"
  echo "  Python deps install proxy: $PROXY_URL"
else
  unset PIP_PROXY
  echo "  Python deps install proxy: none"
fi

install_python_deps() {
  local no_deps="${1:-0}" backend
  backend="${POOL_SERVER_INSTALL_BACKEND:-pip}"
  if [ "$backend" != "auto" ] && [ "$backend" != "uv" ] && [ "$backend" != "pip" ]; then
    die "Unsupported POOL_SERVER_INSTALL_BACKEND=${backend}; expected auto|uv|pip"
  fi
  if [ "$backend" != "pip" ] && command -v uv &>/dev/null; then
    local uv_args=(pip install --python "$VENV_PYTHON" --link-mode copy --index-strategy unsafe-best-match)
    if [ "$no_deps" = "1" ]; then
      uv_args+=(--no-deps)
    fi
    if UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-openclaw}" UV_LINK_MODE=copy \
      uv "${uv_args[@]}" "${PIP_DEPENDENCIES[@]}"; then
      return 0
    fi
    if [ "$backend" = "uv" ]; then
      return 1
    fi
    echo "  [WARN] uv pip install failed; falling back to python -m pip" >&2
  elif [ "$backend" = "uv" ]; then
    die "POOL_SERVER_INSTALL_BACKEND=uv requested but uv is not available"
  fi

  local pip_args=("${PIP_INSTALL_ARGS[@]}")
  if [ "$no_deps" = "1" ]; then
    pip_args+=(--no-deps)
  fi
  "$VENV_PYTHON" -m pip install "${pip_args[@]}" "${PIP_DEPENDENCIES[@]}"
}

# Install pool_server dependencies
if venv_deps_ok "${REQUIRED_PY_MODULES[@]}"; then
  echo "  [OK] Required Python deps already installed"
else
  echo "  Installing missing dependencies into ${POOL_SERVER_VENV}..."
  INSTALL_LOG="$(mktemp /tmp/openclaw_python_deps.XXXXXX.log)"
  echo "  install log: ${INSTALL_LOG}"
  if install_python_deps 0 > "${INSTALL_LOG}" 2>&1; then
    tail -20 "${INSTALL_LOG}" | sed 's/^/    /'
  else
    echo "  [WARN] Python dependency install failed; last 80 log lines:"
    tail -80 "${INSTALL_LOG}" | sed 's/^/    /'
    echo "  Retrying without dependency resolution..."
    if install_python_deps 1 >> "${INSTALL_LOG}" 2>&1; then
      tail -20 "${INSTALL_LOG}" | sed 's/^/    /'
    else
      echo "  [FAIL] Python dependency install fallback failed; last 120 log lines:"
      tail -120 "${INSTALL_LOG}" | sed 's/^/    /'
      die "Python dependency installation failed. Full install log: ${INSTALL_LOG}"
    fi
  fi
  venv_deps_ok "${REQUIRED_PY_MODULES[@]}" || die "Python deps are still missing after install"
  echo "  [OK] Python deps installed"
fi
echo ""

# ── 7. Optional proxy hardening + apt-proxied base images ───────────
echo "=== 7. Proxy Hardening ==="
if [ "${RUN_PROXY_FIX}" = "1" ] && systemd_available; then
  echo "  Running fix_dockerd_and_proxy.sh (writes /etc/seta_build_proxy.env and wraps base images)..."
  if [ "$(id -u)" -eq 0 ]; then
    PROXY_URL="${PROXY_URL}" NO_PROXY_LIST="${NO_PROXY_LIST}" DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" SKIP_VERIFY="${SKIP_VERIFY}" \
      bash "${SCRIPT_DIR}/fix_dockerd_and_proxy.sh"
  else
    sudo env PROXY_URL="${PROXY_URL}" NO_PROXY_LIST="${NO_PROXY_LIST}" DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" SKIP_VERIFY="${SKIP_VERIFY}" \
      bash "${SCRIPT_DIR}/fix_dockerd_and_proxy.sh"
  fi
elif [ "${RUN_PROXY_FIX}" = "1" ]; then
  echo "  [SKIP] systemd is not available; /etc/seta_build_proxy.env was written in Step 4"
else
  echo "  [SKIP] RUN_PROXY_FIX=0"
fi
echo ""

# ── 8. Install docker-watchdog systemd service ──────────────────────
echo "=== 8. Docker Watchdog ==="
if [ "${INSTALL_WATCHDOG}" = "1" ] && systemd_available; then
  run_sudo cp "${SCRIPT_DIR}/docker-watchdog.service" /etc/systemd/system/docker-watchdog.service
  run_sudo systemctl daemon-reload
  run_sudo systemctl enable --now docker-watchdog
  if systemctl is-active --quiet docker-watchdog; then
    echo "  [OK] docker-watchdog active"
  else
    echo "  [WARN] docker-watchdog did not become active; inspect: journalctl -u docker-watchdog -n 80 --no-pager"
  fi
elif [ "${INSTALL_WATCHDOG}" = "1" ]; then
  echo "  [SKIP] systemd is not available; docker-watchdog.service cannot be installed"
else
  echo "  [SKIP] INSTALL_WATCHDOG=0"
fi
echo ""

# ── 9. Verify build works ───────────────────────────────────────────
echo "=== 9. Verify Docker Build ==="
if [ "${SKIP_VERIFY}" = "1" ]; then
  echo "  [SKIP] SKIP_VERIFY=1"
else
  export DATASET_DIR="terminal-rl/dataset"
  echo "  Building task 100 (simple task)..."
  BUILD_LOG="$(mktemp)"
  if timeout 300 docker compose -p test_build \
    -f terminal-rl/dataset/seta_env/100/docker-compose.yaml build > "${BUILD_LOG}" 2>&1; then
    tail -5 "${BUILD_LOG}"
    echo "  [OK] Build succeeded"
    timeout 60 docker compose -p test_build -f terminal-rl/dataset/seta_env/100/docker-compose.yaml down 2>/dev/null || true
  else
    tail -20 "${BUILD_LOG}"
    echo "  [WARN] Build failed - run: sudo bash terminal-rl/remote/fix_dockerd_and_proxy.sh"
  fi
  rm -f "${BUILD_LOG}"
fi
echo ""

# ── 10. Summary ─────────────────────────────────────────────────────
MY_IP=$(hostname -I | awk '{print $1}')
echo "============================================================"
echo " Setup Complete!"
echo ""
echo " Machine IP: $MY_IP"
DOCKER_ROOT_ACTUAL="$(timeout "${DOCKER_INFO_TIMEOUT}" docker info 2>/dev/null | awk -F': ' '/Docker Root Dir/{print $2; exit}' || echo unknown)"
COMPOSE_VERSION_SUMMARY="$(docker_compose_version | awk '{print $NF}' || echo unknown)"
echo " Docker root: ${DOCKER_ROOT_ACTUAL:-unknown}"
echo " Docker disk: $(df -h "$DOCKER_DATA_ROOT" --output=avail | tail -1 | xargs) available"
echo " Base images: pre-pulled"
echo " Compose V2: ${COMPOSE_VERSION_SUMMARY:-unknown}"
echo ""
echo " To start pool_server:"
echo "   cd $REPO_ROOT"
echo "   bash terminal-rl/remote/run_pool_server_pu_v2.sh"
echo ""
echo " Then on GPU worker, set:"
echo "   export WORKER_URLS=\"http://${MY_IP}:18081\""
echo "   bash terminal-rl/terminal-rl_qwen3-8b_pu.sh"
echo "============================================================"
