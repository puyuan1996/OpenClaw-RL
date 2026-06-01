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
#   NO_PROXY_LIST      - no_proxy list for internal network bypass.
#   DOCKER_DATA_ROOT   - Docker data root. DOCKER_ROOT is accepted as legacy alias.
#   ASSUME_YES         - 1 to continue through low-disk warning non-interactively.
#   RUN_PROXY_FIX      - 1 to run fix_dockerd_and_proxy.sh at the end. Default: 1.
#   INSTALL_WATCHDOG   - 1 to install and start docker-watchdog.service. Default: 1.
#   SKIP_VERIFY        - 1 to skip docker build verification. Default: 0.
#   DOCKER_INFO_TIMEOUT - Timeout for docker info probes. Default: 10.
#   DOCKER_PULL_TIMEOUT - Timeout for docker pulls. Default: 900.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "[ERROR] $*"; exit 1; }
docker_info_ok() { timeout "${DOCKER_INFO_TIMEOUT}" docker info >/dev/null 2>&1; }
docker_compose_version() { timeout 10 docker compose version 2>/dev/null; }
run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}
force_restart_docker() {
  if [ "$(id -u)" -eq 0 ]; then
    DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" bash "${SCRIPT_DIR}/restart_docker_force.sh"
  else
    sudo env DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" bash "${SCRIPT_DIR}/restart_docker_force.sh"
  fi
}

echo "============================================================"
echo " Terminal-RL Pool Server - New Worker Setup"
echo " $(date)"
echo "============================================================"
echo ""

# ── 0. Detect proxy ─────────────────────────────────────────────────
PROXY_URL="${PROXY_URL:-}"
NO_PROXY_LIST="${NO_PROXY_LIST:-localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc}"
if [ -z "$PROXY_URL" ]; then
  # Try the pjlab proxy setup script
  if curl -fsS --max-time 3 "http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh" >/dev/null 2>&1; then
    PROXY_URL="http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
    echo "[auto] Detected pjlab proxy: $PROXY_URL"
  fi
fi
if [ -n "$PROXY_URL" ]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
  export no_proxy="$NO_PROXY_LIST"
  export NO_PROXY="$no_proxy"
fi
ASSUME_YES="${ASSUME_YES:-0}"
RUN_PROXY_FIX="${RUN_PROXY_FIX:-1}"
INSTALL_WATCHDOG="${INSTALL_WATCHDOG:-1}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
DOCKER_INFO_TIMEOUT="${DOCKER_INFO_TIMEOUT:-10}"
DOCKER_PULL_TIMEOUT="${DOCKER_PULL_TIMEOUT:-900}"

# ── 1. Check disk space ─────────────────────────────────────────────
echo "=== 1. Disk Space Check ==="
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/var/lib/docker}}"
DOCKER_ROOT="${DOCKER_DATA_ROOT}"
run_sudo mkdir -p "$DOCKER_DATA_ROOT"
DOCKER_PARTITION=$(df "$DOCKER_DATA_ROOT" --output=target 2>/dev/null | tail -1)
AVAIL_GB=$(df -BG --output=avail "$DOCKER_DATA_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')

echo "  Docker data root: $DOCKER_DATA_ROOT"
echo "  Partition: $DOCKER_PARTITION"
echo "  Available: ${AVAIL_GB}GB"
echo ""

if [ "${AVAIL_GB:-0}" -lt 150 ]; then
  echo "  [WARN] Less than 150GB available. Full training needs ~100-200GB for docker images."
  echo "         Consider setting DOCKER_ROOT to a larger partition."
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
  run_sudo systemctl enable docker
  run_sudo systemctl start docker
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
  run_sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<EOF
[Service]
Environment="HTTP_PROXY=${PROXY_URL}"
Environment="HTTPS_PROXY=${PROXY_URL}"
Environment="NO_PROXY=${NO_PROXY_LIST}"
EOF
  echo "  [OK] Docker daemon proxy configured"
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
  "storage-driver": "overlay2",
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
echo "  [OK] Docker daemon.json configured (data-root=${DOCKER_DATA_ROOT}, address-pool=10.200.0.0/12)"

# 4c. Restart docker
echo "  Restarting docker daemon..."
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
if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
  echo "  [OK] .venv exists"
else
  echo "  Creating .venv..."
  if command -v uv &>/dev/null; then
    uv venv .venv --python 3.12
  else
    python3 -m venv .venv
  fi
fi
source .venv/bin/activate

# Install pool_server dependencies
echo "  Installing dependencies..."
pip install --quiet terminal-bench fastapi uvicorn camel-ai 2>&1 | tail -3 || \
  pip install --quiet terminal-bench fastapi uvicorn camel-ai --no-deps 2>&1 | tail -3
echo "  [OK] Python deps installed"
echo ""

# ── 7. Optional proxy hardening + apt-proxied base images ───────────
echo "=== 7. Proxy Hardening ==="
if [ "${RUN_PROXY_FIX}" = "1" ]; then
  echo "  Running fix_dockerd_and_proxy.sh (writes /etc/seta_build_proxy.env and wraps base images)..."
  if [ "$(id -u)" -eq 0 ]; then
    PROXY_URL="${PROXY_URL}" NO_PROXY_LIST="${NO_PROXY_LIST}" DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" SKIP_VERIFY="${SKIP_VERIFY}" \
      bash "${SCRIPT_DIR}/fix_dockerd_and_proxy.sh"
  else
    sudo env PROXY_URL="${PROXY_URL}" NO_PROXY_LIST="${NO_PROXY_LIST}" DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" SKIP_VERIFY="${SKIP_VERIFY}" \
      bash "${SCRIPT_DIR}/fix_dockerd_and_proxy.sh"
  fi
else
  echo "  [SKIP] RUN_PROXY_FIX=0"
fi
echo ""

# ── 8. Install docker-watchdog systemd service ──────────────────────
echo "=== 8. Docker Watchdog ==="
if [ "${INSTALL_WATCHDOG}" = "1" ]; then
  run_sudo cp "${SCRIPT_DIR}/docker-watchdog.service" /etc/systemd/system/docker-watchdog.service
  run_sudo systemctl daemon-reload
  run_sudo systemctl enable --now docker-watchdog
  if systemctl is-active --quiet docker-watchdog; then
    echo "  [OK] docker-watchdog active"
  else
    echo "  [WARN] docker-watchdog did not become active; inspect: journalctl -u docker-watchdog -n 80 --no-pager"
  fi
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
