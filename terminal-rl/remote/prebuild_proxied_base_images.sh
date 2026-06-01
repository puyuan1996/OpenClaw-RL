#!/usr/bin/env bash
# prebuild_proxied_base_images.sh — Wrap base images with apt proxy injection.
#
# Why this exists:
#   On Ubuntu 24.04 (noble), apt-get does NOT honor the HTTP_PROXY env var by
#   default. It only reads /etc/apt/apt.conf.d/*. So even if dockerd / build
#   ENV / ~/.docker/config.json all have HTTP_PROXY set correctly, `apt-get
#   update` inside the build still tries direct egress to archive.ubuntu.com
#   and times out behind a firewall.
#
#   seta_env/0_my/Dockerfile works because it explicitly writes
#   /etc/apt/apt.conf.d/95proxies. The original 1377 seta_env Dockerfiles do
#   NOT, and we cannot patch all of them.
#
# Solution:
#   For each base image tag used in the dataset (top frequencies below),
#   build a local "shadow" image with the SAME tag that adds:
#     - /etc/apt/apt.conf.d/95proxies            (apt http/https proxy)
#     - ENV HTTP_PROXY/HTTPS_PROXY               (for non-apt tools)
#     - DEBIAN_FRONTEND=noninteractive           (matches 0_my behavior)
#
#   Docker prefers local images over registry pulls, so all subsequent
#   `FROM <tag>` builds inherit the proxy config WITHOUT modifying ANY
#   of the 1377 Dockerfiles.
#
# Caveat:
#   `docker pull <tag>` will overwrite our shadow back to upstream.
#   Re-run this script if base images get re-pulled.
#
# Usage:
#   sudo bash terminal-rl/remote/prebuild_proxied_base_images.sh
#
# Env:
#   PROXY_URL       default: pjlab proxy
#   NO_PROXY_LIST   default: internal loopback / pjlab network bypass list
#   BASE_IMAGES     space-separated override (default: top 4 from seta_env scan)

set -uo pipefail

PROXY_URL="${PROXY_URL:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}"
NO_PROXY_LIST="${NO_PROXY_LIST:-localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc}"

# Frequencies measured in seta_env/*/Dockerfile (1377 total):
#   1317  ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624
#     45  ubuntu:22.04
#      8  ghcr.io/laude-institute/t-bench/python-3-13:20250620
#      4  ubuntu:24.04
DEFAULT_BASE_IMAGES=(
  "ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624"
  "ubuntu:22.04"
  "ghcr.io/laude-institute/t-bench/python-3-13:20250620"
  "ubuntu:24.04"
)
if [ -n "${BASE_IMAGES:-}" ]; then
  read -r -a BASE_IMAGES_ARR <<< "${BASE_IMAGES}"
else
  BASE_IMAGES_ARR=("${DEFAULT_BASE_IMAGES[@]}")
fi

log() { echo "[$(date '+%F %T')] $*"; }

if [ "$(id -u)" -ne 0 ]; then
  log "[ERROR] must run as root (so docker socket is accessible)"; exit 1
fi
if ! timeout 5 docker info >/dev/null 2>&1; then
  log "[ERROR] docker daemon not responsive; run fix_dockerd_and_proxy.sh first"; exit 2
fi

log "Wrapping ${#BASE_IMAGES_ARR[@]} base image(s) with apt proxy injection"
log "  PROXY_URL=${PROXY_URL}"

WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

# Use legacy builder to match what pool_server uses (and avoid BuildKit
# frontend image pull dependency).
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

OK=0
FAIL=0
for BASE in "${BASE_IMAGES_ARR[@]}"; do
  log "─── ${BASE} ──────────────────────────────────────────"

  # 1. Make sure original image exists locally; if not, pull it through proxy.
  if ! docker image inspect "${BASE}" >/dev/null 2>&1; then
    log "  not present locally, pulling ..."
    if ! HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}" \
         NO_PROXY="${NO_PROXY_LIST}" \
         timeout 600 docker pull "${BASE}"; then
      log "  [FAIL] pull ${BASE}"
      FAIL=$((FAIL+1))
      continue
    fi
  fi

  # 2. Build a Dockerfile that mirrors what 0_my/Dockerfile does, but FROM
  #    the existing base. Tag the result with the SAME tag as the base.
  cat > "${WORK}/Dockerfile" <<EOF
FROM ${BASE}

# Match seta_env/0_my/Dockerfile L18-35 behavior:
ARG HTTP_PROXY="${PROXY_URL}"
ARG HTTPS_PROXY="${PROXY_URL}"
ARG NO_PROXY="${NO_PROXY_LIST}"

ENV HTTP_PROXY=\$HTTP_PROXY
ENV HTTPS_PROXY=\$HTTPS_PROXY
ENV http_proxy=\$HTTP_PROXY
ENV https_proxy=\$HTTPS_PROXY
ENV NO_PROXY=\$NO_PROXY
ENV no_proxy=\$NO_PROXY
ENV DEBIAN_FRONTEND=noninteractive

# Critical: apt does NOT read HTTP_PROXY env var on Ubuntu 24.04.
# Configure apt explicitly via apt.conf.d. This is the missing piece
# that distinguishes seta_env/0 (fails) from seta_env/0_my (works).
RUN if [ -n "\$HTTP_PROXY" ]; then \\
        echo 'Acquire::http::Proxy "'\$HTTP_PROXY'";'  > /etc/apt/apt.conf.d/95proxies && \\
        echo 'Acquire::https::Proxy "'\$HTTPS_PROXY'";' >> /etc/apt/apt.conf.d/95proxies; \\
    fi
EOF

  # 3. Build and tag with the SAME name to shadow the upstream tag.
  log "  building proxied wrapper ..."
  if timeout 300 docker build \
       --build-arg "HTTP_PROXY=${PROXY_URL}" \
       --build-arg "HTTPS_PROXY=${PROXY_URL}" \
       --build-arg "NO_PROXY=${NO_PROXY_LIST}" \
       -t "${BASE}" \
       -f "${WORK}/Dockerfile" "${WORK}" \
       2>&1 | tail -3 | sed 's/^/    /'; then
    log "  [OK] ${BASE} now has apt proxy injected"

    # 4. Quick smoke test: apt-get update inside this image should succeed.
    if docker run --rm "${BASE}" sh -c 'apt-get update >/dev/null 2>&1' 2>/dev/null; then
      log "  [OK] apt-get update verified inside ${BASE}"
      OK=$((OK+1))
    else
      log "  [WARN] apt-get update still failing in shadow image; proxy may still be wrong"
      FAIL=$((FAIL+1))
    fi
  else
    log "  [FAIL] build wrapper for ${BASE}"
    FAIL=$((FAIL+1))
  fi
done

log "─── summary ──────────────────────────────────────────"
log "  ok=${OK}  fail=${FAIL}  total=${#BASE_IMAGES_ARR[@]}"
if [ "${FAIL}" -gt 0 ]; then
  log "  Some images failed; seta_env tasks using them will still fail to build."
  exit 3
fi
log "  All shadow images built. Now retry seta_env/0 build:"
log "    cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL"
log "    set -a; . /etc/seta_build_proxy.env; set +a"
log "    DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 \\"
log "      docker compose -p t0 -f terminal-rl/dataset/seta_env/0/docker-compose.yaml build"
