#!/bin/bash
# Setup ECS Docker node on a worker that has Docker installed.
# Run this script on the 4-GPU worker (or any machine with Docker).
#
# Usage:
#   bash setup_ecs_on_worker.sh          # pull all 100 images
#   N=5 bash setup_ecs_on_worker.sh      # pull first 5 images only

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SWE_RL_DIR="${SCRIPT_DIR}"
TRAIN="${TRAIN:-${SWE_RL_DIR}/swe_data/swe_gym_subset/train.jsonl}"
N="${N:-0}"

echo "========================================"
echo "SWE ECS Worker Setup"
echo "SWE_RL_DIR: ${SWE_RL_DIR}"
echo "TRAIN:      ${TRAIN}"
echo "N:          ${N}"
echo "========================================"

# Step 1: Check Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. This script must run on a machine with Docker."
    exit 1
fi
echo "[1/3] Docker OK: $(docker --version)"

# Step 2: Install Flask and start swe_exec_server
if ! python3 -c "import flask" 2>/dev/null; then
    echo "[2/3] Installing Flask..."
    pip3 install flask --ignore-installed blinker --trusted-host mirrors.i.h.pjlab.org.cn -i  http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/  2>/dev/null || pip3 install flask
else
    echo "[2/3] Flask already installed"
fi

# Kill any existing swe_exec_server
pkill -f "swe_exec_server.py" 2>/dev/null || true
sleep 1

echo "[2/3] Starting swe_exec_server on :5000..."
nohup python3 "${SWE_RL_DIR}/server/swe_exec_server.py" --port 5000 --host 0.0.0.0 \
    > /tmp/swe_exec_server.log 2>&1 &
SWE_PID=$!
echo "  PID: ${SWE_PID}, log: /tmp/swe_exec_server.log"

# Wait for server to be ready
for i in $(seq 1 15); do
    if curl -fsS http://localhost:5000/healthz >/dev/null 2>&1; then
        echo "  swe_exec_server is ready!"
        break
    fi
    if ! kill -0 "${SWE_PID}" 2>/dev/null; then
        echo "ERROR: swe_exec_server exited. Check /tmp/swe_exec_server.log:"
        cat /tmp/swe_exec_server.log
        exit 1
    fi
    sleep 1
done

if ! curl -fsS http://localhost:5000/healthz >/dev/null 2>&1; then
    echo "ERROR: swe_exec_server failed to start. Log:"
    cat /tmp/swe_exec_server.log
    exit 1
fi

# Step 3: Pull SWE-Bench Docker images
if [ ! -f "${TRAIN}" ]; then
    echo "ERROR: Training data not found: ${TRAIN}"
    echo "  Make sure the shared storage is mounted."
    exit 1
fi

echo "[3/3] Pulling SWE-Bench Docker images (N=${N})..."
N="${N}" TRAIN="${TRAIN}" bash "${SWE_RL_DIR}/data/pull_swe_images.sh"

echo ""
echo "========================================"
echo "Setup complete!"
echo "  Verify: curl http://localhost:5000/healthz"
echo "  Images: curl http://localhost:5000/images | python3 -m json.tool | head"
echo ""
echo "  On the 8-GPU training worker, set:"
echo "    SWE_EXEC_SERVER_URLS=http://$(hostname -I | awk '{print $1}'):5000"
echo "========================================"
