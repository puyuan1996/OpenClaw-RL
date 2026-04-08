#!/bin/bash
# 在宿主机（有网+有Docker daemon）上执行，拉取所有 SWE-Bench 镜像。
# GPU worker 通过 Docker Remote API 共享同一个 daemon，镜像自动可见。
#
# Usage:
#   bash pull_images_on_host.sh          # 拉取全部 100 个
#   N=5 bash pull_images_on_host.sh      # 先拉 5 个测试

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TRAIN="${TRAIN:-${SCRIPT_DIR}/swe_data/swe_gym_subset/train.jsonl}"
N="${N:-0}"

echo "========================================"
echo "Pull SWE images on host machine"
echo "TRAIN: ${TRAIN}"
echo "N:     ${N} (0=all)"
echo "========================================"

if [ ! -f "${TRAIN}" ]; then
    echo "ERROR: ${TRAIN} not found"
    exit 1
fi

TRAIN="${TRAIN}" N="${N}" bash "${SCRIPT_DIR}/data/pull_swe_images.sh"
