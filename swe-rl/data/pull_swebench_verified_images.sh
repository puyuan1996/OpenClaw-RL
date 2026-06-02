#!/usr/bin/env bash
# Pull SWE-bench Verified images from an image list.
#
# Usage:
#   IMAGE_LIST=~/swe_verified/docker_images.txt bash pull_swebench_verified_images.sh
#   N=100 IMAGE_LIST=~/swe_verified/docker_images.txt bash pull_swebench_verified_images.sh
#
# Optional proxy mode:
#   SRC_PREFIX=docker.io/swebench
#   PROXY_PREFIX=dockerproxy.net/swebench
#   -> pull proxy path, then tag back to canonical docker.io/swebench path

set -euo pipefail

IMAGE_LIST=${IMAGE_LIST:-}
N=${N:-0}
MAX_RETRIES=${MAX_RETRIES:-5}
RETRY_SLEEP=${RETRY_SLEEP:-15}
SRC_PREFIX=${SRC_PREFIX:-docker.io/swebench}
PROXY_PREFIX=${PROXY_PREFIX:-}
LOG_DIR=${LOG_DIR:-./output/swe_images}

if [[ -z "${IMAGE_LIST}" ]]; then
  echo "ERROR: IMAGE_LIST is required."
  echo "Example: IMAGE_LIST=~/swe_verified/docker_images.txt bash pull_swebench_verified_images.sh"
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/pull_verified_$(date +%F_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

python3 - <<PY
from pathlib import Path
import subprocess
import time
import sys

image_list = Path("${IMAGE_LIST}").expanduser().resolve()
n = int("${N}")
retries = int("${MAX_RETRIES}")
sleep_s = int("${RETRY_SLEEP}")
src_prefix = "${SRC_PREFIX}".rstrip("/")
proxy_prefix = "${PROXY_PREFIX}".rstrip("/")

if not image_list.exists():
    print(f"ERROR: IMAGE_LIST does not exist: {image_list}")
    sys.exit(1)

images = [x.strip() for x in image_list.read_text().splitlines() if x.strip()]
if n > 0:
    images = images[:n]

def to_proxy_image(dst: str) -> str:
    if not proxy_prefix:
        return dst
    needle = src_prefix + "/"
    if dst.startswith(needle):
        suffix = dst[len(needle):]
        return f"{proxy_prefix}/{suffix}"
    return dst

failed = []
total = len(images)
for i, dst in enumerate(images, start=1):
    src = to_proxy_image(dst)
    exists = subprocess.run(["docker", "image", "inspect", dst], capture_output=True).returncode == 0
    if exists:
        print(f"[{i}/{total}] skip (exists): {dst}", flush=True)
        continue

    ok = False
    print(f"[{i}/{total}] pull {src}", flush=True)
    for attempt in range(1, retries + 1):
        rc = subprocess.run(["docker", "pull", src]).returncode
        if rc == 0:
            ok = True
            break
        if attempt < retries:
            print(f"  attempt {attempt}/{retries} failed, retry in {sleep_s}s...", flush=True)
            time.sleep(sleep_s)
        else:
            print(f"  attempt {attempt}/{retries} failed", flush=True)

    if not ok:
        failed.append(src)
        continue

    if src != dst:
        subprocess.run(["docker", "tag", src, dst], check=True)
    print(f"[{i}/{total}] done: {dst}", flush=True)

print()
print(f"finished: {total - len(failed)}/{total} images OK")
if failed:
    print(f"failed ({len(failed)}):")
    for img in failed:
        print(f"  {img}")
    sys.exit(1)
PY

echo "log: ${LOG}"
