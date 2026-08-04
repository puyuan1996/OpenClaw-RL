#!/usr/bin/env python3
"""Audit overlay2 directories that are unreachable from Docker layerdb metadata.

Default mode is read-only. The quarantine mode is intentionally offline-only:
stop pool_server/watchdog/docker first, move candidates aside, restart Docker,
and only delete the quarantine after validation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def read_one_line(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return ""


def docker_is_running(timeout: int = 5) -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
        )
        return True
    except Exception:
        return False


def du_bytes(path: Path, timeout: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["du", "-xsB1", str(path)],
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            text=True,
        )
        return int(out.split()[0])
    except Exception:
        return None


def overlay_dirs(overlay: Path) -> set[str]:
    return {
        p.name
        for p in overlay.iterdir()
        if p.is_dir(follow_symlinks=False) and p.name != "l"
    }


def layerdb_ids(layerdb: Path) -> tuple[set[str], set[str], int, int]:
    cache_ids: set[str] = set()
    mount_ids: set[str] = set()
    cache_files = 0
    mount_files = 0
    for p in layerdb.rglob("cache-id"):
        if p.is_file():
            cache_files += 1
            value = read_one_line(p)
            if value:
                cache_ids.add(value)
    mounts = layerdb / "mounts"
    if mounts.exists():
        for p in mounts.rglob("mount-id"):
            if p.is_file():
                mount_files += 1
                value = read_one_line(p)
                if value:
                    mount_ids.add(value)
    return cache_ids, mount_ids, cache_files, mount_files


def resolve_lower_entry(overlay: Path, entry: str) -> str | None:
    entry = entry.strip()
    if not entry:
        return None
    target: Path
    if entry.startswith("l/"):
        target = overlay / entry
    elif entry.startswith("/"):
        target = Path(entry)
    else:
        target = overlay / entry
    try:
        resolved = target.resolve(strict=False)
    except OSError:
        return None
    parts = resolved.parts
    for idx, part in enumerate(parts):
        if part == "overlay2" and idx + 1 < len(parts):
            layer = parts[idx + 1]
            if layer != "l":
                return layer
    return None


def reachable_overlay_dirs(overlay: Path, roots: set[str], all_dirs: set[str]) -> set[str]:
    reachable: set[str] = set()
    queue = deque([x for x in roots if x in all_dirs])
    while queue:
        layer = queue.popleft()
        if layer in reachable:
            continue
        reachable.add(layer)
        lower = overlay / layer / "lower"
        if not lower.exists():
            continue
        try:
            entries = lower.read_text(errors="replace").strip().split(":")
        except OSError:
            continue
        for entry in entries:
            parent = resolve_lower_entry(overlay, entry)
            if parent and parent in all_dirs and parent not in reachable:
                queue.append(parent)
    return reachable


def mounted_overlay_dirs(docker_root: Path, all_dirs: set[str]) -> set[str]:
    mounted: set[str] = set()
    try:
        text = Path("/proc/self/mountinfo").read_text(errors="replace")
    except OSError:
        return mounted
    prefix = str(docker_root / "overlay2") + "/"
    for line in text.splitlines():
        if prefix not in line:
            continue
        for layer in all_dirs:
            if f"{prefix}{layer}/" in line or line.endswith(f"{prefix}{layer}"):
                mounted.add(layer)
    return mounted


def move_to_quarantine(
    docker_root: Path,
    candidates: list[str],
    quarantine: Path,
    size_timeout: int,
) -> None:
    overlay = docker_root / "overlay2"
    q_overlay = quarantine / "overlay2"
    q_links = quarantine / "l"
    q_overlay.mkdir(parents=True, exist_ok=True)
    q_links.mkdir(parents=True, exist_ok=True)
    manifest = []
    for layer in candidates:
        src = overlay / layer
        if not src.exists():
            continue
        size = du_bytes(src, size_timeout)
        link_name = read_one_line(src / "link")
        dst = q_overlay / layer
        print(f"quarantine {src} -> {dst}")
        src.rename(dst)
        link_moved = None
        if link_name:
            link_src = overlay / "l" / link_name
            if link_src.exists() or link_src.is_symlink():
                link_dst = q_links / link_name
                link_src.rename(link_dst)
                link_moved = str(link_dst)
        manifest.append(
            {
                "layer": layer,
                "size_bytes": size,
                "from": str(src),
                "to": str(dst),
                "link": link_name,
                "link_to": link_moved,
            }
        )
    (quarantine / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote manifest: {quarantine / 'manifest.json'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-root", default=os.environ.get("DOCKER_DATA_ROOT", "/data"))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--size-timeout", type=int, default=20)
    parser.add_argument("--size-top", action="store_true", help="run du for top candidates")
    parser.add_argument("--quarantine", action="store_true", help="move candidates aside")
    parser.add_argument("--max-quarantine", type=int, default=0, help="0 means all candidates")
    parser.add_argument("--quarantine-dir", default="")
    parser.add_argument("--allow-running-dockerd", action="store_true")
    args = parser.parse_args()

    docker_root = Path(args.docker_root)
    overlay = docker_root / "overlay2"
    layerdb = docker_root / "image" / "overlay2" / "layerdb"
    if not overlay.is_dir():
        print(f"missing overlay2 directory: {overlay}", file=sys.stderr)
        return 2
    if not layerdb.is_dir():
        print(f"missing layerdb directory: {layerdb}", file=sys.stderr)
        return 2

    all_dirs = overlay_dirs(overlay)
    cache_ids, mount_ids, cache_files, mount_files = layerdb_ids(layerdb)
    roots = cache_ids | mount_ids
    reachable = reachable_overlay_dirs(overlay, roots, all_dirs)
    mounted = mounted_overlay_dirs(docker_root, all_dirs)
    candidates = sorted((all_dirs - reachable) - mounted)

    print("overlay2 orphan audit")
    print(f"docker_root={docker_root}")
    print(f"overlay2_dirs={len(all_dirs)}")
    print(f"cache_id_files={cache_files} unique_cache_ids={len(cache_ids)}")
    print(f"mount_id_files={mount_files} unique_mount_ids={len(mount_ids)}")
    print(f"reachable_dirs={len(reachable)}")
    print(f"mounted_excluded={len(mounted)}")
    print(f"orphan_candidates={len(candidates)}")

    if args.size_top and candidates:
        rows = []
        for layer in candidates[: args.top_n * 4]:
            size = du_bytes(overlay / layer, args.size_timeout)
            rows.append((-1 if size is None else size, layer))
        rows.sort(reverse=True)
        print("largest candidate sizes:")
        for size, layer in rows[: args.top_n]:
            size_text = "timeout" if size < 0 else f"{size / 1024**3:.2f}G"
            print(f"  {size_text} {layer}")
    else:
        print("candidate sample:")
        for layer in candidates[: args.top_n]:
            print(f"  {layer}")

    if not args.quarantine:
        print("read-only mode; pass --quarantine after stopping Docker to move candidates aside")
        return 0

    if docker_is_running() and not args.allow_running_dockerd:
        print("refusing quarantine while Docker daemon is running", file=sys.stderr)
        print("stop pool_server/watchdog/docker first, or pass --allow-running-dockerd only if you accept the risk", file=sys.stderr)
        return 3

    selected = candidates if args.max_quarantine <= 0 else candidates[: args.max_quarantine]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    quarantine = Path(args.quarantine_dir) if args.quarantine_dir else docker_root / f"overlay2.orphan-quarantine.{stamp}"
    move_to_quarantine(docker_root, selected, quarantine, args.size_timeout)
    print(f"moved {len(selected)} candidates to {quarantine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
