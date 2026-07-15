#!/usr/bin/env python3
"""Create a SETA-env supplement dataset for samples missing eval results.

By default the script keeps metadata.task_name unchanged and rewrites only
metadata.task_path to a symlink alias. Terminal-Bench image names are derived
from the task_path basename, so this avoids worker-side build blacklist/cache
without changing the task name shown to the agent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAIL_RE = re.compile(
    r"\[task=(?P<task>\S+).*?uid=(?P<uid>\S+).*?sample_idx=(?P<sample_idx>[^\]]+)\] "
    r"Generate failed \((?P<error_type>[^)]+)\): (?P<error>.*)"
)


@dataclass
class FailedEvent:
    task_name: str
    uid: str
    sample_index: int | None
    error_type: str
    error: str


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_index_sample_indices(run: Path) -> set[int]:
    p = run / "trajectories" / "index.jsonl"
    seen: set[int] = set()
    if not p.exists():
        return seen
    with p.open(errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            idx = d.get("sample_index")
            if isinstance(idx, int):
                seen.add(idx)
    return seen


def parse_failed_events(run: Path) -> list[FailedEvent]:
    p = run / "logs" / "train.log"
    events: list[FailedEvent] = []
    if not p.exists():
        return events
    with p.open(errors="replace") as f:
        for line in f:
            m = FAIL_RE.search(line)
            if not m:
                continue
            gd = m.groupdict()
            try:
                idx: int | None = int(gd["sample_idx"])
            except Exception:
                idx = None
            events.append(
                FailedEvent(
                    task_name=gd["task"],
                    uid=gd["uid"],
                    sample_index=idx,
                    error_type=gd["error_type"],
                    error=gd["error"].strip(),
                )
            )
    return events


def safe_alias_name(task_name: str, sample_index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_name).strip("-_.")
    if not base:
        base = "task"
    return f"{base}__retry_s{sample_index}"


def ensure_alias(
    dataset_root: Path,
    alias_rel_path: str,
    original_rel_path: str,
    *,
    copy: bool,
) -> None:
    alias_abs = dataset_root / alias_rel_path
    original_abs = dataset_root / original_rel_path
    if not original_abs.exists():
        raise FileNotFoundError(f"original task path not found: {original_abs}")
    alias_abs.parent.mkdir(parents=True, exist_ok=True)
    if alias_abs.exists() or alias_abs.is_symlink():
        return
    if copy:
        import shutil

        shutil.copytree(original_abs, alias_abs, symlinks=True)
    else:
        rel_target = os.path.relpath(original_abs, alias_abs.parent)
        alias_abs.symlink_to(rel_target, target_is_directory=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, type=Path)
    ap.add_argument("--dataset-root", default=Path("terminal-rl/dataset"), type=Path)
    ap.add_argument("--alias-prefix", required=True, help="directory under dataset root, e.g. seta_env_retry/runid")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--include-result-failed-events", action="store_true")
    ap.add_argument("--copy", action="store_true", help="copy task dirs instead of symlinking")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    result_indices: set[int] = set()
    events: list[FailedEvent] = []
    for run in args.run:
        result_indices.update(parse_index_sample_indices(run))
        events.extend(parse_failed_events(run))

    failed_indices = {e.sample_index for e in events if e.sample_index is not None}
    missing_indices = set(range(len(dataset))) - result_indices
    wanted = set(missing_indices)
    if args.include_result_failed_events:
        wanted.update(int(i) for i in failed_indices if i is not None)

    out_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    dataset_root = args.dataset_root.resolve()
    for idx in sorted(wanted):
        row = json.loads(json.dumps(dataset[idx], ensure_ascii=False))
        meta = row.setdefault("metadata", {})
        if not isinstance(meta, dict):
            raise TypeError(f"row {idx} metadata is not an object")
        original_task_name = str(meta.get("task_name") or "")
        original_task_path = str(meta.get("task_path") or "")
        if not original_task_path:
            raise ValueError(f"row {idx} has no metadata.task_path")
        alias_name = safe_alias_name(original_task_name or Path(original_task_path).name, idx)
        alias_rel_path = f"{args.alias_prefix.strip('/')}/{alias_name}"
        ensure_alias(
            dataset_root,
            alias_rel_path,
            original_task_path,
            copy=args.copy,
        )
        meta["original_task_name"] = original_task_name
        meta["original_task_path"] = original_task_path
        meta["task_path"] = alias_rel_path
        meta["supplement_sample_index"] = idx
        meta["supplement_alias"] = True
        row["metadata"] = meta
        out_rows.append(row)
        matching = [e for e in events if e.sample_index == idx]
        manifest_rows.append(
            {
                "sample_index": idx,
                "task_name": original_task_name,
                "original_task_path": original_task_path,
                "alias_task_path": alias_rel_path,
                "reason": "missing_result" if idx in missing_indices else "failed_event",
                "failed_event_count": len(matching),
                "failed_event_types": ";".join(sorted({e.error_type for e in matching})),
                "failed_event_uids": ";".join(e.uid for e in matching[:10]),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "dataset_rows": len(dataset),
                "result_indices": len(result_indices),
                "failed_events": len(events),
                "missing_indices": len(missing_indices),
                "supplement_rows": len(out_rows),
                "out": str(args.out),
                "manifest": str(args.manifest),
                "alias_prefix": args.alias_prefix,
                "copy": args.copy,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
