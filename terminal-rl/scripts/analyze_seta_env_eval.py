#!/usr/bin/env python3
"""Analyze terminal-rl SETA-env eval runs.

The script treats raw_score as the verifier accuracy signal and keeps shaped
task/total rewards as secondary diagnostics. Multiple runs can be merged in
order; later runs replace earlier rows with the same sample_index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAIL_RE = re.compile(
    r"\[task=(?P<task>\S+).*?uid=(?P<uid>\S+).*?sample_idx=(?P<sample_idx>[^\]]+)\] "
    r"Generate failed \((?P<error_type>[^)]+)\): (?P<error>.*)"
)
SUPPLEMENT_SAMPLE_RE = re.compile(r"__retry_s(?P<sample_index>\d+)(?:/)?$")


@dataclass
class RunInput:
    path: Path
    label: str


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _status_name(value: Any) -> str:
    text = str(value or "")
    if text.startswith("Status."):
        return text.split(".", 1)[1]
    return text or "UNKNOWN"


def _sha256(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _sample_identity_from_metadata(
    *,
    meta_path: Path | None = None,
    sample_metadata: dict[str, Any] | None = None,
    task_path: str = "",
    fallback_sample_index: int,
    fallback_task_name: str = "",
) -> dict[str, Any]:
    """Resolve supplement rows back to original dataset sample indices.

    Supplement datasets are small JSONL files, so runtime `sample_index` values
    are local to the supplement. The original full-dataset index is stored in
    `metadata.supplement_sample_index`; older aliases also encode it as
    `...__retry_sNNN`.
    """

    meta: dict[str, Any] = {}
    if sample_metadata is None and meta_path is not None and meta_path.exists():
        meta = _load_json(meta_path)
        sample_metadata = meta.get("sample_metadata") if isinstance(meta.get("sample_metadata"), dict) else {}
    sample_metadata = sample_metadata or {}

    resolved = _int_or_none(sample_metadata.get("supplement_sample_index"))
    source = "sample_metadata.supplement_sample_index" if resolved is not None else ""
    if resolved is None:
        m = SUPPLEMENT_SAMPLE_RE.search(task_path)
        if m:
            resolved = int(m.group("sample_index"))
            source = "task_path.retry_suffix"
    if resolved is None:
        resolved = fallback_sample_index
        source = "index.sample_index"

    original_task_name = str(sample_metadata.get("original_task_name") or fallback_task_name)
    original_task_path = str(sample_metadata.get("original_task_path") or task_path)

    return {
        "sample_index": resolved,
        "sample_index_source": source,
        "original_task_name": original_task_name,
        "original_task_path": original_task_path,
        "supplement_alias": bool(sample_metadata.get("supplement_alias")),
    }


def _run_sample_index_map(run: RunInput) -> dict[int, dict[str, Any]]:
    cfg = _load_json(run.path / "config" / "run_config.json")
    prompt_data = cfg.get("prompt_data")
    if not prompt_data:
        return {}
    path = Path(str(prompt_data))
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with path.open(errors="replace") as f:
        for run_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
            task_path = str(meta.get("task_path") or "")
            task_name = str(meta.get("task_name") or "")
            out[run_idx] = _sample_identity_from_metadata(
                sample_metadata=meta,
                task_path=task_path,
                fallback_sample_index=run_idx,
                fallback_task_name=task_name,
            )
    return out


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(errors="replace") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
            rows.append(
                {
                    "sample_index": i,
                    "task_name": str(meta.get("task_name", "")),
                    "task_path": str(meta.get("task_path", "")),
                    "instruction": str(meta.get("instruction", "")),
                    "data_source": str(meta.get("data_source", "")),
                }
            )
    return rows


def parse_failures(run: RunInput) -> list[dict[str, Any]]:
    log = run.path / "logs" / "train.log"
    if not log.exists():
        return []
    sample_map = _run_sample_index_map(run)
    out: list[dict[str, Any]] = []
    for line in log.open(errors="replace"):
        m = FAIL_RE.search(line)
        if not m:
            continue
        gd = m.groupdict()
        try:
            run_sample_index: int | None = int(gd["sample_idx"])
        except Exception:
            run_sample_index = None
        identity = sample_map.get(run_sample_index) if run_sample_index is not None else None
        sample_index = int(identity["sample_index"]) if identity else run_sample_index
        out.append(
            {
                "run_label": run.label,
                "task_name": gd["task"],
                "uid": gd["uid"],
                "run_sample_index": run_sample_index,
                "sample_index": sample_index,
                "sample_index_source": identity.get("sample_index_source") if identity else "log.sample_idx",
                "original_task_path": identity.get("original_task_path") if identity else "",
                "error_type": gd["error_type"],
                "error": gd["error"].strip(),
            }
        )
    return out


def load_index(run: RunInput) -> list[dict[str, Any]]:
    path = run.path / "trajectories" / "index.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            sample_index = d.get("sample_index")
            if not isinstance(sample_index, int):
                continue
            traj_path = Path(str(d.get("traj_path") or ""))
            meta_path = Path(str(d.get("meta_path") or ""))
            task_name = str(d.get("task_name") or "")
            task_path = str(d.get("task_path") or "")
            identity = _sample_identity_from_metadata(
                meta_path=meta_path,
                task_path=task_path,
                fallback_sample_index=sample_index,
                fallback_task_name=task_name,
            )
            raw_score = _safe_float(d.get("raw_score"))
            row = {
                "run_label": run.label,
                "run_dir": str(run.path),
                "index_line": line_no,
                "run_sample_index": sample_index,
                "sample_index": int(identity["sample_index"]),
                "sample_index_source": str(identity["sample_index_source"]),
                "task_name": str(identity["original_task_name"] or task_name),
                "task_path": str(identity["original_task_path"] or task_path),
                "result_task_name": task_name,
                "result_task_path": task_path,
                "supplement_alias": int(bool(identity["supplement_alias"])),
                "uid": str(d.get("uid") or ""),
                "status": _status_name(d.get("status")),
                "num_turns": _safe_float(d.get("num_turns")),
                "raw_score": raw_score,
                "raw_reward": _safe_float(d.get("raw_reward")),
                "task_reward": _safe_float(d.get("task_reward")),
                "total_reward": _safe_float(d.get("total_reward")),
                "exact_pass": int(raw_score == 1.0) if raw_score is not None else "",
                "nonzero_score": int(raw_score > 0.0) if raw_score is not None else "",
                "traj_path": str(traj_path) if str(traj_path) else "",
                "meta_path": str(meta_path) if str(meta_path) else "",
                "traj_sha256": _sha256(traj_path),
                "meta_sha256": _sha256(meta_path),
            }
            enrich_from_traj(row, traj_path)
            rows.append(row)
    return rows


def enrich_from_traj(row: dict[str, Any], path: Path) -> None:
    row.update(
        {
            "tool_calls": "",
            "parse_error_turns": "",
            "input_tokens": "",
            "output_tokens": "",
            "eval_error": "",
        }
    )
    if not path.exists():
        return
    d = _load_json(path)
    info = d.get("info") if isinstance(d.get("info"), dict) else {}
    turns = d.get("turns") if isinstance(d.get("turns"), list) else []
    tool_calls = 0
    parse_errors = 0
    input_tokens = 0
    output_tokens = 0
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        calls = turn.get("tool_calls")
        if isinstance(calls, list):
            tool_calls += len(calls)
        elif isinstance(turn.get("sdk_tool_calls_count"), int):
            tool_calls += int(turn["sdk_tool_calls_count"])
        if turn.get("parse_error_recorded"):
            parse_errors += 1
        for key, target in (("n_input_tokens", "input"), ("n_output_tokens", "output")):
            value = turn.get(key)
            if isinstance(value, (int, float)):
                if target == "input":
                    input_tokens += int(value)
                else:
                    output_tokens += int(value)
    row["tool_calls"] = tool_calls
    row["parse_error_turns"] = parse_errors
    row["input_tokens"] = input_tokens
    row["output_tokens"] = output_tokens
    row["eval_error"] = str(info.get("eval_error") or "")


def merge_rows(runs: list[RunInput]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[int, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for run_order, run in enumerate(runs):
        rows = load_index(run)
        for row in rows:
            row["run_order"] = run_order
            all_rows.append(row)
            selected[int(row["sample_index"])] = row
    return list(selected.values()), all_rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pct(num: int | float, den: int | float) -> float:
    return float(num) / float(den) * 100.0 if den else 0.0


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def build_outputs(
    dataset: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_by_idx = {int(r["sample_index"]): r for r in dataset}
    selected_by_idx = {int(r["sample_index"]): r for r in selected_rows}

    per_sample: list[dict[str, Any]] = []
    for idx in sorted(dataset_by_idx):
        base = dict(dataset_by_idx[idx])
        row = selected_by_idx.get(idx)
        if row:
            base.update(row)
            base["has_result"] = 1
        else:
            base.update(
                {
                    "run_label": "",
                    "run_dir": "",
                    "status": "MISSING",
                    "num_turns": "",
                    "raw_score": "",
                    "raw_reward": "",
                    "task_reward": "",
                    "total_reward": "",
                    "exact_pass": "",
                    "nonzero_score": "",
                    "tool_calls": "",
                    "parse_error_turns": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "eval_error": "",
                    "traj_path": "",
                    "meta_path": "",
                    "traj_sha256": "",
                    "meta_sha256": "",
                    "has_result": 0,
                }
            )
        per_sample.append(base)

    raw_scores = [_safe_float(r.get("raw_score")) for r in per_sample]
    raw_scores_f = [x for x in raw_scores if x is not None]
    exact = sum(1 for x in raw_scores_f if x == 1.0)
    nonzero = sum(1 for x in raw_scores_f if x and x > 0.0)
    total = len(dataset)
    missing = total - len(raw_scores_f)

    status_counts = Counter(str(r.get("status") or "UNKNOWN") for r in per_sample)
    run_counts = Counter(str(r.get("run_label") or "MISSING") for r in per_sample)
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_sample:
        task_rows[str(r.get("task_name") or "")].append(r)

    task_summary: list[dict[str, Any]] = []
    for task, rows in sorted(task_rows.items(), key=lambda kv: kv[0]):
        scores = [_safe_float(r.get("raw_score")) for r in rows]
        scores = [x for x in scores if x is not None]
        task_summary.append(
            {
                "task_name": task,
                "task_path": rows[0].get("task_path", ""),
                "n": len(rows),
                "result_n": len(scores),
                "missing_n": len(rows) - len(scores),
                "raw_score_mean": mean(scores),
                "exact_pass_n": sum(1 for x in scores if x == 1.0),
                "nonzero_n": sum(1 for x in scores if x > 0.0),
                "status_counts": ";".join(
                    f"{k}:{v}" for k, v in sorted(Counter(str(r.get("status")) for r in rows).items())
                ),
            }
        )

    failure_counts = Counter(str(f.get("error_type") or "") for f in failures)
    failure_by_sample = defaultdict(list)
    for f in failures:
        if f.get("sample_index") is not None:
            failure_by_sample[int(f["sample_index"])].append(f)

    summary = {
        "dataset_total": total,
        "result_count": len(raw_scores_f),
        "missing_count": missing,
        "raw_score_mean_completed_rows": mean(raw_scores_f),
        "raw_score_sum_completed_rows": sum(raw_scores_f),
        "raw_score_mean_all_dataset_missing_as_zero": sum(raw_scores_f) / total if total else None,
        "exact_pass_count": exact,
        "exact_pass_rate_completed_rows": exact / len(raw_scores_f) if raw_scores_f else None,
        "exact_pass_rate_all_dataset_missing_as_zero": exact / total if total else None,
        "nonzero_score_count": nonzero,
        "nonzero_score_rate_completed_rows": nonzero / len(raw_scores_f) if raw_scores_f else None,
        "nonzero_score_rate_all_dataset_missing_as_zero": nonzero / total if total else None,
        "raw_score_distribution": {str(k): v for k, v in sorted(Counter(raw_scores_f).items())},
        "status_counts": dict(status_counts),
        "run_counts": dict(run_counts),
        "failure_event_count": len(failures),
        "failure_event_counts": dict(failure_counts),
        "turns": quantiles([float(r["num_turns"]) for r in per_sample if _safe_float(r.get("num_turns")) is not None]),
        "tool_calls": quantiles([float(r["tool_calls"]) for r in per_sample if _safe_float(r.get("tool_calls")) is not None]),
        "input_tokens": quantiles([float(r["input_tokens"]) for r in per_sample if _safe_float(r.get("input_tokens")) is not None]),
        "output_tokens": quantiles([float(r["output_tokens"]) for r in per_sample if _safe_float(r.get("output_tokens")) is not None]),
    }

    write_csv(out_dir / "per_sample.csv", per_sample)
    write_csv(out_dir / "all_index_rows.csv", all_rows)
    write_csv(out_dir / "task_summary.csv", task_summary)
    write_csv(out_dir / "failure_events.csv", failures)
    write_csv(
        out_dir / "status_counts.csv",
        [{"status": k, "count": v, "pct_dataset": pct(v, total)} for k, v in status_counts.most_common()],
    )
    write_csv(
        out_dir / "failure_event_counts.csv",
        [{"error_type": k, "count": v} for k, v in failure_counts.most_common()],
    )

    selected_manifest = select_manifest_rows(per_sample, failure_by_sample)
    write_csv(out_dir / "selected_trajectory_manifest.csv", selected_manifest)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")
    make_plots(out_dir, summary, per_sample, task_summary)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def select_manifest_rows(
    per_sample: list[dict[str, Any]], failure_by_sample: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    scored = [r for r in per_sample if _safe_float(r.get("raw_score")) is not None]
    passes = [r for r in scored if _safe_float(r.get("raw_score")) == 1.0][:20]
    zeroes = [r for r in scored if _safe_float(r.get("raw_score")) == 0.0][:20]
    partial = [
        r
        for r in scored
        if (x := _safe_float(r.get("raw_score"))) is not None and 0.0 < x < 1.0
    ][:20]
    missing = [r for r in per_sample if not r.get("has_result")][:20]
    selected = passes + partial + zeroes + missing
    out = []
    seen: set[int] = set()
    for r in selected:
        idx = int(r["sample_index"])
        if idx in seen:
            continue
        seen.add(idx)
        failures = failure_by_sample.get(idx, [])
        out.append(
            {
                "sample_index": idx,
                "task_name": r.get("task_name", ""),
                "task_path": r.get("task_path", ""),
                "raw_score": r.get("raw_score", ""),
                "status": r.get("status", ""),
                "run_label": r.get("run_label", ""),
                "traj_path": r.get("traj_path", ""),
                "traj_sha256": r.get("traj_sha256", ""),
                "meta_path": r.get("meta_path", ""),
                "meta_sha256": r.get("meta_sha256", ""),
                "failure_types": ";".join(str(f.get("error_type") or "") for f in failures),
            }
        )
    return out


def render_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SETA-env Eval Analysis",
            "",
            f"- dataset_total: {summary['dataset_total']}",
            f"- result_count: {summary['result_count']}",
            f"- missing_count: {summary['missing_count']}",
            f"- raw_score_mean_completed_rows: {summary['raw_score_mean_completed_rows']}",
            f"- raw_score_mean_all_dataset_missing_as_zero: {summary['raw_score_mean_all_dataset_missing_as_zero']}",
            f"- exact_pass: {summary['exact_pass_count']} / {summary['dataset_total']}",
            f"- nonzero_score: {summary['nonzero_score_count']} / {summary['dataset_total']}",
            "",
            "Files:",
            "",
            "- `summary.json`: machine-readable aggregate metrics.",
            "- `per_sample.csv`: one merged row per dataset sample.",
            "- `all_index_rows.csv`: all trajectory index rows before merge.",
            "- `task_summary.csv`: grouped by task_name.",
            "- `failure_events.csv`: Generate failed events parsed from train logs.",
            "- `selected_trajectory_manifest.csv`: trajectory paths and hashes for audit sampling.",
        ]
    ) + "\n"


def make_plots(
    out_dir: Path,
    summary: dict[str, Any],
    per_sample: list[dict[str, Any]],
    task_summary: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    scores = [_safe_float(r.get("raw_score")) for r in per_sample]
    scores = [x for x in scores if x is not None]
    if scores:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(scores, bins=[-0.01, 0.01, 0.21, 0.41, 0.61, 0.81, 0.99, 1.01], color="#4c78a8")
        ax.set_title("SETA-env raw_score distribution")
        ax.set_xlabel("raw_score")
        ax.set_ylabel("samples")
        fig.tight_layout()
        fig.savefig(out_dir / "score_hist.png", dpi=180)
        plt.close(fig)

    status_counts = Counter(str(r.get("status") or "UNKNOWN") for r in per_sample)
    if status_counts:
        labels = [k for k, _ in status_counts.most_common()]
        values = [v for _, v in status_counts.most_common()]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#59a14f")
        ax.set_title("Rollout status counts")
        ax.set_ylabel("samples")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / "status_counts.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["mean raw_score", "exact pass", "nonzero"]
    values = [
        summary["raw_score_mean_all_dataset_missing_as_zero"] or 0.0,
        summary["exact_pass_rate_all_dataset_missing_as_zero"] or 0.0,
        summary["nonzero_score_rate_all_dataset_missing_as_zero"] or 0.0,
    ]
    ax.bar(labels, [v * 100 for v in values], color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylabel("percent")
    ax.set_title("SETA-env accuracy summary")
    for i, v in enumerate(values):
        ax.text(i, v * 100, f"{v * 100:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "acc_summary.png", dpi=180)
    plt.close(fig)

    scored_tasks = [r for r in task_summary if _safe_float(r.get("raw_score_mean")) is not None]
    if scored_tasks:
        scored_tasks.sort(key=lambda r: float(r["raw_score_mean"]), reverse=True)
        show = scored_tasks[:15]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh([str(r["task_name"]) for r in reversed(show)], [float(r["raw_score_mean"]) for r in reversed(show)])
        ax.set_xlabel("mean raw_score")
        ax.set_title("Top task mean raw_score")
        fig.tight_layout()
        fig.savefig(out_dir / "top_tasks.png", dpi=180)
        plt.close(fig)


def parse_run_arg(raw: str) -> RunInput:
    if "=" in raw:
        label, path = raw.split("=", 1)
    else:
        path = raw
        label = Path(path).name
    return RunInput(Path(path).resolve(), label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, help="LABEL=/path/to/run; later runs override earlier rows")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    runs = [parse_run_arg(x) for x in args.run]
    dataset = load_dataset(args.dataset)
    selected_rows, all_rows = merge_rows(runs)
    failures: list[dict[str, Any]] = []
    for run in runs:
        failures.extend(parse_failures(run))
    summary = build_outputs(dataset, selected_rows, all_rows, failures, args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
