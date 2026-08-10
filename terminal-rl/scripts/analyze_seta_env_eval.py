#!/usr/bin/env python3
"""Merge SETA-env eval runs into per-sample rows and aggregate accuracy.

A full SETA-env eval rarely lands in one pass: remote Docker resets fail for
some samples, those are retried in supplement runs, and the final accuracy has
to be computed over the union. This script does that merge and the aggregation.

The reported metric is ``raw_score``, the fraction of the task's own verifier
checks that passed, as returned by the SETA verifier. Shaped rewards
(``task_reward`` / ``total_reward``) are carried through for reference but are
never used as accuracy. ``exact_pass`` means ``raw_score == 1.0``.

Usage:

    python terminal-rl/scripts/analyze_seta_env_eval.py \\
        --dataset terminal-rl/dataset/seta_env_convert/train.filtered.jsonl \\
        --run main=runs/eval_seta_full_... \\
        --run supp1=runs/eval_seta_full_..._supp1_... \\
        --run supp2=runs/eval_seta_full_..._supp2_... \\
        --out runs/eval_seta_full_.../final_analysis

Later ``--run`` arguments win, so pass them in chronological order.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# Emitted by generate.py when a rollout raises before producing a result, e.g.
#   [task=1080 uid=1b9bfb5a group_idx=-1 sample_idx=17] Generate failed (HTTPStatusError): ...
FAILURE_EVENT_RE = re.compile(
    r"\[task=(?P<task_name>[^\s\]]+)\s+"
    r"uid=(?P<uid>[^\s\]]+)\s+"
    r"group_idx=(?P<group_index>-?\d+)\s+"
    r"sample_idx=(?P<run_sample_index>-?\d+)\]\s+"
    r"Generate failed \((?P<error_type>[A-Za-z_][A-Za-z0-9_]*)\):\s*(?P<error>.*)"
)

PER_SAMPLE_COLUMNS = [
    "sample_index",
    "task_name",
    "task_path",
    "data_source",
    "run_label",
    "run_order",
    "run_sample_index",
    "sample_index_source",
    "uid",
    "status",
    "num_turns",
    "raw_score",
    "raw_reward",
    "task_reward",
    "total_reward",
    "exact_pass",
    "nonzero_score",
    "tool_calls",
    "parse_error_turns",
    "input_tokens",
    "output_tokens",
    "eval_error",
    "traj_path",
    "has_result",
]

MISSING_STATUS = "MISSING"


@dataclass(frozen=True)
class DatasetSample:
    sample_index: int
    task_name: str
    task_path: str
    data_source: str


@dataclass
class IndexRow:
    """One trajectory, keyed back to the dataset row it came from."""

    sample_index: int
    sample_index_source: str
    run_label: str
    run_order: int
    run_sample_index: int | None
    task_name: str
    task_path: str
    uid: str
    status: str
    raw_score: float | None
    raw_reward: float | None
    task_reward: float | None
    total_reward: float | None
    num_turns: float | None
    tool_calls: int | None
    parse_error_turns: int | None
    input_tokens: int | None
    output_tokens: int | None
    eval_error: str
    traj_path: str


@dataclass
class FailureEvent:
    run_label: str
    task_name: str
    uid: str
    run_sample_index: int
    group_index: int
    error_type: str
    error: str


@dataclass
class Analysis:
    per_sample: list[dict[str, Any]]
    summary: dict[str, Any]
    failure_events: list[FailureEvent] = field(default_factory=list)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except (OverflowError, ValueError):
        # NaN and infinity reach here; treat them like any other unusable value
        # rather than aborting a whole run on one malformed meta.json.
        return None


def read_dataset(path: Path) -> list[DatasetSample]:
    """Read the eval JSONL. The 0-based line number is the sample index."""
    samples: list[DatasetSample] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            metadata = json.loads(line).get("metadata") or {}
            samples.append(
                DatasetSample(
                    sample_index=line_no,
                    task_name=str(metadata.get("task_name", "")),
                    task_path=str(metadata.get("task_path", "")),
                    data_source=str(metadata.get("data_source", "")),
                )
            )
    return samples


def derive_turn_metrics(trajectory: dict[str, Any]) -> dict[str, int | float]:
    """Per-trajectory counters, summed over turns.

    Validated against the 60 trajectories published with issue #33: every field
    below reproduces the corresponding column of that run's per_sample.csv exactly.
    """
    turns = trajectory.get("turns") or []
    return {
        "num_turns": float(len(turns)),
        "tool_calls": sum(len(turn.get("tool_calls") or []) for turn in turns),
        "parse_error_turns": sum(1 for turn in turns if turn.get("parse_error_recorded")),
        "input_tokens": sum(turn.get("n_input_tokens") or 0 for turn in turns),
        "output_tokens": sum(turn.get("n_output_tokens") or 0 for turn in turns),
    }


def read_run(run_dir: Path, run_label: str, run_order: int) -> Iterator[IndexRow]:
    """Yield one IndexRow per trajectory directory under ``run_dir``.

    Supplement runs are driven by a filtered JSONL, so their ``sample_index`` is
    local to that run. The original dataset index is recovered from
    ``sample_metadata.supplement_sample_index``, which the eval driver writes.
    """
    trajectories = run_dir / "trajectories"
    if not trajectories.is_dir():
        raise FileNotFoundError(f"{trajectories} is not a directory")

    for meta_path in sorted(trajectories.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_metadata = meta.get("sample_metadata") or {}

        supplement_index = sample_metadata.get("supplement_sample_index")
        if supplement_index is not None:
            sample_index = int(supplement_index)
            sample_index_source = "sample_metadata.supplement_sample_index"
        else:
            sample_index = int(meta.get("sample_index", -1))
            sample_index_source = "index.sample_index"

        traj_path = meta_path.parent / "traj.json"
        metrics: dict[str, int | float] = {}
        if traj_path.is_file():
            metrics = derive_turn_metrics(json.loads(traj_path.read_text(encoding="utf-8")))

        yield IndexRow(
            sample_index=sample_index,
            sample_index_source=sample_index_source,
            run_label=run_label,
            run_order=run_order,
            run_sample_index=_as_int(meta.get("sample_index")),
            task_name=str(meta.get("task_name", "")),
            task_path=str(meta.get("task_path", "")),
            uid=str(meta.get("uid", "")),
            # meta.json stores the enum repr, e.g. "Status.COMPLETED".
            status=str(meta.get("status", "")).replace("Status.", ""),
            raw_score=_as_float(meta.get("raw_score")),
            raw_reward=_as_float(meta.get("raw_reward")),
            task_reward=_as_float(meta.get("task_reward")),
            total_reward=_as_float(meta.get("total_reward")),
            num_turns=metrics.get("num_turns"),
            tool_calls=metrics.get("tool_calls"),
            parse_error_turns=metrics.get("parse_error_turns"),
            input_tokens=metrics.get("input_tokens"),
            output_tokens=metrics.get("output_tokens"),
            eval_error=str(meta.get("eval_error") or ""),
            traj_path=str(traj_path),
        )


def read_failure_events(log_path: Path, run_label: str) -> list[FailureEvent]:
    """Parse 'Generate failed' lines, deduplicated per (uid, sample, error type).

    A single failed rollout retries and therefore logs several lines; the audit
    counts one event per attempt group, not per line.
    """
    if not log_path.is_file():
        return []
    seen: dict[tuple[str, str, str], FailureEvent] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FAILURE_EVENT_RE.search(line)
        if not match:
            continue
        key = (match["uid"], match["run_sample_index"], match["error_type"])
        seen.setdefault(
            key,
            FailureEvent(
                run_label=run_label,
                task_name=match["task_name"],
                uid=match["uid"],
                run_sample_index=int(match["run_sample_index"]),
                group_index=int(match["group_index"]),
                error_type=match["error_type"],
                error=match["error"].strip(),
            ),
        )
    return list(seen.values())


def merge(dataset: Sequence[DatasetSample], index_rows: Iterable[IndexRow]) -> list[dict[str, Any]]:
    """One row per dataset sample; the highest run_order that produced a score wins.

    "Produced a score" matters: a retry that itself failed to reach the verifier
    must not displace an earlier scored attempt, or the sample would be reported
    as present-but-unscored -- neither counted in the mean nor listed as missing
    nor re-queued by :func:`write_supplement_jsonl`.
    """
    best: dict[int, IndexRow] = {}
    for row in index_rows:
        current = best.get(row.sample_index)
        if current is None:
            best[row.sample_index] = row
            continue
        if current.raw_score is not None and row.raw_score is None:
            continue
        if row.raw_score is not None and current.raw_score is None:
            best[row.sample_index] = row
            continue
        if row.run_order >= current.run_order:
            best[row.sample_index] = row

    per_sample: list[dict[str, Any]] = []
    for sample in dataset:
        row = best.get(sample.sample_index)
        if row is None:
            per_sample.append(
                {
                    "sample_index": sample.sample_index,
                    "task_name": sample.task_name,
                    "task_path": sample.task_path,
                    "data_source": sample.data_source,
                    "run_label": "",
                    "run_order": "",
                    "run_sample_index": "",
                    "sample_index_source": "",
                    "uid": "",
                    "status": MISSING_STATUS,
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
                    "has_result": 0,
                }
            )
            continue

        score = row.raw_score
        per_sample.append(
            {
                "sample_index": sample.sample_index,
                "task_name": sample.task_name,
                "task_path": sample.task_path,
                "data_source": sample.data_source,
                "run_label": row.run_label,
                "run_order": row.run_order,
                "run_sample_index": row.run_sample_index,
                "sample_index_source": row.sample_index_source,
                "uid": row.uid,
                "status": row.status,
                "num_turns": row.num_turns,
                "raw_score": score,
                "raw_reward": row.raw_reward,
                "task_reward": row.task_reward,
                "total_reward": row.total_reward,
                "exact_pass": int(score == 1.0) if score is not None else "",
                "nonzero_score": int(score > 0) if score is not None else "",
                "tool_calls": row.tool_calls,
                "parse_error_turns": row.parse_error_turns,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "eval_error": row.eval_error,
                "traj_path": row.traj_path,
                "has_result": 1,
            }
        )
    return per_sample


def _distribution(values: Sequence[float]) -> dict[str, int]:
    return {str(score): count for score, count in sorted(Counter(values).items())}


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(
    per_sample: Sequence[dict[str, Any]],
    failure_events: Sequence[FailureEvent] = (),
) -> dict[str, Any]:
    """Aggregate per-sample rows into the accuracy report.

    Two denominators are reported for every rate. ``*_completed_rows`` divides by
    the samples that produced a result; ``*_all_dataset_missing_as_zero`` divides
    by the whole dataset and scores infrastructure-missing samples as 0. The
    second is the conservative headline number.
    """
    total = len(per_sample)
    present = [row for row in per_sample if row.get("has_result") in (1, "1")]
    scores = [float(row["raw_score"]) for row in present if row.get("raw_score") not in ("", None)]

    score_sum = sum(scores)
    exact_pass = sum(1 for score in scores if score == 1.0)
    nonzero = sum(1 for score in scores if score > 0)

    def _numeric(column: str) -> list[float]:
        return [
            float(row[column])
            for row in present
            if row.get(column) not in ("", None)
        ]

    summary: dict[str, Any] = {
        "dataset_total": total,
        "result_count": len(present),
        "missing_count": total - len(present),
        # Denominator of every *_completed_rows rate. Equal to result_count in a
        # healthy run; smaller if a present row somehow carries no raw_score, and
        # the two must be visibly distinct when that happens.
        "scored_count": len(scores),
        "raw_score_sum_completed_rows": score_sum,
        "raw_score_mean_completed_rows": (score_sum / len(scores)) if scores else None,
        "raw_score_mean_all_dataset_missing_as_zero": (score_sum / total) if total else None,
        "exact_pass_count": exact_pass,
        "exact_pass_rate_completed_rows": (exact_pass / len(scores)) if scores else None,
        "exact_pass_rate_all_dataset_missing_as_zero": (exact_pass / total) if total else None,
        "nonzero_score_count": nonzero,
        "nonzero_score_rate_completed_rows": (nonzero / len(scores)) if scores else None,
        "nonzero_score_rate_all_dataset_missing_as_zero": (nonzero / total) if total else None,
        "raw_score_distribution": _distribution(scores),
        "status_counts": dict(Counter(str(row["status"]) for row in per_sample)),
        "run_counts": dict(
            Counter(str(row["run_label"]) or MISSING_STATUS for row in per_sample)
        ),
        "failure_event_count": len(failure_events),
        "failure_event_counts": dict(Counter(event.error_type for event in failure_events)),
    }
    for column in ("num_turns", "tool_calls", "input_tokens", "output_tokens"):
        key = "turns" if column == "num_turns" else column
        summary[key] = _stats(_numeric(column))
    return summary


def build_task_summary(per_sample: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_sample:
        grouped.setdefault((str(row["task_name"]), str(row["task_path"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (task_name, task_path), rows in sorted(grouped.items()):
        scores = [float(r["raw_score"]) for r in rows if r.get("raw_score") not in ("", None)]
        summary.append(
            {
                "task_name": task_name,
                "task_path": task_path,
                "n": len(rows),
                "result_n": sum(1 for r in rows if r.get("has_result") in (1, "1")),
                "missing_n": sum(1 for r in rows if r.get("has_result") in (0, "0")),
                "raw_score_mean": (sum(scores) / len(scores)) if scores else "",
                "exact_pass_n": sum(1 for s in scores if s == 1.0),
                "nonzero_n": sum(1 for s in scores if s > 0),
                "status_counts": ";".join(
                    f"{status}:{count}"
                    for status, count in sorted(Counter(str(r["status"]) for r in rows).items())
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_outputs(out_dir: Path, analysis: Analysis) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(analysis.summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "per_sample.csv", analysis.per_sample, PER_SAMPLE_COLUMNS)

    task_rows = build_task_summary(analysis.per_sample)
    _write_csv(
        out_dir / "task_summary.csv",
        task_rows,
        ["task_name", "task_path", "n", "result_n", "missing_n", "raw_score_mean",
         "exact_pass_n", "nonzero_n", "status_counts"],
    )

    total = analysis.summary["dataset_total"] or 1
    _write_csv(
        out_dir / "status_counts.csv",
        [
            {"status": status, "count": count, "pct_dataset": 100.0 * count / total}
            for status, count in sorted(analysis.summary["status_counts"].items())
        ],
        ["status", "count", "pct_dataset"],
    )

    _write_csv(
        out_dir / "failure_events.csv",
        [
            {
                "run_label": e.run_label,
                "task_name": e.task_name,
                "uid": e.uid,
                "run_sample_index": e.run_sample_index,
                "group_index": e.group_index,
                "error_type": e.error_type,
                "error": e.error,
            }
            for e in analysis.failure_events
        ],
        ["run_label", "task_name", "uid", "run_sample_index", "group_index", "error_type", "error"],
    )


def write_supplement_jsonl(
    dataset_path: Path,
    per_sample: Sequence[dict[str, Any]],
    out_path: Path,
) -> int:
    """Write a retry JSONL holding only the samples that produced no result.

    A sample qualifies when it has no trajectory at all, and equally when every
    run produced a trajectory that never reached the verifier: both leave it
    without a score, and both need the same retry.

    The supplement is a filtered subset, so the rollout's own sample index no
    longer matches the dataset. ``supplement_sample_index`` is injected into each
    row's metadata and travels into the trajectory's ``sample_metadata``, which is
    how :func:`read_run` maps a supplement trajectory back to its dataset row.
    """
    missing = {
        int(row["sample_index"])
        for row in per_sample
        if row.get("has_result") in (0, "0") or row.get("raw_score") in ("", None)
    }
    if not missing:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dataset_path.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src):
            line = line.strip()
            if not line or line_no not in missing:
                continue
            record = json.loads(line)
            # setdefault is not enough: the key may be present but null.
            if not isinstance(record.get("metadata"), dict):
                record["metadata"] = {}
            record["metadata"]["supplement_sample_index"] = line_no
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def analyze(dataset_path: Path, runs: Sequence[tuple[str, Path]]) -> Analysis:
    dataset = read_dataset(dataset_path)
    index_rows: list[IndexRow] = []
    failure_events: list[FailureEvent] = []
    for order, (label, run_dir) in enumerate(runs):
        index_rows.extend(read_run(run_dir, label, order))
        failure_events.extend(read_failure_events(run_dir / "logs" / "train.log", label))
    per_sample = merge(dataset, index_rows)
    return Analysis(
        per_sample=per_sample,
        summary=summarize(per_sample, failure_events),
        failure_events=failure_events,
    )


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--run expects label=path, got {value!r}")
    label, _, path = value.partition("=")
    if not label:
        raise argparse.ArgumentTypeError(f"--run label must be non-empty, got {value!r}")
    return label, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="eval JSONL; line number is the sample index")
    parser.add_argument(
        "--run",
        type=_parse_run,
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="run directory, repeatable; later runs win on conflict",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--supplement-out",
        type=Path,
        help="also write a retry JSONL containing only the samples with no result",
    )
    args = parser.parse_args(argv)

    analysis = analyze(args.dataset, args.run)
    write_outputs(args.out, analysis)
    if args.supplement_out is not None:
        written = write_supplement_jsonl(args.dataset, analysis.per_sample, args.supplement_out)
        print(f"supplement rows         {written} -> {args.supplement_out}")

    summary = analysis.summary

    def _rate(key: str) -> str:
        # None whenever the dataset is empty; formatting it would crash.
        value = summary[key]
        return "n/a" if value is None else f"{value:.6f}"

    print(f"dataset_total            {summary['dataset_total']}")
    print(f"result_count             {summary['result_count']}")
    print(f"scored_count             {summary['scored_count']}")
    print(f"missing_count            {summary['missing_count']}")
    print(f"raw_score mean (all)     {_rate('raw_score_mean_all_dataset_missing_as_zero')}")
    print(f"exact_pass (all)         {summary['exact_pass_count']} "
          f"({_rate('exact_pass_rate_all_dataset_missing_as_zero')})")
    print(f"status_counts            {summary['status_counts']}")
    print(f"outputs                  {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
