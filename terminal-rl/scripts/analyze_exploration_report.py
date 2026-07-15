#!/usr/bin/env python3
"""Build a focused exploration-run report against a SETA-DAPO baseline.

This script complements ``plot_training_metrics.py`` and
``analyze_trajectories.py``. It reads the structured ``logs/metrics.jsonl``
records when available, falls back to the training log parser from
``plot_training_metrics.py`` when needed, and writes run-specific report assets
under ``<run_dir>/metrics/analysis``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sqlite3
import statistics
import sys
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def metric(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = num(record.get(key))
        if value is not None:
            return value
    return None


def rid(record: dict[str, Any]) -> int:
    for key in ("rollout_id", "global_step"):
        value = record.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return -1


def fmt(value: Any, digits: int = 4) -> str:
    value = num(value)
    if value is None:
        return "NA"
    if value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.{max(2, digits)}e}"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.{digits}f}"


def fmt_int(value: Any) -> str:
    value = num(value)
    if value is None:
        return "NA"
    return f"{int(round(value)):,}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    value = num(value)
    if value is None:
        return "NA"
    return f"{value * 100:.{digits}f}%"


def pct_delta(new: Any, base: Any) -> float | None:
    new_v = num(new)
    base_v = num(base)
    if new_v is None or base_v is None or abs(base_v) < 1e-12:
        return None
    return (new_v - base_v) / abs(base_v)


def delta_text(exp: Any, base: Any, unit: str = "rollout") -> str:
    exp_v = num(exp)
    base_v = num(base)
    if exp_v is None or base_v is None:
        return "无法比较"
    diff = exp_v - base_v
    if abs(diff) < 1e-12:
        return f"持平（同为 {fmt(exp_v, 0)} {unit}）"
    direction = "更早" if diff < 0 else "更晚"
    return f"{direction} **{fmt(abs(diff), 0)}** 个 {unit}"


def rel_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def mtime_text(path: Path) -> str:
    if not path.exists():
        return "NA"
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(path.stat().st_mtime))


def load_metrics_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("phase") and record.get("phase") != "train":
                continue
            dataset = str(record.get("dataset") or "").strip()
            if dataset and dataset != "seta":
                continue
            record["_jsonl_line"] = line_no
            records.append(record)
    return sorted(records, key=rid)


def load_metrics_from_train_log(run_dir: Path, log_file: Path) -> list[dict[str, Any]]:
    script_path = Path(__file__).resolve().with_name("plot_training_metrics.py")
    spec = importlib.util.spec_from_file_location("plot_training_metrics", script_path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    sys.modules["plot_training_metrics"] = module
    spec.loader.exec_module(module)
    parsed = module._parse_log(log_file)  # type: ignore[attr-defined]
    records = module._structured_train_records(parsed)  # type: ignore[attr-defined]
    return sorted(records, key=rid)


def dedupe_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("schema"),
        record.get("phase"),
        record.get("dataset"),
        record.get("rollout_id"),
        record.get("global_step"),
        record.get("sample_count"),
        record.get("trainable_count"),
    )


def merge_records(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for record in group:
            key = dedupe_key(record)
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)
    return sorted(merged, key=rid)


def load_records(run_dir: Path, log_file: Path | None = None) -> tuple[list[dict[str, Any]], str]:
    jsonl = run_dir / "logs" / "metrics.jsonl"
    records = load_metrics_from_jsonl(jsonl)
    train_records: list[dict[str, Any]] = []
    log_path = log_file if log_file and log_file.is_file() else run_dir / "logs" / "train.log"
    if log_path.is_file():
        train_records = load_metrics_from_train_log(run_dir, log_path)
    if records or train_records:
        merged = merge_records(records, train_records)
        if records and train_records:
            return merged, f"{jsonl} + {log_path}"
        if records:
            return merged, str(jsonl)
        return merged, str(log_path)
    if log_file and log_file.is_file():
        records = load_metrics_from_train_log(run_dir, log_file)
        return records, str(log_file)
    train_log = run_dir / "logs" / "train.log"
    if train_log.is_file():
        records = load_metrics_from_train_log(run_dir, train_log)
        return records, str(train_log)
    return [], "missing"


def is_valid(record: dict[str, Any]) -> bool:
    trainable = metric(record, "trainable_count")
    raw = metric(record, "raw_reward", "reward/raw", "test_acc", "pass_rate")
    return trainable is not None and trainable > 0 and raw is not None


def values(records: list[dict[str, Any]], *keys: str) -> list[float]:
    out: list[float] = []
    for record in records:
        value = metric(record, *keys)
        if value is not None:
            out.append(value)
    return out


def stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "first10_mean": None,
            "last10_mean": None,
        }
    head = vals[: min(10, len(vals))]
    tail = vals[-min(10, len(vals)) :]
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "first10_mean": statistics.mean(head),
        "last10_mean": statistics.mean(tail),
    }


def weighted_outcome(records: list[dict[str, Any]], valid: list[dict[str, Any]]) -> dict[str, Any]:
    sample_total = sum(metric(record, "sample_count") or 0.0 for record in records)
    trainable_total = sum(metric(record, "trainable_count") or 0.0 for record in records)
    weighted_raw_num = 0.0
    weighted_raw_den = 0.0
    for record in valid:
        raw = metric(record, *FIELD_KEYS["raw_reward"])
        trainable = metric(record, "trainable_count")
        if raw is None or trainable is None or trainable <= 0:
            continue
        weighted_raw_num += raw * trainable
        weighted_raw_den += trainable
    return {
        "sample_count_total": sample_total,
        "trainable_count_total": trainable_total,
        "valid_raw_weighted": (weighted_raw_num / weighted_raw_den) if weighted_raw_den else None,
        "op_raw": (weighted_raw_num / sample_total) if sample_total else None,
        "fail_fraction": (1.0 - trainable_total / sample_total) if sample_total else None,
    }


def ranges(items: list[int]) -> str:
    if not items:
        return ""
    ordered = sorted(set(items))
    chunks: list[str] = []
    start = prev = ordered[0]
    for item in ordered[1:]:
        if item == prev + 1:
            prev = item
            continue
        chunks.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = item
    chunks.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(chunks)


FIELD_KEYS = {
    "raw_reward": ("raw_reward", "reward/raw", "test_acc", "pass_rate"),
    "task_reward": ("task_reward", "reward/task"),
    "total_reward": ("total_reward", "reward/total"),
    "reward_std": ("reward_std",),
    "truncated_fraction": ("truncated_fraction",),
    "response_length": ("response_length",),
    "trainable_count": ("trainable_count",),
    "sample_count": ("sample_count",),
    "failed": ("failed",),
    "completed": ("completed",),
    "exploration_abs": ("reward/exploration_abs", "exploration_reward_abs"),
    "exploration_signal": ("reward/exploration_signal", "exploration_reward_signal", "agent57/ngu_bonus"),
    "ngu_episodic": ("agent57/ngu_episodic",),
    "lifelong_unique_keys": ("agent57/lifelong_unique_keys",),
    "lifelong_seen_before": ("agent57/lifelong_seen_before",),
    "lifelong_bonus": ("agent57/lifelong_bonus",),
    "lifelong_raw": ("agent57/lifelong_raw",),
    "lifelong_eligible_rate": ("agent57/lifelong_eligible_rate",),
    "lifelong_warmup_remaining": ("agent57/lifelong_warmup_remaining",),
    "ngu_life_mod": ("agent57/ngu_life_mod",),
    "episodic_empty_bucket_rate": ("agent57/episodic_empty_bucket_rate",),
    "episodic_exact_repeat_count": ("agent57/episodic_exact_repeat_count",),
    "ngu_bonus": ("agent57/ngu_bonus",),
    "top_arm": ("agent57/top_arm",),
    "top_arm_ratio": ("agent57/top_arm_ratio",),
    "top_suppressed_ratio": ("agent57/top_suppressed_ratio",),
}


def add_derived(records: list[dict[str, Any]]) -> None:
    for record in records:
        unique = metric(record, "agent57/lifelong_unique_keys")
        seen = metric(record, "agent57/lifelong_seen_before")
        if unique is not None and seen is not None and unique + seen > 0:
            record["_new_state_proxy"] = unique / (unique + seen)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    add_derived(records)
    valid = [record for record in records if is_valid(record)]
    invalid_rollouts = [rid(record) for record in records if not is_valid(record)]
    summary: dict[str, Any] = {
        "n_points": len(records),
        "n_valid": len(valid),
        "n_invalid_or_zero_trainable": len(invalid_rollouts),
        "invalid_fraction": (len(invalid_rollouts) / len(records)) if records else None,
        "invalid_rollouts": invalid_rollouts,
        "invalid_ranges": ranges(invalid_rollouts),
        "first_rollout": min((rid(record) for record in records), default=None),
        "last_rollout": max((rid(record) for record in records), default=None),
        "first_valid_rollout": rid(valid[0]) if valid else None,
        "last_valid_rollout": rid(valid[-1]) if valid else None,
    }
    for name, keys in FIELD_KEYS.items():
        summary[name] = stats(values(valid, *keys))
    summary["new_state_proxy"] = stats(values(valid, "_new_state_proxy"))
    for key, value in weighted_outcome(records, valid).items():
        summary[key] = {"n": len(valid), "mean": value}
    return summary


def window_summary(records: list[dict[str, Any]], *, max_rollout: int | None = None, first_n_valid: int | None = None, last_n_valid: int | None = None) -> dict[str, Any]:
    all_selected = list(records)
    if max_rollout is not None:
        all_selected = [record for record in all_selected if rid(record) <= max_rollout]
    selected = [record for record in all_selected if is_valid(record)]
    if first_n_valid is not None:
        selected = selected[:first_n_valid]
        all_selected = selected
    if last_n_valid is not None:
        selected = selected[-last_n_valid:]
        all_selected = selected
    out = {
        "n_valid": len(selected),
        "first_rollout": rid(selected[0]) if selected else None,
        "last_rollout": rid(selected[-1]) if selected else None,
    }
    for name, keys in FIELD_KEYS.items():
        out[name] = stats(values(selected, *keys))
    out["new_state_proxy"] = stats(values(selected, "_new_state_proxy"))
    for key, value in weighted_outcome(all_selected, selected).items():
        out[key] = {"n": len(selected), "mean": value}
    return out


def rolling_threshold(records: list[dict[str, Any]], threshold: float, window: int = 10) -> dict[str, Any] | None:
    valid = [record for record in records if is_valid(record)]
    vals: list[float] = []
    for idx, record in enumerate(valid, start=1):
        raw = metric(record, *FIELD_KEYS["raw_reward"])
        if raw is None:
            continue
        vals.append(raw)
        if len(vals) >= window:
            mean = statistics.mean(vals[-window:])
            if mean >= threshold:
                return {
                    "threshold": threshold,
                    "window": window,
                    "rollout_id": rid(record),
                    "valid_point_index": idx,
                    "rolling_mean": mean,
                }
    return None


def compare_values(exp: Any, base: Any, lower_is_better: bool = False) -> dict[str, Any]:
    exp_v = num(exp)
    base_v = num(base)
    diff = None if exp_v is None or base_v is None else exp_v - base_v
    pct = pct_delta(exp_v, base_v)
    if lower_is_better and diff is not None:
        verdict = "提升" if diff < 0 else ("退化" if diff > 0 else "持平")
    elif diff is not None:
        verdict = "提升" if diff > 0 else ("退化" if diff < 0 else "持平")
    else:
        verdict = "NA"
    return {"exp": exp_v, "baseline": base_v, "diff": diff, "pct_delta": pct, "verdict": verdict}


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sx = statistics.pstdev(xs)
    sy = statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (sx * sy)


def correlation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if is_valid(record)]
    out: dict[str, Any] = {}
    targets = {
        "exploration_abs": FIELD_KEYS["exploration_abs"],
        "ngu_bonus": FIELD_KEYS["ngu_bonus"],
        "lifelong_unique_keys": FIELD_KEYS["lifelong_unique_keys"],
        "lifelong_seen_before": FIELD_KEYS["lifelong_seen_before"],
        "episodic_empty_bucket_rate": FIELD_KEYS["episodic_empty_bucket_rate"],
        "episodic_exact_repeat_count": FIELD_KEYS["episodic_exact_repeat_count"],
        "top_suppressed_ratio": FIELD_KEYS["top_suppressed_ratio"],
        "new_state_proxy": ("_new_state_proxy",),
    }
    for name, keys in targets.items():
        xs: list[float] = []
        ys: list[float] = []
        for record in valid:
            x = metric(record, *keys)
            y = metric(record, *FIELD_KEYS["raw_reward"])
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        out[name] = {"n": len(xs), "pearson_raw_reward": pearson(xs, ys)}
    return out


def load_arm_events(db_path: Path) -> dict[str, Any]:
    if not db_path.is_file():
        return {"available": False}
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    try:
        n = cur.execute("select count(*) from arm_events").fetchone()[0]
    except sqlite3.Error:
        return {"available": False}
    by_arm = []
    for row in cur.execute(
        """
        select arm_id, count(*) as n,
               avg(normalized_base_score) as normalized_base_score_mean,
               avg(success) as success_rate,
               avg(truncated) as truncated_rate,
               avg(parse_error) as parse_error_rate,
               avg(bonus) as bonus_mean,
               max(bonus) as bonus_max
        from arm_events
        group by arm_id
        order by arm_id
        """
    ):
        by_arm.append(dict(row))
    rows = [
        dict(row)
        for row in cur.execute(
            "select id, ts, arm_id, normalized_base_score, success, truncated, parse_error, bonus from arm_events order by id"
        )
    ]
    bins = []
    if rows:
        bin_count = min(40, max(1, len(rows) // 150))
        size = max(1, math.ceil(len(rows) / bin_count))
        for start in range(0, len(rows), size):
            chunk = rows[start : start + size]
            bins.append(
                {
                    "start_id": chunk[0]["id"],
                    "end_id": chunk[-1]["id"],
                    "n": len(chunk),
                    "success_rate": statistics.mean(float(item["success"]) for item in chunk),
                    "truncated_rate": statistics.mean(float(item["truncated"]) for item in chunk),
                    "parse_error_rate": statistics.mean(float(item["parse_error"]) for item in chunk),
                    "bonus_mean": statistics.mean(float(item["bonus"]) for item in chunk),
                    "arm_mean": statistics.mean(float(item["arm_id"]) for item in chunk),
                    "normalized_base_score_mean": statistics.mean(float(item["normalized_base_score"]) for item in chunk),
                }
            )
    lifelong = None
    try:
        row = cur.execute(
            "select count(*) as n_keys, sum(count) as count_sum, avg(count) as count_mean, max(count) as count_max from lifelong_counts"
        ).fetchone()
        lifelong = dict(row) if row else None
    except sqlite3.Error:
        lifelong = None
    con.close()
    return {"available": True, "n_events": n, "by_arm": by_arm, "time_bins": bins, "lifelong_counts": lifelong}


def setup_matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def plot_core(records: list[dict[str, Any]], baseline: list[dict[str, Any]], out: Path, max_rollout: int, exp_label: str = "experiment") -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    exp_valid = [record for record in records if is_valid(record) and rid(record) <= max_rollout]
    baseline_valid = [record for record in baseline if is_valid(record)][: len(exp_valid)]
    exp_last_rollout = rid(exp_valid[-1]) if exp_valid else None
    baseline_last_rollout = rid(baseline_valid[-1]) if baseline_valid else None

    def plot_series(ax: Any, recs: list[dict[str, Any]], keys: tuple[str, ...], label: str, color: str) -> None:
        pts = [
            (idx, value)
            for idx, record in enumerate(recs)
            for value in [metric(record, *keys)]
            if value is not None
        ]
        if not pts:
            return
        ax.plot([x for x, _ in pts], [y for _, y in pts], ".", ms=3, alpha=0.28, color=color)
        roll_x: list[int] = []
        roll_y: list[float] = []
        vals: list[float] = []
        for x, y in pts:
            vals.append(float(y))
            if len(vals) >= 10:
                roll_x.append(x)
                roll_y.append(statistics.mean(vals[-10:]))
        if roll_y:
            ax.plot(roll_x, roll_y, "-", lw=2.0, label=f"{label} rolling10", color=color)

    plot_series(axes[0, 0], exp_valid, FIELD_KEYS["raw_reward"], exp_label, "tab:blue")
    plot_series(axes[0, 0], baseline_valid, FIELD_KEYS["raw_reward"], "baseline", "tab:orange")
    axes[0, 0].set_title("Pass-rate/raw_reward (valid points)")
    axes[0, 0].set_ylabel("raw_reward")
    axes[0, 0].legend()

    plot_series(axes[0, 1], exp_valid, FIELD_KEYS["total_reward"], exp_label, "tab:blue")
    plot_series(axes[0, 1], baseline_valid, FIELD_KEYS["total_reward"], "baseline", "tab:orange")
    axes[0, 1].axhline(0, color="gray", lw=0.8, ls=":")
    axes[0, 1].set_title("Total reward")
    axes[0, 1].legend()

    plot_series(axes[1, 0], exp_valid, FIELD_KEYS["truncated_fraction"], exp_label, "tab:blue")
    plot_series(axes[1, 0], baseline_valid, FIELD_KEYS["truncated_fraction"], "baseline", "tab:orange")
    axes[1, 0].set_title("Truncated fraction")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 0].set_xlabel("effective valid rollout step")
    axes[1, 0].legend()

    plot_series(axes[1, 1], exp_valid, FIELD_KEYS["response_length"], exp_label, "tab:blue")
    plot_series(axes[1, 1], baseline_valid, FIELD_KEYS["response_length"], "baseline", "tab:orange")
    axes[1, 1].set_title("Response length")
    axes[1, 1].set_xlabel("effective valid rollout step")
    axes[1, 1].legend()

    axes[0, 0].set_xlabel("effective valid rollout step")
    axes[0, 1].set_xlabel("effective valid rollout step")
    fig.suptitle(
        "Core metric comparison by valid rollout step "
        f"(N={len(exp_valid)}, exp rollout<= {exp_last_rollout}, baseline rollout<= {baseline_last_rollout})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_exploration(records: list[dict[str, Any]], out: Path) -> None:
    plt = setup_matplotlib()
    valid = [record for record in records if is_valid(record)]
    xs = [rid(record) for record in valid]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True)

    def series(*keys: str) -> list[float | None]:
        return [metric(record, *keys) for record in valid]

    def transformed_series(*keys: str, scale: float = 1.0, log10p: bool = False) -> list[float | None]:
        out: list[float | None] = []
        for value in series(*keys):
            if value is None:
                out.append(None)
                continue
            if log10p:
                out.append(math.log10(max(0.0, float(value)) + 1.0))
            else:
                out.append(float(value) * scale)
        return out

    def add_line(ax: Any, ys: list[float | None], label: str, *, color: str, marker: str = ".", ls: str = "-", alpha: float = 0.86) -> None:
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            return
        ax.plot([x for x, _ in pts], [float(y) for _, y in pts], marker + ls, label=label, color=color, alpha=alpha, ms=3.2, lw=1.4)

    def legend_for(ax: Any, *extra_axes: Any, loc: str = "best") -> None:
        lines = list(ax.lines)
        for extra in extra_axes:
            lines.extend(extra.lines)
        labels = [line.get_label() for line in lines]
        visible = [(line, label) for line, label in zip(lines, labels) if label and not label.startswith("_")]
        if visible:
            ax.legend([line for line, _ in visible], [label for _, label in visible], fontsize=8, loc=loc)

    # 1) In-episode intrinsic reward: episodic novelty before lifelong modulation.
    ax = axes[0, 0]
    add_line(ax, series("agent57/ngu_episodic"), "NGU episodic", color="tab:blue")
    add_line(ax, series("agent57/episodic_empty_bucket_rate"), "empty-bucket rate", color="tab:orange", alpha=0.75)
    ax2 = ax.twinx()
    add_line(ax2, series("agent57/episodic_exact_repeat_count"), "exact repeats", color="tab:brown", alpha=0.55)
    ax.set_title("1. In-episode intrinsic reward")
    ax.set_ylabel("episodic novelty / ratio")
    ax.set_ylim(-0.03, 1.05)
    ax2.set_ylabel("exact repeat count")
    legend_for(ax, ax2, loc="upper left")

    # 2) Across-episode/lifelong intrinsic reward: novelty memory and life modifier.
    ax = axes[0, 1]
    add_line(ax, series("agent57/lifelong_raw"), "lifelong raw novelty", color="tab:blue")
    add_line(ax, series("agent57/ngu_life_mod"), "NGU life modifier", color="tab:purple", alpha=0.75)
    add_line(ax, transformed_series("agent57/lifelong_bonus", scale=1e5), "lifelong bonus x1e5", color="tab:green", alpha=0.65)
    add_line(ax, transformed_series("_new_state_proxy", scale=10.0), "new-state proxy x10", color="tab:cyan", alpha=0.62)
    ax2 = ax.twinx()
    add_line(ax2, transformed_series("agent57/lifelong_unique_keys", log10p=True), "log10(unique keys + 1)", color="tab:olive", alpha=0.72)
    add_line(ax2, transformed_series("agent57/lifelong_seen_before", log10p=True), "log10(seen before + 1)", color="tab:red", alpha=0.62)
    ax.set_title("2. Lifelong intrinsic reward")
    ax.set_ylabel("novelty / modifier / scaled bonus")
    ax2.set_ylabel("coverage proxy count (log10)")
    legend_for(ax, ax2)

    # 3) Fused intrinsic reward: combined NGU signal and reward-space injection.
    ax = axes[1, 0]
    add_line(ax, series("agent57/ngu_bonus"), "fused NGU bonus", color="tab:purple")
    add_line(ax, series("reward/exploration_signal", "exploration_reward_signal"), "exploration signal", color="tab:blue", alpha=0.48)
    ax2 = ax.twinx()
    add_line(ax2, series(*FIELD_KEYS["exploration_abs"]), "abs exploration reward", color="tab:green", alpha=0.72)
    ax.set_title("3. Fused intrinsic reward")
    ax.set_xlabel("rollout")
    ax.set_ylabel("intrinsic signal / NGU bonus")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.set_ylabel("abs reward contribution")
    legend_for(ax, ax2)

    # 4) UCB arm selection and trust/lifelong suppression.
    ax = axes[1, 1]
    add_line(ax, series("agent57/top_arm"), "top arm id", color="tab:gray", marker="o", ls="", alpha=0.55)
    ax.set_title("4. UCB arm selection and suppression")
    ax.set_xlabel("rollout")
    ax.set_ylabel("top arm id")
    ax.set_ylim(-0.5, 7.5)
    ax.set_yticks(range(8))
    ax2 = ax.twinx()
    add_line(ax2, series("agent57/top_arm_ratio"), "top-arm share", color="tab:blue", alpha=0.68)
    add_line(ax2, series("agent57/top_suppressed_ratio"), "top suppressed-reason share", color="tab:red", alpha=0.78)
    ax2.set_ylabel("share / ratio")
    ax2.set_ylim(-0.03, 1.03)
    ax.text(
        0.01,
        0.02,
        "arm beta: 0=.000, 1=.002, 2=.004, 3=.006, 4=.008, 5=.010, 6=.015, 7=.020",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.35",
        va="bottom",
    )
    legend_for(ax, ax2, loc="upper left")

    fig.suptitle("Agent57 exploration metrics: episodic -> lifelong -> fused -> UCB", fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_relationship(records: list[dict[str, Any]], correlations: dict[str, Any], out: Path) -> None:
    plt = setup_matplotlib()
    valid = [record for record in records if is_valid(record)]
    xs_roll = [rid(record) for record in valid]
    raw = [metric(record, *FIELD_KEYS["raw_reward"]) for record in valid]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    def scatter(ax: Any, xkey: tuple[str, ...], title: str, xlabel: str) -> None:
        pts = []
        for record, raw_value, rollout in zip(valid, raw, xs_roll):
            x = metric(record, *xkey)
            if x is None or raw_value is None:
                continue
            pts.append((x, raw_value, rollout))
        if pts:
            sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=[p[2] for p in pts], s=24, cmap="viridis", alpha=0.72)
            fig.colorbar(sc, ax=ax, label="rollout")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("raw_reward")

    scatter(axes[0, 0], FIELD_KEYS["exploration_abs"], "Exploration reward vs raw_reward", "abs exploration reward")
    scatter(axes[0, 1], FIELD_KEYS["lifelong_unique_keys"], "Unique keys vs raw_reward", "lifelong unique keys")
    scatter(axes[1, 0], FIELD_KEYS["top_suppressed_ratio"], "Suppression vs raw_reward", "top suppressed ratio")

    corr_items = [
        (name, item.get("pearson_raw_reward"))
        for name, item in correlations.items()
        if item.get("pearson_raw_reward") is not None
    ]
    corr_items = sorted(corr_items, key=lambda x: abs(float(x[1])), reverse=True)[:8]
    axes[1, 1].barh([name for name, _ in corr_items], [float(value) for _, value in corr_items], color="tab:blue", alpha=0.75)
    axes[1, 1].axvline(0, color="black", lw=0.8)
    axes[1, 1].set_title("Pearson correlation with raw_reward")
    axes[1, 1].set_xlabel("r")
    axes[1, 1].invert_yaxis()

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_arm_events(arm: dict[str, Any], by_arm_out: Path, time_out: Path) -> None:
    if not arm.get("available"):
        return
    plt = setup_matplotlib()
    by_arm = arm.get("by_arm") or []
    if by_arm:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        labels = [int(item["arm_id"]) for item in by_arm]
        counts = [int(item["n"]) for item in by_arm]
        success = [float(item["success_rate"]) for item in by_arm]
        trunc = [float(item["truncated_rate"]) for item in by_arm]
        bonus = [float(item["bonus_mean"]) for item in by_arm]
        axes[0].bar(labels, counts, color="tab:blue", alpha=0.75)
        axes[0].set_title("Arm event counts")
        axes[0].set_xlabel("arm")
        axes[0].set_ylabel("events")
        axes[1].plot(labels, success, "o-", label="success")
        axes[1].plot(labels, trunc, "o-", label="truncated")
        axes[1].set_ylim(0, max(0.5, max(success + trunc) * 1.15))
        axes[1].set_title("Outcome rate by arm")
        axes[1].set_xlabel("arm")
        axes[1].legend()
        axes[2].bar(labels, bonus, color="tab:green", alpha=0.75)
        axes[2].set_title("Mean bonus by arm")
        axes[2].set_xlabel("arm")
        axes[2].set_ylabel("bonus")
        fig.tight_layout()
        fig.savefig(by_arm_out)
        plt.close(fig)

    bins = arm.get("time_bins") or []
    if bins:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        xs = list(range(len(bins)))
        labels = [f"{item['start_id']}-{item['end_id']}" for item in bins]
        axes[0].plot(xs, [item["success_rate"] for item in bins], "o-", label="success")
        axes[0].plot(xs, [item["truncated_rate"] for item in bins], "o-", label="truncated")
        axes[0].plot(xs, [item["parse_error_rate"] for item in bins], "o-", label="parse error", alpha=0.75)
        axes[0].set_title("Arm event outcomes over time")
        axes[0].set_ylabel("rate")
        axes[0].legend()
        axes[1].plot(xs, [item["bonus_mean"] for item in bins], "o-", label="mean bonus", color="tab:green")
        ax2 = axes[1].twinx()
        ax2.plot(xs, [item["arm_mean"] for item in bins], "o-", label="mean arm", color="tab:gray", alpha=0.65)
        axes[1].set_ylabel("bonus")
        ax2.set_ylabel("arm")
        axes[1].set_xlabel("arm event bin")
        step = max(1, len(labels) // 8)
        axes[1].set_xticks(xs[::step])
        axes[1].set_xticklabels([labels[i] for i in xs[::step]], rotation=35, ha="right")
        lines = axes[1].lines + ax2.lines
        axes[1].legend(lines, [line.get_label() for line in lines], fontsize=8)
        fig.tight_layout()
        fig.savefig(time_out)
        plt.close(fig)


def compact(text: Any, limit: int = 20000) -> str:
    raw = str(text if text is not None else "")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    if len(raw) <= limit:
        return raw
    return raw[: limit - 80].rstrip() + "\n... [truncated in report; see trajectory JSON for full raw payload]"


def one_line(text: Any, limit: int = 180) -> str:
    raw = " ".join(str(text if text is not None else "").split())
    return raw if len(raw) <= limit else raw[: limit - 3].rstrip() + "..."


def md_escape(text: Any) -> str:
    return str(text if text is not None else "").replace("|", "\\|")


def load_traj(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_traj_by_name(run_dir: Path, name: str) -> Path | None:
    path = run_dir / "trajectories" / name / "traj.json"
    return path if path.is_file() else None


def scan_trajectories(run_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    traj_dir = run_dir / "trajectories"
    if not traj_dir.is_dir():
        return out
    for path in traj_dir.glob("*/traj.json"):
        data = load_traj(path)
        if not data:
            continue
        info = data.get("info") if isinstance(data.get("info"), dict) else {}
        reward = data.get("reward") if isinstance(data.get("reward"), dict) else {}
        out.append(
            {
                "path": path,
                "name": path.parent.name,
                "task_id": info.get("task_id"),
                "task_name": str(info.get("task_name") or ""),
                "task_path": str(info.get("task_path") or ""),
                "uid": info.get("uid"),
                "group_index": info.get("group_index"),
                "sample_index": info.get("sample_index"),
                "rollout_id": info.get("rollout_id"),
                "train_step": info.get("train_step"),
                "status": str(info.get("status") or "").split(".")[-1],
                "num_turns": info.get("num_turns"),
                "raw_score": reward.get("raw_score"),
                "total_reward": reward.get("total_reward", reward.get("score")),
                "arm_id": reward.get("explore_agent57_arm_id"),
                "beta": reward.get("explore_agent57_beta"),
                "ngu_bonus": reward.get("explore_agent57_ngu_bonus"),
                "intrinsic_signal": reward.get("explore_agent57_intrinsic_signal"),
                "unique_keys": reward.get("explore_agent57_lifelong_unique_keys"),
                "seen_before": reward.get("explore_agent57_lifelong_seen_before"),
                "trust": reward.get("explore_agent57_trust"),
                "suppressed_reason": reward.get("explore_agent57_lifelong_suppressed_reason"),
            }
        )
    return out


def case_strength(item: dict[str, Any]) -> float:
    values = [
        num(item.get("ngu_bonus")),
        num(item.get("unique_keys")),
        num(item.get("beta")),
        num(item.get("intrinsic_signal")),
    ]
    score = 0.0
    for idx, value in enumerate(values):
        if value is not None:
            score += value / (10 ** idx)
    return score


def case_raw(item: dict[str, Any]) -> float:
    value = num(item.get("raw_score"))
    return value if value is not None else 0.0


def case_rollout(item: dict[str, Any]) -> int:
    value = num(item.get("rollout_id"))
    return int(value) if value is not None else -1


def pick_cases(run_dir: Path, baseline_run_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    used_paths: set[Path] = set()

    def add_case(case_id: str, title: str, reason: str, entries: list[tuple[str, Path | None]]) -> None:
        valid_paths = []
        for label, path in entries:
            if path is None or path in used_paths:
                continue
            valid_paths.append((label, path))
            used_paths.add(path)
        if valid_paths:
            cases.append({"case_id": case_id, "title": title, "reason": reason, "paths": valid_paths})

    records = scan_trajectories(run_dir)
    if not records:
        return cases

    success = sorted(
        [item for item in records if item["status"] == "COMPLETED" and case_raw(item) >= 1.0],
        key=lambda item: (case_strength(item), case_rollout(item)),
        reverse=True,
    )
    failures = sorted(
        [item for item in records if item["status"] == "COMPLETED" and case_raw(item) <= 0.0],
        key=lambda item: (case_strength(item), case_rollout(item)),
    )
    truncated = sorted(
        [item for item in records if item["status"] == "TRUNCATED"],
        key=lambda item: (case_strength(item), num(item.get("unique_keys")) or 0.0, case_rollout(item)),
        reverse=True,
    )

    # Strict baseline-vs-experiment match if the saved trajectories expose the
    # same task and sample identifiers. If that is unavailable, use a clearly
    # labeled same-task reference without claiming same-sample causality.
    try:
        baseline_records = scan_trajectories(baseline_run_dir)
    except Exception:
        baseline_records = []
    exact_baseline: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    task_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in baseline_records:
        if item.get("task_name"):
            task_baseline[str(item["task_name"])].append(item)
        if item.get("task_name") and item.get("sample_index") is not None:
            exact_baseline[(str(item["task_name"]), item.get("sample_index"))].append(item)

    matched = False
    for item in success:
        candidates = exact_baseline.get((str(item.get("task_name")), item.get("sample_index")), [])
        candidates = [cand for cand in candidates if case_raw(cand) < 1.0 or cand.get("status") != "COMPLETED"]
        if not candidates:
            continue
        base = sorted(candidates, key=lambda cand: (cand.get("status") == "TRUNCATED", -case_raw(cand)), reverse=True)[0]
        add_case(
            "baseline_exact_sample_reference",
            "baseline 失败 vs 本次实验成功（同 task/sample）",
            "自动匹配到同一 task/sample：baseline 未通过而本次实验通过，可作为最强的单样本提升证据。",
            [("baseline", base["path"]), ("本次实验", item["path"])],
        )
        matched = True
        break

    if not matched:
        for item in success:
            candidates = [
                cand
                for cand in task_baseline.get(str(item.get("task_name")), [])
                if case_raw(cand) < 1.0 or cand.get("status") != "COMPLETED"
            ]
            if not candidates:
                continue
            base = sorted(candidates, key=lambda cand: (cand.get("status") == "TRUNCATED", case_rollout(cand)), reverse=True)[0]
            add_case(
                "baseline_same_task_reference",
                "baseline 失败 vs 本次实验成功（同 task 参考）",
                "自动匹配到同一 task 的 baseline 未通过轨迹与本次实验通过轨迹；这不是同 sample 因果证明，但能展示任务级行为差异。",
                [("baseline 同 task", base["path"]), ("本次实验成功", item["path"])],
            )
            break

    # Within-run high/low exploration contrast on the same task when possible.
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_task[str(item.get("task_name") or item.get("task_path") or item.get("name"))].append(item)
    for task_items in by_task.values():
        task_success = [item for item in task_items if item["status"] == "COMPLETED" and case_raw(item) >= 1.0]
        task_fail = [item for item in task_items if case_raw(item) <= 0.0 and item["status"] in {"COMPLETED", "TRUNCATED", "FAILED"}]
        if not task_success or not task_fail:
            continue
        high = max(task_success, key=case_strength)
        low = min(task_fail, key=case_strength)
        if high["path"] == low["path"]:
            continue
        add_case(
            "within_run_high_low_exploration",
            "同 task 高探索成功 vs 低探索失败",
            "自动选择同一 task 中探索强度较高且通过的轨迹，以及探索强度较低且失败/截断的轨迹，用于观察探索 arm 对决策路径的影响。",
            [("低探索失败/截断", low["path"]), ("高探索成功", high["path"])],
        )
        break

    if success:
        item = success[0]
        add_case(
            "high_exploration_success",
            "高探索成功样本",
            "自动选择 NGU/unique/beta 综合探索强度最高的成功轨迹，展示探索机制带来有效完成的正例。",
            [("本次实验高探索成功", item["path"])],
        )
    if truncated:
        item = truncated[0]
        add_case(
            "high_exploration_truncated",
            "高覆盖但截断的负例",
            "自动选择探索强度/覆盖度较高但被截断的轨迹，展示探索不能替代任务完成，且需要截断约束与 trust gate。",
            [("本次实验高探索截断", item["path"])],
        )
    if failures:
        item = failures[0]
        add_case(
            "low_exploration_failure",
            "低探索失败样本",
            "自动选择探索强度较低且 raw=0 的完成轨迹，作为高探索成功样本的反面对照。",
            [("本次实验低探索失败", item["path"])],
        )
    return cases


def traj_summary(data: dict[str, Any]) -> dict[str, Any]:
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    reward = data.get("reward") if isinstance(data.get("reward"), dict) else {}
    return {
        "task": info.get("task_name"),
        "status": str(info.get("status") or "").split(".")[-1],
        "rollout_id": info.get("rollout_id"),
        "train_step": info.get("train_step"),
        "turns": info.get("num_turns", len(data.get("turns") or [])),
        "raw_score": reward.get("raw_score"),
        "task_reward": reward.get("task_reward"),
        "total_reward": reward.get("total_reward", reward.get("score")),
        "arm_id": reward.get("explore_agent57_arm_id"),
        "beta": reward.get("explore_agent57_beta"),
        "trust": reward.get("explore_agent57_trust"),
        "suppressed_reason": reward.get("explore_agent57_lifelong_suppressed_reason"),
        "intrinsic_signal": reward.get("explore_agent57_intrinsic_signal"),
        "ngu_bonus": reward.get("explore_agent57_ngu_bonus"),
        "unique_keys": reward.get("explore_agent57_lifelong_unique_keys"),
        "seen_before": reward.get("explore_agent57_lifelong_seen_before"),
        "empty_bucket_rate": reward.get("explore_agent57_episodic_empty_bucket_rate"),
        "exact_repeat_count": reward.get("explore_agent57_episodic_exact_repeat_count"),
    }


def render_trajectory(label: str, path: Path, output_root: Path) -> tuple[list[str], dict[str, Any]]:
    data = load_traj(path) or {}
    summary = traj_summary(data)
    lines: list[str] = []
    lines.append(f"### {label}: `{path.parent.name}`")
    lines.append("")
    lines.append(f"- 文件：`{rel_path(path, output_root)}`")
    lines.append(
        "- 摘要："
        f"task=`{summary.get('task')}` status=`{summary.get('status')}` "
        f"rollout=`{summary.get('rollout_id')}` train_step=`{summary.get('train_step')}` "
        f"turns=`{summary.get('turns')}` raw=`{fmt(summary.get('raw_score'), 3)}` "
        f"total=`{fmt(summary.get('total_reward'), 3)}`"
    )
    lines.append(
        "- 探索："
        f"arm=`{summary.get('arm_id')}` beta=`{fmt(summary.get('beta'), 3)}` "
        f"trust=`{fmt(summary.get('trust'), 3)}` suppressed=`{summary.get('suppressed_reason') or ''}` "
        f"intrinsic=`{fmt(summary.get('intrinsic_signal'), 3)}` "
        f"ngu=`{fmt(summary.get('ngu_bonus'), 7)}` "
        f"unique/seen=`{summary.get('unique_keys')}/{summary.get('seen_before')}`"
    )
    lines.append("")
    turns = data.get("turns") if isinstance(data.get("turns"), list) else []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        uncertainty = turn.get("uncertainty") if isinstance(turn.get("uncertainty"), dict) else {}
        highlight = []
        if turn.get("parse_error_recorded"):
            highlight.append("parse_error")
        if uncertainty.get("low_progress_from_prev"):
            highlight.append("low_progress")
        tool_calls = [tc for tc in (turn.get("tool_calls") or []) if isinstance(tc, dict)]
        err_count = 0
        for tc in tool_calls:
            result = str(tc.get("result") or "").lower()
            if any(token in result for token in ("error", "traceback", "failed", "permission denied", "not found", "timeout", "500")):
                err_count += 1
        if err_count:
            highlight.append(f"tool_error={err_count}")
        marker = f" **高亮：{', '.join(highlight)}**" if highlight else ""
        lines.append(
            f"#### Turn {turn.get('turn_idx')} {marker}\n"
            f"- finish=`{turn.get('finish_reason')}` latency_ms=`{fmt(turn.get('latency_ms'), 1)}` "
            f"tokens=`{turn.get('n_input_tokens')}/{turn.get('n_output_tokens')}` "
            f"uncertainty=`{fmt(uncertainty.get('turn_level_uncertainty'), 4)}`"
        )
        lines.append("")
        assistant_output = turn.get("assistant_output")
        if assistant_output:
            lines.append("Assistant:")
            lines.append("```text")
            lines.append(compact(assistant_output))
            lines.append("```")
        if tool_calls:
            lines.append("Tool calls / observations:")
            for idx, tc in enumerate(tool_calls, start=1):
                lines.append(f"- call {idx}: `{tc.get('tool_name') or tc.get('name')}`")
                lines.append("  - args:")
                lines.append("```json")
                args = tc.get("args")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False, default=str, indent=2)
                lines.append(compact(args))
                lines.append("```")
                lines.append("  - observation:")
                lines.append("```text")
                result = tc.get("result")
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False, default=str, indent=2)
                lines.append(compact(result))
                lines.append("```")
        lines.append("")
    return lines, summary


def render_case_studies(cases: list[dict[str, Any]], out_path: Path, output_root: Path) -> list[dict[str, Any]]:
    lines: list[str] = []
    case_summaries: list[dict[str, Any]] = []
    lines.append("# 典型轨迹 Case-Study 详情")
    lines.append("")
    lines.append("说明：本文件按完整 turn 顺序展示代表轨迹。极长工具 observation 会在本 Markdown 中截断，原始完整 JSON 保留在对应 `traj.json`。")
    lines.append("")
    for case in cases:
        lines.append(f"## {case['title']}")
        lines.append("")
        lines.append(case["reason"])
        lines.append("")
        entries = []
        for label, path in case["paths"]:
            rendered, summary = render_trajectory(label, path, output_root)
            entries.append({"label": label, "path": str(path), "summary": summary})
            lines.extend(rendered)
        case_summaries.append({"case_id": case["case_id"], "title": case["title"], "reason": case["reason"], "entries": entries})
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return case_summaries


def m(stats_obj: dict[str, Any], field: str, key: str = "mean") -> Any:
    item = stats_obj.get(field) if isinstance(stats_obj, dict) else None
    return item.get(key) if isinstance(item, dict) else None


def comparison_rows(same: dict[str, Any], same_base: dict[str, Any], last50: dict[str, Any], last50_base: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("raw_reward 均值", "raw_reward", False, same, same_base),
        ("valid_raw_weighted", "valid_raw_weighted", False, same, same_base),
        ("op_raw", "op_raw", False, same, same_base),
        ("raw_reward 后50有效点", "raw_reward", False, last50, last50_base),
        ("total_reward 均值", "total_reward", False, same, same_base),
        ("truncated_fraction 均值", "truncated_fraction", True, same, same_base),
        ("fail_fraction", "fail_fraction", True, same, same_base),
        ("response_length 均值", "response_length", True, same, same_base),
        ("raw_reward std", "raw_reward", True, same, same_base),
        ("trainable_count 均值", "trainable_count", False, same, same_base),
    ]
    rows = []
    for label, field, lower, left, right in specs:
        key = "std" if label.endswith("std") else "mean"
        row = compare_values(m(left, field, key), m(right, field, key), lower_is_better=lower)
        row["label"] = label
        row["lower_is_better"] = lower
        rows.append(row)
    return rows


def md_compare_table(rows: list[dict[str, Any]], exp_label: str = "本次实验") -> str:
    lines = [
        f"| 指标 | {exp_label} | baseline | 差值 | 相对变化 | 判断 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {fmt(row['exp'], 4)} | {fmt(row['baseline'], 4)} | "
            f"{fmt(row['diff'], 4)} | {fmt_pct(row['pct_delta'], 1)} | {row['verdict']} |"
        )
    return "\n".join(lines)


def md_stats_table(title: str, fields: list[tuple[str, str]], summary: dict[str, Any]) -> str:
    lines = [f"**{title}**", "", "| 指标 | n | mean | first10 | last10 | std |", "|---|---:|---:|---:|---:|---:|"]
    for label, field in fields:
        item = summary.get(field) or {}
        lines.append(
            f"| {label} | {item.get('n', 0)} | {fmt(item.get('mean'), 4)} | "
            f"{fmt(item.get('first10_mean'), 4)} | {fmt(item.get('last10_mean'), 4)} | {fmt(item.get('std'), 4)} |"
        )
    return "\n".join(lines)


def render_report(
    *,
    run_dir: Path,
    baseline_run_dir: Path,
    out_dir: Path,
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    same: dict[str, Any],
    same_base: dict[str, Any],
    first_n: dict[str, Any],
    first_n_base: dict[str, Any],
    last50: dict[str, Any],
    last50_base: dict[str, Any],
    compare_rows_data: list[dict[str, Any]],
    thresholds: dict[str, Any],
    correlations: dict[str, Any],
    arm: dict[str, Any],
    case_summaries: list[dict[str, Any]],
    standard_summary: dict[str, Any] | None,
    trajectory_classification: dict[str, Any] | None,
    hang: dict[str, Any] | None,
    figs: dict[str, Path],
) -> str:
    exp_label = "本次实验"
    max_common = same.get("last_rollout")
    raw_cmp = compare_values(m(same, "raw_reward"), m(same_base, "raw_reward"))
    op_cmp = compare_values(m(same, "op_raw"), m(same_base, "op_raw"))
    trunc_cmp = compare_values(m(same, "truncated_fraction"), m(same_base, "truncated_fraction"), lower_is_better=True)
    fail_cmp = compare_values(m(same, "fail_fraction"), m(same_base, "fail_fraction"), lower_is_better=True)
    final_cmp = compare_values(m(last50, "raw_reward"), m(last50_base, "raw_reward"))
    threshold_04 = thresholds.get("0.40") or {}
    threshold_04_base = thresholds.get("baseline_0.40") or {}
    threshold_04_text = delta_text(threshold_04.get("rollout_id"), threshold_04_base.get("rollout_id"))
    trunc_direction = "降低" if num(trunc_cmp.get("diff")) is not None and num(trunc_cmp.get("diff")) < 0 else ("升高" if num(trunc_cmp.get("diff")) is not None and num(trunc_cmp.get("diff")) > 0 else "持平")
    fail_direction = "降低" if num(fail_cmp.get("diff")) is not None and num(fail_cmp.get("diff")) < 0 else ("升高" if num(fail_cmp.get("diff")) is not None and num(fail_cmp.get("diff")) > 0 else "持平")
    expl_first = m(summary, "exploration_abs", "first10_mean")
    expl_last = m(summary, "exploration_abs", "last10_mean")
    episodic_first = m(summary, "ngu_episodic", "first10_mean")
    episodic_last = m(summary, "ngu_episodic", "last10_mean")
    life_mod_first = m(summary, "ngu_life_mod", "first10_mean")
    life_mod_last = m(summary, "ngu_life_mod", "last10_mean")
    fused_first = m(summary, "ngu_bonus", "first10_mean")
    fused_last = m(summary, "ngu_bonus", "last10_mean")
    suppress_first = m(summary, "top_suppressed_ratio", "first10_mean")
    suppress_last = m(summary, "top_suppressed_ratio", "last10_mean")
    reset500 = None
    gen_failed = None
    if standard_summary:
        reset500 = ((standard_summary.get("reset500") or {}).get("total"))
        gen_failed = (((standard_summary.get("no_training_diagnostics") or {}).get("error_counts") or {}).get("generate_failed"))
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    train_log_path = run_dir / "logs" / "train.log"
    train_newer_than_metrics = (
        metrics_path.exists()
        and train_log_path.exists()
        and train_log_path.stat().st_mtime > metrics_path.stat().st_mtime + 60
    )

    lines: list[str] = []
    lines.append("# 探索算法实验分析报告")
    lines.append("")
    lines.append(f"- 本次实验：`{run_dir.name}`")
    lines.append(f"- Baseline：`{baseline_run_dir.name}`")
    lines.append(f"- 输出目录：`{out_dir}`")
    lines.append(f"- 结构化指标截止：rollout `{summary['last_rollout']}`，有效点 `{summary['n_valid']}` / `{summary['n_points']}`")
    lines.append(f"- 文件时间：`metrics.jsonl` {mtime_text(metrics_path)}；`train.log` {mtime_text(train_log_path)}")
    if train_newer_than_metrics:
        lines.append("- 运行状态说明：`train.log` 晚于 `metrics.jsonl`，说明下一轮 rollout 可能仍在生成；本报告只统计已经写入 `metrics.jsonl` 的完成 rollout。")
    lines.append("")
    lines.append("## 执行摘要")
    lines.append("")
    lines.append(
        f"1. 同 rollout<=`{max_common}` 的有效训练点比较，{exp_label} raw_reward/pass rate 均值为 **{fmt(raw_cmp['exp'], 4)}**，baseline 为 **{fmt(raw_cmp['baseline'], 4)}**，差值 **{fmt(raw_cmp['diff'], 4)} ({fmt_pct(raw_cmp['pct_delta'], 1)})**，判断为 **{raw_cmp['verdict']}**。"
    )
    lines.append(
        f"2. operational pass (`op_raw`) 为 **{fmt(op_cmp['exp'], 4)}** vs baseline **{fmt(op_cmp['baseline'], 4)}**，差值 **{fmt(op_cmp['diff'], 4)} ({fmt_pct(op_cmp['pct_delta'], 1)})**；环境/空批次失败率 `fail_fraction` {fail_direction}到 **{fmt(fail_cmp['exp'], 4)}**，baseline 为 **{fmt(fail_cmp['baseline'], 4)}**。"
    )
    lines.append(
        f"3. truncated_fraction 从 baseline **{fmt(trunc_cmp['baseline'], 4)}** 到 {exp_label} **{fmt(trunc_cmp['exp'], 4)}**，{trunc_direction} **{fmt(abs(trunc_cmp['diff'] or 0), 4)} ({fmt_pct(abs(trunc_cmp['pct_delta'] or 0), 1)})**，判断为 **{trunc_cmp['verdict']}**。rolling10 raw_reward 达到 0.40 的 rollout 为 {exp_label} `{threshold_04.get('rollout_id')}`、baseline `{threshold_04_base.get('rollout_id')}`，{threshold_04_text}。"
    )
    lines.append(
        f"4. 探索链路已按“局内 episodic -> 局间 lifelong -> 融合 NGU -> UCB arm”拆解：局内 NGU episodic first10/last10 为 **{fmt(episodic_first, 4)} -> {fmt(episodic_last, 4)}**，局间 life modifier 为 **{fmt(life_mod_first, 4)} -> {fmt(life_mod_last, 4)}**，融合 NGU bonus 为 **{fmt(fused_first, 7)} -> {fmt(fused_last, 7)}**，abs exploration reward 为 **{fmt(expl_first, 5)} -> {fmt(expl_last, 5)}**，top suppressed ratio 为 **{fmt(suppress_first, 4)} -> {fmt(suppress_last, 4)}**。探索信号与 raw_reward 的 Pearson r 约 **{fmt((correlations.get('exploration_abs') or {}).get('pearson_raw_reward'), 3)}**，只是弱正相关。"
    )
    lines.append(
        f"5. 数据质量仍是主要混杂因素：{exp_label}的空/不可训练批次 **{summary['n_invalid_or_zero_trainable']} / {summary['n_points']} ({fmt_pct(summary['invalid_fraction'])})**；标准日志还记录 reset_500=`{reset500}`、generate_failed=`{gen_failed}`，因此报告中的算法结论均按有效点、同 rollout 窗口和 operational 口径同时报告。"
    )
    lines.append("")
    lines.append("## 数据说明")
    lines.append("")
    lines.append("- 主要训练指标来自 `logs/metrics.jsonl`；若字段缺失，才回退到 `train.log` 解析。")
    lines.append("- SETA 的 `raw_reward`/`test_acc` 在该日志中表示当前 rollout 任务 unit-test pass rate，范围为 0-1，不是 held-out test set。")
    lines.append("- 有效训练点定义为 `trainable_count > 0` 且 `raw_reward != null`；`trainable_count=0` 的兼容空批次不计入算法性能均值。")
    lines.append("- `valid_raw_weighted` 用 `trainable_count` 加权；`op_raw=sum(raw_reward*trainable_count)/sum(sample_count)`，把环境失败/空样本纳入分母；`fail_fraction=1-sum(trainable_count)/sum(sample_count)`。")
    lines.append("- 本 run 没有独立 `simhash_coverage` 或 `fp_*` 聚合字段；报告用 Agent57 的 episodic/lifelong 聚合字段作为 SimHash 覆盖、局内新颖性和局间新颖性的代理指标。")
    lines.append("- 保存轨迹中的 `exploration_reward` 多数为 0，因为保存阶段标记为 `generate_pre_reward_postprocess`；轨迹 case-study 使用 `explore_agent57_*` 字段解释探索状态。")
    lines.append("")
    lines.append("## 提升评估")
    lines.append("")
    lines.append(f"对齐窗口：{exp_label}最后一个有效 rollout 为 `{summary['last_valid_rollout']}`，因此主比较使用 rollout<=`{max_common}` 的有效点。")
    lines.append("")
    lines.append(md_compare_table(compare_rows_data, exp_label=exp_label))
    lines.append("")
    lines.append(f"补充：按“前 N 个有效点”对齐，N 等于{exp_label}有效点数。")
    lines.append("")
    lines.append(f"| 指标 | {exp_label}前N有效点 | baseline 前N有效点 | 差值 | 相对变化 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, field in (("raw_reward", "raw_reward"), ("valid_raw_weighted", "valid_raw_weighted"), ("op_raw", "op_raw"), ("truncated_fraction", "truncated_fraction"), ("fail_fraction", "fail_fraction"), ("response_length", "response_length")):
        lower = label == "truncated_fraction" or label == "response_length"
        if label == "fail_fraction":
            lower = True
        row = compare_values(m(first_n, field), m(first_n_base, field), lower_is_better=lower)
        lines.append(f"| {label} | {fmt(row['exp'], 4)} | {fmt(row['baseline'], 4)} | {fmt(row['diff'], 4)} | {fmt_pct(row['pct_delta'])} |")
    lines.append("")
    lines.append("**收敛速度（rolling10 raw_reward）**")
    lines.append("")
    lines.append(f"| 阈值 | {exp_label} rollout | baseline rollout | {exp_label} rolling mean | baseline rolling mean |")
    lines.append("|---:|---:|---:|---:|---:|")
    for threshold in ("0.30", "0.35", "0.40", "0.45"):
        item = thresholds.get(threshold) or {}
        base_item = thresholds.get(f"baseline_{threshold}") or {}
        lines.append(
            f"| {threshold} | {item.get('rollout_id', 'NA')} | {base_item.get('rollout_id', 'NA')} | "
            f"{fmt(item.get('rolling_mean'), 4)} | {fmt(base_item.get('rolling_mean'), 4)} |"
        )
    lines.append("")
    lines.append("图表：")
    lines.append(f"- ![core comparison]({rel_path(figs['core'], out_dir)})")
    lines.append(f"- 标准训练 overview：![overview]({rel_path(figs['overview'], out_dir)})")
    lines.append("")
    lines.append("说明：`baseline_core_comparison.png` 使用“有效 rollout step index”作为横轴，即两条曲线都按 `trainable_count>0` 的有效点顺序绘制；baseline 原始 rollout 98-213 段存在大量空/不可训练点，如果按原始 rollout_id 截到 160 多会自然停在约 100，因此图中改用 first-N 有效点保证曲线覆盖到相同横轴末端。上方表格仍保留同原始 rollout 窗口和前 N 有效点两种定量口径。")
    lines.append("")
    lines.append(f"结论：{exp_label} raw_reward 同窗口判断为 **{raw_cmp['verdict']}**，后 50 有效点判断为 **{final_cmp['verdict']}**；truncation 判断为 **{trunc_cmp['verdict']}**，operational pass 判断为 **{op_cmp['verdict']}**。最终是否优于 baseline 需要同时看 raw/pass、op_raw、fail/trunc 和收敛速度，而不能只看单个成功样本。")
    lines.append("")
    lines.append("## 探索指标")
    lines.append("")
    lines.append(
        md_stats_table(
            f"{exp_label}有效点探索统计",
            [
                ("局内: NGU episodic", "ngu_episodic"),
                ("局内: empty-bucket rate", "episodic_empty_bucket_rate"),
                ("局内: exact repeats", "episodic_exact_repeat_count"),
                ("局间: lifelong raw novelty", "lifelong_raw"),
                ("局间: NGU life modifier", "ngu_life_mod"),
                ("局间: lifelong bonus", "lifelong_bonus"),
                ("局间: new-state proxy", "new_state_proxy"),
                ("融合: NGU bonus", "ngu_bonus"),
                ("融合: abs exploration reward", "exploration_abs"),
                ("UCB: top arm id", "top_arm"),
                ("UCB: top-arm share", "top_arm_ratio"),
                ("UCB: top suppressed ratio", "top_suppressed_ratio"),
            ],
            summary,
        )
    )
    lines.append("")
    lines.append("**探索-性能相关性（Pearson r, raw_reward）**")
    lines.append("")
    lines.append("| 探索指标 | n | r |")
    lines.append("|---|---:|---:|")
    for name, item in correlations.items():
        lines.append(f"| {name} | {item.get('n', 0)} | {fmt(item.get('pearson_raw_reward'), 3)} |")
    lines.append("")
    if arm.get("available"):
        lines.append(f"SQLite `arm_events` 共 **{fmt_int(arm.get('n_events'))}** 条。lifelong_counts key 数为 **{fmt_int((arm.get('lifelong_counts') or {}).get('n_keys'))}**，平均计数 **{fmt((arm.get('lifelong_counts') or {}).get('count_mean'), 3)}**。")
        lines.append("")
        lines.append("| arm | n | normalized_base | success_rate | trunc_rate | parse_rate | bonus_mean |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for item in arm.get("by_arm") or []:
            lines.append(
                f"| {item['arm_id']} | {item['n']} | {fmt(item['normalized_base_score_mean'], 4)} | "
                f"{fmt(item['success_rate'], 4)} | {fmt(item['truncated_rate'], 4)} | "
                f"{fmt(item['parse_error_rate'], 4)} | {fmt(item['bonus_mean'], 7)} |"
            )
        lines.append("")
    lines.append("图表：")
    lines.append(f"- ![exploration trends]({rel_path(figs['exploration'], out_dir)})")
    lines.append(f"- ![exploration relationship]({rel_path(figs['relationship'], out_dir)})")
    lines.append(f"- ![arm by arm]({rel_path(figs['arm_by_arm'], out_dir)})")
    lines.append(f"- ![arm over time]({rel_path(figs['arm_time'], out_dir)})")
    lines.append("")
    lines.append(f"解释：`exploration_metrics_trends.png` 现在按 Agent57 计算链路组织：左上是局内 episodic novelty，右上是局间 lifelong novelty/modifier，并同时显示 `unique_keys`、`seen_before`、`new-state proxy` 等 coverage 代理指标；左下是融合后的 NGU/实际探索奖励，右下是 UCB arm 选择与 trust/lifelong suppression。`top_suppressed_ratio` 若为负相关，通常说明被截断/parse error 等信任门控压制的探索样本表现更差。baseline 没有 Agent57/SimHash/UCB 字段，因此探索行为差异只能通过{exp_label}的专属指标与 baseline 的空 exploration 字段对照。")
    lines.append("")
    lines.append("## Case-Study")
    lines.append("")
    if trajectory_classification:
        dist = trajectory_classification.get("class_distribution") or {}
        lines.append(
            f"{exp_label}保存轨迹共 `{trajectory_classification.get('n_trajectories')}` 条：pass `{dist.get('pass', 0)}`，fail_eval_normal `{dist.get('fail_eval_normal', 0)}`，truncated `{dist.get('truncated', 0)}`，fail_eval_500 `{dist.get('fail_eval_500', 0)}`。"
        )
        lines.append("")
    lines.append(f"完整 turn 级内容见 [`case_study_details.md`]({rel_path(out_dir / 'case_study_details.md', out_dir)})。")
    lines.append("")
    lines.append("| Case | 轨迹 | status | raw | total | arm/beta | trust | 关键解释 |")
    lines.append("|---|---|---|---:|---:|---|---:|---|")
    for case in case_summaries:
        for entry in case.get("entries") or []:
            s = entry["summary"]
            arm_beta = "无" if s.get("arm_id") is None and num(s.get("beta")) is None else f"{s.get('arm_id')}/{fmt(s.get('beta'), 3)}"
            lines.append(
                f"| {md_escape(case['title'])} | {md_escape(entry['label'])} | `{s.get('status')}` | "
                f"{fmt(s.get('raw_score'), 3)} | {fmt(s.get('total_reward'), 3)} | "
                f"{arm_beta} | {fmt(s.get('trust'), 3)} | "
                f"{md_escape(one_line(case['reason'], 110))} |"
            )
    lines.append("")
    if case_summaries:
        case_titles = "、".join(case.get("title", "") for case in case_summaries[:3])
        lines.append(f"Case 结论：本次自动选取了 `{case_titles}` 等代表轨迹。成功 case 可证明探索 arm 在个别样本上能产生有效完成；失败/截断 case 则显示覆盖度或 bonus 升高并不等价于任务完成，仍需要 trust gate、截断约束和最终 pass/raw_reward 共同判断。若报告中只有同 task 而非同 sample 的 baseline 对照，该 case 只作为行为参考，不作为严格因果证据。")
    else:
        lines.append("Case 结论：未在保存轨迹中找到可用代表 case；请以训练曲线和结构化指标为主。")
    lines.append("")
    lines.append("## 结论与建议")
    lines.append("")
    lines.append(f"1. 相对 SETA-DAPO baseline，{exp_label} raw_reward/pass rate 为 **{raw_cmp['verdict']}**，op_raw 为 **{op_cmp['verdict']}**，后 50 有效点 raw_reward 为 **{final_cmp['verdict']}**；这些是判断整体提升的主依据。")
    lines.append(f"2. 探索侧的主要信号是 abs exploration reward、lifelong unique keys/new-state proxy、UCB top suppressed ratio 与 arm_events 的趋势；若这些指标上升但 raw/op_raw 未同步提升，应视为“探索活跃但尚未稳定转化为性能”。")
    lines.append(f"3. 风险侧重点看 truncation、fail_fraction 与 response_length：本次 truncation 为 **{trunc_cmp['verdict']}**，fail_fraction 为 **{fail_cmp['verdict']}**。若二者任一退化，建议优先调低探索强度或增加 explicit truncation penalty。")
    lines.append("4. 下一轮建议固定同 rollout 预算、同有效样本数、后 50 有效点和 operational pass 四套口径，并继续保留 task-level case-study；对 `trainable_count=0` 批次建议从优化 step 中跳过或单独标记。")
    lines.append("")
    lines.append("## 生成文件")
    lines.append("")
    generated = [
        ("report.md", "最终中文分析报告"),
        ("exploration_analysis.json", "结构化统计与对比数据"),
        ("case_study_details.md", "代表轨迹完整 turn 级 case-study"),
        ("figs/baseline_core_comparison.png", "核心指标与 baseline first-N 有效 rollout step 对比"),
        ("figs/exploration_metrics_trends.png", "按局内/局间/融合/UCB 拆解的探索指标趋势"),
        ("figs/exploration_performance_relationship.png", "探索强度与 raw_reward 关系"),
        ("figs/agent57_arm_events_by_arm.png", "SQLite arm_events 按 arm 聚合"),
        ("figs/agent57_arm_events_over_time.png", "SQLite arm_events 随时间变化"),
        ("summary_stats.json", "复用标准脚本生成的训练统计"),
        ("trajectory_classification.json", "复用标准脚本生成的轨迹分类"),
        ("hang_diagnosis.json", "复用标准脚本生成的 hang 诊断"),
    ]
    lines.append("| 文件 | 说明 |")
    lines.append("|---|---|")
    for path, desc in generated:
        lines.append(f"| `{path}` | {desc} |")
    lines.append("")
    if hang:
        assessment = hang.get("assessment") if isinstance(hang.get("assessment"), dict) else {}
        likelihood = (
            hang.get("likelihood")
            or assessment.get("likelihood")
            or assessment.get("similar_dynamic_sampling_env_hang_likelihood")
        )
        lines.append(f"Hang 诊断：`{likelihood}`。")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    baseline_log = args.baseline_log.expanduser().resolve()
    baseline_run_dir = baseline_log.parent.parent
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else run_dir / "metrics" / "analysis"
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    records, source = load_records(run_dir, run_dir / "logs" / "train.log")
    baseline_records, baseline_source = load_records(baseline_run_dir, baseline_log)
    if not records:
        raise SystemExit(f"no metrics parsed for {run_dir}")
    if not baseline_records:
        raise SystemExit(f"no metrics parsed for baseline {baseline_run_dir}")
    add_derived(records)
    add_derived(baseline_records)

    summary = summarize_records(records)
    baseline_summary = summarize_records(baseline_records)
    max_common = int(min(summary["last_valid_rollout"], baseline_summary["last_valid_rollout"]))
    same = window_summary(records, max_rollout=max_common)
    same_base = window_summary(baseline_records, max_rollout=max_common)
    first_n_count = int(summary["n_valid"])
    first_n = window_summary(records, first_n_valid=first_n_count)
    first_n_base = window_summary(baseline_records, first_n_valid=first_n_count)
    last50 = window_summary(records, max_rollout=max_common, last_n_valid=50)
    last50_base = window_summary(baseline_records, max_rollout=max_common, last_n_valid=50)
    compare_rows_data = comparison_rows(same, same_base, last50, last50_base)

    thresholds: dict[str, Any] = {}
    for threshold in (0.30, 0.35, 0.40, 0.45):
        key = f"{threshold:.2f}"
        thresholds[key] = rolling_threshold(records, threshold)
        thresholds[f"baseline_{key}"] = rolling_threshold(baseline_records, threshold)

    correlations = correlation_summary(records)
    arm = load_arm_events(run_dir / "agent57_lite.sqlite3")

    figs = {
        "core": figs_dir / "baseline_core_comparison.png",
        "exploration": figs_dir / "exploration_metrics_trends.png",
        "relationship": figs_dir / "exploration_performance_relationship.png",
        "arm_by_arm": figs_dir / "agent57_arm_events_by_arm.png",
        "arm_time": figs_dir / "agent57_arm_events_over_time.png",
        "overview": figs_dir / "overview.png",
    }
    plot_core(records, baseline_records, figs["core"], max_common, exp_label="experiment")
    plot_exploration(records, figs["exploration"])
    plot_relationship(records, correlations, figs["relationship"])
    plot_arm_events(arm, figs["arm_by_arm"], figs["arm_time"])

    cases = pick_cases(run_dir, baseline_run_dir)
    case_summaries = render_case_studies(cases, out_dir / "case_study_details.md", ROOT)

    standard_summary = load_json(out_dir / "summary_stats.json")
    trajectory_classification = load_json(out_dir / "trajectory_classification.json")
    hang = load_json(out_dir / "hang_diagnosis.json")

    analysis = {
        "schema": "openclaw.exploration_report.v1",
        "run_dir": str(run_dir),
        "baseline_run_dir": str(baseline_run_dir),
        "metric_source": source,
        "baseline_metric_source": baseline_source,
        "summary": summary,
        "baseline_summary": baseline_summary,
        "same_rollout_window": {"max_rollout": max_common, "run": same, "baseline": same_base},
        "first_n_valid_window": {"n": first_n_count, "run": first_n, "baseline": first_n_base},
        "last50_same_rollout_window": {"run": last50, "baseline": last50_base},
        "comparison_rows": compare_rows_data,
        "convergence_thresholds": thresholds,
        "correlations": correlations,
        "agent57_arm_events": arm,
        "case_studies": case_summaries,
        "figures": {name: str(path) for name, path in figs.items()},
    }
    (out_dir / "exploration_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    report = render_report(
        run_dir=run_dir,
        baseline_run_dir=baseline_run_dir,
        out_dir=out_dir,
        summary=summary,
        baseline_summary=baseline_summary,
        same=same,
        same_base=same_base,
        first_n=first_n,
        first_n_base=first_n_base,
        last50=last50,
        last50_base=last50_base,
        compare_rows_data=compare_rows_data,
        thresholds=thresholds,
        correlations=correlations,
        arm=arm,
        case_summaries=case_summaries,
        standard_summary=standard_summary,
        trajectory_classification=trajectory_classification,
        hang=hang,
        figs=figs,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"[+] wrote {out_dir / 'exploration_analysis.json'}")
    print(f"[+] wrote {out_dir / 'case_study_details.md'}")
    print(f"[+] wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
