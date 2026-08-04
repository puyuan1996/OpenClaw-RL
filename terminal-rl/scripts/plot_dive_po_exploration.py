#!/usr/bin/env python3
"""Refresh DiVE-PO-specific exploration figures and a concise run report."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openclaw-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXP_COLOR = "#4C78A8"
BASE_COLOR = "#F58518"
ARM_BETAS = [0.000, 0.004, 0.008, 0.012, 0.016, 0.022]


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value: Any, default: float = float("nan")) -> float:
    return float(value) if finite(value) else default


def values(records: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([number(record.get(field)) for record in records], dtype=float)


def mean(array: Iterable[float]) -> float:
    data = np.asarray(list(array), dtype=float)
    data = data[np.isfinite(data)]
    return float(np.mean(data)) if data.size else float("nan")


def std(array: Iterable[float]) -> float:
    data = np.asarray(list(array), dtype=float)
    data = data[np.isfinite(data)]
    return float(np.std(data)) if data.size else float("nan")


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def rolling(array: Iterable[float], window: int = 10) -> np.ndarray:
    data = np.asarray(list(array), dtype=float)
    result = np.full(data.shape, np.nan, dtype=float)
    for index in range(data.size):
        chunk = data[max(0, index - window + 1) : index + 1]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            result[index] = float(np.mean(chunk))
    return result


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    if int(mask.sum()) < 3 or np.std(x_array[mask]) == 0 or np.std(y_array[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x_array[mask], y_array[mask])[0, 1])


def load_metrics(path: Path) -> list[dict[str, Any]]:
    by_rollout: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rollout_id = record.get("rollout_id")
            if record.get("dataset") == "seta" and record.get("phase") == "train" and isinstance(rollout_id, int):
                by_rollout[rollout_id] = record
    return [by_rollout[key] for key in sorted(by_rollout)]


def valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if number(record.get("trainable_count"), 0.0) > 0 and finite(record.get("raw_reward"))
    ]


def metric_stats(records: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    data = values(records, field)
    data = data[np.isfinite(data)]
    return {
        "n": int(data.size),
        "mean": mean(data),
        "first10": mean(data[:10]),
        "last10": mean(data[-10:]),
        "std": std(data),
        "min": float(np.min(data)) if data.size else float("nan"),
        "max": float(np.max(data)) if data.size else float("nan"),
    }


def operational_pass(records: list[dict[str, Any]]) -> float:
    weighted = sum(
        number(record.get("raw_reward"), 0.0) * max(number(record.get("trainable_count"), 0.0), 0.0)
        for record in records
    )
    sampled = sum(max(number(record.get("sample_count"), 0.0), 0.0) for record in records)
    return ratio(weighted, sampled)


def save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def add_series(axis: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: str) -> None:
    axis.scatter(x, y, color=color, s=8, alpha=0.16, linewidths=0)
    axis.plot(x, rolling(y), color=color, linewidth=2.0, label=label)


def plot_core(
    experiment: list[dict[str, Any]], baseline: list[dict[str, Any]], path: Path
) -> None:
    common = min(len(experiment), len(baseline))
    exp = experiment[:common]
    base = baseline[:common]
    x = np.arange(1, common + 1)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    specs = (
        ("raw_reward", "Pass-rate/raw_reward", "raw_reward"),
        ("total_reward", "Total reward", "total_reward"),
        ("truncated_fraction", "Truncated fraction", "fraction"),
        ("response_length", "Response length", "tokens"),
    )
    for axis, (field, title, ylabel) in zip(axes.flat, specs):
        add_series(axis, x, values(exp, field), "experiment rolling10", EXP_COLOR)
        add_series(axis, x, values(base, field), "baseline rolling10", BASE_COLOR)
        if field == "total_reward":
            axis.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("effective valid rollout step")
        axis.legend(loc="best", fontsize=8)
        axis.grid(alpha=0.22)
    figure.suptitle(
        f"Core metric comparison by valid rollout step (N={common}, "
        f"experiment rollout<= {exp[-1]['rollout_id']}, baseline rollout<= {base[-1]['rollout_id']})",
        fontsize=15,
    )
    save(figure, path)


def plot_exploration(records: list[dict[str, Any]], path: Path) -> None:
    x = np.asarray([record["rollout_id"] for record in records], dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.7))

    # 1. Episodic novelty and exact repeats.
    axis = axes[0, 0]
    episodic = values(records, "agent57/ngu_episodic")
    empty = values(records, "agent57/episodic_empty_bucket_rate")
    repeats = values(records, "agent57/episodic_exact_repeat_count")
    axis.plot(x, episodic, marker=".", markersize=2.5, linewidth=1.0, label="NGU episodic")
    axis.plot(x, empty, marker=".", markersize=2.5, linewidth=1.0, label="empty-bucket rate")
    axis.set_title("1. In-episode intrinsic reward")
    axis.set_ylabel("episodic novelty / ratio")
    twin = axis.twinx()
    twin.plot(x, repeats, color="#9C755F", alpha=0.65, linewidth=1.0, label="exact repeats")
    twin.set_ylabel("exact repeat count")
    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axis.legend(handles + twin_handles, labels + twin_labels, loc="best", fontsize=8)

    # 2. Lifelong novelty, modifier, and coverage proxies.
    axis = axes[0, 1]
    life_raw = values(records, "agent57/lifelong_raw")
    life_mod = values(records, "agent57/ngu_life_mod")
    life_bonus = values(records, "agent57/lifelong_bonus") * 1e5
    seen = values(records, "agent57/lifelong_seen_before")
    unique = values(records, "agent57/lifelong_unique_keys")
    new_state = 10.0 / (np.maximum(seen, 0.0) + 1.0)
    axis.plot(x, life_raw, marker=".", markersize=2.5, linewidth=1.0, label="lifelong raw novelty")
    axis.plot(x, life_mod, marker=".", markersize=2.5, linewidth=1.0, label="NGU life modifier")
    axis.plot(x, life_bonus, marker=".", markersize=2.5, linewidth=1.0, label="lifelong bonus x1e5")
    axis.plot(x, new_state, marker=".", markersize=2.5, linewidth=1.0, label="new-state proxy x10")
    axis.set_title("2. Lifelong intrinsic reward")
    axis.set_ylabel("novelty / modifier / scaled bonus")
    twin = axis.twinx()
    twin.plot(x, np.log10(np.maximum(unique, 0.0) + 1.0), color="#BCBD22", linewidth=1.0, label="log10(unique keys + 1)")
    twin.plot(x, np.log10(np.maximum(seen, 0.0) + 1.0), color="#E45756", linewidth=1.0, label="log10(seen before + 1)")
    twin.set_ylabel("coverage proxy count (log10)")
    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axis.legend(handles + twin_handles, labels + twin_labels, loc="best", fontsize=7)

    # 3. Fused signal and the magnitude actually injected into the advantage stream.
    axis = axes[1, 0]
    ngu = values(records, "agent57/ngu_bonus")
    signal = values(records, "exploration_reward_signal")
    exploration_abs = values(records, "exploration_reward_abs")
    axis.plot(x, ngu, marker=".", markersize=2.5, linewidth=1.0, label="fused NGU bonus")
    axis.plot(x, signal, marker=".", markersize=2.5, linewidth=1.0, label="exploration signal")
    axis.set_title("3. Fused intrinsic reward")
    axis.set_ylabel("intrinsic signal / NGU bonus")
    twin = axis.twinx()
    twin.plot(x, exploration_abs, color="#54A24B", marker=".", markersize=2.5, linewidth=1.0, label="abs exploration reward")
    twin.set_ylabel("abs reward contribution")
    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axis.legend(handles + twin_handles, labels + twin_labels, loc="best", fontsize=8)

    # 4. UCB allocation and outcome-aware suppression.
    axis = axes[1, 1]
    top_arm = values(records, "agent57/top_arm")
    top_share = values(records, "agent57/top_arm_ratio")
    suppressed = values(records, "agent57/top_suppressed_ratio")
    trust = values(records, "agent57/trust_mean")
    axis.scatter(x, top_arm, color="gray", s=9, alpha=0.65, label="top arm id")
    axis.set_ylabel("top arm id")
    twin = axis.twinx()
    twin.plot(x, top_share, linewidth=1.0, label="top-arm share")
    twin.plot(x, suppressed, color="#E45756", linewidth=1.0, label="top suppressed share")
    twin.plot(x, trust, color="#54A24B", linewidth=1.0, alpha=0.8, label="mean trust")
    twin.set_ylim(-0.02, 1.02)
    twin.set_ylabel("share / ratio")
    axis.set_title("4. UCB arm selection and suppression")
    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axis.legend(handles + twin_handles, labels + twin_labels, loc="best", fontsize=8)
    axis.text(
        0.0,
        0.01,
        "arm beta: 0=.000, 1=.004, 2=.008, 3=.012, 4=.016, 5=.022",
        transform=axis.transAxes,
        fontsize=7,
        color="dimgray",
    )

    for axis in axes.flat:
        axis.set_xlabel("rollout_id")
        axis.grid(alpha=0.22)
    figure.suptitle("DiVE-PO exploration metrics: episodic -> lifelong -> fused -> UCB", fontsize=15)
    save(figure, path)


def correlation_inputs(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    seen = values(records, "agent57/lifelong_seen_before")
    return {
        "exploration_abs": values(records, "exploration_reward_abs"),
        "new_state_proxy": 1.0 / (np.maximum(seen, 0.0) + 1.0),
        "ngu_bonus": values(records, "agent57/ngu_bonus"),
        "episodic_exact_repeat_count": values(records, "agent57/episodic_exact_repeat_count"),
        "top_suppressed_ratio": values(records, "agent57/top_suppressed_ratio"),
        "lifelong_seen_before": seen,
        "lifelong_unique_keys": values(records, "agent57/lifelong_unique_keys"),
        "episodic_empty_bucket_rate": values(records, "agent57/episodic_empty_bucket_rate"),
    }


def plot_relationship(records: list[dict[str, Any]], path: Path) -> dict[str, float]:
    raw = values(records, "raw_reward")
    rollout = np.asarray([record["rollout_id"] for record in records], dtype=float)
    inputs = correlation_inputs(records)
    correlations = {name: pearson(data, raw) for name, data in inputs.items()}

    figure, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    scatters = (
        ("exploration_abs", "Exploration reward vs raw_reward", "abs exploration reward"),
        ("lifelong_unique_keys", "Unique keys vs raw_reward", "lifelong unique keys"),
        ("top_suppressed_ratio", "Suppression vs raw_reward", "top suppressed ratio"),
    )
    for axis, (name, title, xlabel) in zip(axes.flat[:3], scatters):
        mask = np.isfinite(inputs[name]) & np.isfinite(raw)
        points = axis.scatter(inputs[name][mask], raw[mask], c=rollout[mask], cmap="viridis", s=32, alpha=0.72)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("raw_reward")
        axis.grid(alpha=0.22)
        figure.colorbar(points, ax=axis, label="rollout_id")

    axis = axes[1, 1]
    ordered = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
    names = [item[0] for item in ordered]
    data = [item[1] for item in ordered]
    y = np.arange(len(names))
    axis.barh(y, data, color="#5B9BD5")
    axis.set_yticks(y, labels=names)
    axis.invert_yaxis()
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Pearson r")
    axis.set_title("Pearson correlation with raw_reward")
    axis.grid(axis="x", alpha=0.22)
    save(figure, path)
    return correlations


def load_arm_events(path: Path) -> tuple[list[dict[str, float]], int, dict[str, float | int]]:
    if not path.is_file():
        return [], 0, {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute("SELECT * FROM arm_events ORDER BY id")]
    lifelong_count = int(connection.execute("SELECT COUNT(*) FROM lifelong_counts").fetchone()[0])
    life = connection.execute("SELECT AVG(count), MAX(count), SUM(count) FROM lifelong_counts").fetchone()
    connection.close()
    return rows, lifelong_count, {
        "key_count": lifelong_count,
        "mean_count": number(life[0]),
        "max_count": number(life[1]),
        "sum_count": number(life[2]),
    }


def aggregate_arms(events: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    result = []
    for arm in sorted({int(event["arm_id"]) for event in events}):
        rows = [event for event in events if int(event["arm_id"]) == arm]
        result.append(
            {
                "arm": arm,
                "n": len(rows),
                "normalized_base": mean(number(row.get("normalized_base_score")) for row in rows),
                "success_rate": mean(number(row.get("success")) for row in rows),
                "truncated_rate": mean(number(row.get("truncated")) for row in rows),
                "parse_rate": mean(number(row.get("parse_error")) for row in rows),
                "infra_failure_rate": mean(number(row.get("infra_failure")) for row in rows),
                "bonus_mean": mean(number(row.get("bonus")) for row in rows),
            }
        )
    return result


def plot_arm_by_arm(aggregates: list[dict[str, Any]], path: Path) -> None:
    arms = np.asarray([row["arm"] for row in aggregates], dtype=int)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    axes[0].bar(arms, [row["n"] for row in aggregates], color="#5B9BD5")
    axes[0].set_title("Arm event counts")
    axes[0].set_ylabel("events")
    axes[1].plot(arms, [row["success_rate"] for row in aggregates], marker="o", label="success")
    axes[1].plot(arms, [row["truncated_rate"] for row in aggregates], marker="o", label="truncated")
    axes[1].plot(arms, [row["parse_rate"] for row in aggregates], marker="o", label="parse error")
    axes[1].set_title("Outcome rate by arm")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    axes[2].bar(arms, [row["bonus_mean"] for row in aggregates], color="#60B75D")
    axes[2].set_title("Mean bonus by arm")
    axes[2].set_ylabel("bonus")
    for axis in axes:
        axis.set_xlabel("arm")
        axis.grid(alpha=0.22)
    save(figure, path)


def time_bins(events: list[dict[str, Any]], bins: int = 50) -> list[dict[str, Any]]:
    if not events:
        return []
    result = []
    indexes = np.array_split(np.arange(len(events)), min(bins, len(events)))
    for indexes_in_bin in indexes:
        rows = [events[int(index)] for index in indexes_in_bin]
        result.append(
            {
                "first_id": int(rows[0]["id"]),
                "last_id": int(rows[-1]["id"]),
                "n": len(rows),
                "success_rate": mean(number(row.get("success")) for row in rows),
                "truncated_rate": mean(number(row.get("truncated")) for row in rows),
                "parse_rate": mean(number(row.get("parse_error")) for row in rows),
                "bonus_mean": mean(number(row.get("bonus")) for row in rows),
                "arm_mean": mean(number(row.get("arm_id")) for row in rows),
            }
        )
    return result


def plot_arm_time(bins: list[dict[str, Any]], path: Path) -> None:
    x = np.arange(len(bins))
    labels = [f"{row['first_id']}-{row['last_id']}" for row in bins]
    figure, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    axes[0].plot(x, [row["success_rate"] for row in bins], marker="o", label="success")
    axes[0].plot(x, [row["truncated_rate"] for row in bins], marker="o", label="truncated")
    axes[0].plot(x, [row["parse_rate"] for row in bins], marker="o", label="parse error")
    axes[0].set_title("Arm event outcomes over time")
    axes[0].set_ylabel("rate")
    axes[0].legend(loc="best")
    axes[1].plot(x, [row["bonus_mean"] for row in bins], color="#2CA02C", marker="o", label="mean bonus")
    axes[1].set_ylabel("bonus")
    twin = axes[1].twinx()
    twin.plot(x, [row["arm_mean"] for row in bins], color="gray", marker="o", alpha=0.7, label="mean arm")
    twin.set_ylabel("mean arm")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[1].legend(handles + twin_handles, legend_labels + twin_labels, loc="best")
    tick_indexes = np.linspace(0, len(bins) - 1, min(9, len(bins)), dtype=int)
    axes[1].set_xticks(tick_indexes, labels=[labels[index] for index in tick_indexes], rotation=35, ha="right")
    axes[1].set_xlabel("arm event bin")
    for axis in axes:
        axis.grid(alpha=0.22)
    save(figure, path)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fmt(value: float, digits: int = 4) -> str:
    return "NA" if not finite(value) else f"{value:.{digits}f}"


def fmt_metric(value: float) -> str:
    if not finite(value):
        return "NA"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def write_outputs(
    run_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    all_records: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    baseline_valid: list[dict[str, Any]],
    correlations: dict[str, float],
    arm_aggregates: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    lifelong: dict[str, Any],
    figures: dict[str, Path],
) -> None:
    common = min(len(valid), len(baseline_valid))
    exp = valid[:common]
    base = baseline_valid[:common]
    exp_raw = mean(values(exp, "raw_reward"))
    base_raw = mean(values(base, "raw_reward"))
    raw_delta = exp_raw - base_raw
    raw_relative = ratio(raw_delta, base_raw)
    exp_trunc = mean(values(exp, "truncated_fraction"))
    base_trunc = mean(values(base, "truncated_fraction"))
    invalid = len(all_records) - len(valid)
    generated = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M:%S HKT")
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    train_log = run_dir / "logs" / "train.log"
    metric_mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime, ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M:%S HKT")
    train_mtime = datetime.fromtimestamp(train_log.stat().st_mtime, ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M:%S HKT")

    metric_fields = {
        "raw_reward": "raw_reward",
        "task_reward": "task_reward",
        "total_reward": "total_reward",
        "truncated_fraction": "truncated_fraction",
        "response_length": "response_length",
        "exploration_abs": "exploration_reward_abs",
        "exploration_signal": "exploration_reward_signal",
        "ngu_episodic": "agent57/ngu_episodic",
        "episodic_empty_bucket_rate": "agent57/episodic_empty_bucket_rate",
        "episodic_exact_repeat_count": "agent57/episodic_exact_repeat_count",
        "lifelong_raw": "agent57/lifelong_raw",
        "lifelong_unique_keys": "agent57/lifelong_unique_keys",
        "lifelong_seen_before": "agent57/lifelong_seen_before",
        "lifelong_bonus": "agent57/lifelong_bonus",
        "ngu_life_mod": "agent57/ngu_life_mod",
        "ngu_bonus": "agent57/ngu_bonus",
        "top_arm": "agent57/top_arm",
        "top_arm_ratio": "agent57/top_arm_ratio",
        "top_suppressed_ratio": "agent57/top_suppressed_ratio",
        "trust_mean": "agent57/trust_mean",
    }
    metric_summary = {name: metric_stats(valid, field) for name, field in metric_fields.items()}
    metadata = {
        "schema": "openclaw.dive_po_exploration_analysis.v2",
        "generated_at": generated,
        "snapshot": {
            "metrics_mtime": metric_mtime,
            "train_log_mtime": train_mtime,
            "last_completed_rollout_id": int(all_records[-1]["rollout_id"]),
            "attempts": len(all_records),
            "valid_rollout_steps": len(valid),
            "invalid_or_untrainable": invalid,
        },
        "run_dir": str(run_dir.resolve()),
        "baseline_run_dir": str(baseline_dir.resolve()),
        "valid_filter": "dataset=seta, phase=train, trainable_count > 0, finite raw_reward",
        "common_valid_budget": common,
        "common_comparison": {
            "experiment_raw_reward": exp_raw,
            "baseline_raw_reward": base_raw,
            "raw_delta": raw_delta,
            "raw_relative": raw_relative,
            "experiment_truncated_fraction": exp_trunc,
            "baseline_truncated_fraction": base_trunc,
            "experiment_total_reward": mean(values(exp, "total_reward")),
            "baseline_total_reward": mean(values(base, "total_reward")),
            "experiment_response_length": mean(values(exp, "response_length")),
            "baseline_response_length": mean(values(base, "response_length")),
            "experiment_last50_raw": mean(values(exp[-50:], "raw_reward")),
            "baseline_last50_raw": mean(values(base[-50:], "raw_reward")),
            "experiment_operational_pass": operational_pass(exp),
            "baseline_operational_pass": operational_pass(base),
        },
        "metrics": metric_summary,
        "correlations_with_raw_reward": correlations,
        "agent57_arm_events": {
            "n_events": sum(int(row["n"]) for row in arm_aggregates),
            "by_arm": arm_aggregates,
            "time_bins": bins,
            "lifelong_counts": lifelong,
        },
        "figures": {key: str(path.resolve()) for key, path in figures.items()},
    }
    (output_dir / "exploration_analysis.json").write_text(
        json.dumps(clean_json(metadata), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    exploration_rows = [
        ("局内 NGU episodic", metric_summary["ngu_episodic"]),
        ("局内 empty-bucket rate", metric_summary["episodic_empty_bucket_rate"]),
        ("局内 exact repeats", metric_summary["episodic_exact_repeat_count"]),
        ("局间 lifelong raw novelty", metric_summary["lifelong_raw"]),
        ("局间 NGU life modifier", metric_summary["ngu_life_mod"]),
        ("融合 NGU bonus", metric_summary["ngu_bonus"]),
        ("abs exploration reward", metric_summary["exploration_abs"]),
        ("UCB top-arm share", metric_summary["top_arm_ratio"]),
        ("outcome gate suppressed share", metric_summary["top_suppressed_ratio"]),
        ("outcome gate trust mean", metric_summary["trust_mean"]),
    ]
    exploration_table = "\n".join(
        f"| {label} | {row['n']} | {fmt_metric(row['mean'])} | {fmt_metric(row['first10'])} | {fmt_metric(row['last10'])} | {fmt_metric(row['std'])} |"
        for label, row in exploration_rows
    )
    arm_table = "\n".join(
        f"| {row['arm']} | {row['n']} | {fmt(row['normalized_base'])} | {fmt(row['success_rate'])} | {fmt(row['truncated_rate'])} | {fmt(row['parse_rate'])} | {row['bonus_mean']:.7f} |"
        for row in arm_aggregates
    )
    correlation_table = "\n".join(
        f"| {name} | {fmt(value, 3)} |" for name, value in sorted(correlations.items(), key=lambda item: item[1], reverse=True)
    )

    report = f"""# DiVE-PO v0716 centered-gate 最新训练分析

生成时间：`{generated}`。

- 结构化指标截止 rollout：`{all_records[-1]['rollout_id']}`；已记录 `{len(all_records)}` 次尝试，正式有效 rollout-step `{len(valid)}`，空/不可训练 `{invalid}`（`{invalid/len(all_records):.1%}`）。
- `metrics.jsonl` 时间：`{metric_mtime}`；`train.log` 时间：`{train_mtime}`。
- `train.log` 晚于结构化指标文件，说明下一轮仍可能在生成；所有正式均值与公平曲线只统计已写入 `metrics.jsonl` 的完成 rollout。
- 有效点过滤：`dataset=seta`、`phase=train`、`trainable_count > 0`、`raw_reward` finite。原始 `rollout_id` 的空运行增长不计为训练步。

## 最新结论

共同前 `{common}` 个有效 rollout-step 上，DiVE-PO `raw_reward={exp_raw:.4f}`，baseline `raw_reward={base_raw:.4f}`，提升 `{raw_delta:+.4f}`（`{raw_relative:+.1%}`）。但共同窗口后 50 点为 `{mean(values(exp[-50:], 'raw_reward')):.4f} vs {mean(values(base[-50:], 'raw_reward')):.4f}`，尾部已经基本持平，因此更准确的判断是：**累计均值仍领先，最新局部性能未继续扩大优势**。

| 公平比较指标（前 {common} 个有效点） | DiVE-PO | baseline | 差值 |
|---|---:|---:|---:|
| raw_reward | {exp_raw:.4f} | {base_raw:.4f} | {raw_delta:+.4f} |
| total_reward | {mean(values(exp, 'total_reward')):.4f} | {mean(values(base, 'total_reward')):.4f} | {mean(values(exp, 'total_reward'))-mean(values(base, 'total_reward')):+.4f} |
| truncated_fraction | {exp_trunc:.4f} | {base_trunc:.4f} | {exp_trunc-base_trunc:+.4f} |
| response_length | {mean(values(exp, 'response_length')):.1f} | {mean(values(base, 'response_length')):.1f} | {mean(values(exp, 'response_length'))-mean(values(base, 'response_length')):+.1f} |
| operational pass | {operational_pass(exp):.4f} | {operational_pass(base):.4f} | {operational_pass(exp)-operational_pass(base):+.4f} |
| common-window 后50 raw_reward | {mean(values(exp[-50:], 'raw_reward')):.4f} | {mean(values(base[-50:], 'raw_reward')):.4f} | {mean(values(exp[-50:], 'raw_reward'))-mean(values(base[-50:], 'raw_reward')):+.4f} |

解释：累计 raw/pass 与 operational pass 的优势仍然明确；同时 truncation 比 baseline 高 `{exp_trunc-base_trunc:+.4f}`、response length 高 `{mean(values(exp, 'response_length'))-mean(values(base, 'response_length')):+.1f}`，且后 50 点持平，说明后续优化应优先关注尾部探索转化率和截断控制，而不是继续放大 intrinsic 权重。

## DiVE-PO 探索链路

| 指标 | n | mean | first10 | last10 | std |
|---|---:|---:|---:|---:|---:|
{exploration_table}

![exploration metrics](figs/{figures['exploration'].name})

## 探索信号与性能相关性

相关性是 rollout 聚合层面的 Pearson r，只用于定位关联，不作为因果证据。

| 指标 | r(raw_reward) |
|---|---:|
{correlation_table}

![exploration relationship](figs/{figures['relationship'].name})

## UCB arm 与 lifelong 存储

SQLite 当前包含 `{sum(int(row['n']) for row in arm_aggregates):,}` 条 arm events、`{int(lifelong.get('key_count', 0)):,}` 个 lifelong keys；count 均值 `{fmt(number(lifelong.get('mean_count')))}`。

| arm | n | normalized base | success | truncated | parse error | mean bonus |
|---:|---:|---:|---:|---:|---:|---:|
{arm_table}

![arm by arm](figs/{figures['arm_by_arm'].name})

![arm over time](figs/{figures['arm_time'].name})

## 曲线索引

- 标准训练曲线：[`overview.png`](figs/overview.png)、[`reward_curve.png`](figs/reward_curve.png)、[`loss_curve.png`](figs/loss_curve.png)、[`grad_norm.png`](figs/grad_norm.png)、[`kl_entropy.png`](figs/kl_entropy.png)。
- 公平核心对比：[`baseline_core_comparison.png`](figs/{figures['core'].name})。
- 最终过滤 reward 对比：[`baseline_vs_dive_po_filtered_reward_vs_rollout_step.png`](figs/baseline_vs_dive_po_filtered_reward_vs_rollout_step.png)。
- 空运行与原始 ID 诊断：[`baseline_vs_dive_po_actual_rollout_id.png`](figs/baseline_vs_dive_po_actual_rollout_id.png)。
- 详细过滤统计：[`baseline_vs_dive_po_rollout_step_report.md`](baseline_vs_dive_po_rollout_step_report.md)。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    baseline_dir = args.baseline_run.resolve()
    output_dir = (args.out_dir or run_dir / "metrics" / "analysis").resolve()
    figures_dir = output_dir / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_records = load_metrics(run_dir / "logs" / "metrics.jsonl")
    baseline_all = load_metrics(baseline_dir / "logs" / "metrics.jsonl")
    valid = valid_records(all_records)
    baseline_valid = valid_records(baseline_all)
    if not valid or not baseline_valid:
        raise RuntimeError("no valid SETA train records found")

    figures = {
        "core": figures_dir / "baseline_core_comparison.png",
        "exploration": figures_dir / "exploration_metrics_trends.png",
        "relationship": figures_dir / "exploration_performance_relationship.png",
        "arm_by_arm": figures_dir / "agent57_arm_events_by_arm.png",
        "arm_time": figures_dir / "agent57_arm_events_over_time.png",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_core(valid, baseline_valid, figures["core"])
    plot_exploration(valid, figures["exploration"])
    correlations = plot_relationship(valid, figures["relationship"])
    events, _, lifelong = load_arm_events(run_dir / "agent57_lite.sqlite3")
    aggregates = aggregate_arms(events)
    bins = time_bins(events)
    if aggregates:
        plot_arm_by_arm(aggregates, figures["arm_by_arm"])
        plot_arm_time(bins, figures["arm_time"])
    write_outputs(
        run_dir,
        baseline_dir,
        output_dir,
        all_records,
        valid,
        baseline_valid,
        correlations,
        aggregates,
        bins,
        lifelong,
        figures,
    )
    print(f"snapshot: rollout_id={all_records[-1]['rollout_id']}, attempts={len(all_records)}, valid={len(valid)}")
    print(f"arm events: {len(events)}, lifelong keys: {lifelong.get('key_count', 0)}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
