#!/usr/bin/env python3
"""Compare two terminal-rl runs after removing empty/untrainable rollouts.

This script intentionally keeps two views of the same logs:

1. the original rollout_id history, where empty Docker/server-failure attempts
   remain visible; and
2. the effective rollout-step history, where only trainable SETA train records
   with a finite raw_reward are retained and re-indexed contiguously.

The second view is the correct x-axis for algorithm-performance comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openclaw-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"baseline": "#4C78A8", "experiment": "#B279A2"}
ROLLING_WINDOW = 10


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def as_float(value: Any, default: float = float("nan")) -> float:
    return float(value) if finite_number(value) else default


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def rolling(values: Iterable[float], window: int = ROLLING_WINDOW) -> np.ndarray:
    """NaN-aware trailing rolling mean with min_periods=1."""
    array = np.asarray(list(values), dtype=float)
    result = np.full(array.shape, np.nan, dtype=float)
    for index in range(array.size):
        chunk = array[max(0, index - window + 1) : index + 1]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            result[index] = float(np.mean(chunk))
    return result


def load_records(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Load and de-duplicate SETA train records, keeping the last occurrence."""
    records_by_id: dict[int, dict[str, Any]] = {}
    malformed = 0
    ignored = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if record.get("dataset") != "seta" or record.get("phase") != "train":
                ignored += 1
                continue
            rollout_id = record.get("rollout_id")
            if not isinstance(rollout_id, int):
                ignored += 1
                continue
            records_by_id[rollout_id] = record
    return [records_by_id[key] for key in sorted(records_by_id)], malformed, ignored


def is_valid(record: dict[str, Any]) -> bool:
    trainable = record.get("trainable_count")
    return finite_number(trainable) and float(trainable) > 0 and finite_number(record.get("raw_reward"))


def read_configured_rollouts(run_dir: Path) -> int | None:
    config = run_dir / "config" / "run_config.json"
    if not config.exists():
        return None
    try:
        value = json.loads(config.read_text(encoding="utf-8")).get("num_rollout")
    except (json.JSONDecodeError, OSError):
        return None
    return int(value) if finite_number(value) else None


@dataclass
class RunData:
    key: str
    label: str
    run_dir: Path
    metrics_path: Path
    configured_rollouts: int | None
    records: list[dict[str, Any]]
    valid: list[dict[str, Any]]
    malformed_lines: int
    ignored_lines: int

    @classmethod
    def load(cls, key: str, label: str, run_dir: Path) -> "RunData":
        metrics_path = run_dir / "logs" / "metrics.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"metrics file not found: {metrics_path}")
        records, malformed, ignored = load_records(metrics_path)
        if not records:
            raise ValueError(f"no dataset=seta, phase=train rollout records in {metrics_path}")
        valid = [record for record in records if is_valid(record)]
        if not valid:
            raise ValueError(f"no valid trainable rollout records in {metrics_path}")
        return cls(
            key=key,
            label=label,
            run_dir=run_dir.resolve(),
            metrics_path=metrics_path.resolve(),
            configured_rollouts=read_configured_rollouts(run_dir),
            records=records,
            valid=valid,
            malformed_lines=malformed,
            ignored_lines=ignored,
        )

    def values(self, field: str, valid_only: bool = True) -> np.ndarray:
        source = self.valid if valid_only else self.records
        return np.asarray([as_float(record.get(field)) for record in source], dtype=float)

    def ids(self, valid_only: bool = True) -> np.ndarray:
        source = self.valid if valid_only else self.records
        return np.asarray([int(record["rollout_id"]) for record in source], dtype=int)

    def valid_efficiency(self) -> float:
        return safe_ratio(len(self.valid), len(self.records))

    def trailing_invalid(self) -> int:
        count = 0
        for record in reversed(self.records):
            if is_valid(record):
                break
            count += 1
        return count

    def operational_pass(self, records: list[dict[str, Any]] | None = None) -> float:
        source = records if records is not None else self.records
        weighted_reward = 0.0
        sampled = 0.0
        for record in source:
            sample_count = as_float(record.get("sample_count"), 0.0)
            trainable_count = as_float(record.get("trainable_count"), 0.0)
            raw_reward = as_float(record.get("raw_reward"), 0.0)
            sampled += max(sample_count, 0.0)
            weighted_reward += raw_reward * max(trainable_count, 0.0)
        return safe_ratio(weighted_reward, sampled)

    def trainable_sample_ratio(self, records: list[dict[str, Any]] | None = None) -> float:
        source = records if records is not None else self.records
        trainable = sum(max(as_float(record.get("trainable_count"), 0.0), 0.0) for record in source)
        sampled = sum(max(as_float(record.get("sample_count"), 0.0), 0.0) for record in source)
        return safe_ratio(trainable, sampled)


def plot_points_and_rolling(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str,
    *,
    scatter: bool = True,
) -> None:
    if scatter:
        axis.scatter(x, y, s=8, alpha=0.13, color=color, linewidths=0)
    axis.plot(x, rolling(y), color=color, linewidth=2.1, label=label)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_actual_history(baseline: RunData, experiment: RunData, path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for run in (baseline, experiment):
        color = COLORS[run.key]
        ids = run.ids(valid_only=False)
        valid_mask = np.asarray([is_valid(record) for record in run.records], dtype=bool)

        # Invalid attempts are zero-filled only in this diagnostic view so long
        # server-failure spans remain visible. They are removed from formal curves.
        raw = np.asarray(
            [as_float(record.get("raw_reward"), 0.0) if valid else 0.0 for record, valid in zip(run.records, valid_mask)],
            dtype=float,
        )
        trunc = np.asarray(
            [as_float(record.get("truncated_fraction")) if valid else float("nan") for record, valid in zip(run.records, valid_mask)],
            dtype=float,
        )
        ratio = np.asarray(
            [safe_ratio(max(as_float(record.get("trainable_count"), 0.0), 0.0), max(as_float(record.get("sample_count"), 0.0), 0.0)) for record in run.records],
            dtype=float,
        )

        axes[0].scatter(ids[valid_mask], raw[valid_mask], s=7, alpha=0.13, color=color, linewidths=0)
        axes[0].plot(ids, rolling(raw), color=color, linewidth=2.0, label=f"{run.label} rolling10")
        plot_points_and_rolling(axes[1], ids, trunc, color, f"{run.label} rolling10")
        plot_points_and_rolling(axes[2], ids, ratio, color, f"{run.label} rolling10", scatter=False)
        axes[2].axvline(run.ids()[-1], color=color, linestyle="--", alpha=0.7, linewidth=1.1)

    figure.suptitle(
        f"SETA+DAPO baseline vs DiVE-PO — full actual rollout history\n"
        f"Baseline: {len(baseline.records)} attempts (id {baseline.ids(False)[0]}–{baseline.ids(False)[-1]}); "
        f"DiVE-PO: {len(experiment.records)} attempts (id {experiment.ids(False)[0]}–{experiment.ids(False)[-1]}), "
        f"last trainable id {experiment.ids()[-1]}",
        fontsize=16,
    )
    titles = (
        "Pass rate / raw_reward (invalid attempts zero-filled in this diagnostic only)",
        "Truncation rate (trainable rollouts)",
        "Trainable sample ratio (zero exposes failed/empty rollout attempts)",
    )
    ylabels = ("raw_reward", "truncated_fraction", "trainable / sampled")
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("actual rollout_id (zero-based; unfiltered)")
    save_figure(figure, path)


def plot_all_valid(baseline: RunData, experiment: RunData, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 7.6), sharex=True)
    for run in (baseline, experiment):
        x = np.arange(1, len(run.valid) + 1)
        color = COLORS[run.key]
        plot_points_and_rolling(
            axes[0], x, run.values("raw_reward"), color, f"{run.label} rolling10 (N={len(run.valid)})"
        )
        plot_points_and_rolling(
            axes[1], x, run.values("truncated_fraction"), color, f"{run.label} rolling10 (N={len(run.valid)})"
        )
    figure.suptitle(
        "SETA+DAPO baseline vs DiVE-PO — all effective trainable rollout steps\n"
        f"Baseline N={len(baseline.valid)}; DiVE-PO N={len(experiment.valid)}",
        fontsize=16,
    )
    for axis, title, ylabel in zip(
        axes,
        ("Pass rate / raw_reward", "Truncated fraction"),
        ("raw_reward", "truncated_fraction"),
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("effective valid rollout step (empty/untrainable attempts removed; one-based)")
    save_figure(figure, path)


def plot_common_budget(baseline: RunData, experiment: RunData, common: int, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    fields = (
        ("raw_reward", "raw_reward"),
        ("total_reward", "total_reward"),
        ("truncated_fraction", "truncated_fraction"),
        ("response_length", "response_length"),
    )
    x = np.arange(1, common + 1)
    for axis, (field, ylabel) in zip(axes.flat, fields):
        for run in (baseline, experiment):
            y = run.values(field)[:common]
            plot_points_and_rolling(axis, x, y, COLORS[run.key], f"{run.label} rolling10")
        axis.set_title(field)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("effective valid rollout step")
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"SETA+DAPO baseline vs DiVE-PO — common first {common} valid rollout steps",
        fontsize=16,
    )
    save_figure(figure, path)


def plot_filtered_reward(baseline: RunData, experiment: RunData, common: int, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(15.75, 10.125))
    for run in (baseline, experiment):
        x = np.arange(1, len(run.valid) + 1)
        y = run.values("raw_reward")
        color = COLORS[run.key]
        plot_points_and_rolling(
            axes[0], x, y, color, f"{run.label} rolling10 (N={len(run.valid)})"
        )
        axes[0].axvline(len(run.valid), color=color, linestyle="--", alpha=0.65, linewidth=1.0)

        common_y = y[:common]
        plot_points_and_rolling(
            axes[1],
            np.arange(1, common + 1),
            common_y,
            color,
            f"{run.label} rolling10; mean={mean(common_y):.4f}",
        )

    figure.suptitle(
        "Final filtered reward vs rollout-step: SETA+DAPO baseline vs DiVE-PO\n"
        "Filter: dataset=seta, phase=train, trainable_count > 0 and finite raw_reward; original rollout_id gaps removed",
        fontsize=16,
    )
    axes[0].set_title("All filtered trainable rollout steps")
    axes[1].set_title(f"Fair comparison: common first {common} filtered rollout steps")
    for axis in axes:
        axis.set_ylabel("raw_reward / pass rate")
        axis.set_xlabel("filtered rollout-step (empty attempts removed; re-indexed from 1)")
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=9)
    save_figure(figure, path)


def metrics_mtime(path: Path) -> str:
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=ZoneInfo("Asia/Hong_Kong"))
    return timestamp.strftime("%Y-%m-%d %H:%M:%S HKT")


def run_summary(run: RunData, common: int) -> dict[str, Any]:
    common_valid = run.valid[:common]
    common_last50 = common_valid[-min(50, len(common_valid)) :]
    all_last50 = run.valid[-min(50, len(run.valid)) :]
    return {
        "label": run.label,
        "run_dir": str(run.run_dir),
        "metrics_path": str(run.metrics_path),
        "configured_num_rollout": run.configured_rollouts,
        "actual_rollout_id_first": int(run.ids(False)[0]),
        "actual_rollout_id_last": int(run.ids(False)[-1]),
        "actual_rollout_attempts": len(run.records),
        "valid_rollout_count": len(run.valid),
        "last_valid_rollout_id": int(run.ids()[-1]),
        "invalid_or_untrainable_attempts": len(run.records) - len(run.valid),
        "trailing_untrainable_attempts": run.trailing_invalid(),
        "valid_attempt_efficiency": run.valid_efficiency(),
        "valid_raw_reward_mean": mean(run.values("raw_reward")),
        "valid_truncated_fraction_mean": mean(run.values("truncated_fraction")),
        "valid_total_reward_mean": mean(run.values("total_reward")),
        "valid_response_length_mean": mean(run.values("response_length")),
        "valid_raw_reward_last50": mean(as_float(record.get("raw_reward")) for record in all_last50),
        "valid_truncated_fraction_last50": mean(
            as_float(record.get("truncated_fraction")) for record in all_last50
        ),
        "first_common_valid_raw_reward_mean": mean(
            as_float(record.get("raw_reward")) for record in common_valid
        ),
        "first_common_valid_truncated_fraction_mean": mean(
            as_float(record.get("truncated_fraction")) for record in common_valid
        ),
        "first_common_valid_total_reward_mean": mean(
            as_float(record.get("total_reward")) for record in common_valid
        ),
        "first_common_valid_response_length_mean": mean(
            as_float(record.get("response_length")) for record in common_valid
        ),
        "first_common_last50_raw_reward_mean": mean(
            as_float(record.get("raw_reward")) for record in common_last50
        ),
        "first_common_operational_pass_rate": run.operational_pass(common_valid),
        "first_common_trainable_sample_ratio": run.trainable_sample_ratio(common_valid),
        "operational_pass_rate": run.operational_pass(),
        "trainable_sample_ratio": run.trainable_sample_ratio(),
        "rolling10_final": float(rolling(run.values("raw_reward"))[-1]),
        "malformed_json_lines_skipped": run.malformed_lines,
        "non_seta_train_lines_ignored": run.ignored_lines,
        "metrics_mtime": metrics_mtime(run.metrics_path),
    }


def fmt(value: float, digits: int = 4) -> str:
    return "NA" if not finite_number(value) else f"{value:.{digits}f}"


def relative_change(new: float, old: float) -> float:
    return safe_ratio(new - old, old)


def write_outputs(
    baseline: RunData,
    experiment: RunData,
    output_dir: Path,
    figures: dict[str, Path],
    generated_at: str,
    common: int,
) -> None:
    summaries = {
        "baseline": run_summary(baseline, common),
        "dive_po": run_summary(experiment, common),
    }
    baseline_summary = summaries["baseline"]
    experiment_summary = summaries["dive_po"]

    rollout_meta = {
        "schema": "openclaw.baseline_vs_dive_po_rollout_step.v2",
        "generated_at": generated_at,
        "snapshot_rule": "only completed records already present in logs/metrics.jsonl",
        "valid_filter": "dataset=seta, phase=train, trainable_count > 0 and raw_reward is finite",
        "common_valid_budget": common,
        "runs": summaries,
        "figures": [str(figures[key].resolve()) for key in ("actual", "valid", "common")],
    }
    (output_dir / "baseline_vs_dive_po_rollout_step_meta.json").write_text(
        json.dumps(rollout_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    filtered_meta = {
        "schema": "openclaw.filtered_reward_rollout_comparison.v2",
        "generated_at": generated_at,
        "snapshot_rule": "only completed records already present in logs/metrics.jsonl",
        "filter": "dataset=seta, phase=train, trainable_count > 0, raw_reward finite",
        "x_axis": "one-based contiguous index after filtering; original rollout_id is not used as x",
        "common_valid_steps": common,
        "figure": str(figures["filtered"].resolve()),
        "runs": {
            key: {
                "label": summary["label"],
                "metrics_path": summary["metrics_path"],
                "logged_attempts": summary["actual_rollout_attempts"],
                "empty_or_untrainable_removed": summary["invalid_or_untrainable_attempts"],
                "filtered_rollout_steps": summary["valid_rollout_count"],
                "first_source_rollout_id": summary["actual_rollout_id_first"],
                "last_source_rollout_id": summary["last_valid_rollout_id"],
                "reward_mean_all_filtered": summary["valid_raw_reward_mean"],
                "reward_mean_first_common": summary["first_common_valid_raw_reward_mean"],
                "reward_last50_mean": summary["valid_raw_reward_last50"],
                "reward_first_common_last50_mean": summary["first_common_last50_raw_reward_mean"],
                "rolling10_final": summary["rolling10_final"],
                "metrics_mtime": summary["metrics_mtime"],
            }
            for key, summary in summaries.items()
        },
    }
    (output_dir / "baseline_vs_dive_po_filtered_reward_meta.json").write_text(
        json.dumps(filtered_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    baseline_raw = baseline_summary["first_common_valid_raw_reward_mean"]
    experiment_raw = experiment_summary["first_common_valid_raw_reward_mean"]
    raw_delta = experiment_raw - baseline_raw
    raw_relative = relative_change(experiment_raw, baseline_raw)
    baseline_trunc = baseline_summary["first_common_valid_truncated_fraction_mean"]
    experiment_trunc = experiment_summary["first_common_valid_truncated_fraction_mean"]
    trunc_delta = experiment_trunc - baseline_trunc

    rollout_report = f"""# SETA+DAPO baseline vs DiVE-PO：过滤后的实际 rollout-step 核对

生成时间：`{generated_at}`。

正式有效点口径：`dataset=seta`、`phase=train`、`trainable_count > 0` 且 `raw_reward` 为有限值。Docker/server 中断造成的空批次只保留在原始 ID 诊断图中，不计为正式 rollout-step。

| 实验 | 配置 num_rollout | 已记录 rollout_id | 已记录尝试数 | 有效 rollout-step | 空/不可训练 | 有效率 | 最后有效 ID | 尾部连续无效 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| {baseline.label} | {baseline.configured_rollouts or 'NA'} | {baseline.ids(False)[0]}–{baseline.ids(False)[-1]} | {len(baseline.records)} | {len(baseline.valid)} | {len(baseline.records)-len(baseline.valid)} | {baseline.valid_efficiency():.1%} | {baseline.ids()[-1]} | {baseline.trailing_invalid()} |
| {experiment.label} | {experiment.configured_rollouts or 'NA'} | {experiment.ids(False)[0]}–{experiment.ids(False)[-1]} | {len(experiment.records)} | {len(experiment.valid)} | {len(experiment.records)-len(experiment.valid)} | {experiment.valid_efficiency():.1%} | {experiment.ids()[-1]} | {experiment.trailing_invalid()} |

## 公平预算：共同前 {common} 个有效 rollout-step

| 指标 | {baseline.label} | {experiment.label} | DiVE-PO - baseline |
|---|---:|---:|---:|
| raw_reward | {baseline_raw:.4f} | {experiment_raw:.4f} | {raw_delta:+.4f} ({raw_relative:+.1%}) |
| truncated_fraction | {baseline_trunc:.4f} | {experiment_trunc:.4f} | {trunc_delta:+.4f} |
| total_reward | {fmt(baseline_summary['first_common_valid_total_reward_mean'])} | {fmt(experiment_summary['first_common_valid_total_reward_mean'])} | {experiment_summary['first_common_valid_total_reward_mean']-baseline_summary['first_common_valid_total_reward_mean']:+.4f} |
| response_length | {fmt(baseline_summary['first_common_valid_response_length_mean'], 1)} | {fmt(experiment_summary['first_common_valid_response_length_mean'], 1)} | {experiment_summary['first_common_valid_response_length_mean']-baseline_summary['first_common_valid_response_length_mean']:+.1f} |
| common-window 后50点 raw_reward | {baseline_summary['first_common_last50_raw_reward_mean']:.4f} | {experiment_summary['first_common_last50_raw_reward_mean']:.4f} | {experiment_summary['first_common_last50_raw_reward_mean']-baseline_summary['first_common_last50_raw_reward_mean']:+.4f} |
| operational pass | {baseline_summary['first_common_operational_pass_rate']:.4f} | {experiment_summary['first_common_operational_pass_rate']:.4f} | {experiment_summary['first_common_operational_pass_rate']-baseline_summary['first_common_operational_pass_rate']:+.4f} |

结论：共同前 {common} 个过滤后有效步上，DiVE-PO raw_reward 提升 `{raw_delta:.4f}`（`{raw_relative:+.1%}`）；截断率变化 `{trunc_delta:+.4f}`。本次 v0716 快照仅有 {len(experiment.valid)} 个有效点，说明当前结论是同预算早中期比较，不应据此声称已经达到长期收敛。

## 图表

- `figs/{figures['actual'].name}`：完整原始 rollout_id；空/不可训练尝试不会被误当成有效训练步。
- `figs/{figures['valid'].name}`：双方各自全部有效 rollout-step，横轴连续重编号。
- `figs/{figures['common'].name}`：共同前 {common} 个有效点上的 raw、total reward、truncation、response length。
- `figs/{figures['filtered'].name}`：最终正式 reward vs filtered rollout-step；下图是严格同预算对齐。

数据快照：baseline `metrics.jsonl` mtime `{baseline_summary['metrics_mtime']}`；DiVE-PO `metrics.jsonl` mtime `{experiment_summary['metrics_mtime']}`。只统计已经写入结构化指标文件的完成 rollout。
"""
    (output_dir / "baseline_vs_dive_po_rollout_step_report.md").write_text(
        rollout_report, encoding="utf-8"
    )

    filtered_report = f"""# 最终过滤版 reward vs rollout-step 对比

生成时间：`{generated_at}`。

过滤器：`dataset=seta`、`phase=train`、`trainable_count > 0`、`raw_reward` finite。过滤后横轴从 1 连续编号，原始 `rollout_id` 的空运行增长不作为训练进度。

| run | logged attempts | removed empty/untrainable | filtered rollout steps | last original valid rollout_id | reward mean, first {common} | common-window last50 |
|---|---:|---:|---:|---:|---:|---:|
| {baseline.label} | {len(baseline.records)} | {len(baseline.records)-len(baseline.valid)} | {len(baseline.valid)} | {baseline.ids()[-1]} | {baseline_raw:.4f} | {baseline_summary['first_common_last50_raw_reward_mean']:.4f} |
| {experiment.label} | {len(experiment.records)} | {len(experiment.records)-len(experiment.valid)} | {len(experiment.valid)} | {experiment.ids()[-1]} | {experiment_raw:.4f} | {experiment_summary['first_common_last50_raw_reward_mean']:.4f} |

共同前 {common} 个过滤后 rollout-step 上，DiVE-PO raw_reward 相比 baseline 提升 `{raw_delta:.4f}`（`{raw_relative:+.1%}`）。

图：`figs/{figures['filtered'].name}`。
"""
    (output_dir / "baseline_vs_dive_po_filtered_reward_report.md").write_text(
        filtered_report, encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--experiment-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="SETA+DAPO baseline")
    parser.add_argument("--experiment-label", default="DiVE-PO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline = RunData.load("baseline", args.baseline_label, args.baseline_run)
    experiment = RunData.load("experiment", args.experiment_label, args.experiment_run)
    common = min(len(baseline.valid), len(experiment.valid))
    generated_at = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M:%S HKT")

    figures = {
        "actual": figures_dir / "baseline_vs_dive_po_actual_rollout_id.png",
        "valid": figures_dir / "baseline_vs_dive_po_effective_valid_rollout_step.png",
        "common": figures_dir / "baseline_vs_dive_po_common_valid_budget.png",
        "filtered": figures_dir / "baseline_vs_dive_po_filtered_reward_vs_rollout_step.png",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_actual_history(baseline, experiment, figures["actual"])
    plot_all_valid(baseline, experiment, figures["valid"])
    plot_common_budget(baseline, experiment, common, figures["common"])
    plot_filtered_reward(baseline, experiment, common, figures["filtered"])
    write_outputs(baseline, experiment, output_dir, figures, generated_at, common)

    print(f"baseline: attempts={len(baseline.records)}, valid={len(baseline.valid)}")
    print(f"experiment: attempts={len(experiment.records)}, valid={len(experiment.valid)}")
    print(f"common valid budget: {common}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
