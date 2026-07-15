#!/usr/bin/env python3
"""Analyze the v0626 SETA exploration run against the SETA-DAPO baseline.

This script is intentionally a thin orchestrator around the reusable analysis
helpers in ``analyze_exploration_report.py`` and the standard terminal-rl
metrics scripts. It writes all final artifacts under
``<run_dir>/metrics/analysis`` and appends a latest-run section to the rolling
Chinese exploration report.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs/terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0626_riskcal_turninv_truncpenalty_dualadv_think_2026-06-26_183129"
DEFAULT_BASELINE_LOG = ROOT / "runs/terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_think_mt10_2026-06-11_092726/logs/train.log"
DEFAULT_REVIEW_DOC = ROOT / "terminal-rl/docs/exploration_review_optimization_2026-06-26_175803.md"
DEFAULT_HISTORY_DOC = ROOT / "terminal-rl/docs/exploration_exp_report_zh.md"
PREV_ANALYSIS_DIR = ROOT / "runs/terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0623_envtolerant_fastwarm_dualadv_think_2026-06-24_164820/metrics/analysis"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP = load_module(SCRIPTS_DIR / "analyze_exploration_report.py", "analyze_exploration_report_v0626_base")
PLOT = load_module(SCRIPTS_DIR / "plot_training_metrics.py", "plot_training_metrics_v0626_base")


def num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 4) -> str:
    value = num(value)
    if value is None:
        return "NA"
    if value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.{max(2, digits)}e}"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    value = num(value)
    if value is None:
        return "NA"
    return f"{value * 100:.{digits}f}%"


def fmt_int(value: Any) -> str:
    value = num(value)
    if value is None:
        return "NA"
    return f"{int(round(value)):,}"


def m(stats_obj: dict[str, Any], field: str, key: str = "mean") -> Any:
    item = stats_obj.get(field) if isinstance(stats_obj, dict) else None
    return item.get(key) if isinstance(item, dict) else None


def compare(exp_value: Any, base_value: Any, *, lower_is_better: bool = False) -> dict[str, Any]:
    exp_v = num(exp_value)
    base_v = num(base_value)
    diff = None if exp_v is None or base_v is None else exp_v - base_v
    pct = None if diff is None or base_v is None or abs(base_v) < 1e-12 else diff / abs(base_v)
    if diff is None:
        verdict = "缺失，已跳过"
    elif lower_is_better:
        verdict = "提升" if diff < 0 else ("退化" if diff > 0 else "持平")
    else:
        verdict = "提升" if diff > 0 else ("退化" if diff < 0 else "持平")
    return {"exp": exp_v, "baseline": base_v, "diff": diff, "pct_delta": pct, "verdict": verdict}


def values(records: list[dict[str, Any]], *keys: str) -> list[float]:
    out: list[float] = []
    for record in records:
        value = EXP.metric(record, *keys)
        if value is not None:
            out.append(float(value))
    return out


def stat_values(vals: list[float]) -> dict[str, Any]:
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


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def run_cmd(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("[+] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_standard_scripts(run_dir: Path, out_dir: Path) -> None:
    py = sys.executable
    run_cmd([py, str(SCRIPTS_DIR / "plot_training_metrics.py"), "--run-dir", str(run_dir), "--out-dir", str(out_dir)])
    run_cmd([py, str(SCRIPTS_DIR / "analyze_hang_diagnostics.py"), "--run-dir", str(run_dir), "--out-dir", str(out_dir)])
    run_cmd([py, str(SCRIPTS_DIR / "analyze_trajectories.py"), "--run-dir", str(run_dir), "--out-dir", str(out_dir), "--max-iter-hint", "10"])


def load_rollout_metrics(run_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    log_path = run_dir / "logs/train.log"
    if not log_path.is_file():
        return {}, {"bad_lines": 0}
    parsed = PLOT._parse_log(log_path)
    rollout_metrics = parsed.get("rollout_metrics") or {}
    return {int(k): v for k, v in rollout_metrics.items()}, {"bad_lines": 0}


def augment_rollout_metrics(records: list[dict[str, Any]], rollout_metrics: dict[int, dict[str, Any]]) -> None:
    for record in records:
        item = rollout_metrics.get(EXP.rid(record)) or {}
        for src, dst in (
            ("rollout/returns", "rollout_returns"),
            ("rollout/rewards", "rollout_rewards"),
            ("rollout/advantages", "rollout_advantages"),
            ("rollout/raw_reward", "rollout_raw_reward_train_space"),
        ):
            value = num(item.get(src))
            if value is not None:
                record[dst] = value


def window_extra_stats(records: list[dict[str, Any]], *, max_rollout: int | None = None, first_n_valid: int | None = None, last_n_valid: int | None = None) -> dict[str, Any]:
    selected_all = list(records)
    if max_rollout is not None:
        selected_all = [record for record in selected_all if EXP.rid(record) <= max_rollout]
    selected = [record for record in selected_all if EXP.is_valid(record)]
    if first_n_valid is not None:
        selected = selected[:first_n_valid]
    if last_n_valid is not None:
        selected = selected[-last_n_valid:]
    return {
        "rollout_returns": stat_values(values(selected, "rollout_returns")),
        "rollout_rewards": stat_values(values(selected, "rollout_rewards")),
        "rollout_advantages": stat_values(values(selected, "rollout_advantages")),
    }


def first_matching_line(path: Path, needle: str) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if needle in line:
                return line_no
    return None


def jsonl_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def list_files(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return [str(item.relative_to(path)) for item in sorted(path.rglob("*")) if item.is_file() and "__pycache__" not in item.parts]


def script_inventory() -> list[dict[str, Any]]:
    return [
        {
            "file": "plot_training_metrics.py",
            "entry": "python terminal-rl/scripts/plot_training_metrics.py --run-dir <run>",
            "data_source": "logs/train.log + logs/metrics.jsonl",
            "outputs": "summary_stats.json; figs/overview.png, reward_curve.png, response_length.png, loss_curve.png, grad_norm.png, kl_entropy.png",
            "reuse": "复用：核心训练曲线与标准 summary",
        },
        {
            "file": "analyze_exploration_report.py",
            "entry": "python terminal-rl/scripts/analyze_exploration_report.py --run-dir <run> --baseline-log <baseline/logs/train.log>",
            "data_source": "logs/metrics.jsonl, train.log, agent57_lite.sqlite3, trajectories/*/traj.json",
            "outputs": "exploration_analysis.json; report.md; case_study_details.md; exploration/UCB PNG",
            "reuse": "复用并扩展：指标定义、图表风格、SQLite arm_events 与自动 case 选择",
        },
        {
            "file": "analyze_trajectories.py",
            "entry": "python terminal-rl/scripts/analyze_trajectories.py --run-dir <run>",
            "data_source": "trajectories/*/traj.json",
            "outputs": "trajectory_classification.json; case_analysis.md",
            "reuse": "复用：轨迹状态分类",
        },
        {
            "file": "analyze_hang_diagnostics.py",
            "entry": "python terminal-rl/scripts/analyze_hang_diagnostics.py --run-dir <run>",
            "data_source": "logs/train.log tail",
            "outputs": "hang_diagnosis.json; hang_diagnosis.md",
            "reuse": "复用：环境/挂起诊断",
        },
        {
            "file": "analyze_case_study.py / compare_case_study.py / run_case_study.sh",
            "entry": "bash terminal-rl/scripts/run_case_study.sh <run>",
            "data_source": "固定样本配置 case_study_samples.yaml + trajectories",
            "outputs": "case_study/*.md/json/jsonl/csv",
            "reuse": "部分复用：本任务需要自动代表样本，因此使用 analyze_exploration_report.py 的 pick_cases",
        },
    ]


def phase0_inventory(
    *,
    run_dir: Path,
    baseline_run_dir: Path,
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    out_dir: Path,
    source: str,
    baseline_source: str,
) -> dict[str, Any]:
    sample_metric_path = run_dir / "logs/metrics.jsonl"
    baseline_metric_path = baseline_run_dir / "logs/metrics.jsonl"
    train_log = run_dir / "logs/train.log"
    baseline_log = baseline_run_dir / "logs/train.log"
    first_record = records[0] if records else {}
    first_baseline = baseline_records[0] if baseline_records else {}
    inventory = {
        "analysis_dir_0624": str(PREV_ANALYSIS_DIR),
        "analysis_dir_0624_files": list_files(PREV_ANALYSIS_DIR),
        "scripts_dir": str(SCRIPTS_DIR),
        "scripts_dir_files": list_files(SCRIPTS_DIR),
        "script_inventory": script_inventory(),
        "metric_sources": {
            "experiment": {
                "records": len(records),
                "source": source,
                "metrics_jsonl": str(sample_metric_path),
                "metrics_jsonl_lines": jsonl_line_count(sample_metric_path),
                "train_log": str(train_log),
                "train_log_metric_line": first_matching_line(train_log, "TERMINAL_RL_METRIC_JSON"),
                "train_log_rollout_line": first_matching_line(train_log, "data.py:"),
                "train_log_train_step_line": first_matching_line(train_log, "model.py:"),
                "field_count": len(set().union(*(record.keys() for record in records))) if records else 0,
                "fields": sorted(set().union(*(record.keys() for record in records))) if records else [],
                "example": first_record,
            },
            "baseline": {
                "records": len(baseline_records),
                "source": baseline_source,
                "metrics_jsonl": str(baseline_metric_path),
                "metrics_jsonl_lines": jsonl_line_count(baseline_metric_path),
                "train_log": str(baseline_log),
                "train_log_metric_line": first_matching_line(baseline_log, "TERMINAL_RL_METRIC_JSON"),
                "train_log_rollout_line": first_matching_line(baseline_log, "data.py:"),
                "train_log_train_step_line": first_matching_line(baseline_log, "model.py:"),
                "field_count": len(set().union(*(record.keys() for record in baseline_records))) if baseline_records else 0,
                "fields": sorted(set().union(*(record.keys() for record in baseline_records))) if baseline_records else [],
                "example": first_baseline,
            },
        },
        "output_dir": str(out_dir),
    }
    (out_dir / "phase0_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return inventory


ALIGNMENT_ROWS = [
    ("训练步 / 横轴", "rollout_id / global_step", "rollout_id / global_step", "可对齐；主表用 rollout<=共同 last_valid，曲线另用 effective_valid_index"),
    ("pass rate / raw_reward", "raw_reward, reward/raw, test_acc, pass_rate", "raw_reward, reward/raw, test_acc, pass_rate", "可对齐；SETA 当前 rollout unit-test pass rate"),
    ("success rate", "pass_rate / unit_test_pass_rate", "raw_reward/test_acc 等价字段；unit_test_pass_rate 缺失", "可对齐到 raw_reward；baseline unit_test_pass_rate 字段缺失，已用 raw_reward"),
    ("total reward", "total_reward, reward/total", "total_reward, reward/total", "可对齐"),
    ("return", "train.log data.py payload: rollout/returns", "train.log data.py payload: rollout/returns", "可对齐；训练侧 rollout aggregate，不作为 SETA pass rate"),
    ("稳定性", "raw_reward std, total_reward std", "raw_reward std, total_reward std", "可对齐；基于有效点总体 std"),
    ("truncation", "truncated_fraction, truncated/sample_count", "truncated_fraction, truncated/sample_count", "可对齐；越低越好"),
    ("operational pass", "derived op_raw=sum(raw_reward*trainable_count)/sum(sample_count)", "同左", "可对齐；环境失败纳入分母"),
    ("环境失败率", "derived fail_fraction=1-sum(trainable_count)/sum(sample_count)", "同左", "可对齐；越低越好"),
    ("response length", "response_length", "response_length", "可对齐；越低通常越省成本/越少截断风险"),
    ("SimHash/NGU 局内", "agent57/ngu_episodic, agent57/episodic_empty_bucket_rate, exact_repeat_count", "缺失", "baseline 无探索字段，缺失，已跳过直接对比"),
    ("lifelong 覆盖", "agent57/lifelong_unique_keys, lifelong_seen_before, derived new_state_proxy", "缺失", "baseline 无探索字段，缺失，已跳过直接对比"),
    ("exploration bonus", "reward/exploration_abs, exploration_reward_abs, agent57/ngu_bonus", "reward/exploration 为 null 或 0，无 Agent57", "仅对本次实验分析趋势；baseline 缺失，已跳过"),
    ("UCB", "agent57/top_arm, top_arm_ratio, top_suppressed_ratio, SQLite arm_events", "缺失", "baseline 无 UCB，缺失，已跳过"),
]


def write_aligned_csv(records: list[dict[str, Any]], baseline_records: list[dict[str, Any]], out_dir: Path) -> Path:
    columns = [
        "run_label",
        "valid_index",
        "rollout_id",
        "global_step",
        "raw_reward",
        "pass_rate",
        "unit_test_pass_rate",
        "total_reward",
        "task_reward",
        "rollout_returns",
        "rollout_rewards",
        "reward_std",
        "truncated_fraction",
        "response_length",
        "sample_count",
        "trainable_count",
        "completed",
        "failed",
        "exploration_abs",
        "exploration_signal",
        "ngu_episodic",
        "lifelong_unique_keys",
        "lifelong_seen_before",
        "new_state_proxy",
        "ngu_bonus",
        "top_arm",
        "top_arm_ratio",
        "top_suppressed_ratio",
    ]
    out_path = out_dir / "aligned_metrics.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for label, group in (("experiment", records), ("baseline", baseline_records)):
            valid_index = 0
            for record in group:
                if not EXP.is_valid(record):
                    continue
                row = {
                    "run_label": label,
                    "valid_index": valid_index,
                    "rollout_id": EXP.rid(record),
                    "global_step": record.get("global_step"),
                    "raw_reward": EXP.metric(record, *EXP.FIELD_KEYS["raw_reward"]),
                    "pass_rate": record.get("pass_rate"),
                    "unit_test_pass_rate": record.get("unit_test_pass_rate"),
                    "total_reward": EXP.metric(record, *EXP.FIELD_KEYS["total_reward"]),
                    "task_reward": EXP.metric(record, *EXP.FIELD_KEYS["task_reward"]),
                    "rollout_returns": record.get("rollout_returns"),
                    "rollout_rewards": record.get("rollout_rewards"),
                    "reward_std": EXP.metric(record, *EXP.FIELD_KEYS["reward_std"]),
                    "truncated_fraction": EXP.metric(record, *EXP.FIELD_KEYS["truncated_fraction"]),
                    "response_length": EXP.metric(record, *EXP.FIELD_KEYS["response_length"]),
                    "sample_count": EXP.metric(record, *EXP.FIELD_KEYS["sample_count"]),
                    "trainable_count": EXP.metric(record, *EXP.FIELD_KEYS["trainable_count"]),
                    "completed": EXP.metric(record, *EXP.FIELD_KEYS["completed"]),
                    "failed": EXP.metric(record, *EXP.FIELD_KEYS["failed"]),
                    "exploration_abs": EXP.metric(record, *EXP.FIELD_KEYS["exploration_abs"]),
                    "exploration_signal": EXP.metric(record, *EXP.FIELD_KEYS["exploration_signal"]),
                    "ngu_episodic": EXP.metric(record, *EXP.FIELD_KEYS["ngu_episodic"]),
                    "lifelong_unique_keys": EXP.metric(record, *EXP.FIELD_KEYS["lifelong_unique_keys"]),
                    "lifelong_seen_before": EXP.metric(record, *EXP.FIELD_KEYS["lifelong_seen_before"]),
                    "new_state_proxy": record.get("_new_state_proxy"),
                    "ngu_bonus": EXP.metric(record, *EXP.FIELD_KEYS["ngu_bonus"]),
                    "top_arm": EXP.metric(record, *EXP.FIELD_KEYS["top_arm"]),
                    "top_arm_ratio": EXP.metric(record, *EXP.FIELD_KEYS["top_arm_ratio"]),
                    "top_suppressed_ratio": EXP.metric(record, *EXP.FIELD_KEYS["top_suppressed_ratio"]),
                }
                writer.writerow(row)
                valid_index += 1
    return out_path


def write_comparison_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    out_path = out_dir / "comparison_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "experiment", "baseline", "diff", "pct_delta", "verdict", "lower_is_better"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "metric": row.get("label"),
                    "experiment": row.get("exp"),
                    "baseline": row.get("baseline"),
                    "diff": row.get("diff"),
                    "pct_delta": row.get("pct_delta"),
                    "verdict": row.get("verdict"),
                    "lower_is_better": row.get("lower_is_better"),
                }
            )
    return out_path


def md_table_compare(rows: list[dict[str, Any]]) -> str:
    lines = ["| 指标 | 本次实验 | baseline | 差值 | 相对变化 | 判断 |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['label']} | {fmt(row['exp'], 4)} | {fmt(row['baseline'], 4)} | "
            f"{fmt(row['diff'], 4)} | {fmt_pct(row['pct_delta'], 1)} | {row['verdict']} |"
        )
    return "\n".join(lines)


def md_exploration_stats(summary: dict[str, Any]) -> str:
    fields = [
        ("局内 NGU episodic", "ngu_episodic"),
        ("局内 empty-bucket rate", "episodic_empty_bucket_rate"),
        ("局内 exact repeats", "episodic_exact_repeat_count"),
        ("局间 lifelong raw novelty", "lifelong_raw"),
        ("局间 lifelong unique keys", "lifelong_unique_keys"),
        ("局间 lifelong seen_before", "lifelong_seen_before"),
        ("局间 new-state proxy", "new_state_proxy"),
        ("融合 NGU bonus", "ngu_bonus"),
        ("abs exploration reward", "exploration_abs"),
        ("UCB top arm id", "top_arm"),
        ("UCB top-arm share", "top_arm_ratio"),
        ("UCB top suppressed ratio", "top_suppressed_ratio"),
    ]
    lines = ["| 指标 | n | mean | first10 | last10 | std |", "|---|---:|---:|---:|---:|---:|"]
    for label, field in fields:
        item = summary.get(field) or {}
        lines.append(
            f"| {label} | {item.get('n', 0)} | {fmt(item.get('mean'), 4)} | "
            f"{fmt(item.get('first10_mean'), 4)} | {fmt(item.get('last10_mean'), 4)} | {fmt(item.get('std'), 4)} |"
        )
    return "\n".join(lines)


def md_missing_table() -> str:
    return "\n".join(
        [
            "| 指标 | 状态 | 处理 |",
            "|---|---|---|",
            "| held-out test pass rate | 日志中 `test_acc` 明确为当前 rollout unit-test pass rate，不是 held-out split | 缺失，已跳过；报告只称 pass rate/raw_reward |",
            "| 独立 SimHash coverage | 未发现 `simhash_coverage` 或 fingerprint coverage 聚合字段 | 缺失，已跳过；用 Agent57 episodic/lifelong 代理覆盖趋势 |",
            "| baseline Agent57/SimHash/UCB | baseline 结构化记录无 `agent57/*` 字段 | 缺失，已跳过；不对探索字段做 baseline 数值对比 |",
            "| 逐 step credit assignment | 保存轨迹只有 turn 和 trajectory 级 reward/metadata | 无法逐 token/step 还原，退化为 turn 级 case 展示 |",
        ]
    )


def improvement_suggestions(raw_cmp: dict[str, Any], trunc_cmp: dict[str, Any], corr: dict[str, Any]) -> list[dict[str, str]]:
    corr_expl = num((corr.get("exploration_abs") or {}).get("pearson_raw_reward"))
    return [
        {
            "rank": "1",
            "title": "固定 v0626 风险约束，做同环境并行 baseline vs v0626 复现",
            "motivation": "本次 run 的 fail/op_raw 与 raw 后段可能受运行日期和环境失败率影响；需要同一天同 worker 池消除混杂。",
            "change": "同一 rjob 窗口启动 baseline 复刻和 v0626，固定 rollout 0-95/96-191/后50有效点三段比较。",
            "expected": "最高收益：确认 raw/op_raw 改善是否可复现，避免把 infra 差异误判为算法收益。",
            "metrics": "raw_reward、valid_raw_weighted、op_raw、fail_fraction、truncated_fraction、rolling10 阈值。",
        },
        {
            "rank": "2",
            "title": "v0626 风险约束拆解 ablation，优先恢复 raw_reward",
            "motivation": "本次 truncation 明显降低，但 raw/op_raw 大幅退化，说明风险约束可能过强或把部分可成功长轨迹压掉。",
            "change": "以 v0626 为底座，分别回退 `EXPLORE_TRUNCATION_PENALTY=0`、`TRUST_TRUNCATED=0.2`、`TRUNCATED_INTRINSIC_SCALE=0.05/0.1`，保留 turn-invariant 与低 UCB epsilon。",
            "expected": "预期找到 raw_reward 与 truncation 的 Pareto 点，避免为了低截断牺牲 pass rate。",
            "metrics": "raw_reward、op_raw、后50 raw_reward、truncated_fraction、response_length。",
        },
        {
            "rank": "3",
            "title": "turn-level intrinsic 只奖励产生有效工具进展的 turn",
            "motivation": "v0626 已用 `explore_agent57_turn_intrinsic_signal`，但 turn-level novelty 仍可能奖励空转或重复工具序列。",
            "change": "在生成端将 low_progress/工具错误 turn 的 intrinsic 乘 0 或小系数，保留 completed turn 的正向信号。",
            "expected": "预期提高探索信号与 raw_reward 的相关性，减少高覆盖但截断 case。",
            "metrics": "exploration_abs 与 raw_reward Pearson r、low_progress_fraction、case 中工具错误 turn 占比。",
        },
        {
            "rank": "4",
            "title": "UCB value 改成 raw-preserving 风险约束",
            "motivation": "当前各 arm success_rate 都偏低，单纯压低截断不足以提升任务完成；UCB 应更快偏向 normalized_base 高的 arm。",
            "change": "做 `normalized_base - 0.25*trunc_rate - 0.5*parse_rate` 与 `success` value ablation，和 v0626 风险约束拆开评估。",
            "expected": "预期在保持低截断的同时恢复高 raw arm 的采样比例。",
            "metrics": "arm_events success_rate/trunc_rate by arm、top_arm 分布、raw_reward 后50。",
        },
        {
            "rank": "5",
            "title": "lifelong new-state proxy 退火或按 task 重置小窗口",
            "motivation": "new-state proxy 后期接近 0，后期 lifelong 主要在 seen-before 上波动，未必提供有效覆盖增益。",
            "change": "做 lifelong coef 0.003/0.005 和 task-local decay ablation，或对 seen_before 高的 task 降低 life modifier。",
            "expected": "预期降低无效覆盖追逐，提升后期 raw_reward 稳定性。",
            "metrics": "new_state_proxy last10、lifelong_unique_keys、raw_reward std、后50 raw_reward。",
        },
    ]


def adjust_case_selection(run_dir: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer case-study examples that match their explanatory label."""
    records = EXP.scan_trajectories(run_dir)
    used: set[Path] = {
        path
        for case in cases
        for _, path in case.get("paths", [])
        if isinstance(path, Path)
    }

    def replace_case(case_id: str, label: str, path: Path) -> None:
        for case in cases:
            if case.get("case_id") == case_id:
                old_paths = [p for _, p in case.get("paths", []) if isinstance(p, Path)]
                for old in old_paths:
                    used.discard(old)
                case["paths"] = [(label, path)]
                used.add(path)
                return

    truncated_failures = sorted(
        [
            item
            for item in records
            if item.get("status") == "TRUNCATED"
            and (num(item.get("raw_score")) is None or float(num(item.get("raw_score")) or 0.0) < 1.0)
        ],
        key=lambda item: (EXP.case_strength(item), num(item.get("unique_keys")) or 0.0, EXP.case_rollout(item)),
        reverse=True,
    )
    if truncated_failures:
        replace_case("high_exploration_truncated", "本次实验高探索截断负例", truncated_failures[0]["path"])

    if not any(case.get("case_id") == "high_exploration_success" for case in cases):
        successes = sorted(
            [
                item
                for item in records
                if item.get("status") == "COMPLETED"
                and (num(item.get("raw_score")) or 0.0) >= 1.0
                and item.get("path") not in used
            ],
            key=lambda item: (EXP.case_strength(item), EXP.case_rollout(item)),
            reverse=True,
        )
        if successes:
            item = successes[0]
            cases.append(
                {
                    "case_id": "high_exploration_success",
                    "title": "高探索成功样本",
                    "reason": "自动选择未被其他 case 使用、且 NGU/unique/beta 综合探索强度较高的成功轨迹，展示探索机制的正例。",
                    "paths": [("本次实验高探索成功", item["path"])],
                }
            )
    return cases


def render_report(
    *,
    run_dir: Path,
    baseline_run_dir: Path,
    out_dir: Path,
    review_doc: Path,
    history_doc: Path,
    inventory: dict[str, Any],
    analysis: dict[str, Any],
    standard_summary: dict[str, Any] | None,
    trajectory_classification: dict[str, Any] | None,
    hang: dict[str, Any] | None,
    generated_files: list[tuple[Path, str, str]],
) -> str:
    summary = analysis["summary"]
    baseline_summary = analysis["baseline_summary"]
    same = analysis["same_rollout_window"]["run"]
    same_base = analysis["same_rollout_window"]["baseline"]
    first_n = analysis["first_n_valid_window"]["run"]
    first_n_base = analysis["first_n_valid_window"]["baseline"]
    last50 = analysis["last50_same_rollout_window"]["run"]
    last50_base = analysis["last50_same_rollout_window"]["baseline"]
    rows = analysis["comparison_rows"]
    thresholds = analysis["convergence_thresholds"]
    corr = analysis["correlations"]
    arm = analysis["agent57_arm_events"]
    cases = analysis["case_studies"]
    figs = {k: Path(v) for k, v in analysis["figures"].items()}
    raw_cmp = compare(m(same, "raw_reward"), m(same_base, "raw_reward"))
    op_cmp = compare(m(same, "op_raw"), m(same_base, "op_raw"))
    trunc_cmp = compare(m(same, "truncated_fraction"), m(same_base, "truncated_fraction"), lower_is_better=True)
    final_cmp = compare(m(last50, "raw_reward"), m(last50_base, "raw_reward"))
    ret_cmp = compare(m(same, "rollout_returns"), m(same_base, "rollout_returns"))
    threshold_04 = thresholds.get("0.40") or {}
    threshold_04_base = thresholds.get("baseline_0.40") or {}
    threshold_diff = None
    if threshold_04.get("rollout_id") is not None and threshold_04_base.get("rollout_id") is not None:
        threshold_diff = threshold_04["rollout_id"] - threshold_04_base["rollout_id"]
    corr_expl = (corr.get("exploration_abs") or {}).get("pearson_raw_reward")
    reset500 = ((standard_summary or {}).get("reset500") or {}).get("total")
    error_counts = (((standard_summary or {}).get("no_training_diagnostics") or {}).get("error_counts") or {})
    gen_failed = error_counts.get("generate_failed")
    suggestions = improvement_suggestions(raw_cmp, trunc_cmp, corr)

    lines: list[str] = []
    lines.append("# 探索算法实验分析报告")
    lines.append("")
    lines.append(f"- 本次实验：`{run_dir.name}`")
    lines.append(f"- Baseline：`{baseline_run_dir.name}`")
    lines.append(f"- 输出目录：`{out_dir}`")
    lines.append(f"- 生成时间：`{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`")
    lines.append("")

    lines.append("## 执行摘要")
    lines.append("")
    lines.append(f"1. 同 rollout<=`{analysis['same_rollout_window']['max_rollout']}` 的有效点比较，raw_reward/pass rate 为 **{fmt(raw_cmp['exp'], 4)}** vs baseline **{fmt(raw_cmp['baseline'], 4)}**，差值 **{fmt(raw_cmp['diff'], 4)} ({fmt_pct(raw_cmp['pct_delta'], 1)})**，判断为 **{raw_cmp['verdict']}**。")
    lines.append(f"2. operational pass `op_raw` 为 **{fmt(op_cmp['exp'], 4)}** vs baseline **{fmt(op_cmp['baseline'], 4)}**，差值 **{fmt(op_cmp['diff'], 4)} ({fmt_pct(op_cmp['pct_delta'], 1)})**；后 50 有效点 raw_reward 为 **{fmt(final_cmp['exp'], 4)}** vs **{fmt(final_cmp['baseline'], 4)}**，判断为 **{final_cmp['verdict']}**。")
    lines.append(f"3. 风险侧，truncated_fraction 为 **{fmt(trunc_cmp['exp'], 4)}** vs baseline **{fmt(trunc_cmp['baseline'], 4)}**，差值 **{fmt(trunc_cmp['diff'], 4)} ({fmt_pct(trunc_cmp['pct_delta'], 1)})**，判断为 **{trunc_cmp['verdict']}**；rolling10 raw_reward 达到 0.40 的 rollout 为 `{threshold_04.get('rollout_id')}` vs baseline `{threshold_04_base.get('rollout_id')}`，差 `{threshold_diff}`。")
    lines.append(f"4. 探索信号与 raw_reward 的 Pearson r：`exploration_abs` 为 **{fmt(corr_expl, 3)}**；baseline 缺少 Agent57/SimHash/UCB 字段，探索指标不做直接数值对比，已显式跳过。")
    lines.append(f"5. 数据质量混杂项：本次有效点 **{summary['n_valid']} / {summary['n_points']}**，空/不可训练批次 **{summary['n_invalid_or_zero_trainable']}**；标准日志记录 reset_500=`{reset500}`、generate_failed=`{gen_failed}`。")
    lines.append("")

    lines.append("## 提升评估")
    lines.append("")
    lines.append("主比较使用共同原始 rollout 窗口；曲线图使用 effective valid rollout index，避免 baseline 中间空批次造成曲线提前断开。")
    lines.append("")
    lines.append(md_table_compare(rows))
    lines.append("")
    lines.append("**按前 N 个有效点对齐**")
    lines.append("")
    lines.append("| 指标 | 本次实验前N | baseline前N | 差值 | 相对变化 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, field, lower in (
        ("raw_reward", "raw_reward", False),
        ("valid_raw_weighted", "valid_raw_weighted", False),
        ("op_raw", "op_raw", False),
        ("rollout_returns", "rollout_returns", False),
        ("truncated_fraction", "truncated_fraction", True),
        ("fail_fraction", "fail_fraction", True),
        ("response_length", "response_length", True),
    ):
        row = compare(m(first_n, field), m(first_n_base, field), lower_is_better=lower)
        lines.append(f"| {label} | {fmt(row['exp'], 4)} | {fmt(row['baseline'], 4)} | {fmt(row['diff'], 4)} | {fmt_pct(row['pct_delta'], 1)} |")
    lines.append("")
    lines.append("**收敛速度（rolling10 raw_reward）**")
    lines.append("")
    lines.append("| 阈值 | 本次实验 rollout | baseline rollout | 本次 rolling mean | baseline rolling mean |")
    lines.append("|---:|---:|---:|---:|---:|")
    for threshold in ("0.30", "0.35", "0.40", "0.45"):
        item = thresholds.get(threshold) or {}
        base = thresholds.get(f"baseline_{threshold}") or {}
        lines.append(f"| {threshold} | {item.get('rollout_id', 'NA')} | {base.get('rollout_id', 'NA')} | {fmt(item.get('rolling_mean'), 4)} | {fmt(base.get('rolling_mean'), 4)} |")
    lines.append("")
    lines.append(f"return 对比：`rollout/returns` 来自 `train.log` 的 `data.py` payload，同窗口均值为 **{fmt(ret_cmp['exp'], 4)}** vs baseline **{fmt(ret_cmp['baseline'], 4)}**，差值 **{fmt(ret_cmp['diff'], 4)} ({fmt_pct(ret_cmp['pct_delta'], 1)})**。该指标是训练侧 rollout aggregate，不等同于 SETA pass rate。")
    lines.append("")
    lines.append(f"![核心指标对比]({rel(figs['core'], out_dir)})")
    lines.append("")
    lines.append(f"![标准 overview]({rel(figs['overview'], out_dir)})")
    lines.append("")

    lines.append("## 探索指标")
    lines.append("")
    lines.append(md_exploration_stats(summary))
    lines.append("")
    lines.append("**探索强度与 raw_reward 相关性**")
    lines.append("")
    lines.append("| 探索指标 | n | Pearson r |")
    lines.append("|---|---:|---:|")
    for name, item in corr.items():
        lines.append(f"| {name} | {item.get('n', 0)} | {fmt(item.get('pearson_raw_reward'), 3)} |")
    lines.append("")
    if arm.get("available"):
        lines.append(f"SQLite `arm_events` 共 **{fmt_int(arm.get('n_events'))}** 条；`lifelong_counts` key 数 **{fmt_int((arm.get('lifelong_counts') or {}).get('n_keys'))}**，平均计数 **{fmt((arm.get('lifelong_counts') or {}).get('count_mean'), 3)}**。")
        lines.append("")
        lines.append("| arm | n | normalized_base | success_rate | trunc_rate | parse_rate | bonus_mean |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for item in arm.get("by_arm") or []:
            lines.append(
                f"| {item['arm_id']} | {item['n']} | {fmt(item['normalized_base_score_mean'], 4)} | "
                f"{fmt(item['success_rate'], 4)} | {fmt(item['truncated_rate'], 4)} | {fmt(item['parse_error_rate'], 4)} | {fmt(item['bonus_mean'], 7)} |"
            )
        lines.append("")
    lines.append(f"![探索链路趋势]({rel(figs['exploration'], out_dir)})")
    lines.append("")
    lines.append(f"![探索-性能关系]({rel(figs['relationship'], out_dir)})")
    lines.append("")
    lines.append(f"![UCB arm 聚合]({rel(figs['arm_by_arm'], out_dir)})")
    lines.append("")
    lines.append(f"![UCB arm 时间趋势]({rel(figs['arm_time'], out_dir)})")
    lines.append("")

    lines.append("## Case-Study")
    lines.append("")
    if trajectory_classification:
        dist = trajectory_classification.get("class_distribution") or {}
        lines.append(f"保存轨迹共 **{trajectory_classification.get('n_trajectories')}** 条：pass `{dist.get('pass', 0)}`，fail_eval_normal `{dist.get('fail_eval_normal', 0)}`，truncated `{dist.get('truncated', 0)}`，fail_eval_500 `{dist.get('fail_eval_500', 0)}`。")
        lines.append("")
    lines.append(f"完整逐 turn 轨迹见 [`case_study_details.md`]({rel(out_dir / 'case_study_details.md', out_dir)})。")
    lines.append("")
    lines.append("| Case | 轨迹 | status | raw | total | arm/beta | trust | 关键解释 |")
    lines.append("|---|---|---|---:|---:|---|---:|---|")
    for case in cases:
        for entry in case.get("entries") or []:
            s = entry["summary"]
            arm_beta = "无" if s.get("arm_id") is None and num(s.get("beta")) is None else f"{s.get('arm_id')}/{fmt(s.get('beta'), 3)}"
            reason = " ".join(str(case.get("reason", "")).split())
            lines.append(f"| {case['title']} | {entry['label']} | `{s.get('status')}` | {fmt(s.get('raw_score'), 3)} | {fmt(s.get('total_reward'), 3)} | {arm_beta} | {fmt(s.get('trust'), 3)} | {reason[:120]} |")
    lines.append("")
    lines.append("解释：成功 case 只能证明探索 arm 在个别样本上可产生有效完成；失败/截断 case 显示覆盖度或 bonus 升高并不自动转化为 pass rate。若 case 只匹配同 task 而非同 sample，本报告不把它当作严格因果证据。")
    lines.append("")

    lines.append("## 结论与后续建议")
    lines.append("")
    lines.append(f"结论：本次实验相对 baseline 的主结论是 raw_reward **{raw_cmp['verdict']}**、op_raw **{op_cmp['verdict']}**、后 50 有效点 raw_reward **{final_cmp['verdict']}**、truncation **{trunc_cmp['verdict']}**。因此它是否优于 baseline 不能只看探索指标，必须以 pass/op_raw 和风险指标共同判断。")
    lines.append("")
    lines.append("**按预期收益排序的下一步实验**")
    lines.append("")
    lines.append("| 优先级 | 实验 | 动机 | 改动点 | 预期影响 | 验证指标 |")
    lines.append("|---:|---|---|---|---|---|")
    for item in suggestions:
        lines.append(f"| {item['rank']} | {item['title']} | {item['motivation']} | {item['change']} | {item['expected']} | {item['metrics']} |")
    lines.append("")
    lines.append("代码 review 依据：v0626 启动脚本降低 UCB pressure/epsilon、关闭 episodic include_turn、启用 turn-level intrinsic、cosine lambda schedule、truncated intrinsic scale=0 和 `EXPLORE_TRUNCATION_PENALTY=-0.03`；对应配置见 `terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0626.sh:77-167`。训练后处理实现见 `reward_postprocess.py:42-58`、`:136-182`、`:263-337`；turn-level intrinsic 写入见 `generate.py:3784-3802`；UCB value/arm event 见 `explore_agent57_lite.py:1695-1731`、`:1769-1842`。")
    lines.append("")

    lines.append("## 数据说明")
    lines.append("")
    lines.append("**数据来源与解析代码**")
    lines.append("")
    lines.append(f"- 新实验指标源：`{run_dir / 'logs/metrics.jsonl'}` + `{run_dir / 'logs/train.log'}`；首条结构化 train.log 记录在行 `{inventory['metric_sources']['experiment']['train_log_metric_line']}`。")
    lines.append(f"- Baseline 指标源：`{baseline_run_dir / 'logs/metrics.jsonl'}` + `{baseline_run_dir / 'logs/train.log'}`；首条结构化 train.log 记录在行 `{inventory['metric_sources']['baseline']['train_log_metric_line']}`。")
    lines.append("- 解析入口复用 `terminal-rl/scripts/analyze_exploration_report.py:126-209` 合并 `metrics.jsonl` 与 `train.log`，`plot_training_metrics.py:34-49` 定义日志正则，`plot_training_metrics.py:91-220` 抽取 rollout/train/structured metric。")
    lines.append("- 字段映射复用 `analyze_exploration_report.py:290-317`；`valid_raw_weighted/op_raw/fail_fraction` 派生公式见 `analyze_exploration_report.py:253-270`；有效点定义见 `analyze_exploration_report.py:212-215`。")
    lines.append(f"- 中间数据：`{rel(out_dir / 'aligned_metrics.csv', out_dir)}`、`{rel(out_dir / 'comparison_summary.csv', out_dir)}`、`{rel(out_dir / 'exploration_analysis.json', out_dir)}`。")
    lines.append(f"- 启动前 review 文档：`{review_doc}`；历史报告：`{history_doc}`。")
    lines.append("")
    lines.append("**Phase 0 探查结论**")
    lines.append("")
    lines.append(f"- 0624 analysis 目录共 `{len(inventory['analysis_dir_0624_files'])}` 个文件；典型产物包括 `report.md`、`summary_stats.json`、`exploration_analysis.json`、`trajectory_classification.json` 和 `figs/*.png`。完整清单见 `phase0_inventory.json`。")
    lines.append(f"- scripts 目录共 `{len(inventory['scripts_dir_files'])}` 个非 pycache 文件；关键脚本入口、数据源和复用决策如下。")
    lines.append("")
    lines.append("| 脚本 | 入口 | 数据源 | 产出 | 复用/扩展决策 |")
    lines.append("|---|---|---|---|---|")
    for item in inventory["script_inventory"]:
        lines.append(f"| `{item['file']}` | `{item['entry']}` | {item['data_source']} | {item['outputs']} | {item['reuse']} |")
    lines.append("")
    lines.append(f"- 新实验可用字段 `{inventory['metric_sources']['experiment']['field_count']}` 个，baseline 可用字段 `{inventory['metric_sources']['baseline']['field_count']}` 个；完整字段列表与示例记录写入 `phase0_inventory.json`。")
    lines.append("- 分析计划：生成核心指标对比图、探索链路趋势图、探索-性能相关图、UCB arm 图；定量评估收敛速度/最终性能/稳定性；自动选择 baseline 对照、高低探索、成功/失败/截断代表轨迹。")
    lines.append("")
    lines.append("**指标对齐表**")
    lines.append("")
    lines.append("| 分析指标 | 新实验字段 | baseline 字段 | 对齐策略 |")
    lines.append("|---|---|---|---|")
    for metric_name, exp_field, base_field, strategy in ALIGNMENT_ROWS:
        lines.append(f"| {metric_name} | `{exp_field}` | `{base_field}` | {strategy} |")
    lines.append("")
    lines.append("**缺失/异常处理**")
    lines.append("")
    lines.append(md_missing_table())
    lines.append("")
    lines.append(f"- 解析异常：JSONL 坏行按现有 parser 跳过；本次 `n_invalid_or_zero_trainable={summary['n_invalid_or_zero_trainable']}`，baseline `n_invalid_or_zero_trainable={baseline_summary['n_invalid_or_zero_trainable']}`。")
    lines.append(f"- 对齐策略：主表使用共同原始 rollout<=`{analysis['same_rollout_window']['max_rollout']}`；补充表使用前 `{analysis['first_n_valid_window']['n']}` 个有效点；后段性能使用同窗口后 50 有效点。")
    if hang:
        assessment = hang.get("assessment") if isinstance(hang.get("assessment"), dict) else {}
        likelihood = hang.get("likelihood") or assessment.get("likelihood") or assessment.get("similar_dynamic_sampling_env_hang_likelihood")
        lines.append(f"- hang 诊断：`{likelihood}`；详见 `hang_diagnosis.json/md`。")
    lines.append("")
    lines.append("**生成文件**")
    lines.append("")
    lines.append("| 状态 | 文件 | 说明 |")
    lines.append("|---|---|---|")
    for status, path, desc in generated_files:
        lines.append(f"| {status} | `{path}` | {desc} |")
    lines.append("")
    return "\n".join(lines)


def update_history_report(history_doc: Path, analysis: dict[str, Any], report_path: Path) -> None:
    summary = analysis["summary"]
    same = analysis["same_rollout_window"]["run"]
    same_base = analysis["same_rollout_window"]["baseline"]
    last50 = analysis["last50_same_rollout_window"]["run"]
    last50_base = analysis["last50_same_rollout_window"]["baseline"]
    raw_cmp = compare(m(same, "raw_reward"), m(same_base, "raw_reward"))
    op_cmp = compare(m(same, "op_raw"), m(same_base, "op_raw"))
    trunc_cmp = compare(m(same, "truncated_fraction"), m(same_base, "truncated_fraction"), lower_is_better=True)
    final_cmp = compare(m(last50, "raw_reward"), m(last50_base, "raw_reward"))
    corr_expl = ((analysis.get("correlations") or {}).get("exploration_abs") or {}).get("pearson_raw_reward")
    section = f"""
## 0626 riskcal/turninv/truncpenalty 诊断

证据报告：`{rel(report_path, ROOT)}`

本次 run：`runs/{DEFAULT_RUN.name}`

同 rollout<=`{analysis['same_rollout_window']['max_rollout']}` 有效点对齐：

| 指标 | 0626 | baseline | 差值 | 相对变化 | 判断 |
|---|---:|---:|---:|---:|---|
| raw_reward 均值 | {fmt(raw_cmp['exp'], 4)} | {fmt(raw_cmp['baseline'], 4)} | {fmt(raw_cmp['diff'], 4)} | {fmt_pct(raw_cmp['pct_delta'], 1)} | {raw_cmp['verdict']} |
| raw_reward 后50有效点 | {fmt(final_cmp['exp'], 4)} | {fmt(final_cmp['baseline'], 4)} | {fmt(final_cmp['diff'], 4)} | {fmt_pct(final_cmp['pct_delta'], 1)} | {final_cmp['verdict']} |
| op_raw | {fmt(op_cmp['exp'], 4)} | {fmt(op_cmp['baseline'], 4)} | {fmt(op_cmp['diff'], 4)} | {fmt_pct(op_cmp['pct_delta'], 1)} | {op_cmp['verdict']} |
| truncated_fraction | {fmt(trunc_cmp['exp'], 4)} | {fmt(trunc_cmp['baseline'], 4)} | {fmt(trunc_cmp['diff'], 4)} | {fmt_pct(trunc_cmp['pct_delta'], 1)} | {trunc_cmp['verdict']} |

探索摘要：

- 结构化指标截止 rollout `{summary['last_rollout']}`，有效点 `{summary['n_valid']}/{summary['n_points']}`。
- exploration_abs 与 raw_reward 的 Pearson r 为 `{fmt(corr_expl, 3)}`。
- baseline 无 Agent57/SimHash/UCB 字段，探索字段缺失，直接数值对比已跳过。

判断：0626 的 risk-calibrated/turn-invariant/trunc-penalty 改动显著压低了 truncation/response length，但本次 raw_reward、op_raw 和后50有效点均退化。下一步不应继续加大风险惩罚，优先做同环境并行复现，并拆解 trunc penalty、truncated intrinsic scale、trust_truncated 三个风险约束以恢复 raw_reward。
""".strip()

    old = history_doc.read_text(encoding="utf-8") if history_doc.is_file() else ""
    heading = "## 0626 riskcal/turninv/truncpenalty 诊断"
    if heading in old:
        start = old.index(heading)
        next_marker = old.find("\n## ", start + 1)
        if next_marker == -1:
            new_text = old[:start].rstrip() + "\n\n" + section + "\n"
        else:
            new_text = old[:start].rstrip() + "\n\n" + section + "\n\n" + old[next_marker + 1 :].lstrip()
    else:
        insert_before = "\n## 新 Run 更新流程"
        if insert_before in old:
            idx = old.index(insert_before)
            new_text = old[:idx].rstrip() + "\n\n" + section + "\n" + old[idx:]
        else:
            new_text = old.rstrip() + "\n\n" + section + "\n"
    history_doc.write_text(new_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-log", type=Path, default=DEFAULT_BASELINE_LOG)
    parser.add_argument("--review-doc", type=Path, default=DEFAULT_REVIEW_DOC)
    parser.add_argument("--history-doc", type=Path, default=DEFAULT_HISTORY_DOC)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-standard", action="store_true", help="Do not rerun standard summary/trajectory/hang scripts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    baseline_log = args.baseline_log.expanduser().resolve()
    baseline_run_dir = baseline_log.parent.parent
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else run_dir / "metrics/analysis"
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    if not args.skip_standard:
        run_standard_scripts(run_dir, out_dir)

    records, source = EXP.load_records(run_dir, run_dir / "logs/train.log")
    baseline_records, baseline_source = EXP.load_records(baseline_run_dir, baseline_log)
    if not records:
        raise SystemExit(f"no records parsed for {run_dir}")
    if not baseline_records:
        raise SystemExit(f"no records parsed for baseline {baseline_run_dir}")

    run_rollouts, run_parse = load_rollout_metrics(run_dir)
    base_rollouts, base_parse = load_rollout_metrics(baseline_run_dir)
    augment_rollout_metrics(records, run_rollouts)
    augment_rollout_metrics(baseline_records, base_rollouts)
    EXP.add_derived(records)
    EXP.add_derived(baseline_records)

    summary = EXP.summarize_records(records)
    baseline_summary = EXP.summarize_records(baseline_records)
    max_common = int(min(summary["last_valid_rollout"], baseline_summary["last_valid_rollout"]))
    same = EXP.window_summary(records, max_rollout=max_common)
    same_base = EXP.window_summary(baseline_records, max_rollout=max_common)
    first_n_count = int(summary["n_valid"])
    first_n = EXP.window_summary(records, first_n_valid=first_n_count)
    first_n_base = EXP.window_summary(baseline_records, first_n_valid=first_n_count)
    last50 = EXP.window_summary(records, max_rollout=max_common, last_n_valid=50)
    last50_base = EXP.window_summary(baseline_records, max_rollout=max_common, last_n_valid=50)
    for target, extra in (
        (summary, window_extra_stats(records)),
        (baseline_summary, window_extra_stats(baseline_records)),
        (same, window_extra_stats(records, max_rollout=max_common)),
        (same_base, window_extra_stats(baseline_records, max_rollout=max_common)),
        (first_n, window_extra_stats(records, first_n_valid=first_n_count)),
        (first_n_base, window_extra_stats(baseline_records, first_n_valid=first_n_count)),
        (last50, window_extra_stats(records, max_rollout=max_common, last_n_valid=50)),
        (last50_base, window_extra_stats(baseline_records, max_rollout=max_common, last_n_valid=50)),
    ):
        target.update(extra)

    rows = EXP.comparison_rows(same, same_base, last50, last50_base)
    ret_row = compare(m(same, "rollout_returns"), m(same_base, "rollout_returns"))
    ret_row.update({"label": "rollout_returns 均值", "lower_is_better": False})
    rows.insert(4, ret_row)

    thresholds: dict[str, Any] = {}
    for threshold in (0.30, 0.35, 0.40, 0.45):
        key = f"{threshold:.2f}"
        thresholds[key] = EXP.rolling_threshold(records, threshold)
        thresholds[f"baseline_{key}"] = EXP.rolling_threshold(baseline_records, threshold)

    correlations = EXP.correlation_summary(records)
    arm = EXP.load_arm_events(run_dir / "agent57_lite.sqlite3")
    figs = {
        "core": figs_dir / "baseline_core_comparison.png",
        "exploration": figs_dir / "exploration_metrics_trends.png",
        "relationship": figs_dir / "exploration_performance_relationship.png",
        "arm_by_arm": figs_dir / "agent57_arm_events_by_arm.png",
        "arm_time": figs_dir / "agent57_arm_events_over_time.png",
        "overview": figs_dir / "overview.png",
    }
    EXP.plot_core(records, baseline_records, figs["core"], max_common, exp_label="experiment")
    EXP.plot_exploration(records, figs["exploration"])
    EXP.plot_relationship(records, correlations, figs["relationship"])
    EXP.plot_arm_events(arm, figs["arm_by_arm"], figs["arm_time"])

    cases = adjust_case_selection(run_dir, EXP.pick_cases(run_dir, baseline_run_dir))
    case_summaries = EXP.render_case_studies(cases, out_dir / "case_study_details.md", ROOT)

    aligned_csv = write_aligned_csv(records, baseline_records, out_dir)
    comparison_csv = write_comparison_csv(rows, out_dir)
    inventory = phase0_inventory(
        run_dir=run_dir,
        baseline_run_dir=baseline_run_dir,
        records=records,
        baseline_records=baseline_records,
        out_dir=out_dir,
        source=source,
        baseline_source=baseline_source,
    )

    standard_summary = EXP.load_json(out_dir / "summary_stats.json")
    trajectory_classification = EXP.load_json(out_dir / "trajectory_classification.json")
    hang = EXP.load_json(out_dir / "hang_diagnosis.json")

    analysis = {
        "schema": "openclaw.exploration_v0626_report.v1",
        "run_dir": str(run_dir),
        "baseline_run_dir": str(baseline_run_dir),
        "metric_source": source,
        "baseline_metric_source": baseline_source,
        "train_log_rollout_parse": {"experiment": run_parse, "baseline": base_parse},
        "summary": summary,
        "baseline_summary": baseline_summary,
        "same_rollout_window": {"max_rollout": max_common, "run": same, "baseline": same_base},
        "first_n_valid_window": {"n": first_n_count, "run": first_n, "baseline": first_n_base},
        "last50_same_rollout_window": {"run": last50, "baseline": last50_base},
        "comparison_rows": rows,
        "convergence_thresholds": thresholds,
        "correlations": correlations,
        "agent57_arm_events": arm,
        "case_studies": case_summaries,
        "alignment_table": ALIGNMENT_ROWS,
        "figures": {name: str(path) for name, path in figs.items()},
        "intermediate_files": {"aligned_metrics_csv": str(aligned_csv), "comparison_summary_csv": str(comparison_csv)},
    }
    analysis_path = out_dir / "exploration_analysis.json"
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    generated_files = [
        ("新增/覆盖", out_dir / "report.md", "最终中文分析报告"),
        ("新增/覆盖", analysis_path, "结构化统计、对比、相关性、case 和图表路径"),
        ("新增/覆盖", aligned_csv, "两组实验有效点对齐后的中间 CSV"),
        ("新增/覆盖", comparison_csv, "核心指标对比中间 CSV"),
        ("新增/覆盖", out_dir / "phase0_inventory.json", "Phase 0 探查清单、字段列表、示例记录"),
        ("新增/覆盖", out_dir / "case_study_details.md", "代表轨迹完整 turn 级 case study"),
        ("新增/覆盖", figs["core"], "核心指标 vs baseline 对比图"),
        ("新增/覆盖", figs["exploration"], "Agent57/SimHash 探索链路趋势图"),
        ("新增/覆盖", figs["relationship"], "探索强度与 raw_reward 关系图"),
        ("新增/覆盖", figs["arm_by_arm"], "SQLite arm_events 按 arm 聚合图"),
        ("新增/覆盖", figs["arm_time"], "SQLite arm_events 时间趋势图"),
        ("新增/覆盖", out_dir / "summary_stats.json", "标准训练统计"),
        ("新增/覆盖", out_dir / "trajectory_classification.json", "轨迹分类统计"),
        ("新增/覆盖", out_dir / "hang_diagnosis.json", "hang/环境异常诊断"),
        ("修改", args.history_doc.resolve(), "历史探索报告追加本次实验小节"),
    ]

    report = render_report(
        run_dir=run_dir,
        baseline_run_dir=baseline_run_dir,
        out_dir=out_dir,
        review_doc=args.review_doc.resolve(),
        history_doc=args.history_doc.resolve(),
        inventory=inventory,
        analysis=analysis,
        standard_summary=standard_summary,
        trajectory_classification=trajectory_classification,
        hang=hang,
        generated_files=generated_files,
    )
    report_path = out_dir / "report.md"
    report_path.write_text(report + "\n", encoding="utf-8")
    update_history_report(args.history_doc.resolve(), analysis, report_path)

    print(f"[+] wrote {report_path}")
    print(f"[+] wrote {analysis_path}")
    print(f"[+] updated {args.history_doc.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
