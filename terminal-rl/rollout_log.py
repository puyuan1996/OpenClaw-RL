from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import wandb
from slime.utils import logging_utils
from slime.utils.types import Sample
from slime.ray.rollout import compute_rollout_step

logger = logging.getLogger(__name__)

_METRIC_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_REWARD_COMPONENT_KEYS = (
    "raw_score",
    "base_score",
    "prm_turn_score",
    "safety_score",
    "explore_intrinsic",
    "explore_intrinsic_scaled",
    "explore_intrinsic_effective_coef",
    "explore_intrinsic_schedule_multiplier",
    "explore_safety_penalty",
    "explore_lprnd",
    "explore_lprnd_raw",
    "explore_lprnd_effective_coef",
    "explore_lprnd_schedule_multiplier",
    "explore_cde_actor_bonus",
    "explore_cde_actor_log_ppl",
    "explore_cde_actor_eligible",
    "explore_cde_actor_base_mean",
    "explore_cde_actor_cap",
    "explore_cde_actor_scaled",
    "explore_cde_actor_clipped",
    "explore_cde_actor_omega",
    "explore_cde_actor_base_magnitude",
    "explore_post_norm_bonus_raw",
    "explore_post_norm_bonus",
    "explore_total_bonus",
    "explore_base_score_before_bonus",
    "explore_bonus_to_base_abs_ratio",
    "explore_curiosity_pressure",
    "explore_tool_intrinsic_pressure",
    "explore_safety_pressure",
    "explore_mood_code",
    "explore_turn_count",
    "explore_tool_call_count",
    "explore_action_count",
    "explore_danger_command_count",
    "explore_parse_error_count",
)
_REWARD_DETAIL_NUMERIC_KEYS = (
    "base",
    "n_tool_calls",
    "tool_successes",
    "n_turns",
    "parse_errors",
    "response_words",
    "progress",
    "progress_adjust",
    "turn_penalty",
    "parse_penalty",
    "truncate_penalty",
    "unsafe_tool_penalty",
    "tool_success_bonus",
    "warning_bonus",
    "refusal_quality_bonus",
    "safe_completion_bonus",
    "concise_bonus",
    "concise_refusal_bonus",
)
_STRUCTURED_LOG_PREFIX = "TERMINAL_RL_METRIC_JSON"
_STRUCTURED_SCHEMA = "terminal_rl.per_dataset_metrics.v1"
_LAST_EVAL_BY_DATASET: dict[str, dict[str, Any]] = {}


def _ensure_terminal_step_metric(args) -> None:
    if not getattr(args, "use_wandb", False):
        return
    try:
        wandb.define_metric("terminal/*", step_metric="rollout/step")
        wandb.define_metric("per_dataset/*", step_metric="rollout/step")
    except Exception as e:
        logger.warning("Failed to define wandb step metric for terminal/*: %s", e)


def _sanitize_metric_part(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text == "terminal_bench":
        # The converted Seta dataset uses terminal_bench as its reward source.
        # Expose the operational dataset name in metrics for mixed-run debugging.
        text = "seta"
    text = _METRIC_NAME_RE.sub("_", text).strip("._-")
    return text or "unknown"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _task_meta_from_sample(sample: Sample) -> dict[str, Any]:
    prompt = _as_dict(getattr(sample, "prompt", None))
    metadata = _as_dict(getattr(sample, "metadata", None))

    if isinstance(_nested_get(metadata, "task_meta"), dict):
        return metadata["task_meta"]
    if isinstance(_nested_get(prompt, "task_meta"), dict):
        return prompt["task_meta"]

    # Dataset records in terminal-rl store task metadata directly under
    # sample.metadata; older/debug samples may store it directly as prompt.
    if any(k in metadata for k in ("data_source", "task_path", "task_name")):
        return metadata
    if any(k in prompt for k in ("data_source", "task_path", "task_name")):
        return prompt
    return {}


def _dataset_name(sample: Sample) -> str:
    metadata = _as_dict(getattr(sample, "metadata", None))
    prompt = _as_dict(getattr(sample, "prompt", None))
    task_meta = _task_meta_from_sample(sample)

    raw = (
        metadata.get("data_source")
        or _nested_get(metadata, "task_meta", "data_source")
        or prompt.get("data_source")
        or _nested_get(prompt, "task_meta", "data_source")
        or task_meta.get("data_source")
    )

    task_path = str(
        metadata.get("task_path")
        or _nested_get(metadata, "task_meta", "task_path")
        or prompt.get("task_path")
        or _nested_get(prompt, "task_meta", "task_path")
        or task_meta.get("task_path")
        or ""
    )
    if not raw:
        if task_path.startswith("agent_safetybench/"):
            raw = "agent_safetybench"
        elif task_path.startswith("seta_env/") or "seta" in task_path:
            raw = "seta"
        elif task_path:
            raw = task_path.split("/", 1)[0]

    return _sanitize_metric_part(raw)


def _reward_details(sample: Sample) -> dict[str, Any]:
    metadata = _as_dict(getattr(sample, "metadata", None))
    details = metadata.get("reward_details")
    return details if isinstance(details, dict) else {}


def _safety_split(sample: Sample) -> str:
    metadata = _as_dict(getattr(sample, "metadata", None))
    raw_split = metadata.get("safety_split")
    if raw_split:
        return _sanitize_metric_part(raw_split)

    task_meta = _task_meta_from_sample(sample)
    data_source = _dataset_name(sample)
    if data_source not in {"agent_safetybench", "agentharm"}:
        return "agentic"

    raw = task_meta.get("fulfillable")
    try:
        fulfillable = int(raw)
    except (TypeError, ValueError):
        task_type = str(task_meta.get("agentharm_task_type") or "").lower()
        fulfillable = 1 if task_type == "benign" else 0
    return "benign_should_comply" if fulfillable == 1 else "harmful_should_refuse"


def _bool_detail(sample: Sample, key: str) -> bool | None:
    details = _reward_details(sample)
    value = details.get(key)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _reward_value(sample: Sample, key: str = "score") -> float | None:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        return _to_float(reward.get(key))
    if key in ("score", "reward"):
        return _to_float(reward)
    return None


def _reward_raw(sample: Sample, key: str) -> Any:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        return reward.get(key)
    return None


def _reward_bool(sample: Sample, key: str) -> bool | None:
    value = _reward_raw(sample, key)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _response_length(sample: Sample) -> float | None:
    value = getattr(sample, "effective_response_length", None)
    if value is None:
        value = getattr(sample, "response_length", None)
    return _to_float(value)


def _status_name(sample: Sample) -> str:
    status = getattr(sample, "status", None)
    if isinstance(status, Sample.Status):
        return status.value
    return str(status or "unknown").lower()


def _mean_token_logprob(sample: Sample) -> float | None:
    values = getattr(sample, "rollout_log_probs", None)
    if not values:
        return None
    nums = [_to_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _stats(values: List[float]) -> dict[str, float] | None:
    nums = []
    for value in values:
        num = _to_float(value)
        if num is not None:
            nums.append(num)
    if not nums:
        return None
    count = len(nums)
    mean = sum(nums) / count
    variance = sum((x - mean) ** 2 for x in nums) / count
    sorted_nums = sorted(nums)

    def percentile(pct: float) -> float:
        if count == 1:
            return sorted_nums[0]
        idx = (count - 1) * pct
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return sorted_nums[lo]
        weight = idx - lo
        return sorted_nums[lo] * (1.0 - weight) + sorted_nums[hi] * weight

    return {
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": min(nums),
        "max": max(nums),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
    }


def _add_stats(
    log_dict: Dict[str, Any],
    prefix: str,
    values: List[float],
    *,
    include_percentiles: bool = False,
) -> dict[str, float] | None:
    stats = _stats(values)
    if not stats:
        return None
    keys = (
        ("mean", "std", "min", "max", "p50", "p90")
        if include_percentiles
        else ("mean", "std", "min", "max")
    )
    for key in keys:
        log_dict[f"{prefix}/{key}"] = stats[key]
    return stats


def _add_exploration_debug_metrics(
    log_dict: Dict[str, Any],
    prefix: str,
    samples: List[Sample],
) -> dict[str, Any]:
    """Add structured exploration/exploitation health metrics.

    The "mood" fields are intentionally coarse: they make live logs scannable
    during mixed training without replacing the lower-level numeric components.
    """
    source = [s for s in samples if not getattr(s, "remove_sample", False)] or samples
    summary: dict[str, Any] = {}
    if not source:
        return summary

    numeric_keys = (
        "explore_total_bonus",
        "explore_base_score_before_bonus",
        "explore_bonus_to_base_abs_ratio",
        "explore_curiosity_pressure",
        "explore_tool_intrinsic_pressure",
        "explore_safety_pressure",
        "explore_action_count",
        "explore_tool_call_count",
        "explore_danger_command_count",
        "explore_parse_error_count",
        "explore_cde_actor_log_ppl",
        "explore_lprnd_raw",
    )
    for key in numeric_keys:
        values = [v for v in (_reward_value(s, key) for s in source) if v is not None]
        if values:
            stats = _add_stats(log_dict, f"{prefix}/explore/{key}", values)
            if stats and key in {
                "explore_total_bonus",
                "explore_bonus_to_base_abs_ratio",
                "explore_curiosity_pressure",
                "explore_safety_pressure",
            }:
                summary[f"{key}_mean"] = stats["mean"]

    for key in (
        "explore_reward_hacking_risk",
        "explore_over_exploration_risk",
        "explore_safety_tension",
    ):
        values = [v for v in (_reward_bool(s, key) for s in source) if v is not None]
        if values:
            rate = sum(1 for v in values if v) / len(values)
            log_dict[f"{prefix}/explore/{key}_rate"] = rate
            summary[f"{key}_rate"] = rate

    mood_counts: dict[str, int] = defaultdict(int)
    for sample in source:
        mood = _reward_raw(sample, "explore_mood")
        if mood:
            mood_counts[_sanitize_metric_part(mood)] += 1
    if mood_counts:
        total = sum(mood_counts.values())
        top_mood, top_count = max(mood_counts.items(), key=lambda item: item[1])
        summary["top_mood"] = top_mood
        summary["top_mood_ratio"] = top_count / total if total else 0.0
        for mood, count in sorted(mood_counts.items()):
            log_dict[f"{prefix}/explore/mood/{mood}"] = count
            log_dict[f"{prefix}/explore/mood_ratio/{mood}"] = count / total if total else 0.0

    return summary


def _stats_mean(stats: dict[str, float] | None) -> float | None:
    return stats["mean"] if stats else None


def _env_enabled(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _structured_metrics_path() -> Path | None:
    configured = os.getenv("TERMINAL_METRICS_JSONL")
    if configured:
        return Path(configured)
    run_dir = os.getenv("RUN_DIR")
    if run_dir:
        return Path(run_dir) / "logs" / "metrics.jsonl"
    return None


def _write_structured_metrics(records: list[dict[str, Any]]) -> None:
    if not _env_enabled("TERMINAL_STRUCTURED_METRICS", "1"):
        return
    if not records:
        return

    lines = [
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for record in records
    ]
    for text in lines:
        logger.info("%s %s", _STRUCTURED_LOG_PREFIX, text)

    path = _structured_metrics_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for text in lines:
                f.write(text)
                f.write("\n")
    except Exception as e:
        logger.warning("Failed to write structured rollout metrics to %s: %s", path, e)


def _run_name() -> str | None:
    if os.getenv("RUN_ID"):
        return os.getenv("RUN_ID")
    if os.getenv("RUN_NAME"):
        return os.getenv("RUN_NAME")
    if os.getenv("RUN_DIR"):
        return Path(os.getenv("RUN_DIR", "")).name
    return None


def _epoch(args: Any) -> int | float | None:
    for key in ("epoch", "data_epoch", "train_epoch"):
        value = getattr(args, key, None)
        num = _to_float(value)
        if num is not None:
            return int(num) if float(num).is_integer() else num
    return None


def _analysis_dataset_name_from_raw(raw_name: Any) -> str:
    name = _sanitize_metric_part(raw_name)
    if name in {"terminal_bench", "seta", "seta_env"}:
        return "seta"
    if name.startswith("seta_"):
        return "seta"
    if name in {
        "agent_safetybench",
        "asb",
        "agentharm",
        "safety",
        "security",
        "mcpsafety",
        "harmbench",
    }:
        return "security"
    if name.startswith("agent_safetybench") or name.startswith("agentharm") or name.startswith("safety"):
        return "security"
    return name


def _analysis_dataset_name(sample: Sample) -> str:
    return _analysis_dataset_name_from_raw(_dataset_name(sample))


def _mean_from_samples(samples: list[Sample], key: str) -> float | None:
    values = [v for v in (_reward_value(s, key) for s in samples) if v is not None]
    stats = _stats(values)
    return _stats_mean(stats)


def _metric_record_from_samples(
    *,
    args: Any,
    phase: str,
    dataset_name: str,
    source_datasets: list[str],
    rollout_id: int,
    step: int,
    samples: list[Sample],
    rollout_time: float | None = None,
    kl: float | None = None,
    entropy: float | None = None,
) -> dict[str, Any]:
    trainable = [s for s in samples if not getattr(s, "remove_sample", False)]
    reward_source = trainable or samples
    total_values = [v for v in (_reward_value(s, "score") for s in reward_source) if v is not None]
    total_stats = _stats(total_values)
    raw_score = _mean_from_samples(reward_source, "raw_score")
    task_reward = _mean_from_samples(reward_source, "base_score")
    if task_reward is None:
        task_reward = raw_score
    if task_reward is None:
        task_reward = _stats_mean(total_stats)

    exploration_values = [
        v for v in (_reward_value(s, "explore_total_bonus") for s in reward_source)
        if v is not None
    ]
    exploration_stats = _stats(exploration_values)
    exploration_reward = _stats_mean(exploration_stats)
    denom = None
    if task_reward is not None and exploration_reward is not None:
        denom = task_reward + exploration_reward

    status_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        status_counts[_status_name(sample)] += 1

    # Exploration reward is additive reward shaping produced in generate.py:
    # count/signature novelty, optional LP-RND logprob surprise, optional CDE
    # actor/PPL bonus, and optional safety penalties are accumulated as
    # `explore_total_bonus`. Baseline runs keep the field absent, so it is
    # reported as null rather than omitted.
    return {
        "schema": _STRUCTURED_SCHEMA,
        "run": _run_name(),
        "phase": phase,
        "dataset": dataset_name,
        "source_datasets": sorted(set(source_datasets)),
        "global_step": int(step),
        "epoch": _epoch(args),
        "rollout_id": int(rollout_id),
        "sample_count": len(samples),
        "trainable_count": len(trainable),
        "completed": int(status_counts.get(Sample.Status.COMPLETED.value, 0)),
        "truncated": int(status_counts.get(Sample.Status.TRUNCATED.value, 0)),
        "failed": int(status_counts.get(Sample.Status.FAILED.value, 0)),
        "aborted": int(status_counts.get(Sample.Status.ABORTED.value, 0)),
        "reward/total": _stats_mean(total_stats),
        "reward/task": task_reward,
        "reward/raw": raw_score,
        "reward/exploration": exploration_reward,
        "reward/exploration_ratio": (
            exploration_reward / denom if denom and abs(denom) > 1e-12 else None
        ),
        "test_acc": _mean_from_samples(reward_source, "accuracy"),
        "reward_std": total_stats["std"] if total_stats else None,
        "response_length": _stats_mean(
            _stats([v for v in (_response_length(s) for s in samples) if v is not None])
        ),
        "kl": kl,
        "entropy": entropy,
        "rollout_time_sec": rollout_time,
    }


def _metric_record_from_rewards(
    *,
    args: Any,
    phase: str,
    dataset_name: str,
    rollout_id: int,
    step: int,
    rewards: list[Any],
    kl: float | None = None,
    entropy: float | None = None,
) -> dict[str, Any]:
    reward_values = [_to_float(v) for v in rewards]
    reward_values = [v for v in reward_values if v is not None]
    stats = _stats(reward_values)
    return {
        "schema": _STRUCTURED_SCHEMA,
        "run": _run_name(),
        "phase": phase,
        "dataset": dataset_name,
        "source_datasets": [dataset_name],
        "global_step": int(step),
        "epoch": _epoch(args),
        "rollout_id": int(rollout_id),
        "sample_count": len(rewards),
        "trainable_count": len(rewards),
        "completed": None,
        "truncated": None,
        "failed": None,
        "aborted": None,
        "reward/total": _stats_mean(stats),
        "reward/task": _stats_mean(stats),
        "reward/raw": _stats_mean(stats),
        "reward/exploration": None,
        "reward/exploration_ratio": None,
        "test_acc": _stats_mean(stats),
        "reward_std": stats["std"] if stats else None,
        "response_length": None,
        "kl": kl,
        "entropy": entropy,
        "rollout_time_sec": None,
    }


def _aggregate_metric_records(
    *,
    args: Any,
    phase: str,
    rollout_id: int,
    step: int,
    samples: list[Sample],
    rollout_time: float | None = None,
    kl: float | None = None,
    entropy: float | None = None,
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[Sample]] = defaultdict(list)
    source_by_dataset: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        raw_source = _dataset_name(sample)
        dataset_name = _analysis_dataset_name_from_raw(raw_source)
        by_dataset[dataset_name].append(sample)
        source_by_dataset[dataset_name].add(raw_source)

    records = [
        _metric_record_from_samples(
            args=args,
            phase=phase,
            dataset_name=dataset_name,
            source_datasets=sorted(source_by_dataset[dataset_name]),
            rollout_id=rollout_id,
            step=step,
            samples=by_dataset[dataset_name],
            rollout_time=rollout_time,
            kl=kl,
            entropy=entropy,
        )
        for dataset_name in sorted(by_dataset)
    ]

    if len(records) > 1:
        records.append(
            _metric_record_from_samples(
                args=args,
                phase=phase,
                dataset_name="mixed-all",
                source_datasets=sorted({_dataset_name(s) for s in samples}),
                rollout_id=rollout_id,
                step=step,
                samples=samples,
                rollout_time=rollout_time,
                kl=kl,
                entropy=entropy,
            )
        )
    return records


def _add_per_dataset_log_dict(log_dict: Dict[str, Any], records: list[dict[str, Any]]) -> None:
    for record in records:
        dataset = _sanitize_metric_part(record["dataset"])
        for key in (
            "reward/total",
            "reward/task",
            "reward/raw",
            "reward/exploration",
            "reward/exploration_ratio",
            "test_acc",
            "reward_std",
            "response_length",
            "kl",
            "entropy",
        ):
            value = record.get(key)
            if value is not None:
                log_dict[f"per_dataset/{dataset}/{key}"] = value
        log_dict[f"per_dataset/{dataset}/sample_count"] = record.get("sample_count", 0)
        log_dict[f"per_dataset/{dataset}/trainable_count"] = record.get("trainable_count", 0)


def _format_per_dataset_table(
    records: list[dict[str, Any]],
    *,
    phase: str,
    step: int,
) -> str:
    if not records:
        return ""
    run = _run_name() or "-"
    epoch = records[0].get("epoch")
    title = f"========== step {step} | epoch {epoch if epoch is not None else '-'} | phase: {phase} | run: {run} =========="
    header = (
        "dataset          | reward/total | reward/task | reward/explore | test_acc | "
        "resp_len | kl     | entropy"
    )
    sep = "-----------------+--------------+-------------+----------------+----------+----------+--------+--------"
    body = []
    for record in records:
        body.append(
            f"{str(record['dataset'])[:16]:16} | "
            f"{_format_float(record.get('reward/total'), width=12)} | "
            f"{_format_float(record.get('reward/task'), width=11)} | "
            f"{_format_float(record.get('reward/exploration'), width=14)} | "
            f"{_format_float(record.get('test_acc'), width=8)} | "
            f"{_format_float(record.get('response_length'), width=8)} | "
            f"{_format_float(record.get('kl'), width=6)} | "
            f"{_format_float(record.get('entropy'), width=6)}"
        )
    return "\n".join([title, header, sep, *body, "=" * len(title)])


def _format_eval_diag(records: list[dict[str, Any]]) -> str:
    eval_records = [r for r in records if r.get("dataset") != "mixed-all"]
    if not eval_records:
        return ""

    lines = ["[DIAG] per-dataset eval reward movement"]
    task_by_dataset: dict[str, float] = {}
    explore_ratios: list[float] = []
    conclusions: list[str] = []

    for record in eval_records:
        dataset = str(record["dataset"])
        task = _to_float(record.get("reward/task"))
        explore = _to_float(record.get("reward/exploration"))
        ratio = _to_float(record.get("reward/exploration_ratio"))
        if task is not None:
            task_by_dataset[dataset] = task
        if ratio is not None:
            explore_ratios.append(ratio)

        prev = _LAST_EVAL_BY_DATASET.get(dataset)
        delta = None
        if prev is not None and task is not None:
            prev_task = _to_float(prev.get("reward/task"))
            if prev_task is not None:
                delta = task - prev_task

        lines.append(
            f"[DIAG] dataset={dataset} delta_reward/task={_format_float(delta)} "
            f"explore_ratio={_format_float(ratio)}"
        )
        if delta is not None and abs(delta) <= 0.01 and ratio is not None and ratio > 0.2:
            conclusions.append(f"{dataset}: 疑似只在探索没在学习")
        baseline_delta = _recent_eval_task_delta_from_jsonl(
            os.getenv("TERMINAL_BASELINE_METRICS_JSONL"),
            dataset,
        )
        if baseline_delta is not None:
            lines.append(
                f"[DIAG] dataset={dataset} baseline_delta_reward/task="
                f"{_format_float(baseline_delta)}"
            )
            if baseline_delta > 0.01 and (delta is None or delta <= 0.01):
                conclusions.append(f"{dataset}: baseline 涨而 exploration 不涨, 疑似探索奖励干扰主任务")

    if "seta" in task_by_dataset and "security" in task_by_dataset:
        gap = task_by_dataset["seta"] - task_by_dataset["security"]
        lines.append(f"[DIAG] task_reward_gap_seta_minus_security={gap:.4f}")
    else:
        lines.append("[DIAG] task_reward_gap_seta_minus_security=NA")

    if not conclusions:
        if explore_ratios and max(explore_ratios) > 0.5:
            conclusions.append("探索奖励占比偏高,需结合下一次 eval 的 task reward 变化判断")
        else:
            conclusions.append("未触发探索压过任务学习的启发式告警")
    lines.append("[DIAG] conclusion=" + "；".join(conclusions))

    for record in eval_records:
        _LAST_EVAL_BY_DATASET[str(record["dataset"])] = dict(record)
    return "\n".join(lines)


def _recent_eval_task_delta_from_jsonl(path_raw: str | None, dataset: str) -> float | None:
    if not path_raw:
        return None
    path = Path(path_raw)
    if not path.exists():
        return None

    recent: list[float] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("schema") != _STRUCTURED_SCHEMA
                    or record.get("phase") != "eval"
                    or record.get("dataset") != dataset
                ):
                    continue
                task = _to_float(record.get("reward/task"))
                if task is None:
                    continue
                recent.append(task)
                if len(recent) > 2:
                    recent = recent[-2:]
    except Exception as e:
        logger.warning("Failed to read baseline metrics from %s: %s", path, e)
        return None

    if len(recent) < 2:
        return None
    return recent[-1] - recent[-2]


def _format_float(value: Any, width: int = 8) -> str:
    num = _to_float(value)
    if num is None:
        return " " * (width - 1) + "-"
    return f"{num:{width}.3f}"


def _dataset_metrics(
    samples: List[Sample],
) -> tuple[Dict[str, Any], List[dict[str, Any]], List[dict[str, Any]]]:
    log_dict: Dict[str, Any] = {}
    rows: List[dict[str, Any]] = []
    split_rows: List[dict[str, Any]] = []
    total = len(samples)
    by_dataset: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_dataset[_dataset_name(sample)].append(sample)

    for dataset_name in sorted(by_dataset):
        dataset_samples = by_dataset[dataset_name]
        trainable = [s for s in dataset_samples if not getattr(s, "remove_sample", False)]
        prefix = f"terminal/dataset/{dataset_name}"

        count = len(dataset_samples)
        trainable_count = len(trainable)
        ratio = count / total if total else 0.0
        log_dict[f"{prefix}/sample_count"] = count
        log_dict[f"{prefix}/sample_ratio"] = ratio
        log_dict[f"{prefix}/trainable_count"] = trainable_count
        log_dict[f"{prefix}/trainable_ratio"] = trainable_count / count if count else 0.0

        status_counts = {status.value: 0 for status in Sample.Status}
        status_counts["unknown"] = 0
        for sample in dataset_samples:
            status = _status_name(sample)
            status_counts[status] = status_counts.get(status, 0) + 1
        for status, status_count in sorted(status_counts.items()):
            if status_count:
                log_dict[f"{prefix}/status/{status}"] = status_count
                log_dict[f"{prefix}/status_ratio/{status}"] = status_count / count if count else 0.0

        reward_stats = _add_stats(
            log_dict,
            f"{prefix}/reward",
            [v for v in (_reward_value(s, "score") for s in trainable) if v is not None],
        )

        acc_stats = _add_stats(
            log_dict,
            f"{prefix}/acc",
            [v for v in (_reward_value(s, "accuracy") for s in trainable) if v is not None],
        )

        response_stats = _add_stats(
            log_dict,
            f"{prefix}/response_length",
            [v for v in (_response_length(s) for s in dataset_samples) if v is not None],
            include_percentiles=True,
        )

        component_stats: dict[str, dict[str, float]] = {}
        for reward_key in _REWARD_COMPONENT_KEYS:
            values = [v for v in (_reward_value(s, reward_key) for s in trainable) if v is not None]
            if values:
                stats = _add_stats(log_dict, f"{prefix}/reward_component/{reward_key}", values)
                if stats:
                    component_stats[reward_key] = stats

        explore_summary = _add_exploration_debug_metrics(log_dict, prefix, dataset_samples)

        safety_values = [v for v in (_reward_value(s, "safety_score") for s in trainable) if v is not None]
        if safety_values:
            log_dict[f"{prefix}/safety_negative_ratio"] = sum(1 for x in safety_values if x < 0) / len(
                safety_values
            )
            safety_coef = next(
                (coef for coef in (_reward_value(s, "safety_coef") for s in trainable) if coef is not None),
                None,
            )
            if safety_coef is not None:
                log_dict[f"{prefix}/safety_coef"] = safety_coef

        mean_logprobs = [v for v in (_mean_token_logprob(s) for s in dataset_samples) if v is not None]
        if mean_logprobs:
            _add_stats(log_dict, f"{prefix}/rollout_logprob", mean_logprobs)
            _add_stats(log_dict, f"{prefix}/rollout_neg_logprob", [-x for x in mean_logprobs])

        n_cs_calls = 0
        n_cs_errors = 0
        for sample in dataset_samples:
            safety_meta = _as_dict(_as_dict(getattr(sample, "metadata", None)).get("safety"))
            n_cs_calls += int(safety_meta.get("n_calls", 0) or 0)
            n_cs_errors += int(safety_meta.get("n_errors", 0) or 0)
        if n_cs_calls > 0:
            log_dict[f"{prefix}/clawsentry_calls_total"] = n_cs_calls
            log_dict[f"{prefix}/clawsentry_errors_total"] = n_cs_errors
            log_dict[f"{prefix}/clawsentry_error_rate"] = n_cs_errors / n_cs_calls

        reason_counts: dict[str, int] = defaultdict(int)
        for sample in trainable:
            reason = _reward_details(sample).get("reason")
            if reason:
                reason_counts[_sanitize_metric_part(reason)] += 1
        for reason, reason_count in sorted(reason_counts.items()):
            log_dict[f"{prefix}/reward_reason/{reason}"] = reason_count
            log_dict[f"{prefix}/reward_reason_ratio/{reason}"] = (
                reason_count / trainable_count if trainable_count else 0.0
            )

        by_split: dict[str, list[Sample]] = defaultdict(list)
        for sample in dataset_samples:
            by_split[_safety_split(sample)].append(sample)
        for split_name in sorted(by_split):
            split_samples = by_split[split_name]
            split_trainable = [s for s in split_samples if not getattr(s, "remove_sample", False)]
            split_prefix = f"{prefix}/split/{split_name}"
            split_count = len(split_samples)
            split_trainable_count = len(split_trainable)
            log_dict[f"{split_prefix}/sample_count"] = split_count
            log_dict[f"{split_prefix}/sample_ratio"] = split_count / total if total else 0.0
            log_dict[f"{split_prefix}/dataset_ratio"] = split_count / count if count else 0.0
            log_dict[f"{split_prefix}/trainable_count"] = split_trainable_count

            split_reward_stats = _add_stats(
                log_dict,
                f"{split_prefix}/reward",
                [v for v in (_reward_value(s, "score") for s in split_trainable) if v is not None],
            )
            split_acc_stats = _add_stats(
                log_dict,
                f"{split_prefix}/acc",
                [v for v in (_reward_value(s, "accuracy") for s in split_trainable) if v is not None],
            )
            split_component_stats: dict[str, dict[str, float]] = {}
            for reward_key in (
                "raw_score",
                "base_score",
                "safety_score",
                "explore_total_bonus",
                "explore_intrinsic_scaled",
                "explore_safety_penalty",
            ):
                values = [
                    v for v in (_reward_value(s, reward_key) for s in split_trainable)
                    if v is not None
                ]
                if values:
                    stats = _add_stats(log_dict, f"{split_prefix}/reward_component/{reward_key}", values)
                    if stats:
                        split_component_stats[reward_key] = stats
            _add_stats(
                log_dict,
                f"{split_prefix}/response_length",
                [v for v in (_response_length(s) for s in split_samples) if v is not None],
                include_percentiles=True,
            )

            split_reason_counts: dict[str, int] = defaultdict(int)
            for sample in split_trainable:
                reason = _reward_details(sample).get("reason")
                if reason:
                    split_reason_counts[_sanitize_metric_part(reason)] += 1
            top_reason = "-"
            top_reason_count = 0
            if split_reason_counts:
                top_reason, top_reason_count = max(
                    split_reason_counts.items(), key=lambda item: item[1]
                )
            for reason, reason_count in sorted(split_reason_counts.items()):
                log_dict[f"{split_prefix}/reward_reason/{reason}"] = reason_count
                log_dict[f"{split_prefix}/reward_reason_ratio/{reason}"] = (
                    reason_count / split_trainable_count if split_trainable_count else 0.0
                )

            def bool_rate(detail_key: str) -> float | None:
                vals = [
                    value
                    for value in (_bool_detail(s, detail_key) for s in split_trainable)
                    if value is not None
                ]
                if not vals:
                    return None
                rate = sum(1 for value in vals if value) / len(vals)
                log_dict[f"{split_prefix}/{detail_key}_rate"] = rate
                return rate

            verbal_refused_rate = bool_rate("verbal_refused")
            attempted_tool_rate = bool_rate("attempted_tool_use")
            nonempty_rate = bool_rate("nonempty_response")
            bool_rate("truncation_cap_applied")
            bool_rate("safe_truncated_refusal")
            if nonempty_rate is not None:
                log_dict[f"{split_prefix}/empty_response_rate"] = 1.0 - nonempty_rate

            for detail_key in _REWARD_DETAIL_NUMERIC_KEYS:
                values = [
                    v
                    for v in (
                        _to_float(_reward_details(s).get(detail_key))
                        for s in split_trainable
                    )
                    if v is not None
                ]
                if values:
                    _add_stats(log_dict, f"{split_prefix}/detail/{detail_key}", values)

            split_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "count": split_count,
                    "ratio": split_count / total if total else 0.0,
                    "trainable": split_trainable_count,
                    "reward_mean": split_reward_stats["mean"] if split_reward_stats else None,
                    "test_acc_mean": split_acc_stats["mean"] if split_acc_stats else None,
                    "acc_mean": split_acc_stats["mean"] if split_acc_stats else None,
                    "raw_score_mean": _stats_mean(split_component_stats.get("raw_score")),
                    "base_reward_mean": _stats_mean(split_component_stats.get("base_score")),
                    "safety_score_mean": _stats_mean(split_component_stats.get("safety_score")),
                    "exploration_reward_mean": _stats_mean(
                        split_component_stats.get("explore_total_bonus")
                    ),
                    "explore_intrinsic_scaled_mean": _stats_mean(
                        split_component_stats.get("explore_intrinsic_scaled")
                    ),
                    "explore_safety_penalty_mean": _stats_mean(
                        split_component_stats.get("explore_safety_penalty")
                    ),
                    "verbal_refused_rate": verbal_refused_rate,
                    "attempted_tool_rate": attempted_tool_rate,
                    "empty_response_rate": (1.0 - nonempty_rate) if nonempty_rate is not None else None,
                    "top_reason": top_reason,
                    "top_reason_ratio": (
                        top_reason_count / split_trainable_count
                        if split_trainable_count
                        else None
                    ),
                }
            )

        rows.append(
            {
                "dataset": dataset_name,
                "count": count,
                "ratio": ratio,
                "trainable": trainable_count,
                "reward_mean": _stats_mean(reward_stats),
                "reward_std": reward_stats["std"] if reward_stats else None,
                "test_acc_mean": _stats_mean(acc_stats),
                "acc_mean": _stats_mean(acc_stats),
                "raw_score_mean": _stats_mean(component_stats.get("raw_score")),
                "base_reward_mean": _stats_mean(component_stats.get("base_score")),
                "safety_score_mean": _stats_mean(component_stats.get("safety_score")),
                "exploration_reward_mean": _stats_mean(component_stats.get("explore_total_bonus")),
                "explore_intrinsic_scaled_mean": _stats_mean(
                    component_stats.get("explore_intrinsic_scaled")
                ),
                "explore_safety_penalty_mean": _stats_mean(
                    component_stats.get("explore_safety_penalty")
                ),
                "explore_lprnd_mean": _stats_mean(component_stats.get("explore_lprnd")),
                "explore_cde_actor_bonus_mean": _stats_mean(
                    component_stats.get("explore_cde_actor_bonus")
                ),
                "response_mean": response_stats["mean"] if response_stats else None,
                "completed": status_counts.get(Sample.Status.COMPLETED.value, 0),
                "truncated": status_counts.get(Sample.Status.TRUNCATED.value, 0),
                "failed": status_counts.get(Sample.Status.FAILED.value, 0),
                "aborted": status_counts.get(Sample.Status.ABORTED.value, 0),
                "explore_mood": explore_summary.get("top_mood"),
                "explore_pressure": explore_summary.get("explore_bonus_to_base_abs_ratio_mean"),
                "reward_hack_risk": explore_summary.get("explore_reward_hacking_risk_rate"),
            }
        )

    return log_dict, rows, split_rows


def _format_dataset_table(rows: List[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = (
        "dataset                 n  ratio train  rew_mean  rew_std      acc resp_len  "
        "comp trunc fail abort mood              xpress rhack"
    )
    line = "-" * len(header)
    body = []
    for row in rows:
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{int(row['count']):4d} "
            f"{row['ratio']:6.2%} "
            f"{int(row['trainable']):5d} "
            f"{_format_float(row['reward_mean'])} "
            f"{_format_float(row['reward_std'])} "
            f"{_format_float(row['acc_mean'])} "
            f"{_format_float(row['response_mean'])} "
            f"{int(row['completed']):5d} "
            f"{int(row['truncated']):5d} "
            f"{int(row['failed']):4d} "
            f"{int(row['aborted']):5d} "
            f"{str(row.get('explore_mood') or '-')[:16]:16} "
            f"{_format_float(row.get('explore_pressure'), width=6)} "
            f"{_format_float(row.get('reward_hack_risk'), width=5)}"
        )
    return "\n".join([header, line, *body])


def _format_reward_breakdown_table(rows: List[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = (
        "dataset                 n train final_reward test_acc raw_score base_reward "
        "safety explore_reward intrinsic safepen   cde lprnd"
    )
    line = "-" * len(header)
    body = []
    for row in rows:
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{int(row['count']):4d} "
            f"{int(row['trainable']):5d} "
            f"{_format_float(row.get('reward_mean'), width=12)} "
            f"{_format_float(row.get('test_acc_mean'), width=8)} "
            f"{_format_float(row.get('raw_score_mean'), width=9)} "
            f"{_format_float(row.get('base_reward_mean'), width=11)} "
            f"{_format_float(row.get('safety_score_mean'), width=6)} "
            f"{_format_float(row.get('exploration_reward_mean'), width=14)} "
            f"{_format_float(row.get('explore_intrinsic_scaled_mean'), width=9)} "
            f"{_format_float(row.get('explore_safety_penalty_mean'), width=7)} "
            f"{_format_float(row.get('explore_cde_actor_bonus_mean'), width=5)} "
            f"{_format_float(row.get('explore_lprnd_mean'), width=5)}"
        )
    return "\n".join([header, line, *body])


def _format_split_table(rows: List[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = (
        "dataset                 split                    n  ratio train  rew_mean      acc "
        "refuse   tools   empty top_reason"
    )
    line = "-" * len(header)
    body = []
    for row in rows:
        top_reason = str(row.get("top_reason") or "-")
        top_ratio = _to_float(row.get("top_reason_ratio"))
        if top_ratio is not None and top_reason != "-":
            top_reason = f"{top_reason[:24]}:{top_ratio:.0%}"
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{str(row['split'])[:24]:24} "
            f"{int(row['count']):4d} "
            f"{row['ratio']:6.2%} "
            f"{int(row['trainable']):5d} "
            f"{_format_float(row['reward_mean'])} "
            f"{_format_float(row['acc_mean'])} "
            f"{_format_float(row['verbal_refused_rate'])} "
            f"{_format_float(row['attempted_tool_rate'])} "
            f"{_format_float(row['empty_response_rate'])} "
            f"{top_reason}"
        )
    return "\n".join([header, line, *body])


def rollout_log(rollout_id, args, samples, rollout_extra_metrics, rollout_time):

    trainable = [s for s in samples if not getattr(s, "remove_sample", False)]
    non_trainable = [s for s in samples if getattr(s, "remove_sample", False)]

    log_dict: Dict[str, Any] = {}

    total = len(samples)
    n_failed = sum(1 for s in samples if s.status == Sample.Status.FAILED)
    n_aborted = sum(1 for s in samples if s.status == Sample.Status.ABORTED)
    n_truncated = sum(1 for s in samples if s.status == Sample.Status.TRUNCATED)
    n_completed = sum(1 for s in samples if s.status == Sample.Status.COMPLETED)

    log_dict["terminal/total_samples"] = total
    log_dict["terminal/completed"] = n_completed
    log_dict["terminal/truncated"] = n_truncated
    log_dict["terminal/failed"] = n_failed
    log_dict["terminal/aborted"] = n_aborted
    log_dict["terminal/failed_ratio"] = n_failed / total if total else 0.0
    log_dict["terminal/non_trainable_ratio"] = (
        len(non_trainable) / total if total else 0.0
    )

    if trainable:
        trainable_rewards = [
            v for v in (_reward_value(s, "score") for s in trainable) if v is not None
        ]
        log_dict["terminal/reward_mean"] = sum(trainable_rewards) / len(
            trainable_rewards
        ) if trainable_rewards else 0.0
        if trainable_rewards:
            reward_stats = _stats(trainable_rewards)
            log_dict["terminal/reward_std"] = reward_stats["std"]
            log_dict["terminal/reward_min"] = reward_stats["min"]
            log_dict["terminal/reward_max"] = reward_stats["max"]

        trainable_accs = []
        for s in trainable:
            if isinstance(s.reward, dict) and "accuracy" in s.reward:
                trainable_accs.append(float(s.reward["accuracy"]))
        if trainable_accs:
            log_dict["terminal/accuracy"] = sum(trainable_accs) / len(trainable_accs)

        trainable_prm = []
        for s in trainable:
            if isinstance(s.reward, dict) and "prm_turn_score" in s.reward:
                trainable_prm.append(float(s.reward["prm_turn_score"]))
        if trainable_prm:
            log_dict["terminal/prm_turn_score"] = sum(trainable_prm) / len(
                trainable_prm
            )

        trainable_safety = []
        trainable_safety_coef = None
        for s in trainable:
            if isinstance(s.reward, dict) and "safety_score" in s.reward:
                trainable_safety.append(float(s.reward["safety_score"]))
                if trainable_safety_coef is None:
                    trainable_safety_coef = float(s.reward.get("safety_coef", 0.0))
        if trainable_safety:
            n = len(trainable_safety)
            log_dict["terminal/safety_score_mean"] = sum(trainable_safety) / n
            log_dict["terminal/safety_score_min"] = min(trainable_safety)
            log_dict["terminal/safety_score_max"] = max(trainable_safety)
            log_dict["terminal/safety_negative_ratio"] = (
                sum(1 for x in trainable_safety if x < 0) / n
            )
            if trainable_safety_coef is not None:
                log_dict["terminal/safety_coef"] = trainable_safety_coef

    dataset_log_dict, dataset_rows, split_rows = _dataset_metrics(samples)
    log_dict.update(dataset_log_dict)
    _add_exploration_debug_metrics(log_dict, "terminal", samples)

    n_cs_calls = 0
    n_cs_errors = 0
    for s in samples:
        safety_meta = (s.metadata or {}).get("safety") if s.metadata else None
        if isinstance(safety_meta, dict):
            n_cs_calls += int(safety_meta.get("n_calls", 0) or 0)
            n_cs_errors += int(safety_meta.get("n_errors", 0) or 0)
    if n_cs_calls > 0:
        log_dict["terminal/clawsentry_calls_total"] = n_cs_calls
        log_dict["terminal/clawsentry_errors_total"] = n_cs_errors
        log_dict["terminal/clawsentry_error_rate"] = n_cs_errors / n_cs_calls

    log_dict["terminal/rollout_time"] = rollout_time

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    metric_records = _aggregate_metric_records(
        args=args,
        phase="train",
        rollout_id=rollout_id,
        step=step,
        samples=samples,
        rollout_time=rollout_time,
    )
    _add_per_dataset_log_dict(log_dict, metric_records)
    per_dataset_table = _format_per_dataset_table(metric_records, phase="train", step=step)
    if per_dataset_table:
        logger.info("per-dataset metrics\n%s", per_dataset_table)
    table = _format_dataset_table(dataset_rows)
    if table:
        logger.info("dataset metrics rollout=%s step=%s\n%s", rollout_id, step, table)
    reward_table = _format_reward_breakdown_table(dataset_rows)
    if reward_table:
        logger.info(
            "dataset reward breakdown rollout=%s step=%s\n%s",
            rollout_id,
            step,
            reward_table,
        )
    split_table = _format_split_table(split_rows)
    if split_table:
        logger.info("dataset split metrics rollout=%s step=%s\n%s", rollout_id, step, split_table)
    _write_structured_metrics(metric_records)
    _ensure_terminal_step_metric(args)
    logging_utils.log(args, log_dict, step_key="rollout/step")

    return False


def eval_rollout_log(rollout_id, args, data, extra_metrics=None):
    """Per-dataset eval logging hook.

    The hook writes the same JSONL schema as training rollouts. It returns
    False so slime's default eval logger still emits its legacy metrics.
    """
    step = compute_rollout_step(args, rollout_id)
    records: list[dict[str, Any]] = []
    all_samples: list[Sample] = []
    all_rewards: list[Any] = []

    for raw_name, info in sorted((data or {}).items()):
        dataset_name = _analysis_dataset_name_from_raw(raw_name)
        samples = info.get("samples") if isinstance(info, dict) else None
        if samples:
            all_samples.extend(samples)
            records.append(
                _metric_record_from_samples(
                    args=args,
                    phase="eval",
                    dataset_name=dataset_name,
                    source_datasets=[str(raw_name)],
                    rollout_id=rollout_id,
                    step=step,
                    samples=samples,
                )
            )
            continue

        rewards = info.get("rewards", []) if isinstance(info, dict) else []
        all_rewards.extend(rewards)
        records.append(
            _metric_record_from_rewards(
                args=args,
                phase="eval",
                dataset_name=dataset_name,
                rollout_id=rollout_id,
                step=step,
                rewards=rewards,
            )
        )

    if len(records) > 1:
        if all_samples:
            records.append(
                _metric_record_from_samples(
                    args=args,
                    phase="eval",
                    dataset_name="mixed-all",
                    source_datasets=sorted({str(name) for name in (data or {}).keys()}),
                    rollout_id=rollout_id,
                    step=step,
                    samples=all_samples,
                )
            )
        else:
            records.append(
                _metric_record_from_rewards(
                    args=args,
                    phase="eval",
                    dataset_name="mixed-all",
                    rollout_id=rollout_id,
                    step=step,
                    rewards=all_rewards,
                )
            )

    log_dict: Dict[str, Any] = dict(extra_metrics or {})
    log_dict["eval/step"] = step
    # `per_dataset/*` is defined on rollout/step so train/eval points share the
    # same x-axis when inspecting dataset-specific curves in wandb.
    log_dict["rollout/step"] = step
    _add_per_dataset_log_dict(log_dict, records)

    table = _format_per_dataset_table(records, phase="eval", step=step)
    if table:
        logger.info("per-dataset eval metrics\n%s", table)
    diag = _format_eval_diag(records)
    if diag:
        logger.info("%s", diag)

    _write_structured_metrics(records)
    _ensure_terminal_step_metric(args)
    logging_utils.log(args, log_dict, step_key="eval/step")
    return False
