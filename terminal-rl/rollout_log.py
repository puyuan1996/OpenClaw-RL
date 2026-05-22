from __future__ import annotations

import logging
from typing import Any, Dict, List

import wandb
from slime.utils import logging_utils
from slime.utils.types import Sample
from slime.ray.rollout import compute_rollout_step

logger = logging.getLogger(__name__)


def _ensure_terminal_step_metric(args) -> None:
    if not getattr(args, "use_wandb", False):
        return
    try:
        wandb.define_metric("terminal/*", step_metric="rollout/step")
    except Exception as e:
        logger.warning("Failed to define wandb step metric for terminal/*: %s", e)


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
        trainable_rewards = [s.reward["score"] for s in trainable]
        log_dict["terminal/reward_mean"] = sum(trainable_rewards) / len(
            trainable_rewards
        )
        log_dict["terminal/reward_min"] = min(trainable_rewards)
        log_dict["terminal/reward_max"] = max(trainable_rewards)

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
    _ensure_terminal_step_metric(args)
    logging_utils.log(args, log_dict, step_key="rollout/step")

    return False
