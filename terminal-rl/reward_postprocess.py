from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.4f", name, raw, default)
        return default


def _reward_value(args: Any, sample: Any) -> float:
    reward = getattr(sample, "reward", None)
    key = getattr(args, "reward_key", None)
    if key:
        if not isinstance(reward, dict):
            return 0.0
        return float(reward.get(key, 0.0) or 0.0)
    return float(reward or 0.0)


def _component_value(sample: Any, key: str) -> float:
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, dict):
        return 0.0
    value = reward.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _normalize_values(values: list[float], use_std: bool) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    centered = [v - mean for v in values]
    if not use_std:
        return centered
    if len(values) <= 1:
        return [0.0 for _ in values]
    # Match torch.std default semantics used by slime: unbiased sample std.
    var = sum(v * v for v in centered) / max(1, len(values) - 1)
    std = math.sqrt(max(var, 0.0))
    return [v / (std + 1e-6) for v in centered]


def _default_post_process(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Mirror slime's default reward post-process for GRPO/GSPO.

    This function is only used when EXPLORE_ADVANTAGE_BONUS is enabled; keeping
    the default math here lets us add post-normalization exploration bonuses
    without replacing the rest of slime's behavior.
    """
    raw_rewards = [_reward_value(args, sample) for sample in samples]
    if (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]
        and getattr(args, "rewards_normalization", False)
    ):
        if getattr(args, "dynamic_history", False):
            traj_reward_by_key: dict[tuple[int, int], float] = {}
            group_to_keys: dict[int, list[tuple[int, int]]] = {}
            key_by_sample: list[tuple[int, int]] = []
            for i, sample in enumerate(samples):
                group_idx = int(sample.group_index) if sample.group_index is not None else -1
                traj_idx = int(sample.index) if sample.index is not None else i
                key = (group_idx, traj_idx)
                key_by_sample.append(key)
                if key not in traj_reward_by_key:
                    traj_reward_by_key[key] = float(raw_rewards[i])
                    group_to_keys.setdefault(group_idx, []).append(key)

            normalized_by_key: dict[tuple[int, int], float] = {}
            for keys in group_to_keys.values():
                vals = _normalize_values(
                    [traj_reward_by_key[k] for k in keys],
                    bool(getattr(args, "grpo_std_normalization", False)),
                )
                for j, key in enumerate(keys):
                    normalized_by_key[key] = float(vals[j])
            return raw_rewards, [normalized_by_key[key] for key in key_by_sample]

        group_to_indices: dict[int, list[int]] = {}
        for i, sample in enumerate(samples):
            group_idx = int(sample.group_index) if sample.group_index is not None else -1
            group_to_indices.setdefault(group_idx, []).append(i)

        rewards = list(raw_rewards)
        for idxs in group_to_indices.values():
            vals = _normalize_values(
                [raw_rewards[i] for i in idxs],
                bool(getattr(args, "grpo_std_normalization", False)),
            )
            for j, sample_idx in enumerate(idxs):
                rewards[sample_idx] = float(vals[j])
        return raw_rewards, rewards

    return raw_rewards, raw_rewards


def post_process_rewards(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    raw_rewards, rewards = _default_post_process(args, samples)
    if not _env_flag("EXPLORE_ADVANTAGE_BONUS_ENABLED", os.getenv("EXPLORE_ADVANTAGE_BONUS", "0")):
        return raw_rewards, rewards

    component_names = [
        part.strip()
        for part in os.getenv("EXPLORE_ADVANTAGE_BONUS_COMPONENTS", "explore_intrinsic_scaled").split(",")
        if part.strip()
    ]
    coef = _env_float("EXPLORE_ADVANTAGE_BONUS_COEF", 1.0)
    clip = _env_float("EXPLORE_ADVANTAGE_BONUS_CLIP", 0.25)

    adjusted = list(rewards)
    for i, sample in enumerate(samples):
        raw_bonus = sum(_component_value(sample, key) for key in component_names)
        clipped_bonus = max(-clip, min(clip, raw_bonus)) if clip > 0 else raw_bonus
        bonus = coef * clipped_bonus
        adjusted[i] += bonus
        reward = getattr(sample, "reward", None)
        if isinstance(reward, dict):
            reward["explore_post_norm_bonus_raw"] = raw_bonus
            reward["explore_post_norm_bonus"] = bonus
            reward["explore_post_norm_bonus_coef"] = coef
            reward["explore_post_norm_bonus_clip"] = clip
            reward["explore_post_norm_bonus_components"] = ",".join(component_names)
    return raw_rewards, adjusted
