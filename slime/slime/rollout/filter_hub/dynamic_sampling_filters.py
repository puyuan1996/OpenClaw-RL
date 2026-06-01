import os

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std"]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _reward_component(sample: Sample, key: str) -> float:
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, dict):
        return 0.0
    value = reward.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _post_norm_bonus_for_filter(sample: Sample) -> float:
    if not _env_flag("EXPLORE_ADVANTAGE_BONUS_ENABLED", os.getenv("EXPLORE_ADVANTAGE_BONUS", "0")):
        return 0.0
    components = [
        part.strip()
        for part in os.getenv("EXPLORE_ADVANTAGE_BONUS_COMPONENTS", "explore_intrinsic_scaled").split(",")
        if part.strip()
    ]
    raw_bonus = sum(_reward_component(sample, key) for key in components)
    clip = _env_float("EXPLORE_ADVANTAGE_BONUS_CLIP", 0.25)
    coef = _env_float("EXPLORE_ADVANTAGE_BONUS_COEF", 1.0)
    clipped = max(-clip, min(clip, raw_bonus)) if clip > 0 else raw_bonus
    return coef * clipped


def _representative_sample(sample_or_turns) -> Sample | None:
    """Return one reward-bearing Sample per original completion.

    terminal-rl multi-turn generation returns one list of per-turn training
    samples for each original completion. Dynamic sampling should compare
    completions inside the prompt group, not flatten all per-turn samples into
    artificial extra completions.
    """
    if isinstance(sample_or_turns, Sample):
        return sample_or_turns
    if isinstance(sample_or_turns, list):
        for item in reversed(sample_or_turns):
            sample = _representative_sample(item)
            if sample is not None:
                return sample
    return None


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    representatives = [_representative_sample(sample) for sample in samples]
    representatives = [sample for sample in representatives if sample is not None]
    if not representatives:
        return DynamicFilterOutput(keep=False, reason="empty_reward_group")

    rewards = [
        sample.get_reward_value(args) + _post_norm_bonus_for_filter(sample)
        for sample in representatives
    ]
    keep = bool(torch.tensor(rewards, dtype=torch.float).std(unbiased=False) > 0.0)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
