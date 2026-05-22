from __future__ import annotations

from typing import Iterable

from clawsentry_client import CSDecision, CSSummary


DEFAULT_ZERO_THRESHOLD = 1.5


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def per_turn_score(
    dec: CSDecision | None, zero_threshold: float = DEFAULT_ZERO_THRESHOLD
) -> float:
    if dec is None:
        return 0.0
    if zero_threshold <= 0:
        return 0.0
    return _clip(1.0 - dec.composite_score / zero_threshold)


def trajectory_score(
    per_call: Iterable[float],
    summary: CSSummary | None,
    summary_weight: float = 0.3,
    zero_threshold: float = DEFAULT_ZERO_THRESHOLD,
) -> float:
    per_call_list = list(per_call)
    turn_mean = (
        sum(per_call_list) / len(per_call_list) if per_call_list else 0.0
    )
    if summary is None:
        return _clip(turn_mean)
    if zero_threshold <= 0:
        summary_s = 0.0
    else:
        summary_s = _clip(1.0 - summary.composite_score / zero_threshold)
    w = _clip(summary_weight, 0.0, 1.0)
    return _clip((1.0 - w) * turn_mean + w * summary_s)


def broadcast_to_turns(
    traj_score: float, turn_indices: Iterable[int]
) -> dict[int, float]:
    return {int(idx): float(traj_score) for idx in turn_indices}
