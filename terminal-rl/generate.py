from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from agent.prm_agent import TerminalPRMAgent
from clawsentry_client import ClawSentryClient
from custom_types import (
    Interaction,
    RunContext,
    TaskSpec,
    TaskTimeouts,
    TurnContext,
    TurnResult,
)
from inference_client import SGLangTurnClient
from agent_runner import create_agent_runner
from env_client import TerminalEnvClient
from safety_reward import (
    DEFAULT_ZERO_THRESHOLD as _SAFETY_ZERO_THRESHOLD,
    broadcast_to_turns as _safety_broadcast,
    per_turn_score as _safety_per_turn_score,
    trajectory_score as _safety_trajectory_score,
)

logger = logging.getLogger(__name__)

_DIRECT_SCORE_DATA_SOURCES = {"agent_safetybench", "agentharm"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.4f", name, raw, default)
        return default


# ── Exploration: count-based intrinsic reward (MERCI simplified) ──────────────
_EXPLORE_INTRINSIC_ENABLED = _env_bool("EXPLORE_INTRINSIC_ENABLED", False)
_EXPLORE_INTRINSIC_COEF = _env_float("EXPLORE_INTRINSIC_COEF", 0.1)
_EXPLORE_INTRINSIC_SCHEDULE = os.getenv("EXPLORE_INTRINSIC_SCHEDULE", "constant").strip().lower()
_EXPLORE_INTRINSIC_DECAY_STEPS = _env_int("EXPLORE_INTRINSIC_DECAY_STEPS", 0)
# Granularity for novelty hashing:
#   "raw"        = full command string (default, matches v1)
#   "signature"  = tool-call signature (cmd name + first 2 args), Agent57-style
#                  sub-goal/skill granularity per the LaMer/Agent57 analysis.
_EXPLORE_INTRINSIC_GRANULARITY = os.getenv("EXPLORE_INTRINSIC_GRANULARITY", "raw").strip().lower()
_EXPLORE_INTRINSIC_SCOPE = os.getenv("EXPLORE_INTRINSIC_SCOPE", "process").strip().lower()
_CMD_COUNTER: Dict[str, int] = {}  # process-level counter for command novelty

# ── Exploration: LP-RND lifelong novelty (草案 C, zero-extra-param) ───────────
# Reuses the rollout_log_probs already computed by slime (no extra forward pass).
# Bonus is proportional to how surprised the *current* policy is by the trajectory:
# higher mean negative-logprob → more novel → larger bonus, clipped to [0, L].
# This is the LLM analog of RND: "how surprising is this trajectory under the
# current rollout policy?" implemented without maintaining a separate net.
_EXPLORE_LPRND_ENABLED = _env_bool("EXPLORE_LPRND_ENABLED", False)
_EXPLORE_LPRND_COEF = _env_float("EXPLORE_LPRND_COEF", 0.05)
_EXPLORE_LPRND_SCHEDULE = os.getenv("EXPLORE_LPRND_SCHEDULE", "constant").strip().lower()
_EXPLORE_LPRND_DECAY_STEPS = _env_int("EXPLORE_LPRND_DECAY_STEPS", 0)
_EXPLORE_LPRND_CLIP = _env_float("EXPLORE_LPRND_CLIP", 3.0)
_EXPLORE_LPRND_WARMUP = _env_int("EXPLORE_LPRND_WARMUP", 32)
# Running stats for normalization (process-level, updated online).
_LPRND_STATS = {"warmup": 0, "n": 0, "mean": 0.0, "m2": 0.0}

# ── Exploration: CDE actor curiosity bonus (RLVR PPL bonus) ──────────────────
# Optional actor-side Curiosity-Driven Exploration bonus:
#   B_actor(q,o) = -mean_t log pi(o_t | o_<t, q)
#   r_hat = r + omega * min(|r| / kappa, alpha * B_actor)
#
# The cap is based on the pre-exploration task reward magnitude. That keeps this
# as a supplement to verifiable rewards and prevents empty/infra-failed rollouts
# with score=0 from receiving curiosity reward.
_EXPLORE_CDE_ACTOR_ENABLED = (
    os.getenv("EXPLORE_CDE_ACTOR_ENABLED", os.getenv("EXPLORE_CDE_ACTOR", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
)
_EXPLORE_CDE_ACTOR_OMEGA = _env_float("EXPLORE_CDE_ACTOR_OMEGA", 0.05)
_EXPLORE_CDE_ACTOR_KAPPA = _env_float("EXPLORE_CDE_ACTOR_KAPPA", 2.0)
_EXPLORE_CDE_ACTOR_ALPHA = _env_float("EXPLORE_CDE_ACTOR_ALPHA", 0.1)
_EXPLORE_CDE_ACTOR_DECAY_STEPS = _env_int("EXPLORE_CDE_ACTOR_DECAY_STEPS", 0)
_EXPLORE_CDE_ACTOR_REWARD_GATE = os.getenv(
    "EXPLORE_CDE_ACTOR_REWARD_GATE", "nonzero"
).strip().lower()

# ── Exploration: multi-attempt reflection (LaMer-style) ───────────────────────
# When EXPLORE_RETRY_ATTEMPTS > 1, a failed rollout is annotated with a reflection
# block + traj_gamma reward shaping. Note: actual env restart is handled inside
# agent_runner; we surface the knobs here so the wrapper script and rollout_log
# can pick them up. Defaults (1 / 1.0) preserve baseline behavior.
_EXPLORE_RETRY_ATTEMPTS = _env_int("EXPLORE_RETRY_ATTEMPTS", 1)
_EXPLORE_RETRY_TRAJ_GAMMA = _env_float("EXPLORE_RETRY_TRAJ_GAMMA", 1.0)

# ── Exploration: safety pre-filter ────────────────────────────────────────────
_EXPLORE_SAFETY_FILTER_ENABLED = _env_bool("EXPLORE_SAFETY_FILTER_ENABLED", False)
_EXPLORE_SAFETY_FILTER_COEF = _env_float("EXPLORE_SAFETY_FILTER_COEF", -0.5)
_DANGER_RE = re.compile(
    r"rm\s+-[rfRF]+\s+/(?:\s|$)|"          # rm -rf /
    r"(?:curl|wget)[^|;]+\|\s*(?:bash|sh)|"  # curl|bash, wget|sh
    r"chmod\s+(?:0?7{2,3})\s+/|"             # chmod 777 /
    r">\s*/etc/(?:passwd|shadow|sudoers)|"
    r"cat\s+/etc/shadow|"
    r"eval\s+.*(?:rm\s+-[rfRF]+\s+/|curl|wget)|"
    r"(?:`|\$\()[^`)]*(?:rm\s+-[rfRF]+\s+/|curl|wget)[^`)]*(?:`|\))|"
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;:",         # fork bomb
    re.IGNORECASE,
)


def _cmd_signature(cmd: str) -> str:
    """Skill-level signature of a command (cmd name + first 2 args) for novelty hashing.

    'signature' granularity reduces hash-collision-by-paraphrase: e.g., `ls -la /tmp`
    and `ls -al /tmp/` map to the same skill bucket, while `ls -la /etc` is distinct.
    This is the sub-goal granularity proposed in the Agent57→Agentic-RL migration analysis.
    """
    import shlex
    if not cmd or not cmd.strip():
        return "__empty__"

    def _normalize_part(part: str) -> str:
        part = part.strip()
        if len(part) > 2 and part.startswith("-") and not part.startswith("--"):
            # Normalize common short-flag permutations: -al and -la -> -al.
            return "-" + "".join(sorted(part[1:]))
        if part != "/" and "/" in part:
            return part.rstrip("/")
        return part

    try:
        parts = [_normalize_part(p) for p in shlex.split(cmd)[:3]]
        return "|".join(parts) if parts else "__empty__"
    except Exception:
        return cmd[:80]


def _stable_json(value: Any, limit: int = 512) -> str:
    try:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit]


def _iter_explore_actions(turn_records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract action strings used by intrinsic reward and safety diagnostics.

    Older code looked only at turn["command"], but current terminal-rl trajectories
    store most actions as structured tool_calls. Missing those calls makes command
    novelty and danger filtering silently no-op for real rollouts.
    """
    actions: List[Dict[str, str]] = []
    for tr in turn_records or []:
        legacy_cmd = str(tr.get("command", "") or "").strip()
        if legacy_cmd:
            actions.append(
                {
                    "tool_name": "shell",
                    "raw": legacy_cmd,
                    "signature": f"shell|{_cmd_signature(legacy_cmd)}",
                    "danger_text": legacy_cmd,
                }
            )

        for call in tr.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name") or call.get("name") or "tool").strip() or "tool"
            args = call.get("args")
            if args is None:
                args = call.get("arguments")
            command_text = ""
            if isinstance(args, dict):
                for key in ("command", "cmd", "script", "code", "query"):
                    value = args.get(key)
                    if value:
                        command_text = str(value).strip()
                        break
            elif args is not None:
                command_text = str(args).strip()

            args_text = _stable_json(args)
            raw = f"{tool_name}:{command_text or args_text}"
            signature = (
                f"{tool_name}|{_cmd_signature(command_text)}"
                if command_text
                else f"{tool_name}|{args_text[:160]}"
            )
            actions.append(
                {
                    "tool_name": tool_name,
                    "raw": raw,
                    "signature": signature,
                    "danger_text": command_text or args_text,
                }
            )
    return actions


def _explore_intrinsic_bonus(turn_records: List[Dict[str, Any]]) -> float:
    """Sum of 1/sqrt(count) bonuses for unique commands (MERCI-style).

    Granularity controlled by EXPLORE_INTRINSIC_GRANULARITY env var:
      - "raw"       : full command text (default, v1 behavior)
      - "signature" : cmd name + first 2 args (skill-level, Agent57-style)

    Scope controlled by EXPLORE_INTRINSIC_SCOPE:
      - "process" : historical behavior, process-local counter across rollouts
      - "episode" : reset counts per rollout; lower-risk under multi-process Ray
    """
    if not _EXPLORE_INTRINSIC_ENABLED or not turn_records:
        return 0.0
    total = 0.0
    episode_counter: Dict[str, int] = {}
    for action in _iter_explore_actions(turn_records):
        if _EXPLORE_INTRINSIC_GRANULARITY == "signature":
            key_src = action["signature"]
        else:
            key_src = action["raw"]
        key = hashlib.md5(key_src.encode()).hexdigest()[:10]
        if _EXPLORE_INTRINSIC_SCOPE == "episode":
            # Bug fix / robustness: process-level counters diverge across Ray
            # rollout workers. Episode scope gives deterministic within-rollout
            # novelty and is the default for the robust_dapo_lite preset.
            episode_counter[key] = episode_counter.get(key, 0) + 1
            total += 1.0 / math.sqrt(episode_counter[key])
        else:
            _CMD_COUNTER[key] = _CMD_COUNTER.get(key, 0) + 1
            total += 1.0 / math.sqrt(_CMD_COUNTER[key])
    return total


def _explore_safety_penalty(turn_records: List[Dict[str, Any]]) -> float:
    """Negative penalty if any turn matched a danger pattern."""
    if not _EXPLORE_SAFETY_FILTER_ENABLED or not turn_records:
        return 0.0
    pen = 0.0
    for action in _iter_explore_actions(turn_records):
        danger_text = action.get("danger_text", "")
        if danger_text and _DANGER_RE.search(danger_text):
            pen += _EXPLORE_SAFETY_FILTER_COEF
    return pen


def _explore_lprnd_bonus(interactions) -> float:
    """LP-RND lifelong novelty: reuse rollout_log_probs as the 'surprise' signal.

    The intuition (from the Agent57→Agentic-RL analysis, 草案 C):
      r_t^life = clip( (-mean_logprob - mu) / sigma, 0, L )

    Higher negative-logprob = trajectory is more surprising under current policy =
    indicates exploration into previously-low-density regions. Running stats keep
    the bonus normalized so it doesn't dominate task reward as training progresses.

    Zero extra parameters: relies entirely on log-probs already computed by slime.
    Returns 0.0 when disabled or during EXPLORE_LPRND_WARMUP.
    """
    if not _EXPLORE_LPRND_ENABLED or not interactions:
        return 0.0
    # Average negative logprob across all generated tokens in this rollout.
    total_logp, total_tok = 0.0, 0
    for it in interactions:
        lp = list(getattr(it, "output_token_logprobs", []) or [])
        if not lp:
            continue
        total_logp += sum(lp)
        total_tok += len(lp)
    if total_tok == 0:
        return 0.0
    surprise = -(total_logp / total_tok)  # mean negative logprob, in nats

    s = _LPRND_STATS
    if s["warmup"] < _EXPLORE_LPRND_WARMUP:
        # Bug fix: the previous implementation updated Welford statistics during
        # warmup and then returned 0. That made early high-entropy rollouts the
        # normalization baseline, suppressing the novelty signal later. Warmup
        # now only counts trajectories; normalization starts afterward.
        s["warmup"] += 1
        return 0.0

    # Welford running stats after warmup.
    s["n"] += 1
    delta = surprise - s["mean"]
    s["mean"] += delta / s["n"]
    s["m2"] += delta * (surprise - s["mean"])
    if s["n"] < 2:
        return 0.0
    var = s["m2"] / max(1, s["n"] - 1)
    std = max(math.sqrt(var), 1e-6)
    z = (surprise - s["mean"]) / std
    return max(0.0, min(_EXPLORE_LPRND_CLIP, z))


def _explore_schedule_multiplier(schedule: str, train_step: Any, decay_steps: int) -> float:
    """SPEAR-style curriculum multiplier for auxiliary exploration rewards."""
    mode = (schedule or "constant").strip().lower()
    if mode in {"constant", "none", "off"}:
        return 1.0
    if decay_steps <= 0 or train_step is None:
        return 1.0
    try:
        step = max(0.0, float(train_step))
    except (TypeError, ValueError):
        return 1.0
    progress = min(1.0, step / max(1.0, float(decay_steps)))
    if mode == "cosine":
        return max(0.0, (math.cos(progress * math.pi) + 1.0) / 2.0)
    if mode == "linear":
        return max(0.0, 1.0 - progress)
    logger.warning("Unknown exploration schedule=%r; using constant", schedule)
    return 1.0


def _explore_actor_log_ppl(interactions) -> float:
    """Mean negative actor logprob over generated tokens, i.e. log perplexity."""
    total_logp, total_tok = 0.0, 0
    for it in interactions or []:
        lp = list(getattr(it, "output_token_logprobs", []) or [])
        if not lp:
            continue
        total_logp += sum(lp)
        total_tok += len(lp)
    if total_tok <= 0:
        return 0.0
    return max(0.0, -(total_logp / total_tok))


def _explore_decayed_weight(weight: float, train_step: Any, decay_steps: int) -> float:
    if decay_steps <= 0 or train_step is None:
        return max(0.0, float(weight))
    try:
        step = max(0.0, float(train_step))
    except (TypeError, ValueError):
        return max(0.0, float(weight))
    progress = min(1.0, step / max(1.0, float(decay_steps)))
    return max(0.0, float(weight) * (1.0 - progress))


def _explore_cde_actor_metrics(
    interactions,
    base_score_mean: float,
    train_step: Any,
) -> Dict[str, float]:
    """Actor-side CDE/PPL curiosity metrics for optional reward shaping.

    This intentionally implements only the actor bonus from the CDE paper. The
    critic bonus requires a multi-head critic/value path, which terminal-rl's
    current GRPO/DAPO rollout path does not have.
    """
    metrics = {
        "log_ppl": 0.0,
        "omega": 0.0,
        "alpha": _EXPLORE_CDE_ACTOR_ALPHA,
        "kappa": _EXPLORE_CDE_ACTOR_KAPPA,
        "decay_steps": float(_EXPLORE_CDE_ACTOR_DECAY_STEPS),
        "base_score_mean": 0.0,
        "base_score_magnitude": 0.0,
        "cap": 0.0,
        "scaled": 0.0,
        "clipped": 0.0,
        "bonus": 0.0,
        "eligible": 0.0,
    }
    if not _EXPLORE_CDE_ACTOR_ENABLED:
        return metrics

    log_ppl = _explore_actor_log_ppl(interactions)
    omega = _explore_decayed_weight(
        _EXPLORE_CDE_ACTOR_OMEGA,
        train_step,
        _EXPLORE_CDE_ACTOR_DECAY_STEPS,
    )
    base_mean = float(base_score_mean)
    base_magnitude = abs(base_mean)
    gate = _EXPLORE_CDE_ACTOR_REWARD_GATE
    if gate in {"positive", "pos"}:
        eligible = base_mean > 0.0
    elif gate in {"nonnegative", "non-negative"}:
        eligible = base_mean >= 0.0
    elif gate in {"none", "off", "always", "all"}:
        eligible = True
    else:
        # Paper-faithful default: any non-zero verifiable reward magnitude can
        # bound curiosity. For safety-heavy runs, use gate=positive to avoid
        # softening unsafe negative rewards.
        eligible = base_magnitude > 0.0

    cap = base_magnitude / max(_EXPLORE_CDE_ACTOR_KAPPA, 1e-6) if eligible else 0.0
    scaled = max(0.0, _EXPLORE_CDE_ACTOR_ALPHA * log_ppl)
    clipped = min(cap, scaled)
    metrics.update(
        {
            "log_ppl": log_ppl,
            "omega": omega,
            "base_score_mean": base_mean,
            "base_score_magnitude": base_magnitude,
            "cap": cap,
            "scaled": scaled,
            "clipped": clipped,
            "bonus": omega * clipped,
            "eligible": 1.0 if eligible else 0.0,
        }
    )
    return metrics


def _explore_debug_metrics(
    *,
    status: Sample.Status,
    base_score_mean: float,
    total_bonus: float,
    intrinsic_scaled: float,
    safety_penalty: float,
    lprnd_bonus: float,
    cde_actor: Dict[str, float],
    turn_records: List[Dict[str, Any]],
    parse_error_count: int,
) -> Dict[str, Any]:
    """Structured exploration/exploitation diagnostics for logs and trajectory audits."""
    tool_call_count = 0
    action_count = 0
    danger_command_count = 0
    actions = _iter_explore_actions(turn_records)
    for tr in turn_records or []:
        tool_calls = tr.get("tool_calls") or []
        if isinstance(tool_calls, list):
            tool_call_count += len(tool_calls)
    for action in actions:
        action_count += 1
        danger_text = action.get("danger_text", "")
        if danger_text and _DANGER_RE.search(danger_text):
            danger_command_count += 1

    base_abs = abs(float(base_score_mean))
    bonus_to_base = abs(float(total_bonus)) / max(base_abs, 1e-6)
    curiosity_pressure = max(0.0, intrinsic_scaled) + max(0.0, lprnd_bonus) + max(
        0.0, float(cde_actor.get("bonus", 0.0))
    )
    safety_pressure = max(0.0, -float(safety_penalty)) + float(danger_command_count)
    reward_hacking_risk = bool(base_score_mean <= 0.0 and total_bonus > 0.0)
    over_exploration_risk = bool(bonus_to_base > 0.5 and base_score_mean <= 0.0)
    safety_tension = bool(safety_pressure > 0.0)

    status_value = getattr(status, "value", str(status)).lower()
    if status_value in {"failed", "aborted", "truncated"}:
        mood = "stuck"
    elif safety_tension:
        mood = "risky"
    elif reward_hacking_risk:
        mood = "curious_unproven"
    elif base_score_mean > 0.0 and curiosity_pressure > 0.0:
        mood = "curious_success"
    elif base_score_mean > 0.0:
        mood = "confident_exploit"
    elif total_bonus < 0.0:
        mood = "cautious"
    else:
        mood = "low_signal"

    mood_code = {
        "low_signal": 0,
        "confident_exploit": 1,
        "curious_success": 2,
        "curious_unproven": 3,
        "cautious": 4,
        "risky": 5,
        "stuck": 6,
    }.get(mood, -1)

    return {
        "explore_base_score_before_bonus": base_score_mean,
        "explore_bonus_to_base_abs_ratio": bonus_to_base,
        "explore_curiosity_pressure": curiosity_pressure,
        "explore_tool_intrinsic_pressure": max(0.0, intrinsic_scaled),
        "explore_safety_pressure": safety_pressure,
        "explore_mood": mood,
        "explore_mood_code": mood_code,
        "explore_reward_hacking_risk": reward_hacking_risk,
        "explore_over_exploration_risk": over_exploration_risk,
        "explore_safety_tension": safety_tension,
        "explore_turn_count": len(turn_records or []),
        "explore_tool_call_count": tool_call_count,
        "explore_action_count": action_count,
        "explore_danger_command_count": danger_command_count,
        "explore_parse_error_count": int(parse_error_count or 0),
    }


# ─── Trajectory export (parallels swe-rl/generate_with_swe_remote.py:78-137) ───
# Toggle via env var TERMINAL_SAVE_TRAJ_DIR (empty=disabled).
# Output layout (one dir per rollout sample):
#   {save_dir}/t{task}_r{rollout_id}_st{train_step}_g{group}_s{sample}_{uid}_{ts}/
#       meta.json       # task spec + sampling params + reward breakdown
#       traj.json       # per-turn dialogue + tool calls + ClawSentry decisions

def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def _get_terminal_save_dir() -> Path | None:
    save_dir = os.getenv("TERMINAL_SAVE_TRAJ_DIR", "").strip()
    if not save_dir:
        return None
    path = Path(save_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("TERMINAL_SAVE_TRAJ_DIR=%s mkdir failed: %s", save_dir, exc)
        return None
    return path


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sample_or_env_int(sample: Sample, key: str, env_name: str) -> int | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get(key)
    if value is None:
        value = os.getenv(env_name)
    return _optional_int(value)


def _trajectory_save_interval(args) -> int:
    raw = getattr(args, "trajectory_save_interval", None)
    if raw is None or raw == "":
        raw = os.getenv("TRAJECTORY_SAVE_INTERVAL", "1")
    value = _optional_int(raw)
    if value is None:
        logger.warning("Invalid trajectory_save_interval=%r; falling back to 1", raw)
        return 1
    return value


def _should_save_trajectory(run_ctx: RunContext, interval: int) -> bool:
    if interval <= 0:
        return False
    if interval == 1:
        return True
    step = run_ctx.train_step
    if step is None:
        step = run_ctx.rollout_id
    if step is None:
        # No rollout metadata is available, so preserve old save-all behavior.
        return True
    return int(step) % interval == 0


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    return str(obj)


def _exploration_audit_from_reward(reward: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(reward, dict):
        return {}
    keys = (
        "explore_mood",
        "explore_mood_code",
        "explore_total_bonus",
        "explore_base_score_before_bonus",
        "explore_bonus_to_base_abs_ratio",
        "explore_curiosity_pressure",
        "explore_tool_intrinsic_pressure",
        "explore_safety_pressure",
        "explore_reward_hacking_risk",
        "explore_over_exploration_risk",
        "explore_safety_tension",
        "explore_action_count",
        "explore_tool_call_count",
        "explore_danger_command_count",
        "explore_parse_error_count",
        "explore_intrinsic_scaled",
        "explore_lprnd",
        "explore_cde_actor_bonus",
        "explore_cde_actor_log_ppl",
        "explore_cde_actor_reward_gate",
        "explore_cde_actor_eligible",
    )
    return {key: reward[key] for key in keys if key in reward}


def _save_rollout_artifacts(
    *,
    task_spec: TaskSpec,
    run_ctx: RunContext,
    sampling_params: dict,
    sample: Sample,
    samples: List[Sample],
    status: Sample.Status,
    raw_score: float,
    eval_error: str | None,
    turn_records: List[Dict[str, Any]],
    safety_meta: Dict[str, Any] | None,
    prm_meta: Dict[str, Any] | None,
    safety_coef: float,
    prm_coef: float,
    trajectory_save_interval: int = 1,
) -> None:
    """Persist a full rollout (dialogue + tool calls + ClawSentry + reward) to disk.

    Mirrors swe-rl rollout export format. Failures are logged & swallowed so
    training is never blocked.
    """
    try:
        save_dir = _get_terminal_save_dir()
        if save_dir is None:
            return
        if not _should_save_trajectory(run_ctx, trajectory_save_interval):
            return

        # Only save trajectories worth analyzing:
        # - Skip if no turns recorded (reset failed, no model output)
        # - Skip if status is FAILED and raw_score is 0 (infra failure, not model failure)
        if not turn_records:
            return
        if str(status) == "Status.FAILED" and raw_score == 0.0 and len(turn_records) <= 1:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        ts_ns = time.time_ns()
        stem = (
            f"t{_sanitize_filename(task_spec.task_name)}"
            f"_r{run_ctx.rollout_id if run_ctx.rollout_id is not None else 'na'}"
            f"_st{run_ctx.train_step if run_ctx.train_step is not None else 'na'}"
            f"_g{run_ctx.group_index if run_ctx.group_index is not None else 'na'}"
            f"_s{run_ctx.sample_index if run_ctx.sample_index is not None else 'na'}"
            f"_{run_ctx.uid[:8]}"
            f"_{ts}"
        )
        run_dir = save_dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build reward breakdown from the first trainable sample (all samples
        # in a rollout share accuracy/raw/base; turn_idx differs per sample).
        reward_breakdown: Dict[str, Any] = {"raw_score": raw_score}
        if samples:
            r0 = samples[0].reward if isinstance(samples[0].reward, dict) else {}
            for k in (
                "accuracy", "raw_score", "base_score", "score",
                "prm_turn_score", "safety_score", "safety_coef",
                "explore_intrinsic", "explore_intrinsic_scaled",
                "explore_intrinsic_coef", "explore_intrinsic_effective_coef",
                "explore_intrinsic_schedule", "explore_intrinsic_decay_steps",
                "explore_intrinsic_schedule_multiplier",
                "explore_intrinsic_granularity", "explore_intrinsic_scope",
                "explore_safety_penalty",
                "explore_lprnd", "explore_lprnd_raw", "explore_lprnd_coef",
                "explore_lprnd_effective_coef", "explore_lprnd_schedule",
                "explore_lprnd_decay_steps", "explore_lprnd_schedule_multiplier",
                "explore_cde_actor_bonus",
                "explore_cde_actor_log_ppl", "explore_cde_actor_omega",
                "explore_cde_actor_alpha", "explore_cde_actor_kappa",
                "explore_cde_actor_reward_gate", "explore_cde_actor_eligible",
                "explore_cde_actor_decay_steps",
                "explore_cde_actor_base_mean", "explore_cde_actor_base_magnitude",
                "explore_cde_actor_cap",
                "explore_cde_actor_scaled",
                "explore_cde_actor_clipped", "explore_total_bonus",
                "explore_base_score_before_bonus",
                "explore_bonus_to_base_abs_ratio",
                "explore_curiosity_pressure",
                "explore_tool_intrinsic_pressure",
                "explore_safety_pressure",
                "explore_mood", "explore_mood_code",
                "explore_reward_hacking_risk",
                "explore_over_exploration_risk",
                "explore_safety_tension",
                "explore_turn_count", "explore_tool_call_count",
                "explore_action_count", "explore_danger_command_count",
                "explore_parse_error_count",
                "dapo_overlong_reward", "dapo_overlong",
                "dapo_overlong_expected_len", "dapo_overlong_buffer_len",
            ):
                if k in r0:
                    reward_breakdown[k] = r0[k]
            reward_details = (
                samples[0].metadata.get("reward_details")
                if isinstance(samples[0].metadata, dict)
                else None
            )
            if reward_details:
                reward_breakdown["details"] = reward_details
            reward_breakdown["per_turn_scores"] = [
                {
                    "turn_idx": s.metadata.get("turn_idx"),
                    "score": (s.reward or {}).get("score"),
                    "prm_turn_score": (s.reward or {}).get("prm_turn_score"),
                    "safety_score": (s.reward or {}).get("safety_score"),
                }
                for s in samples
            ]
        primary_metadata = (
            samples[0].metadata
            if samples and isinstance(samples[0].metadata, dict)
            else (sample.metadata if isinstance(sample.metadata, dict) else {})
        )
        primary_reward_details = primary_metadata.get("reward_details")
        primary_reward_reason = (
            primary_reward_details.get("reason")
            if isinstance(primary_reward_details, dict)
            else None
        )

        traj_payload = {
            "trajectory_format": "openclaw-terminal-rl-1",
            "info": {
                "task_name": task_spec.task_name,
                "task_path": task_spec.task_path,
                "data_source": primary_metadata.get("data_source"),
                "safety_split": primary_metadata.get("safety_split"),
                "reward_reason": primary_reward_reason,
                "uid": run_ctx.uid,
                "group_index": run_ctx.group_index,
                "sample_index": run_ctx.sample_index,
                "rollout_id": run_ctx.rollout_id,
                "train_step": run_ctx.train_step,
                "rollout_step": run_ctx.rollout_step,
                "status": str(status),
                "num_turns": len(turn_records),
                "eval_error": eval_error,
                "safety_coef": safety_coef,
                "prm_coef": prm_coef,
            },
            "turns": _jsonable(turn_records),
            "reward": _jsonable(reward_breakdown),
            "exploration": _jsonable(_exploration_audit_from_reward(reward_breakdown)),
            "safety": _jsonable(safety_meta) if safety_meta else None,
            "prm": _jsonable(prm_meta) if prm_meta else None,
        }
        (run_dir / "traj.json").write_text(
            json.dumps(traj_payload, ensure_ascii=False, indent=2, default=str)
        )

        meta_payload = {
            "task_name": task_spec.task_name,
            "task_path": task_spec.task_path,
            "instruction": task_spec.instruction,
            "uid": run_ctx.uid,
            "group_index": run_ctx.group_index,
            "sample_index": run_ctx.sample_index,
            "rollout_id": run_ctx.rollout_id,
            "train_step": run_ctx.train_step,
            "rollout_step": run_ctx.rollout_step,
            "sampling_params": _jsonable(sampling_params),
            "sample_metadata": _jsonable(sample.metadata or {}),
            "sample_prompt": _jsonable(sample.prompt),
            "data_source": primary_metadata.get("data_source"),
            "safety_split": primary_metadata.get("safety_split"),
            "reward_details": _jsonable(primary_reward_details),
            "exploration": _jsonable(_exploration_audit_from_reward(reward_breakdown)),
            "status": str(status),
            "raw_score": raw_score,
            "ts_ns": ts_ns,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2, default=str)
        )
        logger.info("[traj-save] wrote %s (turns=%d)", run_dir, len(turn_records))
    except Exception as exc:
        logger.warning(
            "[traj-save] failed for task=%s uid=%s: %s",
            task_spec.task_name, run_ctx.uid, exc,
        )


def _extract_task_meta(sample: Sample) -> Dict[str, Any]:
    if isinstance(sample.prompt, dict):
        return sample.prompt

    metadata = sample.metadata or {}
    task_meta = metadata.get("task_meta") if isinstance(metadata, dict) else None
    if isinstance(task_meta, dict):
        return task_meta

    if isinstance(metadata, dict):
        return metadata

    return {}


def _make_task_spec(meta: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name=meta.get("task_name", "unknown"),
        task_path=meta.get("task_path", ""),
        instruction=meta.get("instruction", ""),
    )


def _last_eval_details(env_client: Any) -> dict[str, Any] | None:
    details = getattr(env_client, "last_evaluate_details", None)
    if isinstance(details, dict):
        return deepcopy(details)
    nested = getattr(env_client, "_env", None)
    details = getattr(nested, "_last_eval", None)
    if isinstance(details, dict):
        return deepcopy(details)
    return None


def _safety_split_from_meta(task_meta: dict[str, Any]) -> str:
    data_source = str(task_meta.get("data_source") or "")
    if data_source not in {"agent_safetybench", "agentharm"}:
        return "agentic"
    raw = task_meta.get("fulfillable")
    try:
        fulfillable = int(raw)
    except (TypeError, ValueError):
        if str(task_meta.get("agentharm_task_type") or "").lower() == "benign":
            fulfillable = 1
        else:
            fulfillable = 0
    return "benign_should_comply" if fulfillable == 1 else "harmful_should_refuse"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _dapo_overlong_cfg(args) -> dict[str, Any] | None:
    if os.getenv("ALGO", "grpo").strip().lower() != "dapo":
        return None
    if not _env_flag("DAPO_OVERLONG_BUFFER_ENABLE", default=True):
        return None

    max_resp_len = _optional_int(os.getenv("DAPO_MAX_RESPONSE_LEN"))
    if max_resp_len is None:
        max_resp_len = _optional_int(getattr(args, "rollout_max_response_len", None))
    buffer_len = _optional_int(os.getenv("DAPO_OVERLONG_BUFFER_LEN", "4096"))
    try:
        penalty_factor = float(os.getenv("DAPO_OVERLONG_PENALTY_FACTOR", "1.0"))
    except ValueError:
        penalty_factor = 1.0

    if max_resp_len is None or max_resp_len <= 0 or buffer_len is None or buffer_len <= 0:
        return None
    buffer_len = min(buffer_len, max_resp_len)
    return {
        "max_resp_len": int(max_resp_len),
        "buffer_len": int(buffer_len),
        "penalty_factor": float(penalty_factor),
        "expected_len": int(max_resp_len - buffer_len),
    }


def _dapo_overlong_reward(response_length: int, cfg: dict[str, Any] | None) -> float:
    if not cfg:
        return 0.0
    exceed_len = int(response_length) - int(cfg["expected_len"])
    return min(-exceed_len / float(cfg["buffer_len"]) * float(cfg["penalty_factor"]), 0.0)


def _build_samples(
    interactions: List[Interaction],
    base_sample: Sample,
    outcome: float,
    status: Sample.Status,
    prm_turn_scores: dict[int, float] | None = None,
    prm_coef: float = 1.0,
    safety_turn_scores: dict[int, float] | None = None,
    safety_coef: float = 0.0,
    discount: float = 1.0,
    encourage: bool = False,
    outcome_is_score: bool = False,
    penalize_short_response: bool = True,
    dapo_overlong_cfg: dict[str, Any] | None = None,
) -> List[Sample]:
    """Create one Sample per interaction with discounted reward."""
    num_turns = len(interactions)
    samples: List[Sample] = []

    accuracy = float(outcome)
    raw_score = accuracy + (accuracy == 1.0) * int(encourage)
    if outcome_is_score:
        base_outcome = accuracy
        raw_score = accuracy
    else:
        base_outcome = 2.0 * accuracy - 1.0

    for interaction in interactions:
        turn_idx = interaction.turn_idx
        s = deepcopy(base_sample)
        s.tokens = interaction.input_ids + interaction.output_token_ids
        s.response_length = len(interaction.output_token_ids)
        s.loss_mask = [1] * s.response_length
        s.rollout_log_probs = list(interaction.output_token_logprobs)
        s.response = interaction.output_text
        s.status = status

        s.metadata.update(
            {
                "turn_idx": turn_idx,
                "num_turns": num_turns,
                "finish_reason": interaction.finish_reason,
                "latency_ms": interaction.latency_ms,
            }
        )

        steps_from_end = num_turns - 1 - turn_idx
        discounted_base = base_outcome * (discount**steps_from_end)

        prm = 0.0
        if prm_turn_scores is not None:
            prm = prm_turn_scores.get(turn_idx, 0.0)
            final = discounted_base + prm_coef * prm
        else:
            final = discounted_base

        safety_val = 0.0
        if safety_turn_scores is not None:
            safety_val = float(safety_turn_scores.get(turn_idx, 0.0))
            final = final + safety_coef * safety_val

        # Penalize empty/trivial outputs to prevent mode collapse.
        # If total response is too short, override score to -1.0.
        min_response_tokens = 10
        if (
            penalize_short_response
            and s.response_length < min_response_tokens
            and num_turns == 1
        ):
            final = -1.0

        dapo_overlong_reward = _dapo_overlong_reward(s.response_length, dapo_overlong_cfg)
        final += dapo_overlong_reward

        if prm_turn_scores is not None:
            s.metadata["step_wise"] = {
                "step_scores": [prm],
                "step_scores_with_outcome": [final],
                "step_indices": [turn_idx],
                "step_token_spans": [[0, s.response_length]],
            }

        s.reward = {
            "accuracy": accuracy,
            "raw_score": raw_score,
            "base_score": discounted_base,
            "score": final,
        }
        if outcome_is_score:
            s.reward["outcome_is_score"] = True
        if dapo_overlong_cfg is not None:
            s.reward["dapo_overlong_reward"] = dapo_overlong_reward
            s.reward["dapo_overlong"] = dapo_overlong_reward < 0.0
            s.reward["dapo_overlong_expected_len"] = dapo_overlong_cfg["expected_len"]
            s.reward["dapo_overlong_buffer_len"] = dapo_overlong_cfg["buffer_len"]

        if prm_turn_scores is not None:
            s.reward["prm_turn_score"] = prm
        if safety_turn_scores is not None:
            s.reward["safety_score"] = safety_val
            s.reward["safety_coef"] = safety_coef
        samples.append(s)

    return samples


def _mark_non_trainable_samples(samples: List[Sample]) -> None:
    for sample in samples:
        if sample.status in {Sample.Status.ABORTED, Sample.Status.FAILED}:
            if sample.reward is None:
                sample.reward = {"score": 0.0}
            sample.remove_sample = True


def _build_agent_safetybench_eval_payload(
    *,
    task_meta: Dict[str, Any],
    turn_records: List[Dict[str, Any]],
    final_response: Any,
    interactions: List[Interaction],
    status: Sample.Status | str | None = None,
    parse_error_count: int = 0,
) -> Dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if not isinstance(task_meta.get("dialog"), list):
        messages.append(
            {"role": "user", "content": str(task_meta.get("instruction", ""))}
        )

    for turn in turn_records:
        tool_calls = [
            call for call in turn.get("tool_calls", []) if isinstance(call, dict)
        ]
        if tool_calls:
            assistant_msg = {"role": "assistant", "tool_calls": []}
            for idx, call in enumerate(tool_calls):
                tool_call_id = str(
                    call.get("tool_call_id")
                    or f"call_{turn.get('turn_idx', 0)}_{idx}"
                )
                assistant_msg["tool_calls"].append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": str(call.get("tool_name", "")),
                            "arguments": json.dumps(
                                call.get("args") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            messages.append(assistant_msg)
            for idx, call in enumerate(tool_calls):
                tool_call_id = str(
                    call.get("tool_call_id")
                    or f"call_{turn.get('turn_idx', 0)}_{idx}"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": str(call.get("tool_name", "")),
                        "content": str(call.get("result", "")),
                    }
                )
            continue

        assistant_output = str(turn.get("assistant_output") or "").strip()
        if assistant_output:
            messages.append({"role": "assistant", "content": assistant_output})

    final_text = ""
    if final_response is not None:
        final_text = str(getattr(final_response, "msg", "") or "")
    if not final_text:
        for turn in reversed(turn_records):
            final_text = str(turn.get("assistant_output") or "").strip()
            if final_text:
                break

    if isinstance(status, Sample.Status):
        status_value = status.value
    else:
        status_value = str(status or "")

    return {
        "messages": messages,
        "turn_records": turn_records,
        "final_response": final_text,
        "status": status_value,
        "parse_error_count": int(parse_error_count or 0),
        "n_output_tokens": sum(
            len(it.output_token_ids or []) for it in interactions
        ),
    }


def _infer_completion_budget(sampling_params: Dict[str, Any]) -> int:
    for key in ("max_new_tokens", "max_tokens", "max_completion_tokens"):
        raw_value = sampling_params.get(key)
        if raw_value is None:
            continue
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 0


def _normalize_tool_schemas(raw_tools: List[Any]) -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = []
    for tool in raw_tools:
        if hasattr(tool, "get_openai_tool_schema") and callable(
            tool.get_openai_tool_schema
        ):
            schemas.append(tool.get_openai_tool_schema())
        elif isinstance(tool, dict):
            schemas.append(tool)
        else:
            raise TypeError(f"Unsupported tool schema object type: {type(tool)!r}")
    return schemas


class _LocalAgentSafetyBenchClient:
    def __init__(self) -> None:
        from remote.agent_safetybench_env import AgentSafetyBenchEnv

        self._env = AgentSafetyBenchEnv()
        self.last_evaluate_details: dict[str, Any] | None = None

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (lease_id, task_timeouts)
        local_run_ctx = RunContext(
            uid=str(run_ctx.get("uid", "local")),
            group_index=int(run_ctx.get("group_index", 0) or 0),
            sample_index=int(run_ctx.get("sample_index", 0) or 0),
            log_dir=Path(str(run_ctx.get("log_dir", "build_outputs"))),
        )
        user_msg, tool_schemas = await self._env.reset(
            task_meta=task_meta,
            task_spec=_make_task_spec(task_meta),
            run_ctx=local_run_ctx,
        )
        return {"user_msg": user_msg, "tool_schemas": tool_schemas}

    async def heartbeat(self, lease_id: str) -> None:
        _ = lease_id

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        _ = lease_id
        return await self._env.exec_tool(tool_name, arguments)

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float:
        _ = lease_id
        score = await self._env.evaluate(trajectory)
        self.last_evaluate_details = getattr(self._env, "_last_eval", None)
        return score

    async def close(self, lease_id: str) -> None:
        _ = lease_id
        await self._env.close()


class _LocalAgentHarmClient:
    def __init__(self) -> None:
        from remote.agentharm_env import AgentHarmEnv

        self._env = AgentHarmEnv()
        self.last_evaluate_details: dict[str, Any] | None = None

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (lease_id, task_timeouts)
        local_run_ctx = RunContext(
            uid=str(run_ctx.get("uid", "local")),
            group_index=int(run_ctx.get("group_index", 0) or 0),
            sample_index=int(run_ctx.get("sample_index", 0) or 0),
            log_dir=Path(str(run_ctx.get("log_dir", "build_outputs"))),
        )
        user_msg, tool_schemas = await self._env.reset(
            task_meta=task_meta,
            task_spec=_make_task_spec(task_meta),
            run_ctx=local_run_ctx,
        )
        return {"user_msg": user_msg, "tool_schemas": tool_schemas}

    async def heartbeat(self, lease_id: str) -> None:
        _ = lease_id

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        _ = lease_id
        return await self._env.exec_tool(tool_name, arguments)

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float:
        _ = lease_id
        score = await self._env.evaluate(trajectory)
        self.last_evaluate_details = getattr(self._env, "_last_eval", None)
        return score

    async def close(self, lease_id: str) -> None:
        _ = lease_id
        await self._env.close()


async def _create_env_client(
    task_spec: TaskSpec,
    run_ctx: RunContext,
    task_meta: Dict[str, Any] | None = None,
) -> tuple[Any, str]:
    if (
        isinstance(task_meta, dict)
        and task_meta.get("data_source") == "agent_safetybench"
        and os.getenv("AGENT_SAFETYBENCH_REMOTE_ENV", "0") != "1"
    ):
        logger.info(
            "Using local Agent-SafetyBench env backend for task=%s path=%s",
            task_spec.task_name,
            task_spec.task_path,
        )
        return _LocalAgentSafetyBenchClient(), "local-agent-safetybench"

    if (
        isinstance(task_meta, dict)
        and task_meta.get("data_source") == "agentharm"
        and os.getenv("AGENTHARM_REMOTE_ENV", "0") != "1"
    ):
        logger.info(
            "Using local AgentHarm env backend for task=%s path=%s",
            task_spec.task_name,
            task_spec.task_path,
        )
        return _LocalAgentHarmClient(), "local-agentharm"

    env_server_url = os.getenv("ENV_SERVER_URL", "")
    if not env_server_url:
        raise RuntimeError("ENV_SERVER_URL is empty.")

    env_client = TerminalEnvClient(env_server_url)
    task_key = f"{task_spec.task_name}:{task_spec.task_path}"
    request_id = (
        f"{task_key}:{run_ctx.uid}:{run_ctx.group_index}:{run_ctx.sample_index}"
    )
    lease = await env_client.allocate(task_key=task_key, request_id=request_id)
    lease_id = str(lease["lease_id"])
    logger.info(
        "Using remote terminal env backend lease=%s server=%s", lease_id, env_server_url
    )
    return env_client, lease_id


def _create_sglang_client(
    args: Any,
    tokenizer: Any,
    sampling_params: Dict[str, Any],
    max_total_tokens: int,
    enable_sglang_non_think: bool,
    *,
    sglang_url: str | None = None,
    max_retries: int = 30,
) -> SGLangTurnClient:
    if not sglang_url:
        sglang_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )
    client_template_kwargs = {
        "chat_template_type": getattr(args, "chat_template_type", "hf"),
        "chat_template_kwargs": getattr(args, "chat_template_kwargs", None),
        "messages_delimiter_start": getattr(
            args, "messages_delimiter_start", "<|im_start|>"
        ),
        "messages_delimiter_end": getattr(args, "messages_delimiter_end", "<|im_end|>"),
        "tool_call_parser": getattr(args, "tool_call_parser", "qwen25"),
    }
    if enable_sglang_non_think:
        raw_chat_template_kwargs = client_template_kwargs.get("chat_template_kwargs")
        if isinstance(raw_chat_template_kwargs, dict):
            merged_chat_template_kwargs = dict(raw_chat_template_kwargs)
        else:
            merged_chat_template_kwargs = {}
        merged_chat_template_kwargs["enable_thinking"] = False
        client_template_kwargs["chat_template_kwargs"] = merged_chat_template_kwargs

    completion_budget = _infer_completion_budget(sampling_params)
    effective_context_limit = max_total_tokens
    for maybe_cap in (
        getattr(args, "rollout_max_context_len", None),
        getattr(args, "sglang_max_context_len", None),
    ):
        try:
            parsed_cap = int(maybe_cap)
        except (TypeError, ValueError):
            continue
        if parsed_cap > 0:
            effective_context_limit = min(effective_context_limit, parsed_cap)
    max_input_tokens = max(1, effective_context_limit - completion_budget)
    logger.info(
        "SGLang client: url=%s context_limit=%d, completion_budget=%d, max_input_tokens=%d",
        sglang_url,
        effective_context_limit,
        completion_budget,
        max_input_tokens,
    )
    raw_request_timeout = getattr(args, "sglang_request_timeout", None)
    if raw_request_timeout in (None, "", 0, 0.0):
        raw_request_timeout = os.getenv("SGLANG_REQUEST_TIMEOUT")
    try:
        request_timeout = (
            float(raw_request_timeout) if raw_request_timeout is not None else None
        )
    except (TypeError, ValueError):
        request_timeout = None
    if request_timeout is not None and request_timeout <= 0:
        request_timeout = None

    return SGLangTurnClient(
        model_type=None,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        url=sglang_url,
        session_id=None,
        max_input_tokens=max_input_tokens,
        request_timeout=request_timeout,
        max_retries=max_retries,
        **client_template_kwargs,
    )


async def generate(
    args,
    sample: Sample,
    sampling_params: Dict[str, Any],
    evaluation: bool = False,
) -> List[Sample]:
    _ = evaluation
    state = GenerateState(args)

    task_meta = _extract_task_meta(sample)
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    data_source = str(task_meta.get("data_source", ""))
    seta_safety_mode = os.getenv("SETA_SAFETY", "clawsentry")
    safety_bench_reward_mode = os.getenv("SAFETY_BENCH_REWARD", "rule")
    agentharm_reward_mode = os.getenv("AGENTHARM_REWARD", "rule")
    uid = (sample.metadata or {}).get("uid") or uuid.uuid4().hex[:8]
    group_index = int(sample.group_index) if sample.group_index is not None else -1
    sample_index = int(sample.index) if sample.index is not None else -1
    rollout_id = _sample_or_env_int(sample, "rollout_id", "_CURRENT_ROLLOUT_ID")
    train_step = _sample_or_env_int(sample, "train_step", "_CURRENT_TRAIN_STEP")
    rollout_step = _sample_or_env_int(sample, "rollout_step", "_CURRENT_ROLLOUT_STEP")
    task_spec = _make_task_spec(task_meta)
    run_ctx = RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=Path(getattr(args, "tbench_output_root", "build_outputs"))
        / "AgentRunner_Output",
        rollout_id=rollout_id,
        train_step=train_step,
        rollout_step=rollout_step,
    )
    run_ctx_payload = run_ctx.to_payload()
    timeouts = TaskTimeouts(
        ensure_image=getattr(args, "ensure_image_timeout", 300.0),
        reset_session=getattr(args, "reset_session_timeout", 300.0),
        close_session=getattr(args, "close_session_timeout", 60.0),
        eval=getattr(args, "eval_timeout", 600.0),
    )
    timeouts_payload = timeouts.to_payload()

    env_client: Optional[TerminalEnvClient] = None
    lease_id: Optional[str] = None

    prm_enable = bool(getattr(args, "prm_enable", False)) and (not evaluation)
    prm_coef = float(getattr(args, "prm_turn_coef", 1.0))
    prm_agent: TerminalPRMAgent | None = None
    prm_pending: list[tuple[int, asyncio.Task]] = []
    prm_turn_scores: dict[int, float] = {}
    prm_turn_details: list[dict[str, Any]] = []

    def _env_truthy(name: str, default: str = "0") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    if data_source == "agent_safetybench":
        safety_enable = safety_bench_reward_mode == "clawsentry"
    elif data_source == "agentharm":
        safety_enable = agentharm_reward_mode == "clawsentry"
    else:
        safety_enable = seta_safety_mode == "clawsentry"
    safety_enable = safety_enable and (not evaluation)
    safety_coef = _env_float("SAFETY_REWARD_COEF", 0.0)
    traj_save_interval = _trajectory_save_interval(args)
    safety_summary_weight = _env_float("SAFETY_REWARD_SUMMARY_WEIGHT", 0.3)
    safety_zero_threshold = _env_float(
        "SAFETY_REWARD_ZERO_THRESHOLD", _SAFETY_ZERO_THRESHOLD
    )
    cs_client: ClawSentryClient | None = None
    cs_per_call: list[tuple[int, float]] = []
    cs_per_call_full: list[dict[str, Any]] = []
    turn_records: list[dict[str, Any]] = []

    _log_tag = f"[task={task_spec.task_name} uid={run_ctx.uid} group_idx={run_ctx.group_index} sample_idx={run_ctx.sample_index}]"

    try:
        env_client, lease_id = await _create_env_client(
            task_spec, run_ctx, task_meta=task_meta
        )
        reset_payload = await env_client.reset(
            lease_id=lease_id,
            task_meta=task_meta,
            run_ctx=run_ctx_payload,
            task_timeouts=timeouts_payload,
        )
        user_msg = str(reset_payload.get("user_msg", ""))
        raw_tools = list(reset_payload.get("tool_schemas", []))
        logger.info("%s Start terminal rollout", _log_tag)

        tool_schemas = _normalize_tool_schemas(raw_tools)
        agent_type = str(getattr(args, "terminal_agent_type", "camel_agent"))
        model_type = str(getattr(args, "model_type", "slime-sglang"))
        non_think_mode = bool(getattr(args, "non_think_mode", True))
        non_think_mode_source = str(
            getattr(args, "non_think_mode_source", "prompt")
        ).lower()
        if non_think_mode_source not in {"prompt", "sglang", "both"}:
            non_think_mode_source = "prompt"
        enable_prompt_non_think = non_think_mode and non_think_mode_source in {
            "prompt",
            "both",
        }
        enable_sglang_non_think = non_think_mode and non_think_mode_source in {
            "sglang",
            "both",
        }

        terminal_max_iterations = max(1, int(getattr(args, "max_iteration", 10)))
        terminal_max_parse_errors = max(1, int(getattr(args, "max_parse_errors", 3)))
        max_total_tokens = int(getattr(args, "max_total_tokens", 32768))
        sglang_client = _create_sglang_client(
            args=args,
            tokenizer=state.tokenizer,
            sampling_params=sampling_params,
            max_total_tokens=max_total_tokens,
            enable_sglang_non_think=enable_sglang_non_think,
        )

        if prm_enable:
            prm_router_ip = getattr(args, "prm_router_ip", None)
            prm_router_port = getattr(args, "prm_router_port", None)
            if prm_router_ip and prm_router_port:
                prm_sglang_url = f"http://{prm_router_ip}:{prm_router_port}/generate"
            else:
                prm_sglang_url = getattr(args, "prm_sglang_url", None) or os.getenv(
                    "PRM_SGLANG_URL", ""
                )
            if not prm_sglang_url:
                raise RuntimeError(
                    "prm_enable=True but no PRM endpoint: set prm_router_ip/port, "
                    "prm_sglang_url, or PRM_SGLANG_URL env var."
                )
            prm_sampling_params = {
                "temperature": float(getattr(args, "prm_temperature", 0.0)),
                "max_new_tokens": int(getattr(args, "prm_max_new_tokens", 4096)),
            }
            prm_max_total_tokens = int(getattr(args, "prm_max_total_tokens", 16384))
            prm_sglang_client = _create_sglang_client(
                args=args,
                tokenizer=state.tokenizer,
                sampling_params=prm_sampling_params,
                max_total_tokens=prm_max_total_tokens,
                enable_sglang_non_think=True,
                sglang_url=prm_sglang_url,
                max_retries=10,
            )
            prm_agent = TerminalPRMAgent(
                sglang_client=prm_sglang_client,
                task_instruction=task_spec.instruction,
                history_mode=str(getattr(args, "prm_history_mode", "head_tail")),
            )
            logger.info(
                "%s PRM enabled: url=%s coef=%.3f", _log_tag, prm_sglang_url, prm_coef
            )

        if safety_enable:
            cs_base = os.getenv("CS_HTTP_URL", "http://127.0.0.1:8090")
            cs_session_id = (
                f"openclaw-rl:{task_spec.task_name}:{run_ctx.uid}"
                f":g{run_ctx.group_index}:s{run_ctx.sample_index}"
            )
            cs_timeout = _env_float("SAFETY_REWARD_TIMEOUT", 2.0)
            cs_client = ClawSentryClient(
                base_url=cs_base,
                session_id=cs_session_id,
                agent_id="openclaw-rl-trainer",
                auth_token=os.getenv("CS_AUTH_TOKEN") or None,
                timeout=cs_timeout,
                enabled=True,
            )
            logger.info(
                "%s ClawSentry enabled: url=%s coef=%.3f sid=%s",
                _log_tag,
                cs_base,
                safety_coef,
                cs_session_id,
            )

        agent_runner = create_agent_runner(
            agent_type=agent_type,
            sglang_client=sglang_client,
            model_type=model_type,
            tool_schemas=tool_schemas,
            non_think_mode=enable_prompt_non_think,
            max_total_tokens=max_total_tokens,
        )
        agent_runner.reset(user_msg)
        agent_runner.set_max_parse_errors(terminal_max_parse_errors)
        agent_runner.set_max_iterations(terminal_max_iterations)

        # Loop
        interactions: List[Interaction] = []
        final_model_response = None
        final_response = None
        reached_iteration_limit = False
        reached_parse_error_limit = False

        while True:
            context_result: TurnContext = await agent_runner.get_turn_context()
            if context_result.terminated_response is not None:
                logger.warning("%s Rollout pre-terminated before model turn.", _log_tag)
                final_response = context_result.terminated_response
                break
            if context_result.context_messages is None:
                logger.warning("%s Rollout context is empty; aborting loop.", _log_tag)
                break

            turn_state: TurnResult = await agent_runner.run_model_turn(
                context_result.context_messages
            )
            interaction = turn_state.interaction
            turn_idx = int(interaction.turn_idx)
            interactions.append(interaction)

            current_turn_record: dict[str, Any] = {
                "turn_idx": turn_idx,
                "context_messages": context_result.context_messages,
                "assistant_output": interaction.output_text or "",
                "finish_reason": interaction.finish_reason,
                "latency_ms": float(interaction.latency_ms),
                "n_input_tokens": len(interaction.input_ids or []),
                "n_output_tokens": len(interaction.output_token_ids or []),
                "parse_error_recorded": bool(turn_state.parse_error_recorded),
                "tool_calls": [],
            }
            turn_records.append(current_turn_record)

            if prm_agent is not None:
                tool_calls_for_prm = [
                    {"tool_name": tc.tool_name, "args": tc.args}
                    for tc in (turn_state.tool_call_requests or [])
                ]
                prm_agent.record_model_turn(
                    turn_idx,
                    assistant_text=interaction.output_text or "",
                    tool_calls=tool_calls_for_prm or None,
                    parse_error_recorded=turn_state.parse_error_recorded,
                    finish_reason=interaction.finish_reason,
                )

            if turn_state.terminated_response is not None:
                logger.warning(
                    "%s Rollout terminated during model turn %d.", _log_tag, turn_idx
                )
                final_response = turn_state.terminated_response
                break
            if turn_state.model_response is None:
                logger.warning(
                    "%s Model turn %d returned empty model_response.",
                    _log_tag,
                    turn_idx,
                )
                break

            should_continue_loop = False
            if tool_call_requests := turn_state.tool_call_requests:
                logger.info(
                    "%s Turn %d: executing %d tool call(s).",
                    _log_tag,
                    turn_idx,
                    len(tool_call_requests),
                )
                for tool_call_request in tool_call_requests:
                    assert env_client is not None and lease_id is not None
                    await env_client.heartbeat(lease_id)
                    cs_dec_dict: dict[str, Any] | None = None
                    if cs_client is not None:
                        cs_dec = await cs_client.pre_action(
                            tool_call_request.tool_name,
                            tool_call_request.args,
                        )
                        cs_score = _safety_per_turn_score(
                            cs_dec, zero_threshold=safety_zero_threshold
                        )
                        cs_per_call.append((turn_idx, cs_score))
                        if cs_dec is not None:
                            cs_dec_dict = {
                                "decision": cs_dec.decision,
                                "risk_level": cs_dec.risk_level,
                                "composite_score": cs_dec.composite_score,
                                "reason": cs_dec.reason,
                                "safety_score": cs_score,
                            }
                            cs_per_call_full.append(cs_dec_dict)
                    raw_result = await env_client.exec_tool(
                        lease_id,
                        tool_call_request.tool_name,
                        tool_call_request.args,
                    )
                    agent_runner.record_tool_result(tool_call_request, raw_result)
                    if prm_agent is not None:
                        prm_agent.record_tool_result(
                            turn_idx, tool_call_request, raw_result
                        )
                    current_turn_record["tool_calls"].append({
                        "tool_call_id": getattr(tool_call_request, "tool_call_id", None),
                        "tool_name": tool_call_request.tool_name,
                        "args": tool_call_request.args,
                        "result": raw_result[:4096] if isinstance(raw_result, str) else str(raw_result)[:4096],
                        "clawsentry": cs_dec_dict,
                    })
                should_continue_loop = True

            if turn_state.parse_error_recorded:
                logger.warning(
                    "%s Turn %d: tool-call parse error.",
                    _log_tag,
                    turn_idx,
                )
                should_continue_loop = True

            if prm_agent is not None:
                task = asyncio.create_task(prm_agent.judge_turn(turn_idx))
                prm_pending.append((turn_idx, task))

            if should_continue_loop:
                if (
                    turn_state.parse_error_recorded
                    and agent_runner.reached_parse_error_limit()
                ):
                    logger.error(
                        "%s Max parse errors (%d) reached at turn %d.",
                        _log_tag,
                        agent_runner.max_parse_errors,
                        turn_idx,
                    )
                    reached_parse_error_limit = True
                    final_model_response = turn_state.model_response
                    break
                if agent_runner.reached_iteration_limit():
                    logger.warning(
                        "%s Max iterations (%d) reached.",
                        _log_tag,
                        agent_runner.max_iterations,
                    )
                    reached_iteration_limit = True
                    final_model_response = turn_state.model_response
                    break
                continue

            final_model_response = turn_state.model_response
            break

        if final_response is None and final_model_response is not None:
            final_response = agent_runner.finalize_response(final_model_response)

        if final_response is None:
            logger.error(
                "%s No final response produced; mark sample aborted.", _log_tag
            )
            sample.status = Sample.Status.ABORTED
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            return [sample]

        finish_reasons = final_response.info.get("termination_reasons", [])
        is_aborted = not final_response.msg

        if final_response.terminated and "max_tokens_exceeded" in finish_reasons:
            status = Sample.Status.TRUNCATED
        elif reached_iteration_limit:
            status = Sample.Status.TRUNCATED
        elif reached_parse_error_limit:
            status = Sample.Status.FAILED
        elif is_aborted:
            status = Sample.Status.ABORTED
        else:
            status = Sample.Status.COMPLETED
        logger.info(
            "%s Rollout finished: status=%s turns=%d parse_errors=%d",
            _log_tag,
            status,
            agent_runner.model_turn_count,
            agent_runner.parse_error_count,
        )

        # Evaluation & Reward
        reward = 0.0
        eval_error: str | None = None
        eval_details: dict[str, Any] | None = None

        if (not is_aborted) and status != Sample.Status.FAILED:
            try:
                assert env_client is not None and lease_id is not None
                await env_client.heartbeat(lease_id)
                eval_payload = None
                if data_source in _DIRECT_SCORE_DATA_SOURCES:
                    eval_payload = _build_agent_safetybench_eval_payload(
                        task_meta=task_meta,
                        turn_records=turn_records,
                        final_response=final_response,
                        interactions=interactions,
                        status=status,
                        parse_error_count=agent_runner.parse_error_count,
                    )
                raw_score = await env_client.evaluate(lease_id, trajectory=eval_payload)
                reward = float(raw_score)
                eval_details = _last_eval_details(env_client)
                logger.info("%s Evaluation reward=%.4f", _log_tag, reward)
                if eval_details:
                    reason = eval_details.get("reason")
                    base = eval_details.get("base")
                    split = _safety_split_from_meta(task_meta)
                    logger.info(
                        "%s Reward details: source=%s split=%s mode=%s reason=%s base=%s "
                        "refused=%s verbal_refused=%s tools=%s turns=%s parse_errors=%s",
                        _log_tag,
                        data_source or "seta",
                        split,
                        eval_details.get("mode"),
                        reason,
                        base,
                        eval_details.get("refused"),
                        eval_details.get("verbal_refused", eval_details.get("text_refused")),
                        eval_details.get("n_tool_calls"),
                        eval_details.get("n_turns"),
                        eval_details.get("parse_errors"),
                    )
            except Exception as exc:
                eval_error = f"{type(exc).__name__}: {exc}"
                status = Sample.Status.FAILED
                reward = 0.0
                logger.error(
                    "%s Evaluation failed, marking FAILED: %s",
                    _log_tag,
                    eval_error,
                )

        if not interactions:
            logger.warning("%s No interactions recorded; remove sample.", _log_tag)
            sample.status = status
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            return [sample]

        if prm_agent is not None and prm_pending:
            for turn_idx, prm_task in prm_pending:
                try:
                    output_text, score = await prm_task
                    prm_turn_scores[turn_idx] = float(score)
                    prm_turn_details.append(
                        {
                            "turn_idx": turn_idx,
                            "score": float(score),
                            "output_text": output_text,
                        }
                    )
                    logger.info(
                        "%s PRM judge turn %d score=%.4f, output_text=%s",
                        _log_tag,
                        turn_idx,
                        float(score),
                        output_text.replace("\n", ""),
                    )
                except Exception as exc:
                    logger.warning(
                        "%s PRM judge failed for turn %d (ignored): %s",
                        _log_tag,
                        turn_idx,
                        exc,
                    )
                    prm_turn_scores[turn_idx] = 0.0
                    prm_turn_details.append(
                        {"turn_idx": turn_idx, "score": 0.0, "error": str(exc)}
                    )

        if prm_agent is not None:
            sample.metadata["prm"] = {
                "enabled": True,
                "coef": prm_coef,
                "turn_scores": prm_turn_scores,
                "turn_details": prm_turn_details,
            }

        safety_turn_scores: dict[int, float] | None = None
        if cs_client is not None:
            cs_summary = await cs_client.fetch_summary()
            per_call_scores = [score for (_idx, score) in cs_per_call]
            safety_traj = _safety_trajectory_score(
                per_call_scores,
                cs_summary,
                summary_weight=safety_summary_weight,
                zero_threshold=safety_zero_threshold,
            )
            turn_indices = [it.turn_idx for it in interactions]
            safety_turn_scores = _safety_broadcast(safety_traj, turn_indices)
            cs_stats = cs_client.stats()
            sample.metadata["safety"] = {
                "enabled": True,
                "coef": safety_coef,
                "summary_weight": safety_summary_weight,
                "zero_threshold": safety_zero_threshold,
                "trajectory_score": safety_traj,
                "per_call_scores": cs_per_call,
                "summary_composite_score": (
                    cs_summary.composite_score if cs_summary is not None else None
                ),
                "summary_dimensions": (
                    cs_summary.dimensions if cs_summary is not None else None
                ),
                "n_calls": cs_stats["calls"],
                "n_errors": cs_stats["errors"],
                "decisions": cs_stats["decisions"],
            }
            logger.info(
                "%s ClawSentry trajectory_score=%.4f calls=%d errors=%d",
                _log_tag,
                safety_traj,
                cs_stats["calls"],
                cs_stats["errors"],
            )

        # Build training samples
        dapo_overlong_cfg = _dapo_overlong_cfg(args)
        samples = _build_samples(
            interactions=interactions,
            base_sample=sample,
            outcome=reward,
            status=status,
            prm_turn_scores=(prm_turn_scores if prm_agent is not None else None),
            prm_coef=prm_coef,
            safety_turn_scores=safety_turn_scores,
            safety_coef=safety_coef,
            discount=1.0,
            encourage=False,
            outcome_is_score=(data_source in _DIRECT_SCORE_DATA_SOURCES),
            penalize_short_response=(data_source not in _DIRECT_SCORE_DATA_SOURCES),
            dapo_overlong_cfg=dapo_overlong_cfg,
        )
        if dapo_overlong_cfg is not None:
            logger.info(
                "%s DAPO overlong cfg: max_resp_len=%s buffer_len=%s expected_len=%s penalty_factor=%s",
                _log_tag,
                dapo_overlong_cfg["max_resp_len"],
                dapo_overlong_cfg["buffer_len"],
                dapo_overlong_cfg["expected_len"],
                dapo_overlong_cfg["penalty_factor"],
            )

        # ── Exploration: add intrinsic + safety + LP-RND + CDE actor bonuses (no-op when disabled) ────
        if (
            _EXPLORE_INTRINSIC_ENABLED
            or _EXPLORE_SAFETY_FILTER_ENABLED
            or _EXPLORE_LPRND_ENABLED
            or _EXPLORE_CDE_ACTOR_ENABLED
        ):
            _intr_bonus = _explore_intrinsic_bonus(turn_records)
            _intr_schedule_multiplier = _explore_schedule_multiplier(
                _EXPLORE_INTRINSIC_SCHEDULE,
                run_ctx.train_step,
                _EXPLORE_INTRINSIC_DECAY_STEPS,
            )
            _intr_effective_coef = _EXPLORE_INTRINSIC_COEF * _intr_schedule_multiplier
            _intr_scaled = _intr_bonus * _intr_effective_coef
            _safe_penalty = _explore_safety_penalty(turn_records)
            _lprnd_raw = _explore_lprnd_bonus(interactions)
            _lprnd_schedule_multiplier = _explore_schedule_multiplier(
                _EXPLORE_LPRND_SCHEDULE,
                run_ctx.train_step,
                _EXPLORE_LPRND_DECAY_STEPS,
            )
            _lprnd_effective_coef = _EXPLORE_LPRND_COEF * _lprnd_schedule_multiplier
            _lprnd_bonus = _lprnd_raw * _lprnd_effective_coef
            _base_score_values = []
            for _sample in samples:
                if isinstance(_sample.reward, dict) and "score" in _sample.reward:
                    try:
                        _base_score_values.append(float(_sample.reward["score"]))
                    except (TypeError, ValueError):
                        pass
            _base_score_mean = (
                sum(_base_score_values) / len(_base_score_values)
                if _base_score_values
                else 0.0
            )
            _cde_actor = _explore_cde_actor_metrics(
                interactions,
                _base_score_mean,
                run_ctx.train_step,
            )
            _cde_actor_bonus = _cde_actor["bonus"]
            _explore_total = _intr_scaled + _safe_penalty + _lprnd_bonus + _cde_actor_bonus
            _explore_debug = _explore_debug_metrics(
                status=status,
                base_score_mean=_base_score_mean,
                total_bonus=_explore_total,
                intrinsic_scaled=_intr_scaled,
                safety_penalty=_safe_penalty,
                lprnd_bonus=_lprnd_bonus,
                cde_actor=_cde_actor,
                turn_records=turn_records,
                parse_error_count=agent_runner.parse_error_count,
            )
            for s in samples:
                if isinstance(s.reward, dict) and "score" in s.reward:
                    s.reward["score"] += _explore_total
                    s.reward["explore_intrinsic"] = _intr_bonus
                    s.reward["explore_intrinsic_scaled"] = _intr_scaled
                    s.reward["explore_intrinsic_coef"] = _EXPLORE_INTRINSIC_COEF
                    s.reward["explore_intrinsic_effective_coef"] = _intr_effective_coef
                    s.reward["explore_intrinsic_schedule"] = _EXPLORE_INTRINSIC_SCHEDULE
                    s.reward["explore_intrinsic_decay_steps"] = _EXPLORE_INTRINSIC_DECAY_STEPS
                    s.reward["explore_intrinsic_schedule_multiplier"] = _intr_schedule_multiplier
                    s.reward["explore_intrinsic_granularity"] = _EXPLORE_INTRINSIC_GRANULARITY
                    s.reward["explore_intrinsic_scope"] = _EXPLORE_INTRINSIC_SCOPE
                    s.reward["explore_safety_penalty"] = _safe_penalty
                    s.reward["explore_lprnd"] = _lprnd_bonus
                    s.reward["explore_lprnd_raw"] = _lprnd_raw
                    s.reward["explore_lprnd_coef"] = _EXPLORE_LPRND_COEF
                    s.reward["explore_lprnd_effective_coef"] = _lprnd_effective_coef
                    s.reward["explore_lprnd_schedule"] = _EXPLORE_LPRND_SCHEDULE
                    s.reward["explore_lprnd_decay_steps"] = _EXPLORE_LPRND_DECAY_STEPS
                    s.reward["explore_lprnd_schedule_multiplier"] = _lprnd_schedule_multiplier
                    if _EXPLORE_CDE_ACTOR_ENABLED:
                        s.reward["explore_cde_actor_bonus"] = _cde_actor_bonus
                        s.reward["explore_cde_actor_log_ppl"] = _cde_actor["log_ppl"]
                        s.reward["explore_cde_actor_omega"] = _cde_actor["omega"]
                        s.reward["explore_cde_actor_alpha"] = _cde_actor["alpha"]
                        s.reward["explore_cde_actor_kappa"] = _cde_actor["kappa"]
                        s.reward["explore_cde_actor_reward_gate"] = _EXPLORE_CDE_ACTOR_REWARD_GATE
                        s.reward["explore_cde_actor_eligible"] = _cde_actor["eligible"]
                        s.reward["explore_cde_actor_decay_steps"] = _cde_actor["decay_steps"]
                        s.reward["explore_cde_actor_base_mean"] = _cde_actor["base_score_mean"]
                        s.reward["explore_cde_actor_base_magnitude"] = _cde_actor["base_score_magnitude"]
                        s.reward["explore_cde_actor_cap"] = _cde_actor["cap"]
                        s.reward["explore_cde_actor_scaled"] = _cde_actor["scaled"]
                        s.reward["explore_cde_actor_clipped"] = _cde_actor["clipped"]
                    s.reward["explore_total_bonus"] = _explore_total
                    s.reward.update(_explore_debug)

        for s in samples:
            s.metadata["model_turn_count"] = agent_runner.model_turn_count
            s.metadata["parse_error_count"] = agent_runner.parse_error_count
            s.metadata["data_source"] = data_source or s.metadata.get("data_source")
            s.metadata["safety_split"] = _safety_split_from_meta(task_meta)
            if eval_details is not None:
                s.metadata["reward_details"] = _jsonable(eval_details)
            if eval_error is not None:
                s.metadata["evaluation_failed"] = True
                s.metadata["evaluation_error"] = eval_error
        _mark_non_trainable_samples(samples)

        _save_rollout_artifacts(
            task_spec=task_spec,
            run_ctx=run_ctx,
            sampling_params=sampling_params,
            sample=sample,
            samples=samples,
            status=status,
            raw_score=reward,
            eval_error=eval_error,
            turn_records=turn_records,
            safety_meta=sample.metadata.get("safety") if sample.metadata else None,
            prm_meta=sample.metadata.get("prm") if sample.metadata else None,
            safety_coef=safety_coef,
            prm_coef=prm_coef,
            trajectory_save_interval=traj_save_interval,
        )

        return samples

    except Exception as exc:
        logger.error(
            "%s Generate failed (%s): %s",
            _log_tag,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        sample.status = Sample.Status.FAILED
        sample.remove_sample = True
        sample.reward = {"score": 0.0}

        eos = state.tokenizer.eos_token_id
        if eos is None:
            sample.tokens = []
            sample.response_length = 0
            sample.rollout_log_probs = []
            sample.loss_mask = []
        else:
            sample.tokens = [eos, eos]
            sample.response_length = 1
            sample.rollout_log_probs = [0.0]
            sample.loss_mask = [0]
        return [sample]

    finally:
        for _turn_idx, t in prm_pending:
            if not t.done():
                t.cancel()

        if cs_client is not None:
            try:
                await cs_client.aclose()
            except Exception as exc:
                logger.debug("ClawSentry aclose ignored: %s", exc)

        if env_client is not None and lease_id is not None:
            try:
                await env_client.close(lease_id)
            except Exception as exc:
                logger.debug(
                    "%s Best-effort remote close failed lease=%s: %s",
                    _log_tag,
                    lease_id,
                    exc,
                )
