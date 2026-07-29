from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


_SECRET_NAME = (
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|password|passwd|secret|"
    r"openai_api_key|github_token|gh_token|hf_token|aws_access_key_id|aws_secret_access_key|"
    r"[a-z0-9_]*(?:_token|_secret|_password|_api_key))"
)
_QUOTED_SECRET_RE = re.compile(rf"([\"']{_SECRET_NAME}[\"']\s*:\s*[\"'])(.*?)([\"'])", re.IGNORECASE)
_QUOTED_ASSIGNMENT_RE = re.compile(rf"(\b{_SECRET_NAME}\b\s*=\s*[\"'])(.*?)([\"'])", re.IGNORECASE)
_BARE_SECRET_RE = re.compile(rf"(\b{_SECRET_NAME}\b\s*[:=]\s*)([^\s,;}}\"']+)", re.IGNORECASE)
_AUTHORIZATION_SCHEME_RE = re.compile(
    r"(?i)(\bauthorization\b\s*[:=]\s*)(?:basic|bearer|token)\s+[a-z0-9._~+/:=-]{4,}"
)
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[a-z0-9._~+/=-]{8,}")
_URL_CREDENTIAL_RE = re.compile(r"(?i)(https?://[^:/\s]+:)[^@\s]+@")
_TOKEN_LITERAL_RE = re.compile(
    r"(?i)\b(?:sk-[a-z0-9_-]{12,}|gh[pousr]_[a-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"
)


def is_world_model_enabled(args: Any) -> bool:
    raw = getattr(args, "world_model_enable", False)
    if isinstance(raw, str):
        raw = raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def stable_hash(value: Any, *, digest_size: int = 16) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_size).hexdigest()


def canonicalize_context_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Derive split identity from the canonical text passed to the encoder tokenizer."""
    normalized = dict(record)
    context_text = normalized.get("context_text")
    if context_text is not None and not isinstance(context_text, str):
        context_text = json.dumps(context_text, ensure_ascii=False, sort_keys=True, default=str)
        normalized["context_text"] = context_text
    previous_hash = normalized.get("context_hash")
    if not context_text:
        if previous_hash:
            normalized["source_context_hash"] = previous_hash
        normalized["context_hash"] = None
        normalized["context_hash_schema"] = "missing_context_text"
        return normalized
    canonical_hash = stable_hash(context_text)
    if previous_hash and previous_hash != canonical_hash:
        normalized["source_context_hash"] = previous_hash
    normalized["context_hash"] = canonical_hash
    normalized["context_hash_schema"] = "canonical_context_text_v1"
    return normalized


def redact_sensitive_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = _AUTHORIZATION_SCHEME_RE.sub(r"\1[REDACTED]", text)
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _URL_CREDENTIAL_RE.sub(r"\1[REDACTED]@", text)
    text = _QUOTED_SECRET_RE.sub(r"\1[REDACTED]\3", text)
    text = _QUOTED_ASSIGNMENT_RE.sub(r"\1[REDACTED]\3", text)
    text = _BARE_SECRET_RE.sub(r"\1[REDACTED]", text)
    return _TOKEN_LITERAL_RE.sub("[REDACTED]", text)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truncate(text: Any, max_chars: int, *, strategy: str = "head") -> str:
    text = "" if text is None else str(text)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if strategy == "tail":
        return text[-max_chars:]
    if strategy == "head_tail":
        marker = "\n[openclaw_truncated_middle]\n"
        if max_chars <= len(marker) + 2:
            return text[:max_chars]
        keep = max_chars - len(marker)
        head_chars = max(1, keep // 4)
        tail_chars = keep - head_chars
        return text[:head_chars] + marker + text[-tail_chars:]
    return text[:max_chars]


def truncate_head_tail_text(value: Any, max_chars: int) -> str:
    """Bound stored text while retaining evidence from both ends."""
    return _truncate(value, max_chars, strategy="head_tail")


def _tool_action_text(call: dict[str, Any], max_chars: int) -> str:
    name = str(call.get("tool_name") or call.get("name") or "tool")
    args = call.get("args")
    if args is None:
        args = call.get("arguments")
    args_text = json.dumps(_jsonable(args), ensure_ascii=False, sort_keys=True, default=str)
    return _truncate(redact_sensitive_text(f"{name}({args_text})"), max_chars, strategy="head_tail")


def _extract_action_text(turn: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    assistant = str(turn.get("assistant_output") or "").strip()
    if assistant:
        parts.append(redact_sensitive_text(assistant))
    for call in turn.get("tool_calls") or []:
        if isinstance(call, dict):
            parts.append(_tool_action_text(call, max_chars))
    return _truncate("\n".join(parts), max_chars, strategy="head_tail")


def _extract_context_text(turn: dict[str, Any], *, task_meta: dict[str, Any], max_chars: int) -> str:
    context_messages = turn.get("context_messages") or []
    payload = {
        "task_name": task_meta.get("task_name"),
        "task_path": task_meta.get("task_path"),
        "data_source": task_meta.get("data_source"),
        "turn_idx": turn.get("turn_idx"),
        "context_messages": context_messages,
    }
    return _truncate(
        redact_sensitive_text(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, default=str)),
        max_chars,
        strategy="head_tail",
    )


def _extract_observation_text(
    turn: dict[str, Any],
    *,
    status: Any,
    reward: dict[str, Any],
    eval_details: dict[str, Any] | None,
    eval_error: str | None,
    max_chars: int,
    is_terminal: bool,
) -> str:
    observations: list[str] = []
    for call in turn.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("result") is not None:
            result_text = redact_sensitive_text(call.get("result"))
            observations.append(result_text or '{"status": "tool_result", "result": ""}')
    if not observations:
        if is_terminal:
            reason = None
            if isinstance(eval_details, dict):
                reason = eval_details.get("reason") or eval_details.get("message")
            observations.append(
                json.dumps(
                    {
                        "status": getattr(status, "value", str(status)),
                        "score": _finite_number(reward.get("score")),
                        "raw_score": _finite_number(reward.get("raw_score")),
                        "eval_reason": redact_sensitive_text(reason) if reason else None,
                        "eval_error": redact_sensitive_text(eval_error) if eval_error else None,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
        else:
            observations.append('{"status": "no_tool_result"}')
    return _truncate("\n".join(observations), max_chars, strategy="head_tail")


def build_terminal_world_model_record(
    *,
    sample: Any,
    turn: dict[str, Any],
    task_meta: dict[str, Any],
    run_ctx: Any,
    status: Any,
    eval_details: dict[str, Any] | None,
    eval_error: str | None,
    max_chars: int,
    is_last_turn: bool | None = None,
) -> dict[str, Any]:
    reward = sample.reward if isinstance(getattr(sample, "reward", None), dict) else {}
    sample_metadata = getattr(sample, "metadata", {}) if isinstance(getattr(sample, "metadata", {}), dict) else {}
    turn_idx = int(turn.get("turn_idx", sample_metadata.get("turn_idx", 0) or 0))
    num_turns = sample_metadata.get("num_turns")
    if num_turns is not None:
        final_turn = turn_idx >= int(num_turns) - 1
    else:
        final_turn = bool(is_last_turn)
    context_text = _extract_context_text(turn, task_meta=task_meta, max_chars=max_chars)
    action_text = _extract_action_text(turn, max_chars)
    next_observation_text = _extract_observation_text(
        turn,
        status=status,
        reward=reward,
        eval_details=eval_details,
        eval_error=eval_error,
        max_chars=max_chars,
        is_terminal=final_turn,
    )
    context_messages = turn.get("context_messages") or []
    response_length = int(getattr(sample, "response_length", 0) or 0)
    token_count = len(getattr(sample, "tokens", []) or [])
    terminal_status = getattr(status, "value", str(status))
    status_value = terminal_status if final_turn else "in_progress"
    return {
        "schema": "openclaw_text_jepa_world_model_v2",
        "hidden_source": "cached_or_frozen_encoder",
        "task_name": task_meta.get("task_name"),
        "task_path": task_meta.get("task_path"),
        "data_source": task_meta.get("data_source"),
        "uid": getattr(run_ctx, "uid", None),
        "group_index": getattr(run_ctx, "group_index", None),
        "sample_index": getattr(run_ctx, "sample_index", None),
        "rollout_id": getattr(run_ctx, "rollout_id", None),
        "train_step": getattr(run_ctx, "train_step", None),
        "turn_idx": turn_idx,
        "num_turns": num_turns,
        "status": status_value,
        "trajectory_status": terminal_status,
        "done": bool(final_turn),
        "context_hash": stable_hash(context_text),
        "action_hash": stable_hash(action_text),
        "next_observation_hash": stable_hash(next_observation_text),
        "context_token_len": max(0, token_count - response_length),
        "action_token_len": response_length,
        "context_text": context_text,
        "context_text_source": "context_messages.head_tail",
        "context_text_truncation": "head_tail",
        "action_text": action_text,
        "next_observation_text": next_observation_text,
        "reward_score": _finite_number(reward.get("score")),
        "reward_base_score": _finite_number(reward.get("base_score")),
        "reward_raw_score": _finite_number(reward.get("raw_score")),
        "reward_label_scope": "turn_return_or_step_score",
        "reward_label_source": "sample.reward.score",
        "reward_label_semantics": "training_reward_unspecified",
        "reward_label_is_execution_outcome": None,
        "reward_label_terminal": bool(final_turn),
        "has_tool_result": any(
            isinstance(call, dict) and call.get("result") is not None
            for call in (turn.get("tool_calls") or [])
        ),
    }


def attach_terminal_world_model_metadata(
    *,
    args: Any,
    samples: list[Any],
    turn_records: list[dict[str, Any]],
    task_meta: dict[str, Any],
    run_ctx: Any,
    status: Any,
    eval_details: dict[str, Any] | None = None,
    eval_error: str | None = None,
) -> None:
    if not is_world_model_enabled(args) or not samples:
        return
    if any(
        isinstance(turn, dict) and len(turn.get("sdk_model_turns") or []) > 1
        for turn in (turn_records or [])
    ):
        for sample in samples:
            skip = {
                "schema": "openclaw_text_jepa_world_model_skip_v1",
                "reason": "multi_interaction_turn_requires_harness_adapter",
            }
            metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
            sample.metadata = metadata
            sample.metadata["world_model_skipped"] = skip
            train_metadata = dict(sample.train_metadata or {})
            train_metadata["world_model_skipped"] = skip
            sample.train_metadata = train_metadata
        return
    max_chars = int(getattr(args, "world_model_metadata_max_chars", 4096) or 4096)
    turn_by_idx = {int(turn.get("turn_idx", i)): turn for i, turn in enumerate(turn_records or [])}
    last_turn_idx = max(turn_by_idx, default=None)
    for sample in samples:
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        sample.metadata = metadata
        turn_idx = int(metadata.get("turn_idx", 0) or 0)
        turn = turn_by_idx.get(turn_idx)
        if turn is None:
            continue
        record = build_terminal_world_model_record(
            sample=sample,
            turn=turn,
            task_meta=task_meta,
            run_ctx=run_ctx,
            status=status,
            eval_details=eval_details,
            eval_error=eval_error,
            max_chars=max_chars,
            is_last_turn=last_turn_idx is not None and turn_idx == last_turn_idx,
        )
        sample.metadata["world_model"] = record
        train_metadata = dict(sample.train_metadata or {})
        train_metadata["world_model"] = record
        sample.train_metadata = train_metadata
