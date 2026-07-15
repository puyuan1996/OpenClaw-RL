from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .metadata import stable_hash


@dataclass(frozen=True)
class TerminalTransition:
    """A turn-level terminal transition used by the latent world model."""

    trajectory_id: str
    task_name: str | None
    data_source: str | None
    turn_idx: int
    context_messages: list[dict[str, Any]]
    action_text: str
    feedback_text: str
    next_context_messages: list[dict[str, Any]] | None
    done: bool
    reward: float | None
    status: str | None
    source_path: str
    rollout_id: int | None = None
    train_step: int | None = None
    group_index: int | None = None
    sample_index: int | None = None

    @property
    def transition_id(self) -> str:
        return stable_hash(
            {
                "trajectory_id": self.trajectory_id,
                "turn_idx": self.turn_idx,
                "action_text": self.action_text,
                "feedback_text": self.feedback_text,
            }
        )

    @property
    def has_next(self) -> bool:
        return bool(self.next_context_messages)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["transition_id"] = self.transition_id
        value["has_next"] = self.has_next
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TerminalTransition":
        fields = cls.__dataclass_fields__
        return cls(**{key: value.get(key) for key in fields})


def _as_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            messages.append(dict(item))
        else:
            messages.append({"role": "user", "content": str(item)})
    return messages


def _action_text(turn: dict[str, Any]) -> str:
    parts: list[str] = []
    assistant_output = str(turn.get("assistant_output") or "").strip()
    if assistant_output:
        parts.append(assistant_output)
    for call in turn.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool_name") or call.get("name") or "tool")
        args = call.get("args", call.get("arguments", {}))
        rendered = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        signature = f"{name}({rendered})"
        # Some trajectory formats keep the parsed tool call outside the raw
        # assistant output.  Avoid duplicating it when it is already present.
        if signature not in assistant_output:
            parts.append(signature)
    return "\n".join(parts)


def _feedback_text(turn: dict[str, Any], *, status: Any, reward: dict[str, Any]) -> str:
    parts: list[str] = []
    for call in turn.get("tool_calls") or []:
        if not isinstance(call, dict) or call.get("result") is None:
            continue
        name = str(call.get("tool_name") or call.get("name") or "tool")
        parts.append(f"<tool_result name={name}>\n{call.get('result')}\n</tool_result>")
    if parts:
        return "\n\n".join(parts)
    return json.dumps(
        {
            "status": status,
            "score": reward.get("score"),
            "raw_score": reward.get("raw_score"),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _turn_reward(reward: dict[str, Any], turn_idx: int) -> float | None:
    for row in reward.get("per_turn_scores") or []:
        if isinstance(row, dict) and int(row.get("turn_idx", -1)) == turn_idx and row.get("score") is not None:
            return float(row["score"])
    for key in ("score", "base_score", "raw_score"):
        if reward.get(key) is not None:
            return float(reward[key])
    return None


def transitions_from_seta_trajectory(payload: dict[str, Any], *, source_path: str) -> list[TerminalTransition]:
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    reward = payload.get("reward") if isinstance(payload.get("reward"), dict) else {}
    turns = [turn for turn in (payload.get("turns") or []) if isinstance(turn, dict)]
    trajectory_id = str(info.get("uid") or Path(source_path).parent.name or stable_hash(source_path))
    status = str(info.get("status")) if info.get("status") is not None else None
    transitions: list[TerminalTransition] = []
    for index, turn in enumerate(turns):
        turn_idx = int(turn.get("turn_idx", index))
        next_messages = None
        if index + 1 < len(turns):
            next_messages = _as_messages(turns[index + 1].get("context_messages"))
        transitions.append(
            TerminalTransition(
                trajectory_id=trajectory_id,
                task_name=str(info.get("task_name")) if info.get("task_name") is not None else None,
                data_source=str(info.get("data_source")) if info.get("data_source") is not None else None,
                turn_idx=turn_idx,
                context_messages=_as_messages(turn.get("context_messages")),
                action_text=_action_text(turn),
                feedback_text=_feedback_text(turn, status=status, reward=reward),
                next_context_messages=next_messages,
                done=index == len(turns) - 1,
                reward=_turn_reward(reward, turn_idx),
                status=status,
                source_path=source_path,
                rollout_id=info.get("rollout_id"),
                train_step=info.get("train_step"),
                group_index=info.get("group_index"),
                sample_index=info.get("sample_index"),
            )
        )
    return transitions


def _messages_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("context_messages"):
        return _as_messages(record["context_messages"])
    text = record.get("context_text")
    if not text:
        return []
    try:
        parsed = json.loads(str(text))
    except (TypeError, json.JSONDecodeError):
        return [{"role": "user", "content": str(text)}]
    if isinstance(parsed, dict) and "context_messages" in parsed:
        return _as_messages(parsed["context_messages"])
    return _as_messages(parsed)


def transition_from_world_model_record(record: dict[str, Any], *, source_path: str) -> TerminalTransition:
    status = str(record.get("status")) if record.get("status") is not None else None
    reward = record.get("reward_score")
    next_context_messages = None
    if record.get("next_context_messages"):
        next_context_messages = _as_messages(record.get("next_context_messages"))
    elif record.get("next_context_text"):
        next_context_messages = _messages_from_record({"context_text": record.get("next_context_text")})
    return TerminalTransition(
        trajectory_id=str(record.get("uid") or record.get("trajectory_id") or stable_hash(record)),
        task_name=str(record.get("task_name")) if record.get("task_name") is not None else None,
        data_source=str(record.get("data_source")) if record.get("data_source") is not None else None,
        turn_idx=int(record.get("turn_idx", 0) or 0),
        context_messages=_messages_from_record(record),
        action_text=str(record.get("action_text") or ""),
        feedback_text=str(record.get("next_observation_text") or record.get("feedback_text") or ""),
        next_context_messages=next_context_messages,
        done=bool(record.get("done", False)),
        reward=float(reward) if reward is not None else None,
        status=status,
        source_path=source_path,
        rollout_id=record.get("rollout_id"),
        train_step=record.get("train_step"),
        group_index=record.get("group_index"),
        sample_index=record.get("sample_index"),
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _load_pt_records(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "world_model_replay" in payload:
        payload = payload["world_model_replay"]
    if isinstance(payload, dict):
        records = payload.get("records") or payload.get("items")
        if records is not None:
            return [dict(row) for row in records if isinstance(row, dict)]
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    raise ValueError(f"No world-model replay records found in {path}")


def load_terminal_transitions(
    input_path: str | Path,
    *,
    max_trajectories: int | None = None,
    max_transitions: int | None = None,
    require_tool_feedback: bool = False,
) -> list[TerminalTransition]:
    """Load raw SETA ``traj.json``, records JSONL, or replay snapshots."""

    root = Path(input_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"World-model input does not exist: {root}")
    if root.is_dir():
        paths = sorted(root.rglob("traj.json"))
        if not paths:
            paths = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.pt"))
    else:
        paths = [root]
    if max_trajectories is not None:
        paths = paths[: max(0, int(max_trajectories))]

    transitions: list[TerminalTransition] = []
    for path in paths:
        if path.suffix in {".pt", ".pth"}:
            rows = _load_pt_records(path)
            batch = [transition_from_world_model_record(row, source_path=str(path)) for row in rows]
        elif path.suffix == ".jsonl":
            batch = [transition_from_world_model_record(row, source_path=str(path)) for row in _iter_jsonl(path)]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            batch = transitions_from_seta_trajectory(payload, source_path=str(path))
        for transition in batch:
            if require_tool_feedback and "<tool_result" not in transition.feedback_text:
                continue
            if not transition.action_text or not transition.feedback_text:
                continue
            transitions.append(transition)
            if max_transitions is not None and len(transitions) >= int(max_transitions):
                return transitions
    return transitions
