#!/usr/bin/env python3
"""Export Terminal-RL AgentSafetyBench trajectories for official ShieldAgent scoring."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_final_assistant_text(traj: dict, meta: dict) -> str:
    turns = traj.get("turns") or []
    if turns:
        text = turns[-1].get("assistant_output")
        if isinstance(text, str) and text:
            return text

    preview = ((meta.get("reward_details") or {}).get("final_text_preview") or "")
    # Handle BaseMessage(... content='...') previews as a fallback.
    match = re.search(r"content=(['\"])(.*)\1", preview, flags=re.DOTALL)
    if match:
        return match.group(2)
    return preview


def extract_output_messages(traj: dict, meta: dict) -> list[dict]:
    messages: list[dict] = []
    turns = traj.get("turns") or []
    for turn in turns:
        assistant_output = turn.get("assistant_output")
        tool_calls = [call for call in turn.get("tool_calls") or [] if isinstance(call, dict)]
        if tool_calls:
            assistant_msg = {"role": "assistant", "tool_calls": []}
            for idx, call in enumerate(tool_calls):
                call_id = str(call.get("tool_call_id") or call.get("id") or f"call_{turn.get('turn_idx', 0)}_{idx}")
                args = call.get("args")
                if args is None:
                    args = call.get("arguments", call.get("tool_arguments", {}))
                assistant_msg["tool_calls"].append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": str(call.get("tool_name") or call.get("name") or ""),
                            "arguments": json.dumps(args or {}, ensure_ascii=False),
                        },
                    }
                )
            messages.append(assistant_msg)
            for idx, call in enumerate(tool_calls):
                call_id = str(call.get("tool_call_id") or call.get("id") or f"call_{turn.get('turn_idx', 0)}_{idx}")
                result = call.get("result")
                if result is None:
                    result = call.get("content", call.get("observation", call.get("output", "")))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": str(result),
                    }
                )
            continue

        parsed_calls: list[dict] = []
        if isinstance(assistant_output, str):
            for idx, match in enumerate(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", assistant_output, re.DOTALL)):
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                parsed_calls.append(
                    {
                        "id": f"parsed_tool_call_{idx}",
                        "type": "function",
                        "function": {
                            "name": payload.get("name") or "",
                            "arguments": json.dumps(payload.get("arguments") or {}, ensure_ascii=False),
                        },
                    }
                )

        if parsed_calls:
            messages.append({"role": "assistant", "tool_calls": parsed_calls})
            continue

        if isinstance(assistant_output, str) and assistant_output:
            messages.append({"role": "assistant", "content": assistant_output})
    if messages:
        return messages
    return [{"role": "assistant", "content": extract_final_assistant_text(traj, meta)}]


def parse_fulfillable(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return 1 if value else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "benign_should_comply"}:
        return 1
    if text in {"0", "false", "no", "harmful_should_refuse"}:
        return 0
    return default


def source_dialog(meta: dict, sample_meta: dict) -> list[dict]:
    dialog = sample_meta.get("dialog")
    if isinstance(dialog, list) and dialog:
        return dialog
    instruction = meta.get("instruction") or sample_meta.get("instruction") or ""
    if instruction:
        return [{"role": "user", "content": instruction}]
    return []


def official_asb_id(meta: dict, sample_meta: dict) -> int:
    for value in (
        sample_meta.get("id"),
        sample_meta.get("task_name"),
        meta.get("task_name"),
    ):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    for value in (sample_meta.get("task_path"), meta.get("task_path")):
        if not value:
            continue
        suffix = str(value).rstrip("/").rsplit("/", 1)[-1]
        try:
            return int(suffix)
        except (TypeError, ValueError):
            pass

    return int(meta.get("sample_index", 0))


def convert_one(meta_path: Path) -> dict | None:
    meta = load_json(meta_path)
    if (meta.get("dataset_slug") or meta.get("data_source")) != "agent_safetybench":
        return None
    traj_path = meta_path.with_name("traj.json")
    traj = load_json(traj_path) if traj_path.exists() else {}
    sample_meta = meta.get("sample_metadata") or {}
    reward_details = meta.get("reward_details") or {}
    fulfillable = parse_fulfillable(sample_meta.get("fulfillable", reward_details.get("fulfillable", 0)))
    safety_split = "benign_should_comply" if fulfillable == 1 else "harmful_should_refuse"
    official_id = official_asb_id(meta, sample_meta)

    return {
        "id": official_id,
        "terminal_rl_uid": meta.get("uid"),
        "terminal_rl_sample_index": meta.get("sample_index"),
        "terminal_rl_run_task": meta.get("task_name"),
        "risks": sample_meta.get("risks") or [],
        "instruction": meta.get("instruction") or sample_meta.get("instruction") or "",
        "environments": sample_meta.get("environments") or [],
        "failure_modes": sample_meta.get("failure_modes", ""),
        "fulfillable": fulfillable,
        "safety_split": safety_split,
        # Official scorer uses 0=safe, 1=unsafe for optional metric reporting.
        "author_human_label": 0 if fulfillable == 1 else 1,
        "dialog": source_dialog(meta, sample_meta),
        "output": extract_output_messages(traj, meta),
        "terminal_rl_reward_details": reward_details,
        "terminal_rl_status": meta.get("status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--filename", default="gen_res.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    items: list[dict] = []
    for meta_path in sorted((args.run_dir / "trajectories").glob("*/meta.json")):
        item = convert_one(meta_path)
        if item is None:
            continue
        items.append(item)
        if args.limit and len(items) >= args.limit:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(items)} AgentSafetyBench examples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
