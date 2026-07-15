from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def build_openai_tool_calls(
    tool_call_requests: Iterable[Any],
) -> List[Dict[str, Any]]:
    """Serialize CAMEL tool requests as one OpenAI assistant tool-call list."""
    tool_calls: List[Dict[str, Any]] = []
    for request in tool_call_requests:
        tool_call: Dict[str, Any] = {
            "id": request.tool_call_id,
            "type": "function",
            "function": {
                "name": request.tool_name,
                "arguments": json.dumps(request.args, ensure_ascii=False),
            },
        }
        extra_content = getattr(request, "extra_content", None)
        if extra_content is not None:
            tool_call["extra_content"] = extra_content
        tool_calls.append(tool_call)
    return tool_calls


def build_openai_assistant_tool_message(
    content: str,
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the assistant half of a structured tool-call exchange."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }


def drop_incomplete_tool_call_groups(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop tool-call groups made incomplete by memory-window truncation.

    A valid group consists of one assistant message containing one or more
    ``tool_calls`` and a later tool result for every call id. Tool results
    without their assistant request are also removed.
    """
    valid_assistant_indexes: set[int] = set()
    valid_tool_result_indexes: set[int] = set()
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue

        call_ids = [
            str(call.get("id"))
            for call in message["tool_calls"]
            if call.get("id") is not None
        ]
        if len(call_ids) != len(message["tool_calls"]):
            continue

        remaining_ids = set(call_ids)
        matching_results: List[int] = []
        for result_index in range(index + 1, len(messages)):
            result_message = messages[result_index]
            if result_message.get("role") != "tool":
                break
            result_id = str(result_message.get("tool_call_id"))
            if result_id in remaining_ids:
                remaining_ids.remove(result_id)
                matching_results.append(result_index)
            if not remaining_ids:
                break

        if not remaining_ids:
            valid_assistant_indexes.add(index)
            valid_tool_result_indexes.update(matching_results)

    normalized: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            if index not in valid_assistant_indexes:
                continue
        elif role == "tool" and index not in valid_tool_result_indexes:
            continue
        normalized.append(message)
    return normalized
