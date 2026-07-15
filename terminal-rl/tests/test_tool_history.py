from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

TERMINAL_RL_DIR = Path(__file__).resolve().parents[1]
if str(TERMINAL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL_DIR))

from agent.tool_history import (  # noqa: E402
    build_openai_assistant_tool_message,
    build_openai_tool_calls,
    drop_incomplete_tool_call_groups,
)


def test_build_openai_tool_calls_keeps_parallel_calls_in_one_list():
    requests = [
        SimpleNamespace(
            tool_call_id="call_1",
            tool_name="shell",
            args={"command": "printf 你好"},
            extra_content=None,
        ),
        SimpleNamespace(
            tool_call_id="call_2",
            tool_name="read_file",
            args={"path": "/tmp/a"},
            extra_content={"provider": "test"},
        ),
    ]

    assert build_openai_tool_calls(requests) == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": '{"command": "printf 你好"}',
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "/tmp/a"}',
            },
            "extra_content": {"provider": "test"},
        },
    ]


def test_build_openai_assistant_tool_message_preserves_reasoning_content():
    tool_calls = [{"id": "call_1", "type": "function"}]

    assert build_openai_assistant_tool_message(
        "<think>full current-turn reasoning</think>", tool_calls
    ) == {
        "role": "assistant",
        "content": "<think>full current-turn reasoning</think>",
        "tool_calls": tool_calls,
    }


def test_drop_incomplete_tool_call_groups_keeps_complete_parallel_group():
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [{"id": "a"}, {"id": "b"}],
        },
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
    ]

    assert drop_incomplete_tool_call_groups(messages) == messages


def test_drop_incomplete_tool_call_groups_removes_orphans_atomically():
    user = {"role": "user", "content": "inspect"}
    messages = [
        user,
        {"role": "tool", "tool_call_id": "old", "content": "orphan"},
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [{"id": "a"}, {"id": "b"}],
        },
        {"role": "tool", "tool_call_id": "b", "content": "partial"},
        {"role": "assistant", "content": "ordinary reply"},
    ]

    assert drop_incomplete_tool_call_groups(messages) == [
        user,
        {"role": "assistant", "content": "ordinary reply"},
    ]


def test_drop_incomplete_tool_call_groups_requires_adjacent_results():
    messages = [
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [{"id": "a"}],
        },
        {"role": "assistant", "content": "interposed"},
        {"role": "tool", "tool_call_id": "a", "content": "late"},
    ]

    assert drop_incomplete_tool_call_groups(messages) == [
        {"role": "assistant", "content": "interposed"}
    ]
