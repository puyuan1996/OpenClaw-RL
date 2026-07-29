import json

import torch

from slime.world_model.build_dataset import (
    _observation_source,
    extract_world_model_records,
    summarize_world_model_records,
)
from slime.world_model.metadata import stable_hash


def test_no_tool_placeholder_is_not_terminal_eval_summary():
    record = {"next_observation_text": '{"status": "no_tool_result"}', "has_tool_result": False}

    assert _observation_source(record) == "no_tool_result"


def test_extract_world_model_records_backfills_legacy_context_text(tmp_path):
    payload = {
        "samples": [
            {
                "status": "completed",
                "prompt": [{"role": "user", "content": "state text"}],
                "metadata": {},
                "train_metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "context_hash": "abc",
                        "action_text": "act",
                        "next_observation_text": "obs",
                    }
                },
            }
        ]
    }
    path = tmp_path / "rollout.pt"
    torch.save(payload, path)

    records = extract_world_model_records(path)

    assert len(records) == 1
    assert json.loads(records[0]["context_text"])[0]["content"] == "state text"
    assert records[0]["context_text_source"] == "sample.prompt"
    assert records[0]["context_hash"] == stable_hash(records[0]["context_text"])
    assert records[0]["source_context_hash"] == "abc"


def test_extract_world_model_records_can_repair_prefix_truncated_context(tmp_path):
    bad_context = "shared system prompt " * 100
    payload = {
        "samples": [
            {
                "status": "completed",
                "prompt": [{"role": "user", "content": "task-specific state"}],
                "metadata": {},
                "train_metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "context_hash": "abc",
                        "context_text": bad_context[:128],
                        "context_text_source": "context_messages",
                        "action_text": "act",
                        "next_observation_text": "obs",
                    }
                },
            }
        ]
    }
    path = tmp_path / "rollout.pt"
    torch.save(payload, path)

    records = extract_world_model_records(path, context_max_chars=128, context_source="sample_prompt")

    assert len(records) == 1
    assert "task-specific state" in records[0]["context_text"]
    assert "shared system prompt" not in records[0]["context_text"]
    assert records[0]["context_text_source"] == "sample.prompt"


def test_auto_context_repair_preserves_v2_head_tail_context(tmp_path):
    context_text = ("head" * 20) + "[openclaw_truncated_middle]" + ("tail" * 20)
    payload = {
        "samples": [
            {
                "status": "completed",
                "prompt": [{"role": "user", "content": "initial prompt"}],
                "metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v2",
                        "context_text": context_text,
                        "context_text_source": "context_messages.head_tail",
                        "context_text_truncation": "head_tail",
                        "action_text": "act",
                        "next_observation_text": "obs",
                    }
                },
            }
        ]
    }
    path = tmp_path / "rollout.pt"
    torch.save(payload, path)

    records = extract_world_model_records(
        path,
        context_max_chars=128,
        context_source="auto",
    )

    assert records[0]["context_text"] == context_text
    assert records[0]["context_text_source"] == "context_messages.head_tail"


def test_extract_world_model_records_redacts_repaired_legacy_prompt(tmp_path):
    secret = "sk-abcdefghijklmnop"
    payload = {
        "samples": [
            {
                "status": "completed",
                "prompt": [{"role": "user", "content": f"OPENAI_API_KEY={secret}"}],
                "metadata": {},
                "train_metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "context_hash": "legacy",
                        "action_text": "Authorization: Basic dXNlcjpwYXNz",
                        "next_observation_text": "password=hunter2",
                    }
                },
            }
        ]
    }
    path = tmp_path / "rollout.pt"
    torch.save(payload, path)

    records = extract_world_model_records(path, context_source="sample_prompt")

    assert len(records) == 1
    assert secret not in records[0]["context_text"]
    assert "dXNlcjpwYXNz" not in records[0]["action_text"]
    assert "hunter2" not in records[0]["next_observation_text"]
    assert "[REDACTED]" in records[0]["context_text"]


def test_extract_world_model_records_unifies_legacy_and_v2_context_hashes(tmp_path):
    context_text = '[{"content":"same state","role":"user"}]'
    payload = {
        "samples": [
            {
                "status": "completed",
                "metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "context_hash": "legacy-hash",
                        "context_text": context_text,
                    }
                },
            },
            {
                "status": "completed",
                "metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v2",
                        "context_hash": "different-source-hash",
                        "context_text": context_text,
                    }
                },
            },
        ]
    }
    path = tmp_path / "mixed.pt"
    torch.save(payload, path)

    records = extract_world_model_records(path)

    assert [record["context_hash"] for record in records] == [stable_hash(context_text)] * 2
    assert all(record["context_hash_schema"] == "canonical_context_text_v1" for record in records)


def test_extract_world_model_records_filters_and_summary(tmp_path):
    payload = {
        "samples": [
            {
                "status": "completed",
                "prompt": "state one",
                "metadata": {},
                "train_metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "status": "completed",
                        "uid": "u1",
                        "task_name": "task-a",
                        "context_hash": "ctx1",
                        "action_text": "act",
                        "next_observation_text": "tool result",
                        "has_tool_result": True,
                        "reward_score": 1.0,
                    }
                },
            },
            {
                "status": "failed",
                "prompt": "state two",
                "metadata": {
                    "world_model": {
                        "schema": "openclaw_text_jepa_world_model_v1",
                        "status": "failed",
                        "uid": "u2",
                        "task_name": "task-b",
                        "context_hash": "ctx2",
                        "action_text": "bad act",
                        "next_observation_text": json.dumps({"eval_reason": "eval_timeout"}),
                        "has_tool_result": False,
                        "reward_score": -2.0,
                    }
                },
                "train_metadata": {},
            },
        ]
    }
    path = tmp_path / "rollout.pt"
    torch.save(payload, path)

    all_records = extract_world_model_records(path)
    filtered = extract_world_model_records(
        path,
        statuses={"completed"},
        exclude_eval_reasons={"eval_timeout"},
        require_tool_result=True,
    )
    summary = summarize_world_model_records(filtered, input_record_count=len(all_records), filter_args={"bucket": "clean"})

    assert len(all_records) == 2
    assert len(filtered) == 1
    assert filtered[0]["uid"] == "u1"
    assert summary["record_count"] == 1
    assert summary["input_record_count"] == 2
    assert summary["dropped_record_count"] == 1
    assert summary["filter_args"]["bucket"] == "clean"
