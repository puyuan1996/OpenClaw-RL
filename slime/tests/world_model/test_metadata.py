from types import SimpleNamespace

from slime.world_model.metadata import attach_terminal_world_model_metadata, is_world_model_enabled


def _sample():
    return SimpleNamespace(
        tokens=[1, 2, 3, 4, 5],
        response_length=2,
        reward={"score": 0.5, "base_score": 0.5, "raw_score": 1.0},
        metadata={"turn_idx": 0, "num_turns": 2},
        train_metadata=None,
    )


def test_attach_terminal_world_model_metadata_default_off():
    sample = _sample()
    attach_terminal_world_model_metadata(
        args=SimpleNamespace(world_model_enable=False),
        samples=[sample],
        turn_records=[],
        task_meta={},
        run_ctx=SimpleNamespace(),
        status=SimpleNamespace(value="completed"),
    )
    assert "world_model" not in sample.metadata
    assert sample.train_metadata is None


def test_world_model_enable_ignores_env_side_channel(monkeypatch):
    monkeypatch.setenv("WORLD_MODEL_ENABLE", "1")

    assert is_world_model_enabled(SimpleNamespace(world_model_enable=False)) is False


def test_attach_terminal_world_model_metadata_enabled():
    sample = _sample()
    attach_terminal_world_model_metadata(
        args=SimpleNamespace(world_model_enable=True, world_model_metadata_max_chars=128),
        samples=[sample],
        turn_records=[
            {
                "turn_idx": 0,
                "context_messages": [{"role": "user", "content": "hi"}],
                "assistant_output": "run tool",
                "tool_calls": [{"tool_name": "bash", "args": {"command": "pwd"}, "result": "/tmp"}],
            }
        ],
        task_meta={"task_name": "task", "data_source": "unit"},
        run_ctx=SimpleNamespace(uid="u", group_index=1, sample_index=2, rollout_id=3, train_step=4),
        status=SimpleNamespace(value="completed"),
    )
    wm = sample.metadata["world_model"]
    assert wm["schema"] == "openclaw_terminal_latent_world_model_v2"
    assert wm["action_text"]
    assert wm["next_observation_text"] == "/tmp"
    assert wm["done"] is False
    assert sample.train_metadata["world_model"] == wm


def test_context_text_preserves_tail_for_long_common_prefix():
    sample = _sample()
    long_system = "common-prefix-" * 200
    task_tail = "unique task state should survive"
    attach_terminal_world_model_metadata(
        args=SimpleNamespace(world_model_enable=True, world_model_metadata_max_chars=256),
        samples=[sample],
        turn_records=[
            {
                "turn_idx": 0,
                "context_messages": [
                    {"role": "system", "content": long_system},
                    {"role": "user", "content": task_tail},
                ],
                "assistant_output": "run",
                "tool_calls": [],
            }
        ],
        task_meta={"task_name": "task"},
        run_ctx=SimpleNamespace(),
        status=SimpleNamespace(value="completed"),
    )

    wm = sample.metadata["world_model"]
    assert len(wm["context_text"]) <= 256
    assert "[openclaw_truncated_middle]" in wm["context_text"]
    assert task_tail in wm["context_text"]
    assert wm["context_text_source"] == "context_messages.head_tail"


def test_attach_terminal_world_model_metadata_normalizes_non_dict_metadata():
    sample = _sample()
    sample.metadata = "legacy"
    attach_terminal_world_model_metadata(
        args=SimpleNamespace(world_model_enable=True, world_model_metadata_max_chars=128),
        samples=[sample],
        turn_records=[
            {
                "turn_idx": 0,
                "context_messages": [{"role": "user", "content": "hi"}],
                "assistant_output": "run",
                "tool_calls": [],
            }
        ],
        task_meta={},
        run_ctx=SimpleNamespace(),
        status=SimpleNamespace(value="completed"),
    )

    assert isinstance(sample.metadata, dict)
    assert sample.metadata["world_model"]["action_text"] == "run"
