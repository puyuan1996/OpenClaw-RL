from __future__ import annotations

from pathlib import Path


TERMINAL_RL = Path(__file__).resolve().parents[1]


def test_training_script_keeps_a3s_code_on_terminal_rl_rollout_path() -> None:
    script = (TERMINAL_RL / "terminal-rl_qwen3-8b_pu.sh").read_text(encoding="utf-8")

    assert 'cfg["harness_option"] = harness_option' in script
    assert '--custom-generate-function-path generate.generate' in script
    assert '--custom-rollout-log-function-path rollout_log.rollout_log' in script
    assert "code_rl_rollout.generate_rollout_code_rl" not in script
    assert "code_rl_api_server.generate" not in script
    assert "code_rl_api_server.reward_func" not in script
    assert "start_a3s_code_traffic" not in script
    assert "run_a3s_code_agent_traffic.sh" not in script
    removed_script = "run_" "a3s_code_rl_4gpu.sh"
    assert removed_script not in script


def test_generate_reads_harness_option_before_legacy_agent_type() -> None:
    source = (TERMINAL_RL / "generate.py").read_text(encoding="utf-8")

    assert '"harness_option"' in source
    assert '"terminal_agent_type"' in source
    assert "create_agent_runner(" in source
