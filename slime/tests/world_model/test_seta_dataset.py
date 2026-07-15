import json

from slime.world_model.seta_dataset import load_terminal_transitions, transitions_from_seta_trajectory


def _trajectory():
    return {
        "info": {
            "uid": "trajectory-1",
            "task_name": "42",
            "data_source": "terminal_bench",
            "status": "Status.COMPLETED",
            "rollout_id": 3,
            "train_step": 6,
        },
        "reward": {"score": 1.0},
        "turns": [
            {
                "turn_idx": 0,
                "context_messages": [{"role": "user", "content": "inspect"}],
                "assistant_output": "run pwd",
                "tool_calls": [{"tool_name": "bash", "args": {"command": "pwd"}, "result": "/tmp"}],
            },
            {
                "turn_idx": 1,
                "context_messages": [
                    {"role": "user", "content": "inspect"},
                    {"role": "tool", "content": "/tmp"},
                ],
                "assistant_output": "done",
                "tool_calls": [],
            },
        ],
    }


def test_seta_transition_boundaries_include_next_context():
    transitions = transitions_from_seta_trajectory(_trajectory(), source_path="/tmp/traj.json")

    assert len(transitions) == 2
    assert transitions[0].action_text.startswith("run pwd")
    assert "/tmp" in transitions[0].feedback_text
    assert transitions[0].has_next is True
    assert transitions[0].next_context_messages[-1]["role"] == "tool"
    assert transitions[1].done is True
    assert transitions[1].has_next is False


def test_load_terminal_transitions_reads_trajectory_directory(tmp_path):
    run = tmp_path / "sample"
    run.mkdir()
    (run / "traj.json").write_text(json.dumps(_trajectory()), encoding="utf-8")

    transitions = load_terminal_transitions(tmp_path, max_transitions=1)

    assert len(transitions) == 1
    assert transitions[0].trajectory_id == "trajectory-1"
