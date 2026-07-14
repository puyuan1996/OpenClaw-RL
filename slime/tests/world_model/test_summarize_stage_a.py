import json
from types import SimpleNamespace

import torch

from slime.world_model.cache_text_hidden import _build_cache_integrity_metadata
from slime.world_model.summarize_stage_a import _summarize_bucket


def _write_json(path, payload):
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _make_stage_bucket(tmp_path, *, evaluation_scope):
    bucket_dir = tmp_path / "clean"
    (bucket_dir / "logs").mkdir(parents=True)
    (bucket_dir / "records.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    _write_json(
        bucket_dir / "records_summary.json",
        {"record_count": 2, "context_text_unique_count": 2, "context_truncated_ratio": 0.0},
    )
    cache_payload = {
        "state_hidden": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        "action_hidden": torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
        "target_hidden": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "record_count": 2,
        "record_metadata": [{"context_hash": "a"}, {"context_hash": "b"}],
    }
    cache_payload["metadata"] = _build_cache_integrity_metadata(
        cache_payload,
        input_records_sha256="records",
        encoder_config={"encoder": "test", "behavior_probe_sha256": "probe"},
    )
    torch.save(cache_payload, bucket_dir / "cached_hidden.pt")
    (bucket_dir / "probe.pt").write_bytes(b"checkpoint")
    _write_json(
        bucket_dir / "eval_summary.json",
        {
            "record_count": 2,
            "evaluation_split": {"scope": evaluation_scope},
            "metrics": {
                "latents": {"state": {"effective_rank": 2.0, "variance_mean": 0.1}},
                "shuffle_gap_mse_shuffled_minus_real": 0.1,
                "zero_action_gap_mse_zero_minus_real": 0.1,
                "action_delta": 0.1,
            },
        },
    )
    _write_json(bucket_dir / "stage_a_config.json", {"value_coef": 0.0})
    (bucket_dir / "logs" / "stage_a.log").write_text("done\n", encoding="utf-8")


def _args(*, allow_non_group_heldout=False):
    return SimpleNamespace(
        min_records=2,
        min_full_records=2,
        min_clean_records=2,
        min_tool_records=2,
        min_context_unique=2,
        max_context_truncated_ratio=0.5,
        min_state_hidden_pairwise_l2=1e-4,
        min_state_latent_rank=1.0,
        min_state_latent_var=1e-9,
        min_shuffle_gap=0.0,
        min_zero_action_gap=0.0,
        min_action_delta=0.0,
        require_execution_rankings=False,
        allow_non_group_heldout=allow_non_group_heldout,
    )


def test_stage_gate_accepts_group_heldout_scope(tmp_path):
    _make_stage_bucket(tmp_path, evaluation_scope="group_heldout")

    row = _summarize_bucket(tmp_path, "clean", _args())

    assert row["checks"]["group_heldout_ok"] is True
    assert row["passed"] is True


def test_stage_gate_rejects_in_sample_scope_by_default(tmp_path):
    _make_stage_bucket(tmp_path, evaluation_scope="in_sample_all")

    row = _summarize_bucket(tmp_path, "clean", _args())

    assert row["checks"]["group_heldout_ok"] is False
    assert row["passed"] is False
    assert "group_heldout_ok" in row["failed_checks"]


def test_stage_gate_rejects_missing_context_truncation_stat(tmp_path):
    _make_stage_bucket(tmp_path, evaluation_scope="group_heldout")
    summary_path = tmp_path / "clean" / "records_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("context_truncated_ratio")
    _write_json(summary_path, summary)

    row = _summarize_bucket(tmp_path, "clean", _args())

    assert row["checks"]["context_not_fully_truncated"] is False
    assert row["passed"] is False


def test_stage_gate_rejects_tampered_cache(tmp_path):
    _make_stage_bucket(tmp_path, evaluation_scope="group_heldout")
    cache_path = tmp_path / "clean" / "cached_hidden.pt"
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    payload["state_hidden"][0, 0] = 99.0
    torch.save(payload, cache_path)

    row = _summarize_bucket(tmp_path, "clean", _args())

    assert row["checks"]["state_hidden_ok"] is False
    assert row["passed"] is False
