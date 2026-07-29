import json

import pytest
import torch

from slime.world_model.evaluate_probe import evaluate_probe
from slime.world_model.modules import TextLatentWorldModel, TextLatentWorldModelConfig


def _cache_metadata():
    return {
        "input_records_sha256": "records",
        "cache_fingerprint_sha256": "cache",
        "encoder_fingerprint_sha256": "encoder",
    }


def _write_checkpoint(path, *, hidden_dim=4, latent_dim=3, metadata=None):
    config = TextLatentWorldModelConfig(
        state_hidden_dim=hidden_dim,
        action_hidden_dim=hidden_dim,
        target_hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        sigreg_num_proj=4,
    )
    model = TextLatentWorldModel(config)
    checkpoint_metadata = dict(metadata or {})
    checkpoint_metadata["cache_metadata"] = _cache_metadata()
    torch.save({"config": config.__dict__, "state_dict": model.state_dict(), "metadata": checkpoint_metadata}, path)


def test_evaluate_probe_single_sample_marks_shuffle_unavailable(tmp_path):
    torch.manual_seed(0)
    ckpt_path = tmp_path / "probe.pt"
    cache_path = tmp_path / "cache.pt"
    out_path = tmp_path / "eval.json"
    _write_checkpoint(ckpt_path)
    torch.save(
        {
            "state_hidden": torch.randn(1, 4),
            "action_hidden": torch.randn(1, 4),
            "target_hidden": torch.randn(1, 4),
            "metadata": _cache_metadata(),
        },
        cache_path,
    )

    summary = evaluate_probe(
        checkpoint=ckpt_path,
        cache=cache_path,
        output=out_path,
        device_name="cpu",
        bootstrap_samples=0,
        split="all",
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert summary["record_count"] == 1
    assert payload["metrics"]["shuffled_action_available"] is False
    assert payload["metrics"]["shuffled_action_reason"] == "fewer_than_two_samples"
    assert payload["metrics"]["shuffle_gap_mse_shuffled_minus_real"] is None
    assert payload["metrics"]["pred_mse_real"] is not None


def test_evaluate_probe_uses_reward_mask_and_constant_reward_reason(tmp_path):
    torch.manual_seed(0)
    ckpt_path = tmp_path / "probe.pt"
    cache_path = tmp_path / "cache.pt"
    out_path = tmp_path / "eval.json"
    _write_checkpoint(
        ckpt_path,
        metadata={
            "has_reward": True,
            "reward_mask_count": 2,
            "train_reward_label_count": 2,
            "train_count": 2,
            "val_count": 0,
            "final_train_loss": 0.1,
            "optimizer_step_count": 1,
            "value_update_step_count": 1,
            "hyperparameters": {"value_coef": 0.05, "epochs": 1, "lr": 1e-4},
        },
    )
    torch.save(
        {
            "state_hidden": torch.randn(3, 4),
            "action_hidden": torch.randn(3, 4),
            "target_hidden": torch.randn(3, 4),
            "reward": torch.tensor([1.0, 1.0, 99.0]),
            "reward_mask": torch.tensor([True, True, False]),
            "record_metadata": [
                {"uid": "u1", "task_name": "task", "status": "completed", "has_tool_result": True},
                {"uid": "u2", "task_name": "task", "status": "completed", "has_tool_result": False},
                {"uid": "u3", "task_name": "task", "status": "failed", "has_tool_result": False},
            ],
            "metadata": _cache_metadata(),
        },
        cache_path,
    )

    summary = evaluate_probe(
        checkpoint=ckpt_path,
        cache=cache_path,
        output=out_path,
        device_name="cpu",
        bootstrap_samples=8,
        split="all",
    )
    metrics = summary["metrics"]

    assert summary["counts"]["n_reward_masked"] == 2
    assert metrics["shuffled_action_available"] is True
    assert metrics["shuffle_gap_mse_shuffled_minus_real"] is not None
    assert metrics["action_delta"] is not None
    assert metrics["value_reward"]["reward_mask_count"] == 2
    assert metrics["value_reward"]["spearman"] is None
    assert metrics["value_reward"]["reason"] == "constant_input"
    assert metrics["uncertainty_error"]["available"] is False
    assert metrics["uncertainty_error"]["reason"] == "uncertainty_head_has_no_dedicated_training_objective"
    assert summary["record_metadata_summary"]["status_hist"]["completed"] == 2


def test_evaluate_probe_rejects_empty_cache(tmp_path):
    ckpt_path = tmp_path / "probe.pt"
    cache_path = tmp_path / "cache.pt"
    out_path = tmp_path / "eval.json"
    _write_checkpoint(ckpt_path)
    torch.save(
        {
            "state_hidden": torch.empty(0, 4),
            "action_hidden": torch.empty(0, 4),
            "target_hidden": torch.empty(0, 4),
        },
        cache_path,
    )

    with pytest.raises(ValueError, match="No cached world-model records"):
        evaluate_probe(checkpoint=ckpt_path, cache=cache_path, output=out_path, device_name="cpu")


def test_evaluate_probe_reports_legacy_value_metric_as_gate_ineligible(tmp_path):
    torch.manual_seed(0)
    ckpt_path = tmp_path / "legacy_probe.pt"
    cache_path = tmp_path / "cache.pt"
    out_path = tmp_path / "eval.json"
    _write_checkpoint(ckpt_path)
    torch.save(
        {
            "state_hidden": torch.randn(3, 4),
            "action_hidden": torch.randn(3, 4),
            "target_hidden": torch.randn(3, 4),
            "reward": torch.tensor([-1.0, 0.0, 1.0]),
            "reward_mask": torch.tensor([True, True, True]),
            "metadata": _cache_metadata(),
        },
        cache_path,
    )

    summary = evaluate_probe(
        checkpoint=ckpt_path,
        cache=cache_path,
        output=out_path,
        device_name="cpu",
        bootstrap_samples=0,
        split="all",
    )
    value_reward = summary["metrics"]["value_reward"]

    assert value_reward["available"] is True
    assert value_reward["gate_eligible"] is False
    assert value_reward["training_status"] == "unknown_legacy"
    assert value_reward["reason"].startswith("gate_ineligible_unknown_legacy")
