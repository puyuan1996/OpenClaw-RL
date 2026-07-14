from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from slime.world_model.checkpoint import (
    select_evaluation_indices,
    trained_value_head_status,
    validate_world_model_configuration,
    value_head_training_status,
)
from slime.world_model.candidate_set_eval import (
    _group_records,
    _require_candidate_groups,
    _reward_label_contract,
    _validate_reward_label_contract,
    _validate_candidate_group_scope,
)
from slime.world_model.train_probe import _reward_label_count, _split_dataset


def _trained_metadata(**overrides):
    metadata = {
        "has_reward": True,
        "reward_mask_count": 4,
        "train_reward_label_count": 3,
        "train_count": 6,
        "val_count": 2,
        "final_train_loss": 0.1,
        "optimizer_step_count": 3,
        "value_update_step_count": 2,
        "hyperparameters": {"value_coef": 0.05, "epochs": 3, "lr": 1e-4},
    }
    metadata.update(overrides)
    return metadata


def _cache_metadata(*, records="digest", cache="cache", encoder="encoder"):
    return {
        "input_records_sha256": records,
        "cache_fingerprint_sha256": cache,
        "encoder_fingerprint_sha256": encoder,
    }


def test_group_records_drops_missing_reward_candidates():
    records = [
        {"context_hash": "ctx", "reward_score": 1.0},
        {"context_hash": "ctx", "reward_score": None},
        {"context_hash": "ctx", "reward_score": -1.0},
    ]

    groups = _group_records(
        records,
        group_key="context_hash",
        min_candidates=2,
        max_candidates=8,
        require_reward_variation=True,
    )

    assert groups == [[0, 2]]


def test_candidate_eval_rejects_empty_candidate_groups():
    with pytest.raises(ValueError, match="no eligible candidate groups"):
        _require_candidate_groups([])


def test_candidate_eval_requires_group_key_to_match_heldout_key():
    with pytest.raises(ValueError, match="split_group_key"):
        _validate_candidate_group_scope(
            "task_name",
            {"scope": "group_heldout", "split_group_key": "context_hash"},
        )


def test_candidate_eval_accepts_matching_heldout_group_key():
    _validate_candidate_group_scope(
        "context_hash",
        {"scope": "group_heldout", "split_group_key": "context_hash"},
    )


def test_reward_label_contract_rejects_mixed_semantics():
    records = [
        {"reward_label_source": "sample.reward.score", "reward_label_semantics": "training_reward_unspecified"},
        {"reward_label_source": "execution.status", "reward_label_semantics": "execution_outcome"},
    ]

    with pytest.raises(ValueError, match="inconsistent reward label contracts"):
        _reward_label_contract(records)


def test_candidate_eval_requires_verified_execution_labels_by_default():
    contract = {"verified_execution_outcome": False}
    with pytest.raises(ValueError, match="verified as execution outcomes"):
        _validate_reward_label_contract(contract, allow_unverified=False)

    assert _validate_reward_label_contract(contract, allow_unverified=True) is False


def test_world_model_configuration_rejects_silent_noop_combinations():
    invalid = [
        SimpleNamespace(world_model_enable=False, world_model_mode="offline", world_model_loss_coef=0.1),
        SimpleNamespace(world_model_enable=True, world_model_mode="offline", world_model_loss_coef=0.1),
        SimpleNamespace(world_model_enable=True, world_model_mode="auxiliary", world_model_loss_coef=0.0),
        SimpleNamespace(
            world_model_enable=True,
            world_model_mode="auxiliary",
            world_model_loss_coef=0.1,
            train_backend="fsdp",
        ),
    ]
    for args in invalid:
        with pytest.raises(ValueError):
            validate_world_model_configuration(args)


def test_world_model_configuration_accepts_offline_and_megatron_auxiliary():
    validate_world_model_configuration(
        SimpleNamespace(world_model_enable=True, world_model_mode="offline", world_model_loss_coef=0.0)
    )
    validate_world_model_configuration(
        SimpleNamespace(
            world_model_enable=True,
            world_model_mode="auxiliary",
            world_model_loss_coef=0.1,
            train_backend="megatron",
        )
    )


def test_trained_value_head_status_requires_positive_coef_and_labels():
    trained, reason = trained_value_head_status(_trained_metadata())

    assert trained is True
    assert reason == "trained with reward supervision"


def test_trained_value_head_status_rejects_empty_reward_mask():
    trained, reason = trained_value_head_status(_trained_metadata(train_reward_label_count=0))

    assert trained is False
    assert reason == "checkpoint train split has no valid reward labels"


def test_trained_value_head_status_rejects_zero_epoch_checkpoint():
    metadata = _trained_metadata()
    metadata["hyperparameters"]["epochs"] = 0

    trained, reason = trained_value_head_status(metadata)

    assert trained is False
    assert reason == "checkpoint has no positive-step training configuration"


def test_trained_value_head_status_rejects_missing_completed_epoch():
    trained, reason = trained_value_head_status(_trained_metadata(final_train_loss=None))

    assert trained is False
    assert reason == "checkpoint training metadata is invalid"


def test_trained_value_head_status_rejects_zero_value_updates():
    trained, reason = trained_value_head_status(_trained_metadata(value_update_step_count=0))

    assert trained is False
    assert reason == "checkpoint has no verified value-head optimizer updates"


def test_value_head_training_status_marks_legacy_metadata_unknown():
    status, reason = value_head_training_status({"hyperparameters": {"value_coef": 0.05}})

    assert status == "unknown_legacy"
    assert reason == "checkpoint lacks verifiable value-head training metadata"


def test_reward_label_count_uses_train_subset_mask():
    dataset = TensorDataset(
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4),
        torch.tensor([True, False, True, False]),
    )

    assert _reward_label_count(Subset(dataset, [1, 2, 3])) == 1


def test_group_holdout_keeps_contexts_disjoint():
    dataset = TensorDataset(torch.arange(6).float().unsqueeze(-1))
    rows = [{"context_hash": value} for value in ["a", "a", "b", "b", "c", "c"]]

    train, val, split = _split_dataset(
        dataset,
        val_ratio=0.34,
        seed=7,
        record_metadata=rows,
        group_key="context_hash",
    )

    assert val is not None
    train_groups = {rows[index]["context_hash"] for index in train.indices}
    val_groups = {rows[index]["context_hash"] for index in val.indices}
    assert train_groups.isdisjoint(val_groups)
    assert split["strategy"] == "group_holdout"
    assert split["group_disjoint"] is True


def test_partial_group_metadata_falls_back_to_record_holdout():
    dataset = TensorDataset(torch.arange(4).float().unsqueeze(-1))
    rows = [{"context_hash": "a"}, {}, {"context_hash": "b"}, {"context_hash": "b"}]

    _train, val, split = _split_dataset(
        dataset,
        val_ratio=0.25,
        seed=7,
        record_metadata=rows,
        group_key="context_hash",
    )

    assert val is not None
    assert split["strategy"] == "record_holdout_fallback"
    assert split["group_values_complete"] is False
    assert split["group_disjoint"] is False


def test_select_evaluation_indices_prefers_matching_validation_split():
    metadata = _trained_metadata(
        cache_metadata=_cache_metadata(),
        split={"strategy": "group_holdout", "group_key": "context_hash", "train_indices": [0, 1], "val_indices": [2]},
    )

    indices, info = select_evaluation_indices(
        metadata,
        _cache_metadata(),
        count=3,
        requested_split="auto",
    )

    assert indices == [2]
    assert info["scope"] == "group_heldout"
    assert info["group_disjoint"] is True
    assert info["resolved"] == "val"


def test_select_evaluation_indices_auto_rejects_non_group_validation():
    metadata = _trained_metadata(
        cache_metadata=_cache_metadata(),
        split={
            "strategy": "record_holdout_fallback",
            "group_key": "context_hash",
            "train_indices": [0, 1],
            "val_indices": [2],
        },
    )

    with pytest.raises(ValueError, match="non-empty group_holdout"):
        select_evaluation_indices(metadata, _cache_metadata(), count=3, requested_split="auto")


def test_select_evaluation_indices_auto_rejects_same_records_with_different_hidden_cache():
    metadata = _trained_metadata(
        cache_metadata=_cache_metadata(cache="train-cache"),
        split={
            "strategy": "group_holdout",
            "group_key": "context_hash",
            "train_indices": [0, 1],
            "val_indices": [2],
        },
    )

    with pytest.raises(ValueError, match="exact hidden cache"):
        select_evaluation_indices(
            metadata,
            _cache_metadata(cache="reencoded-cache"),
            count=3,
            requested_split="auto",
        )


def test_select_evaluation_indices_labels_external_cache_as_unverified():
    metadata = _trained_metadata(cache_metadata=_cache_metadata(records="train", cache="train-cache"))

    indices, info = select_evaluation_indices(
        metadata,
        _cache_metadata(records="eval", cache="eval-cache"),
        count=2,
        requested_split="all",
    )

    assert indices == [0, 1]
    assert info["scope"] == "external_cache_unverified_disjointness"
    assert info["group_disjoint"] is False


def test_select_evaluation_indices_rejects_encoder_mismatch():
    metadata = _trained_metadata(cache_metadata=_cache_metadata(encoder="train-encoder"))

    with pytest.raises(ValueError, match="encoder fingerprint"):
        select_evaluation_indices(
            metadata,
            _cache_metadata(encoder="eval-encoder"),
            count=2,
            requested_split="all",
        )


def test_select_evaluation_indices_rejects_training_indices_on_external_cache():
    metadata = _trained_metadata(
        cache_metadata=_cache_metadata(records="train", cache="train-cache"),
        split={"train_indices": [0], "val_indices": [1]},
    )

    with pytest.raises(ValueError, match="different or unverifiable cache"):
        select_evaluation_indices(
            metadata,
            _cache_metadata(records="eval", cache="eval-cache"),
            count=2,
            requested_split="val",
        )
