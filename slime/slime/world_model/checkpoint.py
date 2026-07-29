from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def validate_world_model_configuration(args: Any) -> None:
    enabled = bool(getattr(args, "world_model_enable", False))
    mode = str(getattr(args, "world_model_mode", "offline"))
    backend = str(getattr(args, "train_backend", "megatron"))
    try:
        coefficient = float(getattr(args, "world_model_loss_coef", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("world-model loss coefficient must be finite and non-negative") from exc
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("world-model loss coefficient must be finite and non-negative")
    if mode not in {"offline", "shadow", "auxiliary"}:
        raise ValueError(f"unknown world-model mode: {mode}")
    if not enabled and (mode != "offline" or coefficient > 0.0):
        raise ValueError("world-model mode/loss requires --world-model-enable")
    if coefficient > 0.0 and mode != "auxiliary":
        raise ValueError("a positive world-model loss coefficient requires --world-model-mode auxiliary")
    if mode == "auxiliary" and coefficient <= 0.0:
        raise ValueError("world-model auxiliary mode requires a positive loss coefficient")
    if coefficient > 0.0 and backend != "megatron":
        raise ValueError("world-model auxiliary loss is supported only by the Megatron backend")


def _validated_indices(raw_indices: Any, *, count: int, name: str) -> list[int]:
    if not isinstance(raw_indices, list):
        raise ValueError(f"checkpoint {name} indices are missing")
    try:
        indices = [int(index) for index in raw_indices]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint {name} indices are invalid") from exc
    if len(indices) != len(set(indices)) or any(index < 0 or index >= count for index in indices):
        raise ValueError(f"checkpoint {name} indices are out of range or duplicated")
    return indices


def _validated_group_holdout_partition(
    split_metadata: Mapping[str, Any],
    *,
    count: int,
) -> tuple[list[int], list[int]]:
    train_indices = _validated_indices(split_metadata.get("train_indices"), count=count, name="train")
    val_indices = _validated_indices(split_metadata.get("val_indices"), count=count, name="val")
    if not train_indices or not val_indices:
        raise ValueError("checkpoint group_holdout split must contain non-empty train and val indices")
    if set(train_indices) & set(val_indices):
        raise ValueError("checkpoint group_holdout train and val indices overlap")
    if set(train_indices) | set(val_indices) != set(range(count)):
        raise ValueError("checkpoint group_holdout train and val indices must partition the cache")
    return train_indices, val_indices


def validate_cache_encoder(
    metadata: Mapping[str, Any] | None,
    cache_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed unless a checkpoint and cache declare the same encoder contract."""
    metadata = metadata if isinstance(metadata, Mapping) else {}
    cache_metadata = cache_metadata if isinstance(cache_metadata, Mapping) else {}
    train_cache_metadata = metadata.get("cache_metadata")
    train_cache_metadata = train_cache_metadata if isinstance(train_cache_metadata, Mapping) else {}
    train_fingerprint = train_cache_metadata.get("encoder_fingerprint_sha256")
    eval_fingerprint = cache_metadata.get("encoder_fingerprint_sha256")
    compatible = bool(train_fingerprint and eval_fingerprint and train_fingerprint == eval_fingerprint)
    if not compatible:
        raise ValueError(
            "checkpoint/cache encoder fingerprint is missing or mismatched; rebuild the cache and checkpoint "
            "with the current cache schema before evaluation"
        )
    return {
        "encoder_compatible": True,
        "encoder_fingerprint_sha256": eval_fingerprint,
    }


def select_evaluation_indices(
    metadata: Mapping[str, Any] | None,
    cache_metadata: Mapping[str, Any] | None,
    *,
    count: int,
    requested_split: str,
) -> tuple[list[int], dict[str, Any]]:
    """Resolve a checkpoint split without applying train indices to another cache."""
    if requested_split not in {"auto", "all", "train", "val"}:
        raise ValueError(f"unknown evaluation split: {requested_split}")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    cache_metadata = cache_metadata if isinstance(cache_metadata, Mapping) else {}
    train_cache_metadata = metadata.get("cache_metadata")
    train_cache_metadata = train_cache_metadata if isinstance(train_cache_metadata, Mapping) else {}
    train_digest = train_cache_metadata.get("input_records_sha256")
    eval_digest = cache_metadata.get("input_records_sha256")
    records_match = bool(train_digest and eval_digest and train_digest == eval_digest)
    encoder_validation = validate_cache_encoder(metadata, cache_metadata)
    encoder_compatible = bool(encoder_validation["encoder_compatible"])
    train_cache_fingerprint = train_cache_metadata.get("cache_fingerprint_sha256")
    eval_cache_fingerprint = cache_metadata.get("cache_fingerprint_sha256")
    same_cache = bool(
        records_match
        and train_cache_fingerprint
        and eval_cache_fingerprint
        and train_cache_fingerprint == eval_cache_fingerprint
    )
    split_metadata = metadata.get("split")
    split_metadata = split_metadata if isinstance(split_metadata, Mapping) else {}
    split_strategy = split_metadata.get("strategy")

    split = requested_split
    if split == "auto":
        val_indices = split_metadata.get("val_indices")
        if not same_cache:
            raise ValueError(
                "split=auto requires the exact hidden cache used for training; use --split all only for an "
                "explicit external-cache or in-sample diagnostic"
            )
        if split_strategy != "group_holdout" or not isinstance(val_indices, list) or not val_indices:
            raise ValueError(
                "split=auto requires a non-empty group_holdout validation split; retrain with complete group "
                "metadata and --val-ratio > 0, or use --split all only for an explicit in-sample diagnostic"
            )
        split = "val"

    if split in {"train", "val"}:
        if not same_cache:
            raise ValueError(f"checkpoint {split} split cannot be applied to a different or unverifiable cache")
        if split_strategy == "group_holdout":
            train_indices, val_indices = _validated_group_holdout_partition(split_metadata, count=count)
            indices = train_indices if split == "train" else val_indices
        else:
            indices = _validated_indices(split_metadata.get(f"{split}_indices"), count=count, name=split)
        if not indices:
            raise ValueError(f"checkpoint {split} split is empty")
        if split == "val":
            scope = "group_heldout" if split_strategy == "group_holdout" else "record_holdout"
        else:
            scope = "in_sample"
    else:
        indices = list(range(count))
        if same_cache:
            scope = "in_sample_all"
        elif records_match:
            scope = "same_records_different_cache"
        elif train_digest and eval_digest:
            scope = "external_cache_unverified_disjointness"
        else:
            scope = "unknown_provenance"

    return indices, {
        "requested": requested_split,
        "resolved": split,
        "scope": scope,
        "record_count": len(indices),
        "same_cache_as_training": same_cache,
        "records_match_training": records_match,
        "encoder_compatible": encoder_compatible,
        "group_disjoint": split == "val" and split_strategy == "group_holdout",
        "split_strategy": split_strategy,
        "split_group_key": split_metadata.get("group_key"),
    }


def value_head_training_status(metadata: Mapping[str, Any] | None) -> tuple[str, str]:
    """Classify value-head training evidence without trusting legacy metadata."""
    if not isinstance(metadata, Mapping):
        return "unknown_legacy", "checkpoint metadata is missing"

    hyperparameters = metadata.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        return "unknown_legacy", "checkpoint hyperparameters are missing"
    try:
        value_coef = float(hyperparameters.get("value_coef", 0.0))
    except (TypeError, ValueError):
        return "unknown_legacy", "checkpoint value_coef is invalid"
    if not math.isfinite(value_coef) or value_coef <= 0.0:
        return "verified_untrained", "checkpoint value_coef must be positive"

    required_metadata = {
        "train_count",
        "val_count",
        "optimizer_step_count",
        "value_update_step_count",
        "final_train_loss",
        "train_reward_label_count",
    }
    if any(key not in metadata for key in required_metadata) or any(
        key not in hyperparameters for key in ("epochs", "lr")
    ):
        return "unknown_legacy", "checkpoint lacks verifiable value-head training metadata"

    try:
        epochs = int(hyperparameters.get("epochs", 0))
        learning_rate = float(hyperparameters.get("lr", 0.0))
        train_count = int(metadata.get("train_count", 0))
        optimizer_step_count = int(metadata.get("optimizer_step_count", 0))
        value_update_step_count = int(metadata.get("value_update_step_count", 0))
        final_train_loss = float(metadata.get("final_train_loss"))
    except (TypeError, ValueError):
        return "unknown_legacy", "checkpoint training metadata is invalid"
    if epochs <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        return "verified_untrained", "checkpoint has no positive-step training configuration"
    if train_count <= 0 or not math.isfinite(final_train_loss):
        return "verified_untrained", "checkpoint has no completed finite training epoch"
    if optimizer_step_count <= 0 or value_update_step_count <= 0:
        return "verified_untrained", "checkpoint has no verified value-head optimizer updates"
    if metadata.get("has_reward") is not True:
        return "verified_untrained", "checkpoint has no reward labels"

    try:
        train_reward_label_count = int(metadata.get("train_reward_label_count"))
    except (TypeError, ValueError):
        return "unknown_legacy", "checkpoint train_reward_label_count is invalid"
    if train_reward_label_count <= 0:
        return "verified_untrained", "checkpoint train split has no valid reward labels"
    return "verified_trained", "trained with reward supervision"


def trained_value_head_status(metadata: Mapping[str, Any] | None) -> tuple[bool, str]:
    """Return whether checkpoint metadata proves that the value head was trained."""
    status, reason = value_head_training_status(metadata)
    return status == "verified_trained", reason
