from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .cache_text_hidden import validate_hidden_cache_integrity
from .checkpoint import select_evaluation_indices, value_head_training_status
from .metrics import action_delta, effective_rank
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig


def _load_checkpoint(path: Path, device: torch.device) -> tuple[TextLatentWorldModel, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected dict checkpoint in {path}, got {type(ckpt).__name__}")
    config = TextLatentWorldModelConfig(**ckpt["config"])
    model = TextLatentWorldModel(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("metadata", {})


def _load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}")
    validate_hidden_cache_integrity(payload)
    required = ["state_hidden", "action_hidden", "target_hidden"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}. Expected {required}.")
    count = int(payload["state_hidden"].shape[0])
    if count == 0:
        raise ValueError(f"No cached world-model records found in {path}.")
    for key in required:
        if int(payload[key].shape[0]) != count:
            raise ValueError(
                f"Inconsistent cache length for {key}: expected {count}, got {int(payload[key].shape[0])}"
            )
    return payload


def _to_device(payload: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state_hidden": payload["state_hidden"].float().to(device),
        "action_hidden": payload["action_hidden"].float().to(device),
        "target_hidden": payload["target_hidden"].float().to(device),
    }


def _subset_payload(payload: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    count = int(payload["state_hidden"].shape[0])
    index_tensor = torch.tensor(indices, dtype=torch.long)
    subset: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == count:
            subset[key] = value[index_tensor]
        elif key == "record_metadata" and isinstance(value, list) and len(value) == count:
            subset[key] = [value[index] for index in indices]
        else:
            subset[key] = value
    subset["record_count"] = len(indices)
    return subset


def _float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().float().cpu().item()
    value = float(value)
    return value if math.isfinite(value) else None


def _mse_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.float() - y.float()).pow(2).mean(dim=-1)


def _cosine_distance_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    return 1.0 - (x * y).sum(dim=-1)


def _tensor_stats(x: torch.Tensor) -> dict[str, float | list[int]]:
    x = x.detach().float()
    return {
        "shape": [int(dim) for dim in x.shape],
        "effective_rank": _float(effective_rank(x)),
        "mean_l2_norm": _float(x.norm(dim=-1).mean()),
        "std_l2_norm": _float(x.norm(dim=-1).std(unbiased=False)),
        "mean_abs": _float(x.abs().mean()),
        "variance_mean": _float(x.var(dim=0, unbiased=False).mean()) if x.size(0) > 1 else 0.0,
    }


def _non_identity_permutation(count: int, *, seed: int, device: torch.device) -> torch.Tensor | None:
    if count < 2:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(count, generator=generator)
    if torch.equal(perm, torch.arange(count)):
        perm = torch.roll(perm, shifts=1, dims=0)
    return perm.to(device)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> tuple[float | None, str | None]:
    x = x.detach().float().flatten().cpu()
    y = y.detach().float().flatten().cpu()
    if x.numel() < 2 or y.numel() < 2:
        return None, "fewer_than_two_samples"
    if x.numel() != y.numel():
        return None, "length_mismatch"
    x = x - x.mean()
    y = y - y.mean()
    denom = x.pow(2).mean().sqrt() * y.pow(2).mean().sqrt()
    if float(denom) <= 0.0:
        return None, "constant_input"
    return float((x * y).mean() / denom), None


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().flatten().cpu()
    ranks = torch.empty_like(values)
    order = torch.argsort(values)
    sorted_values = values[order]
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> tuple[float | None, str | None]:
    if x.numel() < 2 or y.numel() < 2:
        return None, "fewer_than_two_samples"
    return _pearson(_rankdata(x), _rankdata(y))


def _masked_reward(payload: dict[str, Any], device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    reward = payload.get("reward")
    if reward is None:
        return None, None
    reward = reward.float().to(device)
    mask = payload.get("reward_mask")
    if mask is None:
        mask = torch.ones_like(reward, dtype=torch.bool, device=device)
    else:
        mask = mask.bool().to(device)
    return reward, mask


def _bootstrap_ci(values: torch.Tensor, *, seed: int, samples: int) -> dict[str, float | None | int]:
    values = values.detach().float().flatten().cpu()
    if samples <= 0 or values.numel() < 2:
        return {"samples": int(samples), "mean": _float(values.mean()) if values.numel() else None, "low": None, "high": None}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    means = []
    for _ in range(samples):
        idx = torch.randint(0, values.numel(), (values.numel(),), generator=generator)
        means.append(values[idx].mean())
    stacked = torch.stack(means)
    return {
        "samples": int(samples),
        "mean": _float(values.mean()),
        "low": _float(torch.quantile(stacked, 0.025)),
        "high": _float(torch.quantile(stacked, 0.975)),
    }


def _hist(values: list[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(values).items()}


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _record_metadata_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("record_metadata") or []
    if not isinstance(rows, list):
        return {"available": False, "reason": "record_metadata_not_list"}
    uid_values = [row.get("uid") for row in rows if isinstance(row, dict) and row.get("uid") is not None]
    task_values = [
        row.get("task_name") or row.get("task_path")
        for row in rows
        if isinstance(row, dict) and (row.get("task_name") or row.get("task_path"))
    ]
    return {
        "available": True,
        "record_metadata_count": len(rows),
        "uid_count": len(set(uid_values)),
        "task_count": len(set(task_values)),
        "status_hist": _hist([row.get("status") for row in rows if isinstance(row, dict)]),
        "has_tool_result_hist": _hist([bool(row.get("has_tool_result")) for row in rows if isinstance(row, dict)]),
        "records_per_uid": _hist(uid_values),
        "records_per_task": _hist(task_values),
    }


def evaluate_probe(
    *,
    checkpoint: Path,
    cache: Path,
    output: Path,
    device_name: str = "auto",
    seed: int = 42,
    bootstrap_samples: int = 500,
    split: str = "auto",
) -> dict[str, Any]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, checkpoint_metadata = _load_checkpoint(checkpoint, device)
    payload = _load_cache(cache)
    indices, evaluation_split = select_evaluation_indices(
        checkpoint_metadata,
        payload.get("metadata"),
        count=int(payload["state_hidden"].shape[0]),
        requested_split=split,
    )
    payload = _subset_payload(payload, indices)
    tensors = _to_device(payload, device)
    count = int(tensors["state_hidden"].shape[0])

    with torch.no_grad():
        out = model(**tensors)
        state_latent = out["state_latent"]
        action_latent = out["action_latent"]
        pred_latent = out["pred_latent"]
        target_latent = out["target_latent"]
        if target_latent is None:
            raise ValueError("target_hidden is required for evaluate_probe.")

        real_mse_per = _mse_per_sample(pred_latent, target_latent)
        real_cos_per = _cosine_distance_per_sample(pred_latent, target_latent)

        zero_action_pred = model.predictor(state_latent, torch.zeros_like(action_latent))
        zero_mse_per = _mse_per_sample(zero_action_pred, target_latent)
        zero_cos_per = _cosine_distance_per_sample(zero_action_pred, target_latent)

        target_mean = target_latent.mean(dim=0, keepdim=True).expand_as(target_latent)
        target_mean_mse_per = _mse_per_sample(target_mean, target_latent)
        state_latent_mse_per = _mse_per_sample(state_latent, target_latent)
        action_latent_mse_per = _mse_per_sample(action_latent, target_latent)

        perm = _non_identity_permutation(count, seed=seed, device=device)
        if perm is not None:
            shuffled_pred = model.predictor(state_latent, action_latent[perm])
            shuffled_mse_per = _mse_per_sample(shuffled_pred, target_latent)
            shuffled_cos_per = _cosine_distance_per_sample(shuffled_pred, target_latent)
            shuffled_available = True
            shuffled_reason = None
            shuffled_gap_per = shuffled_mse_per - real_mse_per
            shuffled_cos_gap_per = shuffled_cos_per - real_cos_per
            delta = action_delta(pred_latent, shuffled_pred)
        else:
            shuffled_pred = None
            shuffled_mse_per = None
            shuffled_cos_per = None
            shuffled_available = False
            shuffled_reason = "fewer_than_two_samples"
            shuffled_gap_per = None
            shuffled_cos_gap_per = None
            delta = None

        reward, reward_mask = _masked_reward(payload, device)
        value = out["value"]
        value_reward: dict[str, Any]
        value_training_status, value_reason = value_head_training_status(checkpoint_metadata)
        if value_training_status == "verified_untrained":
            value_reward = {
                "available": False,
                "gate_eligible": False,
                "training_status": value_training_status,
                "reason": f"value_head_not_trained: {value_reason}",
                "reward_mask_count": 0 if reward_mask is None else int(reward_mask.sum().item()),
            }
        elif reward is None or value is None:
            value_reward = {
                "available": False,
                "gate_eligible": False,
                "training_status": value_training_status,
                "reason": "missing_reward_or_value_head",
                "reward_mask_count": 0,
            }
        else:
            mask = reward_mask if reward_mask is not None else torch.ones_like(reward, dtype=torch.bool, device=device)
            valid_reward = reward[mask]
            valid_value = value[mask]
            if valid_reward.numel() < 2:
                corr, corr_reason = None, "fewer_than_two_masked_rewards"
            else:
                corr, corr_reason = _spearman(valid_value, valid_reward)
            metric_reason = corr_reason
            if value_training_status == "unknown_legacy":
                metric_reason = f"gate_ineligible_unknown_legacy: {value_reason}"
                if corr_reason is not None:
                    metric_reason += f"; metric_reason={corr_reason}"
            value_reward = {
                "available": valid_reward.numel() >= 2 and corr is not None,
                "gate_eligible": value_training_status == "verified_trained" and corr is not None,
                "training_status": value_training_status,
                "reason": metric_reason,
                "reward_mask_count": int(mask.sum().item()),
                "spearman": corr,
                "mse": _float(_mse_per_sample(valid_value, valid_reward).mean()) if valid_reward.numel() else None,
                "mae": _float((valid_value.float() - valid_reward.float()).abs().mean()) if valid_reward.numel() else None,
                "reward_mean": _float(valid_reward.mean()) if valid_reward.numel() else None,
                "value_mean": _float(valid_value.mean()) if valid_value.numel() else None,
            }

        uncertainty_error = {
            "available": False,
            "reason": "uncertainty_head_has_no_dedicated_training_objective",
            "spearman_uncertainty_vs_pred_mse": None,
            "uncertainty_mean": None,
        }

    metrics: dict[str, Any] = {
        "pred_mse_real": _float(real_mse_per.mean()),
        "pred_cosine_distance_real": _float(real_cos_per.mean()),
        "zero_action_pred_mse": _float(zero_mse_per.mean()),
        "zero_action_gap_mse_zero_minus_real": _float(zero_mse_per.mean() - real_mse_per.mean()),
        "zero_action_cosine_distance": _float(zero_cos_per.mean()),
        "no_action_baseline": {
            "diagnostic_target_mean_mse_eval_leaky": _float(target_mean_mse_per.mean()),
            "state_latent_mse": _float(state_latent_mse_per.mean()),
            "action_latent_mse": _float(action_latent_mse_per.mean()),
            "gate_eligible": False,
            "note": "target_mean uses the evaluation batch target mean and is diagnostic only; state/action latent losses are diagnostics, not trained predictors.",
        },
        "shuffled_action_available": shuffled_available,
        "shuffled_action_reason": shuffled_reason,
        "shuffled_action_pred_mse": _float(shuffled_mse_per.mean()) if shuffled_mse_per is not None else None,
        "shuffle_gap_mse_shuffled_minus_real": _float(shuffled_gap_per.mean()) if shuffled_gap_per is not None else None,
        "shuffle_gap_ratio_mse": (
            _float(shuffled_gap_per.mean() / real_mse_per.mean().clamp_min(1e-12)) if shuffled_gap_per is not None else None
        ),
        "shuffle_gap_positive_fraction": (
            _float((shuffled_gap_per > 0).float().mean()) if shuffled_gap_per is not None else None
        ),
        "shuffle_gap_bootstrap_ci95": (
            _bootstrap_ci(shuffled_gap_per, seed=seed, samples=bootstrap_samples)
            if shuffled_gap_per is not None
            else {"samples": int(bootstrap_samples), "mean": None, "low": None, "high": None}
        ),
        "shuffled_action_cosine_distance": _float(shuffled_cos_per.mean()) if shuffled_cos_per is not None else None,
        "cosine_gap_shuffled_minus_real": (
            _float(shuffled_cos_gap_per.mean()) if shuffled_cos_gap_per is not None else None
        ),
        "action_delta": _float(delta),
        "latents": {
            "state": _tensor_stats(state_latent),
            "action": _tensor_stats(action_latent),
            "pred": _tensor_stats(pred_latent),
            "target": _tensor_stats(target_latent),
        },
        "value_reward": value_reward,
        "uncertainty_error": uncertainty_error,
    }

    summary = {
        "schema_version": "openclaw_text_jepa_probe_eval_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "input": str(cache),
        "device": str(device),
        "seed": int(seed),
        "evaluation_split": evaluation_split,
        "record_count": count,
        "counts": {
            "n_total": count,
            "n_valid_prediction": count,
            "n_reward_total": int(payload["reward"].numel()) if "reward" in payload else 0,
            "n_reward_masked": (
                int(payload["reward_mask"].bool().sum().item())
                if "reward_mask" in payload
                else (int(payload["reward"].numel()) if "reward" in payload else 0)
            ),
            "shuffle_available": shuffled_available,
        },
        "cache_metadata": payload.get("metadata", {}),
        "checkpoint_metadata": checkpoint_metadata,
        "record_metadata_summary": _record_metadata_summary(payload),
        "metrics": metrics,
    }
    summary = _json_sanitize(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a text JEPA world-model probe with action ablations.")
    parser.add_argument("--checkpoint", required=True, help="Probe checkpoint produced by train_probe.py.")
    parser.add_argument("--input", required=True, help="cached_hidden.pt with state/action/target hidden tensors.")
    parser.add_argument("--output", required=True, help="JSON summary path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--split", choices=["auto", "all", "train", "val"], default="auto")
    args = parser.parse_args()

    summary = evaluate_probe(
        checkpoint=Path(args.checkpoint),
        cache=Path(args.input),
        output=Path(args.output),
        device_name=args.device,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        split=args.split,
    )
    metrics = summary["metrics"]
    pred_mse = metrics["pred_mse_real"]
    pred_mse_text = "null" if pred_mse is None else f"{pred_mse:.6f}"
    print(
        "wrote probe evaluation to "
        f"{args.output} "
        f"(n={summary['record_count']} pred_mse_real={pred_mse_text} "
        f"shuffle_gap={metrics['shuffle_gap_mse_shuffled_minus_real']})"
    )


if __name__ == "__main__":
    main()
