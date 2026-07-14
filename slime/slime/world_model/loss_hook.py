from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from slime.utils.misc import load_function


def _as_scalar_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.to(device=device)
        if value.numel() != 1:
            value = value.float().mean()
        return value.reshape(())
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def default_world_model_loss_hook(
    args: Any,
    batch: dict[str, Any],
    logits: torch.Tensor,
) -> dict[str, Any]:
    """No-op online hook unless precomputed latent predictions are provided.

    The v1 online training path intentionally does not extract Megatron hidden
    states. Offline probe training uses `TextLatentWorldModel` directly. This
    hook lets future adapters provide `wm_pred_latents` and `wm_target_latents`
    without replacing the policy loss implementation.
    """
    del args
    device = logits.device
    metadata = batch.get("wm_metadata") or []
    pred = batch.get("wm_pred_latents")
    target = batch.get("wm_target_latents")
    aux_loss = logits.sum() * 0.0
    available = 0.0
    sample_count = max(len(metadata), 1)
    if pred is not None and target is not None:
        if isinstance(pred, list):
            pred = torch.stack(pred)
        if isinstance(target, list):
            target = torch.stack(target)
        if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("world-model latents must be tensors or lists of tensors")
        if pred.shape != target.shape or pred.ndim == 0:
            raise ValueError("world-model prediction/target latents must have the same non-scalar shape")
        response_lengths = batch.get("response_lengths")
        expected_count = len(response_lengths) if response_lengths is not None else len(metadata)
        sample_count = 1 if pred.ndim == 1 else int(pred.shape[0])
        if expected_count > 0 and sample_count != expected_count:
            raise ValueError(
                f"world-model latent sample count mismatch: latents={sample_count} batch={expected_count}"
            )
        aux_loss = F.mse_loss(pred.float(), target.float().detach())
        available = 1.0
    metrics = {
        "wm/metadata_count": torch.tensor(float(len(metadata)), device=device),
        "wm/latent_available": torch.tensor(available * sample_count, device=device),
    }
    return {
        "loss": aux_loss,
        "reduction": "mean",
        "sample_count": sample_count,
        "metrics": metrics,
    }


def apply_world_model_loss(
    *,
    args: Any,
    batch: dict[str, Any],
    logits: torch.Tensor,
    loss: torch.Tensor,
    reported_loss: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Add a sample-sum auxiliary objective before Megatron global-batch scaling.

    Mapping hooks may return ``reduction="mean"`` with ``sample_count``; tuple
    hooks retain the legacy contract and are interpreted as sample sums. Hook
    metrics must also be sample sums because Megatron divides every reported
    field by the global sample count after data-parallel reduction.
    """
    enabled = bool(getattr(args, "world_model_enable", False))
    coef = float(getattr(args, "world_model_loss_coef", 0.0) or 0.0)
    if not enabled:
        return loss, reported_loss
    if not math.isfinite(coef) or coef < 0.0:
        raise ValueError("world-model loss coefficient must be finite and non-negative")
    if coef == 0.0:
        return loss, reported_loss
    if bool(getattr(args, "calculate_per_token_loss", False)):
        raise ValueError("sample-level world-model loss is not supported with calculate_per_token_loss")
    context_parallel_size = int(getattr(args, "context_parallel_size", 1) or 1)
    if context_parallel_size != 1:
        raise ValueError("sample-level world-model loss is not supported with context_parallel_size != 1")

    hook_path = getattr(args, "world_model_loss_hook_path", None)
    hook = load_function(hook_path) if hook_path else default_world_model_loss_hook
    hook_result = hook(args, batch, logits)
    if isinstance(hook_result, Mapping):
        aux_loss = hook_result.get("loss", logits.sum() * 0.0)
        metrics = dict(hook_result.get("metrics", {}))
        reduction = str(hook_result.get("reduction", "sum"))
        sample_count = int(hook_result.get("sample_count", 1))
    else:
        aux_loss, metrics = hook_result
        reduction = "sum"
        sample_count = 1

    aux_loss = _as_scalar_tensor(aux_loss, device=logits.device)
    if not bool(torch.isfinite(aux_loss).all().item()):
        raise ValueError("world-model hook returned a non-finite loss")
    if reduction == "mean":
        if sample_count <= 0:
            raise ValueError("world-model hook sample_count must be positive for mean reduction")
        aux_loss = aux_loss * sample_count
    elif reduction != "sum":
        raise ValueError(f"unknown world-model hook reduction: {reduction}")
    loss = loss + coef * aux_loss
    response_lengths = batch.get("response_lengths")
    report_sample_count = len(response_lengths) if response_lengths is not None else sample_count
    if report_sample_count <= 0:
        report_sample_count = sample_count
    reported_loss["wm/loss"] = aux_loss.detach()
    reported_loss["wm/loss_coef"] = torch.tensor(
        coef * report_sample_count,
        dtype=torch.float32,
        device=logits.device,
    )
    for name, value in dict(metrics or {}).items():
        key = name if str(name).startswith("wm/") else f"wm/{name}"
        reported_loss[key] = _as_scalar_tensor(value, device=logits.device).detach()
    reported_loss["loss"] = loss.clone().detach()
    return loss, reported_loss
