from __future__ import annotations

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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    if pred is not None and target is not None:
        if isinstance(pred, list):
            pred = torch.stack(pred)
        if isinstance(target, list):
            target = torch.stack(target)
        aux_loss = F.mse_loss(pred.float(), target.float().detach())
        available = 1.0
    metrics = {
        "wm/loss": aux_loss.detach(),
        "wm/metadata_count": torch.tensor(float(len(metadata)), device=device),
        "wm/latent_available": torch.tensor(available, device=device),
    }
    return aux_loss, metrics


def apply_world_model_loss(
    *,
    args: Any,
    batch: dict[str, Any],
    logits: torch.Tensor,
    loss: torch.Tensor,
    reported_loss: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    enabled = bool(getattr(args, "world_model_enable", False))
    coef = float(getattr(args, "world_model_loss_coef", 0.0) or 0.0)
    if not enabled or coef == 0.0:
        return loss, reported_loss

    hook_path = getattr(args, "world_model_loss_hook_path", None)
    hook = load_function(hook_path) if hook_path else default_world_model_loss_hook
    hook_result = hook(args, batch, logits)
    if isinstance(hook_result, Mapping):
        aux_loss = hook_result.get("loss", logits.sum() * 0.0)
        metrics = dict(hook_result.get("metrics", {}))
    else:
        aux_loss, metrics = hook_result

    aux_loss = _as_scalar_tensor(aux_loss, device=logits.device)
    loss = loss + coef * aux_loss
    reported_loss["wm/loss"] = aux_loss.detach()
    reported_loss["wm/loss_coef"] = torch.tensor(coef, dtype=torch.float32, device=logits.device)
    for name, value in dict(metrics or {}).items():
        key = name if str(name).startswith("wm/") else f"wm/{name}"
        reported_loss[key] = _as_scalar_tensor(value, device=logits.device).detach()
    reported_loss["loss"] = loss.clone().detach()
    return loss, reported_loss
