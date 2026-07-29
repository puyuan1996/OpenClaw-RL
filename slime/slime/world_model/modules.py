from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .metrics import action_delta, effective_rank, mean_cosine_distance


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() != 3:
        raise ValueError(f"Expected hidden tensor with 2 or 3 dims, got {tuple(x.shape)}")
    if mask is None:
        return x.mean(dim=1)
    mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer adapted for the text-latent probe."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)
        if proj.size(-2) < 2:
            return proj.sum() * 0.0
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        a = a / a.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)
        x_t = (proj @ a).unsqueeze(-1) * self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class StableProjector(nn.Module):
    """Normalize and project LLM hidden states into a controlled latent space."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        *,
        clip_value: float = 30.0,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim, latent_dim)
        self.clip_value = float(clip_value)
        self.output_norm = output_norm
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(min=-self.clip_value, max=self.clip_value)
        z = self.net(x)
        if self.output_norm:
            z = F.normalize(z, dim=-1)
        return z


class ActionConditionedPredictor(nn.Module):
    """Predict next observation latent from belief latent and action latent."""

    def __init__(self, latent_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or latent_dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, state_latent: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        pred = self.net(torch.cat([state_latent, action_latent], dim=-1))
        return F.normalize(pred, dim=-1)


@dataclass
class TextLatentWorldModelConfig:
    state_hidden_dim: int
    action_hidden_dim: int
    target_hidden_dim: int
    latent_dim: int = 1024
    projector_hidden_dim: int | None = None
    predictor_hidden_dim: int | None = None
    clip_value: float = 30.0
    sigreg_num_proj: int = 1024
    action_contrast_margin: float = 0.05
    value_head: bool = True
    uncertainty_head: bool = True
    stop_grad_target: bool = False


class TextLatentWorldModel(nn.Module):
    """JEPA-style text world model for terminal-agent replay data.

    The module treats policy/frozen-encoder hidden states as raw material only.
    All branches pass through explicit projectors before entering the shared
    latent space, matching the controlled hidden-to-belief-latent design.
    """

    def __init__(self, config: TextLatentWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.state_projector = StableProjector(
            config.state_hidden_dim,
            config.latent_dim,
            config.projector_hidden_dim,
            clip_value=config.clip_value,
        )
        self.action_projector = StableProjector(
            config.action_hidden_dim,
            config.latent_dim,
            config.projector_hidden_dim,
            clip_value=config.clip_value,
        )
        self.target_projector = StableProjector(
            config.target_hidden_dim,
            config.latent_dim,
            config.projector_hidden_dim,
            clip_value=config.clip_value,
        )
        self.predictor = ActionConditionedPredictor(config.latent_dim, config.predictor_hidden_dim)
        head_in = config.latent_dim * 2
        self.value_head = nn.Linear(head_in, 1) if config.value_head else None
        self.uncertainty_head = nn.Linear(head_in, 1) if config.uncertainty_head else None
        self.sigreg = SIGReg(num_proj=config.sigreg_num_proj)

    def forward(
        self,
        *,
        state_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        target_hidden: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        state_feat = _masked_mean(state_hidden, state_mask)
        action_feat = _masked_mean(action_hidden, action_mask)
        state_latent = self.state_projector(state_feat)
        action_latent = self.action_projector(action_feat)
        pred_latent = self.predictor(state_latent, action_latent)
        target_latent = None
        if target_hidden is not None:
            target_feat = _masked_mean(target_hidden, target_mask)
            target_latent = self.target_projector(target_feat)

        head_input = torch.cat([state_latent, action_latent], dim=-1)
        value = self.value_head(head_input).squeeze(-1) if self.value_head is not None else None
        uncertainty = None
        if self.uncertainty_head is not None:
            uncertainty = F.softplus(self.uncertainty_head(head_input).squeeze(-1))

        return {
            "state_latent": state_latent,
            "action_latent": action_latent,
            "pred_latent": pred_latent,
            "target_latent": target_latent,
            "value": value,
            "uncertainty": uncertainty,
        }

    def compute_loss(
        self,
        *,
        state_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        reward: torch.Tensor | None = None,
        reward_mask: torch.Tensor | None = None,
        pred_loss_type: str = "mse",
        sigreg_coef: float = 0.1,
        action_contrast_coef: float = 0.1,
        value_coef: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self(
            state_hidden=state_hidden,
            action_hidden=action_hidden,
            target_hidden=target_hidden,
        )
        pred = out["pred_latent"]
        target = out["target_latent"]
        loss_target = target.detach() if self.config.stop_grad_target else target
        if pred_loss_type == "cosine":
            pred_loss = mean_cosine_distance(pred, loss_target)
        elif pred_loss_type == "smooth_l1":
            pred_loss = F.smooth_l1_loss(pred, loss_target)
        else:
            pred_loss = F.mse_loss(pred, loss_target)

        sigreg_loss = self.sigreg(torch.stack([out["state_latent"], loss_target], dim=0))
        contrast_loss = pred_loss * 0.0
        delta = pred_loss * 0.0
        if action_hidden.size(0) > 1 and action_contrast_coef != 0.0:
            shuffled_action = torch.roll(out["action_latent"], shifts=1, dims=0)
            shuffled_pred = self.predictor(out["state_latent"], shuffled_action)
            pos_dist = (pred - loss_target).pow(2).mean(dim=-1)
            neg_dist = (shuffled_pred - loss_target).pow(2).mean(dim=-1)
            margin = float(self.config.action_contrast_margin)
            contrast_loss = F.relu(margin + pos_dist - neg_dist).mean()
            delta = action_delta(pred, shuffled_pred)

        value_loss = pred_loss * 0.0
        if reward is not None and out["value"] is not None and value_coef != 0.0:
            reward = reward.float().view_as(out["value"])
            if reward_mask is not None:
                mask = reward_mask.to(device=out["value"].device, dtype=torch.bool).view_as(out["value"])
                if mask.any():
                    value_loss = F.mse_loss(out["value"][mask], reward[mask])
            else:
                value_loss = F.mse_loss(out["value"], reward)

        loss = pred_loss + sigreg_coef * sigreg_loss + action_contrast_coef * contrast_loss + value_coef * value_loss
        metrics = {
            "wm/pred_loss": pred_loss.detach(),
            "wm/sigreg_loss": sigreg_loss.detach(),
            "wm/action_contrast_loss": contrast_loss.detach(),
            "wm/value_loss": value_loss.detach(),
            "wm/value_mask_count": (
                reward_mask.to(device=pred_loss.device, dtype=torch.float32).sum().detach()
                if reward_mask is not None
                else torch.as_tensor(0.0 if reward is None else reward.numel(), device=pred_loss.device)
            ),
            "wm/effective_rank": effective_rank(out["state_latent"]).detach(),
            "wm/action_delta": delta.detach(),
        }
        return loss, metrics
