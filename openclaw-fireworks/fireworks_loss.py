"""Combined RL (GRPO) + OPD (distillation) loss for Fireworks Training SDK.

Adapts openclaw-combine/combine_loss.py to work with Fireworks's
forward_backward_custom signature: (data, logprobs_list) -> (loss, metrics).

Both branches use the same PPO-style clipped policy gradient objective,
but with different advantages:

- OPD samples: advantage = teacher_logp - old_logp  (token-level distillation)
- RL  samples: advantage = reward broadcast          (GRPO-style)

OPD samples carry reward=0 so GRPO advantage=0; RL samples carry
teacher_logp ~ rollout_logp so teacher advantage ~ 0. Each branch
naturally dominates for its own sample type.
"""

from __future__ import annotations

import torch
import tinker


def make_combine_loss_fn(
    old_logprobs: list[torch.Tensor],
    teacher_logprobs: list[torch.Tensor | None],
    rewards: list[float],
    prompt_lens: list[int],
    response_lens: list[int],
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
):
    """Build a Fireworks-compatible loss function with pre-computed advantages.

    Args:
        old_logprobs: Per-token rollout log-probs for each sample (response
            tokens only, length = response_len).
        teacher_logprobs: Per-token teacher log-probs for each sample. None
            for RL-only samples (in which case old_logprobs are used as
            placeholder so the OPD advantage is zero).
        rewards: Scalar reward per sample (typically +1 or -1 from PRM).
        prompt_lens: Number of prompt tokens per sample.
        response_lens: Number of response tokens per sample.
        w_opd: Weight for the OPD (teacher distillation) advantage.
        w_rl: Weight for the GRPO (reward) advantage.
        eps_clip: Lower PPO clipping bound (ratio >= 1 - eps_clip).
        eps_clip_high: Upper PPO clipping bound (ratio <= 1 + eps_clip_high).

    Returns:
        A loss_fn suitable for ``training_client.forward_backward_custom(datums, loss_fn)``.
    """
    old_lp_tensors = [t.detach().float() for t in old_logprobs]
    teacher_lp_tensors = []
    for i, t in enumerate(teacher_logprobs):
        if t is not None:
            teacher_lp_tensors.append(t.detach().float())
        else:
            teacher_lp_tensors.append(old_lp_tensors[i].clone())

    def loss_fn(
        data: list[tinker.Datum],
        logprobs_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_pg_loss = torch.tensor(0.0)
        total_clipfrac = 0.0
        total_kl = 0.0
        n_samples = len(logprobs_list)

        for i, new_lp in enumerate(logprobs_list):
            p_len = prompt_lens[i]
            r_len = response_lens[i]
            seq_len = min(len(new_lp), p_len + r_len - 1)

            resp_start = max(0, p_len - 1)
            resp_new_lp = new_lp[resp_start:seq_len].float()
            actual_r_len = len(resp_new_lp)

            if actual_r_len == 0:
                continue

            old_lp = old_lp_tensors[i][:actual_r_len]
            t_lp = teacher_lp_tensors[i][:actual_r_len]

            grpo_adv = torch.full((actual_r_len,), rewards[i], dtype=torch.float32)
            teacher_adv = t_lp - old_lp
            combined_adv = w_opd * teacher_adv + w_rl * grpo_adv

            ppo_kl = old_lp - resp_new_lp
            ratio = torch.exp(-ppo_kl)
            clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip_high)
            pg = -torch.min(ratio * combined_adv, clipped * combined_adv)
            total_pg_loss = total_pg_loss + pg.mean()

            with torch.no_grad():
                total_clipfrac += (ratio.detach() != clipped.detach()).float().mean().item()
                total_kl += ppo_kl.detach().mean().item()

        loss = total_pg_loss / max(n_samples, 1)
        metrics = {
            "pg_loss": loss.item(),
            "pg_clipfrac": total_clipfrac / max(n_samples, 1),
            "ppo_kl": total_kl / max(n_samples, 1),
        }
        return loss, metrics

    return loss_fn


def build_datum(
    tokens: list[int],
    prompt_len: int,
    max_length: int = 32768,
) -> tinker.Datum:
    """Build a tinker.Datum from token IDs with appropriate weights.

    Weights are 0 for prompt tokens and 1 for response tokens.
    """
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    if max_length and len(tokens) > max_length:
        tokens = tokens[:max_length]
    mi = tinker.ModelInput.from_ints(tokens)
    w = torch.zeros(len(tokens), dtype=torch.float32)
    w[prompt_len:] = 1.0
    return datum_from_model_input_weights(mi, w, max_length=max_length)
