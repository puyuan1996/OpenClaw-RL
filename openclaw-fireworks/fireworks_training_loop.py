"""Fireworks training loop for OpenClaw-RL.

Replaces the Ray + RolloutManager + MegatronTrainRayActor pipeline with a
simple Python loop that:
  1. Drains training samples from the API server's output queue
  2. Builds tinker Datums
  3. Drives forward_backward_custom + optim_step on the remote Fireworks trainer
  4. Periodically syncs weights to the Fireworks deployment
"""

from __future__ import annotations

import logging
import os
import queue
import time

import tinker
import torch
from fireworks.training.sdk import GradAccNormalization

from fireworks_loss import build_datum, make_combine_loss_fn
from fireworks_server import TrainingSample

logger = logging.getLogger(__name__)


def drain_queue(
    output_queue: queue.Queue,
    batch_size: int,
    poll_interval: float = 0.1,
    progress_interval: float = 30.0,
) -> list[TrainingSample]:
    """Block until batch_size samples are collected from the output queue."""
    samples: list[TrainingSample] = []
    start = time.time()
    last_progress = start

    while len(samples) < batch_size:
        try:
            group_id, group = output_queue.get(timeout=poll_interval)
            for s in group:
                samples.append(s)
        except queue.Empty:
            pass

        if time.time() - last_progress > progress_interval:
            logger.info(
                "[TrainLoop] waiting for samples: %d/%d, queue=%d",
                len(samples), batch_size, output_queue.qsize(),
            )
            last_progress = time.time()

    elapsed = time.time() - start
    samples.sort(key=lambda s: s.index)
    logger.info(
        "[TrainLoop] drained %d samples in %.1fs", len(samples), elapsed,
    )
    return samples[:batch_size]


def build_batch(
    samples: list[TrainingSample],
    max_seq_len: int = 32768,
) -> tuple[
    list[tinker.Datum],
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[float],
    list[int],
    list[int],
]:
    """Convert TrainingSamples into tinker Datums + associated metadata."""
    datums = []
    old_logprobs = []
    teacher_logprobs = []
    rewards = []
    prompt_lens = []
    response_lens = []

    for s in samples:
        datum = build_datum(s.tokens, s.prompt_len, max_length=max_seq_len)
        datums.append(datum)

        old_lp = torch.tensor(s.rollout_log_probs, dtype=torch.float32)
        old_logprobs.append(old_lp)

        if s.teacher_log_probs is not None:
            t_lp = torch.tensor(s.teacher_log_probs, dtype=torch.float32)
            teacher_logprobs.append(t_lp)
        else:
            teacher_logprobs.append(None)

        rewards.append(s.reward)
        prompt_lens.append(s.prompt_len)
        response_lens.append(s.response_len)

    return datums, old_logprobs, teacher_logprobs, rewards, prompt_lens, response_lens


def train_step(
    training_client,
    datums: list[tinker.Datum],
    old_logprobs: list[torch.Tensor],
    teacher_logprobs: list[torch.Tensor | None],
    rewards: list[float],
    prompt_lens: list[int],
    response_lens: list[int],
    *,
    learning_rate: float = 1e-5,
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
) -> dict[str, float]:
    """Run one training step: forward_backward_custom + optim_step."""
    loss_fn = make_combine_loss_fn(
        old_logprobs=old_logprobs,
        teacher_logprobs=teacher_logprobs,
        rewards=rewards,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        w_opd=w_opd,
        w_rl=w_rl,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
    )

    result = training_client.forward_backward_custom(datums, loss_fn).result()
    training_client.optim_step(
        tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
        ),
        grad_accumulation_normalization=GradAccNormalization.NUM_LOSS_TOKENS,
    ).result()

    return result.metrics


def run_training_loop(
    training_client,
    weight_syncer,
    output_queue: queue.Queue,
    *,
    batch_size: int = 16,
    max_seq_len: int = 32768,
    learning_rate: float = 1e-5,
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
    weight_sync_interval: int = 1,
    max_steps: int = 0,
):
    """Main training loop: drain -> train -> weight-sync -> repeat.

    Args:
        training_client: FiretitanTrainingClient instance.
        weight_syncer: WeightSyncer instance for checkpoint + hotload.
        output_queue: Queue receiving (group_id, [TrainingSample]) tuples.
        batch_size: Number of samples per training step.
        max_seq_len: Maximum sequence length for datums.
        learning_rate: Adam learning rate.
        w_opd: Weight for OPD (distillation) advantage.
        w_rl: Weight for GRPO (reward) advantage.
        eps_clip: Lower PPO clip bound.
        eps_clip_high: Upper PPO clip bound.
        weight_sync_interval: Steps between weight syncs to deployment.
        max_steps: Stop after this many steps (0 = run forever).
    """
    step = 0
    logger.info(
        "[TrainLoop] starting: batch_size=%d lr=%.2e w_opd=%.1f w_rl=%.1f "
        "sync_interval=%d max_steps=%d",
        batch_size, learning_rate, w_opd, w_rl, weight_sync_interval,
        max_steps,
    )

    while max_steps == 0 or step < max_steps:
        samples = drain_queue(output_queue, batch_size)
        datums, old_lp, teacher_lp, rewards, p_lens, r_lens = build_batch(
            samples, max_seq_len=max_seq_len,
        )

        t0 = time.time()
        metrics = train_step(
            training_client, datums, old_lp, teacher_lp, rewards, p_lens, r_lens,
            learning_rate=learning_rate,
            w_opd=w_opd, w_rl=w_rl,
            eps_clip=eps_clip, eps_clip_high=eps_clip_high,
        )
        train_time = time.time() - t0

        step += 1
        avg_reward = sum(rewards) / len(rewards)
        logger.info(
            "[TrainLoop] step %d: %s avg_reward=%.2f train_time=%.1fs",
            step, metrics, avg_reward, train_time,
        )

        if weight_sync_interval > 0 and step % weight_sync_interval == 0:
            t0 = time.time()
            weight_syncer.save_and_hotload(f"step-{step:05d}")
            sync_time = time.time() - t0
            logger.info("[TrainLoop] weight sync at step %d (%.1fs)", step, sync_time)

    logger.info("[TrainLoop] finished after %d steps", step)
