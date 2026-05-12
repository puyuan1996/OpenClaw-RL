from __future__ import annotations

from copy import deepcopy
import logging
from typing import List

from slime.utils.types import Sample

from custom_types import Interaction, RolloutOutcome

logger = logging.getLogger(__name__)


def _derive_status(outcome: RolloutOutcome, eval_error: str | None) -> Sample.Status:
    if eval_error is not None:
        return Sample.Status.FAILED
    if outcome.final_response is None:
        return Sample.Status.ABORTED
    if outcome.reached_parse_error_limit:
        return Sample.Status.ABORTED
    if outcome.reached_iteration_limit:
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def build_non_trainable_sample(
    sample: Sample,
    status: Sample.Status,
    *,
    reward: float | None = None,
) -> List[Sample]:
    sample.status = status
    outcome = 0.0 if reward is None else reward
    sample.reward = {
        "accuracy": outcome,
        "discounted_outcome": 2.0 * outcome - 1.0,
        "score": 2.0 * outcome - 1.0,
    }
    sample.remove_sample = True
    return [sample]


def build_samples_from_outcome(
    sample: Sample,
    outcome: RolloutOutcome | None,
    *,
    reward: float,
    rollout_error: str | None = None,
    eval_error: str | None,
    prm_enabled: bool,
    prm_step_coef: float,
    log_tag: str,
) -> List[Sample]:
    if rollout_error is not None:
        sample.metadata["rollout_error"] = rollout_error
    if eval_error is not None:
        sample.metadata["evaluation_error"] = eval_error
    if outcome is None:
        return build_non_trainable_sample(
            sample,
            Sample.Status.FAILED,
            reward=reward,
        )

    status = _derive_status(outcome, eval_error)

    if not outcome.interactions:
        logger.warning("%s No interactions recorded. Remove sample.", log_tag)
        return build_non_trainable_sample(
            sample,
            status,
            reward=reward,
        )

    if prm_enabled:
        sample.metadata["prm"] = {
            "enabled": True,
            "coef": prm_step_coef,
            "turn_scores": outcome.prm_turn_scores,
            "turn_details": outcome.prm_turn_details,
        }

    num_turns = len(outcome.interactions)
    samples: List[Sample] = []
    normalized_outcome = 2.0 * reward - 1.0
    prm_turn_scores = outcome.prm_turn_scores if prm_enabled else None

    for interaction in outcome.interactions:
        turn_idx = interaction.turn_idx
        output_sample = deepcopy(sample)
        output_sample.tokens = interaction.input_ids + interaction.output_token_ids
        output_sample.response_length = len(interaction.output_token_ids)
        output_sample.loss_mask = [1] * output_sample.response_length
        output_sample.rollout_log_probs = list(interaction.output_token_logprobs)
        output_sample.response = interaction.output_text
        output_sample.status = status
        output_sample.metadata.update(
            {
                "turn_idx": turn_idx,
                "num_turns": num_turns,
                "finish_reason": interaction.finish_reason,
                "latency_ms": interaction.latency_ms,
                "model_turn_count": outcome.model_turn_count,
                "parse_error_count": outcome.parse_error_count,
            }
        )

        discounted_outcome = normalized_outcome
        reward_payload = {
            "accuracy": reward,
            "discounted_outcome": discounted_outcome,
            "score": discounted_outcome,
        }
        if prm_turn_scores is not None:
            prm = prm_turn_scores.get(turn_idx, 0.0)
            reward_payload["prm_turn_score"] = prm
            reward_payload["score"] = discounted_outcome + prm_step_coef * prm
            output_sample.metadata["step_wise"] = {
                "step_scores": [prm],
                "step_scores_with_outcome": [reward_payload["score"]],
                "step_indices": [turn_idx],
                "step_token_spans": [[0, output_sample.response_length]],
            }

        output_sample.reward = reward_payload
        if output_sample.status in {Sample.Status.FAILED}:
            output_sample.remove_sample = True
        samples.append(output_sample)

    return samples
