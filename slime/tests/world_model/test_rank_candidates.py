import json

import pytest
import torch

from slime.world_model.metrics import require_finite_tensor
from slime.world_model.rank_candidates import _reward_label_status, _select_scores, _validate_score_mode
from slime.world_model.summarize_stage_a import _ranking_artifact_status


def test_rank_candidates_auto_prefers_value_over_target_error():
    out = {
        "pred_latent": torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        "target_latent": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "value": torch.tensor([0.1, 0.9]),
        "uncertainty": torch.tensor([10.0, 0.1]),
    }

    scores, source = _select_scores(out, score_mode="auto")

    assert source == "value"
    assert torch.equal(scores, out["value"])


def test_rank_candidates_pred_error_requires_explicit_mode():
    out = {
        "pred_latent": torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        "target_latent": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "value": torch.tensor([0.1, 0.9]),
        "uncertainty": torch.tensor([10.0, 0.1]),
    }

    scores, source = _select_scores(out, score_mode="pred_error")

    assert source == "negative_pred_error"
    assert scores[0] > scores[1]


def test_rank_candidates_rejects_untrained_value_head():
    metadata = {
        "has_reward": True,
        "reward_mask_count": 8,
        "train_reward_label_count": 8,
        "train_count": 8,
        "val_count": 0,
        "final_train_loss": 0.1,
        "optimizer_step_count": 1,
        "value_update_step_count": 1,
        "hyperparameters": {"value_coef": 0.0, "epochs": 1, "lr": 1e-4},
    }

    with pytest.raises(ValueError, match="reward-supervised value head"):
        _validate_score_mode("auto", metadata)


def test_rank_candidates_accepts_trained_value_head():
    metadata = {
        "has_reward": True,
        "reward_mask_count": 8,
        "train_reward_label_count": 8,
        "train_count": 8,
        "val_count": 0,
        "final_train_loss": 0.1,
        "optimizer_step_count": 1,
        "value_update_step_count": 1,
        "hyperparameters": {"value_coef": 0.05, "epochs": 1, "lr": 1e-4},
    }

    _validate_score_mode("value", metadata)


def test_rank_candidates_rejects_untrained_uncertainty_head():
    with pytest.raises(ValueError, match="no dedicated loss"):
        _validate_score_mode("uncertainty", {})


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_rank_candidates_rejects_non_finite_scores(bad_value):
    with pytest.raises(ValueError, match="NaN or Inf"):
        require_finite_tensor(torch.tensor([0.0, bad_value]), name="candidate scores")


def test_oracle_ranking_artifact_is_not_execution_eligible(tmp_path):
    path = tmp_path / "rankings.jsonl"
    path.write_text(
        json.dumps(
            {
                "score_source": "negative_pred_error",
                "oracle_only": True,
                "requires_target": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count, eligible = _ranking_artifact_status(path)

    assert count == 1
    assert eligible is False


def test_target_free_unknown_reward_ranking_is_not_execution_eligible(tmp_path):
    path = tmp_path / "rankings.jsonl"
    path.write_text(
        json.dumps(
            {
                "score_source": "value",
                "oracle_only": False,
                "requires_target": False,
                "reward_label_verified_execution_outcome": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count, eligible = _ranking_artifact_status(path)

    assert count == 1
    assert eligible is False


def test_verified_target_free_ranking_is_execution_eligible(tmp_path):
    path = tmp_path / "rankings.jsonl"
    path.write_text(
        json.dumps(
            {
                "score_source": "value",
                "oracle_only": False,
                "requires_target": False,
                "reward_label_verified_execution_outcome": True,
                "evaluation_split_scope": "group_heldout",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count, eligible = _ranking_artifact_status(path)

    assert count == 1
    assert eligible is True


def test_in_sample_target_free_ranking_is_not_execution_eligible(tmp_path):
    path = tmp_path / "rankings.jsonl"
    path.write_text(
        json.dumps(
            {
                "score_source": "value",
                "oracle_only": False,
                "requires_target": False,
                "reward_label_verified_execution_outcome": True,
                "evaluation_split_scope": "in_sample_all",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count, eligible = _ranking_artifact_status(path)

    assert count == 1
    assert eligible is False


def test_reward_label_status_requires_verified_checkpoint_contract():
    contract, verified = _reward_label_status(
        {"cache_metadata": {"reward_label_contract": {"verified_execution_outcome": True}}}
    )

    assert contract["verified_execution_outcome"] is True
    assert verified is True
