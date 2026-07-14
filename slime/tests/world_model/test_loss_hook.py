from types import SimpleNamespace

import pytest
import torch

from slime.world_model.loss_hook import apply_world_model_loss


def test_apply_world_model_loss_disabled_noop():
    logits = torch.ones(2, 3)
    base_loss = logits.sum()
    out_loss, log = apply_world_model_loss(
        args=SimpleNamespace(world_model_enable=False, world_model_loss_coef=1.0),
        batch={},
        logits=logits,
        loss=base_loss,
        reported_loss={"loss": base_loss.detach()},
    )
    assert out_loss is base_loss
    assert list(log.keys()) == ["loss"]


def test_apply_world_model_loss_zero_coefficient_does_not_load_hook():
    logits = torch.ones(2, 3)
    base_loss = logits.sum()
    out_loss, log = apply_world_model_loss(
        args=SimpleNamespace(
            world_model_enable=True,
            world_model_loss_coef=0.0,
            world_model_loss_hook_path="missing.module:hook",
        ),
        batch={},
        logits=logits,
        loss=base_loss,
        reported_loss={"loss": base_loss.detach()},
    )

    assert out_loss is base_loss
    assert list(log) == ["loss"]


def test_apply_world_model_loss_with_precomputed_latents():
    logits = torch.zeros(1)
    base_loss = logits.sum()
    pred = [torch.tensor([1.0, 0.0], requires_grad=True), torch.tensor([0.0, 1.0], requires_grad=True)]
    target = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
    out_loss, log = apply_world_model_loss(
        args=SimpleNamespace(world_model_enable=True, world_model_loss_coef=0.5, world_model_loss_hook_path=None),
        batch={
            "wm_pred_latents": pred,
            "wm_target_latents": target,
            "wm_metadata": [{"a": 1}, {"a": 2}],
            "response_lengths": [2, 2],
        },
        logits=logits,
        loss=base_loss,
        reported_loss={"loss": base_loss.detach()},
    )
    assert torch.isclose(out_loss, torch.tensor(0.5))
    assert torch.isclose(log["wm/loss"], torch.tensor(1.0))
    assert torch.isclose(log["wm/loss_coef"], torch.tensor(1.0))
    assert torch.isclose(log["wm/metadata_count"], torch.tensor(2.0))
    assert torch.isclose(log["wm/latent_available"], torch.tensor(2.0))
    out_loss.backward()
    assert all(row.grad is not None for row in pred)


def test_apply_world_model_loss_rejects_missing_latents_at_positive_coefficient():
    logits = torch.zeros(1, requires_grad=True)
    with pytest.raises(ValueError, match="graph-connected"):
        apply_world_model_loss(
            args=SimpleNamespace(world_model_enable=True, world_model_loss_coef=0.1),
            batch={},
            logits=logits,
            loss=logits.sum(),
            reported_loss={"loss": logits.sum().detach()},
        )


def test_apply_world_model_loss_rejects_detached_latents():
    logits = torch.zeros(1, requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        apply_world_model_loss(
            args=SimpleNamespace(world_model_enable=True, world_model_loss_coef=0.1),
            batch={
                "wm_pred_latents": torch.ones(1, 4),
                "wm_target_latents": torch.zeros(1, 4),
                "response_lengths": [1],
            },
            logits=logits,
            loss=logits.sum(),
            reported_loss={"loss": logits.sum().detach()},
        )


@pytest.mark.parametrize("coefficient", [-1.0, float("nan"), float("inf")])
def test_apply_world_model_loss_rejects_invalid_coefficient(coefficient):
    logits = torch.zeros(1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_world_model_loss(
            args=SimpleNamespace(world_model_enable=True, world_model_loss_coef=coefficient),
            batch={},
            logits=logits,
            loss=logits.sum(),
            reported_loss={"loss": logits.sum()},
        )


def test_apply_world_model_loss_rejects_per_token_normalization():
    logits = torch.zeros(1)
    with pytest.raises(ValueError, match="calculate_per_token_loss"):
        apply_world_model_loss(
            args=SimpleNamespace(
                world_model_enable=True,
                world_model_loss_coef=0.1,
                calculate_per_token_loss=True,
            ),
            batch={},
            logits=logits,
            loss=logits.sum(),
            reported_loss={"loss": logits.sum()},
        )


def test_apply_world_model_loss_rejects_context_parallel_replication():
    logits = torch.zeros(1)
    with pytest.raises(ValueError, match="context_parallel_size"):
        apply_world_model_loss(
            args=SimpleNamespace(
                world_model_enable=True,
                world_model_loss_coef=0.1,
                context_parallel_size=2,
            ),
            batch={},
            logits=logits,
            loss=logits.sum(),
            reported_loss={"loss": logits.sum()},
        )


def test_apply_world_model_loss_treats_unbatched_latent_as_one_sample():
    logits = torch.zeros(1)
    pred = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    out_loss, log = apply_world_model_loss(
        args=SimpleNamespace(world_model_enable=True, world_model_loss_coef=1.0),
        batch={
            "wm_pred_latents": pred,
            "wm_target_latents": torch.zeros(4),
            "wm_metadata": [{}],
            "response_lengths": [1],
        },
        logits=logits,
        loss=logits.sum(),
        reported_loss={"loss": logits.sum()},
    )

    assert torch.isclose(out_loss, torch.tensor(0.25))
    assert torch.isclose(log["wm/loss"], torch.tensor(0.25))
    out_loss.backward()
    assert pred.grad is not None


def test_apply_world_model_loss_rejects_latent_batch_count_mismatch():
    logits = torch.zeros(1)
    with pytest.raises(ValueError, match="sample count mismatch"):
        apply_world_model_loss(
            args=SimpleNamespace(world_model_enable=True, world_model_loss_coef=1.0),
            batch={
                "wm_pred_latents": torch.zeros(2, 4),
                "wm_target_latents": torch.zeros(2, 4),
                "response_lengths": [1],
            },
            logits=logits,
            loss=logits.sum(),
            reported_loss={"loss": logits.sum()},
        )
