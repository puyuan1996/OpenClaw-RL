from types import SimpleNamespace

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


def test_apply_world_model_loss_with_precomputed_latents():
    logits = torch.zeros(1)
    base_loss = logits.sum()
    pred = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    target = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
    out_loss, log = apply_world_model_loss(
        args=SimpleNamespace(world_model_enable=True, world_model_loss_coef=0.5, world_model_loss_hook_path=None),
        batch={"wm_pred_latents": pred, "wm_target_latents": target, "wm_metadata": [{"a": 1}, {"a": 2}]},
        logits=logits,
        loss=base_loss,
        reported_loss={"loss": base_loss.detach()},
    )
    assert torch.isclose(out_loss, torch.tensor(0.25))
    assert torch.isclose(log["wm/loss"], torch.tensor(0.5))
    assert torch.isclose(log["wm/metadata_count"], torch.tensor(2.0))
