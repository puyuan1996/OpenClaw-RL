import torch

from slime.world_model.modules import TextLatentWorldModel, TextLatentWorldModelConfig


def test_text_latent_world_model_loss_backward():
    torch.manual_seed(0)
    config = TextLatentWorldModelConfig(
        state_hidden_dim=8,
        action_hidden_dim=6,
        target_hidden_dim=10,
        latent_dim=4,
        sigreg_num_proj=8,
    )
    model = TextLatentWorldModel(config)
    loss, metrics = model.compute_loss(
        state_hidden=torch.randn(5, 3, 8),
        action_hidden=torch.randn(5, 2, 6),
        target_hidden=torch.randn(5, 4, 10),
        reward=torch.randn(5),
        value_coef=0.1,
    )
    assert loss.ndim == 0
    assert "wm/pred_loss" in metrics
    assert "wm/action_delta" in metrics
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())
    assert any(param.grad is not None for param in model.target_projector.parameters())


def test_text_latent_world_model_value_loss_honors_reward_mask():
    torch.manual_seed(0)
    config = TextLatentWorldModelConfig(
        state_hidden_dim=4,
        action_hidden_dim=4,
        target_hidden_dim=4,
        latent_dim=3,
        sigreg_num_proj=4,
    )
    model = TextLatentWorldModel(config)
    _loss, metrics = model.compute_loss(
        state_hidden=torch.randn(3, 4),
        action_hidden=torch.randn(3, 4),
        target_hidden=torch.randn(3, 4),
        reward=torch.tensor([100.0, -100.0, 3.0]),
        reward_mask=torch.tensor([False, False, False]),
        value_coef=1.0,
    )

    assert torch.allclose(metrics["wm/value_loss"], torch.tensor(0.0))
    assert torch.allclose(metrics["wm/value_mask_count"], torch.tensor(0.0))


def test_stop_grad_target_freezes_target_projector_for_all_losses():
    torch.manual_seed(0)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=4,
            action_hidden_dim=4,
            target_hidden_dim=4,
            latent_dim=3,
            sigreg_num_proj=4,
            stop_grad_target=True,
        )
    )

    loss, _ = model.compute_loss(
        state_hidden=torch.randn(4, 4),
        action_hidden=torch.randn(4, 4),
        target_hidden=torch.randn(4, 4),
        sigreg_coef=0.1,
        action_contrast_coef=0.1,
    )
    loss.backward()

    assert all(parameter.grad is None for parameter in model.target_projector.parameters())
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
