from types import SimpleNamespace

import pytest
import torch

from slime.rollout.data_source import RolloutDataSourceWithBuffer, _iter_sample_leaves
from slime.utils.per_buffer import PrioritizedReplayStrategy
from slime.utils.sil_buffer import SILBuffer, normalize_sil_loss_mask
from slime.utils.topr_utils import compute_topr_seq_weights
from slime.utils.types import Sample


def _sample(reward: float, policy_version: int = 0) -> Sample:
    sample = Sample(reward=reward, response_length=2, loss_mask=[1, 1])
    sample.policy_version = policy_version
    return sample


def test_topr_casts_each_pair_to_the_proximal_logp_device_and_float32():
    proximal = [torch.tensor([0.5, 0.25], dtype=torch.float64)]
    behavior = [torch.tensor([0.0, 0.25], dtype=torch.float32)]
    masks = [torch.tensor([1, 0], dtype=torch.int64)]

    w_seq, w_token = compute_topr_seq_weights(
        proximal,
        behavior,
        masks,
        logw_cap=10.0,
        w_min=0.0,
        w_max=10.0,
    )

    assert w_seq.shape == (1,)
    assert w_token.shape == (2,)
    assert w_seq.device == proximal[0].device
    assert w_token.device == proximal[0].device
    assert w_seq.dtype == torch.float32
    assert w_token.dtype == torch.float32
    assert torch.allclose(w_seq, torch.exp(torch.tensor([0.5])), atol=1e-6)


def test_sil_loss_mask_is_binary_and_response_length_aligned():
    assert normalize_sil_loss_mask([1, 0, 0.2, 1], 3) == [1, 0, 1]
    assert normalize_sil_loss_mask([0], 3) == [0, 1, 1]
    assert normalize_sil_loss_mask(None, 2) == [1, 1]


def test_iter_sample_leaves_handles_nested_rollout_containers():
    a, b, c = _sample(1.0), _sample(0.0), _sample(0.5)
    assert list(_iter_sample_leaves([[a, [b]], c])) == [a, b, c]


def test_per_sampling_attaches_normalized_importance_weights():
    args = SimpleNamespace(
        max_staleness=-1,
        buffer_remove_on_sample=False,
        buffer_reuse_samples=10,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_anneal_steps=1000,
        per_priority_source="reward_dev",
        per_priority_eps=1e-3,
        per_min_priority=1e-6,
        per_max_priority=1e3,
    )
    strategy = PrioritizedReplayStrategy(args, current_policy_version=0)
    buffer = [
        [_sample(0.0), _sample(0.0)],
        [_sample(0.0), _sample(1.0)],
        [_sample(-1.0), _sample(1.0)],
    ]

    sampled = strategy.sample(buffer, 2)

    assert len(sampled) == 2
    for group in sampled:
        for sample in group:
            assert 0.0 <= sample.metadata["per_is_weight"] <= 1.0
            assert 0.0 < sample.metadata["per_sample_prob"] <= 1.0


@pytest.mark.parametrize("raw_mask", ([1, 0], (1, 0), torch.tensor([1, 0]), 1))
def test_sil_loss_mask_accepts_common_sequence_types(raw_mask):
    expected = [1, 1] if isinstance(raw_mask, int) else [1, 0]
    assert normalize_sil_loss_mask(raw_mask, 2) == expected


def test_sil_buffer_preserves_behavior_policy_version():
    sil_buffer = SILBuffer(buffer_size=4, score_threshold=0.0)
    sil_buffer.push(
        [
            {
                "tokens": [1, 2, 3],
                "response_length": 2,
                "loss_mask": [1, 1],
                "rollout_log_probs": torch.tensor([0.1, 0.2]),
                "policy_version": 7,
                "reward": 1.0,
                "advantage": 1.0,
            }
        ],
        current_step=3,
    )

    assert sil_buffer.sample(1, current_step=4)[0]["policy_version"] == 7


def test_sil_candidate_push_defaults_missing_policy_version_to_current_version():
    class CapturingSILBuffer:
        def __init__(self):
            self.entries = []

        def push(self, entries, current_step):
            self.entries.extend(entries)

    data_source = RolloutDataSourceWithBuffer.__new__(RolloutDataSourceWithBuffer)
    data_source.sil_buffer = CapturingSILBuffer()
    data_source.current_policy_version = 5
    data_source.total_added = 0
    data_source.args = SimpleNamespace(reward_key=None)
    sample = Sample(
        reward=1.0,
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[0.1, 0.2],
    )

    data_source._push_sil_candidates([[sample]])

    assert data_source.sil_buffer.entries[0]["policy_version"] == 5
