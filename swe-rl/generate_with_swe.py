"""Default SWE RL algorithm for SWE-Bench (trajectory mode).

Algorithm profile (see docs/four-way-algorithm-comparison.md §1 for rationale):
  * Advantage estimator: GRPO (with std normalization)
  * Sample organization: trajectory mode (1 trajectory = 1 Sample)
  * KL loss: coef=0 (nominally on, effectively off)
  * Clip ratio: 0.2 / 0.28 (DAPO asymmetric)
  * Entropy loss: 0
  * No compact filter, no Dr.GRPO length norm
  * Reward: plain ``{"score": float, "acc": float}`` (outcome only)

Scaffold (bash backtick tool format, submit sentinel, remote Docker pool) is
inherited from ``generate_with_swe_trajectory``.

Usage in training script:
    --custom-generate-function-path generate_with_swe.generate
    --custom-rm-path generate_with_swe.reward_func
"""

from slime.utils.types import Sample

from generate_with_swe_trajectory import generate_trajectory


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    return await generate_trajectory(args, sample, sampling_params)


async def reward_func(args, sample, **kwargs):
    # generate_trajectory already set sample.reward = {"score": float, "acc": float}.
    # No PRM, no discount, no dual-shape dispatch.
    if isinstance(sample, list):
        return [s.reward for s in sample]
    return sample.reward
