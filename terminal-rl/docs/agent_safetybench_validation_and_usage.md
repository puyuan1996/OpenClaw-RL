# Agent-SafetyBench Validation And Usage

## Minimal Validation

Run syntax checks:

```bash
python -m py_compile \
  terminal-rl/remote/agent_safetybench_env.py \
  terminal-rl/remote/terminal_env.py \
  terminal-rl/remote/pool_server.py \
  terminal-rl/env_client.py \
  terminal-rl/generate.py \
  terminal-rl/data_utils/convert_agent_safetybench_to_dataset.py \
  terminal-rl/data_utils/mix_jsonl_datasets.py \
  terminal-rl/scripts/validate_agent_safetybench_backend.py \
  terminal-rl/scripts/validate_agent_safetybench_reward_scale.py
```

Run ASB backend smoke test:

```bash
python terminal-rl/scripts/validate_agent_safetybench_backend.py --sample-id 0
```

Run ASB reward scale smoke test:

```bash
python terminal-rl/scripts/validate_agent_safetybench_reward_scale.py
```

Run ratio mixer smoke test:

```bash
python terminal-rl/data_utils/mix_jsonl_datasets.py \
  --source terminal-rl/dataset/seta_env_convert/train.jsonl:7 \
  --source terminal-rl/dataset/agent_safetybench_convert/train.jsonl:3 \
  --output /tmp/mixed_seta_safety_7_3.jsonl \
  --seed 42
```

## Regenerate Agent-SafetyBench JSONL

```bash
python terminal-rl/data_utils/convert_agent_safetybench_to_dataset.py \
  --input /mnt/shared-storage-user/puyuan/code/Agent-SafetyBench/data/released_data.json \
  --output-dir terminal-rl/dataset/agent_safetybench_convert
```

## Mixed Training Example

Config file:

```text
terminal-rl/configs/mixed_seta_agent_safetybench.env
```

Launch:

```bash
set -a
source terminal-rl/configs/mixed_seta_agent_safetybench.env
set +a
bash terminal-rl/terminal-rl_qwen3-8b_pu.sh
```

## One-step Training Commands

Agent-SafetyBench only:

```bash
DATASET=safety \
SAFETY_BENCH_REWARD=rule \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES=1 \
bash terminal-rl/terminal-rl_qwen3-8b_pu.sh
```

Mixed seta + Agent-SafetyBench:

```bash
DATASET=mixed \
MIX_SETA_RATIO=7 \
MIX_SAFETY_RATIO=3 \
SAFETY_BENCH_REWARD=rule \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES=1 \
bash terminal-rl/terminal-rl_qwen3-8b_pu.sh
```

These commands assume the usual terminal-rl services and model/runtime dependencies are available.
