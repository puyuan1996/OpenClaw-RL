# Agent-SafetyBench Adapter Analysis

## Step 1: Official Agent-SafetyBench

Source repo: `/mnt/shared-storage-user/puyuan/code/Agent-SafetyBench`

Main data file:

- `data/released_data.json`
- 2000 samples
- Fields: `id`, `risks`, `instruction`, `environments`, `failure_modes`, `fulfillable`
- 111 samples also contain `dialog`

Distribution:

- `fulfillable=0`: 1265 harmful or unsafe-to-fulfill samples
- `fulfillable=1`: 735 benign or conditionally fulfillable samples
- 8 risk categories, 250 samples each
- 350 environment names
- 308 no-tool samples use an empty environment name

Interaction flow:

- Entry: `evaluation/eval.py`
- `parse_envs()` initializes environments and collects tool schemas.
- `evaluate()` builds messages, runs up to 10 model rounds, executes one tool call at a time, and stops on final assistant content.
- Tool results are appended as OpenAI-style `tool` messages.

Environment API:

- `environments/EnvManager.py` dynamically imports an environment class by name.
- `environments/BaseEnv.py` loads the matching JSON tool schema file.
- `BaseEnv.call_tool()` validates required parameters, drops extra parameters, coerces JSON schema types, and calls the environment method.

Evaluator:

- Generation does not directly return reward.
- Safety scoring is done by `score/eval_with_shield.py`.
- The scorer formats environment information, dialogue, tool calls, and tool outputs into a ShieldAgent prompt.
- ShieldAgent output is parsed as `[Answer] safe|unsafe`.

## Step 2: terminal-rl seta Flow

Data loading:

- Launch script passes `--prompt-data`, `--input-key task`, and `--reward-key score`.
- `slime/slime/utils/data.py` reads JSONL records into `Sample(prompt=..., metadata=...)`.
- `slime/slime/rollout/data_source.py` duplicates each prompt into GRPO groups via `n_samples_per_prompt`.

Rollout:

- `terminal-rl/generate.py` extracts `metadata` as `task_meta`.
- `TaskSpec` is built from `task_name`, `task_path`, and `instruction`.
- The remote environment is allocated through `ENV_SERVER_URL`.
- `/reset` returns `user_msg` and OpenAI-compatible tool schemas.
- `CamelAgent` receives that message, model generations, tool calls, and tool observations through a single rollout loop.

Reward:

- seta uses `TerminalEnv.evaluate()` to run terminal-bench tests.
- The returned accuracy is mapped to training reward by `2 * accuracy - 1`.
- Optional ClawSentry shaping is added per turn when enabled.

Config:

- `DATASET=seta|safety|mixed`
- `SETA_SAFETY=none|clawsentry`
- `SAFETY_BENCH_REWARD=rule|shield_prompt|clawsentry`
- `mixed` supports legacy full concatenation and optional ratio-based mixing.

## Step 3: Diagnostics

1. Agent-SafetyBench reset failed before reward.
   - File: `terminal-rl/remote/terminal_env.py`
   - Root cause: all samples were treated as terminal-bench Docker tasks and `TrialHandler` looked for `agent_safetybench/<id>/task.yaml`.

2. ASB tools were not exposed.
   - File: `terminal-rl/remote/terminal_env.py`
   - Root cause: reset always returned terminal shell tools, not ASB mock environment tools.

3. ASB reward branch was unreachable.
   - File: `terminal-rl/generate.py`
   - Root cause: the local rule reward ran only after reset and rollout, but reset failed first.

4. ASB reward scale was corrupted.
   - File: `terminal-rl/generate.py`
   - Root cause: `_build_samples()` always interpreted `outcome` as accuracy and mapped it through `2 * accuracy - 1`.

5. Official evaluator semantics were not preserved.
   - File: `terminal-rl/generate.py`
   - Root cause: only a local refusal heuristic existed; no official-style trajectory or ShieldAgent prompt was generated.

6. Converted prompts duplicated tool information.
   - File: `terminal-rl/dataset/agent_safetybench_convert/train.jsonl`
   - Root cause: tool names were appended to the natural-language prompt even though terminal-rl should provide structured tool schemas from env reset.

7. Mixed data had no ratio control.
   - File: `terminal-rl/terminal-rl_qwen3-8b_pu.sh`
   - Root cause: `mixed` simply concatenated full seta and ASB JSONL files.

8. Dataset docs used stale env var names.
   - File: `terminal-rl/dataset/DATASET_ANALYSIS.md`
   - Root cause: docs referred to `DATASET_MODE` and `SAFETY_REWARD_MODE`, while the script uses `DATASET` and `SAFETY_BENCH_REWARD`.
