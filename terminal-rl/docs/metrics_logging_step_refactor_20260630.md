# RL 指标日志系统现状分析与重构说明

## 现状分析

### 三种 step 的定义

1. 文本 log 中旧格式 `========== step N ==========` 来自 `terminal-rl/rollout_log.py::_format_per_dataset_table`。该表由 `rollout_log()` 在 rollout 结束后打印，`step` 参数来自 `compute_rollout_step(args, rollout_id)`，因此语义是 rollout 日志轴，不是 optimizer step。重构后标题显式改为 `rollout-step ... | train-step ... | legacy rollout/step ... | steps/rollout ...`。
   - 源码：`terminal-rl/rollout_log.py:2346-2390`, `terminal-rl/rollout_log.py:3122-3129`
   - 日志抽样：`runs/terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0629_stable_softguard_dualadv_think_2026-06-29_154554/logs/train.log:3395`, `:3844`, `:4337`, `:5887`

2. 文本 log 中旧格式 `step 0: {'train/loss': ...}` 是训练后端的 optimizer train-step 日志。
   - FSDP 后端：`slime/slime/backends/fsdp_utils/actor.py:746-749`，`train/step = self.global_step`，打印后 `self.global_step += 1`。
   - Megatron 后端：`slime/slime/backends/megatron_utils/model.py:638-659`，`train_step_id = args._monotonic_train_step`，同时记录 `train/rollout_id`, `train/rollout_step_id`, `train/num_steps_per_rollout`, `train/legacy_accumulated_step`。
   - 日志抽样：同一 train.log 中 `:3512`, `:3866`, `:5996`, `:5997` 显示 `train/step`, `train/rollout_id`, `train/rollout_step_id`, `train/num_steps_per_rollout=2`。
   - 重构后文本前缀改为 `train-step N: {...}`。

3. W&B 里看到的第三个更大的 step 是 W&B 内部默认 `_step`，不是仓库定义的训练步。原因是统一日志函数调用 `wandb.log(metrics)` 时没有传 `step=`，W&B 会按 log 调用次数自增。
   - 源码：`slime/slime/utils/logging_utils.py:36-41`
   - 因此需要优先看自定义横轴：`axis/rollout_step`, `axis/train_step`，不要用 W&B 默认 `_step` 判断训练进度。

### rollout-step 与 train-step 换算

原有 `compute_rollout_step()`：

```python
if args.wandb_always_use_train_step:
    return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
return rollout_id
```

源码：`slime/slime/utils/metric_utils.py:120-123`。

本次重构新增显式轴：

- `axis/rollout_step = rollout_id`
- `axis/train_step = rollout_id * steps_per_rollout`
- `axis/steps_per_rollout = num_steps_per_rollout`，若未配置则按 `rollout_batch_size * n_samples_per_prompt // global_batch_size` 推导
- `axis/legacy_rollout_step = compute_rollout_step(args, rollout_id)`，保留旧 `rollout/step` 兼容行为

源码：`terminal-rl/rollout_log.py:1758-1777`, `terminal-rl/rollout_log.py:3122-3128`。

### `terminal/reward_mean` 的真实语义

`terminal/reward_mean` 直接读取 trainable sample 的 `reward["score"]` 均值：

```python
trainable_rewards = [
    v for v in (_reward_value(s, "score") for s in trainable) if v is not None
]
log_dict["terminal/reward_mean"] = sum(trainable_rewards) / len(trainable_rewards)
```

源码：`terminal-rl/rollout_log.py:3052-3060`。

结论：`terminal/reward_mean` 是 legacy score 均值，不应再解释为“未经 intrinsic/penalty 修改的原始 task reward”。原始 task reward 的新规范字段是 `reward/task`，来源优先级为 `base_score -> raw_score -> score`。

### 当前混乱点

- rollout 表的旧标题只有 `step`，但该 step 可能是 raw `rollout_id`，也可能在 `wandb_always_use_train_step=1` 时变成 train-step 等价值。
- train loss 文本行旧前缀也叫 `step`，与 rollout 表冲突。
- W&B 默认 `_step` 是 log 调用次数，和 rollout-step / train-step 都不是同一概念。
- `terminal/reward_mean` 名字像 task reward，但实际读的是 `score`。
- 探索奖励字段分散在 `terminal/explore/*`, `per_dataset/*/reward/*`, `reward/adv_*` 等命名空间，缺少一套同时可按 rollout-step 与 train-step 对齐的核心面板。

## 重构后的指标命名与查看方式

核心指标统一由 `terminal-rl/rollout_log.py::_reward_fusion_axis_metrics()` 计算，源码：`terminal-rl/rollout_log.py:917-955`。

规范字段：

- `reward/task`：原始任务奖励，`base_score -> raw_score -> score`
- `reward/raw`：原始 raw score，`raw_score -> base_score -> score`
- `reward/total`：trainer post-process 后总 reward，`postprocess_total_reward -> score`
- `intrinsic/intra`：局内 episodic intrinsic，`explore_agent57_ngu_episodic`
- `intrinsic/inter`：局间 lifelong intrinsic bonus，`explore_agent57_lifelong_bonus`
- `intrinsic/inter_raw`：局间 lifelong raw novelty，`explore_agent57_lifelong_raw`
- `intrinsic/fused`：局内与局间融合后的 intrinsic signal，`explore_agent57_intrinsic_signal`
- `adv/task`：post-normalization intrinsic/penalty 注入前的 task advantage stream
- `adv/intrinsic`：最终加到 task advantage 上的 intrinsic bonus，`explore_post_norm_bonus`
- `adv/final_penalty`：最终加到 task advantage 上的 penalty，当前主要是 truncation penalty
- `adv/exploration_delta`：`adv/intrinsic + adv/final_penalty`
- `adv/with_penalty`：penalty 作用后的最终 advantage stream，优先读 `explore_post_norm_adjusted_reward`

W&B 横轴：

- 默认 rollout 横轴：`reward/*`, `intrinsic/*`, `adv/*`，step metric 为 `axis/rollout_step`
- 显式 rollout 镜像：`rollout_axis/reward/*`, `rollout_axis/intrinsic/*`, `rollout_axis/adv/*`
- 显式 train-step 镜像：`train_axis/reward/*`, `train_axis/intrinsic/*`, `train_axis/adv/*`，step metric 为 `axis/train_step`

W&B 定义位置：

- 初始化公共定义：`slime/slime/utils/wandb_utils.py:155-167`
- terminal hook 兜底定义：`terminal-rl/rollout_log.py:263-276`

文本 log：

- 总览表：`reward fusion metrics`，标题含 `rollout-step`, `train-step`, `legacy rollout/step`, `steps/rollout`
- 分数据集表：`per-dataset metrics`，列出 `task_reward`, `total_reward`, `intra`, `inter`, `fused`, `adv_task`, `adv_intr`, `penalty`, `adv_final`, `pass`, `trunc`

结构化 JSONL：

- schema version 升级到 `8`
- 每条记录新增 `rollout_step`, `train_step`, `steps_per_rollout`, `wandb_rollout_step`
- 每条 sample-level 记录新增同一套 `intrinsic/*` 与 `adv/*` 字段
