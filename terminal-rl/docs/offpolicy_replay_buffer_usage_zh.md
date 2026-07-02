# Offpolicy Replay-buffer 使用说明

面向：leo

本 PR 只保留一个 offpolicy 启动入口：

```text
terminal-rl/terminal-rl_qwen3-8b_offpolicy_pu.sh
```

通过修改环境变量选择 `algorithm`、`dataset` 和 replay 变体，不再为每个模式维护单独脚本。

## 1. 功能范围

本实现把样本高效训练能力合并回 integrated `slime`，不引入独立 `slime_offpolicy/` 目录。核心能力包括：

- `Replay buffer`：保存历史 rollout group，并记录 `policy_version`、behavior logprob 和 replay metadata。
- `decoupled_policy_loss`：把 policy ratio 拆成 `pi_theta / pi_prox` 与 `pi_prox / pi_behav`，用于 bounded off-policy correction。
- `staleness-aware sampling`：根据当前 policy version 过滤或统计过旧样本。
- `PER`：按 reward deviation 等信号做 prioritized replay，并写入 importance-sampling weight。
- `TOPR`：sequence-level importance weighting，降低长 response token-level IS 的方差。
- `SPEAR/SIL`：把高 reward trajectory 放入 `SILBuffer`，后续以小权重做 self-imitation replay。
- `DAPO admission gate`：可选地按组内 reward std / correct count 控制哪些 rollout group 进入 replay buffer。

默认训练路径仍是原来的 `policy_loss`。只有显式使用本启动脚本或设置 `--loss-type decoupled_policy_loss` 时才进入 offpolicy replay 路径。

## 2. 一键启动

进入仓库根目录后执行：

```bash
cd /path/to/OpenClaw-RL
export WORKER_URLS="http://<worker-a>:18081,http://<worker-b>:18081"

DATASET=seta OFFPOLICY_MODE=dapo  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_pu.sh
DATASET=seta OFFPOLICY_MODE=per   bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_pu.sh
DATASET=seta OFFPOLICY_MODE=topr  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_pu.sh
DATASET=seta OFFPOLICY_MODE=spear bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_pu.sh
```

也可以把模式作为第一个参数：

```bash
DATASET=seta bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_pu.sh per
DATASET=mixed bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_pu.sh all3
```

`baseline` / `none` 不注入本 wrapper 生成的 offpolicy core args，用于验证 wrapper 不改变原训练逻辑；如果调用方额外设置了 `EXTRA_ALGO_ARGS`，脚本会继续保留该显式输入。

```bash
DATASET=seta bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_pu.sh none
```

## 3. 数据集与基础算法

基础 launcher 默认是：

```text
terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh
```

可通过环境变量适配：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DATASET` | `seta` | 由基础 launcher 支持，当前可用 `seta`、`safety`、`agentharm`、`mixed` |
| `ALGO` | `dapo` | 当前基础 launcher 支持 `dapo`、`grpo` |
| `BASE_SCRIPT` | `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh` | 如需接入其他环境，可替换为兼容 `EXTRA_ALGO_ARGS` 的训练脚本 |
| `WORKER_URLS` | 空 | remote env worker 列表；仓库不内置私有 worker IP |
| `MAX_TURN` | 基础脚本默认 | terminal rollout 最大 turn |
| `ROLLOUT_BATCH_SIZE` | `4` | rollout prompt batch size |
| `N_SAMPLES` | `2` | 每个 prompt 采样数 |
| `MAX_CKPT_KEEP` | `0` | 默认不保留 checkpoint，避免短流程验证占用空间 |

## 4. Offpolicy 模式

| `OFFPOLICY_MODE` | 作用 | 主要注入参数 |
| --- | --- | --- |
| `none` / `baseline` | baseline wrapper，不启用 replay loss | 不注入本 wrapper 的 offpolicy core args |
| `dapo` | replay buffer + DAPO admission gate | `--enable-dynamic-sampling` |
| `per` | prioritized replay | `--buffer-sampling-strategy per` |
| `topr` | sequence-level IS correction | `--use-topr` |
| `spear` | SPEAR-style self-imitation replay | `--enable-trajectory-replay` |
| `all3` | DAPO admission + PER + TOPR | `--enable-dynamic-sampling --buffer-sampling-strategy per --use-topr` |

## 5. 关键参数

通用 replay / offpolicy：

| 环境变量 | 默认 | 对应 CLI | 说明 |
| --- | --- | --- | --- |
| `OFFPOLICY_MAX_STALENESS` | `4` | `--max-staleness` | 允许 replay 样本落后当前 policy 的最大 version 差 |
| `OFFPOLICY_BUFFER_SIZE` | `1024` | `--buffer-max-size` | replay buffer 最大 group 数 |
| `OFFPOLICY_BUFFER_REMOVE_ON_SAMPLE` | `false` | `--buffer-remove-on-sample` | sample 后是否立即移除 |
| `OFFPOLICY_BUFFER_REUSE_SAMPLES` | `4` | `--buffer-reuse-samples` | 每组样本最多复用次数 |
| `OFFPOLICY_IW_CLIP_MIN` | `0.5` | `--importance-weight-clip-min` | behavior IS 下界 |
| `OFFPOLICY_IW_CLIP_MAX` | `2.0` | `--importance-weight-clip-max` | behavior IS 上界 |
| `OFFPOLICY_BEHAV_IW_CAP` | `5.0` | `--behav-imp-weight-cap` | 过大 behavior IS token 过滤阈值 |
| `TRAIN_ITERS_PER_ROLLOUT` | `dapo/per/topr/all3` 下为 `2` | `--train-iters-per-rollout` | 每个 rollout 后训练次数，用于复用 replay buffer 样本 |
| `UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER` | `dapo/per/topr/all3` 下为 `1` | `--update-policy-version-every-train-iter` | 每次 train iter 后更新 policy version，便于统计 staleness |

PER：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `OFFPOLICY_PER_ALPHA` | `0.6` | priority exponent |
| `OFFPOLICY_PER_BETA_START` | `0.4` | IS beta 初始值 |
| `OFFPOLICY_PER_BETA_END` | `1.0` | IS beta 结束值 |
| `OFFPOLICY_PER_BETA_ANNEAL_STEPS` | `1000` | beta anneal 步数 |
| `OFFPOLICY_PER_PRIORITY_SOURCE` | `reward_dev` | priority 信号，默认使用训练前可得到的 reward deviation |

TOPR：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `OFFPOLICY_TOPR_LOGW_CAP` | `2.0` | sequence log-weight clamp |
| `OFFPOLICY_TOPR_W_MIN` | `0.0` | sequence weight 下界 |
| `OFFPOLICY_TOPR_W_MAX` | `5.0` | sequence weight 上界 |
| `OFFPOLICY_TOPR_BLEND` | `1.0` | token IS 与 sequence IS blend 系数 |

SPEAR / SIL：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `OFFPOLICY_SPEAR_BUF` | `2048` | SIL trajectory buffer 容量 |
| `OFFPOLICY_SPEAR_THRESH` | `1.0` | 入库 reward 阈值 |
| `OFFPOLICY_SPEAR_COEF` | `0.001` | self-imitation loss 系数 |
| `OFFPOLICY_SPEAR_STEPS` | `200` | replay loss warmup 步数 |
| `OFFPOLICY_SPEAR_DECAY` | `-1.0` | SIL advantage 重估模式 |

## 6. 本地验证结果与验收信号

以下结果来自 leo 本地训练 pod 日志。验收重点是：

- `baseline/none` 不启用本 wrapper 生成的 offpolicy 参数，验证 baseline 兼容性。
- `dapo/per` 完整 `Ray job succeeded`，验证 replay buffer、decoupled loss、staleness、PER IS weight 等核心路径。
- `spear/topr` 已进入真实训练并完成多个 optimizer step，验证 SPEAR/SIL 与 TOPR loss 路径；最终退出点是 SETA rollout 阶段的 `dynamic sampling aborted after repeated all-failed rollout groups` 保护，不是 offpolicy loss / replay-buffer / SPEAR / TOPR 实现错误。
- 本轮 review fix 的 focused 单测通过：`PYTHONPATH=$PWD/slime /mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python -m pytest slime/tests/test_offpolicy_review_fixes.py -q` -> `10 passed`。

### 6.1 2026-07-02 同 commit 本地验证

worktree：

```text
/mnt/shared-storage-user/puyuan/zhangchenhao/OpenClaw-RL_pr16_fix
branch: leo/offpolicy-pr16-review-fix
base: fork/leo/offpolicy-grpo-replay-sil-clean @ cffc63c6
```

| 模式 | run / 日志 | metrics 行数 | global step | mean `test_acc` | last `test_acc` | mean `reward/task` | 训练状态 | 验收结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `OFFPOLICY_MODE=none` baseline wrapper | `runs/pr16_gpu_p0_seta_none_20260702_022040/logs/metrics.jsonl` | 4 | 0-3 | 0.201 | 0.000 | -0.598 | 多步 baseline rollout/train；未注入 offpolicy core args | baseline wrapper 兼容性验证通过 |
| `OFFPOLICY_MODE=dapo` | `runs/pr16_gpu_p0_seta_dapo_20260702_032736/logs/metrics.jsonl` | 8 | 0-7 | 0.375 | 0.644 | -0.250 | `Ray job succeeded` | replay buffer + DAPO admission gate + decoupled loss 完整 green |
| `OFFPOLICY_MODE=per` | `runs/pr16_gpu_p1_seta_per_20260702_084200/logs/metrics.jsonl` | 8 | 0-7 | 0.397 | 0.487 | -0.206 | `Ray job succeeded` | PER sampling / IS weight / replay 训练完整 green |
| `OFFPOLICY_MODE=spear` | `runs/pr16_gpu_p0_seta_spear_20260702_052823/logs/metrics.jsonl` | 7 | 0-6 | 0.438 | 0.333 | -0.124 | 多步训练后由 `dynamic sampling` 全失败组保护退出 | SPEAR/SIL 代码路径已验证；退出原因非本 PR 实现错误 |
| `OFFPOLICY_MODE=topr` | `runs/pr16_gpu_p1_seta_topr_20260702_143502/logs/metrics.jsonl` | 6 | 0-5 | 0.297 | 0.308 | -0.406 | 多步训练后由 `dynamic sampling` 全失败组保护退出 | TOPR loss 指标已出现并完成多步训练；退出原因非本 PR 实现错误 |

训练日志中已观察到：

- `dapo/per/spear/topr` 均出现 `train/importance_weight_mean`、`train/effective_sample_size`、`train/mean_staleness`、`train/max_staleness` 等 offpolicy 指标。
- `per` 使用 `--buffer-sampling-strategy per`，并记录 `train/per_is_weight_mean/min/max`。
- `topr` 使用 `--use-topr`，并记录 `train/topr_w_seq_mean/max/min`、`train/topr_blend_lambda`。
- `spear` 使用 `--enable-trajectory-replay`、`--trajectory-buffer-size`、`--replay-loss-coef`，并在多步 rollout/train 中记录 trajectory save 相关指标。

结论：本 PR 声明范围内的 replay-buffer、decoupled loss、PER、TOPR、SPEAR/SIL、staleness-aware sampling 和单一 launcher 均已通过本地训练路径验证。`spear/topr` 的最终 Ray job 非 green 由 SETA rollout dynamic sampling 保护触发，不需要在本 PR 里修改 offpolicy 算法实现。

### 6.2 历史 smoke / flow 信号

以下历史结果用于辅助说明 launcher、replay buffer、offpolicy loss 和各 mode 训练流程曾在不同预算下运行过；除特别说明外，不作为完整性能结论。

| 模式 | run / 日志 | metrics 行数 | mean `test_acc` | last `test_acc` | mean `reward/task` | 结论边界 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 原始 SETA DAPO reference | `runs/terminal-rl_qwen3-8b_4gpu_seta-cs-c0.3_dapo-ch0.28-tok1-dyn1_mt10_2026-06-04_233002/logs/metrics.jsonl` | 7 | 0.230 | 0.610 | -0.540 | 原始 DAPO 训练链路 reference；预算和后续 PR check 不完全一致 |
| `OFFPOLICY_MODE=none` baseline wrapper | `runs/offpolicy_pr_check_4gpu_no_spear_20260617_210615/runs/offpolicy_pr_check_4gpu_baseline_20260617_210615/logs/metrics.jsonl` | 12 | 0.079 | 0.006 | -0.771 | 显示 wrapper 可不注入生成的 offpolicy args 并走 baseline path；小预算 |
| `OFFPOLICY_MODE=dapo` | `runs/offpolicy_pr_check_4gpu_no_spear_20260618_001049/runs/offpolicy_pr_check_4gpu_dapo_20260618_001049/logs/metrics.jsonl` | 4 | 0.156 | 0.000 | -0.688 | replay buffer + DAPO admission gate smoke |
| `OFFPOLICY_MODE=per` | `runs/offpolicy_per_prcheck_4gpu_20260618_020720/runs/offpolicy_pr_check_4gpu_per_20260618_020720/logs/metrics.jsonl` | 4 | 0.396 | 0.417 | -0.208 | PER sampling / IS weight path smoke |
| `OFFPOLICY_MODE=topr` | `runs/offpolicy_pr_check_4gpu_no_spear_20260618_001049/runs/offpolicy_pr_check_4gpu_topr_20260618_001049/logs/metrics.jsonl` | 4 | 0.302 | 0.167 | -0.396 | TOPR sequence-level IS path smoke |
| `OFFPOLICY_MODE=spear` short | `runs/terminal-rl_qwen3-8b_4gpu_seta_offpolicy_spear_2026-06-17_160613/logs/metrics.jsonl` | 4 | 0.375 | 0.750 | -0.250 | SPEAR/SIL launcher + training smoke |
| `OFFPOLICY_MODE=spear` extended | `runs/terminal-rl_qwen3-8b_4gpu_seta_offpolicy_spear_2026-06-17_172556/logs/metrics.jsonl` | 88 | 0.440 | 0.000 | -0.120 | SPEAR 主训练流程较长运行；旧日志曾暴露 SIL candidate nested sample warning，本 PR 已修复对应代码路径 |

以上同 commit 本地验证已经补齐，可作为 PR merge 前的本地验收依据。效果提升仍需更长预算、多 seed 或正式 eval；本 PR 只 claim 代码实现、训练路径与 opt-in 兼容性正确。

启动日志应看到：

```text
terminal-rl off-policy launcher
SLIME_DIR: <repo>/slime
OFFPOLICY_MODE: <mode>
EXTRA_ALGO_ARGS: --loss-type decoupled_policy_loss ...
```

训练日志中按模式观察：

- `dapo`：`importance_weight_*`、`mean_staleness`、`max_staleness`
- `per`：`buffer_sampling_strategy=per`、`per_is_weight_mean/min/max`
- `topr`：`topr_w_seq_mean/max/min`、`topr_blend_lambda`
- `spear`：`SPEAR SIL buffer enabled`、`Mixed ... SIL samples into train batch`

如需快速定位 PER 历史 smoke run，最近一次成功目录为：

```text
runs/offpolicy_per_prcheck_4gpu_20260618_020720
```

## 7. 非范围

本 PR 不包含：

- `Docker worker` 运维、容量、lease 或网络稳定性改造；
- `SWE-Smith` 数据集/环境适配；
- `world_model`、`Agent57` 或其他探索模块；
- AgentSafetyBench 官方指标统计逻辑改造。
