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

`baseline` / `none` 会清空 offpolicy core args，用于验证 wrapper 不改变原训练逻辑：

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
| `none` / `baseline` | baseline wrapper，不启用 replay loss | 清空 offpolicy core args |
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

## 6. 验收信号

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

已在训练 pod 中完成 DAPO、TOPR、SPEAR、PER 流程验证；PER 最近一次成功目录为：

```text
runs/offpolicy_per_prcheck_4gpu_20260618_020720
```

## 7. 非范围

本 PR 不包含：

- `Docker worker` 运维、容量、lease 或网络稳定性改造；
- `SWE-Smith` 数据集/环境适配；
- `world_model`、`Agent57` 或其他探索模块；
- AgentSafetyBench 官方指标统计逻辑改造。
