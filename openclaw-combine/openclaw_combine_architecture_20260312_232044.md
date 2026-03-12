# OpenClaw-Combine 架构文档

**生成时间**: 2026-03-12 23:20:44

## 概述

OpenClaw-Combine 是一个混合训练系统，将 **Binary RL (GRPO)** 和 **On-Policy Distillation (OPD)** 结合在一起，同时利用评估信号（好/坏）和方向性信号（教师纠正）进行强化学习训练。

## 核心组件

### 1. `openclaw_combine_api_server.py`

**功能**: 异步 API 服务器，继承自 `OpenClawOPDAPIServer`

**核心职责**:
- 管理每个对话回合的双重评估流程（hint-judge + eval-judge）
- 根据评估结果生成三种类型的训练样本：
  - **OPD+RL 样本**: hint 被接受 + eval 得分 ±1 → 包含教师 log-probs 和 RL reward
  - **OPD-only 样本**: hint 被接受 + eval 失败 → 仅包含教师 log-probs，reward=0
  - **RL-only 样本**: hint 被拒绝 + eval 得分 ±1 → 使用 rollout log-probs 作为伪教师信号，reward=±1

**与其他组件交互**:
- **继承**: `OpenClawOPDAPIServer` (openclaw-opd/openclaw_opd_api_server.py)
- **输出**: 将样本推送到 `output_queue`，供 rollout worker 消费
- **调用**: SLIME 的 `Sample` 类型构建训练数据

**关键方法**:
- `_submit_turn_sample()`: 提交 OPD 或混合样本（包含真实教师 log-probs）
- `_submit_rl_turn_sample()`: 提交纯 RL 样本（使用 rollout log-probs 作为伪教师信号）
- `_maybe_submit_ready_samples()`: 决策逻辑，根据 hint 和 eval 结果选择样本类型

### 2. `openclaw_combine_rollout.py`

**功能**: Rollout 桥接层，连接 OpenClaw API 服务器和 SLIME 训练器

**核心职责**:
- 管理全局 `AsyncRolloutWorker` 单例
- 控制样本提交的暂停/恢复（训练时开启，非训练时暂停）
- 从 `output_queue` 收集完成的样本组，直到达到 `rollout_batch_size`
- 返回 `RolloutFnTrainOutput` 给 SLIME 训练循环

**与其他组件交互**:
- **调用**: `OpenClawCombineAPIServer` 启动异步服务
- **输出**: SLIME 的 `RolloutFnTrainOutput` 类型
- **配置**: 通过 `args.rollout_batch_size` 控制批次大小

**关键函数**:
- `generate_rollout_openclaw_combine()`: SLIME 调用的主入口点
- `_drain_output_queue()`: 异步收集样本直到达到目标批次大小
- `get_global_worker()`: 获取或创建全局 worker 单例

### 3. `combine_loss.py`

**功能**: 混合损失函数，计算加权优势并应用 PPO 裁剪策略梯度

**核心职责**:
- 计算两种优势的加权和：
  ```
  combined_adv = w_opd * (teacher_logp - old_logp) + w_rl * grpo_advantage
  ```
- 应用与 SLIME 相同的 PPO 裁剪损失
- 支持熵正则化和 KL 散度惩罚

**与其他组件交互**:
- **被调用**: SLIME 训练器通过 `--custom-loss-function-path combine_loss.combine_loss_function`
- **输入**: SLIME 的 batch 字典（包含 `advantages`, `teacher_log_probs`, `rollout_log_probs` 等）
- **输出**: 标量损失和指标字典

**关键参数**:
- `OPENCLAW_COMBINE_W_OPD`: OPD 优势权重（默认 1.0）
- `OPENCLAW_COMBINE_W_RL`: RL 优势权重（默认 1.0）

### 4. `run_qwen3_4b_openclaw_combine_pu.sh`

**功能**: 启动脚本，配置并启动整个训练流程

**核心配置**:
- **GPU 分配**:
  - `NUM_GPUS=4`: 总 GPU 数
  - `ACTOR_GPUS=2`: Actor 模型训练 GPU
  - `ROLLOUT_GPUS=1`: Rollout 生成 GPU
  - `PRM_GPUS=1`: PRM 评估 GPU
- **模型并行**: `--tensor-model-parallel-size 2` (必须整除 ACTOR_GPUS)
- **Rollout 配置**:
  - `--rollout-batch-size 16`: 每批次样本数
  - `--rollout-max-response-len 8192`: 最大响应长度
- **PRM 配置**:
  - `--prm-enable`: 启用 PRM 评估
  - `--prm-m 1`: 每回合评估投票数

**与其他组件交互**:
- **启动**: Ray 集群和训练任务
- **调用**: `train_async.py` (SLIME 主训练脚本)
- **配置**: 通过环境变量和命令行参数传递配置

## 数据流

```
用户交互 / 环境反馈
    ↓
OpenClawCombineAPIServer
    ├─→ Hint Judge (m 次投票)
    ├─→ Eval Judge (m 次投票)
    └─→ 样本决策逻辑
         ├─→ OPD+RL 样本 (hint✓ + eval±1)
         ├─→ OPD-only 样本 (hint✓ + eval✗)
         └─→ RL-only 样本 (hint✗ + eval±1)
    ↓
output_queue
    ↓
AsyncRolloutWorker (_drain_output_queue)
    ↓
RolloutFnTrainOutput (samples + metrics)
    ↓
SLIME Trainer (train_async.py)
    ↓
combine_loss_function
    ├─→ 计算混合优势
    └─→ PPO 裁剪损失
    ↓
Megatron-LM 反向传播
    ↓
模型参数更新
```

## 与 SLIME 集成点

1. **Rollout 函数**: `--rollout-function-path openclaw_combine_rollout.generate_rollout_openclaw_combine`
2. **损失函数**: `--custom-loss-function-path combine_loss.combine_loss_function`
3. **生成函数**: `--custom-generate-function-path openclaw_combine_api_server.generate`
4. **奖励函数**: `--custom-rm-path openclaw_combine_api_server.reward_func`

## 关键设计决策

### 样本类型决策表

| Hint 接受? | Eval ±1? | 结果 |
|-----------|---------|------|
| ✓ | ✓ | 1 个混合样本 (OPD+RL) |
| ✓ | ✗ | 1 个 OPD 样本 |
| ✗ | ✓ | 1 个 RL 样本 |
| ✗ | ✗ | 无样本 |

### 优势计算

- **OPD 样本**: `teacher_logp - old_logp` (token-level)
- **RL 样本**: `reward` 广播到所有 token (sequence-level)
- **混合样本**: 两者相加

### 为什么有效？

- **OPD 样本** 的 `reward=0` → GRPO 优势为 0
- **RL 样本** 的 `teacher_logp ≈ rollout_logp` → 教师优势 ≈ 0
- 每种样本类型自然主导其对应的优势分支

## 配置建议

### GPU 分配约束

```bash
# 必须满足:
ACTOR_GPUS % tensor_model_parallel_size == 0
ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS <= NUM_GPUS
```

### 常见配置

**4 GPU 配置** (当前):
```bash
NUM_GPUS=4
ACTOR_GPUS=2
ROLLOUT_GPUS=1
PRM_GPUS=1
--tensor-model-parallel-size 2
```

**8 GPU 配置** (推荐):
```bash
NUM_GPUS=8
ACTOR_GPUS=4
ROLLOUT_GPUS=2
PRM_GPUS=2
--tensor-model-parallel-size 4
```

## 监控指标

- `rollout/prm_eval_score`: PRM 评估平均分
- `pg_loss`: 策略梯度损失
- `pg_clipfrac`: PPO 裁剪比例
- `ppo_kl`: 新旧策略 KL 散度
- `entropy_loss`: 熵损失
- `train_rollout_logprob_abs_diff`: 训练与 rollout log-prob 漂移

## 故障排查

### AssertionError: world_size not divisible by total_model_size

**原因**: `ACTOR_GPUS` 不能被 `tensor_model_parallel_size` 整除

**解决**: 调整 GPU 分配或修改 `--tensor-model-parallel-size`

### 样本收集超时

**症状**: `waiting for combine samples` 日志持续出现

**可能原因**:
- PRM 服务未启动或响应慢
- Hint judge 拒绝率过高
- Eval judge 失败率过高

**解决**: 检查 `PRM_M`, `ROLLOUT_GPUS`, `PRM_GPUS` 配置
