# Terminal-RL Exploration 技术文档

> 面向接手同事的中文说明，基于当前代码实现（`generate.py`、`terminal-rl_qwen3-8b_exploration_pu.sh`、`terminal-rl_qwen3-8b_pu.sh`）撰写。本文档原为 `README.md`，现重命名为 `README_EXPLORATION.md` 以准确表达主题；历史 `doc_tmp/EXPLORATION.md` 与 `doc_tmp/LAMER_AGENT57_INTEGRATION.md` 保留以供参考，但实际行为以本文档和代码为准。

---

## 一、概览

本模块在 Terminal-RL（基于 slime + Megatron-LM 的 GRPO 在线强化学习框架）之上，叠加多种 **探索增强（Exploration Bonus）** 与训练超参覆盖，目的是改善 LLM Agent 在稀疏奖励终端任务（Terminal-Bench、SETA、CTF、Agent-SafetyBench）上的样本效率与覆盖广度。

核心思路借鉴：

| 工作 | 借鉴点 |
|------|--------|
| **MERCI** (Count Counts) | 1/√N count-based 内在奖励的整体结构 |
| **Agent57** (DeepMind 2020) | 子目标粒度新颖性、生命周期价值的概念 |
| **LaMer** (ICLR '26) | 多次尝试 + 反思（接口已就绪，实际重启需 `agent_runner` 支持） |
| **AEPO** | 熵 bonus 防止 mode collapse |
| **CDE for RLVR** | actor 侧 PPL curiosity bonus：奖励低概率但有 verifiable reward 支撑的探索回复 |
| **SPEAR** | curriculum intrinsic reward + self-imitation 的分层设计；当前先接入低侵入的 intrinsic curriculum |

**设计约束**：所有探索功能默认全部 **关闭**；只要不设任何 `EXPLORE_*` / `EXPLORATION_PROFILE` 环境变量，wrapper 的训练语义与原 baseline 等价（最终仍 `exec` 到主脚本）。

本模块共暴露一组 `EXPLORATION_*` / `EXPLORE_*` 环境变量，分别控制数据/算法对齐、探索预设与多类增强（见 §五速查表）。

### 1.1 当前 exploration option 的关系

现有选项按作用层次分为四类：

| 层次 | 选项 | 作用位置 | 与其它选项的关系 |
|------|------|----------|------------------|
| 采样分布 | `EXPLORE_TEMP_HIGH`、`EXPLORE_THINK_MODE` | rollout 前 | 增加生成多样性或推理长度，不直接改 reward；温度过高会放大 parse error 和无效工具调用 |
| 训练 loss | `EXPLORE_ENTROPY_COEF` | slime actor loss | 与 reward bonus 正交；不看具体样本质量，系全局熵控制，建议小权重 |
| reward shaping | `EXPLORE_INTRINSIC`、`EXPLORE_LPRND`、`EXPLORE_CDE_ACTOR`、`EXPLORE_SAFETY_FILTER` | `generate.py` 合成 reward 后 | 都会改变 `score`；前三个偏探索，`EXPLORE_SAFETY_FILTER` 是负向安全约束 |
| curriculum / schedule | `EXPLORE_INTRINSIC_SCHEDULE`、`EXPLORE_LPRND_SCHEDULE`、`EXPLORE_CDE_ACTOR_DECAY_STEPS` | reward shaping 系数 | 控制早期探索、后期利用；默认关闭以保持兼容 |
| post-normalization bonus | `EXPLORE_ADVANTAGE_BONUS` | slime reward post-process | 可把选定探索项加到 GRPO group normalization 之后，避免同组 baseline 抵消探索信号 |

推荐组合原则：

- `EXPLORE_INTRINSIC` 适合 `seta` 这类有真实工具/命令的 agentic 数据；对纯文本安全样本通常为 0。
- `EXPLORE_INTRINSIC_SCHEDULE=cosine` 是 SPEAR 最容易迁移的部分：早期鼓励工具交互，后期逐步降低辅助奖励，避免和 outcome reward 竞争。
- `EXPLORE_LPRND` 与 `EXPLORE_CDE_ACTOR` 都使用 logprob 信号；前者做 worker-local z-score novelty，后者做论文 CDE 的 bounded PPL bonus。不要同时给太大权重。
- safety / mixed 训练中如启用 `EXPLORE_CDE_ACTOR`，建议设 `EXPLORE_CDE_ACTOR_REWARD_GATE=positive`，避免给负 reward 的不安全轨迹补正向 curiosity。
- `EXPLORE_ADVANTAGE_BONUS=1` 是对 `Bug 6` 的可选补救，只建议消融时打开；默认关闭以保持原有 GRPO reward normalization 行为。
- `EXPLORE_ENTROPY_COEF` 是分布级探索，`EXPLORE_CDE_ACTOR` 是样本级探索；二者可同时开，但建议一个主导、另一个小权重。
- `EXPLORE_SAFETY_FILTER` 与其它探索 bonus 正交，适合混合安全训练时防止探索奖励鼓励危险命令。
- `EXPLORE_RETRY_ATTEMPTS` 当前只是透传旗标，尚未实际改变 `agent_runner` 行为。

---

## 二、Baseline 介绍

### 2.1 框架栈

- **训练框架**：[slime](https://github.com/THUDM/slime)（GRPO 异步在线 RL）+ Megatron-LM（Tensor Parallel）。
- **模型**：Qwen3-8B（默认 `HF_CKPT=.../slime/Qwen3-8B/`），4–8 GPU。
- **Rollout 架构**：CPU worker 上跑 `pool_server`（DooD docker），GPU worker 上跑 actor + sglang rollout。
- **任务集**：`DATASET` 环境变量切换 `seta` / `safety` / `agentharm` / `mixed`。

### 2.2 关键文件

```
terminal-rl/
├── terminal-rl_qwen3-8b_pu.sh              # 主训练脚本（baseline）
├── terminal-rl_qwen3-8b_exploration_pu.sh  # 探索 wrapper（exec 上面那个）
├── generate.py                             # rollout + 奖励合成（已注入探索代码）
├── configs/
│   ├── rollout_qwen3.yaml                  # 默认 rollout 配置
│   └── rollout_qwen3_think.yaml            # Qwen3 think-mode 配置
├── README_EXPLORATION.md                   # ← 本文件：exploration 技术文档
├── README_SETUP.md                         # 项目/远端 worker 启动入口
└── doc_tmp/
    ├── EXPLORATION.md                      # （历史文档，保留）
    └── LAMER_AGENT57_INTEGRATION.md        # （历史文档，保留）
```

### 2.3 Baseline 奖励结构（`generate.py` 中 `_build_samples`）

```
final_score = discounted_base                 # outcome reward, 2*acc-1
            + prm_coef * prm_turn_score       # 可选：PRM 评判
            + safety_coef * safety_val        # 可选：ClawSentry 安全奖励
```

本探索模块在 `_build_samples` 之后再追加：

```
final_score += effective_intrinsic_coef * intrinsic_bonus
            + safety_penalty                  # 危险命令负惩罚
            + EXPLORE_LPRND_COEF * lprnd_bonus
            + cde_actor_bonus                 # 可选：actor PPL curiosity bonus
```

这些字段会写入 `runs/{run_id}/trajectories/.../meta.json` 的 `reward` 子字典，便于事后归因。

---

## 三、Exploration 技术原理

### 3.1 Entropy Bonus（AEPO 风格）

通过 slime 原生 `--entropy-coef` 参数加入 entropy loss：

$$\mathcal{L} = \mathcal{L}_\text{GRPO} - \beta_{ent}\,H(\pi_\theta)$$

防止 mode collapse（baseline 默认 `entropy_coef=0`，曾观察到 entropy 在某些 rollout 前坍缩至 0）。

### 3.2 Think Mode（Qwen3 CoT）

切换 rollout 配置文件至 `configs/rollout_qwen3_think.yaml`，将 `non_think_mode: false`，启用 Qwen3 原生 `<think>...</think>` CoT。

### 3.3 Rollout 温度覆盖

通过环境变量 `ROLLOUT_TEMPERATURE` 覆盖 baseline 的 `--rollout-temperature 1`，提升采样多样性。

### 3.4 Count-based 内在奖励（MERCI 简化版）

$$r^\text{intr} = \sum_{i \in \text{turns}} \frac{1}{\sqrt{N(s_i)}}$$

其中 $N(s_i)$ 为命令 $s_i$ 在进程级计数器中的累积访问次数。

**两种粒度**（`EXPLORE_INTRINSIC_GRANULARITY`）：

- `raw`（默认）：完整命令字符串 MD5。
- `signature`：仅取 `cmd名 | arg1 | arg2`（`shlex.split(cmd)[:3]`），将近义写法（如 `ls -la /tmp` 与 `ls -al /tmp/`）归入同桶 —— 对应 Agent57 子目标粒度。

与 MERCI 原版的关系：MERCI 用 Coin Flipping Network 估计 token-level pseudo-count；本实现是其 **简化版**，用确定性哈希在 command 粒度计数，零额外参数。

**SPEAR-style curriculum**：`EXPLORE_INTRINSIC_SCHEDULE` 可将 `EXPLORE_INTRINSIC_COEF` 变成随训练步变化的有效系数：

$$\mu_h=\mu_0\cdot m(h),\quad R=R_\text{outcome}+\mu_h R_\text{intrinsic}+\cdots$$

支持：

- `constant`：默认值，`m(h)=1`，完全保持旧行为。
- `cosine`：SPEAR 对 `use_toolcall_reward="cosine"` 的低侵入映射，`m(h)=\frac{\cos(\pi h/H)+1}{2}`。
- `linear`：线性退火，`m(h)=1-h/H`。

其中 `H=EXPLORE_INTRINSIC_DECAY_STEPS`。`H=0` 时退化为 constant。`EXPLORATION_PROFILE=spear_lite` 会自动设置 `EXPLORE_INTRINSIC_SCHEDULE=cosine`、`EXPLORE_INTRINSIC_DECAY_STEPS=200`。

### 3.5 LP-RND 生命周期新颖性（草案 C）

复用 slime 已计算的 `output_token_logprobs`，无需额外前向传播：

$$\bar\ell = \frac{1}{T}\sum_t \log\pi_\theta(a_t | s_t),\quad r^\text{lprnd} = \text{clip}\!\left(\frac{-\bar\ell-\mu}{\sigma},\ 0,\ L\right)$$

- $\mu, \sigma$ 由 **Welford 在线算法** 维护，进程级。
- 前 32 条轨迹为 warmup，期间奖励恒为 0。
- $L$ 由 `EXPLORE_LPRND_CLIP` 控制，默认 3.0。
- 可选 `EXPLORE_LPRND_SCHEDULE=cosine|linear` 对 `EXPLORE_LPRND_COEF` 做退火；默认 `constant`。

**直觉**：当前策略对某轨迹"惊讶"（mean negative-logprob 高）→ 探索到低密度区域 → 给正奖励。Agent57 RND 需要额外 random network；LP-RND 用 policy 自身的对数概率作信号，零参数。

### 3.6 CDE Actor/PPL 好奇心奖励

实现论文中 actor 侧 Curiosity-Driven Exploration bonus：

$$B_\text{actor}(q,o)=-\frac{1}{T}\sum_{t=1}^{T}\log\pi(o_t|o_{<t},q)$$

$$\hat r(q,o)=r(q,o)+\omega\min\left(\frac{|r(q,o)|}{\kappa},\alpha B_\text{actor}(q,o)\right)$$

代码使用 rollout 已有的 `output_token_logprobs` 计算 `B_actor`，无需额外 forward。bonus 默认关闭，通过 `EXPLORE_CDE_ACTOR=1` 启用。

关键设计：

- bonus 以 **pre-exploration score 的绝对值** 做上限；如果基础 reward 为 0（例如 infra failure / 空轨迹），bonus 也为 0，避免 reward hacking。
- `EXPLORE_CDE_ACTOR_OMEGA` 默认 0.05，建议早期小权重试跑；`EXPLORE_CDE_ACTOR_DECAY_STEPS>0` 时线性退火到 0。
- `EXPLORE_CDE_ACTOR_ALPHA` 默认 0.1，用于缩放 log-PPL；`EXPLORE_CDE_ACTOR_KAPPA` 默认 2.0，表示 curiosity 上限为 `|r|/2`。
- `EXPLORE_CDE_ACTOR_REWARD_GATE` 默认 `nonzero`，贴近论文形式；安全混合训练推荐 `positive`，即只对基础 reward 为正的轨迹加 PPL bonus。
- 当前只实现 actor bonus；critic bonus 需要多头 critic/value path，和现有 GRPO/DAPO 训练路径不是同一层改动。

落盘/监控字段：

- `explore_cde_actor_bonus`
- `explore_cde_actor_log_ppl`
- `explore_cde_actor_base_magnitude`
- `explore_cde_actor_cap`
- `explore_cde_actor_scaled`
- `explore_cde_actor_clipped`
- `explore_cde_actor_omega`
- `explore_cde_actor_reward_gate`
- `explore_cde_actor_eligible`

额外的结构化调试字段：

- `explore_mood` / `explore_mood_code`：粗粒度运行状态，取值包括 `confident_exploit`、`curious_success`、`curious_unproven`、`cautious`、`risky`、`stuck`、`low_signal`。
- `explore_bonus_to_base_abs_ratio`：探索 bonus 相对基础 reward 的压力；持续过高说明辅助奖励可能盖过 outcome。
- `explore_reward_hacking_risk`：基础 reward 非正但探索 bonus 为正。
- `explore_over_exploration_risk`：非正基础 reward 且探索压力过高。
- `explore_safety_tension`：命中 safety penalty 或危险动作。
- `explore_action_count` / `explore_tool_call_count` / `explore_danger_command_count`：按结构化 tool call 统计，不再只依赖旧的 `command` 字段。

### 3.7 安全预过滤惩罚

对每个 turn 的命令做正则匹配，命中危险模式则施加负奖励 `EXPLORE_SAFETY_FILTER_COEF`（默认 −0.5）。

匹配模式：`rm -rf /` 系列、`curl|bash` 注入、`chmod 777 /`、写 `/etc/passwd|shadow|sudoers`、读 `/etc/shadow`、fork bomb。

与 ClawSentry 安全奖励正交：前者是 reward shaping，本机制是命令级硬惩罚。

### 3.8 多次尝试反思（LaMer，旗标已就绪，env restart 待实现）

`EXPLORE_RETRY_ATTEMPTS > 1` 时透传环境变量并打印 `[WARN] Multi-attempt requires agent_runner support (not yet implemented)`。

后续 P1：在 `agent_runner.py` 的 run 循环里检测 `EXPLORE_RETRY_ATTEMPTS`，失败时追加 reflection turn 并 `env_client.reset()`；按 `EXPLORE_RETRY_TRAJ_GAMMA^attempt_idx` 折扣奖励。

### 3.9 MAX_TURN 覆盖

通过 `EXPLORE_MAX_TURN` 覆盖 baseline 默认的 `max_iteration=10`。脚本生成 per-run yaml 覆盖配置。

### 3.10 SPEAR 迁移边界与 `spear_lite`

参考本地 `/mnt/shared-storage-user/puyuan/code/SPEAR`：

| SPEAR 组件 | 源码位置 | terminal-rl 当前处理 |
|------------|----------|----------------------|
| Tool-call intrinsic reward + cosine curriculum | `verl-agent/agent_system/reward_manager/episode.py` 的 `use_toolcall_reward` | 已低侵入适配为 `EXPLORE_INTRINSIC_SCHEDULE=cosine` |
| Dr.BoT clip-higher / no KL / dynamic filtering | `examples/*_drbot.sh`、`ppo_trainer.yaml` | 大部分已由 `ALGO=dapo` 分支覆盖；可通过 DAPO knobs 消融 |
| Self-imitation replay buffer | `enable_trajectory_replay=True`、`TrajectoryReplayBuffer` | 暂不接入；需要改训练 batch 构造和 off-policy loss |
| P50 advantage recalibration | `weight_decay_trajectory_replay=-1`、`baseline_buffer_size` | 暂不接入；依赖 replay buffer |
| Clip-cov loss | `core_algos.py::compute_policy_loss_clip_cov` | 暂不接入；需要改 slime actor loss/token mask |

`EXPLORATION_PROFILE=spear_lite` 是当前最小兼容版本：

```bash
ALGO=dapo
EXPLORE_INTRINSIC=1
EXPLORE_INTRINSIC_COEF=0.03
EXPLORE_INTRINSIC_SCHEDULE=cosine
EXPLORE_INTRINSIC_DECAY_STEPS=200
EXPLORE_INTRINSIC_GRANULARITY=signature
EXPLORE_INTRINSIC_SCOPE=episode
```

它只改变 reward shaping，不引入 replay buffer，不改变默认 GRPO，不要求新依赖。适用场景是 `seta` 或 mixed 中 agentic 工具任务占比较高的训练；纯 safety 文本任务通常没有 command-level intrinsic signal。

### 3.11 Post-normalization exploration bonus

GRPO 的默认 reward post-process 会在同一 prompt group 内做：

$$A_i \propto R_i-\text{mean}_{j\in group}(R_j)$$

因此，如果同一 prompt 的多个 rollout 都拿到近似相同的 intrinsic bonus，探索项会被 group mean 抵消。`EXPLORE_ADVANTAGE_BONUS=1` 提供一个可选补救：先保持 slime 默认 GRPO 归一化，再把选定探索组件加回 normalized reward：

$$A_i^\prime=A_i+\lambda\cdot \text{clip}\left(\sum_k b_{i,k}, -c, c\right)$$

默认组件是 `explore_intrinsic_scaled`，对应第七节 `Bug 6` 的低侵入修复。该选项会启用 `reward_postprocess.post_process_rewards`，只影响训练用 reward/advantage，不改变轨迹原始 `score`。建议先小范围消融：

```bash
EXPLORE_ADVANTAGE_BONUS=1 \
EXPLORE_ADVANTAGE_BONUS_COMPONENTS=explore_intrinsic_scaled \
EXPLORE_ADVANTAGE_BONUS_CLIP=0.25
```

---

## 四、具体实现介绍

### 4.1 `generate.py` 关键代码段

**模块级常量**（约 line 42–80）：

```python
# Count-based
_EXPLORE_INTRINSIC_ENABLED = os.getenv("EXPLORE_INTRINSIC_ENABLED", "0") == "1"
_EXPLORE_INTRINSIC_COEF = float(os.getenv("EXPLORE_INTRINSIC_COEF", "0.1"))
_EXPLORE_INTRINSIC_SCHEDULE = os.getenv("EXPLORE_INTRINSIC_SCHEDULE", "constant")
_EXPLORE_INTRINSIC_DECAY_STEPS = int(os.getenv("EXPLORE_INTRINSIC_DECAY_STEPS", "0"))
_EXPLORE_INTRINSIC_GRANULARITY = os.getenv("EXPLORE_INTRINSIC_GRANULARITY", "raw")
_CMD_COUNTER: Dict[str, int] = {}

# LP-RND
_EXPLORE_LPRND_ENABLED = os.getenv("EXPLORE_LPRND_ENABLED", "0") == "1"
_EXPLORE_LPRND_COEF = float(os.getenv("EXPLORE_LPRND_COEF", "0.05"))
_EXPLORE_LPRND_SCHEDULE = os.getenv("EXPLORE_LPRND_SCHEDULE", "constant")
_EXPLORE_LPRND_DECAY_STEPS = int(os.getenv("EXPLORE_LPRND_DECAY_STEPS", "0"))
_EXPLORE_LPRND_CLIP = float(os.getenv("EXPLORE_LPRND_CLIP", "3.0"))
_LPRND_STATS = {"n": 0, "mean": 0.0, "m2": 0.0}

# CDE actor
_EXPLORE_CDE_ACTOR_ENABLED = os.getenv("EXPLORE_CDE_ACTOR_ENABLED", "0") == "1"
_EXPLORE_CDE_ACTOR_OMEGA = float(os.getenv("EXPLORE_CDE_ACTOR_OMEGA", "0.05"))
_EXPLORE_CDE_ACTOR_KAPPA = float(os.getenv("EXPLORE_CDE_ACTOR_KAPPA", "2.0"))
_EXPLORE_CDE_ACTOR_ALPHA = float(os.getenv("EXPLORE_CDE_ACTOR_ALPHA", "0.1"))

# Retry (signal-only)
_EXPLORE_RETRY_ATTEMPTS = int(os.getenv("EXPLORE_RETRY_ATTEMPTS", "1"))
_EXPLORE_RETRY_TRAJ_GAMMA = float(os.getenv("EXPLORE_RETRY_TRAJ_GAMMA", "1.0"))

# Safety pre-filter
_EXPLORE_SAFETY_FILTER_ENABLED = os.getenv("EXPLORE_SAFETY_FILTER_ENABLED", "0") == "1"
_EXPLORE_SAFETY_FILTER_COEF = float(os.getenv("EXPLORE_SAFETY_FILTER_COEF", "-0.5"))
_DANGER_RE = re.compile(...)
```

**辅助函数**：

| 函数 | 行号 | 作用 |
|------|------|------|
| `_cmd_signature(cmd)` | ~83 | shlex 切分，取前 3 个 token（cmd + 2 args），拼成签名键 |
| `_explore_intrinsic_bonus(turn_records)` | ~101 | 遍历 turn，按 `EXPLORE_INTRINSIC_GRANULARITY` 选择哈希源，更新 `_CMD_COUNTER`，累加 1/√N |
| `_explore_schedule_multiplier(schedule, train_step, decay_steps)` | ~232 | SPEAR-style curriculum multiplier：constant/cosine/linear |
| `_explore_safety_penalty(turn_records)` | ~125 | 对每条 cmd 做正则匹配，命中则累加 `EXPLORE_SAFETY_FILTER_COEF` |
| `_explore_lprnd_bonus(interactions)` | ~137 | 提取 `output_token_logprobs`，算 mean negative logprob，Welford 归一化 + clip |
| `_explore_cde_actor_metrics(interactions, base_score_magnitude, train_step)` | ~214 | 计算 actor log-PPL、reward cap 和最终 CDE bonus |

**调用点**（约 line 1078–1088，在 `_build_samples` 之后）：

```python
if _EXPLORE_INTRINSIC_ENABLED or _EXPLORE_SAFETY_FILTER_ENABLED or _EXPLORE_LPRND_ENABLED or _EXPLORE_CDE_ACTOR_ENABLED:
    _intr_bonus  = _explore_intrinsic_bonus(turn_records)
    _intr_multiplier = _explore_schedule_multiplier(...)
    _intr_effective_coef = _EXPLORE_INTRINSIC_COEF * _intr_multiplier
    _safe_penalty = _explore_safety_penalty(turn_records)
    _lprnd_raw = _explore_lprnd_bonus(interactions)
    _lprnd_effective_coef = _EXPLORE_LPRND_COEF * _explore_schedule_multiplier(...)
    _lprnd_bonus = _lprnd_raw * _lprnd_effective_coef
    _cde_actor = _explore_cde_actor_metrics(interactions, base_score_magnitude, run_ctx.train_step)
    for s in samples:
        if isinstance(s.reward, dict) and "score" in s.reward:
            s.reward["score"] += (_intr_bonus * _intr_effective_coef
                                  + _safe_penalty + _lprnd_bonus
                                  + _cde_actor["bonus"])
            s.reward["explore_intrinsic"]      = _intr_bonus
            s.reward["explore_safety_penalty"] = _safe_penalty
            s.reward["explore_lprnd"]          = _lprnd_bonus
            s.reward["explore_cde_actor_bonus"] = _cde_actor["bonus"]
```

### 4.2 `terminal-rl_qwen3-8b_exploration_pu.sh` 关键段

纯 wrapper 设计：读取与主脚本一致的 `DATASET` / `ALGO` / reward mode 变量和所有 `EXPLORE_*` 环境变量 → 转换为内部 env vars → 拼装 `RUN_ID` 后缀 → `exec bash terminal-rl_qwen3-8b_pu.sh`。

关键路由：

| 用户输入 | wrapper 行为 |
|----------|-------------|
| `DATASET=safety ALGO=dapo` | 与主脚本同名同义，直接透传 |
| `EXPLORATION_PROFILE=robust_dapo_lite` | 自动设置 `ALGO=dapo`、entropy、episode-local signature intrinsic、safety filter |
| `EXPLORATION_PROFILE=spear_lite` | 自动设置 `ALGO=dapo`、episode-local signature intrinsic、cosine curriculum decay |
| `EXPLORE_ENTROPY_COEF=0.01` | `EXTRA_ALGO_ARGS+="--entropy-coef 0.01"`，GRPO / DAPO 均生效 |
| `EXPLORE_THINK_MODE=1` | `CUSTOM_CONFIG_PATH=configs/rollout_qwen3_think.yaml` |
| `EXPLORE_TEMP_HIGH=1.2` | `ROLLOUT_TEMPERATURE=1.2` |
| `EXPLORE_INTRINSIC=1` | `EXPLORE_INTRINSIC_ENABLED=1` |
| `EXPLORE_INTRINSIC_SCHEDULE=cosine` | 用 SPEAR-style schedule 调节 intrinsic effective coef |
| `EXPLORE_LPRND=1` | `EXPLORE_LPRND_ENABLED=1` |
| `EXPLORE_CDE_ACTOR=1` | `EXPLORE_CDE_ACTOR_ENABLED=1`，启用 actor/PPL curiosity bonus |
| `EXPLORE_SAFETY_FILTER=1` | `EXPLORE_SAFETY_FILTER_ENABLED=1` |
| `EXPLORE_MAX_TURN=15` | `MAX_TURN=15`（覆盖 baseline 默认 10） |
| `EXPLORE_RETRY_ATTEMPTS=3` | 透传 env + 打印 WARN（待 agent_runner 实现） |

### 4.3 `terminal-rl_qwen3-8b_pu.sh` 微改

仅 2 处改动让 wrapper 能透传环境变量：

- L409：`--rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"` （原硬编码 `1`）
- L700：在 `${GRPO_ARGS[@]}` 之后追加 `${EXTRA_GRPO_ARGS:-} \`

两处默认行为不变。

---

## 五、环境变量速查表

| 变量 | 默认值 | 类型 | 作用 |
|------|--------|------|------|
| `DATASET` | `seta` | str | 与主脚本一致：`seta` / `safety` / `agentharm` / `mixed` |
| `ALGO` | `grpo` | str | 与主脚本一致：`grpo` / `dapo` |
| `EXPLORATION_PROFILE` | `off` | str | 探索预设；当前支持 `off` / `robust_dapo_lite` / `spear_lite` |
| `EXPLORE_ENTROPY_COEF` | `0.0` | float | AEPO 熵 bonus 系数；非 0 时透传为 `--entropy-coef X` |
| `EXPLORE_THINK_MODE` | `0` | bool | Qwen3 CoT think mode（切换 rollout yaml） |
| `EXPLORE_TEMP_HIGH` | *（空）* | float | rollout 温度覆盖（空=继承 baseline 1.0） |
| `EXPLORE_INTRINSIC` | `0` | bool | Count-based 内在奖励总开关 |
| `EXPLORE_INTRINSIC_COEF` | `0.1` | float | 内在奖励权重 |
| `EXPLORE_INTRINSIC_SCHEDULE` | `constant` | str | intrinsic coefficient schedule：`constant` / `cosine` / `linear` |
| `EXPLORE_INTRINSIC_DECAY_STEPS` | `0` | int | intrinsic schedule 长度；0 表示不退火 |
| `EXPLORE_INTRINSIC_GRANULARITY` | `raw` | str | `raw` / `signature` 二选一 |
| `EXPLORE_INTRINSIC_SCOPE` | `process` | str | `process` 保留历史跨 rollout 计数；`episode` 每条轨迹内计数，适合多 Ray worker |
| `EXPLORE_LPRND` | `0` | bool | LP-RND 生命周期新颖性开关 |
| `EXPLORE_LPRND_COEF` | `0.05` | float | LP-RND 权重 |
| `EXPLORE_LPRND_SCHEDULE` | `constant` | str | LP-RND coefficient schedule：`constant` / `cosine` / `linear` |
| `EXPLORE_LPRND_DECAY_STEPS` | `0` | int | LP-RND schedule 长度；0 表示不退火 |
| `EXPLORE_LPRND_CLIP` | `3.0` | float | z-score 裁剪上限 |
| `EXPLORE_LPRND_WARMUP` | `32` | int | LP-RND warmup 轨迹数；warmup 期间不更新归一化统计 |
| `EXPLORE_ADVANTAGE_BONUS` | `0` | bool | 将探索组件加到 GRPO normalization 之后，默认关闭 |
| `EXPLORE_ADVANTAGE_BONUS_COMPONENTS` | `explore_intrinsic_scaled` | str | post-normalization bonus 的 reward key 列表，逗号分隔 |
| `EXPLORE_ADVANTAGE_BONUS_COEF` | `1.0` | float | post-normalization bonus 系数 |
| `EXPLORE_ADVANTAGE_BONUS_CLIP` | `0.25` | float | post-normalization bonus 裁剪上限；0 表示不裁剪 |
| `EXPLORE_CDE_ACTOR` | `0` | bool | CDE actor/PPL bonus 开关 |
| `EXPLORE_CDE_ACTOR_OMEGA` | `0.05` | float | CDE actor bonus 权重 $\omega$ |
| `EXPLORE_CDE_ACTOR_KAPPA` | `2.0` | float | curiosity bonus 上限 divisor $\kappa$ |
| `EXPLORE_CDE_ACTOR_ALPHA` | `0.1` | float | log-PPL 缩放因子 $\alpha$ |
| `EXPLORE_CDE_ACTOR_REWARD_GATE` | `nonzero` | str | `nonzero` / `positive` / `none`；安全 mixed 推荐 `positive` |
| `EXPLORE_CDE_ACTOR_DECAY_STEPS` | `0` | int | $\omega$ 线性退火步数；0 表示不退火 |
| `EXPLORE_SAFETY_FILTER` | `0` | bool | 危险命令正则惩罚开关 |
| `EXPLORE_SAFETY_FILTER_COEF` | `-0.5` | float | 危险命令惩罚值（负数） |
| `EXPLORE_RETRY_ATTEMPTS` | `1` | int | 失败轨迹重试次数；当前仅透传，实际重启逻辑待实现 |
| `EXPLORE_RETRY_TRAJ_GAMMA` | `1.0` | float | 跨 attempt 奖励折扣（LaMer 用 0.6） |
| `EXPLORE_MAX_TURN` | *（空）* | int | 覆盖 `max_iteration`（baseline 默认 10） |

---

### 5.1 当前最可能有效的组合

**agentic-heavy / seta 优先**：

```bash
DATASET=seta \
ALGO=dapo \
EXPLORATION_PROFILE=spear_lite \
EXPLORE_ADVANTAGE_BONUS=1 \
EXPLORE_ADVANTAGE_BONUS_COMPONENTS=explore_intrinsic_scaled \
EXPLORE_ADVANTAGE_BONUS_CLIP=0.15 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

理由：DAPO 的 clip-higher / token loss / dynamic sampling 提供稳定 on-policy 更新；`spear_lite` 用 signature + episode scope + cosine intrinsic 促进早期工具探索；post-normalization bonus 小幅保留工具探索信号，避免被 GRPO group mean 完全抵消。

**三数据源 mixed，兼顾 agentic 与 safety**：

```bash
DATASET=mixed \
ALGO=dapo \
EXPLORATION_PROFILE=spear_lite \
EXPLORE_SAFETY_FILTER=1 \
EXPLORE_ADVANTAGE_BONUS=1 \
EXPLORE_ADVANTAGE_BONUS_COMPONENTS=explore_intrinsic_scaled \
EXPLORE_ADVANTAGE_BONUS_CLIP=0.10 \
EXPLORE_CDE_ACTOR=1 \
EXPLORE_CDE_ACTOR_OMEGA=0.02 \
EXPLORE_CDE_ACTOR_ALPHA=0.05 \
EXPLORE_CDE_ACTOR_REWARD_GATE=positive \
EXPLORE_CDE_ACTOR_DECAY_STEPS=200 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

理由：mixed 中 command intrinsic 主要帮助 seta；safety filter 防止探索奖励鼓励危险工具；CDE actor 只给正 reward 轨迹加很小 PPL bonus，用于鼓励“低概率但正确/安全”的输出，不软化负 reward 的 unsafe 行为。

上线观察优先级：`terminal/dataset/*/explore/mood_ratio/*`、`explore_reward_hacking_risk_rate`、`explore_safety_tension_rate`、`explore_bonus_to_base_abs_ratio/mean`。如果 `curious_unproven` 或 `reward_hacking_risk` 持续升高，先降低 `EXPLORE_ADVANTAGE_BONUS_CLIP` 和 `EXPLORE_CDE_ACTOR_OMEGA`。

---

## 六、实验测试命令

以下命令均假定已 `cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL` 且 `WORKER_URLS` 已 export。

### 6.1 兼容性验证（baseline 对照）

所有 `EXPLORE_*` 不设，应与直接跑 baseline 完全等价：

```bash
WORKER_URLS=http://cpu-worker:18081 DEBUG_MODE=1 \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# 对照
WORKER_URLS=http://cpu-worker:18081 DEBUG_MODE=1 \
  bash terminal-rl/terminal-rl_qwen3-8b_pu.sh
# 期望：runs/<id>/config/run_config.json 内容除时间戳外完全一致
```

### 6.2 单技术消融

```bash
# (a) 仅 entropy bonus
EXPLORE_ENTROPY_COEF=0.01 WORKER_URLS=... \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (b) 仅 think mode
EXPLORE_THINK_MODE=1 WORKER_URLS=... \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (c) 仅 count-based intrinsic (raw 粒度)
EXPLORE_INTRINSIC=1 EXPLORE_INTRINSIC_COEF=0.1 \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (d) 仅 count-based intrinsic (signature 粒度, Agent57 风格)
EXPLORE_INTRINSIC=1 EXPLORE_INTRINSIC_COEF=0.1 \
  EXPLORE_INTRINSIC_GRANULARITY=signature \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (e) 仅 SPEAR-style intrinsic curriculum（低侵入版本）
ALGO=dapo EXPLORE_INTRINSIC=1 EXPLORE_INTRINSIC_COEF=0.03 \
  EXPLORE_INTRINSIC_GRANULARITY=signature \
  EXPLORE_INTRINSIC_SCOPE=episode \
  EXPLORE_INTRINSIC_SCHEDULE=cosine \
  EXPLORE_INTRINSIC_DECAY_STEPS=200 \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (f) 仅 LP-RND
EXPLORE_LPRND=1 EXPLORE_LPRND_COEF=0.05 \
  EXPLORE_LPRND_SCHEDULE=cosine \
  EXPLORE_LPRND_DECAY_STEPS=200 \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (f2) post-normalization intrinsic bonus（绕过 GRPO group mean 抵消）
ALGO=dapo EXPLORE_INTRINSIC=1 EXPLORE_INTRINSIC_COEF=0.03 \
  EXPLORE_INTRINSIC_GRANULARITY=signature \
  EXPLORE_INTRINSIC_SCOPE=episode \
  EXPLORE_ADVANTAGE_BONUS=1 \
  EXPLORE_ADVANTAGE_BONUS_COMPONENTS=explore_intrinsic_scaled \
  EXPLORE_ADVANTAGE_BONUS_CLIP=0.25 \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (g) 仅 CDE actor/PPL bonus（推荐先搭配 DAPO 小权重消融）
ALGO=dapo EXPLORE_CDE_ACTOR=1 \
  EXPLORE_CDE_ACTOR_OMEGA=0.05 \
  EXPLORE_CDE_ACTOR_ALPHA=0.1 \
  EXPLORE_CDE_ACTOR_KAPPA=2.0 \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# (h) 仅 safety filter
EXPLORE_SAFETY_FILTER=1 EXPLORE_SAFETY_FILTER_COEF=-0.5 \
  WORKER_URLS=... bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

### 6.3 推荐配置（首选：DAPO + 低风险探索）

首选版本是 `robust_dapo_lite`：不引入额外模型、不依赖跨进程状态、不改变主训练入口的默认 GRPO 行为。

```bash
EXPLORATION_PROFILE=robust_dapo_lite \
DATASET=mixed \
MIX_SETA_RATIO=1 \
MIX_SAFETY_RATIO=1 \
MIX_AGENTHARM_RATIO=1 \
WORKER_URLS=http://cpu-worker:18081 \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

该 profile 等价于：

```bash
ALGO=dapo \
EXPLORE_ENTROPY_COEF=0.01 \
EXPLORE_INTRINSIC=1 \
EXPLORE_INTRINSIC_COEF=0.03 \
EXPLORE_INTRINSIC_GRANULARITY=signature \
EXPLORE_INTRINSIC_SCOPE=episode \
EXPLORE_SAFETY_FILTER=1
```

SPEAR 低侵入 profile：

```bash
EXPLORATION_PROFILE=spear_lite \
DATASET=mixed \
MIX_SETA_RATIO=1 \
MIX_SAFETY_RATIO=1 \
MIX_AGENTHARM_RATIO=1 \
WORKER_URLS=http://cpu-worker:18081 \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

该 profile 等价于：

```bash
ALGO=dapo \
EXPLORE_INTRINSIC=1 \
EXPLORE_INTRINSIC_COEF=0.03 \
EXPLORE_INTRINSIC_SCHEDULE=cosine \
EXPLORE_INTRINSIC_DECAY_STEPS=200 \
EXPLORE_INTRINSIC_GRANULARITY=signature \
EXPLORE_INTRINSIC_SCOPE=episode
```

### 6.4 可选全栈配置

```bash
EXPLORE_ENTROPY_COEF=0.01 \
EXPLORE_THINK_MODE=1 \
EXPLORE_INTRINSIC=1 \
EXPLORE_INTRINSIC_GRANULARITY=signature \
EXPLORE_INTRINSIC_SCHEDULE=cosine \
EXPLORE_INTRINSIC_DECAY_STEPS=200 \
EXPLORE_LPRND=1 \
EXPLORE_LPRND_COEF=0.05 \
EXPLORE_LPRND_SCHEDULE=linear \
EXPLORE_LPRND_DECAY_STEPS=400 \
EXPLORE_CDE_ACTOR=1 \
EXPLORE_CDE_ACTOR_OMEGA=0.03 \
EXPLORE_MAX_TURN=15 \
WORKER_URLS=http://cpu-worker:18081 \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
# 期望 RUN_ID 后缀：_explore_ent0.01_think_int_cosine200_lprnd_linear400_cdeact0.03_<ts>
```

### 6.5 离线单元测试（无需 GPU/CPU worker）

```bash
# 语法检查
python3 -m py_compile terminal-rl/generate.py terminal-rl/rollout_log.py && echo "py OK"
bash -n terminal-rl/terminal-rl_qwen3-8b_pu.sh && echo "main sh OK"
bash -n terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh && echo "explore sh OK"

# DRY_RUN：检查 profile 展开、RUN_ID 后缀、Ray runtime env 透传，不启动训练
DRY_RUN=1 DEBUG_MODE=1 EXPLORATION_PROFILE=spear_lite DATASET=seta \
  WORKER_URLS=http://127.0.0.1:18081 \
  RUN_TIMESTAMP=2026-05-29_spear_lite_test \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

DRY_RUN=1 DEBUG_MODE=1 ALGO=dapo DATASET=safety EXPLORE_CDE_ACTOR=1 \
  RUN_TIMESTAMP=2026-05-29_cde_test \
  bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh

# 内在奖励逻辑最小烟雾测试（不依赖 generate.py 全部依赖）
python3 - <<'PY'
import hashlib, math, shlex
ctr = {}
def bonus(cmds):
    t = 0.0
    for c in cmds:
        sig = "|".join(shlex.split(c)[:3]) if c.strip() else "__empty__"
        k = hashlib.md5(sig.encode()).hexdigest()[:10]
        ctr[k] = ctr.get(k, 0) + 1
        t += 1.0 / math.sqrt(ctr[k])
    return t

print("first  ls -la /tmp:",  bonus(["ls -la /tmp"]))   # ~1.0
print("second ls -al /tmp/:", bonus(["ls -al /tmp/"]))  # ~0.707 (signature 命中同桶)
print("third  ls -la /etc:",  bonus(["ls -la /etc"]))   # ~1.0 (不同桶)
PY
```

### 6.6 验收检查清单

训练启动后请确认：

- [ ] `runs/<run_id>/config/run_config.json` 字段符合预期。
- [ ] `runs/<run_id>/logs/train.log` 出现 `[explore] xxx ON` 日志行。
- [ ] 使用 `spear_lite` 时，日志中 `INTRINSIC` 应显示 `schedule=cosine/200`，run id 应包含 `_spear_lite_int_cosine200`。
- [ ] 一旦有 rollout 完成，`runs/<run_id>/trajectories/t*_r*_st*_g*_s*_*/meta.json` 的 `reward` 字段含 `explore_intrinsic` / `explore_intrinsic_scaled` / `explore_intrinsic_effective_coef` / `explore_intrinsic_schedule_multiplier` / `explore_lprnd_effective_coef` / `explore_total_bonus`。
- [ ] `meta.json` / `traj.json` 应包含 `exploration` 小节，至少能看到 `explore_mood`、`explore_bonus_to_base_abs_ratio`、`explore_action_count`、`explore_reward_hacking_risk`。
- [ ] 使用 `EXPLORE_ADVANTAGE_BONUS=1` 时，dry-run command 应包含 `--custom-reward-post-process-path reward_postprocess.post_process_rewards`。
- [ ] 若 `EXPLORE_LPRND=1`，**前 32 条** rollout 的 `explore_lprnd` 应恒为 0（warmup），第 33 条起才有正值。
- [ ] `wandb`（如启用）能看到 `*/reward_component/explore_intrinsic*`、`*/reward_component/explore_total_bonus*`、`terminal/dataset/*/explore/mood_ratio/*` 等指标。

---

## 七、已修复问题与仍待改进项（对照 MERCI / SPEAR 源码审查）

参考 `/mnt/shared-storage-user/puyuan/code/MERCI/` 与 `/mnt/shared-storage-user/puyuan/code/SPEAR/` 的实现，本轮已处理 P0/P1 中最影响可用性的部分；P2 项保留为下一轮消融和重构方向。

### Fix 1 [P0]：`_CMD_COUNTER` / `_LPRND_STATS` 进程隔离

**原问题**：两者均为模块级 Python dict，slime 异步 rollout（多个 sglang worker / Ray actor）下，每个 worker 进程独立维护一份计数器和 Welford 统计。结果：
- 同一命令在 8 个 worker 上独立累积 N，1/√N 奖励不一致。
- LP-RND 的 μ/σ 在各 worker 上漂移方向不同，归一化基准不同步。

**对照 MERCI**：把 pseudo-count 估计放进全局神经网络 **Coin Flipping Network**，并通过 RL trainer 的 `exploration_model` 子模块跨 worker 同步参数（见 `MERCI/recipe/dapo/example/run_qwen2.5_math_dapo_cfn.sh` 的 `exploration_model.model.pretrain_path`），从根本上避开进程隔离问题。

**当前修复**：新增 `EXPLORE_INTRINSIC_SCOPE=episode`，每条轨迹内重置计数，消除跨 Ray worker 状态不一致。`robust_dapo_lite` 默认使用该模式。历史 `process` 模式仍保留用于消融。

**后续中长期**：用 Ray Actor / Redis 维护全局 `CounterServer`；或参考 MERCI，把计数器换成轻量级 CFN 并跨 worker 同步权重。

### Fix 2 [P0]：LP-RND warmup 期间仍更新统计

**位置**：`_explore_lprnd_bonus`（generate.py 约 L137）。

```python
# Welford 更新（先做）
s["n"] += 1
delta = surprise - s["mean"]
s["mean"] += delta / s["n"]
s["m2"] += delta * (surprise - s["mean"])
if s["n"] < 32:
    return 0.0  # warmup，但统计已被更新
```

**原问题**：warmup 期前 32 条轨迹（训练初期 entropy 最高、最具探索价值）的 surprise 已纳入 μ/σ；之后归一化时它们作为基线，会让后续轨迹的 z-score 显著偏低，削弱奖励信号。

**当前修复**：新增 `EXPLORE_LPRND_WARMUP`，warmup 期间只计数、不更新 Welford 统计；warmup 后才开始建立 μ/σ。

### Fix 3 [P1]：`_cmd_signature` 对空命令未保护

**位置**：generate.py 约 L83。

```python
parts = shlex.split(cmd)[:3]
return "|".join(parts)
```

- `shlex.split("")` → `[]` → `"|".join([]) == ""` → 所有空命令共享同一桶。
- `shlex.split` 在含未配对引号的命令上抛 `ValueError`（已有 `except Exception: return cmd[:80]` 保护，OK）。

注：上游 `_explore_intrinsic_bonus` 已有 `if not cmd: continue` 保护，**当前不会触发**；但 `_cmd_signature` 作为独立 helper 一旦被复用就脆弱。

**当前修复**：空命令返回 `__empty__`；同时规范化短 flag 和路径尾斜杠，使 `ls -la /tmp` 与 `ls -al /tmp/` 落入同一 signature bucket。

### Fix 4 [P1]：safety 正则未覆盖反引号 / `$()` 子 shell 注入

现有正则只匹配字面 `rm -rf /`、`curl|bash` 等。LLM 若学会通过 `eval $(echo "rm -rf /")` 或 `\$(printf 'rm -rf /')` 间接执行，可绕过。当前威胁模型下风险较低（LLM 主动越狱），但属于深度防御缺口。

**当前修复**：正则补充了 `eval ...`、反引号和 `$()` 中包含危险片段的轻量检测。后续仍可在 P2 引入 `bashlex` AST 解析，或把命令文本交给 ClawSentry 做语义判断。

### Fix 5 [P1]：补充 SPEAR 式 intrinsic reward curriculum

**对照 SPEAR**：SPEAR 在内在奖励上设计了 **curriculum decay**：早期权重大（鼓励探索），后期权重渐弱（鼓励 exploitation 已发现的成功轨迹）。历史实现中 `EXPLORE_LPRND_COEF` / `EXPLORE_INTRINSIC_COEF` 都是静态常数。

**当前修复**：新增 `EXPLORE_INTRINSIC_SCHEDULE=constant|cosine|linear` 与 `EXPLORE_INTRINSIC_DECAY_STEPS`。默认 `constant/0` 保持历史行为；`EXPLORATION_PROFILE=spear_lite` 使用 `cosine/200`，对应 SPEAR 的 `use_toolcall_reward="cosine"` / `max_toolcall_steps=200` 思路。

**追加修复**：`EXPLORE_LPRND_COEF` 也支持 `EXPLORE_LPRND_SCHEDULE=constant|cosine|linear` 与 `EXPLORE_LPRND_DECAY_STEPS`。默认 `constant/0`，因此历史 LP-RND 行为不变。

### Fix 6 [P2]：`_explore_intrinsic_bonus` 被 GRPO group mean 抵消

**问题**：GRPO 用 group-internal mean baseline 计算 advantage；当前 intrinsic bonus 直接叠加在 raw reward 上，**会被 baseline 减掉**。同一 prompt 的 8 个 rollout 通常用相似命令，其 intrinsic bonus 高度相关，减完之后剩下的 explore signal 较弱。

**当前修复**：新增 `EXPLORE_ADVANTAGE_BONUS=1`，通过 slime 的 `--custom-reward-post-process-path reward_postprocess.post_process_rewards` 在默认 GRPO normalization 后把选定探索组件加回 normalized reward。默认组件为 `explore_intrinsic_scaled`，并用 `EXPLORE_ADVANTAGE_BONUS_CLIP=0.25` 控制量级。开启 custom reward post-process 时，slime 的 constant-group drop 会跳过 pre-process raw reward 过滤，DAPO dynamic sampling filter 也会把 post-normalization bonus 纳入方差判断。

**默认行为**：关闭。原因是它明确改变 advantage 估计方式，适合做消融，不应默认影响 baseline / `spear_lite`。

### Fix 7 [P1]：tool call 未进入 intrinsic / safety filter

**问题**：早期实现只读取 `turn_record["command"]`，但当前轨迹实际把动作存为 `turn_record["tool_calls"][*]`，例如：

```json
{"tool_name": "add_into_access_list", "args": {"name": "White Smith"}}
```

因此 `EXPLORE_INTRINSIC` 和 `EXPLORE_SAFETY_FILTER` 在很多真实 rollout 上会变成近似 no-op。

**当前修复**：新增统一 action extractor：兼容旧 `command` 字段、shell 风格 `args.command/cmd/script/code/query`，以及 Agent-SafetyBench / agentharm 的结构化 tool name + args。`raw` 粒度使用完整 action，`signature` 粒度使用 tool name + command signature 或 tool name + args 摘要。

### Fix 8 [P1]：补充结构化 exploration mood 监控

新增轨迹和 wandb 字段：

- `explore_mood`：`confident_exploit` / `curious_success` / `curious_unproven` / `cautious` / `risky` / `stuck` / `low_signal`
- `explore_bonus_to_base_abs_ratio`
- `explore_reward_hacking_risk`
- `explore_over_exploration_risk`
- `explore_safety_tension`
- `explore_action_count` / `explore_tool_call_count`

这些字段用于快速判断当前训练是在有效探索、过度探索、卡住，还是安全压力上升。

### Bug 9 [P2]：多次尝试反思（LaMer）未实际接入

`EXPLORE_RETRY_ATTEMPTS=3` 只透传环境变量，`agent_runner.py` 未读取，行为与 `=1` 等价。当前仅打印 WARN。

**修复建议**：见 §3.7 P1 计划；需在 `agent_runner` 主循环增加 "detect failure → inject reflection → reset env → replay" 三段逻辑。预估 ~200 行。

---

## 八、与上游工作的关系小结

| 模块 | 直接复刻 | 简化适配 | 创新 |
|------|---------|---------|------|
| Count-based bonus | MERCI 的整体公式 (1/√N) | 用确定性 hash 替代 CFN | `signature` 粒度 |
| LP-RND | — | RND 思想 | 复用 logprob 作 surprise，零参数 |
| SPEAR intrinsic curriculum | SPEAR `use_toolcall_reward=cosine` | 映射到 command intrinsic coefficient schedule | `spear_lite` profile |
| Safety filter | — | — | 与 ClawSentry 正交的命令级硬惩罚 |
| Multi-attempt | LaMer 概念 | 仅旗标，待实现 | — |
| Entropy bonus | AEPO | slime 原生 `--entropy-coef` 直接打开 | — |

本模块定位是 **轻量化、可消融、零侵入** 的探索工具箱，**不试图复刻 MERCI / SPEAR / LaMer 全部细节**。如需更激进的探索（CFN 跨 worker 同步、SPEAR 的 self-imitation replay buffer），需要更深入的 slime trainer 改造。
