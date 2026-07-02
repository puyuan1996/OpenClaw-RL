# Model-based AgenticRL 前沿调研报告

> 调研范围：2024-2026 年 AI 顶会（ICLR, NeurIPS, ICML, COLM）、arXiv 高引论文、HF Daily Papers、GitHub 高星项目
>
> 调研目标：梳理 model-based RL 思想在 AgenticRL 中的应用现状，识别样本效率方向的研究空白

---

## 摘要

### 一句话结论

当前 AgenticRL 的样本效率瓶颈主要来自 **真实环境 rollout 昂贵、episode-level 奖励稀疏、经验复用不足**。SPEAR 已证明“自模仿 + 课程学习”有效，但仍是 model-free；2024-2026 年最新工作分别从世界模型、过程奖励、环境合成和经验回放切入，尚未形成统一框架。因此，一个围绕 **世界模型想象训练 + TD/PRM 稠密奖励 + 新鲜度感知优先回放** 的统一方案具备明确研究空白。

### 推荐优先级


| 优先级 | 方向                   | 代表工作                           | 对本课题价值                  | 风险                   |
| --- | -------------------- | ------------------------------ | ----------------------- | -------------------- |
| P0  | SPEAR 源码与 SIL buffer | SPEAR                          | 直接基线，可扩展                | 需在远程集群复现             |
| P0  | 世界模型训练               | DynaWeb, Dyna-Mind, RWML       | 样本效率主贡献来源               | sim-to-real gap      |
| P0  | 稠密奖励/信用分配            | AgentPRM, CA Survey            | 缓解稀疏奖励，支持 step-level 学习 | PRM reward hacking   |
| P1  | 优先经验回放               | FreshPER, ReVal, ER for LLM RL | 降低 rollout 计算成本         | off-policy staleness |
| P1  | 环境合成                 | AWM, AutoForge, COVERT         | 后续扩展数据规模                | 首阶段实现成本高             |
| P2  | 树搜索/rollout 控制       | Tree-GRPO, GFCR, ITP           | 提升 rollout 预算利用率        | 与 WM 框架耦合复杂          |


### 强调

1. **不是简单套用 Dreamer/Dyna**：LLM Agent 的状态是文本/工具反馈，世界模型更适合预测“状态变化摘要”或 embedding 对齐，而不是像视觉 RL 那样预测像素。
2. **SPEAR 是最合适的起点**：它已有 SIL buffer、curriculum、tool-use intrinsic reward，刚好暴露出“无世界模型、无优先回放、无稠密过程奖励”的改进空间。
3. **论文创新应聚焦样本效率**：主实验不只看最终成功率，还要看达到同等成功率所需的真实环境交互次数、wall-clock、rollout token 成本。

### 证据等级说明

本文档用于讨论，分为三类证据：


| 等级  | 含义                                        | 使用方式                          |
| --- | ----------------------------------------- | ----------------------------- |
| A   | 已核验源码或论文正文                                | 可作为方案设计依据                     |
| B   | 来自 arXiv/OpenReview/HF paper page 摘要与检索结果 | 可用于调研判断，正式写论文前需补 BibTeX 与原文页码 |
| C   | 代码、会议状态或数值尚未二次核验                          | 文中标注“待确认/待核验”，不作为最终 claim     |


正式论文写作前，应将所有 B/C 类条目统一替换为 BibTeX、论文表格编号或官方代码链接。

---

## 关键补充：LLM hidden → JEPA latent space 的可行性与边界

### 结论

LLM 的 next-token hidden 不能直接等同于 world model 的统一 latent。更稳妥的设计是把 hidden 当作原始表征，经 `normalization / clipping / projector / alignment` 转换为“预测环境反馈所需的 belief latent”。也就是说，本方案不做裸 hidden MSE，而是学习：

```text
context tokens -> LLM/frozen encoder hidden -> controlled state projector -> z_s_t
action tokens  -> action pooling/projector                    -> z_a_t
next obs text  -> frozen encoder hidden -> target projector    -> z_o_t+1

predictor(z_s_t, z_a_t) -> z_hat_o_t+1
loss = JEPA latent prediction + SIGReg + action contrast + optional value/reward heads
```

这个结论支持 OpenClaw-RL 的 v1 实现路线：先做 replay-buffer/offline probe，验证 action-sensitivity 和 latent collapse 指标，再考虑 online auxiliary loss 或 U2 candidate ranking。

### 相关支撑

| 工作 | 证据等级 | 对本方案的支撑 | 链接 |
| --- | --- | --- | --- |
| LLM-JEPA | B | 直接探索把 JEPA 思路迁移到 LLM hidden/embedding 空间，说明 text-only latent prediction 可行，但需要专门 projector/训练目标，而不是裸用 token CE。 | [arXiv](https://arxiv.org/abs/2509.14252), [GitHub](https://github.com/galilai-group/llm-jepa) |
| Pearl | B | 使用 off-the-shelf autoregressive VLM hidden state 编码输入 view 与完整 tool-use trajectory view，再预测 trajectory embedding；支持“agent trajectory latent target”而非只预测 next token。 | [arXiv HTML](https://arxiv.org/html/2604.08065v1) |
| VL-JEPA | B | 选择预测 continuous text embeddings 而不是离散 token reconstruction，支持本方案将世界模型目标放在 latent space。 | [arXiv](https://arxiv.org/abs/2512.10942) |
| Transformers as implicit state estimators | B | 从 POMDP/state-estimation 角度解释 transformer hidden 可承载 belief-state 信息，但需要任务约束来变成可预测环境反馈的 latent。 | [arXiv HTML](https://arxiv.org/html/2410.16546v3) |
| Massive Activations | B | 指出 LLM hidden 有 anisotropy / outlier activation 风险；工程上必须做 clipping、standardization、LayerNorm/RMSNorm 和 projector。 | [arXiv HTML](https://arxiv.org/html/2402.17762v2) |
| ECHO | A/B | 强 baseline：terminal observation token CE auxiliary + GRPO。说明“预测环境反馈”有效，但也意味着本方案必须区别于 token-level next-observation CE。 | [arXiv](https://arxiv.org/abs/2605.24517), [GitHub](https://github.com/microsoft/echo-rl) |
| PaW | B | world-model auxiliary + action entropy / robust loss 进一步挤压简单 next-observation prediction 的 novelty。本方案应强调 queryable latent evaluator 与 candidate ranking。 | [arXiv HTML](https://arxiv.org/html/2606.02388v1) |
| COMAP | B | 支持用世界模型预估 candidate action 的后果，但其更偏 textual future simulation；本方案采用 latent evaluator 以降低文本表面形式偏差。 | [arXiv HTML](https://arxiv.org/html/2606.02372) |
| LeWM | A/B | 提供 SIGReg + JEPA latent prediction 的实现参考；但原域是 pixel/control，不可直接迁移到 text agent，需要替换 encoder/action adapter。 | [arXiv](https://arxiv.org/abs/2603.19312), [GitHub](https://github.com/lucas-maes/le-wm) |

### 对 OpenClaw-RL 方案的修订

1. **latent 定义**：`z_s` 是 policy/context belief latent，`z_a` 是 assistant/tool action latent，`z_o` 是 next observation / tool feedback latent。三者必须通过独立 projector 对齐到统一 latent space。
2. **target branch**：v1 默认使用 frozen encoder 或 cached hidden 得到 `z_o`，并 `stop_gradient`，避免在线训练时 target drift。
3. **action-sensitivity 是硬门槛**：必须报告真实 action 与 shuffled action 的 loss gap、`action_delta`、counterfactual margin；若 gap 不显著，world model 不能进入 online auxiliary。
4. **anti-collapse 不等于 anti-drift**：SIGReg 用于防 collapse，但 online 场景仍需 frozen anchor / EMA / re-anchor。v1 只做 offline/cached hidden。
5. **区别于 ECHO/PaW**：ECHO/PaW 更像 next-observation/token auxiliary baseline。本方案的定位是 action-conditioned latent evaluator，可后续服务 U2 candidate ranking 和 confidence-gated replay。
6. **工程风险**：Megatron PP/CP/SP 下在线抓 middle-layer hidden 风险高；v1 不接入 online hidden hook，只提供 default-off metadata、offline probe、可选 loss hook。

### 进入 online auxiliary 前的验收指标

| 指标 | 目的 | 最低要求 |
| --- | --- | --- |
| shuffled-action gap | 验证预测依赖 action，而不是只复述上下文 | real-action loss 明显低于 shuffled-action loss |
| `action_delta` | 测量替换 action 后 latent prediction 是否变化 | 非零且随 action 类型变化 |
| effective rank / SIGReg | 检查 latent collapse | rank 不能持续退化到极低维 |
| reward/value calibration | 检查 latent 是否携带 outcome 信息 | value head 与 verifier reward 有正相关 |
| ECHO-style token CE baseline | 防止 claim 被简单 CE baseline 覆盖 | 至少作为 ablation 或工程对照 |

---

## 一、AgenticRL 研究全景

### 1.1 问题定义

AgenticRL 指使用强化学习训练 LLM 使其具备在真实/模拟环境中多轮交互、调用工具、做出决策的能力。与 reasoning RL（如数学推理）的核心区别：


| 维度    | Reasoning RL        | Agentic RL               |
| ----- | ------------------- | ------------------------ |
| 环境转移  | 确定性（思维链）            | 随机性（真实环境反馈）              |
| 可观测性  | 完全可观测               | 部分可观测 (POMDP)            |
| 典型长度  | 1 轮，0.5K-30K tokens | 10-100+ 轮，100K-1M tokens |
| 动作类型  | 同质（生成 tokens）       | 异质（工具调用、规划、通信）           |
| 中间验证  | 常可验证（数学答案）          | 极少可验证                    |
| 关键决策点 | 中等频率                | 稀少但决定性                   |


### 1.2 主流技术路线图

```
AgenticRL 技术路线
├── 优化算法
│   ├── GRPO (DeepSeek, 2024) ─── 免值函数、组内相对优势
│   ├── GiGPO ─── step-level 优势估计
│   ├── GSPO (DynaWeb) ─── 序列级重要性采样
│   └── A2C / PPO ─── 经典 actor-critic
├── 探索-利用平衡
│   ├── SPEAR (ICLR 2026) ─── 自模仿 + 课程学习
│   ├── Tree-GRPO (ICLR 2026) ─── 树搜索 rollout
│   └── Dr.BoT ─── 工业级 RL tricks
├── 信用分配 (Credit Assignment)
│   ├── AgentPRM (WWW 2026) ─── TD+GAE step-level 值函数
│   ├── ArCHer (ICML 2024) ─── 离策略 turn-critic
│   └── SWEET-RL ─── 特权 critic
├── 世界模型 / Model-based
│   ├── DynaWeb (2025) ─── LLM 预测 web 页面转移
│   ├── Dyna-Mind (ICLR 2026) ─── 推理内模拟
│   ├── RWML (2026) ─── embedding 空间对齐
│   └── ITP (2025) ─── 自适应想象视野
├── 环境合成
│   ├── AWM (2026, Snowflake) ─── 代码驱动合成环境
│   ├── AutoForge (2025) ─── API 文档 → DAG → 任务
│   └── COVERT / ASTRA ─── 可控数据合成
└── 经验回放
    ├── FreshPER (2025) ─── 新鲜度感知优先级
    ├── ReVal (2026) ─── 值函数 + 回放缓冲
    └── ExGRPO ─── 按正确率和熵组织经验
```

---

## 二、核心论文深度分析

### 2.1 ✨SPEAR：自模仿 + 渐进探索 (ICLR 2026)

**来源**：Tencent Youtu Research | [GitHub](https://github.com/TencentYoutuResearch/SPEAR)

**核心思想**：通过课程调度（curriculum scheduling）协调探索和利用——前期用 intrinsic reward 鼓励工具使用探索，后期加强自模仿利用成功轨迹。

**关键技术组件**：


| 组件                    | 实现细节                                                                                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **SIL Replay Buffer** | `TrajectoryBufferBatch` 类，存储 advantage > 0 的完整轨迹（含 input_ids、old_log_probs、advantages 等），容量 2048，每 1.5x 容量触发回放后清空 |
| **轨迹选择**              | 仅存储 mean advantage > 0 的样本；回放时用历史 P50 百分位重新估计优势                                                                   |
| **新鲜度控制**             | `tolerate_steps=5`：删除与最新条目相差超过 5 个训练步的旧轨迹                                                                         |
| **Intrinsic Reward**  | `r_intrinsic = min(1, 0.1 × num_steps) × (cos(π × t/T) + 1) / 2`，前期满权重、T=200 步后衰减至 0                              |
| **SIL Loss 热身**       | `loss × replay_coef × (1 - cos(π × t/T_warmup)) / 2`，从 0 增长到 replay_coef，防止早期不稳定                                  |
| **On/Off-policy 更新**  | **串行而非混合**：先完成所有 on-policy PPO epochs，再执行 SIL replay epochs                                                       |
| **Clip-Cov Loss**     | 计算 (advantage - mean) × (log_prob - mean) 的协方差，裁剪高协方差 token 以控制熵                                                  |


**实验结果**：


| Benchmark    | GRPO  | + SPEAR | 提升     |
| ------------ | ----- | ------- | ------ |
| ALFWorld     | 72.8% | 88.9%   | +16.1% |
| WebShop      | 56.8% | 77.5%   | +20.7% |
| AIME25 (32B) | 54.0  | 60.1    | +6.1   |


**技术栈**：verl 框架 + Ray 分布式 + FSDP 分片 + vLLM 推理，base model Qwen2.5-1.5B/32B

**源码证据链（已本地核验）**：


| 结论                                                                   | 源码位置                                                                    | 备注                                                                                      |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `TrajectoryBufferBatch` 存储 batch、reward tensor、extra info 和 step     | `baselines/SPEAR/verl-agent/verl/trainer/ppo/ray_trainer_hybrid.py`     | agent 任务分支；math 分支在 `baselines/SPEAR/verl/verl/trainer/ppo/ray_trainer_hybrid.py` 有对应实现 |
| buffer 通过 `remove_old()` 删除超出 tolerate window 的旧轨迹                   | `baselines/SPEAR/verl-agent/verl/trainer/ppo/ray_trainer_hybrid.py`     | 类默认 `tolerate_steps=10` 且取 `min(tolerate_steps, 10)`；WebShop/SPEAR 脚本设为 5               |
| replay loss 使用 cosine warmup                                         | `baselines/SPEAR/verl-agent/verl/workers/actor/dp_actor.py`             | `loss *= replay_loss_coef * (1 - cos(pi * step / T)) / 2`                               |
| toolcall intrinsic reward 使用 cosine decay                            | `baselines/SPEAR/verl-agent/agent_system/reward_manager/episode.py`     | math 分支 reward 在 `baselines/SPEAR/verl/recipe/spear/reward.py`                          |
| `clip_cov` loss 注册为 policy loss                                      | `baselines/SPEAR/verl-agent/verl/trainer/ppo/core_algos.py`             | on-policy 与 replay batch 使用不同 clip-cov 参数                                               |
| 运行脚本中 SPEAR 使用 `use_toolcall_reward=cosine`、`max_toolcall_steps=200` | `baselines/SPEAR/verl-agent/examples/grpo_trainer/run_webshop_spear.sh` | math 32B 脚本也有对应配置                                                                       |


SPEAR 论文 §4.2 对 P50 的唯一论证是这一句（Eq.2 紧随其后）：

> *"As training progresses, **due to the high variance nature of agentic RL**, we utilize the 50-th percentile P50(DR)P50​(DR​) as a **conservative but robust** estimation of the policy baseline..."*

A~ti=Ri−P50(DR)A~ti​=Ri−P50​(DR​)

注意它的论证逻辑是：「因为方差大 → 所以挑个稳健统计量（中位数）」。这是典型的**启发式动机**（robust statistic 的工程直觉），不是从某个目标函数推出「中位数是最优基线」。全文没有针对中位数的最优性/无偏性/方差上界证明。

**局限性**：

- 纯 model-free，每次训练需真实环境交互
- SIL buffer 无优先级采样，仅正优势过滤 + FIFO
- Episode-level 稀疏奖励，无 step-level 稠密信号
- 无世界模型，不能 "想象" 替代方案

---

### 2.2 世界模型方向

#### 2.2.1 DynaWeb：Web Agent 的 Dyna 式 MBRL (2026)

**核心思想**：训练 LLM 作为 web 世界模型，预测页面状态转移，实现 Dyna 式混合真实/想象 rollout 训练。

**关键创新 —— Delta Prediction Decomposition**：

- 不预测完整的下一页面 accessibility tree（token 冗余严重）
- 分解为：(1) 预测自然语言状态变化 Δ(o_t, o_{t+1}) → (2) 将 Δ 应用到 o_t 得到 o_{t+1}

**世界模型架构**：

- 基座：GPT-oss-120B，SFT 微调
- 输入：当前 accessibility tree + 动作 + 系统指令
- 输出：推理 trace + 状态变化描述

**数据混合策略**：50% 真实专家轨迹 + 50% 想象 rollout，40% 真实数据最优

**实验结果**：


| Benchmark  | DynaWeb | 最佳 Baseline   | 提升        |
| ---------- | ------- | ------------- | --------- |
| WebArena   | 31.0%   | 26.7% (WebRL) | +16.1% 相对 |
| WebVoyager | 38.7%   | 32.6% (WebRL) | +18.7% 相对 |


- 训练过的 WM (31.0%) 远优于冻结 GPT-oss-120B (20.9%)，证明 WM 是**显式训练**的（"5.3x 时间 / 6.8x API 节省"这一对比数字在原文未核实到对应出处，引用前需逐条核对）
- 想象 rollout 最优长度 4-5 步，更长则 hallucination 累积

**局限**：WM 120B 参数太大；域特定（WebArena vs WebVoyager 需分别训练）；WM 在 RL rollout 阶段冻结（注意：WM 本身是显式训练的，只是 RL 阶段不再更新），无策略-WM 共演化

---

#### 2.2.2 Dyna-Mind：推理内模拟 (ICLR 2026, Microsoft)

**核心思想**：不训练独立世界模型，而是将环境模拟能力集成到 agent 的推理过程中——agent 自身就是世界模型。

**两阶段训练框架**：


| 阶段                 | 方法     | 核心机制                                          |
| ------------------ | ------ | --------------------------------------------- |
| Stage 1: ReSim     | SFT 蒸馏 | 从真实搜索树（DFS 扩展 + 值函数评估）提取结构化推理 trace，教模型直接生成模拟 |
| Stage 2: Dyna-GRPO | 在线 RL  | 交替模拟改进期（SimRollout：plan→执行→用真实状态修正模拟）和策略改进期   |


**SimRollout 核心流程**：

1. Agent 生成 action + 内部模拟的未来状态
2. 在真实环境中执行 action → 获得 ground truth 状态
3. 用真实状态 prompt agent 修正其模拟
4. 修正后的轨迹用 A_refine 优势进行优化

**Sim Score 指标**：量化模拟能力与任务性能的相关性，r=0.64-0.96 (Sokoban)，r=0.46-0.76 (ALFWorld)

**实验结果**：


| Benchmark    | Dyna-Mind | GRPO  | 提升   |
| ------------ | --------- | ----- | ---- |
| ALFWorld ID  | 92.5%     | 87.0% | +5.5 |
| ALFWorld OOD | 89.1%     | 87.1% | +2.0 |
| AndroidWorld | 40.7%     | 35.3% | +5.4 |


- ReSim 在 ALFWorld 上远超 DeepSeek-R1 (87.7% vs 62.5%)
- token 效率：ReSim 蒸馏相比 R1 蒸馏可大幅减少输出 token（**注：两处资料存在分歧——精读版报 Distill(ReSim) 约为 Distill(R1) 的 1/11；引用核验指出 Dyna-GRPO 终态 token 约为基模 1.9×。引用前需回原文核对口径**）

**局限**：SimRollout 仍需真实环境交互；ReSim 数据收集昂贵（需 rollout model + value function + 聚合 LLM）

---

#### 2.2.3 RWML：强化世界模型学习 (2026, Columbia/Microsoft)

**核心思想**：自监督 RL 训练 LLM 学习 action-conditioned 世界模型，在预训练 embedding 空间中对齐模拟与真实状态转移。

**关键创新 —— Sim-to-Real Alignment in Embedding Space**：

- 不优化 token-level 保真度（易导致 model collapse）
- 用 cosine similarity 在预训练 embedding 空间衡量模拟状态与真实状态的一致性
- 奖励函数：`r = standardized(cos_sim(embed(ŝ_{t+1}), embed(s_{t+1})))`
- 用 GRPO 优化世界模型参数

**实验结果**（Qwen2.5-7B）：


| Benchmark | Base | +RWML (self-supervised) | +RWML+Task RL            |
| --------- | ---- | ----------------------- | ------------------------ |
| ALFWorld  | 49.3 | 68.9 (+19.6)            | 80.4 (+6.9 vs task-only) |
| τ²-Bench  | 44.3 | 51.2 (+6.9)             | 55.1 (+5.7 vs task-only) |


- 纯自监督（无需专家数据、无需任务奖励）即可大幅提升
- 与任务 RL 结合效果叠加，匹配专家数据训练的性能

---

#### 2.2.4 ITP：自适应想象视野 (2025)

**核心思想**：学习何时以及想象多远——用轻量 K-head 预测器自适应选择每个 step 的 imagination horizon。

**POIMDP 形式化**：扩展 POMDP 使 agent 策略条件于观测 + 想象轨迹：`a_t ~ π_θ(· | s_t, τ̂_t^(K_t))`

**两个变体**：

- ITP_I (training-free)：推理时 agent 选 K → WM 想象 → 反思 → 行动
- ITP_R (RL-trained)：pseudo-labeling 最优 horizon → warm-up SFT → online A2C 联合优化 policy + K-head

**实验结果**：


| Benchmark           | ITP_R  | 最佳 Baseline  | 提升    |
| ------------------- | ------ | ------------ | ----- |
| ALFWorld (Qwen3-8B) | 88.57% | 82.14% (IWM) | +6.43 |
| ScienceWorld unseen | 56.95% | 54.30% (IWM) | +2.65 |


- 自适应 K 远优于固定 K：更高成功率 + 更低计算开销
- ALFWorld 平均 K=3，ScienceWorld 平均 K=8（匹配任务复杂度）

---

#### 2.2.5 ✨LeWorldModel (LeWM)：稳定端到端 JEPA 世界模型 (2026, LeCun/Balestriero 等)

**来源**：Mila / NYU / Samsung SAIL / Brown，arXiv 2603.19312（2026-03）| [项目页](https://le-wm.github.io/) | [代码](https://github.com/lucas-maes/le-wm)

**重要定位说明**：LeWM 是 **pixel / 连续控制域**的世界模型（Push-T、OGBench-Cube、Reacher 等 2D/3D control），**不是文本 agent 方法**。

**核心思想**：第一个能从原始像素**端到端、稳定**训练的 JEPA（Joint-Embedding Predictive Architecture）世界模型——编码器把观测映射到紧凑 latent，predictor 在 latent 空间 action-conditioned 地预测下一状态 embedding。**reward-free、reconstruction-free**。

**关键创新 —— 两项损失即稳定，SIGReg 防坍缩**：

```
L_LeWM = L_pred + λ · SIGReg(Z)

L_pred  = ||ẑ_{t+1} - z_{t+1}||²          # 下一 embedding 预测（teacher forcing）
SIGReg  = (1/M) Σ_m T(Z·u^(m))            # 投影到 M 个随机方向做 Epps-Pulley 正态检验
```

- SIGReg（Sketched Isotropic Gaussian Regularizer，源自 LeJEPA 2511.08544）强制 latent 服从各向同性高斯，**由 Cramér-Wold 定理保证：匹配所有一维边缘 = 匹配联合分布**
- **抛弃所有启发式**：无 stop-gradient、无 EMA、无预训练编码器、无多项 VICReg loss
- 可调 loss 超参从 6 个降到 **1 个**（λ，可用对数复杂度 bisection 搜索）

**架构与效率**：ViT-tiny 编码器（192 维 CLS token）+ 6 层 transformer predictor（AdaLN-zero action 条件）；**~15M 参数、单 GPU 几小时**；规划比 DINO-WM 快 **48×**（每观测约 200× 更少 token）。

**latent 规划**：CEM + MPC，在 latent 空间 rollout 到 horizon H，用终态 latent 与目标 embedding 的距离作 cost；只执行前 K 步再重规划（缓解自回归误差累积）。

**关键的两点**：

1. **Surprise / Violation-of-Expectation**：用 latent 预测误差度量"惊讶度"，能显著检测物理不可能事件（teleport 扰动 p<0.01）。
2. **稳定的 latent WM 范式**：LeWM 给出一个**在 latent 空间预测 + 高斯正则防坍缩**的稳定替代。

**局限**（对迁移到文本 agent 很关键）：仅短 horizon 规划；需有覆盖度的离线数据；需 action label；低内在维度环境下 SIGReg 反而变差；**全部实验在像素控制域，未触及文本/工具 agent**。

---

### 2.3 信用分配与过程奖励模型

#### 2.3.1 AgentPRM：TD+GAE Step-level 值函数 (WWW 2026)

**核心问题**：MC-based PRM 在 agentic 场景下失败——需要从每个中间状态重新执行环境交互，代价过高。

**技术方案 —— TD+GAE 值估计**：


| 概念                   | 公式                                                          | 含义              |
| -------------------- | ----------------------------------------------------------- | --------------- |
| Promise (Q-value)    | Q^π(s_t, a_t) = E[r(u, τ)]                                  | 当前状态-动作对达成目标的概率 |
| Progress (Advantage) | A^π(s_t, a_t) = Q(s_t, a_t) - V(s_t)                        | 动作相对于平均水平的进展    |
| TD 残差                | δ(s_t, a_t) = r_t + γ·M_φ(s_t, a_t) - M_φ(s_{t-1}, a_{t-1}) | 单步预测误差          |
| GAE                  | Â = Σ(γλ)^k · δ(s_{t+k}, a_{t+k})                           | 平衡偏差-方差的优势估计    |


**训练损失**：L = L_Q (MSE on Q-value) + β·L_A (MSE on consecutive Q-value differences)

**8x 样本效率**：AgentPRM (TD) 用 1.0x token 开销 vs MC-based 用 1.5-2.8x，在 BoN@64 上性能相当或更优。

**应用方式**：Best-of-N 重排序 / 束搜索 / PPO 奖励信号

---

#### 2.3.2 信用分配全景 (Survey, 2604.09459)

**覆盖**：2024-2026 年发表的 47 篇信用分配方法论文

**分类体系**：5 个粒度级别 × 5 种方法论


| 粒度            | 最相关方法                                    |
| ------------- | ---------------------------------------- |
| Token         | VinePPO, RED, T-REG                      |
| Segment       | SPO, SCAR, TEMPO                         |
| **Step/Turn** | **AgentPRM, ArCHer, SWEET-RL, Turn-PPO** |
| Multi-Agent   | M-GRPO, SHARP                            |


**新兴范式 —— Hindsight Counterfactual**：2026 年 3 月一周内出现 3 篇独立论文 (HCAPO, C3, CCPO)，标志社区共识

**领域共识**：

1. Turn 是 agentic RL 中信用分配的自然原子单位
2. Hindsight 分析 > 前向预测（在随机环境中）
3. 层次结构（plan/execute/verify）应指导信用分配结构

---

### 2.4 环境合成

#### 2.4.1 AWM：代码驱动合成环境 (2026, Snowflake)

**规模**：1,000 个环境、35,062 个工具、10,000 个任务

**Pipeline**：Scenario → Task → Database (SQLite) → Interface (FastAPI + MCP) → Verification Code

**核心优势**：

- 代码驱动 vs LLM 模拟：BFCLv3 上 65.94 vs 52.53 (+13.4)
- 确定性状态转移、无 hallucination
- 推理开销极低（无需每步 LLM 调用）
- 支持 1,024 并行实例

**OOD 泛化**：所有评估 benchmark (BFCLv3, τ²-bench, MCP-Universe) 均与训练环境零重叠

---

### 2.5 经验回放

#### 2.5.1 Experience Replay for LLM RL (Meta FAIR, 2026)

**核心发现**：推理计算占训练 GPU 时间的 **>80%**，on-policy 逐次丢弃 rollout 是计算次优的。

**最优缓冲区设计**：

- FIFO 循环缓冲区，均匀采样
- 中等回放比率（2-5x）最优
- 可节省 **40% 计算开销**，同时保持或提升准确率
- 缓冲区起正则化作用：稳定训练、保留输出多样性

#### 2.5.2 FreshPER：新鲜度感知优先回放 (KAUST, 2025)

**核心问题**：标准 PER 在 LLM RL 中失败——高优先级旧轨迹主导采样，但已不具信息量。

**解决方案 —— 指数年龄衰减**：

```
p_i = p_i^{base} × exp(-Δ_i / τ)
```

- `p_i^{base}`：基于信息量的基础优先级
- `Δ_i`：自采集以来的梯度步数
- `τ`：年龄衰减常数（半衰期 = τ ln 2）

**效果**：在 NQ Search 上 +46%，Sokoban 上 +367%（vs 标准 PER 一致性下降）

#### 2.5.3 ReVal：值函数 + 回放缓冲 (南京大学, 2026)

**核心创新**：将 LLM logits 解释为 Q-values → 天然支持 off-policy 学习

**校准初始化**：当 r=0 且 π_θ = π_ref 时，loss 恰好为 0（无虚假漂移）

**加速**：4.3x 收敛速度提升（easy 5.2x / medium 4.1x / hard 3.6x）

---

### 2.6 Rollout 策略 (GFCR Survey, 2026)

**四阶段分类**：


| 阶段           | 功能        | 关键机制                   |
| ------------ | --------- | ---------------------- |
| **Generate** | 生成候选轨迹和拓扑 | 单/组/树/图 rollout；引导式脚手架 |
| **Filter**   | 构建中间信号    | 验证器、评委、critic；过程评分     |
| **Control**  | 分配计算、管理决策 | prompt 优先级；自适应分配；提前退出  |
| **Replay**   | 保留和重用历史产物 | 自演化课程；轨迹库；组合重用         |


**病理诊断索引**：

- 覆盖不足 → Generate（采样多样性不足）
- 奖励噪声 → Filter（验证器不可靠）
- 计算浪费 → Control（无自适应停止）
- 遗忘/停滞 → Replay（无课程演化）

---

### 2.7 2026 新增竞品与并行工作（novelty 审计补充）

> 这些工作在最初调研中被遗漏，但对论文 novelty 和 baseline 至关重要，**必须纳入 related work 与主表对照**。arXiv ID 标注为待二次核验。


| 工作 (arXiv，待核验)                                          | 类型                                    | 与本课题的关系 / 为什么必须加                                                |
| ------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------- |
| **PaW** (2606.02388)                                    | WM + policy 协同训练（辅助 next-obs 预测 loss） | **最大威胁**：在 ALFWorld/WebShop 上仅 2% 开销拿到 +8%。审稿人会问"为何要重型三模块"。必引必比 |
| **3SPO** (2606.09961)                                   | step-level，免 PRM/免 value，主打样本效率       | 2026 强 baseline，ALFWorld +22.6%/WebShop +15.6%，与"样本效率"主张正面撞车    |
| **GiGPO** (2505.10978, NeurIPS'25)                      | step-level 信用分配                       | SPEAR 的 step-level 基座本身，必备对照                                    |
| **Rollout-level Advantage-PER** (2606.04560)            | rollout 级 advantage 优先 + age eviction | P3 的**并行发明**（math 域），不区分则 P3 novelty 被削                         |
| **Efficient-RL-Replay** (2604.08706)                    | fresh-anchored replay                 | 同上，需明确差异（我们是 agentic 三源 + WM 置信度）                               |
| **Dyna-Think** (2506.00320)                             | 内部 WM 模拟 + critique generation        | WM-in-reasoning 代表，补全世界模型谱系                                     |
| **WebWorld** (2602.14721) / **WebEvolver** (2504.21024) | WM 作为大规模数据生成器                         | 强化"WM-as-experience-source"相关工作                                 |


**对 novelty 的影响**：P1（世界模型想象）已被 DynaWeb/PaW/Dyna-Mind 占领；P2（counterfactual dense reward）≈ 经典 MVE/STEVE 移植；**唯一干净的增量是 P3——confidence-gated 三源优先想象回放**。详见 `paper-proposal.md` §0 的定位修订。

---

### 2.8 顶会最新补充（2025-2026 多会场扫掠）

> 来自 ICLR/NeurIPS/ICML/COLM/RLC 2025-2026 的定向检索。**注**：COLM 2025/2026 未命中高置信的 "world-model + agentic RL" 正会论文（COLM 更偏 reasoning/tool-use RL）。arXiv ID 待二次核验。下表分两类：LLM/VLM agent 域（直接竞品/baseline）与通用 RL/控制域（方法论来源 + 关键负结果）。

**A. LLM/VLM agent 域（直接相关）**


| 工作                               | 会议/年份        | arXiv      | 核心                                                                                                                   | 对 DREAM 的意义                                                                                     |
| -------------------------------- | ------------ | ---------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **VAGEN**                        | NeurIPS 2025 | 2510.16907 | VLM agent 推理拆成 StateEstimation+TransitionModeling，WorldModeling Reward 做 turn-level dense supervision + Bi-Level GAE | **P2 dense reward 直接对照**：证明 turn-level world-modeling reward 能替代稀疏终局奖励（3B 模型 0.82 > GPT-5 0.75） |
| **CoMap**                        | 2026         | 2606.02372 | textual WM 与 policy 闭环**共演化**，policy 用 WM future feedback 反思，on-policy 轨迹自蒸馏更新 WM                                    | **挑战 DREAM 固定 WM 设定**：需补 fixed-WM / periodic-update / co-evolving 三种对照（Qwen3-8B 69.53%→72.11%）  |
| **ProAct**                       | 2026         | 2602.05327 | 环境 MCTS 生成 grounded lookahead → 压缩成 reasoning chain SFT → MC-Critic 给 PPO/GRPO 低方差 value                             | "真实轻量 rollout + critic" 作 dense reward 校准器，减少 hallucinated imagination                          |
| **TSR**                          | 2026         | 2602.11767 | 把 best-of-N/beam/lookahead 从测试时移到训练 rollout 生成阶段                                                                     | prioritized rollout construction 的强 baseline（不学 WM）；Sokoban/WebShop 最高 +15%                     |
| **WAC: WM-Augmented Web Agents** | 2026         | 2602.15384 | 执行前用 WM 模拟 action 后果 + judge 风险评估 + action correction                                                                | inference-time consequence simulation 对照（VisualWebArena 24.5%）                                  |


**B. 通用 RL / 控制域（方法论来源 + 必答负结果）**


| 工作                                         | 会议/年份     | arXiv      | 核心                                                                                                               | 对 DREAM 的意义                                                                                 |
| ------------------------------------------ | --------- | ---------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Stealing That Free Lunch**               | ICML 2025 | 2412.14312 | 系统揭示 Dyna-style 合成 rollout 在多数 DMC 环境**反而降性能**                                                                   | **必须正面回应的负结果**：DREAM 的说服力不在"想象有用"，而在"哪些想象值得进 buffer、何时进、多大权重"                               |
| **WIMLE**                                  | ICLR 2026 | 2602.14351 | IMLE 学多模态 stochastic dynamics + ensemble/latent 采样估不确定性，对 synthetic transition 做 **confidence weighting**        | **P3 的通用 RL 侧最强 baseline**：直接对应 confidence-gated imagination replay（Humanoid-run 样本效率 +50%） |
| **Simulus**                                | 2025      | 2502.11537 | 组合 multi-modality tokenization + intrinsic motivation + **prioritized WM replay** + regression-as-classification | **最贴近的 prioritized WM replay 竞品**：DREAM 必须强调 agentic 三源 + WM 置信度 + dense reward 的差异         |
| **EAWM (Obs→Events)**                      | ICLR 2026 | 2601.19336 | 自动从观测生成 event + event boundary 分段塑造 WM 表示                                                                        | **event-level replay 粒度**来源：agentic replay 可按关键事件分段/优先级，而非纯 token/step                      |
| **JEDI**                                   | 2026      | 2605.13013 | diffusion WM 移到紧凑 latent space 端到端学习                                                                             | 支持 latent WM（与 LeWM 同向），降 hallucination 与 rollout cost（vs DIAMOND 省 43% VRAM）               |
| **Horizon Imagination**                    | ICLR 2026 | 2602.08032 | diffusion WM 并行 horizon denoising，解耦 denoise 预算与 rollout horizon                                                 | 若 DREAM 用生成式 WM，降低 imagined rollout 成本的可迁移技术                                                |
| **Improving Transformer WM (Dyna-warmup)** | ICML 2025 | 2502.01591 | Dyna with warmup + nearest-neighbor tokenizer + block teacher forcing                                            | **Dyna warmup 印证**：imagination replay 不应训练早期就强混入（Craftax 69.66% > DreamerV3 53.2%）          |
| **Simulus / SPlaTES (skill-WM)**           | RLC 2025  | 待核验        | 学 temporally extended predictable skills，在 abstract skill WM 上 MPC                                               | 长程 agentic 想象单步易误差累积，可把 replay 单位提升到 option/skill/subtask                                   |


---

### 2.9 经验回放 / off-policy / 样本效率最新补充（P3 竞争格局）

> 定向检索 LLM RL 经验回放与样本效率方向（ICLR/NeurIPS/ICML 2025-2026 + 强预印本）。直接关系到 P3「confidence-gated 三源优先想象回放」的 novelty 防守。arXiv ID 待二次核验。


| 工作                                                  | 会议/年份          | arXiv                       | 核心                                                                            | 对 P3 的意义                                                                                    |
| --------------------------------------------------- | -------------- | --------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Fatemi: Prioritized Replay for RL Post-training** | 2026           | 2601.02648                  | **问题级**（非轨迹级）优先化调度，明确主张"transition/trajectory 级 PER 不适合序列模型"                  | **理论挑战者**：P3 必须正面回应"为何 trajectory-level 优先回放对 LLM agentic 有效"；二者正交（可在 Fatemi 选出的问题上再做轨迹级优先） |
| **VESPO**                                           | 2026           | 2602.10693                  | 序列级 off-policy IS 的闭式 reshaping kernel，可证方差上界，staleness 达 64× 仍收敛             | **可替代 P3 的简单 clip-ratio**：为历史/想象源提供理论无偏的 IS 校正                                              |
| **TBRM**                                            | NeurIPS 2025   | 待核验                         | Trajectory Bellman Residual Minimization，用 logit 作 Q，天然支持 off-policy + replay | value-based 信号可作 P3 置信度的替代度量（Q 代替 reward-model 置信度）                                         |
| **VLM-RB**                                          | 2026           | 2602.01915                  | 冻结 VLM 对 sub-trajectory 语义打分作回放优先级（通用 RL，游戏/机器人 +11-52%）                      | **置信度门控的视觉先驱（支持性佐证非竞品）**：P3 差异 = 用训练中 LLM **在线**置信度（随策略进化）+ 文本 agentic + 想象源                |
| **EAPO**                                            | ICML 2026      | 待核验 (OpenReview QOoQ0Bo2ls) | 经验以策略自适应方式在关键决策点注入（rollout 生成阶段）                                              | 与 P3 管线位置不同（注入 vs 采样），可串联：EAPO 生成期注入 + P3 训练期优先采样                                           |
| **RLEP**                                            | 2025           | 2507.07451                  | 仅重放**验证正确**的轨迹，mini-batch 混合新鲜 + 成功历史（G→G+M）                                  | P3 可视为 RLEP 超集：优先化（非均匀）+ 置信度门控（非 binary）+ 想象源                                               |
| **SLEA-RL**                                         | 2026           | 2603.18079                  | 多步 agentic 自演化**步级**经验库，按步聚类检索注入（ALFWorld/WebShop 类）                          | "步级经验检索注入"是 P3 想象源的**记忆式替代**（用真实成功步替代生成想象步）                                                 |
| **Siri**                                            | 2026           | 2606.02355                  | GiGPO 热身 → 从成功轨迹自挖 compact skills → 轨迹 utility×动作 advantage 蒸馏                | 双重加权（utility×advantage）是 P3 segment 优先级设计的近亲；ALFWorld 0.908→0.930                           |
| **AgentFlow / Flow-GRPO**                           | ICLR 2026 Oral | 2510.05592                  | 四模块 agentic + 演化记忆；Flow-GRPO 把终局奖励广播到每步解决长程信用分配                               | 多步 agentic 信用广播机制，可指导 P3 历史回放轨迹各步优先级权重                                                      |
| **LoRR**                                            | ICLR 2026      | 2508.06412                  | 高回放比 + 周期性参数重置对抗 primacy bias                                                 | 与 P3 正交互补：reset"刷新"对陈旧数据适应性 vs P3 置信度过滤陈旧                                                   |
| **DOTS+RR**                                         | NeurIPS 2025   | 2506.05316                  | 难度自适应选择 + FIFO Rollout Replay，训练时间 -23~65%                                    | RR 的 FIFO 无优先级回放是 P3 历史源前序；DOTS 难度估计可作门控的高效替代                                               |
| **Pilot-Commit / AERO**                             | 2026           | 2605.26606 / 2602.14338     | 预算感知 rollout 分配（Pilot 估方差 → Commit 高信号；Beta-Binomial 后验自适应采样）                 | "在线可学性/置信度估计"的轻量实现，可决定是否值得进入完整生成 + 写入 buffer                                                |


**P3 差异化矩阵（必须写进论文）**：


| 维度      | FreshPER   | Rollout-Adv-PER | Fatemi         | VLM-RB        | EAPO  | **P3 (ours)**          |
| ------- | ---------- | --------------- | -------------- | ------------- | ----- | ---------------------- |
| 数据来源    | 历史 rollout | 历史 rollout      | 问题调度（无 buffer） | 历史 transition | 先验注入  | **真实 on/off + 想象三源**   |
| 优先信号    | PER×年龄     |                 | advantage      |               | 问题成功率 | 冻结 VLM 分               |
| 世界模型想象源 | ❌          | ❌               | ❌              | ❌             | ❌     | **✅**                  |
| 在线置信度门控 | ❌（年龄代替）    | ❌               | ✅间接            | ✅（外部固定）       | ✅     | **✅（自身 LLM 在线，随策略进化）** |
| 目标域     | 推理         | 推理              | 推理             | 视觉控制          | 推理    | **agentic 多步**         |


**结论**：P3 的两个不可被现有工作覆盖的独特点是 **(1) 世界模型想象作为第三回放源** 与 **(2) 随训练进化的在线 WM 置信度门控**；其余维度（年龄衰减、advantage 优先、问题调度）均已有强工作，必须在论文中显式区分并做对应消融。

---

## 三、对比分析与研究空白

### 3.1 世界模型方法对比


| 维度         | DynaWeb     | Dyna-Mind     | RWML          | ITP            | LeWM (pixel)               |
| ---------- | ----------- | ------------- | ------------- | -------------- | -------------------------- |
| WM 形态      | 独立 120B LLM | 集成于推理过程       | Policy LLM 自身 | 独立 LLM         | 独立 15M JEPA                |
| 预测目标       | 页面 delta    | 无（agent 自行模拟） | Embedding 对齐  | 文本观测           | latent embedding           |
| 想象 horizon | 固定 5 步      | 固定（按环境）       | 无想象 rollout   | **自适应 K-head** | CEM+MPC 短 horizon          |
| RL 中需真实环境？ | 否（纯想象）      | 是（SimRollout） | 是（训练数据）       | 是（A2C rollout） | 否（离线 + latent 规划）          |
| RL 算法      | GSPO        | Dyna-GRPO     | GRPO          | A2C            | 无（MPC 规划，reward-free）      |
| WM 共演化？    | 否（冻结）       | 无独立 WM        | 否（预训练）        | 否（冻结）          | 否（离线训练）                    |
| 防坍缩/校准     | -           | -             | embedding cos | -              | **SIGReg 高斯正则 + surprise** |
| 模态         | 文本(web)     | 文本+视觉         | 文本            | 文本             | **像素(控制)**                 |


### 3.2 经验回放方法对比


| 方法        | Buffer 类型    | 采样策略       | Off-policy 校正 | 最适场景          |
| --------- | ------------ | ---------- | ------------- | ------------- |
| SPEAR SIL | FIFO + 正优势过滤 | 均匀随机       | P50 优势重校准     | 探索-利用平衡       |
| ER (Meta) | FIFO 循环      | 均匀随机       | 无 / AsymRE    | 计算效率          |
| ReVal     | FIFO         | 均匀         | Bellman 残差    | 值学习           |
| FreshPER  | 轨迹级          | 优先级 + 年龄衰减 | IS 权重         | 多轮 agentic 任务 |
| ExGRPO    | 按正确率/熵组织     | 优先级        | 混合策略目标        | RLVR 推理       |


### 3.3 关键研究空白

```
┌──────────────────────────────────────────────────────────┐
│                  尚未被统一解决的问题                        │
├──────────────────────────────────────────────────────────┤
│ 1. 无工作统一了世界模型、优先回放、稠密奖励三大支柱            │
│    → DynaWeb 有 WM 但无 PRM 和优先回放                     │
│    → SPEAR 有回放但无 WM 和稠密奖励                        │
│    → AgentPRM 有稠密奖励但未与 model-based 训练结合          │
│                                                          │
│ 2. 无工作实现了 WM 与策略的共演化                           │
│    → 所有 WM 方法训练后冻结或无独立 WM                      │
│                                                          │
│ 3. 想象 rollout 的自适应控制不足                            │
│    → ITP 提出了 K-head 但未与 GRPO 系方法结合               │
│    → 无工作基于 WM 不确定性调节想象比例                      │
│                                                          │
│ 4. Agentic 场景下的经验回放研究不足                         │
│    → 现有回放文献主要针对 reasoning 任务                     │
│    → FreshPER 在 agentic 上有初步验证但未与 SIL 结合        │
│                                                          │
│ 5. 跨域通用框架缺失                                        │
│    → 所有方法在单一/少数域上验证                             │
│    → 无统一的多 benchmark 样本效率评测                       │
└──────────────────────────────────────────────────────────┘
```

### 3.4 Takeaway

1. **最稳妥的论文切入点**：不是“做一个更大的 Agent 框架”，而是围绕“真实环境交互样本效率”建立清晰指标和消融。
2. **最自然的技术基线**：SPEAR，因为它已把 self-imitation、curriculum 和 tool-use reward 做进 verl 训练 loop。
3. **最需要导师拍板的问题**：是先做低风险的 `SPEAR + Prioritized Replay + Dense Reward`，还是直接把 world model imagination 作为主贡献。

---

## 四、完整参考文献

> arXiv ID 与 venue 已经引用核验子代理用 HF Papers API + web 核对（第 1-21 条）；第 22-28 条为 novelty 审计新增竞品，arXiv ID 标注"待核验"。


| #   | 论文                                                                | 会议/年份             | arXiv ID            | 关键贡献                                                 | 代码                                                                     |
| --- | ----------------------------------------------------------------- | ----------------- | ------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------- |
| 1   | SPEAR: Self-imitation with Progressive Exploration for Agentic RL | ICLR 2026         | 2509.22601          | SIL + curriculum + intrinsic reward                  | [GitHub](https://github.com/TencentYoutuResearch/SPEAR)                |
| 2   | DynaWeb: Model-Based RL of Web Agents                             | 2026 (预印本)        | 2601.22149          | Dyna MBRL, delta prediction                          | -                                                                      |
| 3   | Dyna-Mind: Learning to Simulate from Experience                   | ICLR 2026         | 2510.09577          | ReSim + Dyna-GRPO + Sim Score                        | [GitHub](https://github.com/jasonyux/Dyna-Mind)                        |
| 4   | RWML: Reinforcement World Model Learning                          | 2026 (预印本)        | 2602.05842          | Embedding 空间 sim-to-real 对齐                          | -                                                                      |
| 5   | ITP: Imagine-then-Plan with Adaptive Lookahead                    | 2026 (预印本)        | 2601.08955          | POIMDP + K-head 自适应 horizon                          | [GitHub](https://github.com/loyiv/ITP)                                 |
| 6   | Dyna-Think: World Model Simulation in AI Agents                   | 2025 (OpenReview) | 2506.00320          | 内部 WM + critique generation                          | -                                                                      |
| 7   | AWM: Agent World Model                                            | 2026              | 2602.10090          | 1000 合成环境 + MCP                                      | [GitHub](https://github.com/Snowflake-Labs/agent-world-model)          |
| 8   | AutoForge: Automated Environment Synthesis                        | 2025              | 2512.22857          | API → DAG → 任务 + ERPO                                | -                                                                      |
| 9   | COVERT: Controllable Tool-Use Data Synthesis                      | 2026 (预印本)        | 2604.09813          | Oracle-preserving augmentation                       | -                                                                      |
| 10  | ASTRA: Automated Synthesis of Trajectories and Arenas             | 2026              | 2601.21558          | Tool-call graph → 代码可执行环境                            | -                                                                      |
| 11  | AgentPRM (Xi et al.): Step-Wise Promise and Progress              | WWW 2026          | 2511.08325          | TD+GAE, 8x 效率, Promise+Progress                      | 待确认                                                                    |
| 12  | AgentPRM (Choudhury): Practical Framework                         | 2025              | 2502.10325          | MC-based + InversePRM + 迭代训练                         | [GitHub](https://github.com/sanjibanc/agent_prm)                       |
| 13  | Tree-GRPO: Tree Search for LLM Agent RL                           | ICLR 2026         | 2509.21240          | 树搜索 rollout, 1/4 预算                                  | [GitHub](https://github.com/AMAP-ML/Tree-GRPO)                         |
| 14  | Credit Assignment Survey (From Reasoning to Agentic)              | 2026              | 2604.09459          | 47 方法, 5×5 分类体系                                      | [GitHub](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL) |
| 15  | Efficient RL Training for LLMs with Experience Replay (Meta FAIR) | 2026              | 2604.08706          | 回放缓冲理论分析, 40% 计算节省                                   | -                                                                      |
| 16  | Off-Policy Value-Based RL for LLMs (ReVal)                        | 2026              | 2603.23355          | Logit-as-Q, 4.3x 加速                                  | -                                                                      |
| 17  | FreshPER: Freshness-Aware Prioritized ER                          | 2026              | 2604.16918          | 指数年龄衰减                                               | [GitHub](https://github.com/Vision-CAIR/Freshness-Aware-PER)           |
| 18  | ExGRPO: Learning to Reason from Experience                        | ICLR 2026         | 2510.02245          | 经验组织 + 混合策略目标                                        | -                                                                      |
| 19  | GFCR Survey: Rollout Strategies for LLM RL                        | 2026              | 2605.02913          | Generate-Filter-Control-Replay 分类                    | -                                                                      |
| 20  | Dreamer 4: Scalable World Model Agents                            | 2025              | 2509.24527          | 纯想象训练, Minecraft 获钻石                                 | [项目页](https://danijar.com/project/dreamer4/)                           |
| 21  | WKM: Agent Planning with World Knowledge Model                    | NeurIPS 2024      | 2405.14205          | 参数化世界知识辅助规划                                          | [GitHub](https://github.com/zjunlp/WKM)                                |
| 22  | PaW: Policy and World Modeling Co-Training                        | 2026              | 2606.02388 (待核验)    | WM+policy 协同训练，2% 开销                                 | -                                                                      |
| 23  | 3SPO: step-level sample-efficient policy opt.                     | 2026              | 2606.09961 (待核验)    | 免 PRM/免 value step-level，主打样本效率                      | -                                                                      |
| 24  | GiGPO: step-level group policy opt.                               | NeurIPS 2025      | 2505.10978 (待核验)    | SPEAR 的 step-level 基座                                | -                                                                      |
| 25  | Rollout-level Advantage-PER                                       | 2026              | 2606.04560 (待核验)    | rollout 级 advantage 优先 + age eviction                | -                                                                      |
| 26  | WebWorld: WM as data generator                                    | 2026              | 2602.14721 (待核验)    | WM 作为大规模数据生成器                                        | -                                                                      |
| 27  | WebEvolver                                                        | 2025              | 2504.21024 (待核验)    | WM 驱动的 web agent 自进化                                 | -                                                                      |
| 28  | LeWorldModel (LeWM)                                               | 2026              | 2603.19312          | 稳定端到端 JEPA from pixels + SIGReg + surprise           | [GitHub](https://github.com/lucas-maes/le-wm)                          |
| 29  | LeJEPA                                                            | 2025              | 2511.08544          | SIGReg 各向同性高斯正则，可证明防坍缩 SSL                           | -                                                                      |
| 30  | VAGEN: WM Reasoning for Multi-Turn VLM Agents                     | NeurIPS 2025      | 2510.16907          | WorldModeling Reward + Bi-Level GAE turn-level dense | -                                                                      |
| 31  | CoMap: Co-Evolving WM and Policy for LLM Agents                   | 2026              | 2606.02372          | textual WM 与 policy 闭环共演化                            | -                                                                      |
| 32  | ProAct: Agentic Lookahead in Interactive Env                      | 2026              | 2602.05327          | 环境 MCTS lookahead + MC-Critic                        | -                                                                      |
| 33  | TSR: Trajectory-Search Rollouts for Multi-Turn RL                 | 2026              | 2602.11767          | 训练期高质量 rollout 构造                                    | -                                                                      |
| 34  | WAC: WM-Augmented Web Agents w/ Action Correction                 | 2026              | 2602.15384          | 执行前 consequence simulation + correction              | -                                                                      |
| 35  | Stealing That Free Lunch (Dyna 负结果)                               | ICML 2025         | 2412.14312          | Dyna 合成 rollout 在多数 DMC 反而有害（必答）                     | -                                                                      |
| 36  | WIMLE: Uncertainty-Aware WM with IMLE                             | ICLR 2026         | 2602.14351          | 不确定性加权 synthetic transition（P3 通用 baseline）          | -                                                                      |
| 37  | Simulus: Sample-Efficient WM Agents                               | 2025              | 2502.11537          | prioritized WM replay + intrinsic motivation         | -                                                                      |
| 38  | EAWM: Event-Aware World Models                                    | ICLR 2026         | 2601.19336          | event-level 分段 WM 表示                                 | -                                                                      |
| 39  | JEDI: Joint Embedding Diffusion WM                                | 2026              | 2605.13013          | latent diffusion WM，降 hallucination/cost             | -                                                                      |
| 40  | Horizon Imagination (diffusion WM rollout)                        | ICLR 2026         | 2602.08032          | 并行 horizon denoising 降 rollout 成本                    | -                                                                      |
| 41  | Improving Transformer WM (Dyna-warmup)                            | ICML 2025         | 2502.01591          | Dyna warmup + NN tokenizer + block TF                | -                                                                      |
| 42  | Fatemi: Prioritized Replay for RL Post-training                   | 2026              | 2601.02648          | 问题级优先调度（反对 trajectory-PER）                           | -                                                                      |
| 43  | VESPO: Variational Sequence-level Soft PO                         | 2026              | 2602.10693          | 序列级 off-policy IS 闭式 reshaping                       | -                                                                      |
| 44  | TBRM: Trajectory Bellman Residual Minimization                    | NeurIPS 2025      | 待核验                 | logit-as-Q，off-policy + replay                       | -                                                                      |
| 45  | VLM-RB: VLM-Guided Experience Replay                              | 2026              | 2602.01915          | 冻结 VLM 语义优先级（置信度门控视觉先驱）                              | -                                                                      |
| 46  | EAPO: Experience Augmented Policy Optimization                    | ICML 2026         | 待核验 (OR:QOoQ0Bo2ls) | 经验自适应注入关键决策点                                         | -                                                                      |
| 47  | RLEP: RL with Experience Replay for Reasoning                     | 2025              | 2507.07451          | 仅重放验证正确轨迹                                            | -                                                                      |
| 48  | SLEA-RL: Step-Level Experience-Augmented RL                       | 2026              | 2603.18079          | 多步 agentic 步级自演化经验库                                  | -                                                                      |
| 49  | Siri: Self-Internalizing RL with Intrinsic Skills                 | 2026              | 2606.02355          | 成功轨迹自挖技能 + utility×advantage 蒸馏                      | -                                                                      |
| 50  | AgentFlow / Flow-GRPO                                             | ICLR 2026 Oral    | 2510.05592          | 四模块 agentic + 终局奖励逐步广播                               | -                                                                      |
| 51  | LoRR: LLM Optimization with Reset Replay                          | ICLR 2026         | 2508.06412          | 高回放比 + 周期参数重置                                        | -                                                                      |
| 52  | DOTS+RR: Difficulty-targeted Selection + Rollout Replay           | NeurIPS 2025      | 2506.05316          | 难度选择 + FIFO 回放，-23~65% 时间                            | -                                                                      |
