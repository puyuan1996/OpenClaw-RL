# Terminal-Agent Latent World Model 完整设计方案（v2，已整合两轮评审）

> **合并自**：`terminal-rl-latent-reward-wm.md`（可行性 + 文献定位）与 `0616 SETA + DAPO Latent World Model 调研与集成方案.md`（代码事实 + 集成方案），融入三路并行调研（实现机制 / JEPA 稳定性与跨源对齐 / 竞品扫描）与两轮 reviewer（技术审阅 Opus + 新颖性红队 GPT-5.5）反馈后的统一设计。
>
> 生成 / 修订：2026-06-16 | 状态：设计稿 v2（已据评审下调过度声明、修正核心 loss 退化问题、重定位 novelty 到 U2）
>
> **重定位后的单一最强 claim**：
> > **A policy-hidden-conditioned latent action-value world model for terminal agents that enables *pre-execution command screening* (calibration measured, not assumed) and dense shaping — going beyond ECHO-style token-level environment prediction by providing a queryable latent predictor + value head for counterfactual command ranking, while preserving the unit-test verifier as ground truth.**
>
> **评审整合要点**：①核心科学问题 = **「动作敏感性」**（predictor 是否真依赖 `a_t`，而非退化成 state→mean 预测）；②头牌从「latent 统一 dynamics+reward/value」改为 **U2 校准式执行前命令筛选**（ECHO 结构上做不到的廉价能力）；③ECHO(2605.24517, 已核实) 为**头号必比 baseline**；④下调过度声明（judge 稠密性、SIGReg surprise=校准、在线替代 judge）；⑤修正 §4.6 loss 退化解、§4.5 折扣淹没末端信号、§7 PP/层矛盾。

---

## 0. 执行摘要（TL;DR）

1. **原理可行、非新发明**：「latent dynamics + reward/value head」是 MuZero/TD-MPC2/UniZero/DreamerV3 标准做法；LeWM(2603.19312) 是「去掉 reward/value 头、只留 dynamics+SIGReg」的极简版。本方案 = 把头装回 LeWM 风格 backbone + 迁移到 terminal/LLM-agent + 用 LLM hidden 当 encoder。
2. **但顶会 novelty 不在「统一 latent」**（已被 MuZero/TD-MPC2/PriorZero 范式化），**而在「policy-hidden latent 对 terminal 候选命令做 counterfactual 打分 + 校准式执行前筛选 + 保留 verifier」**。
3. **头号竞品 ECHO（已核实）**：向 GRPO 加「环境观测 token 的 CE 辅助损失」，同 forward、零额外开销，TerminalBench-2.0 pass@1 近翻倍，λ=0.05；甚至「单用环境预测损失可 verifier-free 自提升」。**ECHO 已占据「terminal 反馈做稠密监督」这块地**。我们必须证明 latent/hidden/value/U2 能**超越 ECHO 的极简 token-CE**——这是全文成败点。
4. **用途边界**：✅ **U2 校准式候选命令筛选（头牌）**、✅ U1 稠密 value/progress 信用分配（支撑）；⚠️ U3 imagination replay（移出核心论文，留作扩展）；❌ **U4 用 latent 替代单测出末端真值（不可行）**；❌ **不把 `R̂` 放进 RL 梯度路径替代 judge（reward-hacking 雷区）**。
5. **三大胜负手**（评审确认）：
   - **动作敏感性**：predictor 必须真依赖 `a_t`，否则退化成 `E[z^o|state]`（终端反馈给定 state 多为低熵 → mean 预测即低 loss，SIGReg 管不了条件依赖）。→ 加 **counterfactual 动作对比项** + **Δ-sensitivity 硬门控** + 按反馈熵重加权（压低 boilerplate turn）。
   - **抗坍缩 ≠ 抗漂移**：SIGReg 只抗坍缩；在线 policy hidden 漂移需 **EMA/周期 re-anchor**（「无需 EMA」仅限离线 Stage A）。
   - **实现**：slime `--custom-loss-function-path` 只拿 logits；**中后层 hidden 不在末段 PP 阶段**（与「末段取 hidden」矛盾）→ 在线先用末层、中后层留给离线 probe（PP=1 规避）。

---

## 1. 背景与动机

### 1.1 terminal-RL 的四个结构性特征

| 特征 | 事实（来源） | 含义 |
| --- | --- | --- |
| **真实交互极贵且不可逆** | 每 rollout 起独立 Docker 跑真实命令；Terminal-Bench-RL 估全量 32×H100 ~£30–50k | latent 里给候选命令打分而**不真执行** = 收益最大处（**U2**） |
| **奖励稀疏 + 超长程** | 末端单测；Endless Terminals(2601.16443) 16-turn PPO，真实任务 50–100+ turn | value 头提供 per-turn 信用分配（U1）；但**长程折扣会淹没末端信号**（见 §4.5） |
| **观测巨大且噪声重** | 终端输出 = 长 stack trace/dump；上下文 32K | token 级预测困难；latent + `H_θ` 压缩契合 |
| **自带（部分）稠密监督** | Terminal-Bench-RL 奖励 = 65% 加权单测 + 35% LLM-judge | ⚠️ **下调声明**：judge 子项**未必严格 per-turn 稠密**，可能 trajectory/behavior 级、噪声大、风格偏置；视为**带噪辅助监督**，需验证（§9 judge-label validity） |

### 1.2 四种用途与边界（U2 升为头牌）

| 用途 | 形式 | 定位 | 可行性 |
| --- | --- | --- | --- |
| **U2 候选命令筛选（头牌）** | 同 state 下对 N 个候选命令在 latent 预测 `R̂/V̂` 排序，仅 top-k 真执行 | **核心贡献** | ✅ 但需**counterfactual ranking** 证据（预测 observed transition ≠ 会排序候选，见 §9） |
| **U1 dense shaping / 信用分配** | `V̂(z^s_t)` 每 turn 估进度 | 支撑 | ✅ 但须正面对比 ECHO + judge 稠密信号 |
| U3 imagination replay | 生成 imagined latent 轨迹 | **移出核心论文**（DREAM 扩展） | ⚠️ 长程误差；2412.14312 负结果 |
| U4 替代 verifier 出末端真值 | `R̂` 代替单测 | **禁止** | ❌ 依赖精确字节 |
| （新增禁区）`R̂` 进 RL 梯度替代 judge | 训练目标里用 `R̂` 省 judge | **禁止** | ❌ Goodhart/reward-hacking |

---

## 2. 文献定位与新颖性

### 2.1 谱系：latent dynamics 是否带 reward/value head

| 模型 | dynamics | reward head | value head | 编码器 | 抗坍缩 | 域 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuZero | ✅ | ✅ | ✅(+policy) | 学习 | 监督驱动 | 游戏 | 1911.08265 |
| TD-MPC2 | ✅ | ✅ | ✅(ens.Q) | decoder-free | SimNorm | 控制 | 2310.16828 |
| UniZero | ✅ | ✅ | ✅ | 学习 | 一致性 | 通用 | 2406.10667(待核验) |
| **LeWM** | ✅ | ❌ | ❌ | 端到端 ViT | **SIGReg(单超参)** | 像素 | 2603.19312 |
| TD-JEPA | ✅(policy-cond) | ❌(测试任意 reward) | 隐式 | JEPA | 理论非坍缩 | 控制 | 2510.00739 |
| **本方案** | ✅ | ✅(env reward) | ✅(progress) | **LLM hidden + SIGReg** | SIGReg+在线 EMA | **terminal agent** | — |

> **诚实定位**：仅「latent 预测 reward/value」**不是顶会级新意**（已范式化）。novelty 须落在 **policy-coupled latent 对 terminal 命令的 counterfactual 打分 + U2 校准筛选 + 保留 verifier**。

### 2.2 支撑性工作

| 工作 | arXiv | 支撑 |
| --- | --- | --- |
| LLM-JEPA | 2509.14252(待核验) | 语言侧 JEPA 目标，embedding 预测提升表征 |
| Sparse Reward Subsystem in LLMs | 2602.00986(待核验) | LLM hidden 内已有 value/TD-error 式神经元 → 支撑 hidden-conditioned 头 |
| Pearl | 2604.08065(待核验) | JEPA 式学工具 latent 后果、推理免显式调用（U2 概念近邻） |
| ELHSR / SWIFT | 代码 `aster2024/SWIFT` | LLM hidden/logits 上线性 reward/gating 头可行 |
| Agents Fail to Leverage WM | 2601.03905(待核验) | naive 接 WM 反而更差 → 论证需校准式准入（U2 门控） |

### 2.3 最近邻竞品与差异化

| 竞品 | arXiv | 做了什么 | 威胁 | 差异化（必须用实验证明，非口头） |
| --- | --- | --- | --- | --- |
| **ECHO**（已核实，MSR） | **2605.24517** | GRPO 加**环境观测 token CE**辅助损失，同 forward、零开销，TerminalBench-2.0 近翻倍，λ=0.05；环境预测损失单用可 verifier-free 自提升 | **最高（同域、更简、已发表）** | ECHO 是**纯 token-CE 塑形 policy 自身**，无独立可查询 predictor、无显式 value、**无法廉价做 U2**（要 imagine 候选输出需逐 token 生成，昂贵）。我们 = **独立 latent predictor + value 头 → 一次前向给候选命令打分**。**全文成败 = 证明 latent/hidden/value/U2 超越 ECHO** |
| **RWML** | 2602.05842(待核验) | 动作条件**文本** WM + sim-to-real embedding reward + 任务 RL | 高 | RWML 外部 embedding 预测语义 next-state；我们 policy-hidden 绑定 latent + U2，须证 hidden-conditioning 优于外部 encoder |
| **CWM** | 2510.02387(待核验) | 32B 代码模型，Python trace+Docker 轨迹 mid-train + RL | 高（生态） | CWM 是大规模 WM 预训练**基座**；我们是 policy-coupled latent critic 做 SETA+DAPO 信用分配 + U2 筛选 |
| **PriorZero**（已核实） | 2605.12289 | LLM 与环境间放 UniZero 式 WM（reward+分布式 value+latent consistency）做 credit assignment，交替更新 | 高（概念近） | 我们**不引入独立大 WM，condition 在 policy `H_θ`**；JEPA/SIGReg（reconstruction-free、单超参）而非 UniZero 生成/类别值；须证 hidden-coupled 的增量 |
| WebWorld / ProAct / ToolTree | 2602.14721 / 2602.05327 / 2603.12740(均待核验) | web 模拟器 lookahead / MC-Critic / 训练-free 工具树筛选 | 中-高（U2 近邻） | 域 + 机制（训练好的 hidden-latent WM vs 训练-free 搜索 / web / 蒸馏 critic） |

> **不可约 novelty（评审收敛）**：**(1)** policy-hidden latent 对 terminal 命令的 **counterfactual 打分 → U2 校准式执行前筛选**（ECHO/RWML 结构上做不到或昂贵）；**(2)** 用 surprise/选择性弃权做**校准准入**而非裸接 WM；**(3)** 保留 verifier 的 U4 边界。「dense supervision 有用」已被 ECHO 占据，**不能作为主 claim**。

---

## 3. 代码事实基础（来自 0616 调研，保留）

### 3.1 `le-wm` 数据流

| 模块 | 位置 | 输入→输出 | 作用 |
| --- | --- | --- | --- |
| obs encoder | `jepa.py:29-40` | `pixels(B,T,..)`→`emb(B,T,D)` | ViT 取 CLS+projector |
| action encoder | `jepa.py:42-43`,`module.py:189-214` | `action`→`act_emb` | 映射到 state latent 维度 |
| AR predictor | `module.py:244-285` | `x=(B,T,D)`,`c=(B,T,D)`→`(B,T,D)` | Transformer+AdaLN，`(z_t,a_t)`→未来 latent |
| loss | `train.py:17-42` | → `pred_loss+0.09*sigreg` | 同训 encoder/projector/predictor/action |
| SIGReg | `module.py:10-36` | `emb.T(0,1)`→标量 | 随机投影 Epps-Pulley 正态检验 |

核心：$\hat z_{2:T+1}=P(z_{1:T},e^a_{1:T})$；$\mathcal{L}=\|\hat z-z_{tgt}\|^2+\lambda_{sig}\mathcal{L}_{sigreg}(z),\ \lambda_{sig}=0.09$。le-wm 原码对 target **未 detach**（在线须改 sg/EMA）；SIGReg 作用在 encoder `emb`，**不在 predictor 输出**。公开代码 `github.com/lucas-maes/le-wm`。

### 3.2 SETA+DAPO baseline（关键事实）

- wrapper `..._seta_dapo_nodynamic_pu.sh:21-33`：`ROLLOUT_BATCH_SIZE=8`、`N_SAMPLES=8`、`MAX_TURN=10`。
- DAPO（`...:1242-1254`）：`grpo`、`eps-clip 0.2/high 0.28`、`--calculate-per-token-loss`。
- `_build_samples()`（`generate.py:2384-2488`）：`s.tokens=input_ids+output_token_ids`（**仅 $h_t+a_t$，不含反馈**）。
- 后端 **Megatron**；`model.py:342-418` 只返回 logits、不暴露 hidden；`loss.py:575-715` PPO-clipped policy loss。**DAPO/GRPO 是有意 critic-free**（§4.5 重新引入 value 须论证净收益）。

### 3.3 transition 边界

$h_t$=turn 前 `context_messages`；$a_t$=`assistant_output`/`tool_calls`；$o_{t+1}$=`tool_calls[*].result`+末端 evaluate。`Sample.tokens` 不含 $o_{t+1}$ → **预测任务天然无泄漏**（强项）。

---

## 4. 核心设计：LLM-hidden-conditioned LeWM + reward/value 双头

### 4.1 总体架构

```
turn sample: input_ids=h_t ; output_token_ids=a_t ; feedback_text=o_{t+1} ; next_input_ids=h_{t+1}(opt)

LLM branch (detached/frozen H_θ):
  h_t       -> hidden[layer] @prompt span -> pool -> A_h -> C -> z^s_t
  h_t+a_t   -> hidden[layer] @action span -> (multi-vec) A_a -> e^a_t      # 见 §4.2 action bandwidth
  h_{t+1}   -> hidden @next prompt span    -> pool -> A_h -> C -> z^s_{t+1} (predictor 输入/value 输入, 非主 target)
Feedback branch (frozen anchor T_fix):
  o_{t+1}   -> T_fix -> A_o -> C -> z^o_{t+1}   (MAIN target / 锚)
Predict: ARPredictor(z^s_t, e^a_t) -> ẑ^o_{t+1}      # 预测“环境响应”，主 target=z^o
Heads:   R̂_ψ(ẑ^o_{t+1})->reward ;  V̂_ψ(z^s_t)->return（训练）；U2 排序用 V̂_ψ(ẑ^o_{t+1})（候选相关，§4.7）
Train:   logits->原 DAPO loss(数值不变) ; z/pred/heads->auxiliary（小权重）
```

### 4.2 token → world latent 编码机制

- **(a) 粒度 + 数据布局（评审修正歧义）**：turn-level `z`，`ARPredictor` 的 **T=turns**。**明确选单步还是 AR-over-turns**：
  - 首版用 **单步 `(z^s_t,e^a_t)→ẑ^o_{t+1}`**（与 replay buffer 的独立 transition 采样、value TD 一致）；
  - AR-over-turns 作扩展，需 turn 维 padding + causal mask（且 SIGReg batch 要跨轨迹打散，见 §4.6）。
- **(b) 取哪层**：中后层（~60–80%）语义更丰富。**但在线 Megatron 取中后层 ≠ 末段 PP（见 §7 矛盾）** → **在线首版用末层 hidden；中后层仅离线 probe（PP=1）**。超参 `--world-model-hidden-layer`。
- **(c) 池化**：probe 用 last-token；主模型用 **learned-query attention pooling**。**action bandwidth（评审）**：终端 action 是长多工具 turn，pool 成单个 `e^a_t` 极损、直接削弱动作敏感性 → **用多向量 / 对 action token 的 cross-attention 条件**，而非单向量。超参 `--world-model-pool`。
- **(d) 各向异性/离群维**：cone effect(1907.12009) + massive activations(2402.17762) 会让裸 projector 被离群维主导。→ `A_source` 前 **先 winsorize/clip 离群维，再 per-dim standardize**；**ZCA 白化为最后手段**（求逆协方差对重尾不稳）。**标准化统计须用固定参考集**（勿随漂移 policy 每 batch 重算）。

### 4.3 跨源对齐（抗坍缩 + 抗漂移，评审强化）

- 进 latent 且需对齐：`z^s_t,z^s_{t+1},z^o_{t+1}`；`e^a_t` 是 AdaLN 条件、不对齐。
- **三个力**：Anchor（frozen `T_fix` 的 `z^o` 做 sg target）+ Pull（预测/对齐 loss）+ Spread（SIGReg）。
- **源头降 gap**：首版 `T_fix`=**frozen Qwen3 快照**（同几何）；ablation 换 code-aware encoder。
- **抗坍缩 vs 抗漂移（关键修正）**：SIGReg-无-EMA **仅离线 Stage A 成立**；**在线** policy hidden 漂移 → `A_h` 把移动输入回归进（Phase-1 冻结的）`C` 空间，SIGReg 管不了对齐漂移。→ 在线**对 `C`/标准化统计加慢速 EMA / 周期 re-anchor**，`A_h` 受控刷新；**按训练步监控 CKA/校准**，U2/Stage-C 仅在**近期**校准达标时启用。
- **解耦跨源静态偏移**：`ẑ^o=z^s_t⊕Δ̂` 预设两源同尺度，否则 `Δ̂` 要吸收一个**漂移的常数源偏移** → 要么**直接预测 `z^o`（不做 delta）**，要么**显式学一个单独正则的 source-offset 项**让 `Δ̂` 只载动态。
- **两阶段（推荐）**：Phase-0 仅 `o` 分支定义 `C`；Phase-1 冻结/EMA `A_o,C`，训 `A_h,A_a,P,heads`。⚠️ 锚在「反馈流形」（多 boilerplate、低维）可能**欠服务更丰富的 state** → 验证 `z^s` 所需 rank 落在 `C` 像内，必要时锚在并集。

### 4.4 predictor 与动作敏感性（核心修正）

> **评审一致认定的致命点**：预测 `z^o`+delta **并不能从根上阻断坍缩**。两个低 loss 退化解：①**mean 预测** `E[z^o_{t+1}|z^s_t]`，无视 `e^a_t`（终端反馈给定 state 多低熵 → mean 即低 loss，SIGReg 只约束边际、管不了对 `a_t` 的条件依赖）；②**低 surprise turn 上 trivial-copy**（`ẑ^o≈z^s_t` 近最优；若用 delta 参数化即 `Δ̂≈0`，而这类 turn 占多数）。**两者都对 U1/U2 致命**（U2 本质=按动作排序候选；predictor 若不依赖动作，排序即噪声）。

**修正方案（必做）**：
1. **counterfactual 动作对比项**：`ẑ^o(z^s_t,a_t)` 须比 `ẑ^o(z^s_t,a'_t)`（同 state 的其他动作，in-batch 负例/margin）更接近真 `z^o_{t+1}` → 强制 $I(\hat z^o_{t+1};a_t\mid z^s_t)>0$（首版**直接预测 `z^o`、不做 delta**，与 §4.3/§4.6 一致）。
2. **Δ-sensitivity 度量 + 硬门控**：报告 `ẑ^o` 在「动作置换」vs「state 置换」下的方差比；**「动作效应 ≫ 0（held-out）」设为 Stage-A 硬门控**，不止 MSE/retrieval。
3. **按反馈熵重加权 $w(H(o_{t+1}))$（移入 §4.6 目标）**：对低熵 boilerplate turn 降权，**且用同一权重 gate `L_action-contrast`**——低熵 turn 上真 `z^o` 本就与动作无关，强加 margin 会**注入伪动作依赖**；同时避免 loss 被「copy 即赢」的多数样本主导。

### 4.5 reward / value 双头（评审修正）

```
# 修正 t/t+1 约定一致：
reward head:  R̂_ψ(ẑ^o_{t+1}) → 预测 r^env_{t+1}（per-turn judge 稠密分，视为带噪辅助）
value head:   V̂_ψ(z^s_t)      → 预测 progress（见下，非 success prob）
TD target:    y_t = r^env_{t+1} + γ(1-d_t) sg(V̂_ψ(z^s_{t+1}))
```

- **分离两类信号（关键修正）**：**per-turn judge reward** 与 **末端单测 return** 用**独立头/目标**，**不要相加成单一 `R^env`**——否则 50–100 turn 下 γ^k 会湮灭早期 turn 的末端信号（γ=0.95→γ^100≈0.006）。
- **长程 return 用 TD(λ)/n-step 或 horizon-normalized/average-reward**，显式声明 γ 与有效 horizon。
- **value 范围降级为 "progress shaping"**，非「成功概率」：`z^s_t` 是 belief latent、丢决定成败的字节 → 末端成功有大 aleatoric 方差；显式声明并 bound value error。
- **重新引入 critic 的论证**：DAPO/GRPO 有意 critic-free；Stage-C 把（有偏的）belief-value 注入 advantage 须实证**净降方差而非注入偏置**。
- **`R̂` 严格不进 policy 梯度路径（禁区，与 `V̂` 区分）**：`R̂` 只用于**推理期 U2 + U1 诊断**，**绝不作为被优化的目标**（Goodhart）；若省 judge 成本，仅在 **surprise/OOD 低**时用 `R̂` 且**周期真 judge 审计/锚定**；删除原「在线替代 judge」表述。
- **`V̂` 仅经 Stage-C 门控 control-variate 进梯度**（§6.3，`η=0` 默认）：这是与 `R̂`-hacking **不同的风险**——`V̂` 作有偏 baseline 须实证净降方差（§6.3），并非完全禁入。
- ⚠️ reward 头的 per-turn judge 目标**预设 §9.7 judge-label validity 通过**；若 judge 实为 trajectory 级/风格偏置，reward 头退化为弱辅助，则主用 value + 末端单测。

### 4.6 Loss 设计（评审修正版）

> 几何二选一（不混）：**①MSE+SIGReg（高斯，主线）** 或 **②L2-norm+cosine/InfoNCE-or-VICReg（球面）**。

主损失（方案①，**已删冗余/有害的 term 2**）：

$$\mathcal{L}=\underbrace{\mathcal{L}_{DAPO}}_{\text{数值不变}}+\alpha(t)\Big[\underbrace{\lambda_o\|\hat z^o_{t+1}-\mathrm{sg}(z^o_{t+1})\|^2}_{\text{预测反馈(主)}}+\underbrace{\lambda_{cf}\,w(H(o_{t+1}))\,\mathcal{L}_{\text{action-contrast}}}_{\text{动作敏感(必加,§4.4;熵门控)}}+\underbrace{\lambda_a\mathbf{1}_{has\_next}\|z^s_{t+1}-\mathrm{sg}(z^o_{t+1})\|^2}_{\text{对齐}}+\lambda_{sig}\mathrm{SIGReg}(z^s)\Big]+\lambda_v^{*}\mathrm{Huber}(\hat V_t,y_t)$$

评审修正点：
- **删原 term 2**（`λ_s‖ẑ^o−sg(z^s_{t+1})‖`）：与 term 3 冲突/冗余，且 `z^s_{t+1}≈z^s_t` 使其**奖励 copy 捷径**。next-state 监督已通过「`z^s_{t+1}` 作 predictor/value 输入」免费获得。
- **SIGReg 只对 `z^s`**：Phase-1 下 `A_o,C,T_fix` 冻结，对 `z^o` 的 SIGReg 是 no-op（明示）。
- **value 项移出 α(t) 全局 warmup**（记 `λ_v^*` 独立 LR/权重）：value 头用 detached hidden、不扰动 policy，不应被表征 warmup 拖到无法达标。
- **SIGReg batch 组成**：`8×8` 且 within-trajectory 强相关 → Epps-Pulley 投影 CDF 估计有偏 → **跨轨迹打散 / 设最小有效 batch**。
- **反馈熵重加权 $w(H(o_{t+1}))$（§4.4 必需，已入目标且专门 gate `L_action-contrast`）**：动作无关的低熵 turn 上强加对比 margin 会注入伪动作依赖，故按反馈熵降权。

默认超参：`WORLD_MODEL_ENABLED=0`(默认) / `DETACH_LLM_HIDDEN=1` / `α(t):0→0.05`（对齐 ECHO 的 λ=0.05 经验）/ `λ_o=1` / `λ_cf=0.5` / `λ_a=0.1` / `λ_sig=0.03–0.09` / `λ_v^*` 独立 `0.1` / `VALUE_TO_ADVANTAGE η=0`(首版)。

### 4.7 推理：U2 筛选 + 校准门控（下调声明）

- **U2 latent MPC**：同 `z^s_t` 下对候选 `{a^{(i)}}` 算 `ẑ^o_{(i)}=P(z^s_t,e^{a(i)})`，按 **`R̂(ẑ^o_{(i)})` 与 `V̂(ẑ^o_{(i)})`** 排序（**value 作用在预测的 next-state 上**——`V̂(z^s_t)` 对候选恒定、无法排序），仅 top-k 真执行；`ẑ^o↔z^s` 可比性由 §4.6 对齐项保证。
- **surprise 门控**：`surprise=‖ẑ^o−z^o‖²` 按**拟合的 held-out 协方差**标准化（**不假设 Σ=I**）；低置信→回退真执行。
- ⚠️ **下调声明**：**SIGReg 各向同性 ≠ 认识论校准**。surprise 是否可信须由 **ECE / 选择性风险(risk-coverage) / AUROC** 实测，不能因 latent 高斯化就声称「已校准」。

---

## 5. 训练数据：复用已存储 replay buffer

| 维度 | 说明 |
| --- | --- |
| 现成字段 | `input_ids`($h_t$)、`output_token_ids`($a_t$)、`log_probs`、`advantages`、`rewards`、turn/traj 结构 |
| 优势 | 已结构化、token 边界对齐；可放大全量，Stage A 首选 |
| hidden 来源 | buffer 不存 hidden；**frozen LLM 单快照离线重算**（PP=1 规避 §7 难题）。**这是优点**：单快照给 off-policy 数据**一致几何**，非 drift 负担——真正的 drift 是**在线部署**时快照≠现策略 |

**两个注意**：
1. **分布偏置（关键）**：SPEAR-SIL buffer 仅正优势/成功轨迹 → WM 只见成功、reward/value 在差动作上无法校准、U2 偏乐观。→ **WM 训练用全量 rollout（含失败）**；只有 SIL 时按比例混入失败/低分轨迹。
2. **turn 链接**：state target 需 `(h_t,h_{t+1})` 配对，buffer 须保留 traj→turn 顺序；若 flatten 则退化为仅 feedback target。
3. ⚠️ **「tokens 确定性映射」过强**：bf16+FlashAttention+packing 仅近似可复现，probe 够用，**勿依赖 bitwise 确定性**。

---

## 6. 训练阶段（A→B→C）

### 6.1 Stage A：offline probe（门控加严，评审）
从 replay buffer（含失败）抽 transition → frozen LLM 取 hidden → frozen `T_fix` 编码 $o_{t+1}$ → 训 `A_h/A_a/A_o/C/P/heads`。
**进入 online 的硬门控（全部在标准化/白化后的 latent 上测）**：
- **动作敏感性（新增、最重要）**：`ẑ^o` 动作置换方差 ≫ state 置换方差；shuffled-action predictor 明显劣于真 action。
- value Spearman **≥0.4–0.5（per-turn within-trajectory，非跨轨迹池化）**。
- cross-source CKA **≥0.3–0.5 且显著优于 shuffled-label 控制**。
- positive vs negative cosine gap（**标准化后**，避免 cone effect 虚高）。
- effective rank **> k%·D**（定量下限），不坍塌。

### 6.2 Stage B：online auxiliary-only
`WORLD_MODEL_ENABLED=1`、`DETACH_LLM_HIDDEN=1`、`η=0`、`α(t):0→0.05`。对照 baseline vs `..._worldmodel_aux_pu.sh`。观察主指标（`test_acc`/`reward/raw`/`pg_loss`/`ppo_kl`/`entropy`）**不显著降**；`wm/*` 改善；**CKA/校准按步监控**（抗漂移）。

### 6.3 Stage C：value-assisted advantage（小权重、可选）
仅 Stage B 成立且**近期**校准达标：$A^{hybrid}=A^{DAPO}+\eta\,\mathrm{GroupNorm}(y-\hat V)$，`η=0.02` 起、warmup、**仅 non-final turn**（防末端 label 泄漏）；实证**降方差不注入偏置**才保留。

---

## 7. 实现接入点（slime/Megatron）

| 文件 | 修改 | 理由 |
| --- | --- | --- |
| `terminal-rl/generate.py` | turn record/`Sample` 存 `feedback_text`、`next_input_ids`、`prompt_len`、`response_len`、`raw_score/base_score/status` | train_data 需结构化字段 |
| `slime/ray/rollout.py` | `_convert_samples_to_train_data()` 增 `wm_*` 字段 | iterator 只消费显式字段 |
| `slime/backends/megatron_utils/data.py` | `get_batch()` keys 增 WM 字段 | 走 Megatron |
| `slime/backends/megatron_utils/model.py` | **取 hidden**：在线优先末层（`output_processor`/末段 PP forward hook）；**中后层须在其所属 PP rank 上 hook 并跨 stage 路由**（见下） | 现只返回 logits |
| `model_provider.py` | **WM 头在 `get_megatron_optimizer` 之前注册**（参考 critic `LinearForLastLayer`），保留 policy LM 头 | optimizer 后加参数不被优化 |
| `loss.py` | `--custom-loss-function-path` 包装：先调原 DAPO loss + `λ_wm·wm_loss` | 主 loss 不变 |
| `arguments.py` | 加 `--world-model-*`（`hidden-layer`/`pool`/`lambda-*`） | ablation |
| （新增）`scripts/wm_build_dataset.py` | buffer→frozen LLM(PP=1) 重算 hidden+池化→frozen `T_fix` 编码→落盘 `wm_*` | Stage A 离线、零侵入、**de-risk 全部分布式难题** |
| `..._worldmodel_pu.sh` | 新建脚本 | 对照 |

**实现 gotchas（评审修正）**：
- **§4.2(b) 矛盾的解法**：中后层 hidden **不在末段 PP** → 在线要么 (a) **承诺用末层 hidden**（中后层降级到离线 probe），要么 (b) 在**所属 PP rank** hook 中间层并显式设计**跨 PP 激活路由 + masking**（Megatron 只在 stage 边界传激活，这是真·跨 stage 通信，非「加 hook」）。
- **CP**：单 turn 的 span 可能被 CP rank 切分 → span 池化是 **CP group 内带 mask 的 all-reduce**，不止「验证顺序」。
- **TP/SP**：SP 下须 `gather_from_sequence_parallel_region` 再全局池化；`thd`/`packed_seq_params` 偏移须复用 logprob/value 同款逻辑（**正确点，保留**）。
- **checkpoint**：base Qwen ckpt 无 WM 头 key，bootstrap 处理 missing-key（`sharded_state_dict`+宽松 strict）。
- **离线 vs 在线**：离线最安全（PP=1、一致几何），约 2× 前向；在线省算力但内存敏感（**只池化 span 边界向量**）。

**开源可复用**：le-wm `lucas-maes/le-wm`、TD-JEPA `facebookresearch/td_jepa`、ELHSR `aster2024/SWIFT`、ArCHer `YifeiZhou02/ArCHer`、Terminal-Bench-RL `Danau5tin/terminal-bench-rl`、PriorZero `opendilab/LightZero`(PR 待确认)、RLVR-World `thuml/RLVR-World`。**未发现** policy-hidden JEPA latent WM for terminal-agent 的现成实现。

---

## 8. 最小可发表版本（MVP，评审强烈建议）

**砍掉**：U3 imagination replay、DREAM 集成、Stage-C 在线 advantage 修改（降为附录）、「替代 judge」表述、ALFWorld/WebShop 大扩展（仅作必要 baseline）、「SIGReg surprise 天然校准」声明。

**保留为一篇窄而硬的论文**：
- 方法：policy-hidden-conditioned latent 模型，预测 next-feedback latent + value/return。
- 用例：**浅层执行前命令筛选（U2）**。
- 核心 claim：**hidden-conditioned latent 打分，在「每次真实命令执行的成功率」上超越 ECHO 式稠密监督与简单 value probe**。
- baseline：GRPO/DAPO、**ECHO（主）**、ECHO+value probe、RWML 式 embedding target、（可行则）PriorZero/UniZero 式外部 WM、LLM 预评/ToolTree 式筛选。
- 证据：counterfactual 命令排序、在线 top-k 筛选成本-成功曲线、校准/选择性弃权、等算力样本效率。
> 若 U2 成立 → 强应用/agentic-RL 论文；若仅 U1 成立 → 大概率 workshop（ECHO 已占主线）。

---

## 9. 实验设计（围绕「证明动作敏感 + U2 超越 ECHO」）

**Baselines**：GRPO/DAPO ｜ **ECHO(2605.24517, 主)** ｜ ECHO+value/critic ｜ RWML 式 embedding target ｜ PriorZero/UniZero 式外部 WM（可行则）｜ LLM 预评/ToolTree 式筛选 ｜ SPEAR。

**必做实验（评审 missing-experiments）**：
1. **latent vs token-level**：GRPO ｜ ECHO token-CE ｜ latent feedback 预测(无头) ｜ latent+reward/value ｜ ECHO+reward/value。**等 rollout 预算/前向/aux 权重**。
2. **dynamics 是否真起作用（核心）**：value-only-on-current-hidden ｜ reward/value-on-predicted-latent ｜ **shuffled-action predictor** ｜ no-action predictor。**若 shuffled-action 与真 action 相当 → 「world model」claim 崩塌**。
3. **hidden-conditioning vs 外部 encoder**：policy hidden pooling ｜ frozen 文本 encoder over state/action ｜ RWML 式外部 embedding ｜ 单独小可训 encoder。
4. **U2 counterfactual ranking**：每 state 采 N 候选命令，**全部沙箱执行得标签**，测 top-1/top-k 排序 vs LM logprob / LLM 预评 / ECHO-as-generator / 随机 / ToolTree。
5. **在线筛选收益**：成功率 vs 真实执行数 / Docker 成本 / wall-clock / 不可逆-错误命令率 / 弃权覆盖；**top-k 必须改善成本调整后成功率**。
6. **校准**：ECE、risk-coverage、surprise/error 的 AUROC、OOD task split。
7. **judge-label validity**：分离末端单测 reward / judge 行为分 / 最终成功，证明模型**非仅学 judge 风格合规**。

**指标**：样本效率（到 X% SR 的真实执行数/rollout/wall-clock）；动作敏感性方差比；预测质量；对齐健康（CKA/retrieval/effective-rank/标准化 cosine gap）；校准（ECE/AUROC）；U2 成本-成功曲线。

---

## 10. 风险与缓解

| 风险 | 表现 | 缓解 |
| --- | --- | --- |
| **动作不敏感退化** | predictor 出 state→mean，U2 排序=噪声 | counterfactual 对比项 + Δ-sensitivity 硬门控 + 熵重加权（§4.4） |
| **在线漂移**（≠坍缩） | 离线达标、在线 calib 退化 | EMA/周期 re-anchor `C`；按步监控 CKA/校准；U2 仅近期达标时启用 |
| 长程折扣淹没末端信号 | 早期 turn value 无意义 | 分离 per-turn judge / 末端 return 两头；TD(λ)/horizon-norm（§4.5） |
| **`R̂` 进梯度 → reward hacking** | policy 迁移到 `R̂` 高估区 | `R̂/V̂` 仅推理/诊断；surprise 门控+真 judge 审计（§4.5） |
| 末端依赖精确字节 | latent 出不了末端真值 | **U4 禁区**，真值跑单测 |
| 跨源/坍缩/trivial-copy | 不对齐/低秩/copy | 锚+SIGReg+两阶段；主 target=z^o；删 term 2 |
| ECHO 撞车 | novelty 被「稠密监督有用」覆盖 | 打 U2+counterfactual+校准；ECHO 为主 baseline；实验 1/2/4 |
| §7 PP/层 + CP 路由 | 在线中后层提取被低估 | 在线用末层；中后层离线 probe；CP all-reduce 池化 |
| Stage-A 门控过松 | 假阳进 online | 加严阈值 + shuffled 控制 + 标准化后测（§6.1） |
| arXiv ID 未核 | 引用风险 | ECHO/PriorZero 已核实；其余 §11 标待核验 |

---

## 11. 可行性结论与下一步

**结论**：可行，但**顶会价值取决于 U2 能否跑赢 ECHO**。le-wm 归纳偏置可迁移；transition 边界无泄漏；replay buffer 提供离线数据；`ARPredictor/SIGReg` 可复用。**最大科学风险 = 动作敏感性**（§4.4）；**最大工程风险 = 在线漂移 + Megatron 中后层提取**（§4.3/§7）；**最大 novelty 风险 = ECHO**（§2.3）。

**下一步**：
1. `scripts/wm_build_dataset.py` + offline probe（含失败轨迹），跑通 §6.1 **全部硬门控（尤其动作敏感性 + shuffled-action 对照）**。
2. 达标 → Stage B online auxiliary-only（detach、`T_fix` frozen、`α→0.05`、DAPO 不变、按步监控漂移）。
3. **优先做 U2 counterfactual ranking 实验（实验 4）**——这是 novelty 的命门，应早做以决定是否继续。
4. value 头近期校准达标后，才 `η=0.02` 进 hybrid advantage。
5. 论文：ECHO 为主 baseline，补实验 1/2/3/4/5。

---

## 12. 参考文献（已核实者去标记；其余「待核验」投稿前逐条核对）

**已核实**：ECHO **2605.24517**（MSR，terminal env-token CE，TerminalBench-2.0 近翻倍，λ=0.05）｜PriorZero **2605.12289**｜LeWM **2603.19312**｜TD-JEPA **2510.00739**｜SPEAR **2509.22601**｜TD-MPC2 **2310.16828**｜MuZero **1911.08265**｜Massive Activations **2402.17762**｜cone effect **1907.12009**｜Endless Terminals **2601.16443**｜Stealing-That-Free-Lunch **2412.14312**。

**待核验**：UniZero 2406.10667｜DreamerV3 2301.04104｜WIMLE 2602.14351｜AgentPRM(Xi) 2511.08325｜ArCHer 2402.19446｜VAGEN 2510.16907｜PaW 2606.02388｜RWML 2602.05842｜CWM 2510.02387｜WebWorld 2602.14721｜ProAct 2602.05327｜ToolTree 2603.12740｜Imagine-then-Plan 2601.08955｜Sparse-Reward-Subsystem 2602.00986｜LLM-JEPA 2509.14252｜Pearl 2604.08065｜Agents-Fail-to-Leverage-WM 2601.03905。

**开源代码**：le-wm `lucas-maes/le-wm`｜TD-JEPA `facebookresearch/td_jepa`｜ELHSR `aster2024/SWIFT`｜ArCHer `YifeiZhou02/ArCHer`｜Terminal-Bench-RL `Danau5tin/terminal-bench-rl`｜RLVR-World `thuml/RLVR-World`｜PriorZero→`opendilab/LightZero`(PR 待确认)。

---

> **评审整合说明**：v2 已整合技术审阅（Opus）与新颖性红队（GPT-5.5）的两轮反馈，核心修正：①§4.4 动作敏感性退化解 + counterfactual 对比项；②§4.6 删冗余 term 2、value 解耦 warmup、SIGReg batch；③§4.5 分离 judge/末端两头、折扣修正、`R̂` 出梯度路径；④§4.3 抗漂移 EMA/re-anchor；⑤§7 PP/层矛盾；⑥novelty 重定位到 U2、ECHO 为主 baseline；⑦§9 围绕 shuffled-action + U2 counterfactual 的关键实验；⑧§8 MVP。**ECHO/PriorZero 已联网核实**；其余 2025-2026 ID 投稿前须逐条核对。
