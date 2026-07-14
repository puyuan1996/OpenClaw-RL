# DIVE-PO：Policy Optimization with Dual-stream Intrinsic adVantage Exploration

**基于双流内在优势探索的策略优化**

## 摘要

带有可验证奖励的强化学习（RLVR）已经成为提升语言模型智能体工具使用、代码执行和数学推理能力的重要范式。然而，在 group-based policy optimization 中，稀疏的任务奖励常使同一 prompt 下的多条 rollout 过早坍塌到相似轨迹；直接把探索 bonus 加到任务分数又会污染 verifier reward，使训练信号更难解释。本文提出 **DIVE-PO**，一种面向 RLVR 的双流内在优势探索方法。DIVE-PO 的核心主张是：探索信号应作为独立归一化的 intrinsic advantage 流注入策略优化，而不是作为 score-space bonus 直接修改任务奖励。具体而言，DIVE-PO 将局内 SimHash-KNN 新颖性与局间 hierarchical decayed count 组合为轨迹级内在信号，在 prompt group 内独立归一化，并通过 outcome-aware gate 与 Agent57-style beta arm 控制其进入 DAPO clipped objective 的强度。DIVE-PO 采用固定 rollout group 以稳定 arm 分配；UCB 控制器自适应选择不同 beta arm，并配合轻量温度阶梯增加采样侧差异。我们计划在 SETA 终端智能体任务、公开终端/软件工程 benchmark，以及 MATH、AMC23、AIME24、AIME25 数学 RLVR 任务上验证 DIVE-PO。[TODO: 填入主实验与消融结果]

## 1 引言

语言模型智能体在终端环境中需要执行连续决策：读取文件、运行命令、解析反馈、修复错误，并在有限 turn budget 内完成目标。这类任务的奖励通常稀疏且延迟，失败轨迹高度相似，标准 RLVR 容易在训练早期收敛到少数高频动作模式。数学 RLVR 中也存在类似问题：模型可能快速强化少数熟悉推理模板，导致采样多样性下降，pass@k 提升受限。

探索奖励是缓解这一问题的自然工具，但语言智能体中的探索设计存在三个困难。第一，状态不是低维环境变量，而是由自然语言、工具调用、观察文本、退出状态和任务上下文共同构成。第二，低质量的“新颖”行为很容易被奖励，例如无意义命令、格式扰动或超长输出。第三，若把内在奖励直接加到 scalar task reward，verifier reward 的语义会被混合，组内归一化后的优势可能同时反映任务质量和探索启发式，难以定位改进来源。

本文的核心 claim 是：

> 在 group-based RLVR 中，探索信号应作为独立归一化的 advantage stream 注入策略优化，而非直接注入 score space。

这一 claim 形成了 DIVE-PO 的主线。DIVE-PO 保持任务奖励本身不变，先计算 task advantage；再从轨迹新颖性计算 intrinsic signal 并独立做组内归一化；最后将受质量门控和 beta arm 缩放的 intrinsic advantage 加入 DAPO 的 clipped surrogate objective。SimHash-KNN、lifelong count、UCB arm 和温度阶梯都是服务于这一主线的实现机制，而不是独立贡献的简单堆叠。

本文的贡献如下：

- 提出 DIVE-PO，将内在探索信号作为独立归一化的 intrinsic advantage 注入 DAPO，避免直接污染 task reward。
- 给出面向终端智能体的轻量新颖性估计：局内 SimHash-KNN episodic novelty 与局间 hierarchical decayed lifelong novelty。
- 将 Agent57-style beta arm 作用于 post-normalized intrinsic advantage，使不同 arm 对应不同探索强度，并用 UCB 自适应分配探索预算。
- 明确讨论默认实现中的 correctness 边界，包括温度采样的 off-policy 偏差、混 arm 组内归一化、轨迹级 credit assignment 和 novelty hacking 风险。
- 给出面向 ICLR 投稿的实验、消融和分析计划，覆盖 SETA、公开 agent benchmark、MATH、AMC23、AIME24 和 AIME25。

## 2 预备知识与问题设定

### 2.1 Group-based RLVR

给定 prompt $q$，行为策略生成 $G$ 条候选轨迹：

$$
\{\tau_i\}_{i=1}^{G}\sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid q).
$$

任务 verifier 返回轨迹级任务奖励：

$$
R_i^{task}=R(q,\tau_i).
$$

在 GRPO/DAPO 类方法中，同一 prompt group 内的奖励被转换为相对优势：

$$
A_i^{task}=\operatorname{Norm}_G(R_i^{task}),
$$

其中本文使用

$$
\operatorname{Norm}_G(x_i)=
\frac{x_i-\frac{1}{G}\sum_{j=1}^{G}x_j}
{\sqrt{\frac{1}{G-1}\sum_{j=1}^{G}(x_j-\bar x)^2}+\epsilon}.
$$

本文设置启用标准差归一化，$\epsilon=10^{-6}$。若组内样本数不足或方差为零，优势退化为零或中心化值。

### 2.2 DAPO Objective

DIVE-PO 建立在 DAPO 上。设 $\rho_{i,t}$ 为 token 级 importance ratio：

$$
\rho_{i,t}=
\exp\left(
\log\pi_{\theta}(a_{i,t}\mid h_{i,t})
-
\log\pi_{\theta_{\mathrm{old}}}(a_{i,t}\mid h_{i,t})
\right).
$$

DAPO 使用非对称 clipping：

$$
\rho_{i,t}^{clip}=
\operatorname{clip}(\rho_{i,t},1-\epsilon_{low},1+\epsilon_{high}),
$$

本文默认设置为

$$
\epsilon_{low}=0.2,\qquad \epsilon_{high}=0.28.
$$

令 $\hat A_i$ 表示 DIVE-PO 后处理后的轨迹级训练优势。该优势被广播到轨迹中所有参与 loss 的生成 token。策略目标可写为：

$$
\mathcal{J}(\theta)=
\mathbb{E}_{q,\{\tau_i\}}
\left[
\sum_{i=1}^{G}\sum_{t\in\tau_i}
\min\left(
\rho_{i,t}\hat A_i,\,
\rho_{i,t}^{clip}\hat A_i
\right)
\right].
$$

本文训练使用 token-level loss，因此长轨迹会贡献更多 token 级项。这是实现事实，也是后文 limitations 中讨论的 credit assignment 风险。

### 2.3 为什么不使用 Score-space Bonus

一种直接做法是把内在奖励 $I_i$ 加到任务奖励：

$$
\tilde R_i = R_i^{task}+\alpha I_i.
$$

该做法的问题是，$\operatorname{Norm}_G(\tilde R_i)$ 同时混合了任务质量和探索启发式。若 $I_i$ 奖励了低质量但表面新颖的轨迹，模型可能学习到 reward hacking 行为；若 $I_i$ 尺度随训练阶段变化，任务 reward 的可解释性也会下降。DIVE-PO 因此保留

$$
R_i^{score}=R_i^{task},
$$

并只在 advantage 空间注入探索信号。

## 3 方法

### 3.1 方法总览

图 1 展示 DIVE-PO 的数据流。

**图 1：[TODO] DIVE-PO 方法总览。** Rollout group 先经过 verifier 得到 task reward；同时从轨迹中抽取 action states，计算 episodic novelty 和 lifelong novelty；二者组合成 intrinsic signal 并在 group 内独立归一化；quality gate 与 beta arm weight 缩放 intrinsic advantage；最终与 task advantage 和 truncation penalty 相加，进入 DAPO clipped objective。

DIVE-PO 的训练优势为：

$$
\hat A_i
=
A_i^{task}
+
B_i^{int}
+
P_i^{trunc}.
$$

其中 $A_i^{task}$ 来自 verifier reward 的组内归一化，$B_i^{int}$ 是独立归一化后的 intrinsic advantage bonus，$P_i^{trunc}$ 是 outcome-aware truncation penalty。本文默认不启用 CDE actor PPL bonus，不启用 multi-head critic bonus，也不把 Agent57/NGU bonus 写入 score space。

### 3.2 Dual-stream Advantage Fusion

设 $I_i$ 为轨迹级内在信号。DIVE-PO 先独立归一化：

$$
A_i^{int}=\operatorname{Norm}_G(I_i).
$$

然后计算质量门控 $q_i$ 和 arm weight $w_i$，得到 intrinsic bonus：

$$
B_i^{int}
=
\operatorname{clip}_{[-c,c]}
\left(
\lambda\, w_i\, q_i\, A_i^{int}
\right).
$$

默认超参为：

$$
\lambda=0.08,\qquad c=0.35.
$$

#### Proposition 1：未门控双流的均值保持

若 $B_i^{int}=\lambda A_i^{int}$，且 $\operatorname{Norm}_G$ 含组内去均值，则

$$
\frac{1}{G}\sum_{i=1}^{G}B_i^{int}=0.
$$

因此，未门控的 intrinsic stream 不改变同一 prompt group 的平均 advantage，只改变组内相对排序。

**证明。** 由 $\operatorname{Norm}_G$ 的定义，$\sum_i A_i^{int}=0$。乘以常数 $\lambda$ 后仍为零。证毕。

#### 默认门控实现的边界

默认门控实现为了区分探索强度并抑制低质量探索，使用 sample-dependent 的 $w_i q_i$。因此严格的零均值性质对最终 $B_i^{int}$ 不再成立：

$$
\sum_i w_i q_i A_i^{int}\neq 0.
$$

不过，DIVE-PO 仍保持两个性质。第一，task reward 本身不被修改，任务分数与探索分数可分开记录。第二，最终扰动被 $\lambda$、$q_i$、$w_i$ 和 $c$ 限制，避免 intrinsic stream 主导 task stream。投稿前应加入一个消融：在 $w_iq_iA_i^{int}$ 之后再次组内中心化，比较 “center-after-gate” 与默认非二次中心化实现的稳定性。

### 3.3 局内新颖性：SimHash-KNN Episodic Novelty

终端轨迹由多个 action state 组成。第 $t$ 步 action state 记为：

$$
s_t=(tool, signature, observation, exit, turn).
$$

其中 `turn` 使用粗粒度 bucket，例如第一步、早期、中期和后期，避免精确位置本身成为可刷新的 novelty 来源。将状态映射到 $d=256$ 维向量：

$$
z_t=\phi(s_t)\in\mathbb{R}^{256}.
$$

SimHash 使用随机超平面矩阵 $P\in\mathbb{R}^{64\times 256}$：

$$
h_t=\mathbf{1}[(Pz_t)_j\ge0]_{j=1}^{64}.
$$

候选集合来自当前 bucket 及 Hamming radius 1 的 probe buckets：

$$
C_t=\{z_j\mid h_j\in\mathcal{N}_1(h_t)\}.
$$

若 $C_t$ 为空，说明该状态落入未访问区域，取最大局内 novelty：

$$
r_t^{epi}=1.
$$

否则，用最近 $k=5$ 个候选的平均 cosine distance：

$$
d(z,z')=\frac{1-\cos(z,z')}{2},
\qquad
\bar d_k=\frac{1}{k}\sum_{j\in\operatorname{KNN}_k(t)}d(z_t,z_j),
$$

$$
r_t^{epi}=\max\left(\eta,\frac{\bar d_k}{\bar d_k+1}\right),
\qquad \eta=0.02.
$$

轨迹级局内新颖性为：

$$
r_i^{epi}=\frac{1}{T_i}\sum_{t=1}^{T_i}r_t^{epi}.
$$

实现采用 compute-then-add：先计算当前 action 的 novelty，再写入 episodic memory，避免当前 action 污染自己的 novelty。

### 3.4 局间新颖性：Hierarchical Decayed Lifelong Count

局间新颖性衡量跨轨迹的历史访问频率。DIVE-PO 为每个状态构造三层 key：

$$
l\in\{\mathrm{task},\mathrm{skill},\mathrm{global}\}.
$$

task 层最具体，包含任务、动作族、命令签名、观察 fingerprint 与退出状态；skill 层抽象到工具/动作模式；global 层只保留跨任务的粗粒度动作族信息。

对 key $k$，读取写入当前轨迹前的 decayed count：

$$
\tilde c_k=c_k\cdot\delta^{\Delta_k},
\qquad \delta=0.995,
$$

其中 $\Delta_k$ 是距离上次访问的轨迹间隔。单 key novelty 为：

$$
u(k)=\frac{1}{\sqrt{\tilde c_k+1}}.
$$

每层 raw novelty 为：

$$
r_l=\frac{1}{|K_l|}\sum_{k\in K_l}u(k).
$$

三层融合为：

$$
r_i^{life}=
\operatorname{clip}_{[0,2]}
\left(
0.50r_{task}+0.35r_{skill}+0.15r_{global}
\right).
$$

权重设计反映三个优先级：task 层鼓励当前任务中的新策略；skill 层鼓励可迁移工具模式；global 层保留少量跨任务探索压力，但避免泛泛的新动作获得过高奖励。

随后用历史 running mean/std 标准化，并经过 softplus 得到 lifelong modifier：

$$
z_i=
\operatorname{clip}
\left(
\frac{r_i^{life}-\mu_{before}}{\sigma_{before}},
-5,5
\right),
$$

$$
m_i^{life}=
\operatorname{clip}_{[1,5]}
\left(1+\operatorname{softplus}(z_i)\right).
$$

最终轨迹级 intrinsic signal 为：

$$
I_i=r_i^{epi}\cdot m_i^{life}.
$$

Decayed count 的作用是避免早期访问永久压制后续探索。长期未出现的行为会随时间恢复部分新颖性，更符合非平稳策略训练过程。

### 3.5 Outcome-aware Gate 与 Beta Arm

DIVE-PO 不鼓励所有新颖轨迹，而是用 outcome-aware gate 抑制低质量探索。令 $o_i\in[0,1]$ 为 outcome score，优先使用任务 raw score；若缺失，则回退到 accuracy、success score、unit-test pass rate 等任务相关指标。

不同轨迹状态的 floor 为：

$$
f_i=
\begin{cases}
0.50, & \text{completed},\\
0.15, & \text{truncated},\\
0, & \text{failed or aborted}.
\end{cases}
$$

质量门控为：

$$
q_i=f_i+(1-f_i)o_i.
$$

Agent57-style beta arm 控制探索强度。默认设置使用 8 个 arm：

$$
\beta\in
\{0,0.004,0.006,0.008,0.010,0.012,0.016,0.020\}.
$$

arm weight 为：

$$
w_i=\frac{\beta_{a_i}}{\max_a\beta_a}.
$$

因此 arm 0 是 task-only baseline，最大 arm 的 intrinsic advantage 有效系数为 $\lambda=0.08$。这不是把原始 Agent57 的 $\beta$ 直接加到 reward，而是在 post-normalized advantage 空间中使用 normalized beta 作为相对探索强度。

### 3.6 Truncation Penalty

DIVE-PO 保留 outcome-aware truncation penalty：

$$
P_i^{trunc}
=
-0.01\cdot\mathbf{1}[\mathrm{truncated}_i]\cdot(1-o_i).
$$

这避免把高 outcome 的 truncated trajectory 一律视为坏样本，同时惩罚低 outcome 的超长或未完成轨迹。

### 3.7 UCB Arm Controller

UCB 只选择 rollout 使用哪个 arm，不直接产生 reward。每个非 evaluation group 保留一个 arm 0 baseline，其余位置由 UCB 或小概率随机探索选择。

对 arm $a$，默认 score 为：

$$
UCB_a=
\bar R_a^{base}
-0.5\cdot parse\_rate_a
-0.5\cdot trunc\_rate_a
+0.5\sqrt{\frac{\log(N+1)}{n_a}}.
$$

其中 $\bar R_a^{base}$ 是窗口内 normalized base reward，窗口大小为 256；$n_a$ 是 arm 访问次数；若 $n_a<4$，score 设为 $+\infty$ 以保证冷启动覆盖；随机探索概率为 0.02。两个错误率惩罚系数是经验设定，投稿前必须做敏感性分析。

### 3.8 轻量温度阶梯与 Off-policy 边界

默认设置还为 arm 配置轻量温度阶梯：

$$
T_a\in
\{1.00,1.00,1.005,1.010,1.015,1.020,1.025,1.030\}.
$$

该阶梯在 24 个 rollout warmup 后生效，top-p 均为 1。其目的是让不同 arm 不只在训练梯度中因 beta 不同而区分，也在采样侧产生轻量差异。

这一设计带来 correctness 问题：若 rollout 实际来自 tempered policy $\pi_{\theta_{\mathrm{old}}}^{T_a}$，但训练 ratio 使用未按 arm 温度修正的 $\pi_{\theta_{\mathrm{old}}}$，则 PPO/DAPO 的 on-policy 假设存在偏差。默认温度范围很小，但论文不能默认其无影响。投稿前必须报告：

$$
\left|\log p_{\mathrm{train\ old}}-\log p_{\mathrm{rollout}}\right|,
\qquad
KL(\pi_{\theta_{\mathrm{old}}}^{T_a}\|\pi_{\theta_{\mathrm{old}}}),
$$

以及各 arm 的 ratio 分布。若偏差不可忽略，应采用 temperature-aware old log-prob、启用 rollout log-prob ratio，或删除温度阶梯以保持方法更干净。

### 3.9 算法伪代码

**Algorithm 1：DIVE-PO**

```text
Input: prompts q, group size G, policy pi_theta, beta arms, UCB controller
for each training iteration do
    for each prompt q do
        assign G arms with one task-only baseline arm and UCB-selected non-baseline arms
        sample trajectories tau_i with arm-specific beta and optional temperature
        compute task reward R_i^task with verifier
        extract action states s_t from each trajectory
        compute episodic novelty r_i^epi with SimHash-KNN
        compute lifelong novelty modifier m_i^life with decayed hierarchical counts
        set intrinsic signal I_i = r_i^epi * m_i^life
    end for

    for each prompt group do
        compute A_i^task = Norm_G(R_i^task)
        compute A_i^int  = Norm_G(I_i)
        compute quality gate q_i from outcome and trajectory status
        compute arm weight w_i = beta_i / max(beta)
        compute B_i^int = clip(lambda * w_i * q_i * A_i^int, -c, c)
        compute truncation penalty P_i^trunc
        set training advantage Ahat_i = A_i^task + B_i^int + P_i^trunc
    end for

    update policy with DAPO clipped objective using token-level loss
    update UCB arm statistics with base reward, parse rate, and truncation rate
end for
```

## 4 实验设计

本节给出投稿前应完成的实验。所有表格中的数值均为占位符，不代表实验结论。

### 4.1 主实验：终端智能体

主实验应覆盖内部 SETA 与至少一个公开 benchmark。若只报告内部 SETA，审稿人难以校准结果。

建议 benchmark：

| Benchmark | 目的 | 指标 |
|---|---|---|
| SETA terminal tasks | 验证目标训练场景中的任务完成能力 | raw score、success rate、truncation rate、parse error |
| Terminal-Bench 或同类公开终端 benchmark | 外部可复现验证 | pass rate、normalized score、turns、wall-clock |
| SWE 类软件工程任务，可选 | 验证方法是否能迁移到更复杂代码修改任务 | resolved rate、test pass rate |

主表应同时报告等训练步数和等 wall-clock 两种口径，并报告 intrinsic 计算 overhead。

| 方法 | score-space bonus | intrinsic advantage | beta arm | UCB | SETA score | SETA pass | Public pass | overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base / SFT | 否 | 否 | 否 | 否 | [TODO] | [TODO] | [TODO] | [TODO] |
| GRPO | 否 | 否 | 否 | 否 | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO | 否 | 否 | 否 | 否 | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO + score-space intrinsic | 是 | 否 | 可选 | 可选 | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO + dual-stream, no beta | 否 | 是 | 否 | 是 | [TODO] | [TODO] | [TODO] | [TODO] |
| DIVE-PO | 否 | 是 | 是 | 是 | [TODO] | [TODO] | [TODO] | [TODO] |

### 4.2 数学 RLVR：MATH、AMC23、AIME24、AIME25

为了与 CDE 类 RLVR 工作可比，数学实验应至少包含 MATH、AIME24 和 AIME25；AMC23 可作为中等难度补充。数学任务没有 terminal command state，因此需要使用附录 A 的 reasoning-state novelty 适配。

| Benchmark | 推荐指标 | 说明 |
|---|---|---|
| MATH | Avg@1，Avg@8/16 | 标准数学推理能力 |
| AMC23 | Avg@16，Pass@16 | 中等难度补充 |
| AIME24 | Avg@16，Pass@16 | 高难竞赛推理，多样采样收益 |
| AIME25 | Avg@16，Pass@16 | 更新、更难泛化集 |

数学主表：

| 方法 | MATH Avg@1 | AMC23 Avg@16 | AIME24 Avg@16 | AIME24 Pass@16 | AIME25 Avg@16 | AIME25 Pass@16 |
|---|---:|---:|---:|---:|---:|---:|
| Base / SFT | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| GRPO | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO + entropy bonus | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO + CDE-style PPL bonus | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| DIVE-PO-math | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

CDE-style baseline 必须真实复现 actor PPL bonus 或使用作者公开实现，不能只作为占位行。

### 4.3 探索指标

只报告 Avg@1 或 pass rate 不足以支撑探索 claim。应加入：

| 指标 | 目的 |
|---|---|
| pass@k 曲线，$k=1,\ldots,16$ | 验证多样采样是否带来更多可解轨迹 |
| policy entropy 曲线 | 观察熵坍塌是否缓解 |
| distinct command/action rate | 衡量终端动作多样性 |
| exact repeat rate | 衡量重复动作是否减少 |
| episodic novelty / empty bucket rate | 验证局内探索 |
| lifelong raw / modifier | 验证跨轨迹探索 |
| arm distribution | 验证 UCB 是否自适应分配探索预算 |
| parse/truncation rate | 排除探索导致格式或长度失败 |

### 4.4 多 Seed 与显著性

探索方法方差通常较大。主实验至少使用 3 个 seeds，报告 mean±std；关键对比使用 paired bootstrap 或近似随机化检验。若算力不足，至少对 DAPO、DIVE-PO 和两个核心消融做多 seed。

## 5 消融实验

消融应围绕核心 claim 排优先级，而不是平均铺开。

### 5.1 核心消融

| 消融 | 验证假设 | 设置 | 预期结论 | 支撑 claim |
|---|---|---|---|---|
| Score-space vs dual-stream | advantage-space 注入比 reward-space 注入更稳定 | 将 $I_i$ 加到 score 后再归一化，对比默认双流 | dual-stream 更稳，parse/truncation 更低 | 核心 claim |
| 去掉 beta weighting | arm 必须对应不同训练探索强度 | 令 $w_i=1$ 或固定非零值 | arm 差异变弱，UCB 作用下降 | beta arm 有效 |
| 去掉 outcome gate | 质量门控抑制低质量新颖性 | 令 $q_i=1$ 或只用 status scale | 高 novelty 低 outcome 样本增多 | gate 必要 |
| episodic-only / lifelong-only | 两级 novelty 互补 | 分别关闭 lifelong 或 episodic | 单独使用时探索覆盖下降 | 双级 novelty 必要 |

### 5.2 次级消融

| 消融 | 目的 | 处理建议 |
|---|---|---|
| UCB vs round-robin/fixed arm | 验证自适应 arm selection | 若收益小，可简化 controller |
| 温度阶梯 | 验证采样侧差异是否值得保留 | 若 off-policy 偏差或收益不明显，应删除 |
| $\lambda$ sweep | 验证 intrinsic 强度 | 测试 $0.04,0.08,0.12,0.16$ |
| beta ladder 密度 | 验证低端密集 ladder 的必要性 | 对比线性 ladder 与默认低端密集 ladder |
| center-after-gate | 验证 Proposition 1 的严格变体 | 对 $w_iq_iA_i^{int}$ 二次中心化 |
| clip $c=0.35$ | 验证 clip 是否实际激活 | 报告 clip 激活率；若接近 0，可只作为安全网或删除 |

## 6 分析实验

### 6.1 Pass@k 与多样性曲线

目的：证明 DIVE-PO 真正提升探索，而不是只改变平均分。

设置：每个 checkpoint 评估 pass@k、策略熵、distinct command rate、distinct reasoning path rate 和 exact repeat rate。

预期：DIVE-PO 的 pass@k 曲线在较大 $k$ 上更明显，重复率下降，熵下降速度更慢。

### 6.2 Off-policy 与温度偏差

目的：回答不同 arm 温度是否破坏 DAPO ratio 假设。

设置：按 arm 记录 rollout log-prob、训练侧 old log-prob、二者差值、sequence-level KL、clip fraction 和 ratio 分位数。

预期：默认温度 1.00-1.03 的偏差应较小；若高温 arm 显著偏离，应改为 temperature-aware ratio 或删除温度阶梯。

### 6.3 混 Arm 组内归一化

目的：分析 $\operatorname{Norm}_G(I_i)$ 在同一 group 内混合不同 beta/temperature arm 的影响。

设置：统计每个 arm 的 $I_i$、$A_i^{int}$、$B_i^{int}$、task advantage 分布，并比较 group-level normalization 与 per-arm normalization。

预期：若高温 arm 系统性获得正 intrinsic advantage、低温 arm 系统性为负，需要判断这是有意的跨 arm 比较，还是对 baseline arm 的不公平压制。

### 6.4 Reward Hacking 审计

目的：验证 novelty 不会被表面变化刷高。

设置：抽样高 novelty 低 outcome 轨迹，按模式分类：无意义 echo、参数顺序扰动、重复查看文件、过长命令、格式失败等。

预期：outcome gate 应显著降低这些样本的最终 bonus；若仍大量出现，需要改进 state fingerprint 或 gate。

### 6.5 Intrinsic 与 Task Advantage 量级

目的：确认 intrinsic stream 足够可见但不主导任务优化。

设置：记录 $A_i^{task}$、$A_i^{int}$、$B_i^{int}$、$\hat A_i$ 的均值、标准差、分位数，以及 clip 激活频率。

预期：最大有效系数 0.08 使 $B_i^{int}$ 处于 task advantage 的辅助量级；若 clip 0.35 几乎不激活，应将其解释为安全网或在最终实现中简化。

### 6.6 UCB 系数敏感性

目的：减少 ad hoc 超参风险。

设置：扫描 UCB 探索系数与 parse/truncation 惩罚系数，例如 $\{0.25,0.5,1.0\}$。

预期：DIVE-PO 对小范围变化应相对鲁棒；若性能高度依赖某一组系数，应在正文中降低 UCB 的贡献定位。

### 6.7 Compute-matched 对比

目的：排除额外计算带来的不公平收益。

设置：报告等训练步数与等 wall-clock 两种结果，并拆分 SimHash-KNN、sqlite lifelong store、UCB controller 的时间开销。

预期：DIVE-PO 的主要收益不应只来自更多 wall-clock 或更多有效样本。

## 7 相关工作

### 7.1 RLVR、GRPO 与 DAPO

PPO（Schulman et al., 2017）奠定了 clipped policy optimization 的基础。GRPO 在 RLVR 中用组内相对优势替代显式 critic，被 DeepSeekMath 等工作用于数学推理训练。DAPO 在 GRPO 类估计器上加入更适合长输出任务的 clipping、token-level loss、overlong shaping 和可选 dynamic sampling。DIVE-PO 建立在 DAPO objective 上，但关注点不同：它研究如何把探索信号作为独立 advantage stream 注入 group-based RLVR。

### 7.2 Count-based 与 Intrinsic Motivation

经典 count-based exploration 使用访问次数或伪计数奖励罕见状态（Bellemare et al., 2016；Tang et al., 2017）。ICM（Pathak et al., 2017）和 RND（Burda et al., 2019）使用预测误差作为新颖性信号。DIVE-PO 与这些方法共享“访问不足区域应获得探索压力”的思想，但不训练额外 dynamics 或 predictor；它使用 action-state fingerprint、SimHash-KNN 和 hierarchical count 作为轻量估计。

### 7.3 NGU、Agent57 与 UCB

NGU 和 Agent57（Badia et al., 2020a,b）结合 episodic novelty、lifelong novelty 和不同探索强度的策略族，在 Atari 等环境中实现深度探索。DIVE-PO 借鉴其 episodic/lifelong 和 beta arm 思路，但不训练多个独立策略；所有 arm 共享同一语言模型，只通过 post-normalized intrinsic advantage、UCB arm selection 和轻量采样参数产生差异。UCB 源自多臂老虎机中的 optimism under uncertainty（Auer et al., 2002），在本文中只作为探索预算分配器。

### 7.4 语言模型中的探索与 CDE

近期 RLVR 探索工作研究了 entropy bonus、actor perplexity bonus、critic uncertainty 和多头 critic。CDE 类方法把 actor PPL 或 critic 方差作为好奇心信号。DIVE-PO 的定位不同：默认实现不使用 actor PPL，也不使用多头 critic，而是把可解释的轨迹新颖性变成独立 intrinsic advantage。与 CDE 的实验对比应聚焦“PPL/uncertainty curiosity”与“state novelty advantage stream”的差异。

## 8 局限性

DIVE-PO 仍有若干限制。第一，intrinsic signal 是轨迹级的，而 token-level loss 会把同一 bonus 广播到所有生成 token，credit assignment 较粗。第二，轻量温度阶梯可能引入 off-policy 偏差，必须用 log-prob/KL 统计验证。第三，SimHash novelty 可能被表面扰动 hack，outcome gate 只能缓解，不能完全解决。第四，state fingerprint 的设计与终端环境强相关，迁移到数学或软件工程任务需要重新定义状态。第五，lifelong sqlite store 在分布式训练下存在一致性和锁竞争问题，需要报告 overhead 和失败恢复策略。第六，$\lambda$、UCB 惩罚系数、hierarchy weights 等仍是经验设定，需要更系统的敏感性分析。

## 9 可复现性声明

最终投稿应报告以下信息：base model 与 checkpoint；训练数据可得性；SETA 是否可公开；公开 benchmark 的版本、prompt、verifier 和容器环境；训练 GPU 型号、GPU 数、总卡时；随机种子；rollout group size；采样温度；DAPO clip；是否使用 rollout log-probs；所有 exploration 超参；sqlite lifelong store 的持久化路径与清理策略。若 SETA 无法公开，主结论必须由公开 benchmark 和数学 benchmark 支撑。

## 10 结论

本文提出 DIVE-PO：Policy Optimization with Dual-stream Intrinsic adVantage Exploration。DIVE-PO 的核心不是简单拼接多个探索模块，而是把探索信号从 score space 移到 independently normalized advantage stream。局内 SimHash-KNN、局间 decayed count、outcome-aware gate、beta arm 与 UCB controller 共同服务于这一目标：在不污染 verifier reward 的前提下，为 group-based RLVR 提供可控探索压力。投稿前的关键工作是完成 SETA 与公开 benchmark 主实验、四项核心消融、pass@k/熵/多样性曲线，以及 off-policy 和混 arm 归一化分析。

# 附录 A：数学 RLVR 中的 DIVE-PO 适配

数学任务没有终端命令和观察文本，因此不能直接复用 terminal action state。适配原则是把“工具动作新颖性”替换为“推理状态新颖性”，同时保留 dual-stream advantage fusion。

## A.1 Reasoning State

将模型输出切分为 reasoning states：

$$
s_t^{math}=(problem\_type, step\_role, equation\_pattern, answer\_format, position\_bucket).
$$

其中 $step\_role$ 可包括 lemma、case split、calculation、verification、final answer 等；$position\_bucket$ 可取 early/mid/late/final，避免仅因输出长度获得 novelty。

## A.2 数学 SimHash-KNN

将 $s_t^{math}$ 映射为向量 $z_t=\phi(s_t^{math})$，复用 64-bit SimHash 与 Hamming radius 1 multi-probe：

$$
h_t=\mathbf{1}[(Pz_t)_j\ge0]_{j=1}^{64}.
$$

step novelty 与 terminal 设置相同，但建议对错误答案使用更强 gate，因为数学中“新颖但错误”的推理路径很多。

## A.3 Lifelong Key

数学 lifelong key 可分三层：

| 层级 | key 示例 |
|---|---|
| task | dataset/problem id/problem type/step fingerprint/equation pattern/final answer type |
| skill | problem type/operation family/equation pattern/answer format |
| global | operation family/answer format |

operation family 可包括 algebra、geometry、number theory、combinatorics、probability 等。

## A.4 Outcome Gate

若只有最终答案 verifier，可设：

$$
o_i=\mathbf{1}[\mathrm{answer\ correct}].
$$

若有 partial verifier 或 process reward，可使用 $[0,1]$ 连续分数。无效格式、无法抽取答案和超长输出应使用低 floor。最终仍使用：

$$
B_i^{int}=
\operatorname{clip}_{[-c,c]}
\left(
\lambda w_i q_i A_i^{int}
\right).
$$

## A.5 数学实验注意事项

数学实验若无法在投稿前完成，不应作为未兑现 promise 放在正文中。若完成，应并入主实验，并与 CDE-style PPL bonus 做严格对比：相同 base model、相同 prompt、相同 verifier、相同采样预算和相同 pass@k 统计。

# 附录 B：当前实现配置映射

本附录仅用于代码 release 和复现实验；正文不依赖环境变量名叙事。

| 模块 | 当前实现 |
|---|---|
| base model | Qwen3-8B |
| policy optimization | DAPO with GRPO advantage estimator |
| dynamic sampling | off |
| group size | 8 samples per prompt |
| rollout batch size | 8 prompts |
| max turn | 10 |
| max response/context length | 8192 / 16384 |
| DAPO clip | $\epsilon_{low}=0.2,\epsilon_{high}=0.28$ |
| token-level loss | on |
| KL loss | off by default |
| score-space exploration bonus | off |
| intrinsic fusion | dual-stream post-normalized advantage |
| intrinsic lambda / clip | 0.08 / 0.35 |
| arm weight | normalized beta |
| beta arms | 0, 0.004, 0.006, 0.008, 0.010, 0.012, 0.016, 0.020 |
| outcome gate floors | completed 0.50, truncated 0.15, failed 0, aborted 0 |
| truncation penalty | $-0.01(1-o_i)$ for truncated trajectories |
| episodic backend | SimHash-KNN |
| episodic bits / dim / k | 64 / 256 / 5 |
| episodic distance / radius | cosine / Hamming radius 1 |
| episodic novelty floor | 0.02 |
| lifelong backend | sqlite |
| lifelong decay / capacity | 0.995 / 200000 |
| lifelong hierarchy weights | task 0.50, skill 0.35, global 0.15 |
| UCB C / window / epsilon / min | 0.5 / 256 / 0.02 / 4 |
| UCB value | normalized base reward |
| arm temperatures | 1.00, 1.00, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030 |
| temperature warmup | 24 rollouts |

# 附录 C：投稿前优先级

1. 完成 SETA 主实验、公开 benchmark 主实验、四项核心消融、pass@k 与熵曲线。
2. 补 off-policy 温度偏差与混 arm 组内归一化分析，决定是否保留温度阶梯。
3. 补 reward hacking 审计、intrinsic/task advantage 量级、clip 激活频率和 UCB 敏感性。
4. 落实 CDE-style baseline 与相关工作引用。
5. 视算力决定数学实验是否并入正文；未完成时不要把 MATH/AIME 作为核心 claim。

# 附录 D：参考文献占位

- Schulman et al. Proximal Policy Optimization Algorithms. 2017.
- Auer et al. Finite-time Analysis of the Multiarmed Bandit Problem. 2002.
- Bellemare et al. Unifying Count-Based Exploration and Intrinsic Motivation. 2016.
- Tang et al. #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning. 2017.
- Pathak et al. Curiosity-driven Exploration by Self-supervised Prediction. 2017.
- Burda et al. Exploration by Random Network Distillation. 2019.
- Badia et al. Never Give Up: Learning Directed Exploration Strategies. 2020.
- Badia et al. Agent57: Outperforming the Atari Human Benchmark. 2020.
- Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. 2024.
- [TODO] DAPO 原始论文或技术报告。
- [TODO] GSPO 原始论文或技术报告。
- [TODO] CDE 参考论文正式 BibTeX。
