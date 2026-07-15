# SETA + DAPO Latent World Model 调研与集成方案

> **历史设计稿**：本文保留 2026-06-16 的调研与方案推导。当前实现、配置和命令请以
> [`latent_world_model_guide_zh.md`](latent_world_model_guide_zh.md) 为准；精简研究摘要见
> [`latent_world_model_research_notes_zh.md`](latent_world_model_research_notes_zh.md)。

生成时间：2026-06-16 15:46:23 HKT

本文基于当前代码与 baseline 轨迹撰写，目标是在 OpenClaw-RL/terminal-rl 的 SETA + DAPO 训练链路中，最小侵入式集成一个预测环境反馈的 latent world model。

## 1. 代码调研事实

### 1.1 `le-wm` 的 world model 数据流

`le-wm` 是显式分离的 JEPA/world-model 结构：

| 模块 | 实现位置 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| observation encoder | `/mnt/shared-storage-user/puyuan/code/le-wm/jepa.py:29-40` | `info["pixels"]`，按 `(B,T,...) -> (B*T,...)` 展平 | `info["emb"]`，形状 `(B,T,D)` | ViT 编码观测，取 CLS，再过 projector |
| action encoder | `/mnt/shared-storage-user/puyuan/code/le-wm/jepa.py:42-43`，`module.py:189-214` | `info["action"]` | `info["act_emb"]` | 将动作序列映射到与 state latent 对齐的维度 |
| AR predictor | `/mnt/shared-storage-user/puyuan/code/le-wm/module.py:244-285` | 历史 latent `x=(B,T,D)` 与动作 embedding `c=(B,T,D)` | 预测 latent `(B,T,D)` | Transformer + AdaLN conditioning，根据 `(z_t,a_t)` 预测未来 latent |
| training loss | `/mnt/shared-storage-user/puyuan/code/le-wm/train.py:17-42` | `emb`、`act_emb` | `pred_loss + 0.09 * sigreg_loss` | 同时训练 encoder/projector/predictor/action encoder |
| SigReg | `/mnt/shared-storage-user/puyuan/code/le-wm/module.py:10-36` | `emb.transpose(0,1)`，形状注释为 `(T,B,D)` | 标量 regularizer | 用随机投影的 Epps-Pulley 统计约束 embedding 分布接近各向同性 Gaussian |

核心训练公式来自代码：

$$
z_{1:T}=E(o_{1:T}),\quad e^a_{1:T}=A(a_{1:T})
$$

$$
\hat z_{2:T+1}=P(z_{1:T}, e^a_{1:T})
$$

$$
\mathcal{L}_{pred}=\|\hat z - z_{target}\|_2^2
$$

$$
\mathcal{L}_{lewm}=\mathcal{L}_{pred}+\lambda_{sig}\mathcal{L}_{sigreg}(z),\quad \lambda_{sig}=0.09
$$

关键注意点：

- 训练代码中 `pred_loss = (pred_emb - tgt_emb).pow(2).mean()` 没有对 `tgt_emb` detach（`train.py:35-41`），因此预测损失会回传到 predictor，也会回传到 encoder/projector 产出的 target embedding。
- `jepa.py:120-123` 里的 `goal_emb.detach()` 只出现在 inference cost，不是训练 loss。
- SigReg 作用在 encoder 产出的 `emb` 上，而不是 predictor 输出上；它通过随机单位投影约束 batch/time 上的投影分布，防止所有 embedding 退化到常数或低秩子空间。

### 1.2 当前 SETA + DAPO baseline 训练链路

baseline wrapper：

- `/mnt/shared-storage-user/puyuan/code/OpenClaw-RL/terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh:21-33` 固定 `ROLLOUT_BATCH_SIZE=8`、`N_SAMPLES=8`、`MAX_TURN=10`，设置 `DAPO_DYNAMIC_SAMPLING=0`，并委托 mixed nodynamic 脚本。

实际 mixed 脚本的关键参数：

| 阶段 | 代码位置 | 事实 |
|---|---|---|
| rollout 参数 | `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:1174-1192` | `--reward-key score`，`--rollout-batch-size 8`，`--n-samples-per-prompt 8`，`--rollout-temperature 1`，`--num-steps-per-rollout 2` |
| DAPO 参数 | `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:1242-1254` | `--advantage-estimator grpo`，`--dynamic_history`，`--eps-clip 0.2`，`--eps-clip-high 0.28`，默认 `--calculate-per-token-loss` |
| custom rollout | `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:1336-1347` | 使用 `generate.generate`，`rollout_log.rollout_log`；只有探索 post-norm bonus 开启时才加 `reward_postprocess.post_process_rewards` |
| 实际 run 参数 | `runs/...092726/logs/train.log:864` | 该 baseline run 使用 Megatron backend、Qwen3-8B、8 GPU，其中 actor 4 GPU、rollout 4 GPU |

rollout 内部数据流：

1. `generate.generate()` 构造 `RunContext` 和任务环境，见 `terminal-rl/generate.py:2832-2866`。
2. 远程环境通过 `TerminalEnvClient.allocate/reset/exec_tool/evaluate` 调用，见 `terminal-rl/env_client.py:74-160`。
3. SGLang 生成请求由 `SGLangTurnClient.generate_turn()` 发出，payload 包含 `input_ids`、`sampling_params`、`return_logprob=True`，见 `terminal-rl/inference_client.py:207-220`。
4. `generate_turn()` 保存 `input_ids`、`output_token_ids`、`output_token_logprobs` 到 `Interaction`，见 `terminal-rl/inference_client.py:260-319` 和 `terminal-rl/custom_types.py:59-69`。
5. `AgentRunner` 每轮拿历史消息、调用模型、执行工具并把工具结果写回 agent memory，见 `terminal-rl/agent_runner.py:103-141`。
6. `generate.py` 每轮保存 `context_messages`、`assistant_output`、`tool_calls[*].result` 到 trajectory record，见 `terminal-rl/generate.py:3179-3300`。
7. 结束后调用环境 evaluate 得到 raw score，见 `terminal-rl/generate.py:3377-3428`。
8. `_build_samples()` 将每个 model turn 转成一个 `Sample`，见 `terminal-rl/generate.py:2384-2488`：
   - `s.tokens = interaction.input_ids + interaction.output_token_ids`
   - `s.response_length = len(interaction.output_token_ids)`
   - `s.loss_mask = [1] * response_length`
   - `s.rollout_log_probs = interaction.output_token_logprobs`
   - `s.reward["score"]` 来自 task reward / DAPO overlong / optional exploration。

训练数据与 DAPO loss：

- `slime/slime/ray/rollout.py:395-470` 做 reward post-process。GRPO/GSPO 且 reward normalization 开启时，按 group 做 reward centering/std normalization。
- `slime/slime/ray/rollout.py:684-754` 转成 train data：`tokens`、`response_lengths`、`rewards`、`raw_reward`、`loss_masks`、`rollout_log_probs` 等。
- 当前 run 实际走 Megatron actor。`slime/slime/backends/megatron_utils/actor.py:474-530` 在 actor train 前计算 old/ref logprob 和 GRPO advantage。
- `slime/slime/backends/megatron_utils/loss.py:320-326` 对 GRPO/GSP0 使用 reward/KL 得到 returns/advantages。
- `slime/slime/backends/megatron_utils/model.py:342-418` 的 train forward 当前只返回 logits 给 loss；没有暴露 hidden states。
- `slime/slime/backends/megatron_utils/loss.py:575-715` 计算 policy loss：current logprob、entropy、PPO-style clipped policy gradient、optional KL/TIS。

### 1.3 真实轨迹中的 \(h_t,a_t,o_{t+1}\) 边界

真实 trajectory 格式来自：

`/mnt/shared-storage-user/puyuan/code/OpenClaw-RL/runs/terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_think_mt10_2026-06-11_092726/trajectories`

样例 1，失败轨迹：

`seta_t407_r5_st10_g46_s369_e644ba3e_20260611_110711/traj.json`

- `info.status = Status.COMPLETED` 或 `Status.TRUNCATED/FAILED` 等保存在 `info`。
- `turns[0].context_messages` 有 system/user 两条，长度分别约 5819/1683 字符。
- `turns[0].assistant_output` 是完整 LLM 输出，包含 `<think>` 和 tool call，长度约 9319 字符。
- `turns[0].tool_calls[0]`：
  - `tool_name = shell_exec`
  - `args = {"id": "1", "command": "sudo ls -la ..."}`
  - `result = "OCI runtime exec failed ... chdir to cwd ..."`
- 下一轮 `context_messages` 会追加该 `tool` result，因此 \(o_{t+1}\) 明确进入下一轮历史。

样例 2，成功轨迹：

`seta_t1021_r90_st180_g720_s5761_e7c47e3a_20260613_010719/traj.json`

- `info`: `task_name=1021`，`data_source=terminal_bench`，`rollout_id=90`，`train_step=180`，`status=Status.COMPLETED`，`num_turns=2`。
- `reward`: `raw_score=1.0`，`accuracy=1.0`，`base_score=1.0`，`score=0.93115234375`，`total_reward=0.93115234375`。
- turn 0 调用多个 shell tools，例如 `cryptsetup luksDump`、`cryptsetup luksAddKey`、`cryptsetup luksRemoveKey`，每个工具结果保存在 `tool_calls[*].result`。

因此，推荐的 transition 定义是 turn 级：

$$
h_t = \text{context\_messages before model turn }t
$$

$$
a_t = \text{assistant output / parsed tool calls generated at turn }t
$$

$$
o_{t+1} = \text{tool\_calls[*].result and/or terminal evaluation feedback after }a_t
$$

当前 `Sample.tokens` 只包含 \(h_t+a_t\)，不包含工具返回；这正好适合用 causal LLM hidden 预测下一步环境反馈。

## 2. 更新后的核心设计判断

### 2.1 旧方案的保留结论

旧方案是用 `H_\theta(h_t,a_t)` 的最后 action hidden 直接接一个 predictor，预测 fixed text encoder 得到的 feedback latent。这个方案可以作为最低成本 baseline，但不再作为主方案。

它的问题是：

- 没有显式的 \(z_t\) state encoder，结构上不像 `le-wm` 的 \(z_t,a_t\rightarrow z_{t+1}\)。
- SigReg 不能直接加在 LLM hidden 上；否则会把 policy backbone 的 hidden 推向各向同性 Gaussian，可能干扰 DAPO 训练。
- predictor 输出空间与 target feedback encoder 空间缺少共享 latent 约束，容易变成“黑盒回归器”，研究价值弱于显式 latent world model。

旧方案仍有价值：可作为 offline probe 的最小 sanity check，用来确认 LLM hidden 是否包含可预测下一步环境反馈的信息。

### 2.2 新主方案：LLM-hidden-conditioned LEWM

推荐采用“尽可能复用 `le-wm` 原理和代码”的结构：从 LLM 中分别取 \(h_t\)、\(a_t\)、\(h_{t+1}\) 对应 hidden，再通过源特定 adapter 和共享 latent projector 映射到同一个 world latent 空间；环境反馈 \(o_{t+1}\) 先过 fixed text encoder，再进入同一个 latent 空间。

核心形式：

$$
u^h_t = \text{Pool}\left(H_\theta(h_t)\right)
$$

$$
u^a_t = \text{Pool}\left(H_\theta(h_t,a_t)[a_t]\right)
$$

$$
z^s_t = C(A_h(u^h_t)),\quad e^a_t=A_a(u^a_t)
$$

$$
\hat z^s_{t+1}=P(z^s_t,e^a_t)
$$

feedback target：

$$
r_{t+1}=T_{fix}(o_{t+1})
$$

$$
z^o_{t+1}=C(A_o(r_{t+1}))
$$

next-state target：

$$
z^s_{t+1}=C(A_h(\text{Pool}(H_\theta(h_{t+1}))))
$$

其中：

- \(H_\theta\)：当前训练的 LLM hidden。
- \(A_h\)：LLM state hidden adapter。
- \(A_a\)：LLM action hidden adapter/action encoder。
- \(A_o\)：feedback text embedding adapter。
- \(C\)：共享 latent projector，把不同来源 embedding 映射到统一 world latent。
- \(P\)：复用 `le-wm` 的 `ARPredictor`。
- \(T_{fix}\)：冻结文本 encoder，用于稳定编码工具反馈。

这比旧方案更接近 `le-wm`：

| `le-wm` | terminal-agent 版本 |
|---|---|
| observation encoder 输出 `emb` | LLM hidden / feedback text encoder 输出经 adapter + shared projector 得到 \(z\) |
| action encoder 输出 `act_emb` | action hidden 经 \(A_a\) 得到 \(e^a_t\) |
| `ARPredictor(emb, act_emb)` | `ARPredictor(z^s_t, e^a_t)` |
| `SIGReg(emb.transpose(0,1))` | `SIGReg([z^s_t,z^s_{t+1},z^o_{t+1}].transpose(0,1))` |

### 2.3 为什么不使用完全同一个裸 projector

用户提出的“\(h_t\) hidden 和 fixed text encoder embedding 都通过同一个 encoder/projector”方向是合理的，但需要一个工程修正：不要让两种 raw embedding 直接进入完全相同的第一层 projector。

原因：

- \(u^h_t\) 来自 Qwen3 policy hidden，维度和分布由 LLM 决定。
- \(r_{t+1}=T_{fix}(o_{t+1})\) 来自另一个 text encoder，语义空间和数值分布不同。
- 如果直接共享裸 projector，projector 同时承担“跨 encoder 对齐”和“world latent 建模”，训练不稳定且难以诊断。

更稳的结构是：

$$
z = C(A_{source}(x))
$$

其中 \(A_{source}\) 是源特定 adapter，负责把不同 raw embedding 规整到统一维度/尺度；\(C\) 是共享 latent projector，负责定义真正的 world-model latent 空间。这样既保留“公用 encoder/projector”的研究意图，也避免 raw distribution mismatch。

### 2.4 hidden 边界与 target 定义

当前每个 `Sample.tokens` 是：

$$
\text{tokens}=h_t+a_t
$$

不包含工具反馈。因此可以安全取：

- `prompt_end = len(input_ids) - 1`：该位置 hidden 只看 \(h_t\)，可用于 \(z^s_t\)。
- `action_span = [len(input_ids), len(input_ids)+len(output_token_ids)-1]`：该 span hidden 看 \(h_t+a_t\)，可用于 \(e^a_t\)。
- `action_end = total_length - 1`：最低成本 action pooling 位置。

\(h_{t+1}\) 有两种来源：

| 来源 | 适用性 | 说明 |
|---|---|---|
| 下一轮 `Interaction.input_ids` | 非 final turn 最好 | 真实包含 tool result 后的下一轮 context，可抽取 \(z^s_{t+1}\) |
| 当前 turn 的 `tool_calls[*].result` | 所有 turn 可用 | 用 fixed text encoder 得到 \(z^o_{t+1}\)，final turn 也可训练 feedback prediction |

因此训练时建议使用双 target：

1. state target \(z^s_{t+1}\)：只对存在下一轮 context 的 turn 启用。
2. feedback target \(z^o_{t+1}\)：对有 tool feedback 的 turn 启用。

这避免 final turn 没有下一状态时丢样本，也让模型同时学习“下一 belief state”和“环境反馈语义”。

## 3. 优化版架构与数据流

### 3.1 总体结构

```text
current turn sample:
  input_ids = h_t
  output_token_ids = a_t
  feedback_text = o_{t+1}
  next_input_ids = h_{t+1}  # optional, non-final turn only

LLM branch:
  h_t tokens            -> LLM hidden[prompt_end]      -> A_h -> C -> z^s_t
  h_t + a_t tokens      -> LLM hidden[action_span]     -> A_a     -> e^a_t
  h_{t+1} tokens        -> LLM hidden[next_prompt_end] -> A_h -> C -> z^s_{t+1}

Feedback branch:
  o_{t+1} text -> fixed text encoder T_fix -> A_o -> C -> z^o_{t+1}

Prediction:
  ARPredictor(z^s_t, e^a_t) -> \hat z^s_{t+1}
  feedback_head(\hat z^s_{t+1}) -> \hat z^o_{t+1}

Training:
  logits -> existing DAPO/GRPO loss
  z/pred/value -> auxiliary LEWM losses
```

模块建议：

| 模块 | 推荐复用/新增 | 输入 | 输出 | 梯度策略 |
|---|---|---|---|---|
| `StateAdapter A_h` | 复用 `le-wm` 的 `MLP` 风格 | LLM hidden | adapter hidden | MVP detach LLM，只训练 adapter |
| `ActionAdapter A_a` | 复用 `MLP` 或 `Embedder` 思路 | action span hidden | action embedding \(e^a_t\) | 训练 |
| `FeedbackAdapter A_o` | 新增轻量 MLP | fixed text encoder embedding | adapter hidden | 训练或 pretrain 后冻结 |
| `SharedProjector C` | 复用 `le-wm` projector 思路 | adapter hidden | latent \(z\) | 训练 |
| `ARPredictor P` | 直接复用 `module.py:244-285` | \(z^s_t,e^a_t\) | \(\hat z^s_{t+1}\) | 训练 |
| `SIGReg` | 直接复用 `module.py:10-36` | latent sequence | regularizer | 训练 |
| `ValueHead` | 新增 MLP | \(z^s_t\) 或 \(\hat z^s_{t+1}\) | \(\hat V_t\) | 训练 |

### 3.2 与 `le-wm` 代码的复用边界

可以直接复用：

- `SIGReg`：输入仍是 `(T,B,D)` latent。
- `ARPredictor`：接口仍是 `x=(B,T,D)` 和 `c=(B,T,D_a)`。
- `MLP`/`Embedder`：可作为 adapter/action encoder 基础。
- `JEPA.predict()` 的 reshape 和 `pred_proj` 逻辑。
- `train.py` 的 loss 组织方式：`pred_loss + lambda * sigreg_loss`。

需要新写一个 wrapper，而不是原样使用 `JEPA.encode()`：

```text
HiddenLEWM.encode(batch):
  u_h_t       = batch["wm_state_hidden"]
  u_a_t       = batch["wm_action_hidden"]
  u_h_next    = batch["wm_next_state_hidden"]      # optional
  feedback_e  = batch["wm_feedback_embed"]         # from fixed text encoder

  z_t         = shared_projector(state_adapter(u_h_t))
  act_emb     = action_adapter(u_a_t)
  z_next_s    = shared_projector(state_adapter(u_h_next))
  z_next_o    = shared_projector(feedback_adapter(feedback_e))

HiddenLEWM.predict(z_t, act_emb):
  return ARPredictor(z_t, act_emb)
```

原因是 `le-wm/jepa.py:29-40` 的 `encode()` 写死了 `pixels -> encoder -> CLS -> projector`，而 terminal-agent 的输入已经是 LLM hidden 或 text encoder embedding。

### 3.3 Loss 设计

主预测目标：

$$
\hat z^s_{t+1}=P(z^s_t,e^a_t)
$$

feedback 预测目标：

$$
\hat z^o_{t+1}=G_o(\hat z^s_{t+1})
$$

state prediction loss：

$$
\mathcal{L}_{state}
= \mathbf{1}_{has\_next}
\left(1-\cos(\hat z^s_{t+1}, \text{sg}(z^s_{t+1}))\right)
$$

feedback prediction loss：

$$
\mathcal{L}_{feedback}
= 1-\cos(\hat z^o_{t+1}, \text{sg}(z^o_{t+1}))
$$

latent alignment loss：

$$
\mathcal{L}_{align}
= \mathbf{1}_{has\_next}
\left(1-\cos(z^s_{t+1}, \text{sg}(z^o_{t+1}))\right)
$$

SigReg：

$$
\mathcal{L}_{sigreg}
= \text{SIGReg}\left(
\left[z^s_t,z^s_{t+1},z^o_{t+1}\right]^\top
\right)
$$

value loss：

$$
\mathcal{L}_{value}
= \text{Huber}(\hat V_t,y_t)
$$

总 loss：

$$
\mathcal{L}_{total}
= \mathcal{L}_{DAPO}
+ \alpha(t)\left[
\lambda_s\mathcal{L}_{state}
+ \lambda_o\mathcal{L}_{feedback}
+ \lambda_a\mathcal{L}_{align}
+ \lambda_{sig}\mathcal{L}_{sigreg}
+ \lambda_v\mathcal{L}_{value}
\right]
$$

建议默认：

| 参数 | 建议值 | 理由 |
|---|---:|---|
| `WORLD_MODEL_ENABLED` | `0` 默认，ablation 开启 | 不影响 baseline |
| `WORLD_MODEL_DETACH_LLM_HIDDEN` | `1` | 首先验证 hidden 是否有可预测性，避免扰动 policy |
| `WORLD_MODEL_LOSS_COEF` | warmup `0 -> 0.01/0.05` | auxiliary loss 不应压过 RL |
| `WORLD_MODEL_STATE_COEF` | `1.0` | 主 next-state 预测 |
| `WORLD_MODEL_FEEDBACK_COEF` | `1.0` | 直接预测环境反馈语义 |
| `WORLD_MODEL_ALIGN_COEF` | `0.1` | 让 next-state latent 和 feedback latent 在同一空间对齐 |
| `WORLD_MODEL_SIGREG_COEF` | `0.03-0.09` | 接近 `le-wm` 的 `0.09`，但在线 RL 初期可更小 |
| `WORLD_MODEL_VALUE_COEF` | `0.05-0.1` | value 先轻量辅助 |
| `WORLD_MODEL_VALUE_TO_ADVANTAGE_COEF` | `0` | 首版不改 DAPO advantage |

说明：

- 如果所有 latent 做 L2 normalize，MSE 与 cosine loss 等价性更强；可以用 `le-wm` 风格的 MSE 作为 ablation。
- `pred_loss` 首版建议对 target 加 stop-gradient。离线预训练阶段可以更接近 `le-wm`，允许 adapter/projector 从 target branch 收梯度，但在线 RL 阶段建议固定或 EMA target，避免 moving target。
- SigReg 应加在 encoder/projector 产出的 latent 上，而不是原始 LLM hidden 上；这与 `le-wm/train.py:39-41` 正则 `emb` 的逻辑一致。

### 3.4 MuZero 式 value predictor

当前 baseline 没有 critic。DAPO/GRPO 的优势主要来自 group-normalized reward，reward postprocess 在 `slime/slime/ray/rollout.py:395-470`，GRPO returns/advantages 在 `slime/slime/backends/megatron_utils/loss.py:320-326`。

首版 value target 使用 Monte-Carlo final return：

$$
y_t = \text{raw\_score}
\quad \text{or} \quad
y_t = \text{base\_score}
$$

不建议首版使用 group-normalized reward，因为它依赖同组采样组合，不是稳定环境 value。

后续可加入 n-step/bootstrap：

$$
y_t^{(n)}
= \sum_{k=0}^{n-1}\gamma^k r_{t+k+1}
+ \gamma^n\text{sg}(V_{\bar\phi}(z^s_{t+n}))
$$

但 SETA 当前主要是 final evaluate score，没有天然 dense reward，所以 n-step 的收益取决于是否引入 step-wise PRM、安全分或探索内在奖励。

与 DAPO 的安全结合方式：

$$
A^{hybrid}_{i,t}
= A^{DAPO}_{i}
+ \eta\cdot\text{GroupNorm}_{group,turn}
\left(y_{i,t}-\text{sg}(\hat V_{i,t})\right)
$$

其中 \(\eta\) 必须从 0 warmup 到很小值，例如 `0.02`。只有当 heldout value correlation 稳定为正，才启用该项。

## 4. 训练阶段建议

### 4.1 Stage A：offline LEWM probe/pretrain

目标：验证 LLM hidden + shared latent projector 能否预测环境反馈。

流程：

1. 从 baseline trajectories 抽取 1k-5k turn transitions。
2. 重建 `input_ids`、`output_token_ids`、`next_input_ids`、`feedback_text`。
3. 用 frozen LLM 取：
   - `prompt_end` hidden 得到 \(u^h_t\)
   - action span mean/last hidden 得到 \(u^a_t\)
   - next prompt hidden 得到 \(u^h_{t+1}\)
4. 用 frozen text encoder 得到 \(r_{t+1}=T_{fix}(o_{t+1})\)。
5. 训练 `A_h/A_a/A_o/C/P/G_o/ValueHead`，LLM 和 text encoder 都冻结。
6. 记录：
   - `wm/state_pred_cos`
   - `wm/feedback_pred_cos`
   - `wm/feedback_retrieval_top1/top5`
   - `wm/effective_rank`
   - `wm/value_spearman_to_raw_score`

进入 online 的最低标准：

- feedback positive cosine 明显高于 in-batch negative。
- retrieval top-k 高于随机。
- effective rank 不坍塌。
- value Spearman 稳定为正，最好 \(>0.2\)。

### 4.2 Stage B：online auxiliary-only

目标：确认 world model loss 不破坏 DAPO。

配置：

- `WORLD_MODEL_ENABLED=1`
- `WORLD_MODEL_DETACH_LLM_HIDDEN=1`
- `WORLD_MODEL_VALUE_TO_ADVANTAGE_COEF=0`
- `WORLD_MODEL_LOSS_COEF` warmup `0 -> 0.01/0.05`

对照：

- baseline：`terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh`
- world-model auxiliary：新建 `terminal-rl_qwen3-8b_seta_dapo_nodynamic_worldmodel_aux_pu.sh`

观察：

- 主训练指标不应显著下降：`terminal/test_acc`、`terminal/reward/raw`、`train/pg_loss`、`train/ppo_kl`、`train/entropy_loss`。
- world-model 指标应改善：`wm/state_pred_loss`、`wm/feedback_pred_loss` 下降，`wm/pred_cos`、`wm/value_corr` 上升。

### 4.3 Stage C：value-assisted advantage 小权重 ablation

仅当 Stage B 成立后启用：

$$
A^{hybrid}
= A^{DAPO}
+ \eta\cdot\text{GroupNorm}(y-\hat V)
$$

建议：

- \(\eta=0.02\) 起。
- warmup 50-100 rollouts。
- 先只对 non-final turn 生效，避免 final evaluate label 泄漏式过拟合。

结果解释：

| 结果 | 结论 |
|---|---|
| auxiliary 指标好，RL 指标不变 | latent world model 可行；继续优化 value 使用方式 |
| auxiliary 指标好，RL 指标下降 | loss 或 advantage 耦合干扰 policy；保持 detach 或降低 coef |
| auxiliary 指标差 | hidden/feedback target 定义不足；需要更强 state summary 或更合适 text encoder |

## 5. 最小侵入式代码接入点

推荐新增/修改：

| 文件 | 修改 | 理由 |
|---|---|---|
| `terminal-rl/generate.py` | 在 turn record 和 `Sample` 上保存 `feedback_text`、`next_input_ids` 或 next-turn reference、`prompt_len`、`response_len`、`raw_score/base_score/status` | trajectory 已有文本，但 train_data 需要结构化字段 |
| `slime/slime/ray/rollout.py` | `_convert_samples_to_train_data()` 增加 `wm_feedback_embed`、`wm_has_next`、`wm_next_input_ids`、`wm_value_targets` 等字段 | Megatron/FSDP data iterator 只能安全消费显式 train_data 字段 |
| `slime/slime/backends/megatron_utils/data.py` | `get_batch()` keys 增加 world-model 字段 | 当前实际 run 走 Megatron backend |
| `slime/slime/backends/megatron_utils/model.py` | 可选暴露 hidden states，或增加 custom hook 捕获 prompt/action hidden | 现有 `model(**forward_kwargs)` 只把输出作为 logits 传给 loss |
| `slime/slime/backends/megatron_utils/loss.py` | 在 policy loss 后加 auxiliary LEWM loss，或通过 `custom_loss_function_path` 包装 | 保持 DAPO 主 loss 不变 |
| `slime/slime/utils/arguments.py` | 添加 `--world-model-*` 参数 | 支持 ablation |
| `terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_worldmodel_pu.sh` | 新建实验脚本 | 保证和 baseline 对照 |

实现细节：

- Megatron `DataIterator.get_next()` 会按 keys 切 micro-batch（`slime/slime/backends/megatron_utils/data.py:286-315`），world-model 字段应是 per-sample list/tensor，不应只塞 nested metadata。
- 字符串 feedback 不应直接进 GPU；首版建议离线或 rollout 侧预编码为 `wm_feedback_embed`。
- 若在线取 \(h_{t+1}\) 成本太高，首版可以只训练 feedback target，state target 在 offline probe 中验证。
- hidden 边界：
  - `prompt_end = len(input_ids) - 1`
  - `action_span = [len(input_ids), total_length - 1]`
  - `action_end = total_length - 1`
  - 当前 turn sample 不含 tool feedback，因此 action hidden 不泄漏 \(o_{t+1}\)。

## 6. 可行性结论

结论：支持采用 LLM-hidden-conditioned LEWM，而不是旧的 hidden-direct-regression 方案。

支持理由：

- `le-wm` 的关键归纳偏置是显式 latent、动作条件预测、SigReg 防坍塌；这些都可以迁移到 terminal-agent。
- 当前 rollout 已保存 turn 级 `h_t,a_t,o_{t+1}`，并且 sample tokens 不含下一步 tool feedback，预测任务边界干净。
- 通过 `A_h/A_o + shared C` 可以把 LLM hidden 与 feedback text encoder embedding 对齐到同一 latent 空间。
- `ARPredictor` 和 `SIGReg` 可以基本直接复用，工程上只需要新增 hidden/text embedding wrapper。

保留限制：

- \(z^s_t\) 是 belief latent，不是完整环境状态。终端文件系统的真实状态只通过历史工具反馈间接进入 context。
- policy LLM hidden 会随 RL 更新漂移；在线首版必须 detach hidden 或使用很小 loss coef。
- shared latent projector 需要 SigReg/对齐/stop-gradient，否则仍可能发生跨分支坍塌。

## 7. 风险与缓解

| 风险 | 表现 | 缓解 |
|---|---|---|
| 跨源 embedding 分布不匹配 | shared projector 难收敛，feedback/state latent 不对齐 | 源特定 adapter + shared projector；LayerNorm；alignment loss |
| 表征坍塌 | latent 方差趋近 0，effective rank 下降 | 复用 `SIGReg`；监控 rank；必要时加 VICReg variance/cov |
| target moving | online target 随 LLM 更新漂移 | MVP detach LLM hidden；feedback text encoder frozen；target branch stop-gradient/EMA |
| 目标泄漏 | 模型直接看到 tool feedback，预测任务变 trivial | 只用当前 turn `input_ids+output_token_ids` 取 predictor hidden；不要把下一轮 tool result 拼入 predictor input |
| final turn 缺 next state | \(z^s_{t+1}\) 缺失 | 用 `has_next` mask；final turn 只训练 feedback/value |
| value 估计误导 advantage | value corr 低但进入 policy | 首版 `VALUE_TO_ADVANTAGE_COEF=0`；heldout corr 达标后小权重 warmup |
| hidden 获取工程成本高 | Megatron forward 当前只返回 logits | 先 offline probe；online 只加可选 hidden return/custom hook |

## 8. 推荐下一步

1. 先实现 offline `HiddenLEWM` probe，复用 `SIGReg`、`ARPredictor`、`MLP`，不改训练主链路。
2. 若 probe 成立，做 online auxiliary-only：LLM hidden detach，feedback target frozen，DAPO loss 不变。
3. 第一次 online 实验只记录 world-model 指标，不改 reward/advantage。
4. value predictor 只有在 heldout correlation 稳定后，才以小权重进入 hybrid advantage。
