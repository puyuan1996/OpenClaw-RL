# Terminal-RL Latent World Model 实现与使用

> 状态：v2 已实现（2026-07-14）
> 权威实现：`slime/slime/world_model/`
> 历史设计：[`latent_world_model_design_20260616_154623_HKT.md`](latent_world_model_design_20260616_154623_HKT.md)

## 1. 当前实现解决什么问题

本实现从 SETA terminal-agent 的 turn 轨迹构造

\[
(h_t,a_t,o_{t+1},h_{t+1},r_t,d_t),
\]

用 policy LLM hidden 得到 state/action/feedback 表征，再在统一 latent space 预测动作后果：

\[
z^s_t=C(A_s(H(h_t))),\quad e^a_t=A_a(H(h_t+a_t)[a_t]),
\]

\[
z^o_{t+1}=C(A_o(H(o_{t+1}))),\quad
\hat z^o_{t+1}=P_{\text{AdaLN}}(z^s_t,e^a_t).
\]

它不替换 SETA verifier，也不默认修改 DAPO/GRPO policy loss。默认路径是离线训练或 shadow 诊断。

## 2. SETA 中 observation 与 action

- `h_t`：第 `t` 轮生成前的 `turns[t].context_messages`，即 agent belief observation。
- `a_t`：`assistant_output`，必要时补充解析后的 `tool_calls[*].tool_name/args`。
- `o_{t+1}`：本轮 `tool_calls[*].result`；无工具结果时使用终止状态与 verifier 分数摘要。
- `h_{t+1}`：下一轮 `context_messages`，其中已经包含工具反馈；final turn 没有该字段。
- `r_t`：优先使用 `reward.per_turn_scores[t].score`，否则回退到 trajectory `score/base_score/raw_score`。

一条实际形态：

```text
h_t: context_messages=[system, user: "检查当前目录"]
a_t: assistant_output="..." + bash({"command":"pwd"})
o_{t+1}: <tool_result name=bash>\n/tmp\n</tool_result>
h_{t+1}: [system, user, assistant/tool-call, tool-result]
```

适配器支持 SETA 原始 `*/traj.json`、world-model records JSONL 和 DAPO world-model replay `.pt`。

## 3. hidden 与 latent 如何得到

HF policy 路径在一次 causal forward 中编码 `h_t+a_t`：

```text
prompt tokens | action tokens
      ^              ^
 prompt_end      action_span
      |              |
 state hidden    pooled action hidden
```

- state hidden：`hidden[prompt_end]`；该位置因 causal mask 看不到 action。
- action hidden：`hidden[action_span]` 的 mean 或 last pooling；它可看到 state 与已生成 action。
- feedback hidden：无梯度 target forward 编码 `<environment_observation> + o_{t+1}`。
- next-state hidden：无梯度 target forward 取 `h_{t+1}` 的 prompt-end；仅在 `has_next=true` 时参加 alignment。
- `hidden_layer` 可配置，默认 `-1`；离线 HF 可安全选中间层。

raw hidden 不直接做 MSE。state 和 feedback 先经过源特定 adapter，再通过共享 projector `C`；action 经独立 projector 得到条件向量。
target forward 与 current branch 使用同一 policy checkpoint，但计算图始终 detached；它不是另行加载的独立文本 encoder。开启 backbone 更新时，target geometry 会随 policy 参数更新，因此不应称为固定 EMA teacher。

## 4. action 如何与 observation latent 融合

当前默认 predictor 是 LeWM 风格的 action-conditioned AdaLN Transformer：

```text
z_state tokens ───────────────> self-attention Q/K/V ──> predicted feedback latent
                                      ▲
e_action ─> SiLU + Linear ─> shift / scale / residual gates（每层）
```

因此 action：

- 不作为独立 token 进入 self-attention；
- 不与 state 做 `torch.cat([state, action], dim=-1)`；
- 只生成每层 attention/MLP 的 AdaLN shift、scale 和残差 gate。

`--predictor-type mlp` 保留旧 concat-MLP，仅用于兼容和 ablation；默认是 `adaln`。

## 5. 整体数据流

```text
SETA traj.json / replay.pt
        |
        v
TerminalTransition(h_t, a_t, o_t+1, h_t+1, reward, done)
        |
        +--> policy forward(h_t+a_t)
        |       +--> prompt_end hidden --> state adapter --+
        |       +--> action_span hidden --> action adapter  |
        |                                                  v
        +--> no-grad target forward(o_t+1) --> feedback adapter --> shared latent C
        +--> no-grad target forward(h_t+1) --> state adapter ----> shared latent C
                                                           |
state latent ------------------------------------------> Q/K/V
action latent --> per-layer AdaLN ---------------------> predictor
                                                           |
                                                           v
                                               predicted feedback latent
                                                           |
                                      prediction / contrast / value diagnostics
```

核心 loss：

\[
L=L_{pred}+\lambda_{sig}L_{SIGReg}+\lambda_{cf}L_{action\ contrast}
+\lambda_{align}L_{next/feedback}+\lambda_vL_{value}.
\]

- `L_pred`：预测 feedback latent；
- `L_action contrast`：真实 action 应优于 batch 内 shuffled action，防止 predictor 忽略动作；
- `SIGReg`：只约束 state latent，防止低秩/常数坍塌；
- `L_align`：有下一轮时对齐 next-state latent 与 feedback latent；
- `L_value`：可选 Smooth-L1；仅当 `--value-coef` 非零时创建并训练 value head，默认不创建。

## 6. 两个开关

### 6.1 是否回传 LLM backbone

CLI：`--backprop-to-llm`，在 slime 总参数中对应 `--world-model-backprop-to-llm`。

默认 `false`。此时 LLM forward 使用 `no_grad`，hidden 被缓存，只训练 adapter/projector/predictor/head；显存占用低、target geometry 稳定，适合第一阶段。

开启后，state/action hidden 保留计算图，optimizer 以独立 `--llm-lr` 更新 backbone；feedback/next-state 仍是 detached target。该模式显著增加激活与优化器显存。若要持久化更新后的 HF backbone，另加 `--save-updated-llm`。
若不开 `--save-updated-llm`，world-model checkpoint 不包含更新后的 8B backbone，不能单独复现该次端到端训练的 hidden geometry。

### 6.2 是否使用 DAPO replay buffer

离线 CLI：`--use-dapo-replay-buffer`；DAPO 总参数：`--world-model-use-dapo-replay-buffer`，默认 `false`。

启用后：

1. rollout 已附加的 world-model record 进入 `TrajectoryReplayBuffer`；
2. buffer 采用固定容量 FIFO、去重、随机 sample；成功与失败默认都入库；
3. checkpoint 保存到 `${SAVE}/rollout/world_model_replay_<rollout_id>.pt`；
4. `train_latent.py --input <replay.pt>` 可直接训练。

接口沿用本地 PR #16 的 `push(entries, current_step)`、`sample(n, current_step, baseline_reward)` 形状，但不复用其 SIL 成功阈值：world model 需要完整 outcome 分布。参考来源为本地提交 `cffc63c6`，无需网络访问。

## 7. 使用

### 7.1 用指定 SETA 轨迹跑一次 smoke

```bash
WM_USE_DAPO_REPLAY_BUFFER=1 \
WM_MAX_TRAJECTORIES=2 \
WM_MAX_TRANSITIONS=8 \
WM_EPOCHS=1 \
terminal-rl/scripts/run_world_model_seta_latent.sh
```

默认 `hash` hidden 仅验证数据/replay/AdaLN/loss/prediction 闭环。输出包含 `hidden_cache.pt`、可选 `dapo_replay.pt`、`latent_world_model.pt`、`predictions.jsonl` 和 `run_summary.json`。

### 7.2 冻结 Qwen3-8B hidden（推荐正式起点）

```bash
WM_ENCODER=hf-policy \
WM_HF_MODEL=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B \
WM_MAX_TRAJECTORIES=100 \
WM_MAX_TRANSITIONS=1000 \
WM_OUTPUT_DIR=runs/world_model_seta_latent/qwen_frozen \
terminal-rl/scripts/run_world_model_seta_latent.sh
```

### 7.3 开启 LLM 梯度

```bash
WM_ENCODER=hf-policy \
WM_BACKPROP_TO_LLM=1 \
WM_SAVE_UPDATED_LLM=1 \
WM_LLM_LR=1e-6 \
WM_HF_MODEL=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B \
WM_OUTPUT_DIR=runs/world_model_seta_latent/qwen_e2e \
terminal-rl/scripts/run_world_model_seta_latent.sh
```

建议先减小 batch/context，并使用 `--llm-lr 1e-6` 或更低。8B 全参数端到端训练需要明显多于冻结模式的显存。

### 7.4 DAPO 采集 replay

```bash
EXTRA_ALGO_ARGS="--world-model-enable \
  --world-model-use-dapo-replay-buffer \
  --world-model-replay-buffer-size 4096" \
bash terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh
```

这只收集/保存 world-model replay，不开启辅助 policy loss。现有在线 default loss hook 仍只消费显式提供的 `wm_pred_latents/wm_target_latents`。

## 8. ECHO、Qwen-AgentWorld 与 LeWM 的借鉴边界

- ECHO：借鉴 observation 定义、terminal prompt/role 兼容检查和“环境反馈是免费监督”的观点；ECHO 自身优化 observation-token CE，本实现预测连续 latent。
- Qwen-AgentWorld：借鉴 terminal simulator 的“历史上下文 + 当前状态 + action -> 完整 next observation”边界，以及 Format/Factuality/Consistency/Realism/Quality 评估维度。本实现不把其长 system prompt 注入 policy hidden：SETA 已保存真实 agent context，latent predictor 也不生成 observation 文本；该 prompt 更适合作为后续离散生成 baseline 或 latent decoder 的模板。
- LeWM：借鉴 joint-embedding prediction、SIGReg、action-conditioned AdaLN；把 pixel encoder 替换为 policy LLM span hidden。
- 本实现的差异：显式 state/action/feedback latent、可查询 predictor、shuffled-action 约束、SETA replay 适配。潜在优势是候选 action 可在 latent 中并行评分；该优势仍需真实同-state candidate 执行实验验证，不能由 smoke loss 直接推出。

## 9. 已知边界

- SETA `traj.json` 没保存原始 `input_ids`，HF 路径用同 tokenizer/chat template 重建 token 边界，不保证与历史 rollout bitwise 一致。
- Megatron 在线训练仍未默认暴露 middle-layer hidden；当前端到端 backbone 梯度入口是独立 HF latent trainer。
- hash smoke 没有语义结论；正式结论必须使用 policy hidden，并报告 shuffled-action gap、effective rank、heldout prediction 与 reward calibration。
- world-model value 尚未接管 GRPO advantage；默认保持 DAPO critic-free 行为。

## 10. 代码索引

| 文件 | 作用 |
| --- | --- |
| `slime/slime/world_model/seta_dataset.py` | SETA/raw records/replay 数据适配 |
| `slime/slime/world_model/hidden_encoder.py` | prompt-end/action-span/target hidden 提取 |
| `slime/slime/world_model/modules.py` | shared latent、AdaLN predictor、loss/head |
| `slime/slime/world_model/replay_buffer.py` | DAPO world-model replay |
| `slime/slime/world_model/train_latent.py` | 训练、预测和 checkpoint 入口 |
| `terminal-rl/scripts/run_world_model_seta_latent.sh` | 指定 SETA 轨迹的一键入口 |
