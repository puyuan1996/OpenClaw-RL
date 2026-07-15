# Terminal-Agent Latent World Model 调研摘要

> 本文由原 `rl_doc/survey-notes.md` 与阶段报告精简迁移；实现事实以 [`latent_world_model_guide_zh.md`](latent_world_model_guide_zh.md) 为准。

## 1. 研究定位

terminal agent 的真实环境交互昂贵、反馈长且奖励稀疏。连续 latent predictor 的主要研究价值不是替代 verifier，而是：

1. 学习 action-conditioned environment consequence；
2. 为同 state 多候选 action 提供低成本排序信号；
3. 作为 DAPO 的辅助表征目标或后续 value/control-variate 输入。

“latent dynamics + value/reward head”本身已有 MuZero、TD-MPC 等先例。可验证的差异应落在 terminal policy-hidden、动作敏感性和执行前候选筛选。

## 2. 最近邻

| 工作 | 可借鉴点 | 与本实现的边界 |
| --- | --- | --- |
| ECHO (2605.24517) | terminal observation token、环境 CE 辅助目标、prompt/role 处理 | 预测离散 observation token；没有独立 latent consequence predictor |
| Qwen-AgentWorld (2606.24597) | history/state/action 序列化、terminal simulator prompt、五维 next-state 评估 | 原模型生成完整 observation；本实现直接预测连续 latent，不注入其长 simulator prompt |
| LeWM (2603.19312) | JEPA、SIGReg、action-conditioned AdaLN | 原输入是 pixel/action；本实现替换为 LLM span hidden |
| RWML / PriorZero | agent world model、credit assignment | 架构和训练域不同；需实验而非文字证明优势 |
| SPEAR / PR #16 | 成功轨迹 replay、P50 advantage、off-policy 接口 | world model replay 默认保留成功和失败，不直接做 SIL policy update |

本地实现依据：`/mnt/shared-storage-user/puyuan/code/echo-rl/echo_rl/`、`/mnt/shared-storage-user/puyuan/code/Qwen-AgentWorld/prompts/terminal/system_prompt.txt`、`/mnt/shared-storage-user/puyuan/code/le-wm/`；replay 接口依据本地提交 `cffc63c6`。

## 3. 必须报告的证据

- real action prediction loss 与 shuffled/zero action 对照；
- action 替换后的 latent delta；
- state latent effective rank/variance；
- heldout prediction/retrieval；
- value 与 verifier reward 的相关和校准；
- 同 state 多候选真实执行的 top-1、random、oracle 对比。

低 prediction loss 不足以证明模型依赖 action；SIGReg 防坍缩也不等于 uncertainty 已校准。

## 4. Replay 原则

- world model 应覆盖成功、失败、异常工具输出和不同 task；只用 SIL 成功轨迹会造成乐观偏差。
- 记录 trajectory/turn 顺序，避免丢失 `h_{t+1}`。
- policy hidden 应尽量由同一冻结快照重算，减少跨 policy-version geometry drift。
- 在线更新时需监控 staleness，并考虑 EMA/re-anchor；这与防坍缩是不同问题。

## 5. 安全边界

- verifier 仍提供末端 ground truth；不使用 world-model reward 替代真实单测。
- value 接入 GRPO advantage 前先做 heldout gate，默认系数为零。
- candidate screening 需要低置信回退真执行，不能把 latent distance 直接当作已校准概率。

## 6. 历史阶段结论

2026-06-26 的 v1 probe 已打通 metadata、text/hash cache、MLP predictor、Stage-A 与候选 ranking 工具，但当时没有 policy span hidden、AdaLN predictor 或 DAPO WM replay。v2 补齐这些工程路径；历史候选排序正信号仍只算离线诊断，不代表在线收益。
