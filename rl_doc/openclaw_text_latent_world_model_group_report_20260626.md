# OpenClaw Text Latent World Model 阶段总结

> 日期：2026-06-26
> 范围：`slime/slime/world_model/`、`terminal-rl` world-model metadata 接入、离线 Stage-A/P2 评估
> 定位：默认关闭的 JEPA-style text latent world model，用于 terminal-agent RL 的离线验证和未来 shadow candidate screening。

## 1. 目标

本阶段目标不是替换 verifier 或 reward model，而是验证：

1. LLM/text hidden 是否可以经过受控 projector/alignment 层，进入统一的 belief latent space。
2. 该 latent 是否能支持 action-conditioned next-state prediction。
3. 该 latent 是否能为 agentic-RL 中的 candidate action 排序、execution result prediction 提供可测信号。
4. 工程接入是否保持默认 no-op，不影响现有 RL 训练主路径。

## 2. 已实现内容

核心实现位于 `slime/slime/world_model/`：

| 模块 | 作用 |
| --- | --- |
| `metadata.py` | 从 terminal rollout 附加 state/action/next observation/reward/tool-result metadata |
| `build_dataset.py` | 从 debug rollout `.pt` 抽取 world-model `records.jsonl` |
| `cache_text_hidden.py` | 将 text records 转为 `state_hidden/action_hidden/target_hidden` cache |
| `modules.py` | JEPA-style projector、action-conditioned predictor、SIGReg、value/uncertainty heads |
| `train_probe.py` | 训练离线 world-model probe |
| `evaluate_probe.py` | Stage-A action-sensitivity、collapse、reward calibration 诊断 |
| `rank_candidates.py` | 执行前可用的 value/uncertainty candidate ranking interface |
| `candidate_set_eval.py` | 同 context 多候选离线 P2 eval |
| `loss_hook.py` | 在线 auxiliary loss 的默认关闭 hook 边界 |

训练接入点：

| 文件 | 说明 |
| --- | --- |
| `terminal-rl/generate.py` | 可选记录 world-model metadata |
| `slime/slime/ray/rollout.py` | 可选携带 `wm_*` fields |
| `slime/slime/backends/megatron_utils/model.py` | 可选 batch keys |
| `slime/slime/backends/megatron_utils/loss.py` | 可选 auxiliary loss hook |
| `slime/slime/backends/megatron_utils/data.py` | 跳过 `wm_*` metadata 聚合 |
| `slime/slime/utils/arguments.py` | 增加 `--world-model-*` 参数 |

## 3. 保留的通用脚本

本 PR 只保留可复用入口，删除一次性 H20/H200、overnight、topup、heldout、recovery 包装脚本。

| 脚本 | 用途 |
| --- | --- |
| `terminal-rl/scripts/run_world_model_seta_smoke.sh` | 采集 SETA rollout metadata；默认复用目标分支已有 SETA DAPO wrapper，可用 `WM_TRAIN_SCRIPT` 覆盖 |
| `terminal-rl/scripts/run_world_model_offline_probe_smoke.sh` | 使用 `hash` encoder 快速验证 dataset/cache/train/ranking 闭环 |
| `terminal-rl/scripts/run_world_model_batch_probe.sh` | 从多个 rollout 或 records 批量构建 cache 并训练 probe |
| `terminal-rl/scripts/run_world_model_stage_a_eval.sh` | full/clean/tool-only bucket 的 Stage-A 诊断 |
| `terminal-rl/scripts/run_world_model_p2_candidate_set_eval.sh` | 对同 context candidate set 做离线 ranking eval |

默认安全策略：

- `WM_ENCODER=hash`，只验证链路，不默认加载大模型。
- 使用 HF/LLM encoder 必须显式设置 `WM_ENCODER=hf`、`WM_ALLOW_HF=1`、`WM_HF_MODEL=/path/to/model`。
- P2 candidate-set eval 必须显式提供 `WM_P2_BASE_EXP` 或 `WM_P2_RECORDS/WM_P2_CACHE/WM_P2_CHECKPOINT`。
- 所有实验输出写入 `runs/`，不会进入提交。

## 4. 阶段实验结论

### 4.1 Stage-A 数据规模

已从 1-16 条 smoke 样本扩展到千级 records，并完成 full/clean/tool-only bucket 验证。

| Bucket | Records |
| --- | ---: |
| full | 2878 |
| clean | 472 |
| tool_only | 2591 |

结论：数据 schema、hidden cache、probe training、Stage-A 诊断和 summary 聚合已经闭环；clean bucket 规模仍偏小，需要继续积累高质量 replay-buffer 数据。

### 4.2 P2 candidate-set ranking

离线 same-context candidate-set eval 已出现稳定正信号。

| 指标 | 当前最好结果 |
| --- | ---: |
| `WM - random` | +0.3058 |
| `Spearman(score, reward)` | 0.6538 |
| `hit_oracle` | 0.8333 |

结论：当前 world model 已能在离线 candidate-set 中学习到部分 action quality ordering，但这仍不是 online selector 证据。

### 4.3 Heldout / robustness

Repeated heldout recovery gate 中，3 个 clean seeds 均保持正收益。

| 指标 | 结果 |
| --- | ---: |
| `WM - random mean` | 0.2908 |
| `WM - random min` | 0.2826 |
| `hit_oracle mean` | 0.7234 |

结论：结果不是单一 seed 偶然性，但仍需要更严格的 real-execution shadow eval。

### 4.4 Tool-use / execution-result diagnostic

CPU 线性 probe diagnostic 显示：

| 任务 | 结论 |
| --- | --- |
| `pred_latent -> reward_bin` | 有初步正信号；clean split 上明显超过 majority baseline |
| `state_latent -> tool-use/tool-name` | 尚未稳定超过 majority baseline |

代表性结果：

| Split | Probe | Accuracy | Majority |
| --- | --- | ---: | ---: |
| full | `pred_latent -> reward_bin` | 0.8278 | 0.8167 |
| clean | `pred_latent -> reward_bin` | 0.7034 | 0.5339 |

结论：execution result prediction 比 tool-use prediction 更接近当前 latent 的能力边界；tool-use 需要更结构化、更均衡的 labels。

## 5. 工程安全性

- 默认 `world_model_enable=False`，不改变现有训练行为。
- `world_model_loss_coef=0.0` 时 auxiliary hook 为 no-op。
- `rank_candidates.py` 默认使用 execution-time 可用的 value/uncertainty score；只有显式 `--score-mode pred_error` 才使用 target prediction error 做 oracle diagnostic。
- `candidate_set_eval.py` 会过滤缺失或非法 `reward_score` 的候选，避免 replay 数据中部分 missing reward 导致崩溃。
- `target_projector` 默认可学习，并保留 `stop_grad_target` 配置供后续 EMA/frozen-target 实验。

## 6. 当前不足

1. 仍是 offline replay/candidate-set eval，不是在线决策收益。
2. clean records 规模仍有限，reward/task 分布不够均衡。
3. tool-use labels 目前较稀疏，majority baseline 偏强。
4. hidden cache 仍依赖 frozen encoder/offline extraction，v1 不直接在线抓 Megatron middle-layer hidden。
5. P2 positive signal 不等价于 P2b shadow real-execution 成功。

## 7. 下一步

优先级建议：

1. **P2b shadow real-execution candidate screening**：同一 state/context 生成多个候选 action，全部真实执行，world model 只做执行前排序，事后评估 `WM top1` vs `random` vs `oracle`。
2. **Replay-buffer 数据扩展**：提高 clean/tool-result records 覆盖，增加 task/reward 多样性。
3. **Tool-use labels 结构化**：抽取 tool-call/tool-name/tool-result/status，做均衡 split。
4. **Value/uncertainty calibration**：将 value head 与 uncertainty head 的排序稳定性作为 P2b gate。
5. **Auxiliary online hook 小流量 shadow**：保持 `world_model_loss_coef=0` 到完成 P2b 证据后再打开。

## 8. 验证命令

本阶段 PR 验证：

```bash
PYTHONPATH=slime python -m pytest slime/tests/world_model -q

python -m py_compile \
  slime/slime/backends/megatron_utils/data.py \
  slime/slime/backends/megatron_utils/loss.py \
  slime/slime/backends/megatron_utils/model.py \
  slime/slime/ray/rollout.py \
  slime/slime/utils/arguments.py \
  terminal-rl/generate.py \
  slime/slime/world_model/modules.py \
  slime/slime/world_model/candidate_set_eval.py \
  slime/slime/world_model/rank_candidates.py

for f in terminal-rl/scripts/*world_model*.sh; do
  bash -n "$f"
done
```

通过标准：

- world-model unit tests 通过。
- shell scripts 语法检查通过。
- 无 `runs/`、`build_outputs/`、模型权重、日志、cache、nested `le-wm/` 提交。
