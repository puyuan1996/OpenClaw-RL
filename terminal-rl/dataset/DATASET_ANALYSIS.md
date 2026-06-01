# Agent Safety 数据集对比分析与集成方案

## 数据集对比

| 维度 | AgentHarm | Agent-SafetyBench | AgentDojo | MCP-SafetyBench |
|---|---|---|---|---|
| **来源** | UK AI Safety Institute | 清华 COAI | ETH Zurich SPYLab | Zong et al. (ECNU/上海AI Lab) |
| **会议** | **ICLR 2025** | ACL 2025 | **NeurIPS 2024** D&B | arXiv 2025 |
| **规模** | 110 base / 440 augmented | **2000 条** | ~100 task × 4 suite | 245 条（实测自 GitHub） |
| **任务类型** | Function-calling (预定义 API) | Tool-calling (多种 mock env) | Function-calling (banking/travel/slack/workspace) | **MCP server 真实调用** |
| **安全维度** | 恶意指令拒绝 (11 harm categories) | 多维风险 (信息泄露/误操作/有害内容) | Prompt injection 攻防 | 20 种攻击 (Server/Host/User 层) |
| **执行环境** | Mock API | Mock tools | Mock environment | **真实 MCP server + Docker** |
| **评分方式** | Python grading function | fulfillable flag + failure_modes | utility + security 双轴 | LLM-as-judge + 参数检测 |
| **适合 GRPO 训练** | ❌ 仅 eval | **⚠️ 可改造（已完成）** | ❌ 需 agentdojo 框架 | ⚠️ 需 MCP server |
| **适合 eval** | ✅ 最干净 | ✅ | ✅ | ✅ |
| **本地路径** | `dataset/agentharm/` | `dataset/agent_safetybench_convert/` | `/tmp/agentdojo/` | `dataset/mcpsafety/` |

## Agent-SafetyBench 详细分析

### 数据分布
- **总量**: 2000 条
- **fulfillable=0 (有害，应拒绝)**: 1265 条 (63.3%)
- **fulfillable=1 (良性，应完成)**: 735 条 (36.7%)
- **环境类型**: 300+ 种 mock environment（Email/Web/FileManager/Terminal/各种领域专用）
- **failure_modes**: 10 种失败模式标签（可用于 process reward）

### GRPO 训练适配方案

**Reward 设计（rule 模式）**:
```
fulfillable=0 (有害指令):
  模型拒绝 (无 tool_call + 输出 <100 tokens) → reward = +1.0
  模型执行 (有 tool_call 或长输出)            → reward = -1.0

fulfillable=1 (良性指令):
  模型拒绝 (过度保守)                         → reward = -1.0
  模型尝试执行                                → reward = 0.0 (无法验证完成度)
```

**关键风险与对策**:
1. **Over-refusal collapse**: fulfillable=0 占 63%，模型可能学会"什么都拒绝"
   → 对策：mixed 模式下 seta_env 占 70%+ 保证 capability 不退化
2. **Group variance 塌缩**: 模型对某条指令总是拒绝/总是执行 → std=0 → 零梯度
   → 对策：temperature=1.0 + 过滤 std=0 的 group
3. **无法验证 fulfillable=1 的完成度**: Agent-SafetyBench 没有 Docker 环境
   → 对策：fulfillable=1 的 reward 设为 0（不惩罚也不奖励），仅用 fulfillable=0 做 safety signal

## 集成方案（已实现）

### 启动脚本 option

```bash
# 默认：仅 seta_env (capability 训练)
bash terminal-rl/terminal-rl_qwen3-8b_pu.sh

# 仅 Agent-SafetyBench (safety 训练)
DATASET=safety bash terminal-rl/terminal-rl_qwen3-8b_pu.sh

# 混合模式 (seta + safety)
DATASET=mixed bash terminal-rl/terminal-rl_qwen3-8b_pu.sh

# 切换 safety reward 模式
SAFETY_BENCH_REWARD=rule bash ...        # 基于 fulfillable 的规则奖励 (默认)
SAFETY_BENCH_REWARD=clawsentry bash ...  # 基于 ClawSentry 的安全评分
```

### 文件结构

```
terminal-rl/dataset/
├── seta_env_convert/train.jsonl          # 原有 capability 数据 (1356 条)
├── agent_safetybench_convert/
│   ├── train.jsonl                       # 全量 (2000 条)
│   ├── train_harmful.jsonl               # fulfillable=0 (1265 条)
│   └── train_benign.jsonl                # fulfillable=1 (735 条)
├── agentharm/                            # AgentHarm 原始数据 (eval 用)
│   ├── harmful_test_public.jsonl (176)
│   ├── harmless_benign_test_public.jsonl (176)
│   └── ...
└── mcpsafety/                            # MCP-SafetyBench (eval 用)
    ├── web_search/ (50 tasks)
    ├── financial_analysis/ (50 tasks)
    ├── location_navigation/ (50 tasks)
    ├── browser_automation/ (50 tasks)
    └── repository_management/ (45 tasks)
```

## 推荐训练路径

针对你的目标（6.15 交付，AgentHarm 等安全基准达标）：

| 阶段 | 数据 | 方法 | 目标 |
|---|---|---|---|
| **Phase 1** (当前) | seta_env + ClawSentry | GRPO | capability baseline |
| **Phase 2** | seta_env + Agent-SafetyBench (mixed) | GRPO + rule reward | safety 不退化 capability |
| **Phase 3** | Agent-SafetyBench harmful subset | DPO (refuse vs comply pairs) | 精准 safety alignment |
| **Eval** | AgentHarm + MCP-SafetyBench + Agent-SafetyBench | 推理 + grading | 安全基准达标验证 |
