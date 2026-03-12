# OpenClaw-Combine 反馈机制分析

## 核心发现：你没有直接传递反馈！

**重要结论：** 你的 `👍` 或 `👎` 反馈**不是直接传递给训练系统**的，而是作为**下一轮用户消息（next state）**被系统自动分析和评分的。

## 完整工作流程

### 1. 对话序列

```
Turn 1: User → "2+2等于几？"
        Assistant → "2+2等于4"

Turn 2: User → "👍 Thanks!"  ← 这是 next_state
        (系统此时触发 Turn 1 的评估)
```

### 2. 自动评估流程

当 Turn 2 的用户消息到达时（第738-741行）：

```python
if turn_type == "main":
    prev_turn_num = self._turn_counts.get(session_id, 0)
    if prev_turn_num > 0 and messages:
        # 1. 保存 Turn 1 的 next_state (Turn 2 的用户消息)
        self._flush_pending_record(session_id, messages[-1])

        # 2. 触发 Turn 1 的评估任务
        prev_turn_data = self._pending_turn_data.get(session_id, {}).get(prev_turn_num)
        if prev_turn_data is not None:
            self._fire_opd_task(session_id, prev_turn_num, prev_turn_data, messages[-1])
```

**关键点：** `messages[-1]` 就是 Turn 2 的用户消息（例如 "👍 Thanks!"）

### 3. PRM 自动打分

评估任务 `_opd_evaluate()` 会：

**步骤 A：构建 PRM 评估 Prompt（第580-586行）**

```python
eval_msgs = _build_prm_eval_prompt(
    turn_data["response_text"],  # Turn 1 的模型回答
    next_state_text,              # Turn 2 的用户消息 (你的反馈)
    next_state_role               # "user"
)
```

**步骤 B：PRM Prompt 内容（第141-183行）**

```python
system = """
You are a process reward model (PRM) evaluating an AI assistant.
...
## Scoring rules
- \\boxed{1} (good): The next state shows the task progressed as expected —
  e.g. the user moves on, says thanks, ...

- \\boxed{-1} (bad): The next state signals the assistant's output was wrong,
  incomplete, or unwanted. **Key negative signals include:**
  * The user asks the assistant to **redo, retry, or repeat**
  * The user requests a **correction or modification**
  * The user **rephrases or restates** the same request
  ...
"""

user = f"""
## Assistant output
{response_text}    # "2+2等于4"

## Next state [role: user]
{next_state_text}  # "👍 Thanks!"

First, classify the next state: is it (a) positive progression,
(b) a correction / redo / change request, or (c) ambiguous?
Then assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}.
"""
```

**步骤 C：PRM 模型推理**

PRM 模型（也是 Qwen3-4B）会：
1. 阅读模型回答 + 用户反馈
2. 判断用户的反应是正面还是负面
3. 输出 `\boxed{1}` 或 `\boxed{-1}` 或 `\boxed{0}`

**步骤 D：多数投票（第196-204行）**

```python
def _prm_eval_majority_vote(scores: list[int | None]) -> float:
    # 运行 m 次（默认 m=1）
    valid = [s for s in scores if s is not None]
    counter = collections.Counter(valid)
    # 返回最高票数的分数
    return float(top[0])  # 1.0 或 -1.0 或 0.0
```

### 4. PRM 如何判断你的反馈

| 你输入的消息 | PRM 看到的 next_state | PRM 可能的判断 |
|------------|---------------------|---------------|
| `👍` 或 `Thanks!` | "👍 Thanks!" | `\boxed{1}` - 正面 |
| `👎` 或 `Not right` | "👎 Not right." | `\boxed{-1}` - 负面 |
| `Can you try again?` | "Can you try again?" | `\boxed{-1}` - redo 请求 |
| `What is Python?` | "What is Python?" | `\boxed{0}` - 新话题，无关 |

**关键洞察：** PRM 是通过**语义理解**来判断用户反应的好坏，而不是依赖特殊标记！

## 为什么你的测试脚本有效？

回到你的测试脚本 `simple_chat.py:feedback()`:

```python
def feedback(self, is_good):
    if is_good:
        msg = "👍 Good!"
    else:
        msg = "👎 Not quite right."

    # 发送反馈消息（创建 next_state）
    self.client.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "messages": self.messages + [{"role": "user", "content": msg}],
            ...
        },
        headers={"X-Session-ID": self.session_id},
    )
```

**工作原理：**
1. 发送 `"👍 Good!"` 作为下一条用户消息
2. API 服务器触发对上一轮的评估
3. PRM 看到 "👍 Good!"，理解为正面反馈
4. 输出 `\boxed{1}`，奖励 = +1.0
5. 生成训练样本并提交

## 对比：显式 vs 隐式反馈

### 显式反馈（你的方式）
```
User: 2+2等于几？
Assistant: 2+2等于4
User: 👍 Good!  ← PRM 分析这条消息，判断为正面
→ 训练样本 reward=+1.0
```

### 隐式反馈（README 中的场景）
```
User: 帮我写个排序函数
Assistant: [给出代码]
User: 这个代码有bug，请修复  ← PRM 分析，判断为负面（correction）
→ 训练样本 reward=-1.0
```

### 自然对话反馈（最真实）
```
User: 解释一下量子计算
Assistant: [给出解释]
User: 很清楚，我还想问...  ← PRM 判断为正面（继续话题）
→ 训练样本 reward=+1.0
```

## PRM 评分的智能性

PRM 模型会识别各种模式：

| Next State 类型 | 示例 | PRM 判断 |
|----------------|------|---------|
| **感谢** | "Thanks!", "谢谢", "好的" | +1 |
| **继续话题** | "那么接下来...", "我还想问..." | +1 |
| **重复请求** | "再说一遍", "你能再试试吗" | -1 |
| **纠正** | "不对", "应该是...", "你错了" | -1 |
| **修改请求** | "改成...", "换个方式" | -1 |
| **新话题** | 完全无关的新问题 | 0 |

## 代码位置索引

| 功能 | 文件:行号 |
|-----|----------|
| 触发评估 | `openclaw_opd_api_server.py:738-741` |
| 构建 PRM Prompt | `openclaw_opd_api_server.py:141-183` |
| PRM 推理 | `openclaw_opd_api_server.py:429-457` |
| 解析分数 | `openclaw_opd_api_server.py:186-193` |
| 多数投票 | `openclaw_opd_api_server.py:196-204` |
| 评估流程 | `openclaw_opd_api_server.py:568-672` |

## 总结

### ✅ 你理解对了
- 系统确实是**自动判断**反馈好坏
- **不需要**显式传递 "+1" 或 "-1" 标签
- 你的对话自然流转就能触发训练

### ✅ 你的测试脚本设计正确
- 发送 "👍 Good!" 作为用户消息
- PRM 自动识别为正面反馈
- 生成 reward=+1.0 的训练样本

### 💡 设计哲学
这种设计非常巧妙：
1. **无需额外标注**：用户的自然反应就是反馈
2. **灵活理解**：PRM 能理解各种表达方式
3. **真实场景**：符合实际对话中的反馈模式

### 🎯 最佳实践
对于测试，你可以使用任何自然的反馈表达：
- 正面：`"好的"`, `"谢谢"`, `"很有帮助"`, `"👍"`
- 负面：`"不对"`, `"再试试"`, `"应该是..."`, `"👎"`
- 中性：完全不相关的新话题

PRM 会自动理解并打分！
