# 真实交互式对话测试 - 完整指南

## 🎯 功能说明

与正在训练的 Qwen3-4B 模型实时对话，通过反馈（👍/👎）让模型实时学习和改进。

**工作原理：**
1. 你的每次对话都会发送到 RL 训练服务器
2. 给出反馈（👍/👎）后，系统会：
   - 运行 PRM 评估（+1/-1 奖励）
   - 提取 hint 用于 OPD 蒸馏
   - 生成训练样本
3. 累积 16 个样本后自动触发一次模型更新
4. 更新后的模型会在下次对话中使用

## 🚀 快速启动

### 方法 1：一键启动（推荐）

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/openclaw-combine
bash start_chat.sh
```

### 方法 2：直接运行

```bash
python simple_chat.py
```

## 💬 使用说明

### 界面示例

```
============================================================
  Interactive Chat with RL Training Model
============================================================

Server: http://0.0.0.0:30000

Commands:
  👍 or +    → Positive feedback (模型回答得好)
  👎 or -    → Negative feedback (模型回答不好)
  /new       → New session (开始新对话)
  /quit      → Exit (退出)

============================================================

✓ Connected
Session: chat_a3f8b2c1

You: _
```

### 对话流程示例

```
You: 什么是机器学习？