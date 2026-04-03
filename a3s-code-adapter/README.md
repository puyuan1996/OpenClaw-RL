# a3s-code-adapter for OpenClaw-RL

从 `a3s-code-rl` 迁移的 a3s-code Agent 数据采集适配层，用于自动生成 RL 训练样本。

## 功能

通过真实的 a3s-code Agent 会话自动生成训练样本，替代手动交互。

## 目录结构

```
a3s-code-adapter/
├── code_rl_api_server.py           # RL Proxy，拦截请求并记录样本
├── code_rl_rollout.py              # Rollout 接口，管理样本队列
├── a3s_code_agent_traffic_driver.py # 流量驱动器，持续生成会话
├── check_simulated_user_backends.py # 模拟用户后端健康检查
├── run_a3s_code_rl_4gpu.sh         # 主启动脚本（RL 训练）
├── run_a3s_code_agent_traffic.sh   # 流量驱动启动脚本
├── refresh_simulated_user_backends.sh # 刷新模拟用户后端（自动调用）
├── seed_data/                      # 种子任务数据
│   └── code_task_seeds.json
├── task_templates/                 # 任务模板
│   └── mini_taskboard/
├── simulated_user_backends.json    # 模拟用户后端配置（自动生成）
├── generated_workspaces/           # 运行时生成的会话工作区
├── generated_configs/              # 运行时生成的配置文件
├── results/                        # 运行日志（按时间戳命名）
└── docs/                           # 文档目录
```

## 快速启动

### 前置条件

1. **安装 a3s-code SDK**:
   ```bash
   pip install a3s-code --trusted-host mirrors.i.h.pjlab.org.cn \
     -i http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/

   # 验证
   python3 -c "import a3s_code; print('OK')"
   ```

2. **确保 RL 服务运行** (端口 30000)

### 启动方式

**方式 1：一键启动（推荐）**
```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL
bash start_training.sh a3s-code
```

**方式 2：分离启动（调试用）**
```bash
# 终端 1：启动 RL 训练服务
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/slime
bash ../a3s-code-adapter/run_a3s_code_rl_4gpu.sh

# 终端 2：启动流量驱动（等 RL 服务启动后）
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/a3s-code-adapter
bash run_a3s_code_agent_traffic.sh
```

## 关键配置

### RL 训练配置
```bash
NUM_GPUS=4              # GPU 总数
ACTOR_GPUS=2            # Actor 使用的 GPU 数
ROLLOUT_GPUS=2          # Rollout 使用的 GPU 数
ENABLE_PRM=1            # 启用 PRM 打分
PRM_BACKEND=external_openai  # PRM 后端类型
```

### 流量驱动配置
```bash
A3S_CODE_TRAFFIC_CONCURRENCY=1      # 并发会话数
A3S_CODE_MAX_MAIN_TURNS=4           # 最大主轮次
A3S_CODE_MAX_TOOL_ROUNDS=16         # 最大工具调用轮次
CODE_RL_MATCHED_CONTEXT_TOKENS=8192 # 上下文长度
```

### 模拟用户配置
模拟用户后端会在启动时自动检查和更新，无需手动配置。

## 监控训练进度

```bash
# 查看运行信息
cat runs/<run_id>/launch_info.json

# 查看训练样本记录（按时间戳命名）
ls -lt results/
tail -f results/a3s_code_agent_traffic_<timestamp>.jsonl

# 查看 PRM 打分记录
tail -f runs/<run_id>/code_rl_prm_record.jsonl

# Ray Dashboard
# 浏览器访问 http://localhost:8265
```

## 与 openclaw-combine 的区别

| 特性 | openclaw-combine | a3s-code-adapter |
|------|------------------|------------------|
| 数据来源 | 手动交互式对话 | 自动化 Agent 任务执行 |
| 种子数据 | 无 | `seed_data/code_task_seeds.json` |
| 工作区隔离 | 无 | 每个会话独立工作区 |
| 模拟用户 | 无 | 支持任务改写增加多样性 |

## 故障排查

### a3s_code 模块导入失败
```bash
# 检查安装
python3 -c "import a3s_code; print(a3s_code.__file__)"

# 重新安装
pip install a3s-code --trusted-host mirrors.i.h.pjlab.org.cn \
  -i http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/
```

### RL Proxy 端口 30000 被占用
```bash
# 检查端口
ss -ltnp | grep 30000

# 清理旧进程
pkill -f "code_rl_api_server"
ray stop --force
```

### 训练报错 UnboundLocalError
已修复。如遇到，请重启训练服务。

## 文档

详细文档位于 `docs/` 目录：
- `README.md` - 文档索引
- `architecture_analysis_*.md` - 系统架构
- `dataflow_training_*.md` - 训练逻辑
- `optimization_guide_*.md` - 优化建议
- `fixes_summary_*.md` - 问题修复总结

## 迁移说明

本适配层从 `/mnt/shared-storage-user/puyuan/SafeCode/a3s-code-rl` 迁移，主要调整：
- 路径适配 OpenClaw-RL 目录结构
- 模型配置使用 `qwen3-4B.sh` (rotary_base=1M)
- 启动方式使用 `ray job submit` + `train_async.py`
- 日志文件按时间戳命名
- 修复消息格式兼容性问题
