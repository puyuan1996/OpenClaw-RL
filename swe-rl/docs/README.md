# Commands

Replace placeholders before running:

| Placeholder | Description |
| --- | --- |
| `<ROOT>` | Data root directory |
| `<CONDA_ENV>` | Path to conda environment |
| `<ENV_FILE>` | Path to `.env` file with API keys |
| `<MODEL_PATH>` | Local path to model weights (HuggingFace snapshot) |
| `<EXEC_SERVER>` | SWE exec server URL, e.g. `http://10.0.0.1:5000` |
| `<DATA_DIR>` | Directory containing dataset JSONL files |
| `<REPO_ROOT>` | Path to the repo root |

## PR-safe script overrides (required before run)

The default 4B scripts no longer ship hardcoded private infra values (internal IPs,
proxy endpoints, local SSH key paths, or machine-specific absolute paths). Set
these env vars in your shell or `<ENV_FILE>` before launching:

| Variable | Required | Used by | Example |
| --- | --- | --- | --- |
| `SWE_EXEC_SERVER_URLS` | Yes | train + eval | `http://10.0.0.1:5000,http://10.0.0.2:5000` |
| `HF_CKPT` | Yes (train), recommended (eval) | train + eval | `<REPO_ROOT>/export/hf/swe-rl-4b_iter180` |
| `CES_SSH_USER` | Optional | train + eval | `root` |
| `CES_SSH_KEY` | Optional | train + eval | `<REPO_ROOT>/swe.pem` |
| `HTTP_PROXY` / `HTTPS_PROXY` | Optional | train + eval | `http://<proxy-host>:<port>` |
| `VERIFIED_DATA` | Optional | eval | `<REPO_ROOT>/data/swebench_verified/verified_eval_with_script.jsonl` |
| `SWEGYM_DATA` | Optional | eval | `<REPO_ROOT>/data/swegym_293/train_with_eval_script.parquet` |
| `PROMPT_DATA` | Optional | train | `<REPO_ROOT>/data/swegym_293/train_with_eval_script.parquet` |
| `HF_HOME` | Optional | train + eval | `<ROOT>/systems/huggingface` |

## Eval: SWE-Bench Verified (sample)

```bash
source activate <CONDA_ENV>
set -a; source <ENV_FILE>; set +a

export EVAL_MODEL_PATH=<MODEL_PATH>
export EVAL_MODEL_NAME=openai/Qwen3-32B
export EVAL_TP=8
export EVAL_GPU_IDS=0,1,2,3,4,5,6,7
export SWE_EXEC_SERVER_URLS=<EXEC_SERVER>
export MAX_CONCURRENT=15
export STEP_LIMIT=20
export MAX_TOKENS=4096
export TEMPERATURE=0
export PROMPT_DATA=<DATA_DIR>/swe_verified_full/train.jsonl
export DATA_SOURCE=swe-bench
export MAX_INSTANCES=20

cd <REPO_ROOT>
bash swe-rl/eval/run_eval_swe.sh
```

## Eval: SWE-Bench Verified (full)

```bash
source activate <CONDA_ENV>
set -a; source <ENV_FILE>; set +a

export EVAL_MODEL_PATH=<MODEL_PATH>
export EVAL_MODEL_NAME=openai/Qwen3-32B
export EVAL_TP=8
export EVAL_GPU_IDS=0,1,2,3,4,5,6,7
export SWE_EXEC_SERVER_URLS=<EXEC_SERVER>
export MAX_CONCURRENT=15
export STEP_LIMIT=20
export MAX_TOKENS=4096
export TEMPERATURE=0
export PROMPT_DATA=<DATA_DIR>/swe_verified_full/train.jsonl
export DATA_SOURCE=swe-bench
unset MAX_INSTANCES

cd <REPO_ROOT>
bash swe-rl/eval/run_eval_swe.sh
```

## Eval: 4B multi-node script

```bash
source activate <CONDA_ENV>
set -a; source <ENV_FILE>; set +a

# Optional: evaluate a specific checkpoint
export HF_CKPT=<REPO_ROOT>/export/hf/swe-rl-4b_iter180

cd <REPO_ROOT>
bash swe-rl/scripts/eval_4b_4nodes.sh
```

`swe-rl/scripts/eval_4b_4nodes.sh` evaluation logic:

- Starts remote SWE env pool server and joins Ray workers on 4 GPU nodes
- Runs two eval passes in sequence:
  - SWE-Bench Verified (`text/metadata` payload)
  - SWE-Gym 293 (`prompt/instance` payload)
- Each eval trajectory calls `generate_with_swe.generate` + `reward_func`
- Cleans remote Docker containers between phases and writes artifacts to:
  - `${EXPORT_ROOT}/swe_rollouts/eval_<run_tag>_verified_<ts>`
  - `${EXPORT_ROOT}/swe_rollouts/eval_<run_tag>_swegym_<ts>`

## Train: 8 nodes (Qwen3-32B)

```bash
# Platform injects automatically:
# MLP_ROLE_INDEX=0/1/2/3/4/5/6/7
# MLP_WORKER_0_HOST=<head_ip>
# MLP_WORKER_<i>_HOST=<self_ip>

export HF_HOME=<ROOT>/systems/huggingface
export HF_CKPT=<MODEL_PATH>
export RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}
export SAVE_CKPT=<REPO_ROOT>/swe-rl/output/ckpt/swe-rl-32b-remote-8nodes_${RUN_TIMESTAMP}
export SWE_SAVE_TRAJ_DIR=<REPO_ROOT>/swe-rl/output/swe_rollouts/swe-rl-32b-remote-8nodes_${RUN_TIMESTAMP}
export PROMPT_DATA=<DATA_DIR>/swe_verified_full/train_subset_360.jsonl
if [ ! -f "${PROMPT_DATA}" ]; then
  echo "Missing dataset: ${PROMPT_DATA}"
  exit 1
fi

set -a; source <ENV_FILE>; set +a
export WANDB_KEY="${WANDB_API_KEY}"
export WANDB_PROJECT=${WANDB_PROJECT:-slime_swe}

source activate <CONDA_ENV>

cd <REPO_ROOT>
bash swe-rl/scripts/run_swe_rl_32b_remote_8nodes.sh || true
```

Key training parameters used by the script:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `NUM_NODES` | `8` | Total training nodes in the Ray cluster |
| `NUM_GPUS_PER_NODE` | `8` | Total GPUs available on each node |
| `ACTOR_GPUS_PER_NODE` | `4` | GPUs per node reserved for actor training |
| `ROLLOUT_GPUS_PER_NODE` | `4` | GPUs per node reserved for rollout inference |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | `4` | Tensor parallel size per rollout engine |
| `SWE_EXEC_SERVER_URLS` | (built-in list) | Comma-separated SWE exec server endpoints (`http://<ip>:5000`) |
| `SWE_MAX_CONTAINERS_PER_NODE` | `15` | Max active SWE containers per exec server node |
| `SWE_MAX_CONCURRENT` | `128` | Global max concurrent SWE rollouts |
| `EXPORT_ROOT` | `<REPO_ROOT>/export` | Output base directory; used to build default output paths when `SAVE_CKPT`/`SWE_SAVE_TRAJ_DIR` are not set |
| `SAVE_CKPT` | `${EXPORT_ROOT}/ckpt/swe-rl-32b-remote-8nodes_${RUN_TIMESTAMP}` | Final training checkpoint save path (passed to `--save`) |
| `SWE_SAVE_TRAJ_DIR` | `${EXPORT_ROOT}/swe_rollouts/swe-rl-32b-remote-8nodes_${RUN_TIMESTAMP}` | Final SWE rollout artifact path (`traj.json`, `patch.diff`, `meta.json`) |
| `WANDB_PROJECT` | `slime_swe` | Weights & Biases project name used by `--wandb-project` |
| `PROMPT_DATA` | `<DATA_DIR>/swe_verified_full/train_subset_360.jsonl` | Training JSONL file |
| `--rollout-max-context-len` | `16384` | Context window budget for rollout history |
| `--rollout-max-response-len` | `4096` | Max new tokens generated per step |
| `--n-samples-per-prompt` | `8` | GRPO group size (samples per prompt) |
| `--num-rollout` | `2000` (or `40` in `DEBUG_MODE=1`) | Rollout instances per training |
