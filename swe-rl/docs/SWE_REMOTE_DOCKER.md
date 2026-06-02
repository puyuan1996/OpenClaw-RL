# Remote Docker Architecture

SWE RL decouples the Docker execution layer onto separate Docker nodes, leaving GPU nodes solely responsible for LLM inference and training.

## Architecture

```
┌─ Docker Node(s) ──────────────────────┐
│  server/swe_exec_server.py (:5000)    │  ← pre-installed on each Docker node
│    /container/create                  │
│    /container/exec                    │
│    /container/diff                    │
│    /container/evaluate                │
│    /container/destroy                 │
└────────────────▲──────────────────────┘
                 │ HTTP
┌─ GPU Head Node─┼──────────────────────┐
│  server/swe_env_pool_server.py        │  ← started by training script
│    (:18090) load-balancing + leases   │
└────────────────▲──────────────────────┘
                 │ HTTP
┌─ RolloutManager┼──────────────────────┐
│  swe_env_client.py                    │
│  generate_with_swe_remote.py          │
└───────────────────────────────────────┘
```

## Data Flow

Full lifecycle of one SWE-Bench instance:

1. `generate()` is called by the RolloutManager
2. `SweEnvClient.allocate(image)` → pool_server picks the least-loaded node → `docker run` → returns `lease_id`
3. Multi-turn agent loop:
   - LLM inference
   - Parse bash command
   - `SweEnvClient.exec(command)` → pool_server → Docker Node → `docker exec` → returns output
   - Build observation → back to LLM inference
4. Agent submits patch (or `diff()` for fallback patch)
5. Close agent container, allocate a fresh eval container
6. `SweEnvClient.evaluate(patch, eval_script)` → Docker Node runs `git apply` + test suite → `resolved?`
7. `SweEnvClient.close()` → pool_server → Docker Node → `docker rm -f`
8. Encode tokens + loss_mask + reward → return `Sample`

## Docker Node Setup

Upload files to the node:

```bash
NODE_IP=<node IP>
scp server/swe_exec_server.py    root@${NODE_IP}:~/
scp server/setup_ecs_seed.sh     root@${NODE_IP}:~/
scp data/pull_swe_images.sh      root@${NODE_IP}:~/
scp ~/data/train.jsonl           root@${NODE_IP}:~/train.jsonl
```

SSH in and run:

```bash
bash ~/setup_ecs_seed.sh
```

Verify:

```bash
curl http://localhost:5000/healthz
curl http://localhost:5000/images | python3 -m json.tool | head
```
