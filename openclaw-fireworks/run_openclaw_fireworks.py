#!/usr/bin/env python3
"""Entry point for OpenClaw-RL with Fireworks backend.

Bootstraps:
  1. Fireworks Deployment (policy inference + PRM)
  2. Fireworks Trainer (training + teacher logprobs)
  3. OpenClaw API server (proxy)
  4. Training loop (drain queue -> train -> weight sync)

Environment / CLI configuration (all have sensible defaults):
  FIREWORKS_API_KEY         -- required
  FIREWORKS_BASE_URL        -- default https://api.fireworks.ai
  BASE_MODEL                -- default accounts/fireworks/models/qwen3-8b
  TRAINING_SHAPE_ID         -- default accounts/fireworks/trainingShapes/qwen3-8b-128k
  DEPLOYMENT_ID             -- default openclaw-serving
  TOKENIZER_MODEL           -- HuggingFace model for tokenizer (default Qwen/Qwen3-8B)
  ROLLOUT_BATCH_SIZE        -- default 16
  LEARNING_RATE             -- default 1e-5
  W_OPD                     -- OPD advantage weight (default 1.0)
  W_RL                      -- GRPO advantage weight (default 1.0)
  EPS_CLIP                  -- PPO lower clip (default 0.2)
  EPS_CLIP_HIGH             -- PPO upper clip (default 0.28)
  WEIGHT_SYNC_INTERVAL      -- steps between hotloads (default 1)
  MAX_STEPS                 -- 0 = run forever (default 0)
  MAX_SEQ_LEN               -- datum max length (default 32768)
  SERVER_PORT               -- proxy port (default 30000)
  PRM_ENABLED               -- 1/0 (default 1)
  PRM_M                     -- number of PRM judge votes (default 3)
  PRM_TEMPERATURE           -- PRM sampling temperature (default 0.6)
  PRM_MAX_TOKENS            -- PRM max generation tokens (default 4096)
  LORA_RANK                 -- 0 for full-parameter (default 0)
  GRADIENT_ACCUMULATION     -- grad accumulation steps (default 1)
  RECORD_DIR                -- directory for conversation/PRM records (default "")
  SERVED_MODEL_NAME         -- model name returned by the proxy (default "default")
  EXPECTED_API_KEY          -- proxy auth key (default "" = no auth)
  MAX_STEPS                 -- stop after N training steps (default 0 = forever)
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("openclaw-fireworks")


def main():
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        logger.error("FIREWORKS_API_KEY is required")
        sys.exit(1)

    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    base_model = os.environ.get("BASE_MODEL", "accounts/fireworks/models/qwen3-8b")
    shape_id = os.environ.get(
        "TRAINING_SHAPE_ID",
        "accounts/fireworks/trainingShapes/qwen3-8b-128k",
    )
    deployment_id = os.environ.get("DEPLOYMENT_ID", "openclaw-serving")
    tokenizer_model = os.environ.get("TOKENIZER_MODEL", "Qwen/Qwen3-8B")
    batch_size = int(os.environ.get("ROLLOUT_BATCH_SIZE", "16"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-5"))
    w_opd = float(os.environ.get("W_OPD", "1.0"))
    w_rl = float(os.environ.get("W_RL", "1.0"))
    eps_clip = float(os.environ.get("EPS_CLIP", "0.2"))
    eps_clip_high = float(os.environ.get("EPS_CLIP_HIGH", "0.28"))
    weight_sync_interval = int(os.environ.get("WEIGHT_SYNC_INTERVAL", "1"))
    max_steps = int(os.environ.get("MAX_STEPS", "0"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "32768"))
    server_port = int(os.environ.get("SERVER_PORT", "30000"))
    prm_enabled = os.environ.get("PRM_ENABLED", "1") == "1"
    prm_m = int(os.environ.get("PRM_M", "3"))
    prm_temperature = float(os.environ.get("PRM_TEMPERATURE", "0.6"))
    prm_max_tokens = int(os.environ.get("PRM_MAX_TOKENS", "4096"))
    lora_rank = int(os.environ.get("LORA_RANK", "0"))
    grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION", "1"))
    record_dir = os.environ.get("RECORD_DIR", "")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "default")
    expected_api_key = os.environ.get("EXPECTED_API_KEY", "")

    from fireworks.training.sdk import (
        DeploymentConfig,
        DeploymentManager,
        FiretitanServiceClient,
        TrainerJobConfig,
        TrainerJobManager,
        WeightSyncer,
    )

    logger.info("=" * 70)
    logger.info("  OpenClaw-RL Fireworks Backend")
    logger.info("  base_model:    %s", base_model)
    logger.info("  shape_id:      %s", shape_id)
    logger.info("  deployment_id: %s", deployment_id)
    logger.info("  batch_size:    %d", batch_size)
    logger.info("  learning_rate: %.2e", learning_rate)
    logger.info("  w_opd=%.1f  w_rl=%.1f  eps_clip=%.2f/%.2f", w_opd, w_rl, eps_clip, eps_clip_high)
    logger.info("  lora_rank:     %d", lora_rank)
    logger.info("  server_port:   %d", server_port)
    logger.info("=" * 70)

    # ----- 1. Resolve training shape -----
    logger.info("[Bootstrap] Resolving training shape '%s'...", shape_id)
    trainer_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    profile = trainer_mgr.resolve_training_profile(shape_id)
    logger.info("[Bootstrap] Resolved shape version: %s", profile.training_shape_version)
    deployment_shape = getattr(profile, "deployment_shape_version", None)
    if deployment_shape:
        logger.info("[Bootstrap] Linked deployment shape: %s", deployment_shape)

    # ----- 2. Create deployment -----
    logger.info("[Bootstrap] Creating/getting deployment '%s'...", deployment_id)
    deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)
    deploy_config = DeploymentConfig(
        deployment_id=deployment_id,
        base_model=base_model,
        min_replica_count=0,
        max_replica_count=1,
    )
    if deployment_shape:
        deploy_config.deployment_shape = deployment_shape
    deploy_mgr.create_or_get(deploy_config)
    deploy_mgr.wait_for_ready(deployment_id)
    logger.info("[Bootstrap] Deployment ready: %s", deploy_mgr.inference_url)

    # ----- 3. Create trainer -----
    logger.info("[Bootstrap] Creating trainer...")
    endpoint = trainer_mgr.create_and_wait(TrainerJobConfig(
        base_model=base_model,
        training_shape_ref=profile.training_shape_version,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        gradient_accumulation_steps=grad_accum,
        hot_load_deployment_id=deployment_id,
    ))
    logger.info("[Bootstrap] Trainer ready at %s", endpoint.base_url)

    # ----- 4. Connect training client -----
    service = FiretitanServiceClient(base_url=endpoint.base_url, api_key=api_key)
    training_client = service.create_training_client(
        base_model=base_model, lora_rank=lora_rank,
    )
    logger.info("[Bootstrap] Training client connected")

    # ----- 5. Set up weight syncer -----
    weight_syncer = WeightSyncer(
        policy_client=training_client,
        deploy_mgr=deploy_mgr,
        deployment_id=deployment_id,
        base_model=base_model,
        hotload_timeout=600,
        first_checkpoint_type="base",
    )

    # ----- 6. Resolve deployment chat URL -----
    account_id = deploy_mgr.account_id
    inference_base = deploy_mgr.inference_url.rstrip("/")
    if not inference_base.endswith("/inference"):
        inference_base = f"{inference_base}/inference"
    deployment_chat_url = f"{inference_base}/v1/chat/completions"
    deployment_model = f"accounts/{account_id}/deployments/{deployment_id}"

    # ----- 7. Start API server -----
    from fireworks_server import OpenClawFireworksServer

    output_queue: queue.Queue = queue.Queue(maxsize=100000)
    submission_enabled = threading.Event()
    submission_enabled.set()

    server = OpenClawFireworksServer(
        output_queue=output_queue,
        submission_enabled=submission_enabled,
        tokenizer_model=tokenizer_model,
        deployment_chat_url=deployment_chat_url,
        deployment_model=deployment_model,
        fw_api_key=api_key,
        training_client=training_client,
        prm_enabled=prm_enabled,
        prm_m=prm_m,
        prm_temperature=prm_temperature,
        prm_max_tokens=prm_max_tokens,
        host="0.0.0.0",
        port=server_port,
        served_model_name=served_model_name,
        expected_api_key=expected_api_key,
        record_dir=record_dir,
    )
    server.start()
    logger.info("[Bootstrap] API server started on port %d", server_port)

    # ----- 8. Run training loop (blocks) -----
    from fireworks_training_loop import run_training_loop

    try:
        run_training_loop(
            training_client=training_client,
            weight_syncer=weight_syncer,
            output_queue=output_queue,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            learning_rate=learning_rate,
            w_opd=w_opd,
            w_rl=w_rl,
            eps_clip=eps_clip,
            eps_clip_high=eps_clip_high,
            weight_sync_interval=weight_sync_interval,
            max_steps=max_steps,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted -- shutting down")
    finally:
        server.stop()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
