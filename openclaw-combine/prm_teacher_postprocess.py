"""Fill ``rollout_data[\"teacher_log_probs_megatron\"]`` for OPD (Megatron-teacher mode).

Runs as ``--rollout-data-postprocess-path`` after slime packs rollout tensors and
before ``log_rollout_data``. **Does nothing** unless
``OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron`` (or alias ``ref`` / ``mcore``).

Uses **HuggingFace** causal LM on **OPENCLAW_PRM_TEACHER_HF_DEVICE** so the actor
Megatron process does not load a second mcore model. Log-probs are for the same
packed ``tokens`` / response span as training.

Env: ``OPENCLAW_PRM_TEACHER_HF_DEVICE`` (e.g. ``6`` or ``cuda:6``) — must be a
GPU not used by this process’s other work.

If actor **data parallel > 1**, the hook skips (single HF device would race).

Signature: ``postprocess_openclaw_prm_teacher(args, rollout_data)`` (2 args; slime
falls back to ``(args)`` for older single-arg hooks).
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from slime.utils.types import RolloutBatch

logger = logging.getLogger("openclaw.prm_teacher_postprocess")

_prm_hf_model = None
_prm_hf_tokenizer = None


def _want_megatron_teacher() -> bool:
    v = os.getenv("OPENCLAW_COMBINE_OPD_TEACHER_SOURCE", "inference").strip().lower()
    return v in ("megatron", "ref", "mcore")


def _hf_device() -> torch.device | None:
    raw = os.getenv("OPENCLAW_PRM_TEACHER_HF_DEVICE", "").strip()
    if not raw:
        return None
    if raw.startswith("cuda"):
        return torch.device(raw)
    return torch.device(f"cuda:{int(raw)}")


def _get_teacher_model_path(args: Namespace) -> str | None:
    # Prefer an explicit teacher/ref path. Falling back to prm_model_path can
    # silently load a different judge model and blow up teacher-student log-p deltas.
    for value in (
        os.getenv("OPENCLAW_PRM_TEACHER_HF_MODEL_PATH"),
        getattr(args, "ref_load", None),
        os.getenv("REF_LOAD"),
        getattr(args, "prm_model_path", None),
        os.getenv("PRM_MODEL_PATH"),
    ):
        if value:
            return str(value)
    return None


def _load_hf_prm(path: str, device: torch.device):
    global _prm_hf_model, _prm_hf_tokenizer
    if _prm_hf_model is not None:
        return _prm_hf_model, _prm_hf_tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    _prm_hf_model = model
    _prm_hf_tokenizer = tok
    logger.info("Loaded PRM HF teacher on %s from %s", device, path)
    return model, tok


@torch.inference_mode()
def _response_logprobs_hf(
    model,
    input_ids_1d: torch.Tensor,
    response_len: int,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    if response_len <= 0:
        return torch.zeros(0, dtype=torch.float32)
    ids = input_ids_1d.to(device).unsqueeze(0)
    out = model(ids)
    logits = out.logits[0]
    if temperature != 1.0:
        logits = logits / temperature
    logp = torch.log_softmax(logits.float(), dim=-1)
    tlen = ids.size(1)
    pl = tlen - response_len
    lps: list[float] = []
    for k in range(response_len):
        pos = pl - 1 + k
        tok = int(ids[0, pl + k].item())
        lps.append(float(logp[pos, tok].item()))
    return torch.tensor(lps, dtype=torch.float32)


def postprocess_openclaw_prm_teacher(args: Namespace, rollout_data: RolloutBatch) -> None:
    if not _want_megatron_teacher():
        return

    from megatron.core import mpu

    if not mpu.is_pipeline_last_stage(ignore_virtual=True):
        return

    if mpu.get_data_parallel_world_size(with_context_parallel=False) > 1:
        logger.warning(
            "teacher_log_probs_megatron: data_parallel_world_size>1 unsupported; skip"
        )
        return

    if "tokens" not in rollout_data or not rollout_data["tokens"]:
        return

    dev = _hf_device()
    if dev is None:
        logger.warning(
            "OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron but OPENCLAW_PRM_TEACHER_HF_DEVICE "
            "unset; set e.g. OPENCLAW_PRM_TEACHER_HF_DEVICE=6 (free GPU). Skip."
        )
        return

    path = _get_teacher_model_path(args)
    if not path:
        logger.warning(
            "teacher HF model path missing; set OPENCLAW_PRM_TEACHER_HF_MODEL_PATH "
            "or provide ref_load / REF_LOAD. Skip teacher log-p."
        )
        return

    lengths = [int(x) for x in rollout_data["response_lengths"]]
    tp_world = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_src = mpu.get_tensor_model_parallel_src_rank()
    train_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    temperature = float(getattr(args, "rollout_temperature", 1.0))

    broadcast_list: list[torch.Tensor] = []
    if tp_rank == 0:
        model, _ = _load_hf_prm(path, dev)
        for i, toks in enumerate(rollout_data["tokens"]):
            if isinstance(toks, torch.Tensor):
                row = toks.detach().long().flatten()
            else:
                row = torch.tensor(toks, dtype=torch.long)
            rl = lengths[i]
            cpu_vec = _response_logprobs_hf(
                model,
                row,
                rl,
                dev,
                temperature=temperature,
            )
            broadcast_list.append(cpu_vec.to(train_dev))
    else:
        broadcast_list = [
            torch.zeros(rl, dtype=torch.float32, device=train_dev) for rl in lengths
        ]

    if tp_world > 1:
        for t in broadcast_list:
            dist.broadcast(t, src=tp_src, group=tp_group)

    rollout_data["teacher_log_probs_megatron"] = [x.detach().cpu().clone() for x in broadcast_list]
