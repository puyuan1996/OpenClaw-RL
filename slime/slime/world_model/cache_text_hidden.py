from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .metadata import canonicalize_context_identity


_ENCODER_BEHAVIOR_PROBES = [
    "OpenClaw encoder identity probe: inspect terminal state.",
    'OpenClaw encoder identity probe: {"tool":"bash","command":"pwd"}',
]


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hidden_tensors_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def _encoder_behavior_sha256(hidden: torch.Tensor) -> str:
    # Quantization avoids false mismatches from insignificant cross-device rounding.
    quantized = torch.round(hidden.detach().cpu().float() * 10_000.0).to(torch.int32)
    return _hidden_tensors_sha256({"encoder_behavior_probe": quantized})


def _reward_label_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "reward_label_scope",
        "reward_label_source",
        "reward_label_semantics",
        "reward_label_is_execution_outcome",
    ]
    contracts: dict[str, dict[str, Any]] = {}
    for record in records:
        contract = {field: record.get(field) for field in fields}
        key = json.dumps(contract, ensure_ascii=False, sort_keys=True, default=str)
        contracts[key] = contract
    consistent = len(contracts) == 1
    contract = next(iter(contracts.values())) if consistent else {field: None for field in fields}
    return {
        **contract,
        "consistent": consistent,
        "contract_count": len(contracts),
        "verified_execution_outcome": consistent
        and contract.get("reward_label_is_execution_outcome") is True,
    }


def _sample_payload_fingerprints(payload: dict[str, Any]) -> dict[str, str]:
    record_count = int(payload.get("record_count", -1))
    if record_count < 0:
        raise ValueError("cache record_count is missing or invalid")
    record_metadata = payload.get("record_metadata")
    if not isinstance(record_metadata, list) or len(record_metadata) != record_count:
        raise ValueError("cache record_metadata must match record_count")

    has_reward = "reward" in payload
    has_reward_mask = "reward_mask" in payload
    if has_reward != has_reward_mask:
        raise ValueError("cache reward and reward_mask must either both be present or both be absent")
    supervision_tensors: dict[str, torch.Tensor] = {}
    if has_reward:
        for key in ["reward", "reward_mask"]:
            tensor = payload[key]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or int(tensor.shape[0]) != record_count:
                raise ValueError(f"cache {key} must be a tensor whose first dimension matches record_count")
            supervision_tensors[key] = tensor

    record_metadata_sha256 = _json_sha256(record_metadata)
    supervision_tensors_sha256 = _hidden_tensors_sha256(supervision_tensors)
    sample_payload_sha256 = _json_sha256(
        {
            "record_count": record_count,
            "record_metadata_sha256": record_metadata_sha256,
            "supervision_tensors_sha256": supervision_tensors_sha256,
        }
    )
    return {
        "record_metadata_sha256": record_metadata_sha256,
        "supervision_tensors_sha256": supervision_tensors_sha256,
        "sample_payload_sha256": sample_payload_sha256,
    }


def _build_cache_integrity_metadata(
    payload: dict[str, Any],
    *,
    input_records_sha256: str,
    encoder_config: dict[str, Any],
) -> dict[str, Any]:
    tensor_keys = ["action_hidden", "state_hidden", "target_hidden"]
    missing = [key for key in tensor_keys if not isinstance(payload.get(key), torch.Tensor)]
    if missing:
        raise ValueError(f"cannot fingerprint cache; hidden tensors are missing: {missing}")
    encoder_fingerprint_sha256 = _json_sha256(encoder_config)
    hidden_tensors_sha256 = _hidden_tensors_sha256({key: payload[key] for key in tensor_keys})
    sample_fingerprints = _sample_payload_fingerprints(payload)
    cache_fingerprint_sha256 = _json_sha256(
        {
            "encoder_fingerprint_sha256": encoder_fingerprint_sha256,
            "hidden_tensors_sha256": hidden_tensors_sha256,
            "input_records_sha256": input_records_sha256,
            "sample_payload_sha256": sample_fingerprints["sample_payload_sha256"],
        }
    )
    return {
        "schema_version": "openclaw_text_jepa_hidden_cache_v4",
        "input_records_sha256": input_records_sha256,
        "encoder_config": encoder_config,
        "encoder_fingerprint_sha256": encoder_fingerprint_sha256,
        "encoder_behavior_probe_sha256": encoder_config.get("behavior_probe_sha256"),
        "hidden_tensors_sha256": hidden_tensors_sha256,
        "fingerprint_tensor_keys": tensor_keys,
        **sample_fingerprints,
        "cache_fingerprint_sha256": cache_fingerprint_sha256,
        "reward_label_contract": _reward_label_contract(payload["record_metadata"]),
    }


def validate_hidden_cache_integrity(
    payload: dict[str, Any],
    *,
    require_verified: bool = False,
) -> dict[str, Any]:
    """Recompute cache digests; strict consumers reject legacy partial fingerprints."""
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    schema_version = metadata.get("schema_version")
    expected_tensor_digest = metadata.get("hidden_tensors_sha256")
    if not expected_tensor_digest:
        if schema_version in {
            "openclaw_text_jepa_hidden_cache_v3",
            "openclaw_text_jepa_hidden_cache_v4",
        }:
            raise ValueError("current-schema cache is missing hidden_tensors_sha256")
        if require_verified:
            raise ValueError("hidden cache lacks a complete integrity fingerprint; rebuild the cache")
        return {"verified": False, "reason": "hidden_tensors_sha256_missing"}

    tensor_keys = metadata.get("fingerprint_tensor_keys") or [
        "action_hidden",
        "state_hidden",
        "target_hidden",
    ]
    if not isinstance(tensor_keys, list) or not tensor_keys:
        raise ValueError("cache fingerprint_tensor_keys is invalid")
    missing = [key for key in tensor_keys if not isinstance(payload.get(key), torch.Tensor)]
    if missing:
        raise ValueError(f"cache fingerprint tensors are missing: {missing}")
    actual_tensor_digest = _hidden_tensors_sha256({key: payload[key] for key in tensor_keys})
    if actual_tensor_digest != expected_tensor_digest:
        raise ValueError("hidden cache tensor digest mismatch")
    hidden_count = int(payload[tensor_keys[0]].shape[0])
    if int(payload.get("record_count", hidden_count)) != hidden_count:
        raise ValueError("cache record_count does not match hidden tensors")
    for key in tensor_keys[1:]:
        if int(payload[key].shape[0]) != hidden_count:
            raise ValueError("cache fingerprint tensor first dimensions do not match")

    encoder_config = metadata.get("encoder_config")
    if not isinstance(encoder_config, dict):
        raise ValueError("hidden cache encoder_config is missing or invalid")
    actual_encoder_fingerprint = _json_sha256(encoder_config)
    if actual_encoder_fingerprint != metadata.get("encoder_fingerprint_sha256"):
        raise ValueError("hidden cache encoder fingerprint mismatch")
    behavior_fingerprint = encoder_config.get("behavior_probe_sha256")
    if behavior_fingerprint and behavior_fingerprint != metadata.get("encoder_behavior_probe_sha256"):
        raise ValueError("hidden cache encoder behavior fingerprint mismatch")

    expected_cache_fingerprint = metadata.get("cache_fingerprint_sha256")
    if not expected_cache_fingerprint:
        raise ValueError("cache_fingerprint_sha256 is missing")
    expected_sample_payload = metadata.get("sample_payload_sha256")
    if not expected_sample_payload:
        if schema_version == "openclaw_text_jepa_hidden_cache_v4":
            raise ValueError("hidden cache lacks sample/reward/group fingerprints; rebuild the cache")
        legacy_cache_fingerprint = _json_sha256(
            {
                "encoder_fingerprint_sha256": actual_encoder_fingerprint,
                "hidden_tensors_sha256": actual_tensor_digest,
                "input_records_sha256": metadata.get("input_records_sha256"),
            }
        )
        if legacy_cache_fingerprint != expected_cache_fingerprint:
            raise ValueError("legacy hidden cache metadata fingerprint mismatch")
        if require_verified:
            raise ValueError("hidden cache lacks sample/reward/group fingerprints; rebuild the cache")
        return {"verified": False, "reason": "sample_payload_sha256_missing"}

    sample_fingerprints = _sample_payload_fingerprints(payload)
    for key, actual in sample_fingerprints.items():
        if metadata.get(key) != actual:
            raise ValueError(f"hidden cache {key} mismatch")
    actual_reward_contract = _reward_label_contract(payload["record_metadata"])
    if metadata.get("reward_label_contract") != actual_reward_contract:
        raise ValueError("hidden cache reward label contract mismatch")
    actual_cache_fingerprint = _json_sha256(
        {
            "encoder_fingerprint_sha256": actual_encoder_fingerprint,
            "hidden_tensors_sha256": actual_tensor_digest,
            "input_records_sha256": metadata.get("input_records_sha256"),
            "sample_payload_sha256": sample_fingerprints["sample_payload_sha256"],
        }
    )
    if actual_cache_fingerprint != expected_cache_fingerprint:
        raise ValueError("hidden cache metadata fingerprint mismatch")
    return {"verified": True, "reason": None}


def _finite_reward(value: Any) -> float | None:
    try:
        reward = float(value)
    except (TypeError, ValueError):
        return None
    return reward if math.isfinite(reward) else None


def _record_state_text(record: dict) -> str:
    text = record.get("context_text")
    if text:
        return str(text)
    # Compatibility fallback for records created before context_text existed.
    return json.dumps(
        {
            "context_hash": record.get("context_hash"),
            "data_source": record.get("data_source"),
            "task_name": record.get("task_name"),
            "task_path": record.get("task_path"),
            "turn_idx": record.get("turn_idx"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _record_action_text(record: dict) -> str:
    return str(record.get("action_text") or record.get("action_hash") or "")


def _record_target_text(record: dict) -> str:
    return str(record.get("next_observation_text") or record.get("next_observation_hash") or "")


def _light_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "schema",
        "hidden_source",
        "task_name",
        "task_path",
        "data_source",
        "uid",
        "group_index",
        "sample_index",
        "rollout_id",
        "train_step",
        "turn_idx",
        "num_turns",
        "status",
        "trajectory_status",
        "done",
        "has_tool_result",
        "reward_score",
        "reward_base_score",
        "reward_raw_score",
        "reward_label_scope",
        "reward_label_source",
        "reward_label_semantics",
        "reward_label_is_execution_outcome",
        "reward_label_terminal",
        "context_hash",
        "context_hash_schema",
        "source_context_hash",
        "action_hash",
        "next_observation_hash",
        "context_token_len",
        "action_token_len",
        "context_text_source",
    ]
    return {key: record.get(key) for key in keys if key in record}


def _stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _hash_encode(texts: Iterable[str], hidden_dim: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for text in texts:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(text))
        vec = torch.randn(hidden_dim, generator=generator)
        rows.append(F.normalize(vec, dim=0))
    if not rows:
        return torch.empty(0, hidden_dim)
    return torch.stack(rows, dim=0)


def _pool_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError("hidden and attention_mask shapes are incompatible for last-token pooling")
    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0).expand_as(mask)
    last_indices = positions.masked_fill(~mask, -1).max(dim=1).values
    if bool((last_indices < 0).any()):
        raise ValueError("last-token pooling requires at least one unmasked token per row")
    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_indices, last_indices]


def _hf_encode(
    texts: list[str],
    *,
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].to(hidden.dtype)
        if pooling == "last":
            pooled = _pool_last_token(hidden, enc["attention_mask"])
        elif pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        rows.append(pooled.detach().cpu().float())
    if not rows:
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        return torch.empty(0, hidden_size)
    return torch.cat(rows, dim=0)


def _encode_texts(args: argparse.Namespace, texts: list[str]) -> torch.Tensor:
    if args.encoder == "hash":
        return _hash_encode(texts, args.hidden_dim)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return _hf_encode(
        texts,
        model_name_or_path=args.hf_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        pooling=args.pooling,
        local_files_only=args.hf_local_files_only,
        trust_remote_code=args.hf_trust_remote_code,
    )


def _encode_record_texts(
    args: argparse.Namespace,
    state_texts: list[str],
    action_texts: list[str],
    target_texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    all_texts = state_texts + action_texts + target_texts + _ENCODER_BEHAVIOR_PROBES
    all_hidden = _encode_texts(args, all_texts)
    count = len(state_texts)
    expected_count = count * 3 + len(_ENCODER_BEHAVIOR_PROBES)
    if int(all_hidden.shape[0]) != expected_count:
        raise ValueError(f"encoder returned {int(all_hidden.shape[0])} rows; expected {expected_count}")
    probe_hidden = all_hidden[count * 3 :]
    return (
        all_hidden[:count],
        all_hidden[count : count * 2],
        all_hidden[count * 2 : count * 3],
        _encoder_behavior_sha256(probe_hidden),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache state/action/target hidden tensors for text JEPA world-model probes.")
    parser.add_argument("--input", required=True, help="World-model records JSONL.")
    parser.add_argument("--output", required=True, help="Torch output with state_hidden/action_hidden/target_hidden tensors.")
    parser.add_argument("--encoder", choices=["hash", "hf"], default="hash")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hash encoder output dim.")
    parser.add_argument("--hf-model", default=None, help="HF model path/name used when --encoder hf.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pooling", choices=["mean", "last", "cls"], default="mean")
    parser.set_defaults(hf_local_files_only=True)
    parser.add_argument(
        "--hf-local-files-only",
        action="store_true",
        dest="hf_local_files_only",
        help="Use only local HF files (default).",
    )
    parser.add_argument(
        "--hf-allow-downloads",
        action="store_false",
        dest="hf_local_files_only",
        help="Explicitly allow HF model/tokenizer downloads.",
    )
    parser.add_argument(
        "--hf-trust-remote-code",
        action="store_true",
        help="Allow custom Python code from the HF model repository. Disabled by default.",
    )
    args = parser.parse_args()

    if args.encoder == "hf" and not args.hf_model:
        raise ValueError("--hf-model is required when --encoder hf")

    input_path = Path(args.input)
    records = _read_jsonl(input_path)
    if not records:
        raise ValueError(f"No world-model records found in {input_path}. Check filters or input path.")
    records = [canonicalize_context_identity(record) for record in records]
    state_texts = [_record_state_text(record) for record in records]
    action_texts = [_record_action_text(record) for record in records]
    target_texts = [_record_target_text(record) for record in records]

    state_hidden, action_hidden, target_hidden, encoder_behavior_sha256 = _encode_record_texts(
        args,
        state_texts,
        action_texts,
        target_texts,
    )
    rewards = [_finite_reward(record.get("reward_score")) for record in records]
    reward_mask = [value is not None for value in rewards]
    has_reward = any(value is not None for value in rewards)

    input_records_sha256 = _file_sha256(input_path)
    encoder_config = {
        "schema_version": "openclaw_text_jepa_encoder_config_v1",
        "encoder": args.encoder,
        "hidden_dim": args.hidden_dim if args.encoder == "hash" else int(state_hidden.shape[-1]),
        "hf_model": args.hf_model if args.encoder == "hf" else None,
        "pooling": args.pooling if args.encoder == "hf" else None,
        "max_length": args.max_length if args.encoder == "hf" else None,
        "behavior_probe_schema": "openclaw_text_encoder_behavior_probe_v1",
        "behavior_probe_sha256": encoder_behavior_sha256,
    }
    record_metadata = [_light_record_metadata(record) for record in records]
    payload = {
        "state_hidden": state_hidden,
        "action_hidden": action_hidden,
        "target_hidden": target_hidden,
        "record_count": len(records),
        "encoder": args.encoder,
        "input": str(input_path),
        "record_metadata": record_metadata,
    }
    if has_reward:
        payload["reward"] = torch.tensor([0.0 if value is None else float(value) for value in rewards], dtype=torch.float32)
        payload["reward_mask"] = torch.tensor(reward_mask, dtype=torch.bool)
    integrity_metadata = _build_cache_integrity_metadata(
        payload,
        input_records_sha256=input_records_sha256,
        encoder_config=encoder_config,
    )
    payload["metadata"] = {
        **integrity_metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "record_count": len(records),
        "encoder": args.encoder,
        "hidden_dim": args.hidden_dim if args.encoder == "hash" else int(state_hidden.shape[-1]),
        "hf_model": args.hf_model if args.encoder == "hf" else None,
        "pooling": args.pooling if args.encoder == "hf" else None,
        "hf_local_files_only": bool(args.hf_local_files_only) if args.encoder == "hf" else None,
        "hf_trust_remote_code": bool(args.hf_trust_remote_code) if args.encoder == "hf" else None,
        "max_length": args.max_length if args.encoder == "hf" else None,
        "batch_size": args.batch_size,
        "state_hidden_shape": tuple(state_hidden.shape),
        "action_hidden_shape": tuple(action_hidden.shape),
        "target_hidden_shape": tuple(target_hidden.shape),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(
        "cached "
        f"{len(records)} records to {out} "
        f"(state={tuple(state_hidden.shape)} action={tuple(action_hidden.shape)} target={tuple(target_hidden.shape)} encoder={args.encoder})"
    )


if __name__ == "__main__":
    main()
