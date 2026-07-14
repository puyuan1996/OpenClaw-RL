from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .cache_text_hidden import validate_hidden_cache_integrity
from .checkpoint import trained_value_head_status, validate_cache_encoder
from .metrics import require_finite_tensor
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig


def _select_scores(
    out: dict[str, torch.Tensor | None],
    *,
    score_mode: str,
) -> tuple[torch.Tensor, str]:
    if score_mode == "auto":
        if out["value"] is not None:
            return out["value"], "value"
        raise ValueError("score_mode=auto requires a checkpoint with value_head enabled")
    if score_mode == "value":
        if out["value"] is None:
            raise ValueError("score_mode=value requires a checkpoint with value_head enabled")
        return out["value"], "value"
    if score_mode == "uncertainty":
        if out["uncertainty"] is None:
            raise ValueError("score_mode=uncertainty requires a checkpoint with uncertainty_head enabled")
        return -out["uncertainty"], "negative_uncertainty"
    if score_mode == "pred_error":
        if out["target_latent"] is None:
            raise ValueError("score_mode=pred_error requires target_hidden in the input cache")
        return -((out["pred_latent"] - out["target_latent"]).pow(2).mean(dim=-1)), "negative_pred_error"
    raise ValueError(f"unknown score mode: {score_mode}")


def _validate_score_mode(score_mode: str, checkpoint_metadata: dict) -> None:
    if score_mode in {"auto", "value"}:
        trained, reason = trained_value_head_status(checkpoint_metadata)
        if not trained:
            raise ValueError(
                f"score_mode={score_mode} requires a reward-supervised value head: {reason}. "
                "Use --score-mode pred_error only for target-aware oracle diagnostics."
            )
    if score_mode == "uncertainty":
        raise ValueError(
            "score_mode=uncertainty is unavailable: v1 checkpoints have no dedicated loss for the uncertainty head"
        )


def _reward_label_status(checkpoint_metadata: dict) -> tuple[dict, bool]:
    cache_metadata = checkpoint_metadata.get("cache_metadata")
    cache_metadata = cache_metadata if isinstance(cache_metadata, dict) else {}
    contract = cache_metadata.get("reward_label_contract")
    contract = contract if isinstance(contract, dict) else {}
    return contract, contract.get("verified_execution_outcome") is True


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate actions with a trained text latent world model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Torch file with state_hidden, action_hidden, optional target_hidden.")
    parser.add_argument("--output", required=True, help="JSONL ranking output.")
    parser.add_argument(
        "--score-mode",
        choices=["auto", "value", "uncertainty", "pred_error"],
        default="auto",
        help=(
            "Ranking score. auto requires a reward-supervised value head. pred_error is an explicit, "
            "target-aware oracle diagnostic and is not an execution-time score."
        ),
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_metadata = ckpt.get("metadata", {})
    _validate_score_mode(args.score_mode, checkpoint_metadata)
    reward_label_contract, execution_outcome_verified = _reward_label_status(checkpoint_metadata)
    config = TextLatentWorldModelConfig(**ckpt["config"])
    model = TextLatentWorldModel(config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    validate_hidden_cache_integrity(payload)
    validate_cache_encoder(checkpoint_metadata, payload.get("metadata"))
    target_hidden = payload.get("target_hidden", None) if args.score_mode == "pred_error" else None
    with torch.no_grad():
        out = model(
            state_hidden=payload["state_hidden"].float(),
            action_hidden=payload["action_hidden"].float(),
            target_hidden=target_hidden,
        )
        scores, score_source = _select_scores(out, score_mode=args.score_mode)
        scores = require_finite_tensor(scores, name="candidate scores")
        order = torch.argsort(scores, descending=True).tolist()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rank, idx in enumerate(order):
            row = {
                "rank": rank,
                "candidate_index": int(idx),
                "score": float(scores[idx].item()),
                "score_source": score_source,
                "oracle_only": args.score_mode == "pred_error",
                "requires_target": args.score_mode == "pred_error",
                "reward_label_contract": reward_label_contract,
                "reward_label_verified_execution_outcome": execution_outcome_verified,
            }
            if args.score_mode == "uncertainty" and out["uncertainty"] is not None:
                row["uncertainty"] = float(out["uncertainty"][idx].item())
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {len(order)} candidate rankings to {out_path}")


if __name__ == "__main__":
    main()
