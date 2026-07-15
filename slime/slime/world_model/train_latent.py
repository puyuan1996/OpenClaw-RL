from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
from typing import Any, Sequence

import torch

from .hidden_encoder import PolicyHiddenEncoder, hash_hidden_batch
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig
from .replay_buffer import TrajectoryReplayBuffer
from .seta_dataset import TerminalTransition, load_terminal_transitions


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _split_indices(count: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    if count < 2 or val_ratio <= 0:
        return indices, []
    val_count = max(1, min(count - 1, int(round(count * val_ratio))))
    return indices[val_count:], indices[:val_count]


def _batches(indices: Sequence[int], batch_size: int, *, shuffle: bool, seed: int) -> list[list[int]]:
    indices = list(indices)
    if shuffle:
        random.Random(seed).shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _select_hidden(hidden: dict[str, torch.Tensor], indices: Sequence[int], device: torch.device):
    index = torch.tensor(indices, dtype=torch.long)
    return {key: value.index_select(0, index).to(device) for key, value in hidden.items()}


def _cache_hidden(
    transitions: Sequence[TerminalTransition],
    *,
    encoder_kind: str,
    hash_hidden_dim: int,
    policy_encoder: PolicyHiddenEncoder | None,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if encoder_kind == "hash":
        return hash_hidden_batch(transitions, hash_hidden_dim)
    if policy_encoder is None:
        raise ValueError("policy_encoder is required for hf-policy encoding")
    rows: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(transitions), batch_size):
        batch = policy_encoder(transitions[start : start + batch_size])
        for key, value in batch.items():
            rows.setdefault(key, []).append(value.detach().cpu())
    return {key: torch.cat(values, dim=0) for key, values in rows.items()}


def _run_epoch(
    *,
    model: TextLatentWorldModel,
    transitions: Sequence[TerminalTransition],
    indices: Sequence[int],
    cached_hidden: dict[str, torch.Tensor] | None,
    policy_encoder: PolicyHiddenEncoder | None,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    device: torch.device,
    seed: int,
    sigreg_coef: float,
    action_contrast_coef: float,
    alignment_coef: float,
    value_coef: float,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    total_loss = 0.0
    total_count = 0
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch_indices in _batches(indices, batch_size, shuffle=training, seed=seed):
            batch_transitions = [transitions[index] for index in batch_indices]
            if cached_hidden is not None:
                hidden = _select_hidden(cached_hidden, batch_indices, device)
            else:
                if policy_encoder is None:
                    raise RuntimeError("End-to-end training requires a policy hidden encoder")
                hidden = policy_encoder(batch_transitions)
            rewards = torch.tensor(
                [0.0 if row.reward is None else float(row.reward) for row in batch_transitions],
                dtype=torch.float32,
                device=device,
            )
            reward_mask = torch.tensor(
                [row.reward is not None for row in batch_transitions],
                dtype=torch.bool,
                device=device,
            )
            loss, metrics = model.compute_loss(
                state_hidden=hidden["state_hidden"],
                action_hidden=hidden["action_hidden"],
                target_hidden=hidden["target_hidden"],
                next_state_hidden=hidden["next_state_hidden"],
                has_next=hidden["has_next"],
                reward=rewards,
                reward_mask=reward_mask,
                sigreg_coef=sigreg_coef,
                action_contrast_coef=action_contrast_coef,
                alignment_coef=alignment_coef,
                value_coef=value_coef,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count = len(batch_indices)
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * count
    averaged = {key: value / max(total_count, 1) for key, value in totals.items()}
    return total_loss / max(total_count, 1), averaged


@torch.no_grad()
def _write_predictions(
    *,
    path: Path,
    model: TextLatentWorldModel,
    transitions: Sequence[TerminalTransition],
    cached_hidden: dict[str, torch.Tensor] | None,
    policy_encoder: PolicyHiddenEncoder | None,
    batch_size: int,
    device: torch.device,
) -> None:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for indices in _batches(range(len(transitions)), batch_size, shuffle=False, seed=0):
            batch_transitions = [transitions[index] for index in indices]
            hidden = (
                _select_hidden(cached_hidden, indices, device)
                if cached_hidden is not None
                else policy_encoder(batch_transitions)
            )
            output = model(
                state_hidden=hidden["state_hidden"],
                action_hidden=hidden["action_hidden"],
                target_hidden=hidden["target_hidden"],
                next_state_hidden=hidden["next_state_hidden"],
            )
            error = (output["pred_latent"] - output["target_latent"]).pow(2).mean(dim=-1)
            for offset, transition in enumerate(batch_transitions):
                row = {
                    "transition_id": transition.transition_id,
                    "trajectory_id": transition.trajectory_id,
                    "task_name": transition.task_name,
                    "turn_idx": transition.turn_idx,
                    "done": transition.done,
                    "has_next": transition.has_next,
                    "reward": transition.reward,
                    "latent_mse": float(error[offset].cpu()),
                    "pred_latent": output["pred_latent"][offset].cpu().tolist(),
                    "target_latent": output["target_latent"][offset].cpu().tolist(),
                    "value": None if output["value"] is None else float(output["value"][offset].cpu()),
                    "uncertainty": (
                        None if output["uncertainty"] is None else float(output["uncertainty"][offset].cpu())
                    ),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an action-conditioned latent world model on SETA trajectories.")
    parser.add_argument("--input", required=True, help="SETA trajectories directory, records JSONL, or replay .pt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", choices=["hash", "hf-policy"], default="hash")
    parser.add_argument("--hash-hidden-dim", type=int, default=256)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--hf-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--action-pool", choices=["mean", "last"], default="mean")
    parser.add_argument("--max-context-tokens", type=int, default=1536)
    parser.add_argument("--max-action-tokens", type=int, default=512)
    parser.add_argument("--max-feedback-tokens", type=int, default=512)
    parser.add_argument(
        "--backprop-to-llm",
        "--world-model-backprop-to-llm",
        dest="backprop_to_llm",
        action="store_true",
        default=False,
        help="Allow latent losses to update the policy LLM backbone. Default: false.",
    )
    parser.add_argument("--save-updated-llm", action="store_true")
    parser.add_argument(
        "--use-dapo-replay-buffer",
        "--world-model-use-dapo-replay-buffer",
        dest="use_dapo_replay_buffer",
        action="store_true",
        default=False,
        help="Route DAPO-collected transitions through the PR #16-compatible replay interface.",
    )
    parser.add_argument("--replay-buffer-size", type=int, default=2048)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--require-tool-feedback", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--predictor-type", choices=["adaln", "mlp"], default="adaln")
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--predictor-num-heads", type=int, default=4)
    parser.add_argument("--predictor-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--stop-grad-target", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--llm-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sigreg-coef", type=float, default=0.09)
    parser.add_argument("--action-contrast-coef", type=float, default=0.1)
    parser.add_argument("--alignment-coef", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.encoder == "hf-policy" and not args.hf_model:
        raise ValueError("--hf-model is required when --encoder hf-policy")
    if args.encoder == "hash" and args.backprop_to_llm:
        raise ValueError("--backprop-to-llm requires --encoder hf-policy")
    if args.latent_dim % args.predictor_num_heads != 0 and args.predictor_type == "adaln":
        raise ValueError("--latent-dim must be divisible by --predictor-num-heads")

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    transitions = load_terminal_transitions(
        args.input,
        max_trajectories=args.max_trajectories,
        max_transitions=args.max_transitions,
        require_tool_feedback=args.require_tool_feedback,
    )
    if not transitions:
        raise ValueError(f"No valid terminal transitions found in {args.input}")

    replay_stats = None
    if args.use_dapo_replay_buffer:
        replay = TrajectoryReplayBuffer(args.replay_buffer_size, seed=args.seed)
        replay.push(transitions, current_step=0)
        replay.save(output_dir / "dapo_replay.pt")
        records = replay.sample(len(replay), current_step=0)
        transitions = [TerminalTransition.from_dict(record) for record in records]
        replay_stats = replay.stats()

    device = _device(args.device)
    policy_encoder: PolicyHiddenEncoder | None = None
    if args.encoder == "hf-policy":
        policy_encoder = PolicyHiddenEncoder.from_pretrained(
            args.hf_model,
            device=str(device),
            dtype=args.hf_dtype,
            local_files_only=args.hf_local_files_only,
            hidden_layer=args.hidden_layer,
            action_pool=args.action_pool,
            max_context_tokens=args.max_context_tokens,
            max_action_tokens=args.max_action_tokens,
            max_feedback_tokens=args.max_feedback_tokens,
            backprop_to_llm=args.backprop_to_llm,
        )
        hidden_dim = policy_encoder.hidden_size
    else:
        hidden_dim = args.hash_hidden_dim

    cached_hidden = None
    if not args.backprop_to_llm:
        cached_hidden = _cache_hidden(
            transitions,
            encoder_kind=args.encoder,
            hash_hidden_dim=args.hash_hidden_dim,
            policy_encoder=policy_encoder,
            batch_size=args.encode_batch_size,
        )
        torch.save(
            {
                **cached_hidden,
                "record_metadata": [row.to_dict() for row in transitions],
                "encoder": args.encoder,
                "hf_model": args.hf_model,
            },
            output_dir / "hidden_cache.pt",
        )
        if policy_encoder is not None:
            del policy_encoder
            policy_encoder = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    config = TextLatentWorldModelConfig(
        state_hidden_dim=hidden_dim,
        action_hidden_dim=hidden_dim,
        target_hidden_dim=hidden_dim,
        latent_dim=args.latent_dim,
        adapter_dim=args.adapter_dim,
        predictor_type=args.predictor_type,
        predictor_depth=args.predictor_depth,
        predictor_num_heads=args.predictor_num_heads,
        predictor_mlp_ratio=args.predictor_mlp_ratio,
        value_head=args.value_coef != 0.0,
        uncertainty_head=False,
        stop_grad_target=args.stop_grad_target,
    )
    model = TextLatentWorldModel(config).to(device)
    parameter_groups: list[dict[str, Any]] = [{"params": model.parameters(), "lr": args.lr}]
    if args.backprop_to_llm:
        parameter_groups.append({"params": policy_encoder.model.parameters(), "lr": args.llm_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    train_indices, val_indices = _split_indices(len(transitions), args.val_ratio, args.seed)

    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        train_loss, train_metrics = _run_epoch(
            model=model,
            transitions=transitions,
            indices=train_indices,
            cached_hidden=cached_hidden,
            policy_encoder=policy_encoder,
            optimizer=optimizer,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + epoch,
            sigreg_coef=args.sigreg_coef,
            action_contrast_coef=args.action_contrast_coef,
            alignment_coef=args.alignment_coef,
            value_coef=args.value_coef,
        )
        val_loss = None
        val_metrics: dict[str, float] = {}
        if val_indices:
            val_loss, val_metrics = _run_epoch(
                model=model,
                transitions=transitions,
                indices=val_indices,
                cached_hidden=cached_hidden,
                policy_encoder=policy_encoder,
                optimizer=None,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed,
                sigreg_coef=args.sigreg_coef,
                action_contrast_coef=args.action_contrast_coef,
                alignment_coef=args.alignment_coef,
                value_coef=args.value_coef,
            )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    checkpoint = {
        "schema_version": "openclaw_terminal_latent_wm_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.__dict__,
        "state_dict": model.state_dict(),
        "runtime": vars(args),
        "record_count": len(transitions),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "history": history,
        "replay_stats": replay_stats,
        "backbone_updates_saved": bool(args.backprop_to_llm and args.save_updated_llm),
    }
    torch.save(checkpoint, output_dir / "latent_world_model.pt")
    _write_predictions(
        path=output_dir / "predictions.jsonl",
        model=model,
        transitions=transitions,
        cached_hidden=cached_hidden,
        policy_encoder=policy_encoder,
        batch_size=args.batch_size,
        device=device,
    )
    if args.backprop_to_llm and args.save_updated_llm:
        policy_encoder.model.save_pretrained(output_dir / "updated_llm")
        policy_encoder.tokenizer.save_pretrained(output_dir / "updated_llm")
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "record_count": len(transitions),
                "train_count": len(train_indices),
                "val_count": len(val_indices),
                "encoder": args.encoder,
                "backprop_to_llm": args.backprop_to_llm,
                "use_dapo_replay_buffer": args.use_dapo_replay_buffer,
                "replay_stats": replay_stats,
                "final": history[-1],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved latent world-model outputs to {output_dir}")


if __name__ == "__main__":
    main()
