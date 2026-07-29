from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from .cache_text_hidden import validate_hidden_cache_integrity
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig


def _load_cache_payload(path: Path, *, require_verified: bool = False) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}")
    validate_hidden_cache_integrity(payload, require_verified=require_verified)
    return payload


def _load_tensor_dataset(payload: dict, path: Path) -> TensorDataset:
    required = ["state_hidden", "action_hidden", "target_hidden"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}. Expected {required}.")
    tensors = [payload[key].float() for key in required]
    count = int(tensors[0].shape[0])
    for key, tensor in zip(required, tensors):
        if int(tensor.shape[0]) != count:
            raise ValueError(f"Inconsistent first dimension for {key}: expected {count}, got {int(tensor.shape[0])}")
    if "reward" in payload:
        reward = payload["reward"].float()
        if int(reward.shape[0]) != count:
            raise ValueError(f"Inconsistent first dimension for reward: expected {count}, got {int(reward.shape[0])}")
        tensors.append(reward)
        if "reward_mask" in payload:
            reward_mask = payload["reward_mask"].bool()
            if int(reward_mask.shape[0]) != count:
                raise ValueError(
                    f"Inconsistent first dimension for reward_mask: expected {count}, got {int(reward_mask.shape[0])}"
                )
            tensors.append(reward_mask)
    return TensorDataset(*tensors)


def _split_dataset(
    dataset: TensorDataset,
    *,
    val_ratio: float,
    seed: int,
    record_metadata: list[dict] | None = None,
    group_key: str = "context_hash",
):
    if val_ratio <= 0.0 or len(dataset) < 2:
        indices = list(range(len(dataset)))
        return Subset(dataset, indices), None, {
            "strategy": "no_validation",
            "group_key": group_key,
            "group_values_complete": False,
            "group_disjoint": False,
            "train_indices": indices,
            "val_indices": [],
        }

    groups: dict[str, list[int]] = defaultdict(list)
    group_values: list[str | None] = []
    if isinstance(record_metadata, list) and len(record_metadata) == len(dataset):
        for row in record_metadata:
            value = row.get(group_key) if isinstance(row, dict) else None
            value = str(value).strip() if value is not None else ""
            group_values.append(value or None)
    group_values_complete = bool(group_values) and all(value is not None for value in group_values)
    if group_values_complete:
        for idx, value in enumerate(group_values):
            groups[str(value)].append(idx)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if group_values_complete and len(groups) >= 2:
        group_names = list(groups)
        order = torch.randperm(len(group_names), generator=generator).tolist()
        target_val_size = max(1, min(int(round(len(dataset) * val_ratio)), len(dataset) - 1))
        val_indices: list[int] = []
        for position in order[:-1]:
            val_indices.extend(groups[group_names[position]])
            if len(val_indices) >= target_val_size:
                break
        val_set = set(val_indices)
        train_indices = [idx for idx in range(len(dataset)) if idx not in val_set]
        strategy = "group_holdout"
    elif group_values_complete:
        indices = list(range(len(dataset)))
        return Subset(dataset, indices), None, {
            "strategy": "no_validation_insufficient_groups",
            "group_key": group_key,
            "group_values_complete": True,
            "group_disjoint": False,
            "train_indices": indices,
            "val_indices": [],
        }
    else:
        val_size = max(1, min(int(round(len(dataset) * val_ratio)), len(dataset) - 1))
        permutation = torch.randperm(len(dataset), generator=generator).tolist()
        val_indices = permutation[:val_size]
        train_indices = permutation[val_size:]
        strategy = "record_holdout_fallback"

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices), {
        "strategy": strategy,
        "group_key": group_key,
        "group_values_complete": group_values_complete,
        "group_disjoint": strategy == "group_holdout",
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


def _reward_label_count(dataset: TensorDataset | Subset) -> int:
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if not isinstance(base_dataset, TensorDataset) or len(base_dataset.tensors) < 4:
        return 0
    if len(base_dataset.tensors) == 4:
        return len(dataset)

    reward_mask = base_dataset.tensors[4].bool()
    if isinstance(dataset, Subset):
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        reward_mask = reward_mask[indices]
    return int(reward_mask.sum().item())


def _run_epoch(
    *,
    model: TextLatentWorldModel,
    loader: DataLoader,
    device: torch.device,
    optim: torch.optim.Optimizer | None,
    sigreg_coef: float,
    action_contrast_coef: float,
    value_coef: float,
) -> tuple[float, int, int]:
    total_loss = 0.0
    total_count = 0
    optimizer_step_count = 0
    value_update_step_count = 0
    train = optim is not None
    model.train(train)
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            batch = [x.to(device) for x in batch]
            reward = batch[3] if len(batch) > 3 else None
            reward_mask = batch[4] if len(batch) > 4 else None
            loss, _metrics = model.compute_loss(
                state_hidden=batch[0],
                action_hidden=batch[1],
                target_hidden=batch[2],
                reward=reward,
                reward_mask=reward_mask,
                sigreg_coef=sigreg_coef,
                action_contrast_coef=action_contrast_coef,
                value_coef=value_coef,
            )
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                optimizer_step_count += 1
                if value_coef > 0.0 and reward is not None and (
                    reward_mask is None or bool(reward_mask.bool().any().item())
                ):
                    value_update_step_count += 1
            total_loss += float(loss.detach().cpu()) * batch[0].size(0)
            total_count += batch[0].size(0)
    return total_loss / max(total_count, 1), optimizer_step_count, value_update_step_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small JEPA-style text latent world-model probe.")
    parser.add_argument("--input", required=True, help="Torch file with state_hidden/action_hidden/target_hidden tensors.")
    parser.add_argument("--output", required=True, help="Checkpoint path for the trained probe.")
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigreg-coef", type=float, default=0.1)
    parser.add_argument("--action-contrast-coef", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--split-group-key", default="context_hash")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("--lr must be finite and positive")

    input_path = Path(args.input)
    payload = _load_cache_payload(
        input_path,
        require_verified=args.value_coef > 0.0 or args.val_ratio > 0.0,
    )
    dataset = _load_tensor_dataset(payload, input_path)
    if len(dataset) == 0:
        raise ValueError(f"No cached world-model records found in {input_path}.")
    first = dataset[0]
    config = TextLatentWorldModelConfig(
        state_hidden_dim=int(first[0].shape[-1]),
        action_hidden_dim=int(first[1].shape[-1]),
        target_hidden_dim=int(first[2].shape[-1]),
        latent_dim=args.latent_dim,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextLatentWorldModel(config).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dataset, val_dataset, split_metadata = _split_dataset(
        dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        record_metadata=payload.get("record_metadata"),
        group_key=args.split_group_key,
    )
    train_reward_label_count = _reward_label_count(train_dataset)
    if args.value_coef > 0.0 and train_reward_label_count <= 0:
        raise ValueError("--value-coef is positive but the train split has no valid reward labels")
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(
        "dataset "
        f"total={len(dataset)} train={len(train_dataset)} val={0 if val_dataset is None else len(val_dataset)} "
        f"device={device}"
    )

    final_train_loss = None
    final_val_loss = None
    optimizer_step_count = 0
    value_update_step_count = 0
    for epoch in range(args.epochs):
        train_loss, epoch_optimizer_steps, epoch_value_steps = _run_epoch(
            model=model,
            loader=loader,
            device=device,
            optim=optim,
            sigreg_coef=args.sigreg_coef,
            action_contrast_coef=args.action_contrast_coef,
            value_coef=args.value_coef,
        )
        optimizer_step_count += epoch_optimizer_steps
        value_update_step_count += epoch_value_steps
        if val_loader is not None:
            val_loss, _, _ = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optim=None,
                sigreg_coef=args.sigreg_coef,
                action_contrast_coef=args.action_contrast_coef,
                value_coef=args.value_coef,
            )
            print(f"epoch={epoch + 1} loss={train_loss:.6f} val_loss={val_loss:.6f}")
            final_val_loss = val_loss
        else:
            print(f"epoch={epoch + 1} loss={train_loss:.6f}")
        final_train_loss = train_loss

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "openclaw_text_jepa_probe_checkpoint_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "cache_metadata": payload.get("metadata", {}),
        "record_count": len(dataset),
        "has_reward": "reward" in payload,
        "reward_mask_count": int(payload.get("reward_mask", torch.zeros(0, dtype=torch.bool)).sum().item())
        if "reward_mask" in payload
        else None,
        "state_hidden_shape": tuple(payload["state_hidden"].shape),
        "action_hidden_shape": tuple(payload["action_hidden"].shape),
        "target_hidden_shape": tuple(payload["target_hidden"].shape),
        "train_count": len(train_dataset),
        "train_reward_label_count": train_reward_label_count,
        "val_count": 0 if val_dataset is None else len(val_dataset),
        "split": split_metadata,
        "optimizer_step_count": optimizer_step_count,
        "value_update_step_count": value_update_step_count,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "hyperparameters": {
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "sigreg_coef": args.sigreg_coef,
            "action_contrast_coef": args.action_contrast_coef,
            "value_coef": args.value_coef,
            "val_ratio": args.val_ratio,
            "split_group_key": args.split_group_key,
            "seed": args.seed,
        },
    }
    torch.save({"config": config.__dict__, "state_dict": model.state_dict(), "metadata": metadata}, out)
    print(f"saved probe checkpoint to {out}")


if __name__ == "__main__":
    main()
