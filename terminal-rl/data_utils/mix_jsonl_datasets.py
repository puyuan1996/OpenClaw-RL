#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_source(raw: str) -> tuple[Path, float]:
    if ":" not in raw:
        raise ValueError(f"source must be PATH:RATIO, got {raw!r}")
    path_raw, ratio_raw = raw.rsplit(":", 1)
    ratio = float(ratio_raw)
    if ratio <= 0:
        raise ValueError(f"source ratio must be positive, got {raw!r}")
    return Path(path_raw), ratio


def allocate_counts(
    lengths: list[int], ratios: list[float], total: int | None, oversample: bool
) -> list[int]:
    ratio_sum = sum(ratios)
    if total is None:
        if oversample:
            total = sum(lengths)
        else:
            total = int(min(length / (ratio / ratio_sum) for length, ratio in zip(lengths, ratios)))

    counts = [int(total * ratio / ratio_sum) for ratio in ratios]
    while sum(counts) < total:
        remainders = [
            total * ratio / ratio_sum - count for ratio, count in zip(ratios, counts)
        ]
        idx = max(range(len(remainders)), key=remainders.__getitem__)
        counts[idx] += 1

    if not oversample:
        for count, length in zip(counts, lengths):
            if count > length:
                raise ValueError(
                    f"requested {count} samples from source with only {length}; "
                    "lower --total or pass --oversample"
                )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically mix JSONL datasets by source ratios."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Input as PATH:RATIO. Pass multiple times.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--oversample", action="store_true")
    args = parser.parse_args()

    parsed_sources = [parse_source(x) for x in args.source]
    rng = random.Random(args.seed)
    datasets = [read_jsonl(path) for path, _ratio in parsed_sources]
    ratios = [ratio for _path, ratio in parsed_sources]
    counts = allocate_counts(
        [len(dataset) for dataset in datasets], ratios, args.total, args.oversample
    )

    mixed: list[dict[str, Any]] = []
    summary = []
    for (path, ratio), dataset, count in zip(parsed_sources, datasets, counts):
        if args.oversample and count > len(dataset):
            selected = [rng.choice(dataset) for _ in range(count)]
        else:
            selected = rng.sample(dataset, count)
        mixed.extend(selected)
        summary.append(
            {
                "path": str(path),
                "available": len(dataset),
                "selected": count,
                "ratio": ratio,
            }
        )

    rng.shuffle(mixed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in mixed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "total": len(mixed),
                "seed": args.seed,
                "sources": summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
