from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "weekly_metrics.csv"
OUTPUT_PATH = ROOT / "outputs" / "weekly_report.json"


def load_rows() -> list[dict[str, str]]:
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_report(rows: list[dict[str, str]]) -> dict[str, object]:
    total_tickets = sum(int(row["tickets"]) for row in rows)
    total_escalations = sum(int(row["customer_escalations"]) for row in rows)
    channel_totals: dict[str, int] = defaultdict(int)
    for row in rows:
        channel_totals[row["channel"]] += int(row["tickets"])
    top_channel = max(channel_totals.items(), key=lambda item: item[1])[0]
    return {
        "weeks": [row["week"] for row in rows],
        "total_tickets": total_tickets,
        "total_customer_escalations": total_escalations,
        "top_channel": top_channel,
        "channels": channel_totals,
    }


def main() -> int:
    rows = load_rows()
    report = build_report(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
