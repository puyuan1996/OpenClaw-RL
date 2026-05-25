from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_export_report_writes_summary_file() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/export_report.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads((ROOT / "outputs" / "weekly_report.json").read_text(encoding="utf-8"))
    assert payload["total_tickets"] == 680
    assert payload["total_customer_escalations"] == 18
    assert payload["top_channel"] == "email"
    assert payload["channels"]["chat"] == 282
