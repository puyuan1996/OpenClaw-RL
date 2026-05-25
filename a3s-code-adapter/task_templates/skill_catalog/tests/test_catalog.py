from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_catalog_writes_expected_entries(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_catalog.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    output_path = ROOT / "catalog" / "skills_index.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert [item["slug"] for item in payload] == ["calendar_planning", "email_triage"]
    assert payload[0]["summary"]
    assert "calendar" in payload[0]["tags"]
    assert payload[1]["scripts"] == ["scripts/filter_priority.py"]
