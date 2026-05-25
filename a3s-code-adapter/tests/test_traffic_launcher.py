from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def test_traffic_launcher_does_not_require_adjacent_a3s_code_checkout(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "run_a3s_code_agent_traffic.sh"
    project_root = tmp_path / "OpenClaw-RL"
    adapter_dir = project_root / "a3s-code-adapter"
    adapter_dir.mkdir(parents=True)
    launcher = adapter_dir / source.name
    shutil.copy2(source, launcher)

    env = os.environ.copy()
    env.pop("A3S_CODE_REPO_ROOT", None)
    env.update(
        {
            "A3S_CODE_PYTHON_BIN": "/bin/true",
            "A3S_CODE_REFRESH_SIMULATED_USER_BACKENDS_ON_START": "0",
            "A3S_CODE_REQUIRED_VERSION": "",
            "A3S_CODE_TRAFFIC_SESSION_LIMIT": "1",
            "ROLLOUT_BATCH_SIZE": "0",
            "N_SAMPLES_PER_PROMPT": "1",
            "NUM_ROLLOUT": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
