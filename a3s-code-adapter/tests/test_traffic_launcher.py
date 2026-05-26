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


def test_traffic_launcher_does_not_auto_bind_adjacent_a3s_code_checkout(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "run_a3s_code_agent_traffic.sh"
    project_root = tmp_path / "OpenClaw-RL"
    adapter_dir = project_root / "a3s-code-adapter"
    adapter_dir.mkdir(parents=True)
    launcher = adapter_dir / source.name
    shutil.copy2(source, launcher)

    adjacent_sdk = tmp_path / "a3s-lab" / "Code" / "sdk" / "python"
    adjacent_sdk.mkdir(parents=True)
    (adjacent_sdk / "pyproject.toml").write_text(
        '[project]\nname = "a3s-code"\nversion = "999.0.0"\n',
        encoding="utf-8",
    )
    checker = tmp_path / "check_no_repo_root.py"
    checker.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "sys.exit(1 if os.environ.get('A3S_CODE_REPO_ROOT') else 0)\n",
        encoding="utf-8",
    )
    checker.chmod(0o755)

    env = os.environ.copy()
    env.pop("A3S_CODE_REPO_ROOT", None)
    env.update(
        {
            "A3S_CODE_PYTHON_BIN": str(checker),
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
