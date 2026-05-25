from __future__ import annotations

import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
A3S_CODE_ROOT = Path(os.getenv("A3S_CODE_REPO_ROOT", str(Path.home() / "workspace" / "a3s-lab" / "Code")))
A3S_CODE_SDK_PYTHON = A3S_CODE_ROOT / "sdk" / "python"
DEFAULT_CONDA_ENV = Path(
    os.getenv(
        "CONDA_ENV",
        os.getenv("CONDA_PREFIX", sys.prefix),
    )
)
DEFAULT_PYTHON_BIN = Path(os.getenv("A3S_CODE_PYTHON", str(DEFAULT_CONDA_ENV / "bin" / "python3")))
if not DEFAULT_PYTHON_BIN.exists():
    DEFAULT_PYTHON_BIN = Path(sys.executable)
DEFAULT_MATURIN_BIN = Path(os.getenv("A3S_CODE_MATURIN", shutil.which("maturin") or str(DEFAULT_CONDA_ENV / "bin" / "maturin")))
DEFAULT_WHEEL_DIR = REPO_ROOT / ".artifacts" / "a3s_code_wheels"
PROJECT_VERSION_RE = re.compile(r"(?m)^version\s*=\s*[\"']([^\"']+)[\"']")
WHEEL_FILENAME_RE = re.compile(
    r"^a3s_code-(?P<version>[^-]+)-(?P<python>cp\d+)-(?P<abi>cp\d+)-(?P<platform>.+)\.whl$"
)


def run_checked(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        check=True,
        text=True,
        capture_output=True,
    )


def current_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def target_python_tag() -> str:
    if DEFAULT_PYTHON_BIN.exists():
        try:
            result = run_checked(
                [
                    str(DEFAULT_PYTHON_BIN),
                    "-c",
                    "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
                ]
            )
            tag = (result.stdout or "").strip()
            if tag:
                return tag
        except subprocess.CalledProcessError:
            pass
    return current_python_tag()


def current_a3s_code_version() -> str | None:
    pyproject = A3S_CODE_SDK_PYTHON / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    match = PROJECT_VERSION_RE.search(text)
    if not match:
        return None
    return match.group(1).strip() or None


def find_existing_wheel(wheel_dir: Path) -> Path | None:
    if not wheel_dir.exists():
        return None
    preferred_tag = target_python_tag()
    current_version = current_a3s_code_version()
    version_pattern = current_version or "*"
    preferred_pattern = f"a3s_code-{version_pattern}-{preferred_tag}-{preferred_tag}-*.whl"
    wheels = sorted(wheel_dir.glob(preferred_pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if wheels:
        return wheels[0]
    if current_version:
        return None
    fallback_wheels = sorted(wheel_dir.glob("a3s_code-*.whl"), key=lambda path: path.stat().st_mtime, reverse=True)
    return fallback_wheels[0] if fallback_wheels else None


def _existing_wheel_from_env(name: str) -> Path | None:
    explicit = os.getenv(name, "").strip()
    if not explicit:
        return None
    path = Path(explicit).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def ensure_a3s_code_wheel(wheel_dir: Path | None = None) -> Path:
    explicit = _existing_wheel_from_env("A3S_CODE_WHEEL_PATH")
    if explicit is not None:
        return explicit

    wheel_dir = (wheel_dir or DEFAULT_WHEEL_DIR).resolve()
    existing = find_existing_wheel(wheel_dir)
    if existing is not None:
        return existing

    wheel_dir.mkdir(parents=True, exist_ok=True)
    python_bin = str(DEFAULT_PYTHON_BIN)
    maturin_bin = str(DEFAULT_MATURIN_BIN)
    env = {
        "PATH": f"{DEFAULT_CONDA_ENV / 'bin'}:{Path.home() / '.cargo' / 'bin'}:{os.environ.get('PATH', '')}",
        "CONDA_PREFIX": str(DEFAULT_CONDA_ENV),
        "VIRTUAL_ENV": str(DEFAULT_CONDA_ENV),
    }
    run_checked(
        [maturin_bin, "build", "--release", "-o", str(wheel_dir)],
        cwd=A3S_CODE_SDK_PYTHON,
        env=env,
    )
    built = find_existing_wheel(wheel_dir)
    if built is None:
        raise RuntimeError(f"maturin build completed but no wheel was produced in {wheel_dir}")
    return built


def _wheel_version(path: Path) -> str | None:
    match = WHEEL_FILENAME_RE.match(path.name)
    return match.group("version") if match else None


def find_skillsbench_compatible_wheel(default_wheel: Path) -> Path:
    """Prefer a wheel that works across older SkillsBench task images."""

    version = _wheel_version(default_wheel)
    if not version:
        return default_wheel

    preferred_python_tag = os.getenv("A3S_CODE_SKILLSBENCH_PYTHON_TAG", "cp311").strip() or "cp311"
    platform_order = (
        "manylinux_2_31",
        "manylinux_2_28",
        "manylinux_2_17",
        "manylinux2014",
    )
    for platform in platform_order:
        candidates = sorted(
            default_wheel.parent.glob(
                f"a3s_code-{version}-{preferred_python_tag}-{preferred_python_tag}-*{platform}*.whl"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return default_wheel


def ensure_skillsbench_a3s_code_wheel(wheel_dir: Path | None = None) -> Path:
    explicit = _existing_wheel_from_env("A3S_CODE_SKILLSBENCH_WHEEL_PATH")
    if explicit is not None:
        return explicit
    return find_skillsbench_compatible_wheel(ensure_a3s_code_wheel(wheel_dir))


def detect_host_ip() -> str:
    override = os.getenv("A3S_CODE_BENCHMARK_HOST_IP", "").strip()
    if override:
        return override

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host_ip = sock.getsockname()[0]
            if host_ip:
                return host_ip
    except OSError:
        pass

    return os.getenv("SLIME_HOST_IP", "127.0.0.1")


def detect_model_base_url(default_port: int = 30000) -> str:
    explicit = (
        os.getenv("A3S_CODE_MODEL_BASE_URL", "").strip()
        or os.getenv("A3S_CODE_BENCHMARK_BASE_URL", "").strip()
    )
    if explicit:
        return explicit.rstrip("/")
    return f"http://{detect_host_ip()}:{default_port}"


def render_agent_config(
    *,
    provider: str,
    base_url: str,
    model_name: str,
    api_key: str,
    context_tokens: int = 16384,
    output_tokens: int = 4096,
    session_id_header: str | None = "X-Session-Id",
) -> str:
    session_header_line = ""
    if session_id_header:
        session_header_line = f'  session_id_header = "{session_id_header}"\n'
    return (
        f'default_model = "{provider}/{model_name}"\n\n'
        f'providers "{provider}" {{\n'
        f'  api_key = "{api_key}"\n'
        f'  base_url = "{base_url}"\n'
        f"{session_header_line}"
        f'  models "{model_name}" {{\n'
        f'    name = "{model_name}"\n'
        f"    max_tokens = {output_tokens}\n"
        f"    context_tokens = {context_tokens}\n"
        "    tool_call = true\n\n"
        "  }\n"
        "}\n"
    )


def render_openai_agent_config(
    *,
    base_url: str,
    model_name: str,
    api_key: str,
    context_tokens: int = 16384,
    output_tokens: int = 4096,
    session_id_header: str | None = "X-Session-Id",
) -> str:
    return render_agent_config(
        provider="openai",
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        session_id_header=session_id_header,
    )


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)
