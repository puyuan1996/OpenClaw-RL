#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


DEFAULT_SKILLSBENCH_ROOT = Path.home() / "workspace" / "skillsbench"
DEFAULT_OUTPUT_DIR = Path.home() / "workspace" / "OpenClaw-RL" / "runs" / "skillsbench_image_builds"
DEFAULT_HTTP_PROXY = os.getenv(
    "A3S_CODE_DOCKER_BUILD_PROXY",
    "http://httpproxy-headless.kubebrain.svc.pjlab.local:3128",
)
DEFAULT_NO_PROXY = os.getenv(
    "A3S_CODE_DOCKER_BUILD_NO_PROXY",
    "localhost,127.0.0.1,0.0.0.0,::1,10.0.0.0/8,10.140.158.149,"
    "192.168.0.0/16,172.16.0.0/12,*.local,100.96.0.0/12,.pjlab.org.cn,"
    ".i.h.pjlab.org.cn,mirrors.i.h.pjlab.org.cn,pypi.i.h.pjlab.org.cn",
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class BuildRecord:
    task_id: str
    image: str
    status: str
    attempts: int
    duration_sec: float
    log_path: str
    command: list[str]
    error_tail: str = ""


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _image_name(task_id: str, prefix: str) -> str:
    return f"{prefix}{_sanitize(task_id)}"


def _tasks(skillsbench_root: Path, selected: set[str] | None) -> list[Path]:
    tasks_root = skillsbench_root / "tasks"
    tasks = []
    for task_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        if selected and task_dir.name not in selected:
            continue
        if (task_dir / "task.toml").exists() and (task_dir / "environment" / "Dockerfile").exists():
            tasks.append(task_dir)
    return tasks


def _read_completed(status_path: Path) -> set[str]:
    completed: set[str] = set()
    if not status_path.exists():
        return completed
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "ok":
            completed.add(str(payload.get("task_id")))
    return completed


def _image_exists(image: str) -> bool:
    result = subprocess.run(["docker", "image", "inspect", image], text=True, capture_output=True, check=False)
    return result.returncode == 0


def _proxy_build_args(http_proxy: str, no_proxy: str) -> list[str]:
    if not http_proxy:
        return []
    args: list[str] = []
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        args.extend(["--build-arg", f"{key}={http_proxy}"])
    for key in ("NO_PROXY", "no_proxy"):
        args.extend(["--build-arg", f"{key}={no_proxy}"])
    return args


def _compose_proxy_build_args(http_proxy: str, no_proxy: str) -> list[str]:
    if not http_proxy:
        return []
    args: list[str] = []
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        args.extend(["--build-arg", f"{key}={http_proxy}"])
    for key in ("NO_PROXY", "no_proxy"):
        args.extend(["--build-arg", f"{key}={no_proxy}"])
    return args


def _proxy_host_port(http_proxy: str) -> tuple[str, int] | None:
    if not http_proxy:
        return None
    parsed = urlparse(http_proxy)
    if not parsed.hostname:
        return None
    return parsed.hostname, parsed.port or 3128


def _coursier_proxy_args(http_proxy: str) -> str:
    proxy = _proxy_host_port(http_proxy)
    if not proxy:
        return ""
    host, port = proxy
    return (
        f"-J-Dhttp.proxyHost={host} -J-Dhttp.proxyPort={port} "
        f"-J-Dhttps.proxyHost={host} -J-Dhttps.proxyPort={port}"
    )


def _strip_pip_no_cache(line: str) -> str:
    return re.sub(r"(?<!\S)--no-cache-dir(?!\S)\s*", "", line)


def _patch_dockerfile_for_package_manager_cache(dockerfile_text: str) -> str:
    """Add BuildKit cache mounts for download-heavy package manager RUN steps.

    Docker layer cache only reuses identical RUN layers. SkillsBench tasks often
    use unique pip/npm install commands, and many pip commands pass
    --no-cache-dir, so downloads are repeated across images. BuildKit cache
    mounts keep the package-manager download cache outside the image while still
    allowing installed packages to be committed into the image layer.
    """

    lines = dockerfile_text.splitlines()
    patched: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if not stripped.startswith("RUN "):
            patched.append(line)
            index += 1
            continue

        block = [line]
        while block[-1].rstrip().endswith("\\") and index + 1 < len(lines):
            index += 1
            block.append(lines[index])

        block_text = "\n".join(block)
        mounts: list[str] = []
        if "pip install" in block_text or "pip3 install" in block_text or "python3 -m pip install" in block_text:
            mounts.append("--mount=type=cache,target=/root/.cache/pip,sharing=locked")
            block = [_strip_pip_no_cache(item) for item in block]
        if "npm install" in block_text or "npm ci" in block_text:
            mounts.append("--mount=type=cache,target=/root/.npm,sharing=locked")

        first_stripped = block[0].lstrip()
        if mounts and "--mount=type=cache" not in first_stripped:
            indent = block[0][: len(block[0]) - len(first_stripped)]
            after_run = first_stripped[len("RUN ") :]
            block[0] = f"{indent}RUN {' '.join(mounts)} {after_run}"
        patched.extend(block)
        index += 1

    return "\n".join(patched) + ("\n" if dockerfile_text.endswith("\n") else "")


def _patch_dockerfile_for_apt_mirrors(dockerfile_text: str) -> str:
    mirror_step = (
        "RUN for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources; do "
        "[ -f \"$f\" ] || continue; "
        "sed -i "
        "-e 's|https://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "
        "-e 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "
        "-e 's|https://ports.ubuntu.com/ubuntu-ports|http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports|g' "
        "-e 's|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports|g' "
        "-e 's|https://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "
        "-e 's|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "
        "-e 's|https://deb.debian.org/debian-security|http://mirrors.tuna.tsinghua.edu.cn/debian-security|g' "
        "-e 's|http://deb.debian.org/debian-security|http://mirrors.tuna.tsinghua.edu.cn/debian-security|g' "
        "-e 's|https://security.debian.org/debian-security|http://mirrors.tuna.tsinghua.edu.cn/debian-security|g' "
        "-e 's|http://security.debian.org/debian-security|http://mirrors.tuna.tsinghua.edu.cn/debian-security|g' "
        "-e 's|https://deb.debian.org/debian|http://mirrors.tuna.tsinghua.edu.cn/debian|g' "
        "-e 's|http://deb.debian.org/debian|http://mirrors.tuna.tsinghua.edu.cn/debian|g' "
        "\"$f\"; done"
    )
    patched: list[str] = []
    for line in dockerfile_text.splitlines():
        patched.append(line)
        if line.lstrip().upper().startswith("FROM "):
            patched.append(mirror_step)
            patched.append(
                "ENV PIP_INDEX_URL=http://mirrors.i.h.pjlab.org.cn/pypi/simple/ "
                "PIP_TRUSTED_HOST=mirrors.i.h.pjlab.org.cn"
            )
    return "\n".join(patched) + ("\n" if dockerfile_text.endswith("\n") else "")


def _patch_dockerfile_for_task(
    *,
    task_id: str,
    dockerfile_name: str,
    dockerfile_text: str,
    http_proxy: str,
) -> str:
    text = dockerfile_text
    if task_id == "fix-visual-stability" and dockerfile_name == "Dockerfile":
        replacements = {
            "libatk1.0-0": "libatk1.0-0t64",
            "libatk-bridge2.0-0": "libatk-bridge2.0-0t64",
            "libcups2": "libcups2t64",
            "libgtk-3-0": "libgtk-3-0t64",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

    if task_id == "speaker-diarization-subtitles" and dockerfile_name == "Dockerfile":
        text = text.replace(
            "RUN pip install --no-cache-dir -U pip setuptools wheel",
            'RUN pip install --no-cache-dir -U "pip<26" "setuptools<81" wheel',
        ).replace(
            "RUN pip install -U pip setuptools wheel",
            'RUN pip install -U "pip<26" "setuptools<81" wheel',
        )
        text = text.replace(
            "RUN pip install --no-cache-dir \\\n    speechbrain==1.0.3",
            "RUN pip install --no-cache-dir --no-build-isolation \\\n    speechbrain==1.0.3",
        ).replace(
            "RUN pip install \\\n    speechbrain==1.0.3",
            "RUN pip install --no-build-isolation \\\n    speechbrain==1.0.3",
        )
        text = text.replace(
            'python -c "import whisper; whisper.load_model(\'large-v3\')"',
            'python -c "import os, whisper; whisper._download(whisper._MODELS[\'large-v3\'], os.path.expanduser(\'~/.cache/whisper\'), False)"',
        )

    if task_id == "multilingual-video-dubbing" and dockerfile_name == "Dockerfile":
        text = text.replace(
            "    mojimoji==0.0.13",
            "    mojimoji==0.0.13 \\\n    transformers==4.49.0",
        )

    if task_id == "python-scala-translation" and dockerfile_name == "Dockerfile":
        proxy = _proxy_host_port(http_proxy)
        if proxy:
            host, port = proxy
            java_opts = (
                f"-Dhttp.proxyHost={host} -Dhttp.proxyPort={port} "
                f"-Dhttps.proxyHost={host} -Dhttps.proxyPort={port} "
                "-Dhttp.nonProxyHosts=localhost|127.*|*.local|*.pjlab.org.cn"
            )
            text = text.replace(
                'ENV PATH="/root/.local/share/coursier/bin:$PATH"\n',
                f'ENV PATH="/root/.local/share/coursier/bin:$PATH"\nENV JAVA_OPTS="{java_opts}"\n',
            )
            coursier_args = _coursier_proxy_args(http_proxy)
            text = text.replace(
                "&& ./cs setup --yes && ./cs install scala:2.13.12 scalac:2.13.12",
                f"&& ./cs {coursier_args} setup --yes && ./cs {coursier_args} install scala:2.13.12 scalac:2.13.12",
            )
    if task_id == "debug-trl-grpo" and dockerfile_name == "Dockerfile":
        text = text.replace(
            "pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu",
            "pip install --no-cache-dir --default-timeout 180 --retries 10 "
            "torch==2.4.1 --index-url http://mirrors.i.h.pjlab.org.cn/pypi/simple/ "
            "--trusted-host mirrors.i.h.pjlab.org.cn",
        )
    return text


def _prepare_patched_context(
    *,
    task_dir: Path,
    output_dir: Path,
    http_proxy: str,
    patch_package_manager_cache: bool,
) -> Path:
    env_dir = task_dir / "environment"
    patched_root = output_dir / "patched_contexts" / _sanitize(task_dir.name)
    if patched_root.exists():
        shutil.rmtree(patched_root)
    shutil.copytree(env_dir, patched_root, symlinks=True)

    for dockerfile in sorted(patched_root.glob("Dockerfile*")):
        if not dockerfile.is_file():
            continue
        dockerfile_text = dockerfile.read_text(encoding="utf-8", errors="ignore")
        if not dockerfile_text.startswith("# syntax="):
            dockerfile_text = "# syntax=docker/dockerfile:1\n" + dockerfile_text
        dockerfile_text = _patch_dockerfile_for_task(
            task_id=task_dir.name,
            dockerfile_name=dockerfile.name,
            dockerfile_text=dockerfile_text,
            http_proxy=http_proxy,
        )
        dockerfile_text = _patch_dockerfile_for_apt_mirrors(dockerfile_text)
        if patch_package_manager_cache:
            dockerfile_text = _patch_dockerfile_for_package_manager_cache(dockerfile_text)
        dockerfile.write_text(dockerfile_text, encoding="utf-8")
    return patched_root


def _base_env(http_proxy: str, no_proxy: str, *, buildkit: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "1" if buildkit else "0"
    env["COMPOSE_DOCKER_CLI_BUILD"] = "1" if buildkit else "0"
    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
        env["HTTPS_PROXY"] = http_proxy
        env["https_proxy"] = http_proxy
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    return env


def _compose_env(
    task_dir: Path,
    image: str,
    http_proxy: str,
    no_proxy: str,
    output_dir: Path,
    *,
    buildkit: bool,
    context_dir: Path,
) -> dict[str, str]:
    env = _base_env(http_proxy, no_proxy, buildkit=buildkit)
    logs_root = output_dir / "compose_mounts" / _sanitize(task_dir.name)
    verifier = logs_root / "verifier"
    agent = logs_root / "agent"
    verifier.mkdir(parents=True, exist_ok=True)
    agent.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "MAIN_IMAGE_NAME": image,
            "CONTEXT_DIR": str(context_dir.resolve()),
            "TEST_DIR": "/tests",
            "HOST_VERIFIER_LOGS_PATH": str(verifier.resolve()),
            "HOST_AGENT_LOGS_PATH": str(agent.resolve()),
            "ENV_VERIFIER_LOGS_PATH": "/logs/verifier",
            "ENV_AGENT_LOGS_PATH": "/logs/agent",
            "CPUS": "2",
            "MEMORY": "8192M",
            "NETWORK_MODE": "bridge",
            "GOOGLE_AUTH_PATH": str((Path.home() / ".config" / "gcloud").resolve()),
        }
    )
    return env


def _build_command(
    task_dir: Path,
    image: str,
    *,
    http_proxy: str,
    no_proxy: str,
    use_host_network: bool,
    output_dir: Path,
    buildkit: bool,
    patch_package_manager_cache: bool,
) -> tuple[list[str], dict[str, str], Path]:
    env_dir = task_dir / "environment"
    compose_path = env_dir / "docker-compose.yaml"
    context_dir = env_dir
    if buildkit:
        context_dir = _prepare_patched_context(
            task_dir=task_dir,
            output_dir=output_dir,
            http_proxy=http_proxy,
            patch_package_manager_cache=patch_package_manager_cache,
        )
        compose_path = context_dir / "docker-compose.yaml"
    if compose_path.exists():
        command = [
            "docker",
            "compose",
            "-p",
            f"skillsbench-prebuild-{_sanitize(task_dir.name).lower()}",
            "-f",
            str(compose_path.resolve()),
            "build",
        ]
        command.extend(_compose_proxy_build_args(http_proxy, no_proxy))
        return (
            command,
            _compose_env(
                task_dir,
                image,
                http_proxy,
                no_proxy,
                output_dir,
                buildkit=buildkit,
                context_dir=context_dir,
            ),
            context_dir,
        )

    cmd = ["docker", "build"]
    if buildkit:
        cmd.append("--progress=plain")
    if use_host_network:
        cmd.append("--network=host")
    cmd.extend(_proxy_build_args(http_proxy, no_proxy))
    cmd.extend(["-t", image, str(context_dir.resolve())])
    return cmd, _base_env(http_proxy, no_proxy, buildkit=buildkit), context_dir


def _tail(text: str, max_chars: int = 4000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run_build(
    task_dir: Path,
    *,
    image: str,
    output_dir: Path,
    attempts: int,
    retry_sleep_sec: int,
    timeout_sec: int | None,
    http_proxy: str,
    no_proxy: str,
    use_host_network: bool,
    patch_package_manager_cache: bool,
) -> BuildRecord:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{_sanitize(task_dir.name)}.log"
    start = time.time()
    last_tail = ""
    used_attempts = 0
    buildkit = patch_package_manager_cache
    for attempt in range(1, attempts + 1):
        used_attempts = attempt
        with log_path.open("a", encoding="utf-8") as log:
            command, env, cwd = _build_command(
                task_dir,
                image,
                http_proxy=http_proxy,
                no_proxy=no_proxy,
                use_host_network=use_host_network,
                output_dir=output_dir,
                buildkit=buildkit,
                patch_package_manager_cache=patch_package_manager_cache,
            )
            log.write(f"\n\n===== attempt {attempt}/{attempts} buildkit={int(buildkit)} {datetime.now().isoformat()} =====\n")
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            try:
                result = subprocess.run(
                    command,
                    cwd=str(cwd),
                    env=env,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_sec,
                    check=False,
                )
                return_code = result.returncode
            except subprocess.TimeoutExpired:
                return_code = -1
                log.write(f"\nTIMEOUT after {timeout_sec}s\n")
        last_tail = _tail(log_path.read_text(encoding="utf-8", errors="ignore"))
        if return_code == 0 and _image_exists(image):
            return BuildRecord(
                task_id=task_dir.name,
                image=image,
                status="ok",
                attempts=used_attempts,
                duration_sec=time.time() - start,
                log_path=str(log_path),
                command=command,
            )
        if not buildkit and (
            "dockerfile parse error" in last_tail.lower()
            or "unknown instruction" in last_tail.lower()
            or "buildkit isn't enabled" in last_tail.lower()
        ):
            buildkit = True
            continue
        if attempt < attempts:
            time.sleep(retry_sleep_sec)

    return BuildRecord(
        task_id=task_dir.name,
        image=image,
        status="failed",
        attempts=used_attempts,
        duration_sec=time.time() - start,
        log_path=str(log_path),
        command=command,
        error_tail=last_tail,
    )


def _write_prebuilt_task_overlay(
    *,
    skillsbench_root: Path,
    output_root: Path,
    tasks: Iterable[Path],
    image_prefix: str,
) -> Path:
    overlay_root = output_root / "prebuilt_tasks"
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_root.mkdir(parents=True, exist_ok=True)

    docker_image_re = re.compile(r"^docker_image\s*=.*$", re.MULTILINE)
    env_section_re = re.compile(r"^\[environment\]\s*$", re.MULTILINE)

    for task_dir in tasks:
        target = overlay_root / task_dir.name
        shutil.copytree(task_dir, target, symlinks=True)
        task_toml = target / "task.toml"
        text = task_toml.read_text(encoding="utf-8")
        image = _image_name(task_dir.name, image_prefix)
        if docker_image_re.search(text):
            text = docker_image_re.sub(f'docker_image = "{image}"', text)
        elif env_section_re.search(text):
            text = env_section_re.sub(f'[environment]\ndocker_image = "{image}"', text, count=1)
        else:
            text += f'\n[environment]\ndocker_image = "{image}"\n'
        task_toml.write_text(text, encoding="utf-8")
    return overlay_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prebuild reusable SkillsBench Docker images for Harbor/a3s-code eval.")
    parser.add_argument("--skillsbench-root", type=Path, default=Path(os.getenv("A3S_CODE_SKILLSBENCH_ROOT", DEFAULT_SKILLSBENCH_ROOT)))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--task", action="append", default=[], help="Task id to build. Can be repeated. Defaults to all tasks.")
    parser.add_argument("--image-prefix", default=os.getenv("A3S_CODE_SKILLSBENCH_IMAGE_PREFIX", "hb__"))
    parser.add_argument("--attempts", type=int, default=int(os.getenv("A3S_CODE_SKILLSBENCH_BUILD_ATTEMPTS", "3")))
    parser.add_argument("--retry-sleep-sec", type=int, default=int(os.getenv("A3S_CODE_SKILLSBENCH_BUILD_RETRY_SLEEP_SEC", "20")))
    parser.add_argument("--timeout-sec", type=int, default=int(os.getenv("A3S_CODE_SKILLSBENCH_BUILD_TIMEOUT_SEC", "7200")))
    parser.add_argument("--http-proxy", default=DEFAULT_HTTP_PROXY)
    parser.add_argument("--no-proxy", default=DEFAULT_NO_PROXY)
    parser.add_argument("--no-host-network", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the target image already exists.")
    parser.add_argument("--write-prebuilt-task-overlay", action="store_true")
    parser.add_argument(
        "--patch-package-manager-cache",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("A3S_CODE_SKILLSBENCH_PATCH_PACKAGE_MANAGER_CACHE", True),
        help="Patch direct Dockerfiles to use BuildKit cache mounts for pip/npm downloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.task) if args.task else None
    tasks = _tasks(args.skillsbench_root, selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.jsonl"
    completed = _read_completed(status_path)

    if not tasks:
        print("No matching SkillsBench tasks found.", file=sys.stderr)
        return 2

    print(f"output_dir={args.output_dir}")
    print(f"tasks={len(tasks)}")
    print(f"http_proxy={'set' if args.http_proxy else 'unset'} host_network={not args.no_host_network}")
    print(f"patch_package_manager_cache={args.patch_package_manager_cache}")

    ok = 0
    failed = 0
    for index, task_dir in enumerate(tasks, start=1):
        image = _image_name(task_dir.name, args.image_prefix)
        if not args.force and (task_dir.name in completed or _image_exists(image)):
            record = BuildRecord(
                task_id=task_dir.name,
                image=image,
                status="skipped",
                attempts=0,
                duration_sec=0.0,
                log_path="",
                command=[],
            )
            print(f"[{index}/{len(tasks)}] skip {task_dir.name} -> {image}")
        else:
            print(f"[{index}/{len(tasks)}] build {task_dir.name} -> {image}", flush=True)
            record = _run_build(
                task_dir,
                image=image,
                output_dir=args.output_dir,
                attempts=max(1, args.attempts),
                retry_sleep_sec=max(0, args.retry_sleep_sec),
                timeout_sec=args.timeout_sec if args.timeout_sec > 0 else None,
                http_proxy=args.http_proxy,
                no_proxy=args.no_proxy,
                use_host_network=not args.no_host_network,
                patch_package_manager_cache=args.patch_package_manager_cache,
            )
            print(f"  {record.status} attempts={record.attempts} duration={record.duration_sec:.1f}s log={record.log_path}")
        with status_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        if record.status in {"ok", "skipped"}:
            ok += 1
        else:
            failed += 1

    if args.write_prebuilt_task_overlay:
        overlay = _write_prebuilt_task_overlay(
            skillsbench_root=args.skillsbench_root,
            output_root=args.output_dir,
            tasks=tasks,
            image_prefix=args.image_prefix,
        )
        print(f"prebuilt_task_overlay={overlay}")

    print(f"ok_or_skipped={ok} failed={failed} status={status_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
