#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import toml
import yaml

if __package__ in {None, ""}:  # pragma: no cover - script entrypoint path bootstrap
    PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from a3s_code_benchmarks.benchmark_runtime_utils import shell_join
from a3s_code_benchmarks.official.worker_local_docker import start_worker_local_docker


DEFAULT_IMAGE_REPO = "registry.example.com/openclaw-rl/a3s-code"


@dataclass
class ImageSpec:
    task: str
    service: str
    image: str
    context: str
    dockerfile: str


@dataclass
class BuildRecord:
    task: str
    service: str
    image: str
    status: str
    build_returncode: int | None = None
    push_returncode: int | None = None
    elapsed_sec: float = 0.0
    log_dir: str | None = None
    error: str | None = None


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip(".-").lower()
    return cleaned or "task"


def _task_env_hash(environment_dir: Path) -> str:
    digest = hashlib.sha1()
    for path in sorted(environment_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(environment_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:10]


def _image_for_task(repo: str, prefix: str, task: str, service: str, digest: str) -> str:
    service_suffix = "" if service == "main" else f"-{service}"
    tag = _sanitize(f"{prefix}-{task}{service_suffix}-{digest}")
    if len(tag) > 128:
        keep = 128 - len(digest) - len(service_suffix) - len(prefix) - 3
        tag = _sanitize(f"{prefix}-{task[:max(8, keep)]}{service_suffix}-{digest}")[:128]
    return f"{repo}:{tag}"


_ENV_RE = re.compile(r"\$\{([^}]+)\}")
_FROM_RE = re.compile(r"^(?P<prefix>\s*FROM\s+(?:--platform=\S+\s+)?)(?P<image>\S+)(?P<suffix>.*)$")
_PIP_ENV_LINES = [
    "ENV PIP_INDEX_URL=http://mirrors.i.h.pjlab.org.cn/pypi/simple/",
    "ENV PIP_EXTRA_INDEX_URL=http://pypi.i.h.pjlab.org.cn/brain/dev/+simple",
    "ENV PIP_TRUSTED_HOST=\"mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn\"",
    "ENV PIP_DISABLE_PIP_VERSION_CHECK=1",
    "ENV PIP_PROGRESS_BAR=off",
]
_APT_MIRROR_SETUP_LINE = (
    "RUN set -eu; "
    "for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do "
    "[ -e \"$file\" ] || continue; "
    "sed -i "
    "-e 's#http://deb.debian.org/debian-security#http://mirrors.tuna.tsinghua.edu.cn/debian-security#g' "
    "-e 's#https://deb.debian.org/debian-security#http://mirrors.tuna.tsinghua.edu.cn/debian-security#g' "
    "-e 's#http://security.debian.org/debian-security#http://mirrors.tuna.tsinghua.edu.cn/debian-security#g' "
    "-e 's#https://security.debian.org/debian-security#http://mirrors.tuna.tsinghua.edu.cn/debian-security#g' "
    "-e 's#http://deb.debian.org/debian#http://mirrors.tuna.tsinghua.edu.cn/debian#g' "
    "-e 's#https://deb.debian.org/debian#http://mirrors.tuna.tsinghua.edu.cn/debian#g' "
    "\"$file\"; "
    "done; "
    "if [ -d /etc/apt/apt.conf.d ]; then "
    "printf '%s\\n' "
    "'Acquire::http::Proxy \"false\";' "
    "'Acquire::https::Proxy \"false\";' "
    "> /etc/apt/apt.conf.d/99a3s-direct-proxy; "
    "fi"
)


def _resolve_compose_scalar(value: Any, *, environment_dir: Path) -> str:
    text = str(value)

    def replace(match: re.Match[str]) -> str:
        expr = match.group(1)
        if ":-" in expr:
            name, default = expr.split(":-", 1)
            return os.getenv(name, default)
        if expr == "CONTEXT_DIR":
            return str(environment_dir)
        return os.getenv(expr, "")

    return _ENV_RE.sub(replace, text)


def _build_spec_from_compose(
    *,
    task: str,
    service: str,
    build: Any,
    image: str,
    environment_dir: Path,
) -> ImageSpec:
    if isinstance(build, str):
        context_raw = build
        dockerfile_raw = "Dockerfile"
    elif isinstance(build, dict):
        context_raw = build.get("context") or "."
        dockerfile_raw = build.get("dockerfile") or "Dockerfile"
    else:
        raise ValueError(f"Unsupported compose build spec for {task}/{service}: {build!r}")

    context_text = _resolve_compose_scalar(context_raw, environment_dir=environment_dir)
    context = Path(context_text)
    if not context.is_absolute():
        context = environment_dir / context
    dockerfile_text = _resolve_compose_scalar(dockerfile_raw, environment_dir=environment_dir)
    dockerfile = Path(dockerfile_text)
    if not dockerfile.is_absolute():
        dockerfile = context / dockerfile
    return ImageSpec(
        task=task,
        service=service,
        image=image,
        context=str(context.resolve()),
        dockerfile=str(dockerfile.resolve()),
    )


def _image_specs(task_dir: Path, repo: str, prefix: str) -> tuple[list[ImageSpec], str | None]:
    task = task_dir.name
    environment_dir = task_dir / "environment"
    digest = _task_env_hash(environment_dir)
    main_image = _image_for_task(repo, prefix, task, "main", digest)
    compose_path = environment_dir / "docker-compose.yaml"
    specs: list[ImageSpec] = []
    patched_compose: str | None = None

    if compose_path.exists():
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
        services = compose.setdefault("services", {})
        main = services.get("main") or {}
        if "build" in main:
            specs.append(
                _build_spec_from_compose(
                    task=task,
                    service="main",
                    build=main["build"],
                    image=main_image,
                    environment_dir=environment_dir,
                )
            )
        else:
            specs.append(
                ImageSpec(
                    task=task,
                    service="main",
                    image=main_image,
                    context=str(environment_dir.resolve()),
                    dockerfile=str((environment_dir / "Dockerfile").resolve()),
                )
            )
        main.pop("build", None)
        main.pop("image", None)

        for service, config in sorted(services.items()):
            if service == "main" or not isinstance(config, dict) or "build" not in config:
                continue
            image = _image_for_task(repo, prefix, task, service, digest)
            specs.append(
                _build_spec_from_compose(
                    task=task,
                    service=service,
                    build=config["build"],
                    image=image,
                    environment_dir=environment_dir,
                )
            )
            config.pop("build", None)
            config["image"] = image
        patched_compose = yaml.safe_dump(compose, sort_keys=False)
        return specs, patched_compose

    specs.append(
        ImageSpec(
            task=task,
            service="main",
            image=main_image,
            context=str(environment_dir.resolve()),
            dockerfile=str((environment_dir / "Dockerfile").resolve()),
        )
    )
    return specs, None


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _write_prebuilt_task(task_dir: Path, target_dir: Path, main_image: str, patched_compose: str | None) -> None:
    _remove_path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(task_dir.iterdir()):
        target = target_dir / child.name
        if child.name == "task.toml":
            cfg = toml.loads(child.read_text(encoding="utf-8"))
            cfg.setdefault("environment", {})["docker_image"] = main_image
            target.write_text(toml.dumps(cfg), encoding="utf-8")
        elif child.name == "environment" and patched_compose is not None:
            target.mkdir(parents=True, exist_ok=True)
            for env_child in sorted(child.iterdir()):
                env_target = target / env_child.name
                if env_child.name == "docker-compose.yaml":
                    env_target.write_text(patched_compose, encoding="utf-8")
                else:
                    env_target.symlink_to(env_child)
        else:
            target.symlink_to(child)


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], log_dir: Path) -> subprocess.CompletedProcess[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "command.txt").write_text(shell_join(argv) + "\n", encoding="utf-8")
    output_path = log_dir / "output.txt"
    with output_path.open("w", encoding="utf-8", errors="ignore") as output:
        result = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    (log_dir / "return_code.txt").write_text(str(result.returncode), encoding="utf-8")
    with output_path.open("rb") as output:
        output.seek(0, os.SEEK_END)
        size = output.tell()
        output.seek(max(0, size - 8000))
        tail = output.read().decode("utf-8", errors="ignore")[-4000:]
    return subprocess.CompletedProcess(argv, result.returncode, stdout=tail, stderr=None)


def _manifest_exists(image: str, env: dict[str, str]) -> bool:
    result = subprocess.run(
        ["docker", "manifest", "inspect", image],
        env=env,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _proxy_build_args(env: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        value = env.get(key)
        if value:
            args.extend(["--build-arg", f"{key}={value}"])
    return args


def _docker_build_network(env: dict[str, str]) -> list[str]:
    value = env.get("A3S_CODE_DOCKER_BUILD_NETWORK", "").strip()
    if not value:
        return []
    if value.lower() in {"none", "default"}:
        return []
    return ["--network", value]


def _base_image_rewrites(env: dict[str, str]) -> dict[str, str]:
    raw = env.get("A3S_CODE_DOCKER_BASE_REWRITE", "").strip()
    if not raw:
        return {}
    rewrites: dict[str, str] = {}
    for item in re.split(r"[,\n]+", raw):
        item = item.strip()
        if not item or "=" not in item:
            continue
        source, target = item.split("=", 1)
        source = source.strip()
        target = target.strip()
        if source and target:
            rewrites[source] = target
    return rewrites


def _strip_optional_agent_tooling(lines: list[str]) -> tuple[list[str], bool]:
    stripped: list[str] = []
    changed = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.lstrip().upper().startswith("RUN "):
            stripped.append(line)
            index += 1
            continue

        block = [line]
        index += 1
        while block[-1].rstrip().endswith("\\") and index < len(lines):
            block.append(lines[index])
            index += 1
        if "@zed-industries/claude-agent-acp" in "".join(block):
            changed = True
            continue
        stripped.extend(block)
    return stripped, changed


def _rewritten_dockerfile(dockerfile: Path, *, env: dict[str, str], target: Path) -> Path:
    rewrites = _base_image_rewrites(env)
    inject_pip_env = env.get("A3S_CODE_DOCKER_INJECT_PIP_ENV", "1").strip().lower() not in {"0", "false", "no", "off"}
    inject_apt_mirror = env.get("A3S_CODE_DOCKER_INJECT_APT_MIRROR", "1").strip().lower() not in {"0", "false", "no", "off"}
    strip_optional_agent_tooling = (
        env.get("A3S_CODE_DOCKER_STRIP_OPTIONAL_AGENT_TOOLING", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    if not rewrites and not inject_pip_env and not inject_apt_mirror and not strip_optional_agent_tooling:
        return dockerfile

    changed = False
    lines: list[str] = []
    source_lines = dockerfile.read_text(encoding="utf-8").splitlines(keepends=True)
    if strip_optional_agent_tooling:
        source_lines, stripped_tooling = _strip_optional_agent_tooling(source_lines)
        changed = changed or stripped_tooling
    for line in source_lines:
        match = _FROM_RE.match(line.rstrip("\n"))
        if match and match.group("image") in rewrites:
            newline = "\n" if line.endswith("\n") else ""
            line = f"{match.group('prefix')}{rewrites[match.group('image')]}{match.group('suffix')}{newline}"
            changed = True
        lines.append(line)
        if inject_pip_env and match:
            lines.extend(f"{item}\n" for item in _PIP_ENV_LINES)
            changed = True
        if inject_apt_mirror and match:
            lines.append(f"{_APT_MIRROR_SETUP_LINE}\n")
            changed = True
    if not changed:
        return dockerfile

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(lines), encoding="utf-8")
    return target


def _run_with_retries(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    attempts: int,
) -> subprocess.CompletedProcess[str]:
    last: subprocess.CompletedProcess[str] | None = None
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        attempt_log_dir = log_dir if attempt == 1 else log_dir.with_name(f"{log_dir.name}_attempt_{attempt}")
        last = _run(argv, cwd=cwd, env=env, log_dir=attempt_log_dir)
        if last.returncode == 0:
            if attempt_log_dir != log_dir:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "command.txt").write_text(" ".join(shlex.quote(arg) for arg in argv) + "\n", encoding="utf-8")
                (log_dir / "output.txt").write_text(last.stdout or "", encoding="utf-8", errors="ignore")
                (log_dir / "return_code.txt").write_text(str(last.returncode), encoding="utf-8")
            return last
        if attempt < attempts:
            time.sleep(min(60, 10 * attempt))
    assert last is not None
    return last


def _build_and_push(spec: ImageSpec, *, push: bool, skip_existing: bool, logs_dir: Path, env: dict[str, str]) -> BuildRecord:
    started = time.time()
    log_dir = logs_dir / spec.task / spec.service
    if skip_existing and _manifest_exists(spec.image, env):
        return BuildRecord(
            task=spec.task,
            service=spec.service,
            image=spec.image,
            status="skipped_existing",
            elapsed_sec=time.time() - started,
            log_dir=str(log_dir),
        )

    dockerfile = _rewritten_dockerfile(Path(spec.dockerfile), env=env, target=log_dir / "Dockerfile.rewritten")
    build_cmd = [
        "docker",
        "build",
        *_docker_build_network(env),
        *_proxy_build_args(env),
        "-t",
        spec.image,
        "-f",
        str(dockerfile),
        spec.context,
    ]
    if env.get("DOCKER_BUILDKIT") not in {"0", "false", "False"}:
        build_cmd[2:2] = ["--progress=plain"]
    attempts = int(env.get("A3S_CODE_DOCKER_BUILD_ATTEMPTS", "3"))
    build = _run_with_retries(build_cmd, cwd=Path(spec.context), env=env, log_dir=log_dir / "build", attempts=attempts)
    if build.returncode != 0:
        return BuildRecord(
            task=spec.task,
            service=spec.service,
            image=spec.image,
            status="build_failed",
            build_returncode=build.returncode,
            elapsed_sec=time.time() - started,
            log_dir=str(log_dir),
            error=(build.stdout or "")[-4000:],
        )

    push_returncode = None
    if push:
        pushed = _run_with_retries(
            ["docker", "push", spec.image],
            cwd=Path(spec.context),
            env=env,
            log_dir=log_dir / "push",
            attempts=int(env.get("A3S_CODE_DOCKER_PUSH_ATTEMPTS", "3")),
        )
        push_returncode = pushed.returncode
        if pushed.returncode != 0:
            return BuildRecord(
                task=spec.task,
                service=spec.service,
                image=spec.image,
                status="push_failed",
                build_returncode=build.returncode,
                push_returncode=pushed.returncode,
                elapsed_sec=time.time() - started,
                log_dir=str(log_dir),
                error=(pushed.stdout or "")[-4000:],
            )

    return BuildRecord(
        task=spec.task,
        service=spec.service,
        image=spec.image,
        status="pushed" if push else "built",
        build_returncode=build.returncode,
        push_returncode=push_returncode,
        elapsed_sec=time.time() - started,
        log_dir=str(log_dir),
    )


def _task_dirs(tasks_root: Path, task_names: list[str], limit: int) -> list[Path]:
    if task_names:
        selected = [tasks_root / name for name in task_names]
    else:
        selected = sorted(path for path in tasks_root.iterdir() if path.is_dir())
    valid = [
        path.resolve()
        for path in selected
        if (path / "task.toml").exists() and (path / "instruction.md").exists() and (path / "environment").exists()
    ]
    return valid[:limit] if limit > 0 else valid


def _git_short(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return (result.stdout or "unknown").strip() or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prebuild SkillsBench Docker images and create a prebuilt task tree.")
    parser.add_argument("--skillsbench-root", type=Path, default=Path(os.getenv("A3S_CODE_SKILLSBENCH_ROOT", Path.home() / "workspace" / "skillsbench")))
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--output-tasks-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--registry-repo", type=str, default=os.getenv("A3S_CODE_SKILLSBENCH_IMAGE_REPO", DEFAULT_IMAGE_REPO))
    parser.add_argument("--tag-prefix", type=str, default=os.getenv("A3S_CODE_SKILLSBENCH_IMAGE_TAG_PREFIX", ""))
    parser.add_argument("--task-names", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--push", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true", help="Create the prebuilt task tree and manifest without building images.")
    parser.add_argument("--worker-local-docker", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.skillsbench_root = args.skillsbench_root.expanduser().resolve()
    tasks_root = (args.tasks_dir or (args.skillsbench_root / "tasks")).expanduser().resolve()
    args.output_tasks_dir = args.output_tasks_dir.expanduser().resolve()
    logs_dir = (args.logs_dir or (args.output_tasks_dir.parent / "prebuild_logs")).expanduser().resolve()
    manifest_path = (args.manifest_path or (args.output_tasks_dir.parent / "prebuild_manifest.json")).expanduser().resolve()
    tag_prefix = args.tag_prefix.strip() or f"sb-{_git_short(args.skillsbench_root)}"
    task_names = [item.strip() for item in args.task_names.split(",") if item.strip()]
    tasks = _task_dirs(tasks_root, task_names, args.limit)
    if not tasks:
        raise RuntimeError(f"No SkillsBench tasks found under {tasks_root}")

    worker_docker = None
    if args.worker_local_docker:
        worker_docker = start_worker_local_docker(log_dir=logs_dir / "worker_local_docker")

    env = os.environ.copy()
    env.setdefault("DOCKER_BUILDKIT", "0")
    env.setdefault("COMPOSE_DOCKER_CLI_BUILD", "0")
    env.setdefault("COMPOSE_BAKE", "false")
    records: list[BuildRecord] = []
    task_manifest: list[dict[str, Any]] = []
    try:
        for task_dir in tasks:
            specs, patched_compose = _image_specs(task_dir, args.registry_repo, tag_prefix)
            main_image = next(spec.image for spec in specs if spec.service == "main")
            _write_prebuilt_task(task_dir, args.output_tasks_dir / task_dir.name, main_image, patched_compose)
            task_manifest.append(
                {
                    "task": task_dir.name,
                    "source": str(task_dir),
                    "prebuilt_task": str(args.output_tasks_dir / task_dir.name),
                    "main_image": main_image,
                    "images": [asdict(spec) for spec in specs],
                }
            )
            if args.plan_only:
                for spec in specs:
                    records.append(
                        BuildRecord(
                            task=spec.task,
                            service=spec.service,
                            image=spec.image,
                            status="planned",
                            elapsed_sec=0.0,
                            log_dir=str(logs_dir / spec.task / spec.service),
                        )
                    )
                    print(json.dumps(asdict(records[-1]), ensure_ascii=False), flush=True)
            else:
                for spec in specs:
                    record = _build_and_push(
                        spec,
                        push=args.push,
                        skip_existing=args.skip_existing,
                        logs_dir=logs_dir,
                        env=env,
                    )
                    records.append(record)
                    print(json.dumps(asdict(record), ensure_ascii=False), flush=True)
                    if record.status.endswith("failed"):
                        break
                if records and records[-1].status.endswith("failed"):
                    break
    finally:
        if worker_docker is not None:
            worker_docker.stop()

    payload = {
        "skillsbench_root": str(args.skillsbench_root),
        "tasks_root": str(tasks_root),
        "output_tasks_dir": str(args.output_tasks_dir),
        "registry_repo": args.registry_repo,
        "tag_prefix": tag_prefix,
        "push": args.push,
        "skip_existing": args.skip_existing,
        "plan_only": args.plan_only,
        "task_count": len(tasks),
        "tasks": task_manifest,
        "records": [asdict(record) for record in records],
        "counts": {},
    }
    for record in records:
        payload["counts"][record.status] = payload["counts"].get(record.status, 0) + 1
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "counts": payload["counts"]}, ensure_ascii=False), flush=True)
    return 1 if any(record.status.endswith("failed") for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
