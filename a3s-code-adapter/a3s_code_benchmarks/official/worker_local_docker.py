from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOCKET = Path("/tmp/a3s-worker-docker/docker.sock")
DEFAULT_DATA_ROOT = Path("/tmp/a3s-worker-docker/root")
DEFAULT_EXEC_ROOT = Path("/tmp/a3s-worker-docker/exec")
DEFAULT_CLIENT_CONFIG = Path("/tmp/a3s-worker-docker/client-config")
DEFAULT_DOCKER_PROXY = ""
DEFAULT_NO_PROXY = (
    "localhost,127.0.0.1,0.0.0.0,::1,*.local,.pjlab.org.cn,"
    ".i.h.pjlab.org.cn,mirrors.i.h.pjlab.org.cn,pypi.i.h.pjlab.org.cn"
)
DEFAULT_ADDRESS_POOLS = [
    {"base": "100.80.0.0/12", "size": 24},
    {"base": "172.30.0.0/15", "size": 24},
]
DISABLE_PROXY_VALUES = {"", "0", "false", "no", "none", "off", "direct"}
ENABLE_VALUES = {"1", "true", "yes", "on"}


def _docker_host(socket_path: Path) -> str:
    return f"unix://{socket_path}"


def _docker_info(socket_path: Path, timeout_sec: int = 5) -> subprocess.CompletedProcess[str]:
    command = ["docker", "-H", _docker_host(socket_path), "info", "--format", "Name={{.Name}} Root={{.DockerRootDir}}"]
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            -1,
            stdout=exc.stdout or "",
            stderr=f"docker info timed out after {timeout_sec}s",
        )


def _read_pidfile(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _proc_state(pid: int) -> str:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("State:"):
                return line
    except OSError:
        return ""
    return ""


def _proc_looks_like_dockerd(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text(encoding="utf-8", errors="ignore").replace("\x00", " ")
    except OSError:
        cmdline = ""
    try:
        comm = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        comm = ""
    return "dockerd" in cmdline or "dockerd" in comm


def _remove_unreachable_dockerd_pidfile(socket_path: Path, log_path: Path) -> None:
    """Clear a stale dockerd pidfile when the socket is already unusable.

    A registry pull can leave dockerd as a zombie with no Unix socket. The next
    restart then fails because dockerd refuses to start while its old pidfile
    points at that zombie. This cleanup is scoped to the worker-local Docker
    directory and only acts on processes that look like dockerd.
    """

    pidfile = socket_path.parent / "dockerd.pid"
    pid = _read_pidfile(pidfile)
    if pid is None or not _proc_looks_like_dockerd(pid):
        return

    state = _proc_state(pid)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[cleanup] removing unreachable dockerd pidfile pid={pid} state={state}\n")
    if "zombie" not in state.lower() and "\tZ" not in state:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
        for _ in range(20):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.25)
        if Path(f"/proc/{pid}").exists():
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
    try:
        pidfile.unlink()
    except OSError:
        pass
    try:
        socket_path.unlink()
    except OSError:
        pass


def _storage_drivers() -> list[str]:
    configured = os.getenv("A3S_CODE_WORKER_DOCKER_STORAGE_DRIVER", "").strip()
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return ["overlay2", "vfs"]


def _network_modes() -> list[str]:
    configured = os.getenv("A3S_CODE_WORKER_DOCKER_NETWORK_MODE", "").strip()
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return ["nat", "isolated"]


def _address_pools() -> list[dict[str, object]]:
    configured = os.getenv("A3S_CODE_WORKER_DOCKER_ADDRESS_POOLS", "").strip()
    if not configured:
        return DEFAULT_ADDRESS_POOLS
    try:
        loaded = json.loads(configured)
    except json.JSONDecodeError as exc:
        raise ValueError("A3S_CODE_WORKER_DOCKER_ADDRESS_POOLS must be JSON") from exc
    if not isinstance(loaded, list) or not all(isinstance(item, dict) for item in loaded):
        raise ValueError("A3S_CODE_WORKER_DOCKER_ADDRESS_POOLS must be a JSON list of objects")
    return loaded


def _configured_proxy() -> str | None:
    explicit = os.getenv("A3S_CODE_DOCKERD_PROXY")
    if explicit is not None:
        value = explicit.strip()
        if value.lower() in DISABLE_PROXY_VALUES:
            return None
        return value
    for key in ("A3S_CODE_BENCHMARK_PROXY", "BENCHMARK_HTTP_PROXY"):
        value = os.getenv(key, "").strip()
        if value:
            return None if value.lower() in DISABLE_PROXY_VALUES else value
    default = DEFAULT_DOCKER_PROXY.strip()
    if not default or default.lower() in DISABLE_PROXY_VALUES:
        return None
    return default


def _proxy_env() -> dict[str, str]:
    proxy = _configured_proxy()
    no_proxy_values = [DEFAULT_NO_PROXY]
    for key in ("A3S_CODE_NO_PROXY", "NO_PROXY", "no_proxy"):
        value = os.getenv(key, "").strip()
        if value:
            no_proxy_values.append(value)
    no_proxy_entries = []
    for raw_value in no_proxy_values:
        for entry in raw_value.split(","):
            entry = entry.strip()
            if entry and entry != "*":
                no_proxy_entries.append(entry)
    no_proxy = ",".join(dict.fromkeys(no_proxy_entries))
    env = {
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
    }
    if proxy:
        env.update(
            {
                "HTTP_PROXY": proxy,
                "http_proxy": proxy,
                "HTTPS_PROXY": proxy,
                "https_proxy": proxy,
                "ALL_PROXY": proxy,
                "all_proxy": proxy,
            }
        )
    else:
        env.update(
            {
                "HTTP_PROXY": "",
                "http_proxy": "",
                "HTTPS_PROXY": "",
                "https_proxy": "",
                "ALL_PROXY": "",
                "all_proxy": "",
            }
        )
    return env


def _prepare_client_config(proxy_env: dict[str, str]) -> Path:
    """Create a worker-local Docker client config with auth plus build proxies."""

    target = Path(os.getenv("A3S_CODE_WORKER_DOCKER_CONFIG", str(DEFAULT_CLIENT_CONFIG)))
    source_raw = os.getenv("DOCKER_CONFIG", "").strip()
    source = Path(source_raw).expanduser() if source_raw else Path.home() / ".docker"
    target.mkdir(parents=True, exist_ok=True)

    try:
        if source.exists() and source.resolve() != target.resolve():
            shutil.copytree(source, target, dirs_exist_ok=True)
    except OSError:
        # Registry auth is useful but not required for public base-image pulls.
        # Continue with a fresh config if the inherited config is unreadable.
        pass

    config_path = target / "config.json"
    config: dict[str, object] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config = loaded
        except (OSError, json.JSONDecodeError):
            config = {}

    proxies = config.setdefault("proxies", {})
    if not isinstance(proxies, dict):
        proxies = {}
        config["proxies"] = proxies
    client_proxy_enabled = os.getenv("A3S_CODE_DOCKER_CLIENT_PROXY", "").strip().lower() in ENABLE_VALUES
    if proxy_env.get("HTTP_PROXY") and client_proxy_enabled:
        proxies["default"] = {
            "httpProxy": proxy_env["HTTP_PROXY"],
            "httpsProxy": proxy_env["HTTPS_PROXY"],
            "noProxy": proxy_env["NO_PROXY"],
        }
    else:
        proxies.pop("default", None)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


@dataclass
class WorkerLocalDocker:
    socket_path: Path
    data_root: Path
    exec_root: Path
    log_path: Path
    process: subprocess.Popen[str] | None = None

    @property
    def docker_host(self) -> str:
        return _docker_host(self.socket_path)

    def activate_env(self) -> None:
        proxy_env = _proxy_env()
        client_config = _prepare_client_config(proxy_env)
        os.environ["DOCKER_HOST"] = self.docker_host
        os.environ["DOCKER_CONFIG"] = str(client_config)
        os.environ["A3S_CODE_WORKER_LOCAL_DOCKER_ACTIVE"] = "1"
        # BuildKit/Bake can bypass the daemon proxy for registry auth requests
        # in this cluster. The legacy builder follows the worker-local dockerd
        # proxy path that we configure below.
        os.environ.setdefault("DOCKER_BUILDKIT", "0")
        os.environ.setdefault("COMPOSE_DOCKER_CLI_BUILD", "0")
        os.environ.setdefault("COMPOSE_BAKE", "false")
        os.environ.update(proxy_env)

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)


def start_worker_local_docker(
    *,
    log_dir: Path,
    socket_path: Path | None = None,
    data_root: Path | None = None,
    exec_root: Path | None = None,
    timeout_sec: int = 60,
) -> WorkerLocalDocker:
    """Start a Docker daemon inside the current privileged worker.

    PJLab rjob wrappers may inject a remote DOCKER_HOST that points back to the
    dev machine. This helper intentionally overrides that with a Unix socket
    owned by a dockerd process running inside the current worker pod.
    """

    socket_path = Path(os.getenv("A3S_CODE_WORKER_DOCKER_SOCKET", str(socket_path or DEFAULT_SOCKET)))
    data_root = Path(os.getenv("A3S_CODE_WORKER_DOCKER_DATA_ROOT", str(data_root or DEFAULT_DATA_ROOT)))
    exec_root = Path(os.getenv("A3S_CODE_WORKER_DOCKER_EXEC_ROOT", str(exec_root or DEFAULT_EXEC_ROOT)))
    timeout_sec = int(os.getenv("A3S_CODE_WORKER_DOCKER_START_TIMEOUT_SEC", str(timeout_sec)))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dockerd.log"

    existing = _docker_info(socket_path)
    if existing.returncode == 0:
        manager = WorkerLocalDocker(socket_path=socket_path, data_root=data_root, exec_root=exec_root, log_path=log_path)
        manager.activate_env()
        return manager

    _remove_unreachable_dockerd_pidfile(socket_path, log_path)

    env = os.environ.copy()
    env.pop("DOCKER_HOST", None)
    env.update(_proxy_env())

    last_error = ""
    for storage_driver in _storage_drivers():
        for network_mode in _network_modes():
            driver_data_root = data_root / storage_driver / network_mode
            driver_exec_root = exec_root / storage_driver / network_mode
            daemon_config_path = socket_path.parent / "daemon.json"
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            driver_data_root.mkdir(parents=True, exist_ok=True)
            driver_exec_root.mkdir(parents=True, exist_ok=True)
            if socket_path.exists():
                socket_path.unlink()
            proxy_env = _proxy_env()
            daemon_config = {}
            if proxy_env.get("HTTP_PROXY"):
                daemon_config["proxies"] = {
                    "http-proxy": proxy_env["HTTP_PROXY"],
                    "https-proxy": proxy_env["HTTPS_PROXY"],
                    "no-proxy": proxy_env["NO_PROXY"],
                }
            if network_mode == "nat":
                daemon_config["default-address-pools"] = _address_pools()
            daemon_config_path.write_text(
                json.dumps(daemon_config, indent=2),
                encoding="utf-8",
            )

            command = [
                "dockerd",
                f"--config-file={daemon_config_path}",
                f"--host=unix://{socket_path}",
                f"--data-root={driver_data_root}",
                f"--exec-root={driver_exec_root}",
                f"--pidfile={socket_path.parent / 'dockerd.pid'}",
                f"--storage-driver={storage_driver}",
                "--userland-proxy=false",
            ]
            if network_mode == "isolated":
                command.extend(["--iptables=false", "--ip-masq=false"])
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n[start] {' '.join(command)}\n")
                log_handle.flush()
                process = subprocess.Popen(
                    command,
                    text=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                if process.poll() is not None:
                    break
                probe = _docker_info(socket_path)
                if probe.returncode == 0:
                    manager = WorkerLocalDocker(
                        socket_path=socket_path,
                        data_root=driver_data_root,
                        exec_root=driver_exec_root,
                        log_path=log_path,
                        process=process,
                    )
                    manager.activate_env()
                    atexit.register(manager.stop)
                    return manager
                last_error = (probe.stderr or probe.stdout or "").strip()
                time.sleep(1)

            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    log_tail = ""
    try:
        log_tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
    except OSError:
        pass
    raise RuntimeError(
        "Failed to start worker-local dockerd. This path requires a privileged rjob "
        "container and writable cgroups. "
        f"Last docker error: {last_error}\nDockerd log tail:\n{log_tail}"
    )
