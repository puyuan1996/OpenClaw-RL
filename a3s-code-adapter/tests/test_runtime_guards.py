from __future__ import annotations

import asyncio
import collections
import importlib
import importlib.util
import queue
import sys
import threading
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_UTILS_PATH = ROOT / "a3s_code_benchmarks" / "benchmark_runtime_utils.py"


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens: bool = False):
        if isinstance(text, str):
            return {"input_ids": list(range(len(text.split())))}
        return {"input_ids": [0]}

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        rendered = "\n".join(
            f"{msg.get('role', 'user')}:{msg.get('content', '')}" for msg in messages
        )
        if tokenize:
            return list(range(len(rendered.split())))
        return rendered

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "decoded"


def _install_slime_stubs(monkeypatch) -> None:
    slime_pkg = types.ModuleType("slime")
    utils_pkg = types.ModuleType("slime.utils")
    rollout_pkg = types.ModuleType("slime.rollout")
    metric_utils = types.ModuleType("slime.utils.metric_utils")
    processing_utils = types.ModuleType("slime.utils.processing_utils")
    types_mod = types.ModuleType("slime.utils.types")
    async_utils = types.ModuleType("slime.utils.async_utils")
    base_types = types.ModuleType("slime.rollout.base_types")
    sglang_rollout = types.ModuleType("slime.rollout.sglang_rollout")

    class FakeSample:
        class Status:
            COMPLETED = "completed"
            ABORTED = "aborted"

        def __init__(self):
            self.reward = {}

    class FakeRolloutFnTrainOutput:
        def __init__(self, samples, metrics=None):
            self.samples = samples
            self.metrics = metrics

    async def fake_eval_rollout(*args, **kwargs):
        return [], None

    metric_utils.has_repetition = lambda text: False
    processing_utils.load_tokenizer = lambda *args, **kwargs: _FakeTokenizer()
    types_mod.Sample = FakeSample
    async_utils.run = lambda awaitable: asyncio.run(awaitable)
    base_types.RolloutFnTrainOutput = FakeRolloutFnTrainOutput
    sglang_rollout.eval_rollout = fake_eval_rollout

    monkeypatch.setitem(sys.modules, "slime", slime_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "slime.rollout", rollout_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils.metric_utils", metric_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", processing_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.types", types_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.async_utils", async_utils)
    monkeypatch.setitem(sys.modules, "slime.rollout.base_types", base_types)
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", sglang_rollout)


def _install_service_stubs(monkeypatch) -> None:
    fastapi = types.ModuleType("fastapi")
    fastapi_responses = types.ModuleType("fastapi.responses")
    httpx = types.ModuleType("httpx")
    uvicorn = types.ModuleType("uvicorn")

    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        def on_event(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def get(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def post(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    class FakeHTTPException(Exception):
        def __init__(self, status_code: int, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeResponse:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeServer:
        def __init__(self, config=None):
            self.config = config
            self.should_exit = False

        def run(self):
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fastapi.FastAPI = FakeFastAPI
    fastapi.Header = lambda default=None: default
    fastapi.HTTPException = FakeHTTPException
    fastapi.Request = object
    fastapi_responses.JSONResponse = FakeResponse
    fastapi_responses.StreamingResponse = FakeResponse
    httpx.AsyncClient = FakeAsyncClient
    httpx.Client = FakeClient
    httpx.get = lambda *args, **kwargs: FakeResponse(*args, **kwargs)
    uvicorn.Config = FakeConfig
    uvicorn.Server = FakeServer

    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", fastapi_responses)
    monkeypatch.setitem(sys.modules, "httpx", httpx)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)


def _install_sglang_stubs(monkeypatch) -> None:
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    function_call = types.ModuleType("sglang.srt.function_call")
    parser_mod = types.ModuleType("sglang.srt.function_call.function_call_parser")
    managers = types.ModuleType("sglang.srt.managers")
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class FakeFunctionCallParser:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def parse_non_stream(self, response_text):
            return response_text, []

    class FakeFunction:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeTool:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    parser_mod.FunctionCallParser = FakeFunctionCallParser
    io_struct.Function = FakeFunction
    io_struct.Tool = FakeTool

    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.function_call", function_call)
    monkeypatch.setitem(sys.modules, "sglang.srt.function_call.function_call_parser", parser_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers", managers)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)


def _install_a3s_code_stub(monkeypatch) -> None:
    a3s_code = types.ModuleType("a3s_code")
    a3s_code.Agent = type("Agent", (), {})
    a3s_code.PermissionPolicy = type("PermissionPolicy", (), {})
    a3s_code.SessionOptions = type("SessionOptions", (), {})
    monkeypatch.setitem(sys.modules, "a3s_code", a3s_code)


def _load_module(monkeypatch, module_name: str):
    monkeypatch.syspath_prepend(str(ROOT))
    _install_slime_stubs(monkeypatch)
    _install_service_stubs(monkeypatch)
    _install_sglang_stubs(monkeypatch)
    if module_name == "a3s_code_agent_traffic_driver":
        _install_a3s_code_stub(monkeypatch)

    sys.modules.pop(module_name, None)
    if module_name == "code_rl_rollout":
        sys.modules.pop("code_rl_api_server", None)
    return importlib.import_module(module_name)


def _load_runtime_utils(monkeypatch, a3s_code_root: Path):
    monkeypatch.setenv("A3S_CODE_REPO_ROOT", str(a3s_code_root))
    module_name = f"benchmark_runtime_utils_test_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, RUNTIME_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_find_existing_wheel_requires_current_a3s_code_version(monkeypatch, tmp_path: Path) -> None:
    a3s_root = tmp_path / "Code"
    sdk_dir = a3s_root / "sdk" / "python"
    sdk_dir.mkdir(parents=True)
    (sdk_dir / "pyproject.toml").write_text(
        '[project]\nname = "a3s-code"\nversion = "2.3.0"\n',
        encoding="utf-8",
    )
    module = _load_runtime_utils(monkeypatch, a3s_root)
    tag = module.target_python_tag()
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    stale = wheel_dir / f"a3s_code-1.9.0-{tag}-{tag}-manylinux_2_34_x86_64.whl"
    fresh = wheel_dir / f"a3s_code-2.3.0-{tag}-{tag}-manylinux_2_34_x86_64.whl"
    stale.write_text("old", encoding="utf-8")
    fresh.write_text("new", encoding="utf-8")

    assert module.find_existing_wheel(wheel_dir) == fresh


def test_find_existing_wheel_rejects_stale_only_wheel(monkeypatch, tmp_path: Path) -> None:
    a3s_root = tmp_path / "Code"
    sdk_dir = a3s_root / "sdk" / "python"
    sdk_dir.mkdir(parents=True)
    (sdk_dir / "pyproject.toml").write_text(
        '[project]\nname = "a3s-code"\nversion = "2.3.0"\n',
        encoding="utf-8",
    )
    module = _load_runtime_utils(monkeypatch, a3s_root)
    tag = module.target_python_tag()
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    (wheel_dir / f"a3s_code-1.9.0-{tag}-{tag}-manylinux_2_34_x86_64.whl").write_text(
        "old",
        encoding="utf-8",
    )

    assert module.find_existing_wheel(wheel_dir) is None


def test_pause_submission_discards_queued_groups(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_rollout")
    worker = object.__new__(module.AsyncRolloutWorker)
    worker._submission_enabled = threading.Event()
    worker._submission_enabled.set()
    worker.output_queue = queue.Queue()
    worker.output_queue.put((1, ["a", "b"]))
    worker.output_queue.put((2, ["c"]))
    worker.partial_groups = {}

    class FakeServer:
        def __init__(self):
            self.purge_calls = 0
            self.drain_calls = 0
            self.epoch_calls = 0

        def advance_submission_epoch(self):
            self.epoch_calls += 1
            return self.epoch_calls

        def wait_for_inflight_generation_requests(self):
            self.drain_calls += 1
            return True

        def purge_record_files(self):
            self.purge_calls += 1

    worker._server = FakeServer()

    module.AsyncRolloutWorker.pause_submission(worker)

    assert not worker._submission_enabled.is_set()
    assert worker.output_queue.empty()
    assert worker._server.epoch_calls == 1
    assert worker._server.drain_calls == 1
    assert worker._server.purge_calls == 1


def test_drain_output_queue_times_out_without_progress(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_rollout")
    monkeypatch.setenv("CODE_RL_ROLLOUT_IDLE_TIMEOUT_SEC", "0.01")
    monkeypatch.setenv("CODE_RL_ROLLOUT_DRAIN_TIMEOUT_SEC", "0")

    class FakeWorker:
        partial_groups = {}

        @staticmethod
        def get_completed_groups():
            return []

        @staticmethod
        def get_queue_size():
            return 0

    args = types.SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=1)

    try:
        asyncio.run(module._drain_output_queue(args, FakeWorker()))
    except TimeoutError as exc:
        assert "no accepted group progress" in str(exc)
    else:
        raise AssertionError("expected rollout drain idle timeout")


def test_short_repetition_guard_flags_degenerate_outputs(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")

    degenerate = "\n".join(["In"] + ["The"] * 40)
    normal = (
        "I will inspect the input files, implement the requested script, "
        "run the verifier, and then write the required outputs under output/."
    )

    assert module._has_short_repetition(degenerate)
    assert not module._has_short_repetition(normal)


def test_default_model_tool_filter_drops_delegation_tools(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    monkeypatch.delenv("CODE_RL_DROP_MODEL_TOOLS", raising=False)
    server = object.__new__(module.CodeRLAPIServer)
    server.drop_model_tools = module._csv_env_set(
        "CODE_RL_DROP_MODEL_TOOLS",
        module._DEFAULT_DROPPED_MODEL_TOOLS,
    )
    server._stats = module.collections.Counter()
    server._stats_lock = threading.Lock()

    tools = [
        {"type": "function", "function": {"name": "read"}},
        {"type": "function", "function": {"name": "task"}},
        {"type": "function", "function": {"name": "parallel_task"}},
    ]

    filtered = module.CodeRLAPIServer._filter_model_tools(server, tools)

    assert [tool["function"]["name"] for tool in filtered] == ["read"]
    assert server._stats["dropped_model_tool_task"] == 1
    assert server._stats["dropped_model_tool_parallel_task"] == 1


def test_snapshot_stats_reports_rollout_proxy_state(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    server.submission_enabled = threading.Event()
    server.submission_enabled.set()
    server._inflight_generation_condition = threading.Condition()
    server._inflight_generation_requests = 3
    server._submission_epoch = 2
    server.output_queue = queue.Queue()
    server.output_queue.put((0, ["sample"]))
    server._stats = module.collections.Counter({"submitted_samples_total": 2})
    server._stats_lock = threading.Lock()
    server._eval_scores = [1.0, -1.0]
    server._eval_scores_lock = threading.Lock()
    server._pending_turn_data = {"s1": {1: {}, 2: {}}}
    server._pending_records = {"s1": {}}
    server._finalizing_sessions = {"s1"}
    server._prm_tasks = {"s1": {1: object()}}

    class FakeTask:
        def done(self) -> bool:
            return False

    server._submit_tasks = {"s1": {FakeTask()}}
    server._reward_mode = "verifier"
    server._prm_backend = "disabled"
    server._require_verifier_feedback = True

    snapshot = module.CodeRLAPIServer.snapshot_stats(server)

    assert snapshot["submission_enabled"] is True
    assert snapshot["submission_epoch"] == 2
    assert snapshot["inflight_generation_requests"] == 3
    assert snapshot["queue_size"] == 1
    assert snapshot["counters"]["submitted_samples_total"] == 2
    assert snapshot["pending"]["turns"] == 2
    assert snapshot["pending"]["live_submit_tasks"] == 1
    assert snapshot["reward"]["eval_scores_count"] == 2
    assert snapshot["reward"]["eval_score_mean"] == 0.0
    assert snapshot["reward"]["require_verifier_feedback"] is True


def test_feedback_record_infers_group_and_replica_from_session_id(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    server._turn_counts = {"a3s-code-run-grp000027-rep03-abcd": 1}
    server._pending_turn_data = {}
    server._turn_feedback = {}
    server._stats = module.collections.Counter()
    server._stats_lock = threading.Lock()
    server._feedback_record_file = ""
    records = []
    server._append_jsonl = lambda _path, payload: records.append(payload)
    server._append_trace_event = lambda *_args, **_kwargs: None

    server._record_feedback(
        "a3s-code-run-grp000027-rep03-abcd",
        1,
        {
            "event_type": "task_verifier_reward",
            "details": {"score": 0.75},
        },
    )

    details = records[0]["details"]
    assert details["score"] == 0.75
    assert details["sample_group_index"] == 27
    assert details["sample_replica_index"] == 3


def test_pause_wait_blocks_until_submission_resumes(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    server.submission_enabled = threading.Event()
    server.pause_wait_timeout_sec = 1.0

    async def scenario():
        async def resume_soon():
            await asyncio.sleep(0.01)
            server.submission_enabled.set()

        resume_task = asyncio.create_task(resume_soon())
        await server._wait_for_submission_enabled(session_id="s1", turn_type="main")
        await resume_task

    asyncio.run(scenario())


def test_pause_wait_times_out_as_retryable_503(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    server.submission_enabled = threading.Event()
    server.pause_wait_timeout_sec = 0.01

    try:
        asyncio.run(server._wait_for_submission_enabled(session_id="s1", turn_type="main"))
    except module.HTTPException as exc:
        assert exc.status_code == 503
        assert "retry after resume" in exc.detail
    else:
        raise AssertionError("expected pause wait timeout")


def test_submit_turn_sample_drops_ready_sample_when_paused(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    server.submission_enabled = threading.Event()
    server._inflight_generation_condition = threading.Condition()
    server._submission_epoch = 0
    server.drop_repetitive_samples = False
    server._eval_scores = []
    server._eval_scores_lock = threading.Lock()
    server._session_effective = {}
    server.max_response_tokens = 0
    server.response_trim_margin_tokens = 0
    server.max_train_tokens = 0
    server.tokenizer = _FakeTokenizer()
    server.output_queue = queue.Queue()
    server._index_counter = iter([0])
    server._group_counter = iter([0])
    decisions: list[str] = []
    server._append_sample_trace = lambda **kwargs: decisions.append(kwargs["decision"])
    server._maybe_cleanup_session = lambda session_id: None
    server._consume_turn_feedback = lambda session_id, turn_num: []
    server._aggregate_feedback = lambda feedback_events: {
        "sanitized": False,
        "redaction_count": 0,
        "permission_denied": False,
        "injection_blocked": False,
        "event_types": [],
    }
    server._resolve_reward = lambda turn_data, prm_result, feedback_events: {
        "score": 1.0,
        "source": "next_state_prm",
        "details": {},
    }
    server._build_metadata = lambda **kwargs: {"train_metadata": {}}

    turn_data = {
        "turn_num": 1,
        "turn_type": "main",
        "prompt_ids": [1, 2],
        "response_ids": [3, 4],
        "response_logprobs": [-0.1, -0.2],
        "prompt_text": "prompt",
        "response_text": "response",
        "tool_calls": [],
        "channel": "api",
        "session_done": False,
        "has_next_state": True,
        "next_state_role": "user",
        "response_has_repetition": False,
    }

    asyncio.run(server._submit_turn_sample(turn_data=turn_data, session_id="s1", prm_result=None))

    assert server.output_queue.empty()
    assert decisions == ["dropped_paused"]
    assert server._eval_scores == []


def test_submit_turn_sample_drops_stale_epoch_after_resume(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    server.submission_enabled = threading.Event()
    server.submission_enabled.set()
    server._inflight_generation_condition = threading.Condition()
    server._submission_epoch = 1
    server.drop_repetitive_samples = False
    server._eval_scores = []
    server._eval_scores_lock = threading.Lock()
    server._session_effective = {}
    server.max_response_tokens = 0
    server.response_trim_margin_tokens = 0
    server.max_train_tokens = 0
    server.tokenizer = _FakeTokenizer()
    server.output_queue = queue.Queue()
    server._index_counter = iter([0])
    server._group_counter = iter([0])
    decisions: list[str] = []
    server._append_sample_trace = lambda **kwargs: decisions.append(kwargs["decision"])
    server._maybe_cleanup_session = lambda session_id: None
    server._consume_turn_feedback = lambda session_id, turn_num: []
    server._aggregate_feedback = lambda feedback_events: {
        "sanitized": False,
        "redaction_count": 0,
        "permission_denied": False,
        "injection_blocked": False,
        "event_types": [],
    }
    server._resolve_reward = lambda turn_data, prm_result, feedback_events: {
        "score": 1.0,
        "source": "next_state_prm",
        "details": {},
    }
    server._build_metadata = lambda **kwargs: {"train_metadata": {}}

    turn_data = {
        "turn_num": 1,
        "turn_type": "main",
        "prompt_ids": [1, 2],
        "response_ids": [3, 4],
        "response_logprobs": [-0.1, -0.2],
        "prompt_text": "prompt",
        "response_text": "response",
        "tool_calls": [],
        "channel": "api",
        "session_done": False,
        "has_next_state": True,
        "next_state_role": "user",
        "response_has_repetition": False,
        "submission_epoch": 0,
    }

    asyncio.run(server._submit_turn_sample(turn_data=turn_data, session_id="s1", prm_result=None))

    assert server.output_queue.empty()
    assert decisions == ["dropped_stale_epoch"]
    assert server._eval_scores == []


def test_verifier_required_samples_wait_for_feedback(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    server._require_verifier_feedback = True
    server._prm_enabled = False
    server._prm_tasks = {}
    server._pending_turn_data = {
        "s1": {
            1: {
                "turn_num": 1,
                "turn_type": "main",
                "response_text": "",
                "tool_calls": [{"function": {"name": "read"}}],
            }
        }
    }
    server._turn_feedback = {"s1": {}}
    server._submit_tasks = {}
    server._maybe_cleanup_session = lambda session_id: None
    created = []

    class FakeTask:
        def add_done_callback(self, callback):
            return None

        def done(self):
            return False

    def fake_create_task(coro):
        created.append(coro)
        coro.close()
        return FakeTask()

    server._safe_create_task = fake_create_task

    module.CodeRLAPIServer._maybe_submit_ready_samples(server, "s1")
    assert created == []
    assert 1 in server._pending_turn_data["s1"]

    server._turn_feedback["s1"][1] = [
        {
            "event_type": "task_verifier_reward",
            "details": {"score": 0.5},
        }
    ]
    module.CodeRLAPIServer._maybe_submit_ready_samples(server, "s1", force_no_prm=True)

    assert len(created) == 1
    assert server._pending_turn_data["s1"] == {}


def test_tool_call_only_turn_is_not_marked_repetitive(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    monkeypatch.setattr(module, "has_repetition", lambda text: True)
    server = object.__new__(module.CodeRLAPIServer)
    server.submission_enabled = threading.Event()
    server.submission_enabled.set()
    server._inflight_generation_condition = threading.Condition()
    server._submission_epoch = 0
    server.drop_repetitive_samples = True
    server._eval_scores = []
    server._eval_scores_lock = threading.Lock()
    server._session_effective = {}
    server.max_response_tokens = 0
    server.response_trim_margin_tokens = 0
    server.max_train_tokens = 0
    server.tokenizer = _FakeTokenizer()
    server.output_queue = queue.Queue()
    server._index_counter = iter([0])
    server._group_counter = iter([0])
    decisions: list[str] = []
    server._append_sample_trace = lambda **kwargs: decisions.append(kwargs["decision"])
    server._maybe_cleanup_session = lambda session_id: None
    server._consume_turn_feedback = lambda session_id, turn_num: [
        {"event_type": "task_verifier_reward", "details": {"score": 1.0}}
    ]
    server._aggregate_feedback = lambda feedback_events: {
        "sanitized": False,
        "redaction_count": 0,
        "permission_denied": False,
        "injection_blocked": False,
        "event_types": ["task_verifier_reward"],
    }
    server._resolve_reward = lambda turn_data, prm_result, feedback_events: {
        "score": 1.0,
        "source": "verifier",
        "details": {},
    }
    server._build_metadata = lambda **kwargs: {"train_metadata": {}}

    turn_data = {
        "turn_num": 1,
        "turn_type": "main",
        "prompt_ids": [1, 2],
        "response_ids": [3, 4],
        "response_logprobs": [-0.1, -0.2],
        "prompt_text": "prompt",
        "response_text": "",
        "tool_calls": [{"function": {"name": "write"}}],
        "channel": "api",
        "session_done": False,
        "has_next_state": True,
        "next_state_role": "tool",
        "response_has_repetition": False,
    }

    asyncio.run(server._submit_turn_sample(turn_data=turn_data, session_id="s1", prm_result=None))

    assert decisions == ["submitted"]
    assert not server.output_queue.empty()


def test_verifier_labeled_repetitive_failure_is_trainable(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    monkeypatch.setattr(module, "has_repetition", lambda text: True)
    server = object.__new__(module.CodeRLAPIServer)
    server.submission_enabled = threading.Event()
    server.submission_enabled.set()
    server._inflight_generation_condition = threading.Condition()
    server._submission_epoch = 0
    server.drop_repetitive_samples = True
    server._eval_scores = []
    server._eval_scores_lock = threading.Lock()
    server._session_effective = {}
    server.max_response_tokens = 0
    server.response_trim_margin_tokens = 0
    server.max_train_tokens = 0
    server.tokenizer = _FakeTokenizer()
    server.output_queue = queue.Queue()
    server._index_counter = iter([0])
    server._group_counter = iter([0])
    decisions: list[str] = []
    metadata_store: dict = {}
    server._append_sample_trace = lambda **kwargs: decisions.append(kwargs["decision"])
    server._maybe_cleanup_session = lambda session_id: None
    server._consume_turn_feedback = lambda session_id, turn_num: [
        {"event_type": "task_verifier_reward", "details": {"score": 0.0}}
    ]
    server._aggregate_feedback = lambda feedback_events: {
        "sanitized": False,
        "redaction_count": 0,
        "permission_denied": False,
        "injection_blocked": False,
        "event_types": ["task_verifier_reward"],
    }
    server._resolve_reward = lambda turn_data, prm_result, feedback_events: {
        "score": 0.0,
        "source": "verifier",
        "details": {},
    }

    def fake_build_metadata(**kwargs):
        metadata_store.update({"train_metadata": {}})
        return metadata_store

    server._build_metadata = fake_build_metadata

    turn_data = {
        "turn_num": 1,
        "turn_type": "main",
        "prompt_ids": [1, 2],
        "response_ids": [3, 4],
        "response_logprobs": [-0.1, -0.2],
        "prompt_text": "prompt",
        "response_text": "retry retry retry retry retry retry retry retry",
        "tool_calls": [],
        "channel": "api",
        "session_done": True,
        "has_next_state": True,
        "next_state_role": "tool",
        "response_has_repetition": False,
    }

    asyncio.run(server._submit_turn_sample(turn_data=turn_data, session_id="s1", prm_result=None))

    assert decisions == ["submitted"]
    assert not server.output_queue.empty()
    _, samples = server.output_queue.get_nowait()
    assert samples[0].reward == {"score": 0.0}
    assert metadata_store["train_metadata"]["repetition_kept_by_verifier"] is True


def test_finalize_drops_pending_turn_without_required_verifier_feedback(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    server._require_verifier_feedback = True
    server._prm_enabled = False
    server._pending_records = {}
    server._pending_turn_data = {
        "s1": {
            1: {
                "turn_num": 1,
                "turn_type": "main",
                "prompt_ids": [1],
                "response_ids": [2],
                "response_text": "unfinished",
                "tool_calls": [],
                "session_done": False,
            }
        }
    }
    server._turn_feedback = {"s1": {}}
    server._prm_tasks = {}
    server._submit_tasks = {}
    server._finalizing_sessions = set()
    server._overflow_terminated_sessions = set()
    server._session_latest_messages = {"s1": []}
    server._session_last_activity = {"s1": 1.0}
    server._session_effective = {"s1": 0}
    server._turn_counts = {"s1": 1}
    server._stats = collections.Counter()
    server._stats_lock = threading.Lock()
    server._trace_record_file = None
    server._trace_counter = iter([1])

    module.CodeRLAPIServer._finalize_session(server, "s1", reason="idle_timeout")

    assert "s1" not in server._pending_turn_data
    assert "s1" not in server._finalizing_sessions
    assert server._stats["dropped_missing_verifier_feedback"] == 1
    assert server._stats["dropped_samples_total"] == 1


def test_cleanup_waits_for_submit_tasks(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    session_id = "s1"

    class FakeTask:
        def __init__(self, done: bool):
            self._done = done

        def done(self) -> bool:
            return self._done

    server._finalizing_sessions = {session_id}
    server._pending_records = {}
    server._pending_turn_data = {}
    server._prm_tasks = {}
    server._submit_tasks = {session_id: {FakeTask(False)}}
    server._overflow_terminated_sessions = {session_id}
    server._session_latest_messages = {session_id: []}
    server._session_last_activity = {session_id: time.time()}
    server._turn_feedback = {session_id: {}}
    server._session_effective = {session_id: 1}
    server._turn_counts = {session_id: 2}

    module.CodeRLAPIServer._maybe_cleanup_session(server, session_id)
    assert session_id in server._finalizing_sessions

    server._submit_tasks = {session_id: {FakeTask(True)}}
    module.CodeRLAPIServer._maybe_cleanup_session(server, session_id)
    assert session_id not in server._finalizing_sessions
    assert session_id not in server._session_latest_messages


def test_prepare_workspace_template_cache_serializes_creation(monkeypatch, tmp_path: Path) -> None:
    module = _load_module(monkeypatch, "a3s_code_agent_traffic_driver")
    template_root = tmp_path / "task_templates"
    cache_root = tmp_path / "workspace_template_cache"
    source = template_root / "demo_template"
    source.mkdir(parents=True)
    (source / "README.md").write_text("demo\n", encoding="utf-8")

    monkeypatch.setattr(module, "TEMPLATE_ROOT", template_root)
    monkeypatch.setattr(module, "WORKSPACE_TEMPLATE_CACHE_ROOT", cache_root)
    monkeypatch.setattr(module, "WORKSPACE_COPY_MODE", "copy")
    monkeypatch.setattr(module, "TEMPLATE_CACHE_LOCK", threading.Lock())

    copy_calls = 0
    init_calls = 0
    counter_lock = threading.Lock()

    def fake_copy_dir(src: Path, dst: Path, *, mode: str, strip_template_meta: bool = False) -> None:
        nonlocal copy_calls
        with counter_lock:
            copy_calls += 1
        time.sleep(0.05)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "README.md").write_text((src / "README.md").read_text(encoding="utf-8"), encoding="utf-8")

    def fake_init_git_repo(workspace: Path) -> None:
        nonlocal init_calls
        with counter_lock:
            init_calls += 1

    monkeypatch.setattr(module, "_copy_dir", fake_copy_dir)
    monkeypatch.setattr(module, "_init_git_repo", fake_init_git_repo)

    start = threading.Event()
    results: list[Path] = []
    errors: list[Exception] = []

    def target() -> None:
        try:
            start.wait()
            results.append(module._prepare_workspace_template_cache("demo_template"))
        except Exception as exc:  # pragma: no cover - assertion below covers this path
            errors.append(exc)

    threads = [threading.Thread(target=target) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == 2
    assert results[0] == results[1] == cache_root / "demo_template"
    assert copy_calls == 1
    assert init_calls == 1
