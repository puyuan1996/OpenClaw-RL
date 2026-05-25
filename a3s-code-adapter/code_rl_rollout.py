import asyncio
import atexit
import os
import queue
import threading
import time

from code_rl_api_server import CodeRLAPIServer
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import eval_rollout
from slime.utils.async_utils import run
from slime.utils.types import Sample

_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer):
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            _global_worker = AsyncRolloutWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    def __init__(self, args, data_buffer):
        self.args = args
        self.data_buffer = data_buffer
        self.running = True
        self.output_queue = queue.Queue(maxsize=100000)
        self.partial_groups: dict[int, list[Sample]] = {}
        self.worker_thread = None
        self._submission_enabled = threading.Event()
        self._server = CodeRLAPIServer(
            args=args,
            output_queue=self.output_queue,
            submission_enabled=self._submission_enabled,
        )

    async def continuous_worker_loop(self):
        while self.running:
            await asyncio.sleep(1.0)

    def worker_thread_func(self):
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        self._server.start()
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()

    def stop(self):
        self.running = False
        self._submission_enabled.clear()
        self._server.stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def pause_submission(self):
        if self._submission_enabled.is_set():
            self._submission_enabled.clear()
            epoch = self._server.advance_submission_epoch()
            drained = self._server.wait_for_inflight_generation_requests()
            discarded_groups = 0
            discarded_samples = 0
            if self.partial_groups:
                discarded_groups += len(self.partial_groups)
                discarded_samples += sum(len(group) for group in self.partial_groups.values())
                self.partial_groups.clear()
            while True:
                try:
                    _, group = self.output_queue.get_nowait()
                except queue.Empty:
                    break
                discarded_groups += 1
                discarded_samples += len(group)
            self._server.purge_record_files()
            if discarded_groups:
                print(
                    "[CodeRLWorker] submission paused; "
                    f"epoch={epoch} drained={drained} "
                    f"discarded {discarded_groups} queued groups / {discarded_samples} samples"
                )
            else:
                print(f"[CodeRLWorker] submission paused; epoch={epoch} drained={drained}")

    def resume_submission(self):
        if not self._submission_enabled.is_set():
            self._submission_enabled.set()
            print("[CodeRLWorker] submission resumed")

    def get_completed_groups(self) -> list[tuple]:
        completed = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


async def _drain_output_queue(args, worker: AsyncRolloutWorker) -> list[list[Sample]]:
    target_data_size = args.rollout_batch_size
    target_group_size = max(1, int(getattr(args, "n_samples_per_prompt", 1)))
    idle_timeout_sec = float(os.getenv("CODE_RL_ROLLOUT_IDLE_TIMEOUT_SEC", "0") or 0)
    hard_timeout_sec = float(os.getenv("CODE_RL_ROLLOUT_DRAIN_TIMEOUT_SEC", "0") or 0)
    data: list[list[Sample]] = []
    completed_groups = worker.partial_groups
    start = time.time()
    last_activity = start
    last_status_log = start

    while len(data) < target_data_size:
        completed = worker.get_completed_groups()
        if completed:
            last_activity = time.time()
            for group_id, group in completed:
                completed_groups.setdefault(group_id, []).extend(group)

        for group_id in list(completed_groups.keys()):
            if len(data) >= target_data_size:
                break
            group = completed_groups[group_id]
            if len(group) < target_group_size:
                continue
            group = completed_groups.pop(group_id)
            if len(group) > target_group_size:
                group, remainder = group[:target_group_size], group[target_group_size:]
                if remainder:
                    completed_groups[group_id] = remainder
            if any(sample.status == Sample.Status.ABORTED for sample in group):
                continue
            data.append(group)
            last_activity = time.time()

        now = time.time()
        if hard_timeout_sec > 0 and now - start > hard_timeout_sec:
            raise TimeoutError(
                "Timed out waiting for API-produced samples: "
                f"groups={len(data)}/{target_data_size}, group_size={target_group_size}, "
                f"elapsed={now - start:.1f}s, hard_timeout={hard_timeout_sec:.1f}s"
            )
        if idle_timeout_sec > 0 and now - last_activity > idle_timeout_sec:
            raise TimeoutError(
                "Timed out waiting for API-produced samples after no accepted group progress: "
                f"groups={len(data)}/{target_data_size}, group_size={target_group_size}, "
                f"idle={now - last_activity:.1f}s, idle_timeout={idle_timeout_sec:.1f}s, "
                f"pending_groups={len(completed_groups)}, queue={worker.get_queue_size()}"
            )

        if now - last_status_log > 30:
            stats = None
            snapshot = getattr(getattr(worker, "_server", None), "snapshot_stats", None)
            if callable(snapshot):
                try:
                    stats = snapshot()
                except Exception:
                    stats = None
            print(
                f"[CodeRLWorker] waiting for API-produced samples: {len(data)}/{target_data_size}, "
                f"group_size={target_group_size} pending_groups={len(completed_groups)} "
                f"queue={worker.get_queue_size()} idle={now - last_activity:.1f}s "
                f"stats={stats}",
                flush=True,
            )
            last_status_log = now

        if len(data) < target_data_size:
            await asyncio.sleep(0.05)

    data.sort(key=lambda group: group[0].index if group and group[0].index is not None else -1)
    print(f"[CodeRLWorker] drained {len(data)} groups in {time.time() - start:.2f}s", flush=True)
    return data


def generate_rollout_code_rl(args, rollout_id, data_buffer, evaluation=False):
    worker = get_global_worker(args, data_buffer)

    if evaluation:
        eval_output, _ = run(eval_rollout(args, rollout_id))
        return eval_output

    worker._server.reset_eval_scores()
    worker.resume_submission()
    completed_samples = run(_drain_output_queue(args, worker))
    worker.pause_submission()

    extra_metrics = None
    eval_scores = worker._server.drain_eval_scores()
    if eval_scores:
        extra_metrics = {"rollout/reward_mean": sum(eval_scores) / len(eval_scores)}
        print(
            f"[CodeRLWorker] reward_mean={extra_metrics['rollout/reward_mean']:.4f} "
            f"(n={len(eval_scores)})",
            flush=True,
        )

    return RolloutFnTrainOutput(samples=completed_samples, metrics=extra_metrics)


atexit.register(stop_global_worker)
