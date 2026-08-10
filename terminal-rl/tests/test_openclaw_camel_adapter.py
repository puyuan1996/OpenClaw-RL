"""Guards for the mode B aligned Harbor adapter.

The adapter's value is that its defaults reproduce the training-time harness, so
the evals recorded in docs/HARBOR_CAMEL_MODE_B_zh.md stay comparable. Silently
changing one of those defaults would not break anything at runtime, it would just
make every future number incomparable to the recorded ones -- which is exactly the
kind of drift a test has to catch.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_RL = ROOT / "terminal-rl"
if str(TERMINAL_RL / "scripts") not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL / "scripts"))
MODE_B = TERMINAL_RL / "eval" / "mode_b_aligned"
ADAPTER_DIR = MODE_B / "adapter"
ADAPTER_PY = ADAPTER_DIR / "openclaw_camel_adapter.py"
LAUNCHERS = sorted((MODE_B / "launchers").glob("*.sh"))

# Importing the adapter pulls in harbor, camel and transformers. Skip rather than
# fail where those are absent, so the rest of the suite stays runnable.
pytest.importorskip("harbor", reason="harbor is required to import the adapter")
pytest.importorskip("camel", reason="camel-ai is required to import the adapter")

for _extra in (ADAPTER_DIR, MODE_B):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import openclaw_camel_adapter as adapter_module  # noqa: E402

OpenClawCamelAgent = adapter_module.OpenClawCamelAgent


def _agent(**kwargs):
    return OpenClawCamelAgent(
        logs_dir=Path(tempfile.mkdtemp()),
        sglang_served_name="test-served-name",
        hf_model_dir="/nonexistent/hf-dir",
        **kwargs,
    )


def test_terminal_rl_root_resolves_from_the_adapter_location():
    """The adapter must work from any checkout and cwd, without env setup."""
    resolved = Path(adapter_module._resolve_terminal_rl_dir())
    assert resolved == TERMINAL_RL
    assert (resolved / "agent" / "camel_agent.py").is_file()


def test_terminal_rl_root_override_reports_the_bad_path(tmp_path, monkeypatch):
    """A wrong OPENCLAW_TERMINAL_RL_DIR must fail loudly, not as ModuleNotFoundError."""
    monkeypatch.setenv("OPENCLAW_TERMINAL_RL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        adapter_module._resolve_terminal_rl_dir()
    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "OPENCLAW_TERMINAL_RL_DIR" in message


@pytest.mark.parametrize(
    "kwargs, missing",
    [
        ({}, "sglang_served_name"),
        ({"sglang_served_name": "some-model"}, "hf_model_dir"),
    ],
)
def test_checkpoint_identity_kwargs_are_required(kwargs, missing):
    """A wrong served name or tokenizer dir evaluates the wrong thing silently."""
    with pytest.raises(ValueError, match=missing):
        OpenClawCamelAgent(logs_dir=Path(tempfile.mkdtemp()), **kwargs)


def test_aligned_knob_defaults_match_the_recorded_evals():
    """Pins knobs 5-13 of the alignment table in docs/HARBOR_CAMEL_MODE_B_zh.md."""
    agent = _agent()
    assert agent.max_iteration == 10
    assert agent.max_parse_errors == 3
    assert agent.temperature == 1.0
    assert agent.top_p == 1.0
    assert agent.top_k == -1
    assert agent.max_new_tokens == 8192
    assert agent.max_total_tokens == 16384
    assert agent.rollout_skip_special_tokens is False
    assert agent.tool_call_parser == "qwen25"
    assert agent.non_think_mode is False


def test_sampling_params_carry_exactly_the_knobs_that_reach_sglang():
    """rollout_seed is metadata only, so the payload must not imply seeded sampling.

    top_k is omitted rather than sent as -1, matching the eight recorded evals.
    """
    params = _agent()._build_sampling_params()
    assert params == {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 8192,
        "skip_special_tokens": False,
    }
    assert "seed" not in params
    assert _agent(top_k=20)._build_sampling_params()["top_k"] == 20


def test_sglang_url_is_normalised_to_the_generate_endpoint():
    """Callers pass a server root; the client needs the /generate path."""
    assert _agent(sglang_url="http://127.0.0.1:30000").sglang_url.endswith("/generate")
    assert _agent(sglang_url="http://127.0.0.1:30000/generate").sglang_url.count("/generate") == 1


def test_adapter_reports_a_stable_identity():
    """Harbor writes these into the job manifest, so results stay attributable."""
    assert OpenClawCamelAgent.name() == "openclaw-camel-agent"
    assert _agent().version() == "0.1.0"


def test_launcher_scripts_exist():
    """Without this, the parametrized checks below would vacuously pass on an empty glob."""
    assert {p.name for p in LAUNCHERS} == {"launch_sglang.sh", "run_harbor_eval.sh"}


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_launchers_parse_and_are_executable(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert script.stat().st_mode & 0o111, f"{script.name} is not executable"


@pytest.mark.parametrize(
    "path",
    [ADAPTER_PY] + LAUNCHERS,
    ids=lambda p: p.name,
)
def test_runtime_files_carry_no_site_specific_absolute_paths(path):
    """Site paths belong in docs and env vars, never in code the next site runs."""
    offenders = [
        f"{path.name}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if "/mnt/shared-storage-user/" in line
        or "/mnt/data/deepghs/" in line
        or "/nfs/eval_results/" in line
    ]
    assert not offenders, "site-specific absolute paths must not be hardcoded:\n" + "\n".join(offenders)


# --- Harbor job report -------------------------------------------------------

import harbor_job_report  # noqa: E402


def _write_harbor_job(
    root: Path,
    *,
    n_total_trials: int,
    solved: list[str],
    zero_reward: int,
    timeouts: int,
    no_reward: int,
    finished: bool = True,
) -> Path:
    """Build a Harbor job directory with a known shape."""
    job = root / "job"
    job.mkdir(parents=True)
    aggregate: dict[str, object] = {
        "started_at": "2026-07-01T22:00:13.674888",
        "n_total_trials": n_total_trials,
    }
    if finished:
        aggregate["finished_at"] = "2026-07-02T07:22:00.608529"
    (job / "result.json").write_text(json.dumps(aggregate), encoding="utf-8")

    def _trial(name: str, payload: dict[str, object]) -> None:
        directory = job / name
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    for name in solved:
        _trial(name, {"verifier_result": {"rewards": {"reward": 1.0}}})
    for index in range(zero_reward):
        _trial(f"zero-{index}", {"verifier_result": {"rewards": {"reward": 0.0}}})
    for index in range(timeouts):
        _trial(
            f"timeout-{index}",
            {
                "verifier_result": {"rewards": {"reward": 0.0}},
                "exception_info": {
                    "exception_type": "AgentTimeoutError",
                    "exception_message": "AgentTimeoutError: 20",
                },
            },
        )
    for index in range(no_reward):
        _trial(f"no-reward-{index}", {"exception_info": {"exception_type": "RewardFileNotFoundError"}})
    return job


@pytest.fixture
def recorded_tbv21_job(tmp_path):
    """The run recorded in docs/TBV21_HARBOR_FULL_EVAL_zh.md.

    89 trials, reward_sum 2.0, 20 AgentTimeoutError, and exactly one trial that
    never reached the verifier -- which is what makes the two denominators differ.
    """
    return _write_harbor_job(
        tmp_path,
        n_total_trials=89,
        solved=["configure-git-webserver__zVAaSVv", "hf-model-inference__72rnD2H"],
        zero_reward=66,
        timeouts=20,
        no_reward=1,
    )


def test_job_report_reproduces_the_recorded_tbv21_run(recorded_tbv21_job):
    report = harbor_job_report.read_job(recorded_tbv21_job)
    assert len(report.trials) == 89
    assert report.n_total_trials == 89
    assert report.reward_sum == 2.0
    assert report.error_counts["AgentTimeoutError"] == 20
    assert sorted(t.name for t in report.solved) == [
        "configure-git-webserver__zVAaSVv",
        "hf-model-inference__72rnD2H",
    ]
    assert report.is_finished


def test_job_report_denominator_is_every_intended_trial(recorded_tbv21_job):
    """2.0/89, not 2.0/88: a trial that errored before the verifier still counts."""
    report = harbor_job_report.read_job(recorded_tbv21_job)
    assert len(report.rewarded_trials) == 88
    assert report.score == pytest.approx(0.0224719101, abs=1e-10)
    assert report.score_over_rewarded_only == pytest.approx(0.0227272727, abs=1e-10)
    assert report.score < report.score_over_rewarded_only


def test_job_report_attaches_each_label_to_the_right_number(recorded_tbv21_job):
    """Binding matters: marking 2.0/88 as the one to report would be the bug."""
    text = harbor_job_report.format_report(harbor_job_report.read_job(recorded_tbv21_job))
    assert "2.0 / 89 = 0.0224719101   <- report this one" in text
    assert "over the 88 trials that reached the verifier: 0.0227272727" in text
    assert "not the reporting number" in text


def test_job_report_marks_an_unfinished_job(tmp_path):
    job = _write_harbor_job(
        tmp_path, n_total_trials=89, solved=[], zero_reward=3, timeouts=0, no_reward=0,
        finished=False,
    )
    report = harbor_job_report.read_job(job)
    assert report.is_finished is False
    assert len(report.trials) == 3


def test_job_report_survives_a_half_written_trial_result(tmp_path):
    """A watch loop must not die on a file Harbor is mid-write."""
    job = _write_harbor_job(
        tmp_path, n_total_trials=2, solved=["ok"], zero_reward=0, timeouts=0, no_reward=0,
    )
    partial = job / "half-written"
    partial.mkdir()
    (partial / "result.json").write_text('{"verifier_result": {"rew', encoding="utf-8")

    report = harbor_job_report.read_job(job)
    assert report.reward_sum == 1.0
    assert len(report.trials) == 2


def test_job_report_cli_watch_stops_once_the_job_is_finished(recorded_tbv21_job, capsys):
    assert harbor_job_report.main([str(recorded_tbv21_job), "--watch", "--interval", "0"]) == 0
    assert "poll 1" in capsys.readouterr().out


def _one_trial_job(tmp_path, reward) -> Path:
    job = tmp_path / "job"
    (job / "t0").mkdir(parents=True)
    (job / "result.json").write_text(json.dumps({"n_total_trials": 1}), encoding="utf-8")
    (job / "t0" / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": reward}}}), encoding="utf-8"
    )
    return job


@pytest.mark.parametrize("reward", ["1.0", 1, 1.0])
def test_job_report_accepts_a_numeric_reward_in_any_json_form(tmp_path, reward):
    assert harbor_job_report.read_job(_one_trial_job(tmp_path, reward)).reward_sum == 1.0


@pytest.mark.parametrize(
    "reward",
    [None, True, [], {"a": 1}, "abc", float("nan"), float("inf")],
    ids=["null", "bool", "list", "dict", "text", "nan", "inf"],
)
def test_an_unusable_reward_scores_zero_rather_than_poisoning_the_job(tmp_path, reward):
    """NaN would otherwise flow into reward_sum and annihilate every other trial,
    and json.dumps would emit a bare NaN token that is not valid JSON."""
    report = harbor_job_report.read_job(_one_trial_job(tmp_path, reward))
    assert report.reward_sum == 0.0
    assert report.score == 0.0
    assert report.rewarded_trials == []
    # Scan the two score lines only: the full report embeds job_dir, so a path
    # containing "nan" would otherwise decide this assertion.
    numeric = [
        line for line in harbor_job_report.format_report(report).splitlines()
        if line.startswith(("reward_sum", "score"))
    ]
    assert numeric and not any("nan" in line.lower() for line in numeric)
    assert "NaN" not in json.dumps(report.to_dict())


def test_job_report_watch_that_hits_its_poll_cap_exits_nonzero(tmp_path):
    """An unfinished job must not report success just because polling stopped."""
    job = _write_harbor_job(
        tmp_path, n_total_trials=2, solved=[], zero_reward=1, timeouts=0, no_reward=0,
        finished=False,
    )
    assert harbor_job_report.main(
        [str(job), "--watch", "--interval", "0", "--max-polls", "2"]
    ) == 1


def test_job_report_watch_json_stays_parseable_line_by_line(tmp_path, capsys):
    """Several indent=2 documents concatenated would not parse, so watch emits one per line."""
    job = _write_harbor_job(
        tmp_path, n_total_trials=9, solved=[], zero_reward=1, timeouts=0, no_reward=0,
        finished=False,
    )
    harbor_job_report.main(
        [str(job), "--watch", "--interval", "0", "--max-polls", "3", "--json"]
    )
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 3
    assert all(json.loads(line)["is_finished"] is False for line in lines)


def test_job_report_cli_emits_json(recorded_tbv21_job, capsys):
    assert harbor_job_report.main([str(recorded_tbv21_job), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["score"] == pytest.approx(2.0 / 89)
    # The trial that never reached the verifier is an error too, and is what makes
    # n_rewarded_trials 88 rather than 89.
    assert payload["error_counts"] == {"AgentTimeoutError": 20, "RewardFileNotFoundError": 1}
    assert payload["n_result_files"] == 89
    assert payload["n_rewarded_trials"] == 88


# --- eval-history figure -----------------------------------------------------

import plot_modeb_eval_history as history_plot  # noqa: E402


@pytest.mark.parametrize(
    "issue, successes, trials, published_low, published_high",
    [
        # Percentages as printed in each issue body's TL;DR.
        ("#27 pass@1", 5, 267, 0.80, 4.31),
        ("#27 pass@3", 3, 89, 1.15, 9.45),
        ("#28 pass@1", 6, 267, 1.03, 4.82),
        ("#28 pass@3", 4, 89, 1.76, 10.99),
        ("#29 pass@3", 5, 89, 2.42, 12.49),
        ("#31 pass@1", 3, 267, 0.38, 3.25),
        ("#31 pass@3", 3, 89, 1.15, 9.45),
    ],
)
def test_wilson_interval_reproduces_the_published_intervals(
    issue, successes, trials, published_low, published_high
):
    """The figure must not draw intervals that disagree with the issues it cites."""
    low, high = history_plot.wilson_interval(successes, trials)
    assert round(low * 100, 2) == published_low, issue
    assert round(high * 100, 2) == published_high, issue


def test_eval_history_rows_match_the_documented_pass_at_1():
    """Keeps the figure's table and the doc's table from drifting apart."""
    documented = {
        "#21": (3, 267), "#22": (8, 267), "#24": (6, 267), "#25": (3, 267),
        "#27": (5, 267), "#28": (6, 267), "#29": (5, 267), "#31": (3, 267),
    }
    assert {e.issue: (e.successes, e.trials) for e in history_plot.EVALS} == documented

    doc = (TERMINAL_RL / "docs" / "HARBOR_CAMEL_MODE_B_zh.md").read_text(encoding="utf-8")
    for row in history_plot.EVALS:
        rendered = f"{row.successes / row.trials * 100:.2f}%（{row.successes}/{row.trials}）"
        assert rendered in doc, f"{row.issue}: {rendered} missing from the history table"
