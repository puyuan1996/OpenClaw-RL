"""Regression guards for the SETA-env eval analyzer.

The fixtures under tests/data are derived from the audit pack published with
issue #33 (seta_qwen3_8b_base_core_audit_20260709_101409.tar.gz, sha256
889f634decddfb681c1cc8b2c52b1c5dbad005313abb218812120893093ce110). The tests
below assert that this analyzer reproduces that run's published aggregates, so
the reported 38.77% / 21.61% baseline stays reproducible from source.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_RL = ROOT / "terminal-rl"
DATA = Path(__file__).resolve().parent / "data"
GOLDEN_PER_SAMPLE = DATA / "seta_env_eval_20260709_per_sample.csv"
GOLDEN_SUMMARY = DATA / "seta_env_eval_20260709_summary.json"

if str(TERMINAL_RL / "scripts") not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL / "scripts"))

import analyze_seta_env_eval as analyzer  # noqa: E402


@pytest.fixture(scope="module")
def golden_rows() -> list[dict[str, object]]:
    with GOLDEN_PER_SAMPLE.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def golden_summary() -> dict[str, object]:
    return json.loads(GOLDEN_SUMMARY.read_text(encoding="utf-8"))


def test_golden_fixture_is_intact(golden_rows, golden_summary):
    """Fixture integrity only. Asserts nothing about the analyzer."""
    assert len(golden_rows) == golden_summary["dataset_total"] == 1356


@pytest.mark.parametrize(
    "key",
    [
        "dataset_total",
        "result_count",
        "missing_count",
        "exact_pass_count",
        "nonzero_score_count",
    ],
)
def test_summarize_reproduces_published_counts(golden_rows, golden_summary, key):
    assert analyzer.summarize(golden_rows)[key] == golden_summary[key]


@pytest.mark.parametrize(
    "key",
    [
        "raw_score_sum_completed_rows",
        "raw_score_mean_completed_rows",
        "raw_score_mean_all_dataset_missing_as_zero",
        "exact_pass_rate_completed_rows",
        "exact_pass_rate_all_dataset_missing_as_zero",
        "nonzero_score_rate_completed_rows",
        "nonzero_score_rate_all_dataset_missing_as_zero",
    ],
)
def test_summarize_reproduces_published_rates(golden_rows, golden_summary, key):
    # Relative tolerance, not equality: the published numbers were produced by a
    # float summation in a different row order, which differs in the last ulp.
    assert analyzer.summarize(golden_rows)[key] == pytest.approx(golden_summary[key], rel=1e-12)


def test_summarize_reproduces_the_headline_baseline(golden_rows):
    """The two numbers issue #33 reports in its title."""
    summary = analyzer.summarize(golden_rows)
    assert round(summary["raw_score_mean_all_dataset_missing_as_zero"], 4) == 0.3877
    assert round(summary["exact_pass_rate_all_dataset_missing_as_zero"], 4) == 0.2161


@pytest.mark.parametrize(
    "key",
    [
        "status_counts",
        "raw_score_distribution",
        "run_counts",
        "turns",
        "tool_calls",
        "input_tokens",
        "output_tokens",
    ],
)
def test_summarize_reproduces_published_distributions(golden_rows, golden_summary, key):
    assert analyzer.summarize(golden_rows)[key] == golden_summary[key]


def test_exact_pass_is_stricter_than_nonzero(golden_rows, golden_summary):
    """Partial verifier credit must not be counted as solving the task."""
    summary = analyzer.summarize(golden_rows)
    assert summary["exact_pass_count"] < summary["nonzero_score_count"]
    assert summary["exact_pass_count"] == golden_summary["exact_pass_count"]
    # 0.5 scored a partial pass in this run and must not be an exact pass.
    assert golden_summary["raw_score_distribution"]["0.5"] > 0


def test_dataset_sample_index_is_the_line_number():
    dataset_path = TERMINAL_RL / "dataset" / "seta_env_convert" / "train.filtered.jsonl"
    if not dataset_path.is_file():
        pytest.skip(f"{dataset_path} is not checked out")
    samples = analyzer.read_dataset(dataset_path)
    assert [s.sample_index for s in samples] == list(range(len(samples)))
    assert samples[0].task_path == f"seta_env/{samples[0].task_name}"


def test_failure_event_regex_matches_the_real_log_format():
    line = (
        "(RolloutManager pid=283525) [2026-07-09 00:14:24] generate.py:4059 - "
        "[task=1080 uid=1b9bfb5a group_idx=-1 sample_idx=17] Generate failed "
        "(HTTPStatusError): Server error '500 Internal Server Error' for url "
        "'http://env-server:18080/reset'"
    )
    match = analyzer.FAILURE_EVENT_RE.search(line)
    assert match is not None
    assert match["task_name"] == "1080"
    assert match["uid"] == "1b9bfb5a"
    assert match["run_sample_index"] == "17"
    assert match["error_type"] == "HTTPStatusError"


def test_failure_events_deduplicate_retries_of_one_rollout(tmp_path):
    """One failing rollout logs a line per retry; the count is per rollout."""
    template = (
        "[task=806 uid=a3954d9a group_idx=-1 sample_idx={idx}] "
        "Generate failed (HTTPStatusError): boom\n"
    )
    log = tmp_path / "train.log"
    log.write_text(template.format(idx=27) * 3 + template.format(idx=28), encoding="utf-8")
    events = analyzer.read_failure_events(log, "main")
    assert [e.run_sample_index for e in events] == [27, 28]


def test_derive_turn_metrics_sums_over_turns():
    trajectory = {
        "turns": [
            {"tool_calls": [{}, {}], "n_input_tokens": 10, "n_output_tokens": 4,
             "parse_error_recorded": False},
            {"tool_calls": [{}], "n_input_tokens": 20, "n_output_tokens": 6,
             "parse_error_recorded": True},
        ]
    }
    assert analyzer.derive_turn_metrics(trajectory) == {
        "num_turns": 2.0,
        "tool_calls": 3,
        "parse_error_turns": 1,
        "input_tokens": 30,
        "output_tokens": 10,
    }


def _sample(index: int) -> analyzer.DatasetSample:
    return analyzer.DatasetSample(
        sample_index=index, task_name=str(index), task_path=f"seta_env/{index}",
        data_source="terminal_bench",
    )


def _index_row(index: int, run_label: str, run_order: int, score: float) -> analyzer.IndexRow:
    return analyzer.IndexRow(
        sample_index=index, sample_index_source="index.sample_index", run_label=run_label,
        run_order=run_order, run_sample_index=index, task_name=str(index),
        task_path=f"seta_env/{index}", uid=f"uid{index}{run_label}", status="COMPLETED",
        raw_score=score, raw_reward=score, task_reward=score, total_reward=score,
        num_turns=1.0, tool_calls=1, parse_error_turns=0, input_tokens=1, output_tokens=1,
        eval_error="", traj_path="",
    )


def test_later_runs_win_so_retries_replace_the_original_attempt():
    merged = analyzer.merge(
        [_sample(0)],
        [_index_row(0, "main", 0, 0.0), _index_row(0, "supp1", 1, 1.0)],
    )
    assert [(r["run_label"], r["raw_score"], r["exact_pass"]) for r in merged] == [("supp1", 1.0, 1)]


def test_an_unscored_retry_does_not_displace_a_scored_attempt():
    """Otherwise the sample is neither scored, nor missing, nor re-queued."""
    unscored = _index_row(0, "supp1", 1, 0.0)
    unscored.raw_score = None
    unscored.status = "FAILED"
    merged = analyzer.merge([_sample(0)], [_index_row(0, "main", 0, 1.0), unscored])

    assert [(r["run_label"], r["raw_score"]) for r in merged] == [("main", 1.0)]
    summary = analyzer.summarize(merged)
    assert summary["scored_count"] == 1
    assert summary["raw_score_mean_all_dataset_missing_as_zero"] == 1.0


def test_scored_count_exposes_a_present_row_that_carries_no_score():
    """result_count and the *_completed_rows denominator are not the same set."""
    unscored = _index_row(0, "main", 0, 0.0)
    unscored.raw_score = None
    summary = analyzer.summarize(analyzer.merge([_sample(0)], [unscored]))
    assert summary["result_count"] == 1
    assert summary["scored_count"] == 0
    assert summary["missing_count"] == 0


def test_samples_with_no_trajectory_are_missing_and_score_zero():
    merged = analyzer.merge([_sample(0), _sample(1)], [_index_row(0, "main", 0, 1.0)])
    missing = [row for row in merged if row["has_result"] == 0]
    assert [row["status"] for row in missing] == [analyzer.MISSING_STATUS]

    summary = analyzer.summarize(merged)
    assert summary["dataset_total"] == 2
    assert summary["missing_count"] == 1
    # Conservative denominator: the missing sample drags the mean to 0.5, and is
    # not silently dropped from the report.
    assert summary["raw_score_mean_all_dataset_missing_as_zero"] == 0.5
    assert summary["raw_score_mean_completed_rows"] == 1.0


def _write_dataset(path: Path, count: int) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "task": [{"role": "user", "content": f"task {i}"}],
                    "metadata": {
                        "task_name": str(i),
                        "task_path": f"seta_env/{i}",
                        "instruction": f"task {i}",
                        "data_source": "terminal_bench",
                    },
                }
            )
            + "\n"
            for i in range(count)
        ),
        encoding="utf-8",
    )


def test_supplement_jsonl_keeps_only_missing_rows_and_records_their_dataset_index(tmp_path):
    dataset = tmp_path / "train.jsonl"
    _write_dataset(dataset, 4)
    per_sample = analyzer.merge(
        [_sample(i) for i in range(4)],
        [_index_row(0, "main", 0, 1.0), _index_row(2, "main", 0, 0.0)],
    )

    out = tmp_path / "supp.jsonl"
    assert analyzer.write_supplement_jsonl(dataset, per_sample, out) == 2

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["metadata"]["supplement_sample_index"] for r in records] == [1, 3]
    assert [r["metadata"]["task_name"] for r in records] == ["1", "3"]


def test_a_sample_whose_every_run_went_unscored_is_still_re_queued(tmp_path):
    """Having a trajectory is not the same as having a score."""
    dataset = tmp_path / "train.jsonl"
    _write_dataset(dataset, 2)
    unscored = _index_row(0, "main", 0, 0.0)
    unscored.raw_score = None
    unscored.status = "FAILED"
    per_sample = analyzer.merge([_sample(0), _sample(1)], [unscored])

    out = tmp_path / "supp.jsonl"
    assert analyzer.write_supplement_jsonl(dataset, per_sample, out) == 2
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["metadata"]["supplement_sample_index"] for r in records] == [0, 1]


def test_supplement_index_is_what_maps_a_retry_back_to_its_dataset_row(tmp_path):
    """Closes the loop: what write_supplement_jsonl emits is what read_run reads."""
    run_dir = tmp_path / "supp1" / "trajectories" / "seta_task-9_uid"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                # Local index inside the filtered supplement, not the dataset index.
                "sample_index": 0,
                "task_name": "9",
                "task_path": "seta_env/9",
                "uid": "abc",
                "status": "Status.COMPLETED",
                "raw_score": 1.0,
                "sample_metadata": {"supplement_sample_index": 1192},
            }
        ),
        encoding="utf-8",
    )
    (rows,) = list(analyzer.read_run(tmp_path / "supp1", "supp1", 1))
    assert rows.sample_index == 1192
    assert rows.run_sample_index == 0
    assert rows.sample_index_source == "sample_metadata.supplement_sample_index"


def test_status_enum_prefix_is_stripped(tmp_path):
    run_dir = tmp_path / "main" / "trajectories" / "t0"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"sample_index": 0, "status": "Status.TRUNCATED", "raw_score": 0.5}),
        encoding="utf-8",
    )
    (row,) = list(analyzer.read_run(tmp_path / "main", "main", 0))
    assert row.status == "TRUNCATED"


def test_analyzer_cli_writes_the_expected_output_files(tmp_path):
    dataset = tmp_path / "train.jsonl"
    _write_dataset(dataset, 2)
    run_dir = tmp_path / "main" / "trajectories" / "t0"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"sample_index": 0, "task_name": "0", "task_path": "seta_env/0",
                    "status": "Status.COMPLETED", "raw_score": 1.0}),
        encoding="utf-8",
    )
    out = tmp_path / "analysis"
    assert analyzer.main(
        ["--dataset", str(dataset), "--run", f"main={tmp_path / 'main'}", "--out", str(out)]
    ) == 0
    assert {p.name for p in out.iterdir()} == {
        "summary.json", "per_sample.csv", "task_summary.csv",
        "status_counts.csv", "failure_events.csv",
    }
    assert json.loads((out / "summary.json").read_text())["missing_count"] == 1


def test_driver_script_parses_and_is_executable():
    script = TERMINAL_RL / "scripts" / "run_seta_env_eval.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert script.stat().st_mode & 0o111, "run_seta_env_eval.sh is not executable"


def _driver_defaults() -> dict[str, str]:
    """Parse `export NAME="${NAME:-literal}"` defaults out of the driver script.

    Defaults containing a nested expansion are skipped; the callers below only
    need the literal ones, and a missing key raises KeyError rather than passing.
    """
    source = (TERMINAL_RL / "scripts" / "run_seta_env_eval.sh").read_text(encoding="utf-8")
    return dict(re.findall(r'export (\w+)="\$\{\1:-([^${}]*)\}"', source))


def test_driver_script_disables_checkpoint_writing_by_default():
    """Nothing is trained, and the default checkpoint dir may not be writable."""
    assert _driver_defaults()["MAX_CKPT_KEEP"] == "0"
    assert "eval_only.py" in (TERMINAL_RL / "scripts" / "run_seta_env_eval.sh").read_text()


# --- cross-run comparison ----------------------------------------------------

import compare_seta_env_evals as compare  # noqa: E402


@pytest.mark.parametrize(
    "successes, trials, low, high",
    [
        # The intervals published in issues #27/#28/#29, as printed there.
        (5, 267, 0.80, 4.31),
        (3, 89, 1.15, 9.45),
        (6, 267, 1.03, 4.82),
        (4, 89, 1.76, 10.99),
        (5, 89, 2.42, 12.49),
    ],
)
def test_wilson_interval_reproduces_published_intervals(successes, trials, low, high):
    """A comparison tool must not draw intervals that disagree with the record."""
    got_low, got_high = compare.wilson_interval(successes, trials)
    assert round(got_low * 100, 2) == low
    assert round(got_high * 100, 2) == high


def test_comparison_reports_the_published_baseline(tmp_path):
    out = tmp_path / "summary.json"
    out.write_text(GOLDEN_SUMMARY.read_text(encoding="utf-8"), encoding="utf-8")
    run = compare.load_run("baseline", out)
    assert run.total == 1356
    assert run.exact_pass == 293
    assert round(run.exact_pass_rate * 100, 2) == 21.61
    assert round(run.raw_score_mean * 100, 2) == 38.77
    low, high = run.exact_pass_interval
    assert low < run.exact_pass_rate < high


def _summary(tmp_path, name: str, exact_pass: int, total: int = 1356) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "dataset_total": total,
                "exact_pass_count": exact_pass,
                "raw_score_mean_all_dataset_missing_as_zero": exact_pass / total,
                "missing_count": 0,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_gap_within_noise_is_reported_as_no_evidence(tmp_path):
    """293 vs 320 out of 1356: under 2 pp, p = 0.22."""
    runs = [
        compare.load_run("baseline", _summary(tmp_path, "a.json", 293)),
        compare.load_run("rl", _summary(tmp_path, "b.json", 320)),
    ]
    (pair,) = compare.compare_pairs(runs)
    assert pair.p_value == pytest.approx(0.2151, abs=5e-4)
    assert pair.is_significant is False
    text = compare.format_comparison(runs)
    assert "no evidence of a difference" in text
    assert "+1.99 pp" in text


def test_overlapping_intervals_do_not_override_a_significant_test(tmp_path):
    """The overlap fallacy, pinned.

    293 vs 352 out of 1356 have overlapping Wilson intervals and p = 0.008.
    Reading that pair off the intervals would discard a real effect, so the
    verdict must come from the test and the disagreement must be called out.
    """
    runs = [
        compare.load_run("baseline", _summary(tmp_path, "a.json", 293)),
        compare.load_run("rl", _summary(tmp_path, "b.json", 352)),
    ]
    (pair,) = compare.compare_pairs(runs)
    assert pair.intervals_overlap is True
    assert pair.p_value == pytest.approx(0.0078, abs=5e-4)
    assert pair.is_significant is True

    text = compare.format_comparison(runs)
    assert "differ (p < 0.05)" in text
    assert "the test, not the intervals, decides" in text
    assert "overlap does not imply" in text


def test_a_large_gap_is_significant(tmp_path):
    runs = [
        compare.load_run("baseline", _summary(tmp_path, "a.json", 293)),
        compare.load_run("much-better", _summary(tmp_path, "b.json", 700)),
    ]
    (pair,) = compare.compare_pairs(runs)
    assert pair.is_significant
    assert pair.intervals_overlap is False
    assert "differ (p < 0.05)" in compare.format_comparison(runs)


def test_two_proportion_test_matches_a_known_unequal_n_case():
    """Pins the pooling and the 1/n1 + 1/n2 term, which equal-n cases cannot."""
    z, p_value = compare.two_proportion_test(50, 500, 90, 1500)
    assert p_value == pytest.approx(0.00239832, rel=1e-6)
    assert z == pytest.approx(-3.0358837, rel=1e-6)


@pytest.mark.parametrize(
    "only_left, only_right, expected",
    [
        (0, 0, 1.0),        # no discordant items carries no information
        (5, 5, 1.0),
        (1, 0, 1.0),        # 2 * 0.5 = 1.0
        (0, 10, 0.001953125),
        (8, 35, 4.1934157934520044e-05),   # == scipy binomtest(8, 43, 0.5)
    ],
)
def test_mcnemar_exact_matches_the_binomial_sign_test(only_left, only_right, expected):
    assert compare.mcnemar_exact(only_left, only_right) == pytest.approx(expected, rel=1e-4)


def _per_sample(tmp_path, name: str, passes: list[bool]) -> Path:
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "raw_score", "has_result"])
        writer.writeheader()
        for index, passed in enumerate(passes):
            writer.writerow(
                {"sample_index": index, "raw_score": 1.0 if passed else 0.0, "has_result": 1}
            )
    return path


def test_per_sample_input_switches_to_the_paired_test(tmp_path):
    """Two runs over the same dataset are paired; the unpaired test loses power.

    Same 1356 items, 8 items only the baseline solves and 35 only the RL run
    solves. Unpaired that is +1.99 pp at p = 0.215, reported as no evidence;
    McNemar on the discordant items is p < 0.001.
    """
    total = 1356
    baseline = [index < 293 for index in range(total)]
    rl = list(baseline)
    for index in [i for i in range(total) if not baseline[i]][:35]:
        rl[index] = True
    for index in range(8):
        rl[index] = False

    runs = [
        compare.load_run("baseline", _per_sample(tmp_path, "a.csv", baseline)),
        compare.load_run("rl", _per_sample(tmp_path, "b.csv", rl)),
    ]
    (pair,) = compare.compare_pairs(runs)
    assert pair.test == "mcnemar-exact"
    assert pair.discordant == (8, 35)
    assert pair.p_value < 0.001
    assert pair.is_significant

    unpaired_p = compare.two_proportion_test(293, total, 320, total)[1]
    assert unpaired_p > 0.2, "the unpaired test on the same data finds no evidence"

    text = compare.format_comparison(runs)
    assert "McNemar exact   discordant 8/35" in text
    assert "the test, not the intervals, decides" in text


def test_mcnemar_survives_a_discordant_count_that_overflows_a_float():
    """math.comb returns an exact int that overflows float from 1025 pairs up."""
    assert compare.mcnemar_exact(512, 513) == pytest.approx(1.0, abs=1e-9)
    assert compare.mcnemar_exact(400, 800) < 1e-20
    assert compare.mcnemar_exact(2000, 2000) == pytest.approx(1.0, abs=1e-9)


def test_paired_join_uses_only_the_shared_samples(tmp_path):
    """A union or a left-only join would count items the other run never saw."""
    left = _per_sample(tmp_path, "a.csv", [True, True, False, False])
    right = tmp_path / "b.csv"
    with right.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "raw_score", "has_result"])
        writer.writeheader()
        # Overlaps on 2 and 3 only; 4 and 5 are unique to this run.
        for index, score in [(2, 1.0), (3, 0.0), (4, 1.0), (5, 1.0)]:
            writer.writerow({"sample_index": index, "raw_score": score, "has_result": 1})

    runs = [compare.load_run("a", left), compare.load_run("b", right)]
    (pair,) = compare.compare_pairs(runs)
    assert pair.test == "mcnemar-exact"
    # Shared indices are 2 and 3. On 2: a fails, b passes. On 3: both fail.
    assert pair.discordant == (0, 1)


def test_disjoint_sample_sets_fall_back_to_the_unpaired_test(tmp_path):
    left = _per_sample(tmp_path, "a.csv", [True, False])
    right = tmp_path / "b.csv"
    with right.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "raw_score", "has_result"])
        writer.writeheader()
        for index in (100, 101):
            writer.writerow({"sample_index": index, "raw_score": 1.0, "has_result": 1})
    (pair,) = compare.compare_pairs(
        [compare.load_run("a", left), compare.load_run("b", right)]
    )
    assert pair.test == "two-proportion-z"


def test_one_sided_per_sample_input_falls_back_to_the_unpaired_test(tmp_path):
    runs = [
        compare.load_run("csv", _per_sample(tmp_path, "a.csv", [True, False])),
        compare.load_run("summary", _summary(tmp_path, "b.json", 1, total=2)),
    ]
    (pair,) = compare.compare_pairs(runs)
    assert pair.test == "two-proportion-z"


def test_per_sample_rows_without_a_score_count_as_missing_non_passes(tmp_path):
    path = tmp_path / "a.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "raw_score", "has_result"])
        writer.writeheader()
        writer.writerow({"sample_index": 0, "raw_score": 1.0, "has_result": 1})
        writer.writerow({"sample_index": 1, "raw_score": 0.5, "has_result": 1})
        writer.writerow({"sample_index": 2, "raw_score": "", "has_result": 0})

    run = compare.load_run("a", path)
    assert run.total == 3
    assert run.missing == 1
    assert run.exact_pass == 1                      # 0.5 is partial credit, not a pass
    assert run.raw_score_mean == pytest.approx(0.5)  # (1.0 + 0.5 + 0) / 3


def test_a_repeated_sample_index_is_rejected(tmp_path):
    path = tmp_path / "dup.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "raw_score", "has_result"])
        writer.writeheader()
        for _ in range(2):
            writer.writerow({"sample_index": 0, "raw_score": 1.0, "has_result": 1})
    with pytest.raises(ValueError, match="repeats sample_index 0"):
        compare.load_run("dup", path)


def test_summary_input_says_the_unpaired_test_is_the_weaker_choice(tmp_path):
    runs = [
        compare.load_run("baseline", _summary(tmp_path, "a.json", 293)),
        compare.load_run("rl", _summary(tmp_path, "b.json", 320)),
    ]
    text = compare.format_comparison(runs)
    assert "two-proportion z (unpaired)" in text
    assert "assumes independent samples" in text
    assert "pass per_sample.csv" in text


def test_two_proportion_test_is_symmetric_and_zero_for_identical_runs(tmp_path):
    z, p_value = compare.two_proportion_test(293, 1356, 293, 1356)
    assert z == 0.0
    assert p_value == 1.0
    forward = compare.two_proportion_test(293, 1356, 352, 1356)
    backward = compare.two_proportion_test(352, 1356, 293, 1356)
    assert forward[0] == pytest.approx(-backward[0])
    assert forward[1] == pytest.approx(backward[1])


def test_a_degenerate_run_is_flagged_not_declared_different(tmp_path):
    """dataset_total = 0 must not read as 'separable from everything'."""
    empty = tmp_path / "empty.json"
    empty.write_text(
        json.dumps(
            {
                "dataset_total": 0,
                "exact_pass_count": 0,
                "raw_score_mean_all_dataset_missing_as_zero": None,
                "missing_count": 0,
            }
        ),
        encoding="utf-8",
    )
    runs = [
        compare.load_run("baseline", _summary(tmp_path, "a.json", 293)),
        compare.load_run("empty", empty),
    ]
    (pair,) = compare.compare_pairs(runs)
    assert pair.is_significant is False
    text = compare.format_comparison(runs)
    assert "WARNING: a run has dataset_total = 0" in text
    # The intervals do NOT overlap here, so the note must not claim they do, and
    # an undefined z must not be printed as a confident +0.000.
    # The table prints n/a for a zero-sample interval, so the per-pair note must
    # not claim anything about overlap. The closing general remark still may.
    assert "[Wilson intervals" not in text
    assert pair.z is None
    assert "+0.000" not in text
    # A null mean on an empty dataset renders as n/a, not as a real 0.00%.
    assert "n/a" in text


def test_load_run_names_the_file_and_the_missing_keys(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"dataset_total": 10}), encoding="utf-8")
    with pytest.raises(KeyError) as excinfo:
        compare.load_run("bad", bad)
    message = str(excinfo.value)
    assert "bad.json" in message
    assert "exact_pass_count" in message


def test_comparison_cli_emits_json(tmp_path, capsys):
    args = [
        f"baseline={_summary(tmp_path, 'a.json', 293)}",
        f"rl={_summary(tmp_path, 'b.json', 352)}",
        "--json",
    ]
    assert compare.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["label"] for r in payload["runs"]] == ["baseline", "rl"]
    (pair,) = payload["pairs"]
    assert pair["significant_at_0_05"] is True
    assert pair["wilson_intervals_overlap"] is True
    assert pair["p_value"] == pytest.approx(0.0078, abs=5e-4)


def test_comparison_cli_rejects_a_missing_label(tmp_path):
    with pytest.raises(SystemExit):
        compare.main([str(_summary(tmp_path, "a.json", 293))])


def test_driver_script_pins_one_rollout_per_prompt():
    """The launcher this delegates to defaults EVAL_N_SAMPLES to 16.

    The published baseline ran n_samples=1, and the analyzer keeps one trajectory
    per sample, so inheriting 16 would cost 16x and report one arbitrary rollout
    out of sixteen. Guard the override rather than the comment explaining it.
    """
    launcher = (TERMINAL_RL / "terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh").read_text()
    assert 'EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-16}"' in launcher, (
        "the downstream default changed; re-check what this driver must override"
    )
    assert _driver_defaults()["EVAL_N_SAMPLES"] == "1"
