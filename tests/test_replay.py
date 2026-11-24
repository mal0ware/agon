"""The committed logs rebuild the committed outcomes, and nothing else does."""

import gzip
import json
from pathlib import Path

import pytest

from agon.backends import FakeBackend, Request, fake_key
from agon.filings import synthetic_statements
from agon.questions import build_questions, save_questions
from agon.replay import RUN_MODELS, ReplayBackend, export_logs, log_name, rebuild_outcomes
from agon.report import build_report, collect, lean_table, plain_name, read_outcomes, write_outcomes
from agon.runner import CallLog, run_experiment
from agon.systems import Protocol

ROOT = Path(__file__).resolve().parents[1]
PROTO = Protocol(cap=30)
QUESTIONS = build_questions(synthetic_statements(12, 7), 7)
TRUTH = {fake_key(q.company, q.claim): q.truth for q in QUESTIONS}
MODELS = ["a", "b"]


def _run(tmp_path: Path) -> tuple[Path, dict]:  # type: ignore[type-arg]
    run = tmp_path / "run"
    backend = FakeBackend(TRUTH, {"a": 0.7, "b": 0.6}, herding=0.5)
    outcomes = run_experiment(QUESTIONS, MODELS, backend, PROTO, CallLog(run / "calls.jsonl"))
    save_questions(QUESTIONS, run / "questions.json")
    return run, outcomes


def test_exported_logs_rebuild_the_same_outcomes(tmp_path: Path) -> None:
    run, outcomes = _run(tmp_path)
    counts = export_logs(run / "calls.jsonl", run / "questions.json", tmp_path / "logs")
    assert set(counts) == {log_name("a"), log_name("b")}
    questions, rebuilt = rebuild_outcomes(tmp_path / "logs", MODELS, PROTO)
    assert questions == QUESTIONS
    assert rebuilt == outcomes


def test_export_is_byte_for_byte_repeatable(tmp_path: Path) -> None:
    run, _ = _run(tmp_path)
    export_logs(run / "calls.jsonl", run / "questions.json", tmp_path / "one")
    export_logs(run / "calls.jsonl", run / "questions.json", tmp_path / "two")
    for name in (log_name("a"), log_name("b"), "questions.json.gz"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_a_missing_call_cannot_be_replayed(tmp_path: Path) -> None:
    run, _ = _run(tmp_path)
    export_logs(run / "calls.jsonl", run / "questions.json", tmp_path / "logs", ["a", "b"])
    with gzip.open(tmp_path / "logs" / log_name("b"), "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    with gzip.open(tmp_path / "logs" / log_name("b"), "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines[1:]) + "\n")
    with pytest.raises(KeyError):
        rebuild_outcomes(tmp_path / "logs", MODELS, PROTO)
    with pytest.raises(KeyError):
        ReplayBackend().complete(Request("a", "p", 1, 0.0, 0))


def test_an_edited_record_is_refused(tmp_path: Path) -> None:
    run, _ = _run(tmp_path)
    lines = (run / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["request"]["prompt"] += " edited"
    lines[0] = json.dumps(record)
    (run / "calls.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        export_logs(run / "calls.jsonl", None, tmp_path / "logs")


@pytest.mark.skipif(
    not (ROOT / "logs" / "questions.json.gz").exists(), reason="no committed logs in this checkout"
)
def test_committed_logs_rebuild_the_committed_outcomes(tmp_path: Path) -> None:
    questions, outcomes = rebuild_outcomes(ROOT / "logs", RUN_MODELS)
    write_outcomes(tmp_path / "outcomes.csv", questions, outcomes)
    committed = (ROOT / "results" / "outcomes.csv").read_bytes()
    assert (tmp_path / "outcomes.csv").read_bytes() == committed


def test_lean_table_counts_true_verdicts_and_accuracy_by_truth(tmp_path: Path) -> None:
    always_true = FakeBackend(TRUTH, {"a": 1.0, "b": 0.0})
    outcomes = run_experiment(QUESTIONS, MODELS, always_true, PROTO)
    write_outcomes(tmp_path / "outcomes.csv", QUESTIONS, outcomes)
    rows = {(r["config"], r["system"]): r for r in lean_table(collect(read_outcomes(
        tmp_path / "outcomes.csv")))}  # fmt: skip
    share = sum(q.truth for q in QUESTIONS) / len(QUESTIONS)
    perfect, wrong = rows["a", "single"], rows["b", "single"]
    assert float(perfect["said_true"]) == pytest.approx(share, abs=1e-3)
    assert float(wrong["said_true"]) == pytest.approx(1 - share, abs=1e-3)
    assert (perfect["acc_true"], perfect["acc_false"]) == ("1.000", "1.000")
    assert (wrong["acc_true"], wrong["acc_false"]) == ("0.000", "0.000")
    assert float(perfect["lo"]) < float(perfect["said_true"]) < float(perfect["hi"])
    build_report(tmp_path)
    assert (tmp_path / "verdict_lean.md").exists()
    assert (tmp_path / "verdict_lean_shift.md").exists()


def test_plain_name_removes_non_breaking_spaces() -> None:
    assert plain_name("Matson," + chr(0xA0) + "Inc.") == "Matson, Inc."
    assert plain_name("  A  B ") == "A B"
