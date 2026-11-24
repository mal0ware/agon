"""The harness end to end against a fake model with known behaviour."""

import json
import math
from pathlib import Path

import pytest

from agon.backends import Completion, FakeBackend, OllamaBackend, Request, fake_key
from agon.filings import synthetic_statements
from agon.parsing import parse_answer
from agon.questions import build_questions
from agon.report import build_report, read_outcomes, write_outcomes
from agon.runner import CallLog, run_experiment
from agon.systems import Protocol, _answer_request

PROTO = Protocol(cap=40)
QUESTIONS = build_questions(synthetic_statements(300, 4), 4)
TRUTH = {fake_key(q.company, q.claim): q.truth for q in QUESTIONS}


def accuracy(outcomes: dict, config: str, system: str) -> float:  # type: ignore[type-arg]
    truth = {q.qid: q.truth for q in QUESTIONS}
    hits = [
        o.verdict == truth[qid] for (qid, c, s), o in outcomes.items() if (c, s) == (config, system)
    ]
    return sum(hits) / len(hits)


def test_fake_backend_is_deterministic_and_has_the_stated_skill() -> None:
    backend = FakeBackend(TRUTH, {"a": 0.7})
    outcomes = run_experiment(QUESTIONS, ["a"], backend, PROTO)
    again = run_experiment(QUESTIONS, ["a"], FakeBackend(TRUTH, {"a": 0.7}), PROTO)
    assert outcomes == again
    assert accuracy(outcomes, "a", "single") == pytest.approx(0.7, abs=0.07)
    assert accuracy(outcomes, "a", "closed_book") == pytest.approx(0.5, abs=0.08)


def test_a_perfect_model_scores_one_and_a_wrong_one_scores_zero() -> None:
    perfect = run_experiment(QUESTIONS[:40], ["a"], FakeBackend(TRUTH, {"a": 1.0}), PROTO)
    wrong = run_experiment(QUESTIONS[:40], ["a"], FakeBackend(TRUTH, {"a": 0.0}), PROTO)
    for system in ("single", "long", "vote5", "vote6", "judge", "debate"):
        truth = {q.qid: q.truth for q in QUESTIONS}
        assert all(o.verdict == truth[k[0]] for k, o in perfect.items() if k[2] == system)
        assert all(o.verdict != truth[k[0]] for k, o in wrong.items() if k[2] == system)


def test_majority_vote_follows_the_binomial() -> None:
    # Independent members each right with probability p: a vote of five is right
    # with sum_{k>=3} C(5,k) p^k (1-p)^(5-k), a vote of three with 3p^2 - 2p^3.
    backend = FakeBackend(TRUTH, {"a": 0.7})
    outcomes = run_experiment(QUESTIONS, ["a"], backend, PROTO)
    members = [
        [
            parse_answer(backend.complete(_answer_request(q, ("a", i), PROTO)).text).verdict
            == q.truth
            for i in range(5)
        ]
        for q in QUESTIONS
    ]
    p = sum(map(sum, members)) / (5 * len(QUESTIONS))
    assert p == pytest.approx(0.7, abs=0.04)
    five = sum(math.comb(5, k) * p**k * (1 - p) ** (5 - k) for k in (3, 4, 5))
    assert accuracy(outcomes, "a", "vote5") == pytest.approx(five, abs=0.06)
    assert accuracy(outcomes, "a", "vote3") == pytest.approx(3 * p**2 - 2 * p**3, abs=0.06)
    # And exactly: the vote is the majority of its own members.
    assert accuracy(outcomes, "a", "vote5") == sum(sum(m) >= 3 for m in members) / len(members)
    assert accuracy(outcomes, "a", "vote5") > accuracy(outcomes, "a", "single")


def test_a_fully_herding_debate_cannot_overturn_a_unanimous_first_round() -> None:
    # With full herding each agent copies the other two when they agree.
    backend = FakeBackend(TRUTH, {"a": 0.7}, herding=1.0)
    outcomes = run_experiment(QUESTIONS, ["a"], backend, PROTO)
    unanimous = 0
    for q in QUESTIONS:
        first = [
            parse_answer(backend.complete(_answer_request(q, ("a", i), PROTO)).text).verdict
            for i in range(3)
        ]
        if len(set(first)) == 1:
            unanimous += 1
            assert outcomes[q.qid, "a", "debate"].verdict == first[0]
            assert outcomes[q.qid, "a", "vote3"].verdict == first[0]
    assert unanimous > 50


def test_shared_calls_run_once_and_the_log_is_a_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    backend = FakeBackend(TRUTH, {"a": 0.7, "b": 0.6})
    first = run_experiment(QUESTIONS[:10], ["a", "b"], backend, PROTO, CallLog(path))
    # Per question: 2 x (closed book, single, 6 samples, bull, bear, judge, 3 revisions)
    # = 28, plus 2 long answers, 6 mixed revisions and 4 mixed judges = 40, plus
    # 2 x (6 five-line samples and 3 five-line revisions) = 18 for the follow-up.
    assert backend.calls == 10 * 58
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == backend.calls
    record = json.loads(lines[0])
    assert set(record) == {"key", "request", "completion"}
    assert record["request"]["prompt"] and "text" in record["completion"]

    resumed_backend = FakeBackend(TRUTH, {"a": 0.7, "b": 0.6})
    resumed = run_experiment(QUESTIONS[:10], ["a", "b"], resumed_backend, PROTO, CallLog(path))
    assert resumed_backend.calls == 0
    assert resumed == first


def test_request_key_depends_on_every_field() -> None:
    base = Request("m", "p", 10, 0.7, 1)
    variants = [
        Request("n", "p", 10, 0.7, 1),
        Request("m", "q", 10, 0.7, 1),
        Request("m", "p", 11, 0.7, 1),
        Request("m", "p", 10, 0.0, 1),
        Request("m", "p", 10, 0.7, 2),
    ]
    assert len({base.key(), *(v.key() for v in variants)}) == 6
    assert base.key() == Request("m", "p", 10, 0.7, 1).key()


def test_local_backend_refuses_remote_hosts() -> None:
    OllamaBackend("http://127.0.0.1:11434")
    OllamaBackend("http://localhost:11434")
    with pytest.raises(ValueError):
        OllamaBackend("https://api.example.com")


def test_fake_backend_respects_max_tokens() -> None:
    backend = FakeBackend(TRUTH, {"a": 1.0}, filler_words=100)
    out = backend.complete(Request("a", "Claim: x", 7, 0.0, 0))
    assert isinstance(out, Completion)
    assert out.completion_tokens == 7 and out.truncated


def test_report_is_identical_when_built_twice(tmp_path: Path) -> None:
    backend = FakeBackend(TRUTH, {"a": 0.75, "b": 0.65}, herding=0.5)
    outcomes = run_experiment(QUESTIONS[:60], ["a", "b"], backend, PROTO)
    folders = []
    for name in ("one", "two"):
        folder = tmp_path / name
        write_outcomes(folder / "outcomes.csv", QUESTIONS[:60], outcomes)
        build_report(folder)
        folders.append(folder)
    for file in sorted(p.name for p in folders[0].iterdir() if p.suffix in {".csv", ".md"}):
        assert (folders[0] / file).read_bytes() == (folders[1] / file).read_bytes(), file
    assert (folders[0] / "fig_headline.png").stat().st_size > 5_000

    rows = read_outcomes(folders[0] / "outcomes.csv")
    assert len(rows) == 60 * (11 + 11 + 9)
    summary = (folders[0] / "summary.md").read_text(encoding="utf-8")
    assert "debate" in summary and "mixed" in summary
