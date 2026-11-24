"""Token budgets are enforced, not assumed."""

import pytest

from agon.backends import FakeBackend, RunawayBackend, fake_key
from agon.budget import BudgetError, Ledger
from agon.filings import synthetic_statements
from agon.questions import build_questions
from agon.report import base_system
from agon.runner import run_experiment, run_jobs
from agon.systems import Protocol, _answer_request, _ask, debate, judged, lineup, vote

PROTO = Protocol(cap=20)
QUESTIONS = build_questions(synthetic_statements(8, 1), 1)
TRUTH = {fake_key(q.company, q.claim): q.truth for q in QUESTIONS}


def test_ledger_refuses_a_call_larger_than_what_is_left() -> None:
    ledger = Ledger(100)
    ledger.reserve(60)
    with pytest.raises(BudgetError):
        ledger.reserve(41)
    ledger.settle(60, 10, 5)
    ledger.reserve(90)
    assert ledger.remaining == 0


def test_ledger_refuses_a_backend_that_overruns_its_cap() -> None:
    ledger = Ledger(100)
    ledger.reserve(50)
    with pytest.raises(BudgetError):
        ledger.settle(50, 51, 0)


def test_ledger_refuses_an_unreserved_settlement() -> None:
    with pytest.raises(BudgetError):
        Ledger(100).settle(10, 5, 0)


def test_runner_stops_a_backend_that_ignores_max_tokens() -> None:
    with pytest.raises(BudgetError):
        run_experiment(QUESTIONS[:1], ["a"], RunawayBackend(), PROTO)


def test_every_system_stays_inside_its_budget() -> None:
    # filler_words is far above the cap, so every reply is cut at the cap.
    backend = FakeBackend(TRUTH, {"a": 0.8, "b": 0.6}, filler_words=500)
    outcomes = run_experiment(QUESTIONS, ["a", "b"], backend, PROTO)
    units = {"single": 1, "closed_book": 1, "long": 2, "vote3": 3, "vote5": 5, "vote6": 6}
    units |= {"vote3_long": 6, "vote6_long": 12, "debate_long": 12}
    for (_, _, system), o in outcomes.items():
        assert o.budget == units.get(base_system(system), 6) * PROTO.cap
        assert o.completion_tokens == o.budget  # the fake fills every cap here
        assert o.completion_tokens <= 2 * PROTO.budget
        assert o.truncated == o.calls


def test_debate_judge_and_vote6_share_one_budget() -> None:
    backend = FakeBackend(TRUTH, {"a": 0.8})
    q = QUESTIONS[0]
    jobs = {
        ("q", "a", "vote6"): vote(q, [("a", i) for i in range(6)], PROTO),
        ("q", "a", "judge"): judged(q, "a", "a", "a", PROTO),
        ("q", "a", "debate"): debate(q, [("a", i) for i in range(3)], PROTO),
    }
    outcomes = run_jobs(jobs, backend)
    assert {o.budget for o in outcomes.values()} == {PROTO.budget}
    assert [o.calls for o in outcomes.values()] == [6, 3, 6]


def test_five_line_vote6_and_debate_share_a_doubled_budget() -> None:
    backend = FakeBackend(TRUTH, {"a": 0.8})
    q = QUESTIONS[0]
    jobs = {
        ("q", "a", "vote6_long"): vote(q, [("a", i) for i in range(6)], PROTO, long=True),
        ("q", "a", "debate_long"): debate(q, [("a", i) for i in range(3)], PROTO, long=True),
    }
    outcomes = run_jobs(jobs, backend)
    assert {o.budget for o in outcomes.values()} == {2 * PROTO.budget}
    assert [o.calls for o in outcomes.values()] == [6, 6]


def test_a_system_that_overspends_is_stopped() -> None:
    # A vote of seven does not fit a ledger sized for six answers.
    ledger = Ledger(PROTO.budget)
    requests = [_answer_request(QUESTIONS[0], ("a", i), PROTO) for i in range(7)]
    with pytest.raises(BudgetError):
        next(_ask(ledger, requests))


def test_lineup_has_the_expected_jobs() -> None:
    jobs = lineup(QUESTIONS[0], ["a", "b"], PROTO)
    assert sum(1 for c, _ in jobs if c == "a") == 11
    assert sum(1 for c, _ in jobs if c == "mixed") == 9
