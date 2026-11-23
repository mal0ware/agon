"""The question set: joins, rounding, and truth values that follow from the table."""

import pytest

from agon.filings import (
    BALANCE_CONCEPTS,
    CURRENT_DURATION,
    CURRENT_INSTANT,
    INCOME_CONCEPTS,
    PRIOR_DURATION,
    PRIOR_INSTANT,
    Row,
    build_statements,
    fmt,
    frame_names,
    render,
    synthetic_statements,
    to_thousands,
)
from agon.questions import (
    TEMPLATES,
    build_questions,
    evaluate,
    load_questions,
    order_statements,
    quantity,
    save_questions,
)


def _frames(accession_prior: str = "A-1") -> dict[tuple[str, str], list[dict[str, object]]]:
    """One company, cik 7, reporting every concept from one filing."""
    frames: dict[tuple[str, str], list[dict[str, object]]] = {}
    base = {"cik": 7, "entityName": "TEST CO"}
    for i, concept in enumerate(INCOME_CONCEPTS):
        frames[concept, CURRENT_DURATION] = [
            {**base, "accn": "A-1", "start": "2026-07-01", "end": "2026-09-30",
             "val": 1_000_000 * (i + 2)}
        ]  # fmt: skip
        frames[concept, PRIOR_DURATION] = [
            {**base, "accn": accession_prior, "start": "2025-07-01", "end": "2025-09-30",
             "val": 900_000 * (i + 2)}
        ]  # fmt: skip
    for i, concept in enumerate(BALANCE_CONCEPTS):
        frames[concept, CURRENT_INSTANT] = [
            {**base, "accn": "A-1", "end": "2026-09-30", "val": 5_000_000 * (i + 2)}
        ]
        frames[concept, PRIOR_INSTANT] = [
            {**base, "accn": "A-1", "end": "2025-12-31", "val": 4_000_000 * (i + 2)}
        ]
    return frames


def test_rounding_to_thousands() -> None:
    assert to_thousands(58_013_000) == 58_013
    assert to_thousands(1_499) == 1
    assert to_thousands(1_500) == 2
    assert to_thousands(-1_500) == -2
    assert to_thousands(499) == 0


def test_filing_style_formatting() -> None:
    assert fmt(1234567) == "1,234,567"
    assert fmt(-3092) == "(3,092)"


def test_frame_list_covers_both_periods() -> None:
    names = frame_names()
    assert len(names) == 2 * (len(INCOME_CONCEPTS) + len(BALANCE_CONCEPTS))
    assert ("Revenues", CURRENT_DURATION) in names
    assert ("Assets", PRIOR_INSTANT) in names


def test_statement_is_joined_from_one_filing() -> None:
    (s,) = build_statements(_frames())
    assert (s.cik, s.name, s.accession) == (7, "TEST CO", "A-1")
    revenue = s.row("Revenues")
    assert revenue == Row("Revenues", "Revenues", 2000, 1800)
    assert len(s.balance) == len(BALANCE_CONCEPTS)


def test_rows_from_another_filing_are_dropped() -> None:
    # Prior-period income values tagged to a different filing: no income rows survive.
    assert build_statements(_frames(accession_prior="B-2")) == []


def test_off_calendar_periods_are_dropped() -> None:
    frames = _frames()
    for (_, period), facts in frames.items():
        if period == CURRENT_INSTANT:
            facts[0]["end"] = "2026-08-31"
    assert build_statements(frames) == []


def test_render_shows_every_row() -> None:
    (s,) = build_statements(_frames())
    text = render(s)
    assert "Revenues | 2,000 | 1,800" in text
    assert "Total assets | " in text
    assert "A-1" in text


def test_evaluate_matches_hand_arithmetic() -> None:
    a = Row("Revenues", "Revenues", 120, 100)
    b = Row("Assets", "Total assets", 400, 500)
    c = Row("Liabilities", "Total liabilities", 200, 250)
    assert quantity("change", a, None, None) == pytest.approx(20.0)
    assert evaluate("change", a, None, None, "higher", 19.9) is True
    assert evaluate("change", a, None, None, "higher", 20.1) is False
    assert evaluate("change", a, None, None, "lower", 5.0) is False
    assert evaluate("change", b, None, None, "lower", 19.9) is True  # 400 against 500
    assert evaluate("change", b, None, None, "lower", 20.1) is False
    assert quantity("share", a, b, None) == pytest.approx(30.0)
    assert evaluate("share", a, b, None, "", 29.9) is True
    assert evaluate("share", a, b, None, "", 30.1) is False
    assert evaluate("difference", b, a, None, "", 279) is True
    assert evaluate("difference", b, a, None, "", 280) is False
    # (400 - 120) / 200 = 140%
    assert quantity("spread", b, a, c) == pytest.approx(140.0)
    assert evaluate("spread", b, a, c, "", 139.0) is True
    assert evaluate("spread", b, a, c, "", 141.0) is False
    # 280 now against 400 then: down 30%
    assert quantity("spread_change", b, a, None) == pytest.approx(-30.0)
    assert evaluate("spread_change", b, a, None, "lower", 29.0) is True
    assert evaluate("spread_change", b, a, None, "higher", 1.0) is False
    # 30% now against 20% then: the ratio rose by half
    assert quantity("shift", a, b, None) == pytest.approx(50.0)
    assert evaluate("shift", a, b, None, "higher", 0.0) is True
    assert evaluate("shift", a, b, None, "lower", 0.0) is False
    assert evaluate("change", Row("x", "x", 5, -5), None, None, "higher", 1.0) is None


def test_questions_are_balanced_true_to_the_table_and_reproducible() -> None:
    statements = synthetic_statements(120, 11)
    questions = build_questions(statements, 11)
    assert len(questions) == 120
    assert sum(q.truth for q in questions) == 60
    assert {q.template for q in questions} == set(TEMPLATES)
    by_cik = {s.cik: s for s in statements}
    for q in questions:
        s = by_cik[q.cik]
        a = s.row(q.a)
        assert a is not None
        b = s.row(q.b) if q.b else None
        c = s.row(q.c) if q.c else None
        assert evaluate(q.template, a, b, c, q.direction, q.threshold) is q.truth
        value = quantity(q.template, a, b, c)
        assert value is not None
        if q.template != "shift":
            # The threshold is close to the real value, on the side that makes the truth.
            assert abs(q.threshold - abs(value)) / abs(value) <= 0.05
            assert 0.004 <= q.margin <= 0.05
        assert s.name in q.context
    assert questions == build_questions(statements, 11)
    assert questions != build_questions(statements, 12)


def test_pilot_and_evaluation_slices_do_not_overlap() -> None:
    ordered = order_statements(synthetic_statements(50, 2), 5)
    assert ordered == order_statements(synthetic_statements(50, 2), 5)
    pilot, evaluation = ordered[:10], ordered[10:]
    assert not {s.cik for s in pilot} & {s.cik for s in evaluation}


def test_questions_round_trip_through_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    questions = build_questions(synthetic_statements(5, 3), 3)
    save_questions(questions, tmp_path / "q.json")
    assert load_questions(tmp_path / "q.json") == questions
