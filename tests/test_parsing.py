"""Scoring is deterministic for a fixed set of responses."""

import pytest

from agon.parsing import Answer, parse_answer
from agon.systems import aggregate, member_p


def test_tagged_answer_and_confidence() -> None:
    assert parse_answer("18.2% > 15%\nANSWER: TRUE\nCONFIDENCE: 90") == Answer(True, 0.9)
    a = parse_answer("ANSWER: FALSE\nCONFIDENCE: 80%")
    assert a.verdict is False
    assert a.p_true == pytest.approx(0.2)


def test_last_tag_wins() -> None:
    assert parse_answer("ANSWER: TRUE ... on reflection\nANSWER: FALSE").verdict is False


def test_markdown_and_case_are_tolerated() -> None:
    assert parse_answer("**Answer:** true\n**Confidence:** 95").verdict is True


def test_bare_word_only_when_unambiguous() -> None:
    assert parse_answer("TRUE\nThe figure is higher.\nCONFIDENCE: 95") == Answer(True, 0.95)
    assert parse_answer("It could be TRUE or FALSE").verdict is None
    assert parse_answer("no verdict here") == Answer(None, None)


def test_confidence_is_clamped_and_fractions_are_read() -> None:
    assert parse_answer("ANSWER: TRUE\nCONFIDENCE: 30").p_true == 0.5
    assert parse_answer("ANSWER: TRUE\nCONFIDENCE: 250").p_true == 1.0
    assert parse_answer("ANSWER: TRUE\nCONFIDENCE: 0.8").p_true == pytest.approx(0.8)


def test_missing_confidence_gets_the_default() -> None:
    assert member_p(parse_answer("ANSWER: TRUE")) == 0.75
    assert member_p(parse_answer("ANSWER: FALSE")) == 0.25
    assert member_p(parse_answer("nothing")) is None


def test_majority_vote() -> None:
    t, f, none = Answer(True, 0.9), Answer(False, 0.2), Answer(None, None)
    assert aggregate([t, t, f]) == (True, pytest.approx((0.9 + 0.9 + 0.2) / 3))
    assert aggregate([t, f, f, none])[0] is False
    assert aggregate([none, none]) == (None, 0.5)


def test_ties_go_to_the_mean_probability() -> None:
    assert aggregate([Answer(True, 0.9), Answer(False, 0.4)])[0] is True
    assert aggregate([Answer(True, 0.6), Answer(False, 0.1)])[0] is False
    assert aggregate([Answer(True, 0.75), Answer(False, 0.25)]) == (None, 0.5)


def test_parsing_is_deterministic() -> None:
    text = "Gross profit 40 / revenue 100 = 40%\nANSWER: TRUE\nCONFIDENCE: 85"
    assert len({parse_answer(text) for _ in range(20)}) == 1
