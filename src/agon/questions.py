"""Build true-or-false claims about a filing, each with a checkable answer.

Every claim is a statement about numbers printed in the filing table, with a
threshold set close to the real value. Whether it is true follows from
arithmetic on the table and nothing else. Half the claims are true and half
are false.

Six templates, two of them needing two steps of arithmetic:

``change``         X was more than t% higher (or lower) than in the prior period.
``share``          A was more than t% of B.
``difference``     A exceeded B by more than N thousand dollars.
``spread``         A minus B, as a percentage of C, was more than t%.
``spread_change``  A minus B was more than t% higher (or lower) than in the prior period.
``shift``          A as a percentage of B was higher (or lower) than in the prior period.

The threshold sits a relative ``margin`` of one, two or four percent away from
the real value, so a model has to do the sum to about that precision. ``shift``
has no threshold; its ``margin`` records how far the ratio moved.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from agon.filings import INCOME_CONCEPTS, Row, Statement, render

TEMPLATES = ("change", "share", "difference", "spread", "spread_change", "shift")
# How far the threshold sits from the real value, as a fraction of the value.
MARGINS = (0.01, 0.02, 0.04)
# Allowed size of the move in a shift claim, in percent of the earlier ratio.
SHIFT_GAP = (0.5, 8.0)

# (part, whole) pairs for the share and shift templates.
SHARE_PAIRS = (
    ("GrossProfit", "Revenues"),
    ("CostOfRevenue", "Revenues"),
    ("SellingGeneralAndAdministrativeExpense", "Revenues"),
    ("ResearchAndDevelopmentExpense", "Revenues"),
    ("OperatingExpenses", "Revenues"),
    ("OperatingIncomeLoss", "Revenues"),
    ("NetIncomeLoss", "Revenues"),
    ("InterestExpense", "OperatingIncomeLoss"),
    ("CashAndCashEquivalentsAtCarryingValue", "Assets"),
    ("AssetsCurrent", "Assets"),
    ("Goodwill", "Assets"),
    ("PropertyPlantAndEquipmentNet", "Assets"),
    ("Liabilities", "Assets"),
    ("StockholdersEquity", "Assets"),
    ("LiabilitiesCurrent", "Liabilities"),
    ("LongTermDebtNoncurrent", "Liabilities"),
    ("AccountsReceivableNetCurrent", "AssetsCurrent"),
    ("InventoryNet", "AssetsCurrent"),
    ("CashAndCashEquivalentsAtCarryingValue", "LiabilitiesCurrent"),
    ("AccountsPayableCurrent", "LiabilitiesCurrent"),
)
# (larger, smaller) pairs for the difference template.
DIFFERENCE_PAIRS = (
    ("Assets", "Liabilities"),
    ("AssetsCurrent", "LiabilitiesCurrent"),
    ("Revenues", "CostOfRevenue"),
    ("Revenues", "OperatingExpenses"),
    ("GrossProfit", "SellingGeneralAndAdministrativeExpense"),
    ("Assets", "AssetsCurrent"),
    ("Liabilities", "LiabilitiesCurrent"),
    ("AssetsCurrent", "CashAndCashEquivalentsAtCarryingValue"),
    ("StockholdersEquity", "Goodwill"),
)


# (A, B, C) triples for the spread template: (A minus B) as a percentage of C.
SPREAD_TRIPLES = (
    ("Revenues", "CostOfRevenue", "Revenues"),
    ("Revenues", "OperatingExpenses", "Revenues"),
    ("GrossProfit", "SellingGeneralAndAdministrativeExpense", "Revenues"),
    ("GrossProfit", "OperatingExpenses", "Revenues"),
    ("Assets", "Liabilities", "Assets"),
    ("AssetsCurrent", "LiabilitiesCurrent", "Assets"),
    ("AssetsCurrent", "LiabilitiesCurrent", "Liabilities"),
    ("AssetsCurrent", "InventoryNet", "LiabilitiesCurrent"),
    ("Assets", "Goodwill", "Liabilities"),
    ("Liabilities", "LiabilitiesCurrent", "Assets"),
    ("Assets", "AssetsCurrent", "Liabilities"),
)


@dataclass(frozen=True)
class Question:
    """A claim, its truth, and the filing text it is checked against."""

    qid: str
    cik: int
    company: str
    accession: str
    template: str
    a: str
    b: str
    c: str
    direction: str
    threshold: float
    margin: float
    claim: str
    truth: bool
    context: str


def _is_income(concept: str) -> bool:
    return concept in INCOME_CONCEPTS


def _when(concept: str, current: bool) -> str:
    if _is_income(concept):
        year = "2026" if current else "2025"
        return f"for the three months ended September 30, {year}"
    return "at September 30, 2026" if current else "at December 31, 2025"


def _lower(label: str) -> str:
    return label[0].lower() + label[1:]


def quantity(template: str, a: Row, b: Row | None, c: Row | None) -> float | None:
    """The number a claim is about, from the displayed figures. None if undefined.

    ``change``        percent change in A from the prior period
    ``share``         A as a percent of B
    ``difference``    A minus B, in thousands of dollars
    ``spread``        (A minus B) as a percent of C
    ``spread_change`` percent change in (A minus B) from the prior period
    ``shift``         A/B now divided by A/B in the prior period, minus one, in percent
    """
    if template == "change":
        if a.prior <= 0 or a.current <= 0:
            return None
        return (a.current - a.prior) / a.prior * 100.0
    if b is None:
        return None
    if template == "share":
        if b.current <= 0 or a.current <= 0:
            return None
        return a.current / b.current * 100.0
    if template == "difference":
        return float(a.current - b.current)
    if template == "spread":
        if c is None or c.current <= 0:
            return None
        return (a.current - b.current) / c.current * 100.0
    if template == "spread_change":
        now, then = a.current - b.current, a.prior - b.prior
        if now <= 0 or then <= 0:
            return None
        return (now - then) / then * 100.0
    if template == "shift":
        if min(a.current, a.prior, b.current, b.prior) <= 0:
            return None
        return ((a.current / b.current) / (a.prior / b.prior) - 1.0) * 100.0
    raise ValueError(f"unknown template {template!r}")


def evaluate(
    template: str, a: Row, b: Row | None, c: Row | None, direction: str, threshold: float
) -> bool | None:
    """Truth of a claim from the displayed numbers. None if it cannot be computed.

    ``direction`` is "higher", "lower" or empty. A "lower" claim is about a fall,
    so the quantity is negated before it is compared with the threshold.
    """
    value = quantity(template, a, b, c)
    if value is None:
        return None
    return (-value if direction == "lower" else value) > threshold


def _threshold(value: float, truth: bool, margin: float) -> float:
    """A threshold just below ``value`` for a true claim, just above for a false one.

    Percentages keep one decimal, or two when the value is under ten.
    """
    raw = value * (1 - margin) if truth else value * (1 + margin)
    return round(raw, 2 if abs(value) < 10 else 1)


def _dollars(value: float, truth: bool, margin: float) -> float:
    """The same for a dollar amount, kept to four significant figures."""
    raw = value * (1 - margin) if truth else value * (1 + margin)
    return float(int(float(f"{raw:.4g}")))


Made = tuple[str, str, str, str, float, float, str]  # a, b, c, direction, threshold, margin, text


def _try(
    template: str, s: Statement, truth: bool, margin: float, rng: random.Random
) -> Made | None:
    """One claim of the given template, or None if the filing cannot support it."""

    def settle(
        a: Row, b: Row | None, c: Row | None, direction: str, size: float, text: str
    ) -> Made | None:
        """Pick the threshold, check the claim has the wanted truth, fill in the text."""
        if template == "difference":
            t = _dollars(size, truth, margin)
        else:
            t = _threshold(size, truth, margin)
        if t <= 0 or abs(t - size) / size < margin / 2:
            return None
        if evaluate(template, a, b, c, direction, t) is not truth:
            return None
        shown = f"{int(t):,}" if template == "difference" else f"{t:g}"
        names = (a.concept, b.concept if b else "", c.concept if c else "")
        return (*names, direction, t, round(abs(t - size) / size, 4), text.format(t=shown))

    if template == "change":
        rows = [r for r in s.income + s.balance if r.current > 0 and r.prior > 0]
        rng.shuffle(rows)
        for r in rows:
            growth = quantity(template, r, None, None)
            if growth is None or not 2.0 <= abs(growth) <= 150.0:
                continue
            direction = "higher" if growth > 0 else "lower"
            text = (
                f"{r.label} {_when(r.concept, True)} was more than {{t}}% {direction} "
                f"than {_when(r.concept, False)}."
            )
            made = settle(r, None, None, direction, abs(growth), text)
            if made:
                return made
        return None

    if template == "spread":
        triples = list(SPREAD_TRIPLES)
        rng.shuffle(triples)
        for ca, cb, cc in triples:
            a, b, c = s.row(ca), s.row(cb), s.row(cc)
            if a is None or b is None or c is None:
                continue
            value = quantity(template, a, b, c)
            if value is None or not 2.0 <= value <= 300.0 or b.current <= 0:
                continue
            text = (
                f"{a.label} minus {_lower(b.label)}, as a percentage of {_lower(c.label)}, "
                f"was more than {{t}}% {_when(ca, True)}."
            )
            made = settle(a, b, c, "", value, text)
            if made:
                return made
        return None

    pairs = list(SHARE_PAIRS if template in ("share", "shift") else DIFFERENCE_PAIRS)
    rng.shuffle(pairs)
    for ca, cb in pairs:
        a, b = s.row(ca), s.row(cb)
        if a is None or b is None:
            continue
        value = quantity(template, a, b, None)
        if value is None:
            continue
        claim: Made | None = None
        if template == "share":
            if not 2.0 <= value <= 98.0:
                continue
            text = f"{a.label} was more than {{t}}% of {_lower(b.label)} {_when(ca, True)}."
            claim = settle(a, b, None, "", value, text)
        elif template == "difference":
            if value < 1000 or b.current <= 0:
                continue
            text = (
                f"{a.label} exceeded {_lower(b.label)} by more than {{t}} thousand dollars "
                f"{_when(ca, True)}."
            )
            claim = settle(a, b, None, "", value, text)
        elif template == "spread_change":
            if not 2.0 <= abs(value) <= 150.0 or min(b.current, b.prior) <= 0:
                continue
            direction = "higher" if value > 0 else "lower"
            text = (
                f"{a.label} minus {_lower(b.label)} was more than {{t}}% {direction} "
                f"{_when(ca, True)} than {_when(ca, False)}."
            )
            claim = settle(a, b, None, direction, abs(value), text)
        else:  # shift: no threshold, the claim is only about the direction
            if not SHIFT_GAP[0] <= abs(value) <= SHIFT_GAP[1]:
                continue
            if not 2.0 <= a.current / b.current * 100.0 <= 98.0:
                continue
            rose = value > 0
            direction = (
                ("higher" if rose else "lower") if truth else ("lower" if rose else "higher")
            )
            if _is_income(ca):
                span = (
                    "in the three months ended September 30, 2026 than in the three months "
                    "ended September 30, 2025"
                )
            else:
                span = "at September 30, 2026 than at December 31, 2025"
            text = f"{a.label} as a percentage of {_lower(b.label)} was {direction} {span}."
            if evaluate(template, a, b, None, direction, 0.0) is not truth:
                continue
            claim = (ca, cb, "", direction, 0.0, round(abs(value) / 100.0, 4), text)
        if claim:
            return claim
    return None


def order_statements(statements: Sequence[Statement], seed: int) -> list[Statement]:
    """A fixed shuffle of the statements. Pilot and evaluation sets are slices of it."""
    ordered = sorted(statements, key=lambda s: s.cik)
    random.Random(seed).shuffle(ordered)
    return ordered


def build_questions(statements: Sequence[Statement], seed: int) -> list[Question]:
    """One claim per statement, templates rotated, truth values balanced."""
    n = len(statements)
    truths = [i % 2 == 0 for i in range(n)]
    random.Random(seed + 1).shuffle(truths)
    questions: list[Question] = []
    for i, s in enumerate(statements):
        rng = random.Random(f"{seed}-{s.cik}")
        margin = rng.choice(MARGINS)
        for k in range(len(TEMPLATES)):
            template = TEMPLATES[(i + k) % len(TEMPLATES)]
            made = _try(template, s, truths[i], margin, rng)
            if made is None:
                continue
            a, b, c, direction, threshold, used_margin, claim = made
            questions.append(
                Question(
                    qid=f"{s.cik}-{template}",
                    cik=s.cik,
                    company=s.name,
                    accession=s.accession,
                    template=template,
                    a=a,
                    b=b,
                    c=c,
                    direction=direction,
                    threshold=threshold,
                    margin=used_margin,
                    claim=claim,
                    truth=truths[i],
                    context=render(s),
                )
            )
            break
    return questions


def save_questions(questions: Sequence[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(q) for q in questions], indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_questions(path: Path) -> list[Question]:
    return [Question(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
