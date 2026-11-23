"""Turn SEC XBRL "frames" into one small financial table per filing.

A frame is the SEC's list of every company's reported value for one tagged
concept in one period (https://data.sec.gov/api/xbrl/frames/). Each value
carries the accession number of the filing it came from. This module joins a
set of frames by company and keeps, for each company, only the values that come
from one single filing: the quarterly report for the period ended
30 September 2026, with its own prior-period comparatives.

Amounts are shown in thousands of US dollars, rounded, the way most filings
print them. Every claim is scored against these displayed numbers.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Periods. Duration frames hold three-month values, instant frames hold balances.
CURRENT_DURATION = "CY2026Q3"
PRIOR_DURATION = "CY2025Q3"
CURRENT_INSTANT = "CY2026Q3I"
PRIOR_INSTANT = "CY2025Q4I"

CURRENT_START, CURRENT_END = "2026-07-01", "2026-09-30"
PRIOR_START, PRIOR_END = "2025-07-01", "2025-09-30"
PRIOR_BALANCE_DATE = "2025-12-31"

# concept -> plain label, in the order the rows are printed.
INCOME_CONCEPTS: dict[str, str] = {
    "Revenues": "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue from contracts with customers",
    "CostOfRevenue": "Cost of revenue",
    "CostOfGoodsAndServicesSold": "Cost of goods and services sold",
    "GrossProfit": "Gross profit",
    "ResearchAndDevelopmentExpense": "Research and development expense",
    "SellingGeneralAndAdministrativeExpense": "Selling, general and administrative expense",
    "OperatingExpenses": "Total operating expenses",
    "OperatingIncomeLoss": "Operating income (loss)",
    "InterestExpense": "Interest expense",
    "IncomeTaxExpenseBenefit": "Income tax expense (benefit)",
    "NetIncomeLoss": "Net income (loss)",
}
BALANCE_CONCEPTS: dict[str, str] = {
    "CashAndCashEquivalentsAtCarryingValue": "Cash and cash equivalents",
    "AccountsReceivableNetCurrent": "Accounts receivable, net",
    "InventoryNet": "Inventory, net",
    "AssetsCurrent": "Total current assets",
    "PropertyPlantAndEquipmentNet": "Property, plant and equipment, net",
    "Goodwill": "Goodwill",
    "Assets": "Total assets",
    "AccountsPayableCurrent": "Accounts payable",
    "LiabilitiesCurrent": "Total current liabilities",
    "LongTermDebtNoncurrent": "Long-term debt, noncurrent",
    "Liabilities": "Total liabilities",
    "RetainedEarningsAccumulatedDeficit": "Retained earnings (accumulated deficit)",
    "StockholdersEquity": "Total stockholders' equity",
}
# When both concepts of a pair are reported with the same numbers, the second is dropped.
_DUPLICATES = (
    ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
)

MIN_INCOME_ROWS = 5
MIN_BALANCE_ROWS = 6


def frame_names() -> list[tuple[str, str]]:
    """Every (concept, period) frame the question set needs."""
    names: list[tuple[str, str]] = []
    for concept in INCOME_CONCEPTS:
        names += [(concept, CURRENT_DURATION), (concept, PRIOR_DURATION)]
    for concept in BALANCE_CONCEPTS:
        names += [(concept, CURRENT_INSTANT), (concept, PRIOR_INSTANT)]
    return names


@dataclass(frozen=True)
class Row:
    """One line of a statement, in thousands of US dollars."""

    concept: str
    label: str
    current: int
    prior: int


@dataclass(frozen=True)
class Statement:
    """The figures taken from one filing."""

    cik: int
    name: str
    accession: str
    income: tuple[Row, ...]
    balance: tuple[Row, ...]

    def row(self, concept: str) -> Row | None:
        for r in self.income + self.balance:
            if r.concept == concept:
                return r
        return None


def to_thousands(value: float) -> int:
    """Round a dollar amount to thousands, halves away from zero."""
    sign = -1 if value < 0 else 1
    return sign * int((abs(value) + 500) // 1000)


def fmt(value: int) -> str:
    """Print the way a filing does: commas, negatives in brackets."""
    return f"({abs(value):,})" if value < 0 else f"{value:,}"


Fact = Mapping[str, object]
Frames = Mapping[tuple[str, str], Sequence[Fact]]


def _index(facts: Sequence[Fact], start: str | None, end: str) -> dict[int, Fact]:
    out: dict[int, Fact] = {}
    for f in facts:
        if f.get("end") != end:
            continue
        if start is not None and f.get("start") != start:
            continue
        cik = f.get("cik")
        if isinstance(cik, int):
            out[cik] = f
    return out


def build_statements(frames: Frames) -> list[Statement]:
    """Join frames into statements, one per company, sorted by CIK.

    A row is kept only if its current and prior values both come from the same
    filing, and that filing is the one most of the company's current values
    come from. A company is kept only if enough rows survive.
    """
    cur: dict[str, dict[int, Fact]] = {}
    pri: dict[str, dict[int, Fact]] = {}
    for concept in INCOME_CONCEPTS:
        cur[concept] = _index(
            frames.get((concept, CURRENT_DURATION), []), CURRENT_START, CURRENT_END
        )
        pri[concept] = _index(frames.get((concept, PRIOR_DURATION), []), PRIOR_START, PRIOR_END)
    for concept in BALANCE_CONCEPTS:
        cur[concept] = _index(frames.get((concept, CURRENT_INSTANT), []), None, CURRENT_END)
        pri[concept] = _index(frames.get((concept, PRIOR_INSTANT), []), None, PRIOR_BALANCE_DATE)

    ciks = sorted({cik for by_cik in cur.values() for cik in by_cik})
    statements: list[Statement] = []
    for cik in ciks:
        accessions = Counter(
            str(by_cik[cik].get("accn")) for by_cik in cur.values() if cik in by_cik
        )
        accession = sorted(accessions.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        name = ""
        rows: dict[str, Row] = {}
        for concept, label in (INCOME_CONCEPTS | BALANCE_CONCEPTS).items():
            c, p = cur[concept].get(cik), pri[concept].get(cik)
            if c is None or p is None:
                continue
            if c.get("accn") != accession or p.get("accn") != accession:
                continue
            cv, pv = c.get("val"), p.get("val")
            if not isinstance(cv, int | float) or not isinstance(pv, int | float):
                continue
            name = name or str(c.get("entityName", ""))
            rows[concept] = Row(concept, label, to_thousands(cv), to_thousands(pv))
        for first, second in _DUPLICATES:
            a, b = rows.get(first), rows.get(second)
            if a and b and (a.current, a.prior) == (b.current, b.prior):
                del rows[second]
        income = tuple(rows[c] for c in INCOME_CONCEPTS if c in rows)
        balance = tuple(rows[c] for c in BALANCE_CONCEPTS if c in rows)
        if len(income) >= MIN_INCOME_ROWS and len(balance) >= MIN_BALANCE_ROWS:
            statements.append(Statement(cik, name, accession, income, balance))
    return statements


def render(statement: Statement) -> str:
    """The text a model is shown for a filing."""
    lines = [
        f"Company: {statement.name}",
        "Source: quarterly filing with the SEC for the period ended September 30, 2026 "
        f"(accession {statement.accession}).",
        "All amounts are in thousands of US dollars. Brackets mean a negative number.",
        "",
        "Income statement, three months ended September 30",
        "Item | 2026 | 2025",
    ]
    lines += [f"{r.label} | {fmt(r.current)} | {fmt(r.prior)}" for r in statement.income]
    lines += ["", "Balance sheet", "Item | September 30, 2026 | December 31, 2025"]
    lines += [f"{r.label} | {fmt(r.current)} | {fmt(r.prior)}" for r in statement.balance]
    return "\n".join(lines)


def synthetic_statements(n: int, seed: int) -> list[Statement]:
    """Made-up companies with consistent numbers, for tests and the fast mode."""
    rng = random.Random(seed)
    out: list[Statement] = []
    for i in range(n):

        def pair(base: float, rng: random.Random = rng) -> tuple[int, int]:
            return int(base), int(base * rng.uniform(0.7, 1.3))

        revenue = rng.uniform(20_000, 900_000)
        cost = revenue * rng.uniform(0.35, 0.8)
        sga = revenue * rng.uniform(0.08, 0.3)
        rnd = revenue * rng.uniform(0.02, 0.15)
        assets = revenue * rng.uniform(2.0, 6.0)
        current_assets = assets * rng.uniform(0.25, 0.6)
        liabilities = assets * rng.uniform(0.3, 0.8)
        values = {
            "Revenues": pair(revenue),
            "CostOfRevenue": pair(cost),
            "GrossProfit": pair(revenue - cost),
            "ResearchAndDevelopmentExpense": pair(rnd),
            "SellingGeneralAndAdministrativeExpense": pair(sga),
            "OperatingExpenses": pair(sga + rnd),
            "OperatingIncomeLoss": pair(revenue - cost - sga - rnd),
            "CashAndCashEquivalentsAtCarryingValue": pair(current_assets * rng.uniform(0.2, 0.6)),
            "AccountsReceivableNetCurrent": pair(current_assets * rng.uniform(0.1, 0.3)),
            "AssetsCurrent": pair(current_assets),
            "Goodwill": pair(assets * rng.uniform(0.05, 0.3)),
            "Assets": pair(assets),
            "LiabilitiesCurrent": pair(liabilities * rng.uniform(0.3, 0.6)),
            "Liabilities": pair(liabilities),
            "StockholdersEquity": pair(assets - liabilities),
        }
        labels = INCOME_CONCEPTS | BALANCE_CONCEPTS
        rows = {c: Row(c, labels[c], v[0], v[1]) for c, v in values.items()}
        out.append(
            Statement(
                cik=900_000 + i,
                name=f"SYNTHETIC CO {i:03d}",
                accession=f"0000000000-26-{i:06d}",
                income=tuple(rows[c] for c in INCOME_CONCEPTS if c in rows),
                balance=tuple(rows[c] for c in BALANCE_CONCEPTS if c in rows),
            )
        )
    return out
