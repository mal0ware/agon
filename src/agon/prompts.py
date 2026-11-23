"""Every prompt the systems send. The filing text always comes first."""

from __future__ import annotations

from collections.abc import Sequence

from agon.questions import Question

ANSWER_FORMAT = (
    "End with exactly these two lines:\n"
    "ANSWER: TRUE or FALSE\n"
    "CONFIDENCE: a number from 50 to 100, how sure you are of your answer"
)
BRIEF = (
    "Show only the calculation, in one or two short lines of plain text. "
    "Do not restate the claim or the data."
)
PEERS_HEADER = "These are the answers from other agents:"
PEERS_FOOTER = "Using the other agents' answers as additional advice"
BULL_ROLE = "You argue that the claim is TRUE."
BEAR_ROLE = "You argue that the claim is FALSE."
CLOSED_BOOK = "CLOSED BOOK. No filing data is provided."


def _head(q: Question) -> str:
    return f"{q.context}\n\nClaim: {q.claim}\n\n"


def solo(q: Question) -> str:
    """One agent answering on its own. Also round one of the debate."""
    return (
        _head(q) + "Decide whether the claim is true or false using only the filing data above. "
        f"{BRIEF}\n" + ANSWER_FORMAT
    )


def solo_long(q: Question) -> str:
    """One agent with the room an advocate or a judge gets: five lines, not two."""
    return (
        _head(q) + "Decide whether the claim is true or false using only the filing data above. "
        "Show the calculation step by step, in at most five short lines of plain text.\n"
        + ANSWER_FORMAT
    )


def closed_book(q: Question) -> str:
    """The recall control: the claim without the filing table."""
    return (
        f"{CLOSED_BOOK}\nCompany: {q.company}\n"
        "The claim is about its quarterly filing with the SEC for the period ended "
        "September 30, 2026. Amounts are in thousands of US dollars.\n\n"
        f"Claim: {q.claim}\n\n"
        "From what you know, decide whether the claim is true or false. "
        "Reply in one short line.\n" + ANSWER_FORMAT
    )


def advocate(q: Question, side: bool) -> str:
    """The bull (side True) or the bear (side False)."""
    role = BULL_ROLE if side else BEAR_ROLE
    return (
        _head(q) + f"{role} Make the strongest case you can from the filing data above, "
        "with the calculation, in at most five short lines. Do not give a verdict."
    )


def judge(q: Question, bull: str, bear: str) -> str:
    return (
        _head(q) + "Two advocates were each told which side to argue.\n\n"
        f"Argument that the claim is TRUE:\n{bull.strip()}\n\n"
        f"Argument that the claim is FALSE:\n{bear.strip()}\n\n"
        "You are the judge. Check both arguments against the filing data and decide. "
        f"{BRIEF}\n" + ANSWER_FORMAT
    )


def revise(q: Question, own: str, others: Sequence[str]) -> str:
    """Round two of the debate, after Du et al. (2023), section 2.1."""
    peers = "\n\n".join(f"Agent {i + 1}:\n{text.strip()}" for i, text in enumerate(others))
    return (
        _head(q) + f"Your earlier answer:\n{own.strip()}\n\n"
        f"{PEERS_HEADER}\n\n{peers}\n\n"
        f"{PEERS_FOOTER}, give an updated answer. "
        f"{BRIEF}\n" + ANSWER_FORMAT
    )
