"""Read a verdict and a confidence out of a model reply."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ANSWER = re.compile(r"ANSWER\W{0,6}(TRUE|FALSE)\b", re.IGNORECASE)
_BARE = re.compile(r"\b(TRUE|FALSE)\b")
_CONFIDENCE = re.compile(r"CONFIDENCE\W{0,6}(\d{1,3}(?:\.\d+)?)\s*(%?)", re.IGNORECASE)


@dataclass(frozen=True)
class Answer:
    """A parsed reply.

    ``verdict`` is None when no verdict could be read. ``p_true`` is the
    probability the reply puts on the claim being true, or None when the reply
    gave a verdict but no usable confidence.
    """

    verdict: bool | None
    p_true: float | None


def parse_answer(text: str) -> Answer:
    """Parse ``ANSWER: TRUE|FALSE`` and ``CONFIDENCE: 50-100``.

    Rules, in order:

    1. The last ``ANSWER: TRUE`` or ``ANSWER: FALSE`` in the text is the verdict.
    2. With no such tag, the verdict is the bare upper-case word TRUE or FALSE,
       but only if exactly one of the two words appears.
    3. Otherwise there is no verdict.

    The confidence is how sure the model is of its own verdict. It is clamped
    to [50, 100] and turned into a probability that the claim is true. A value
    of at most 1 is read as a fraction.
    """
    tagged = _ANSWER.findall(text)
    verdict: bool | None
    if tagged:
        verdict = tagged[-1].upper() == "TRUE"
    else:
        bare = set(_BARE.findall(text))
        verdict = (bare == {"TRUE"}) if len(bare) == 1 else None
    if verdict is None:
        return Answer(None, None)

    found = _CONFIDENCE.findall(text)
    if not found:
        return Answer(verdict, None)
    raw = float(found[-1][0])
    if raw <= 1.0 and not found[-1][1]:
        raw *= 100.0
    sure = min(max(raw, 50.0), 100.0) / 100.0
    return Answer(verdict, sure if verdict else 1.0 - sure)
