"""The systems under test.

Each system is a generator. It yields a batch of requests, receives the
completions, and may yield again. It returns an :class:`Outcome`. Writing them
this way lets the runner execute one stage for every question before moving
on, so the GPU loads each model a handful of times instead of once per call.

With ``c`` the per-answer cap (:attr:`Protocol.cap`), the budgets are:

==========  =====================================  =========
system      calls                                  budget
==========  =====================================  =========
single      1 answer, greedy                       c
vote3       3 sampled answers                      3c
vote5       5 sampled answers                      5c
vote6       6 sampled answers                      6c
judge       bull 2c, bear 2c, judge 2c             6c
debate      3 answers, then 3 revised answers      6c
==========  =====================================  =========

The debate's first round is the same three calls as vote3, with the same
prompts and seeds. So debate against vote6 asks one thing: once three agents
have answered, is it better to let them read each other and answer again, or
to draw three more independent answers?
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass

from agon import prompts
from agon.backends import Completion, Request
from agon.budget import Ledger
from agon.parsing import Answer, parse_answer
from agon.questions import Question

# A confidence that could not be read counts as this sure of the verdict.
DEFAULT_SURE = 0.75


@dataclass(frozen=True)
class Protocol:
    """Decoding settings and budgets, fixed for a whole run."""

    cap: int = 128
    temperature: float = 0.7
    base_seed: int = 20261123

    @property
    def budget(self) -> int:
        """The shared budget: six capped answers."""
        return 6 * self.cap


@dataclass(frozen=True)
class Outcome:
    """What a system concluded on one question, and what it cost."""

    verdict: bool | None
    p_true: float
    completion_tokens: int
    prompt_tokens: int
    calls: int
    truncated: int
    unparsed: int
    budget: int


Member = tuple[str, int]  # (model, sample index)
Steps = Generator[list[Request], list[Completion], Outcome]


def member_p(answer: Answer) -> float | None:
    """One member's probability that the claim is true."""
    if answer.verdict is None:
        return None
    if answer.p_true is not None:
        return answer.p_true
    return DEFAULT_SURE if answer.verdict else 1.0 - DEFAULT_SURE


def aggregate(answers: Sequence[Answer]) -> tuple[bool | None, float]:
    """Majority vote, ties broken by the mean stated probability.

    Members with no readable verdict do not vote. The probability is the mean
    of the voting members' probabilities, or 0.5 when nobody voted. A tie with
    a mean of exactly 0.5 has no verdict and is scored as wrong.
    """
    ps = [p for p in (member_p(a) for a in answers) if p is not None]
    if not ps:
        return None, 0.5
    p = sum(ps) / len(ps)
    yes = sum(1 for a in answers if a.verdict is True)
    no = sum(1 for a in answers if a.verdict is False)
    if yes != no:
        return yes > no, p
    if p == 0.5:
        return None, p
    return p > 0.5, p


def _answer_request(q: Question, member: Member, proto: Protocol) -> Request:
    model, sample = member
    return Request(model, prompts.solo(q), proto.cap, proto.temperature, proto.base_seed + sample)


def _outcome(
    verdict: bool | None, p: float, ledger: Ledger, done: Sequence[Completion], unparsed: int
) -> Outcome:
    return Outcome(
        verdict=verdict,
        p_true=p,
        completion_tokens=ledger.spent,
        prompt_tokens=ledger.prompt_tokens,
        calls=ledger.calls,
        truncated=sum(1 for c in done if c.truncated),
        unparsed=unparsed,
        budget=ledger.budget,
    )


def _ask(
    ledger: Ledger, requests: list[Request]
) -> Generator[list[Request], list[Completion], list[Completion]]:
    """Reserve, yield, settle. Every call a system makes goes through here."""
    for r in requests:
        ledger.reserve(r.max_tokens)
    completions = yield requests
    if len(completions) != len(requests):
        raise RuntimeError("runner returned the wrong number of completions")
    for r, c in zip(requests, completions, strict=True):
        ledger.settle(r.max_tokens, c.completion_tokens, c.prompt_tokens)
    return completions


def single(q: Question, model: str, proto: Protocol) -> Steps:
    """One greedy answer."""
    ledger = Ledger(proto.cap)
    request = Request(model, prompts.solo(q), proto.cap, 0.0, proto.base_seed)
    done = yield from _ask(ledger, [request])
    answer = parse_answer(done[0].text)
    verdict, p = aggregate([answer])
    return _outcome(verdict, p, ledger, done, int(answer.verdict is None))


def closed_book(q: Question, model: str, proto: Protocol) -> Steps:
    """One greedy answer without the filing table. Measures recall and guessing."""
    ledger = Ledger(proto.cap)
    request = Request(model, prompts.closed_book(q), proto.cap, 0.0, proto.base_seed)
    done = yield from _ask(ledger, [request])
    answer = parse_answer(done[0].text)
    verdict, p = aggregate([answer])
    return _outcome(verdict, p, ledger, done, int(answer.verdict is None))


def vote(q: Question, members: Sequence[Member], proto: Protocol) -> Steps:
    """Independent sampled answers and a majority vote."""
    ledger = Ledger(len(members) * proto.cap)
    done = yield from _ask(ledger, [_answer_request(q, m, proto) for m in members])
    answers = [parse_answer(c.text) for c in done]
    verdict, p = aggregate(answers)
    return _outcome(verdict, p, ledger, done, sum(a.verdict is None for a in answers))


def debate(q: Question, agents: Sequence[Member], proto: Protocol) -> Steps:
    """Two rounds. Each agent answers, reads the others, and answers again."""
    ledger = Ledger(2 * len(agents) * proto.cap)
    first = yield from _ask(ledger, [_answer_request(q, m, proto) for m in agents])
    requests = []
    for i, (model, _) in enumerate(agents):
        others = [c.text for j, c in enumerate(first) if j != i]
        requests.append(
            Request(
                model,
                prompts.revise(q, first[i].text, others),
                proto.cap,
                proto.temperature,
                proto.base_seed + 100 + i,
            )
        )
    second = yield from _ask(ledger, requests)
    answers = [parse_answer(c.text) for c in second]
    verdict, p = aggregate(answers)
    unparsed = sum(a.verdict is None for a in answers)
    return _outcome(verdict, p, ledger, [*first, *second], unparsed)


def judged(q: Question, bull: str, bear: str, judge: str, proto: Protocol) -> Steps:
    """A bull argues true, a bear argues false, a judge reads both and decides."""
    ledger = Ledger(6 * proto.cap)
    cap = 2 * proto.cap
    cases = yield from _ask(
        ledger,
        [
            Request(bull, prompts.advocate(q, True), cap, proto.temperature, proto.base_seed + 200),
            Request(
                bear, prompts.advocate(q, False), cap, proto.temperature, proto.base_seed + 201
            ),
        ],
    )
    ruling = yield from _ask(
        ledger,
        [Request(judge, prompts.judge(q, cases[0].text, cases[1].text), cap, 0.0, proto.base_seed)],
    )
    answer = parse_answer(ruling[0].text)
    verdict, p = aggregate([answer])
    return _outcome(verdict, p, ledger, [*cases, *ruling], int(answer.verdict is None))


def lineup(q: Question, models: Sequence[str], proto: Protocol) -> dict[tuple[str, str], Steps]:
    """Every (configuration, system) job for one question.

    Each model gets a configuration of its own. With two models there is also a
    ``mixed`` configuration: a three-and-three vote, the debate in both
    two-against-one line-ups, and the judge panel in all four arrangements that
    put the bull and the bear on different models.
    """
    jobs: dict[tuple[str, str], Steps] = {}
    for m in models:
        jobs[m, "closed_book"] = closed_book(q, m, proto)
        jobs[m, "single"] = single(q, m, proto)
        for k in (3, 5, 6):
            jobs[m, f"vote{k}"] = vote(q, [(m, i) for i in range(k)], proto)
        jobs[m, "judge"] = judged(q, m, m, m, proto)
        jobs[m, "debate"] = debate(q, [(m, i) for i in range(3)], proto)
    if len(models) == 2:
        a, b = models
        six = [(a, 0), (b, 0), (a, 1), (b, 1), (a, 2), (b, 2)]
        jobs["mixed", "vote6"] = vote(q, six, proto)
        for tag, agents in (("aba", [(a, 0), (b, 0), (a, 1)]), ("bab", [(b, 0), (a, 0), (b, 1)])):
            jobs["mixed", f"vote3_{tag}"] = vote(q, agents, proto)
            jobs["mixed", f"debate_{tag}"] = debate(q, agents, proto)
        for tag, (bull, bear, jud) in {
            "aba": (a, b, a),
            "abb": (a, b, b),
            "baa": (b, a, a),
            "bab": (b, a, b),
        }.items():
            jobs["mixed", f"judge_{tag}"] = judged(q, bull, bear, jud, proto)
    return jobs
