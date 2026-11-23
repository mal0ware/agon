"""Drive every system over every question, one stage at a time.

The runner keeps a call cache keyed by the full request. Two systems that make
the same call (same model, prompt, cap, temperature and seed) share one
execution, and each is still charged for it on its own ledger. Every executed
call is appended to a raw JSON-lines log with its prompt and response. The log
doubles as a checkpoint: a restarted run reads it back and skips what is done.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from agon.backends import Backend, Completion, Request
from agon.budget import BudgetError
from agon.questions import Question
from agon.systems import Outcome, Protocol, Steps, lineup

JobKey = tuple[str, str, str]  # (question id, configuration, system)


class CallLog:
    """Raw log of every prompt and response. Never committed."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.cache: dict[str, Completion] = {}
        if path is not None and path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self.cache[record["key"]] = Completion(**record["completion"])

    def get(self, request: Request) -> Completion | None:
        return self.cache.get(request.key())

    def put(self, request: Request, completion: Completion) -> None:
        self.cache[request.key()] = completion
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": request.key(),
            "request": asdict(request),
            "completion": asdict(completion),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_jobs(
    jobs: dict[JobKey, Steps],
    backend: Backend,
    log: CallLog | None = None,
    model_order: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> dict[JobKey, Outcome]:
    """Run every job to completion and return its outcome."""
    log = log or CallLog(None)
    say = progress or (lambda _msg: None)
    waiting: dict[JobKey, list[Request]] = {}
    outcomes: dict[JobKey, Outcome] = {}
    for key, steps in jobs.items():
        try:
            waiting[key] = next(steps)
        except StopIteration as stop:
            outcomes[key] = stop.value

    rank = {m: i for i, m in enumerate(model_order)}
    stage = 0
    while waiting:
        stage += 1
        todo: dict[str, Request] = {}
        for requests in waiting.values():
            for r in requests:
                if log.get(r) is None:
                    todo.setdefault(r.key(), r)
        ordered = sorted(todo.values(), key=lambda r: rank.get(r.model, len(rank)))
        say(f"stage {stage}: {len(waiting)} jobs waiting, {len(ordered)} new calls")
        started = time.time()
        for i, r in enumerate(ordered, start=1):
            completion = backend.complete(r)
            if completion.completion_tokens > r.max_tokens:
                raise BudgetError(
                    f"backend returned {completion.completion_tokens} tokens "
                    f"for a call capped at {r.max_tokens}"
                )
            log.put(r, completion)
            if i % 50 == 0 or i == len(ordered):
                rate = (time.time() - started) / i
                say(f"stage {stage}: {i}/{len(ordered)} calls, {rate:.2f} s per call ({r.model})")

        still: dict[JobKey, list[Request]] = {}
        for key, requests in waiting.items():
            completions = [c for c in (log.get(r) for r in requests) if c is not None]
            try:
                still[key] = jobs[key].send(completions)
            except StopIteration as stop:
                outcomes[key] = stop.value
        waiting = still
    return {key: outcomes[key] for key in jobs}


def run_experiment(
    questions: Sequence[Question],
    models: Sequence[str],
    backend: Backend,
    proto: Protocol,
    log: CallLog | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[JobKey, Outcome]:
    """Every system in the line-up, for every question."""
    jobs: dict[JobKey, Steps] = {}
    for q in questions:
        for (config, system), steps in lineup(q, models, proto).items():
            jobs[q.qid, config, system] = steps
    return run_jobs(jobs, backend, log, models, progress)
