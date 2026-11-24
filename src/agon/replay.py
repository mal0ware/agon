"""The committed raw logs, and rebuilding the outcomes from them.

A run writes every prompt and response to ``runs/<name>/calls.jsonl``, which
git ignores. :func:`export_logs` copies that log into ``logs/`` as one gzipped
JSON-lines file per model, plus the question set with the filing tables the
models were shown. :func:`rebuild_outcomes` reads those files back and runs
every system again with a backend that can only replay logged calls. If a call
is missing it raises, so the rebuilt ``outcomes.csv`` is the logged run or
nothing.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from agon.backends import Completion, Request
from agon.questions import Question
from agon.runner import CallLog, JobKey, run_experiment
from agon.systems import Outcome, Protocol

QUESTIONS_FILE = "questions.json.gz"
# The two models of the committed run, in the order the line-up uses them.
RUN_MODELS = ("qwen2.5:7b-instruct", "gemma4:e4b-it-q4_K_M")


def log_name(model: str) -> str:
    """``qwen2.5:7b-instruct`` -> ``calls_qwen2.5-7b-instruct.jsonl.gz``."""
    return "calls_" + model.replace(":", "-").replace("/", "-") + ".jsonl.gz"


def _write_gz(path: Path, text: str) -> None:
    """Gzip with a fixed timestamp and no file name, so the bytes are repeatable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as gz:
        gz.write(text.encode("utf-8"))
    path.write_bytes(buffer.getvalue())


def _read_lines(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line


def export_logs(
    calls: Path, questions: Path | None, out: Path, models: Sequence[str] | None = None
) -> dict[str, int]:
    """Split a raw call log by model into ``out``. Returns calls written per file.

    Records keep the request and the completion as logged. Each line is checked
    against its stored key first, so an edited record cannot be exported.
    """
    by_model: dict[str, list[str]] = {}
    with calls.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            request = Request(**record["request"])
            if request.key() != record["key"]:
                raise ValueError(f"log record does not match its key: {record['key']}")
            if models is not None and request.model not in models:
                continue
            clean = {
                "key": record["key"],
                "request": record["request"],
                "completion": record["completion"],
            }
            by_model.setdefault(request.model, []).append(json.dumps(clean, ensure_ascii=False))
    counts: dict[str, int] = {}
    for model, lines in sorted(by_model.items()):
        name = log_name(model)
        _write_gz(out / name, "\n".join(lines) + "\n")
        counts[name] = len(lines)
    if questions is not None:
        _write_gz(out / QUESTIONS_FILE, questions.read_text(encoding="utf-8"))
    return counts


def load_logs(paths: Sequence[Path]) -> CallLog:
    """Read gzipped call logs into an in-memory call cache."""
    log = CallLog(None)
    for path in paths:
        for line in _read_lines(path):
            record = json.loads(line)
            request = Request(**record["request"])
            if request.key() != record["key"]:
                raise ValueError(f"{path.name}: record does not match its key")
            log.cache[record["key"]] = Completion(**record["completion"])
    return log


def load_logged_questions(logs: Path) -> list[Question]:
    """The questions, with the filing tables the models were shown."""
    with gzip.open(logs / QUESTIONS_FILE, "rt", encoding="utf-8") as handle:
        return [Question(**item) for item in json.load(handle)]


class ReplayBackend:
    """A backend that has no model. Any call that reaches it was not logged."""

    def complete(self, request: Request) -> Completion:
        raise KeyError(f"call not in the logs: {request.model}, key {request.key()}")


def rebuild_outcomes(
    logs: Path, models: Sequence[str], proto: Protocol | None = None
) -> tuple[list[Question], dict[JobKey, Outcome]]:
    """Run every system again from the committed logs alone."""
    questions = load_logged_questions(logs)
    log = load_logs([logs / log_name(m) for m in models])
    outcomes = run_experiment(questions, models, ReplayBackend(), proto or Protocol(), log)
    return questions, outcomes
