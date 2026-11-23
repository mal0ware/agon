"""Rebuild the results.

Three modes:

    uv run python scripts/reproduce.py
        Rebuild every table and figure in results/ from the committed
        per-question outcomes (results/outcomes.csv). Seconds, no GPU.

    uv run python scripts/reproduce.py --fast
        Also run the whole pipeline end to end against the fake backend on
        made-up filings, into runs/fast/. Under two minutes, no GPU, no network.
        This is what CI runs.

    uv run python scripts/reproduce.py --run-models
        Run the experiment again on local models through Ollama. Needs
        scripts/fetch_data.py to have been run, the two models pulled, and a
        few hours of GPU time. Overwrites results/. Raw prompts and responses
        go to runs/, which git ignores.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from agon import backends
from agon.backends import FakeBackend, OllamaBackend, fake_key
from agon.filings import build_statements, frame_names, synthetic_statements
from agon.questions import build_questions, order_statements, save_questions
from agon.report import build_report, write_outcomes, write_questions
from agon.runner import CallLog, run_experiment
from agon.systems import Protocol

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_SEED = 20261123
PILOT = 30  # companies at the head of the fixed shuffle, used only to set up the run
MODELS = ("qwen2.5:7b-instruct", "gemma4:e4b-it-q4_K_M")


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fast() -> None:
    """End-to-end smoke run with the fake backend."""
    out = ROOT / "runs" / "fast"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calls.jsonl").unlink(missing_ok=True)
    questions = build_questions(synthetic_statements(60, DATA_SEED), DATA_SEED)
    truth = {fake_key(q.company, q.claim): q.truth for q in questions}
    backend = FakeBackend(truth, {"fake-a": 0.8, "fake-b": 0.65}, herding=0.5)
    outcomes = run_experiment(
        questions, ["fake-a", "fake-b"], backend, Protocol(), CallLog(out / "calls.jsonl")
    )
    write_questions(out / "questions.csv", questions)
    write_outcomes(out / "outcomes.csv", questions, outcomes)
    tables = build_report(out)
    say(f"fast run: {len(questions)} questions, {backend.calls} fake calls, tables in {out}")
    for row in tables["summary"]:
        say(f"  {row['config']:8s} {row['system']:12s} accuracy {row['accuracy']}")


def gpu_name() -> str:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return done.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def run_models(args: argparse.Namespace) -> None:
    """The real run on local models."""
    frames = {}
    for concept, period in frame_names():
        path = ROOT / "data" / "frames" / f"{concept}_{period}.json"
        if not path.exists():
            sys.exit("data/frames is incomplete: run scripts/fetch_data.py first")
        frames[concept, period] = json.loads(path.read_text(encoding="utf-8"))["data"]
    statements = order_statements(build_statements(frames), DATA_SEED)
    chosen = statements[PILOT : PILOT + args.n]
    questions = build_questions(chosen, DATA_SEED)
    run_dir = ROOT / "runs" / args.name
    save_questions(questions, run_dir / "questions.json")
    say(f"{len(statements)} filings available, {len(questions)} questions built")

    backend = OllamaBackend(args.host)
    for model in args.models:  # start with nothing loaded, so the first model gets the GPU
        backend.unload(model)
    proto = Protocol(cap=args.cap)
    log = CallLog(run_dir / "calls.jsonl")
    already = len(log.cache)
    started = time.time()
    outcomes = run_experiment(questions, list(args.models), backend, proto, log, say)
    wall = time.time() - started
    for model in args.models:
        backend.unload(model)

    calls = list(log.cache.values())
    info = backend.model_info()
    RESULTS.mkdir(exist_ok=True)
    write_questions(RESULTS / "questions.csv", questions)
    write_outcomes(RESULTS / "outcomes.csv", questions, outcomes)
    run_info = {
        "date": date.today().isoformat(),
        "questions": len(questions),
        "filings_available": len(statements),
        "pilot_filings_skipped": PILOT,
        "data_seed": DATA_SEED,
        "models": {m: info.get(m, {}) for m in args.models},
        "runner": "Ollama, local, one model in memory at a time",
        "gpu": gpu_name(),
        "decoding": {
            "temperature_sampled": proto.temperature,
            "temperature_greedy": 0.0,
            "top_p": backends.TOP_P,
            "top_k": backends.TOP_K,
            "repeat_penalty": backends.REPEAT_PENALTY,
            "context_tokens": backends.CONTEXT_TOKENS,
            "base_seed": proto.base_seed,
            "thinking": "off",
        },
        "cap_tokens_per_answer": proto.cap,
        "budget_tokens_per_question": proto.budget,
        "model_calls": len(calls),
        "calls_reused_from_an_earlier_start": already,
        "generated_tokens": sum(c.completion_tokens for c in calls),
        "prompt_tokens": sum(c.prompt_tokens for c in calls),
        "model_seconds": round(sum(c.seconds for c in calls)),
        "wall_seconds_this_start": round(wall),
        "paid_api_calls": 0,
    }
    (RESULTS / "run_info.json").write_text(
        json.dumps(run_info, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    say(f"run finished in {wall / 60:.1f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fast", action="store_true", help="fake-backend smoke run as well")
    parser.add_argument("--run-models", action="store_true", help="rerun on local models")
    parser.add_argument("--n", type=int, default=250, help="filings to draw questions from")
    parser.add_argument("--cap", type=int, default=128, help="token cap per answer")
    parser.add_argument("--models", nargs=2, default=list(MODELS))
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--name", default="main", help="folder under runs/ for the raw log")
    args = parser.parse_args()

    if args.fast:
        fast()
    if args.run_models:
        run_models(args)
    if (RESULTS / "outcomes.csv").exists():
        tables = build_report(RESULTS)
        say(f"rebuilt {len(tables)} tables and 3 figures in {RESULTS}")
    else:
        say("no results/outcomes.csv yet: nothing to rebuild")


if __name__ == "__main__":
    main()
