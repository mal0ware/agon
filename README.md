# Agon

**Three language models argue about a filing. Is that better than asking one model five times?**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

Multi-agent debate is a popular design for AI trading systems. I could not find a head-to-head test of it against a single model on finance tasks. General studies say debate does not reliably win. So: does it?

## What this will contain

- An evaluation set of questions about public filings, each with an answer that can be checked in the filing itself
- Four systems: one model; one model asked five times with a majority vote; a bull, a bear and a judge; a two-round debate
- Every system held to the same token budget
- Accuracy, calibration and cost for each

## The result it is built to show

Does debate beat simply asking five times, at the same cost?

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Xiao, Sun, Luo and Wang, "TradingAgents: Multi-Agents LLM Financial Trading Framework", arXiv 2412.20138, 2024
- Smit et al., "Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs", arXiv 2311.17371
- "Stop Overvaluing Multi-Agent Debate", arXiv 2502.08788

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

Public company filings. The question set is built and published here.

## How it is tested

The maths is tested before any result is trusted.

- The harness runs end to end against a fake model with known behaviour
- Token budgets are enforced, not assumed
- Scoring is deterministic for a fixed set of responses

## Layout

```
src/agon/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [ ] Project skeleton, CI, and the reproduce command
- [ ] An evaluation set of questions about public filings, each with an answer that can be checked in the filing itself
- [ ] Four systems: one model; one model asked five times with a majority vote; a bull, a bear and a judge; a two-round debate
- [ ] Every system held to the same token budget
- [ ] Accuracy, calibration and cost for each
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- Questions with checkable answers are easier than real investment judgement. A win here does not show that debate picks better stocks.
- Results belong to the models tested on the date tested.
- If debate loses, that is the result that gets published.
- This is research code. It is not investment advice and it does not trade.

## The name

Agon is the Greek word for a contest held in front of judges. It is where "protagonist" and "antagonist" come from.

## Licence

MIT. See `LICENSE`.
