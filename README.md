# Agon

**Three language models argue about a filing. Is that better than asking one model several times?**

On 250 true-or-false claims about fresh SEC filings, with two small local models, mostly no. At the same budget of generated tokens, a two-round debate scored 1.2 to 2.8 points above a majority vote when every answer was held to two lines. None of those gaps survives a multiple-testing correction, and most of gemma's gap sits on questions where the vote of six split three to three and had no verdict. The largest effect in the experiment was answer length. Letting each answer show five lines of working instead of two raised the same system by 16 to 24 points. With five-line answers, debate beat the vote by 2.0 points for gemma and 4.8 points for qwen (interval 1.2 to 8.8, Holm p 0.12).

![Accuracy of each system with 95% intervals, for each model and for the two mixed](results/fig_headline.png)

| | single | single, long | vote of 6 | debate | bull, bear, judge | vote of 6, long | debate, long |
|---|---|---|---|---|---|---|---|
| gemma4 e4b | 78.4% | 95.6% | 77.2% | 80.0% | 92.4% | 95.6% | 97.6% |
| qwen2.5 7B | 66.4% | 82.4% | 66.4% | 67.6% | 72.8% | 86.8% | 91.6% |
| both, mixed | | | 77.2% | 78.2% | 89.2% | | |

| debate minus vote of 6, same budget | two-line answers | five-line answers |
|---|---|---|
| gemma4 e4b | +2.8 points (0.4 to 5.2), Holm p 0.20 | +2.0 points (0.0 to 4.4), Holm p 0.37 |
| qwen2.5 7B | +1.2 points (-2.8 to 5.2), Holm p 1.0 | +4.8 points (1.2 to 8.8), Holm p 0.12 |
| both, mixed | +1.0 points (-2.0 to 4.0), Holm p 1.0 | not run |

Brackets are paired bootstrap 95% intervals over the 250 questions. Chance is 50%. "Long" means the answer may show five lines of working with a cap of 256 tokens; every other answer was told to use one or two lines with a cap of 128. All three long systems were added after the first results. `single, long` was a control. The long vote and debate came in a follow-up whose design was committed in `4a96a5f` before it ran.

The main limits: these are 4-bit 7B-class models on a home GPU, the task is arithmetic on a small table, and gemma is near its ceiling with five-line answers, which leaves little room for any system to beat another.

## The question

Multi-agent debate is a popular design for AI trading systems. I could not find a head-to-head test of it against a single model on finance tasks at the same cost. General studies say debate does not reliably win. This repository tests it on a task with checkable answers. Every model runs locally through Ollama on one RTX 3080, and nothing was paid for.

## Reproduce

```
uv sync
uv run python scripts/reproduce.py          # replay the logged run and rebuild every table and figure
uv run python scripts/reproduce.py --fast   # also run the whole pipeline against a fake model
```

Neither command needs a GPU, a model or the network. The first one reads every prompt and response from `logs/`, runs each system again with a backend that can only return logged responses, writes `results/outcomes.csv`, and rebuilds the tables and figures from it. A call missing from the logs stops it with an error. CI checks that the rebuilt files match the committed ones.

To run the models again:

```
ollama pull qwen2.5:7b-instruct
ollama pull gemma4:e4b-it-q4_K_M
uv run python scripts/fetch_data.py
uv run python scripts/reproduce.py --run-models
```

That overwrites `results/` and `logs/`. The SEC adds filings to the source files over time, so a later download can draw a different set of 250. `logs/questions.json.gz` holds the questions this run used, with the tables the models were shown.

## What was built

- A question set: 250 true-or-false claims, one per filing, each about numbers in that filing's income statement and balance sheet for the quarter ended 30 September 2026. Half are true. The threshold in each claim sits 1%, 2% or 4% from the real value, so the model has to do the sum.
- Systems, each run on each model:
  - `single`: one greedy answer.
  - `vote of 5` and `vote of 6`: independent sampled answers and a majority vote.
  - `bull, bear, judge`: one agent told to argue the claim is true, one told to argue it is false, and a judge who reads both.
  - `debate`: three agents answer, read each other's answers, and answer again. Majority of the second round.
  - `single, long`: one greedy answer with five lines of working allowed and twice the token cap. Added after the first results.
  - `vote of 6, long` and `debate, long`: the vote and the debate with five-line answers and twice the cap. The follow-up run.
- A mixed set-up that puts both models in the same vote, debate or judge panel. It was only run with two-line answers.
- A token ledger. The vote of 6, the judge panel and the debate each get 768 generated tokens per question; the long vote and the long debate get 1,536. A call that would break the budget raises an error.
- Accuracy, Brier score, calibration error and token counts for every system, with paired comparisons.

The debate's first round is the same three calls as the first three samples of the vote, with the same seeds. So debate against the vote of 6 compares two ways to spend the second half of the budget: let three agents read each other and answer again, or draw three more independent answers. The long versions keep that pairing.

## What the run found

Numbers are from `results/summary.md`, `results/comparisons.md`, `results/no_verdict.md` and `results/verdict_lean.md`.

### Debate against the vote, two-line answers

The second round fixed more answers than it broke (16 against 9 for qwen, 7 against 2 for gemma, 34 against 10 across the two mixed line-ups). Three more independent votes did about as well. The gap is +1.0 to +2.8 points, and none survives the Holm correction.

### Ties in the vote of 6

Six voters can split three to three. When their stated confidences also cancel to exactly 50%, the protocol gives no verdict and scores the question wrong. That happened on 5 to 7 of 250 questions in each single-model vote of 6, and on 10 in the mixed one. A debate of three agents cannot tie. For gemma, 5 of the debate's 9 wins over the vote were on such ties with two-line answers, and all 6 of its wins with five-line answers. Scoring a question with no verdict as half right, the debate's lead over the vote drops to +1.4 points for gemma (-0.8 to 3.6) and 0.0 for qwen with two-line answers, and to +0.6 (-0.8 to 1.8) and +3.8 (0.4 to 7.4) with five-line answers. This check was added after the results were read.

### Answer length

Every two-line system was 16 to 24 points below its five-line version. A single long answer beat the two-line debate by 15.6 points for gemma and 14.8 for qwen, on about a third of the generated tokens. Once answers had room, gemma was near its ceiling: 95.6% for one answer, 95.6% for a vote of six and 97.6% for debate. Qwen gained more from extra agents: 82.4% for one answer, 86.8% for a vote of six and 91.6% for debate. Qwen's second round with five-line answers fixed 21 majorities and broke 6, a gain of 6.0 points over its first round (2.0 to 10.0, Holm p 0.039 within the follow-up). It is the one debate result here that survives the correction.

### The judge panel

With gemma as judge the panel scored 92.4%, 15 points above the vote of 6. But the bull, the bear and the judge were each allowed five lines and 256 tokens. A single answer with the same allowance scored 3.2 points above the panel for gemma (-6.8 to 0.4) and 9.6 points above it for qwen (3.2 to 16.0). The long answers averaged about 100 tokens, under the 128-token cap the short answers already had. What held the short answers back was the instruction to be brief.

### Short answers lean TRUE

The claims are half true. With two-line answers, gemma's single answer said TRUE on 59.6% of claims (53.4 to 65.5) and qwen's on 56.4% (50.2 to 62.4). The errors fall on the false claims: gemma's single answer was right on 88.0% of true claims and 68.8% of false ones. The vote and the debate carry the same lean. Five-line answers removed it for gemma (48.8%, change -10.8 points, -16.4 to -5.2) but not for qwen (58.8%, change +2.4, -4.8 to 9.6). Qwen's long answers still went wrong mostly on false claims: 91.2% right on true claims, 73.6% on false ones. This table was added after the results were read.

### The recall control

With the table hidden, both models scored 50.0% and 49.6%. They said FALSE on almost every claim (gemma on all 250, qwen on 249), so that score is the base rate. They were not recalling the filings.

### Mixing models

A mixed vote of 6 scored 77.2%, the same as gemma alone and 11 points above qwen alone. Mixed debate was 78.2%. Mixing did not lift debate above the vote.

### Calibration

In their two-line single answers both models said they were 95% to 100% sure almost every time, right or wrong. Brier scores follow accuracy: 0.021 for gemma's long debate, 0.044 for its long single answer, 0.18 for its two-line vote and debate. See `results/fig_calibration.png`.

### Cost

Per question, the two-line vote of 6 generated 304 to 328 tokens and the two-line debate 303 to 316. The long vote of 6 generated 562 to 671 and the long debate 577 to 680. The long single answer generated 92 to 110. Debate also reads the most: about 4,700 prompt tokens per question for the long debate against 3,700 for the long vote and 630 for a single answer. See `results/fig_cost.png`.

## Sources

- Xiao, Sun, Luo and Wang, "TradingAgents: Multi-Agents LLM Financial Trading Framework", arXiv 2412.20138, 2024
- Smit et al., "Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs", arXiv 2311.17371
- Zhang et al., "Stop Overvaluing Multi-Agent Debate", arXiv 2502.08788
- Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate", arXiv 2305.14325, for the debate procedure

In brief: the result agrees with Smit et al. and Zhang et al. that debate does not reliably beat simpler baselines at the same cost. It does not reproduce Zhang et al.'s finding that mixing models helps debate. It says nothing about TradingAgents' trading results, which were not tested. `PAPER.md` has the protocol, the formulas, the full comparison and every variant tried.

## Data

Public company filings, through the SEC's XBRL frames API at `data.sec.gov`. It is free, needs no key or account, and the data is US government work. `scripts/fetch_data.py` downloads about 23 MB into `data/`, which git ignores.

Every filing is for the quarter ended 30 September 2026 and was filed in October or November 2026. The qwen model dates from 2024. I do not know gemma4's training cutoff, which is why the closed-book control is there.

`logs/` holds every prompt and response from the run, gzipped, one file per model: 14,500 calls in about 2 MB. Each prompt includes the filing table the model saw, about 20 numbers from one filing. `logs/questions.json.gz` has the 250 questions with their tables, and `logs/pilot_a/` and `logs/pilot_b/` have the two pilots. The logs hold the model name, prompt, decoding settings, response, token counts and time per call, and nothing about the machine. Some company names in the source contain a non-breaking space; the prompts keep it, and `results/questions.csv` writes a plain space.

## How it is tested

59 tests, none of which need a GPU or model weights.

- The harness runs end to end against a fake model with known behaviour. A fake with 70% accuracy produces the vote accuracy the binomial formula predicts.
- Token budgets are enforced. A system that asks for more than its budget is stopped, and so is a backend that returns more tokens than its cap.
- The committed logs rebuild the committed `outcomes.csv` byte for byte. A log record whose request does not match its stored hash is refused, and a call missing from the logs raises an error.
- Scoring is deterministic for a fixed set of responses. Building the report twice gives identical tables.
- The Wilson interval, Brier score, calibration error, McNemar test, sign-flip test and Holm correction are checked against closed forms and hand-worked values.
- Every generated claim is re-evaluated from the table by a separate function and must match its stored truth.

CI runs ruff, mypy, the tests and the fast reproduce on a CPU runner, replays the logs, and checks that the committed tables rebuild unchanged.

## Layout

```
src/agon/              the code
tests/                 maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
logs/                  every prompt and response, gzipped
results/               committed figures and small tables
PAPER.md               citations, protocol, formulas, comparison with the sources
```

The model sits behind a one-method interface in `src/agon/backends.py`. Only the local backend, the fake and the log replay are included. The local one refuses any host that is not this machine.

## Limits

- The models are qwen2.5 7B and gemma4 e4b at 4-bit quantisation. A stronger model would likely get nearly all of these claims right alone, which leaves no room to measure anything. Gemma already reaches 95.6% to 98.0% with five-line answers. Nothing here says how debate behaves between frontier models.
- The long single answer and the long vote and debate were added after I saw the first results. The follow-up design is in `PAPER.md`, committed in `4a96a5f` before the follow-up ran. The tie check and the TRUE-lean tables were added after the results were read and are not corrected for multiple testing.
- The tie rule matters. A 3-to-3 vote of six with confidences that cancel is scored wrong, by a rule fixed before the first run. It decides 5 to 10 questions per vote of six, which is as large as most of the gaps between systems.
- One prompt wording per system, one seed, one run. One instruction moved accuracy by 16 points, so other wordings could move the rest.
- 250 questions, one per filing. Intervals on the paired differences are 2 to 4 points either side. A real effect smaller than that would not show.
- The mixed line-ups were only run with two-line answers.
- The task is arithmetic on a table of about 20 numbers. Questions with checkable answers are easier than real investment judgement. A win here would not show that debate picks better stocks, and a loss does not show that it picks worse ones.
- The bull and the bear are assigned a side on a claim that has a right answer. In the pilot log I saw a bull do the sum and concede that the claim was false. That is a weaker version of a bull and bear debate than the one in TradingAgents.
- The judge panel with gemma hit its token cap on 44 of 750 calls, and 8 rulings could not be read and were scored wrong. Its 92.4% would be a little higher with more room.
- The recall control shows the models cannot answer without the table. The thresholds are also my own, so the claims exist nowhere else. I could not check what gemma4 was trained on.
- Truth is computed from the tagged values shown to the model. I did not compare those with the printed filings. A tagging error would change what the claim is about, not whether it is scored correctly.
- The same seed gave the same output in a spot check. I did not run the models twice to confirm that every response repeats. The replay from `logs/` is exact; a rerun on the models may not be.
- The GPU was shared with other work during parts of both runs, so the per-call times in the logs vary widely and are not a clean benchmark. `results/run_info.md` gives the totals: 14,500 calls, 4.3 hours of model time.
- Results belong to the models tested on the dates tested, 23 and 24 November 2026.
- This is research code. It is not investment advice and it does not trade.

Agon is the Greek word for a contest held in front of judges.

## Licence

MIT. See `LICENSE`.
