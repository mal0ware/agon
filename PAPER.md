# Sources, protocol and comparison

This file has four parts: the sources and what each one says, the protocol that was fixed before the run, the formulas, and the comparison of the results with the sources. The last part also lists every variant that was tried.

## 1. Sources

All four were read on 2026-11-23 from the arXiv HTML pages. Where a detail could not be found on the page, that is said.

**Du, Li, Torralba, Tenenbaum and Mordatch (2023), "Improving Factuality and Reasoning in Language Models through Multiagent Debate", arXiv 2305.14325.** Section 2.1 gives the procedure: several agents answer, then each agent is shown the other agents' responses and asked for an updated response. Section 3.1 says the experiments mainly use three agents and two rounds. The debate in this repository follows that shape: three agents, two rounds. I did not find a stated rule for picking the final answer on the page I read, so the majority vote over the last round used here is my choice, not theirs.

**Smit, Duckworth, Grinsztajn, Barrett and Pretorius (2023), "Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs", arXiv 2311.17371.** The abstract says multi-agent debate systems in their current form "do not reliably outperform other proposed prompting strategies", such as self-consistency and ensembling. They add that tuning, mainly how much the agents agree, can change the ranking.

**Zhang, Cui, Chen, Wang, Zhang, Wang, Wu and Hu (2025), "Stop Overvaluing Multi-Agent Debate -- We Must Rethink Evaluation and Embrace Model Heterogeneity", arXiv 2502.08788.** The abstract reports that across 5 debate methods, 9 benchmarks and 4 models, debate often fails to beat chain-of-thought and self-consistency while using more computation, and that mixing different models helps debate. Only the abstract was readable on the page I fetched, so I cite no section numbers.

**Xiao, Sun, Luo and Wang (2024), "TradingAgents: Multi-Agents LLM Financial Trading Framework", arXiv 2412.20138.** Section 4.2 describes one bullish and one bearish researcher who debate for a number of rounds set by a facilitator, and the facilitator picks the prevailing view. The evaluation in section 5 is a backtest on a few large stocks over January to March 2024 against rule-based baselines. I found no comparison with a single model sampled several times at the same token cost. That missing comparison is what this repository measures, on a much simpler task.

## 2. Protocol

Everything in this section was fixed and committed before the evaluation run started. The pilot that informed it is described in section 5.

### Questions

- Source: SEC XBRL frames (`https://data.sec.gov/api/xbrl/frames/`), concepts and periods listed in `src/agon/filings.py`.
- A filing qualifies if it reports at least five income-statement rows and six balance-sheet rows from the lists, for the quarter 1 July to 30 September 2026, with comparatives for the same quarter of 2025 and for the balance sheet at 31 December 2025, all tagged to the same accession number.
- Qualifying filings are shuffled once with seed 20261123. The first 30 are the pilot set. The evaluation set is the next 250, a size chosen from the pilot's speed (about 36 seconds a question) to fit about two and a half hours of GPU time.
- One claim per filing. Six templates in rotation (`src/agon/questions.py`). Truth values are balanced by a seeded shuffle.
- The threshold in a claim sits 1%, 2% or 4% (relative) from the real value. Shift claims have no threshold; the ratio they ask about moved by 0.5% to 8%.
- The truth is computed from the numbers shown to the model, which are the filing's tagged values rounded to thousands of dollars.

### Models and decoding

- Two open-weight models run locally through Ollama: `qwen2.5:7b-instruct` (Q4_K_M) and `gemma4:e4b-it-q4_K_M` (Q4_K_M). One model in memory at a time.
- Sampled calls: temperature 0.7, top-p 0.95, top-k 40, repeat penalty 1.0. Greedy calls: temperature 0. Thinking mode off. Context 4096 tokens.
- Seeds: base seed 20261123. Sample `i` of a vote uses seed base + i. Debate round two uses base + 100 + agent index. Bull and bear use base + 200 and base + 201.

### Systems and budgets

`c` is the cap on generated tokens for one answer, set to 128. The shared budget is `6c` = 768 generated tokens per question.

| system | calls | cap per call | budget |
|---|---|---|---|
| single | 1 greedy answer | c | c |
| vote of 5 | 5 sampled answers | c | 5c |
| vote of 6 | 6 sampled answers | c | 6c |
| bull, bear, judge | bull (sampled), bear (sampled), judge (greedy) | 2c each | 6c |
| debate | 3 sampled answers, then 3 revised answers | c | 6c |

- The debate's first round is the same three calls as the first three samples of the vote: same prompt, same seed. The runner executes each distinct call once and charges every system that uses it.
- So debate against the vote of 6 is a clean question. Three agents have answered. Spend the other half of the budget on letting them read each other and answer again, or on three more independent answers?
- "Vote of 3" is the debate's own first round. Debate minus vote of 3 is what the second round added.
- The vote of 5 is the README's "asked five times". It uses five sixths of the budget.
- Final answers: majority of the verdicts. A tie goes to the side with the higher mean stated probability. An answer with no readable verdict does not vote. If nothing is readable the system has no verdict and is scored wrong.
- The bull is told to argue the claim is true and the bear that it is false, whatever the numbers say. The judge sees both and decides.
- Mixed configuration (both models): a vote of 6 with three samples from each model; the debate in both two-against-one line-ups; the judge panel in the four arrangements that put bull and bear on different models. Mixed debate and mixed judge are reported as the mean over their line-ups.
- Closed-book control: each model is asked each claim with the company name but without the table, greedy.

### Budget enforcement

A ledger per system per question counts generated tokens. A call must be reserved before it is made. A request beyond what is left raises an error, and so does a backend that returns more tokens than the call's cap. Prompt tokens are counted and reported but are not capped: the debate and the judge read more than the vote does.

### Scoring

- Accuracy: share of claims where the verdict equals the truth. No verdict counts as wrong.
- Probability that the claim is true: each answer states how sure it is of its verdict, 50 to 100. That is clamped to that range and flipped for a FALSE verdict. A verdict with no readable confidence counts as 75. A system's probability is the mean over its voting members.
- Primary comparison: debate minus vote of 6, paired by question, in each of the three configurations.
- Secondary: debate minus vote of 5, debate minus vote of 3, judge minus vote of 6, judge minus single, vote of 5 minus single.
- Intervals: Wilson for a single accuracy, percentile bootstrap over questions (10,000 resamples, seed 20261123) for paired differences, Brier scores and averaged mixed systems. Exact McNemar p-values for paired binary comparisons. Nothing is corrected for multiple comparisons; there are 15 paired comparisons in the table.

## 3. Formulas

| quantity | formula | source |
|---|---|---|
| Wilson interval | centre `(p + z²/2n) / (1 + z²/n)`, half-width `z·sqrt(p(1−p)/n + z²/4n²) / (1 + z²/n)`, `z = 1.96` | Wilson (1927), "Probable inference, the law of succession, and statistical inference", JASA 22. Formula taken from a secondary statement (Newcombe 1998, method 3); I did not read the 1927 paper. |
| Brier score | `mean((p − y)²)`, `y` = 1 if the claim is true | Brier (1950), "Verification of forecasts expressed in terms of probability", Monthly Weather Review 78. Secondary source: this is the two-class form with one probability, half of Brier's original two-category sum. |
| Expected calibration error | `Σ_b (n_b / n) · abs(acc_b − conf_b)` over 5 equal bins of confidence on [0.5, 1] | Naeini, Cooper and Hauskrecht (2015); Guo et al. (2017), "On Calibration of Modern Neural Networks", equation 3. Written from memory of equation 3 and not re-read today. |
| McNemar exact test | two-sided binomial tail on the discordant pairs with success probability 1/2 | McNemar (1947), Psychometrika 12; the exact binomial form is the standard small-sample version. Secondary. |
| Majority of independent voters | `Σ_{k>n/2} C(n,k) p^k (1−p)^(n−k)` | Condorcet's jury theorem. Used only in a test of the fake backend. |
| Paired bootstrap | resample questions with replacement, recompute the mean difference, take the 2.5 and 97.5 percentiles | Efron and Tibshirani (1993), percentile method. Secondary. |

The tests in `tests/test_scoring.py` check each of these against closed forms or hand-worked values.
