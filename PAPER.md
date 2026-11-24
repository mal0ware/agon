# Sources, protocol and comparison

This file has four parts: the sources and what each one says, the protocol that was fixed before the run, the formulas, and the comparison of the results with the sources. The last part also lists every variant that was tried.

## 1. Sources

All four were read on 2026-11-23 from the arXiv HTML pages. Where a detail could not be found on the page, that is said.

**Du, Li, Torralba, Tenenbaum and Mordatch (2023), "Improving Factuality and Reasoning in Language Models through Multiagent Debate", arXiv 2305.14325.** Section 2.1 gives the procedure: several agents answer, then each agent is shown the other agents' responses and asked for an updated response. Section 3.1 says the experiments mainly use three agents and two rounds. The debate in this repository follows that shape: three agents, two rounds. I did not find a stated rule for picking the final answer on the page I read, so the majority vote over the last round used here is my choice, not theirs.

**Smit, Duckworth, Grinsztajn, Barrett and Pretorius (2023), "Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs", arXiv 2311.17371.** The abstract says multi-agent debate systems in their current form "do not reliably outperform other proposed prompting strategies", such as self-consistency and ensembling. They add that tuning, mainly how much the agents agree, can change the ranking.

**Zhang, Cui, Chen, Wang, Zhang, Wang, Wu and Hu (2025), "Stop Overvaluing Multi-Agent Debate -- We Must Rethink Evaluation and Embrace Model Heterogeneity", arXiv 2502.08788.** The abstract reports that across 5 debate methods, 9 benchmarks and 4 models, debate often fails to beat chain-of-thought and self-consistency while using more computation, and that mixing different models helps debate. Only the abstract was readable on the page I fetched, so I cite no section numbers.

**Xiao, Sun, Luo and Wang (2024), "TradingAgents: Multi-Agents LLM Financial Trading Framework", arXiv 2412.20138.** Section 4.2 describes one bullish and one bearish researcher who debate for a number of rounds set by a facilitator, and the facilitator picks the prevailing view. The evaluation in section 5 is a backtest on a few large stocks over January to March 2024 against rule-based baselines. I found no comparison with a single model sampled several times at the same token cost. That missing comparison is what this repository measures, on a much simpler task.

## 2. Protocol

Everything in this section was fixed and committed before the evaluation run started (commit `e0dbf4b`), and the text is unchanged since. The pilot that informed it is in section 5. What was added after the run is in section 4.

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
| Sign-flip permutation test | flip the sign of each question's paired difference at random 20,000 times; p = (number of flips with an absolute sum at least the observed one, plus 1) / 20,001 | Fisher's randomisation test for paired data, in the usual Monte Carlo form with the add-one correction. Secondary. |
| Holm adjustment | sort the m p-values ascending, multiply the i-th (counting from 0) by m − i, take a running maximum, cap at 1 | Holm (1979), "A simple sequentially rejective multiple test procedure", Scand. J. Statist. 6. Secondary statement; I did not read the paper. |
| Paired bootstrap | resample questions with replacement, recompute the mean difference, take the 2.5 and 97.5 percentiles | Efron and Tibshirani (1993), percentile method. Secondary. |

The tests in `tests/test_scoring.py` check each of these against closed forms or hand-worked values.

## 4. Results and comparison with the sources

Run on 2026-11-23. 250 questions, 10,000 model calls, 603,370 generated tokens, 3.02 hours of model time on one RTX 3080. Full tables are in `results/`.

### Added after the run

Two things are not in the protocol above.

1. The system `long`: one greedy answer with the prompt "Show the calculation step by step, in at most five short lines" and a cap of 2c = 256 tokens, the allowance each call of the judge panel has. I added it after seeing the judge panel score well above the vote, to find out whether the panel or the allowance was responsible. It reused the first run's raw log, so the 9,500 earlier calls were not repeated and their outcomes are byte-identical. Every comparison that involves it is marked "post hoc".
2. The sign-flip p-values and the Holm adjustment. The protocol said nothing would be corrected. A table of 21 comparisons without a correction invites over-reading, so the table now gives a Holm-adjusted p-value within each family: primary (3 rows), secondary (12), post hoc (6).

### Primary comparison: debate minus vote of 6, same budget

| configuration | difference | 95% interval | wins / losses | p | Holm p |
|---|---|---|---|---|---|
| gemma4 e4b | +2.8 points | 0.4 to 5.2 | 9 / 2 | 0.066 | 0.20 |
| qwen2.5 7B | +1.2 points | -2.8 to 5.2 | 15 / 12 | 0.70 | 1.0 |
| mixed | +1.0 points | -2.0 to 4.0 | 22 / 21 | 0.61 | 1.0 |

No configuration shows a difference that survives the correction. The gemma interval excludes zero by a hair while its permutation p-value is 0.066. The two methods disagree at the margin because the result rests on 11 discordant questions. I read that as weak evidence of a small gain at most.

### Claim by claim

| source | claim | here |
|---|---|---|
| Smit et al., abstract | Debate does "not reliably outperform" simpler strategies such as self-consistency. | Agrees. Debate is within 3 points of a vote at the same budget in all three configurations. |
| Zhang et al., abstract | Debate often fails to beat chain-of-thought and self-consistency while using more computation. | Agrees. A single answer with five lines of working beat debate by 15.6 points for gemma (10.4 to 20.8) and 14.8 for qwen (8.0 to 21.6), using about a third of the generated tokens. This is a post hoc comparison. |
| Zhang et al., abstract | Mixing different models improves debate. | Not reproduced. Mixed debate scored 78.2%, against 80.0% for gemma debating itself and 77.2% for a mixed vote. One mixed result does stand out: the second round added 4.8 points over the mixed first round (2.2 to 7.4, Holm p 0.006). A mixed vote of 6 got the same lift without the debate. |
| Du et al., sections 2.1 and 3 | A second round of debate improves on the agents' first answers. | Direction agrees, size is small. The second round changed the majority from wrong to right more often than the reverse in every line-up (7 to 2, 16 to 9, 24 to 6, 10 to 4). The gain over the first round is +2.0 and +2.8 points for the single-model debates, with intervals that include zero. |
| Xiao et al., section 4.2 | A bull and bear debate settled by a facilitator is part of a framework that beats rule-based baselines in a backtest. | Not tested. Nothing here trades. On checkable claims the bull, bear and judge panel beat the vote for gemma (+15.2 points, 9.2 to 20.8). It did not beat one model given the same room to work: -3.2 points for gemma (-6.8 to 0.4) and -9.6 for qwen (-16.0 to -3.2). The panel's edge over the vote came from the allowance. |

### Why the numbers differ from the sources

- The models are 7B-class at 4-bit quantisation. The sources use much larger hosted models.
- The task is arithmetic on a table with a true-or-false answer. The sources use question-answering and reasoning benchmarks, and TradingAgents uses a trading backtest.
- The budget here is equal generated tokens with a two-line limit per answer. Du et al. do not hold cost equal. That limit turned out to cost 16 to 17 points, so the debate and the vote were both compared in a handicapped state.
- Voting needs errors that differ between samples. Here the vote of 6 agreed with the greedy answer on 224 to 230 of 250 questions, so there was little for it to average away.

## 5. Variants tried

Counted in full. Nothing was tuned on the evaluation set.

1. Pilot A, 12 filings from the pilot slice. Four claim templates, thresholds 5%, 10% and 20% from the real value, cap 128. Single answers scored 92% and 100%. That is too easy to separate any two systems. Dropped.
2. Pilot B, 18 filings from the pilot slice. The current six templates, thresholds 1%, 2% and 4%, and the brevity sentence reworded from "at most two short lines" to "only the calculation, in one or two short lines" to reduce truncated judge rulings. Single answers scored 67% on both models. Kept. This is the protocol in section 2.
3. A second Ollama server with four parallel slots, to raise throughput. It was slower than one slot on this GPU. Dropped before any evaluation call.
4. The evaluation run, once, as in section 2.
5. The `long` control and the adjusted p-values, added after the run, as described in section 4.

The 30 pilot filings are excluded from the evaluation set. The cap, the sample size and the templates were set from the pilots. No evaluation question was seen before the run, and the evaluation run was not repeated with other settings.

## 6. Not verified

- The Wilson, Brier, ECE, McNemar and Holm formulas are from secondary statements or memory, as marked in section 3. The tests check them against closed forms and hand-worked values.
- Only the abstract of Zhang et al. was read. The summaries of Du et al. and Xiao et al. rest on the arXiv HTML pages, parts of which reached me as a summary of the page.
- The author list of TradingAgents is taken from the arXiv metadata.
- Du et al.'s rule for the final answer was not found, so the majority vote over the last round is my assumption.
- The tagged values were not compared with the printed filings.
- That a full rerun reproduces every response was not tested. The committed tables do rebuild exactly from `results/outcomes.csv`.
- I do not know gemma4's training cutoff. The closed-book control (50.0% and 49.6%) is the evidence against recall.
