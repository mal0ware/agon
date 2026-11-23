"""Turn per-question outcomes into the committed tables and figures.

The input is ``outcomes.csv``: one row per question, configuration and system,
with the verdict, the probability, the truth and the token counts. It holds no
prompt or response text. Everything here is deterministic.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agon.questions import Question
from agon.runner import JobKey
from agon.scoring import bootstrap_ece, bootstrap_mean, brier, mcnemar_exact, wilson_interval
from agon.systems import Outcome

OUTCOME_FIELDS = (
    "qid",
    "template",
    "config",
    "system",
    "truth",
    "verdict",
    "p_true",
    "completion_tokens",
    "prompt_tokens",
    "calls",
    "truncated",
    "unparsed",
    "budget",
)
QUESTION_FIELDS = ("qid", "cik", "company", "accession", "template", "margin", "truth", "claim")

# Systems shown in the headline, in order. Mixed variants are averaged into these names.
HEADLINE = ("single", "long", "vote5", "vote6", "judge", "debate")
LABELS = {
    "closed_book": "closed book",
    "single": "single",
    "long": "single, long",
    "vote3": "vote of 3",
    "vote5": "vote of 5",
    "vote6": "vote of 6",
    "judge": "bull, bear, judge",
    "debate": "debate",
}
COMPARISONS = (
    ("debate", "vote6", "debate vs vote of 6 (same budget)"),
    ("debate", "vote5", "debate vs vote of 5"),
    ("debate", "vote3", "debate vs its own first round"),
    ("judge", "vote6", "bull, bear, judge vs vote of 6 (same budget)"),
    ("judge", "single", "bull, bear, judge vs single"),
    ("judge", "long", "bull, bear, judge vs single, long"),
    ("long", "single", "single, long vs single"),
    ("vote5", "single", "vote of 5 vs single"),
)
BOOT_SEED = 20261123

# Chart colours: one fixed hue per configuration, text in ink colours.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")


def write_outcomes(
    path: Path, questions: Sequence[Question], outcomes: Mapping[JobKey, Outcome]
) -> None:
    """Write the per-question outcome table."""
    by_id = {q.qid: q for q in questions}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTCOME_FIELDS)
        for (qid, config, system), o in outcomes.items():
            q = by_id[qid]
            verdict = "" if o.verdict is None else int(o.verdict)
            writer.writerow(
                [
                    qid,
                    q.template,
                    config,
                    system,
                    int(q.truth),
                    verdict,
                    f"{o.p_true:.4f}",
                    o.completion_tokens,
                    o.prompt_tokens,
                    o.calls,
                    o.truncated,
                    o.unparsed,
                    o.budget,
                ]
            )


def write_questions(path: Path, questions: Sequence[Question]) -> None:
    """Write the published question set: claims and answers, without the tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(QUESTION_FIELDS)
        for q in questions:
            writer.writerow(
                [q.qid, q.cik, q.company, q.accession, q.template, q.margin, int(q.truth), q.claim]
            )


def read_outcomes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Cell:
    """All questions for one configuration and one (possibly averaged) system."""

    def __init__(self) -> None:
        self.score: dict[str, list[float]] = defaultdict(list)
        self.p: dict[str, list[float]] = defaultdict(list)
        self.truth: dict[str, bool] = {}
        self.template: dict[str, str] = {}
        self.completion: list[float] = []
        self.prompt: list[float] = []
        self.calls: list[float] = []
        self.truncated = 0
        self.unparsed = 0
        self.budget = 0
        self.variants: set[str] = set()

    def add(self, row: Mapping[str, str]) -> None:
        qid = row["qid"]
        truth = row["truth"] == "1"
        self.truth[qid] = truth
        self.template[qid] = row["template"]
        self.score[qid].append(float(row["verdict"] != "" and (row["verdict"] == "1") == truth))
        self.p[qid].append(float(row["p_true"]))
        self.completion.append(float(row["completion_tokens"]))
        self.prompt.append(float(row["prompt_tokens"]))
        self.calls.append(float(row["calls"]))
        self.truncated += int(row["truncated"])
        self.unparsed += int(row["unparsed"])
        self.budget = max(self.budget, int(row["budget"]))
        self.variants.add(row["system"])

    @property
    def qids(self) -> list[str]:
        return sorted(self.score)

    def scores(self) -> list[float]:
        """Per-question score: 1 right, 0 wrong, or the mean over averaged variants."""
        return [float(np.mean(self.score[q])) for q in self.qids]

    def probabilities(self) -> tuple[list[float], list[bool]]:
        """Every (probability, truth) pair, one per variant per question."""
        ps: list[float] = []
        ys: list[bool] = []
        for q in self.qids:
            ps += self.p[q]
            ys += [self.truth[q]] * len(self.p[q])
        return ps, ys


def base_system(system: str) -> str:
    """``debate_aba`` and ``debate_bab`` both report as ``debate``."""
    return system.split("_")[0] if system != "closed_book" else system


def collect(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Cell]:
    cells: dict[tuple[str, str], Cell] = defaultdict(Cell)
    for row in rows:
        cells[row["config"], base_system(row["system"])].add(row)
    return dict(cells)


def configs_of(cells: Mapping[tuple[str, str], Cell]) -> list[str]:
    names = sorted({config for config, _ in cells})
    return [c for c in names if c != "mixed"] + (["mixed"] if "mixed" in names else [])


def summary_table(cells: Mapping[tuple[str, str], Cell]) -> list[dict[str, str]]:
    """Accuracy, calibration and cost for every configuration and system."""
    table: list[dict[str, str]] = []
    for config in configs_of(cells):
        for system in (
            "closed_book",
            "single",
            "long",
            "vote3",
            "vote5",
            "vote6",
            "judge",
            "debate",
        ):
            cell = cells.get((config, system))
            if cell is None:
                continue
            scores = cell.scores()
            n = len(scores)
            acc = float(np.mean(scores))
            if len(cell.variants) == 1:
                lo, hi = wilson_interval(sum(scores), n)
            else:
                _, lo, hi = bootstrap_mean(scores, seed=BOOT_SEED)
            ps, ys = cell.probabilities()
            sq = [(p - float(y)) ** 2 for p, y in zip(ps, ys, strict=True)]
            per_q = np.asarray(sq).reshape(n, -1).mean(axis=1)
            _, b_lo, b_hi = bootstrap_mean(list(per_q), seed=BOOT_SEED)
            e, e_lo, e_hi = bootstrap_ece(ps, ys, seed=BOOT_SEED)
            table.append(
                {
                    "config": config,
                    "system": system,
                    "variants": str(len(cell.variants)),
                    "n": str(n),
                    "accuracy": f"{acc:.3f}",
                    "acc_lo": f"{lo:.3f}",
                    "acc_hi": f"{hi:.3f}",
                    "brier": f"{brier(ps, ys):.3f}",
                    "brier_lo": f"{b_lo:.3f}",
                    "brier_hi": f"{b_hi:.3f}",
                    "ece": f"{e:.3f}",
                    "ece_lo": f"{e_lo:.3f}",
                    "ece_hi": f"{e_hi:.3f}",
                    "budget": str(cell.budget),
                    "completion_tokens": f"{np.mean(cell.completion):.0f}",
                    "prompt_tokens": f"{np.mean(cell.prompt):.0f}",
                    "calls": f"{np.mean(cell.calls):.0f}",
                    "truncated_calls": str(cell.truncated),
                    "unparsed_answers": str(cell.unparsed),
                }
            )
    return table


def comparison_table(cells: Mapping[tuple[str, str], Cell]) -> list[dict[str, str]]:
    """Paired accuracy differences on the same questions."""
    table: list[dict[str, str]] = []
    for config in configs_of(cells):
        for first, second, label in COMPARISONS:
            a, b = cells.get((config, first)), cells.get((config, second))
            if a is None or b is None:
                continue
            sa, sb = np.asarray(a.scores()), np.asarray(b.scores())
            diff, lo, hi = bootstrap_mean(list(sa - sb), seed=BOOT_SEED)
            binary = len(a.variants) == 1 and len(b.variants) == 1
            wins, losses = int(np.sum(sa > sb)), int(np.sum(sa < sb))
            table.append(
                {
                    "config": config,
                    "comparison": label,
                    "n": str(len(sa)),
                    "difference": f"{diff:+.3f}",
                    "lo": f"{lo:+.3f}",
                    "hi": f"{hi:+.3f}",
                    "first_better": str(wins),
                    "second_better": str(losses),
                    "mcnemar_p": f"{mcnemar_exact(wins, losses):.3f}" if binary else "",
                }
            )
    return table


def template_table(cells: Mapping[tuple[str, str], Cell]) -> list[dict[str, str]]:
    """Accuracy by claim template."""
    table: list[dict[str, str]] = []
    for config in configs_of(cells):
        for system in HEADLINE:
            cell = cells.get((config, system))
            if cell is None:
                continue
            groups: dict[str, list[float]] = defaultdict(list)
            for qid, score in zip(cell.qids, cell.scores(), strict=True):
                groups[cell.template[qid]].append(score)
            row = {"config": config, "system": system}
            for template in sorted(groups):
                row[template] = f"{np.mean(groups[template]):.3f}"
                row[f"n_{template}"] = str(len(groups[template]))
            table.append(row)
    return table


def flow_table(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """What the second debate round did to the first round's majority."""
    verdicts: dict[tuple[str, str, str], bool] = {}
    for row in rows:
        right = row["verdict"] != "" and row["verdict"] == row["truth"]
        verdicts[row["config"], row["system"], row["qid"]] = right
    table: list[dict[str, str]] = []
    pairs = sorted({(c, s) for c, s, _ in verdicts if s.startswith("debate")})
    for config, system in pairs:
        before_name = system.replace("debate", "vote3")
        counts = {"stayed_right": 0, "fixed": 0, "broken": 0, "stayed_wrong": 0}
        for (c, s, qid), after in verdicts.items():
            if (c, s) != (config, system):
                continue
            before = verdicts[c, before_name, qid]
            name = {
                (True, True): "stayed_right",
                (False, True): "fixed",
                (True, False): "broken",
                (False, False): "stayed_wrong",
            }[before, after]
            counts[name] += 1
        table.append({"config": config, "debate": system, **{k: str(v) for k, v in counts.items()}})
    return table


def write_table(path: Path, table: Sequence[Mapping[str, str]]) -> None:
    """Write a table as CSV and as a markdown file next to it."""
    if not table:
        return
    fields = list(table[0])
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    lines += ["| " + " | ".join(row.get(f, "") for f in fields) + " |" for row in table]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def short(config: str) -> str:
    """A model tag without its quantisation suffix, for chart titles."""
    return config if config == "mixed" else config.split(":")[0]


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_headline(path: Path, summary: Sequence[Mapping[str, str]]) -> None:
    """Accuracy of each system with its 95% interval, one panel per configuration."""
    configs = list(dict.fromkeys(row["config"] for row in summary))
    fig, axes = plt.subplots(1, len(configs), figsize=(3.6 * len(configs), 3.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, config in zip(np.atleast_1d(axes), configs, strict=True):
        rows = [r for r in summary if r["config"] == config and r["system"] in HEADLINE]
        rows.sort(key=lambda r: HEADLINE.index(r["system"]))
        xs = np.arange(len(rows))
        acc = np.array([float(r["accuracy"]) for r in rows]) * 100
        lo = np.array([float(r["acc_lo"]) for r in rows]) * 100
        hi = np.array([float(r["acc_hi"]) for r in rows]) * 100
        ax.errorbar(
            xs, acc, yerr=[acc - lo, hi - acc], fmt="o", color=SERIES[0], ecolor=SERIES[0],
            elinewidth=2, capsize=0, markersize=7,
        )  # fmt: skip
        for x, a, h in zip(xs, acc, hi, strict=True):
            ax.annotate(f"{a:.0f}", (x, h), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)  # fmt: skip
        ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xticks(xs)
        ticks = [LABELS[r["system"]].replace(", ", ",\n").replace(" of ", "\nof ") for r in rows]
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_title(f"{short(config)} (n = {rows[0]['n']})", fontsize=10, color=INK, loc="left")
        _style(ax)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("Accuracy, % of claims (95% interval)", fontsize=9, color=MUTED)
    first.set_ylim(40, 100)
    first.annotate("chance", (-0.5, 50), xytext=(0, 3), textcoords="offset points",
                   fontsize=8, color=MUTED)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_cost(path: Path, summary: Sequence[Mapping[str, str]]) -> None:
    """Accuracy against the generated tokens each system really used."""
    configs = list(dict.fromkeys(row["config"] for row in summary))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(SURFACE)
    for colour, config in zip(SERIES, configs, strict=False):
        rows = [r for r in summary if r["config"] == config and r["system"] != "closed_book"]
        xs = [float(r["completion_tokens"]) for r in rows]
        ys = [float(r["accuracy"]) * 100 for r in rows]
        ax.scatter(xs, ys, s=46, color=colour, label=short(config), edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)  # fmt: skip
        for r, x, y in zip(rows, xs, ys, strict=True):
            ax.annotate(LABELS[r["system"]], (x, y), xytext=(6, 3), textcoords="offset points",
                        fontsize=7.5, color=MUTED)  # fmt: skip
    ax.set_xlabel("Generated tokens per question (mean)", fontsize=9, color=MUTED)
    ax.set_ylabel("Accuracy, % of claims", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_calibration(path: Path, cells: Mapping[tuple[str, str], Cell]) -> None:
    """Stated confidence against accuracy, one panel per configuration."""
    configs = configs_of(cells)
    shown = ("single", "vote6", "judge", "debate")
    edges = np.linspace(0.5, 1.0, 6)
    fig, axes = plt.subplots(1, len(configs), figsize=(3.6 * len(configs), 3.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, config in zip(np.atleast_1d(axes), configs, strict=True):
        ax.plot([50, 100], [50, 100], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        for colour, system in zip(SERIES, shown, strict=True):
            cell = cells.get((config, system))
            if cell is None:
                continue
            ps, ys = cell.probabilities()
            p, y = np.asarray(ps), np.asarray(ys, dtype=bool)
            conf = np.maximum(p, 1 - p)
            right = np.where(p == 0.5, False, (p > 0.5) == y)
            index = np.clip(np.digitize(conf, edges[1:-1]), 0, 4)
            xs = [conf[index == b].mean() * 100 for b in range(5) if (index == b).sum() >= 5]
            zs = [right[index == b].mean() * 100 for b in range(5) if (index == b).sum() >= 5]
            ax.plot(xs, zs, marker="o", markersize=5, linewidth=2, color=colour,
                    label=LABELS[system])  # fmt: skip
        ax.set_title(short(config), fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Stated confidence, %", fontsize=9, color=MUTED)
        ax.set_xlim(48, 102)
        _style(ax)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("Accuracy in the bin, %", fontsize=9, color=MUTED)
    first.set_ylim(30, 102)
    first.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_run_info(path: Path, info: Mapping[str, object]) -> None:
    """Render run_info.json as a short markdown note."""
    lines = ["# Run information", ""]
    for key, value in info.items():
        if isinstance(value, dict):
            lines.append(f"- {key}:")
            lines += [f"  - {k}: {json.dumps(v) if isinstance(v, dict) else v}"
                      for k, v in value.items()]  # fmt: skip
        else:
            lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_report(results: Path) -> dict[str, list[dict[str, str]]]:
    """Rebuild every table and figure in ``results`` from ``outcomes.csv``."""
    rows = read_outcomes(results / "outcomes.csv")
    cells = collect(rows)
    tables = {
        "summary": summary_table(cells),
        "comparisons": comparison_table(cells),
        "by_template": template_table(cells),
        "debate_flow": flow_table(rows),
    }
    for name, table in tables.items():
        write_table(results / name, table)
    figure_headline(results / "fig_headline.png", tables["summary"])
    figure_cost(results / "fig_cost.png", tables["summary"])
    figure_calibration(results / "fig_calibration.png", cells)
    info = results / "run_info.json"
    if info.exists():
        write_run_info(results / "run_info.md", json.loads(info.read_text(encoding="utf-8")))
    return tables
