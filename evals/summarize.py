"""Spread of the scores across repeats of a case and across evaluator re-scores of a run.

Reads a runs folder (as run_matrix writes it) and makes no model calls. Runs are grouped by case id
without the trailing _rN. For every metric it reports:

    reps        how many runs of the case have the metric
    mean, sd    across those runs (each run counts once: the mean of its evaluations eval1, eval2, ...)
    eval_sd     the mean, over runs with two or more evaluations, of the spread between them
                (how much the evaluator alone moves the score)

Metrics: overall, invented-content, completeness, fit, CV-standards and cover-letter scores, the
evaluator's counts (unsupported and stretched claims, high-relevance omissions, overstated gaps,
invented work arrangement), and the truth check's counts when a truth_check.json exists.

Usage:
    .venv/bin/python -m evals.summarize RUNS_DIR [--out RUNS_DIR/repeats_summary.json] [--only-repeated]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_REP_SUFFIX = re.compile(r"_r\d+$")

EVAL_METRICS = {
    "overall": lambda r: r["evaluation"]["overall_score"],
    "invented_content": lambda r: r["evaluation"]["invented_content_score"],
    "completeness": lambda r: r["evaluation"]["completeness_score"],
    "fit": lambda r: r["evaluation"]["fit"]["score"],
    "cv_standards": lambda r: r["evaluation"]["cv_standards"]["score"],
    "cover_letter": lambda r: r["evaluation"]["cover_letter"]["score"],
    "cv_unsupported": lambda r: r["counts"]["cv_unsupported"],
    "cv_partly": lambda r: r["counts"]["cv_partly"],
    "cl_unsupported": lambda r: r["counts"]["cl_unsupported"],
    "cl_partly": lambda r: r["counts"]["cl_partly"],
    "omissions_high": lambda r: r["counts"]["omissions_high"],
    "gaps_overstated": lambda r: r["counts"]["gaps_overstated"],
    "work_arrangement_invented": lambda r: r["counts"]["work_arrangement_invented"],
}
TRUTH_METRICS = ("gaps_claimed", "traps_hit", "unclear", "unverified_quotes")


def _get(fn, record):
    try:
        value = fn(record)
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _sd(values: list[float]) -> float | None:
    return round(statistics.stdev(values), 3) if len(values) >= 2 else None


def read_run(run_dir: Path) -> dict:
    """{metric: [value per evaluation or truth check]} for one run folder."""
    values: dict[str, list[float]] = {}
    for f in sorted(run_dir.glob("evaluation/*/evaluation.json")):
        record = json.loads(f.read_text(encoding="utf-8"))
        for name, fn in EVAL_METRICS.items():
            v = _get(fn, record)
            if v is not None:
                values.setdefault(name, []).append(v)
    for f in sorted(run_dir.glob("evaluation/*/truth_check.json")):
        counts = json.loads(f.read_text(encoding="utf-8")).get("counts") or {}
        for name in TRUTH_METRICS:
            if isinstance(counts.get(name), (int, float)):
                values.setdefault(f"truth_{name}", []).append(counts[name])
    return values


def summarize(runs_dir: Path) -> dict:
    """{case: {"runs": [run ids], "metrics": {metric: {reps, mean, sd, min, max, eval_sd}}}}."""
    groups: dict[str, dict[str, dict]] = {}
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "manifest.json").is_file()):
        groups.setdefault(_REP_SUFFIX.sub("", run.name), {})[run.name] = read_run(run)
    out = {}
    for case, runs in groups.items():
        metrics = {}
        names = sorted({m for values in runs.values() for m in values}, key=_metric_order)
        for m in names:
            per_run = [statistics.fmean(v[m]) for v in runs.values() if v.get(m)]
            within = [_sd(v[m]) for v in runs.values() if len(v.get(m) or []) >= 2]
            metrics[m] = {
                "reps": len(per_run),
                "mean": round(statistics.fmean(per_run), 3),
                "sd": _sd(per_run),
                "min": min(per_run), "max": max(per_run),
                "eval_sd": round(statistics.fmean(within), 3) if within else None,
                "evals_per_run": max(len(v.get(m) or []) for v in runs.values()),
            }
        out[case] = {"runs": sorted(runs), "metrics": metrics}
    return out


def _metric_order(name: str) -> int:
    order = [*EVAL_METRICS, *(f"truth_{m}" for m in TRUTH_METRICS)]
    return order.index(name) if name in order else len(order)


def format_table(summary: dict, metrics=("overall", "invented_content", "completeness", "fit", "cv_unsupported",
                                         "truth_gaps_claimed", "truth_traps_hit")) -> str:
    def cell(stats):
        if not stats:
            return "-"
        text = f"{stats['mean']:g}"
        if stats["sd"] is not None:
            text += f"±{stats['sd']:g}"
        if stats["eval_sd"] is not None:
            text += f" (ev±{stats['eval_sd']:g})"
        return text
    header = ["case", "reps", *metrics]
    rows = [header]
    for case, data in summary.items():
        reps = max((s["reps"] for s in data["metrics"].values()), default=0)
        rows.append([case, str(reps), *(cell(data["metrics"].get(m)) for m in metrics)])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, help="JSON file to write (default RUNS_DIR/repeats_summary.json)")
    parser.add_argument("--only-repeated", action="store_true",
                        help="only cases with two or more runs or a run with two or more evaluations")
    args = parser.parse_args(argv)
    summary = summarize(args.runs_dir)
    if args.only_repeated:
        summary = {c: d for c, d in summary.items()
                   if len(d["runs"]) >= 2 or any(s["evals_per_run"] >= 2 for s in d["metrics"].values())}
    out = args.out or args.runs_dir / "repeats_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(format_table(summary))
    print("mean±sd across repeats; (ev±x) = mean spread between evaluator re-scores of the same run", file=sys.stderr)
    print(f"written to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
