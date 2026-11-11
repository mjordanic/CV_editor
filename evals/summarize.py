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

It also writes RUNS_DIR/sessions_summary.json (`session_totals`): over all sessions, how many
actually redrafted (critique-driven redrafts and user-driven redrafts per document, from
events.jsonl via evals/iterations.py), the mean of the new rubric scores (visual layout, language,
additional information, user requests) and the user-request outcomes per script (followed, ignored,
untruthful requests refused or followed). Per script it also counts, from each session's script and
transcript alone, the scripted requests that were sent and those "not exercised" (never sent, so never
graded): those are not a result about the app, and a script with many needs its cases rerun.

It also writes RUNS_DIR/methods_summary.json (`method_summary`): the metrics split by method (the
manifest's "method": "single_prompt" from evals/single_prompt.py, else "multi_agent") and, for every
single-prompt run with a "baseline" label, the pair: its scores next to those of the multi-agent run it
is compared with. `--with OTHER_RUNS_DIR` (repeatable) adds the runs of other folders to this split, e.g.
the single-prompt runs folder next to the main round's.

Sessions interrupted because the provider ran out of credit (manifest status "interrupted: out of
credit") are left out of every number and listed under "interrupted".

Usage:
    .venv/bin/python -m evals.summarize RUNS_DIR [--out RUNS_DIR/repeats_summary.json] [--only-repeated]
        [--with OTHER_RUNS_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from evals.evaluate import scripted_request_status

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
    "visual_layout": lambda r: r["evaluation"]["visual_layout"]["score"],
    "language_quality": lambda r: r["evaluation"]["language_quality"]["score"],
    "additional_information": lambda r: r["evaluation"]["additional_information"]["score"],
    "user_requests": lambda r: r["evaluation"]["user_requests"]["score"],
    "untruthful_followed": lambda r: r["counts"]["untruthful_followed"],
    "requests_ignored": lambda r: r["counts"]["requests_ignored"],
}
REQUEST_COUNTS = ("requests", "requests_followed", "requests_partly", "requests_ignored", "requests_refused_wrongly",
                  "untruthful_refused", "untruthful_followed")
# How many scripted requests each session sent, by status (evals.evaluate.scripted_request_status).
SCRIPTED_STATUS_COUNTS = {"sent": "scripted_sent", "replaced": "scripted_replaced",
                          "not exercised": "scripted_not_exercised"}
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


def _manifest(run: Path) -> dict:
    try:
        return json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def is_interrupted(run: Path) -> bool:
    return str(_manifest(run).get("status") or "").startswith("interrupted")


def run_folders(runs_dir: Path) -> tuple[list[Path], list[str]]:
    """(run folders to count, names of interrupted runs left out)."""
    runs = sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "manifest.json").is_file())
    return [r for r in runs if not is_interrupted(r)], [r.name for r in runs if is_interrupted(r)]


def session_totals(runs_dir: Path) -> dict:
    """Redrafts and user-request outcomes over every (not interrupted) session of a runs folder."""
    from evals.iterations import KINDS, redrafts
    runs, interrupted = run_folders(runs_dir)
    out = {"sessions": len(runs), "interrupted": interrupted, "redrafts": {}, "by_script": {}}
    per_run = {r.name: redrafts(r) for r in runs}
    for kind in KINDS:
        out["redrafts"][kind] = {
            "sessions_with_a_draft": sum(1 for d in per_run.values() if d[kind]["drafts"]),
            "sessions_with_critique_redraft": sum(1 for d in per_run.values() if d[kind]["critique_redrafts"]),
            "sessions_with_user_redraft": sum(1 for d in per_run.values() if d[kind]["user_redrafts"]),
            "critique_redrafts": sum(d[kind]["critique_redrafts"] for d in per_run.values()),
            "critique_skipped": sum(d[kind]["critique_skipped"] for d in per_run.values()),
            "user_redrafts": sum(d[kind]["user_redrafts"] for d in per_run.values()),
            "removal_passes": sum(d[kind]["removal_passes"] for d in per_run.values()),
        }
    out["redrafts"]["sessions_with_any_critique_redraft"] = sum(
        1 for d in per_run.values() if any(d[k]["critique_redrafts"] for k in KINDS))
    for run in runs:
        script = (_manifest(run).get("labels") or {}).get("script") or "default"
        bucket = out["by_script"].setdefault(script, {"sessions": 0, "evaluated": 0, **{k: 0 for k in REQUEST_COUNTS},
                                                      **{k: 0 for k in SCRIPTED_STATUS_COUNTS.values()}})
        bucket["sessions"] += 1
        for item in scripted_request_status(run):
            bucket[SCRIPTED_STATUS_COUNTS[item["status"]]] += 1
        for f in sorted(run.glob("evaluation/*/evaluation.json"))[:1]:  # the first evaluation of each run
            counts = json.loads(f.read_text(encoding="utf-8")).get("counts") or {}
            bucket["evaluated"] += 1
            for k in REQUEST_COUNTS:
                bucket[k] += counts.get(k) or 0
    return out


def summarize(runs_dir: Path) -> dict:
    """{case: {"runs": [run ids], "metrics": {metric: {reps, mean, sd, min, max, eval_sd}}}}.
    Interrupted (out-of-credit) runs are left out."""
    groups: dict[str, dict[str, dict]] = {}
    for run in run_folders(runs_dir)[0]:
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


def method_of(run: Path) -> str:
    """single_prompt (evals/single_prompt.py) or multi_agent (evals/run_case.py, whose manifest has no method)."""
    manifest = _manifest(run)
    return manifest.get("method") or (manifest.get("labels") or {}).get("method") or "multi_agent"


def _run_means(run: Path) -> dict:
    """{metric: mean over the run's evaluations}."""
    return {m: round(statistics.fmean(v), 3) for m, v in read_run(run).items() if v}


def method_summary(runs_dirs: list[Path]) -> dict:
    """The metrics split by method over every (not interrupted) run of the folders, and the like-for-like pairs:
    each single-prompt run with a "baseline" label next to that multi-agent run (when it is in the folders)."""
    runs = [r for d in runs_dirs for r in run_folders(d)[0]]
    by_method: dict[str, dict] = {}
    for run in runs:
        bucket = by_method.setdefault(method_of(run), {"runs": 0, "evaluated": 0, "values": {}})
        bucket["runs"] += 1
        means = _run_means(run)
        bucket["evaluated"] += bool(means)
        for m, v in means.items():
            bucket["values"].setdefault(m, []).append(v)
    methods = {
        method: {"runs": b["runs"], "evaluated": b["evaluated"],
                 "metrics": {m: {"n": len(v), "mean": round(statistics.fmean(v), 3), "sd": _sd(v)}
                             for m, v in sorted(b["values"].items(), key=lambda kv: _metric_order(kv[0]))}}
        for method, b in sorted(by_method.items())
    }
    by_name = {r.name: r for r in runs}
    pairs = []
    for run in runs:
        baseline = (_manifest(run).get("labels") or {}).get("baseline")
        if method_of(run) != "single_prompt" or not baseline:
            continue
        other = by_name.get(baseline)
        pair = {"single_prompt": run.name, "multi_agent": baseline, "found": other is not None,
                "single_prompt_metrics": _run_means(run), "multi_agent_metrics": _run_means(other) if other else {}}
        pair["diff"] = {m: round(v - pair["multi_agent_metrics"][m], 3) for m, v in pair["single_prompt_metrics"].items()
                        if m in pair["multi_agent_metrics"]}
        pairs.append(pair)
    paired = [p for p in pairs if p["diff"]]
    mean_diff = {}
    for m in {m for p in paired for m in p["diff"]}:
        values = [p["diff"][m] for p in paired if m in p["diff"]]
        mean_diff[m] = {"n": len(values), "mean": round(statistics.fmean(values), 3), "sd": _sd(values)}
    return {"methods": methods, "pairs": pairs,
            "paired_mean_diff": dict(sorted(mean_diff.items(), key=lambda kv: _metric_order(kv[0])))}


def format_methods(summary: dict, metrics=("overall", "invented_content", "completeness", "fit", "cv_unsupported",
                                          "visual_layout")) -> str:
    rows = [["method", "runs", *metrics]]
    for method, data in summary["methods"].items():
        cell = lambda s: "-" if not s else f"{s['mean']:g}" + (f"±{s['sd']:g}" if s["sd"] is not None else "")  # noqa: E731
        rows.append([method, str(data["runs"]), *(cell(data["metrics"].get(m)) for m in metrics)])
    if summary["paired_mean_diff"]:
        d = summary["paired_mean_diff"]
        n = max(s["n"] for s in d.values())
        rows.append([f"pairs: sp-ma (n={n})", "", *(f"{d[m]['mean']:+g}" if m in d else "-" for m in metrics)])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


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
    parser.add_argument("--with", dest="with_dirs", type=Path, action="append", default=[],
                        help="another runs folder whose runs join the split by method (repeatable)")
    args = parser.parse_args(argv)
    summary = summarize(args.runs_dir)
    if args.only_repeated:
        summary = {c: d for c, d in summary.items()
                   if len(d["runs"]) >= 2 or any(s["evals_per_run"] >= 2 for s in d["metrics"].values())}
    out = args.out or args.runs_dir / "repeats_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(format_table(summary))
    totals = session_totals(args.runs_dir)
    (args.runs_dir / "sessions_summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    r = totals["redrafts"]
    print(f"\n{totals['sessions']} sessions; critique-driven redraft: CV in {r['cv']['sessions_with_critique_redraft']}, "
          f"cover letter in {r['cover_letter']['sessions_with_critique_redraft']}, either in "
          f"{r['sessions_with_any_critique_redraft']}; user-driven redraft: CV in {r['cv']['sessions_with_user_redraft']}, "
          f"cover letter in {r['cover_letter']['sessions_with_user_redraft']}")
    unsent = {name: b["scripted_not_exercised"] for name, b in totals["by_script"].items() if b["scripted_not_exercised"]}
    if unsent:
        print("scripted requests not exercised (never sent, not graded): "
              + ", ".join(f"{name} {n}" for name, n in sorted(unsent.items())))
    methods = method_summary([args.runs_dir, *args.with_dirs])
    (args.runs_dir / "methods_summary.json").write_text(json.dumps(methods, indent=2), encoding="utf-8")
    if len(methods["methods"]) > 1 or "single_prompt" in methods["methods"]:
        print("\nby method (mean±sd over runs; pairs: single-prompt minus its multi-agent baseline)")
        print(format_methods(methods))
    if totals["interrupted"]:
        print(f"left out (interrupted, out of credit): {', '.join(totals['interrupted'])}")
    print("mean±sd across repeats; (ev±x) = mean spread between evaluator re-scores of the same run", file=sys.stderr)
    print(f"written to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
