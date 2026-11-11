"""Grader calibration checks of the plan's section 6, on the composite score. Standard library and numpy only.

The *primary* grader grades every session; a *second* grader grades a subset (the calibration partner). The checks:

1. primary self-agreement, composite: grader SD = sqrt(mean(d^2) / 2) over runs the primary graded twice, <= 0.3;
2. primary self-agreement, overall: share of those runs whose two overall scores are within 1 point, >= 90%;
3. primary vs second: Spearman of the composite on the runs both graded, >= 0.7;
4. no bias by arm: per contrast (A, B), the mean per-cell difference B - A under each grader differs by <= 0.3 and
   the per-cell signs agree in >= 80% of the cells non-tied under both graders;
5. claim audit: >= 80% of the claims the second grader flags are also found by the primary. A person reads them,
   so this row only counts the flagged claims and never passes or fails by itself.

Checks 3-5 use the *first* grading of each grader per run (the lowest evaluation name: eval1, not eval2), on the
runs graded by both. Every check returns one row {check, measure, result, threshold, passed}; a check without data
says so in `result` and has passed None.
"""
from __future__ import annotations

import math
import re
import statistics

import numpy as np

from . import load, stats
from .load import COMPOSITE

SD_MAX = 0.3            # check 1: composite grader SD
WITHIN_1_MIN = 0.9      # check 2: overall scores within one point
SPEARMAN_MIN = 0.7      # check 3
GAP_MAX = 0.3           # check 4: |mean diff under primary - mean diff under second|
SAME_SIGN_MIN = 0.8     # check 4: per-cell sign agreement
_EPS = 1e-9             # so that a value equal to a threshold is not lost to rounding


def composite_of(row: dict) -> float | None:
    """The row's composite (load.read_rows sets it), computed from the sub-scores when the row has none."""
    return row["composite"] if "composite" in row else load.composite(row)


def _value(row: dict, metric: str) -> float | None:
    v = composite_of(row) if metric == "composite" else row.get(metric)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _eval_order(name: str) -> tuple:
    """eval1 < eval2 < eval10 (natural order of evaluation names)."""
    m = re.match(r"(.*?)(\d+)$", name or "")
    return (m.group(1), int(m.group(2))) if m else (name or "", -1)


def gradings_by_run(rows: list[dict], grader: str) -> dict[str, list[dict]]:
    """{run_id: that grader's grading rows of the run, in evaluation-name order}."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        if grader and r.get("grader") == grader:
            out.setdefault(r["run_id"], []).append(r)
    return {run: sorted(g, key=lambda r: _eval_order(r.get("eval_name"))) for run, g in out.items()}


def first_gradings(rows: list[dict], grader: str) -> dict[str, dict]:
    """{run_id: the grader's first grading of the run (lowest evaluation name)}."""
    return {run: g[0] for run, g in gradings_by_run(rows, grader).items()}


def _row(check, measure, result, threshold, passed):
    return {"check": check, "measure": measure, "result": result, "threshold": threshold, "passed": passed}


def _self_agreement(rows, primary):
    """Checks 1 and 2 on the first two gradings of every run the primary graded twice or more."""
    twice = [g[:2] for g in gradings_by_run(rows, primary).values() if len(g) >= 2]
    comp = [(a, b) for a, b in ((_value(x, "composite"), _value(y, "composite")) for x, y in twice)
            if a is not None and b is not None]
    over = [(a, b) for a, b in ((_value(x, "overall_score"), _value(y, "overall_score")) for x, y in twice)
            if a is not None and b is not None]
    out = []
    check = "primary self-agreement (composite)"
    measure = f"grader SD of the composite, sqrt(mean(d^2)/2), over runs graded twice by {primary}"
    if comp:
        sd = math.sqrt(statistics.fmean((a - b) ** 2 for a, b in comp) / 2)
        out.append(_row(check, f"{measure} ({len(comp)} runs)", f"{sd:.3f}", f"<= {SD_MAX}", sd <= SD_MAX + _EPS))
    else:
        out.append(_row(check, measure, f"no data: no run graded twice by {primary}", f"<= {SD_MAX}", None))
    check = "primary self-agreement (overall)"
    measure = f"share of runs graded twice by {primary} whose two overall scores are within 1 point"
    if over:
        k = sum(abs(a - b) <= 1 for a, b in over)
        share = k / len(over)
        out.append(_row(check, measure, f"{share:.0%} ({k}/{len(over)})", f">= {WITHIN_1_MIN:.0%}",
                        share >= WITHIN_1_MIN - _EPS))
    else:
        out.append(_row(check, measure, f"no data: no run graded twice by {primary}", f">= {WITHIN_1_MIN:.0%}", None))
    return out


def _spearman(first_p, first_s, primary, second):
    check = "primary vs second (rank agreement)"
    measure = f"Spearman of the composite, {primary} vs {second}, first grading each, on runs graded by both"
    both = [(_value(first_p[r], "composite"), _value(first_s[r], "composite")) for r in sorted(set(first_p) & set(first_s))]
    both = [(a, b) for a, b in both if a is not None and b is not None]
    if not both:
        return _row(check, measure, f"no data: no run graded by both {primary} and {second}", f">= {SPEARMAN_MIN}", None)
    ag = stats.agreement([a for a, _ in both], [b for _, b in both])
    rho = ag.get("spearman")
    if rho is None or math.isnan(rho):
        return _row(check, f"{measure} ({len(both)} runs)", "no data: not computable (too few runs or no spread)",
                    f">= {SPEARMAN_MIN}", None)
    return _row(check, f"{measure} ({len(both)} runs)", f"{rho:.3f}", f">= {SPEARMAN_MIN}", rho >= SPEARMAN_MIN - _EPS)


def _arm_bias(first_p, first_s, contrast, primary, second):
    a, b = contrast[0], contrast[1]
    check = f"no bias by arm: {b} - {a}"
    threshold = f"gap <= {GAP_MAX} and same-sign share >= {SAME_SIGN_MIN:.0%}"
    runs = set(first_p) & set(first_s)
    cells = {}
    for name, first in ((primary, first_p), (second, first_s)):
        recs = [{**first[r], "composite": _value(first[r], "composite")} for r in runs]
        cells[name] = {x["cell"]: x["diff"] for x in load.pair_arms(recs, a, b, "composite")["pairs"]}
    common = sorted(set(cells[primary]) & set(cells[second]))
    measure = "mean per-cell difference of the composite under each grader, first gradings, runs graded by both"
    if not common:
        return _row(check, measure, "no data: no cell with both arms graded by both graders", threshold, None)
    dp = [cells[primary][c] for c in common]
    ds = [cells[second][c] for c in common]
    mp, ms = statistics.fmean(dp), statistics.fmean(ds)
    gap = abs(mp - ms)
    non_tied = [(x, y) for x, y in zip(dp, ds) if x != 0 and y != 0]
    same = sum(bool(np.sign(x) == np.sign(y)) for x, y in non_tied)
    share = same / len(non_tied) if non_tied else None
    passed = gap <= GAP_MAX + _EPS and (share is None or share >= SAME_SIGN_MIN - _EPS)
    measure += f" ({len(common)} cells): {primary} {mp:+.3f}, {second} {ms:+.3f}"
    signs = (f"same sign {share:.0%} ({same}/{len(non_tied)} non-tied cells)" if non_tied
             else "no cell non-tied under both graders")
    return _row(check, measure, f"gap {gap:.3f}; {signs}", threshold, passed)


def _claim_audit(first_p, first_s, primary, second):
    check = "claim audit"
    threshold = ">= 80% also found by primary (a person reads them)"
    runs = sorted(set(first_p) & set(first_s))

    def flagged(first):
        counts = [(first[r].get("n_unsupported"), first[r].get("n_partly")) for r in runs]
        counts = [(u or 0) + (p or 0) for u, p in counts if u is not None or p is not None]
        return sum(counts), sum(c > 0 for c in counts), len(counts)

    n_s, runs_s, known = flagged(first_s)
    measure = f"unsupported + partly-supported claims {second} flagged on the runs graded by both"
    if not known:
        return _row(check, measure, f"no data: no claim counts from {second} on runs graded by both", threshold, None)
    n_p, _, _ = flagged(first_p)
    return _row(check, measure, f"{n_s} claims in {runs_s} of {known} runs ({primary} flagged {n_p})", threshold, None)


def section6_checks(rows: list[dict], primary: str, second: str | None, contrasts) -> list[dict]:
    """The section-6 calibration table: one row per check (see the module docstring). `contrasts` are (A, B, ...)
    tuples, e.g. the notebook's CONTRASTS; only those whose two arms both appear in `rows` get a row."""
    out = _self_agreement(rows, primary)
    first_p = first_gradings(rows, primary)
    first_s = first_gradings(rows, second) if second else {}
    second_name = second or "(no second grader)"
    out.append(_spearman(first_p, first_s, primary, second_name))
    present = {r["arm"] for r in rows}
    for c in contrasts:
        if c[0] in present and c[1] in present:
            out.append(_arm_bias(first_p, first_s, c, primary, second_name))
    out.append(_claim_audit(first_p, first_s, primary, second_name))
    return out
