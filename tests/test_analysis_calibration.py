"""Offline tests for the section-6 grader calibration checks (evals/analysis/calibration.py): small synthetic grading
rows for pass and fail cases, and the published pilot round 2026-11-11 against the pilot gate's numbers."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evals.analysis import calibration, load

PRIMARY, SECOND = "sonnet", "opus"
A, B = "multi_agent/gpt-6-luna", "single_prompt/gpt-6-luna"


def _grading(run, arm, persona, eval_name, grader, comp, overall=None, flagged=0):
    """One grading row whose composite sub-scores all equal `comp` (so the composite is `comp`)."""
    return {"run_id": run, "arm": arm, "persona": persona, "ad": "a1", "input_set": "all4", "script": "default",
            "work_preference": "", "rep": 1, "eval_name": eval_name, "grader": grader,
            "overall_score": comp if overall is None else overall, **{c: comp for c in load.COMPOSITE},
            "n_unsupported": flagged, "n_partly": 0}


def _round(primary_a, primary_b, second_a, second_b, regrade_delta=0.0, overall_delta=0):
    """Rows for cells p0, p1, ...: arm A and B sessions graded by the primary (eval1, plus eval2 shifted by the
    deltas) and by the second grader (opus1). Arguments are the per-cell composites."""
    rows = []
    for i, (pa, pb, sa, sb) in enumerate(zip(primary_a, primary_b, second_a, second_b)):
        for arm, p, s in ((A, pa, sa), (B, pb, sb)):
            run = f"{arm.split('/')[0]}_p{i}"
            rows.append(_grading(run, arm, f"p{i}", "eval1", PRIMARY, p, flagged=1))
            rows.append(_grading(run, arm, f"p{i}", "eval2", PRIMARY, p + regrade_delta, overall=p + overall_delta))
            rows.append(_grading(run, arm, f"p{i}", "opus1", SECOND, s, flagged=i % 2))
    return rows


def _by_check(table):
    return {r["check"]: r for r in table}


def test_composite_is_the_mean_of_its_subscores_and_none_when_one_is_missing():
    row = {c: 8 for c in load.COMPOSITE}
    row["fit"] = 1
    assert load.composite(row) == pytest.approx((8 * 6 + 1) / 7)
    row["visual_layout"] = None
    assert load.composite(row) is None


def test_every_check_passes_when_graders_agree():
    rows = _round([6, 7, 8, 5], [7, 8, 9, 6], [6, 7, 8, 5], [7, 8, 9, 6], regrade_delta=0.2, overall_delta=1)
    t = _by_check(calibration.section6_checks(rows, PRIMARY, SECOND, [(A, B, "q", True)]))
    assert t["primary self-agreement (composite)"]["result"] == f"{(0.2 ** 2 / 2) ** 0.5:.3f}"
    assert t["primary self-agreement (composite)"]["passed"] is True
    assert t["primary self-agreement (overall)"]["result"] == "100% (8/8)"
    assert t["primary self-agreement (overall)"]["passed"] is True
    assert t["primary vs second (rank agreement)"]["passed"] is True
    bias = t[f"no bias by arm: {B} - {A}"]
    assert bias["passed"] is True and bias["result"].startswith("gap 0.000; same sign 100% (4/4")
    audit = t["claim audit"]
    assert audit["passed"] is None and audit["result"].startswith("4 claims in 4 of 8 runs")


def test_noisy_primary_and_disagreeing_graders_fail():
    # the primary re-grades 0.5 apart and 2 overall points apart; the second grader reverses the ranking and
    # sees arm B as worse than A
    rows = _round([6, 7, 8, 5], [7, 8, 9, 6], [8, 7, 6, 9], [7, 6, 5, 8], regrade_delta=0.5, overall_delta=2)
    t = _by_check(calibration.section6_checks(rows, PRIMARY, SECOND, [(A, B, "q", True)]))
    assert t["primary self-agreement (composite)"]["passed"] is False      # sqrt(0.25/2) = 0.354 > 0.3
    assert t["primary self-agreement (overall)"]["result"] == "0% (0/8)"
    assert t["primary self-agreement (overall)"]["passed"] is False
    assert t["primary vs second (rank agreement)"]["passed"] is False
    bias = t[f"no bias by arm: {B} - {A}"]
    assert bias["passed"] is False and "gap 2.000; same sign 0% (0/4" in bias["result"]


def test_arm_bias_fails_on_sign_disagreement_even_with_a_small_gap():
    # mean differences +0.5 under both graders (gap 0) but the signs agree in only 2 of 4 cells
    rows = _round([5, 5, 5, 5], [6, 6, 4.5, 5.5], [5, 5, 5, 5], [4.5, 5.5, 6, 6])
    bias = _by_check(calibration.section6_checks(rows, PRIMARY, SECOND, [(A, B)]))[f"no bias by arm: {B} - {A}"]
    assert bias["result"].startswith("gap 0.000; same sign 50% (2/4") and bias["passed"] is False


def test_arm_bias_without_non_tied_cells_passes_on_the_gap_alone():
    rows = _round([5, 6], [5, 6], [5, 6], [5, 6])
    bias = _by_check(calibration.section6_checks(rows, PRIMARY, SECOND, [(A, B)]))[f"no bias by arm: {B} - {A}"]
    assert "no cell non-tied" in bias["result"] and bias["passed"] is True


def test_first_grading_is_the_lowest_evaluation_name():
    # the second grading (eval2) disagrees completely, but checks 3 and 4 only use eval1
    rows = _round([6, 7, 8, 5], [7, 8, 9, 6], [6, 7, 8, 5], [7, 8, 9, 6], regrade_delta=3)
    t = _by_check(calibration.section6_checks(list(reversed(rows)), PRIMARY, SECOND, [(A, B)]))
    assert t["primary vs second (rank agreement)"]["result"] == "1.000"
    assert t[f"no bias by arm: {B} - {A}"]["result"].startswith("gap 0.000")
    assert calibration._eval_order("eval10") > calibration._eval_order("eval2")


def test_checks_without_data_say_so_and_do_not_pass_or_fail():
    rows = [r for r in _round([6, 7], [7, 8], [6, 7], [7, 8]) if r["eval_name"] == "eval1"]
    t = calibration.section6_checks(rows, PRIMARY, None, [(A, B), (A, "single_prompt/absent")])
    assert [r["check"] for r in t] == ["primary self-agreement (composite)", "primary self-agreement (overall)",
                                       "primary vs second (rank agreement)", f"no bias by arm: {B} - {A}",
                                       "claim audit"]
    assert all(r["passed"] is None and r["result"].startswith("no data") for r in t)


def test_pilot_round_reproduces_the_pilot_gate():
    rows = load.read_rows(REPO / "evals" / "results" / "2026-11-11")
    contrasts = [(A, B), (A, "single_prompt/gpt-6.1-sol"), ("multi_agent/gpt-6.1-terra", "single_prompt/gpt-6.1-sol")]
    t = calibration.section6_checks(rows, "claude-sonnet-5-5", "claude-opus-5-5", contrasts)
    by = _by_check(t)
    assert len(t) == 6   # no multi-agent terra sessions in the pilot: that contrast has no row
    assert by["primary self-agreement (composite)"]["result"] == "0.195"
    assert by["primary self-agreement (overall)"]["result"] == "100% (18/18)"
    assert by["primary vs second (rank agreement)"]["result"] == "0.781"
    assert by[f"no bias by arm: {B} - {A}"]["result"].startswith("gap 0.214")
    assert by[f"no bias by arm: single_prompt/gpt-6.1-sol - {A}"]["result"].startswith("gap 0.214")
    assert by["claim audit"]["result"].startswith("7 claims in 5 of 18 runs")
    assert all(r["passed"] is True for r in t if r["check"] != "claim audit")
