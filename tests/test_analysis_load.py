"""Offline tests for the analysis notebook's loading, pairing and statistics helpers (evals/analysis).
A tiny runs folder is written to tmp_path: two arms, repeats, re-gradings, a second grader, cells only one
arm has, and an interrupted session. No model calls; numpy is the only third-party import."""
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evals.analysis import load, stats

MA, SP = "multi_agent/gpt-6-luna", "single_prompt/gpt-6.1-sol"
SOL, OTHER = "gpt-6.1-sol", "grader-b"


def _run(runs: Path, name: str, labels: dict, gradings: dict, status="completed", method=None, cost=0.03, ats=None):
    folder = runs / name
    folder.mkdir(parents=True)
    manifest = {"status": status, "labels": {"script": "default", **labels}, "cost_usd": cost, "minutes": 2.0}
    if method:
        manifest["method"] = method
        manifest["labels"]["method"] = method
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "summary.json").write_text(json.dumps({"llm_calls": 1 if method else 22, "llm_errors": 0}))
    for ev_name, (grader, overall) in gradings.items():
        d = folder / "evaluation" / ev_name
        d.mkdir(parents=True)
        evaluation = {"overall_score": overall, "invented_content_score": 10, "completeness_score": 8,
                      "fit": {"score": overall - 1}, "visual_layout": {"score": 7}, "ats": ats}
        counts = {"claims": 50, "cv_unsupported": 1 if overall < 7 else 0, "cl_unsupported": 0,
                  "cv_partly": 1, "cl_partly": 1}
        (d / "evaluation.json").write_text(json.dumps({"evaluator_model": grader, "name": ev_name,
                                                       "evaluation": evaluation, "counts": counts,
                                                       "cost_usd": 0.1}))


@pytest.fixture
def round_dir(tmp_path):
    runs = tmp_path / "round" / "runs"
    cell = {"persona": "p01", "ad": "a1", "input_set": "all4"}
    _run(runs, "p01_a1_all4_luna_r1", {**cell, "model": "openai:gpt-6-luna", "rep": "1"},
         {"eval1": (f"openai:{SOL}", 7), "eval2": (f"openai:{SOL}", 8), "b1": (f"other:{OTHER}", 6)},
         ats={"ad_keywords": list("abcdefghij"), "keywords_in_cv": list("abcd"),
              "keywords_missing_but_candidate_has": ["e"], "keyword_coverage_pct": 40, "pass_likelihood": "medium"})
    _run(runs, "p01_a1_all4_luna_r2", {**cell, "model": "openai:gpt-6-luna", "rep": "2"},
         {"eval1": (f"openai:{SOL}", 9), "b1": (f"other:{OTHER}", 8)})
    _run(runs, "sp_p01_a1_all4_sol_r1", {**cell, "model": "openai:gpt-6.1-sol", "rep": "1",
                                         "baseline": "p01_a1_all4_luna_r1"},
         {"eval1": (f"openai:{SOL}", 8)}, method="single_prompt", cost=0.11)
    # a cell only the multi-agent arm has, and the same persona/ad/inputs with another script
    _run(runs, "p02_a5_cv_only_luna_r1", {"persona": "p02", "ad": "a5", "input_set": "cv_only",
                                          "model": "openai:gpt-6-luna", "rep": "1"}, {"eval1": (f"openai:{SOL}", 6)})
    _run(runs, "p01_a1_all4_trap-p01_luna_r1", {**cell, "script": "trap-p01", "model": "openai:gpt-6-luna",
                                                "rep": "1"}, {"eval1": (f"openai:{SOL}", 5)})
    # a cell only the single-prompt arm has, and an interrupted single-prompt session
    _run(runs, "sp_p04_a6_all4_sol_r1", {"persona": "p04", "ad": "a6", "input_set": "all4",
                                         "model": "openai:gpt-6.1-sol", "rep": "1"},
         {"eval1": (f"openai:{SOL}", 8)}, method="single_prompt")
    _run(runs, "sp_p03_a3_no_cv_sol_r1", {"persona": "p03", "ad": "a3", "input_set": "no_cv",
                                          "model": "openai:gpt-6.1-sol", "rep": "1"},
         {}, status="interrupted: out of credit", method="single_prompt")
    return tmp_path / "round"


def test_rows_are_one_per_grading_with_arms_and_interrupted_sessions_left_out(round_dir):
    rows = load.read_rows(round_dir)
    assert load.runs_dir_of(round_dir) == round_dir / "runs"
    assert len({r["run_id"] for r in rows}) == 6 and len(rows) == 9
    assert {r["arm"] for r in rows} == {MA, SP}
    assert {r["grader"] for r in rows} == {SOL, OTHER}
    sp = next(r for r in rows if r["run_id"] == "sp_p01_a1_all4_sol_r1")
    assert sp["method"] == "single_prompt" and sp["baseline"] == "p01_a1_all4_luna_r1" and sp["llm_calls"] == 1
    assert sp["round"] == "round"
    with_interrupted = load.read_rows(round_dir, include_interrupted=True)
    assert any(r["interrupted"] and r["eval_name"] == "" for r in with_interrupted)


def test_pairing_averages_regradings_then_repeats_and_lists_unpaired_cells(round_dir):
    rows = load.read_rows(round_dir)
    p = load.pair_arms(rows, MA, SP, "overall_score", grader=SOL)
    assert len(p["pairs"]) == 1
    pair = p["pairs"][0]
    # luna r1: (7 + 8) / 2 = 7.5 from the two sol gradings; r2: 9; the cell: (7.5 + 9) / 2 = 8.25
    assert pair["a"] == pytest.approx(8.25) and pair["b"] == 8 and pair["diff"] == pytest.approx(-0.25)
    assert (pair["reps_a"], pair["reps_b"]) == (2, 1)
    assert ("p02", "a5", "cv_only", "default", "") in p["only_a"]
    assert ("p01", "a1", "all4", "trap-p01", "") in p["only_a"]  # another script is another cell
    assert p["only_b"] == [("p04", "a6", "all4", "default", "")]
    # without a grader filter every grading counts: r1 = (7 + 8 + 6) / 3 = 7, r2 = (9 + 8) / 2 = 8.5
    assert load.pair_arms(rows, MA, SP, "overall_score")["pairs"][0]["a"] == pytest.approx(7.75)
    # sub-scores pair the same way
    assert load.pair_arms(rows, MA, SP, "fit", grader=SOL)["pairs"][0]["diff"] == pytest.approx(-0.25)


def test_repeat_regrade_and_grader_groups(round_dir):
    rows = load.read_rows(round_dir)
    reps = load.repeat_groups(rows, "overall_score", grader=SOL)
    assert len(reps) == 1 and reps[0]["arm"] == MA and reps[0]["values"] == [7.5, 9.0]
    regr = load.regrade_groups(rows, "overall_score", grader=SOL)
    assert [(g["run_id"], g["values"]) for g in regr] == [("p01_a1_all4_luna_r1", [7, 8])]
    gp = load.grader_pairs(rows, "overall_score", SOL, OTHER)
    assert [(g["run_id"], g["a"], g["b"]) for g in gp] == [("p01_a1_all4_luna_r1", 7.5, 6.0),
                                                          ("p01_a1_all4_luna_r2", 9.0, 8.0)]


def test_ats_and_truthfulness_fields(round_dir):
    rows = load.read_rows(round_dir)
    r = next(x for x in rows if x["run_id"] == "p01_a1_all4_luna_r1" and x["eval_name"] == "eval1")
    assert r["ats_honest_coverage"] == pytest.approx(0.8)    # 4 in the CV of 5 claimable
    assert r["ats_reachable_share"] == pytest.approx(0.5)    # 5 claimable of 10 in the ad
    assert r["ats_pass"] == 1 and r["ats_keyword_coverage_pct"] == 40
    assert r["n_unsupported"] == 0 and r["any_unsupported"] == 0 and r["n_partly"] == 2
    low = next(x for x in rows if x["run_id"] == "p02_a5_cv_only_luna_r1")
    assert low["any_unsupported"] == 1 and low["ats_pass"] is None


def test_paired_statistics():
    diffs = [1, 1, 1, 0, -1, 2, 1, 0]
    s = stats.paired_summary(diffs, clusters=list("aabbccdd"), seed=1)
    assert s["n_pairs"] == 8 and s["mean_diff"] == pytest.approx(0.625)
    assert s["ci_low"] <= s["mean_diff"] <= s["ci_high"]
    assert (s["wins_b"], s["ties"], s["wins_a"]) == (5, 2, 1)
    assert s["p_sign"] == pytest.approx(0.21875)   # exact two-sided: 2 * P(X <= 1 | n = 6)
    assert 0 < s["p_signflip"] <= 1
    assert stats.holm({"a": 0.01, "b": 0.04, "c": None}) == {"a": 0.02, "b": 0.04, "c": None}
    p, lo, hi = stats.wilson_ci(0, 20)
    assert p == 0 and lo == 0 and 0.1 < hi < 0.2
    assert stats.weighted_kappa([5, 6, 7, 8], [5, 6, 7, 8]) == pytest.approx(1.0)
    assert stats.agreement([5, 6, 7, 8], [6, 7, 8, 9])["bias"] == pytest.approx(1.0)
    var, dof = stats.pooled_within_var([[7, 9], [8, 8], [6]])
    assert dof == 2 and var == pytest.approx(1.0)
    assert stats.pairs_needed(1.0, 1.0) < stats.pairs_needed(0.5, 1.0)
    assert stats.diff_sd(0.5, 0.4, 0.3, reps=2, gradings=2) < stats.diff_sd(0.5, 0.4, 0.3)
    d = stats.simulate_diffs(30, 0.5, 0.5, 0.3, 0.3, n_sim=4)
    assert d.shape == (4, 30) and not math.isnan(float(d.mean()))


def test_the_published_round_loads_and_matches_its_results_csv():
    round_dir = REPO / "evals" / "results" / "2026-11-10"
    rows = load.read_rows(round_dir)
    published = {r["run_id"]: r for r in load.read_results_csv(round_dir / "results.csv")}
    assert len(rows) == len(published) == 110
    for r in rows:
        for col in ("overall_score", "invented_content_score", "fit", "visual_layout", "n_cv_unsupported"):
            assert float(published[r["run_id"]][col]) == r[col], (r["run_id"], col)
    p = load.pair_arms(rows, "multi_agent/gpt-6-luna", "multi_agent/gpt-6.1-terra", "overall_score")
    assert len(p["pairs"]) == 19   # as in REPORT.md: terra +0.21 on 19 paired cases, 6 better, 9 equal, 4 worse
    assert sum(x["diff"] for x in p["pairs"]) / 19 == pytest.approx(0.2105, abs=1e-3)
    assert stats.sign_test([x["diff"] for x in p["pairs"]])["wins"] == 6
