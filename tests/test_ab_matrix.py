"""Offline tests for the staged comparison matrices (evals/build_ab_matrix.py, matrix_ab_stage12.json,
matrix_ab_stage3.json) and run_matrix's stages, gradings, caps, freeze and estimate.

No paid call of any kind: nothing here builds a model, and run_matrix's subprocess is faked."""
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STAGE12 = REPO / "evals" / "matrix_ab_stage12.json"
STAGE3 = REPO / "evals" / "matrix_ab_stage3.json"
SONNET, OPUS, HAIKU = "anthropic:claude-sonnet-5-5", "anthropic:claude-opus-5-5", "anthropic:claude-haiku-5-5"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _names(case, grader=None):
    return [g["name"] for g in case["gradings"] if grader is None or g["grader"] == grader]


# --- the matrix files ---------------------------------------------------------------------------------
def test_the_matrix_files_are_what_the_builder_makes():
    from evals import build_ab_matrix
    assert build_ab_matrix.main(["--check"]) == 0


def test_stage12_resolves_and_has_the_plans_sessions():
    from evals.cases import INPUT_SETS, resolve_case
    m = _load(STAGE12)
    cases = m["cases"]
    assert len(cases) == 147 and len({c["id"] for c in cases}) == 147
    assert Counter(c["stage"] for c in cases) == {"pilot": 36, "core": 111}
    assert Counter(c["arm"] for c in cases) == {"A": 49, "C": 49, "E": 49}
    arms = {"A": ("multi_agent", "openai:gpt-6-luna"), "C": ("single_prompt", "openai:gpt-6-luna"),
            "E": ("single_prompt", "openai:gpt-6.1-sol")}
    for c in cases:
        resolve_case(c)
        assert (c["method"], c["model"]) == arms[c["arm"]] and not c.get("script") and not c.get("work_preference")
        assert c["writer_model"] == c["model"].split(":", 1)[1]
        assert c["case"] == f"{c['persona']}_{c['ad']}_{c['input_set']}" and c["inputs"] == INPUT_SETS[c["input_set"]]
        assert c["id"].endswith(f"_{c['case']}_r{c['rep']}")
        truth = json.loads(resolve_case(c)["persona_truth"].read_text())
        assert truth["primary_ad"].split("-", 1)[0] == c["ad"]
    # the 39 matched cases, each in all three arms
    first = [c for c in cases if c["rep"] == 1]
    assert {(c["persona"], c["input_set"]) for c in first} == {
        (f"p{i:02d}", s) for i in range(1, 14) for s in ("all4", "cv_only", "no_cv")}
    assert Counter(c["case"] for c in first) == dict.fromkeys({c["case"] for c in first}, 3)


def test_stage12_pilot_check_repeat_and_opus_cases_follow_the_plan():
    m = _load(STAGE12)
    d, cases = m["design"], m["cases"]
    assert d["pilot"] == ["p01_a1_all4", "p03_a3_all4", "p04_a6_all4", "p11_a1_all4", "p12_a7_all4", "p13_a8_all4"]
    assert {c["case"] for c in cases if c["stage"] == "pilot"} == set(d["pilot"])
    pilot_personas = {c.split("_")[0] for c in d["pilot"]}
    check_personas = [c.split("_")[0] for c in d["check"]]
    assert len(set(check_personas)) == 6 and not set(check_personas) & pilot_personas
    assert Counter(c.split("_", 2)[2] for c in d["check"]) == {"all4": 2, "cv_only": 2, "no_cv": 2}
    extra = [c for c in d["repeat"] if c not in d["pilot"]]
    assert len(d["repeat"]) == 10 and len(extra) == 4 and set(extra) <= set(d["check"])
    assert d["opus"] == d["pilot"] + d["check"]
    # writer runs: 2 for the repeat cases, 1 otherwise; second runs are graded once by Sonnet
    reps = Counter((c["case"], c["arm"]) for c in cases)
    assert all(n == (2 if case in d["repeat"] else 1) for (case, _), n in reps.items())
    for c in cases:
        if c["rep"] == 2:
            assert _names(c) == ["eval1"]
    for c in (c for c in cases if c["rep"] == 1):
        assert _names(c, "sonnet") == (["eval1", "eval2"] if c["case"] in d["repeat"] else ["eval1"])
        assert _names(c, "opus") == (["opus1"] if c["case"] in d["opus"] else [])
        assert _names(c, "haiku") == (["haiku1", "haiku2"] if c["case"] in d["pilot"] else [])
    # the plan's totals: 30 re-graded first runs, Opus on 36, Haiku x2 on 18, the trap check on every session
    graded = Counter((g["grader"], g["name"]) for c in cases for g in c["gradings"])
    assert graded == {("sonnet", "eval1"): 147, ("sonnet", "eval2"): 30, ("opus", "opus1"): 36,
                      ("haiku", "haiku1"): 18, ("haiku", "haiku2"): 18}
    models = {"sonnet": SONNET, "opus": OPUS, "haiku": HAIKU}
    assert all(g["model"] == models[g["grader"]] for c in cases for g in c["gradings"])
    assert all(c["truth"] == {"name": "truth1", "grader": "trap", "model": HAIKU} for c in cases)
    # the pilot's 18 first runs carry the calibration; the core holds the other 12 re-grades and 18 Opus gradings,
    # and no Haiku grading (Haiku was dropped as a re-grader after the pilot)
    core = [g for c in cases if c["stage"] == "core" for g in c["gradings"]]
    assert Counter(g["name"] for g in core if g["name"] != "eval1") == {"eval2": 12, "opus1": 18}


def test_each_cases_sessions_are_back_to_back_in_a_shuffled_arm_order():
    cases = _load(STAGE12)["cases"]
    assert [c["stage"] for c in cases] == ["pilot"] * 36 + ["core"] * 111  # the pilot first
    blocks = [c["case"] for i, c in enumerate(cases) if i == 0 or cases[i - 1]["case"] != c["case"]]
    assert len(blocks) == len(set(blocks)) == 39  # one contiguous block per matched case
    arm_orders = {tuple(c["arm"] for c in cases if c["case"] == case and c["rep"] == 1) for case in blocks}
    assert len(arm_orders) > 1  # not always the same arm first


def test_the_design_is_deterministic():
    from evals import build_ab_matrix
    assert build_ab_matrix.design() == build_ab_matrix.design() == _load(STAGE12)["design"]
    assert build_ab_matrix.stage12()["cases"] == _load(STAGE12)["cases"]
    assert build_ab_matrix.design(seed=1) != build_ab_matrix.design()


def test_stage3_is_a_separate_matrix_of_terra_arms():
    from evals.cases import resolve_case
    m, d = _load(STAGE3), _load(STAGE12)["design"]
    cases = m["cases"]
    assert len(cases) == 78 and set(m["stages"]) == {"stage3"}
    assert Counter((c["arm"], c["method"], c["model"]) for c in cases) == {
        ("B", "multi_agent", "openai:gpt-6.1-terra"): 39, ("D", "single_prompt", "openai:gpt-6.1-terra"): 39}
    for c in cases:
        resolve_case(c)
        assert c["rep"] == 1 and c["truth"]["model"] == HAIKU
    opus = m["design"]["stage3_opus"]
    assert len(opus) == 10 and set(opus) <= set(d["opus"])
    assert Counter(g["name"] for c in cases for g in c["gradings"]) == {"eval1": 78, "opus1": 20}
    assert {c["case"] for c in cases if "opus1" in _names(c)} == set(opus)
    assert not set(m["stages"]) & set(_load(STAGE12)["stages"])


# --- the estimate -------------------------------------------------------------------------------------
def _estimate(path, stages=None):
    from evals import run_matrix
    meta = run_matrix.load_meta(path)
    cases = run_matrix.select_stages(_load(path)["cases"], meta, stages)
    return run_matrix.estimate_stages(cases, meta)


def test_the_estimate_reprices_the_pilot_to_its_spend_and_matches_the_core_plan():
    est = _estimate(STAGE12)
    # at the pilot's measured unit costs, the pilot's own sessions and gradings come to what it spent (6.78 Batch)
    assert est["stages"]["pilot"]["batch"] == 6.78 and est["stages"]["pilot"]["plan_usd"]["batch"] == 8.81
    core = est["stages"]["core"]
    assert (core["standard"], core["batch"]) == (20.67, 11.73) == tuple(core["plan_usd"].values())
    assert core["within_cap"] and est["total"]["within_cap"]
    assert est["total"]["cap_usd"] == {"standard": 57.0, "batch": 34.0}
    a = est["arms"]["A"]  # 49 sessions: 37 Sonnet eval1, 10 eval2, 12 opus1, 6 x2 haiku, 49 trap checks
    assert a["sessions"] == 49 and a["app"] == pytest.approx(49 * 0.0286) and a["grading"] == pytest.approx(49 * 0.093)
    assert a["regrades"] == pytest.approx(10 * 0.093) and a["opus"] == pytest.approx(12 * 0.352)
    assert a["haiku"] == pytest.approx(12 * 0.0098) and a["trap"] == pytest.approx(49 * 0.001)
    assert a["batch"] == pytest.approx(49 * 0.0286 + (49 * 0.093 + 10 * 0.093 + 12 * 0.352 + 12 * 0.0098
                                                      + 49 * 0.001) / 2, abs=0.01)  # sessions direct
    s3 = _estimate(STAGE3)
    assert (s3["total"]["standard"], s3["total"]["batch"]) == (20.03, 12.84)


def test_the_estimate_of_a_selection_has_no_total_cap_line_and_the_stage_caps():
    est = _estimate(STAGE12, ["pilot"])
    assert set(est["stages"]) == {"pilot"} and "plan_usd" not in est["total"]
    assert est["stages"]["pilot"]["cap_usd"] == {"standard": 20.0, "batch": 12.0}


def test_the_unit_prices_are_the_pilots_measured_means():
    from evals import run_matrix
    u = run_matrix.AB_UNIT_USD
    assert u["multi_agent"]["gpt-6-luna"] == 0.0286
    assert u["single_prompt"] == {"gpt-6-luna": 0.0028, "gpt-6.1-sol": 0.0439,
                                  "gpt-6.1-terra": 0.06}  # terra: not in the pilot, 6k in / 10k out
    # the gradings were measured through the Batch API (half price): the units are list price
    assert u["grading"] == pytest.approx({"claude-sonnet-5-5": 0.093, "claude-opus-5-5": 0.352,
                                          "claude-haiku-5-5": 0.0098})
    assert u["trap"] == pytest.approx({"claude-haiku-5-5": 0.001})
    with pytest.raises(ValueError, match="no price"):
        run_matrix.sp_session_usd("openai:unknown-model")


def test_estimate_prints_the_plans_table(capsys):
    from evals import run_matrix
    assert run_matrix.main(["--matrix", str(STAGE12), "--out", "unused", "--estimate"]) == 0
    out = capsys.readouterr().out
    assert "| A MA-luna" in out and "| total" in out and "Haiku calibration" in out
    assert "stage core: 111 sessions, $20.67 standard / $11.73 Batch (plan $20.67 / $11.73); cap $38.00 / $23.00: within" in out
    assert run_matrix.main(["--matrix", str(STAGE12), "--out", "unused", "--estimate", "--stage", "core", "--json"]) == 0
    est = json.loads(capsys.readouterr().out)
    assert est["total"]["sessions"] == 111 and est["total"]["standard"] == 20.67


# --- run_matrix on a staged matrix --------------------------------------------------------------------
def test_the_session_command_carries_the_comparison_labels():
    from evals import run_matrix
    case = next(c for c in _load(STAGE12)["cases"] if c["arm"] == "C" and c["rep"] == 2)
    cmd = run_matrix.session_command(case, Path("/runs") / case["id"], None, 1.5,
                                     {"app_commit": "abc", "app_head": "abc", "app_dirty": False})
    labels = dict(cmd[i + 1].split("=", 1) for i, x in enumerate(cmd) if x == "--label")
    assert cmd[2] == "evals.single_prompt" and cmd[cmd.index("--model") + 1] == "openai:gpt-6-luna"
    assert labels == {"persona": case["persona"], "ad": case["ad"], "model": "openai:gpt-6-luna", "rep": "2",
                      "input_set": case["input_set"], "case": case["case"], "script": "default",
                      "method": "single_prompt", "arm": "C", "stage": case["stage"], "writer_model": "gpt-6-luna",
                      "app_commit": "abc", "app_head": "abc", "app_dirty": "False"}
    ma = next(c for c in _load(STAGE12)["cases"] if c["arm"] == "A")
    assert run_matrix.session_command(ma, Path("/r"), None, 1.5)[2] == "evals.run_case"


def _fake_subprocess(monkeypatch, commands, session_cost=0.03, eval_cost=0.15, exit_codes=None):
    from evals import run_matrix

    def fake_run(cmd, *a, **kw):
        commands.append(cmd)
        module = cmd[2]
        code = (exit_codes or {}).get(module, 0)
        if module in ("evals.run_case", "evals.single_prompt"):
            run_dir = Path(cmd[cmd.index("--run-dir") + 1])
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(json.dumps({"status": "completed", "cost_usd": session_cost}))
        elif code == 0:
            name = cmd[cmd.index("--eval-name" if "--eval-name" in cmd else "--name") + 1]
            folder = Path(cmd[3]) / "evaluation" / name
            folder.mkdir(parents=True)
            target = "truth_check.json" if module == "evals.score_truth" else "evaluation.json"
            (folder / target).write_text(json.dumps({"cost_usd": eval_cost}))
        return type("Done", (), {"stdout": "", "stderr": "", "returncode": code})()

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)


def _opts(out, all_cases, sessions_only=False, total_cap=None):
    from evals import run_matrix
    opts = Namespace(out=str(out), env_file=None, total_cap=total_cap, cost_cap=1.5, rescore=None, eval_name="eval1",
                     truth_check=False, sessions_only=sessions_only, app_labels={"app_commit": "abc"})
    run_matrix.configure_staged(opts, run_matrix.load_meta(STAGE12), all_cases)
    return opts


def test_a_pilot_first_run_gets_its_named_gradings_and_the_trap_check(tmp_path, monkeypatch):
    from evals import run_matrix
    cases = _load(STAGE12)["cases"]
    case = next(c for c in cases if c["stage"] == "pilot" and c["rep"] == 1 and c["arm"] == "A")
    commands = []
    _fake_subprocess(monkeypatch, commands)
    opts = _opts(tmp_path, cases)
    assert opts.total_cap == 57.0 and opts.stage_caps == {"pilot": 20.0, "core": 38.0}
    assert run_matrix.run_one(case, opts) == (case["id"], "ok")
    graded = [(c[2], c[c.index("--eval-name" if "--eval-name" in c else "--name") + 1],
               c[c.index("--grader-model") + 1]) for c in commands[1:]]
    assert graded == [("evals.evaluate", "eval1", SONNET), ("evals.evaluate", "eval2", SONNET),
                      ("evals.evaluate", "opus1", OPUS), ("evals.evaluate", "haiku1", HAIKU),
                      ("evals.evaluate", "haiku2", HAIKU), ("evals.score_truth", "truth1", HAIKU)]
    assert "app_commit=abc" in commands[0]
    ledger = [line.split("\t")[:2] for line in (tmp_path / "ledger.tsv").read_text().splitlines()]
    assert ledger == [[case["id"], k] for k in ("app", "eval:eval1", "eval:eval2", "eval:opus1", "eval:haiku1",
                                                "eval:haiku2", "truth")]
    assert run_matrix.run_one(case, opts)[1] == "skip (done)"


def test_sessions_only_runs_no_grading_and_resumes(tmp_path, monkeypatch):
    from evals import run_matrix
    cases = _load(STAGE12)["cases"]
    case = cases[0]
    commands = []
    _fake_subprocess(monkeypatch, commands)
    opts = _opts(tmp_path, cases, sessions_only=True)
    assert opts.total_cap == 34.0 and opts.stage_caps == {"pilot": 12.0, "core": 23.0}
    assert run_matrix.run_one(case, opts) == (case["id"], "session done (not graded)")
    assert [c[2] for c in commands] == [run_matrix.session_command(case, Path("x"), None, 1)[2]]
    assert run_matrix.run_one(case, opts)[1] == "skip (session done)" and len(commands) == 1
    # later, grading here picks up the finished session without redoing it
    opts.sessions_only = False
    assert run_matrix.run_one(case, opts)[1] == "ok"
    assert all(c[2] != "evals.run_case" and c[2] != "evals.single_prompt" for c in commands[1:])


def test_the_stage_cap_counts_only_that_stages_spend(tmp_path, monkeypatch):
    from evals import run_matrix
    cases = _load(STAGE12)["cases"]
    pilot = [c for c in cases if c["stage"] == "pilot"]
    core = next(c for c in cases if c["stage"] == "core")
    commands = []
    _fake_subprocess(monkeypatch, commands)
    opts = _opts(tmp_path, cases)
    with open(tmp_path / "ledger.tsv", "w") as f:  # the pilot spent past its own cap
        f.write(f"{pilot[0]['id']}\tapp\t20.5\tt\n")
    assert run_matrix.run_one(pilot[1], opts)[1] == "skip (stage pilot cap reached)"
    assert run_matrix.run_one(core, opts)[1] == "ok"  # the core has its own cap
    with open(tmp_path / "ledger.tsv", "a") as f:
        f.write(f"{core['id']}\tapp\t40\tt\n")
    assert run_matrix.run_one(next(c for c in cases if c["stage"] == "core" and c is not core), opts)[1] == \
        "skip (total cap reached)"


def test_the_cost_of_a_failed_grading_or_trap_check_stays_in_the_ledger(tmp_path, monkeypatch):
    """A refusal or unparseable answer exits 1 after writing summary.json with its billed cost; the retry on the next
    run deletes that folder, so the cost must already be in the ledger (--total-cap counts it)."""
    from evals import run_matrix
    cases = _load(STAGE12)["cases"]
    case = next(c for c in cases if c["stage"] == "pilot" and c["rep"] == 1 and c["arm"] == "A")
    commands = []
    _fake_subprocess(monkeypatch, commands)
    passing = run_matrix.subprocess.run
    failing = {"eval1", "truth1"}

    def fake_run(cmd, *a, **kw):
        name = next((cmd[cmd.index(f) + 1] for f in ("--eval-name", "--name") if f in cmd), None)
        if name not in failing:
            return passing(cmd, *a, **kw)
        commands.append(cmd)
        folder = Path(cmd[3]) / "evaluation" / name
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"status": "parse_error", "cost_usd": 0.35}))
        return type("Done", (), {"stdout": "", "stderr": "", "returncode": 1})()
    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    opts = _opts(tmp_path, cases)
    assert run_matrix.run_one(case, opts) == (case["id"], "failed: eval1, truth1")
    ledger = lambda: [line.split("\t")[1:3] for line in (tmp_path / "ledger.tsv").read_text().splitlines()]  # noqa: E731
    assert ["eval:eval1-failed", "0.350000"] in ledger() and ["truth-failed", "0.350000"] in ledger()
    failing.clear()  # the retry deletes the failed folders and succeeds; the failed costs stay counted
    assert run_matrix.run_one(case, opts) == (case["id"], "ok")
    assert run_matrix.spent(tmp_path / "ledger.tsv") == pytest.approx(0.03 + 6 * 0.15 + 2 * 0.35)


def test_out_of_credit_in_a_named_grading_stops_the_matrix(tmp_path, monkeypatch):
    from evals import run_matrix
    from evals.credit import EXIT_OUT_OF_CREDIT
    cases = _load(STAGE12)["cases"]
    commands = []
    _fake_subprocess(monkeypatch, commands, exit_codes={"evals.evaluate": EXIT_OUT_OF_CREDIT})
    run_matrix.OUT_OF_CREDIT.clear()
    try:
        status = run_matrix.run_one(cases[0], _opts(tmp_path, cases))[1]
        assert status.startswith("interrupted (out of credit) during evaluation eval1")
        assert run_matrix.run_one(cases[1], _opts(tmp_path, cases))[1] == "not started (out of credit)"
    finally:
        run_matrix.OUT_OF_CREDIT.clear()


def test_a_staged_matrix_needs_a_stage_and_refuses_the_flat_options(tmp_path):
    from evals import run_matrix
    base = ["--matrix", str(STAGE12), "--out", str(tmp_path / "runs")]
    with pytest.raises(SystemExit, match="choose the stage"):
        run_matrix.main(base)
    for extra in (["--rescore", "2"], ["--repeats", "2"], ["--truth-check"], ["--eval-name", "x"]):
        with pytest.raises(SystemExit, match="do not apply"):
            run_matrix.main(base + extra)
    with pytest.raises(SystemExit, match="unknown stage"):
        run_matrix.main(base + ["--stage", "stage3", "--estimate"])
    with pytest.raises(SystemExit, match="no stages"):
        run_matrix.main(["--matrix", str(REPO / "evals" / "matrix.json"), "--out", "x", "--stage", "pilot"])
    assert not (tmp_path / "runs").exists()


def test_a_frozen_app_commit_that_differs_stops_before_any_session(tmp_path, monkeypatch):
    from evals import run_matrix
    monkeypatch.setattr(run_matrix, "app_version", lambda frozen=None: {
        "app_head": "def", "app_dirty": False, "app_commit": "abc", "matches_frozen": False})
    monkeypatch.setattr(run_matrix.subprocess, "run", lambda *a, **k: pytest.fail("nothing may run"))
    with pytest.raises(SystemExit, match="differ from the frozen commit"):
        run_matrix.main(["--matrix", str(STAGE12), "--out", str(tmp_path), "--stage", "pilot"])


def test_app_version_reads_git_and_compares_the_app_files():
    from evals import run_matrix
    v = run_matrix.app_version()
    assert v["app_commit"] == v["app_head"] and len(v["app_head"]) == 40 and "matches_frozen" not in v
    frozen = run_matrix.app_version(v["app_head"])
    assert frozen["app_commit"] == v["app_head"] and frozen["matches_frozen"] == (not v["app_dirty"])


def test_dry_run_prints_each_named_grading(tmp_path, capsys):
    from evals import run_matrix
    case = next(c for c in _load(STAGE12)["cases"] if c["stage"] == "pilot" and c["rep"] == 1)
    assert run_matrix.main(["--matrix", str(STAGE12), "--out", str(tmp_path), "--stage", "pilot", "--dry-run",
                            "--only", case["id"]]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7 and "--label arm=" in lines[0] and "--label app_commit=" in lines[0]
    assert lines[3].endswith(f"--eval-name opus1 --grader-model {OPUS}")
    assert lines[6].endswith(f"--name truth1 --grader-model {HAIKU}")
    assert not list(tmp_path.iterdir())


def test_gradings_out_lists_the_jobs_case_by_case(tmp_path):
    from evals import run_matrix
    target = tmp_path / "jobs.json"
    assert run_matrix.main(["--matrix", str(STAGE12), "--out", str(tmp_path / "runs"), "--stage", "pilot",
                            "--gradings-out", str(target)]) == 0
    jobs = json.loads(target.read_text())
    assert len(jobs) == 36 + 18 + 18 + 36 + 36 and [j["order"] for j in jobs] == list(range(1, len(jobs) + 1))
    blocks = [j["case"] for i, j in enumerate(jobs) if i == 0 or jobs[i - 1]["case"] != j["case"]]
    assert len(blocks) == len(set(blocks)) == 6
    assert Counter(j["kind"] for j in jobs) == {"evaluation": 108, "truth_check": 36}
    assert all(j["run_dir"] == str((tmp_path / "runs" / j["id"]).resolve()) for j in jobs)
