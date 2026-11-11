"""Offline tests for the evaluation harness: non-IT fixtures, the revision script, the truth check,
repeats and re-scores in run_matrix, the spread summary, and the evaluator reusing the app's helpers.
No model, Tavily or subprocess calls: models are stubbed and run_matrix's subprocess is faked."""
import json
import statistics
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# --- scripted user: content wish, one revision, missing details ---------------------------------
def _revision_script():
    from evals.cases import find_script
    from evals.run_case import DEFAULT_SCRIPT
    return {**DEFAULT_SCRIPT, **json.loads(find_script("revision").read_text(encoding="utf-8"))}


def test_revision_script_sends_one_revision_after_the_cv_pdf(tmp_path):
    from evals.run_case import DEFAULT_SCRIPT, ScriptedUser
    script = _revision_script()
    assert script["first"] != DEFAULT_SCRIPT["first"] and "cover letter" in script["first"]
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "cv.pdf").write_bytes(b"%PDF-1.4")
    user = ScriptedUser(script, "/ad.txt", tmp_path)
    with_cv = {"generated_cv": "CV", "cv_pdf_path": "output/cv.pdf"}

    assert user.reply(("analyze_job_description",), {}) == "/ad.txt"
    assert user.reply(("collect_user_input",), {}) == script["first"]
    assert user.reply(("collect_user_input",), with_cv) == script["revision"]
    assert user.reply(("collect_user_input",), with_cv) == script["cover_letter"]  # the revision is sent once
    assert user.reply(("collect_user_input",), {**with_cv, "generated_cover_letter": "CL"}) == script["exit"]
    assert user.used["revision"] == 1 and user.observations == []


def test_revision_waits_for_the_pdf_and_is_noted_when_it_never_comes(tmp_path):
    from evals.run_case import ScriptedUser
    script = _revision_script()
    user = ScriptedUser(script, "/ad.txt", tmp_path)
    user.reply(("collect_user_input",), {})
    # a path in the state whose file does not exist does not count
    assert user.reply(("collect_user_input",), {"generated_cv": "CV", "cv_pdf_path": "missing.pdf"}) == script["cover_letter"]
    assert "revision" not in user.used
    assert user.observations == ["the revision was not sent: there was no CV PDF before the cover letter"]


def test_default_script_never_revises_and_answers_missing_details(tmp_path):
    from evals.run_case import DEFAULT_SCRIPT, ScriptedUser
    assert "revision" not in DEFAULT_SCRIPT and DEFAULT_SCRIPT["missing_details"]
    (tmp_path / "cv.pdf").write_bytes(b"%PDF-1.4")
    user = ScriptedUser(DEFAULT_SCRIPT, "/ad.txt", tmp_path)
    assert user.reply(("collect_user_input",), {"generated_cv": "CV", "cv_pdf_path": "cv.pdf"}) == DEFAULT_SCRIPT["cover_letter"]
    for node in ("ask_cv_missing_details", "ask_cover_letter_missing_details"):
        # asked before or after the documents exist: always the missing-details answer
        assert user.reply((node,), {}) == DEFAULT_SCRIPT["missing_details"]
        assert user.reply((node,), {"generated_cv": "CV", "generated_cover_letter": "CL"}) == DEFAULT_SCRIPT["missing_details"]
    assert user.used["missing_details"] == 4
    assert user.observations == ["the app asked for missing details more than twice"]
    assert user.reply(("ask_something_else",), {"generated_cv": "CV", "generated_cover_letter": "CL"}) == DEFAULT_SCRIPT["exit"]


def test_a_script_is_found_by_name_or_path(tmp_path):
    from evals.cases import SCRIPTS, find_script
    assert find_script("revision") == SCRIPTS / "revision.json"
    own = tmp_path / "mine.json"
    own.write_text("{}", encoding="utf-8")
    assert find_script(str(own)) == own
    with pytest.raises(ValueError):
        find_script("no-such-script")


# --- non-IT fixtures and matrix cases --------------------------------------------------------------
@pytest.mark.parametrize("persona_id,ad_id,language", [("p12", "a7", "en"), ("p13", "a8", "de")])
def test_non_it_fixtures_are_complete_and_fictional(persona_id, ad_id, language):
    from evals.cases import INPUT_FILES, find_ad, find_persona
    persona = find_persona(persona_id)
    truth = json.loads((persona / "persona.json").read_text(encoding="utf-8"))
    assert truth["fictional"] is True and truth["id"] == persona_id
    assert truth["primary_ad"] == find_ad(ad_id).name
    assert truth["language"].startswith(language)
    for key in ("true_skills", "deliberate_gaps", "traps", "facts_only_in_portfolio", "facts_only_in_notes"):
        assert truth[key], key
    for name in INPUT_FILES.values():
        assert (persona / name).read_text(encoding="utf-8").strip(), name
    ad = find_ad(ad_id)
    assert (ad / "ad.md").read_text(encoding="utf-8").strip() and (ad / "company.md").read_text(encoding="utf-8").strip()
    fixture = json.loads((ad / "search_fixture.json").read_text(encoding="utf-8"))
    for result in fixture["searches"][0]["response"]["results"]:
        host = urlparse("https://" + result["url"] if "://" not in result["url"] else result["url"]).hostname
        assert host.endswith((".example", ".invalid")) and result["raw_content"].strip()


def test_butcher_persona_is_the_repo_sample():
    from evals.cases import find_ad, find_persona
    sample_cv = (REPO / "examples" / "sample_CV_butcher.txt").read_text(encoding="utf-8")
    cv = (find_persona("p12") / "cv.txt").read_text(encoding="utf-8")
    assert cv.strip() and cv.strip() in sample_cv  # the sample without its "SAMPLE CV" header line
    ad = (find_ad("a7") / "ad.md").read_text(encoding="utf-8")
    assert "FreshCart" in ad and "Forklift licence" in ad
    # each persona keeps its own portfolio
    assert (find_persona("p12") / "portfolio.txt").read_text() != (find_persona("p13") / "portfolio.txt").read_text()


def test_matrix_has_non_it_and_revision_cases():
    from evals.cases import load_matrix, resolve_case
    cases = {c["id"]: c for c in load_matrix()}
    for pair in ("p12_a7", "p13_a8"):
        for input_set in ("all4", "cv_only"):
            case = cases[f"{pair}_{input_set}_luna_r1"]
            assert case["model"] == "openai:gpt-6-luna" and case["input_set"] == input_set
    revision = [c for c in cases.values() if c.get("script") == "revision"]
    assert {c["persona"] for c in revision} >= {"p12", "p13"}
    for case in revision:
        assert resolve_case(case)["script"].name == "revision.json"


# --- truth check -------------------------------------------------------------------------------------
class _StubChecker:
    """Stands in for the chat model: returns a fixed TruthCheck and keeps the prompt."""

    def __init__(self, items):
        self.items, self.messages = items, None

    def with_structured_output(self, schema, include_raw=False):
        from langchain_core.runnables import RunnableLambda

        def answer(messages):
            self.messages = messages
            return {"raw": None, "parsed": schema(items=self.items), "parsing_error": None}
        return RunnableLambda(answer)


def _run_with_outputs(tmp_path, cv: str, cover_letter: str, persona="p12") -> Path:
    run = tmp_path / "p12_a7_all4_luna_r1"
    (run / "outputs").mkdir(parents=True)
    (run / "outputs" / "final_cv.txt").write_text(cv, encoding="utf-8")
    (run / "outputs" / "final_cover_letter.txt").write_text(cover_letter, encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"labels": {"persona": persona}}), encoding="utf-8")
    return run


def test_truth_check_verifies_quotes_and_downgrades_a_fabricated_one(tmp_path):
    from evals.score_truth import score_truth
    from evals.cases import find_persona
    truth = json.loads((find_persona("p12") / "persona.json").read_text(encoding="utf-8"))
    run = _run_with_outputs(
        tmp_path,
        cv="Tomislav Mesarić\nMaster   Butcher with 9 years\nForklift licence (valid since 2020)\n",
        cover_letter="Dear FreshCart team,\nI am available for every shift, including all Saturdays.\n")
    n_gaps, n_traps = len(truth["deliberate_gaps"]), len(truth["traps"])
    items = [
        # true claim, quoted with other case, spacing and typographic dashes
        {"id": "G1", "verdict": "claimed_as_met", "document": "cv", "quote": "forklift licence (valid since 2020)", "note": ""},
        # quote spread over a line break and with "..."
        {"id": "G3", "verdict": "claimed_as_met", "document": "cv", "quote": "Master Butcher … 9 years", "note": ""},
        # fabricated quote: not in the outputs, so downgraded to unclear
        {"id": "T1", "verdict": "claimed_as_met", "document": "cv", "quote": "Licensed forklift operator", "note": ""},
        {"id": "T4", "verdict": "claimed_as_met", "document": "cover_letter",
         "quote": "available for every shift, including all Saturdays", "note": ""},
        # claimed without a quote: cannot be verified
        {"id": "T2", "verdict": "claimed_as_met", "document": "cv", "quote": "", "note": ""},
    ]
    items += [{"id": f"G{i}", "verdict": "not_claimed", "document": "none", "quote": "", "note": ""}
              for i in range(1, n_gaps + 1) if f"G{i}" not in {"G1", "G3"}]
    items += [{"id": f"T{i}", "verdict": "not_claimed", "document": "none", "quote": "", "note": ""}
              for i in range(1, n_traps) if f"T{i}" not in {"T1", "T2", "T4"}]  # the last trap is left out
    stub = _StubChecker(items)
    record = score_truth(run, name="truth1", llm=stub)

    saved = json.loads((run / "evaluation" / "truth1" / "truth_check.json").read_text(encoding="utf-8"))
    assert saved == record and record["persona"] == "p12"
    by_id = {r["id"]: r for r in record["items"]}
    assert by_id["G1"]["verdict"] == "claimed_as_met" and by_id["G1"]["quote_found"] is True
    assert by_id["G3"]["verdict"] == "claimed_as_met"
    assert by_id["T1"]["verdict"] == "unclear" and by_id["T1"]["model_verdict"] == "claimed_as_met"
    assert by_id["T1"]["flag"] == "quote_not_in_outputs"
    assert by_id["T2"]["flag"] == "quote_not_in_outputs"
    assert by_id[f"T{n_traps}"]["flag"] == "missing_from_answer" and by_id[f"T{n_traps}"]["verdict"] == "unclear"
    assert by_id["G1"]["item"] == truth["deliberate_gaps"][0] and by_id["T1"]["kind"] == "trap"
    counts = record["counts"]
    assert counts["gaps"] == n_gaps and counts["traps"] == n_traps
    assert counts["gaps_claimed"] == 2 and counts["traps_hit"] == 1
    assert counts["unverified_quotes"] == 2 and counts["missing_from_answer"] == 1 and counts["unclear"] == 3
    assert counts["cost_usd"] == 0 and record["cost_usd"] == 0
    # the checker saw every gap and trap and both outputs
    prompt = "\n".join(m.content for m in stub.messages)
    assert all(g in prompt for g in truth["deliberate_gaps"]) and all(t in prompt for t in truth["traps"])
    assert "Forklift licence (valid since 2020)" in prompt and "Dear FreshCart team" in prompt
    assert (run / "evaluation" / "truth1" / "prompt.json").is_file()
    with pytest.raises(FileExistsError):
        score_truth(run, name="truth1", llm=stub)


def test_truth_check_needs_a_persona(tmp_path):
    from evals.score_truth import score_truth
    run = _run_with_outputs(tmp_path, "CV", "CL", persona=None)
    with pytest.raises(ValueError):
        score_truth(run, llm=_StubChecker([]))
    # an explicit persona works without a label
    from evals.cases import find_persona
    record = score_truth(run, persona=str(find_persona("p13") / "persona.json"), llm=_StubChecker([]))
    assert record["persona"] == "p13" and record["counts"]["missing_from_answer"] == record["counts"]["gaps"] + record["counts"]["traps"]


def _p01_run(tmp_path, inputs_given=None) -> Path:
    """A p01 session whose CV names Debezium (the pilot's all4 sessions did: the portfolio and notes state it)."""
    run = _run_with_outputs(tmp_path, "Lena Varga-Holm\nCDC pipelines with Debezium and Kafka Connect\n", "Dear team",
                            persona="p01")
    manifest = {"labels": {"persona": "p01"}}
    if inputs_given is not None:
        manifest["inputs_given"] = inputs_given
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


DEBEZIUM_HIT = {"id": "T6", "verdict": "claimed_as_met", "document": "cv", "quote": "CDC pipelines with Debezium",
                "note": ""}


def test_truth_check_skips_a_conditional_trap_the_input_set_does_not_meet(tmp_path):
    """p01 T6 ("Debezium/CDC when portfolio and notes are omitted") fired in every all4 pilot session, where the
    portfolio and notes do state Debezium. It is only asked when both are left out, and keeps its id T6."""
    from evals.score_truth import score_truth
    all4 = _p01_run(tmp_path / "all4", ["cv", "portfolio", "cover_letter", "notes"])
    stub = _StubChecker([DEBEZIUM_HIT])
    record = score_truth(all4, llm=stub)
    assert "Debezium/CDC" not in "\n".join(m.content for m in stub.messages)
    assert record["not_applicable"] == ["T6"] and "T6" not in {r["id"] for r in record["items"]}
    assert record["counts"]["traps"] == 5 and record["counts"]["traps_hit"] == 0

    cv_only = _p01_run(tmp_path / "cv_only", ["cv"])
    stub = _StubChecker([DEBEZIUM_HIT])
    record = score_truth(cv_only, llm=stub)
    assert "T6: Attributing Debezium/CDC" in "\n".join(m.content for m in stub.messages)
    assert record["not_applicable"] == [] and record["counts"]["traps"] == 6 and record["counts"]["traps_hit"] == 1


def test_trap_conditions_on_given_inputs():
    from evals.cases import find_persona
    from evals.score_truth import trap_applies
    visa = json.loads((find_persona("p04") / "persona.json").read_text(encoding="utf-8"))["traps"][5]
    assert visa["when_given"] == ["notes"]
    assert trap_applies(visa, ["cv", "notes"]) and not trap_applies(visa, ["cv"])
    assert trap_applies(visa, None) and trap_applies("an unconditional trap", ["cv"])


def test_refilter_drops_collected_verdicts_of_traps_that_do_not_apply(tmp_path):
    """The pilot's checks asked about every trap; --refilter re-applies the conditions with no model call."""
    from evals.score_truth import main, refilter, score_truth
    run = _p01_run(tmp_path)  # no inputs_given: every trap is asked, as before the conditions existed
    before = score_truth(run, llm=_StubChecker([DEBEZIUM_HIT]))
    assert before["counts"]["traps_hit"] == 1 and before["not_applicable"] == []
    (run / "manifest.json").write_text(json.dumps({"labels": {"persona": "p01"},
                                                   "inputs_given": ["cv", "portfolio", "cover_letter", "notes"]}))
    assert main([str(run), "--refilter"]) == 0
    after = json.loads((run / "evaluation" / "truth1" / "truth_check.json").read_text(encoding="utf-8"))
    assert after["not_applicable"] == ["T6"] and [r["id"] for r in after["items"]][-1] == "T5"
    assert after["counts"]["traps"] == 5 and after["counts"]["traps_hit"] == 0
    assert after["counts"]["gaps"] == before["counts"]["gaps"] and after["cost_usd"] == before["cost_usd"]
    assert refilter(run) == after  # idempotent


def test_quote_matching_ignores_case_spacing_and_typography():
    from evals.score_truth import quote_found
    docs = ["Ich habe den „Führerschein Klasse B“ – seit 2020.", None]
    assert quote_found('führerschein  klasse b" - seit 2020', docs)
    assert quote_found("Ich habe ... seit 2020", docs)
    assert not quote_found("Führerschein Klasse C", docs)
    assert not quote_found("   ", docs)


# --- run_matrix: repeats, re-scores, truth check -----------------------------------------------------
def _case(cid, rep=1, **extra):
    return {"id": cid, "persona": "p12", "ad": "a7", "input_set": "all4",
            "inputs": ["cv", "portfolio", "cover_letter", "notes"], "model": "openai:gpt-6-luna", "rep": rep, **extra}


def test_repeats_keep_rep_one_ids_and_add_suffixed_reps():
    from evals.run_matrix import base_id, expand_repeats
    cases = [_case("x_all4_luna_r1"), _case("x_all4_luna_r2", rep=2), _case("plain")]
    assert expand_repeats(cases, 1) == cases
    expanded = expand_repeats(cases, 3)
    assert [c["id"] for c in expanded] == ["x_all4_luna_r1", "x_all4_luna_r2", "x_all4_luna_r3",
                                           "plain", "plain_r2", "plain_r3"]
    assert [c["rep"] for c in expanded] == [1, 2, 3, 1, 2, 3]
    assert base_id("x_all4_luna_r12") == "x_all4_luna" and base_id("plain") == "plain"


def test_dry_run_prints_repeats_rescores_and_truth_checks(tmp_path, capsys):
    from evals import run_matrix
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"cases": [_case("p12_a7_all4_revision_luna_r1", script="revision"), _case("other_r1")]}))
    code = run_matrix.main(["--matrix", str(matrix), "--out", str(tmp_path / "runs"), "--dry-run", "--repeats", "2",
                            "--rescore", "2", "--truth-check", "--only", "p12_a7_all4_revision_luna_r1"])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0 and not (tmp_path / "runs").exists()
    sessions = [line for line in lines if "evals.run_case" in line]
    assert len(sessions) == 2 and all("revision.json" in line and "--label script=revision" in line for line in sessions)
    assert "p12_a7_all4_revision_luna_r2 " in sessions[1] and "--label rep=2" in sessions[1]
    assert sum("evals.evaluate" in line for line in lines) == 4
    names = {line.split("--eval-name ")[1].split()[0] for line in lines if "evals.evaluate" in line}
    assert names == {"eval1", "eval2"}
    assert sum("evals.score_truth" in line for line in lines) == 2
    with pytest.raises(SystemExit):
        run_matrix.main(["--matrix", str(matrix), "--out", str(tmp_path / "runs"), "--dry-run", "--only", "nope"])


def test_rescore_and_truth_check_reuse_a_finished_session(tmp_path, monkeypatch):
    from argparse import Namespace
    from evals import run_matrix
    out = tmp_path / "runs"
    run = out / "p12_a7_all4_luna_r1"
    (run / "evaluation" / "eval1").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"status": "completed", "cost_usd": 0.1}))
    (run / "evaluation" / "eval1" / "evaluation.json").write_text(json.dumps({"cost_usd": 0.5}))
    (run / "evaluation" / "eval2").mkdir()  # a failed earlier attempt: removed and redone
    (run / "evaluation" / "eval2" / "raw_response.json").write_text("{}")
    commands = []

    def fake_run(cmd, *a, **kw):
        commands.append(cmd)
        target, name = Path(cmd[3]), cmd[cmd.index("--eval-name" if "--eval-name" in cmd else "--name") + 1]
        folder = target / "evaluation" / name
        assert not folder.exists()
        folder.mkdir(parents=True)
        result = "truth_check.json" if cmd[2] == "evals.score_truth" else "evaluation.json"
        (folder / result).write_text(json.dumps({"cost_usd": 0.01}))
        return type("Done", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    opts = Namespace(out=str(out), env_file=None, total_cap=10.0, cost_cap=1.0, rescore=3, eval_name="eval1",
                     truth_check=True)
    assert run_matrix.run_one(_case("p12_a7_all4_luna_r1"), opts) == ("p12_a7_all4_luna_r1", "ok")
    assert [c[2] for c in commands] == ["evals.evaluate", "evals.evaluate", "evals.score_truth"]  # no new session
    ledger = [line.split("\t")[:3] for line in (out / "ledger.tsv").read_text().splitlines()]
    assert ledger == [["p12_a7_all4_luna_r1", "eval", "0.010000"], ["p12_a7_all4_luna_r1", "eval", "0.010000"],
                      ["p12_a7_all4_luna_r1", "truth", "0.010000"]]
    assert run_matrix.run_one(_case("p12_a7_all4_luna_r1"), opts)[1] == "skip (done)"


# --- summarize -----------------------------------------------------------------------------------------
def _write_eval(run: Path, name: str, overall: int, cv_unsupported: int, grader="anthropic:claude-sonnet-5-5"):
    folder = run / "evaluation" / name
    folder.mkdir(parents=True)
    ev = {"overall_score": overall, "invented_content_score": 7, "completeness_score": 6,
          "fit": {"score": 5}, "cv_standards": {"score": 8}, "cover_letter": {"score": 7}}
    counts = {"cv_unsupported": cv_unsupported, "cv_partly": 0, "cl_unsupported": 0, "cl_partly": 1,
              "omissions_high": 2, "gaps_overstated": 0, "work_arrangement_invented": 0}
    (folder / "evaluation.json").write_text(json.dumps({"evaluator_model": grader, "evaluation": ev, "counts": counts,
                                                        "cost_usd": 0.1}))


def test_summarize_reports_spread_across_reps_and_rescores(tmp_path, capsys):
    from evals import summarize
    runs = tmp_path / "runs"
    for rid in ("p12_a7_all4_luna_r1", "p12_a7_all4_luna_r2", "p13_a8_cv_only_luna_r1"):
        (runs / rid).mkdir(parents=True)
        (runs / rid / "manifest.json").write_text("{}")
    (runs / "ledger.tsv").write_text("")
    (runs / "not_a_run").mkdir()  # no manifest: ignored
    _write_eval(runs / "p12_a7_all4_luna_r1", "eval1", 6, 1)
    _write_eval(runs / "p12_a7_all4_luna_r1", "eval2", 8, 3)
    _write_eval(runs / "p12_a7_all4_luna_r2", "eval1", 4, 0)
    _write_eval(runs / "p13_a8_cv_only_luna_r1", "eval1", 9, 0)
    _write_eval(runs / "p12_a7_all4_luna_r1", "opus1", 0, 9, grader="anthropic:claude-opus-5-5")  # never averaged in
    truth = runs / "p12_a7_all4_luna_r1" / "evaluation" / "truth1"
    truth.mkdir()
    (truth / "truth_check.json").write_text(json.dumps({"counts": {"gaps_claimed": 1, "traps_hit": 2, "unclear": 0,
                                                                   "unverified_quotes": 1}}))
    summary = summarize.summarize(runs)
    assert sorted(summary) == ["p12_a7_all4_luna", "p13_a8_cv_only_luna"]
    p12 = summary["p12_a7_all4_luna"]
    assert p12["runs"] == ["p12_a7_all4_luna_r1", "p12_a7_all4_luna_r2"]
    overall = p12["metrics"]["overall"]
    # run means 7 (6 and 8) and 4: across reps; 6 vs 8 within the first run: evaluator spread
    assert overall["reps"] == 2 and overall["mean"] == 5.5 and overall["sd"] == round(statistics.stdev([7, 4]), 3)
    assert overall["eval_sd"] == round(statistics.stdev([6, 8]), 3) and overall["evals_per_run"] == 2
    assert p12["metrics"]["cv_unsupported"]["mean"] == 1.0
    assert p12["metrics"]["truth_traps_hit"] == {"reps": 1, "mean": 2, "sd": None, "min": 2, "max": 2,
                                                  "eval_sd": None, "evals_per_run": 1}
    single = summary["p13_a8_cv_only_luna"]["metrics"]["overall"]
    assert single["sd"] is None and single["eval_sd"] is None

    assert summarize.main([str(runs), "--only-repeated"]) == 0
    table = capsys.readouterr().out
    assert "p12_a7_all4_luna" in table and "p13_a8" not in table and "5.5±2.121" in table
    written = json.loads((runs / "repeats_summary.json").read_text())
    assert list(written) == ["p12_a7_all4_luna"]
    opus = summarize.summarize(runs, grader="claude-opus-5-5")["p12_a7_all4_luna"]["metrics"]["overall"]
    assert opus["reps"] == 1 and opus["mean"] == 0


# --- evaluator reuses the app's helpers ---------------------------------------------------------------
def test_company_text_is_what_the_app_reads(tmp_path):
    from agents.search import CHARS_PER_RESULT, search_result_texts
    from evals.evaluate import company_text
    response = {"results": [
        {"title": "Page", "url": "a.example/x", "content": "short snippet", "raw_content": "FULL PAGE " * 500},
        {"title": "Snippet only", "url": "b.example/y", "content": "only the snippet"},
        {"title": "Empty", "url": "c.example/z", "content": ""},
    ]}
    fixture = tmp_path / "search_fixture.json"
    fixture.write_text(json.dumps({"fixture_version": 1, "searches": [{"query": "*", "response": response}]}))
    text = company_text(fixture)
    assert "FULL PAGE" in text and "short snippet" not in text and "only the snippet" in text and "Empty" not in text
    expected = search_result_texts(response)
    assert len(expected[0]["text"]) == CHARS_PER_RESULT and expected[0]["text"] in text
    assert company_text(tmp_path / "missing.json") == "(none)"


def test_pdf_checks_use_the_apps_check_and_keep_the_old_keys(tmp_path, monkeypatch):
    from agents import pdf_text_check
    from evals.evaluate import pdf_checks
    pdf = tmp_path / "cv.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    monkeypatch.setattr(pdf_text_check, "extract_pdf_text", lambda path: ("Jane Doe Butcher [PHONE]", 1, None))
    checks = pdf_checks(pdf, "Jane Doe Butcher")
    for key in ("exists", "pages", "chars", "text_words_found_in_pdf_pct", "bracket_placeholders"):
        assert key in checks, key
    assert checks["exists"] is True and checks["pages"] == 1 and checks["text_words_found_in_pdf_pct"] == 100.0
    assert checks["bracket_placeholders"] == ["[PHONE]"] and "extra_words" in checks
    assert "text_words_found_in_pdf_pct" not in pdf_checks(pdf, None)  # nothing to compare with
    monkeypatch.setattr(pdf_text_check, "extract_pdf_text", lambda path: (None, None, "PDFSyntaxError: bad"))
    assert pdf_checks(pdf, "x") == {"exists": True, "error": "PDFSyntaxError: bad"}
    assert pdf_checks(tmp_path / "none.pdf", "x") == {"exists": False}
