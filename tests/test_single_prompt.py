"""Offline tests for the single-prompt baseline (evals/single_prompt.py, evals/matrix_single_prompt.json).

No paid call of any kind: the chat models are stubbed, pdflatex is not run, no .env is read (an empty
--env-file is passed), and run_matrix's subprocess is faked."""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SP_MATRIX = REPO / "evals" / "matrix_single_prompt.json"
CV_TEXT = "Marta Example\nDevOps Engineer\nSUMMARY_FROM_STUB: CI pipelines with GitHub Actions."
LETTER_TEXT = "Dear hiring team,\nLETTER_FROM_STUB: I would like to apply.\nKind regards, Marta"


CALLS: list = []  # every FakeChat call of the current test


class FakeChat(BaseChatModel):
    """A chat model that answers every structured-output request with fixed JSON and logs each call."""

    model_name: str = "gpt-6.1-sol"
    schema_name: str = ""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        CALLS.append({"schema": self.schema_name, "messages": messages})
        answers = {
            "SinglePromptResponse": {"cv_content": CV_TEXT, "cover_letter_content": LETTER_TEXT, "notes": "n"},
            "CVLatexDocumentResponse": {"latex_document": "\\documentclass{article}\\begin{document}CV\\end{document}"},
            "CoverLetterLatexDocumentResponse": {"latex_document": "\\documentclass{article}\\begin{document}L\\end{document}"},
        }
        message = AIMessage(content=json.dumps(answers[self.schema_name]),
                            usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
                            response_metadata={"model_name": self.model_name})
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        bound = self.model_copy(update={"schema_name": schema.__name__})

        def parse(message):
            parsed = schema.model_validate_json(message.content)
            return {"raw": message, "parsed": parsed, "parsing_error": None} if include_raw else parsed

        return bound | RunnableLambda(parse)


def _blank_pdf(path: Path) -> None:
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        writer.write(f)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Every model the runner and the app's PDF path could build is FakeChat; LaTeX is never compiled."""
    import agents.latex_fixer as latex_fixer
    import agents.latex_pdf_generator as lpg
    from evals import single_prompt
    CALLS.clear()

    def fake_init(model, **kwargs):
        return FakeChat(model_name=model.split(":", 1)[-1])

    monkeypatch.setattr(single_prompt, "init_chat_model", fake_init)
    monkeypatch.setattr(lpg, "init_chat_model", fake_init)
    monkeypatch.setattr(latex_fixer, "init_chat_model", fake_init)

    def fake_compile(self, latex, output_name, max_fix_iterations=1):
        pdf = Path(self.output_folder) / f"{output_name}.pdf"
        _blank_pdf(pdf)
        return str(pdf), 0, None

    monkeypatch.setattr(lpg.LatexPDFGenerator, "_compile_latex", fake_compile)
    monkeypatch.setattr(lpg, "check_pdf", lambda path, text: {"ok": True, "pages": 1})
    env = tmp_path / "empty.env"
    env.write_text("")
    return {"calls": CALLS, "env": str(env)}


def _persona_texts(persona: str) -> dict:
    from evals.cases import INPUT_FILES, find_persona
    folder = find_persona(persona)
    return {k: (folder / f).read_text(encoding="utf-8") for k, f in INPUT_FILES.items()}


# --- the prompt ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("input_set", ["all4", "cv_only", "no_cv"])
def test_the_prompt_holds_exactly_the_given_inputs_and_the_search_fixture(input_set):
    from agents.document_writer import TRUTHFULNESS_PROTOCOL
    from evals.cases import INPUT_SETS, find_ad
    from evals.evaluate import company_text
    from evals.single_prompt import build_messages, writer_rules
    texts = _persona_texts("p03")
    given = {k: texts[k] for k in INPUT_SETS[input_set]}
    ad = find_ad("a3")
    company = company_text(ad / "search_fixture.json")
    system, human = build_messages(given, (ad / "ad.md").read_text(encoding="utf-8"), company)
    for key, text in texts.items():
        assert (text.strip() in human.content) == (key in given), key
    assert company.strip() in human.content and company != "(none)"
    first_result = json.loads((ad / "search_fixture.json").read_text())["searches"][0]["response"]["results"][0]
    assert first_result["url"] in human.content
    assert (ad / "ad.md").read_text(encoding="utf-8").strip() in human.content
    # the writers' own rules, not a copy
    assert TRUTHFULNESS_PROTOCOL.strip() in system.content
    rules = writer_rules()
    for key in ("cv_guidelines", "cover_letter_guidelines"):
        assert rules[key] in human.content
    for key in ("cv_goal", "cover_letter_goal"):
        assert rules[key] in system.content
    # never the grader's ground truth
    truth = json.loads((REPO / "evals/fixtures/personas/p03-marta-qa-to-devops/persona.json").read_text())
    assert "deliberate_gaps" not in human.content and truth["traps"][0] not in human.content


def test_without_a_cv_the_prompt_says_so_and_the_missing_details_are_declined():
    from agents.cv_writer import NO_CV_TEXT
    from evals.run_case import DEFAULT_SCRIPT
    from evals.single_prompt import build_messages
    _, human = build_messages({"notes": "some notes"}, "ad", "(the search found nothing)")
    assert NO_CV_TEXT in human.content and DEFAULT_SCRIPT["missing_details"] in human.content
    _, with_cv = build_messages({"cv": "a cv"}, "ad", "(none)")
    assert NO_CV_TEXT not in with_cv.content


def test_the_rules_are_sliced_from_the_writer_prompts():
    from evals.single_prompt import writer_rules
    rules = writer_rules()
    assert rules["cv_goal"].startswith("Your PRIMARY GOAL") and "ATS" in rules["cv_goal"]
    assert rules["cover_letter_guidelines"].startswith("General guidelines") and "3-4 paragraphs" in rules["cover_letter_guidelines"]
    assert "CRITICAL" not in rules["cv_guidelines"]


# --- one run, end to end, stubbed ---------------------------------------------------------------------
def _run(tmp_path, stubbed, inputs="cv_only", extra=()):
    from evals import single_prompt
    run_dir = tmp_path / "run"
    code = single_prompt.main(["--run-dir", str(run_dir), "--ad", "a3", "--persona", "p03", "--inputs", inputs,
                               "--env-file", stubbed["env"], "--label", "method=single_prompt", *extra])
    return code, run_dir


def test_a_run_makes_one_generation_call_and_logs_every_rendering_call(tmp_path, stubbed):
    code, run_dir = _run(tmp_path, stubbed)
    assert code == 0
    schemas = [c["schema"] for c in stubbed["calls"]]
    assert schemas.count("SinglePromptResponse") == 1
    assert schemas == ["SinglePromptResponse", "CVLatexDocumentResponse", "CoverLetterLatexDocumentResponse"]
    logged = [json.loads(line) for line in (run_dir / "llm_calls.jsonl").read_text().splitlines()]
    assert [c["node"] for c in logged] == ["single_prompt", "generate_cv_pdf", "generate_cover_letter_pdf"]
    assert len(logged) == len(stubbed["calls"])  # no model call outside the log
    assert logged[0]["model"] == "gpt-6.1-sol" and logged[1]["model"] == "gpt-6-luna"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["method"] == "single_prompt"
    assert manifest["labels"]["method"] == "single_prompt"
    assert manifest["inputs_given"] == ["cv"] and manifest["inputs_omitted"] == ["portfolio", "cover_letter", "notes"]
    assert manifest["model"] == "openai:gpt-6.1-sol" and manifest["render_model"] == "openai:gpt-6-luna"
    assert manifest["calls"]["generation"]["calls"] == 1 and manifest["calls"]["rendering"]["calls"] == 2
    assert manifest["calls"]["other"]["calls"] == 0
    assert manifest["cost_usd"] == pytest.approx(sum(c["cost_usd"] for c in logged))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["llm_calls"] == 3 and summary["by_node"]["single_prompt"]["calls"] == 1


def test_the_run_folder_is_graded_unchanged_and_blind(tmp_path, stubbed):
    from evals import evaluate
    code, run_dir = _run(tmp_path, stubbed, inputs="no_cv")
    assert code == 0
    for rel in ("manifest.json", "config.json", "llm_calls.jsonl", "summary.json", "transcript.json",
                "inputs/job_ad.txt", "inputs/search_fixture.json", "inputs/notes.txt",
                "outputs/final_cv.txt", "outputs/final_cv.pdf", "outputs/final_cover_letter.txt",
                "outputs/final_cover_letter.pdf"):
        assert (run_dir / rel).is_file(), rel
    assert not (run_dir / "inputs" / "cv.txt").exists()
    values, checks = evaluate.build_prompt(run_dir)
    assert "SUMMARY_FROM_STUB" in values["cv"] and "LETTER_FROM_STUB" in values["cover_letter"]
    assert checks["cv_pdf"]["exists"] and checks["cover_letter_pdf"]["exists"]
    assert _persona_texts("p03")["notes"].strip() in values["candidate"]
    assert _persona_texts("p03")["cv"].strip() not in values["candidate"]
    shown = json.dumps([m.content for m in evaluate.format_messages(values)])
    for leak in ("single_prompt", "single prompt", "multi_agent", "gpt-6.1-sol", "gpt-6-luna", "sp_p03"):
        assert leak not in shown, leak


def test_the_cost_cap_stops_before_rendering(tmp_path, stubbed):
    code, run_dir = _run(tmp_path, stubbed, extra=("--cost-cap", "0.0001"))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert code == 1 and manifest["status"].startswith("stopped: cost cap")
    assert [c["schema"] for c in stubbed["calls"]] == ["SinglePromptResponse"]
    assert (run_dir / "outputs" / "final_cv.txt").is_file() and not (run_dir / "outputs" / "final_cv.pdf").exists()


def test_out_of_credit_marks_the_run_interrupted_and_exits_75(tmp_path, stubbed, monkeypatch):
    from evals import single_prompt
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS, OutOfCredit

    def no_credit(llm, messages, callbacks):
        raise OutOfCredit("insufficient_quota")

    monkeypatch.setattr(single_prompt, "generate", no_credit)
    code, run_dir = _run(tmp_path, stubbed)
    assert code == EXIT_OUT_OF_CREDIT
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == INTERRUPTED_STATUS


# --- the matrix ---------------------------------------------------------------------------------------
def test_the_single_prompt_matrix_resolves_and_mirrors_the_main_baselines():
    from evals.cases import load_matrix, matrix_total_cap, resolve_case
    cases = load_matrix(SP_MATRIX)
    main = {c["id"]: c for c in load_matrix()}
    assert len(cases) == 39 and len({c["id"] for c in cases}) == 39
    assert not {c["id"] for c in cases} & set(main)
    assert {(c["persona"], c["input_set"]) for c in cases} == {
        (f"p{i:02d}", s) for i in range(1, 14) for s in ("all4", "cv_only", "no_cv")}
    for case in cases:
        resolve_case(case)
        assert case["method"] == "single_prompt" and case["model"] == "openai:gpt-6.1-sol" and not case.get("script")
        base = main[case["baseline"]]
        assert (base["persona"], base["ad"], base["inputs"]) == (case["persona"], case["ad"], case["inputs"])
        assert not base.get("script") and not base.get("work_preference")
        truth = json.loads(resolve_case(case)["persona_truth"].read_text())
        assert truth["primary_ad"].split("-", 1)[0] == case["ad"]
    assert matrix_total_cap(SP_MATRIX) == 12.0


def test_a_single_prompt_case_cannot_have_a_script():
    from evals.cases import resolve_case
    with pytest.raises(ValueError, match="cannot have a script"):
        resolve_case({"id": "x", "method": "single_prompt", "persona": "p03", "ad": "a3", "inputs": "all4",
                      "script": "terse"})


def test_the_estimate_stays_under_the_matrix_cap(capsys):
    from evals import run_matrix
    assert run_matrix.main(["--matrix", str(SP_MATRIX), "--out", "unused", "--estimate"]) == 0
    est = json.loads(capsys.readouterr().out)
    assert est["cases"] == 39 and est["total_cap"] == 12.0
    assert est["total_usd"] <= est["total_cap"]
    assert est["by_method"] == {"single_prompt": est["total_usd"]}
    assert round(est["sessions"] + est["evaluations"], 2) == est["total_usd"]


def test_run_matrix_runs_the_single_prompt_runner_with_the_same_ledger(tmp_path, monkeypatch):
    from argparse import Namespace
    from evals import run_matrix
    from evals.cases import load_matrix
    case = load_matrix(SP_MATRIX)[0]
    commands = []

    def fake_run(cmd, *a, **kw):
        commands.append(cmd)
        if cmd[2] == "evals.single_prompt":
            run_dir = Path(cmd[cmd.index("--run-dir") + 1])
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(json.dumps({"status": "completed", "cost_usd": 0.08}))
        else:
            folder = Path(cmd[3]) / "evaluation" / cmd[cmd.index("--name") + 1]
            folder.mkdir(parents=True)
            (folder / "evaluation.json").write_text(json.dumps({"cost_usd": 0.17}))
        return type("Done", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    out = tmp_path / "runs"
    out.mkdir()
    opts = Namespace(out=str(out), env_file=None, total_cap=12.0, cost_cap=1.5, rescore=None, eval_name="eval1",
                     truth_check=False)
    assert run_matrix.run_one(case, opts) == (case["id"], "ok")
    assert [c[2] for c in commands] == ["evals.single_prompt", "evals.evaluate"]
    assert "method=single_prompt" in commands[0] and f"baseline={case['baseline']}" in commands[0]
    ledger = [line.split("\t")[:3] for line in (out / "ledger.tsv").read_text().splitlines()]
    assert ledger == [[case["id"], "app", "0.080000"], [case["id"], "eval", "0.170000"]]
    assert run_matrix.run_one(case, opts)[1] == "skip (done)"


def test_the_main_matrix_keeps_its_24_usd_default_cap():
    from evals import run_matrix
    from evals.cases import DEFAULT_MATRIX
    assert run_matrix.default_total_cap(DEFAULT_MATRIX) == 24.0
    assert run_matrix.default_total_cap(SP_MATRIX) == 12.0


# --- summarize: split by method -----------------------------------------------------------------------
def _fake_run(folder: Path, overall: int, manifest: dict):
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"status": "completed", **manifest}))
    ev_dir = folder / "evaluation" / "eval1"
    ev_dir.mkdir(parents=True)
    ev = {"overall_score": overall, "invented_content_score": 8, "completeness_score": 6,
          "fit": {"score": 5}, "cv_standards": {"score": 8}, "cover_letter": {"score": 7}}
    (ev_dir / "evaluation.json").write_text(json.dumps({"evaluation": ev, "counts": {"cv_unsupported": 1},
                                                        "cost_usd": 0.1}))


def test_summarize_splits_by_method_and_pairs_each_run_with_its_baseline(tmp_path, capsys):
    from evals import summarize
    main, sp = tmp_path / "main", tmp_path / "sp"
    _fake_run(main / "p03_a3_cv_only_luna_r1", 6, {"labels": {"script": "default"}})
    _fake_run(main / "p03_a3_all4_luna_r1", 7, {"labels": {"script": "default"}})
    _fake_run(sp / "sp_p03_a3_cv_only_sol_r1", 8, {"method": "single_prompt",
                                                  "labels": {"method": "single_prompt", "baseline": "p03_a3_cv_only_luna_r1"}})
    result = summarize.method_summary([sp, main])
    assert result["methods"]["single_prompt"]["runs"] == 1 and result["methods"]["multi_agent"]["runs"] == 2
    assert result["methods"]["multi_agent"]["metrics"]["overall"]["mean"] == 6.5
    (pair,) = result["pairs"]
    assert pair["found"] and pair["diff"]["overall"] == 2 and pair["diff"]["invented_content"] == 0
    assert result["paired_mean_diff"]["overall"]["mean"] == 2
    assert summarize.main([str(sp), "--with", str(main)]) == 0
    assert "pairs: sp-ma (n=1)" in capsys.readouterr().out
    assert json.loads((sp / "methods_summary.json").read_text())["pairs"][0]["multi_agent"] == "p03_a3_cv_only_luna_r1"
