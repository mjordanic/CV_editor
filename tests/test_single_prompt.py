"""Offline tests for the single-prompt baseline (evals/single_prompt.py) and its place in run_matrix.

No paid call of any kind: every chat model the code could build is a counting stub, no .env is read (an
empty --env-file is passed), and run_matrix's subprocess is faked. pdflatex is stubbed in the run tests;
the two layout tests that compile for real are skipped where pdflatex is missing."""
import json
import shutil
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

CV_TEXT = "Marta Example\nDevOps Engineer\nSUMMARY_FROM_STUB: CI pipelines with GitHub Actions."
LETTER_TEXT = "Dear hiring team,\nLETTER_FROM_STUB: I would like to apply.\nKind regards, Marta"
GOOD_LATEX = "\\documentclass{article}\\begin{document}GOOD\\end{document}"
BROKEN_LATEX = "\\documentclass{article}\\begin{document}\\undefinedcommand BROKEN\\end{document}"

CALLS: list = []  # every model call of the current test, whatever built the model


class FakeChat(BaseChatModel):
    """Answers the single prompt with fixed JSON; any other request is recorded and fails the test."""

    model_name: str = "gpt-6.1-sol"
    schema_name: str = ""
    answer: dict = {}

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        CALLS.append(self.schema_name or "plain")
        message = AIMessage(content=json.dumps(self.answer),
                            usage_metadata={"input_tokens": 6000, "output_tokens": 9000, "total_tokens": 15000},
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
    with open(path, "wb") as f:
        writer.write(f)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """init_chat_model everywhere (the runner, every app agent module, langchain itself) is a counting stub;
    pdflatex "compiles" any LaTeX without BROKEN in it."""
    import importlib
    import pkgutil

    import langchain.chat_models
    import agents
    from evals import single_prompt
    CALLS.clear()
    answer = {"cv_content": CV_TEXT, "cv_latex": GOOD_LATEX, "cover_letter_content": LETTER_TEXT,
              "cover_letter_latex": GOOD_LATEX, "notes": "n"}

    def fake_init(model, **kwargs):
        return FakeChat(model_name=model.split(":", 1)[-1], answer=answer)

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init)
    for info in pkgutil.iter_modules(agents.__path__):
        module = importlib.import_module(f"agents.{info.name}")
        if hasattr(module, "init_chat_model"):
            monkeypatch.setattr(module, "init_chat_model", fake_init)
    monkeypatch.setattr(single_prompt, "init_chat_model", fake_init)
    compiled = []

    def fake_pdflatex(tex: Path):
        compiled.append(tex.name)
        if "BROKEN" in tex.read_text():
            return False, "! Undefined control sequence."
        _blank_pdf(tex.with_suffix(".pdf"))
        return True, ""

    monkeypatch.setattr(single_prompt, "pdflatex", fake_pdflatex)
    env = tmp_path / "empty.env"
    env.write_text("")
    return {"answer": answer, "compiled": compiled, "env": str(env)}


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
    # the app's own rules, not a copy, and both LaTeX templates
    assert TRUTHFULNESS_PROTOCOL.strip() in system.content
    rules = writer_rules()
    for key in ("cv_guidelines", "cover_letter_guidelines", "cv_latex", "cover_letter_latex"):
        assert rules[key] in human.content
    for key in ("cv_goal", "cover_letter_goal"):
        assert rules[key] in system.content
    for template in ("cv_template.tex", "cover_letter_template.tex"):
        assert (REPO / "templates" / template).read_text(encoding="utf-8") in human.content
    # never the grader's ground truth
    truth = json.loads((REPO / "evals/fixtures/personas/p03-marta-qa-to-devops/persona.json").read_text())
    assert "deliberate_gaps" not in human.content and truth["traps"][0] not in human.content


def test_without_a_cv_the_prompt_says_so_and_never_asks_for_placeholders():
    from evals.cases import INPUT_SETS
    from evals.run_case import DEFAULT_SCRIPT
    from evals.single_prompt import NO_CV_TEXT, build_messages
    for i in range(1, 14):  # every persona's no-CV case: one instruction about placeholders, never two
        texts = _persona_texts(f"p{i:02d}")
        _, human = build_messages({k: texts[k] for k in INPUT_SETS["no_cv"]}, "ad", "(the search found nothing)")
        assert NO_CV_TEXT in human.content and DEFAULT_SCRIPT["missing_details"] in human.content
        assert "leave no placeholder" in human.content and "use placeholders" not in human.content
    _, with_cv = build_messages({"cv": "a cv"}, "ad", "(none)")
    assert NO_CV_TEXT not in with_cv.content


def test_the_rules_are_sliced_from_the_app_prompts():
    from evals.single_prompt import writer_rules
    rules = writer_rules()
    assert rules["cv_goal"].startswith("Your PRIMARY GOAL") and "ATS" in rules["cv_goal"]
    assert rules["cover_letter_guidelines"].startswith("General guidelines") and "3-4 paragraphs" in rules["cover_letter_guidelines"]
    assert "CRITICAL" not in rules["cv_guidelines"]
    assert rules["cv_latex"].startswith("CRITICAL RULES:") and "word for word" in rules["cv_latex"]
    assert "{{" not in rules["cv_latex"] and "\\begin{document}" in rules["cover_letter_latex"]


# --- one run, end to end, stubbed ---------------------------------------------------------------------
def _run(tmp_path, stubbed, inputs="cv_only", extra=()):
    from evals import single_prompt
    run_dir = tmp_path / "run"
    code = single_prompt.main(["--run-dir", str(run_dir), "--ad", "a3", "--persona", "p03", "--inputs", inputs,
                               "--env-file", stubbed["env"], "--label", "method=single_prompt", *extra])
    return code, run_dir


def test_a_case_is_exactly_one_model_call(tmp_path, stubbed):
    code, run_dir = _run(tmp_path, stubbed)
    assert code == 0
    assert CALLS == ["SinglePromptResponse"]  # fails on any further model call, from any module
    logged = [json.loads(line) for line in (run_dir / "llm_calls.jsonl").read_text().splitlines()]
    assert [c["node"] for c in logged] == ["single_prompt"] and logged[0]["model"] == "gpt-6.1-sol"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["calls"] == 1 and manifest["calls_by_node"] == {"single_prompt": 1}
    assert manifest["status"] == "completed" and manifest["method"] == "single_prompt"
    assert manifest["labels"]["method"] == "single_prompt"
    assert manifest["inputs_given"] == ["cv"] and manifest["inputs_omitted"] == ["portfolio", "cover_letter", "notes"]
    assert manifest["model"] == "openai:gpt-6.1-sol" and "render_model" not in manifest
    assert manifest["cost_usd"] == pytest.approx(logged[0]["cost_usd"])
    assert manifest["fallback_pdfs"] == [] and manifest["compile_failures"] == []
    assert {k: v["source"] for k, v in manifest["pdfs"].items()} == {"cv": "model_latex", "cover_letter": "model_latex"}
    assert stubbed["compiled"] == ["final_cv.tex", "final_cover_letter.tex"]
    assert json.loads((run_dir / "summary.json").read_text())["llm_calls"] == 1


def test_latex_that_does_not_compile_gets_a_marked_fallback_pdf_and_no_model_call(tmp_path, stubbed):
    stubbed["answer"]["cv_latex"] = BROKEN_LATEX
    code, run_dir = _run(tmp_path, stubbed)
    assert code == 0 and CALLS == ["SinglePromptResponse"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["calls"] == 1
    assert manifest["compile_failures"] == ["cv"] and manifest["fallback_pdfs"] == ["cv"]
    cv = manifest["pdfs"]["cv"]
    assert cv["compiled"] is False and "Undefined control sequence" in cv["compile_error"]
    assert cv["source"] == "fallback_layout" and cv["pdf"] == "outputs/final_cv.pdf"
    assert manifest["pdfs"]["cover_letter"]["source"] == "model_latex"
    assert stubbed["compiled"] == ["final_cv.tex", "final_cv_fallback.tex", "final_cover_letter.tex"]
    assert (run_dir / "outputs" / "final_cv.pdf").is_file()
    assert (run_dir / "outputs" / "final_cv.tex").read_text() == BROKEN_LATEX  # as the model returned it


def test_the_run_folder_is_graded_unchanged_and_blind(tmp_path, stubbed):
    from evals import evaluate
    code, run_dir = _run(tmp_path, stubbed, inputs="no_cv")
    assert code == 0
    for rel in ("manifest.json", "config.json", "llm_calls.jsonl", "summary.json", "transcript.json", "prompt.json",
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
    shown = json.dumps([m.content for m in evaluate.format_messages(values)])  # document-only, as compared
    for leak in ("single_prompt", "single prompt", "multi_agent", "gpt-6.1-sol", "fallback", "sp_p03"):
        assert leak not in shown, leak


def test_an_infrastructure_error_marks_the_run_interrupted_not_failed(tmp_path, stubbed, monkeypatch):
    import httpx
    import openai
    from evals import evaluate, single_prompt
    from evals.credit import INFRASTRUCTURE_STATUS

    def connection_lost(llm, messages, callbacks):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))

    monkeypatch.setattr(single_prompt, "generate", connection_lost)
    code, run_dir = _run(tmp_path, stubbed)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert code == 1 and manifest["status"] == INFRASTRUCTURE_STATUS and manifest["failure"] is None
    from evals.run_matrix import session_done
    assert not session_done(run_dir)  # never graded as "no CV was produced"; run_matrix runs it again
    assert evaluate.build_prompt(run_dir)[0]["cv"] == "(no CV was produced)"  # what grading it would have meant


def test_an_answer_cut_off_at_the_output_limit_is_a_parse_error_with_its_finish_reason(tmp_path, stubbed, monkeypatch):
    import openai
    from types import SimpleNamespace

    def cut_off(self, messages, stop=None, run_manager=None, **kwargs):
        raise openai.LengthFinishReasonError(completion=SimpleNamespace(usage=None))

    monkeypatch.setattr(FakeChat, "_generate", cut_off)
    code, run_dir = _run(tmp_path, stubbed)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert code == 1 and manifest["failure"] == "parse_error" and manifest["finish_reason"] == "length"


def test_out_of_credit_marks_the_run_interrupted_and_exits_75(tmp_path, stubbed, monkeypatch):
    from evals import single_prompt
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS, OutOfCredit

    def no_credit(llm, messages, callbacks):
        raise OutOfCredit("insufficient_quota")

    monkeypatch.setattr(single_prompt, "generate", no_credit)
    code, run_dir = _run(tmp_path, stubbed)
    assert code == EXIT_OUT_OF_CREDIT
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == INTERRUPTED_STATUS


@pytest.mark.parametrize("model", ["openai:gpt-6-luna", "openai:gpt-6.1-terra", "openai:gpt-6.1-sol"])
def test_each_comparison_model_runs_built_as_the_app_builds_its_writers(tmp_path, stubbed, monkeypatch, model):
    from evals import single_prompt
    from provenance import cost_usd
    built = []
    inner = single_prompt.init_chat_model

    def recording_init(name, **kwargs):
        built.append((name, kwargs))
        return inner(name, **kwargs)

    monkeypatch.setattr(single_prompt, "init_chat_model", recording_init)
    code, run_dir = _run(tmp_path, stubbed, extra=("--model", model))
    assert code == 0 and CALLS == ["SinglePromptResponse"]
    assert built == [(model, {"temperature": None})]  # luna accepts only its default temperature
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["model"] == model and manifest["failure"] is None and manifest["finish_reason"] is None
    logged = json.loads((run_dir / "llm_calls.jsonl").read_text().splitlines()[0])
    assert logged["model"] == model.split(":", 1)[1]
    assert manifest["cost_usd"] == pytest.approx(cost_usd(model, 6000, 9000))  # priced from provenance


def test_a_model_without_a_price_is_refused_before_anything_runs(tmp_path, stubbed):
    with pytest.raises(SystemExit, match="no price"):
        _run(tmp_path, stubbed, extra=("--model", "openai:not-a-priced-model"))
    assert CALLS == [] and not (tmp_path / "run").exists()


def test_the_prompt_and_the_transcript_carry_the_default_users_words(tmp_path, stubbed):
    from evals import evaluate
    from evals.run_case import DEFAULT_SCRIPT
    from evals.single_prompt import build_messages
    _, human = build_messages({"cv": "a cv"}, "ad", "(none)")
    assert f'"{DEFAULT_SCRIPT["first"]}"' in human.content
    assert f'"{DEFAULT_SCRIPT["missing_details"]}"' in human.content
    code, run_dir = _run(tmp_path, stubbed)
    assert code == 0
    transcript = json.loads((run_dir / "transcript.json").read_text())
    assert [t["user"] for t in transcript["turns"]] == [
        "<path to job ad>", DEFAULT_SCRIPT["first"], DEFAULT_SCRIPT["missing_details"]]
    assert all(t["assistant"] is None for t in transcript["turns"])
    assert transcript["user_instructions"] == [{"key": "first", "text": DEFAULT_SCRIPT["first"]},
                                               {"key": "missing_details", "text": DEFAULT_SCRIPT["missing_details"]}]
    assert transcript["requests"] == [] and evaluate.request_checklist(run_dir) == []
    shown = evaluate.conversation_text(run_dir)
    assert DEFAULT_SCRIPT["first"] in shown and "no conversation was recorded" not in shown
    # the same user turns as an app session in which the app asked for missing details
    assert json.loads((run_dir / "inputs" / "script.json").read_text()) == DEFAULT_SCRIPT


def test_an_answer_that_does_not_parse_is_a_failed_session_that_is_still_graded(tmp_path, stubbed, monkeypatch):
    from evals import evaluate, single_prompt

    def unparsed(llm, messages, callbacks):
        raise single_prompt.AnswerNotParsed("the model's answer did not parse: truncated")

    monkeypatch.setattr(single_prompt, "generate", unparsed)
    code, run_dir = _run(tmp_path, stubbed)
    assert code == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"].startswith("error: AnswerNotParsed") and manifest["failure"] == "parse_error"
    values, _ = evaluate.build_prompt(run_dir)
    assert values["cv"] == "(no CV was produced)" and values["cover_letter"] == "(no cover letter was produced)"


# --- the fallback layout, compiled for real -----------------------------------------------------------
needs_pdflatex = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")


def test_the_fallback_layout_escapes_everything_and_drops_only_unprintable_characters():
    from evals.single_prompt import fallback_latex
    latex, dropped = fallback_latex("Ana Kovačić — Zagreb\nC# & R&D, 30% $5 a_b {x} ~ ^ \\ → ok \U0001F600\n\nNext")
    body = latex.split("\\begin{document}", 1)[1]
    for raw in ("\\#", "\\&", "\\%", "\\$", "\\_", "\\{", "\\}", "\\textbackslash{}"):
        assert raw in body
    assert "Kovačić" in body and "---" in body and "->" in body
    assert dropped == ["\U0001F600"]


@needs_pdflatex
def test_the_fallback_layout_compiles_and_keeps_the_text(tmp_path):
    from agents.pdf_text_check import check_pdf
    from evals.single_prompt import make_pdfs
    texts = {"cv": "Ana Kovačić\nData engineer, 30% faster C# & SQL\n\nZagreb, Straße 5",
             "cover_letter": "Sehr geehrte Damen und Herren,\nich bewerbe mich."}
    good = (REPO / "templates" / "cover_letter_template.tex").read_text(encoding="utf-8").replace(
        "{{COVER_LETTER_HEADER}}", "").replace("{{COVER_LETTER_CONTENT}}", texts["cover_letter"])
    result = make_pdfs(texts, {"cv": BROKEN_LATEX, "cover_letter": good}, tmp_path)
    assert result["cv"]["source"] == "fallback_layout" and result["cv"]["fallback_error"] is None
    assert result["cover_letter"]["source"] == "model_latex"
    check = check_pdf(result["cv"]["pdf"], texts["cv"])
    assert check["ok"], check


# --- the matrix ---------------------------------------------------------------------------------------
def test_a_single_prompt_case_cannot_have_a_script():
    from evals.cases import resolve_case
    with pytest.raises(ValueError, match="cannot have a script"):
        resolve_case({"id": "x", "method": "single_prompt", "persona": "p03", "ad": "a3", "inputs": "all4",
                      "script": "terse"})


def test_run_matrix_runs_the_single_prompt_runner_with_the_same_ledger(tmp_path, monkeypatch):
    from argparse import Namespace
    from evals import run_matrix
    case = {"id": "sp_p03_a3_all4_sol_r1", "method": "single_prompt", "persona": "p03", "ad": "a3",
            "input_set": "all4", "inputs": "all4", "model": "openai:gpt-6.1-sol", "rep": 1}
    commands = []

    def fake_run(cmd, *a, **kw):
        commands.append(cmd)
        if cmd[2] == "evals.single_prompt":
            run_dir = Path(cmd[cmd.index("--run-dir") + 1])
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(json.dumps({"status": "completed", "cost_usd": 0.08}))
        else:
            folder = Path(cmd[3]) / "evaluation" / cmd[cmd.index("--eval-name") + 1]
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
    assert "method=single_prompt" in commands[0]
    ledger = [line.split("\t")[:3] for line in (out / "ledger.tsv").read_text().splitlines()]
    assert ledger == [[case["id"], "app", "0.080000"], [case["id"], "eval", "0.170000"]]
    assert run_matrix.run_one(case, opts)[1] == "skip (done)"


def test_the_main_matrix_keeps_its_24_usd_default_cap():
    from evals import run_matrix
    from evals.cases import DEFAULT_MATRIX
    assert run_matrix.default_total_cap(DEFAULT_MATRIX) == 24.0


@pytest.mark.parametrize("method", ["evals.run_case", "evals.single_prompt"])
def test_run_matrix_runs_an_infrastructure_failure_again_once_for_both_methods(tmp_path, monkeypatch, method):
    from argparse import Namespace
    from evals import run_matrix
    from evals.credit import INFRASTRUCTURE_STATUS
    case = {"id": "c_r1", "persona": "p03", "ad": "a3", "input_set": "all4", "inputs": "all4", "rep": 1,
            "model": "openai:gpt-6-luna", **({"method": "single_prompt"} if method == "evals.single_prompt" else {})}
    statuses, commands = [INFRASTRUCTURE_STATUS, "completed"], []

    def fake_run(cmd, *a, **kw):
        commands.append(cmd)
        if cmd[2] == method:
            run_dir = Path(cmd[cmd.index("--run-dir") + 1])
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(json.dumps({"status": statuses.pop(0), "cost_usd": 0.01}))
        else:
            folder = Path(cmd[3]) / "evaluation" / cmd[cmd.index("--eval-name") + 1]
            folder.mkdir(parents=True)
            (folder / "evaluation.json").write_text(json.dumps({"cost_usd": 0.1}))
        return type("Done", (), {"stdout": "", "stderr": "", "returncode": 1})()

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    out = tmp_path / "runs"
    out.mkdir()
    opts = Namespace(out=str(out), env_file=None, total_cap=5.0, cost_cap=1.5, rescore=None, eval_name="eval1",
                     truth_check=False)
    assert run_matrix.run_one(case, opts) == ("c_r1", "ok")  # the second attempt completed and was graded
    assert [c[2] for c in commands] == [method, method, "evals.evaluate"]
    assert "attempt=2" in commands[1] and "attempt=2" not in commands[0]
    commands.clear()
    statuses[:] = [INFRASTRUCTURE_STATUS, INFRASTRUCTURE_STATUS]
    case["id"] = "d_r1"
    assert "infrastructure error, twice" in run_matrix.run_one(case, opts)[1]
    assert [c[2] for c in commands] == [method, method]  # never graded
