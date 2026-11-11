"""The grader's ATS score: judged by the model inside the one grading call, from the plain text a standard
extractor pulls from the CV PDF ("what an ATS would read"), in both grading modes and for every provider.
Offline: scripted grader replies, no model call."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.test_eval_diversity import _evaluation
from tests.test_grader import _case_run, _Client, _message

REPO = Path(__file__).resolve().parent.parent
# A real session of the published round: its final_cv.pdf was compiled by the app's LaTeX templates.
REAL_RUN = REPO / "evals" / "results" / "2026-11-10" / "runs" / "p01_a1_all4_luna_r1"
ATS_HEADING = "# OUTPUT 1b: WHAT AN ATS WOULD READ"
MODES = ("document-only", "conversation")


def _real_run(tmp_path: Path) -> Path:
    """A copy of the real session without its gradings."""
    run = tmp_path / REAL_RUN.name
    shutil.copytree(REAL_RUN, run, ignore=shutil.ignore_patterns("evaluation"))
    return run


def _ats_section(text: str) -> str:
    """The body of the ATS section of a grader prompt: from its heading to the next heading."""
    start = text.index(ATS_HEADING)
    body = text[text.index("\n", start) + 1:]
    return body[:body.index("\n# OUTPUT 2:")].strip()


class _FakeLLM:
    """Stands in for a LangChain chat model (an openai:* grader): returns `answer` as the structured output."""

    def __init__(self, answer: dict):
        self.answer, self.calls = answer, []

    def with_structured_output(self, schema, include_raw=False):
        llm = self

        class _Chain:
            def invoke(self, messages, config=None):
                llm.calls.append(messages)
                from types import SimpleNamespace
                raw = SimpleNamespace(content="", additional_kwargs={}, usage_metadata={},
                                      response_metadata={"model_name": "gpt-6.1-sol", "finish_reason": "stop"})
                return {"raw": raw, "parsed": llm.answer, "parsing_error": None}
        return _Chain()


# --- the input -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_the_ats_section_is_the_text_extracted_from_a_real_compiled_cv_pdf(tmp_path, mode):
    from agents.pdf_text_check import extract_pdf_text
    from evals.evaluate import build_grading_request
    run = _real_run(tmp_path)
    extracted, pages, error = extract_pdf_text(run / "outputs" / "final_cv.pdf")
    assert error is None and pages and len(extracted) > 500
    for grader in ("anthropic:claude-sonnet-5-5", "anthropic:claude-haiku-5-5", "openai:gpt-6.1-sol"):
        request, _, _ = build_grading_request(run, grader, tmp_path / "graded" / mode / grader.replace(":", "_"), mode)
        assert _ats_section(request.document_text) == extracted.strip()
        assert "Judge the ats section from WHAT AN ATS WOULD READ" in request.system
        if request.grader.provider == "anthropic":
            # the Batch params carry the same input, unchanged, next to the page images
            params = request.anthropic_params()
            assert params["system"][0]["text"] == request.system
            assert [b["text"] for b in params["messages"][0]["content"] if b["type"] == "text"] == \
                [request.shared_text, request.document_text]
            assert [b["type"] for b in params["messages"][0]["content"]].count("image") >= 2
        else:
            # the LangChain (OpenAI) form carries it too, next to the page images
            human = request.langchain_messages()[1].content
            assert any(isinstance(p, dict) and p.get("type") == "text" and ATS_HEADING in p["text"] for p in human)
            assert [p.get("type") for p in human if isinstance(p, dict)].count("image_url") >= 2


def test_the_ats_section_reads_the_pdf_not_the_apps_own_cv_text(tmp_path):
    from evals.evaluate import build_grading_request
    run = _case_run(tmp_path, "ma", cv="A CV text that only the app has, never in the PDF.")
    request, _, _ = build_grading_request(run, "anthropic:claude-sonnet-5-5", tmp_path / "graded", "document-only")
    section = _ats_section(request.document_text)
    assert "Backend developer with five years of Python." in section  # the PDF's own words (CLEAN_PAGE)
    assert "only the app has" not in section
    assert "A CV text that only the app has" in request.document_text  # still shown as the CV text


def test_a_cv_without_a_pdf_tells_the_grader_there_is_nothing_to_read(tmp_path):
    from evals.evaluate import build_grading_request
    run = _case_run(tmp_path, "ma")
    (run / "outputs" / "final_cv.pdf").unlink()
    request, _, _ = build_grading_request(run, "anthropic:claude-sonnet-5-5", tmp_path / "graded", "document-only")
    assert _ats_section(request.document_text) == "(no CV PDF was produced)"


# --- the answer ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_the_ats_answer_lands_in_evaluation_json_and_counts(tmp_path, mode):
    from evals import evaluate as ev_mod
    run = _real_run(tmp_path)
    client = _Client(_message(_evaluation()))
    record = ev_mod.evaluate(run, "anthropic:claude-sonnet-5-5", "eval1", mode=mode, client=client)
    assert len(client.calls) == 1  # the ATS score is inside the one grading call
    assert ATS_HEADING in client.calls[0]["messages"][0]["content"][1]["text"]
    ats = record["evaluation"]["ats"]
    assert ats["score"] == 6 and ats["reason"].startswith("Clean text")
    assert ats["text_readable_in_order"] is True and ats["clear_sections"] is True
    assert ats["covered_in_searchable_wording"] == ["Python"] and ats["unsupported_keywords"] == ["Kubernetes"]
    assert record["counts"]["ats_score"] == 6
    assert [k for k in record["counts"] if "ats" in k] == ["ats_score"]  # one ATS number in the counts
    on_disk = json.loads((run / "evaluation" / "eval1" / "evaluation.json").read_text())
    assert on_disk["evaluation"]["ats"] == ats and on_disk["counts"]["ats_score"] == 6
    assert on_disk["grading_mode"] == mode


def test_an_openai_grader_answers_the_same_ats_fields(tmp_path):
    from evals.evaluate import build_grading_request, finish_evaluation, write_prompt
    from evals.grader import send
    from provenance import RunRecorder
    run = _real_run(tmp_path)
    out_dir = run / "evaluation" / "sol1"
    request, checks, images = build_grading_request(run, "openai:gpt-6.1-sol", out_dir, "document-only")
    recorder = RunRecorder(out_dir)
    write_prompt(recorder, request, "document-only", images)
    llm = _FakeLLM(_evaluation())
    result = send(request, recorder, node="evaluator", llm=llm)
    record = finish_evaluation(run, "sol1", request, result, checks, "document-only", recorder)
    sent = llm.calls[0][1].content
    assert any(isinstance(p, dict) and ATS_HEADING in p.get("text", "") for p in sent)
    assert record["evaluation"]["ats"]["score"] == 6 and record["counts"]["ats_score"] == 6


def test_an_ats_score_out_of_range_or_the_old_ats_shape_does_not_parse():
    from evals.evaluate import Evaluation
    from evals.grader import parse_answer
    bad = _evaluation()
    bad["ats"] = {**bad["ats"], "score": 11}
    assert parse_answer(json.dumps(bad), Evaluation)[0] is None
    old = _evaluation()
    old["ats"] = {"pdf_text_parseable": True, "standard_headings": True, "headings_found": [], "ad_keywords": [],
                  "keywords_in_cv": [], "keywords_missing_from_cv": [], "keywords_missing_but_candidate_has": [],
                  "keyword_coverage_pct": 50, "pass_likelihood": "medium", "reasons": []}
    assert parse_answer(json.dumps(old), Evaluation)[0] is None


def test_the_ats_fields_are_in_the_structured_output_schema():
    from evals.evaluate import Evaluation
    from evals.grader import json_schema_for
    schema = json_schema_for(Evaluation)
    ats = schema["$defs"]["ATS"]
    assert ats["required"] == ["text_readable_in_order", "clear_sections", "headings_found", "parse_issues",
                               "important_requirements", "covered_in_searchable_wording",
                               "missed_but_candidate_has", "keyword_stuffing", "unsupported_keywords", "score",
                               "reason"]
    assert ats["additionalProperties"] is False
