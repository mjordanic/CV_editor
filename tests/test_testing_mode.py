"""Offline tests for testing mode: free-form notes input, search fixtures, provenance, PDF text layer,
the scripted user and the evaluator's (blind) input. No model or web calls."""
import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents import critique, document_reader, search
from agents.latex_fixer import ensure_text_extractable

REPO = Path(__file__).resolve().parent.parent


# --- free-form notes -------------------------------------------------------------------------
def test_notes_are_read_and_a_missing_cv_is_not_asked_for_when_notes_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(document_reader, "write_to_debug", lambda *_, **__: None)
    (tmp_path / "notes.md").write_text("- 2019-2024 support engineer, wrote Ansible playbooks", encoding="utf-8")

    def no_question(*_, **__):
        raise AssertionError("asked for a CV although notes were given")

    monkeypatch.setattr(document_reader, "ask_for_document", no_question)
    out = document_reader.DocumentReaderAgent(cv_folder=str(tmp_path)).run({"messages": []})

    assert out["candidate_text"]["cv"] is None
    assert "Ansible playbooks" in out["candidate_text"]["notes"]


def test_without_cv_and_notes_the_cv_is_still_asked_for(tmp_path, monkeypatch):
    monkeypatch.setattr(document_reader, "write_to_debug", lambda *_, **__: None)
    (tmp_path / "notes.txt").write_text("   \n", encoding="utf-8")  # empty notes do not count
    monkeypatch.setattr(document_reader, "ask_for_document", lambda message, what: "pasted rough CV")
    out = document_reader.DocumentReaderAgent(cv_folder=str(tmp_path)).run({"messages": []})
    assert out["candidate_text"]["cv"] == "pasted rough CV"
    assert out["candidate_text"]["notes"] is None


class _Capture:
    """Stands in for a writer model: remembers the prompt, returns a fixed document."""

    def __init__(self, schema_field):
        self.prompts, self.schema_field = [], schema_field

    def with_structured_output(self, schema):
        from langchain_core.runnables import RunnableLambda

        def answer(prompt):
            self.prompts.append(prompt.to_string())
            return schema(**{self.schema_field: "DOC", "notes": ""})
        return RunnableLambda(answer)


@pytest.mark.parametrize("module_name,cls,field", [
    ("cv_writer", "CVWriterAgent", "cv_content"),
    ("cover_letter_writer", "CoverLetterWriterAgent", "cover_letter_content"),
])
def test_writers_get_the_notes(module_name, cls, field, tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module(f"agents.{module_name}")
    llm = _Capture(field)
    monkeypatch.setattr(module, "init_chat_model", lambda *_, **__: llm)
    monkeypatch.setattr(module, "write_to_debug", lambda *_, **__: None)
    agent = getattr(module, cls)(output_folder=str(tmp_path))
    agent.run({"candidate_text": {"cv": None, "cover_letter": None, "notes": "NOTE-FACT: ran the on-call rota"},
               "job_description_info": {"job_title": "SRE"}, "messages": []})
    assert "NOTE-FACT: ran the on-call rota" in llm.prompts[0]
    if module_name == "cv_writer":
        assert "only from the facts in the candidate's notes" in llm.prompts[0]


def test_grounding_and_critique_see_the_notes():
    agent = critique.CritiqueAgent.__new__(critique.CritiqueAgent)
    text = agent._format_candidate_info({"cv": "CV", "notes": "NOTE-FACT"})
    assert "Candidate's Own Notes:\nNOTE-FACT" in text


# --- search fixtures -------------------------------------------------------------------------
def _fixture(tmp_path, searches):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"fixture_version": 1, "company": "X", "searches": searches}), encoding="utf-8")
    return str(path)


def test_fixture_replay_never_calls_tavily(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "init_chat_model", lambda *_, **__: None)
    monkeypatch.setattr(search, "TavilyClient", lambda **_: pytest.fail("live search in replay mode"))
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")
    path = _fixture(tmp_path, [{"query": "a", "response": {"results": [{"content": "A"}]}},
                               {"query": "b", "response": {"results": [{"content": "B"}]}}])
    agent = search.SearchAgent(fixture_path=path)
    assert agent.search_tavily("b")["results"][0]["content"] == "B"  # exact query
    assert agent.search_tavily("other")["results"][0]["content"] == "A"  # else the first answer


def test_record_mode_saves_live_answers_in_fixture_format(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "init_chat_model", lambda *_, **__: None)

    class FakeTavily:
        def __init__(self, **_):
            pass

        def search(self, query, **_):
            return {"results": [{"content": f"answer to {query}"}]}

    monkeypatch.setattr(search, "TavilyClient", FakeTavily)
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")
    out = tmp_path / "rec" / "fixture.json"
    agent = search.SearchAgent(record_path=str(out))
    agent.search_tavily("q1")
    replay = search.SearchAgent(fixture_path=str(out))
    assert replay.search_tavily("q1")["results"][0]["content"] == "answer to q1"


def test_empty_fixture_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        search.load_search_fixture(_fixture(tmp_path, []))


def test_fixture_from_pages(tmp_path):
    from evals.search_fixtures import fixture_from_pages
    page = tmp_path / "about.md"
    page.write_text("# About Tessel\n(fictional page - tessel.example/about)\nWe build software.", encoding="utf-8")
    fx = fixture_from_pages("Tessel", [page])
    result = fx["searches"][0]["response"]["results"][0]
    assert result["title"] == "About Tessel" and "We build software." in result["content"]
    assert result["url"].startswith("tessel.example")


# --- provenance ------------------------------------------------------------------------------
def test_recorder_logs_model_calls_with_node_and_scrubs_keys(tmp_path):
    from provenance import RunRecorder, scrub, stream_turn

    class S(TypedDict, total=False):
        x: str

    llm = FakeListChatModel(responses=["the key is sk-abcdefghijklmnopqrstuvwxyz123456"])
    graph = StateGraph(S)
    graph.add_node("writer", lambda state: {"x": llm.invoke("hi").content})
    graph.add_edge(START, "writer")
    graph.add_edge("writer", END)
    recorder = RunRecorder(tmp_path)
    assert stream_turn(graph.compile(checkpointer=MemorySaver()), {"x": ""}, recorder.graph_config("t"), recorder) is None

    calls = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and calls[0]["node"] == "writer"
    assert "sk-abcdef" not in (tmp_path / "llm_calls.jsonl").read_text()
    assert "sk-abcdef" not in (tmp_path / "events.jsonl").read_text()
    nodes = [json.loads(line).get("node") for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "writer" in nodes
    assert scrub({"openai_api_key": "x", "n": 1}) == {"openai_api_key": "[REDACTED]", "n": 1}


def test_cost_uses_longest_price_prefix():
    from provenance import cost_usd
    assert cost_usd("gpt-6.1-terra-2026-10-01", 1_000_000, 0) == pytest.approx(1.0)
    assert cost_usd("openai:gpt-6-luna", 0, 1_000_000) == pytest.approx(0.5)
    assert cost_usd("unknown-model", 10, 10) is None


# --- PDF text layer --------------------------------------------------------------------------
def test_generated_latex_gets_an_extractable_text_layer_once():
    doc = "\\documentclass{article}\n\\usepackage{hyperref}\n\\begin{document}\nHello\n\\end{document}"
    fixed = ensure_text_extractable(doc)
    head, body = fixed.split("\\begin{document}")
    assert "\\pdfgentounicode" in head and body.lstrip().startswith("\\raggedright")
    assert ensure_text_extractable(fixed) == fixed
    assert ensure_text_extractable("no document") == "no document"


# --- scripted user and evaluator input -------------------------------------------------------
def test_scripted_user_follows_the_session():
    from evals.run_case import DEFAULT_SCRIPT, ScriptedUser
    user = ScriptedUser(DEFAULT_SCRIPT, "/ad.txt")
    assert user.reply(("analyze_job_description",), {}) == "/ad.txt"
    with pytest.raises(RuntimeError):
        user.reply(("analyze_job_description",), {})
    assert user.reply(("collect_user_input",), {}) == DEFAULT_SCRIPT["first"]
    assert user.reply(("collect_user_input",), {"generated_cv": "CV"}) == DEFAULT_SCRIPT["cover_letter"]
    assert user.reply(("collect_user_input",), {"generated_cv": "CV", "generated_cover_letter": "CL"}) == DEFAULT_SCRIPT["exit"]
    with pytest.raises(RuntimeError):
        user.reply(("load_candidate_documents",), {})


def test_evaluator_input_is_blind_and_has_only_given_inputs(tmp_path):
    from evals.evaluate import build_prompt, counts
    run = tmp_path / "p01_ad01_cv_only_luna_r1"
    (run / "inputs").mkdir(parents=True)
    (run / "outputs").mkdir()
    (run / "inputs" / "cv.txt").write_text("CV TEXT", encoding="utf-8")
    (run / "inputs" / "job_ad.txt").write_text("AD TEXT", encoding="utf-8")
    (run / "outputs" / "final_cv.txt").write_text("OUT CV", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"model_override": "openai:gpt-6-luna"}), encoding="utf-8")
    values, checks = build_prompt(run)
    joined = json.dumps(values)
    assert "CV TEXT" in joined and "AD TEXT" in joined and "OUT CV" in joined
    assert "PORTFOLIO" not in joined and "NOTES" not in joined
    assert "luna" not in joined and "cv_only" not in joined
    assert checks["cv_pdf"] == {"exists": False}
    ev = {"claims": [{"document": "cv", "verdict": "unsupported"}], "omissions": [],
          "fit": {"requirements": [{"priority": "must_have", "shown_in_cv": "yes", "gap_handling": "not_a_gap"}]}}
    assert counts(ev)["cv_unsupported"] == 1 and counts(ev)["must_haves_shown"] == 1


@pytest.mark.parametrize("line,expected", [
    # found in evaluation runs: the rest of the line vanished, or the amounts became math
    ("by approximately 30%. In personal projects, I built", "by approximately 30\\%. In personal projects, I built"),
    ("bei höchstens 28 %. HACCP-Dokumentation", "bei höchstens 28 \\%. HACCP-Dokumentation"),
    ("from approximately $38,000 to $24,000 per month", "from approximately \\$38,000 to \\$24,000 per month"),
    ("a budget of $40,000.", "a budget of \\$40,000."),
    # left alone: already escaped, real math, comments
    ("cut costs by 30\\% and $5$", "cut costs by 30\\% and $5$"),
    ("cold room ($0$--$4\\,^\\circ\\mathrm{C}$) checks", "cold room ($0$--$4\\,^\\circ\\mathrm{C}$) checks"),
    ("Python $\\cdot$ SQL $\\cdot$ dbt", "Python $\\cdot$ SQL $\\cdot$ dbt"),
    ("\\cvheader{A}{B}{C}{D}% layout comment", "\\cvheader{A}{B}{C}{D}% layout comment"),
])
def test_plain_text_percent_and_dollar_are_escaped(line, expected):
    from agents.latex_fixer import escape_text_specials
    doc = "\\documentclass{article}\n% 30% in the preamble stays\n\\begin{document}\n" + line + "\n\\end{document}"
    out = escape_text_specials(doc)
    assert out.split("\\begin{document}\n", 1)[1].split("\n")[0] == expected
    assert "% 30% in the preamble stays" in out
    assert escape_text_specials(out) == out


# --- testing mode end to end, offline -------------------------------------------------------
class _FakeModels:
    """Every chat model in the app: answers by schema, drives CV -> cover letter -> exit."""

    def __init__(self):
        self.router = ["collect_user_input", "draft_cv", "collect_user_input", "draft_cover_letter",
                       "collect_user_input", "exit"]
        self.prompts = []

    def with_structured_output(self, schema):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda prompt, schema=schema: self._answer(schema, prompt))

    def invoke(self, prompt):
        return type("Msg", (), {"content": "YES"})()

    def _answer(self, schema, prompt):
        name = schema.__name__
        self.prompts.append((name, prompt.to_string()))
        if name == "JobDescriptionInfo":
            return schema(company_name="Tessel Cargo Software", job_title="Backend Engineer", work_type="hybrid")
        if name == "RouterResponse":
            action = self.router.pop(0)
            return schema(next_action=action, message_to_user=f"next: {action}",
                          needs_user_input=action == "collect_user_input")
        if name == "CVGenerationResponse":
            return schema(cv_content="Jane Doe\nBackend developer, 5 years Python", notes="")
        if name == "CoverLetterGenerationResponse":
            return schema(cover_letter_content="Dear Tessel team, ...", notes="")
        if name == "CritiqueResponse":
            return schema(quality_score=90, content_quality_feedback="ok", ats_compatibility_feedback="ok",
                          job_alignment_feedback="ok", improvement_instructions="", critical_issues="")
        if name == "GroundingResponse":
            return schema(unsupported_claims=[])
        if name == "RemoteWorkResponse":
            return schema(supports_remote="NA")
        raise AssertionError(f"unexpected structured call: {name}")


def test_run_case_writes_a_complete_run_folder_offline(tmp_path, monkeypatch):
    import subprocess
    from evals import run_case
    from agents import (cover_letter_pdf_generator, cover_letter_writer, cv_pdf_generator, cv_writer,
                        job_description, latex_fixer, rag_agent, router)

    # restore everything run_case changes globally
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", subprocess.run)
    monkeypatch.setattr(rag_agent, "_embeddings_for", rag_agent._embeddings_for)
    for var in ("CV_EDITOR_MODEL", "CV_EDITOR_SEARCH_FIXTURE", "CV_EDITOR_SEARCH_RECORD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-no-calls")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy-no-calls")

    fake = _FakeModels()
    for module in (router, cv_writer, cover_letter_writer, critique, job_description, search, rag_agent,
                   cv_pdf_generator, cover_letter_pdf_generator, latex_fixer):
        monkeypatch.setattr(module, "init_chat_model", lambda *_, **__: fake)

    def fake_compile(self, latex, output_name, max_fix_iterations=1):
        path = Path(self.output_folder) / f"{output_name}.pdf"
        path.write_text(latex, encoding="utf-8")
        return str(path), 0, None

    for gen in (cv_pdf_generator.CVPDFGenerator, cover_letter_pdf_generator.CoverLetterPDFGenerator):
        monkeypatch.setattr(gen, "_build_latex_document", lambda self, template, text, style: text)
        monkeypatch.setattr(gen, "_compile_latex", fake_compile)

    fixtures = REPO / "evals" / "fixtures"
    persona = fixtures / "personas" / "p07-samuel-cloud-gaps"
    env = tmp_path / "empty.env"
    env.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
    code = run_case.main([
        "--run-dir", str(run_dir), "--job-ad", str(fixtures / "ads" / "a5-backend-engineer" / "ad.md"),
        "--search-fixture", str(fixtures / "ads" / "a5-backend-engineer" / "search_fixture.json"),
        "--cv", str(persona / "cv.txt"), "--notes", str(persona / "notes.txt"),
        "--env-file", str(env), "--label", "case=offline",
    ])

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert code == 0, manifest["status"]
    assert manifest["inputs_given"] == ["cv", "notes"]
    assert manifest["inputs_omitted"] == ["portfolio", "cover_letter"]
    assert sorted(p.name for p in (run_dir / "inputs").iterdir()) == [
        "cv.txt", "job_ad.txt", "notes.txt", "script.json", "search_fixture.json"]
    assert (run_dir / "outputs" / "final_cv.txt").read_text().startswith("Jane Doe")
    assert (run_dir / "outputs" / "final_cover_letter.txt").exists()
    transcript = json.loads((run_dir / "transcript.json").read_text())
    assert [t["user"] for t in transcript["turns"]][0] == "<path to job ad>"
    assert transcript["observations"] == []
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["router_decision"] for e in events if e.get("router_decision")][:2] == ["collect_user_input", "draft_cv"]
    company = json.loads((run_dir / "outputs" / "company_info.json").read_text())
    assert company["search_source"].startswith("fixture:")
    # the notes reached the CV writer; the omitted cover letter did not
    cv_prompt = next(p for n, p in fake.prompts if n == "CVGenerationResponse")
    assert "Candidate's Own Notes" in cv_prompt and (persona / "notes.txt").read_text()[:40] in cv_prompt
    config = json.loads((run_dir / "config.json").read_text())
    assert "agents" in config and "OPENAI_API_KEY" not in json.dumps(config)
