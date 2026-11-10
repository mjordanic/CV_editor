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
