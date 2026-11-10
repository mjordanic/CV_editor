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


@pytest.mark.parametrize("fonts", ["\\usepackage{helvet}\n\\renewcommand{\\familydefault}{\\sfdefault}",
                                   "\\usepackage{mathptmx}", "\\usepackage{fontspec}\n\\setmainfont{Arial}"])
def test_the_text_layer_never_overrides_a_font_the_document_chose(fonts):
    doc = f"\\documentclass[11pt,a4paper]{{article}}\n{fonts}\n\\begin{{document}}\nHello\n\\end{{document}}"
    fixed = ensure_text_extractable(doc)
    assert fixed.startswith("\\documentclass[11pt,a4paper]{article}\n")
    assert fixed.index("\\usepackage{lmodern}") < fixed.index(fonts) < fixed.index("\\begin{document}")


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
        if name == "WorkContext":
            return schema(arrangement="on-site", location_limits="Antwerp office, 4 days a week",
                          culture_signals=["small teams that own what they run"])
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
        "--env-file", str(env), "--label", "case=offline", "--work-preference", "remote",
    ])

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert code == 0, manifest["status"]
    assert manifest["inputs_given"] == ["cv", "notes"]
    assert manifest["inputs_omitted"] == ["portfolio", "cover_letter"]
    assert sorted(p.name for p in (run_dir / "inputs").iterdir()) == [
        "cv.txt", "job_ad.txt", "notes.txt", "script.json", "search_fixture.json", "work_preference.txt"]
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
    # arrangement, culture and the stated preference reach the writers, with the conflict spelled out
    assert "Work arrangement (from the ad and company pages): on-site" in cv_prompt
    assert "Candidate's own work-arrangement preference (stated by the candidate): remote" in cv_prompt
    assert "Preference conflict: You said you prefer remote work, but this job is on-site" in cv_prompt
    assert manifest["work_preference"] == "remote" and manifest["work_context"]["arrangement"] == "on-site"
    assert (run_dir / "inputs" / "work_preference.txt").read_text().strip() == "remote"
    from evals.evaluate import build_prompt
    assert build_prompt(run_dir)[0]["work_preference"] == "remote"
    config = json.loads((run_dir / "config.json").read_text())
    assert "agents" in config and "OPENAI_API_KEY" not in json.dumps(config)


# --- work arrangement, culture and preference -------------------------------------------------
def test_work_context_lines_never_assume_a_preference():
    from agents.work_context import describe, preference_conflict
    info = {"work_context": {"arrangement": "hybrid", "location_limits": "not stated", "culture_signals": []}}
    text = describe(info, None)
    assert "Work arrangement (from the ad and company pages): hybrid" in text
    assert "Culture signals: not stated" in text
    assert "preference: not given. Treat the candidate as open to this job's arrangement" in text
    assert "Preference conflict" not in describe(info, "hybrid")
    assert "Preference conflict" in describe(info, "remote")
    assert preference_conflict("no preference", "on-site") is None
    assert preference_conflict("remote", "not stated") is None


@pytest.mark.parametrize("preference", [None, "no preference"])
@pytest.mark.parametrize("arrangement", ["remote", "hybrid", "on-site", "not stated"])
def test_without_a_stated_preference_the_default_is_inclusive(preference, arrangement):
    from agents.work_context import describe, preference_conflict
    info = {"work_context": {"arrangement": arrangement, "location_limits": "Kraków office", "culture_signals": []}}
    text = describe(info, preference)
    # open to every arrangement, relocation included; never flagged as a conflict or weakness
    assert "open to this job's arrangement (remote, hybrid, on-site or relocation)" in text
    assert "Never present the arrangement or location as a conflict, gap or weakness" in text
    assert "never narrow the candidate's options" in text
    assert "Preference conflict" not in text and preference_conflict(preference, arrangement) is None
    # honest: no specific commitment the candidate did not make; neutral open wording allowed
    assert "Do not invent a specific commitment" in text and "willing to relocate to <place>" in text
    assert "neutral, open wording" in text


def test_the_inclusive_default_reaches_every_prompt_even_without_company_research():
    from agents import cover_letter_writer, cv_writer, router
    from agents.work_context import with_preference
    info = with_preference(None, None)
    assert info == {"work_preference": None}
    for formatter in (cv_writer.CVWriterAgent._format_company_info, cover_letter_writer.CoverLetterWriterAgent._format_company_info,
                      critique.CritiqueAgent._format_company_info, router.RouterAgent._format_company_info):
        assert "Treat the candidate as open to this job's arrangement" in formatter(None, info)


def _router_prompt(monkeypatch, state) -> str:
    from langchain_core.runnables import RunnableLambda
    from agents import router as router_module

    seen = []

    class CapturingRouterLLM:
        def with_structured_output(self, schema):
            def reply(prompt):
                seen.append(prompt.to_string())
                return router_module.RouterResponse(next_action="draft_cv", message_to_user="", needs_user_input=False)
            return RunnableLambda(reply)

    monkeypatch.setattr(router_module, "init_chat_model", lambda *_, **__: CapturingRouterLLM())
    router_module.RouterAgent(model="fake", temperature=None).run(state)
    return seen[-1]


def test_the_router_prompt_keeps_an_unstated_preference_open(monkeypatch):
    prompt = _router_prompt(monkeypatch, {"messages": [{"role": "user", "content": "Write my CV."}],
                                          "job_description_info": {"job_title": "X"}, "company_info": None})
    assert "never raise the arrangement, location or relocation as a problem" in prompt
    assert "Treat the candidate as open to this job's arrangement" in prompt


def test_the_router_sees_the_candidates_notes(monkeypatch):
    prompt = _router_prompt(monkeypatch, {"messages": [{"role": "user", "content": "Write my CV."}],
                                          "job_description_info": {"job_title": "X"}, "company_info": None,
                                          "candidate_text": {"cv": None, "cover_letter": None,
                                                             "notes": "- 2019-2024 support engineer, Ansible"}})
    assert "Original CV: Not provided" in prompt
    assert "Candidate's Own Notes: - 2019-2024 support engineer, Ansible" in prompt




def test_a_stated_preference_replaces_the_open_default():
    from agents.work_context import describe
    info = {"work_context": {"arrangement": "on-site", "location_limits": "not stated", "culture_signals": []}}
    text = describe(info, "remote")
    assert "Treat the candidate as open" not in text
    assert "Candidate's own work-arrangement preference (stated by the candidate): remote" in text
    assert "Preference conflict" in text


def test_evaluator_treats_a_missing_preference_as_open(tmp_path):
    from evals.evaluate import build_prompt, format_messages
    (tmp_path / "inputs").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "inputs" / "job_ad.txt").write_text("On-site in Antwerp.", encoding="utf-8")
    values, _ = build_prompt(tmp_path)
    assert values["work_preference"].startswith("(not given")
    system, human = format_messages(values)
    assert "candidate counts as open to the job's arrangement" in system.content
    assert values["work_preference"] in human.content and "On-site in Antwerp." in human.content


def _search_agent(monkeypatch, llm, tmp_path):
    monkeypatch.setattr(search, "init_chat_model", lambda *_, **__: llm)
    monkeypatch.setattr(search, "write_to_debug", lambda *_, **__: None)
    path = _fixture(tmp_path, [{"query": "*", "response": {"results": [{"content": "We are remote-first."}]}}])
    return search.SearchAgent(fixture_path=path)


def test_search_extracts_work_context_from_ad_and_pages(monkeypatch, tmp_path):
    fake = _FakeModels()
    fake.prompts.clear()
    monkeypatch.setattr(search.SearchAgent, "process_search_results", lambda self, results: "A company.")
    agent = _search_agent(monkeypatch, fake, tmp_path)
    out = agent.run({"job_description_info": {"company_name": "X", "job_description": "AD-TEXT: office in Antwerp"},
                     "messages": []})
    info = out["company_info"]
    assert info["work_context"]["arrangement"] == "on-site" and info["remote_work"] == "no"
    prompt = next(p for n, p in fake.prompts if n == "WorkContext")
    assert "AD-TEXT: office in Antwerp" in prompt and "We are remote-first." in prompt
    assert "Work arrangement (from the ad and company pages): on-site" in out["messages"][-1]["content"]


def test_failed_extraction_falls_back_to_the_ads_work_type(monkeypatch, tmp_path):
    class Broken:
        def with_structured_output(self, schema):
            raise RuntimeError("model down")

    monkeypatch.setattr(search.SearchAgent, "process_search_results", lambda self, results: "A company.")
    agent = _search_agent(monkeypatch, Broken(), tmp_path)
    out = agent.run({"job_description_info": {"company_name": "X", "work_type": "unknown"}, "messages": []})
    assert out["company_info"]["work_context"]["arrangement"] == "not stated"
    assert out["company_info"]["remote_work"] == "NA"


def test_router_keeps_a_preference_only_when_the_user_states_it(monkeypatch):
    from langchain_core.runnables import RunnableLambda
    from agents import router as router_module

    replies = []

    class FakeRouterLLM:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda prompt: replies.pop(0))

    monkeypatch.setattr(router_module, "init_chat_model", lambda *_, **__: FakeRouterLLM())
    agent = router_module.RouterAgent(model="fake", temperature=None)
    state = {"messages": [{"role": "user", "content": "I only want remote jobs. Write my CV."}],
             "job_description_info": {"job_title": "X"}, "company_info": None}
    replies.append(router_module.RouterResponse(next_action="draft_cv", message_to_user="", needs_user_input=False,
                                                work_preference="remote"))
    assert agent.run(state)["work_preference"] == "remote"
    replies.append(router_module.RouterResponse(next_action="draft_cv", message_to_user="", needs_user_input=False))
    assert "work_preference" not in agent.run(state)


# --- tracked fixtures and the evaluation matrix ------------------------------------------------
def _tracked_files() -> set[Path]:
    import subprocess
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "evals"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return {(REPO / p).resolve() for p in out.split("\0") if p}


def test_every_matrix_case_resolves_to_tracked_fixture_files():
    from evals.cases import load_matrix, resolve_case
    tracked = _tracked_files()
    cases = load_matrix()
    assert cases and len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        paths = resolve_case(case)
        files = [paths["job_ad"], paths["search_fixture"], paths["persona_truth"], *paths["inputs"].values()]
        assert paths["inputs"], case["id"]
        for f in files:
            assert f.resolve().is_relative_to((REPO / "evals" / "fixtures").resolve()), (case["id"], f)
            assert f.resolve() in tracked, f"{case['id']}: {f} is not tracked (or ignored)"
            assert f.read_text(encoding="utf-8").strip(), f"{case['id']}: {f} is empty"
        # the ids the case uses match the folders it resolves to
        assert paths["job_ad"].parent.name.split("-")[0] == case["ad"]
        assert paths["persona_truth"].parent.name.split("-")[0] == case["persona"]


def test_all_eleven_personas_and_six_ads_are_tracked():
    from evals.cases import FIXTURES
    personas = sorted(p.name.split("-")[0] for p in (FIXTURES / "personas").iterdir() if p.is_dir())
    ads = sorted(p.name.split("-")[0] for p in (FIXTURES / "ads").iterdir() if p.is_dir())
    assert personas == [f"p{i:02d}" for i in range(1, 12)]
    assert ads == [f"a{i}" for i in range(1, 7)]
    for persona in (FIXTURES / "personas").iterdir():
        assert json.loads((persona / "persona.json").read_text())["fictional"] is True


def test_search_fixtures_have_the_recorded_shape_and_no_real_urls():
    import re
    from urllib.parse import urlparse
    from evals.cases import FIXTURES
    for fixture_file in sorted((FIXTURES / "ads").glob("*/search_fixture.json")):
        fixture = json.loads(fixture_file.read_text(encoding="utf-8"))
        assert fixture["fixture_version"] == 1 and fixture["company"]
        response = fixture["searches"][0]["response"]
        assert response["results"], fixture_file
        for result in response["results"]:
            assert result["content"].strip() and "url" in result and "title" in result
            host = urlparse(result["url"] if "://" in result["url"] else "https://" + result["url"]).hostname or ""
            # reserved test domains only: nothing points at a real site
            assert re.search(r"\.(example|invalid)$", host), (fixture_file.parent.name, result["url"])


def test_run_case_resolves_inputs_from_fixture_ids(tmp_path):
    from argparse import Namespace
    from evals import run_case
    from evals.cases import find_ad, find_persona
    args = Namespace(ad="a3", persona="p07", inputs="cv_only", job_ad=None, search_fixture=None, record_search=None,
                     no_search=False, cv=None, portfolio=None, cover_letter=None, notes=None)
    run_case.apply_fixture_defaults(args)
    assert args.job_ad == str(find_ad("a3") / "ad.md")
    assert args.search_fixture == str(find_ad("a3") / "search_fixture.json")
    assert args.cv == str(find_persona("p07") / "cv.txt") and args.notes is None and args.portfolio is None

    explicit = Namespace(**{**vars(args), "cv": None, "inputs": "cv,notes", "no_search": True, "search_fixture": None,
                            "notes": str(tmp_path / "my_notes.txt")})
    run_case.apply_fixture_defaults(explicit)
    assert explicit.search_fixture is None  # --no-search wins over the ad's fixture
    assert explicit.notes == str(tmp_path / "my_notes.txt")  # an explicit file wins over the persona's
    with pytest.raises(ValueError):
        run_case.apply_fixture_defaults(Namespace(**{**vars(args), "ad": None, "job_ad": None}))
