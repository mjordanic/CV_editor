"""Replay of a recorded session through the real compiled graph, offline.

The models are replaced by a scripted fake, the web search returns nothing and LaTeX compilation
is stubbed, so the test runs the real nodes, edges, interrupts and resumes with no API calls.
Inputs are the repo's fictional samples (examples/sample_CV.txt, job_description/sample_job_ad.txt).

The recorded session (gpt-5-nano) went: job ad -> "CV first" -> drafts 1-3 scored 72, 82, 86 ->
"shorten the summary" -> draft 4 scored 82 -> the program kept draft 3 and the summary was unchanged.
(A draft is now redrafted for a defect, not a low score, so the first cycle here has two drafts.)
"""
import shutil
from pathlib import Path

from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

REPO = Path(__file__).resolve().parent.parent
SHORTEN = "Shorten the professional summary to two sentences."


class ScriptedLLM:
    """Stands in for every chat model; answers by the structured-output schema it is asked for."""

    def __init__(self, router_script, critique_scores):
        self.router_script = list(router_script)
        self.critique_scores = list(critique_scores)
        self.writer_instructions = []

    def with_structured_output(self, schema):
        return RunnableLambda(lambda prompt, schema=schema: self._answer(schema.__name__, schema, prompt))

    def invoke(self, prompt):  # plain calls: the critique filter keeps the instructions as they are
        text = prompt if isinstance(prompt, str) else str(prompt)
        instructions = text.split("**Critique Improvement Instructions:**", 1)[-1].split("**Task:**", 1)[0]
        return type("Msg", (), {"content": instructions.strip()})()

    def _answer(self, name, schema, prompt):
        text = prompt.to_string()
        if name == "JobDescriptionInfo":
            return schema(company_name="Northwind Analytics", job_title="Data Engineer", work_type="remote")
        if name == "GroundingResponse":
            return schema(unsupported_claims=[])
        if name == "DraftComparison":
            return schema(better="B")  # the later of the two drafts
        if name == "RouterResponse":
            return self.router_script.pop(0)
        if name == "CVGenerationResponse":
            n = len(self.writer_instructions) + 1
            instructions = text.split("**Modification Instructions:**", 1)[1].split("**Previous Modification", 1)[0]
            self.writer_instructions.append(instructions)
            base = text.split("**Existing CV (Base Document):**", 1)[1].split("\n**", 1)[0]
            short = SHORTEN in instructions or "SUMMARY: short" in base
            return schema(cv_content=f"CV draft {n}\nSUMMARY: {'short' if short else 'long'}", notes=f"notes {n}")
        if name == "CritiqueResponse":
            score, instructions = self.critique_scores.pop(0)
            return schema(
                quality_score=score,
                content_quality_feedback="ok",
                ats_compatibility_feedback="ok",
                job_alignment_feedback="ok",
                improvement_instructions=instructions,
                critical_issues="Tighten wording." if instructions else "",  # a defect: earns a redraft
            )
        raise AssertionError(f"unexpected structured call: {name}")


def _route(next_action, message="", **feedback):
    from agents.router import RouterResponse

    return RouterResponse(
        next_action=next_action,
        message_to_user=message,
        needs_user_input=next_action == "collect_user_input",
        **feedback,
    )


def _build_agent(monkeypatch, tmp_path, llm):
    """Real MasterAgent in an empty working folder, with every model call going to `llm`."""
    (tmp_path / "templates").symlink_to(REPO / "templates")
    (tmp_path / "data" / "CV").mkdir(parents=True)
    shutil.copy(REPO / "examples" / "sample_CV.txt", tmp_path / "data" / "CV" / "CV.txt")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-no-calls")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy-no-calls")

    import debug_utils
    import langgraph_agent as la
    from agents import (
        cover_letter_pdf_generator, cover_letter_writer, critique, cv_pdf_generator, cv_writer,
        document_writer, job_description, latex_fixer, latex_pdf_generator, rag_agent, router, search,
    )

    monkeypatch.setattr(debug_utils, "_debug_file_path", str(tmp_path / "debug.log"))
    # The writers and PDF generators create their models in their shared base modules.
    for module in (router, document_writer, critique, job_description, search, rag_agent,
                   latex_pdf_generator, latex_fixer):
        monkeypatch.setattr(module, "init_chat_model", lambda *_, **__: llm)
    monkeypatch.setattr(search.SearchAgent, "search_tavily", lambda self, query: {})

    rendered = {}

    def fake_compile(self, latex, output_name, max_fix_iterations=1):
        path = Path(self.output_folder) / f"{output_name}.pdf"
        path.write_text(latex, encoding="utf-8")
        rendered[path.name] = latex
        return str(path), 0, None

    for gen in (cv_pdf_generator.CVPDFGenerator, cover_letter_pdf_generator.CoverLetterPDFGenerator):
        monkeypatch.setattr(gen, "_build_latex_document", lambda self, template, text, style: text)
        monkeypatch.setattr(gen, "_compile_latex", fake_compile)

    return la.MasterAgent(), rendered


def _play(agent, user_replies):
    config = {"configurable": {"thread_id": "test"}}
    result = agent.graph.invoke({"messages": [], "cv_history": [], "cover_letter_history": []}, config)
    replies = list(user_replies)
    while result.get("__interrupt__"):
        result = agent.graph.invoke(Command(resume=replies.pop(0)), config)
    assert not replies, "session ended before all user replies were used"
    return result


def test_requested_cv_change_is_kept_after_earlier_draft_scored_higher(monkeypatch, tmp_path):
    llm = ScriptedLLM(
        router_script=[
            _route("collect_user_input", "What should I write first?"),
            _route("draft_cv", cv_content_feedback=""),
            _route("collect_user_input", "CV draft 2 is ready. Any changes?"),
            _route("draft_cv", cv_content_feedback=SHORTEN),
            _route("exit", "Done."),
        ],
        # drafts 1-2 as in the recorded session; draft 3 (the requested change) scores lower than draft 2
        critique_scores=[(72, "Fix the title."), (86, "Fix the title."), (80, "")],
    )
    agent, rendered = _build_agent(monkeypatch, tmp_path, llm)
    ad = (REPO / "job_description" / "sample_job_ad.txt").read_text(encoding="utf-8")

    final = _play(agent, [ad, "CV first", "Please shorten the summary."])

    assert SHORTEN in llm.writer_instructions[2], "the user's request never reached the CV writer"
    assert final["generated_cv"].endswith("SUMMARY: short"), "finalize threw away the requested change"
    assert final["cv_final_iteration"] == 3
    # the PDF is named after the draft it contains
    assert rendered["generated_CV_draft_2.pdf"].startswith("CV draft 2")
    assert rendered["generated_CV_draft_3.pdf"].startswith("CV draft 3")


def test_user_request_gets_a_fresh_redraft_budget(monkeypatch, tmp_path):
    """A request after the first cycle used up its redrafts still gets critique-driven polishing,
    and the polishing keeps building on the requested change."""
    llm = ScriptedLLM(
        router_script=[
            _route("draft_cv"),
            _route("collect_user_input", "Any changes?"),
            _route("draft_cv", cv_content_feedback=SHORTEN),
            _route("exit", "Done."),
        ],
        critique_scores=[(70, "Fix."), (80, "Fix."), (70, "Fix."), (80, "")],
    )
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)
    ad = (REPO / "job_description" / "sample_job_ad.txt").read_text(encoding="utf-8")

    final = _play(agent, [ad, "Please shorten the summary."])

    assert len(llm.writer_instructions) == 4  # 2 drafts (budget used), then the request plus one redraft
    assert final["cv_final_iteration"] == 4
    assert final["generated_cv"].endswith("SUMMARY: short")
