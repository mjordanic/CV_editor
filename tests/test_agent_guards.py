"""Offline tests for guards that keep a cheap model's slips from derailing a session (no API calls).

All data is the repo's fictional sample material.
"""
import logging
from pathlib import Path

from langchain_core.runnables import RunnableLambda
from langgraph.graph.message import add_messages

import agents.cover_letter_writer as cover_letter_writer
import agents.critique as critique
import agents.cv_writer as cv_writer
import agents.router as router
import agents.search as search
from agents.router import RouterAgent, RouterResponse, _offered_actions

REPO = Path(__file__).resolve().parent.parent
SAMPLE_CV = (REPO / "examples" / "sample_CV.txt").read_text(encoding="utf-8")


def _router_with_answer(answer: RouterResponse) -> RouterAgent:
    agent = RouterAgent.__new__(RouterAgent)
    agent.max_history_messages = 10
    agent.llm = type("LLM", (), {"with_structured_output": lambda self, schema: RunnableLambda(lambda _: answer)})()
    return agent


def _state_after_first_cv(last_message: dict) -> dict:
    return {
        "messages": add_messages([], [{"role": "assistant", "content": "CV PDF generated."}, last_message]),
        "generated_cv": "CV draft 3",
        "cv_history": [{"iteration": i, "score": s, "content": f"CV draft {i}"} for i, s in ((1, 72), (2, 82), (3, 86))],
        "cv_refinement_count": 2,
        "cv_pdf_path": "data/generated_CVs/generated_CV.pdf",
    }


def test_router_stores_request_for_existing_cv_and_starts_new_cycle(monkeypatch):
    monkeypatch.setattr(router, "log_messages", lambda *_, **__: 0)
    agent = _router_with_answer(RouterResponse(
        next_action="draft_cv", message_to_user="", needs_user_input=False,
        cv_content_feedback="Shorten the summary.",
    ))
    out = agent.run(_state_after_first_cv({"role": "user", "content": "Please shorten the summary."}))

    assert out["next"] == "draft_cv"
    assert out["cv_content_feedback"] == "Shorten the summary."
    assert out["cv_refinement_count"] == 0  # fresh critique budget for the new request
    assert out["cv_cycle_start"] == 3  # finalize picks among drafts 4+ only


def test_router_falls_back_to_user_message_when_feedback_field_is_empty(monkeypatch):
    monkeypatch.setattr(router, "log_messages", lambda *_, **__: 0)
    agent = _router_with_answer(RouterResponse(
        next_action="draft_cv", message_to_user="", needs_user_input=False, cv_content_feedback="",
    ))
    out = agent.run(_state_after_first_cv({"role": "user", "content": "Please shorten the summary."}))

    assert out["cv_content_feedback"] == "Please shorten the summary."


def test_router_does_not_store_feedback_it_will_not_act_on(monkeypatch):
    monkeypatch.setattr(router, "log_messages", lambda *_, **__: 0)
    agent = _router_with_answer(RouterResponse(
        next_action="collect_user_input", message_to_user="Shorter how?", needs_user_input=True,
        cv_content_feedback="Shorten the summary.",
    ))
    out = agent.run(_state_after_first_cv({"role": "user", "content": "Please shorten the summary."}))

    assert "cv_content_feedback" not in out
    assert "cv_cycle_start" not in out


def test_finished_actions_are_not_offered_again_without_a_new_user_message():
    """After a re-render or redraft the latest message is the assistant's; the router must not repeat it."""
    agent = RouterAgent.__new__(RouterAgent)
    after_render = add_messages([], [
        {"role": "user", "content": "Section titles in dark blue."},
        {"role": "assistant", "content": "CV PDF generated successfully."},
    ])
    offered = _offered_actions(agent._build_available_actions(
        cv_generated=True, cover_letter_generated=True, messages=after_render,
        cv_pdf_available=True, cover_letter_pdf_available=True,
    ))
    assert offered == {"collect_user_input", "exit"}

    user_asked = add_messages(after_render, [{"role": "user", "content": "Make the CV titles bold."}])
    offered = _offered_actions(agent._build_available_actions(
        cv_generated=True, cover_letter_generated=True, messages=user_asked,
        cv_pdf_available=True, cover_letter_pdf_available=True,
    ))
    assert {"draft_cv", "draft_cover_letter", "update_cv_pdf_style", "update_cover_letter_pdf_style"} <= offered


def test_critique_sees_the_whole_original_cv():
    """The sample CV's SKILLS line starts after character 500; truncation made real skills look invented."""
    assert SAMPLE_CV.index("Kubernetes") > 500
    agent = critique.CritiqueAgent.__new__(critique.CritiqueAgent)
    info = agent._format_candidate_info({"cv": SAMPLE_CV, "cover_letter": None})
    assert "Kubernetes" in info
    assert SAMPLE_CV.strip() in info


def test_finalize_ignores_drafts_from_an_earlier_request():
    agent = cv_writer.CVWriterAgent.__new__(cv_writer.CVWriterAgent)
    agent.save_document = lambda text, filename: filename
    history = [{"iteration": i, "score": s, "content": f"CV draft {i}"} for i, s in ((1, 72), (2, 82), (3, 86), (4, 82))]

    out = agent.finalize_best_version({"cv_history": history, "cv_cycle_start": 3, "messages": []})
    assert out["generated_cv"] == "CV draft 4"
    assert out["cv_final_iteration"] == 4

    out = agent.finalize_best_version({"cv_history": history[:3], "messages": []})
    assert out["cv_final_iteration"] == 3


def test_critique_previous_defect_count_resets_at_new_cycle(monkeypatch):
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    agent = critique.CritiqueAgent.__new__(critique.CritiqueAgent)
    agent.grounding_llm = None  # the check fails and is skipped
    agent.critique_cv = lambda **_: critique.CritiqueResponse(
        quality_score=80, content_quality_feedback="ok", ats_compatibility_feedback="ok",
        job_alignment_feedback="ok", improvement_instructions="", critical_issues="",
    )
    history = [{"iteration": 3, "score": 86, "content": "CV draft 3", "defects": 2}]
    out = agent.run_cv({"generated_cv": "CV draft 4", "cv_history": history, "cv_cycle_start": 1, "messages": []})
    assert out["cv_previous_defect_count"] is None
    out = agent.run_cv({"generated_cv": "CV draft 4", "cv_history": history, "cv_cycle_start": 0, "messages": []})
    assert out["cv_previous_defect_count"] == 2


def _writer_with_filter_returning(cls, answer=""):
    agent = cls.__new__(cls)
    agent.filter_llm = type("F", (), {"invoke": lambda self, _: type("M", (), {"content": answer})()})()
    return agent


def test_fully_filtered_critique_still_uses_up_a_redraft(monkeypatch):
    """Otherwise critique -> redraft -> 'all filtered' -> critique ... repeats without end."""
    for module, cls, kind in (
        (cv_writer, cv_writer.CVWriterAgent, "cv"),
        (cover_letter_writer, cover_letter_writer.CoverLetterWriterAgent, "cover_letter"),
    ):
        agent = _writer_with_filter_returning(cls, '""')  # the two quote characters also mean "nothing"
        out = agent.run({
            f"generated_{kind}": "draft 2",
            f"{kind}_needs_refinement": True,
            f"{kind}_critique_improvement_instructions": "Add an education section.",
            f"{kind}_user_requests": ["Do not add an education section."],
            f"{kind}_refinement_count": 0,
            "messages": [],
        })
        assert out[f"{kind}_refinement_count"] == 1, kind
        assert out[f"generated_{kind}"] == "draft 2"


def test_cover_letter_writer_ignores_cv_feedback(monkeypatch, tmp_path):
    agent = cover_letter_writer.CoverLetterWriterAgent.__new__(cover_letter_writer.CoverLetterWriterAgent)
    agent.output_folder = str(tmp_path)
    seen = {}

    def fake_generate(document, modification_instructions, state, **_):
        seen["modification_instructions"] = modification_instructions
        return "Dear hiring team, ...", "notes"

    agent.generate_document = fake_generate
    out = agent.run({
        "candidate_text": {"cv": SAMPLE_CV, "cover_letter": None},
        "cv_content_feedback": "Add MATLAB to the skills section.",
        "cover_letter_content_feedback": None,
        "cover_letter_history": [],
        "messages": [],
    })
    assert "MATLAB" not in (seen["modification_instructions"] or "")
    assert "cover_letter_user_requests" not in out


def test_failed_company_search_does_not_end_the_session(monkeypatch, caplog):
    monkeypatch.setattr(search, "write_to_debug", lambda *_, **__: None)
    agent = search.SearchAgent.__new__(search.SearchAgent)

    def broken_search(query):
        raise ConnectionError("network unreachable")

    agent.search_tavily = broken_search
    with caplog.at_level(logging.WARNING, logger=search.__name__):
        out = agent.run({"job_description_info": {"company_name": "Northwind Analytics"}, "messages": []})

    assert out["company_info"]["work_context"]["arrangement"] == "not stated"
    assert "remote_work" not in out["company_info"]
    assert "Company research failed" in caplog.text
