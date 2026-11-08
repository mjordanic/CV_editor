"""Offline tests for the defects seen in a gpt-5-nano re-run (no API calls, fictional data only).

1. Tools the candidate never mentioned (Azure, GCP, Airflow, Prefect) reached the final CV and the
   cover letter: the critic proposed them, and a critical "unverified claim" issue came with no
   instructions, so nothing removed them.
2. The critic answered improvement_instructions='""' (the two quote characters); that counted as an
   instruction, and the unchanged draft was critiqued again and stored as a new iteration.
3. The router's fallback message talked about "automatic refinement passes" after a PDF re-render.

The CV, portfolio and job ad are the repo's fictional samples (Alex Example, Northwind Analytics).
"""
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.graph.message import add_messages

import agents.cover_letter_writer as cover_letter_writer
import agents.critique as critique
import agents.cv_writer as cv_writer
import agents.router as router
from agents.router import RouterAgent, RouterResponse
from agents.skill_check import terms_in, unsupported_terms

REPO = Path(__file__).resolve().parent.parent
SAMPLE_CV = (REPO / "examples" / "sample_CV.txt").read_text(encoding="utf-8")
PORTFOLIO = (REPO / "data" / "portfolio" / "portfolio.txt").read_text(encoding="utf-8")
JOB_AD = (REPO / "job_description" / "sample_job_ad.txt").read_text(encoding="utf-8")

# The SKILLS lines gpt-5-nano wrote into draft 5 of the recorded run.
INVENTED_SKILLS = (
    "Cloud platforms: AWS, Azure, GCP (specify which you have hands-on experience with)\n"
    "Workflow orchestration: Airflow or Prefect (familiar)"
)
CV_WITH_INVENTED_SKILLS = SAMPLE_CV.replace("Redis, scikit-learn, Pandas, Git", "Redis, scikit-learn, Pandas, Git\n" + INVENTED_SKILLS)
LETTER_WITH_INVENTED_SKILLS = (
    "Dear Hiring Manager,\n\nI am proficient in Python, SQL, Docker and cloud platforms (AWS, Azure, and GCP), "
    "and I am familiar with workflow orchestration systems such as Airflow or Prefect.\n\nSincerely,\nAlex Example"
)
LETTER_CLEAN = "Dear Hiring Manager,\n\nI build Python services in Docker and dashboards in Tableau.\n\nSincerely,\nAlex Example"


def _state(**extra):
    return {
        "candidate_text": {"cv": SAMPLE_CV, "cover_letter": None},
        "portfolio_text": PORTFOLIO,
        "job_description_info": {"job_description": JOB_AD},
        "messages": [],
        **extra,
    }


def _critique_agent(monkeypatch, *, score=78, instructions="", critical_issues=""):
    """CritiqueAgent whose LLM answers with a fixed critique; counts the LLM calls."""
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    agent = critique.CritiqueAgent.__new__(critique.CritiqueAgent)
    agent.quality_threshold = 85
    agent.calls = 0

    def answer(**_):
        agent.calls += 1
        return critique.CritiqueResponse(
            quality_score=score, content_quality_feedback="ok", ats_compatibility_feedback="ok",
            job_alignment_feedback="ok", improvement_instructions=instructions, critical_issues=critical_issues,
        )

    agent.critique_cv = answer
    agent.critique_cover_letter = answer
    return agent


# --- 1. invented skills -------------------------------------------------------------------------

def test_skill_check_finds_only_tools_missing_from_cv_and_portfolio():
    # AWS is in the portfolio (Healthcare Patient Portal: "AWS (S3, EC2)"), so it is not invented.
    assert unsupported_terms(CV_WITH_INVENTED_SKILLS, _state()) == ["Airflow", "Azure", "GCP", "Prefect"]
    assert unsupported_terms(SAMPLE_CV, _state()) == []
    # Ordinary words that contain or equal a tool name are not tools.
    assert terms_in("I react quickly, sparked a redesign and excel at scalable work.") == set()


def test_skill_the_user_asked_for_is_not_flagged():
    state = _state(messages=add_messages([], [{"role": "user", "content": "Add Airflow to my skills, I used it in 2023."}]))
    assert "Airflow" not in unsupported_terms(CV_WITH_INVENTED_SKILLS, state)


def test_critical_issue_without_instructions_still_removes_invented_tools(monkeypatch):
    """The recorded draft-5 critique: a critical 'unverified claim' issue but no instructions."""
    agent = _critique_agent(monkeypatch, score=78, instructions="",
                            critical_issues="Cloud platform claims are not verified by the original CV.")
    out = agent.run_cv(_state(generated_cv=CV_WITH_INVENTED_SKILLS, cv_history=[]))

    assert out["cv_needs_refinement"] is True
    instructions = out["cv_critique_improvement_instructions"]
    assert "Remove every mention of Airflow, Azure, GCP, Prefect" in instructions
    assert "Cloud platform claims are not verified" in instructions  # the critic's own issue is passed on
    assert out["cv_history"][-1]["unsupported_terms"] == ["Airflow", "Azure", "GCP", "Prefect"]


def test_invented_tools_trigger_a_redraft_even_with_a_good_score(monkeypatch):
    agent = _critique_agent(monkeypatch, score=92, instructions="", critical_issues="")
    out = agent.run_cv(_state(generated_cv=CV_WITH_INVENTED_SKILLS, cv_history=[]))
    assert out["cv_needs_refinement"] is True
    assert "Azure" in out["cv_critique"]["critical_issues"]


def test_critic_suggestion_to_add_job_tools_is_overridden(monkeypatch):
    """The recorded draft-6 critique asked for an 'Airflow DAGs' bullet the candidate never had."""
    agent = _critique_agent(monkeypatch, score=72,
                            instructions="In the Experience section under TechCorp, add a bullet about building Airflow DAGs.")
    out = agent.run_cv(_state(generated_cv=SAMPLE_CV, cv_history=[]))

    instructions = out["cv_critique_improvement_instructions"]
    assert "Do NOT add or keep these tools or skills" in instructions
    forbidden = instructions.split("Do NOT add or keep these tools or skills", 1)[1]
    for tool in ("Airflow", "Azure", "GCP", "Prefect"):
        assert tool in forbidden
    assert "AWS" not in forbidden.split("\n")[0]  # in the portfolio, so allowed


def test_critic_prompt_forbids_proposing_absent_content():
    assert "NEVER suggest adding" in critique.CRITIQUE_SYSTEM_PROMPT


def test_cover_letter_critique_flags_tools_copied_from_the_cv(monkeypatch):
    """The recorded letter scored 86 and its critic said nothing about Azure/GCP/Airflow/Prefect."""
    agent = _critique_agent(monkeypatch, score=86)
    out = agent.run_cover_letter(_state(
        generated_cover_letter=LETTER_WITH_INVENTED_SKILLS, generated_cv=CV_WITH_INVENTED_SKILLS,
        cover_letter_history=[],
    ))
    assert out["cover_letter_needs_refinement"] is True
    assert "Remove every mention of Airflow, Azure, GCP, Prefect" in out["cover_letter_critique_improvement_instructions"]


@pytest.mark.parametrize("module, cls, kind", [
    (cv_writer, cv_writer.CVWriterAgent, "cv"),
    (cover_letter_writer, cover_letter_writer.CoverLetterWriterAgent, "cover_letter"),
])
def test_finalize_prefers_a_draft_without_invented_tools_and_says_so(module, cls, kind):
    agent = cls.__new__(cls)
    agent.save_cv = agent.save_cover_letter = lambda text, filename: filename
    history = [
        {"iteration": 1, "score": 76, "content": "draft 1", "unsupported_terms": []},
        {"iteration": 2, "score": 80, "content": "draft 2", "unsupported_terms": ["Azure", "GCP"]},
    ]
    out = agent.finalize_best_version({f"{kind}_history": history, "messages": []})
    assert out[f"{kind}_final_iteration"] == 1

    # When every draft has them, keep the best one but tell the user which ones to check.
    history[0]["unsupported_terms"] = ["Azure"]
    out = agent.finalize_best_version({f"{kind}_history": history, "messages": []})
    assert out[f"{kind}_final_iteration"] == 1  # fewer unsupported tools beats a higher score
    assert "Azure" in out["messages"][-1]["content"]
    assert "could not find" in out["messages"][-1]["content"]


# --- 2. quote-only instructions and unchanged drafts ----------------------------------------------

@pytest.mark.parametrize("answer", ['""', "''", ' "" ', "``", "None", "N/A", "No improvements needed."])
def test_empty_looking_instructions_are_no_instructions(monkeypatch, answer):
    agent = _critique_agent(monkeypatch, score=78, instructions=answer, critical_issues="")
    out = agent.run_cv(_state(generated_cv=SAMPLE_CV, cv_history=[]))
    assert out["cv_needs_refinement"] is False
    assert out["cv_critique"]["improvement_instructions"] == ""


@pytest.mark.parametrize("kind", ["cv", "cover_letter"])
def test_unchanged_draft_is_not_stored_as_a_new_iteration(monkeypatch, kind):
    """Writer skipped the pass (all critique filtered out) and returned the same text."""
    agent = _critique_agent(monkeypatch, score=78, instructions="Tighten the summary.")
    doc = SAMPLE_CV if kind == "cv" else LETTER_CLEAN
    history = [{"iteration": 5, "score": 78, "content": doc, "unsupported_terms": []}]
    run = agent.run_cv if kind == "cv" else agent.run_cover_letter
    out = run(_state(**{f"generated_{kind}": doc, f"{kind}_history": history, f"{kind}_cycle_start": 0}))

    assert agent.calls == 0, "the unchanged draft was sent to the critic again"
    assert out.get(f"{kind}_history", history) == history
    assert out[f"{kind}_needs_refinement"] is False


def test_same_text_in_a_new_request_cycle_is_a_new_draft(monkeypatch):
    """A new user request starts a cycle; its first draft is kept even if the text did not change."""
    agent = _critique_agent(monkeypatch, score=80)
    history = [{"iteration": 1, "score": 80, "content": SAMPLE_CV, "unsupported_terms": []}]
    out = agent.run_cv(_state(generated_cv=SAMPLE_CV, cv_history=history, cv_cycle_start=1))
    assert agent.calls == 1
    assert [h["iteration"] for h in out["cv_history"]] == [1, 2]


# --- 3. router fallback message --------------------------------------------------------------------

def _router_with_answer(answer: RouterResponse) -> RouterAgent:
    agent = RouterAgent.__new__(RouterAgent)
    agent.max_history_messages = 10
    agent.llm = type("LLM", (), {"with_structured_output": lambda self, schema: RunnableLambda(lambda _: answer)})()
    return agent


@pytest.mark.parametrize("repeated, expected", [
    ("update_cv_pdf_style", "The CV PDF has been updated."),
    ("update_cover_letter_pdf_style", "The cover letter PDF has been updated."),
    ("draft_cv", "The CV draft is ready."),
    ("draft_cover_letter", "The cover letter draft is ready."),
])
def test_router_fallback_message_matches_what_just_ran(monkeypatch, repeated, expected):
    monkeypatch.setattr(router, "log_messages", lambda *_, **__: 0)
    agent = _router_with_answer(RouterResponse(next_action=repeated, message_to_user="", needs_user_input=False))
    state = {
        "messages": add_messages([], [
            {"role": "user", "content": "Make the section titles dark blue."},
            {"role": "assistant", "content": "PDF generated."},
        ]),
        "generated_cv": "CV", "generated_cover_letter": "Letter",
        "cv_pdf_path": "cv.pdf", "cover_letter_pdf_path": "cl.pdf",
    }
    out = agent.run(state)
    assert out["next"] == "collect_user_input"
    message = out["messages"][-1]["content"]
    assert message.startswith(expected)
    assert "refinement passes" not in message


def test_critic_sees_the_portfolio(monkeypatch):
    """The paid re-run's critic called the portfolio's RabbitMQ project a hallucination: it never saw it."""
    agent = _critique_agent(monkeypatch)
    seen = {}
    agent.critique_cv = lambda **kw: (seen.update(kw), critique.CritiqueResponse(
        quality_score=90, content_quality_feedback="ok", ats_compatibility_feedback="ok", job_alignment_feedback="ok"))[1]
    agent.run_cv(_state(generated_cv=SAMPLE_CV, cv_history=[]))
    assert "RabbitMQ" in agent._format_candidate_info(seen["candidate_text"])
