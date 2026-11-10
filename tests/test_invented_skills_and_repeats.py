"""Offline tests for the defects seen in a gpt-5-nano re-run (no API calls, fictional data only).

1. Skills the candidate never mentioned (Azure, GCP, Airflow, Prefect) reached the final CV and the
   cover letter: the critic proposed them, a critical "unverified claim" issue came with no
   instructions, and nothing checked the cover letter. The fix is an LLM grounding check (no fixed
   skill list, so it works for any profession); here the model is a scripted fake.
2. The critic answered improvement_instructions='""' (the two quote characters); that counted as an
   instruction, and the unchanged draft was critiqued again and stored as a new iteration.
3. The router's fallback message talked about "automatic refinement passes" after a PDF re-render.

The CV, portfolio and job ad are the repo's fictional samples (Alex Example, Northwind Analytics).
"""
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.graph.message import add_messages

import agents.critique as critique
import agents.cv_writer as cv_writer
import agents.document_writer as document_writer
import agents.router as router
from agents.router import RouterAgent, RouterResponse

REPO = Path(__file__).resolve().parent.parent
SAMPLE_CV = (REPO / "examples" / "sample_CV.txt").read_text(encoding="utf-8")
# The format example under examples/ (the app never reads it; the repo ships no live portfolio).
PORTFOLIO = (REPO / "examples" / "sample_portfolio.txt").read_text(encoding="utf-8")

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
CLAIMS = ["Cloud platforms: AWS, Azure, GCP", "Workflow orchestration: Airflow or Prefect (familiar)"]


def _state(**extra):
    return {
        "candidate_text": {"cv": SAMPLE_CV, "cover_letter": None},
        "portfolio_text": PORTFOLIO,
        "messages": [],
        **extra,
    }


class GroundingLLM:
    """Fake model for the grounding check: returns fixed claims and keeps the prompts it was given."""

    def __init__(self, claims=()):
        self.claims = list(claims)
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema.__name__ == "GroundingResponse"
        return RunnableLambda(lambda prompt: (self.prompts.append(prompt.to_string()), schema(unsupported_claims=self.claims))[1])


def _critique_agent(monkeypatch, *, score=78, instructions="", critical_issues="", claims=()):
    """CritiqueAgent whose critic answers with a fixed critique; counts the critic calls."""
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    agent = critique.CritiqueAgent.__new__(critique.CritiqueAgent)
    agent.calls = 0
    agent.llm = agent.grounding_llm = GroundingLLM(claims)

    def answer(**_):
        agent.calls += 1
        return critique.CritiqueResponse(
            quality_score=score, content_quality_feedback="ok", ats_compatibility_feedback="ok",
            job_alignment_feedback="ok", improvement_instructions=instructions, critical_issues=critical_issues,
        )

    agent.critique_cv = answer
    agent.critique_cover_letter = answer
    return agent


# --- 1. unsupported claims -----------------------------------------------------------------------

def test_critical_issue_without_instructions_becomes_an_instruction(monkeypatch):
    """The recorded draft-5 critique: a critical 'unverified claim' issue but no instructions."""
    agent = _critique_agent(monkeypatch, score=78, instructions="",
                            critical_issues="Cloud platform claims are not verified by the original CV.")
    out = agent.run_cv(_state(generated_cv=CV_WITH_INVENTED_SKILLS, cv_history=[]))

    assert out["cv_needs_refinement"] is True
    assert "Cloud platform claims are not verified" in out["cv_critique_improvement_instructions"]


def test_grounding_check_claims_become_removal_instructions(monkeypatch):
    agent = _critique_agent(monkeypatch, score=92, claims=CLAIMS)  # good score: would not refine alone
    out = agent.run_cv(_state(generated_cv=CV_WITH_INVENTED_SKILLS, cv_history=[]))

    assert out["cv_needs_refinement"] is True
    # the claims to remove travel separately from the critic's suggestions, so the writers' filter of
    # suggestions against the user's earlier requests can never drop them
    assert out["cv_claims_to_remove"] == CLAIMS
    assert out["cv_critique_improvement_instructions"] in (None, "")
    removal = document_writer.removal_instructions(out["cv_claims_to_remove"])
    assert "Remove these claims" in removal
    for claim in CLAIMS:
        assert claim in removal
        assert claim in out["cv_critique"]["critical_issues"]
    assert out["cv_history"][-1]["unsupported_claims"] == CLAIMS


def test_grounding_check_sees_only_the_candidates_material(monkeypatch):
    """Evidence about the candidate is the original CV, portfolio and the user's own messages; never the
    job ad (which is evidence only for statements about the employer)."""
    agent = _critique_agent(monkeypatch)
    messages = add_messages([], [
        {"role": "assistant", "content": "What should I write?"},
        {"role": "user", "content": "Add my forklift licence (2019) to the CV."},
    ])
    agent.run_cv(_state(generated_cv=SAMPLE_CV, cv_history=[], messages=messages,
                        job_description_info={"job_description": "Must know Airflow."}))
    prompt = agent.llm.prompts[0]
    candidate_part, employer_part = prompt.split("**Job ad and company research**", 1)
    assert "Kubernetes" in candidate_part  # original CV
    assert "RabbitMQ" in candidate_part  # portfolio (the paid re-run's critic called it invented without it)
    assert "forklift licence" in candidate_part  # the user's own request counts as evidence
    assert "What should I write?" not in prompt
    assert "Must know Airflow" not in candidate_part
    assert "Must know Airflow" in employer_part.split("to check:**", 1)[0]


def test_clean_draft_passes_and_a_failed_check_does_not_stop_the_session(monkeypatch):
    agent = _critique_agent(monkeypatch, score=90)
    out = agent.run_cv(_state(generated_cv=SAMPLE_CV, cv_history=[]))
    assert out["cv_needs_refinement"] is False

    agent.llm = None  # check raises: logged, session continues
    out = agent.run_cv(_state(generated_cv=SAMPLE_CV, cv_history=[]))
    assert out["cv_needs_refinement"] is False


def test_redraft_instructions_forbid_adding_what_the_candidate_lacks(monkeypatch, tmp_path):
    """The recorded draft-6 critique asked for an 'Airflow DAGs' bullet the candidate never had."""
    agent = _critique_agent(monkeypatch, score=72, critical_issues="Experience is thin.",
                            instructions="In the Experience section under TechCorp, add a bullet about building Airflow DAGs.")
    out = agent.run_cv(_state(generated_cv=SAMPLE_CV, cv_history=[]))
    # the writer appends the rule to every critique-driven redraft (after the preference filter)
    writer = cv_writer.CVWriterAgent.__new__(cv_writer.CVWriterAgent)
    writer.output_folder = str(tmp_path)
    seen = []

    def fake_generate(document, instructions, state, **_):
        seen.append(instructions)
        return "CV", ""

    writer.generate_document = fake_generate
    writer.run({**_state(), **out, "generated_cv": SAMPLE_CV})
    assert "Airflow DAGs" in seen[0]
    assert seen[0].rstrip().endswith(critique.NO_NEW_CLAIMS_RULE)


def test_critic_prompt_forbids_proposing_absent_content():
    assert "NEVER suggest adding" in critique.CRITIQUE_SYSTEM_PROMPT


def test_cover_letter_claims_are_checked_too(monkeypatch):
    """The recorded letter scored 86 and its critic said nothing about Azure/GCP/Airflow/Prefect."""
    agent = _critique_agent(monkeypatch, score=86, claims=["cloud platforms (AWS, Azure, and GCP)"])
    out = agent.run_cover_letter(_state(
        generated_cover_letter=LETTER_WITH_INVENTED_SKILLS, generated_cv=CV_WITH_INVENTED_SKILLS,
        cover_letter_history=[],
    ))
    assert out["cover_letter_needs_refinement"] is True
    assert "cloud platforms (AWS, Azure, and GCP)" in out["cover_letter_claims_to_remove"]
    assert "cover letter to check" in agent.llm.prompts[0]


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



# --- running against another model (e.g. a local Ollama model) -------------------------------------

def test_model_override_replaces_every_writer_side_model(monkeypatch):
    from config.config_loader import get_agent_config

    monkeypatch.setenv("CV_EDITOR_MODEL", "ollama:gemma4:e4b")
    writer = get_agent_config("cv_writer")
    assert writer["model"] == writer["filter_model"] == "ollama:gemma4:e4b"
    assert get_agent_config("template_customizer")["model"] == "ollama:gemma4:e4b"
    assert get_agent_config("document_reader")["model"] is None  # agents without a model stay without one

    monkeypatch.delenv("CV_EDITOR_MODEL")
    assert get_agent_config("cv_writer")["model"].startswith("openai:")


def test_ollama_model_gets_local_embeddings():
    from agents.rag_agent import _embeddings_for

    assert type(_embeddings_for("ollama:gemma4:e4b")).__name__ == "OllamaEmbeddings"
    assert _embeddings_for("ollama:gemma4:e4b").model == "nomic-embed-text"  # gemma4 cannot embed in Ollama


def test_session_runs_without_a_tavily_key(monkeypatch):
    """A local run has no web-search key: company research is skipped instead of failing at startup."""
    import agents.search as search

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(search, "init_chat_model", lambda *_, **__: None)
    monkeypatch.setattr(search, "write_to_debug", lambda *_, **__: None)
    agent = search.SearchAgent(model="ollama:gemma4:e4b")
    out = agent.run({"job_description_info": {"company_name": "Northwind Analytics"}, "messages": []})
    assert out["company_info"]["work_context"]["arrangement"] == "not stated"
