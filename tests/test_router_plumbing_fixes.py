"""Three router plumbing fixes, run through the real compiled graph with a scripted model (no network).

1. One message with changes for both existing documents: both documents are redrafted, the second
   right after the first is finalized (before, only the routed document's request was kept).
2. A new PDF style request is added to the earlier ones instead of replacing them.
3. The writer's notes on the user's request (e.g. a part it did not apply) reach the router's next
   message to the user, once.
"""
from agents.router import STYLE_REQUESTS_HEADER, _accumulate_style
from tests.test_session_replay import REPO, ScriptedLLM, _build_agent, _play, _route

CV_CHANGE = "Add Kotlin to the skills."
LETTER_CHANGE = "Mention that I am moving to Berlin in March."
WRITERS_HEADING = "**Writers' Notes on the User's Latest Request (not yet shown to the user):**"


class PlumbingLLM(ScriptedLLM):
    """ScriptedLLM that also writes cover letters, keeps every router prompt and can script CV notes."""

    def __init__(self, router_script, critique_scores, cv_notes=()):
        super().__init__(router_script, critique_scores)
        self.letter_instructions = []
        self.router_prompts = []
        self.cv_notes = list(cv_notes)

    def _answer(self, name, schema, prompt):
        text = prompt.to_string()
        if name == "RouterResponse":
            self.router_prompts.append(text)
        if name == "CoverLetterGenerationResponse":
            instructions = text.split("**Modification Instructions:**", 1)[1].split("**Previous Modification", 1)[0]
            self.letter_instructions.append(instructions)
            moving = LETTER_CHANGE in instructions
            return schema(cover_letter_content=f"Dear Northwind team,\nletter draft {len(self.letter_instructions)}"
                          + ("\nI am moving to Berlin in March." if moving else ""), notes="")
        answer = super()._answer(name, schema, prompt)
        if name == "CVGenerationResponse" and self.cv_notes:
            answer.notes = self.cv_notes.pop(0)
        return answer


def _ad():
    return (REPO / "job_description" / "sample_job_ad.txt").read_text(encoding="utf-8")


def _writers_notes(router_prompt: str) -> str:
    return router_prompt.split(WRITERS_HEADING, 1)[1].split("Based on this context", 1)[0].strip()


# --- 1: changes for both documents in one message ------------------------------------------------

def test_one_message_with_changes_for_both_existing_documents_redrafts_both(monkeypatch, tmp_path):
    llm = PlumbingLLM(
        router_script=[
            _route("draft_cv"),
            _route("draft_cover_letter"),
            _route("collect_user_input", "Both documents are ready. Any changes?"),
            _route("draft_cv", cv_content_feedback=CV_CHANGE, cover_letter_content_feedback=LETTER_CHANGE),
            # no router model call between the CV and the letter: the letter's request was kept
            _route("exit", "Both documents are updated."),
        ],
        critique_scores=[(80, "")] * 4,
    )
    agent, rendered = _build_agent(monkeypatch, tmp_path, llm)

    final = _play(agent, [_ad(), f"In the CV: {CV_CHANGE} In the cover letter: {LETTER_CHANGE}"])

    assert not llm.router_script, "the router model was not asked as scripted"
    assert CV_CHANGE in llm.writer_instructions[-1]
    assert len(llm.letter_instructions) == 2
    assert LETTER_CHANGE in llm.letter_instructions[-1], "the cover letter's request was dropped"
    assert final["generated_cover_letter"].endswith("I am moving to Berlin in March.")
    # each request is applied once: finalize cleared both
    assert not final.get("cv_content_feedback") and not final.get("cover_letter_content_feedback")
    assert any("I am moving to Berlin" in latex for latex in rendered.values()), "no new cover letter PDF"


def test_letter_first_then_the_cv_with_its_kept_request(monkeypatch, tmp_path):
    llm = PlumbingLLM(
        router_script=[
            _route("draft_cv"),
            _route("draft_cover_letter"),
            _route("collect_user_input", "Both documents are ready. Any changes?"),
            _route("draft_cover_letter", cv_content_feedback=CV_CHANGE, cover_letter_content_feedback=LETTER_CHANGE),
            _route("exit", "Both documents are updated."),
        ],
        critique_scores=[(80, "")] * 4,
    )
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)

    final = _play(agent, [_ad(), f"Letter: {LETTER_CHANGE} CV: {CV_CHANGE}"])

    assert not llm.router_script
    assert LETTER_CHANGE in llm.letter_instructions[-1]
    assert len(llm.writer_instructions) == 2 and CV_CHANGE in llm.writer_instructions[-1], "the CV's request was dropped"
    assert not final.get("cv_content_feedback") and not final.get("cover_letter_content_feedback")


# --- 2: style requests accumulate ----------------------------------------------------------------

def test_accumulate_style_keeps_earlier_requests_in_order_once():
    assert _accumulate_style(None, " Titles in dark blue ") == "Titles in dark blue"
    both = _accumulate_style("Titles in dark blue", "More compact spacing")
    assert both == f"{STYLE_REQUESTS_HEADER}\n- Titles in dark blue\n- More compact spacing"
    three = _accumulate_style(both, "Use a sans-serif font")
    assert three.splitlines()[1:] == ["- Titles in dark blue", "- More compact spacing", "- Use a sans-serif font"]
    assert _accumulate_style(three, "More compact spacing") == three  # not added twice
    assert _accumulate_style(three, "") == three


def test_second_style_request_keeps_the_first(monkeypatch, tmp_path):
    from agents import cv_pdf_generator

    llm = PlumbingLLM(
        router_script=[
            _route("draft_cv"),
            _route("collect_user_input", "The CV is ready. Any changes?"),
            _route("update_cv_pdf_style", cv_pdf_style="Section titles in dark blue"),
            _route("collect_user_input", "Updated. Anything else?"),
            _route("update_cv_pdf_style", cv_pdf_style="More compact spacing"),
            _route("exit", "Done."),
        ],
        critique_scores=[(80, "")],
    )
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)
    styles = []
    monkeypatch.setattr(cv_pdf_generator.CVPDFGenerator, "_build_latex_document",
                        lambda self, template, text, style: (styles.append(style), text)[1])

    final = _play(agent, [_ad(), "Make the section titles dark blue.", "Make it more compact."])

    assert styles[-1] == final["cv_pdf_style"]
    assert "Section titles in dark blue" in styles[-1] and "More compact spacing" in styles[-1], styles[-1]
    assert styles[-1].index("dark blue") < styles[-1].index("More compact")  # the later request wins conflicts


# --- 3: the writer's notes reach the user ---------------------------------------------------------

def test_writer_notes_on_a_request_reach_the_router_reply_once(monkeypatch, tmp_path):
    declined = "Added the Kafka migration. Did not rename the role to Head of Data: kept your stated title."
    llm = PlumbingLLM(
        router_script=[
            _route("draft_cv"),
            _route("collect_user_input", "The CV is ready. Any changes?"),
            _route("draft_cv", cv_content_feedback="Add the Kafka migration; retitle the role to Head of Data."),
            _route("collect_user_input", "Updated: the title was kept as it was. Anything else?"),
            _route("exit", "Done."),
        ],
        critique_scores=[(80, ""), (80, "")],
        cv_notes=["Tailored the summary to the ad.", declined],
    )
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)

    final = _play(agent, [_ad(), "Add the Kafka migration and call me Head of Data.", "exit"])

    # the first draft had no request from the user: its notes are not about a request
    assert _writers_notes(llm.router_prompts[1]) == "None."
    # the reply after the request sees the writer's notes, with the rule to state what was not applied
    assert _writers_notes(llm.router_prompts[3]) == f"CV writer: {declined}"
    assert "not applied, or was applied differently" in llm.router_prompts[3]
    # shown once: cleared after that reply
    assert _writers_notes(llm.router_prompts[4]) == "None."
    assert not final.get("cv_request_notes")


def test_notes_of_both_documents_reach_the_reply_after_the_second_one(monkeypatch, tmp_path):
    llm = PlumbingLLM(
        router_script=[
            _route("draft_cv"),
            _route("draft_cover_letter"),
            _route("collect_user_input", "Both documents are ready. Any changes?"),
            _route("draft_cv", cv_content_feedback=CV_CHANGE, cover_letter_content_feedback=LETTER_CHANGE),
            _route("exit", "Both documents are updated."),
        ],
        critique_scores=[(80, "")] * 4,
        cv_notes=["", "Kotlin is already listed under skills; left as it was."],
    )
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)

    _play(agent, [_ad(), f"CV: {CV_CHANGE} Letter: {LETTER_CHANGE}"])

    # the bypass to the letter does not show a message, so the CV's notes wait for the final reply
    assert "CV writer: Kotlin is already listed" in _writers_notes(llm.router_prompts[-1])
