"""Offline tests for defects seen in gpt-6-luna sessions for jobs outside IT (no API calls).

Four invented candidate packs were run through the whole flow (CV, cover letter, a content edit,
a style-only PDF request, exit): a butcher applying to a supermarket meat counter, a farm manager
applying to a cooperative, a surgical nurse applying to a private clinic, and a retail shift
supervisor moving to warehouse work. Every person, company and ad is fictional; the butcher pack
is in the repo as examples/sample_CV_butcher.txt and job_description/sample_job_ad_butcher.txt.

1. The final CV or letter was a draft the grounding check had flagged: the redraft that removed the
   unsupported claim scored 93, the flagged draft 94, and finalize picked on score alone.
2. When the redraft budget ran out, the last draft still held an unsupported claim.
3. The butcher's English ad for a Zagreb job got a Croatian cover letter.
4. The critic called requirements the candidate lacks (an ALS certificate, a forklift licence)
   "critical issues", so redrafts could not fix them and the letter ended up saying
   "The role requires an ALS certificate ..."; it also proposed "if true, add: willing to obtain ...".
5. The grounding check flagged "willing to obtain a forklift licence", which the user had asked for.
6. The LaTeX step reworded the letter after it had been checked ("I understand that the role ...").
7. "Cover letter PDF only: use a serif font" also set the CV's PDF style.
8. (Fifth pack, a pastry chef answering a German ad.) pdflatex printed "Süßer" as 8-bit bytes,
   strict UTF-8 decoding raised and no CV PDF was made; the next style request was then refused
   while the user was told the PDF "has been updated".
"""
from pathlib import Path

import pytest

import agents.critique as critique
from agents.cover_letter_pdf_generator import COVER_LETTER_LATEX_HUMAN_PROMPT, COVER_LETTER_LATEX_SYSTEM_PROMPT
from agents.cv_pdf_generator import CV_LATEX_HUMAN_PROMPT, CV_LATEX_SYSTEM_PROMPT
from agents.job_description import JobDescriptionInfo, output_language_instruction
from agents.refinement import best_draft, should_refine
from agents.router import RouterResponse
from tests.test_session_replay import ScriptedLLM, _build_agent, _play, _route

REPO = Path(__file__).resolve().parent.parent
BUTCHER_CV = (REPO / "examples" / "sample_CV_butcher.txt").read_text(encoding="utf-8")
BUTCHER_AD = (REPO / "job_description" / "sample_job_ad_butcher.txt").read_text(encoding="utf-8")
BUDGET = dict(max_refinements=1, max_refinements_hard_cap=2)


class TradesLLM(ScriptedLLM):
    """ScriptedLLM with a scripted grounding check and the ad's language; keeps every writer prompt."""

    def __init__(self, router_script, critique_scores, claims_per_draft=(), language="English"):
        super().__init__(router_script, critique_scores)
        self.claims_per_draft = list(claims_per_draft)
        self.language = language
        self.writer_prompts = []

    def _answer(self, name, schema, prompt):
        if name == "JobDescriptionInfo":
            return schema(company_name="FreshCart Supermarkets", job_title="Butcher", work_type="on-site",
                          language=self.language)
        if name == "GroundingResponse":
            return schema(unsupported_claims=self.claims_per_draft.pop(0) if self.claims_per_draft else [])
        if name in ("CVGenerationResponse", "CoverLetterGenerationResponse"):
            self.writer_prompts.append(prompt.to_string())
        if name == "CoverLetterGenerationResponse":
            return schema(cover_letter_content="Dear FreshCart team, ...", notes="")
        return super()._answer(name, schema, prompt)


def _butcher_agent(monkeypatch, tmp_path, llm):
    agent, rendered = _build_agent(monkeypatch, tmp_path, llm)
    (tmp_path / "data" / "CV" / "CV.txt").write_text(BUTCHER_CV, encoding="utf-8")
    return agent, rendered


# --- 1 and 2: unsupported claims decide which draft is kept, and earn a redraft -------------------

def test_finalize_keeps_the_draft_without_unsupported_claims():
    """Recorded butcher CV cycle: draft 3 (94) said 'advising on quantities', draft 4 (93) did not."""
    drafts = [
        {"iteration": 3, "score": 94, "unsupported_claims": ["quantities"], "content": "draft 3"},
        {"iteration": 4, "score": 93, "unsupported_claims": [], "content": "draft 4"},
    ]
    assert best_draft(drafts)["iteration"] == 4
    # without claims the old rule holds: higher score, then the later draft
    assert best_draft([{"iteration": 1, "score": 91}, {"iteration": 2, "score": 90}])["iteration"] == 1
    assert best_draft([{"iteration": 1, "score": 91}, {"iteration": 2, "score": 91}])["iteration"] == 2


@pytest.mark.parametrize(
    "count, has_claims, expected",
    [
        (1, False, False),  # budget used, no progress on the ad's requirements: stop
        (1, True, True),  # budget used but the draft still has unsupported claims: one more pass
        (2, True, False),  # never past the hard cap
    ],
)
def test_unsupported_claims_earn_a_redraft_within_the_hard_cap(count, has_claims, expected):
    assert should_refine(True, count, 1, 1, **BUDGET, has_unsupported_claims=has_claims)[0] is expected


def test_zero_max_refinements_still_turns_redrafting_off():
    off = dict(BUDGET, max_refinements=0)
    assert should_refine(True, 0, 3, None, **off, has_unsupported_claims=True)[0] is False


def test_session_does_not_end_on_a_flagged_draft(monkeypatch, tmp_path):
    """Whole graph: drafts 1 and 2 are flagged (2 scores highest), draft 3 is clean and is kept."""
    llm = TradesLLM(
        router_script=[_route("draft_cv"), _route("exit", "Done.")],
        critique_scores=[(91, ""), (94, ""), (93, "")],
        claims_per_draft=[["coordinating around 90 orders"], ["advising on quantities"], []],
    )
    agent, rendered = _butcher_agent(monkeypatch, tmp_path, llm)

    final = _play(agent, [BUTCHER_AD])

    assert len(llm.writer_instructions) == 3, "the flagged second draft did not get another redraft"
    assert "advising on quantities" in llm.writer_instructions[2]
    assert final["cv_final_iteration"] == 3
    assert "generated_CV_draft_3.pdf" in rendered


# --- 3: output language ---------------------------------------------------------------------------

def test_writers_are_told_to_write_in_the_language_of_the_ad(monkeypatch, tmp_path):
    llm = TradesLLM(
        router_script=[_route("draft_cv"), _route("draft_cover_letter"), _route("exit", "Done.")],
        critique_scores=[(92, ""), (92, "")],
    )
    agent, _ = _butcher_agent(monkeypatch, tmp_path, llm)

    _play(agent, [BUTCHER_AD])

    cv_prompt, letter_prompt = llm.writer_prompts
    assert "Write the CV in English, the language of the job ad" in cv_prompt
    assert "Write the cover letter in English, the language of the job ad" in letter_prompt
    assert "even if the employer or job is in another country" in letter_prompt


def test_language_instruction_without_a_detected_language():
    text = output_language_instruction({"company_name": "FreshCart Supermarkets"}, "cover letter")
    assert "the language the job ad is written in" in text
    assert "user's instructions" in text  # the user can still ask for another language
    assert "language" in JobDescriptionInfo.model_fields


# --- 4 and 5: gaps are not defects, plans are not claims ------------------------------------------

def test_critic_treats_missing_requirements_as_gaps_not_defects():
    prompt = critique.CRITIQUE_SYSTEM_PROMPT
    assert "GAPS ARE NOT DOCUMENT DEFECTS" in prompt
    assert "Do NOT list them as critical issues" in prompt
    assert "no 'if true, add ...'" in prompt


def test_grounding_check_does_not_count_plans_or_absences_as_claims():
    assert "plans or is willing to obtain it, are not claims to have it" in critique.GROUNDING_PROMPT


# --- 6: the PDF step must not reword checked text -------------------------------------------------

@pytest.mark.parametrize("system, human", [
    (CV_LATEX_SYSTEM_PROMPT, CV_LATEX_HUMAN_PROMPT),
    (COVER_LETTER_LATEX_SYSTEM_PROMPT, COVER_LETTER_LATEX_HUMAN_PROMPT),
])
def test_latex_step_copies_the_text_word_for_word(system, human):
    assert "word for word" in system and "word for word" in human
    assert "do NOT reword" in system
    assert "rephrase" not in system + human and "improve formatting, ordering, and wording" not in system


# --- 7: style for one document only ---------------------------------------------------------------

def test_router_style_fields_are_per_document():
    cv_field = RouterResponse.model_fields["cv_pdf_style"].description
    letter_field = RouterResponse.model_fields["cover_letter_pdf_style"].description
    assert "if the user asks only about the cover letter, leave this empty" in cv_field
    assert "if the user asks only about the CV, leave this empty" in letter_field


# --- 8: pdflatex output that is not UTF-8 ---------------------------------------------------------
# Pastry chef pack (German ad): "Konditorei Süßer Winkel" landed in an overfull-box warning, which
# pdflatex prints as 8-bit bytes; decoding them as strict UTF-8 raised and no CV PDF was made. The
# router then refused the next style request (no PDF) and told the user "The CV PDF has been updated."

BOX_WARNING = b"Overfull \\hbox (12.0pt too wide) in paragraph\n\\T1/cmr/m/n/10 Konditorei S\xfc\xffer Winkel M\xe4rz\n"
LATEX_WITH_UMLAUTS = (
    "\\documentclass{article}\\usepackage[utf8]{inputenc}\\usepackage[T1]{fontenc}"
    "\\begin{document}\\hbox to 2cm{Konditorei Süßer Winkel März}\\end{document}\n"
)


def _generator(gen_class, tmp_path):
    gen = gen_class.__new__(gen_class)
    gen.output_folder = str(tmp_path / "out")
    gen.temp_folder = str(tmp_path / "out" / "temp_latex")
    gen._latex_fixer = None  # must not be needed: the document compiles
    Path(gen.temp_folder).mkdir(parents=True)
    return gen


@pytest.mark.parametrize("module_name, class_name", [
    ("agents.cv_pdf_generator", "CVPDFGenerator"),
    ("agents.cover_letter_pdf_generator", "CoverLetterPDFGenerator"),
])
def test_non_utf8_compiler_output_does_not_fail_the_pdf(monkeypatch, tmp_path, module_name, class_name):
    import importlib
    import subprocess

    module = importlib.import_module(module_name)

    def fake_pdflatex(cmd, **kwargs):
        # what subprocess.run does with text=True: decode the captured bytes with the given settings
        out = BOX_WARNING.decode(kwargs.get("encoding") or "utf-8", kwargs.get("errors") or "strict")
        (Path(kwargs["cwd"]) / cmd[-1].replace(".tex", ".pdf")).write_bytes(b"%PDF-1.5")
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_pdflatex)  # the shared base module runs pdflatex
    gen = _generator(getattr(module, class_name), tmp_path)
    pdf_path, fixes, error = gen._compile_latex(LATEX_WITH_UMLAUTS, "generated_draft_2")
    assert Path(pdf_path).exists() and fixes == 0 and error is None


@pytest.mark.skipif(__import__("shutil").which("pdflatex") is None, reason="needs pdflatex")
def test_real_pdflatex_with_umlauts_in_a_box_warning(tmp_path):
    from agents.cv_pdf_generator import CVPDFGenerator

    pdf_path, fixes, _ = _generator(CVPDFGenerator, tmp_path)._compile_latex(LATEX_WITH_UMLAUTS, "generated_CV_draft_2")
    assert Path(pdf_path).exists() and fixes == 0


def test_style_request_can_rebuild_a_pdf_that_failed(monkeypatch):
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph.message import add_messages

    import agents.router as router
    from agents.router import RouterAgent, _offered_actions

    monkeypatch.setattr(router, "log_messages", lambda *_, **__: 0)
    agent = RouterAgent.__new__(RouterAgent)
    agent.max_history_messages = 10
    answer = RouterResponse(next_action="update_cv_pdf_style", message_to_user="", needs_user_input=False,
                            cv_pdf_style="Narrow sans-serif font, small-caps section titles.")
    prompts = []
    agent.llm = type("LLM", (), {"with_structured_output": lambda self, schema: RunnableLambda(
        lambda prompt: (prompts.append(prompt.to_string()), answer)[1])})()
    state = {
        "messages": add_messages([], [
            {"role": "assistant", "content": "Error generating CV PDF: 'utf-8' codec can't decode byte 0xe4"},
            {"role": "user", "content": "CV PDF only: narrow sans-serif font, small-caps section titles."},
        ]),
        "generated_cv": "Zoë D'Souza-Novak, pastry chef", "cv_pdf_path": None,
    }
    out = agent.run(state)
    assert out["next"] == "update_cv_pdf_style"
    assert "update_cv_pdf_style" in _offered_actions(prompts[0].split("Available actions", 1)[1])

    # an action that is not possible (no cover letter yet) is not reported as done
    answer.next_action = "update_cover_letter_pdf_style"
    out = agent.run(state)
    assert out["next"] == "collect_user_input"
    assert "has been updated" not in out["messages"][-1]["content"]
