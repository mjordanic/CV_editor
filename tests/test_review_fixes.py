"""Offline regression tests for the first-principles review of the CV writer (no API calls).

One test (or a few) per finding: D1, D3-D11 (defects), I1-I9 (design) and C1-C6 (cleanups). D2 was
deliberately not changed: a user's own request steers their CV and letter, also in their first
message, and that behaviour is pinned here too. Models are scripted fakes; the PDF tests use the
real pdflatex when it is installed.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.graph.message import add_messages

from agents import critique, document_writer, job_description, router, search
from agents.cover_letter_writer import CoverLetterWriterAgent
from agents.cv_writer import CVWriterAgent
from agents.notices import NOTICES_KEY, pending_notices
from agents.pdf_text_check import check_pdf, compare_texts, find_placeholders
from tests.test_session_replay import REPO, ScriptedLLM, _build_agent, _play, _route

SAMPLE_CV = (REPO / "examples" / "sample_CV.txt").read_text(encoding="utf-8")
OLD_LETTER = "Dear Brightmill team,\n\nI am currently a QA engineer at Brightmill and want to grow here.\n\nMarta"
AD_INFO = {"job_title": "Butcher", "company_name": "FreshCart Supermarkets", "location": "Zagreb, Croatia",
           "job_description": "FreshCart is hiring a butcher in Zagreb. HACCP required."}


class WriterLLM:
    """Stands in for the writer and filter models: keeps every prompt, returns a fixed document."""

    def __init__(self, text="DOC", filter_answer=None):
        self.prompts, self.filter_prompts = [], []
        self.text, self.filter_answer = text, filter_answer

    def with_structured_output(self, schema):
        field = "cv_content" if "cv_content" in schema.model_fields else "cover_letter_content"

        def answer(prompt):
            self.prompts.append(prompt.to_string())
            return schema(**{field: self.text, "notes": ""})
        return RunnableLambda(answer)

    def invoke(self, prompt):  # the preference filter
        self.filter_prompts.append(prompt)
        return type("M", (), {"content": self.filter_answer if self.filter_answer is not None else "KEPT"})()


def _writer(monkeypatch, tmp_path, cls, llm, **kwargs):
    monkeypatch.setattr(document_writer, "init_chat_model", lambda *_, **__: llm)
    return cls(output_folder=str(tmp_path), **kwargs)


def _instructions(prompt: str) -> str:
    return prompt.split("**Modification Instructions:**", 1)[1].split("**Previous Modification", 1)[0]


def _section(prompt: str, heading: str) -> str:
    return prompt.split(heading, 1)[1].split("\n**", 1)[0]


# --- D1: the first cover letter for a new job is a new letter, the old one only source material ----

def test_first_cover_letter_without_a_request_is_written_fresh_for_this_job(monkeypatch, tmp_path):
    llm = WriterLLM("Dear FreshCart team, ...")
    agent = _writer(monkeypatch, tmp_path, CoverLetterWriterAgent, llm)
    agent.run({"candidate_text": {"cv": SAMPLE_CV, "cover_letter": OLD_LETTER},
               "job_description_info": AD_INFO, "messages": []})
    prompt = llm.prompts[0]
    instructions = _instructions(prompt)
    assert "CREATIVE LIBERTY GRANTED (A NEW LETTER FOR THIS JOB)" in instructions
    assert "written for a different job and employer" in instructions
    assert "Please review the existing cover letter" not in prompt
    assert "Apply ONLY these specific changes" not in instructions
    base = _section(prompt, "**Existing Cover Letter (Base Document):**")
    assert "Brightmill" in base and "Use it only as a source of facts" in base


def test_first_cover_letter_with_a_request_follows_the_request(monkeypatch, tmp_path):
    """D2 is out of scope: the user's own wish steers the letter as they asked, undiluted."""
    llm = WriterLLM()
    agent = _writer(monkeypatch, tmp_path, CoverLetterWriterAgent, llm)
    wish = "Keep my letter as it is, just add that I can start in March."
    out = agent.run({"candidate_text": {"cv": SAMPLE_CV, "cover_letter": OLD_LETTER},
                     "cover_letter_content_feedback": wish, "job_description_info": AD_INFO, "messages": []})
    instructions = _instructions(llm.prompts[0])
    assert wish in instructions and "Apply ONLY these specific changes to the existing cover letter" in instructions
    assert "CREATIVE LIBERTY" not in instructions
    assert "only as a source of facts" not in llm.prompts[0]
    assert out["cover_letter_user_requests"] == [wish]  # C4: kept in state, not on the agent object


class LetterLLM(WriterLLM):
    """A writer that follows the instructions it gets: it readdresses the old letter only when told to."""

    def with_structured_output(self, schema):
        def answer(prompt):
            text = prompt.to_string()
            self.prompts.append(text)
            letter = OLD_LETTER
            if "update the employer, the role and the addressee" in _instructions(text):
                letter = letter.replace("Brightmill team", "FreshCart Supermarkets team").replace(
                    "I am currently a QA engineer at Brightmill and want to grow here.", "I would like to join you as a butcher.")
            if "start in March" in _instructions(text):
                letter = letter.replace("\n\nMarta", "\n\nI can start in March.\n\nMarta")
            return schema(cover_letter_content=letter, notes="")
        return RunnableLambda(answer)


def test_a_wish_on_the_first_letter_readdresses_the_old_letter_to_this_job(monkeypatch, tmp_path):
    """N3 (captain: option a): 'mention I can start in March' with an old letter for another company."""
    llm = LetterLLM()
    agent = _writer(monkeypatch, tmp_path, CoverLetterWriterAgent, llm)
    wish = "In the letter, mention that I can start in March."
    out = agent.run({"candidate_text": {"cv": SAMPLE_CV, "cover_letter": OLD_LETTER},
                     "cover_letter_content_feedback": wish, "job_description_info": AD_INFO, "messages": []})
    instructions = _instructions(llm.prompts[0])
    assert wish in instructions and "Apply ONLY these specific changes" in instructions  # the wish verbatim
    assert "written for a different job and employer" in instructions
    letter = out["generated_cover_letter"]
    assert "FreshCart" in letter and "Brightmill" not in letter and "I can start in March." in letter

    # a later request on the letter already written for this job is applied as asked, nothing more
    llm2 = LetterLLM()
    agent2 = _writer(monkeypatch, tmp_path, CoverLetterWriterAgent, llm2)
    agent2.run({"candidate_text": {"cv": SAMPLE_CV, "cover_letter": OLD_LETTER}, "generated_cover_letter": letter,
                "cover_letter_content_feedback": "Make it shorter.", "job_description_info": AD_INFO, "messages": []})
    assert "different job and employer" not in llm2.prompts[0]


def test_a_wish_on_the_first_cv_does_not_get_the_letter_note(monkeypatch, tmp_path):
    llm = WriterLLM()
    _writer(monkeypatch, tmp_path, CVWriterAgent, llm).run(
        {"candidate_text": {"cv": SAMPLE_CV}, "cv_content_feedback": "Stress my team leadership.",
         "job_description_info": AD_INFO, "messages": []})
    assert "different job and employer" not in llm.prompts[0]


def test_first_cv_with_a_request_follows_the_request(monkeypatch, tmp_path):
    """D2 is out of scope: "write my CV and stress X" applies the wish to the user's CV as asked."""
    llm = WriterLLM()
    agent = _writer(monkeypatch, tmp_path, CVWriterAgent, llm)
    wish = "Stress my team leadership."
    agent.run({"candidate_text": {"cv": SAMPLE_CV}, "cv_content_feedback": wish,
               "job_description_info": AD_INFO, "messages": []})
    instructions = _instructions(llm.prompts[0])
    assert wish in instructions and "Apply ONLY these specific changes to the existing CV" in instructions
    assert "CREATIVE LIBERTY" not in instructions


def test_first_cv_without_a_request_is_tailored(monkeypatch, tmp_path):
    llm = WriterLLM()
    agent = _writer(monkeypatch, tmp_path, CVWriterAgent, llm)
    agent.run({"candidate_text": {"cv": SAMPLE_CV}, "job_description_info": AD_INFO, "messages": []})
    assert "CREATIVE LIBERTY GRANTED" in _instructions(llm.prompts[0])


def test_the_router_keeps_a_first_request_for_the_writer(monkeypatch):
    """The first message's content wish is stored for the not-yet-written CV and reaches its first draft."""
    monkeypatch.setattr(router, "log_messages", lambda *_, **__: 0)
    agent = router.RouterAgent.__new__(router.RouterAgent)
    agent.max_history_messages = 10
    answer = router.RouterResponse(next_action="draft_cv", message_to_user="", needs_user_input=False,
                                   cv_content_feedback="Put my HACCP course first.")
    agent.llm = type("L", (), {"with_structured_output": lambda self, s: RunnableLambda(lambda _: answer)})()
    out = agent.run({"messages": add_messages([], [{"role": "user", "content": "Write my CV; put my HACCP course first."}])})
    assert out["next"] == "draft_cv" and out["cv_content_feedback"] == "Put my HACCP course first."


# --- D7 and the user's chat statements: writers always see the candidate's own material ---------

def test_redrafts_see_the_original_cv_letter_notes_portfolio_and_chat(monkeypatch, tmp_path):
    llm = WriterLLM()
    agent = _writer(monkeypatch, tmp_path, CVWriterAgent, llm)
    state = {
        "candidate_text": {"cv": SAMPLE_CV, "cover_letter": OLD_LETTER, "notes": "NOTE: HACCP course 2021"},
        "relevant_experience": "PROJECT: Smokehouse",
        "generated_cv": "CV DRAFT 2 (dropped the Data Analyst job)",
        "cv_content_feedback": "Shorten the summary.",
        "job_description_info": AD_INFO,
        "messages": [{"role": "user", "content": "I also ran a market stall on weekends in 2019."}],
    }
    agent.run(state)
    prompt = llm.prompts[0]
    assert "CV DRAFT 2" in _section(prompt, "**Existing CV (Base Document):**")
    assert "Data Analyst, DataInc" in _section(prompt, "**Candidate's Original CV")
    assert "Brightmill" in _section(prompt, "**Candidate's Earlier Cover Letter")
    assert "NOTE: HACCP course 2021" in prompt and "PROJECT: Smokehouse" in prompt
    assert "market stall on weekends in 2019" in _section(prompt, "**What the User Said in This Chat")


def test_first_draft_does_not_repeat_the_original_cv(monkeypatch, tmp_path):
    llm = WriterLLM()
    agent = _writer(monkeypatch, tmp_path, CVWriterAgent, llm)
    agent.run({"candidate_text": {"cv": SAMPLE_CV}, "job_description_info": AD_INFO, "messages": []})
    prompt = llm.prompts[0]
    assert prompt.count("Data Analyst, DataInc") == 1
    assert "(The same as the existing document above.)" in _section(prompt, "**Candidate's Original CV")


def test_critic_and_grounding_check_count_the_users_statements_as_material():
    text = critique.format_candidate_material(critique.candidate_material({
        "candidate_text": {"cv": "CV"},
        "messages": [{"role": "user", "content": "I passed my master butcher exam in May."}],
    }))
    assert "User's Own Requests and Statements in This Session:\n- I passed my master butcher exam in May." in text


# --- D5: the preference filter never drops the removal of unsupported claims -----------------------

def test_removals_survive_a_filter_that_drops_every_suggestion(monkeypatch, tmp_path):
    llm = WriterLLM(filter_answer='""')
    agent = _writer(monkeypatch, tmp_path, CVWriterAgent, llm)
    out = agent.run({
        "candidate_text": {"cv": SAMPLE_CV},
        "generated_cv": "CV with an invented forklift licence",
        "cv_needs_refinement": True,
        "cv_critique_improvement_instructions": "Rewrite the summary.",
        "cv_claims_to_remove": ["Forklift licence (2019)"],
        "cv_user_requests": ["Keep my summary exactly as it is."],
        "job_description_info": AD_INFO,
        "messages": [],
    })
    assert "Forklift licence" not in llm.filter_prompts[0]  # the claims never go through the filter
    instructions = _instructions(llm.prompts[0])
    assert "Forklift licence (2019)" in instructions and "Remove these claims" in instructions
    assert critique.NO_NEW_CLAIMS_RULE in instructions
    assert "Rewrite the summary" not in instructions  # the conflicting suggestion was dropped
    assert out["cv_refinement_count"] == 1


# --- D4: a final draft with claims gets a removal-only pass, and the rest is shown to the user ---

def _finalize(monkeypatch, tmp_path, remaining):
    llm = WriterLLM("CV without the claim")
    checked = []

    def checker(label, text, material, employer_info=None):
        checked.append(text)
        return remaining

    agent = _writer(monkeypatch, tmp_path, CVWriterAgent, llm, grounding_checker=checker)
    history = [{"iteration": 1, "score": 90, "content": "CV with claim", "unsupported_claims": ["Forklift licence", "AWS"]}]
    out = agent.finalize_best_version({"cv_history": history, "candidate_text": {"cv": SAMPLE_CV},
                                       "job_description_info": AD_INFO, "messages": []})
    return out, llm, checked


def test_removal_pass_is_kept_when_it_removes_claims(monkeypatch, tmp_path):
    out, llm, checked = _finalize(monkeypatch, tmp_path, remaining=[])
    assert "REMOVAL ONLY" in _instructions(llm.prompts[0]) and "- Forklift licence" in _instructions(llm.prompts[0])
    assert checked == ["CV without the claim"]  # the grounding check ran again on the result
    assert out["generated_cv"] == "CV without the claim" and out["cv_final_iteration"] == 2
    assert out["cv_history"][-1]["removal_pass_of"] == 1
    assert pending_notices(out) == []


def test_claims_that_remain_are_listed_to_the_user_word_for_word(monkeypatch, tmp_path):
    out, _, _ = _finalize(monkeypatch, tmp_path, remaining=["AWS"])
    assert out["generated_cv"] == "CV without the claim"
    (notice,) = pending_notices(out)
    assert "still contains claims" in notice and "- AWS" in notice

    # the router puts the notice in front of the message the user sees, whatever its model writes
    monkeypatch.setattr(router, "log_messages", lambda *_, **__: 0)
    agent = router.RouterAgent.__new__(router.RouterAgent)
    agent.max_history_messages = 10
    answer = router.RouterResponse(next_action="draft_cover_letter", message_to_user="", needs_user_input=False)
    agent.llm = type("L", (), {"with_structured_output": lambda self, s: RunnableLambda(lambda _: answer)})()
    state = {"messages": add_messages([], [{"role": "user", "content": "CV, then the letter."}]), "generated_cv": "CV",
             NOTICES_KEY: out[NOTICES_KEY]}
    routed = agent.run(state)
    assert NOTICES_KEY not in routed  # not shown to the user yet (no question): stays pending
    answer.next_action, answer.message_to_user = "collect_user_input", "Both documents are ready. Any changes?"
    asked = agent.run(state)
    assert asked["user_input_message"].startswith(notice) and asked["user_input_message"].endswith("Any changes?")
    assert asked[NOTICES_KEY] is None


# --- N1: a grounding check that could not run is not read as "no invented claims" -----------------

class BrokenModel:
    """A model whose every call fails (e.g. a mistyped model name or an outage)."""

    def with_structured_output(self, schema):
        def fail(_):
            raise RuntimeError("model not found")
        return RunnableLambda(fail)


def _critic(grounding_llm):
    critic = critique.CritiqueAgent.__new__(critique.CritiqueAgent)
    critic.grounding_llm = grounding_llm
    critic.critique_cv = lambda **_: critique.CritiqueResponse(
        quality_score=95, content_quality_feedback="ok", ats_compatibility_feedback="ok", job_alignment_feedback="ok")
    return critic


def test_a_failed_grounding_check_is_not_an_empty_list(monkeypatch):
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    critic = _critic(BrokenModel())
    assert critic.check_grounding("CV", "Forklift licence (2019)", {"cv": SAMPLE_CV}) is None

    out = critic.run_cv({"generated_cv": "Ana Horvat\nForklift licence (2019)", "cv_history": [], "messages": [],
                         "candidate_text": {"cv": SAMPLE_CV}, "job_description_info": AD_INFO})
    (entry,) = out["cv_history"]
    assert entry["grounding_checked"] is False and out["cv_critique"]["grounding_checked"] is False


def test_an_unchecked_draft_ranks_below_a_checked_draft_with_claims():
    """Probe P1c: draft 1 was flagged, the check on draft 2 failed; draft 2 must not win as 'honest'."""
    from agents.refinement import best_draft

    flagged = {"iteration": 1, "score": 70, "unsupported_claims": ["Forklift licence (2019)"], "defects": 0}
    unchecked = {"iteration": 2, "score": 90, "unsupported_claims": [], "grounding_checked": False, "defects": 0}
    assert best_draft([flagged, unchecked]) is flagged
    clean = {"iteration": 3, "score": 60, "unsupported_claims": [], "grounding_checked": True, "defects": 0}
    assert best_draft([flagged, unchecked, clean]) is clean


def _finalize_with(monkeypatch, tmp_path, history, checker_answers, state=None):
    llm = WriterLLM("CV without the claim")
    answers = list(checker_answers)
    checked = []

    def checker(label, text, material, employer_info=None):
        checked.append(text)
        return answers.pop(0)

    agent = _writer(monkeypatch, tmp_path, CVWriterAgent, llm, grounding_checker=checker)
    out = agent.finalize_best_version({"cv_history": history, "candidate_text": {"cv": SAMPLE_CV},
                                       "job_description_info": AD_INFO, "messages": [], **(state or {})})
    return out, checked


def test_finalize_keeps_the_claims_when_the_recheck_cannot_run(monkeypatch, tmp_path):
    history = [{"iteration": 1, "score": 90, "content": "CV with claim", "unsupported_claims": ["Forklift licence"]}]
    out, checked = _finalize_with(monkeypatch, tmp_path, history, [None])
    assert checked == ["CV without the claim"]  # the removal pass ran, but its check failed
    assert out["generated_cv"] == "CV with claim" and out["cv_final_iteration"] == 1
    (notice,) = pending_notices(out)
    assert "still contains claims" in notice and "- Forklift licence" in notice


def test_a_final_draft_that_was_never_checked_gets_a_fixed_notice(monkeypatch, tmp_path):
    history = [{"iteration": 1, "score": 90, "content": "Unchecked CV", "unsupported_claims": [], "grounding_checked": False}]
    out, checked = _finalize_with(monkeypatch, tmp_path, history, [None])
    assert checked == ["Unchecked CV"]  # one more try at finalize
    assert out["generated_cv"] == "Unchecked CV"
    (notice,) = pending_notices(out)
    assert notice == "Warning: the check for invented claims could not run on the final CV. Read it before you send it."


def test_a_recheck_at_finalize_that_runs_feeds_the_removal_pass(monkeypatch, tmp_path):
    history = [{"iteration": 1, "score": 90, "content": "Unchecked CV", "unsupported_claims": [], "grounding_checked": False}]
    out, checked = _finalize_with(monkeypatch, tmp_path, history, [["Forklift licence"], []])
    assert checked == ["Unchecked CV", "CV without the claim"]
    assert out["generated_cv"] == "CV without the claim" and out["cv_history"][-1]["grounding_checked"] is True
    assert pending_notices(out) == []


# --- N2: each document's warnings are set or cleared by every finalize, so none outlives its draft --

def test_a_stale_claims_warning_is_cleared_when_the_next_final_draft_is_clean(monkeypatch, tmp_path):
    """Probe P2: finalize warned about a claim, the user's reply led to a clean redraft; no false alarm."""
    flagged = [{"iteration": 1, "score": 90, "content": "CV with claim", "unsupported_claims": ["Forklift licence"]}]
    first, _ = _finalize_with(monkeypatch, tmp_path, flagged, [["Forklift licence"]])
    assert len(pending_notices(first)) == 1

    other_doc = "Warning: the cover letter PDF does not match the checked text exactly."
    pending = {**first[NOTICES_KEY], "cv_pdf": "Warning: the CV PDF does not match the checked text exactly.",
               "cover_letter_pdf": other_doc}
    clean = flagged + [{"iteration": 2, "score": 80, "content": "Ana Horvat\nana@example.com", "unsupported_claims": []}]
    second, _ = _finalize_with(monkeypatch, tmp_path, clean, [], state={"cv_cycle_start": 1, NOTICES_KEY: pending})
    assert second["generated_cv"] == "Ana Horvat\nana@example.com"
    assert pending_notices(second) == [other_doc]  # this CV's claim and PDF warnings are gone; the letter's stays


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex")
def test_a_matching_pdf_clears_the_warning_about_an_earlier_pdf(tmp_path):
    from agents.cv_pdf_generator import CVPDFGenerator

    text = "Ana Horvat\nButcher with eight years at the counter."
    gen = CVPDFGenerator.__new__(CVPDFGenerator)
    gen.output_folder, gen.temp_folder = str(tmp_path / "out"), str(tmp_path / "out" / "temp_latex")
    gen.template_path, gen.max_latex_fix_iterations, gen._latex_fixer = str(REPO / "templates" / "cv_template.tex"), 0, None
    os.makedirs(gen.temp_folder)
    gen._build_latex_document = lambda template, plain, style: LATEX.format(body=text.replace("\n", "\\par "))
    out = gen.run({"generated_cv": text, "cv_history": [{"iteration": 1}], "messages": [],
                   NOTICES_KEY: {"cv_pdf": "Warning: the CV PDF does not match the checked text exactly."}})
    assert out["cv_pdf_check"]["ok"] is True and pending_notices(out) == []


# --- D6: the PDF's text is compared with the checked text -------------------------------------------

def test_compare_texts_finds_dropped_and_added_words():
    result = compare_texts("Led a team of five. HACCP certified.", "Led a team of five. Award-winning butcher.")
    assert result["missing_words"] == ["haccp", "certified"]
    assert result["extra_words"] == ["award", "winning", "butcher"]
    assert result["text_words_found_in_pdf_pct"] == 71.4  # 5 of the text's 7 words


def test_compare_texts_ignores_text_extraction_artefacts():
    """Seen in real PDFs: spaces lost between words, and a word split at a line break."""
    text = "I have also used AWS services. Experienced butcher with HACCP training."
    pdf = "Ihavealsousedawsservices. Ex perienced butcher with HACCP training."
    result = compare_texts(text, pdf)
    assert result["missing_words"] == [] and result["extra_words"] == []
    assert result["text_words_found_in_pdf_pct"] == 100.0


def test_a_changed_number_is_seen_even_inside_a_year(tmp_path):
    """Probe P4: '5' is inside '2015', so the joined-text tolerance hid 'team of 5' becoming 'team of 15'."""
    text = "Butcher, 2015-2023. Led a team of 5. Cut 200 kg a day."
    result = compare_texts(text, text.replace("team of 5", "team of 15"))
    assert result["number_changes"] == ["-5", "+15"]
    assert compare_texts(text, text.replace("200 kg", "20 kg"))["number_changes"] == ["-200", "+20"]
    assert compare_texts(text, text)["number_changes"] == []
    # a lost space between numbers is an extraction artefact, not a changed number
    assert compare_texts("Salary 15,000 to 24,000.", "Salary 15,000to24,000.")["number_changes"] == []


def test_a_dropped_or_repeated_line_is_seen_by_its_word_counts():
    """Probe P4: every word of a dropped repeated bullet still occurs elsewhere, so a set comparison missed it."""
    lines = ["Ana Horvat", "- Served customers at the meat counter.", "Lidl, butcher",
             "- Served customers at the meat counter.", "Konzum, butcher"]
    text = "\n".join(lines)
    dropped = compare_texts(text, "\n".join(lines[:3] + lines[4:]))
    assert dropped["missing_word_count"] == 5 and dropped["missing_occurrence_count"] == 5
    assert {"served", "customers", "meat", "counter"} <= set(dropped["missing_words"])
    repeated = compare_texts(text, text + "\n- Served customers at the meat counter.")
    assert repeated["extra_occurrence_count"] == 5


def test_check_pdf_fails_a_changed_number_and_a_dropped_line(monkeypatch):
    from agents import pdf_text_check

    text = "Ana Horvat\nButcher at Lidl 2015-2023, team of 5.\n- Served customers.\n- Served customers."
    for pdf_text in (text.replace("team of 5", "team of 15"), text.rsplit("\n", 1)[0]):
        monkeypatch.setattr(pdf_text_check, "extract_pdf_text", lambda path, t=pdf_text: (t, 1, None))
        assert pdf_text_check.check_pdf("x.pdf", text)["ok"] is False
    monkeypatch.setattr(pdf_text_check, "extract_pdf_text", lambda path: (text, 1, None))
    assert pdf_text_check.check_pdf("x.pdf", text)["ok"] is True


LATEX = "\\documentclass{{article}}\\usepackage[T1]{{fontenc}}\\begin{{document}}{body}\\end{{document}}\n"


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex")
def test_a_pdf_that_drops_text_is_converted_again_and_the_closer_one_kept(monkeypatch, tmp_path):
    from agents.cv_pdf_generator import CVPDFGenerator

    text = "Ana Horvat\nButcher with eight years at the counter.\nHACCP course 2021.\nForklift-free warehouse work."
    conversions = iter([
        LATEX.format(body="Ana Horvat\\par Butcher with eight years at the counter."),  # dropped two lines
        LATEX.format(body=text.replace("\n", "\\par ")),
    ])
    gen = CVPDFGenerator.__new__(CVPDFGenerator)
    gen.output_folder, gen.temp_folder = str(tmp_path / "out"), str(tmp_path / "out" / "temp_latex")
    gen.template_path, gen.max_latex_fix_iterations, gen._latex_fixer = str(REPO / "templates" / "cv_template.tex"), 0, None
    os.makedirs(gen.temp_folder)
    gen._build_latex_document = lambda template, plain, style: next(conversions)
    out = gen.run({"generated_cv": text, "cv_history": [{"iteration": 1}], "messages": []})
    assert out["cv_pdf_check"]["ok"] is True and out["cv_pdf_check"]["converted_twice"] is True
    assert check_pdf(out["cv_pdf_path"], text)["ok"] is True
    assert pending_notices(out) == []


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex")
def test_a_pdf_that_still_differs_is_reported(tmp_path):
    from agents.cv_pdf_generator import CVPDFGenerator

    text = "Ana Horvat\nButcher with eight years at the counter.\nHACCP course 2021."
    gen = CVPDFGenerator.__new__(CVPDFGenerator)
    gen.output_folder, gen.temp_folder = str(tmp_path / "out"), str(tmp_path / "out" / "temp_latex")
    gen.template_path, gen.max_latex_fix_iterations, gen._latex_fixer = str(REPO / "templates" / "cv_template.tex"), 0, None
    os.makedirs(gen.temp_folder)
    gen._build_latex_document = lambda template, plain, style: LATEX.format(body="Ana Horvat\\par Award-winning master butcher.")
    out = gen.run({"generated_cv": text, "cv_history": [{"iteration": 1}], "messages": []})
    assert out["cv_pdf_check"]["ok"] is False
    (notice,) = pending_notices(out)
    assert "does not match the checked text" in notice and "haccp" in notice and "award" in notice


def test_a_retry_that_changes_numbers_does_not_replace_a_first_pdf_that_only_dropped_words(monkeypatch, tmp_path):
    from agents import pdf_text_check
    from agents.cv_pdf_generator import CVPDFGenerator

    sentences = [f"Served customer group number {w} at the meat counter every morning." for w in
                 ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")]
    text = "Ana Horvat\nButcher at Lidl 2019-2023, team of 5.\n" + "\n".join(sentences) + "\nHACCP hygiene course passed with distinction."
    pdf_texts = {
        "first": text.replace("\nHACCP hygiene course passed with distinction.", ""),
        "retry": text.replace("2019", "2018").replace("team of 5", "team of 3"),
    }
    monkeypatch.setattr(pdf_text_check, "extract_pdf_text",
                        lambda path: (pdf_texts["retry" if "_retry" in os.path.basename(path) else "first"], 1, None))

    def compile_latex(latex, name, max_fix_iterations):
        path = os.path.join(gen.temp_folder, f"{name}.pdf")
        open(path, "w", encoding="utf-8").write(name)
        return path, 0, None

    gen = CVPDFGenerator.__new__(CVPDFGenerator)
    gen.output_folder, gen.temp_folder = str(tmp_path / "out"), str(tmp_path / "out" / "temp_latex")
    gen.template_path, gen.max_latex_fix_iterations, gen._latex_fixer = str(REPO / "templates" / "cv_template.tex"), 0, None
    os.makedirs(gen.temp_folder)
    gen._build_latex_document = lambda template, plain, style: "latex"
    gen._compile_latex = compile_latex
    first = pdf_text_check.compare_texts(text, pdf_texts["first"])
    retry = pdf_text_check.compare_texts(text, pdf_texts["retry"])
    assert first["number_changes"] == [] and retry["number_changes"]
    assert retry["text_words_found_in_pdf_pct"] > first["text_words_found_in_pdf_pct"]

    out = gen.run({"generated_cv": text, "cv_history": [{"iteration": 1}], "messages": []})
    assert out["cv_pdf_check"]["ok"] is False and out["cv_pdf_check"]["converted_twice"] is True
    assert out["cv_pdf_check"]["number_changes"] == []
    assert open(out["cv_pdf_path"], encoding="utf-8").read() == "generated_CV_draft_1"


def test_an_unreadable_pdf_is_not_reported_as_a_mismatch(tmp_path):
    fake = tmp_path / "x.pdf"
    fake.write_text("not a pdf", encoding="utf-8")
    assert check_pdf(fake, "text")["ok"] is None


# --- D8: the cover letter writer has the same truthfulness rules as the CV writer ---------------

def test_both_writers_share_the_truthfulness_rules(monkeypatch, tmp_path):
    for cls in (CVWriterAgent, CoverLetterWriterAgent):
        llm = WriterLLM()
        agent = _writer(monkeypatch, tmp_path, cls, llm)
        agent.run({"candidate_text": {"cv": SAMPLE_CV}, "job_description_info": AD_INFO, "messages": []})
        system = llm.prompts[0].split("\nHuman: ", 1)[0]
        assert "NO INVENTED COMMITMENTS OR PREFERENCES" in system, cls.__name__
        assert "Show enthusiasm for the role" not in system, cls.__name__


# --- D9: company research ------------------------------------------------------------------------

def test_no_company_name_means_no_search(monkeypatch):
    monkeypatch.setattr(search, "write_to_debug", lambda *_, **__: None)
    agent = search.SearchAgent.__new__(search.SearchAgent)
    agent.search_tavily = lambda query: pytest.fail("searched without a company name")
    agent.llm = None  # work-context extraction fails and falls back to the ad's work type
    for name in (None, "", "Not specified", "unknown"):
        out = agent.run({"job_description_info": {"company_name": name, "location": "Zagreb", "work_type": "on-site"},
                         "messages": []})
        assert out["company_info"]["company_description"] == search.NO_COMPANY_NAME
        assert out["company_info"]["work_context"]["arrangement"] == "on-site"


def test_summary_is_told_which_company_to_describe(monkeypatch):
    prompts = []
    agent = search.SearchAgent.__new__(search.SearchAgent)
    agent.llm = RunnableLambda(lambda p: (prompts.append(p.to_string()), type("M", (), {"content": "ok"})())[1])
    agent.process_search_results({"results": [{"content": "Other Corp sells shoes."}]}, "FreshCart Supermarkets")
    assert "The company named in the job ad is: FreshCart Supermarkets" in prompts[0]
    assert "Could not identify the company in the search results." in prompts[0]


def test_the_app_reads_the_page_text_it_pays_for():
    results = {"results": [{"content": "snippet A", "raw_content": "FULL PAGE A " * 400},
                           {"content": "snippet B", "raw_content": None}]}
    text = search.SearchAgent._extract_search_content(None, results)
    assert "FULL PAGE A" in text and "snippet A" not in text and "snippet B" in text
    assert len(search.search_result_texts(results)[0]["text"]) == search.CHARS_PER_RESULT


def test_unsupported_claims_about_the_employer_are_flagged(monkeypatch):
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    prompts = []

    class Grounding:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda p: (prompts.append(p.to_string()), schema(
                unsupported_claims=[], unsupported_employer_claims=["your recent Series B"]))[1])

    agent = critique.CritiqueAgent.__new__(critique.CritiqueAgent)
    agent.grounding_llm = Grounding()
    agent.critique_cover_letter = lambda **_: critique.CritiqueResponse(
        quality_score=95, content_quality_feedback="ok", ats_compatibility_feedback="ok", job_alignment_feedback="ok")
    out = agent.run_cover_letter({
        "generated_cover_letter": "I admire your recent Series B.", "cover_letter_history": [], "messages": [],
        "job_description_info": AD_INFO, "company_info": {"company_description": "FreshCart runs 40 shops."},
    })
    assert out["cover_letter_needs_refinement"] is True
    assert out["cover_letter_claims_to_remove"] == [critique.EMPLOYER_CLAIM_PREFIX + "your recent Series B"]
    employer_part = prompts[0].split("**Job ad and company research**", 1)[1]
    assert "FreshCart runs 40 shops." in employer_part and "HACCP required" in employer_part


# --- D10: one failed model call does not end the session -----------------------------------------

def test_a_failed_writer_call_is_retried(monkeypatch, tmp_path):
    calls = []

    class Flaky(WriterLLM):
        def with_structured_output(self, schema):
            inner = super().with_structured_output(schema)

            def answer(prompt):
                calls.append(1)
                if len(calls) == 1:
                    raise ValueError("could not parse the structured output")
                return inner.invoke(prompt)
            return RunnableLambda(answer)

    agent = _writer(monkeypatch, tmp_path, CVWriterAgent, Flaky("CV"))
    out = agent.run({"candidate_text": {"cv": SAMPLE_CV}, "job_description_info": AD_INFO, "messages": []})
    assert out["generated_cv"] == "CV" and len(calls) == 2


# --- D11: the job-ad step does not ask the model to type the whole ad again -----------------------

def test_job_ad_extraction_does_not_echo_the_ad(monkeypatch):
    seen = {}

    class Extractor:
        def with_structured_output(self, schema):
            seen["fields"] = set(schema.model_fields)
            return RunnableLambda(lambda _: schema(job_title="Butcher", must_haves=["HACCP"], nice_to_haves=["Forklift"]))

    monkeypatch.setattr(job_description, "init_chat_model", lambda *_, **__: Extractor())
    monkeypatch.setattr(job_description, "ask_for_document", lambda *_: "We hire a butcher. HACCP required.")
    out = job_description.JobDescriptionAgent().run({"messages": []})
    assert "job_description" not in seen["fields"]
    assert {"must_haves", "nice_to_haves"} <= seen["fields"]  # I2: the ad's requirements, as lists
    assert out["job_description_info"]["job_description"] == "We hire a butcher. HACCP required."
    assert out["job_description_info"]["must_haves"] == ["HACCP"]


# --- I1, I6: the writer and critic aim at a truthful match; the job's country conventions ----------

class CriticLLM:
    """Stands in for the critic model: keeps every prompt it is sent, finds nothing to fix."""

    def __init__(self):
        self.prompts = []

    def with_structured_output(self, schema):
        def answer(prompt):
            self.prompts.append(prompt.to_string())
            if schema is critique.CritiqueResponse:
                return schema(quality_score=90, content_quality_feedback="ok",
                              ats_compatibility_feedback="ok", job_alignment_feedback="ok")
            return schema(unsupported_claims=[])
        return RunnableLambda(answer)


def test_writer_and_critic_aim_at_a_truthful_match_not_keywords(monkeypatch, tmp_path):
    llm = WriterLLM()
    _writer(monkeypatch, tmp_path, CVWriterAgent, llm).run(
        {"candidate_text": {"cv": SAMPLE_CV}, "job_description_info": AD_INFO, "messages": []})
    writer_prompt = llm.prompts[0]

    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    critic = critique.CritiqueAgent.__new__(critique.CritiqueAgent)
    critic.llm, critic.grounding_llm = CriticLLM(), CriticLLM()
    critic.run_cv({"generated_cv": "Ana Horvat, butcher", "cv_history": [], "messages": [],
                   "candidate_text": {"cv": SAMPLE_CV}, "job_description_info": AD_INFO})
    critic_prompt = critic.llm.prompts[0]

    for prompt in (writer_prompt, critic_prompt):
        assert "most convincing TRUTHFUL match to this job ad for a human recruiter" in prompt
        assert "passes AI-based screening" not in prompt
    assert "Check specifically for keyword matching" not in critic_prompt
    assert "never add an ad term the candidate's material does not support" in writer_prompt


def test_cv_writer_sees_the_jobs_location_and_country_conventions(monkeypatch, tmp_path):
    llm = WriterLLM()
    agent = _writer(monkeypatch, tmp_path, CVWriterAgent, llm)
    agent.run({"candidate_text": {"cv": SAMPLE_CV}, "job_description_info": AD_INFO, "messages": []})
    assert "Location: Zagreb, Croatia" in _section(llm.prompts[0], "**Job Description:**")
    assert "usual CV conventions of the job's country" in llm.prompts[0]


# --- I3: the grounding check runs on its own model -------------------------------------------------

def test_grounding_check_has_its_own_model(monkeypatch):
    made = []
    monkeypatch.setattr(critique, "init_chat_model", lambda model, **_: made.append(model) or model)
    agent = critique.CritiqueAgent(model="openai:gpt-6-luna", grounding_model="openai:gpt-5.6-luna")
    assert agent.llm == "openai:gpt-6-luna" and agent.grounding_llm == "openai:gpt-5.6-luna"
    assert critique.CritiqueAgent(model="openai:gpt-6-luna").grounding_llm == "openai:gpt-6-luna"

    from config.config_loader import get_agent_config
    monkeypatch.delenv("CV_EDITOR_MODEL", raising=False)
    monkeypatch.delenv("CV_EDITOR_GROUNDING_MODEL", raising=False)
    assert get_agent_config("grounding")["model"] != get_agent_config("cv_writer")["model"]


# --- the whole critic (quality review and fact check) is independent of the writers ---------------

def _clear_model_overrides(monkeypatch):
    for var in ("CV_EDITOR_MODEL", "CV_EDITOR_CRITIC_MODEL", "CV_EDITOR_GROUNDING_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_shipped_critic_runs_on_a_model_different_from_the_writers(monkeypatch):
    from config.config_loader import get_agent_config

    _clear_model_overrides(monkeypatch)
    writer_models = {get_agent_config(name)[key] for name in ("cv_writer", "cover_letter_writer")
                     for key in ("model", "filter_model")}
    assert get_agent_config("critique")["model"] not in writer_models
    assert get_agent_config("grounding")["model"] not in writer_models


def test_the_model_override_never_replaces_the_critic(monkeypatch):
    from config.config_loader import get_agent_config

    _clear_model_overrides(monkeypatch)
    shipped_critique = get_agent_config("critique")["model"]
    shipped_grounding = get_agent_config("grounding")["model"]

    monkeypatch.setenv("CV_EDITOR_MODEL", "openai:writer")
    assert get_agent_config("cv_writer")["model"] == get_agent_config("router")["model"] == "openai:writer"
    assert get_agent_config("critique")["model"] == shipped_critique
    assert get_agent_config("grounding")["model"] == shipped_grounding


def test_each_critic_call_has_its_own_override(monkeypatch):
    from config.config_loader import get_agent_config

    _clear_model_overrides(monkeypatch)
    shipped_grounding = get_agent_config("grounding")["model"]
    shipped_writer = get_agent_config("cv_writer")["model"]

    monkeypatch.setenv("CV_EDITOR_CRITIC_MODEL", "ollama:critic")
    assert get_agent_config("critique")["model"] == "ollama:critic"
    assert get_agent_config("grounding")["model"] == shipped_grounding  # the fact check is separate
    assert get_agent_config("cv_writer")["model"] == shipped_writer

    monkeypatch.setenv("CV_EDITOR_GROUNDING_MODEL", "ollama:fact-check")
    assert get_agent_config("grounding")["model"] == "ollama:fact-check"
    assert get_agent_config("critique")["model"] == "ollama:critic"


# --- N5: a local CV_EDITOR_MODEL keeps the whole run local ------------------------------------------

def test_a_local_model_override_also_applies_to_the_critic(monkeypatch, caplog):
    from config.config_loader import get_agent_config

    _clear_model_overrides(monkeypatch)
    monkeypatch.setenv("CV_EDITOR_MODEL", "ollama:gemma4:e4b")
    with caplog.at_level("WARNING", logger="config.config_loader"):
        assert get_agent_config("critique")["model"] == "ollama:gemma4:e4b"
        assert get_agent_config("grounding")["model"] == "ollama:gemma4:e4b"
    assert "shares the writer's model" in caplog.text

    # the critic's own variables still win, also for a local run
    monkeypatch.setenv("CV_EDITOR_GROUNDING_MODEL", "ollama:other")
    assert get_agent_config("grounding")["model"] == "ollama:other"
    assert get_agent_config("critique")["model"] == "ollama:gemma4:e4b"


def test_a_local_run_starts_without_an_openai_key(monkeypatch, tmp_path):
    """Probe P6: with only CV_EDITOR_MODEL=ollama:... the grounding check still made an OpenAI client."""
    (tmp_path / "templates").symlink_to(REPO / "templates")
    (tmp_path / "data" / "CV").mkdir(parents=True)
    shutil.copy(REPO / "examples" / "sample_CV.txt", tmp_path / "data" / "CV" / "CV.txt")
    monkeypatch.chdir(tmp_path)
    _clear_model_overrides(monkeypatch)
    monkeypatch.setenv("CV_EDITOR_MODEL", "ollama:gemma4:e4b")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    made = []
    real = critique.init_chat_model
    monkeypatch.setattr(critique, "init_chat_model", lambda model, **kw: made.append(model) or real(model, **kw))

    import langgraph_agent as la
    la.MasterAgent()
    assert made and all(m.startswith("ollama:") for m in made)



# --- I7: placeholders such as [EMAIL] make the program ask once before the PDF -------------------

class PlaceholderLLM(ScriptedLLM):
    def _answer(self, name, schema, prompt):
        if name == "CVGenerationResponse":
            text = prompt.to_string()
            self.writer_instructions.append(_instructions(text))
            filled = "placeholders for missing details" in _instructions(text)
            contact = "ana@example.com" if filled else "[EMAIL] | [PHONE]"
            return schema(cv_content=f"Ana Horvat\n{contact}\nDraft {len(self.writer_instructions)}", notes="")
        return super()._answer(name, schema, prompt)


def test_missing_details_are_asked_for_once_before_the_pdf(monkeypatch, tmp_path):
    assert find_placeholders("Call [PHONE] or mail [EMAIL]; [PHONE]") == ["[PHONE]", "[EMAIL]"]
    llm = PlaceholderLLM(router_script=[_route("draft_cv"), _route("exit", "Done.")], critique_scores=[(90, ""), (90, "")])
    agent, rendered = _build_agent(monkeypatch, tmp_path, llm)
    ad = (REPO / "job_description" / "sample_job_ad.txt").read_text(encoding="utf-8")

    final = _play(agent, [ad, "My email is ana@example.com; leave the phone out."])

    assert "ana@example.com; leave the phone out" in llm.writer_instructions[1]
    assert final["cv_placeholders"] == [] and final["cv_placeholders_asked"] is True
    assert list(rendered) == ["generated_CV_draft_2.pdf"]  # the PDF was made once, from the filled-in draft
    assert "[EMAIL]" not in rendered["generated_CV_draft_2.pdf"]
    asked = [m for m in final["messages"] if "placeholders for details I do not have" in getattr(m, "content", "")]
    assert len(asked) == 1


def test_missing_details_reply_is_applied_when_the_critique_budget_ran_out(monkeypatch, tmp_path):
    """The last critique still asked for a redraft past the budget; the user's reply must win over it."""
    llm = PlaceholderLLM(router_script=[_route("draft_cv"), _route("exit", "Done.")],
                         critique_scores=[(70, "Fix the title."), (70, "Fix the title."), (90, "")])
    agent, rendered = _build_agent(monkeypatch, tmp_path, llm)
    ad = (REPO / "job_description" / "sample_job_ad.txt").read_text(encoding="utf-8")

    final = _play(agent, [ad, "My email is ana@example.com; leave the phone out."])

    assert len(llm.writer_instructions) == 3
    assert "ana@example.com; leave the phone out" in llm.writer_instructions[2]
    assert "Fix the title." not in llm.writer_instructions[2]
    assert final["cv_placeholders"] == []
    [pdf] = rendered.values()  # the PDF was made once, from the filled-in draft
    assert "ana@example.com" in pdf and "[EMAIL]" not in pdf


# --- C5: no import-time side effects ---------------------------------------------------------------

def test_importing_the_app_creates_no_folders_and_configures_no_logging(tmp_path):
    code = ("import logging, langgraph_agent, os, sys; "
            "sys.exit(1 if logging.getLogger().handlers or os.listdir('.') else 0)")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, (result.stderr[-2000:], os.listdir(tmp_path))


def test_shipped_config_has_no_score_threshold_left():
    import yaml

    config = yaml.safe_load((REPO / "config" / "agent_models.yaml").read_text(encoding="utf-8"))
    assert "quality_improvement_threshold" not in config["workflow"]
    assert "quality_threshold" not in config["agents"]["critique"]
    assert json.dumps(config).count("grounding") >= 1
