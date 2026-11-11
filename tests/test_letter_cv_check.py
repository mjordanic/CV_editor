"""Offline tests for the cover-letter check that rejects a letter draft repeating the CV (no API calls).

The fixtures are published drafts from the 2026-11-10 evaluation round (evals/results/2026-11-10):
the first letter draft of p11_a1_all4_luna_r1 (English) and p13_a8_cv_only_luna_r1 (German), which
repeat the whole CV before the letter and were kept as the final letters, and the second draft of
the same sessions, a clean letter. Each folder also holds the session's tailored CV and the
candidate's own CV.
"""
from pathlib import Path

import pytest

from agents import critique
from agents.cover_letter_writer import CoverLetterWriterAgent
from agents.letter_cv_check import CV_REPEAT_INSTRUCTION, MAX_CV_SHARE, cv_share, repeats_cv, strip_cv
from agents.notices import NOTICES_KEY, pending_notices
from agents.refinement import best_draft, should_refine
from tests.test_review_fixes import AD_INFO, WriterLLM, _instructions, _writer

FIXTURES = Path(__file__).parent / "fixtures" / "letter_cv"
SESSIONS = ["p11_a1_all4_luna_r1", "p13_a8_cv_only_luna_r1"]


def _fixture(session: str) -> dict:
    folder = FIXTURES / session
    return {name: (folder / f"{name}.txt").read_text(encoding="utf-8")
            for name in ("tailored_cv", "original_cv", "letter_draft_1", "letter_draft_2")}


def _cvs(f: dict) -> list[str]:
    return [f["tailored_cv"], f["original_cv"]]


# --- the check ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("session", SESSIONS)
def test_a_letter_draft_that_repeats_the_cv_is_detected(session):
    f = _fixture(session)
    assert cv_share(f["letter_draft_1"], _cvs(f)) > 0.9
    assert repeats_cv(f["letter_draft_1"], _cvs(f))


@pytest.mark.parametrize("session", SESSIONS)
def test_a_clean_letter_passes(session):
    f = _fixture(session)
    assert cv_share(f["letter_draft_2"], _cvs(f)) < MAX_CV_SHARE / 2
    assert not repeats_cv(f["letter_draft_2"], _cvs(f))


def test_without_a_cv_nothing_is_judged():
    f = _fixture(SESSIONS[0])
    assert not repeats_cv(f["letter_draft_1"], [None, ""])
    assert not repeats_cv(f["letter_draft_1"], ["Python developer, Lisbon"])  # too short to judge


@pytest.mark.parametrize("session,salutation", [
    ("p11_a1_all4_luna_r1", "Dear Halden Pay Data Platform Hiring Team,"),
    ("p13_a8_cv_only_luna_r1", "Sehr geehrte Damen und Herren,"),
])
def test_the_cv_is_cut_out_of_the_draft_up_to_the_salutation(session, salutation):
    f = _fixture(session)
    letter = strip_cv(f["letter_draft_1"], _cvs(f))
    assert letter is not None
    assert letter.splitlines()[0] == salutation
    assert letter == f["letter_draft_1"][f["letter_draft_1"].index(salutation):].strip()
    assert not repeats_cv(letter, _cvs(f))


def test_a_draft_that_is_only_the_cv_cannot_be_cut():
    f = _fixture(SESSIONS[0])
    assert strip_cv(f["tailored_cv"], _cvs(f)) is None


# --- the critic rejects it before any model call -------------------------------------------------------

def _critic(calls: list) -> critique.CritiqueAgent:
    critic = critique.CritiqueAgent.__new__(critique.CritiqueAgent)

    def critique_cover_letter(**_):
        calls.append("critic")
        return critique.CritiqueResponse(quality_score=95, content_quality_feedback="ok",
                                         ats_compatibility_feedback="ok", job_alignment_feedback="ok")

    def check_grounding(*_, **__):
        calls.append("grounding")
        return []

    critic.critique_cover_letter = critique_cover_letter
    critic.check_grounding = check_grounding
    return critic


def _letter_state(f: dict, letter: str, **extra) -> dict:
    return {"generated_cover_letter": letter, "generated_cv": f["tailored_cv"],
            "candidate_text": {"cv": f["original_cv"]}, "cover_letter_history": [], "messages": [],
            "job_description_info": AD_INFO, **extra}


def test_the_critic_rejects_a_letter_that_repeats_the_cv_without_a_model_call(monkeypatch):
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    f, calls = _fixture(SESSIONS[0]), []
    out = _critic(calls).run_cover_letter(_letter_state(f, f["letter_draft_1"]))
    assert calls == []
    (entry,) = out["cover_letter_history"]
    assert entry["repeats_cv"] is True and entry["content"] == f["letter_draft_1"]
    assert out["cover_letter_needs_refinement"] is True
    assert out["cover_letter_critique_improvement_instructions"] == CV_REPEAT_INSTRUCTION
    assert not out["cover_letter_claims_to_remove"]
    assert "repeats your CV" in out["messages"][-1]["content"]


def test_the_critic_reviews_a_clean_letter_as_before(monkeypatch):
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    f, calls = _fixture(SESSIONS[0]), []
    out = _critic(calls).run_cover_letter(_letter_state(f, f["letter_draft_2"]))
    assert calls == ["critic", "grounding"]
    (entry,) = out["cover_letter_history"]
    assert not entry.get("repeats_cv")


def test_a_rejected_letter_is_redrafted_within_the_existing_budget():
    # The rejection sets needs_refinement; the router's usual rule decides (max_refinements 1, hard cap 2).
    assert should_refine(True, 0, None, None, 1, 2) == (True, False)
    assert should_refine(True, 1, None, 2, 1, 2) == (False, False)


# --- the best-draft choice -------------------------------------------------------------------------------

def test_a_letter_that_repeats_the_cv_is_never_chosen_over_one_that_does_not():
    repeat = {"iteration": 1, "score": 95, "unsupported_claims": [], "grounding_checked": True, "defects": 0, "repeats_cv": True}
    flagged = {"iteration": 2, "score": 60, "unsupported_claims": ["Kafka in production"], "defects": 3}
    unchecked = {"iteration": 3, "score": None, "unsupported_claims": [], "grounding_checked": False, "defects": None}
    assert best_draft([repeat, flagged]) is flagged
    assert best_draft([repeat, unchecked]) is unchecked


def test_two_letters_that_both_repeat_the_cv_are_not_compared_by_the_critic():
    asked = []
    drafts = [{"iteration": i, "score": None, "unsupported_claims": [], "grounding_checked": False, "repeats_cv": True}
              for i in (1, 2)]
    assert best_draft(drafts, compare=lambda a, b: asked.append(1) or "A") is drafts[1]
    assert asked == []


# --- the writer and finalize ---------------------------------------------------------------------------

def test_the_redraft_instruction_is_not_filtered_against_the_users_requests(monkeypatch, tmp_path):
    f = _fixture(SESSIONS[0])
    llm = WriterLLM("Dear team,\n\nA letter.", filter_answer="")  # a filter that would drop everything
    writer = _writer(monkeypatch, tmp_path, CoverLetterWriterAgent, llm)
    state = _letter_state(f, f["letter_draft_1"], cover_letter_needs_refinement=True,
                          cover_letter_critique_improvement_instructions=CV_REPEAT_INSTRUCTION,
                          cover_letter_user_requests=["Keep it short."],
                          cover_letter_history=[{"iteration": 1, "content": f["letter_draft_1"], "repeats_cv": True}])
    out = writer.run(state)
    assert llm.filter_prompts == []
    assert CV_REPEAT_INSTRUCTION in _instructions(llm.prompts[-1])
    assert out["cover_letter_refinement_count"] == 1


def _finalize(monkeypatch, tmp_path, f, history, checker_answers):
    checked = []
    answers = list(checker_answers)

    def checker(label, text, material, employer_info=None):
        checked.append(text)
        return answers.pop(0)

    writer = _writer(monkeypatch, tmp_path, CoverLetterWriterAgent, WriterLLM("unused"), grounding_checker=checker)
    out = writer.finalize_best_version(_letter_state(f, history[-1]["content"], cover_letter_history=history))
    return out, checked


def test_when_every_draft_repeats_the_cv_it_is_cut_out_and_checked(monkeypatch, tmp_path):
    f = _fixture(SESSIONS[0])
    history = [{"iteration": 1, "content": f["letter_draft_1"], "score": None, "unsupported_claims": [],
                "grounding_checked": False, "defects": None, "repeats_cv": True}]
    out, checked = _finalize(monkeypatch, tmp_path, f, history, [[]])
    final = out["generated_cover_letter"]
    assert final.startswith("Dear Halden Pay Data Platform Hiring Team,") and not repeats_cv(final, _cvs(f))
    assert checked == [final]  # the cut letter gets the grounding check it never had
    assert out["cover_letter_final_iteration"] == 2
    assert out["cover_letter_history"][-1]["cv_cut_from"] == 1
    (notice,) = pending_notices(out)
    assert "repeated your CV" in notice and "cut out" in notice


def test_when_the_cv_cannot_be_cut_out_the_user_is_warned(monkeypatch, tmp_path):
    f = _fixture(SESSIONS[0])
    only_cv = f["tailored_cv"]
    history = [{"iteration": 1, "content": only_cv, "score": None, "unsupported_claims": [],
                "grounding_checked": False, "defects": None, "repeats_cv": True}]
    out, _ = _finalize(monkeypatch, tmp_path, f, history, [[]])
    assert out["generated_cover_letter"] == only_cv
    (notice,) = pending_notices(out)
    assert notice.startswith("Warning: the final cover letter repeats your CV")


def test_a_clean_redraft_wins_and_clears_an_earlier_cv_warning(monkeypatch, tmp_path):
    f = _fixture(SESSIONS[0])
    history = [
        {"iteration": 1, "content": f["letter_draft_1"], "score": None, "unsupported_claims": [],
         "grounding_checked": False, "defects": None, "repeats_cv": True},
        {"iteration": 2, "content": f["letter_draft_2"], "score": 80, "unsupported_claims": [],
         "grounding_checked": True, "defects": 1},
    ]
    writer = _writer(monkeypatch, tmp_path, CoverLetterWriterAgent, WriterLLM("unused"))
    state = _letter_state(f, f["letter_draft_2"], cover_letter_history=history,
                          **{NOTICES_KEY: {"cover_letter_cv_repeat": "Warning: old"}})
    out = writer.finalize_best_version(state)
    assert out["generated_cover_letter"] == f["letter_draft_2"]
    assert pending_notices(out) == []
