"""Offline tests for the critique -> redraft routing decision (no API keys, no LLM calls)."""
import pytest

import agents.critique as critique
from agents.refinement import best_draft, should_refine

BUDGET = dict(max_refinements=1, max_refinements_hard_cap=2)


@pytest.mark.parametrize(
    "needs, count, defects, previous, expected",
    [
        (False, 0, 3, None, (False, False)),  # no defect the writer can fix
        (True, 0, 3, None, (True, False)),  # first draft, within max_refinements
        (True, 1, 3, 3, (False, False)),  # budget used, no progress on the ad's requirements
        (True, 1, 1, 3, (True, True)),  # budget used, the redraft showed more requirements: one more pass
        (True, 1, 4, 3, (False, False)),  # it got worse
        (True, 2, 0, 3, (False, True)),  # hard cap reached even though it keeps improving
    ],
)
def test_should_refine(needs, count, defects, previous, expected):
    assert should_refine(needs, count, defects, previous, **BUDGET) == expected


def test_steady_progress_still_stops_at_hard_cap():
    count, previous, defects = 0, None, 10
    while should_refine(True, count, defects, previous, **BUDGET)[0]:
        count, previous, defects = count + 1, defects, defects - 2
        assert count <= BUDGET["max_refinements_hard_cap"]
    assert count == BUDGET["max_refinements_hard_cap"]


def test_score_alone_no_longer_decides_a_redraft(monkeypatch):
    """A low score without a real defect does not send the draft back; a missing requirement does."""
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    agent = critique.CritiqueAgent.__new__(critique.CritiqueAgent)
    agent.grounding_llm = None  # the check fails and is skipped
    answers = iter([
        critique.CritiqueResponse(quality_score=55, content_quality_feedback="meh", ats_compatibility_feedback="ok",
                                  job_alignment_feedback="ok", improvement_instructions="Polish the wording."),
        critique.CritiqueResponse(
            quality_score=92, content_quality_feedback="ok", ats_compatibility_feedback="ok", job_alignment_feedback="ok",
            improvement_instructions="",
            requirements=[critique.RequirementCheck(requirement="HACCP", status="missing", quote="HACCP course 2021"),
                          critique.RequirementCheck(requirement="Forklift licence", status="gap")],
        ),
    ])
    agent.critique_cv = lambda **_: next(answers)
    low = agent.run_cv({"generated_cv": "draft 1", "cv_history": [], "messages": []})
    assert low["cv_needs_refinement"] is False
    high = agent.run_cv({"generated_cv": "draft 2", "cv_history": [], "messages": []})
    assert high["cv_needs_refinement"] is True
    assert high["cv_critique"]["defects"] == 1
    assert "HACCP" in high["cv_critique_improvement_instructions"]  # a missing requirement becomes an instruction
    assert "Forklift licence" not in high["cv_critique_improvement_instructions"]  # a gap is never asked for
    assert "Forklift licence" in high["messages"][-1]["content"]  # but the user is told about it


@pytest.mark.parametrize("kind", ["cv", "cover_letter"])
def test_critique_node_keeps_previous_draft_defect_count(monkeypatch, kind):
    """The node must store the previous draft's defect count, not overwrite it with the new one."""
    monkeypatch.setattr(critique, "init_chat_model", lambda *_, **__: None)
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    agent = critique.CritiqueAgent()

    missing = iter([3, 1])

    def fake_critique(**_):
        return critique.CritiqueResponse(
            quality_score=70, content_quality_feedback="ok", ats_compatibility_feedback="ok",
            job_alignment_feedback="ok", improvement_instructions="Show the requirements.", critical_issues="",
            requirements=[critique.RequirementCheck(requirement=f"R{i}", status="missing") for i in range(next(missing))],
        )

    method = "critique_cv" if kind == "cv" else "critique_cover_letter"
    monkeypatch.setattr(agent, method, fake_critique)
    run = agent.run_cv if kind == "cv" else agent.run_cover_letter

    doc_key = "generated_cv" if kind == "cv" else "generated_cover_letter"
    state = {doc_key: "draft 1", f"{kind}_history": [], "messages": []}
    first = run(state)
    assert first[f"{kind}_previous_defect_count"] is None

    state = {**state, **first, doc_key: "draft 2"}
    second = run(state)
    assert second[f"{kind}_critique"]["defects"] == 1
    assert second[f"{kind}_previous_defect_count"] == 3
    refine, progress = should_refine(
        second[f"{kind}_needs_refinement"], 1, 1, second[f"{kind}_previous_defect_count"], **BUDGET
    )
    assert (refine, progress) == (True, True)


def test_best_draft_prefers_honesty_then_requirements_then_the_side_by_side_comparison():
    drafts = [
        {"iteration": 1, "score": 95, "unsupported_claims": ["x"], "defects": 0},
        {"iteration": 2, "score": 70, "unsupported_claims": [], "defects": 2},
        {"iteration": 3, "score": 90, "unsupported_claims": [], "defects": 1},
        {"iteration": 4, "score": 60, "unsupported_claims": [], "defects": 1},
    ]
    # no comparison: fewest claims, then fewest defects, then score
    assert best_draft(drafts)["iteration"] == 3
    # the comparison sees the last two of the tied drafts (3 and 4) and decides
    seen = []
    pick_b = lambda a, b: (seen.append((a["iteration"], b["iteration"])), "B")[1]  # noqa: E731
    assert best_draft(drafts, compare=pick_b)["iteration"] == 4
    assert seen == [(3, 4)]
    # a failed comparison falls back to the score
    assert best_draft(drafts, compare=lambda a, b: None)["iteration"] == 3
    # a single best draft needs no comparison
    assert best_draft(drafts[:3], compare=lambda a, b: pytest.fail("compared"))["iteration"] == 3
