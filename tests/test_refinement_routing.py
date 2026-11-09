"""Offline tests for the critique -> redraft routing decision (no API keys, no LLM calls)."""
import pytest

import agents.critique as critique
from agents.refinement import should_refine

BUDGET = dict(max_refinements=1, max_refinements_hard_cap=2, quality_improvement_threshold=5)


@pytest.mark.parametrize(
    "needs, count, current, previous, expected",
    [
        (False, 0, 60, None, (False, False)),  # critique is happy
        (True, 0, 60, None, (True, False)),  # first draft, within max_refinements
        (True, 1, 62, 60, (False, False)),  # budget used, score rose by less than the threshold
        (True, 1, 70, 60, (True, True)),  # budget used, score rose by >= threshold: one more pass
        (True, 1, 55, 60, (False, False)),  # score fell
        (True, 2, 90, 70, (False, True)),  # hard cap reached even though the score keeps rising
    ],
)
def test_should_refine(needs, count, current, previous, expected):
    assert should_refine(needs, count, current, previous, **BUDGET) == expected


def test_rising_scores_still_stop_at_hard_cap():
    count, previous, score = 0, None, 40
    while should_refine(True, count, score, previous, **BUDGET)[0]:
        count, previous, score = count + 1, score, score + 20
        assert count <= BUDGET["max_refinements_hard_cap"]
    assert count == BUDGET["max_refinements_hard_cap"]


@pytest.mark.parametrize("kind", ["cv", "cover_letter"])
def test_critique_node_keeps_previous_draft_score(monkeypatch, kind):
    """The node must store the previous draft's score, not overwrite it with the new one."""
    monkeypatch.setattr(critique, "init_chat_model", lambda *_, **__: None)
    monkeypatch.setattr(critique, "write_to_debug", lambda *_, **__: None)
    agent = critique.CritiqueAgent(quality_threshold=85)

    scores = iter([60, 70])

    def fake_critique(**_):
        return critique.CritiqueResponse(
            quality_score=next(scores),
            content_quality_feedback="ok",
            ats_compatibility_feedback="ok",
            job_alignment_feedback="ok",
            improvement_instructions="Tighten the summary.",
            critical_issues="",
        )

    method = "critique_cv" if kind == "cv" else "critique_cover_letter"
    monkeypatch.setattr(agent, method, fake_critique)
    run = agent.run_cv if kind == "cv" else agent.run_cover_letter

    doc_key = "generated_cv" if kind == "cv" else "generated_cover_letter"
    state = {doc_key: "draft 1", f"{kind}_history": [], "messages": []}
    first = run(state)
    assert first[f"{kind}_previous_quality_score"] is None

    state = {**state, **first, doc_key: "draft 2"}
    second = run(state)
    assert second[f"{kind}_critique"]["quality_score"] == 70
    assert second[f"{kind}_previous_quality_score"] == 60
    refine, improved = should_refine(
        second[f"{kind}_needs_refinement"], 1, 70, second[f"{kind}_previous_quality_score"], **BUDGET
    )
    assert (refine, improved) == (True, True)

