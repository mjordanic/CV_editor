"""Decide whether a critiqued document goes back for another critique-driven redraft."""


def best_draft(drafts: list[dict]) -> dict:
    """
    The draft to keep from one request's drafts (history entries).

    A draft with fewer claims the grounding check could not support wins over a higher-scoring
    one with more: the critic's score does not count invented experience against a draft, so on
    score alone a draft the check flagged beat the redraft that removed the claim. Then the higher
    score wins, and on a tie the later draft.
    """
    return max(drafts, key=lambda d: (-len(d.get("unsupported_claims") or []), d.get("score", 0), d.get("iteration", 0)))


def should_refine(
    needs_refinement: bool,
    refinement_count: int,
    current_score: int | None,
    previous_score: int | None,
    max_refinements: int,
    max_refinements_hard_cap: int,
    quality_improvement_threshold: int,
    has_unsupported_claims: bool = False,
) -> tuple[bool, bool]:
    """
    Return (refine, quality_improved).

    A document the critique flagged is redrafted while refinement_count < max_refinements.
    Past that budget it is redrafted again only if the last redraft raised the critique score
    by at least quality_improvement_threshold over the draft before it, or if the latest draft
    still makes claims the candidate's material does not support, and never once
    refinement_count reaches max_refinements_hard_cap, so the loop always ends.
    max_refinements 0 turns redrafting off.

    Args:
        needs_refinement: Whether the latest critique asked for changes.
        refinement_count: Critique-driven redrafts already made in this cycle.
        current_score: Critique score of the latest draft.
        previous_score: Critique score of the draft before it (None for the first draft).
        max_refinements: Redrafts allowed regardless of score.
        max_refinements_hard_cap: Absolute limit on redrafts, even while scores keep rising.
        quality_improvement_threshold: Score gain that earns a redraft past max_refinements.
        has_unsupported_claims: Whether the grounding check found unsupported claims in the latest
            draft; such a draft also earns a redraft past max_refinements (within the hard cap).
    """
    quality_improved = (
        current_score is not None
        and previous_score is not None
        and current_score - previous_score >= quality_improvement_threshold
    )
    if not needs_refinement or max_refinements <= 0 or refinement_count >= max(max_refinements, max_refinements_hard_cap):
        return False, quality_improved
    return refinement_count < max_refinements or quality_improved or has_unsupported_claims, quality_improved
