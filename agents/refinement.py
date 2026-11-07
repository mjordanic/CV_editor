"""Decide whether a critiqued document goes back for another critique-driven redraft."""


def should_refine(
    needs_refinement: bool,
    refinement_count: int,
    current_score: int | None,
    previous_score: int | None,
    max_refinements: int,
    max_refinements_hard_cap: int,
    quality_improvement_threshold: int,
) -> tuple[bool, bool]:
    """
    Return (refine, quality_improved).

    A document the critique flagged is redrafted while refinement_count < max_refinements.
    Past that budget it is redrafted again only if the last redraft raised the critique score
    by at least quality_improvement_threshold over the draft before it, and never once
    refinement_count reaches max_refinements_hard_cap, so the loop always ends.

    Args:
        needs_refinement: Whether the latest critique asked for changes.
        refinement_count: Critique-driven redrafts already made in this cycle.
        current_score: Critique score of the latest draft.
        previous_score: Critique score of the draft before it (None for the first draft).
        max_refinements: Redrafts allowed regardless of score.
        max_refinements_hard_cap: Absolute limit on redrafts, even while scores keep rising.
        quality_improvement_threshold: Score gain that earns a redraft past max_refinements.
    """
    quality_improved = (
        current_score is not None
        and previous_score is not None
        and current_score - previous_score >= quality_improvement_threshold
    )
    if not needs_refinement or refinement_count >= max(max_refinements, max_refinements_hard_cap):
        return False, quality_improved
    return refinement_count < max_refinements or quality_improved, quality_improved
