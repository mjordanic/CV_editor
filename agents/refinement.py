"""Decide whether a critiqued document goes back for another redraft, and which draft is kept."""
from typing import Callable, Optional


def _honesty_key(draft: dict) -> tuple:
    """
    Checked drafts first (a draft whose grounding check could not run has no known claims, which
    is not the same as none), then fewer unsupported claims, then fewer requirements left unshown.
    """
    checked = draft.get("grounding_checked") is not False
    return (checked, -len(draft.get("unsupported_claims") or []), -(draft.get("defects") or 0))


def best_draft(drafts: list[dict], compare: Optional[Callable[[dict, dict], Optional[str]]] = None) -> dict:
    """
    The draft to keep from one request's drafts (history entries).

    A draft with fewer claims the grounding check could not support wins over one with more: the
    critic's score does not count invented experience against a draft, so on score alone a draft
    the check flagged beat the redraft that removed the claim. Next, fewer requirements of the ad
    that the candidate meets but the draft does not show (the critic's requirements check).

    Among drafts still tied, compare(a, b) (the critic reading the last two side by side) decides
    when given; it returns 'A', 'B' or None. Two separate 0-100 scores from a light model are
    noisy, so they decide only without a comparison, and on a tie the later draft wins.
    """
    best_key = max(_honesty_key(d) for d in drafts)
    tied = sorted((d for d in drafts if _honesty_key(d) == best_key), key=lambda d: d.get("iteration", 0))
    if compare is not None and len(tied) >= 2:
        a, b = tied[-2], tied[-1]
        verdict = compare(a, b)
        if verdict in ("A", "B"):
            return a if verdict == "A" else b
    return max(tied, key=lambda d: (d.get("score", 0), d.get("iteration", 0)))


def should_refine(
    needs_refinement: bool,
    refinement_count: int,
    defects: int | None,
    previous_defects: int | None,
    max_refinements: int,
    max_refinements_hard_cap: int,
    has_unsupported_claims: bool = False,
) -> tuple[bool, bool]:
    """
    Return (refine, made_progress).

    A document the critique flagged (a real defect: an unsupported claim, a critical issue, or a
    requirement the candidate meets but the draft does not show) is redrafted while
    refinement_count < max_refinements. Past that budget it is redrafted again only if the latest
    draft still makes claims the candidate's material does not support, or if the last redraft
    made progress (fewer requirements left unshown than the draft before it), and never once
    refinement_count reaches max_refinements_hard_cap, so the loop always ends.
    max_refinements 0 turns redrafting off.

    Args:
        needs_refinement: Whether the latest critique found a defect the writer can fix.
        refinement_count: Critique-driven redrafts already made in this cycle.
        defects: Requirements the latest draft fails to show (None if unknown).
        previous_defects: The same count for the draft before it (None for the first draft).
        max_refinements: Redrafts allowed while the draft has a defect.
        max_refinements_hard_cap: Absolute limit on redrafts.
        has_unsupported_claims: Whether the grounding check found unsupported claims in the latest draft.
    """
    made_progress = defects is not None and previous_defects is not None and defects < previous_defects
    if not needs_refinement or max_refinements <= 0 or refinement_count >= max(max_refinements, max_refinements_hard_cap):
        return False, made_progress
    return refinement_count < max_refinements or has_unsupported_claims or made_progress, made_progress
