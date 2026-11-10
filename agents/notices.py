"""Fixed, non-LLM notices that the user must see (e.g. claims still unsupported in a final draft).

Notices are kept in state['notices_for_user'] by source, e.g. {"cv_claims": text, "cv_pdf": text}.
The node that owns a source sets or clears it every time it runs, so a warning about an earlier
draft cannot outlive that draft: finalize sets or clears '<kind>_claims' (and clears '<kind>_pdf',
because a new PDF follows), and the PDF node sets or clears '<kind>_pdf'. The router puts every
pending notice in front of its next message to the user, word for word, and clears them all. The
router model writes the rest of that message, but it cannot drop or soften a notice.
"""
from __future__ import annotations

from typing import Optional

NOTICES_KEY = "notices_for_user"


def set_notices(state, updates: dict, changes: dict[str, Optional[str]]) -> dict[str, str]:
    """
    The pending notices with each key in `changes` set to its text, or removed when the text is
    empty (for a node's state update under NOTICES_KEY). `updates` is the node's update so far:
    notices it already changed in this step count as pending.
    """
    pending = updates.get(NOTICES_KEY) if NOTICES_KEY in updates else state.get(NOTICES_KEY)
    pending = dict(pending or {})
    for key, text in changes.items():
        if text:
            pending[key] = text
        else:
            pending.pop(key, None)
    return pending


def pending_notices(state) -> list[str]:
    """The texts of every pending notice, in the order they were set."""
    return [text for text in (state.get(NOTICES_KEY) or {}).values() if text]


def with_notices(state, message: str) -> str:
    """`message` with every pending notice in front of it."""
    pending = pending_notices(state)
    if not pending:
        return message
    return "\n\n".join(pending + ([message] if message else []))
