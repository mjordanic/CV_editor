"""Fixed, non-LLM notices that the user must see (e.g. claims still unsupported in a final draft).

A node adds a notice to state['notices_for_user']; the router puts every pending notice in front
of its next message to the user, word for word, and clears the list. The router model writes the
rest of that message, but it cannot drop or soften a notice.
"""
from __future__ import annotations

NOTICES_KEY = "notices_for_user"


def add_notice(state, text: str) -> list[str]:
    """The pending notices plus `text` (for a node's state update under NOTICES_KEY)."""
    pending = list(state.get(NOTICES_KEY) or [])
    if text and text not in pending:
        pending.append(text)
    return pending


def with_notices(state, message: str) -> str:
    """`message` with every pending notice in front of it."""
    pending = [n for n in (state.get(NOTICES_KEY) or []) if n]
    if not pending:
        return message
    return "\n\n".join(pending + ([message] if message else []))
