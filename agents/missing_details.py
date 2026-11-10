"""Ask the user, once per document, for details the writer had to leave as placeholders ([EMAIL]).

Without a CV the writer builds one from notes and uses placeholders such as [EMAIL] for missing
contact details; these reached the PDF. After finalize, a document that still has bracketed
all-caps placeholders goes here before its PDF is made. Whatever the user replies becomes a
request for that document (fill in what they gave, leave out the rest), so this works in any
language without a keyword to type; an empty reply makes the PDF as it is. Asked only once.
"""
from __future__ import annotations

import logging

from langgraph.types import interrupt

logger = logging.getLogger(__name__)

_LABEL = {"cv": "CV", "cover_letter": "cover letter"}


def needs_missing_details(kind: str, state) -> bool:
    return bool(state.get(f"{kind}_placeholders")) and not state.get(f"{kind}_placeholders_asked")


def make_ask_missing_details(kind: str):
    """The graph node that asks about the placeholders of 'cv' or 'cover_letter'."""
    label = _LABEL[kind]

    def ask_missing_details(state):
        from agents.router import _new_draft_cycle

        placeholders = list(state.get(f"{kind}_placeholders") or [])
        listed = ", ".join(placeholders)
        question = (
            f"The {label} still has placeholders for details I do not have: {listed}. "
            f"Reply with the details to fill in, or tell me which ones to leave out. "
            f"An empty reply makes the PDF as it is."
        )
        reply = str(interrupt({"message": question, "required": False}) or "").strip()
        messages = list(state.get("messages", [])) + [{"role": "assistant", "content": question}]
        update = {f"{kind}_placeholders_asked": True, "current_node": f"ask_{kind}_missing_details"}
        if reply:
            messages.append({"role": "user", "content": reply})
            update[f"{kind}_content_feedback"] = (
                f"The {label} has placeholders for missing details: {listed}. The user replied: \"{reply}\". "
                f"Replace each placeholder with the detail the user gave for it, and remove cleanly every "
                f"placeholder the user gave no detail for or asked to leave out."
            )
            update.update(_new_draft_cycle(kind, state))
            logger.info(f"The user answered the placeholder question for the {label} - redrafting it")
        else:
            logger.info(f"No details for the {label} placeholders - making the PDF as it is")
        update["messages"] = messages
        return update

    ask_missing_details.__name__ = f"ask_{kind}_missing_details"
    return ask_missing_details


def route_after_finalize(kind: str):
    """finalize_<kind> -> ask_<kind>_missing_details (once, if placeholders remain) or generate_<kind>_pdf."""
    def route(state):
        return f"ask_{kind}_missing_details" if needs_missing_details(kind, state) else f"generate_{kind}_pdf"
    return route


def route_after_ask(kind: str):
    """After the question: redraft with the user's reply, or make the PDF as it is."""
    def route(state):
        return f"draft_{kind}" if (state.get(f"{kind}_content_feedback") or "").strip() else f"generate_{kind}_pdf"
    return route
