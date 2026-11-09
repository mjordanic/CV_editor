"""Offline tests for router helpers that keep the refinement cap from being bypassed."""
from langgraph.graph.message import add_messages

from agents.router import RouterAgent, _message_role, _offered_actions


def test_message_role_handles_dicts_and_langchain_messages():
    msgs = add_messages([], [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert [_message_role(m) for m in msgs] == ["user", "assistant"]
    assert _message_role({"role": "user", "content": "x"}) == "user"


def test_has_user_feedback_sees_langchain_human_message():
    # Graph state holds LangChain message objects (add_messages converts dicts), which have no .role.
    agent = RouterAgent.__new__(RouterAgent)
    user_last = add_messages([], [{"role": "assistant", "content": "draft ready"}, {"role": "user", "content": "add X"}])
    assistant_last = add_messages([], [{"role": "user", "content": "go"}, {"role": "assistant", "content": "critique"}])
    assert agent._has_user_feedback(user_last) is True
    assert agent._has_user_feedback(assistant_last) is False


def test_offered_actions_parses_available_actions_block():
    agent = RouterAgent.__new__(RouterAgent)
    no_user_feedback = add_messages([], [{"role": "assistant", "content": "CV critique: ..."}])
    block = agent._build_available_actions(
        cv_refinement_allowed=False,
        cover_letter_refinement_allowed=False,
        cv_generated=True,
        cover_letter_generated=False,
        messages=no_user_feedback,
        cv_pdf_available=True,
        cover_letter_pdf_available=False,
    )
    offered = _offered_actions(block)
    # Refinement limit reached and no new user message: redrafting the CV must not be offered.
    assert "draft_cv" not in offered
    assert {"collect_user_input", "exit", "draft_cover_letter"} <= offered
