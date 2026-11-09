"""Offline tests for edge-case inputs found by scripted stress sessions (no API calls).

Each case was seen in a real session on gpt-6-luna with invented candidates and job ads:
- an empty job ad or a wrong file path was accepted, and the session went on tailoring to nothing;
- with no CV on disk the writer built a CV full of "[YOUR NAME]" placeholders;
- an ad with no job title and no requirements ("We're hiring! Apply now.") was used without a word;
- an empty user message cost a router call that only repeated the question;
(A German CV that failed to become a PDF, and the missing retry after a failed PDF, were found here
too; their fixes and tests came with PR 12, see tests/test_trade_scenarios.py.)
"""

import pytest
from langgraph.types import Command

from agents.document_reader import looks_like_path
from agents.job_description import job_ad_lacks_details
from test_session_replay import REPO, ScriptedLLM, _build_agent, _route

AD = (REPO / "job_description" / "sample_job_ad.txt").read_text(encoding="utf-8")
CV = (REPO / "examples" / "sample_CV.txt").read_text(encoding="utf-8")


def _start(agent):
    config = {"configurable": {"thread_id": "edge"}}
    result = agent.graph.invoke({"messages": [], "cv_history": [], "cover_letter_history": []}, config)
    return config, result


def _say(agent, config, reply):
    return agent.graph.invoke(Command(resume=reply), config)


def _question(result) -> str:
    return result["__interrupt__"][0].value["message"]


def test_empty_job_ad_and_wrong_path_are_asked_again(monkeypatch, tmp_path):
    llm = ScriptedLLM(router_script=[_route("exit", "Bye.")], critique_scores=[])
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)
    ad_file = tmp_path / "ad.txt"
    ad_file.write_text(AD, encoding="utf-8")
    config, result = _start(agent)

    result = _say(agent, config, "   ")
    assert "did not receive anything" in _question(result)
    result = _say(agent, config, "/no/such/folder/job_ad.txt")
    assert "could not find the file" in _question(result)
    result = _say(agent, config, str(ad_file))

    assert "__interrupt__" not in result  # the router chose exit
    assert result["job_description_info"]["job_description"] == AD.strip()


def test_missing_cv_is_asked_for_before_the_job_ad(monkeypatch, tmp_path):
    llm = ScriptedLLM(router_script=[_route("exit", "Bye.")], critique_scores=[])
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)
    (tmp_path / "data" / "CV" / "CV.txt").unlink()
    config, result = _start(agent)

    assert "could not find your CV" in _question(result)
    result = _say(agent, config, "")
    assert "did not receive anything" in _question(result)
    result = _say(agent, config, CV)
    assert "job description" in _question(result)
    result = _say(agent, config, AD)

    assert result["candidate_text"]["cv"] == CV.strip()


def test_empty_cv_file_counts_as_missing(monkeypatch, tmp_path):
    llm = ScriptedLLM(router_script=[_route("exit", "Bye.")], critique_scores=[])
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)
    (tmp_path / "data" / "CV" / "CV.txt").write_text("\n  \n", encoding="utf-8")
    _, result = _start(agent)

    assert "could not find your CV" in _question(result)


class ThinAdLLM(ScriptedLLM):
    """Finds no title or requirements in the first ad text, and a title once details are added."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extracted_from = []

    def _answer(self, name, schema, prompt):
        if name == "JobDescriptionInfo":
            text = prompt.to_string()
            self.extracted_from.append(text)
            if "Pastry chef" in text:
                return schema(job_title="Pastry chef", candidate_minimal_requirements="2 years in a bakery")
            return schema(company_name=None, job_title="Not specified", candidate_minimal_requirements=None)
        return super()._answer(name, schema, prompt)


def test_ad_without_title_or_requirements_gets_one_follow_up(monkeypatch, tmp_path):
    llm = ThinAdLLM(router_script=[_route("exit", "Bye.")], critique_scores=[])
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)
    config, result = _start(agent)

    result = _say(agent, config, "We're hiring! Apply now.")
    assert "very little detail" in _question(result)
    result = _say(agent, config, "Pastry chef, 2 years in a bakery needed.")

    info = result["job_description_info"]
    assert info["job_title"] == "Pastry chef"
    assert info["job_description"] == "We're hiring! Apply now.\n\nPastry chef, 2 years in a bakery needed."
    assert len(llm.extracted_from) == 2


def test_follow_up_is_asked_only_once(monkeypatch, tmp_path):
    llm = ThinAdLLM(router_script=[_route("exit", "Bye.")], critique_scores=[])
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)
    config, result = _start(agent)

    result = _say(agent, config, "We're hiring! Apply now.")
    result = _say(agent, config, "continue")

    assert "__interrupt__" not in result  # went on to the router (exit), no second follow-up
    assert result["job_ad_clarification_asked"] is True


@pytest.mark.parametrize("info, thin", [
    ({"job_title": None, "candidate_minimal_requirements": None}, True),
    ({"job_title": "unknown", "candidate_minimal_requirements": "Not specified."}, True),
    ({"job_title": "Butcher", "candidate_minimal_requirements": None}, False),
    ({"job_title": None, "candidate_minimal_requirements": "Forklift licence"}, False),
])
def test_job_ad_lacks_details(info, thin):
    assert job_ad_lacks_details(info) is thin


def test_empty_user_message_is_asked_again_without_a_router_call(monkeypatch, tmp_path):
    llm = ScriptedLLM(
        router_script=[_route("collect_user_input", "CV or cover letter?"), _route("exit", "Bye.")],
        critique_scores=[],
    )
    agent, _ = _build_agent(monkeypatch, tmp_path, llm)
    config, _ = _start(agent)
    _say(agent, config, AD)

    result = _say(agent, config, "  ")

    assert _question(result) == "I did not receive a message. CV or cover letter?"
    assert len(llm.router_script) == 1  # the router was not called for the empty reply
    result = _say(agent, config, "exit")
    assert "__interrupt__" not in result


@pytest.mark.parametrize("text, is_path", [
    ("/Users/someone/job ads/ad.txt", True),
    ("~/ad.pdf", True),
    ("ad.txt", True),
    ("C:\\jobs\\ad.txt", True),
    ("We're hiring! Apply now.", False),
    ("Barista wanted, send your CV as cv.pdf", False),
    ("Line one\n/also text", False),
])
def test_looks_like_path(text, is_path):
    assert looks_like_path(text) is is_path
