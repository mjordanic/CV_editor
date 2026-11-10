"""Offline tests for the portfolio reader and its vector index (no API keys, no paid calls)."""
import logging

from langchain_core.embeddings import DeterministicFakeEmbedding

import agents.rag_agent as rag_agent


def _make_agent(monkeypatch, portfolio_path, **kwargs):
    """Build the agent with fake embeddings and no chat model, so nothing calls OpenAI."""
    monkeypatch.setattr(rag_agent, "OpenAIEmbeddings", lambda **_: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(rag_agent, "init_chat_model", lambda *_, **__: None)
    return rag_agent.ExperienceRetrievalAgent(portfolio_path=str(portfolio_path), **kwargs)


def test_faiss_builds_index_from_a_large_portfolio(monkeypatch, tmp_path):
    portfolio = tmp_path / "portfolio.txt"
    portfolio.write_text("Built a data pipeline in Python.\n\nLed a team of five engineers.\n", encoding="utf-8")

    agent = _make_agent(monkeypatch, portfolio, full_text_limit=10)  # "large": over the limit
    assert agent.vectorstore is None  # nothing is indexed (no paid call) before the session needs it
    agent._index_portfolio()

    assert agent.vectorstore is not None
    assert agent.vectorstore.index.ntotal == 2
    hits = agent.vectorstore.similarity_search("Led a team of five engineers.", k=1)
    assert hits[0].page_content == "Led a team of five engineers."


def test_index_failure_is_logged_as_warning(monkeypatch, tmp_path, caplog):
    portfolio = tmp_path / "portfolio.txt"
    portfolio.write_text("One portfolio item.\n", encoding="utf-8")

    def broken_from_documents(*_, **__):
        raise ImportError("Could not import faiss python package.")

    monkeypatch.setattr(rag_agent.FAISS, "from_documents", broken_from_documents)

    with caplog.at_level(logging.WARNING, logger=rag_agent.__name__):
        agent = _make_agent(monkeypatch, portfolio, full_text_limit=1)
        agent._index_portfolio()

    assert agent.vectorstore is None
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "ImportError" in warning.getMessage()
    assert "RAG is disabled" in warning.getMessage()


def test_without_a_portfolio_nothing_is_read_or_given_to_the_writers(monkeypatch, tmp_path, caplog):
    """The repo ships no portfolio: a user who supplied none gets none (not a sample's projects)."""
    with caplog.at_level(logging.INFO, logger=rag_agent.__name__):
        agent = _make_agent(monkeypatch, tmp_path / "data" / "portfolio" / "portfolio.txt")
        out = agent.run({"job_description_info": {"job_description": "Butcher wanted."}})
    assert out["portfolio_text"] == "" and out["relevant_experience"] == ""
    assert "No portfolio" in caplog.text


def test_the_default_portfolio_path_is_the_users_own_untracked_file():
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    assert rag_agent.DEFAULT_PORTFOLIO_PATH == "data/portfolio/portfolio.txt"
    tracked = subprocess.run(["git", "ls-files", "data"], cwd=repo, capture_output=True, text=True).stdout
    assert tracked.strip() == "", "a portfolio (or any data/ file) is tracked: it would count as every user's evidence"
    ignored = subprocess.run(["git", "check-ignore", "-q", "data/portfolio/portfolio.txt"], cwd=repo)
    assert ignored.returncode == 0, "data/portfolio/portfolio.txt must stay git-ignored"


def test_a_small_portfolio_goes_to_the_writers_whole_without_paid_calls(monkeypatch, tmp_path):
    portfolio = tmp_path / "portfolio.txt"
    text = "PROJECT: Smokehouse\n- Ran the smoking room.\n\nPROJECT: Catering\n- Cut meat for 200 guests."
    portfolio.write_text(text, encoding="utf-8")

    def no_embeddings(*_, **__):
        raise AssertionError("a small portfolio must not be embedded")

    monkeypatch.setattr(rag_agent, "_embeddings_for", no_embeddings)
    agent = _make_agent(monkeypatch, portfolio)
    agent._verify_relevance = no_embeddings
    out = agent.run({"job_description_info": {"job_description": "Butcher wanted."}})
    assert out["relevant_experience"] == text and out["portfolio_text"] == text


def test_relevance_answer_must_start_with_yes(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path / "none.txt")
    for answer, expected in (("YES", True), ("Yes, relevant.", True), ("NO. YES would overstate it.", False), ("no", False)):
        agent.llm = type("L", (), {"invoke": lambda self, _p, a=answer: type("M", (), {"content": a})()})()
        assert agent._verify_relevance("chunk", "ad") is expected, answer
