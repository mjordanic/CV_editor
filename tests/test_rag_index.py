"""Offline smoke tests for the portfolio vector index (no API keys, no paid calls)."""
import logging

from langchain_core.embeddings import DeterministicFakeEmbedding

import agents.rag_agent as rag_agent


def _make_agent(monkeypatch, portfolio_path):
    """Build the agent with fake embeddings and no chat model, so nothing calls OpenAI."""
    monkeypatch.setattr(rag_agent, "OpenAIEmbeddings", lambda **_: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(rag_agent, "init_chat_model", lambda *_, **__: None)
    return rag_agent.ExperienceRetrievalAgent(portfolio_path=str(portfolio_path))


def test_faiss_builds_index_from_portfolio(monkeypatch, tmp_path):
    portfolio = tmp_path / "portfolio.txt"
    portfolio.write_text("Built a data pipeline in Python.\n\nLed a team of five engineers.\n", encoding="utf-8")

    agent = _make_agent(monkeypatch, portfolio)

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
        agent = _make_agent(monkeypatch, portfolio)

    assert agent.vectorstore is None
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "ImportError" in warning.getMessage()
    assert "RAG is disabled" in warning.getMessage()
