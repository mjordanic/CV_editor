"""Offline tests for the console front end (langgraph_agent.py) and the startup checks: how a reply is read, how a
session ends at a prompt, which log lines reach the console, the API-key check and file paths typed by the user."""
import logging
from types import SimpleNamespace

import pytest

import langgraph_agent as app
from agents import document_reader
from config import config_loader


def _feed(monkeypatch, lines, tty=False, waiting=False):
    """Make input() return `lines` one by one, then raise EOFError; stdin looks like a terminal if `tty`."""
    queue = list(lines)

    def fake_input():
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(app.sys, "stdin", SimpleNamespace(isatty=lambda: tty))
    monkeypatch.setattr(app, "_more_input_waiting", lambda timeout=0.05: waiting)
    return queue


AD = ["Job: Data Engineer", "", "", "Requirements: Python, SQL", "END", "next answer"]


def test_piped_text_with_a_double_blank_line_is_read_to_the_end_marker(monkeypatch):
    rest = _feed(monkeypatch, AD, tty=False)
    assert app.read_multiline_input() == "Job: Data Engineer\n\n\nRequirements: Python, SQL"
    assert rest == ["next answer"]


def test_pasted_text_with_more_lines_waiting_keeps_its_double_blank_line(monkeypatch):
    _feed(monkeypatch, AD, tty=True, waiting=True)
    assert app.read_multiline_input().endswith("Requirements: Python, SQL")


def test_enter_twice_at_a_terminal_ends_the_reply(monkeypatch):
    rest = _feed(monkeypatch, AD, tty=True, waiting=False)
    assert app.read_multiline_input() == "Job: Data Engineer"
    assert rest[0] == "Requirements: Python, SQL"


def test_end_of_input_ends_a_reply_with_text_and_raises_without_text(monkeypatch):
    _feed(monkeypatch, ["only line"])
    assert app.read_multiline_input() == "only line"
    with pytest.raises(EOFError):
        app.read_multiline_input()


def test_exit_before_the_chat_ends_the_session_and_in_the_chat_reaches_the_router(monkeypatch):
    _feed(monkeypatch, ["exit", "END", "Exit", "END"])
    with pytest.raises(app.SessionEnded):
        app.ask_user({"message": "Please paste the job description"})
    assert app.ask_user({"message": "Anything else?", "chat": True}) == "Exit"


class _AskingGraph:
    """A compiled graph stand-in that keeps asking the same question and counts the answers it gets."""

    def __init__(self):
        self.answers = []

    def invoke(self, value, config=None):
        if isinstance(value, app.Command):
            self.answers.append(value.resume)
        return {"__interrupt__": [SimpleNamespace(value={"message": "Please paste your CV.", "required": True})]}


@pytest.mark.parametrize("lines, answered", [([], 0), (["exit", "END"], 0), (["my CV", "END"], 1)])
def test_a_session_stops_on_end_of_input_or_exit_instead_of_asking_forever(monkeypatch, caplog, lines, answered):
    _feed(monkeypatch, lines)
    agent = app.MasterAgent.__new__(app.MasterAgent)
    agent.graph = _AskingGraph()
    with caplog.at_level(logging.INFO, logger=app.__name__):
        agent.run()
    assert len(agent.graph.answers) == answered
    assert "Session ended" in caplog.text


def test_ctrl_c_at_a_prompt_ends_the_session(monkeypatch, caplog):
    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)
    agent = app.MasterAgent.__new__(app.MasterAgent)
    agent.graph = _AskingGraph()
    with caplog.at_level(logging.INFO, logger=app.__name__):
        agent.run()
    assert "Session ended (Ctrl+C)" in caplog.text


def test_console_shows_the_programs_lines_and_only_warnings_from_libraries():
    def record(name, level):
        return logging.LogRecord(name, level, __file__, 1, "x", None, None)

    assert app._console_filter(record("agents.router", logging.DEBUG))
    assert app._console_filter(record("__main__", logging.INFO))
    assert not app._console_filter(record("pdfminer.psparser", logging.DEBUG))
    assert not app._console_filter(record("langchain_core", logging.INFO))
    assert app._console_filter(record("pdfminer.pdfpage", logging.WARNING))


def _no_keys(monkeypatch):
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CV_EDITOR_MODEL", "CV_EDITOR_CRITIC_MODEL",
                "CV_EDITOR_GROUNDING_MODEL", "CV_EDITOR_TEMPERATURE"):
        monkeypatch.delenv(env, raising=False)


def test_missing_api_keys_names_the_key_the_configured_models_need(monkeypatch):
    _no_keys(monkeypatch)
    assert config_loader.missing_api_keys() == ["OPENAI_API_KEY"]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert config_loader.missing_api_keys() == []


def test_a_local_model_needs_no_api_key(monkeypatch):
    _no_keys(monkeypatch)
    monkeypatch.setenv("CV_EDITOR_MODEL", "ollama:gemma4:e4b")
    assert config_loader.missing_api_keys() == []
    monkeypatch.setenv("CV_EDITOR_CRITIC_MODEL", "anthropic:claude-sonnet-5-5")
    assert config_loader.missing_api_keys() == ["ANTHROPIC_API_KEY"]


def test_a_malformed_temperature_stops_with_a_clear_message(monkeypatch):
    _no_keys(monkeypatch)
    monkeypatch.setenv("CV_EDITOR_MODEL", "openai:gpt-6-luna")
    monkeypatch.setenv("CV_EDITOR_TEMPERATURE", "0,2")
    with pytest.raises(config_loader.ConfigError, match="CV_EDITOR_TEMPERATURE"):
        config_loader.get_agent_config("cv_writer")
    monkeypatch.setenv("CV_EDITOR_TEMPERATURE", "0.3")
    assert config_loader.get_agent_config("cv_writer")["temperature"] == 0.3


def test_a_relative_path_is_also_found_in_the_folder_the_user_started_in(monkeypatch, tmp_path):
    (tmp_path / "ad.txt").write_text("SAMPLE JOB AD", encoding="utf-8")
    monkeypatch.setattr(document_reader, "_USER_DIRS", [])
    assert document_reader._resolve_user_path("ad.txt") == "ad.txt"
    document_reader.add_user_dir(str(tmp_path))
    assert document_reader._resolve_user_path("'ad.txt'") == str(tmp_path / "ad.txt")


def test_an_earlier_sessions_outputs_are_moved_aside_not_deleted(tmp_path):
    out = tmp_path / "generated_CVs"
    assert app.move_earlier_outputs(str(out)) is None
    (out / "temp_latex").mkdir(parents=True)
    (out / "temp_latex" / "generated_CV_draft_1.tex").write_text("old", encoding="utf-8")
    (out / "generated_CV_draft_3.txt").write_text("old draft 3", encoding="utf-8")
    moved = app.move_earlier_outputs(str(out))
    assert sorted(p.name for p in out.iterdir()) == ["earlier"]
    assert (tmp_path / moved / "generated_CV_draft_3.txt").read_text() == "old draft 3"
    assert (tmp_path / moved / "temp_latex" / "generated_CV_draft_1.tex").is_file()
    assert app.move_earlier_outputs(str(out)) is None  # only "earlier" is left
