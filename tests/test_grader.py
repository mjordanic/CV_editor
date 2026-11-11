"""The provider-agnostic grader (evals/grader.py), document-only grading and the trap check on Claude.

No model is called: the Anthropic client is a stub that returns real SDK Message objects.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_eval_diversity import CLEAN_PAGE, _evaluation, make_pdf


# --- stubs --------------------------------------------------------------------------------------------
def _compact(ev: dict, **overrides) -> dict:
    """The CompactEvaluation a Claude grader answers for the full evaluation `ev`: the full copy as a JSON string
    and the guaranteed fields copied from it (`overrides` replaces some)."""
    from evals.evaluate import GUARANTEED_PATHS
    answer = {"full_evaluation_json": json.dumps(ev)}
    for name, path in GUARANTEED_PATHS.items():
        node = ev
        for key in path:
            node = node[key]
        answer[name] = node
    return {**answer, **overrides}


def _message(answer, *, model="claude-sonnet-5-5", stop_reason="end_turn", usage=None, stop_details=None):
    """A Claude reply; a full evaluation dict is sent as the CompactEvaluation answer a Claude grader gives."""
    from anthropic.types import Message
    if isinstance(answer, dict) and "fit" in answer:
        answer = _compact(answer)
    text = answer if isinstance(answer, str) else json.dumps(answer)
    return Message.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details,
        "usage": usage or {"input_tokens": 1000, "output_tokens": 10000, "cache_creation_input_tokens": 5000,
                           "cache_read_input_tokens": 0},
    })


class _Stream:
    def __init__(self, message):
        self.message, self.response = message, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class _Client:
    """Stands in for anthropic.Anthropic(): each stream() call returns the next message (or raises it)."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        client = self

        class _Messages:
            def stream(self, **params):
                client.calls.append(params)
                reply = client.replies.pop(0)
                if isinstance(reply, BaseException):
                    raise reply
                return _Stream(reply)
        self.messages = _Messages()


def _credit_error():
    import anthropic
    import httpx2
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.example/v1/messages"))
    body = {"type": "error", "error": {"type": "invalid_request_error",
                                       "message": "Your credit balance is too low to access the Anthropic API."}}
    return anthropic.BadRequestError(body["error"]["message"], response=response, body=body)


# --- run folders: one case, made by two methods ----------------------------------------------------------
def _case_run(root: Path, method: str, cv="Jane Doe - Backend developer, Python.", letter="Dear team, ...") -> Path:
    """A run folder of the same case for a multi-agent ("ma") or single-prompt ("sp") session: identical
    inputs and (by default) identical outputs; only the method's own files differ."""
    from evals.run_case import DEFAULT_SCRIPT
    run = root / f"{method}_p03_a1_r1"
    (run / "inputs").mkdir(parents=True)
    (run / "outputs").mkdir()
    (run / "inputs" / "job_ad.txt").write_text("Backend engineer, Python and Kubernetes. Hybrid, Zagreb.")
    (run / "inputs" / "cv.txt").write_text("Jane Doe. Backend developer, five years of Python.")
    (run / "inputs" / "notes.txt").write_text("Available in two months.")
    (run / "inputs" / "script.json").write_text(json.dumps(DEFAULT_SCRIPT))
    (run / "outputs" / "final_cv.txt").write_text(cv)
    (run / "outputs" / "final_cover_letter.txt").write_text(letter)
    make_pdf(run / "outputs" / "final_cv.pdf", [CLEAN_PAGE])
    make_pdf(run / "outputs" / "final_cover_letter.pdf", [[(72, 72, 11, "Dear team")]])
    if method == "ma":
        turns = [{"turn": 0, "assistant": "Here is your CV draft from the multi_agent app (gpt-6-luna).",
                  "user": DEFAULT_SCRIPT["first"]},
                 {"turn": 1, "assistant": "Shall I add your [EMAIL]?", "user": DEFAULT_SCRIPT["missing_details"]},
                 {"turn": 2, "assistant": "Here is the letter.", "user": DEFAULT_SCRIPT["exit"]}]
        manifest = {"status": "completed", "method": "multi_agent", "model": "gpt-6-luna",
                    "labels": {"persona": "p03", "arm": "MA-luna"}}
        (run / "llm_calls.jsonl").write_text(json.dumps({"node": "cv_writer", "model": "gpt-6-luna"}) + "\n")
    else:
        turns = []
        manifest = {"status": "completed", "method": "single_prompt", "model": "gpt-6.1-sol",
                    "labels": {"persona": "p03", "arm": "SP-sol"}}
        (run / "writer_notes.txt").write_text("single_prompt notes from gpt-6.1-sol")
    (run / "transcript.json").write_text(json.dumps({"turns": turns, "observations": [], "requests": [],
                                                     **({"user_instructions": "x"} if method == "sp" else {})}))
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


LEAKS = ("multi_agent", "single_prompt", "single prompt", "gpt-6-luna", "gpt-6.1-sol", "MA-luna", "SP-sol",
         "TOOL:", "no conversation", "Here is your CV draft", "Shall I add", "writer_notes", "_p03_a1_r1")


def _request(run: Path, tmp_path: Path, mode: str, grader="anthropic:claude-sonnet-5-5"):
    from evals.evaluate import build_grading_request
    request, _, _ = build_grading_request(run, grader, tmp_path / "graded" / run.name / mode, mode)
    return request


# --- method blindness ------------------------------------------------------------------------------------
def test_document_only_requests_are_identical_for_a_multi_agent_and_a_single_prompt_run(tmp_path):
    ma, sp = _case_run(tmp_path / "a", "ma"), _case_run(tmp_path / "b", "sp")
    a, b = _request(ma, tmp_path, "document-only"), _request(sp, tmp_path, "document-only")
    # same documents -> byte-identical request, images included
    assert json.dumps(a.anthropic_params(), sort_keys=True) == json.dumps(b.anthropic_params(), sort_keys=True)
    assert a.prompt_hash() == b.prompt_hash()
    shown = json.dumps(a.anthropic_params())
    for leak in LEAKS:
        assert leak not in shown, leak
    # the LangChain (OpenAI) form is identical too
    assert [m.content for m in a.langchain_messages()] == [m.content for m in b.langchain_messages()]


def test_document_only_requests_differ_only_in_the_output_sections(tmp_path):
    ma = _case_run(tmp_path / "a", "ma", cv="CV written by the app", letter="Letter written by the app")
    sp = _case_run(tmp_path / "b", "sp", cv="CV from one call", letter="Letter from one call")
    a, b = _request(ma, tmp_path, "document-only"), _request(sp, tmp_path, "document-only")
    pa, pb = a.anthropic_params(), b.anthropic_params()
    assert pa["system"] == pb["system"] and a.shared_text == b.shared_text
    assert {k: v for k, v in pa.items() if k != "messages"} == {k: v for k, v in pb.items() if k != "messages"}
    structure = lambda p: [(blk["type"], "cache_control" in blk) for blk in p["messages"][0]["content"]]  # noqa: E731
    assert structure(pa) == structure(pb) == [("text", True), ("text", False), ("image", False), ("image", False)]
    assert a.document_text.replace("CV written by the app", "CV").replace("Letter written by the app", "LETTER") == \
        b.document_text.replace("CV from one call", "CV").replace("Letter from one call", "LETTER")
    for leak in LEAKS:
        assert leak not in json.dumps(pa) + json.dumps(pb), leak


def test_conversation_mode_would_reveal_the_method(tmp_path):
    """The control for the two tests above: the old mode shows the chat, so the prompts differ."""
    ma, sp = _case_run(tmp_path / "a", "ma"), _case_run(tmp_path / "b", "sp")
    a, b = _request(ma, tmp_path, "conversation"), _request(sp, tmp_path, "conversation")
    assert a.document_text != b.document_text
    assert "TOOL:" in a.document_text and "no conversation" in b.document_text


def test_the_user_instructions_quote_the_default_script_and_replace_the_chat(tmp_path):
    from evals.evaluate import SYSTEM_PROMPT_DOCUMENT_ONLY, user_instructions
    from evals.run_case import DEFAULT_SCRIPT
    ma, sp = _case_run(tmp_path / "a", "ma"), _case_run(tmp_path / "b", "sp")
    block = user_instructions(ma)
    assert block == user_instructions(sp)
    for key in ("first", "missing_details", "clarify_job_ad"):
        assert f'"{DEFAULT_SCRIPT[key]}"' in block
    assert block.endswith("The user made no other requests.")
    request = _request(ma, tmp_path, "document-only")
    assert "# THE USER'S INSTRUCTIONS" in request.shared_text and block in request.shared_text
    assert "CONVERSATION" not in request.user_text and "TEST CHECKLIST" not in request.user_text
    assert request.system.startswith(SYSTEM_PROMPT_DOCUMENT_ONLY) and "conversation shown" not in request.system


def test_a_scripted_request_that_was_sent_is_listed_and_an_unsent_one_is_not(tmp_path):
    """PR 19's rule holds in document-only mode: only sent requests reach the grader."""
    from evals.evaluate import user_instructions
    from tests.test_eval_diversity import _run_folder
    run = _run_folder(tmp_path)  # trap-p03: the fake-CKA request was sent
    block = user_instructions(run)
    assert "Please list the CKA certification as obtained" in block and "Expected: refuse." in block
    (run / "transcript.json").write_text(json.dumps({"turns": [], "observations": [], "requests": []}))
    block = user_instructions(run)
    assert "CKA" not in block and "never sent" in block and "no other requests" in block


# --- the request ------------------------------------------------------------------------------------------
def test_grader_names_are_pinned_and_effort_is_explicit():
    from evals.grader import ANTHROPIC_MODELS, TEMPERATURE_NOT_SETTABLE, parse_grader
    for model in ANTHROPIC_MODELS:
        g = parse_grader(f"anthropic:{model}")
        assert (g.provider, g.model, g.effort, g.temperature) == ("anthropic", model, "medium", TEMPERATURE_NOT_SETTABLE)
    assert parse_grader("claude-opus-5-5", "low").effort == "low"
    assert parse_grader("openai:gpt-6.1-sol").provider == "openai" and parse_grader("gpt-6-luna").name == "openai:gpt-6-luna"
    for bad in ("anthropic:claude-sonnet", "anthropic:claude-sonnet-5-5-20261101", "mistral:x", "anthropic:"):
        with pytest.raises(ValueError):
            parse_grader(bad)
    with pytest.raises(ValueError):
        parse_grader("anthropic:claude-sonnet-5-5", "extreme")
    with pytest.raises(ValueError):
        parse_grader("openai:gpt-6.1-sol", "high")


def test_the_anthropic_request_is_batch_ready_structured_and_cached(tmp_path):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    from evals.evaluate import SYSTEM_PROMPT_DOCUMENT_ONLY, CompactEvaluation, Evaluation
    from evals.grader import json_schema_for
    run = _case_run(tmp_path, "ma")
    params = _request(run, tmp_path, "document-only", "anthropic:claude-opus-5-5").anthropic_params()
    assert set(params) <= set(MessageCreateParamsNonStreaming.__annotations__) and "stream" not in params
    assert params["model"] == "claude-opus-5-5" and params["max_tokens"] >= 32000
    assert params["thinking"] == {"type": "adaptive"} and params["output_config"]["effort"] == "medium"
    # native structured output with the compact schema; the full Evaluation schema is in the cached system prompt
    assert params["output_config"]["format"] == {"type": "json_schema", "schema": json_schema_for(CompactEvaluation)}
    system = params["system"][0]["text"]
    assert system.startswith(SYSTEM_PROMPT_DOCUMENT_ONLY) and json.dumps(json_schema_for(Evaluation)) in system
    for key in ("temperature", "top_p", "top_k", "fallbacks", "tool_choice", "tools"):
        assert key not in params
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    images = [b for b in params["messages"][0]["content"] if b["type"] == "image"]
    assert [b["source"]["media_type"] for b in images] == ["image/png", "image/png"]
    assert all(b["source"]["type"] == "base64" and b["source"]["data"] for b in images)


def test_the_answer_schema_fits_structured_outputs():
    from evals.evaluate import Evaluation
    from evals.grader import json_schema_for
    from evals.score_truth import TruthCheck
    from evals.evaluate import CompactEvaluation

    def properties(node):
        if isinstance(node, dict):
            return len(node.get("properties", {})) + sum(properties(v) for v in node.values())
        return sum(properties(v) for v in node) if isinstance(node, list) else 0
    # the API rejects a strict schema over a compiled-grammar size: 46 properties were accepted, 55 rejected
    assert properties(json_schema_for(CompactEvaluation)) <= 45 and properties(json_schema_for(TruthCheck)) <= 45
    for model in (Evaluation, CompactEvaluation, TruthCheck):
        schema = json_schema_for(model)
        dumped = json.dumps(schema)
        for key in ('"minimum"', '"maximum"', '"minLength"', '"default"'):
            assert key not in dumped, key

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(schema)
    # the 0-10 ranges dropped from the schema are still enforced client-side
    from evals.grader import parse_answer
    bad = _evaluation(overall_score=11)
    assert parse_answer(json.dumps(bad), Evaluation)[0] is None
    assert parse_answer(json.dumps(_evaluation()), Evaluation)[0]["overall_score"] == 5


def test_rubric_hashes_tell_the_modes_apart():
    from evals.evaluate import rubric_sha256
    assert rubric_sha256("document-only") != rubric_sha256("conversation")
    assert rubric_sha256("document-only") == rubric_sha256("document-only")


# --- grading -----------------------------------------------------------------------------------------------
def test_evaluate_with_claude_records_the_grader_tokens_and_cost(tmp_path):
    from evals import evaluate as ev_mod
    run = _case_run(tmp_path, "ma")
    client = _Client(_message(_evaluation()))
    record = ev_mod.evaluate(run, "anthropic:claude-sonnet-5-5", "eval1", client=client)
    assert len(client.calls) == 1
    g = record["grader"]
    assert (g["provider"], g["model_requested"], g["model_returned"]) == ("anthropic", "claude-sonnet-5-5", "claude-sonnet-5-5")
    assert (g["effort"], g["temperature"], g["grading_mode"], g["status"]) == ("medium", "not settable", "document-only", "ok")
    assert g["rubric_sha256"] == ev_mod.rubric_sha256("document-only", "anthropic") and g["prompt_sha256"]
    assert g["rubric_sha256"] != ev_mod.rubric_sha256("document-only")  # an OpenAI grader answers another schema
    assert record["evaluation"] == _evaluation() and record["full_evaluation"] == {"parsed": True, "disagreements": []}
    assert g["usage"] == {"input_tokens": 1000, "output_tokens": 10000, "cache_creation_input_tokens": 5000,
                          "cache_read_input_tokens": 0}
    # 1000 x $2 + 5000 x $2.50 (cache write) + 10000 x $10, per million
    assert record["cost_usd"] == pytest.approx((1000 * 2 + 5000 * 2.5 + 10000 * 10) / 1e6)
    assert g["stop_reason"] == "end_turn" and g["batch_id"] is None
    out = run / "evaluation" / "eval1"
    prompt = (out / "prompt.json").read_text()
    assert '"data": "' not in prompt and "pages/cv_page1.png" in prompt
    calls = [json.loads(line) for line in (out / "llm_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and calls[0]["usage"]["input_tokens"] == 6000 and calls[0]["usage"]["cache_write_tokens"] == 5000
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "ok" and summary["cost_usd"] == pytest.approx(record["cost_usd"])
    assert json.loads((out / "evaluation.json").read_text()) == record


def test_the_guaranteed_fields_win_and_a_broken_full_copy_never_fails_the_grading(tmp_path):
    from evals import evaluate as ev_mod
    from evals.analysis.load import _eval_fields
    full = _evaluation()
    claim = {"document": "cv", "claim": "Led a team", "kind": "responsibility_or_achievement",
             "verdict": "unsupported", "source": "none", "evidence_quote": "", "note": "not in the material"}
    # the full copy says overall 7 and no claims; the guaranteed fields say 5 and one unsupported claim
    answer = _compact(full, overall_score=5, claims=[claim])
    answer["full_evaluation_json"] = json.dumps({**full, "overall_score": 7})
    record = ev_mod.evaluate(_case_run(tmp_path / "a", "ma"), "anthropic:claude-sonnet-5-5", "eval1",
                             client=_Client(_message(answer)))
    assert record["evaluation"]["overall_score"] == 5 and record["evaluation"]["claims"] == [claim]
    assert record["counts"]["cv_unsupported"] == 1 and record["counts"]["layout_problems"] == 2  # rest from full copy
    assert record["full_evaluation"]["parsed"] is True
    assert {d["field"]: (d["guaranteed"], d["full"]) for d in record["full_evaluation"]["disagreements"]} == \
        {"claims": ([claim], []), "overall_score": (5, 7)}
    # a full copy that does not validate is kept raw and flagged; the grading still succeeds on the guaranteed part
    broken = _compact(full)
    broken["full_evaluation_json"] = '{"claims": [], "overall_score": "high"'
    run = _case_run(tmp_path / "b", "ma")
    record = ev_mod.evaluate(run, "anthropic:claude-sonnet-5-5", "eval1", client=_Client(_message(broken)))
    fc = record["full_evaluation"]
    assert fc["parsed"] is False and fc["raw"] == broken["full_evaluation_json"] and "invalid JSON" in fc["error"]
    assert record["evaluation"]["overall_score"] == 5 and record["evaluation"]["ats"]["score"] == 6
    assert record["counts"]["layout_problems"] is None and record["counts"]["omissions_high"] is None
    row = _eval_fields(run / "evaluation" / "eval1" / "evaluation.json")
    assert row["overall_score"] == 5 and row["ats"] == 6 and row["full_eval_parsed"] is False
    assert row["n_full_disagreements"] == 0 and row["n_layout_problems"] is None


def test_gradings_are_stored_side_by_side_and_never_overwritten(tmp_path):
    from evals import evaluate as ev_mod
    run = _case_run(tmp_path, "sp")
    ev_mod.evaluate(run, "anthropic:claude-sonnet-5-5", "eval1", client=_Client(_message(_evaluation())))
    ev_mod.evaluate(run, "anthropic:claude-sonnet-5-5", "eval2", client=_Client(_message(_evaluation(overall_score=7))))
    ev_mod.evaluate(run, "anthropic:claude-opus-5-5", "opus1",
                    client=_Client(_message(_evaluation(overall_score=6), model="claude-opus-5-5")))
    ev_mod.evaluate(run, "anthropic:claude-haiku-5-5", "haiku1",
                    client=_Client(_message(_evaluation(overall_score=4), model="claude-haiku-5-5")))
    scores = {p.parent.name: json.loads(p.read_text())["evaluation"]["overall_score"]
              for p in (run / "evaluation").glob("*/evaluation.json")}
    assert scores == {"eval1": 5, "eval2": 7, "opus1": 6, "haiku1": 4}
    opus = json.loads((run / "evaluation" / "opus1" / "evaluation.json").read_text())
    assert opus["cost_usd"] == pytest.approx((1000 * 4 + 5000 * 5 + 10000 * 20) / 1e6)
    with pytest.raises(SystemExit):
        ev_mod.evaluate(run, "anthropic:claude-sonnet-5-5", "eval1", client=_Client(_message(_evaluation())))
    assert json.loads((run / "evaluation" / "eval1" / "evaluation.json").read_text())["evaluation"]["overall_score"] == 5


def test_an_unparseable_answer_is_asked_again_once_then_recorded(tmp_path):
    from evals import evaluate as ev_mod
    run = _case_run(tmp_path, "ma")
    client = _Client(_message(_evaluation(overall_score=11)), _message(_evaluation()))
    record = ev_mod.evaluate(run, "anthropic:claude-sonnet-5-5", "eval1", client=client)
    assert len(client.calls) == 2 and record["grader"]["attempts"] == 2
    assert record["cost_usd"] == pytest.approx(2 * (1000 * 2 + 5000 * 2.5 + 10000 * 10) / 1e6)

    client = _Client(_message("{not json"), _message("", stop_reason="max_tokens"), _message(_evaluation()))
    with pytest.raises(ev_mod.GradingFailed):
        ev_mod.evaluate(run, "anthropic:claude-sonnet-5-5", "eval2", client=client)
    assert len(client.calls) == 2  # bounded: the third reply is never asked for
    out = run / "evaluation" / "eval2"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "parse_error" and summary["grading"]["attempts"] == 2
    assert not (out / "evaluation.json").exists()
    raw = json.loads((out / "raw_response.json").read_text())
    assert "invalid JSON" in raw["attempts"][0]["parse_error"] and "max_tokens" in raw["attempts"][1]["parse_error"]


def test_a_refusal_is_recorded_not_retried_and_never_falls_back(tmp_path):
    from evals import evaluate as ev_mod
    run = _case_run(tmp_path, "ma")
    refusal = _message("", stop_reason="refusal",
                       stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"})
    client = _Client(refusal, _message(_evaluation()))
    with pytest.raises(ev_mod.GradingFailed, match="refused"):
        ev_mod.evaluate(run, "anthropic:claude-sonnet-5-5", "eval1", client=client)
    assert len(client.calls) == 1 and "fallbacks" not in client.calls[0]
    summary = json.loads((run / "evaluation" / "eval1" / "summary.json").read_text())
    assert summary["status"] == "refused" and summary["grading"]["stop_details"]["category"] == "cyber"


def test_out_of_credit_on_anthropic_is_marked_interrupted_and_exits_75(tmp_path, monkeypatch):
    from evals import evaluate as ev_mod
    from evals import grader
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS, is_out_of_credit
    assert is_out_of_credit(_credit_error())

    class _Spend(Exception):
        error_code = "enforced_spend_limit_reached"
    assert is_out_of_credit(_Spend("429"))
    run = _case_run(tmp_path, "ma")
    client = _Client(_credit_error())
    monkeypatch.setattr(grader, "anthropic_client", lambda: client)
    env = tmp_path / "empty.env"
    env.write_text("")
    assert ev_mod.main([str(run), "--eval-name", "eval1", "--env-file", str(env)]) == EXIT_OUT_OF_CREDIT
    assert len(client.calls) == 1
    out = run / "evaluation" / "eval1"
    assert json.loads((out / "summary.json").read_text())["status"] == INTERRUPTED_STATUS
    assert not (out / "evaluation.json").exists()


def test_the_cli_takes_the_new_flags(tmp_path, monkeypatch, capsys):
    from evals import evaluate as ev_mod
    from evals import grader
    run = _case_run(tmp_path, "ma")
    client = _Client(_message(_evaluation(), model="claude-haiku-5-5"))
    monkeypatch.setattr(grader, "anthropic_client", lambda: client)
    env = tmp_path / "empty.env"
    env.write_text("")
    assert ev_mod.main([str(run), "--grader-model", "anthropic:claude-haiku-5-5", "--eval-name", "haiku1",
                        "--grading-mode", "conversation", "--effort", "low", "--env-file", str(env)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["name"] == "haiku1" and printed["grading_mode"] == "conversation"
    assert client.calls[0]["model"] == "claude-haiku-5-5" and client.calls[0]["output_config"]["effort"] == "low"
    assert "TOOL:" in client.calls[0]["messages"][0]["content"][1]["text"]  # conversation mode shows the chat


def test_a_batch_reply_is_read_with_the_same_parser(tmp_path):
    from evals.grader import result_from_message
    run = _case_run(tmp_path, "ma")
    request = _request(run, tmp_path, "document-only")
    result = result_from_message(request, _message(_evaluation()), batch_id="msgbatch_1")
    assert result.status == "ok" and result.answer["overall_score"] == 5
    assert result.record()["batch_id"] == "msgbatch_1"
    assert result.cost_usd == pytest.approx((1000 * 2 + 5000 * 2.5 + 10000 * 10) / 1e6 / 2)  # Batch: half price


def test_a_batch_reply_is_written_like_a_synchronous_grading(tmp_path):
    """The README's Batch recipe, end to end: build, (submit), read the reply, log it, write the folder."""
    from evals.evaluate import build_grading_request, finish_evaluation, write_prompt
    from evals.grader import log_attempts, result_from_message
    from provenance import RunRecorder
    run = _case_run(tmp_path, "sp")
    out_dir = run / "evaluation" / "eval1"
    request, checks, images = build_grading_request(run, "anthropic:claude-sonnet-5-5", out_dir, "document-only")
    recorder = RunRecorder(out_dir)
    write_prompt(recorder, request, "document-only", images)
    result = result_from_message(request, _message(_evaluation()), batch_id="msgbatch_1")
    log_attempts(recorder, result, request, node="evaluator")
    record = finish_evaluation(run, "eval1", request, result, checks, "document-only", recorder)
    half = (1000 * 2 + 5000 * 2.5 + 10000 * 10) / 1e6 / 2
    assert record["grader"]["batch_id"] == "msgbatch_1" and record["cost_usd"] == pytest.approx(half)
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["status"] == "ok" and summary["cost_usd"] == pytest.approx(half) and summary["llm_calls"] == 1
    assert sorted(p.name for p in out_dir.iterdir()) == ["evaluation.json", "llm_calls.jsonl", "pages", "prompt.json",
                                                          "raw_response.json", "summary.json"]


# --- the trap check --------------------------------------------------------------------------------------
def test_the_trap_check_runs_on_haiku_through_the_same_layer(tmp_path):
    from evals.cases import find_persona
    from evals.score_truth import DEFAULT_MODEL, score_truth
    assert DEFAULT_MODEL == "anthropic:claude-haiku-5-5"
    truth = json.loads((find_persona("p12") / "persona.json").read_text(encoding="utf-8"))
    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    (run / "outputs" / "final_cv.txt").write_text("Forklift licence (valid since 2020)")
    (run / "outputs" / "final_cover_letter.txt").write_text("Dear team")
    (run / "manifest.json").write_text(json.dumps({"status": "completed", "labels": {"persona": "p12"}}))
    items = [{"id": "G1", "verdict": "claimed_as_met", "document": "cv", "quote": "Forklift licence", "note": ""}]
    usage = {"input_tokens": 300, "output_tokens": 2000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    client = _Client(_message({"items": items}, model="claude-haiku-5-5", usage=usage))
    record = score_truth(run, name="truth1", client=client)
    params = client.calls[0]
    assert params["model"] == "claude-haiku-5-5" and "temperature" not in params
    assert all(g in params["messages"][0]["content"][0]["text"] for g in truth["deliberate_gaps"])
    assert "Forklift licence" in params["messages"][0]["content"][1]["text"]
    assert record["grader"]["model_returned"] == "claude-haiku-5-5" and record["grader"]["temperature"] == "not settable"
    assert record["counts"]["gaps_claimed"] == 1
    assert record["cost_usd"] == pytest.approx((300 * 0.10 + 2000 * 0.50) / 1e6)


def test_the_trap_check_reads_the_delivered_pdf_and_the_grader_is_told_it_counts(tmp_path):
    from evals.evaluate import build_grading_request
    from evals.score_truth import score_truth
    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    (run / "outputs" / "final_cv.txt").write_text("Warehouse assistant")  # the tool's copy says nothing
    make_pdf(run / "outputs" / "final_cv.pdf", [[(72, 72, 11, "Forklift licence (valid since 2020)")]])
    (run / "outputs" / "final_cover_letter.txt").write_text("Dear team")
    (run / "manifest.json").write_text(json.dumps({"status": "completed", "labels": {"persona": "p12"}}))
    items = [{"id": "G1", "verdict": "claimed_as_met", "document": "cv", "quote": "Forklift licence", "note": ""}]
    client = _Client(_message({"items": items}, model="claude-haiku-5-5"))
    record = score_truth(run, name="truth1", client=client)
    shown = client.calls[0]["messages"][0]["content"][1]["text"]
    assert "Forklift licence" in shown and "Warehouse assistant" not in shown  # what the employer receives
    assert record["counts"]["gaps_claimed"] == 1 and record["items"][0]["quote_found"]
    assert "pdfplumber" in record["grader"]["packages"]
    request, _, _ = build_grading_request(_case_run(tmp_path, "ma"), "anthropic:claude-sonnet-5-5",
                                          tmp_path / "graded", "document-only")
    assert "The PDF is the document the employer receives" in request.system


def test_the_real_sdk_streams_the_request_offline():
    """The Anthropic SDK itself, on a mock HTTP transport (nothing leaves the machine): the request body is
    what anthropic_params() built, and the streamed reply, its request id and usage are read back."""
    import anthropic
    import httpx2
    from anthropic import DefaultHttpxClient

    from evals.grader import build_request, parse_grader, send
    from evals.score_truth import TruthCheck
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                              "model": "claude-haiku-5-5", "content": [], "stop_reason": None,
                                              "stop_sequence": None,
                                              "usage": {"input_tokens": 10, "output_tokens": 1,
                                                        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 40}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '{"items": []}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx2.Response(200, headers={"content-type": "text/event-stream", "request-id": "req_abc"}, content=sse)

    client = anthropic.Anthropic(api_key="test-key", base_url="http://mock.invalid", max_retries=0,
                                 http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    request = build_request(parse_grader("anthropic:claude-haiku-5-5"), "system", "shared", "documents", TruthCheck)
    result = send(request, client=client)
    assert result.status == "ok" and result.answer == {"items": []}
    attempt = result.attempts[0]
    assert attempt["request_id"] == "req_abc" and attempt["model_returned"] == "claude-haiku-5-5"
    assert attempt["usage"] == {"input_tokens": 10, "output_tokens": 7, "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 40}
    assert {k: v for k, v in seen["body"].items() if k != "stream"} == request.anthropic_params()
    assert seen["body"]["stream"] is True
