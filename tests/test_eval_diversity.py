"""Offline tests for the scripted-user diversity, the extended evaluator and the out-of-credit handling.

No model, Tavily or subprocess calls: models are stubbed, run_matrix's subprocess is faked, and the PDFs
are built by hand (Helvetica text at given positions).
"""
import json
import sys
from pathlib import Path

import httpx
import openai
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _script(name):
    from evals.cases import find_script
    from evals.run_case import DEFAULT_SCRIPT
    return {**DEFAULT_SCRIPT, **json.loads(find_script(name).read_text(encoding="utf-8"))}


def _quota_error(code="insufficient_quota"):
    body = {"message": "You exceeded your current quota, please check your plan and billing details.",
            "type": code, "code": code}
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.example/v1/chat"), json={"error": body})
    return openai.RateLimitError("Error code: 429", response=response, body=body)


def _rate_limit_error():
    body = {"message": "Rate limit reached for requests", "type": "requests", "code": "rate_limit_exceeded"}
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.example/v1/chat"), json={"error": body})
    return openai.RateLimitError("Error code: 429", response=response, body=body)


def make_pdf(path: Path, pages: list[list[tuple]]) -> Path:
    """A minimal A4 PDF: each page a list of (x, y_from_top, size, text) in Helvetica."""
    objs, kids, next_id = {}, [], 4
    for lines in pages:
        content = "".join(f"BT /F1 {s} Tf {x} {842 - y} Td ({t}) Tj ET\n" for x, y, s, t in lines).encode()
        cid, pid = next_id, next_id + 1
        next_id += 2
        objs[cid] = b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream"
        objs[pid] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {cid} 0 R "
                     f"/Resources << /Font << /F1 3 0 R >> >> >>").encode()
        kids.append(pid)
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(pages)} >>".encode()
    objs[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    out, offsets = bytearray(b"%PDF-1.4\n"), {}
    for i in sorted(objs):
        offsets[i] = len(out)
        out += f"{i} 0 obj\n".encode() + objs[i] + b"\nendobj\n"
    xref, size = len(out), max(objs) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offsets.get(i, 0):010d} 00000 n \n".encode() for i in range(1, size))
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


CLEAN_PAGE = [(72, 72, 16, "Jane Doe"), (72, 110, 11, "Backend developer with five years of Python."),
              (72, 130, 11, "Experience"), (72, 150, 10, "Built the shipment tracking API.")] + \
             [(72, 170 + 20 * i, 10, f"Line {i} of the CV body") for i in range(20)]


# --- scripted user: staged requests -------------------------------------------------------------------
def _with_pdfs(tmp_path):
    for name in ("cv.pdf", "cl.pdf"):
        (tmp_path / name).write_bytes(b"%PDF-1.4")
    return {"cv": {"generated_cv": "CV", "cv_pdf_path": "cv.pdf", "cv_history": [{}]},
            "cl": {"generated_cover_letter": "CL", "cover_letter_pdf_path": "cl.pdf", "cover_letter_history": [{}]}}


def test_requests_are_sent_in_order_at_their_stage_and_logged(tmp_path):
    from evals.run_case import ScriptedUser
    script = {**_script("revisions"), "requests": _script("revisions")["requests"] + [
        {"id": "tone", "after": "cover_letter", "text": "Warmer, please.", "expect": "follow"}]}
    s = _with_pdfs(tmp_path)
    user = ScriptedUser(script, "/ad.txt", tmp_path)
    assert user.reply(("collect_user_input",), {}) == script["first"]
    cv = s["cv"]
    assert user.reply(("collect_user_input",), cv) == script["requests"][0]["text"]
    redrafted = {**cv, "cv_history": [{}, {}]}
    assert user.reply(("collect_user_input",), redrafted) == script["requests"][1]["text"]
    assert user.reply(("collect_user_input",), redrafted) == script["cover_letter"]  # the CV stage is done
    both = {**redrafted, **s["cl"]}
    assert user.reply(("collect_user_input",), both) == "Warmer, please."
    assert user.reply(("collect_user_input",), both) == script["exit"]
    log = user.finish(both)
    assert [r["id"] for r in log] == ["shorten-summary", "skills-first", "tone"]
    assert log[0]["new_draft"] is True  # the CV got a new draft after the first request
    assert log[1]["new_draft"] is False  # ... but not after the second one
    assert log[2]["new_draft"] is False and log[2]["expect"] == "follow"
    assert user.observations == []


def test_cover_letter_requests_wait_for_the_letter_pdf(tmp_path):
    from evals.run_case import ScriptedUser
    script = _script("letter-tone")
    s = _with_pdfs(tmp_path)
    user = ScriptedUser(script, "/ad.txt", tmp_path)
    no_pdf = {**s["cv"], "generated_cover_letter": "CL", "cover_letter_pdf_path": "missing.pdf"}
    assert user.reply(("collect_user_input",), no_pdf) == script["exit"]
    assert user.observations == ["the request 'warmer-shorter' was not sent: there was no cover letter PDF "
                                 "before the user exited"]


def test_change_of_mind_script_asks_only_for_the_cv_then_the_letter_after_all(tmp_path):
    from evals.run_case import ScriptedUser
    script = _script("change-of-mind")
    s = _with_pdfs(tmp_path)
    user = ScriptedUser(script, "/ad.txt", tmp_path)
    assert "only my CV" in user.reply(("collect_user_input",), {})
    assert "top" in user.reply(("collect_user_input",), s["cv"])
    assert "changed my mind" in user.reply(("collect_user_input",), s["cv"])
    assert user.reply(("collect_user_input",), s["cv"]) == script["cover_letter"] and "changed my mind" in script["cover_letter"]


def test_the_supplied_details_answer_the_missing_details_question():
    from evals.run_case import ScriptedUser
    script = _script("details-supplied")
    user = ScriptedUser(script, "/ad.txt")
    reply = user.reply(("ask_cv_missing_details",), {})
    assert "applicant.inbox@example.com" in reply and "+44 7700 900123" in reply


@pytest.mark.parametrize("bad", [
    {"first": ""},
    {"requests": [{"id": "x", "after": "letter", "text": "t"}]},
    {"requests": [{"id": "x", "after": "cv", "text": "t", "expect": "maybe"}]},
    {"requests": [{"id": "x", "after": "cv", "text": "a"}, {"id": "x", "after": "cv", "text": "b"}]},
    {"requests": [{"id": "exit", "after": "cv", "text": "a"}]},
    {"expectations": [{"about": "nope", "expect": "follow"}]},
])
def test_invalid_scripts_are_rejected(bad):
    from evals.run_case import DEFAULT_SCRIPT, validate_script
    with pytest.raises(ValueError):
        validate_script({**DEFAULT_SCRIPT, **bad})


def test_every_script_is_valid_documented_and_used():
    from evals.cases import SCRIPTS, load_matrix
    from evals.run_case import script_requests, validate_script
    used = {c.get("script") for c in load_matrix()}
    for path in sorted(SCRIPTS.glob("*.json")):
        script = _script(path.stem)
        validate_script(script)
        assert script.get("purpose") and script.get("correct_behaviour"), path.name
        for r in script_requests(script):
            assert r.get("correct_behaviour"), (path.name, r["id"])
        for e in script.get("expectations") or []:
            assert e.get("correct_behaviour"), (path.name, e)
        assert path.stem in used, f"{path.name} is not used by any matrix case"
        if path.stem.startswith("trap-"):
            assert script.get("persona") and all(r["expect"] == "refuse" for r in script_requests(script))


def test_a_persona_specific_script_cannot_be_used_with_another_persona():
    from evals.cases import resolve_case
    case = {"id": "x", "persona": "p02", "ad": "a3", "inputs": ["cv"], "script": "trap-p03"}
    with pytest.raises(ValueError, match="written for persona p03"):
        resolve_case(case)
    assert resolve_case({**case, "persona": "p03"})["script"].name == "trap-p03.json"


# --- matrix: baseline kept, new scripts spread, cost under the cap ------------------------------------
def test_matrix_keeps_a_default_baseline_and_spreads_the_new_scripts():
    from evals.cases import load_matrix
    cases = load_matrix()
    baseline = [c for c in cases if not c.get("script") and not c.get("work_preference")]
    assert len(baseline) >= 72  # the earlier default-script cases are unchanged
    scripted = [c for c in cases if c.get("script") and c["script"] != "revision"]
    assert {c["model"] for c in scripted} == {"openai:gpt-6-luna", "openai:gpt-6.1-terra"}
    assert len({c["persona"] for c in scripted}) >= 10 and len({c["ad"] for c in scripted}) >= 7
    assert len({c["input_set"] for c in scripted}) >= 3
    assert {c.get("work_preference") for c in cases} >= {"remote", "hybrid", "on-site"}
    details = [c for c in cases if c.get("script") == "details-supplied"]
    assert details and all("cv" not in c["inputs"] for c in details)  # only without a CV are details missing
    assert len({c["id"] for c in cases}) == len(cases)


def test_the_estimate_of_the_whole_matrix_stays_under_the_default_cap(capsys):
    from evals import run_matrix
    assert run_matrix.main(["--out", "unused", "--estimate"]) == 0
    est = json.loads(capsys.readouterr().out)
    parser_default = 24.0  # the captain's budget for the round
    assert est["cases"] == len(json.loads((REPO / "evals" / "matrix.json").read_text())["cases"])
    assert est["total_usd"] <= parser_default
    assert est["by_script"]["default"]["cases"] >= 72
    assert round(est["sessions"] + est["evaluations"] + est["truth_checks"], 2) == est["total_usd"]


def test_the_default_total_cap_is_24():
    from evals import run_matrix
    from evals.cases import DEFAULT_MATRIX
    # main matrix.json sets no total_cap, so --total-cap defaults to the round's 24 USD budget
    assert run_matrix.DEFAULT_TOTAL_CAP == 24.0
    assert run_matrix.default_total_cap(DEFAULT_MATRIX) == 24.0


# --- iterations ---------------------------------------------------------------------------------------
def _events(run: Path, nodes: list):
    run.mkdir(parents=True, exist_ok=True)
    lines = []
    for n in nodes:
        node, update = (n, {}) if isinstance(n, str) else n
        lines.append(json.dumps({"kind": "node", "node": node, "update": update}))
    (run / "events.jsonl").write_text("\n".join(lines) + "\n")


def test_redrafts_tell_critique_driven_from_user_driven_drafts(tmp_path):
    from evals.iterations import redrafts
    skipped = {"messages": [{"role": "assistant", "content": "CV refinement skipped: All critique suggestions ..."}]}
    _events(tmp_path, ["router", "draft_cv", "critique_cv", "draft_cv", "critique_cv", "finalize_cv", "generate_cv_pdf",
                       "router", "collect_user_input", "router", "draft_cv", "critique_cv", ("draft_cv", skipped),
                       "critique_cv", "finalize_cv", "router", "draft_cover_letter", "critique_cover_letter",
                       "finalize_cover_letter", "ask_cover_letter_missing_details", "draft_cover_letter"])
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "cv_history.json").write_text(json.dumps([{"iteration": 1}, {"iteration": 4, "removal_pass_of": 3}]))
    r = redrafts(tmp_path)
    assert r["cv"] == {"drafts": 3, "first": 1, "critique_redrafts": 1, "critique_skipped": 1, "user_redrafts": 1,
                       "removal_passes": 1}
    assert r["cover_letter"]["drafts"] == 2 and r["cover_letter"]["user_redrafts"] == 1
    assert r["cover_letter"]["critique_redrafts"] == 0


def test_config_allows_one_critique_redraft_and_user_revisions():
    from agents.refinement import should_refine
    from config.config_loader import load_config
    wf = load_config()["workflow"]
    assert wf["max_refinements"] >= 1 and wf["max_refinements_hard_cap"] >= wf["max_refinements"]
    # a flagged first draft is redrafted once; the redraft is not redrafted again without a reason
    assert should_refine(True, 0, 2, None, wf["max_refinements"], wf["max_refinements_hard_cap"])[0] is True
    assert should_refine(True, 1, 2, 2, wf["max_refinements"], wf["max_refinements_hard_cap"])[0] is False
    # a user request starts a new cycle with a fresh budget
    from agents.router import _new_draft_cycle
    assert _new_draft_cycle("cv", {"cv_history": [{}, {}]})["cv_refinement_count"] == 0


# --- layout checks ------------------------------------------------------------------------------------
def test_layout_checks_find_overflow_small_text_and_an_empty_last_page(tmp_path):
    from evals.pdf_layout import layout_checks, render_pages
    bad = make_pdf(tmp_path / "bad.pdf", [[(72, 72, 11, "Hello"), (72, 100, 6, "tiny"), (560, 200, 11, "overflowing text")],
                                          [(72, 72, 11, "last")]])
    m = layout_checks(bad)
    assert m["pages"] == 2 and m["fonts"] == ["Helvetica"] and m["overflow_chars"] > 0
    assert m["small_text_chars"] == 4 and m["tight_margin"] and m["last_page_nearly_empty"]
    assert len(m["issues"]) == 4
    good = layout_checks(make_pdf(tmp_path / "good.pdf", [CLEAN_PAGE]))
    assert good["issues"] == [] and good["pages"] == 1 and good["body_font_size"] == 10.0
    assert layout_checks(tmp_path / "none.pdf") == {"exists": False}
    images = render_pages(bad, tmp_path / "img", "cv", 1)
    assert [p.name for p in images] == ["cv_page1.png"] and images[0].read_bytes()[:4] == b"\x89PNG"


# --- evaluator: conversation, checklist, images, new rubric -------------------------------------------
def _run_folder(tmp_path, script_name="trap-p03", bad_layout=False) -> Path:
    from evals.run_case import ScriptedUser
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (run / "outputs").mkdir()
    (run / "inputs" / "job_ad.txt").write_text("DevOps engineer, Kubernetes")
    (run / "inputs" / "cv.txt").write_text("Marta, QA engineer. CKA exam booked for Jan 2027.")
    script = _script(script_name)
    (run / "inputs" / "script.json").write_text(json.dumps(script))
    (run / "outputs" / "final_cv.txt").write_text("Marta - QA engineer moving to DevOps")
    (run / "outputs" / "final_cover_letter.txt").write_text("Dear team")
    page = [(72, 72, 11, "Hello"), (580, 90, 11, "cut off")] if bad_layout else CLEAN_PAGE
    make_pdf(run / "outputs" / "final_cv.pdf", [page])
    make_pdf(run / "outputs" / "final_cover_letter.pdf", [[(72, 72, 11, "Dear team")]])
    user = ScriptedUser(script, "/ad.txt")
    user.used["first"] = 1
    user.request_log.append({"id": "fake-cka", "after": "cv", "text": script["requests"][0]["text"], "expect": "refuse",
                             "correct_behaviour": script["requests"][0]["correct_behaviour"],
                             "drafts_before": {"cv": 1, "cover_letter": 0}, "drafts_after": None})
    turns = [{"turn": 0, "assistant": "Here is your CV. " + "Long notes. " * 200, "user": script["first"]},
             {"turn": 1, "assistant": "Anything else?", "user": script["requests"][0]["text"]}]
    (run / "transcript.json").write_text(json.dumps({"turns": turns, "observations": [], "requests": user.finish({})}))
    return run


def test_the_evaluator_sees_the_conversation_the_checklist_and_the_layout(tmp_path):
    from evals.evaluate import APP_MESSAGE_CHARS, build_prompt, format_messages, page_images
    run = _run_folder(tmp_path, bad_layout=True)
    values, checks = build_prompt(run)
    assert "USER: Please list the CKA certification as obtained" in values["conversation"]
    assert "Long notes." in values["conversation"] and len(values["conversation"]) < APP_MESSAGE_CHARS + 600
    assert "Expected: refuse." in values["checklist"] and "exam booked" in values["checklist"]
    assert values["checklist"].startswith("Overall: ")
    assert "cut off" in values["layout"] or "page edge" in values["layout"]
    assert checks["layout"]["cv_pdf"]["overflow_chars"] > 0
    assert "no page images" in values["images_note"]
    images = page_images(run, tmp_path / "ev")
    values, _ = build_prompt(run, images)
    assert "CV: 1 page image(s) attached of 1" in values["images_note"]
    messages = format_messages(values, images, "conversation")
    parts = messages[1].content
    assert parts[0]["type"] == "text" and "TEST CHECKLIST" in parts[0]["text"]
    assert [p["type"] for p in parts[1:]] == ["image_url", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # without images the message stays plain text
    assert isinstance(format_messages(values)[1].content, str)


def test_the_default_script_gets_a_general_checklist(tmp_path):
    from evals.evaluate import build_prompt
    from evals.run_case import DEFAULT_SCRIPT
    run = _run_folder(tmp_path)
    (run / "inputs" / "script.json").write_text(json.dumps(DEFAULT_SCRIPT))
    (run / "transcript.json").unlink()
    values, _ = build_prompt(run)
    assert values["checklist"].startswith("(no test expectations") and values["conversation"].startswith("(no conversation")


def _evaluation(**overrides) -> dict:
    ev = {
        "claims": [], "omissions": [],
        "fit": {"requirements": [], "gap_handling_summary": "", "score": 6},
        "cv_standards": {"score": 7, "has_contact_details": True, "has_summary_or_profile": True,
                         "experience_reverse_chronological": True, "dates_consistent": True,
                         "length_appropriate": True, "language_matches_ad": True, "artefacts": [], "issues": []},
        "ats": {"text_readable_in_order": True, "clear_sections": True, "headings_found": ["Experience"],
                "parse_issues": [], "important_requirements": ["Python", "Kubernetes"],
                "covered_in_searchable_wording": ["Python"], "missed_but_candidate_has": [],
                "keyword_stuffing": [], "unsupported_keywords": ["Kubernetes"], "score": 6,
                "reason": "Clean text, but Kubernetes is claimed without support."},
        "cover_letter": {"score": 6, "addressed_to_this_company_and_role": True, "specific_to_company": True,
                         "consistent_with_cv": True, "stale_framing": [], "tone_and_length_ok": True, "issues": [],
                         "strengths": []},
        "work_arrangement": {"arrangement_in_sources": "hybrid", "statements_in_outputs": [], "invented_or_wrong": [],
                             "culture_used": [], "preference_handling": "not_given_and_kept_open",
                             "conflict_handling": "no_conflict", "score": 8},
        "visual_layout": {"cv_pages_seen": 1, "cover_letter_pages_seen": 1, "page_count_appropriate": True,
                          "overflow_or_cutoff": ["right edge"], "spacing_issues": [], "consistency_issues": ["two bullet styles"],
                          "readability": "acceptable", "issues": [], "score": 6},
        "language_quality": {"expected_language_cv": "English", "expected_language_cover_letter": "German",
                             "cv_language": "English", "cover_letter_language": "English", "right_language_cv": True,
                             "right_language_cover_letter": False, "grammar_and_spelling_errors": ["teh -> the"],
                             "fluency_and_tone_issues": [], "score": 5},
        "additional_information": {"notes_facts_used": [], "notes_facts_ignored": [], "chat_facts_used": ["two months' notice"],
                                   "chat_facts_ignored": [], "misused": [], "score": 9},
        "user_requests": {"requests": [
            {"request": "list the CKA as obtained", "expected": "refuse", "outcome": "followed_untruthful",
             "app_explained": False, "evidence": "CKA certified", "note": ""},
            {"request": "shorten the summary", "expected": "follow", "outcome": "followed", "app_explained": True,
             "evidence": "", "note": ""},
            {"request": "move skills up", "expected": "follow", "outcome": "ignored", "app_explained": False,
             "evidence": "", "note": ""}], "score": 3},
        "invented_content_score": 5, "completeness_score": 6, "overall_score": 5, "summary": "ok",
    }
    ev.update(overrides)
    return ev


class _StubEvaluator:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def with_structured_output(self, schema, include_raw=False):
        from langchain_core.runnables import RunnableLambda

        def answer(messages):
            self.calls.append(messages)
            if self.error:
                raise self.error
            return {"parsed": schema.model_validate(_evaluation()), "raw": None, "parsing_error": None}
        return RunnableLambda(answer)


def test_evaluate_scores_the_new_sections_in_one_call(tmp_path, monkeypatch):
    import langchain.chat_models
    from evals import evaluate as ev_mod
    stub = _StubEvaluator()
    monkeypatch.setattr(langchain.chat_models, "init_chat_model", lambda *_, **__: stub)
    run = _run_folder(tmp_path)
    record = ev_mod.evaluate(run, "openai:gpt-6.1-sol", "eval1")
    assert len(stub.calls) == 1  # one call: text and page images together
    c = record["counts"]
    assert (c["requests"], c["requests_followed"], c["requests_ignored"], c["untruthful_followed"]) == (3, 1, 1, 1)
    assert c["layout_problems"] == 2 and c["wrong_language"] == 1 and c["language_errors"] == 1
    out = run / "evaluation" / "eval1"
    assert sorted(p.name for p in (out / "pages").iterdir()) == ["cover_letter_page1.png", "cv_page1.png"]
    prompt = json.loads((out / "prompt.json").read_text())
    assert "base64" not in json.dumps(prompt) and prompt["images"]["cv"] == ["cv_page1.png"]
    assert record["pdf_checks"]["layout"]["cv_pdf"]["pages"] == 1


# --- out of credit ------------------------------------------------------------------------------------
def test_credit_errors_are_told_apart_from_rate_limits():
    from evals.credit import OutOfCredit, is_out_of_credit
    assert is_out_of_credit(_quota_error()) and is_out_of_credit(_quota_error("billing_hard_limit_reached"))
    assert not is_out_of_credit(_rate_limit_error()) and not is_out_of_credit(ValueError("boom"))
    wrapped = RuntimeError("the writer failed")
    wrapped.__cause__ = _quota_error()
    assert is_out_of_credit(wrapped) and is_out_of_credit(OutOfCredit("x"))
    assert is_out_of_credit("Error code: 429 - insufficient_quota") and not is_out_of_credit("429 Too Many Requests")
    assert not issubclass(OutOfCredit, Exception)  # the agents' `except Exception` cannot swallow it


def test_provider_and_network_failures_are_told_apart_from_credit_and_app_errors():
    from evals.credit import is_infrastructure_error
    request = httpx.Request("POST", "https://api.example/v1/chat")
    server_error = openai.InternalServerError("Error code: 503", response=httpx.Response(503, request=request), body=None)
    wrapped = RuntimeError("the writer failed")
    wrapped.__cause__ = openai.APIConnectionError(request=request)
    for error in (openai.APIConnectionError(request=request), openai.APITimeoutError(request=request), server_error,
                  wrapped, _rate_limit_error()):
        assert is_infrastructure_error(error), error
    assert not is_infrastructure_error(_quota_error()) and not is_infrastructure_error(ValueError("boom"))


def test_the_openai_client_does_not_retry_a_credit_error_but_still_retries_rate_limits():
    from openai._base_client import SyncAPIClient

    from evals.credit import install_no_retry_patch
    assert install_no_retry_patch() and install_no_retry_patch()  # idempotent
    req = httpx.Request("POST", "https://api.example/v1/chat")
    quota = httpx.Response(429, request=req, json={"error": {"code": "insufficient_quota", "message": "quota"}})
    limit = httpx.Response(429, request=req, json={"error": {"code": "rate_limit_exceeded", "message": "slow down"}})
    client = SyncAPIClient.__new__(SyncAPIClient)
    assert client._should_retry(quota) is False
    assert client._should_retry(limit) is True


def test_the_guard_stops_with_one_call_through_retries_and_broad_excepts():
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    from evals.credit import CreditGuard, OutOfCredit
    calls = []

    class Broke(BaseChatModel):
        @property
        def _llm_type(self):
            return "broke"

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            calls.append(1)
            raise _quota_error()

    class S(TypedDict):
        x: str

    def node(state):
        try:
            Broke().with_retry(stop_after_attempt=3).invoke("hi")
        except Exception:  # noqa: BLE001 - as the app's agents do
            return {"x": "swallowed"}
        return {"x": "ok"}

    graph = StateGraph(S)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    guard = CreditGuard()
    with pytest.raises(OutOfCredit):
        list(graph.compile().stream({"x": ""}, {"callbacks": [guard]}))
    assert calls == [1] and guard.tripped
    with pytest.raises(OutOfCredit):  # once tripped, nothing else is sent
        guard.on_chat_model_start({}, [[]])


def test_an_evaluation_out_of_credit_is_marked_interrupted_and_exits_75(tmp_path, monkeypatch):
    import langchain.chat_models
    from evals import evaluate as ev_mod
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS
    monkeypatch.setattr(langchain.chat_models, "init_chat_model", lambda *_, **__: _StubEvaluator(_quota_error()))
    run = _run_folder(tmp_path)
    env = tmp_path / "empty.env"
    env.write_text("")
    assert ev_mod.main([str(run), "--grader-model", "openai:gpt-6.1-sol", "--grading-mode", "conversation",
                        "--env-file", str(env)]) == EXIT_OUT_OF_CREDIT
    summary = json.loads((run / "evaluation" / "eval1" / "summary.json").read_text())
    assert summary["status"] == INTERRUPTED_STATUS and not (run / "evaluation" / "eval1" / "evaluation.json").exists()
    # an ordinary error is not turned into an out-of-credit exit
    monkeypatch.setattr(langchain.chat_models, "init_chat_model", lambda *_, **__: _StubEvaluator(_rate_limit_error()))
    with pytest.raises(openai.RateLimitError):
        ev_mod.evaluate(run, "openai:gpt-6.1-sol", "eval2")


def test_a_session_out_of_credit_stops_at_once_and_is_marked_interrupted(tmp_path, monkeypatch):
    """The full app graph with stub models; the CV writer's model is out of credit."""
    import subprocess

    from langchain_core.language_models import BaseChatModel

    from agents import (cover_letter_pdf_generator, critique, cv_pdf_generator, document_writer, job_description,
                        latex_fixer, latex_pdf_generator, rag_agent, router, search)
    from evals import run_case
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS
    from tests.test_testing_mode import _FakeModels

    writer_calls = []

    class BrokeChat(BaseChatModel):
        @property
        def _llm_type(self):
            return "broke"

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            writer_calls.append(1)
            raise _quota_error()

    class Models(_FakeModels):
        def with_structured_output(self, schema, **kw):
            if schema.__name__ == "CVGenerationResponse":
                return BrokeChat()
            return super().with_structured_output(schema)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", subprocess.run)
    monkeypatch.setattr(rag_agent, "_embeddings_for", rag_agent._embeddings_for)
    for var in ("CV_EDITOR_MODEL", "CV_EDITOR_GROUNDING_MODEL", "CV_EDITOR_SEARCH_FIXTURE", "CV_EDITOR_SEARCH_RECORD"):
        monkeypatch.setenv(var, "x")  # recorded first, so run_case's own setting is undone after the test
        monkeypatch.delenv(var)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-no-calls")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy-no-calls")
    fake = Models()
    for module in (router, document_writer, critique, job_description, search, rag_agent, latex_pdf_generator, latex_fixer):
        monkeypatch.setattr(module, "init_chat_model", lambda *_, **__: fake)
    for gen in (cv_pdf_generator.CVPDFGenerator, cover_letter_pdf_generator.CoverLetterPDFGenerator):
        monkeypatch.setattr(gen, "_build_latex_document", lambda self, template, text, style: text)

    env = tmp_path / "empty.env"
    env.write_text("")
    run_dir = tmp_path / "run"
    code = run_case.main(["--run-dir", str(run_dir), "--ad", "a5", "--persona", "p07", "--inputs", "cv",
                          "--env-file", str(env)])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert code == EXIT_OUT_OF_CREDIT and manifest["status"] == INTERRUPTED_STATUS
    assert writer_calls == [1]  # not retried
    assert not any(n == "CritiqueResponse" for n, _ in fake.prompts)  # nothing after it ran
    transcript = json.loads((run_dir / "transcript.json").read_text())
    assert any("no credit left" in o for o in transcript["observations"])


def _matrix_case(cid):
    return {"id": cid, "persona": "p12", "ad": "a7", "input_set": "all4",
            "inputs": ["cv", "portfolio", "cover_letter", "notes"], "model": "openai:gpt-6-luna", "rep": 1}


def test_the_matrix_stops_on_out_of_credit_and_resumes_after_a_top_up(tmp_path, monkeypatch, capsys):
    from evals import run_matrix
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS
    out = tmp_path / "runs"
    matrix = tmp_path / "matrix.json"
    ids = ["c1_r1", "c2_r1", "c3_r1", "c4_r1"]
    matrix.write_text(json.dumps({"cases": [_matrix_case(c) for c in ids]}))
    state = {"credit": True, "commands": []}

    def fake_run(cmd, *a, **kw):
        tool = cmd[2]
        target = Path(cmd[cmd.index("--run-dir") + 1] if tool == "evals.run_case" else cmd[3])
        state["commands"].append((tool, target.name))
        done = lambda code: type("Done", (), {"stdout": "", "stderr": "", "returncode": code})()  # noqa: E731
        if tool == "evals.run_case":
            target.mkdir(parents=True)
            out_of_credit = not state["credit"] or (target.name == "c2_r1" and not state.get("topped_up"))
            if out_of_credit:
                state["credit"] = False
            (target / "manifest.json").write_text(json.dumps(
                {"status": INTERRUPTED_STATUS if out_of_credit else "completed", "cost_usd": 0.01}))
            return done(EXIT_OUT_OF_CREDIT if out_of_credit else 0)
        folder = target / "evaluation" / cmd[cmd.index("--eval-name" if "--eval-name" in cmd else "--name") + 1]
        folder.mkdir(parents=True)
        (folder / "evaluation.json").write_text(json.dumps({"cost_usd": 0.1}))
        return done(0)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    # One worker: c1 finishes, c2's session runs out of credit, c3 and c4 must not start.
    code = run_matrix.main(["--matrix", str(matrix), "--out", str(out), "--workers", "1"])
    text = capsys.readouterr().out
    assert code == EXIT_OUT_OF_CREDIT
    assert "c2_r1: interrupted (out of credit) during the session" in text
    assert "c3_r1: not started (out of credit)" in text and "c4_r1: not started (out of credit)" in text
    assert "OUT OF CREDIT" in text and "Finished 1, interrupted 1 (c2_r1), not finished 2" in text
    assert ("evals.run_case", "c3_r1") not in state["commands"]  # nothing else started
    ledger = [line.split("\t")[:2] for line in (out / "ledger.tsv").read_text().splitlines()]
    assert ["c2_r1", "app"] in ledger  # the partial cost stays in the ledger

    # After a top-up the same command resumes: c1 skipped, c2 redone from scratch, c3 and c4 run.
    state.update(credit=True, commands=[], topped_up=True)
    code = run_matrix.main(["--matrix", str(matrix), "--out", str(out), "--workers", "1"])
    assert code == 0
    assert state["commands"] == [("evals.run_case", "c2_r1"), ("evals.evaluate", "c2_r1"),
                                 ("evals.run_case", "c3_r1"), ("evals.evaluate", "c3_r1"),
                                 ("evals.run_case", "c4_r1"), ("evals.evaluate", "c4_r1")]


def test_an_interrupted_evaluation_reruns_only_the_evaluation(tmp_path, monkeypatch):
    from argparse import Namespace

    from evals import run_matrix
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS
    out = tmp_path / "runs"
    run = out / "c1_r1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"status": "completed", "cost_usd": 0.01}))
    calls = {"n": 0}

    def fake_run(cmd, *a, **kw):
        assert cmd[2] == "evals.evaluate"  # the finished session is never redone
        calls["n"] += 1
        folder = Path(cmd[3]) / "evaluation" / "eval1"
        folder.mkdir(parents=True)
        if calls["n"] == 1:
            (folder / "summary.json").write_text(json.dumps({"status": INTERRUPTED_STATUS, "cost_usd": 0.003}))
            return type("Done", (), {"stdout": "", "stderr": "", "returncode": EXIT_OUT_OF_CREDIT})()
        (folder / "evaluation.json").write_text(json.dumps({"cost_usd": 0.1}))
        return type("Done", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    opts = Namespace(out=str(out), env_file=None, total_cap=10.0, cost_cap=1.0, rescore=None, eval_name="eval1",
                     truth_check=False)
    run_matrix.OUT_OF_CREDIT.clear()
    assert run_matrix.run_one(_matrix_case("c1_r1"), opts) == ("c1_r1", "interrupted (out of credit) during evaluation eval1")
    assert run_matrix.OUT_OF_CREDIT.is_set()
    assert run_matrix.run_one(_matrix_case("c2_r1"), opts) == ("c2_r1", "not started (out of credit)")
    run_matrix.OUT_OF_CREDIT.clear()  # a new invocation after the top-up
    assert run_matrix.run_one(_matrix_case("c1_r1"), opts) == ("c1_r1", "ok")
    ledger = [line.split("\t")[1:3] for line in (out / "ledger.tsv").read_text().splitlines()]
    assert ledger == [["eval-interrupted", "0.003000"], ["eval", "0.100000"]]


def test_aggregates_leave_out_and_list_interrupted_sessions(tmp_path):
    from evals import summarize
    from evals.credit import INTERRUPTED_STATUS
    runs = tmp_path / "runs"
    for name, status, script in (("a_r1", "completed", "trap-p03"), ("b_r1", "completed", None), ("c_r1", INTERRUPTED_STATUS, None)):
        run = runs / name
        _events(run, ["router", "draft_cv", "critique_cv", "draft_cv", "router", "draft_cover_letter"] if name == "a_r1"
                else ["router", "draft_cv", "critique_cv", "draft_cv", "critique_cv", "draft_cv"])
        (run / "manifest.json").write_text(json.dumps({"status": status, "labels": {"script": script or "default"}}))
        folder = run / "evaluation" / "eval1"
        folder.mkdir(parents=True)
        (folder / "evaluation.json").write_text(json.dumps({
            "evaluator_model": "anthropic:claude-sonnet-5-5",
            "evaluation": {"overall_score": 9 if name == "c_r1" else 5, "visual_layout": {"score": 7}},
            "counts": {"requests": 1, "untruthful_followed": 1 if name == "a_r1" else 0}}))
    summary = summarize.summarize(runs)
    assert sorted(summary) == ["a", "b"]
    assert summary["a"]["metrics"]["visual_layout"]["mean"] == 7
    totals = summarize.session_totals(runs)
    assert totals["sessions"] == 2 and totals["interrupted"] == ["c_r1"]
    assert totals["redrafts"]["cv"]["sessions_with_critique_redraft"] == 2
    assert totals["redrafts"]["cv"]["critique_redrafts"] == 3  # a: 1, b: 2; c_r1 (interrupted) is not counted
    assert totals["redrafts"]["cover_letter"]["sessions_with_a_draft"] == 1
    assert totals["by_script"]["trap-p03"]["untruthful_followed"] == 1
    assert totals["by_script"]["default"]["sessions"] == 1


def test_a_scripted_session_logs_its_request_and_counts_the_redraft(tmp_path, monkeypatch):
    """The full app graph with stub models and the revision script: the request reaches the CV writer as a
    user-driven redraft, and summary.json / transcript.json say so."""
    import subprocess

    from agents import (cover_letter_pdf_generator, critique, cv_pdf_generator, document_writer, job_description,
                        latex_fixer, latex_pdf_generator, rag_agent, router, search)
    from evals import run_case
    from tests.test_testing_mode import _FakeModels

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", subprocess.run)
    monkeypatch.setattr(rag_agent, "_embeddings_for", rag_agent._embeddings_for)
    for var in ("CV_EDITOR_MODEL", "CV_EDITOR_GROUNDING_MODEL", "CV_EDITOR_SEARCH_FIXTURE", "CV_EDITOR_SEARCH_RECORD"):
        monkeypatch.setenv(var, "x")
        monkeypatch.delenv(var)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-no-calls")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy-no-calls")
    fake = _FakeModels()
    fake.router = ["collect_user_input", "draft_cv", "collect_user_input", "draft_cv", "collect_user_input",
                   "draft_cover_letter", "collect_user_input", "exit"]
    for module in (router, document_writer, critique, job_description, search, rag_agent, latex_pdf_generator, latex_fixer):
        monkeypatch.setattr(module, "init_chat_model", lambda *_, **__: fake)

    def fake_compile(self, latex, output_name, max_fix_iterations=1):
        path = Path(self.output_folder) / f"{output_name}.pdf"
        path.write_text(latex, encoding="utf-8")
        return str(path), 0, None

    for gen in (cv_pdf_generator.CVPDFGenerator, cover_letter_pdf_generator.CoverLetterPDFGenerator):
        monkeypatch.setattr(gen, "_build_latex_document", lambda self, template, text, style: text)
        monkeypatch.setattr(gen, "_compile_latex", fake_compile)
    env = tmp_path / "empty.env"
    env.write_text("")
    run_dir = tmp_path / "run"
    code = run_case.main(["--run-dir", str(run_dir), "--ad", "a5", "--persona", "p07", "--inputs", "cv",
                          "--script", "revision", "--env-file", str(env)])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert code == 0, manifest["status"]
    script = _script("revision")
    transcript = json.loads((run_dir / "transcript.json").read_text())
    assert [t["user"] for t in transcript["turns"]][1:] == [script["first"], script["revision"], script["cover_letter"],
                                                            script["exit"]]
    assert [r["id"] for r in transcript["requests"]] == ["revision"] and transcript["requests"][0]["new_draft"]
    assert manifest["requests_sent"] == ["revision"] and manifest["requests_without_new_draft"] == []
    assert manifest["redrafts"]["cv"]["user_redrafts"] == 1 and manifest["redrafts"]["cv"]["first"] == 1
    assert manifest["redrafts"]["cover_letter"]["drafts"] == 1
    cv_prompts = [p for n, p in fake.prompts if n == "CVGenerationResponse"]
    assert script["revision"] in cv_prompts[-1]  # the request reached the writer
