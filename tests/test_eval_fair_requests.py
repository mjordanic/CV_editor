"""Offline tests: the evaluator grades only scripted requests that were actually sent, and the language
and contact-details scripts reach their request whichever path the app takes.

No model, Tavily or subprocess calls: the evaluator model is stubbed and the run folders are built by hand.
"""
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.test_eval_diversity import _StubEvaluator, _evaluation, _script, _with_pdfs  # noqa: E402

ASK_ORDER = "Would you like me to generate the tailored CV or cover letter first?"
BOTH_DONE = "Both your CV and cover letter have been generated. Any changes?"


def _run(tmp_path, script_name: str, turns: list[tuple[str, str]], requests: list[dict] | None = None) -> Path:
    """A recorded run folder: inputs/script.json as run_case writes it and a transcript of (app, user) turns."""
    run = tmp_path / script_name
    (run / "inputs").mkdir(parents=True)
    (run / "outputs").mkdir()
    (run / "inputs" / "job_ad.txt").write_text("Machine Learning Engineer, Hamburg. The ad is in English.")
    (run / "inputs" / "cv.txt").write_text("Petra, ML engineer, Python, PyTorch.")
    (run / "inputs" / "script.json").write_text(json.dumps(_script(script_name), ensure_ascii=False))
    (run / "outputs" / "final_cv.txt").write_text("Petra - ML engineer")
    (run / "outputs" / "final_cover_letter.txt").write_text("Dear Hiring Team, I am applying ...")
    transcript = [{"turn": i, "waiting_at": ["collect_user_input"], "assistant": a, "user": u}
                  for i, (a, u) in enumerate(turns)]
    (run / "transcript.json").write_text(json.dumps({"turns": transcript, "observations": [],
                                                     "requests": requests or []}, ensure_ascii=False))
    return run


def _published_letter_de_run(tmp_path) -> Path:
    """As in the 2026-11-10 round (p06_a2_all4_letter-de_luna_r1), with the script of that round: the app
    wrote both documents after the first reply, so the German wish (in the cover-letter reply) was never sent."""
    old = {"purpose": "The user asks for the cover letter in German.",
           "correct_behaviour": "The cover letter is written in correct, natural German; the CV stays in the "
                                "language of the ad.",
           "cover_letter": "Thank you. Now please write the cover letter in German - the hiring manager prefers German.",
           "expectations": [{"about": "cover_letter", "expect": "follow",
                             "correct_behaviour": "The final cover letter is entirely in German."}]}
    run = _run(tmp_path, "letter-de", [(ASK_ORDER, "Please write my CV first, then the cover letter."),
                                       (BOTH_DONE, "Thanks, that is all. Please exit.")])
    from evals.run_case import DEFAULT_SCRIPT
    (run / "inputs" / "script.json").write_text(json.dumps({**DEFAULT_SCRIPT, **old}))
    return run


# --- evaluator: sent is decided from the transcript, unsent is "not exercised" ------------------------------
def test_an_unsent_language_request_is_not_exercised_and_never_reaches_the_evaluator(tmp_path):
    from evals.evaluate import build_prompt, scripted_request_status
    run = _published_letter_de_run(tmp_path)
    status = scripted_request_status(run)
    assert [(i["id"], i["status"]) for i in status] == [("cover_letter", "not exercised")]
    values, _ = build_prompt(run)
    # neither the request nor the script's overall expectation (which names German) is in the checklist
    assert "German" not in values["checklist"] and "Overall:" not in values["checklist"]
    assert "never sent" in values["checklist"] and "must not change any score" in values["checklist"]
    assert "German" not in values["conversation"]


def test_an_unsent_language_request_costs_no_points_and_is_counted_apart(tmp_path, monkeypatch):
    import langchain.chat_models
    from evals import evaluate as ev_mod

    class Grader(_StubEvaluator):
        """Grades like a fair evaluator: German is expected only if the checklist asks for it."""
        def with_structured_output(self, schema, include_raw=False):
            from langchain_core.runnables import RunnableLambda

            def answer(messages):
                self.calls.append(messages)
                asked = "German" in messages[1].content
                ev = _evaluation(user_requests={"requests": [], "score": 10})
                ev["language_quality"].update(expected_language_cover_letter="German" if asked else "English",
                                              right_language_cover_letter=not asked, score=5 if asked else 9,
                                              grammar_and_spelling_errors=[])
                return {"parsed": schema.model_validate(ev), "raw": None, "parsing_error": None}
            return RunnableLambda(answer)

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", lambda *_, **__: Grader())
    record = ev_mod.evaluate(_published_letter_de_run(tmp_path), "openai:gpt-6.1-sol", "eval1")
    assert record["evaluation"]["language_quality"]["score"] == 9
    assert record["evaluation"]["user_requests"] == {"requests": [], "score": 10}
    c = record["counts"]
    assert c["requests_not_exercised"] == 1 and c["requests"] == 0 and c["requests_partly"] == 0
    assert c["wrong_language"] == 0
    assert [i["status"] for i in record["scripted_requests"]] == ["not exercised"]


def test_a_sent_language_request_is_graded_normally(tmp_path):
    from evals.evaluate import build_prompt, counts, scripted_request_status
    script = _script("letter-de")
    run = _run(tmp_path, "letter-de", [(ASK_ORDER, script["first"]),
                                       ("Your CV is ready. Anything else?", script["cover_letter"]),
                                       (BOTH_DONE, script["exit"])])
    status = {i["id"]: i["status"] for i in scripted_request_status(run)}
    assert status == {"cover_letter": "sent", "letter-in-german": "replaced"}
    values, _ = build_prompt(run)
    assert values["checklist"].startswith("Overall: The cover letter is written in correct, natural German")
    assert f"The user said: \"{script['cover_letter']}\"" in values["checklist"]
    assert "never sent" not in values["checklist"]
    assert values["checklist"].count("The user said:") == 1  # the fallback request is not graded twice
    assert counts(_evaluation(), scripted_request_status(run))["requests_not_exercised"] == 0


def test_the_fallback_request_is_graded_when_the_letter_came_unasked(tmp_path):
    from evals.evaluate import build_prompt, scripted_request_status
    script = _script("letter-de")
    fallback = script["requests"][0]
    logged = {"id": fallback["id"], "after": "cover_letter", "text": fallback["text"], "expect": "follow",
              "correct_behaviour": fallback["correct_behaviour"], "drafts_before": {"cv": 1, "cover_letter": 1},
              "drafts_after": {"cv": 1, "cover_letter": 2}, "new_draft": True}
    run = _run(tmp_path, "letter-de", [(ASK_ORDER, script["first"]), (BOTH_DONE, fallback["text"]),
                                       ("Updated. Anything else?", script["exit"])], requests=[logged])
    status = {i["id"]: (i["status"], i.get("replaced_by")) for i in scripted_request_status(run)}
    assert status == {"cover_letter": ("replaced", "letter-in-german"), "letter-in-german": ("sent", None)}
    values, _ = build_prompt(run)
    assert values["checklist"].startswith("Overall: ") and "rewrite the cover letter in German" in values["checklist"]


def test_summaries_count_not_exercised_requests_apart(tmp_path):
    from evals import summarize
    runs = tmp_path / "runs"
    runs.mkdir()
    unsent = _published_letter_de_run(tmp_path)
    unsent.rename(runs / "p06_a2_all4_letter-de_luna_r1")
    script = _script("letter-de")
    sent = _run(tmp_path, "letter-de", [(ASK_ORDER, script["first"]), ("CV ready.", script["cover_letter"]),
                                        (BOTH_DONE, script["exit"])])
    sent.rename(runs / "p10_a2_all4_letter-de_terra_r1")
    for run in runs.iterdir():
        (run / "manifest.json").write_text(json.dumps({"status": "completed", "labels": {"script": "letter-de"}}))
    bucket = summarize.session_totals(runs)["by_script"]["letter-de"]
    assert (bucket["sessions"], bucket["scripted_sent"], bucket["scripted_replaced"],
            bucket["scripted_not_exercised"]) == (2, 1, 1, 1)


def test_a_default_session_has_no_scripted_requests(tmp_path):
    from evals.evaluate import scripted_request_status
    from evals.run_case import DEFAULT_SCRIPT
    run = _run(tmp_path, "letter-de", [(ASK_ORDER, DEFAULT_SCRIPT["first"])])
    (run / "inputs" / "script.json").write_text(json.dumps(DEFAULT_SCRIPT))
    assert scripted_request_status(run) == []


# --- scripted user: the wish reaches the app whichever path it takes ---------------------------------------
@pytest.mark.parametrize("name,language", [("letter-de", "German"), ("letter-hr", "Croatian")])
def test_the_language_wish_is_sent_when_the_letter_is_written_after_the_first_reply(tmp_path, name, language):
    from evals.run_case import ScriptedUser
    script = _script(name)
    s = _with_pdfs(tmp_path)
    user = ScriptedUser(script, "/ad.txt", tmp_path)
    assert user.reply(("collect_user_input",), {}) == script["first"]
    both = {**s["cv"], **s["cl"]}  # the app wrote the CV and the letter at once
    wish = user.reply(("collect_user_input",), both)
    assert language in wish and "cover letter" in wish
    redrafted = {**both, "cover_letter_history": [{}, {}]}
    assert user.reply(("collect_user_input",), redrafted) == script["exit"]
    log = user.finish(redrafted)
    assert [r["id"] for r in log] == [f"letter-in-{language.lower()}"] and log[0]["new_draft"] is True
    assert user.unsent_requests() == [] and user.observations == []


@pytest.mark.parametrize("name,language", [("letter-de", "German"), ("letter-hr", "Croatian")])
def test_the_language_wish_is_sent_once_when_the_app_asks_for_the_letter(tmp_path, name, language):
    from evals.run_case import ScriptedUser
    script = _script(name)
    s = _with_pdfs(tmp_path)
    user = ScriptedUser(script, "/ad.txt", tmp_path)
    first = user.reply(("collect_user_input",), {})
    assert "CV first" in first and "then the cover letter" not in first
    assert user.reply(("collect_user_input",), s["cv"]) == script["cover_letter"] and language in script["cover_letter"]
    assert user.reply(("collect_user_input",), {**s["cv"], **s["cl"]}) == script["exit"]  # no second wish
    assert user.finish({}) == [] and user.unsent_requests() == []


def test_the_contact_details_are_volunteered_when_the_app_never_asks(tmp_path):
    from evals.run_case import ScriptedUser
    script = _script("details-supplied")
    s = _with_pdfs(tmp_path)
    user = ScriptedUser(script, "/ad.txt", tmp_path)
    assert user.reply(("collect_user_input",), {}) == script["first"]
    reply = user.reply(("collect_user_input",), {**s["cv"], **s["cl"]})
    assert "applicant.inbox@example.com" in reply and "+44 7700 900123" in reply
    assert user.reply(("collect_user_input",), {**s["cv"], **s["cl"]}) == script["exit"]
    assert [r["id"] for r in user.finish({})] == ["contact-details"]


def test_the_contact_details_are_not_repeated_when_the_app_asked_for_them(tmp_path):
    from evals.run_case import ScriptedUser
    script = _script("details-supplied")
    s = _with_pdfs(tmp_path)
    user = ScriptedUser(script, "/ad.txt", tmp_path)
    user.reply(("collect_user_input",), {})
    assert user.reply(("ask_cv_missing_details",), {}) == script["missing_details"]
    assert user.reply(("collect_user_input",), s["cv"]) == script["cover_letter"]
    assert user.reply(("collect_user_input",), {**s["cv"], **s["cl"]}) == script["exit"]
    assert user.finish({}) == [] and user.unsent_requests() == []


def test_the_default_user_is_unchanged(tmp_path):
    from evals.run_case import DEFAULT_SCRIPT, ScriptedUser
    assert DEFAULT_SCRIPT == {
        "first": "Please write my CV first, then the cover letter.",
        "cover_letter": "Thank you. Now please write the cover letter.",
        "exit": "Thanks, that is all. Please exit.",
        "clarify_job_ad": "That is all the information the ad gives.",
        "missing_details": "I'd rather leave those details out.",
        "max_turns": 10,
    }
    s = _with_pdfs(tmp_path)
    user = ScriptedUser(DEFAULT_SCRIPT, "/ad.txt", tmp_path)
    assert user.reply(("collect_user_input",), {}) == DEFAULT_SCRIPT["first"]
    assert user.reply(("collect_user_input",), {**s["cv"], **s["cl"]}) == DEFAULT_SCRIPT["exit"]


def test_unless_sent_must_name_an_ordinary_reply():
    from evals.run_case import DEFAULT_SCRIPT, validate_script
    with pytest.raises(ValueError):
        validate_script({**DEFAULT_SCRIPT, "requests": [{"id": "x", "after": "cv", "text": "t", "unless_sent": "nope"}]})
    validate_script({**DEFAULT_SCRIPT, "requests": [{"id": "x", "after": "cv", "text": "t",
                                                     "unless_sent": "missing_details"}]})


def test_details_supplied_cases_give_no_contact_details():
    """The ask for missing details can only happen when the inputs lack them: no e-mail, phone or LinkedIn."""
    from evals.cases import load_matrix, resolve_case
    cases = [c for c in load_matrix() if c.get("script") == "details-supplied"]
    assert cases
    contact = re.compile(r"[\w.+-]+@[\w-]+\.\w+|\+\d[\d ()-]{7,}\d|linkedin", re.I)
    for case in cases:
        for path in resolve_case(case)["inputs"].values():
            assert not contact.search(path.read_text(encoding="utf-8")), (case["id"], path.name)


def test_a_partly_exercised_script_keeps_its_overall_expectation_limited_to_what_was_sent(tmp_path):
    """chatty: the first message (with the tone and length wishes) was sent; the app wrote both documents
    at once, so the cover-letter reply was not, and it never asked for missing details."""
    from evals.evaluate import build_prompt, scripted_request_status
    script = _script("chatty")
    run = _run(tmp_path, "chatty", [(ASK_ORDER, script["first"]), (BOTH_DONE, script["exit"])])
    assert {i["id"]: i["status"] for i in scripted_request_status(run)} == {
        "first": "sent", "cover_letter": "not exercised", "missing_details": "not exercised"}
    checklist = build_prompt(run)[0]["checklist"]
    assert checklist.startswith("Overall (judge this only as far as the requests listed below were sent): ")
    assert checklist.count("The user said:") == 1 and script["cover_letter"] not in checklist
    assert "never sent" in checklist
