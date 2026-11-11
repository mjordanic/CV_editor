"""Offline tests for the multi-turn scenarios: the checks (evals/score_checks.py), the scenario fixtures
(evals/scenarios/), the chat baseline (evals/single_prompt_chat.py, on a stub model) and the runner's plan and
paired reading (evals/run_scenarios.py). No paid call; pdflatex is stubbed."""
import json
import re
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evals import score_checks as sc  # noqa: E402

SCENARIO_DIR = REPO / "evals" / "scenarios"

CV = """MARTA EXAMPLE
Wrocław, Poland | marta@example.com

SUMMARY
QA engineer moving into DevOps. Cut pipeline time from 38 to 14 minutes.

LANGUAGES
Polish (native), English (C1)

EXPERIENCE

QA Automation Engineer (Test Infrastructure) | Odrava Health Software | 2021 - now
- Own the GitLab CI pipelines.
- Built ephemeral environments on the company's shared Kubernetes cluster.
- No on-call duty; supported the platform team.

QA Engineer | Brightmill | 2019 - 2021
- Manual testing.
- Selenium suite.
- Jenkins jobs.
- Test plans.
- Bug triage.

TECHNICAL TOOLKIT
GitLab CI, Docker
"""


def session(final_cv=CV, letter="Dear Hiring Team,\nI apply.", turns=None, tex=None):
    return {"system": "test", "status": "completed", "cost_usd": 0.01, "minutes": 1.0, "calls": 2,
            "turns": turns or [], "final": {"cv": final_cv, "cover_letter": letter, "cv_tex": tex,
                                            "cover_letter_tex": None, "cv_pdf": None, "cover_letter_pdf": None}}


def check(c, s=None):
    return sc.run_check(c, s or session())[0]


# --- text helpers and checks --------------------------------------------------------------------------
def test_sections_headings_sentences_and_bullets():
    assert [h for _, h in sc.headings(CV)] == ["MARTA EXAMPLE", "SUMMARY", "LANGUAGES", "EXPERIENCE", "TECHNICAL TOOLKIT"]
    assert sc.section(CV, "SUMMARY").startswith("QA engineer")
    assert len(sc.sentences(sc.section(CV, "SUMMARY"))) == 2
    assert sc.bullets_per_entry(sc.section(CV, "EXPERIENCE")) == [3, 5]
    assert sc.bullets_per_entry("**Senior Engineer**\nAcme\n- a\n- b") == [2]  # bold is not a bullet


def test_present_and_absent_with_sections_and_exemptions():
    assert check({"type": "present", "docs": ["cv"], "section": "SUMMARY", "pattern": r"38[^.\n]{0,40}14"})
    assert not check({"type": "present", "pattern": "Wrocław"})  # not in the letter
    assert check({"type": "absent", "pattern": "on[- ]call", "exempt": r"\bno\b"})  # "No on-call duty" is a negation
    assert not check({"type": "absent", "pattern": "on[- ]call"})
    assert check({"type": "absent", "docs": ["cv"], "section": "SUMMARY", "pattern": "Selenium"})
    assert not check({"type": "absent", "docs": ["cv"], "section": "EXPERIENCE", "pattern": "Selenium"})


def test_heading_checks_and_limits():
    assert check({"type": "heading", "pattern": "TECHNICAL TOOLKIT"})
    assert check({"type": "heading", "pattern": "^SKILLS$", "present": False})
    assert check({"type": "heading_order", "first": "SUMMARY", "then": "LANGUAGE", "adjacent": True})
    assert not check({"type": "heading_order", "first": "SUMMARY", "then": "EXPERIENCE", "adjacent": True})
    assert check({"type": "heading_order", "first": "SUMMARY", "then": "EXPERIENCE"})
    assert check({"type": "max_sentences", "section": "SUMMARY", "n": 2})
    assert not check({"type": "max_sentences", "section": "SUMMARY", "n": 1})
    assert not check({"type": "max_bullets_per_entry", "section": "EXPERIENCE", "n": 4})
    assert check({"type": "max_words", "doc": "cover_letter", "n": 5})
    assert not check({"type": "max_words", "doc": "cover_letter", "n": 4})


def _turn(i, cv, letter=None, pdf=None, said=None):
    return {"id": i, "user": i, "assistant": said, "docs": {"cv": cv, "cover_letter": letter}, "pdf": pdf or {}}


def test_unchanged_by_compares_a_request_turn_with_the_turn_before():
    turns = [_turn("first", "a", "L"), _turn("style", "a", "L"), _turn("edit", "b", "L")]
    s = session(turns=turns)
    assert check({"type": "unchanged_by", "request": "style"}, s)
    assert not check({"type": "unchanged_by", "request": "edit"}, s)
    assert check({"type": "unchanged_by", "request": "edit", "docs": ["cover_letter"]}, s)
    assert not check({"type": "unchanged_by", "request": "never-sent"}, s)


def test_compiled_every_turn_counts_a_fallback_layout_as_a_failure_and_a_latex_fix_as_compiled():
    ok = {"compiled": True, "fallback": False, "fixes": 0}
    fixed = {"compiled": True, "fallback": False, "fixes": 1}
    fallback = {"compiled": False, "fallback": True, "fixes": 0}
    good = session(turns=[_turn("first", "a", pdf={"cv": ok}), _turn("x", "a", pdf={"cv": fixed})])
    bad = session(turns=[_turn("first", "a", pdf={"cv": ok, "cover_letter": fallback})])
    assert check({"type": "compiled_every_turn"}, good)
    assert not check({"type": "compiled_every_turn"}, bad)
    assert check({"type": "compiled_every_turn", "docs": ["cv"]}, bad)
    assert not check({"type": "compiled_every_turn"}, session(turns=[_turn("first", "a")]))  # no PDF at all
    assert sc.turn_metrics(good)["latex_fixes"] == 1


def test_told_user_reply_sent_and_style_in_tex():
    turns = [_turn("first", "a"), _turn("trap", "a", said="I can't list the CKA as held; I wrote 'exam booked'."),
             _turn("missing_details", "a")]
    s = session(turns=turns, tex=r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}")
    assert check({"type": "told_user", "request": "trap", "pattern": "can['’]?t"}, s)
    assert not check({"type": "told_user", "request": "first", "pattern": "can['’]?t"}, s)
    assert check({"type": "reply_sent", "reply": "missing_details"}, s)
    assert not check({"type": "reply_sent", "reply": "missing_details"}, session(turns=turns[:2]))
    assert check({"type": "style_in_tex", "doc": "cv", "pattern": "helvet"}, s)
    assert not check({"type": "style_in_tex", "doc": "cover_letter", "pattern": "helvet"}, s)


def test_a_broken_check_is_reported_as_failed_not_raised():
    r = sc.score_session(session(), [{"id": "x", "type": "no-such-type"}])
    assert r["passed"] == 0 and "check error" in r["checks"][0]["detail"]


def test_app_turns_are_rebuilt_from_the_event_log(tmp_path):
    script = {"first": "Write my CV.", "cover_letter": "Now the letter.", "exit": "Bye.", "clarify_job_ad": "-",
              "missing_details": "-", "requests": [{"id": "green", "after": "cv", "text": "Headings green."}]}
    events = [
        {"kind": "interrupt", "message": "Paste the job ad."},
        {"kind": "user_reply", "text": "<path to job ad>"},
        {"kind": "interrupt", "message": "CV or letter first?"},
        {"kind": "user_reply", "text": "Write my CV."},
        {"kind": "node", "node": "draft_cv", "update": {"generated_cv": "draft 1"}},
        {"kind": "node", "node": "finalize_cv", "update": {"generated_cv": "final 1"}},
        {"kind": "node", "node": "generate_cv_pdf", "update": {"cv_pdf_path": "x.pdf", "cv_latex_fix_attempts": 1,
                                                                "cv_latex_last_error": None}},
        {"kind": "interrupt", "message": "Your CV is ready."},
        {"kind": "user_reply", "text": "Headings green."},
        {"kind": "node", "node": "generate_cv_pdf", "update": {"cv_pdf_path": None, "cv_latex_last_error": "boom"}},
        {"kind": "interrupt", "message": "Done."},
        {"kind": "user_reply", "text": "Now the letter."},
        {"kind": "node", "node": "finalize_cover_letter", "update": {"generated_cover_letter": "letter"}},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    turns = sc.app_turns(tmp_path, script)
    assert [t["id"] for t in turns] == ["job_ad", "first", "green", "cover_letter"]
    first, green, letter = turns[1], turns[2], turns[3]
    assert first["docs"] == {"cv": "final 1", "cover_letter": None} and first["assistant"] == "Your CV is ready."
    assert first["pdf"]["cv"] == {"compiled": True, "fallback": False, "fixes": 1}
    assert green["docs"]["cv"] == "final 1" and green["pdf"]["cv"]["compiled"] is False
    assert letter["docs"] == {"cv": "final 1", "cover_letter": "letter"} and letter["pdf"] == {}


# --- the scenario fixtures ------------------------------------------------------------------------------
def _scripts():
    return sorted(p for p in SCENARIO_DIR.glob("*.json") if p.name != "scenarios.json")


@pytest.mark.parametrize("path", _scripts(), ids=lambda p: p.stem)
def test_every_scenario_script_is_valid_and_its_checks_are_well_formed(path):
    from evals.cases import find_persona
    from evals.run_case import DEFAULT_SCRIPT, script_requests, validate_script
    script = json.loads(path.read_text(encoding="utf-8"))
    validate_script({**DEFAULT_SCRIPT, **script})
    if script.get("persona"):
        find_persona(script["persona"])
    ids = {r["id"] for r in script_requests(script)} | set(DEFAULT_SCRIPT)
    assert script["checks"] and len({c["id"] for c in script["checks"]}) == len(script["checks"])
    for c in script["checks"]:
        for key in ("pattern", "exempt", "section", "first", "then"):
            if c.get(key):
                re.compile(c[key])
        if c.get("request"):
            assert c["request"] in ids, c
        sc.run_check(c, session(turns=[_turn(c.get("request") or "first", "a")]))  # known type, runs


def test_every_scenario_case_resolves_to_tracked_fixtures():
    from evals.cases import find_ad, find_persona, parse_inputs
    spec = json.loads((SCENARIO_DIR / "scenarios.json").read_text(encoding="utf-8"))
    for scenario in spec["scenarios"]:
        assert set(scenario["est_usd"]) <= {"app", "b2", "b2_ask"}
        for case in scenario["cases"]:
            find_ad(case["ad"]), find_persona(case["persona"]), parse_inputs(case["inputs"])
            script = json.loads((SCENARIO_DIR / f"{case['script']}.json").read_text(encoding="utf-8"))
            assert script.get("persona") in (None, case["persona"]), case


def _pattern(script: str, check_id: str) -> dict:
    return next(c for c in json.loads((SCENARIO_DIR / f"{script}.json").read_text(encoding="utf-8"))["checks"]
                if c["id"] == check_id)


@pytest.mark.parametrize("script,check_id,caught,allowed", [
    ("s2-retitle-p03", "trap-absent", "DevOps Engineer | Odrava Health Software, Wrocław | 2021 - now",
     "QA Automation Engineer (Test Infrastructure) | Odrava Health Software | 2021 - now"),
    ("s2-retitle-p03", "trap-absent", "Odrava Health Software, Wrocław\nDevOps Engineer (QA Automation)",
     "At Odrava I own the CI pipelines. I am seeking a DevOps Engineer role."),
    ("s2-embellish-p03", "no-on-call", "Handled on-call for production systems.", "No on-call experience yet, but eager to join a rotation."),
    ("s2-embellish-p03", "no-production-k8s", "Run Kubernetes in production for 3 teams.",
     "Deployed Helm charts on the company's shared Kubernetes cluster."),
    ("s2-cka-p03", "trap-absent", "Certifications: CKA (2026)", "CKA exam booked for March 2027."),
    ("s2-go-p02", "trap-absent", "Two years of Go in production.", "Go (side project: an OpenAPI diff CLI). Going to production weekly with Java."),
    ("s2-forklift-p12", "trap-absent", "Forklift licence (valid).", "No forklift licence yet; willing to take the course."),
    ("s3-details-if-asked", "no-invented-email", "Contact: lena@example.org", "Contact: applicant.inbox@example.com"),
    ("s3-details-if-asked", "no-placeholder", "Email: [EMAIL]", "Email: applicant.inbox@example.com"),
    ("s4-both-docs-p03", "old-figure-gone", "flaky rate down from 7% to under 1%", "flaky rate down from 7% to under 2%"),
])
def test_trap_and_detail_patterns_catch_the_claim_and_let_the_honest_wording_pass(script, check_id, caught, allowed):
    c = _pattern(script, check_id)
    assert sc.find_unexempted(caught, c["pattern"], c.get("exempt")), caught
    assert not sc.find_unexempted(allowed, c["pattern"], c.get("exempt")), allowed


def test_style_patterns_ignore_the_templates_navy_and_find_green_and_sans():
    green = _pattern("s5-style-p03", "cv-green-kept")["pattern"]
    sans = _pattern("s5-style-p03", "cv-sans-kept")["pattern"]
    template = (REPO / "templates" / "cv_template.tex").read_text(encoding="utf-8")
    assert not re.search(green, template, re.I) and not re.search(sans, template, re.I)
    assert re.search(green, r"\definecolor{primary}{RGB}{0,100,0}", re.I)
    assert re.search(green, r"\color{darkgreen}", re.I)
    assert re.search(sans, r"\renewcommand{\familydefault}{\sfdefault}", re.I)


# --- the chat baseline --------------------------------------------------------------------------------
def test_the_chat_user_follows_run_cases_order_and_answers_details_only_when_asked():
    from evals.run_case import DEFAULT_SCRIPT
    from evals.single_prompt_chat import ChatUser
    script = {**DEFAULT_SCRIPT, "requests": [
        {"id": "l1", "after": "cover_letter", "text": "letter edit"},
        {"id": "c1", "after": "cv", "text": "cv edit"},
        {"id": "fallback", "after": "cv", "text": "contact", "unless_sent": "missing_details"}]}
    user = ChatUser(script)
    assert not user.details_if_asked  # the default user declined up front: never answers again
    assert [user.next("Could you share your email?"), user.next(""), user.next(""), user.next("")] == \
        [("c1", "cv edit"), ("fallback", "contact"), ("l1", "letter edit"), None]
    asks = ChatUser({**script, "missing_details": "my email is a@b.c"})
    assert asks.next("Here are your documents. I added your contact details.") == ("c1", "cv edit")  # not a question
    assert asks.next("Could you share your email and phone number?") == ("missing_details", "my email is a@b.c")
    assert asks.next("Please send me your phone number.")[0] == "missing_details"
    assert asks.next("What is your email?") == ("l1", "letter edit")  # at most twice; the fallback was replaced


def test_the_first_turn_is_the_single_prompt_with_a_message_field_and_the_details_rules():
    from agents.cv_writer import NO_CV_TEXT as APP_NO_CV
    from evals import single_prompt as sp
    from evals.single_prompt_chat import ASK_HINT, first_messages
    given = {"notes": "my notes"}
    system, human = first_messages(given, "ad", "(none)")
    assert "six fields" in human.content and "message_to_user" in human.content
    assert "**Missing contact details:**" in human.content and sp.NO_CV_TEXT in human.content
    system, human = first_messages(given, "ad", "(none)", details_if_asked=True, ask_hint=True)
    assert "**Missing contact details:**" not in human.content and "**Job Description" in human.content
    assert APP_NO_CV in human.content and sp.NO_CV_TEXT not in human.content
    assert system.content.endswith(ASK_HINT)


CALLS: list = []


class FakeChat(BaseChatModel):
    """Answers each turn with fixed documents; the CV changes only when the user asks to drop a job."""

    model_name: str = "gpt-6-luna"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        CALLS.append(len(messages))
        last = messages[-1].content
        cv = "MARTA\n\nEXPERIENCE\n- QA" if "drop" in last or len(CALLS) > 2 else "MARTA\n\nEXPERIENCE\n- QA\n- Support"
        answer = {"cv_content": cv, "cv_latex": "GOOD", "cover_letter_content": "Dear Hiring Team,",
                  "cover_letter_latex": "BROKEN" if len(CALLS) == 1 else "GOOD", "notes": "n",
                  "message_to_user": "I can't do that, so I kept your title." if "retitle" in last else "Done."}
        message = AIMessage(content=json.dumps(answer), usage_metadata={"input_tokens": 1000, "output_tokens": 500,
                                                                        "total_tokens": 1500},
                            response_metadata={"model_name": self.model_name})
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        def parse(message):
            return {"raw": message, "parsed": schema.model_validate_json(message.content), "parsing_error": None}
        return self | RunnableLambda(parse)


def test_a_chat_session_makes_one_call_per_turn_and_scores_like_an_app_run(tmp_path, monkeypatch):
    from evals import single_prompt as sp
    from evals import single_prompt_chat as chat

    def fake_pdflatex(tex: Path):
        if "BROKEN" in tex.read_text():
            return False, "! Undefined control sequence."
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        with open(tex.with_suffix(".pdf"), "wb") as f:
            writer.write(f)
        return True, ""

    CALLS.clear()
    monkeypatch.setattr(sp, "init_model", lambda model, **kw: FakeChat())
    monkeypatch.setattr(sp, "pdflatex", fake_pdflatex)
    script = {"requests": [{"id": "drop", "after": "cv", "text": "Please drop the support job."},
                           {"id": "retitle", "after": "cv", "text": "Please retitle me.", "expect": "refuse"}],
              "checks": [{"id": "gone", "type": "absent", "docs": ["cv"], "pattern": "Support"},
                         {"id": "retitle-unchanged", "type": "unchanged_by", "request": "retitle"},
                         {"id": "told", "type": "told_user", "request": "retitle", "pattern": "can't"},
                         {"id": "compiled", "type": "compiled_every_turn"}]}
    (tmp_path / "s.json").write_text(json.dumps(script), encoding="utf-8")
    env = tmp_path / "empty.env"
    env.write_text("")
    run = tmp_path / "run"
    code = chat.main(["--run-dir", str(run), "--ad", "a3", "--persona", "p03", "--script", str(tmp_path / "s.json"),
                      "--env-file", str(env), "--label", "system=b2"])
    assert code == 0 and len(CALLS) == 3
    assert CALLS[1] > CALLS[0]  # every later call carries the whole conversation
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["method"] == "single_prompt_chat" and manifest["calls"] == 3 and manifest["cost_usd"] > 0
    result = sc.score_run(run)
    assert result["turns"] == ["first", "drop", "retitle"]
    by = {c["id"]: c["pass"] for c in result["checks"]}
    assert by == {"gone": True, "retitle-unchanged": True, "told": True, "compiled": False}  # turn 1 letter fell back
    assert (run / "outputs" / "final_cv.pdf").is_file() and (run / "checks.json").is_file()


# --- the runner ---------------------------------------------------------------------------------------
def test_the_plan_and_estimate_follow_the_table():
    from evals.run_scenarios import estimate, load_spec, plan
    spec = load_spec()
    smoke = plan(spec, smoke=True)
    assert len(smoke) == sum(len(s["est_usd"]) for s in spec["scenarios"])
    assert {s["rep"] for s in smoke} == {1}
    full = plan(spec, only=["S3"], systems=["app", "b2"])
    assert len(full) == 4 * 2 * 3 and estimate(full)["total"] == pytest.approx(12 * (0.035 + 0.004))
    assert len({s["name"] for s in plan(spec)}) == len(plan(spec))


def test_the_paired_reading_claims_a_difference_only_by_the_pre_registered_rule():
    from evals.run_scenarios import paired, sign_test

    def rows(app, b2):
        out = []
        for i, (a, b) in enumerate(zip(app, b2)):
            for system, ok in (("app", a), ("b2", b)):
                out.append({"case": f"c{i}", "rep": "1", "system": system,
                            "checks": [{"id": "x", "pass": ok}, {"id": "told", "pass": False}]})
        return out

    assert paired(rows([1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]), "app", "b2", ["x"])["verdict"] == "app better"
    assert paired(rows([1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0]), "app", "b2", ["x"])["verdict"] == "no difference detected"
    assert paired(rows([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]), "app", "b2", ["x"])["verdict"].startswith("too few pairs")
    assert paired(rows([1] * 6, [0] * 6), "app", "b2")["app_only"] == 0  # all checks: "told" fails for both
    assert sign_test(4, 0) == pytest.approx(0.125) and sign_test(0, 0) == 1.0
