"""Chat baseline ("B2"): one model call per user turn, with the whole conversation, for scripted multi-turn sessions.

The fair single-call comparison for what the multi-agent app does across a conversation (evals/scenarios/). Turn 1 is
the single-prompt baseline's prompt (evals/single_prompt.build_messages: the app's own writer, truthfulness and
LaTeX rules, imported), asking for one more field, `message_to_user`. Every later turn appends the previous answer
(all six fields, as JSON) and the user's next message wrapped in the app writer's own "apply only these changes"
rule (agents/document_writer.DocumentWriterAgent._user_request_instructions) plus three chat rules: keep earlier
requests, an appearance request changes only the LaTeX, and the ground-truth rule the app's truthfulness protocol also
states (what the candidate says about themselves in the chat is true; nothing their material and words do not support
or their own words contradict; say what was not applied). Each answer
returns both documents in full, text and LaTeX; each turn's PDFs go through single_prompt.make_pdfs (pdflatex, no
repair call, the fixed plain layout as fallback, marked).

The scripted user is run_case's script: the same request list, sent one per turn in the same order ("after": "cv"
requests, then "after": "cover_letter" ones; the first answer already has both documents, so "Now please write the
cover letter" is never needed, and a request with "unless_sent": "cover_letter" is sent). Contact details: when the
script has its own "missing_details" answer (the user gives details only if asked), the prompt drops the
single-prompt baseline's "the candidate declined" paragraph and, without a CV, uses the app's own no-CV sentence
(placeholders such as [EMAIL]); the user answers when the model's message asks for an e-mail, phone or contact
details (at most twice). `--ask-hint` ("B2+") adds one line telling the model to ask for missing details.

Run folder: run_case's layout (manifest.json, config.json, inputs/ with script.json, llm_calls.jsonl, events.jsonl,
transcript.json, summary.json, outputs/final_cv.txt/.tex/.pdf, final_cover_letter.txt/.tex/.pdf), plus
outputs/turns.json (every turn's documents, message and PDF result, the shape evals/score_checks.py reads) and
workspace/turns/NN-<id>/ (each turn's LaTeX and PDFs). The manifest says "method": "single_prompt_chat".

Usage (from the repo root; one paid call per user turn):
    .venv/bin/python -m evals.single_prompt_chat --run-dir RUN --ad a3 --persona p03 --script evals/scenarios/s2-retitle-p03.json \\
        [--inputs all4|cv_only|no_cv] [--ask-hint] [--model openai:gpt-6-luna] [--env-file PATH/.env] [--cost-cap 0.5]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from pydantic import Field  # noqa: E402

from evals import single_prompt as sp  # noqa: E402
from evals.run_case import DEFAULT_SCRIPT, script_requests  # noqa: E402

METHOD = "single_prompt_chat"
DEFAULT_MODEL = "openai:gpt-6-luna"
NODE = "single_prompt_chat"
ASK_HINT = ("If a document needs a detail the candidate has not given (such as contact details), leave a placeholder "
            "such as [EMAIL] and ask the candidate for it in message_to_user.")
CHAT_RULES = (
    "Also:\n"
    "- Keep every change the candidate asked for earlier in this chat.\n"
    "- A request about how a PDF looks changes only the LaTeX, never the plain text.\n"
    "- What the candidate tells you about themselves in this chat is true, even if their documents do not say it: "
    "use it. Do not write anything about them that neither their material nor their own words support, or that their "
    "own words contradict; if you do not apply part of a request, say so in message_to_user.\n"
    "Return all six fields in full: both documents, text and LaTeX, even the parts you did not change.")
MESSAGE_FIELD = ("6. message_to_user: what you say to the candidate in this chat (what you did, anything you did not "
                 "do and why, and any question for them).")
_ASKS_DETAILS = re.compile(r"e-?mail|phone|contact (details|information|info)", re.I)
# A request for the details, or a pointer to them ("Replace [EMAIL] with your contact details"): a user would answer.
_IS_REQUEST = re.compile(r"\?|\b(please|could you|can you|let me know|replace|provide|share|send)\b|\[[A-Z][A-Z _/-]{2,}\]",
                         re.I)
MAX_DETAIL_ANSWERS = 2


class ChatTurnResponse(sp.SinglePromptResponse):
    """The single-prompt answer plus what the model says to the candidate."""
    message_to_user: str = Field(..., description="What you say to the candidate this turn: what you changed, anything "
                                                  "you did not do and why, and any question for them.")


def gives_details_only_if_asked(script: dict) -> bool:
    """The script's user has their own answer to a request for contact details (not the default decline)."""
    return script.get("missing_details") != DEFAULT_SCRIPT["missing_details"]


def _drop_paragraph(text: str, start: str) -> str:
    """Remove the prompt paragraph that starts with `start` (up to the next "**" paragraph)."""
    i = text.find(start)
    if i < 0:
        raise ValueError(f"single-prompt prompt changed: no '{start}' paragraph")
    j = text.find("\n**", i + len(start))
    return text[:i] + text[j + 1:]


def first_messages(given: dict, job_ad: str, company: str, work_preference: str | None = None, *,
                   details_if_asked: bool = False, ask_hint: bool = False) -> list:
    """Turn 1: the single-prompt baseline's prompt, with message_to_user added (and the details rules, see above)."""
    from agents.cv_writer import NO_CV_TEXT as APP_NO_CV_TEXT
    system, human = sp.build_messages(given, job_ad, company, work_preference)
    text = human.content
    if details_if_asked:
        text = _drop_paragraph(text, "**Missing contact details:**")
        if not given.get("cv"):
            text = text.replace(sp.NO_CV_TEXT, APP_NO_CV_TEXT)
    if "Return exactly five fields:" not in text:
        raise ValueError("single-prompt prompt changed: no 'Return exactly five fields:'")
    text = text.replace("Return exactly five fields:", "Return exactly six fields:") + "\n" + MESSAGE_FIELD
    return [SystemMessage(content=system.content + ("\n\n" + ASK_HINT if ask_hint else "")), HumanMessage(content=text)]


def followup_message(text: str) -> HumanMessage:
    """A later user turn: the user's words inside the app writer's own preservation rule, plus the chat rules."""
    from agents.document_writer import DocumentWriterAgent
    rule = DocumentWriterAgent._user_request_instructions(SimpleNamespace(label="documents"), text, True)
    return HumanMessage(content=f"The candidate's new message:\n{rule}\n\n{CHAT_RULES}")


class ChatUser:
    """run_case's scripted user for a system whose first answer has both documents: the script's requests one per
    turn in run_case's order, and the script's contact details when the model asks for them (only-if-asked scripts)."""

    def __init__(self, script: dict):
        self.script = script
        self.details_if_asked = gives_details_only_if_asked(script)
        self.pending = ([r for r in script_requests(script) if r["after"] == "cv"]
                        + [r for r in script_requests(script) if r["after"] == "cover_letter"])
        self.used: dict = {"first": 1}
        self.request_log: list = []

    def asks_for_details(self, said: str) -> bool:
        return bool(_ASKS_DETAILS.search(said or "") and _IS_REQUEST.search(said or ""))

    def next(self, said: str) -> tuple[str, str] | None:
        """(id, text) of the user's next message after the model said `said`, or None: the user is done."""
        if (self.details_if_asked and self.used.get("missing_details", 0) < MAX_DETAIL_ANSWERS
                and self.asks_for_details(said)):
            self.used["missing_details"] = self.used.get("missing_details", 0) + 1
            return "missing_details", self.script["missing_details"]
        while self.pending:
            r = self.pending.pop(0)
            if r.get("unless_sent") and self.used.get(r["unless_sent"]):
                continue
            self.used[r["id"]] = 1
            self.request_log.append({"id": r["id"], "after": r["after"], "text": r["text"],
                                     "expect": r.get("expect", "follow"),
                                     "correct_behaviour": r.get("correct_behaviour", "")})
            return r["id"], r["text"]
        return None


def call(llm, messages: list, callbacks: list) -> ChatTurnResponse:
    """One turn's model call (no retry; single_prompt.structured_call)."""
    return sp.structured_call(llm, ChatTurnResponse, messages, callbacks, NODE)[0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="new (or empty) folder for this run's records")
    parser.add_argument("--ad", required=True, help="tracked fixture ad id (e.g. a3)")
    parser.add_argument("--persona", required=True, help="tracked fixture persona id (e.g. p03)")
    parser.add_argument("--inputs", help="all4 (default), cv_only, no_cv, or e.g. cv,notes")
    parser.add_argument("--script", required=True, help="scenario script (a file, or a name in evals/scripts/)")
    parser.add_argument("--ask-hint", action="store_true", help="B2+: tell the model to ask for missing details")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"the model of every turn (default {DEFAULT_MODEL})")
    parser.add_argument("--work-preference", dest="work_preference", choices=["remote", "hybrid", "on-site", "no preference"])
    parser.add_argument("--env-file", help=".env file with API keys (read, never copied)")
    parser.add_argument("--cost-cap", type=float, default=0.5, help="stop the session past this many USD (default 0.5)")
    parser.add_argument("--label", action="append", help="key=value stored in manifest.json (repeatable)")
    args = parser.parse_args(argv)
    for key in ("job_ad", "cv", "portfolio", "cover_letter", "notes", "search_fixture", "record_search",
                "critic_model", "grounding_model"):
        setattr(args, key, None)
    args.no_search = False
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    sp.check_model(args.model)
    from evals.run_case import _git, apply_fixture_defaults, prepare_run_folder
    apply_fixture_defaults(args)
    manifest = prepare_run_folder(args)
    run_dir = Path(manifest["run_dir"])
    script = manifest["script"]
    if script["first"] != DEFAULT_SCRIPT["first"]:
        sys.exit("the chat baseline's first turn is the single prompt, which carries the default 'first' message; "
                 "a script cannot change it")
    system = "b2_ask" if args.ask_hint else "b2"
    manifest.update(method=METHOD, model=args.model, model_override=args.model, ask_hint=args.ask_hint,
                    labels={"system": system, **manifest["labels"], "method": METHOD})
    for key in ("critic_model_override", "grounding_model_override"):
        manifest.pop(key, None)
    from dotenv import load_dotenv
    load_dotenv(args.env_file or REPO / ".env")  # keys stay in this process; nothing from the env is written

    from agents.pdf_text_check import check_pdf
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS, CreditGuard, OutOfCredit, install_no_retry_patch
    from evals.evaluate import company_text
    from provenance import RunRecorder
    recorder = RunRecorder(run_dir, cost_cap_usd=args.cost_cap)
    install_no_retry_patch()
    callbacks = [recorder.callback, CreditGuard()]
    recorder.write_json("config.json", {
        "method": METHOD, "topology": "one model call per user turn, full history; PDFs by pdflatex and a fixed "
                                      "fallback layout, no repair call",
        "generation": {"model": args.model, "ask_hint": args.ask_hint,
                       "details_only_if_asked": gives_details_only_if_asked(script)},
        "code": {"repo": str(REPO), "commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))},
    })
    recorder.write_json("manifest.json", {**manifest, "status": "running"})
    recorder.event("run_start", manifest=manifest)

    inputs = run_dir / "inputs"
    messages = first_messages(sp.read_given_inputs(inputs, manifest["inputs_given"]),
                              (inputs / "job_ad.txt").read_text(encoding="utf-8"),
                              company_text(inputs / "search_fixture.json"), args.work_preference,
                              details_if_asked=gives_details_only_if_asked(script), ask_hint=args.ask_hint)
    recorder.write_json("prompt.json", [{"role": m.type, "content": m.content} for m in messages])

    user = ChatUser(script)
    turns, transcript, status, answer, pdfs, failure, finish_reason = [], [], "completed", None, {}, None, None
    rid, text, said_before = "first", script["first"], None
    t0 = time.time()
    try:
        for n in range(int(script.get("max_turns", 10)) + 1):
            if n == int(script.get("max_turns", 10)):
                status = "stopped: too many turns"
                break
            if n:
                messages.append(followup_message(text))
            transcript.append({"turn": n, "waiting_at": [], "assistant": said_before, "user": text})
            recorder.event("user_reply", turn=n, text=text)
            cost_before, started = recorder.cost, time.time()
            answer = call(sp.init_model(args.model), messages, callbacks)
            messages.append(AIMessage(content=json.dumps(answer.model_dump(), ensure_ascii=False)))
            pdfs = sp.make_pdfs({"cv": answer.cv_content, "cover_letter": answer.cover_letter_content},
                                {"cv": answer.cv_latex, "cover_letter": answer.cover_letter_latex},
                                run_dir / "workspace" / "turns" / f"{n:02d}-{rid}")
            turns.append({"id": rid, "user": text, "assistant": answer.message_to_user,
                          "docs": {"cv": answer.cv_content, "cover_letter": answer.cover_letter_content},
                          "pdf": {k: {"compiled": v["compiled"], "fallback": v["source"] == "fallback_layout",
                                      "fixes": 0} for k, v in pdfs.items()},
                          "seconds": round(time.time() - started, 1), "usd": round(recorder.cost - cost_before, 6)})
            recorder.event("turn", id=rid, compiled={k: v["compiled"] for k, v in pdfs.items()},
                           message_to_user=answer.message_to_user)
            if recorder.over_cap():
                status = f"stopped: cost cap ${recorder.cost:.4f} > ${args.cost_cap}"
                break
            said_before = answer.message_to_user
            nxt = user.next(answer.message_to_user)
            if nxt is None:
                break
            rid, text = nxt
    except OutOfCredit as e:
        status = INTERRUPTED_STATUS
        recorder.record_error("session", e)
    except Exception as e:  # noqa: BLE001 - record what happened
        status, failure, finish_reason = sp.failed_session(e)
        recorder.record_error("session", e)

    out = run_dir / "outputs"
    out.mkdir(exist_ok=True)
    files = {}
    if answer is not None:
        for kind in sp.KINDS:
            stem = sp.PDF_STEMS[kind]
            (out / f"{stem}.txt").write_text(getattr(answer, f"{kind}_content"), encoding="utf-8")
            (out / f"{stem}.tex").write_text(getattr(answer, f"{kind}_latex"), encoding="utf-8")
            files[f"generated_{kind}"] = f"outputs/{stem}.txt"
            entry = pdfs.get(kind) or {}
            if entry.get("pdf") and Path(entry["pdf"]).is_file():
                shutil.copy(entry["pdf"], out / f"{stem}.pdf")
                files[f"{kind}_pdf_path"] = f"outputs/{stem}.pdf"
                entry["check"] = check_pdf(out / f"{stem}.pdf", getattr(answer, f"{kind}_content"))
        (out / "writer_notes.txt").write_text(answer.notes or "", encoding="utf-8")
    recorder.write_json("outputs/pdf_checks.json", pdfs)
    recorder.write_json("outputs/turns.json", turns)
    recorder.write_json("transcript.json", {"turns": transcript, "final_message": said_before if answer is None
                                            else answer.message_to_user, "observations": [],
                                            "requests": user.request_log})
    calls = recorder.totals["llm_calls"] + recorder.totals["llm_errors"]
    summary = {"status": status, "method": METHOD, "system": system, "failure": failure, "finish_reason": finish_reason,
               "minutes": round((time.time() - t0) / 60, 2), "outputs": files, "calls": calls,
               "user_turns": len(turns), "requests_sent": [r["id"] for r in user.request_log],
               "requests_not_sent": [r["id"] for r in user.pending],
               "compile_failures": sum(not p["compiled"] for t in turns for p in t["pdf"].values())}
    recorder.write_summary(**summary)
    recorder.write_json("manifest.json", {**manifest, **summary, "cost_usd": round(recorder.cost, 6),
                                          "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    recorder.event("run_end", **summary, cost_usd=round(recorder.cost, 6))
    print(json.dumps({"run_dir": str(run_dir), "status": status, "method": METHOD, "system": system, "calls": calls,
                      "cost_usd": round(recorder.cost, 4), "user_turns": len(turns)}))
    if status == INTERRUPTED_STATUS:
        return EXIT_OUT_OF_CREDIT
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
