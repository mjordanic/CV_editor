"""Testing mode: run one CV_editor session from prepared inputs, with a scripted user, and record it.

Everything the session sees comes from files you give: any subset of the candidate's CV, portfolio,
earlier cover letter and free-form notes, the job ad, and prepared web-search results (a fixture
file per company) that replace live Tavily calls. A scripted user answers the app's questions.
The run folder gets everything needed to reconstruct the session:

    manifest.json        what was run: inputs given and omitted, model, status, cost, time
    config.json          model and settings per agent, workflow settings, code version
    inputs/              exact copies of the inputs given (only those), the job ad, the search fixture
    events.jsonl         every graph step with its state update, router decisions, questions, replies
    llm_calls.jsonl      every model call: node, model, prompt, response, tokens, cost, seconds
    transcript.json      the conversation (app question -> scripted reply)
    outputs/             final CV and cover letter text and PDFs, every draft with its critique and
                         grounding-check result, extracted job ad fields, company research
    summary.json         tokens and cost per model and per node, errors
    workspace/           the app's working folder (its own logs, debug file, drafts and PDFs)

Usage (from the repo root), with the tracked fixtures in evals/fixtures/:
    .venv/bin/python -m evals.run_case --run-dir RUN --ad a3 --persona p03 [--inputs all4|cv_only|no_cv|cv,notes] \\
        [--model openai:gpt-6-luna]
--ad uses the ad's ad.md and search_fixture.json; --persona gives the chosen candidate files (default all four).
Any file can be given explicitly instead (or to override one): --job-ad, --cv, --portfolio, --cover-letter,
--notes, --search-fixture. Without an --ad and without --search-fixture, company research finds nothing,
unless --record-search FILE (live Tavily, saved to FILE for later replays). This makes paid model calls; --cost-cap stops a run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANDIDATE_INPUTS = ("cv", "portfolio", "cover_letter", "notes")
# Where the app looks for each input, relative to its working folder.
_INPUT_TARGET = {
    "cv": ("data/CV", "CV"),
    "cover_letter": ("data/CV", "cover_letter"),
    "notes": ("data/CV", "notes"),
    "portfolio": ("data/portfolio", "portfolio"),
}

DEFAULT_SCRIPT = {
    "first": "Please write my CV first, then the cover letter.",
    "cover_letter": "Thank you. Now please write the cover letter.",
    "exit": "Thanks, that is all. Please exit.",
    "clarify_job_ad": "That is all the information the ad gives.",
    # The app asks for contact details a document still lacks ([EMAIL] and similar placeholders).
    "missing_details": "I'd rather leave those details out.",
    "max_turns": 10,
}
# Optional keys (see evals/scripts/ and evals/README.md):
#   "purpose", "correct_behaviour"   what the script tests and what the system should do (for the evaluator)
#   "persona"                        the persona a persona-specific script is written for (checked by cases.py)
#   "requests"                       list of {"id", "after": "cv" | "cover_letter", "text", "expect": "follow" |
#                                    "refuse", "correct_behaviour"}: change requests sent once each, in order,
#                                    after that document and its PDF exist ("cv": before the cover letter is
#                                    asked for; "cover_letter": before the user exits). Optional "unless_sent":
#                                    a reply key (e.g. "cover_letter", "missing_details"): the request is the
#                                    fallback for a wish that reply carries, sent only if that reply never was
#                                    (the app wrote the letter without being asked for it, or never asked
#                                    for the missing details), so the wish reaches the app either way
#   "expectations"                   list of {"about": a script key (e.g. "first", "cover_letter",
#                                    "missing_details"), "expect", "correct_behaviour"}: what the system should do
#                                    with one of the ordinary replies
#   "revision"                       legacy: one request sent after the CV (same as one "after": "cv" request)
REQUEST_STAGES = ("cv", "cover_letter")
EXPECT = ("follow", "refuse")
REPLY_KEYS = ("first", "cover_letter", "exit", "clarify_job_ad", "missing_details")


def script_requests(script: dict) -> list[dict]:
    """The script's change requests in order, with the legacy "revision" key as one request after the CV."""
    out = []
    if script.get("revision"):
        out.append({"id": "revision", "after": "cv", "text": script["revision"], "expect": "follow",
                    "correct_behaviour": "The CV is changed as asked, and nothing else is changed."})
    out.extend(dict(r) for r in script.get("requests") or [])
    return out


def validate_script(script: dict) -> None:
    """Raise ValueError for a script the scripted user cannot follow."""
    for key in REPLY_KEYS:
        if not isinstance(script.get(key), str) or not script[key].strip():
            raise ValueError(f"script key '{key}' must be a non-empty string")
    ids = set()
    for r in script_requests(script):
        if not {"id", "after", "text"} <= set(r) or not str(r["text"]).strip():
            raise ValueError(f"request {r} needs id, after and text")
        if r["after"] not in REQUEST_STAGES:
            raise ValueError(f"request '{r['id']}': after must be one of {REQUEST_STAGES}")
        if r.get("expect", "follow") not in EXPECT:
            raise ValueError(f"request '{r['id']}': expect must be one of {EXPECT}")
        if r.get("unless_sent") is not None and r["unless_sent"] not in REPLY_KEYS:
            raise ValueError(f"request '{r['id']}': unless_sent must be one of {REPLY_KEYS}")
        if r["id"] in ids or r["id"] in DEFAULT_SCRIPT or r["id"] in ("job_ad", "requests", "expectations"):
            raise ValueError(f"request id '{r['id']}' is repeated or clashes with a script key")
        ids.add(r["id"])
    for e in script.get("expectations") or []:
        if e.get("about") not in DEFAULT_SCRIPT or e.get("expect", "follow") not in EXPECT:
            raise ValueError(f"expectation {e} must name a script reply key and expect follow or refuse")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _package_versions() -> dict:
    """Versions of the packages a session or a grading depends on (pdfplumber makes the PDF text the grader and
    the trap check read, pypdfium2 the page images), and of pdflatex."""
    from importlib import metadata
    out = {}
    for name in ("langgraph", "langchain", "langchain-core", "langchain-openai", "openai", "anthropic", "tavily-python",
                 "tavily", "pdfplumber", "pypdfium2", "pypdf"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    try:
        out["pdflatex"] = subprocess.run(["pdflatex", "--version"], capture_output=True, text=True,
                                         timeout=10).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        out["pdflatex"] = None
    return out


def degradations(values: dict, workspace: Path) -> list[str]:
    """What the app quietly did without in a session that still completed: company research that found nothing or
    failed (the description is the app's "no information" text), a delivered draft whose fact check could not run,
    or a document whose final PDF was not made."""
    from agents.search import NO_COMPANY_INFO
    out = []
    description = str((values.get("company_info") or {}).get("company_description") or "")
    if description == NO_COMPANY_INFO or description.startswith("Could not identify the company"):
        out.append("company_research_failed")
    unchecked = pdf_failed = False
    for kind in ("cv", "cover_letter"):
        history = values.get(f"{kind}_history") or []
        final = values.get(f"{kind}_final_iteration")
        delivered = [d for d in history if d.get("iteration") == final] if final is not None else history[-1:]
        unchecked |= any(d.get("grounding_checked") is False and not d.get("repeats_cv") for d in delivered)
        pdf = values.get(f"{kind}_pdf_path")
        pdf_failed |= bool(values.get(f"generated_{kind}")) and not (pdf and (workspace / pdf).is_file())
    out += ["grounding_unchecked"] * unchecked + ["pdf_failed"] * pdf_failed
    return out


def prepare_run_folder(args) -> dict:
    """Create run folder, inputs/ and workspace/; returns the manifest skeleton."""
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        sys.exit(f"run folder {run_dir} is not empty")
    inputs = run_dir / "inputs"
    workspace = run_dir / "workspace"
    inputs.mkdir(parents=True)
    workspace.mkdir()

    given, omitted = {}, []
    for key in CANDIDATE_INPUTS:
        src = getattr(args, key)
        if not src:
            omitted.append(key)
            continue
        src = Path(src).resolve()
        if not src.is_file():
            sys.exit(f"--{key.replace('_', '-')} {src} is not a file")
        suffix = src.suffix.lower() if src.suffix.lower() in (".txt", ".md", ".pdf") else ".txt"
        if key == "portfolio":
            suffix = ".txt"  # the portfolio index reads plain text
        shutil.copy(src, inputs / f"{key}{suffix}")
        folder, stem = _INPUT_TARGET[key]
        (workspace / folder).mkdir(parents=True, exist_ok=True)
        shutil.copy(src, workspace / folder / f"{stem}{suffix}")
        given[key] = {"source": str(src), "copied_to": f"inputs/{key}{suffix}", "chars": len(src.read_bytes())}
    (workspace / "data" / "CV").mkdir(parents=True, exist_ok=True)

    ad_src = Path(args.job_ad).resolve()
    shutil.copy(ad_src, inputs / "job_ad.txt")
    shutil.copy(ad_src, workspace / "job_ad.txt")

    search = {"mode": None}
    if args.search_fixture:
        shutil.copy(Path(args.search_fixture).resolve(), inputs / "search_fixture.json")
        search = {"mode": "replay", "source": str(Path(args.search_fixture).resolve()), "copied_to": "inputs/search_fixture.json"}
    elif args.record_search:
        search = {"mode": "record", "record_to": str(Path(args.record_search).resolve())}
    else:
        (inputs / "search_fixture.json").write_text(json.dumps({
            "fixture_version": 1, "company": None, "source": "none",
            "searches": [{"query": "*", "response": {"results": []}}],
        }, indent=2), encoding="utf-8")
        search = {"mode": "none", "copied_to": "inputs/search_fixture.json"}

    if args.work_preference:
        # The candidate's own preference is candidate material: kept with the inputs, seen by the evaluator.
        (inputs / "work_preference.txt").write_text(args.work_preference + "\n", encoding="utf-8")

    script = dict(DEFAULT_SCRIPT)
    if args.script:
        from evals.cases import find_script
        script.update(json.loads(find_script(args.script).read_text(encoding="utf-8")))
    try:
        validate_script(script)
    except ValueError as e:
        sys.exit(f"script {args.script}: {e}")
    (inputs / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")

    (workspace / "templates").symlink_to(REPO / "templates")
    labels = dict(kv.split("=", 1) for kv in (args.label or []))
    return {
        "run_dir": str(run_dir),
        "labels": labels,
        "inputs_given": sorted(given),
        "inputs_omitted": omitted,
        "inputs": given,
        "job_ad": {"source": str(ad_src), "copied_to": "inputs/job_ad.txt"},
        "search": search,
        "script": script,
        "model_override": args.model,
        "critic_model_override": args.critic_model,
        "grounding_model_override": args.grounding_model,
        "work_preference": args.work_preference,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def effective_config() -> dict:
    """Model and settings per agent as the app will use them (after CV_EDITOR_MODEL, CV_EDITOR_CRITIC_MODEL and CV_EDITOR_GROUNDING_MODEL)."""
    from config.config_loader import (CRITIC_MODEL_OVERRIDE_ENV, GROUNDING_MODEL_OVERRIDE_ENV, MODEL_OVERRIDE_ENV,
                                      get_agent_config, load_config)
    full = load_config()
    return {
        "workflow": full.get("workflow", {}),
        "agents": {name: get_agent_config(name) for name in full.get("agents", {})},
        "embeddings": os.environ.get("CV_EDITOR_EMBEDDING_MODEL") or "text-embedding-3-small",
        "env_overrides": {k: os.environ.get(k) for k in (MODEL_OVERRIDE_ENV, CRITIC_MODEL_OVERRIDE_ENV, GROUNDING_MODEL_OVERRIDE_ENV, "CV_EDITOR_TEMPERATURE",
                                                          "CV_EDITOR_SEARCH_FIXTURE", "CV_EDITOR_SEARCH_RECORD")
                          if os.environ.get(k)},
        "code": {"repo": str(REPO), "commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(_git("status", "--porcelain"))},
        "python": platform.python_version(),
        "packages": _package_versions(),
    }


def _asks_missing_details(waiting_at: tuple) -> bool:
    """The app's interrupt nodes that ask for contact details a document lacks (ask_*_missing_details)."""
    return any(n.startswith("ask_") and n.endswith("_missing_details") for n in waiting_at)


class ScriptedUser:
    """Answers the app's questions from the script, by what the app is waiting for.

    Stages: the first message until a CV exists; then the "after": "cv" requests one per turn once the
    CV PDF exists; then the cover-letter message until a letter exists; then the "after":
    "cover_letter" requests once its PDF exists; then exit. A request with "unless_sent" is dropped once
    that ordinary reply has been sent (it carried the same wish). Every request sent is logged with the
    number of drafts of each document before it and at the next turn (did the app act on it?).
    """

    def __init__(self, script: dict, job_ad_path: str, workspace: str | Path | None = None):
        self.script = script
        self.job_ad_path = job_ad_path
        self.workspace = Path(workspace) if workspace else Path.cwd()  # PDF paths in the state are relative to it
        self.used: dict = {}
        self.observations: list = []
        self.requests = script_requests(script)
        self.request_log: list = []
        self._skipped_stage: set = set()

    def _use(self, key: str) -> str:
        self.used[key] = self.used.get(key, 0) + 1
        return self.script[key]

    @staticmethod
    def _drafts(values: dict) -> dict:
        return {k: len(values.get(f"{k}_history") or []) for k in REQUEST_STAGES}

    def _close_open_requests(self, values: dict) -> None:
        for entry in self.request_log:
            if entry.get("drafts_after") is None:
                entry["drafts_after"] = self._drafts(values)
                entry["new_draft"] = entry["drafts_after"] != entry["drafts_before"]

    def finish(self, values: dict) -> list:
        """Close the request log with the final state; returns it."""
        self._close_open_requests(values or {})
        return self.request_log

    def _pdf_exists(self, values: dict, kind: str) -> bool:
        pdf = values.get(f"{kind}_pdf_path")
        return bool(pdf) and (self.workspace / pdf).is_file()

    def _replaced(self, request: dict) -> bool:
        """The request's wish already reached the app through the ordinary reply it stands in for."""
        return bool(request.get("unless_sent") and self.used.get(request["unless_sent"]))

    def unsent_requests(self) -> list[str]:
        """Ids of the script's requests that were neither sent nor replaced by their ordinary reply."""
        return [r["id"] for r in self.requests if not self.used.get(r["id"]) and not self._replaced(r)]

    def _next_request(self, stage: str, values: dict) -> str | None:
        pending = [r for r in self.requests
                   if r["after"] == stage and not self.used.get(r["id"]) and not self._replaced(r)]
        if not pending:
            return None
        if not self._pdf_exists(values, stage):
            if stage not in self._skipped_stage:
                self._skipped_stage.add(stage)
                for r in pending:
                    label = "revision" if r["id"] == "revision" else f"request '{r['id']}'"
                    nxt = "the cover letter" if stage == "cv" else "the user exited"
                    self.observations.append(f"the {label} was not sent: there was no {stage.replace('_', ' ').replace('cv', 'CV')} PDF before {nxt}")
            return None
        r = pending[0]
        self.used[r["id"]] = 1
        self.request_log.append({"id": r["id"], "after": stage, "text": r["text"], "expect": r.get("expect", "follow"),
                                 "correct_behaviour": r.get("correct_behaviour", ""),
                                 "drafts_before": self._drafts(values), "drafts_after": None})
        return r["text"]

    def reply(self, waiting_at: tuple, values: dict) -> str:
        self._close_open_requests(values)
        if "analyze_job_description" in waiting_at:
            if self.used.get("job_ad"):
                raise RuntimeError("the app asked for the job ad again (it could not read it)")
            self.used["job_ad"] = 1
            return self.job_ad_path
        if "load_candidate_documents" in waiting_at:
            raise RuntimeError("the app asked for a CV: the input set has no CV and no notes")
        if "clarify_job_ad" in waiting_at:
            return self._use("clarify_job_ad")
        if _asks_missing_details(waiting_at):
            if self.used.get("missing_details", 0) == 2:
                self.observations.append("the app asked for missing details more than twice")
            return self._use("missing_details")
        if not values.get("generated_cv"):
            if self.used.get("first", 0) >= 2:
                self.observations.append("the app still had no CV after the user asked for it twice")
                return self._use("exit")
            return self._use("first")
        if not values.get("generated_cover_letter"):
            request = self._next_request("cv", values)
            if request:
                return request
            if self.used.get("cover_letter", 0) >= 2:
                self.observations.append("the app still had no cover letter after the user asked for it twice")
                return self._use("exit")
            return self._use("cover_letter")
        request = self._next_request("cv", values) or self._next_request("cover_letter", values)
        if request:
            return request
        if self.used.get("exit"):
            self.observations.append("the app kept asking after the user said exit")
        return self._use("exit")


def _write_outputs(recorder, values: dict, workspace: Path) -> dict:
    out = recorder.run_dir / "outputs"
    out.mkdir(exist_ok=True)
    files = {}
    for key, name in (("generated_cv", "final_cv.txt"), ("generated_cover_letter", "final_cover_letter.txt")):
        if values.get(key):
            (out / name).write_text(values[key], encoding="utf-8")
            files[key] = f"outputs/{name}"
    for key, name in (("cv_pdf_path", "final_cv.pdf"), ("cover_letter_pdf_path", "final_cover_letter.pdf")):
        path = values.get(key)
        if path and (workspace / path).is_file():
            shutil.copy(workspace / path, out / name)
            files[key] = f"outputs/{name}"
    recorder.write_json("outputs/cv_history.json", values.get("cv_history") or [])
    recorder.write_json("outputs/cover_letter_history.json", values.get("cover_letter_history") or [])
    recorder.write_json("outputs/job_description_info.json", values.get("job_description_info"))
    recorder.write_json("outputs/company_info.json", values.get("company_info"))
    recorder.write_json("outputs/candidate_text_seen.json", values.get("candidate_text"))
    (out / "relevant_experience.txt").write_text(values.get("relevant_experience") or "", encoding="utf-8")
    recorder.write_json("outputs/final_state.json", values)
    return files


def apply_fixture_defaults(args) -> None:
    """Fill job ad, search fixture and candidate files from --ad / --persona; explicit paths win."""
    from evals.cases import find_ad, find_persona, parse_inputs, INPUT_FILES
    if args.ad:
        ad = find_ad(args.ad)
        args.job_ad = args.job_ad or str(ad / "ad.md")
        if not (args.search_fixture or args.record_search or args.no_search):
            args.search_fixture = str(ad / "search_fixture.json")
    if args.persona:
        persona = find_persona(args.persona)
        for key in parse_inputs(args.inputs):
            if not getattr(args, key):
                setattr(args, key, str(persona / INPUT_FILES[key]))
    elif args.inputs:
        raise ValueError("--inputs needs --persona")
    if not args.job_ad:
        raise ValueError("give --ad or --job-ad")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="new (or empty) folder for this run's records")
    parser.add_argument("--ad", help="tracked fixture ad id (e.g. a3): its ad.md and search_fixture.json")
    parser.add_argument("--persona", help="tracked fixture persona id (e.g. p03): its candidate files")
    parser.add_argument("--inputs", help="with --persona: all4 (default), cv_only, no_cv, or e.g. cv,notes")
    parser.add_argument("--job-ad", help="job ad text file (overrides --ad's ad.md)")
    parser.add_argument("--cv", help="candidate CV (.txt, .md or .pdf)")
    parser.add_argument("--portfolio", help="candidate portfolio (.txt; projects separated by blank lines)")
    parser.add_argument("--cover-letter", dest="cover_letter", help="candidate's earlier cover letter")
    parser.add_argument("--notes", help="candidate's free-form notes (rough CV draft, bullet notes, anything)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--search-fixture", help="prepared web-search results to replay (no live search)")
    group.add_argument("--record-search", help="live Tavily search, saved to this fixture file")
    group.add_argument("--no-search", action="store_true",
                       help="company research finds nothing (the default when neither --ad nor --search-fixture is given)")
    parser.add_argument("--model", help="use this model for every writer-side agent; never the critic's two calls (sets CV_EDITOR_MODEL)")
    parser.add_argument("--critic-model", dest="critic_model",
                        help="use this model for the critic's quality review, 'critique' (sets CV_EDITOR_CRITIC_MODEL)")
    parser.add_argument("--grounding-model", dest="grounding_model",
                        help="use this model for the critic's fact check, 'grounding' (sets CV_EDITOR_GROUNDING_MODEL)")
    parser.add_argument("--work-preference", dest="work_preference", choices=["remote", "hybrid", "on-site", "no preference"],
                        help="the candidate's own work-arrangement preference (default: not given)")
    parser.add_argument("--script", help="JSON with scripted replies overriding the defaults, or a name in "
                                          "evals/scripts/ (e.g. revision)")
    parser.add_argument("--env-file", help=".env file with API keys (read, never copied)")
    parser.add_argument("--cost-cap", type=float, default=1.0, help="stop the run past this many USD (default 1.0)")
    parser.add_argument("--label", action="append", help="key=value stored in manifest.json (repeatable)")
    args = parser.parse_args(argv)
    try:
        apply_fixture_defaults(args)
    except ValueError as e:
        parser.error(str(e))
    if not any(getattr(args, k) for k in ("cv", "notes")):
        print("note: no --cv and no --notes; the app will ask for a CV and the run will stop there", file=sys.stderr)

    manifest = prepare_run_folder(args)
    run_dir = Path(manifest["run_dir"])
    workspace = run_dir / "workspace"

    from dotenv import load_dotenv
    if args.env_file:
        load_dotenv(args.env_file)  # keys stay in this process; nothing from the env is written
    else:
        load_dotenv(REPO / ".env")
    if args.model:
        os.environ["CV_EDITOR_MODEL"] = args.model
    if args.critic_model:
        os.environ["CV_EDITOR_CRITIC_MODEL"] = args.critic_model
    if args.grounding_model:
        os.environ["CV_EDITOR_GROUNDING_MODEL"] = args.grounding_model
    os.environ.pop("CV_EDITOR_SEARCH_FIXTURE", None)
    os.environ.pop("CV_EDITOR_SEARCH_RECORD", None)
    if manifest["search"]["mode"] == "record":
        os.environ["CV_EDITOR_SEARCH_RECORD"] = manifest["search"]["record_to"]
    else:
        os.environ["CV_EDITOR_SEARCH_FIXTURE"] = str(run_dir / "inputs" / "search_fixture.json")

    os.chdir(workspace)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    real_run = subprocess.run

    def run_with_timeout(cmd, *a, **kw):  # a stuck pdflatex must not hang a batch
        if isinstance(cmd, list) and cmd and cmd[0] == "pdflatex":
            kw.setdefault("timeout", 180)
        return real_run(cmd, *a, **kw)

    subprocess.run = run_with_timeout

    from provenance import CountingEmbeddings, RunRecorder, stream_turn
    from evals.credit import (EXIT_OUT_OF_CREDIT, INFRASTRUCTURE_STATUS, INTERRUPTED_STATUS, CreditGuard, OutOfCredit,
                              install_no_retry_patch, is_infrastructure_error)
    recorder = RunRecorder(run_dir, cost_cap_usd=args.cost_cap)
    credit_guard = CreditGuard()  # out of credit: no retries, the session ends at once (evals/credit.py)
    install_no_retry_patch()

    from agents import rag_agent
    real_embeddings_for = rag_agent._embeddings_for

    class CreditCheckedEmbeddings(CountingEmbeddings):
        def _call(self, fn, *a):
            try:
                return fn(*a)
            except Exception as e:  # noqa: BLE001 - only a credit error is turned into OutOfCredit
                credit_guard.guard(e)
                raise

        def embed_documents(self, texts):
            return self._call(super().embed_documents, texts)

        def embed_query(self, text):
            return self._call(super().embed_query, text)

    def counted_embeddings(model):
        inner = real_embeddings_for(model)
        return CreditCheckedEmbeddings(inner, recorder, getattr(inner, "model", "embeddings"))

    rag_agent._embeddings_for = counted_embeddings

    recorder.write_json("config.json", effective_config())
    recorder.write_json("manifest.json", {**manifest, "status": "running"})
    recorder.event("run_start", manifest=manifest)

    from langgraph.types import Command
    import langgraph_agent as la
    import debug_utils
    if not logging.getLogger().handlers:
        la.setup_logging()  # the app's own logs/ and debug/ files in the workspace, as on a normal run
    if not os.path.isfile(debug_utils.get_debug_file_path() or ""):
        debug_utils.initialize_debug_file()  # logging was set up elsewhere: the debug file still goes here

    import signal

    def stop_on_sigterm(signum, frame):  # a stopped run still records what it did, with its status
        raise RuntimeError("stopped by SIGTERM")

    signal.signal(signal.SIGTERM, stop_on_sigterm)

    user = ScriptedUser(manifest["script"], str(workspace / "job_ad.txt"), workspace)
    transcript, status, values = [], "completed", {}
    t0 = time.time()
    agent = None
    config = recorder.graph_config("testing-mode")
    config["callbacks"] = [*config["callbacks"], credit_guard]
    try:
        agent = la.MasterAgent()
        initial = {k: None for k in la.State.__annotations__}
        initial.update(messages=[], cv_refinement_count=0, cover_letter_refinement_count=0,
                       cv_history=[], cover_letter_history=[], work_preference=manifest["work_preference"])
        graph_input = initial
        for turn in range(int(manifest["script"].get("max_turns", 10)) + 1):
            question = stream_turn(agent.graph, graph_input, config, recorder)
            if question is None:
                break
            snapshot = agent.graph.get_state(config)
            if turn == int(manifest["script"].get("max_turns", 10)):
                status = "stopped: too many turns"
                user.observations.append("the session needed more turns than the script allows")
                break
            reply = user.reply(tuple(snapshot.next or ()), snapshot.values)
            shown = "<path to job ad>" if reply == user.job_ad_path else reply
            transcript.append({"turn": turn, "waiting_at": list(snapshot.next or ()),
                               "assistant": (question or {}).get("message"), "user": shown})
            recorder.event("user_reply", turn=turn, text=shown)
            graph_input = Command(resume=reply)
    except OutOfCredit as e:  # not a failure of the app: the run is redone after a top-up
        status = INTERRUPTED_STATUS
        recorder.record_error("session", e)
        user.observations.append(f"interrupted: the model provider reported no credit left ({e})"[:300])
    except Exception as e:  # noqa: BLE001 - record what happened
        # A provider or network failure is not the app's: the session is not graded and run_matrix re-runs it.
        status = INFRASTRUCTURE_STATUS if is_infrastructure_error(e) else f"error: {type(e).__name__}: {e}"[:500]
        recorder.record_error("session", e)
    if agent is not None:
        try:
            values = agent.graph.get_state(config).values
        except Exception as e:  # noqa: BLE001
            recorder.record_error("final_state", e)

    files = _write_outputs(recorder, values, workspace)
    request_log = user.finish(values)
    recorder.write_json("transcript.json", {"turns": transcript, "observations": user.observations,
                                            "requests": request_log})
    from evals.iterations import redrafts
    summary = {
        "status": status,
        "minutes": round((time.time() - t0) / 60, 2),
        "cv_drafts": len(values.get("cv_history") or []),
        "cover_letter_drafts": len(values.get("cover_letter_history") or []),
        "cv_final_iteration": values.get("cv_final_iteration"),
        "cover_letter_final_iteration": values.get("cover_letter_final_iteration"),
        "outputs": files,
        "observations": user.observations,
        "degraded": degradations(values, workspace),
        "work_context": (values.get("company_info") or {}).get("work_context"),
        "work_preference_final": values.get("work_preference"),
        "redrafts": redrafts(run_dir),
        "requests_sent": [r["id"] for r in request_log],
        "requests_not_sent": user.unsent_requests(),
        "requests_without_new_draft": [r["id"] for r in request_log if not r.get("new_draft")],
    }
    recorder.write_summary(**summary)
    recorder.write_json("manifest.json", {
        **manifest, **summary,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cost_usd": round(recorder.cost, 6),
    })
    recorder.event("run_end", **summary, cost_usd=round(recorder.cost, 6))
    print(json.dumps({"run_dir": str(run_dir), "status": status, "cost_usd": round(recorder.cost, 4),
                      "minutes": summary["minutes"], "outputs": sorted(files)}))
    if status == INTERRUPTED_STATUS:
        print(f"out of credit: the session was interrupted and is marked '{INTERRUPTED_STATUS}'", file=sys.stderr)
        return EXIT_OUT_OF_CREDIT
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
