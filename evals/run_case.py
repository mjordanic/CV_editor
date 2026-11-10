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

Usage (from the repo root):
    .venv/bin/python -m evals.run_case --run-dir RUN --job-ad AD.md --cv CV.txt --portfolio P.txt \\
        --cover-letter CL.txt --notes NOTES.txt --search-fixture FIXTURE.json [--model openai:gpt-6-luna]

Without --search-fixture, pass --no-search (company research finds nothing) or --record-search FILE
(live Tavily, saved to FILE for later replays). This makes paid model calls; --cost-cap stops a run.
"""
from __future__ import annotations

import argparse
import json
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
    "max_turns": 10,
}


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _package_versions() -> dict:
    from importlib import metadata
    out = {}
    for name in ("langgraph", "langchain", "langchain-core", "langchain-openai", "openai", "tavily-python", "tavily"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
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

    script = dict(DEFAULT_SCRIPT)
    if args.script:
        script.update(json.loads(Path(args.script).read_text(encoding="utf-8")))
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
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def effective_config() -> dict:
    """Model and settings per agent as the app will use them (after CV_EDITOR_MODEL)."""
    from config.config_loader import get_agent_config, load_config, MODEL_OVERRIDE_ENV
    full = load_config()
    return {
        "workflow": full.get("workflow", {}),
        "agents": {name: get_agent_config(name) for name in full.get("agents", {})},
        "embeddings": os.environ.get("CV_EDITOR_EMBEDDING_MODEL") or "text-embedding-3-small",
        "env_overrides": {k: os.environ.get(k) for k in (MODEL_OVERRIDE_ENV, "CV_EDITOR_TEMPERATURE",
                                                          "CV_EDITOR_SEARCH_FIXTURE", "CV_EDITOR_SEARCH_RECORD")
                          if os.environ.get(k)},
        "code": {"repo": str(REPO), "commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(_git("status", "--porcelain"))},
        "python": platform.python_version(),
        "packages": _package_versions(),
    }


class ScriptedUser:
    """Answers the app's questions from the script, by what the app is waiting for."""

    def __init__(self, script: dict, job_ad_path: str):
        self.script = script
        self.job_ad_path = job_ad_path
        self.used: dict = {}
        self.observations: list = []

    def _use(self, key: str) -> str:
        self.used[key] = self.used.get(key, 0) + 1
        return self.script[key]

    def reply(self, waiting_at: tuple, values: dict) -> str:
        if "analyze_job_description" in waiting_at:
            if self.used.get("job_ad"):
                raise RuntimeError("the app asked for the job ad again (it could not read it)")
            self.used["job_ad"] = 1
            return self.job_ad_path
        if "load_candidate_documents" in waiting_at:
            raise RuntimeError("the app asked for a CV: the input set has no CV and no notes")
        if "clarify_job_ad" in waiting_at:
            return self._use("clarify_job_ad")
        if not values.get("generated_cv"):
            if self.used.get("first", 0) >= 2:
                self.observations.append("the app still had no CV after the user asked for it twice")
                return self._use("exit")
            return self._use("first")
        if not values.get("generated_cover_letter"):
            if self.used.get("cover_letter", 0) >= 2:
                self.observations.append("the app still had no cover letter after the user asked for it twice")
                return self._use("exit")
            return self._use("cover_letter")
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="new (or empty) folder for this run's records")
    parser.add_argument("--job-ad", required=True, help="job ad text file")
    parser.add_argument("--cv", help="candidate CV (.txt, .md or .pdf)")
    parser.add_argument("--portfolio", help="candidate portfolio (.txt; projects separated by blank lines)")
    parser.add_argument("--cover-letter", dest="cover_letter", help="candidate's earlier cover letter")
    parser.add_argument("--notes", help="candidate's free-form notes (rough CV draft, bullet notes, anything)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--search-fixture", help="prepared web-search results to replay (no live search)")
    group.add_argument("--record-search", help="live Tavily search, saved to this fixture file")
    group.add_argument("--no-search", action="store_true", help="company research finds nothing (default)")
    parser.add_argument("--model", help="use this model for every agent (sets CV_EDITOR_MODEL)")
    parser.add_argument("--script", help="JSON with scripted replies overriding the defaults")
    parser.add_argument("--env-file", help=".env file with API keys (read, never copied)")
    parser.add_argument("--cost-cap", type=float, default=1.0, help="stop the run past this many USD (default 1.0)")
    parser.add_argument("--label", action="append", help="key=value stored in manifest.json (repeatable)")
    args = parser.parse_args(argv)
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
    recorder = RunRecorder(run_dir, cost_cap_usd=args.cost_cap)

    from agents import rag_agent
    real_embeddings_for = rag_agent._embeddings_for

    def counted_embeddings(model):
        inner = real_embeddings_for(model)
        return CountingEmbeddings(inner, recorder, getattr(inner, "model", "embeddings"))

    rag_agent._embeddings_for = counted_embeddings

    recorder.write_json("config.json", effective_config())
    recorder.write_json("manifest.json", {**manifest, "status": "running"})
    recorder.event("run_start", manifest=manifest)

    from langgraph.types import Command
    import langgraph_agent as la

    user = ScriptedUser(manifest["script"], str(workspace / "job_ad.txt"))
    transcript, status, values = [], "completed", {}
    t0 = time.time()
    agent = None
    config = recorder.graph_config("testing-mode")
    try:
        agent = la.MasterAgent()
        initial = {k: None for k in la.State.__annotations__}
        initial.update(messages=[], cv_refinement_count=0, cover_letter_refinement_count=0,
                       cv_history=[], cover_letter_history=[])
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
    except Exception as e:  # noqa: BLE001 - record what happened
        status = f"error: {type(e).__name__}: {e}"[:500]
        recorder.record_error("session", e)
    if agent is not None:
        try:
            values = agent.graph.get_state(config).values
        except Exception as e:  # noqa: BLE001
            recorder.record_error("final_state", e)

    files = _write_outputs(recorder, values, workspace)
    recorder.write_json("transcript.json", {"turns": transcript, "observations": user.observations})
    summary = {
        "status": status,
        "minutes": round((time.time() - t0) / 60, 2),
        "cv_drafts": len(values.get("cv_history") or []),
        "cover_letter_drafts": len(values.get("cover_letter_history") or []),
        "cv_final_iteration": values.get("cv_final_iteration"),
        "cover_letter_final_iteration": values.get("cover_letter_final_iteration"),
        "outputs": files,
        "observations": user.observations,
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
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
