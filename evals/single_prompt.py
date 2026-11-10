"""Single-prompt baseline: one model call writes the tailored CV and cover letter from everything at once.

The comparison point for the multi-agent app (evals/run_case.py). For one case it builds a single prompt
holding all the information the multi-agent session gets:

    - the candidate files the input set gives (omitted ones are absent, as in run_case),
    - the job ad as published,
    - the company search results replayed from the ad's search fixture (the same fixture, read with the
      app's own `agents.search.search_result_texts`, that the multi-agent run and the evaluator see),
    - the candidate's work-arrangement preference rule,

and the writers' own goal and truthfulness rules (imported from agents/cv_writer.py,
agents/cover_letter_writer.py and agents/document_writer.py, not copied). One call to `--model`
(default openai:gpt-6.1-sol) returns the CV, the cover letter and notes.

The PDFs are made by the app's own PDF path (agents/cv_pdf_generator.py, cover_letter_pdf_generator.py:
text -> LaTeX by the `template_customizer` model, pdflatex, the LaTeX fixer when it does not compile,
the PDF text check and its one re-conversion), so they are comparable with the multi-agent PDFs. Those
rendering calls are model calls too: every one is logged in llm_calls.jsonl under the app's node names
(generate_cv_pdf, generate_cover_letter_pdf) and counted separately in the manifest (`calls`).
`--render-model` sets their model (default: the `template_customizer` model in config/agent_models.yaml).

The run folder has run_case's layout (manifest.json, config.json, inputs/, llm_calls.jsonl,
events.jsonl, transcript.json, outputs/final_cv.txt/.pdf, final_cover_letter.txt/.pdf, summary.json,
workspace/), plus prompt.json (the single prompt as sent) and outputs/writer_notes.txt and
outputs/pdf_checks.json, so evals/evaluate.py grades it unchanged and blind: the evaluator reads inputs/, outputs/,
transcript.json and inputs/script.json, never the manifest, so it does not see the method or the model.
The manifest says `"method": "single_prompt"` (also as a label) for evals/summarize.py to split on.
There is no conversation: transcript.json has no turns. The default script's answer to the app's
question for missing contact details (decline) is written into the prompt instead.

Usage (from the repo root; this makes paid model calls):
    .venv/bin/python -m evals.single_prompt --run-dir RUN --ad a3 --persona p03 [--inputs all4|cv_only|no_cv]
        [--model openai:gpt-6.1-sol] [--render-model openai:gpt-6-luna] [--env-file PATH/.env] [--cost-cap 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agents.cover_letter_writer import (COVER_LETTER_GENERATION_HUMAN_PROMPT,  # noqa: E402
                                        COVER_LETTER_GENERATION_SYSTEM_PROMPT, CoverLetterGenerationResponse,
                                        CoverLetterWriterAgent)
from agents.cv_writer import (CV_GENERATION_HUMAN_PROMPT, CV_GENERATION_SYSTEM_PROMPT, NO_CV_TEXT,  # noqa: E402
                              CVGenerationResponse, CVWriterAgent)
from agents.document_writer import TRUTHFULNESS_PROTOCOL  # noqa: E402
from agents.job_description import output_language_instruction  # noqa: E402
from agents.work_context import describe as describe_work_context, with_preference  # noqa: E402

METHOD = "single_prompt"
DEFAULT_MODEL = "openai:gpt-6.1-sol"
GENERATION_NODE = "single_prompt"
RENDER_NODES = {"cv": "generate_cv_pdf", "cover_letter": "generate_cover_letter_pdf"}  # the app's node names

# Section labels as the writers' prompts name the candidate's material (the source-fact wording is theirs).
INPUT_SECTIONS = {
    "cv": "**Candidate's Original CV (the 'Existing CV'; source facts: the candidate's own, as uploaded):**",
    "cover_letter": "**Candidate's Earlier Cover Letter (the 'Existing Cover Letter'; source facts only; "
                    "written for a different job):**",
    "notes": "**Candidate's Own Notes (free-form, possibly rough; facts here are the candidate's own, like the CV):**",
    "portfolio": "**Candidate's Portfolio (projects and past work the candidate supplied; their own facts):**",
}
INPUT_ORDER = ("cv", "cover_letter", "notes", "portfolio")


def _between(text: str, start: str, end: str) -> str:
    """The part of a writer prompt from `start` up to (not including) `end`; fails loudly if the prompt changed."""
    i = text.find(start)
    j = text.find(end, i + len(start)) if i >= 0 else -1
    if i < 0 or j < 0:
        raise ValueError(f"writer prompt changed: cannot find '{start}' ... '{end}'")
    return text[i:j].strip()


def writer_rules() -> dict:
    """The writers' goal, guidelines and truthfulness rules, taken from the app's prompts (not forked)."""
    return {
        "cv_goal": _between(CV_GENERATION_SYSTEM_PROMPT, "Your PRIMARY GOAL", "Your CVs are"),
        "cover_letter_goal": _between(COVER_LETTER_GENERATION_SYSTEM_PROMPT, "Your PRIMARY GOAL", "IMPORTANT:"),
        "cv_guidelines": _between(CV_GENERATION_HUMAN_PROMPT, "General guidelines", "CRITICAL REQUIREMENTS"),
        "cover_letter_guidelines": _between(COVER_LETTER_GENERATION_HUMAN_PROMPT, "General guidelines",
                                            "CRITICAL: Provide your response"),
        "truthfulness": TRUTHFULNESS_PROTOCOL.strip(),
    }


def _fresh_instructions(agent_cls, has_document: bool) -> str:
    # The writers' "creative liberty" instructions for a first draft; the method uses no instance state.
    return agent_cls._fresh_draft_instructions(agent_cls.__new__(agent_cls), has_document)


class SinglePromptResponse(BaseModel):
    """Both documents and the notes, in one answer (field descriptions are the writers' own)."""
    cv_content: str = Field(..., description=CVGenerationResponse.model_fields["cv_content"].description)
    cover_letter_content: str = Field(
        ..., description=CoverLetterGenerationResponse.model_fields["cover_letter_content"].description)
    notes: str = Field(..., description=CVGenerationResponse.model_fields["notes"].description)


def missing_details_reply() -> str:
    """What the default scripted user answers when the app asks for missing contact details."""
    from evals.run_case import DEFAULT_SCRIPT
    return DEFAULT_SCRIPT["missing_details"]


def build_messages(given: dict[str, str], job_ad: str, company: str, work_preference: str | None = None) -> list:
    """The single prompt: [system, human]. `given` maps cv/portfolio/cover_letter/notes to text (omitted keys absent);
    `company` is the search fixture's result text (evals.evaluate.company_text)."""
    rules = writer_rules()
    system = (
        "You are an expert CV and cover letter writer. In one answer you write two documents for the same job "
        "application: a tailored CV and a cover letter that is consistent with it.\n\n"
        f"For the CV: {rules['cv_goal']}\n\n"
        f"For the cover letter: {rules['cover_letter_goal']}\n\n"
        "Keep the documents clean: no explanations, notes or meta-commentary inside them. Put what you did and why "
        "in the separate notes."
        + "\n\n" + rules["truthfulness"]
    )
    parts = ["Create a professional, tailored CV and cover letter for this job from the following information.\n"]
    for key in INPUT_ORDER:
        if given.get(key):
            parts.append(f"{INPUT_SECTIONS[key]}\n{given[key].strip()}\n")
    if not given.get("cv"):
        parts.append(f"**About the CV:** {NO_CV_TEXT}\n")
    parts.append(
        "**Missing contact details:** the candidate was asked for any contact details their material does not give "
        f"and answered: \"{missing_details_reply()}\" Leave such details out; leave no placeholder such as [EMAIL] "
        "in either document.\n")
    parts.append(f"**Job Description (the job ad as published):**\n{job_ad.strip()}\n")
    parts.append("**Company Information (web search results about the employer):**\n"
                 f"{company.strip()}\n{describe_work_context(with_preference({}, work_preference), work_preference)}\n")
    parts.append(f"**Output Language:** {output_language_instruction(None, 'CV')} "
                 f"{output_language_instruction(None, 'cover letter')}\n")
    parts.append(f"**How to write the CV:**\n{_fresh_instructions(CVWriterAgent, bool(given.get('cv')))}\n\n"
                 f"{rules['cv_guidelines']}\n")
    parts.append("**How to write the cover letter:**\n"
                 f"{_fresh_instructions(CoverLetterWriterAgent, bool(given.get('cover_letter')))}\n"
                 "Keep the letter consistent with the CV you write in this answer.\n\n"
                 f"{rules['cover_letter_guidelines']}\n")
    parts.append(
        "Return exactly three fields:\n"
        "1. cv_content: the CV document only.\n"
        "2. cover_letter_content: the cover letter document only.\n"
        "3. notes: for the user, what you did and why (tailoring choices, requirements the candidate does not meet).")
    return [SystemMessage(content=system), HumanMessage(content="\n".join(parts))]


def read_given_inputs(inputs_dir: Path, given_keys: list[str]) -> dict[str, str]:
    """The text of each candidate file copied to inputs/ (PDFs read as the evaluator reads them)."""
    from evals.evaluate import _read
    out = {}
    for key in given_keys:
        for suffix in (".txt", ".md", ".pdf"):
            text = _read(inputs_dir / f"{key}{suffix}")
            if text:
                out[key] = text
                break
    return out


def generate(llm, messages: list, callbacks: list) -> SinglePromptResponse:
    """The one generation call (no retries: a second call would be a hidden extra call)."""
    chain = llm.with_structured_output(SinglePromptResponse, include_raw=True)
    result = chain.invoke(messages, config={"callbacks": callbacks, "metadata": {"langgraph_node": GENERATION_NODE},
                                            "run_name": GENERATION_NODE})
    parsed = result.get("parsed") if isinstance(result, dict) else result
    if parsed is None:
        raise RuntimeError(f"the model's answer did not parse: {result.get('parsing_error')}")
    return parsed


def _logged(runnable, callbacks: list, node: str):
    """A chain whose every model call goes to the run's recorder under the app's node name."""
    return runnable.with_config(callbacks=callbacks, metadata={"langgraph_node": node}, run_name=node)


def render_pdfs(texts: dict[str, str], render_model: str, render_temperature, callbacks: list) -> dict:
    """The app's PDF path for each document, run in the current folder (the run's workspace).

    Returns {kind: {"pdf": path or None, "check": dict, "fix_attempts": int, "error": str or None}}."""
    from agents.cover_letter_pdf_generator import CoverLetterPDFGenerator
    from agents.cv_pdf_generator import CVPDFGenerator
    from agents.latex_fixer import LatexFixer
    out = {}
    for kind, cls in (("cv", CVPDFGenerator), ("cover_letter", CoverLetterPDFGenerator)):
        node = RENDER_NODES[kind]
        try:
            gen = cls(model=render_model, temperature=render_temperature)
            gen._content_chain = _logged(gen._content_chain, callbacks, node)
            fixer: LatexFixer = gen._latex_fixer
            fixer._chain = _logged(fixer._chain, callbacks, node)
            pdf, fixes, check = gen._render_checked(texts[kind], gen._load_template(), None,
                                                    f"{gen.file_stem}_draft_1", gen.max_latex_fix_iterations)
            out[kind] = {"pdf": pdf, "check": check, "fix_attempts": fixes, "error": None}
        except Exception as e:  # noqa: BLE001 - a PDF that fails is recorded, as in the app (no PDF, an error)
            from evals.credit import OutOfCredit, is_out_of_credit
            if is_out_of_credit(e):
                raise OutOfCredit(str(e)) from e
            out[kind] = {"pdf": None, "check": None, "fix_attempts": None, "error": f"{type(e).__name__}: {e}"[:500]}
    return out


def _render_model() -> tuple[str, object]:
    from config.config_loader import get_agent_config
    cfg = get_agent_config("template_customizer", default_model="openai:gpt-5-nano", default_temperature=0.1)
    return cfg["model"], cfg["temperature"]


def calls_by_kind(totals: dict) -> dict:
    """Model calls and cost split into the generation call and the PDF rendering calls (from summary totals)."""
    by_node = totals.get("by_node") or {}
    pick = lambda nodes: {"calls": sum((by_node.get(n) or {}).get("calls", 0) for n in nodes),  # noqa: E731
                          "cost_usd": round(sum((by_node.get(n) or {}).get("cost_usd", 0.0) for n in nodes), 6)}
    return {"generation": pick([GENERATION_NODE]), "rendering": pick(list(RENDER_NODES.values())),
            "other": pick([n for n in by_node if n != GENERATION_NODE and n not in RENDER_NODES.values()])}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="new (or empty) folder for this run's records")
    parser.add_argument("--ad", required=True, help="tracked fixture ad id (e.g. a3): its ad.md and search_fixture.json")
    parser.add_argument("--persona", required=True, help="tracked fixture persona id (e.g. p03): its candidate files")
    parser.add_argument("--inputs", help="all4 (default), cv_only, no_cv, or e.g. cv,notes")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"the one generation call's model (default {DEFAULT_MODEL})")
    parser.add_argument("--render-model", dest="render_model",
                        help="model of the PDF rendering calls (default: template_customizer in config/agent_models.yaml)")
    parser.add_argument("--work-preference", dest="work_preference", choices=["remote", "hybrid", "on-site", "no preference"])
    parser.add_argument("--env-file", help=".env file with API keys (read, never copied)")
    parser.add_argument("--cost-cap", type=float, default=1.0, help="no PDF rendering past this many USD (default 1.0)")
    parser.add_argument("--label", action="append", help="key=value stored in manifest.json (repeatable)")
    args = parser.parse_args(argv)
    # The attributes run_case's folder preparation reads: no script, no recorded search, no file overrides.
    for key in ("job_ad", "cv", "portfolio", "cover_letter", "notes", "search_fixture", "record_search", "script",
                "critic_model", "grounding_model"):
        setattr(args, key, None)
    args.no_search = False
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    from evals.run_case import apply_fixture_defaults, prepare_run_folder
    apply_fixture_defaults(args)
    if not any(getattr(args, k) for k in ("cv", "notes")):
        print("note: no --cv and no --notes; the single prompt has only the portfolio and earlier letter", file=sys.stderr)
    manifest = prepare_run_folder(args)
    run_dir = Path(manifest["run_dir"])
    workspace = run_dir / "workspace"
    labels = {**manifest["labels"], "method": METHOD}
    from dotenv import load_dotenv
    load_dotenv(args.env_file or REPO / ".env")  # keys stay in this process; nothing from the env is written
    for key in ("CV_EDITOR_MODEL", "CV_EDITOR_SEARCH_FIXTURE", "CV_EDITOR_SEARCH_RECORD"):
        os.environ.pop(key, None)  # the rendering models come from the config (or --render-model), as stated
    render_model, render_temperature = _render_model()
    if args.render_model:
        render_model, render_temperature = args.render_model, None
    manifest.update(method=METHOD, labels=labels, model=args.model, render_model=render_model,
                    model_override=args.model, script=None)
    for key in ("critic_model_override", "grounding_model_override"):
        manifest.pop(key, None)

    from provenance import RunRecorder
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS, CreditGuard, OutOfCredit, install_no_retry_patch
    from evals.evaluate import company_text
    from evals.run_case import _git, _package_versions
    recorder = RunRecorder(run_dir, cost_cap_usd=args.cost_cap)
    guard = CreditGuard()
    install_no_retry_patch()
    callbacks = [recorder.callback, guard]
    recorder.write_json("config.json", {
        "method": METHOD, "generation": {"model": args.model, "calls": 1, "retries": 0},
        "rendering": {"model": render_model, "temperature": render_temperature, "agents": ["template_customizer",
                                                                                          "latex_fixer"]},
        "code": {"repo": str(REPO), "commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(_git("status", "--porcelain"))},
        "packages": _package_versions(),
    })
    recorder.write_json("manifest.json", {**manifest, "status": "running"})
    recorder.event("run_start", manifest=manifest)

    inputs = run_dir / "inputs"
    given = read_given_inputs(inputs, manifest["inputs_given"])
    messages = build_messages(given, (inputs / "job_ad.txt").read_text(encoding="utf-8"),
                              company_text(inputs / "search_fixture.json"), args.work_preference)
    recorder.write_json("prompt.json", [{"role": m.type, "content": m.content} for m in messages])

    status, response, rendered = "completed", None, {}
    t0 = time.time()
    real_run = subprocess.run

    def run_with_timeout(cmd, *a, **kw):  # a stuck pdflatex must not hang a batch (as in run_case)
        if isinstance(cmd, list) and cmd and cmd[0] == "pdflatex":
            kw.setdefault("timeout", 180)
        return real_run(cmd, *a, **kw)

    cwd = os.getcwd()
    try:
        llm = init_chat_model(args.model)
        response = generate(llm, messages, callbacks)
        recorder.event("generation", model=args.model, cv_chars=len(response.cv_content),
                       cover_letter_chars=len(response.cover_letter_content))
        if recorder.over_cap():
            status = f"stopped: cost cap {args.cost_cap} USD reached before rendering"
        else:
            os.chdir(workspace)
            subprocess.run = run_with_timeout
            rendered = render_pdfs({"cv": response.cv_content, "cover_letter": response.cover_letter_content},
                                   render_model, render_temperature, callbacks)
            recorder.event("rendering", **{k: {kk: v[kk] for kk in ("pdf", "fix_attempts", "error")}
                                           for k, v in rendered.items()})
    except OutOfCredit as e:
        status = INTERRUPTED_STATUS
        recorder.record_error("session", e)
    except Exception as e:  # noqa: BLE001 - record what happened
        status = f"error: {type(e).__name__}: {e}"[:500]
        recorder.record_error("session", e)
    finally:
        subprocess.run = real_run
        os.chdir(cwd)

    out = run_dir / "outputs"
    out.mkdir(exist_ok=True)
    files = {}
    if response is not None:
        for key, name, text in (("generated_cv", "final_cv.txt", response.cv_content),
                                ("generated_cover_letter", "final_cover_letter.txt", response.cover_letter_content)):
            (out / name).write_text(text, encoding="utf-8")
            files[key] = f"outputs/{name}"
        (out / "writer_notes.txt").write_text(response.notes or "", encoding="utf-8")
    for kind, name in (("cv", "final_cv.pdf"), ("cover_letter", "final_cover_letter.pdf")):
        pdf = (rendered.get(kind) or {}).get("pdf")
        if pdf and (workspace / pdf).is_file():
            shutil.copy(workspace / pdf, out / name)
            files[f"{kind}_pdf_path"] = f"outputs/{name}"
    recorder.write_json("outputs/pdf_checks.json", {k: {kk: v[kk] for kk in ("check", "fix_attempts", "error")}
                                                    for k, v in rendered.items()})
    # No conversation: the evaluator's conversation section says so, and its request checklist is empty.
    recorder.write_json("transcript.json", {"turns": [], "observations": [], "requests": []})
    pdf_errors = {k: v["error"] for k, v in rendered.items() if v.get("error")}
    summary = {"status": status, "method": METHOD, "minutes": round((time.time() - t0) / 60, 2), "outputs": files,
               "pdf_errors": pdf_errors, "calls": calls_by_kind(recorder.totals)}
    recorder.write_summary(**summary)
    recorder.write_json("manifest.json", {
        **manifest, **summary,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cost_usd": round(recorder.cost, 6),
    })
    recorder.event("run_end", **summary, cost_usd=round(recorder.cost, 6))
    print(json.dumps({"run_dir": str(run_dir), "status": status, "method": METHOD, "cost_usd": round(recorder.cost, 4),
                      "calls": summary["calls"], "outputs": sorted(files)}))
    if status == INTERRUPTED_STATUS:
        print(f"out of credit: the run was interrupted and is marked '{INTERRUPTED_STATUS}'", file=sys.stderr)
        return EXIT_OUT_OF_CREDIT
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
