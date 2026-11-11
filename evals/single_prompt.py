"""Single-prompt baseline: exactly ONE model call writes the tailored CV and cover letter, text and LaTeX.

The comparison point for the multi-agent app (evals/run_case.py; a default-script session there
makes a median of 22 model calls on three models). Here a case is one call and nothing else. The
single prompt holds all the information the multi-agent session gets:

    - the candidate files the input set gives (omitted ones are absent, as in run_case),
    - the job ad as published,
    - the company search results replayed from the ad's search fixture (the same fixture, read with the
      app's own `agents.search.search_result_texts`, that the multi-agent run and the evaluator see),
    - the candidate's work-arrangement preference rule,
    - the app's two LaTeX templates (templates/cv_template.tex, cover_letter_template.tex),

and the app's own instructions, imported, not copied: the writers' goal, guidelines and truthfulness
rules (agents/cv_writer.py, cover_letter_writer.py, document_writer.py) and the LaTeX formatters' rules
(agents/cv_pdf_generator.py, cover_letter_pdf_generator.py). The call to `--model` (default
openai:gpt-6.1-sol) returns, per document, the plain text and the full LaTeX source, plus notes.

PDFs: the returned LaTeX goes through the app's deterministic LaTeX helpers (`ensure_text_extractable`,
`escape_text_specials`; no model) and pdflatex, twice as in the app. No repair, retry or re-conversion
call is made. A document whose LaTeX does not compile gets its compile error recorded in the manifest
(`pdfs.<kind>`), and a PDF made from its plain text by a fixed built-in layout (`fallback_latex`, no
model), marked `"source": "fallback_layout"`, so its content can still be graded and the report can
separate fallbacks from model-made PDFs.

The run folder has run_case's layout (manifest.json, config.json, inputs/, llm_calls.jsonl,
events.jsonl, transcript.json, outputs/final_cv.txt/.pdf, final_cover_letter.txt/.pdf, summary.json,
workspace/), plus prompt.json (the prompt as sent), outputs/final_cv.tex, final_cover_letter.tex,
writer_notes.txt and pdf_checks.json. evals/evaluate.py grades it unchanged and blind: it reads inputs/,
outputs/, transcript.json and inputs/script.json, never the manifest, so it sees neither the method nor
the model. The manifest says `"method": "single_prompt"` (also as a label) for evals/summarize.py, and
`"calls"`: the number of model calls made (1).

The same user words as the app's default user: the prompt carries, verbatim, the default scripted user's request
("Please write my CV first, then the cover letter.") and its answer to the app's question for missing contact
details (decline), from `evals.run_case.DEFAULT_SCRIPT`. transcript.json records them as the run's user turns
(after the job ad, shown as the app's run shows it) and as `user_instructions`, so the grader's input builder
reads both arms' run folders alike. There are no tool messages: the turns' `assistant` is empty.

Models: any `--model` with a price in provenance.PRICES_PER_1M (or a local `ollama:` model); the comparison uses
openai:gpt-6-luna, openai:gpt-6.1-terra and openai:gpt-6.1-sol. The model is built as the app builds its writers
(`temperature=None`, the model's default: gpt-6-luna accepts no other). An answer that does not parse is a failed
session (`"failure": "parse_error"`, status `error: ...`, exit 1); it is not retried and is graded as it is.

Usage (from the repo root; this makes one paid model call per case):
    .venv/bin/python -m evals.single_prompt --run-dir RUN --ad a3 --persona p03 [--inputs all4|cv_only|no_cv]
        [--model openai:gpt-6.1-sol] [--env-file PATH/.env] [--cost-cap 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

from agents.cover_letter_pdf_generator import COVER_LETTER_LATEX_SYSTEM_PROMPT  # noqa: E402
from agents.cover_letter_writer import (COVER_LETTER_GENERATION_HUMAN_PROMPT,  # noqa: E402
                                        COVER_LETTER_GENERATION_SYSTEM_PROMPT, CoverLetterGenerationResponse,
                                        CoverLetterWriterAgent)
from agents.cv_pdf_generator import CV_LATEX_SYSTEM_PROMPT  # noqa: E402
from agents.cv_writer import (CV_GENERATION_HUMAN_PROMPT, CV_GENERATION_SYSTEM_PROMPT, NO_CV_TEXT,  # noqa: E402
                              CVGenerationResponse, CVWriterAgent)
from agents.document_writer import TRUTHFULNESS_PROTOCOL  # noqa: E402
from agents.job_description import output_language_instruction  # noqa: E402
from agents.latex_fixer import ensure_text_extractable, escape_text_specials  # noqa: E402 - pure functions, no model
from agents.work_context import describe as describe_work_context, with_preference  # noqa: E402

METHOD = "single_prompt"
DEFAULT_MODEL = "openai:gpt-6.1-sol"
GENERATION_NODE = "single_prompt"
# The default scripted user's replies the single prompt carries, in the order the app's user sends them.
USER_REPLY_KEYS = ("first", "missing_details")
JOB_AD_TURN = "<path to job ad>"  # how run_case's transcript shows the reply that hands over the job ad
KINDS = ("cv", "cover_letter")
TEMPLATES = {"cv": REPO / "templates" / "cv_template.tex",
             "cover_letter": REPO / "templates" / "cover_letter_template.tex"}
PDF_STEMS = {"cv": "final_cv", "cover_letter": "final_cover_letter"}
PDFLATEX_TIMEOUT = 180  # seconds, as run_case gives the app's pdflatex

# Section labels as the writers' prompts name the candidate's material (the source-fact wording is theirs).
INPUT_SECTIONS = {
    "cv": "**Candidate's Original CV (the 'Existing CV'; source facts: the candidate's own, as uploaded):**",
    "cover_letter": "**Candidate's Earlier Cover Letter (the 'Existing Cover Letter'; source facts only; "
                    "written for a different job):**",
    "notes": "**Candidate's Own Notes (free-form, possibly rough; facts here are the candidate's own, like the CV):**",
    "portfolio": "**Candidate's Portfolio (projects and past work the candidate supplied; their own facts):**",
}
INPUT_ORDER = ("cv", "cover_letter", "notes", "portfolio")


def _between(text: str, start: str, end: str | None) -> str:
    """The part of an app prompt from `start` up to (not including) `end` (None: to the end); fails loudly if
    the prompt changed."""
    i = text.find(start)
    j = (text.find(end, i + len(start)) if end else len(text)) if i >= 0 else -1
    if i < 0 or j < 0:
        raise ValueError(f"app prompt changed: cannot find '{start}' ... '{end}'")
    return text[i:j].strip()


def _unbrace(template_text: str) -> str:
    """A ChatPromptTemplate string as the model reads it ({{ }} are its escaped braces)."""
    return template_text.replace("{{", "{").replace("}}", "}")


def writer_rules() -> dict:
    """The app's own instructions, taken from its prompts (not forked)."""
    return {
        "cv_goal": _between(CV_GENERATION_SYSTEM_PROMPT, "Your PRIMARY GOAL", "Your CVs are"),
        "cover_letter_goal": _between(COVER_LETTER_GENERATION_SYSTEM_PROMPT, "Your PRIMARY GOAL", "IMPORTANT:"),
        "cv_guidelines": _between(CV_GENERATION_HUMAN_PROMPT, "General guidelines", "CRITICAL REQUIREMENTS"),
        "cover_letter_guidelines": _between(COVER_LETTER_GENERATION_HUMAN_PROMPT, "General guidelines",
                                            "CRITICAL: Provide your response"),
        "truthfulness": TRUTHFULNESS_PROTOCOL.strip(),
        "cv_latex": _unbrace(_between(CV_LATEX_SYSTEM_PROMPT, "CRITICAL RULES:", None)),
        "cover_letter_latex": _unbrace(_between(COVER_LETTER_LATEX_SYSTEM_PROMPT, "CRITICAL RULES:", None)),
    }


def _fresh_instructions(agent_cls, has_document: bool) -> str:
    # The writers' "creative liberty" instructions for a first draft; the method uses no instance state.
    return agent_cls._fresh_draft_instructions(agent_cls.__new__(agent_cls), has_document)


class SinglePromptResponse(BaseModel):
    """Both documents as plain text and as LaTeX, and the notes, in one answer."""
    cv_content: str = Field(..., description=CVGenerationResponse.model_fields["cv_content"].description)
    cv_latex: str = Field(..., description="The complete LaTeX source of the CV (preamble and document environment), "
                                           "ready for pdflatex, containing cv_content word for word.")
    cover_letter_content: str = Field(
        ..., description=CoverLetterGenerationResponse.model_fields["cover_letter_content"].description)
    cover_letter_latex: str = Field(..., description="The complete LaTeX source of the cover letter (preamble and "
                                                     "document environment), ready for pdflatex, containing "
                                                     "cover_letter_content word for word.")
    notes: str = Field(..., description=CVGenerationResponse.model_fields["notes"].description)


def missing_details_reply() -> str:
    """What the default scripted user answers when the app asks for missing contact details."""
    from evals.run_case import DEFAULT_SCRIPT
    return DEFAULT_SCRIPT["missing_details"]


def user_instructions() -> list[dict]:
    """The default scripted user's words the single prompt carries verbatim: [{"key", "text"}]."""
    from evals.run_case import DEFAULT_SCRIPT
    return [{"key": key, "text": DEFAULT_SCRIPT[key]} for key in USER_REPLY_KEYS]


def transcript_record() -> dict:
    """transcript.json of a single-prompt run: the user's words as turns, as run_case records a conversation
    (the job ad first), with no tool messages; and the same words as `user_instructions`."""
    said = [JOB_AD_TURN] + [i["text"] for i in user_instructions()]
    return {"turns": [{"turn": n, "waiting_at": [], "assistant": None, "user": text} for n, text in enumerate(said)],
            "observations": [], "requests": [], "user_instructions": user_instructions()}


def init_model(model: str):
    """The one call's model, built as the app builds its writers (the model's default temperature)."""
    from config.config_loader import local_model_kwargs
    return init_chat_model(model, temperature=None, **local_model_kwargs(model))


def check_model(model: str) -> None:
    """Refuse a model whose cost the run could not record (no price in provenance, not local)."""
    from provenance import price_for
    if not model.startswith("ollama:") and price_for(model) is None:
        raise SystemExit(f"--model {model}: no price in provenance.PRICES_PER_1M, so its cost could not be "
                         "recorded or capped; add the price first")


def build_messages(given: dict[str, str], job_ad: str, company: str, work_preference: str | None = None) -> list:
    """The single prompt: [system, human]. `given` maps cv/portfolio/cover_letter/notes to text (omitted keys absent);
    `company` is the search fixture's result text (evals.evaluate.company_text)."""
    rules = writer_rules()
    system = (
        "You are an expert CV and cover letter writer and LaTeX formatter. In one answer you write two documents "
        "for the same job application, a tailored CV and a cover letter consistent with it, each as clean plain "
        "text and as a complete LaTeX document that compiles with pdflatex.\n\n"
        f"For the CV: {rules['cv_goal']}\n\n"
        f"For the cover letter: {rules['cover_letter_goal']}\n\n"
        "Keep the documents clean: no explanations, notes or meta-commentary inside them. Put what you did and why "
        "in the separate notes."
        + "\n\n" + rules["truthfulness"]
    )
    parts = ["Create a professional, tailored CV and cover letter for this job from the following information.\n"]
    parts.append(f"**The candidate's request (their words):** \"{user_instructions()[0]['text']}\"\n")
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
    for kind, label in (("cv", "CV"), ("cover_letter", "cover letter")):
        parts.append(f"**LaTeX for the {label}** (start from this template; the plain-text {label} you write is the "
                     f"text to copy):\n{rules[f'{kind}_latex']}\n"
                     f"-------------------------------------\n{TEMPLATES[kind].read_text(encoding='utf-8')}\n"
                     "-------------------------------------\n")
    parts.append(
        "Return exactly five fields:\n"
        "1. cv_content: the CV document only, as plain text.\n"
        "2. cv_latex: the full LaTeX source of that CV (only LaTeX code, no prose).\n"
        "3. cover_letter_content: the cover letter document only, as plain text.\n"
        "4. cover_letter_latex: the full LaTeX source of that cover letter (only LaTeX code, no prose).\n"
        "5. notes: for the user, what you did and why (tailoring choices, requirements the candidate does not meet).")
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


class AnswerNotParsed(RuntimeError):
    """The one call returned an answer that is not the five fields: a failed session, never retried."""


def generate(llm, messages: list, callbacks: list) -> SinglePromptResponse:
    """The one model call of the case (no retries: a second call would change the topology)."""
    chain = llm.with_structured_output(SinglePromptResponse, include_raw=True)
    result = chain.invoke(messages, config={"callbacks": callbacks, "metadata": {"langgraph_node": GENERATION_NODE},
                                            "run_name": GENERATION_NODE})
    parsed = result.get("parsed") if isinstance(result, dict) else result
    if parsed is None:
        raise AnswerNotParsed(f"the model's answer did not parse: {result.get('parsing_error')}")
    return parsed


# ---------------------------------------------------------------------------------------------- PDFs (no model)
def pdflatex(tex: Path) -> tuple[bool, str]:
    """Compile tex in its folder, twice as the app does (references); (ok, compiler output tail)."""
    def once():
        return subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=str(tex.parent), timeout=PDFLATEX_TIMEOUT)
    try:
        first = once()
        if first.returncode != 0:
            return False, (first.stdout or "")[-3000:] + (first.stderr or "")[-1000:]
        once()
    except FileNotFoundError:
        return False, "pdflatex not found"
    except subprocess.TimeoutExpired:
        return False, f"pdflatex timed out after {PDFLATEX_TIMEOUT} s"
    pdf = tex.with_suffix(".pdf")
    return pdf.is_file(), "" if pdf.is_file() else "pdflatex made no PDF"


_LATEX_SPECIALS = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
                   "^": r"\^{}", "_": r"\_", "~": r"\~{}", "%": r"\%"}
_PUNCTUATION = {"\u2013": "--", "\u2014": "---", "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''",
                "\u201e": ",,", "\u2022": r"\textbullet{}", "\u2026": "...", "\u2192": "->", "\u00a0": "~",
                "\u2212": "-", "\u00b7": r"\textperiodcentered{}"}


def _fallback_line(line: str) -> tuple[str, set]:
    out, dropped = [], set()
    for ch in line:
        if ch in _LATEX_SPECIALS:
            out.append(_LATEX_SPECIALS[ch])
        elif ch in _PUNCTUATION:
            out.append(_PUNCTUATION[ch])
        elif ch == "\t":
            out.append(" ")
        elif ord(ch) < 0x250:  # Latin-1 and Latin Extended-A/B: T1 fonts cover the languages of the fixtures
            out.append(ch)
        else:
            dropped.add(ch)
    return "".join(out), dropped


def fallback_latex(text: str) -> tuple[str, list[str]]:
    """A fixed, plain layout for a document's text (no model): blank lines separate paragraphs, other lines
    break where the text breaks. Returns (LaTeX, characters left out because the fonts lack them)."""
    paragraphs, dropped = [], set()
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = []
        for line in block.splitlines():
            escaped, gone = _fallback_line(line.strip())
            dropped |= gone
            if escaped:
                lines.append(escaped)
        if lines:
            paragraphs.append(" \\\\\n".join(lines))
    body = "\n\n".join(paragraphs) or "~"
    latex = ("\\documentclass[11pt,a4paper]{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n"
             "\\usepackage[margin=2cm]{geometry}\n\\pagestyle{empty}\n\\setlength{\\parindent}{0pt}\n"
             "\\setlength{\\parskip}{0.6em}\n\\begin{document}\n" + body + "\n\\end{document}\n")
    return ensure_text_extractable(latex), sorted(dropped)


def make_pdfs(texts: dict[str, str], latex: dict[str, str], out_dir: Path) -> dict:
    """Each document's PDF from the model's LaTeX, else from the fallback layout; never a model call.

    Returns {kind: {"source": "model_latex" | "fallback_layout" | None, "compiled": bool,
    "compile_error": str | None, "fallback_error": str | None, "fallback_dropped_chars": [...], "pdf": path}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind in KINDS:
        stem = PDF_STEMS[kind]
        tex = out_dir / f"{stem}.tex"
        tex.write_text(escape_text_specials(ensure_text_extractable(latex.get(kind) or "")), encoding="utf-8")
        ok, log = pdflatex(tex) if (latex.get(kind) or "").strip() else (False, "the answer had no LaTeX")
        entry = {"source": "model_latex" if ok else None, "compiled": ok, "compile_error": None if ok else log,
                 "fallback_error": None, "fallback_dropped_chars": [], "pdf": str(tex.with_suffix(".pdf")) if ok else None}
        if not ok:
            fb_tex = out_dir / f"{stem}_fallback.tex"
            fb_latex, dropped = fallback_latex(texts.get(kind) or "")
            fb_tex.write_text(fb_latex, encoding="utf-8")
            fb_ok, fb_log = pdflatex(fb_tex)
            entry.update(fallback_dropped_chars=dropped, fallback_error=None if fb_ok else fb_log)
            if fb_ok:
                entry.update(source="fallback_layout", pdf=str(fb_tex.with_suffix(".pdf")))
        results[kind] = entry
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="new (or empty) folder for this run's records")
    parser.add_argument("--ad", required=True, help="tracked fixture ad id (e.g. a3): its ad.md and search_fixture.json")
    parser.add_argument("--persona", required=True, help="tracked fixture persona id (e.g. p03): its candidate files")
    parser.add_argument("--inputs", help="all4 (default), cv_only, no_cv, or e.g. cv,notes")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"the one call's model, e.g. openai:gpt-6-luna, openai:gpt-6.1-terra (default {DEFAULT_MODEL})")
    parser.add_argument("--work-preference", dest="work_preference", choices=["remote", "hybrid", "on-site", "no preference"])
    parser.add_argument("--env-file", help=".env file with API keys (read, never copied)")
    parser.add_argument("--cost-cap", type=float, default=1.0, help="recorded in the manifest; a run is one call (default 1.0)")
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
    check_model(args.model)
    from evals.run_case import apply_fixture_defaults, prepare_run_folder
    apply_fixture_defaults(args)
    if not any(getattr(args, k) for k in ("cv", "notes")):
        print("note: no --cv and no --notes; the single prompt has only the portfolio and earlier letter", file=sys.stderr)
    manifest = prepare_run_folder(args)
    run_dir = Path(manifest["run_dir"])
    manifest.update(method=METHOD, labels={**manifest["labels"], "method": METHOD}, model=args.model,
                    model_override=args.model, script=None)
    for key in ("critic_model_override", "grounding_model_override"):
        manifest.pop(key, None)
    from dotenv import load_dotenv
    load_dotenv(args.env_file or REPO / ".env")  # keys stay in this process; nothing from the env is written

    from provenance import RunRecorder
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS, CreditGuard, OutOfCredit, install_no_retry_patch
    from evals.evaluate import company_text
    from evals.run_case import _git, _package_versions
    recorder = RunRecorder(run_dir, cost_cap_usd=args.cost_cap)
    install_no_retry_patch()
    callbacks = [recorder.callback, CreditGuard()]
    recorder.write_json("config.json", {
        "method": METHOD, "topology": "exactly one model call per case; PDFs by pdflatex and a fixed fallback layout",
        "generation": {"model": args.model, "calls": 1, "retries": 0},
        "templates": {k: str(p.relative_to(REPO)) for k, p in TEMPLATES.items()},
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

    status, response, pdfs, failure = "completed", None, {}, None
    t0 = time.time()
    try:
        response = generate(init_model(args.model), messages, callbacks)
        recorder.event("generation", model=args.model, **{f"{f}_chars": len(getattr(response, f)) for f in
                                                          ("cv_content", "cv_latex", "cover_letter_content",
                                                           "cover_letter_latex")})
        pdfs = make_pdfs({"cv": response.cv_content, "cover_letter": response.cover_letter_content},
                         {"cv": response.cv_latex, "cover_letter": response.cover_letter_latex},
                         run_dir / "workspace" / "latex")
        recorder.event("pdfs", **{k: {kk: v[kk] for kk in ("source", "compiled", "fallback_error")} for k, v in pdfs.items()})
    except OutOfCredit as e:
        status = INTERRUPTED_STATUS
        recorder.record_error("session", e)
    except Exception as e:  # noqa: BLE001 - record what happened
        status = f"error: {type(e).__name__}: {e}"[:500]
        failure = "parse_error" if isinstance(e, AnswerNotParsed) else "error"
        recorder.record_error("session", e)

    out = run_dir / "outputs"
    out.mkdir(exist_ok=True)
    files = {}
    if response is not None:
        for kind, text, latex in (("cv", response.cv_content, response.cv_latex),
                                  ("cover_letter", response.cover_letter_content, response.cover_letter_latex)):
            (out / f"{PDF_STEMS[kind]}.txt").write_text(text, encoding="utf-8")
            (out / f"{PDF_STEMS[kind]}.tex").write_text(latex, encoding="utf-8")  # as the model returned it
            files[f"generated_{kind}"] = f"outputs/{PDF_STEMS[kind]}.txt"
        (out / "writer_notes.txt").write_text(response.notes or "", encoding="utf-8")
    from agents.pdf_text_check import check_pdf
    for kind, entry in pdfs.items():
        if entry.get("pdf") and Path(entry["pdf"]).is_file():
            shutil.copy(entry["pdf"], out / f"{PDF_STEMS[kind]}.pdf")
            files[f"{kind}_pdf_path"] = f"outputs/{PDF_STEMS[kind]}.pdf"
            entry["check"] = check_pdf(out / f"{PDF_STEMS[kind]}.pdf",
                                       response.cv_content if kind == "cv" else response.cover_letter_content)
        entry["pdf"] = files.get(f"{kind}_pdf_path")
    recorder.write_json("outputs/pdf_checks.json", pdfs)
    # The default user's words as the run's user turns (no tool messages); no scripted request, so no checklist.
    recorder.write_json("transcript.json", transcript_record())
    calls = recorder.totals["llm_calls"] + recorder.totals["llm_errors"]
    summary = {"status": status, "method": METHOD, "failure": failure, "minutes": round((time.time() - t0) / 60, 2), "outputs": files,
               "calls": calls, "calls_by_node": {n: v["calls"] for n, v in recorder.totals["by_node"].items()},
               "pdfs": {k: {kk: v.get(kk) for kk in ("source", "compiled", "compile_error", "fallback_error",
                                                       "fallback_dropped_chars", "pdf")} for k, v in pdfs.items()},
               "fallback_pdfs": sorted(k for k, v in pdfs.items() if v.get("source") == "fallback_layout"),
               "compile_failures": sorted(k for k, v in pdfs.items() if not v.get("compiled"))}
    recorder.write_summary(**summary)
    recorder.write_json("manifest.json", {
        **manifest, **summary,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cost_usd": round(recorder.cost, 6),
    })
    recorder.event("run_end", **summary, cost_usd=round(recorder.cost, 6))
    print(json.dumps({"run_dir": str(run_dir), "status": status, "method": METHOD, "calls": calls,
                      "cost_usd": round(recorder.cost, 4), "fallback_pdfs": summary["fallback_pdfs"],
                      "outputs": sorted(files)}))
    if status == INTERRUPTED_STATUS:
        print(f"out of credit: the run was interrupted and is marked '{INTERRUPTED_STATUS}'", file=sys.stderr)
        return EXIT_OUT_OF_CREDIT
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
