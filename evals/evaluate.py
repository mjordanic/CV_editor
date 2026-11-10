"""Independent evaluator: score one testing-mode run folder with a stronger model.

The evaluator is blind to how the run was made: it never sees the model, the input-set name or the
run id. It gets the job ad, the company information the writer had (the replayed search results),
the candidate documents the writer was given (omitted ones are simply absent), the final CV and
cover letter text, and the text extracted from the two PDFs. It answers a fixed rubric as JSON with
quoted evidence:

  (a) invented content   every skill, tool, employer, title, date, number, degree, certificate in the
                         outputs, marked supported (with a quote from the candidate's material) or not
  (b) completeness       facts in the material that matter for this ad but were left out
  (c) fit to the ad      must-have and nice-to-have coverage, and whether gaps are handled honestly
  (d) CV standards       structure, length, contact details, chronology, language, artefacts
  (e) ATS screening      PDF text parseability, standard headings, keyword coverage, pass likelihood
  (f) cover letter       specificity, consistency with the CV, stale framing from the old letter
  (g) work arrangement   remote / hybrid / on-site, location limits and culture used correctly and not
                         invented; the candidate's preference stated only if given, conflicts not hidden

Results go to RUN/evaluation/<name>/: prompt.json, raw_response.json, evaluation.json (the rubric
answer plus deterministic PDF checks and counts), llm_calls.jsonl and summary.json (cost).

Usage:
    .venv/bin/python -m evals.evaluate RUN_DIR [--model openai:gpt-6.1-sol] [--name eval1] [--env-file .env]
This makes one paid model call per evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_MODEL = "openai:gpt-6.1-sol"
INPUT_LABELS = {
    "cv": "CANDIDATE'S CV",
    "portfolio": "CANDIDATE'S PORTFOLIO (projects and past work)",
    "cover_letter": "CANDIDATE'S EARLIER COVER LETTER (written for a different, earlier job application)",
    "notes": "CANDIDATE'S OWN FREE-FORM NOTES",
}


# ---------------------------------------------------------------------------------------------
# Rubric schema
class Claim(BaseModel):
    document: Literal["cv", "cover_letter"]
    claim: str = Field(description="The claim about the candidate, quoted briefly from the output document")
    kind: Literal["skill_or_tool", "employer", "job_title", "date_or_duration", "number_or_metric",
                  "education", "certification", "project", "language", "responsibility_or_achievement", "other"]
    verdict: Literal["supported", "partly_supported", "unsupported"] = Field(
        description="supported: the material states it or it follows obviously; partly_supported: overstated or "
                    "stretched (e.g. a course turned into work experience, 'mentored' into 'led'); unsupported: not in the material")
    source: Literal["cv", "portfolio", "cover_letter", "notes", "none"]
    evidence_quote: str = Field(description="Short verbatim quote from the candidate's material that supports it; empty if none")
    note: str = Field(description="Why partly_supported or unsupported; empty otherwise")


class Omission(BaseModel):
    fact: str = Field(description="A fact about the candidate, relevant to this ad, that neither output uses")
    source: Literal["cv", "portfolio", "cover_letter", "notes"]
    source_quote: str
    relevance: Literal["high", "medium", "low"]
    why_it_matters: str


class Requirement(BaseModel):
    requirement: str = Field(description="One requirement from the job ad, in short form")
    priority: Literal["must_have", "nice_to_have"]
    candidate_has_it: Literal["yes", "partial", "no", "unclear"] = Field(description="Judged from the candidate's material only")
    shown_in_cv: Literal["yes", "partial", "no"]
    evidence_quote: str = Field(description="Quote from the CV that shows it; empty if not shown")
    gap_handling: Literal["not_a_gap", "honest", "overstated", "hidden_or_ignored"] = Field(
        description="For a requirement the candidate lacks or only partly meets: honest (closest real experience shown, "
                    "nothing faked), overstated (the output claims more than the material supports), hidden_or_ignored")


class Fit(BaseModel):
    requirements: list[Requirement]
    gap_handling_summary: str
    score: int = Field(ge=0, le=10, description="How well the CV presents the candidate's real fit to this ad (10 best)")


class CVStandards(BaseModel):
    score: int = Field(ge=0, le=10)
    has_contact_details: bool
    has_summary_or_profile: bool
    experience_reverse_chronological: bool
    dates_consistent: bool
    length_appropriate: bool = Field(description="Roughly 1-2 pages for the candidate's seniority")
    language_matches_ad: bool = Field(description="Written in the language the ad expects")
    artefacts: list[str] = Field(description="Placeholders, leftover instructions, markdown symbols, duplicated sections, "
                                             "garbled text or LaTeX residue found in the CV text or PDF text")
    issues: list[str]


class ATS(BaseModel):
    pdf_text_parseable: bool = Field(description="The PDF text reads in a sensible order with words intact")
    standard_headings: bool
    headings_found: list[str]
    ad_keywords: list[str] = Field(description="The main hard-skill and role keywords of the ad (10-25)")
    keywords_in_cv: list[str]
    keywords_missing_from_cv: list[str]
    keywords_missing_but_candidate_has: list[str] = Field(
        description="Missing keywords the candidate's material shows they have (lost opportunities)")
    keyword_coverage_pct: int = Field(ge=0, le=100)
    pass_likelihood: Literal["low", "medium", "high"] = Field(
        description="How likely the CV passes an automatic screening for this ad")
    reasons: list[str]


class CoverLetter(BaseModel):
    score: int = Field(ge=0, le=10)
    addressed_to_this_company_and_role: bool
    specific_to_company: bool = Field(description="Uses facts about this company, not generic praise")
    consistent_with_cv: bool
    stale_framing: list[str] = Field(description="Quotes where the letter reuses the earlier letter's context wrongly "
                                                 "(old employer, old role, outdated 'currently' statements)")
    tone_and_length_ok: bool
    issues: list[str]
    strengths: list[str]


class WorkArrangement(BaseModel):
    arrangement_in_sources: Literal["remote", "hybrid", "on-site", "not stated"] = Field(
        description="What the job ad and company information say about how the job is done")
    statements_in_outputs: list[str] = Field(
        description="Quotes from the CV or cover letter about work arrangement, location, relocation, availability "
                    "or the candidate's preference")
    invented_or_wrong: list[str] = Field(
        description="Quotes that state an arrangement, location limit, culture trait or candidate preference that the "
                    "sources or the candidate's stated preference do not support, or that contradict them")
    culture_used: list[str] = Field(description="Culture or values from the company information that the cover letter uses")
    preference_handling: Literal["not_given_and_kept_open", "not_given_but_stated", "not_given_but_narrowed",
                                 "given_and_used_honestly", "given_but_misstated", "given_and_not_mentioned"] = Field(
        description="not_given_and_kept_open: no preference given, and the outputs say nothing about it or use only "
                    "neutral, open wording; not_given_but_stated: a specific preference or commitment was invented; "
                    "not_given_but_narrowed: the outputs present the job's arrangement or location as a conflict or "
                    "weakness, or narrow the candidate's options")
    conflict_handling: Literal["no_conflict", "conflict_handled_honestly", "conflict_hidden_or_misstated"] = Field(
        description="If the candidate's preference conflicts with the job's arrangement: do the outputs avoid "
                    "misstating either side?")
    score: int = Field(ge=0, le=10)


class Evaluation(BaseModel):
    claims: list[Claim] = Field(description="Every factual claim about the candidate in both outputs")
    omissions: list[Omission]
    fit: Fit
    cv_standards: CVStandards
    ats: ATS
    cover_letter: CoverLetter
    work_arrangement: WorkArrangement
    invented_content_score: int = Field(ge=0, le=10, description="10 = nothing invented or stretched, 0 = much invented")
    completeness_score: int = Field(ge=0, le=10, description="10 = every relevant fact used")
    overall_score: int = Field(ge=0, le=10)
    summary: str = Field(description="Three to five sentences: the main strengths and problems")


SYSTEM_PROMPT = """You are a strict, experienced technical recruiter and CV auditor. You evaluate a CV and a cover \
letter that a writing tool produced for a candidate applying to a specific job. You judge only from the documents \
given to you. You do not know how the documents were produced and you must not guess.

Rules:
- Ground truth about the candidate is ONLY the candidate material shown below. Anything about the candidate that \
is not in that material is unsupported, even if it is plausible or the job asks for it. Rephrasing and obvious \
inference are supported (e.g. "built a React app" supports "JavaScript"). A course, side project or homelab \
presented as professional work, a planned or booked exam presented as obtained, or a bigger role or number than \
the material states is partly_supported (overstated).
- Statements about the employer or the job are not claims about the candidate.
- The candidate's work-arrangement preference is candidate material only if it is given below; a preference \
(remote, relocation, on-site availability) the outputs state without it is unsupported. Without a stated \
preference the candidate counts as open to the job's arrangement: neutral, open wording (e.g. "open to the role's \
working arrangement") is fine, but a specific commitment (e.g. "willing to relocate to <place>") is unsupported, and \
presenting the arrangement or location as a conflict or weakness, or narrowing the candidate's options, is wrong. \
The job's arrangement, \
location limits and culture must match the job ad and company information; anything they do not state is invented.
- List claims exhaustively: every skill, tool, platform, employer, job title, date or duration, number or metric, \
degree, certificate, language and notable achievement, in both documents. Quote briefly.
- Quote evidence verbatim from the candidate material. Never invent quotes.
- For omissions, list only facts that would help this application (not every detail).
- Be calibrated: use the full score ranges; 10 is rare."""

HUMAN_TEMPLATE = """# JOB AD
{job_ad}

# COMPANY INFORMATION (web search results the writing tool had)
{company}

# CANDIDATE MATERIAL GIVEN TO THE WRITING TOOL
The tool was given only these documents about the candidate:

{candidate}

# CANDIDATE'S OWN WORK-ARRANGEMENT PREFERENCE
{work_preference}

# OUTPUT 1: CV (text)
{cv}

# OUTPUT 1b: TEXT EXTRACTED FROM THE CV PDF
{cv_pdf}

# OUTPUT 2: COVER LETTER (text)
{cover_letter}

# OUTPUT 2b: TEXT EXTRACTED FROM THE COVER LETTER PDF
{cover_letter_pdf}

Evaluate both outputs against the rubric."""


# ---------------------------------------------------------------------------------------------
def pdf_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    from agents.pdf_text_check import extract_pdf_text
    text, _, error = extract_pdf_text(path)
    return text if text is not None else f"[PDF text extraction failed: {error}]"


# Keys only meaningful when there is a text version to compare the PDF with.
_COMPARISON_KEYS = ("text_words_found_in_pdf_pct", "missing_words", "extra_words", "missing_word_count",
                    "extra_word_count", "ok")


def pdf_checks(path: Path, text_version: str | None) -> dict:
    """Deterministic checks on a PDF (the app's own agents/pdf_text_check.py): pages, extractable text,
    how much of the text version survives, words only on one side, leftover [PLACEHOLDERS]."""
    if not path.is_file():
        return {"exists": False}
    from agents.pdf_text_check import check_pdf
    result = check_pdf(path, text_version)
    if "error" in result:
        return {"exists": True, "error": result["error"]}
    if not text_version:
        result = {k: v for k, v in result.items() if k not in _COMPARISON_KEYS}
    return {"exists": True, **result}


def _read(path: Path) -> str | None:
    if not path.is_file():
        return None
    if path.suffix.lower() == ".pdf":
        return pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def company_text(fixture_path: Path) -> str:
    """The search results as the app's company research reads them (agents.search.search_result_texts:
    page text trimmed per result, else the snippet), so the evaluator sees what the writer could see."""
    if not fixture_path.is_file():
        return "(none)"
    from agents.search import search_result_texts
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    parts = []
    for search in fixture.get("searches", []):
        for r in search_result_texts(search.get("response")):
            parts.append(f"## {r['title']}\nURL: {r['url']}\n{r['text']}")
    return "\n\n".join(parts) if parts else "(the search found nothing)"


def build_prompt(run_dir: Path) -> tuple[dict, dict]:
    """The evaluator's input (no model, input-set or run identifiers) and the deterministic checks."""
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"
    candidate = []
    for key, label in INPUT_LABELS.items():
        for suffix in (".txt", ".md", ".pdf"):
            text = _read(inputs / f"{key}{suffix}")
            if text:
                candidate.append(f"## {label}\n{text.strip()}")
                break
    cv = _read(outputs / "final_cv.txt")
    cl = _read(outputs / "final_cover_letter.txt")
    values = {
        "job_ad": (_read(inputs / "job_ad.txt") or "").strip(),
        "company": company_text(inputs / "search_fixture.json"),
        "candidate": "\n\n".join(candidate) or "(none)",
        "work_preference": (_read(inputs / "work_preference.txt") or "").strip()
        or "(not given: the candidate stated no preference, so they count as open to the job's arrangement)",
        "cv": cv or "(no CV was produced)",
        "cv_pdf": pdf_text(outputs / "final_cv.pdf") or "(no CV PDF was produced)",
        "cover_letter": cl or "(no cover letter was produced)",
        "cover_letter_pdf": pdf_text(outputs / "final_cover_letter.pdf") or "(no cover letter PDF was produced)",
    }
    checks = {"cv_pdf": pdf_checks(outputs / "final_cv.pdf", cv),
              "cover_letter_pdf": pdf_checks(outputs / "final_cover_letter.pdf", cl)}
    return values, checks


def counts(ev: dict) -> dict:
    claims = ev.get("claims") or []
    by = lambda doc, v: sum(1 for c in claims if c["document"] == doc and c["verdict"] == v)  # noqa: E731
    reqs = (ev.get("fit") or {}).get("requirements") or []
    must = [r for r in reqs if r["priority"] == "must_have"]
    return {
        "claims": len(claims),
        "cv_unsupported": by("cv", "unsupported"), "cv_partly": by("cv", "partly_supported"),
        "cl_unsupported": by("cover_letter", "unsupported"), "cl_partly": by("cover_letter", "partly_supported"),
        "omissions_high": sum(1 for o in ev.get("omissions") or [] if o["relevance"] == "high"),
        "must_haves": len(must),
        "must_haves_shown": sum(1 for r in must if r["shown_in_cv"] == "yes"),
        "gaps_overstated": sum(1 for r in reqs if r["gap_handling"] == "overstated"),
        "work_arrangement_invented": len((ev.get("work_arrangement") or {}).get("invented_or_wrong") or []),
    }


def format_messages(values: dict) -> list:
    """The system and human messages sent to the evaluator model."""
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)])
    return prompt.format_messages(**values)


def evaluate(run_dir: Path, model: str, name: str) -> dict:
    from langchain.chat_models import init_chat_model
    from provenance import RunRecorder, to_jsonable

    out_dir = run_dir / "evaluation" / name
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} exists; choose another --name")
    recorder = RunRecorder(out_dir)
    values, checks = build_prompt(run_dir)
    messages = format_messages(values)
    recorder.write_json("prompt.json", {"model": model, "messages": [{"role": m.type, "content": m.content} for m in messages]})
    llm = init_chat_model(model)
    chain = llm.with_structured_output(Evaluation, include_raw=True)
    result = chain.invoke(messages, config={"callbacks": [recorder.callback]})
    raw = result.get("raw")
    recorder.write_json("raw_response.json", {
        "content": getattr(raw, "content", None),
        "additional_kwargs": getattr(raw, "additional_kwargs", None),
        "response_metadata": getattr(raw, "response_metadata", None),
        "usage_metadata": getattr(raw, "usage_metadata", None),
        "parsing_error": str(result.get("parsing_error")) if result.get("parsing_error") else None,
    })
    parsed = result.get("parsed")
    if parsed is None:
        recorder.write_summary(status="parse_error")
        raise RuntimeError(f"evaluator output did not parse: {result.get('parsing_error')}")
    ev = to_jsonable(parsed)
    record = {"evaluator_model": model, "name": name, "evaluation": ev, "counts": counts(ev), "pdf_checks": checks,
              "cost_usd": round(recorder.cost, 6)}
    recorder.write_json("evaluation.json", record)
    recorder.write_summary(status="ok")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--name", default="eval1", help="subfolder under RUN/evaluation (use eval2 for a re-score)")
    parser.add_argument("--env-file", help=".env file with the OpenAI key (read, never copied)")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(args.env_file or REPO / ".env")
    record = evaluate(args.run_dir.resolve(), args.model, args.name)
    print(json.dumps({"run_dir": str(args.run_dir), "name": args.name, "counts": record["counts"],
                      "overall": record["evaluation"]["overall_score"],
                      "ats": record["evaluation"]["ats"]["pass_likelihood"], "cost_usd": record["cost_usd"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
