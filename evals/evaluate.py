"""Independent evaluator: score one testing-mode run folder with a stronger model.

The evaluator is blind to how the run was made: it never sees the model, the input-set name or the
run id. It gets the job ad, the company information the writer had (the replayed search results),
the candidate documents the writer was given (omitted ones are simply absent), the final CV and
cover letter text, and the plain text extracted from the two PDFs (pdfplumber, no model; for the CV this is
"what an ATS would read"). It answers a fixed rubric as JSON with
quoted evidence:

  (a) invented content   every skill, tool, employer, title, date, number, degree, certificate in the
                         outputs, marked supported (with a quote from the candidate's material) or not
  (b) completeness       facts in the material that matter for this ad but were left out
  (c) fit to the ad      must-have and nice-to-have coverage, and whether gaps are handled honestly
  (d) CV standards       structure, length, contact details, chronology, language, artefacts
  (e) ATS screening      one 0-10 score with a reason, judged from what an ATS would read (the plain text a
                         standard extractor pulls from the CV PDF): readable, in order, clear sections; the
                         ad's important requirements the candidate has shown in searchable wording; keyword
                         stuffing and unsupported keywords flagged (they lower the score)
  (f) cover letter       specificity, consistency with the CV, stale framing from the old letter
  (g) work arrangement   remote / hybrid / on-site, location limits and culture used correctly and not
                         invented; the candidate's preference stated only if given, conflicts not hidden
  (h) visual layout      from images of the PDF pages: layout, spacing, overflow or cut-off text, page
                         count, consistency, readability (deterministic measurements alongside)
  (i) language quality   grammar, fluency, tone, and the right language for the ad or the user's request
  (j) additional info    facts from the notes and from what the user said in the chat: used, ignored, misused
  (k) user requests      each request the user made in the conversation: followed, partly followed,
                         refused correctly, refused wrongly, ignored, or (for an untruthful one) followed
                         wrongly; the test script's expected behaviour is given for each

Only requests the scripted user actually sent are graded. Which ones were sent is decided in code from
transcript.json (scripted_request_status), never by the model: a scripted request or wish that was
never sent is "not exercised". It is left out of the checklist (and so of user_requests and of any
score it would change, such as the language of a letter asked for in German); the script's overall
expectation is withheld when none of its requests was sent and limited to the sent ones otherwise;
evaluation.json lists every scripted request with its status under "scripted_requests" and counts the
unsent ones in counts.requests_not_exercised.

Two grading modes:
  document-only  (default, the mode of the method comparison) the grader sees the documents (text, PDF text,
                 page images, layout measurements), the inputs, and a fixed user-instructions block built from
                 the script (user_instructions), never the chat, the app's messages, the manifest, the method or
                 the writer model. Every arm of a default-script case gets the same block.
  conversation   (the 2026-11-10 round) the grader also sees the conversation (the app's messages shortened,
                 the user's replies in full) and, from the test script, what each user request expects. What
                 the user says about themselves in the chat is candidate material (source "chat").

The grader is "provider:model" (evals/grader.py): anthropic:claude-sonnet-5-5 (default), anthropic:claude-opus-5-5,
anthropic:claude-haiku-5-5 or an OpenAI model such as openai:gpt-6.1-sol. build_grading_request() builds the
whole request without sending it (Batch grading reuses it); evaluate() sends it and writes the results.

Results go to RUN/evaluation/<name>/ (eval1, eval2, opus1, haiku1, ...; an existing folder is never
overwritten): prompt.json, raw_response.json, evaluation.json (the rubric answer, the grader record -
provider, model requested and returned, effort, temperature, grading mode, rubric and prompt hashes,
tokens, cost, stop reason - plus deterministic PDF checks and counts), llm_calls.jsonl, summary.json
(cost) and pages/ (the PNG page images the evaluator saw).

Usage:
    .venv/bin/python -m evals.evaluate RUN_DIR [--grader-model anthropic:claude-sonnet-5-5] [--eval-name eval1]
        [--grading-mode document-only|conversation] [--effort medium] [--env-file .env]
(--model and --name are the old spellings of --grader-model and --eval-name.)
This makes one paid model call per evaluation (text and page images in the same call), and at most one
more if the answer does not parse. If the provider reports no credit left or a reached spend limit,
nothing is retried, summary.json says "interrupted: out of credit" and the exit code is 75
(evals/credit.py). A refusal or an unparseable answer is recorded in summary.json and exits 1.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"
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
    source: Literal["cv", "portfolio", "cover_letter", "notes", "chat", "none"]
    evidence_quote: str = Field(description="Short verbatim quote from the candidate's material that supports it; empty if none")
    note: str = Field(description="Why partly_supported or unsupported; empty otherwise")


class Omission(BaseModel):
    fact: str = Field(description="A fact about the candidate, relevant to this ad, that neither output uses")
    source: Literal["cv", "portfolio", "cover_letter", "notes", "chat"]
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
    """Judged from WHAT AN ATS WOULD READ (the plain text a standard extractor pulls from the CV PDF), the job
    ad and the candidate's material: one 0-10 score with its reason, and the three judgements behind it."""
    text_readable_in_order: bool = Field(
        description="The extracted text reads in a sensible order with words, dates and contact details intact: "
                    "no garbled characters, merged words, interleaved columns or text missing compared with the "
                    "page images")
    clear_sections: bool = Field(description="Standard section headings a parser can find (e.g. experience, "
                                             "education, skills) and every entry under the right one")
    headings_found: list[str]
    parse_issues: list[str] = Field(description="Quotes from the extracted text that an ATS would misread: garbled, "
                                                "out of order, merged, split or missing; empty if none")
    important_requirements: list[str] = Field(
        description="The ad's important requirements a recruiter's search would look for: role title, must-have "
                    "skills, tools, qualifications, languages (short form, about 5-20)")
    covered_in_searchable_wording: list[str] = Field(
        description="Those the extracted text visibly shows in wording a recruiter's keyword search would find "
                    "(the ad's term or its standard name, not only a paraphrase)")
    missed_but_candidate_has: list[str] = Field(
        description="Those the candidate's material shows they have but the extracted text does not show in "
                    "searchable wording (lost opportunities)")
    keyword_stuffing: list[str] = Field(
        description="Keywords repeated without context, lists of the ad's terms pasted in, or hidden text; quoted")
    unsupported_keywords: list[str] = Field(
        description="Ad keywords or skills the CV claims that the candidate's material does not support")
    score: int = Field(ge=0, le=10, description="How well the CV would do in automatic screening for this ad, "
                                                "honestly: 10 = clean, well-sectioned text that shows every "
                                                "important requirement the candidate really has in searchable "
                                                "wording. Requirements the candidate lacks cost nothing when left "
                                                "out; parse problems and lost opportunities lower the score, and "
                                                "keyword stuffing or unsupported keywords lower it, never raise it")
    reason: str = Field(description="One or two sentences: why this score")


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


class VisualLayout(BaseModel):
    cv_pages_seen: int = Field(ge=0, description="CV pages shown as images")
    cover_letter_pages_seen: int = Field(ge=0)
    page_count_appropriate: bool = Field(description="CV 1-2 pages for the seniority (or what the user asked), "
                                                     "cover letter one page")
    overflow_or_cutoff: list[str] = Field(description="Text cut off, running past a margin or page edge, overlapping "
                                                      "or hidden; where")
    spacing_issues: list[str] = Field(description="Cramped or uneven spacing, large empty areas, orphaned headings "
                                                  "or lines, a nearly empty last page")
    consistency_issues: list[str] = Field(description="Inconsistent fonts, sizes, bullet styles, date formats or "
                                                      "alignment; the CV and letter not looking like one set")
    readability: Literal["good", "acceptable", "poor"]
    issues: list[str]
    score: int = Field(ge=0, le=10, description="10 = clean professional layout; 0 = unusable. If no images were "
                                                "given, judge only from the measurements and say so in issues")


class LanguageQuality(BaseModel):
    expected_language_cv: str = Field(description="The language the CV should be in: the ad's, unless the user asked "
                                                  "for another")
    expected_language_cover_letter: str = Field(description="As for the CV; a user request in the chat for the letter's "
                                                            "language wins")
    cv_language: str
    cover_letter_language: str
    right_language_cv: bool
    right_language_cover_letter: bool
    grammar_and_spelling_errors: list[str] = Field(description="Quotes of real errors, with the correction")
    fluency_and_tone_issues: list[str] = Field(description="Awkward, machine-like, mixed-language or wrong-register "
                                                           "passages, quoted")
    score: int = Field(ge=0, le=10)


class AdditionalInformation(BaseModel):
    notes_facts_used: list[str] = Field(description="Relevant facts only in the candidate's notes that the outputs use")
    notes_facts_ignored: list[str] = Field(description="Relevant facts only in the notes that the outputs leave out")
    chat_facts_used: list[str] = Field(description="Facts the user gave in the conversation (notice period, details, "
                                                   "contact data) that the outputs use")
    chat_facts_ignored: list[str] = Field(description="Relevant facts from the conversation the outputs leave out")
    misused: list[str] = Field(description="Notes or chat facts distorted, overstated, or used where the user said not to")
    score: int = Field(ge=0, le=10, description="10 = every relevant extra fact used correctly; use 10 if there was "
                                                "no extra information")


class RequestCompliance(BaseModel):
    request: str = Field(description="The user's request, quoted briefly from the conversation")
    expected: Literal["follow", "refuse"] = Field(description="From the test script's expectation when given; else "
                                                              "follow, or refuse for a request to state something untrue")
    outcome: Literal["followed", "partly_followed", "refused_correctly", "refused_wrongly", "ignored",
                     "followed_untruthful"] = Field(
        description="followed / partly_followed: the change is in the final outputs (fully / in part); "
                    "refused_correctly: an untruthful request was not followed (ideally the app said so); "
                    "refused_wrongly: a legitimate request was declined; ignored: a legitimate request is not in the "
                    "final outputs and the app did not say why; followed_untruthful: the outputs now state the untrue "
                    "thing")
    app_explained: bool = Field(description="The app told the user what it did or why it did not")
    evidence: str = Field(description="Quote from the final outputs or the app's reply that shows the outcome")
    note: str


class UserRequests(BaseModel):
    requests: list[RequestCompliance] = Field(description="Every request the user made in the conversation, beyond "
                                                          "simply asking for the CV, the letter or to exit")
    score: int = Field(ge=0, le=10, description="10 = every legitimate request followed and every untruthful one "
                                                "refused; use 10 if the user made no requests")


class Evaluation(BaseModel):
    claims: list[Claim] = Field(description="Every factual claim about the candidate in both outputs")
    omissions: list[Omission]
    fit: Fit
    cv_standards: CVStandards
    ats: ATS
    cover_letter: CoverLetter
    work_arrangement: WorkArrangement
    visual_layout: VisualLayout
    language_quality: LanguageQuality
    additional_information: AdditionalInformation
    user_requests: UserRequests
    invented_content_score: int = Field(ge=0, le=10, description="10 = nothing invented or stretched, 0 = much invented")
    completeness_score: int = Field(ge=0, le=10, description="10 = every relevant fact used")
    overall_score: int = Field(ge=0, le=10)
    summary: str = Field(description="Three to five sentences: the main strengths and problems")


_SYSTEM_INTRO = """You are a strict, experienced technical recruiter and CV auditor. You evaluate a CV and a cover \
letter that a writing tool produced for a candidate applying to a specific job. You judge only from the documents \
given to you. You do not know how the documents were produced and you must not guess.

Rules:
"""
_RULES_COMMON = """- Ground truth about the candidate is ONLY the candidate material shown below. Anything about the candidate that \
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
"""
_RULES_CONVERSATION = """- What the user said about themselves in the conversation is candidate material too (source "chat"), except when \
the user asks to state something they themselves say is untrue, or that their material contradicts: such a request \
must not be followed, and following it makes the claim unsupported.
- A user request in the conversation changes what is correct: a CV or letter in another language, shorter, reordered \
or in another tone is right if the user asked for it. The test checklist gives the expected behaviour of the \
requests the test made; judge every request against it. Grade only requests the user actually made in the \
conversation shown: a wish that is not in the user's replies was never asked and changes nothing.
"""
_RULES_DOCUMENT_ONLY = """- The user's instructions below are everything the user told the writing tool. Anything the user says about \
themselves there is candidate material too (source "chat"), except when the user asks to state something they \
themselves say is untrue, or that their material contradicts: such a request must not be followed, and following it \
makes the claim unsupported.
- You see the documents and the user's instructions, not the conversation. A user instruction changes what is \
correct: a CV or letter in another language, shorter, reordered or in another tone is right if the user asked for \
it. Grade user_requests against the user's instructions only, with the expected behaviour where it is given, and \
judge each one by the final documents alone; the tool's replies are not shown, so app_explained is false. Asking \
for the CV and then the cover letter is followed when both documents exist. Nothing else was asked.
"""
_RULES_END = """- The PDF is the document the employer receives; the plain text is the tool's own copy of it. Judge \
claims, layout and ATS from the PDF (its extracted text and page images); a difference between the PDF and the text \
is itself an issue.
- The page images show the PDFs as an employer sees them; the layout measurements come from the PDF itself. Judge \
the visual layout from both.
- Judge the ats section from WHAT AN ATS WOULD READ (the plain text extracted from the CV PDF), not from the CV text; \
the page images only show what that text lost or scrambled. Is that text readable, in order and in clear sections; are the ad's important requirements \
that the candidate really has visible in wording a recruiter's search would find; and is there keyword stuffing or \
a skill the candidate's material does not support. Cheating to match the ad lowers the ATS score.
- Be calibrated: use the full score ranges; 10 is rare."""

SYSTEM_PROMPT = _SYSTEM_INTRO + _RULES_COMMON + _RULES_CONVERSATION + _RULES_END
SYSTEM_PROMPT_DOCUMENT_ONLY = _SYSTEM_INTRO + _RULES_COMMON + _RULES_DOCUMENT_ONLY + _RULES_END

# The prompt is the shared part (identical for every arm of a case and every re-grade: the cache
# breakpoint goes after it) followed by this session's documents.
SHARED_TEMPLATE = """# JOB AD
{job_ad}

# COMPANY INFORMATION (web search results the writing tool had)
{company}

# CANDIDATE MATERIAL GIVEN TO THE WRITING TOOL
The tool was given only these documents about the candidate:

{candidate}

# CANDIDATE'S OWN WORK-ARRANGEMENT PREFERENCE
{work_preference}

"""
_OUTPUTS_TEMPLATE = """# OUTPUT 1: CV (text)
{cv}

# OUTPUT 1b: WHAT AN ATS WOULD READ (plain text extracted from the CV PDF by a standard text extractor)
{cv_pdf}

# OUTPUT 2: COVER LETTER (text)
{cover_letter}

# OUTPUT 2b: TEXT EXTRACTED FROM THE COVER LETTER PDF
{cover_letter_pdf}

"""
_LAYOUT_TEMPLATE = """# LAYOUT MEASUREMENTS OF THE PDFS
{layout}

# PAGE IMAGES
{images_note}

Evaluate both outputs against the rubric."""
DOCUMENTS_TEMPLATE = _OUTPUTS_TEMPLATE + """# CONVERSATION BETWEEN THE USER AND THE WRITING TOOL
(The tool's messages are shortened; the user's replies are complete.)
{conversation}

# TEST CHECKLIST: THE USER'S REQUESTS AND THE CORRECT BEHAVIOUR
{checklist}

""" + _LAYOUT_TEMPLATE

# Document-only mode: the user's instructions (the same words for every arm of a case) take the place of
# the conversation and the checklist, and sit in the shared part.
SHARED_TEMPLATE_DOCUMENT_ONLY = SHARED_TEMPLATE + """# THE USER'S INSTRUCTIONS TO THE WRITING TOOL
{user_instructions}

"""
DOCUMENTS_TEMPLATE_DOCUMENT_ONLY = _OUTPUTS_TEMPLATE + _LAYOUT_TEMPLATE

CONVERSATION, DOCUMENT_ONLY = "conversation", "document-only"
GRADING_MODES = (DOCUMENT_ONLY, CONVERSATION)
DEFAULT_MODE = DOCUMENT_ONLY
_MODE_PROMPTS = {CONVERSATION: (SYSTEM_PROMPT, SHARED_TEMPLATE, DOCUMENTS_TEMPLATE),
                 DOCUMENT_ONLY: (SYSTEM_PROMPT_DOCUMENT_ONLY, SHARED_TEMPLATE_DOCUMENT_ONLY,
                                 DOCUMENTS_TEMPLATE_DOCUMENT_ONLY)}


def rubric_sha256(mode: str = DEFAULT_MODE) -> str:
    """Hash of the rubric a grading in this mode uses (system prompt, templates, answer schema)."""
    from evals.grader import rubric_hash
    system, shared, documents = _MODE_PROMPTS[mode]
    return rubric_hash(system, [shared, documents], Evaluation)


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


APP_MESSAGE_CHARS = 700  # the tool's messages repeat whole documents; the outputs are shown in full above
MAX_CV_PAGE_IMAGES = 3
MAX_LETTER_PAGE_IMAGES = 2


def conversation_text(run_dir: Path) -> str:
    """The transcript as the evaluator reads it: the tool's messages shortened, the user's replies in full."""
    path = run_dir / "transcript.json"
    if not path.is_file():
        return "(no conversation was recorded)"
    turns = json.loads(path.read_text(encoding="utf-8")).get("turns") or []
    lines = []
    for t in turns:
        app = " ".join(str(t.get("assistant") or "").split())
        if len(app) > APP_MESSAGE_CHARS:
            app = app[:APP_MESSAGE_CHARS] + " [...]"
        lines.append(f"TOOL: {app or '(no message)'}\nUSER: {t.get('user')}")
    return "\n\n".join(lines) or "(no conversation was recorded)"


SENT, NOT_EXERCISED, REPLACED = "sent", "not exercised", "replaced"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def scripted_request_status(run_dir: Path) -> list[dict]:
    """Every scripted expectation and request of the run with whether it reached the app, decided from
    inputs/script.json and transcript.json alone (no model):

      sent            the user's replies contain it (an expectation's reply text, or a logged request)
      replaced        not sent itself, but the reply or request that carries the same wish was
                      (a request's "unless_sent" reply, or an expectation whose fallback request was sent)
      not exercised   never sent: the app took another path; it must not be graded
    """
    from evals.run_case import script_requests
    script = _load_json(run_dir / "inputs" / "script.json")
    if not script:
        return []
    transcript = _load_json(run_dir / "transcript.json")
    said = {str(t.get("user")) for t in transcript.get("turns") or []}
    logged = {r.get("id"): r for r in transcript.get("requests") or []}
    requests = script_requests(script)
    sent_keys = {key for key in script if isinstance(script.get(key), str) and script[key] in said}
    out = []
    for e in script.get("expectations") or []:
        key, text = e.get("about"), script.get(e.get("about"))
        fallbacks = [r["id"] for r in requests if r.get("unless_sent") == key and r["id"] in logged]
        status = SENT if key in sent_keys else REPLACED if fallbacks else NOT_EXERCISED
        out.append({"kind": "reply", "id": key, "user_said": text, "expect": e.get("expect", "follow"),
                    "correct_behaviour": e.get("correct_behaviour", ""), "status": status,
                    **({"replaced_by": fallbacks[0]} if status == REPLACED else {})})
    for r in requests:
        if r["id"] in logged:
            status = SENT
        elif r.get("unless_sent") in sent_keys:
            status = REPLACED
        else:
            status = NOT_EXERCISED
        text = logged.get(r["id"], {}).get("text") or r["text"]
        out.append({"kind": "request", "id": r["id"], "user_said": text, "expect": r.get("expect", "follow"),
                    "correct_behaviour": r.get("correct_behaviour", ""), "status": status,
                    **({"replaced_by": r["unless_sent"]} if status == REPLACED else {})})
    return out


def request_checklist(run_dir: Path) -> list[dict]:
    """The test script's expectations for the requests the user actually sent (scripted_request_status):
    what the user said, follow or refuse, and the correct behaviour. Unsent ones are left out."""
    return [{"user_said": i["user_said"], "expect": i["expect"], "correct_behaviour": i["correct_behaviour"]}
            for i in scripted_request_status(run_dir) if i["status"] == SENT]


def _checklist_text(items: list[dict], script: dict, not_exercised: int = 0) -> str:
    """The checklist the evaluator reads. Unsent scripted requests are never named. When none of the
    script's requests was sent, its overall expectation (which describes them) is withheld too; when only
    some were, it is kept but limited to the requests listed, so the evaluator cannot grade the app
    against a wish it never received."""
    parts = []
    if script.get("correct_behaviour") and (items or not not_exercised):
        limit = " (judge this only as far as the requests listed below were sent)" if not_exercised else ""
        parts.append(f"Overall{limit}: {script['correct_behaviour']}")
    for i, item in enumerate(items, 1):
        parts.append(f"{i}. The user said: \"{item['user_said']}\"\n   Expected: {item['expect']}. "
                     f"{item['correct_behaviour']}".rstrip())
    if not_exercised:
        parts.append("(Some of the test's planned requests were never sent in this conversation. They are not "
                     "listed, are not part of this evaluation, and must not change any score.)")
    return "\n".join(parts) or ("(no test expectations: judge every request in the conversation; legitimate ones "
                                 "should be followed, untruthful ones refused)")


# The default user's words, in the order the tool may hear them (evals.run_case.DEFAULT_SCRIPT); a script may
# reword a reply, and its own wording is used then.
_INSTRUCTION_REPLIES = (("first", "The user's request"),
                        ("missing_details", "If the tool asks for contact details the candidate's material lacks"),
                        ("clarify_job_ad", "If the tool asks about the job ad"))


def user_instructions(run_dir: Path) -> str:
    """The user-instructions block of document-only grading. It quotes the script's replies (the default
    script's words unless a script rewords them) and never shows the app's messages, the manifest or the
    method, so every arm of a default-script case gets the same block. A scripted change request is added,
    with the test's expected behaviour, only when it was sent (scripted_request_status, as in conversation
    mode); a default-script run has none."""
    from evals.run_case import DEFAULT_SCRIPT
    script = {**DEFAULT_SCRIPT, **_load_json(run_dir / "inputs" / "script.json")}
    lines = [f'- {label}: "{script[key]}"' for key, label in _INSTRUCTION_REPLIES if script.get(key)]
    status = scripted_request_status(run_dir)
    sent = [i for i in status if i["status"] == SENT]
    not_exercised = sum(1 for i in status if i["status"] == NOT_EXERCISED)
    if not sent:
        return "\n".join(lines) + "\nThe user made no other requests." + (
            "\n(Some of the test's planned requests were never sent. They are not part of this evaluation and "
            "must not change any score.)" if not_exercised else "")
    items = [{"user_said": i["user_said"], "expect": i["expect"], "correct_behaviour": i["correct_behaviour"]}
             for i in sent]
    return ("\n".join(lines) + "\nThe user also made these requests; the test's expected behaviour is given for each:\n"
            + _checklist_text(items, _load_json(run_dir / "inputs" / "script.json"), not_exercised))


def _layout_text(layout: dict) -> str:
    out = []
    for key, label in (("cv_pdf", "CV PDF"), ("cover_letter_pdf", "Cover letter PDF")):
        m = layout.get(key) or {}
        if not m.get("exists"):
            out.append(f"{label}: not produced")
            continue
        if m.get("error"):
            out.append(f"{label}: could not be read ({m['error']})")
            continue
        out.append(f"{label}: {m['pages']} page(s), fonts {', '.join(m['fonts'][:5]) or 'none'}, body "
                   f"{m['body_font_size']} pt, smallest {m['min_font_size']} pt, margins (pt) {m['margins_pt']}, "
                   f"last page {m['last_page_fill_pct']}% filled. Problems: {'; '.join(m['issues']) or 'none found'}")
    return "\n".join(out)


def _rendered_pages(pages: Path, prefix: str) -> list[Path]:
    found = [(int(m.group(1)), p) for p in pages.glob(f"{prefix}_page*.png")
             if (m := re.fullmatch(rf"{prefix}_page(\d+)\.png", p.name))]
    return [p for _, p in sorted(found)]


def page_images(run_dir: Path, out_dir: Path, reuse: bool = False) -> dict:
    """PNG images of the first CV and letter pages, written to out_dir/pages/ ({kind: [paths]}). `reuse`: take the
    images already in out_dir/pages/ (Batch grading reads a reply against the images it submitted)."""
    from evals.pdf_layout import render_pages
    if reuse:
        return {kind: _rendered_pages(out_dir / "pages", kind) for kind in ("cv", "cover_letter")}
    outputs = run_dir / "outputs"
    return {"cv": render_pages(outputs / "final_cv.pdf", out_dir / "pages", "cv", MAX_CV_PAGE_IMAGES),
            "cover_letter": render_pages(outputs / "final_cover_letter.pdf", out_dir / "pages", "cover_letter",
                                         MAX_LETTER_PAGE_IMAGES)}


def _images_note(images: dict | None, layout: dict) -> str:
    if not images:
        return "(no page images: judge the layout from the measurements only)"
    parts = []
    for kind, label in (("cv", "CV"), ("cover_letter", "Cover letter")):
        shown = len(images.get(kind) or [])
        total = (layout.get(f"{kind}_pdf") or {}).get("pages")
        parts.append(f"{label}: {shown} page image(s) attached" + (f" of {total}" if total else "")
                     + (" (no PDF)" if not total and not shown else ""))
    return "; ".join(parts) + ". The images follow this text, CV pages first, in page order."


def build_prompt(run_dir: Path, images: dict | None = None) -> tuple[dict, dict]:
    """The evaluator's input (no model, input-set or run identifiers) and the deterministic checks. The
    values cover both grading modes; the mode's templates decide which are shown."""
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
    from evals.pdf_layout import layout_checks
    layout = {"cv_pdf": layout_checks(outputs / "final_cv.pdf"),
              "cover_letter_pdf": layout_checks(outputs / "final_cover_letter.pdf")}
    checks["layout"] = layout
    script = _load_json(inputs / "script.json")
    status = scripted_request_status(run_dir)
    values.update(
        conversation=conversation_text(run_dir),
        checklist=_checklist_text(request_checklist(run_dir), script,
                                  sum(1 for i in status if i["status"] == NOT_EXERCISED)),
        layout=_layout_text(layout),
        images_note=_images_note(images, layout),
        user_instructions=user_instructions(run_dir),
    )
    return values, checks


def counts(ev: dict, scripted: list[dict] | None = None) -> dict:
    """The evaluator's answer as numbers; `scripted` (scripted_request_status) adds how many scripted
    requests were never sent (requests_not_exercised, never graded)."""
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
        **request_counts(ev),
        "requests_not_exercised": sum(1 for i in scripted or [] if i["status"] == NOT_EXERCISED),
        "layout_problems": sum(len((ev.get("visual_layout") or {}).get(k) or [])
                               for k in ("overflow_or_cutoff", "spacing_issues", "consistency_issues")),
        "wrong_language": sum(1 for k in ("right_language_cv", "right_language_cover_letter")
                              if (ev.get("language_quality") or {}).get(k) is False),
        "language_errors": len((ev.get("language_quality") or {}).get("grammar_and_spelling_errors") or []),
        "ats_score": (ev.get("ats") or {}).get("score"),
    }


def request_counts(ev: dict) -> dict:
    reqs = (ev.get("user_requests") or {}).get("requests") or []
    by = lambda *outcomes: sum(1 for r in reqs if r.get("outcome") in outcomes)  # noqa: E731
    return {
        "requests": len(reqs),
        "requests_followed": by("followed"),
        "requests_partly": by("partly_followed"),
        "requests_ignored": by("ignored"),
        "requests_refused_wrongly": by("refused_wrongly"),
        "untruthful_refused": by("refused_correctly"),
        "untruthful_followed": by("followed_untruthful"),
    }


def _image_paths(images: dict | None) -> list[Path]:
    return [Path(p) for p in (*((images or {}).get("cv") or []), *((images or {}).get("cover_letter") or []))]


def prompt_parts(values: dict, mode: str = DEFAULT_MODE) -> tuple[str, str, str]:
    """(system prompt, shared text, documents text) of a grading in `mode`."""
    if mode not in GRADING_MODES:
        raise ValueError(f"grading mode '{mode}': one of {', '.join(GRADING_MODES)}")
    system, shared, documents = _MODE_PROMPTS[mode]
    return system, shared.format(**values), documents.format(**values)


def format_messages(values: dict, images: dict | None = None, mode: str = DEFAULT_MODE) -> list:
    """The system and human messages for a LangChain (OpenAI) evaluator; page images go in the human message
    after the text, as base64 PNG image parts."""
    from evals.grader import GraderSpec, build_request
    system, shared, documents = prompt_parts(values, mode)
    return build_request(GraderSpec("openai", "-"), system, shared, documents, Evaluation,
                         _image_paths(images)).langchain_messages()


def build_grading_request(run_dir: Path, grader, out_dir: Path, mode: str = DEFAULT_MODE, reuse_pages: bool = False):
    """Everything one grading of `run_dir` sends, built without any model call: renders the page images
    into out_dir/pages/ (`reuse_pages`: takes the ones already there) and returns (evals.grader.GradingRequest,
    deterministic checks, images). `grader` is a GraderSpec or a "provider:model" string. evaluate() sends the
    request; Batch grading (evals/batch_grade.py) submits request.anthropic_params() instead and reads the reply
    with evals.grader.result_from_message."""
    from evals.grader import build_request, parse_grader
    grader = parse_grader(grader) if isinstance(grader, str) else grader
    images = page_images(run_dir, out_dir, reuse=reuse_pages)
    values, checks = build_prompt(run_dir, images)
    system, shared, documents = prompt_parts(values, mode)
    # schema_in_prompt: the Evaluation schema is too large for Anthropic's structured outputs (see evals/grader.py)
    request = build_request(grader, system, shared, documents, Evaluation, _image_paths(images),
                            rubric_sha256=rubric_sha256(mode), schema_in_prompt=True)
    return request, checks, images


class GradingFailed(RuntimeError):
    """The grader's answer did not parse after the allowed attempts, or the grader refused."""


def write_prompt(recorder, request, mode: str, images: dict) -> None:
    recorder.write_json("prompt.json", {"model": request.grader.name, "grading_mode": mode, **request.loggable(),
                                        "images": {k: [Path(p).name for p in v] for k, v in images.items()}})


def finish_evaluation(run_dir: Path, name: str, request, result, checks: dict, mode: str, recorder,
                      extra: dict | None = None) -> dict:
    """Write raw_response.json, evaluation.json and summary.json for one finished grading (`result` is an
    evals.grader.GradingResult from send() or from a batch reply). Raises GradingFailed after recording
    a refusal or an unparseable answer."""
    recorder.write_json("raw_response.json", {"attempts": result.attempts})
    from evals.run_case import _package_versions
    grading = {**result.record(), "grading_mode": mode, "rubric_sha256": request.rubric_sha256,
               "prompt_sha256": request.prompt_hash(), "packages": _package_versions(), **(extra or {})}
    if result.answer is None:
        recorder.write_summary(status=result.status, grading=grading)
        why = "the grader refused" if result.status == "refused" else "evaluator output did not parse"
        raise GradingFailed(f"{why}: {result.attempts[-1].get('parse_error') if result.attempts else ''}")
    ev = result.answer
    scripted = scripted_request_status(run_dir)
    record = {"evaluator_model": request.grader.name, "name": name, "grading_mode": mode, "grader": grading,
              "evaluation": ev, "counts": counts(ev, scripted), "scripted_requests": scripted,
              "pdf_checks": checks, "cost_usd": result.cost_usd}
    recorder.write_json("evaluation.json", record)
    recorder.write_summary(status="ok", grading=grading)
    return record


def evaluate(run_dir: Path, model: str, name: str, mode: str = DEFAULT_MODE, effort: str | None = None,
             client=None) -> dict:
    """Grade one run folder into RUN/evaluation/<name>/ (never overwritten). `client` replaces the
    Anthropic client (tests)."""
    from evals.credit import INTERRUPTED_STATUS, OutOfCredit
    from evals.grader import parse_grader, send
    from provenance import RunRecorder

    grader = parse_grader(model, effort)
    if mode not in GRADING_MODES:
        raise ValueError(f"grading mode '{mode}': one of {', '.join(GRADING_MODES)}")
    out_dir = run_dir / "evaluation" / name
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} exists; choose another --eval-name")
    recorder = RunRecorder(out_dir)
    request, checks, images = build_grading_request(run_dir, grader, out_dir, mode)
    write_prompt(recorder, request, mode, images)
    try:
        result = send(request, recorder, node="evaluator", client=client)
    except OutOfCredit as e:
        recorder.write_summary(status=INTERRUPTED_STATUS, error=str(e)[:500])
        raise
    return finish_evaluation(run_dir, name, request, result, checks, mode, recorder)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--grader-model", "--model", dest="model", default=DEFAULT_MODEL,
                        help=f"provider:model (default {DEFAULT_MODEL}; see evals/grader.py)")
    parser.add_argument("--eval-name", "--name", dest="name", default="eval1",
                        help="subfolder under RUN/evaluation: eval1, eval2, opus1, haiku1, ... (never overwritten)")
    parser.add_argument("--grading-mode", choices=GRADING_MODES, default=DEFAULT_MODE,
                        help="document-only (default): documents and a fixed user-instructions block, no chat; "
                             "conversation: the app's messages and the test checklist, as before")
    parser.add_argument("--effort", help="Anthropic effort (default medium)")
    parser.add_argument("--env-file", help=".env file with the API keys (read, never copied)")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(args.env_file or REPO / ".env")
    from evals.credit import EXIT_OUT_OF_CREDIT, OutOfCredit
    try:
        record = evaluate(args.run_dir.resolve(), args.model, args.name, args.grading_mode, args.effort)
    except ValueError as e:
        sys.exit(str(e))
    except OutOfCredit as e:
        print(f"out of credit: the evaluation was interrupted ({e})", file=sys.stderr)
        return EXIT_OUT_OF_CREDIT
    except GradingFailed as e:
        print(f"grading failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"run_dir": str(args.run_dir), "name": args.name, "grader": args.model,
                      "grading_mode": args.grading_mode, "counts": record["counts"],
                      "overall": record["evaluation"]["overall_score"],
                      "ats": record["evaluation"]["ats"]["score"], "cost_usd": record["cost_usd"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
