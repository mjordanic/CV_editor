"""Deterministic checks of a rendered PDF against the plain text it was made from.

The critic and the invented-claims check validate the plain text, but the employer and the ATS read
the PDF, which a model re-types into LaTeX (and a second model may repair). These checks compare the
words of the two, so a paraphrase, an added sentence or a dropped section during that step is seen.
The app (agents/latex_pdf_generator.py) and the evaluator (evals/evaluate.py) use the same functions.
"""
from __future__ import annotations

import re
from pathlib import Path

# Bracketed all-caps placeholders such as [EMAIL] or [PHONE NUMBER] that a writer leaves for missing details.
PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z][A-Z _/-]{2,}\]")

# Thresholds for a PDF whose text matches its plain text closely enough. Checked on the PDFs of an
# earlier evaluation round: a cover letter whose LaTeX step dropped one sentence had 9 distinct
# words missing (96.3% found); the other 19 PDFs had at most 2 missing and none added.
MIN_WORDS_FOUND_PCT = 97.0  # share of the plain text's words that appear in the PDF text
MAX_MISSING_WORDS = 3       # distinct plain-text words not in the PDF at all
MAX_EXTRA_WORDS = 3         # distinct PDF words not in the plain text at all


def words(text: str | None) -> list[str]:
    """Lower-case word tokens (letters, digits, underscore; any script)."""
    return re.findall(r"\w+", (text or "").lower())


def find_placeholders(text: str | None) -> list[str]:
    """Distinct bracketed all-caps placeholders in the text, in order of first appearance."""
    seen: list[str] = []
    for match in PLACEHOLDER_PATTERN.findall(text or ""):
        if match not in seen:
            seen.append(match)
    return seen


def extract_pdf_text(path: str | Path) -> tuple[str | None, int | None, str | None]:
    """(text, page count, error). Text and pages are None when the file cannot be read as a PDF."""
    path = Path(path)
    if not path.is_file():
        return None, None, "no such file"
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages), len(pdf.pages), None
    except Exception as e:  # noqa: BLE001 - any unreadable file is reported, not raised
        return None, None, f"{type(e).__name__}: {e}"


def compare_texts(text_version: str | None, pdf_text: str | None) -> dict:
    """
    How closely the PDF text matches the plain text it was made from.

    Returns text_words_found_in_pdf_pct (share of the plain text's words, with repeats, found in the
    PDF), missing_words and extra_words (distinct words only on one side; at most 20 each, in text
    order) and bracket_placeholders found in the PDF.

    PDF text extraction sometimes loses the spaces between words ("ihavealsoused...") or splits a
    word at a line break ("ex perienced"). A word counts as present when it is a word of the other
    text or occurs inside that text with all spaces and punctuation removed, so these extraction
    artefacts are not reported, while a dropped or added sentence still is.
    """
    src = words(text_version)
    pdf = words(pdf_text)
    pdf_set, src_set = set(pdf), set(src)
    pdf_joined, src_joined = "".join(pdf), "".join(src)

    def in_pdf(w):
        return w in pdf_set or w in pdf_joined

    def in_text(w):
        return w in src_set or w in src_joined

    missing = list(dict.fromkeys(w for w in src if not in_pdf(w)))
    extra = list(dict.fromkeys(w for w in pdf if not in_text(w)))
    return {
        "text_words_found_in_pdf_pct": round(100 * sum(in_pdf(w) for w in src) / max(len(src), 1), 1),
        "missing_words": missing[:20],
        "extra_words": extra[:20],
        "missing_word_count": len(missing),
        "extra_word_count": len(extra),
        "bracket_placeholders": find_placeholders(pdf_text)[:20],
    }


def check_pdf(path: str | Path, text_version: str | None) -> dict:
    """compare_texts for a PDF file, plus 'pages', 'chars' and 'ok' (None when it could not be read)."""
    text, pages, error = extract_pdf_text(path)
    if text is None:
        return {"ok": None, "error": error}
    result = {"pages": pages, "chars": len(text), **compare_texts(text_version, text)}
    result["ok"] = (result["text_words_found_in_pdf_pct"] >= MIN_WORDS_FOUND_PCT
                    and result["missing_word_count"] <= MAX_MISSING_WORDS
                    and result["extra_word_count"] <= MAX_EXTRA_WORDS)
    return result
