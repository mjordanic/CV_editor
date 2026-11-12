"""A deterministic check that a cover-letter draft does not repeat the CV.

With some writer models the first letter draft sometimes holds the whole CV (or most of it) before
the letter. The critic and the grounding check do not count that as a defect (every line of the CV
is supported by the candidate's material), so such a draft could be kept as the final letter.

The measure is the share of the CV's word 5-grams (five consecutive words, lower-cased, punctuation
ignored) that also occur in the letter, taken against each CV the session has (the tailored CV and
the candidate's own) and keeping the highest. A letter retells the CV in its own sentences and
shares only short runs of words with it; a letter that pastes the CV shares most of them. It needs
no model call and works for any language.

Threshold, measured on the 222 published letter drafts of the 2026-11-10 evaluation round
(evals/results/2026-11-10/runs/*/outputs/cover_letter_history.json): the 9 drafts that repeat the
CV share 0.43-1.00 of the CV's 5-grams with the letter, the 213 clean drafts 0.00-0.19. The
threshold 0.30 sits in that gap.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

NGRAM = 5
# Share of a CV's 5-grams found in the letter at or above which the letter repeats the CV.
MAX_CV_SHARE = 0.30
# A CV this short (in 5-grams, about this many words) cannot be told apart from a letter that
# quotes it; it is not judged.
MIN_CV_GRAMS = 20
# Cutting the CV out of a letter (strip_cv). A block counts as part of the CV when more than this
# share of its 5-grams come from a CV. Clean letters' paragraphs reuse 7% at the median (up to 46%,
# when a sentence retells a CV line closely); the CV blocks pasted into the 9 drafts, 14-100% (lower
# where the CV pasted is a reworded version of the tailored one). The other signals below decide
# the rest.
BLOCK_CV_SHARE = 0.2
# A heading at the top of a block: an all-caps line of at most this many words (e.g. "EXPERIENCE").
HEADING_WORDS = 6
# A salutation: a line of at most this many words that ends with a comma ("Dear Hiring Team,",
# "Sehr geehrte Damen und Herren,"). The letter starts there at the latest.
SALUTATION_WORDS = 8
# What is left after the CV is cut out must still be a letter of at least this many words.
MIN_LETTER_WORDS = 40

# The redraft instruction for a letter that repeats the CV. The writer gets it as it is: it is not
# a critic suggestion, so the preference filter never sees it.
CV_REPEAT_INSTRUCTION = (
    "This draft repeats the CV (in whole or in part) before or around the cover letter. Delete every part of the CV "
    "from it: its sections, headings, bullet lists, dates and skill lists. Keep only the cover letter itself (with "
    "its own header, if it has one), unchanged. The CV is sent as a separate document."
)

_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n")


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text or "").lower())


def _grams(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def session_cvs(state) -> list[Optional[str]]:
    """The CVs a letter of this session must not repeat: the tailored CV and the candidate's own."""
    return [state.get("generated_cv"), (state.get("candidate_text") or {}).get("cv")]


def cv_share(letter: str, cvs: Iterable[Optional[str]]) -> float:
    """The highest share, over the given CVs, of a CV's 5-grams that also occur in the letter."""
    letter_grams = _grams(_words(letter))
    best = 0.0
    for cv in cvs:
        cv_grams = _grams(_words(cv or ""))
        if len(cv_grams) < MIN_CV_GRAMS:
            continue
        best = max(best, len(cv_grams & letter_grams) / len(cv_grams))
    return best


def repeats_cv(letter: str, cvs: Iterable[Optional[str]]) -> bool:
    """Whether the letter repeats one of the CVs (in whole or in large part)."""
    return cv_share(letter, cvs) >= MAX_CV_SHARE


def _is_heading(line: str) -> bool:
    line = line.strip()
    return line.isupper() and 0 < len(_words(line)) <= HEADING_WORDS


def _is_salutation(line: str) -> bool:
    line = line.strip()
    return line.endswith(",") and 0 < len(_words(line)) <= SALUTATION_WORDS


def strip_cv(letter: str, cvs: Iterable[Optional[str]]) -> Optional[str]:
    """
    The letter without the CV pasted before it, or None if it cannot be cut out cleanly.

    The draft is split into blocks at blank lines. Up to the salutation, a block counts towards
    the CV by how many of its 5-grams come from a CV beyond BLOCK_CV_SHARE, and against it by how
    many do not; a block that opens with an all-caps heading ("EXPERIENCE", "CAREER FOCUS") counts
    as the CV whatever its words. The cut goes after the leading run of blocks with the largest
    total, so a CV block that was reworded does not stop it. All-caps heading lines left at the
    top of the letter ("COVER LETTER") go too. The result is used only if it is still a letter
    (MIN_LETTER_WORDS) and no longer repeats the CV.
    """
    cvs = [cv for cv in cvs if cv and cv.strip()]
    cv_grams = set().union(*(_grams(_words(cv)) for cv in cvs)) if cvs else set()
    if not cv_grams:
        return None
    blocks = [b.strip("\n") for b in _BLOCK_SPLIT.split(letter.strip()) if b.strip()]
    best_total, cut, total = 0.0, 0, 0.0
    for i, block in enumerate(blocks):
        lines = [line for line in block.splitlines() if line.strip()]
        if _is_salutation(lines[-1]):
            break
        grams = _grams(_words(block))
        if _is_heading(lines[0]):
            total += (1 - BLOCK_CV_SHARE) * max(len(grams), 1)
        else:
            total += len(grams & cv_grams) - BLOCK_CV_SHARE * len(grams)
        if total > best_total:
            best_total, cut = total, i + 1
    if cut == 0:
        return None
    lines = "\n\n".join(blocks[cut:]).splitlines()
    while lines and (not lines[0].strip() or _is_heading(lines[0])):
        lines = lines[1:]
    stripped = "\n".join(lines).strip()
    if len(_words(stripped)) < MIN_LETTER_WORDS or repeats_cv(stripped, cvs):
        return None
    return stripped
