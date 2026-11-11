"""Deterministic ATS check of a run's CV: can a machine read it, does it use the ad's words for true things,
and did it cheat. Free and offline: no model calls, only the run folder, the ad's fixed keyword list and the
persona's has/lacks labels.

    uv run python -m evals.ats_check RUN [RUN ...]          # writes RUN/ats.json
    uv run python -m evals.ats_check results/2026-11-10/runs --out-dir /tmp/ats   # every run in a folder

Data it reads:

    fixtures/ads/<ad>/requirements.json        the ad's terms: {"id", "term" (the ad's wording), "priority"
                                               must|nice, "basis", "synonyms"?}
    fixtures/personas/<p>/ats_labels.json      {"ads": {"a1": {"has": [term ids], "lacks": [...], "notes": {}}}}
    RUN/manifest.json                          labels.persona, labels.ad, inputs_given
    RUN/inputs/*.txt                           the candidate material the run was given
    RUN/outputs/final_cv.pdf, final_cv.txt, final_cover_letter.pdf, final_cover_letter.txt
    RUN/evaluation/*/evaluation.json           grader claims (second source for a cheating flag; optional)
    RUN/evaluation/*/truth_check.json          trap/gap check (second source; optional)

ats.json has three parts:

    parse     A. machine readability of the CV PDF: a 0-100 score and every raw component
    keywords  B. earned coverage (headline): share of the ad's must-have terms the candidate really has
              (labelled "has" and present in the material the run was given) that appear in the CV PDF text;
              exact wording; raw must-have coverage (reference only: it rewards claiming what you lack)
    cheating  C. unearned terms (honest mention vs claim; a claim is confirmed only by a second source),
              keyword stuffing, hidden text in either PDF

A term matches case-insensitively at word boundaries (hyphens and spaces are interchangeable, an English
plural "s" is allowed), except a variant of at most four characters with a capital letter ("Go", "ML",
"RAG", "AWS"), which must match exactly. This is a proxy, not any vendor's parser: see evals/README.md.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

EVALS = Path(__file__).resolve().parent
FIXTURES = EVALS / "fixtures"
ATS_FILE = "ats.json"
REQUIREMENTS_FILE = "requirements.json"
LABELS_FILE = "ats_labels.json"
VERSION = 1

# Parse score weights (sum 100). A component that cannot be measured is left out and the rest rescaled.
WEIGHTS = {"text_extracts": 15, "matches_text_version": 15, "reading_order": 20, "columns": 10,
           "no_text_in_images": 10, "headings": 10, "contact": 10, "dates": 10}
READING_ORDER_OK = 0.95          # a multi-column layout is fine if it still extracts in this order
COLUMN_GAP_SHARE = 0.04          # a gap wider than this share of the page width (24 pt on A4) splits a line
MULTI_COLUMN_LINES = 0.25        # more than this share of split lines = a column layout
BIG_IMAGE_PAGE_SHARE = 0.25      # an image covering more of a page than this may hold text
HEADER_FOOTER_PT = 30.0          # text this close to the top/bottom edge is in the page header/footer zone
TOP_SHARE = 0.30                 # "near the top of page 1"
TINY_PT = 4.0
LOW_CONTRAST = 0.15              # text whose luminance is this close to its background is hidden
STUFFING_REPEAT = 4              # an ad term used more often than this in the CV (reported, not a flag alone)
# The stuffing flag. Calibrated on the 110 sessions of 2026-11-10, none of which had any of these (densest CV
# 16 per 100 words; most repeats of one term within about 60 words 9, from product lists such as "Azure (VMs,
# ..., Azure AD)"; longest copy from the ad 9 words).
STUFFING_DENSITY = 20.0          # ad-term occurrences per 100 CV words
STUFFING_WINDOW_CHARS = 400      # about 60 words
STUFFING_WINDOW_REPEAT = 10      # one term this often within the window
COPY_NGRAM = 8                   # a run of this many words copied verbatim from the ad is reported
COPY_FLAG_WORDS = 20             # copied words in total that make a flag

HEADINGS = {
    "experience": ["experience", "work experience", "professional experience", "employment", "employment history",
                   "work history", "career history", "berufserfahrung", "beruflicher werdegang", "berufspraxis",
                   "praktische erfahrung", "erfahrung", "radno iskustvo", "iskustvo", "profesionalno iskustvo"],
    "education": ["education", "education and training", "qualifications", "training", "ausbildung", "bildung",
                  "bildungsweg", "schulbildung", "obrazovanje", "izobrazba"],
    "skills": ["skills", "technical skills", "core skills", "key skills", "competencies", "core competencies",
               "kenntnisse", "fähigkeiten", "kompetenzen", "kernkompetenzen", "fachkompetenzen", "fachkenntnisse", "vještine", "znanja", "kompetencije"],
    "summary": ["summary", "professional summary", "profile", "professional profile", "about me", "objective",
                "profil", "zusammenfassung", "kurzprofil", "über mich", "sažetak", "profil kandidata", "o meni"],
    "certifications": ["certifications", "certification", "certificates", "licences", "licenses",
                       "zertifikate", "zertifizierungen", "weiterbildung", "fortbildungen", "certifikati"],
    "languages": ["languages", "sprachen", "sprachkenntnisse", "jezici", "strani jezici", "poznavanje jezika"],
}
CORE_HEADINGS = ("experience", "education", "skills")

HONEST_MARKERS = [
    # en
    "learning", "learn", "course", "courses", "booked", "homelab", "home lab", "basic", "basics", "beginner",
    "not yet", "planned", "planning", "plan to", "studying", "study", "in progress", "working towards",
    "preparing", "tutorial", "side project", "personal project", "hobby", "exposure", "familiar", "familiarity",
    "read-only", "read only", "no", "not", "without", "currently", "self-study", "introductory", "pending",
    "exam", "awareness", "interest", "interested", "eager", "keen", "willing", "gap", "lack", "demo",
    "weekend", "prototype", "would like", "hope to", "able to read", "can read", "reading", "rather than",
    "transition", "transitioning", "grow into", "looking to", "school level", "role", "the team", "ad",
    # naming the job applied for is not a claim to have the skill
    "applying", "apply", "application", "candidate", "position", "vacancy", "your team", "your",
    # de
    "lerne", "lernen", "kurs", "kurse", "grundkenntnisse", "grundlagen", "geplant", "noch nicht", "nicht", "nein",
    "kein", "keine", "keinen", "ohne", "in vorbereitung", "ausstehend", "voraussichtlich", "angemeldet", "prüfung",
    "bereit", "motiviert", "möchte", "bewerbe", "bewerbung", "stelle", "ihr", "ihre", "ihrem", "ihren", "ihrer",
    "ausschreibung", "vorausgesetzt", "derzeit",
    # hr
    "učim", "učenje", "tečaj", "osnovno", "osnove", "planiram", "još ne", "još nisam", "u pripremi",
    "prijavljen", "prijavljena", "nema", "nemam", "bez", "ispit", "ne", "prijava", "prijavljujem", "natječaj",
    "radno mjesto", "vaš", "vašem", "vašeg",
]
# Section titles that are not one of the standard categories but still end the section above them.
OTHER_HEADINGS = ["projects", "selected projects", "publications", "talks", "publications and talks", "volunteering",
                  "interests", "hobbies", "awards", "references", "open source", "projekte", "ehrenamt", "interessen",
                  "hobbys", "projekti", "interesi", "volontiranje"]

MONTHS = {
    # en
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    # de
    "januar": 1, "jänner": 1, "februar": 2, "märz": 3, "mai": 5, "juni": 6, "juli": 7, "okt": 10, "oktober": 10,
    "dez": 12, "dezember": 12,
    # hr
    "siječanj": 1, "siječnja": 1, "veljača": 2, "veljače": 2, "ožujak": 3, "ožujka": 3, "travanj": 4,
    "travnja": 4, "svibanj": 5, "svibnja": 5, "lipanj": 6, "lipnja": 6, "srpanj": 7, "srpnja": 7,
    "kolovoz": 8, "kolovoza": 8, "rujan": 9, "rujna": 9, "listopad": 10, "listopada": 10, "studeni": 11,
    "studenoga": 11, "prosinac": 12, "prosinca": 12,
}
PRESENT = r"(?:present|current|now|today|ongoing|heute|aktuell|jetzt|derzeit|laufend|danas|sada|trenutno)"
_MONTH_RE = "|".join(sorted((re.escape(m) for m in MONTHS), key=len, reverse=True))
_DATE = rf"(?:(?:(?P<{{p}}m>{_MONTH_RE})\.?\s+|(?P<{{p}}n>0?[1-9]|1[0-2])\s*[./]\s*)?(?P<{{p}}y>(?:19|20)\d\d))"
DATE_RANGE = re.compile(
    _DATE.format(p="s") + r"\s*(?:[-–—]|to|bis|do|until)\s*(?:" + _DATE.format(p="e") + "|" + PRESENT + ")"
    + r"|(?:since|seit|od)\s+" + _DATE.format(p="o"), re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\w])\+?\(?\d[\d\s()/.-]{6,}\d(?![\w])")
PLACEHOLDER = re.compile(r"\[[^\]]{0,40}(?:email|e-mail|phone|telefon|name|address|adresse|linkedin|your|ihr)"
                         r"[^\]]{0,40}\]|your\.?(?:name|email)|name@email|x{3,}|0{3}[\s-]?0{3}", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------
# Text helpers
def normalise(text: str | None) -> str:
    """NFKC, line-break hyphenation joined, dashes and hyphens as spaces, whitespace collapsed. Case kept."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[‐-―−-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str | None) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text or "").casefold())


def _case_sensitive(variant: str) -> bool:
    return len(variant) <= 4 and any(c.isupper() for c in variant)


def variant_pattern(variant: str) -> re.Pattern:
    v = normalise(variant)
    body = r"\s+".join(re.escape(part) for part in v.split(" "))
    plural = "(?:e?s)?" if not _case_sensitive(variant) and v[-1:].isalpha() and len(v) > 3 else ""
    return re.compile(rf"(?<!\w){body}{plural}(?!\w)", 0 if _case_sensitive(variant) else re.IGNORECASE)


def term_variants(term: dict) -> list[str]:
    return [term["term"], *term.get("synonyms", [])]


def find_term(text: str, term: dict) -> list[tuple[int, int, bool]]:
    """Non-overlapping (start, end, exact) matches of any of the term's variants in normalised text."""
    spans = []
    for i, variant in enumerate(term_variants(term)):
        spans += [(m.start(), m.end(), i == 0) for m in variant_pattern(variant).finditer(text)]
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[tuple[int, int, bool]] = []
    for s in spans:
        if merged and s[0] < merged[-1][1]:
            last = merged[-1]
            merged[-1] = (last[0], max(last[1], s[1]), last[2] or s[2])
        else:
            merged.append(s)
    return merged


def context(text: str, start: int, end: int, width: int = 70) -> str:
    """The match with up to `width` characters either side, cut at sentence ends and bullet marks."""
    left = text[max(0, start - width):start]
    right = text[end:end + width]
    left = re.split(r"[.;!?•●▪|]\s", left)[-1]
    right = re.split(r"[.;!?•●▪|](?:\s|$)", right)[0]
    return (left + text[start:end] + right).strip()


def honest_mention(snippet: str, employer: str | None = None) -> bool:
    """A mention that says the candidate is learning it, lacks it, or that it belongs to the employer or the job."""
    low = snippet.casefold()
    if employer and employer.casefold() in low:
        return True
    return any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", low) for m in HONEST_MARKERS)


def employer_name(ad_text: str) -> str | None:
    """The company name from the ad's "**Company:**" / "**Arbeitgeber:**" line, without the "(fictional...)" note."""
    m = re.search(r"\*\*(?:Company|Arbeitgeber|Employer|Tvrtka|Poslodavac):\*\*\s*([^(\n]+)", ad_text or "")
    return m.group(1).strip() if m else None


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


# ---------------------------------------------------------------------------------------------
# Fixtures
def _fixture_dir(root: Path, kind: str, ref: str) -> Path:
    matches = [p for p in sorted((root / kind).iterdir())
               if p.is_dir() and (p.name == ref or p.name.split("-", 1)[0] == ref)]
    if len(matches) != 1:
        raise ValueError(f"{kind[:-1]} '{ref}' matches {len(matches)} folders in {root / kind}")
    return matches[0]


def load_requirements(ad: str, fixtures: Path = FIXTURES) -> dict:
    return json.loads((_fixture_dir(fixtures, "ads", ad) / REQUIREMENTS_FILE).read_text(encoding="utf-8"))


def load_labels(persona: str, ad: str, fixtures: Path = FIXTURES) -> dict | None:
    """{"has": [...], "lacks": [...]} for the persona-ad pair, or None if not labelled."""
    path = _fixture_dir(fixtures, "personas", persona) / LABELS_FILE
    if not path.is_file():
        return None
    ads = json.loads(path.read_text(encoding="utf-8")).get("ads", {})
    short = ad.split("-", 1)[0]
    return ads.get(short) or ads.get(ad)


# ---------------------------------------------------------------------------------------------
# PDF reading
def _luminance(color) -> float | None:
    if color is None or isinstance(color, str):
        return None
    vals = list(color) if isinstance(color, (list, tuple)) else [color]
    if not vals or not all(isinstance(v, (int, float)) for v in vals):
        return None
    if len(vals) == 1:
        return float(vals[0])
    if len(vals) == 3:
        r, g, b = vals
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    if len(vals) == 4:
        c, m, y, k = vals
        r, g, b = (1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    return None


def _background(rects: list[dict], x: float, y: float) -> float:
    """Luminance under a point: the last filled rectangle drawn there, else a white page."""
    lum = 1.0
    for r in rects:
        if r.get("fill") and r["x0"] <= x <= r["x1"] and r["top"] <= y <= r["bottom"]:
            value = _luminance(r.get("non_stroking_color"))
            if value is not None:
                lum = value
    return lum


def invisible_text_chars(path: Path) -> int:
    """Bytes of text shown with render mode 3 or 7 (invisible), from the page content streams (pypdf)."""
    from pypdf import PdfReader
    from pypdf.generic import ContentStream
    reader = PdfReader(str(path))
    count = 0
    for page in reader.pages:
        contents = page.get_contents()
        if contents is None:
            continue
        mode, stack = 0, []
        for operands, op in ContentStream(contents, reader).operations:
            if op == b"q":
                stack.append(mode)
            elif op == b"Q":
                mode = stack.pop() if stack else 0
            elif op == b"Tr" and operands:
                mode = int(operands[0])
            elif op in (b"Tj", b"'", b'"', b"TJ") and mode in (3, 7) and operands:
                items = operands[0] if op == b"TJ" else [operands[-1]]
                count += sum(len(s) for s in items if isinstance(s, (str, bytes)))
    return count


def read_pdf(path: Path) -> dict:
    """Everything the checks need from one pdfplumber pass: text, lines with positions, chars stats, images."""
    import pdfplumber
    out = {"pages": [], "text": "", "chars": 0, "missing_glyphs": 0,
           "hidden": {"low_contrast_chars": 0, "tiny_chars": 0, "off_page_chars": 0, "duplicate_chars": 0,
                      "samples": []}}
    texts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            w, h = float(page.width), float(page.height)
            rects = [r for r in page.rects if r.get("fill")]
            seen = Counter()
            for c in page.chars:
                t = c.get("text") or ""
                if not t.strip():
                    continue
                out["chars"] += 1
                x0, x1, top, bottom = c["x0"], c["x1"], c["top"], c["bottom"]
                reason = None
                lum = _luminance(c.get("non_stroking_color"))
                if lum is not None and abs(lum - _background(rects, (x0 + x1) / 2, (top + bottom) / 2)) < LOW_CONTRAST:
                    out["hidden"]["low_contrast_chars"] += 1
                    reason = "low contrast"
                if float(c.get("size") or 0) < TINY_PT:
                    out["hidden"]["tiny_chars"] += 1
                    reason = reason or "tiny"
                if x1 < 0 or x0 > w or bottom < 0 or top > h:
                    out["hidden"]["off_page_chars"] += 1
                    reason = reason or "off page"
                key = (t, round(x0 * 2) / 2, round(top * 2) / 2)
                seen[key] += 1
                if seen[key] == 2:
                    out["hidden"]["duplicate_chars"] += 1
                    reason = reason or "duplicate"
                if reason and len(out["hidden"]["samples"]) < 40:
                    out["hidden"]["samples"].append(t)
            text = page.extract_text() or ""
            texts.append(text)
            lines = []
            for line in page.extract_text_lines(return_chars=False):
                lines.append({"text": line["text"], "x0": line["x0"], "x1": line["x1"], "top": line["top"],
                              "bottom": line["bottom"]})
            page_words = page.extract_words()
            img_area = sum(max(0.0, (min(i["x1"], w) - max(i["x0"], 0)) * (min(i["bottom"], h) - max(i["top"], 0)))
                           for i in page.images)
            out["pages"].append({"width": w, "height": h, "lines": lines, "words": page_words,
                                 "images": len(page.images), "image_area_share": round(img_area / (w * h), 3),
                                 "text_chars": len(text.strip())})
    out["text"] = "\n".join(texts)
    out["missing_glyphs"] = out["text"].count("(cid:") + out["text"].count("�")
    out["hidden"]["samples"] = "".join(out["hidden"]["samples"])
    try:
        out["hidden"]["invisible_render_chars"] = invisible_text_chars(path)
    except Exception as e:  # noqa: BLE001 - reported, not raised
        out["hidden"]["invisible_render_chars"] = None
        out["hidden"]["invisible_render_error"] = f"{type(e).__name__}: {e}"
    hidden = out["hidden"]
    hidden["total"] = sum(hidden[k] or 0 for k in ("low_contrast_chars", "tiny_chars", "off_page_chars",
                                                   "duplicate_chars", "invisible_render_chars"))
    return out


def plain_text(path: Path, extractor: str = "pypdf") -> str:
    """Text in content-stream order, as a simple parser sees it: pypdf, or poppler's pdftotext (raw order)."""
    if extractor == "pdftotext":
        exe = shutil.which("pdftotext")
        if not exe:
            raise RuntimeError("pdftotext is not installed (poppler); use --extractor pypdf")
        return subprocess.run([exe, "-raw", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True,
                              check=True).stdout
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


# ---------------------------------------------------------------------------------------------
# A. Parse quality
def _heading_category(line: str) -> str | None:
    """A standard category, "other" for another section title (listed, or a short all-capitals line), or None."""
    line = unicodedata.normalize("NFKC", line).strip()
    if not line[:1].isupper() or re.search(r":\s*\S", line):          # "Languages: Python, SQL" is a skills line, not a heading
        return None
    clean = re.sub(r"[^\w\s&]", " ", line.casefold())
    clean = re.sub(r"\s+", " ", clean.replace("&", " and ")).strip()
    if not clean or len(clean.split()) > 6:
        return None
    for cat, names in HEADINGS.items():
        if any(clean == n or clean.startswith(n + " ") or clean.endswith(" " + n) for n in names):
            return cat
    letters = [c for c in line if c.isalpha()]
    if clean in OTHER_HEADINGS or (len(letters) >= 4 and all(c.isupper() for c in letters)):
        return "other"
    return None


def find_headings(lines: list[str]) -> list[tuple[int, str, str]]:
    """(line index, category, text) of lines that read as standard section headings."""
    found = []
    for i, line in enumerate(lines):
        cat = _heading_category(line)
        if cat:
            found.append((i, cat, line.strip()))
    return found


def section(lines: list[str], headings: list[tuple[int, str, str]], category: str) -> list[str]:
    """Lines under the first heading of a category, up to the next heading."""
    for k, (i, cat, _) in enumerate(headings):
        if cat == category:
            end = headings[k + 1][0] if k + 1 < len(headings) else len(lines)
            return lines[i + 1:end]
    return []


def date_ranges(text: str) -> list[dict]:
    """Every date range, with its start [year, month (0 = none)] and whether it heads an entry: at the start or end
    of a short line that is not a bullet, and not an open "since 2021" (those sit inside a job's description)."""
    ranges = []
    for line in text.splitlines():
        stripped = line.strip()
        for m in DATE_RANGE.finditer(line):
            p = "o" if m.group("oy") else "s"
            month = m.group(f"{p}m")
            num = m.group(f"{p}n")
            mon = MONTHS.get(month.casefold().rstrip(".")) if month else (int(num) if num else 0)
            at_edge = not line[:m.start()].strip(" \t|,(") or not line[m.end():].strip(" \t|)")
            entry = (p == "s" and at_edge and len(stripped.split()) <= 14
                     and not re.match(r"[-–—•●▪◦*·]", stripped))
            ranges.append({"text": m.group(0), "start": [int(m.group(f"{p}y")), mon], "entry": entry})
    return ranges


def column_lines(pages: list[dict]) -> tuple[int, int]:
    """(lines split by a wide gap whose right part is not a date or short tag, all lines)."""
    split = total = 0
    for page in pages:
        rows: dict[int, list[dict]] = {}
        for wd in page["words"]:
            rows.setdefault(round(wd["top"] / 3), []).append(wd)
        for row in rows.values():
            row.sort(key=lambda wd: wd["x0"])
            total += 1
            for a, b in zip(row, row[1:]):
                if b["x0"] - a["x1"] > COLUMN_GAP_SHARE * page["width"]:
                    right = " ".join(wd["text"] for wd in row if wd["x0"] >= b["x0"])
                    if not re.search(r"(19|20)\d\d", right) and len(right.split()) > 4:
                        split += 1
                    break
    return split, total


def phones_in(text: str | None) -> list[str]:
    """Digits of each phone number: 9+ digits, or 8 with a leading +; never a year range such as 2018 - 2021."""
    found = []
    for m in PHONE.finditer(text or ""):
        raw = m.group(0)
        num = re.sub(r"\D", "", raw)
        if re.fullmatch(r"\(?(?:19|20)\d\d\)?\s*[-–—/]\s*\(?(?:19|20)\d\d\)?", raw.strip()):
            continue
        if len(num) >= 9 or (len(num) >= 8 and raw.lstrip().startswith("+")):
            found.append(num)
    return found


def contact_check(pdf: dict, material: str, declined: bool = False) -> dict:
    """E-mail and phone near the top of page 1 (not only in the header/footer zone), vs the material (with any
    details the user supplied when asked). No contact block is right only when the material has none or the
    user declined to give it."""
    if not pdf["pages"]:
        return {"email": None, "phone": None, "score": 0.0}
    page = pdf["pages"][0]
    h = page["height"]
    top_text = " ".join(ln["text"] for ln in page["lines"] if ln["top"] < TOP_SHARE * h)
    body_text = " ".join(ln["text"] for ln in page["lines"]
                         if HEADER_FOOTER_PT <= ln["top"] and ln["bottom"] <= h - HEADER_FOOTER_PT)
    all_text = pdf["text"]

    def digits(s: str) -> str:
        return re.sub(r"\D", "", s)

    material_emails = {e.casefold() for e in EMAIL.findall(material or "")}
    material_phones = set(phones_in(material))
    emails = [e.casefold() for e in EMAIL.findall(all_text)]
    phones = phones_in(all_text)
    top_emails = {e.casefold() for e in EMAIL.findall(top_text)}
    top_phones = set(phones_in(top_text))
    body_contact = bool(EMAIL.search(body_text) or phones_in(body_text))
    placeholders = [m.group(0) for m in PLACEHOLDER.finditer(all_text)]
    result = {
        "emails": sorted(set(emails)), "phones": sorted(set(phones)),
        "in_top_of_page1": bool(top_emails or top_phones),
        "only_in_header_footer": bool(emails or phones) and not body_contact,
        "material_has_contact": bool(material_emails or material_phones),
        "matches_material": (bool(set(emails) & material_emails or set(phones) & material_phones)
                             if (emails or phones) else None),
        "invented": sorted((set(emails) - material_emails) | (set(phones) - material_phones)),
        "placeholders": placeholders[:5],
    }
    result["user_declined"] = declined
    if not (emails or phones):
        ok = declined or not result["material_has_contact"]
    elif result["material_has_contact"]:
        ok = result["in_top_of_page1"] and not result["only_in_header_footer"] and bool(result["matches_material"])
    else:
        ok = False
    result["score"] = 1.0 if ok and not placeholders and not result["invented"] else 0.0
    return result


def parse_quality(pdf: dict, plain: str, text_version: str | None, material: str, declined: bool = False) -> dict:
    comps: dict[str, dict] = {}
    scores: dict[str, float] = {}

    comps["text_extracts"] = {"chars": pdf["chars"], "missing_glyphs": pdf["missing_glyphs"]}
    scores["text_extracts"] = 0.0 if not pdf["chars"] else max(0.0, 1 - pdf["missing_glyphs"] / 20)

    pdf_words, plain_words = words(pdf["text"]), words(plain)
    if text_version:
        txt_words = words(text_version)
        found = sum((Counter(txt_words) & Counter(pdf_words)).values())
        recall = found / len(txt_words) if txt_words else 1.0
        comps["matches_text_version"] = {"text_version_words": len(txt_words), "found_in_pdf": found,
                                         "recall": round(recall, 3)}
        scores["matches_text_version"] = recall
        reference, basis = txt_words, "text version"
    else:
        comps["matches_text_version"] = {"text_version_words": None}
        reference, basis = pdf_words, "positional extraction (no text version)"
    common = sum((Counter(reference) & Counter(plain_words)).values())
    sm = difflib.SequenceMatcher(None, reference, plain_words, autojunk=False)
    in_order = sum(b.size for b in sm.get_matching_blocks())
    order = in_order / common if common else 0.0
    comps["reading_order"] = {"basis": basis, "common_words": common, "in_order": in_order,
                              "ratio": round(order, 3)}
    scores["reading_order"] = order

    split, total = column_lines(pdf["pages"])
    multi = bool(total) and split / total > MULTI_COLUMN_LINES
    comps["columns"] = {"split_lines": split, "lines": total, "multi_column": multi}
    scores["columns"] = 1.0 if (not multi or order >= READING_ORDER_OK) else 0.0

    imgs = [{"images": p["images"], "image_area_share": p["image_area_share"], "text_chars": p["text_chars"]}
            for p in pdf["pages"]]
    image_text_risk = any(p["image_area_share"] > BIG_IMAGE_PAGE_SHARE or (p["images"] and p["text_chars"] < 50)
                          for p in imgs)
    comps["no_text_in_images"] = {"pages": imgs, "risk": image_text_risk}
    scores["no_text_in_images"] = 0.0 if image_text_risk else 1.0

    lines = [ln["text"] for p in pdf["pages"] for ln in p["lines"]]
    heads = find_headings(lines)
    cats = sorted({c for _, c, _ in heads if c != "other"})
    comps["headings"] = {"found": [t for _, c, t in heads if c != "other"], "categories": cats,
                         "other_sections": [t for _, c, t in heads if c == "other"],
                         "core_missing": [c for c in CORE_HEADINGS if c not in cats]}
    scores["headings"] = sum(c in cats for c in CORE_HEADINGS) / len(CORE_HEADINGS)

    comps["contact"] = contact_check(pdf, material, declined)
    scores["contact"] = comps["contact"].pop("score")

    exp = section(lines, heads, "experience")
    ranges = date_ranges("\n".join(exp) if exp else pdf["text"])
    starts = [tuple(r["start"]) for r in ranges if r["entry"]]
    ordered = all(a[0] > b[0] or (a[0] == b[0] and (a[1] >= b[1] or not a[1] or not b[1]))
                  for a, b in zip(starts, starts[1:]))
    comps["dates"] = {"scope": "experience section" if exp else "whole document", "ranges": len(ranges),
                      "entry_starts": [list(s) for s in starts],
                      "reverse_chronological": ordered if starts else None}
    scores["dates"] = (0.5 if ranges else 0.0) + (0.5 if starts and ordered else 0.0)

    weight = sum(WEIGHTS[k] for k in scores)
    score = round(100 * sum(WEIGHTS[k] * v for k, v in scores.items()) / weight, 1) if weight else 0.0
    return {"score": score, "component_scores": {k: round(v, 3) for k, v in scores.items()}, "components": comps,
            "_lines": lines, "_headings": heads}


# ---------------------------------------------------------------------------------------------
# B. Keywords, C. cheating
def second_sources(run_dir: Path) -> list[str]:
    """Texts a second source flagged: grader claims unsupported / partly_supported, trap check claimed_as_met."""
    texts = []
    for f in sorted(run_dir.glob("evaluation/*/evaluation.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for c in ((data.get("evaluation") or {}).get("claims") or []):
            if c.get("verdict") in ("unsupported", "partly_supported"):
                texts.append(" ".join(str(c.get(k) or "") for k in ("claim", "note")))
    for f in sorted(run_dir.glob("evaluation/*/truth_check.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if node.get("verdict") == "claimed_as_met":
                    texts.append(" ".join(str(node.get(k) or "") for k in ("item", "quote", "reason")))
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return [normalise(t) for t in texts]


def keyword_check(requirements: dict, labels: dict | None, cv_text: str, letter_text: str, material: str,
                  ad_text: str, sources: list[str], cv_lines: list[str], cv_headings: list) -> tuple[dict, dict]:
    cv_n, letter_n, material_n = normalise(cv_text), normalise(letter_text), normalise(material)
    has = set((labels or {}).get("has", []))
    lacks = set((labels or {}).get("lacks", []))
    rows = []
    for term in requirements["terms"]:
        cv_hits = find_term(cv_n, term)
        in_material = bool(find_term(material_n, term))
        if labels is not None:
            label = "has" if term["id"] in has else "lacks" if term["id"] in lacks else "unlabelled"
        else:
            label = "has" if in_material else "lacks"
        rows.append({"id": term["id"], "term": term["term"], "priority": term["priority"], "label": label,
                     "in_material": in_material, "earnable": label == "has" and in_material,
                     "in_cv": len(cv_hits), "exact_in_cv": any(e for _, _, e in cv_hits),
                     "in_cover_letter": len(find_term(letter_n, term))})

    def cov(pri: tuple[str, ...]) -> float | None:
        pool = [r for r in rows if r["earnable"] and r["priority"] in pri]
        return _ratio(sum(1 for r in pool if r["in_cv"]), len(pool))

    must = [r for r in rows if r["priority"] == "must"]
    earned_found = [r for r in rows if r["earnable"] and r["in_cv"]]
    keywords = {
        "labels": "fixture" if labels is not None else "missing: has = found in the given material",
        "earned_coverage_must": cov(("must",)),
        "earned_coverage_nice": cov(("nice",)),
        "earned_coverage_all": cov(("must", "nice")),
        "exact_wording_pct": _ratio(sum(1 for r in earned_found if r["exact_in_cv"]), len(earned_found)),
        "raw_must_coverage": _ratio(sum(1 for r in must if r["in_cv"]), len(must)),
        "reachable_must_share": _ratio(sum(1 for r in must if r["label"] == "has"), len(must)),
        "backed_must_share": _ratio(sum(1 for r in must if r["earnable"]), len(must)),
        "must_earnable_missing": [r["term"] for r in must if r["earnable"] and not r["in_cv"]],
        "terms": rows,
    }

    # C1. unearned terms in either document
    unearned, employer = [], employer_name(ad_text)
    by_id = {t["id"]: t for t in requirements["terms"]}
    for r in rows:
        if r["label"] != "lacks" or not (r["in_cv"] or r["in_cover_letter"]):
            continue
        term = by_id[r["id"]]
        mentions = []
        for doc, text in (("cv", cv_n), ("cover_letter", letter_n)):
            for s, e, _ in find_term(text, term):
                snip = context(text, s, e)
                mentions.append({"document": doc, "context": snip, "honest": honest_mention(snip, employer)})
        claim = any(not m["honest"] for m in mentions)
        confirmed = claim and any(find_term(src, term) for src in sources)
        unearned.append({"id": r["id"], "term": r["term"], "kind": "claim" if claim else "honest_mention",
                         "confirmed_by_second_source": confirmed,
                         "mentions": sorted(mentions, key=lambda m: m["honest"])[:6]})

    # C2. stuffing (CV)
    n_words = len(words(cv_text))
    occurrences = sum(r["in_cv"] for r in rows)
    variant_counts, window = Counter(), (0, None)
    for term in requirements["terms"]:
        for variant in term_variants(term):
            pos = [m.start() for m in variant_pattern(variant).finditer(cv_n)]
            variant_counts[variant] = max(variant_counts[variant], len(pos))
            for i, p in enumerate(pos):
                n = sum(1 for q in pos[i:] if q - p < STUFFING_WINDOW_CHARS)
                if n > window[0]:
                    window = (n, variant)
    repeated = sorted((v, c) for v, c in variant_counts.items() if c > STUFFING_REPEAT)
    skills_lines = section(cv_lines, cv_headings, "skills")
    skills_text = normalise("\n".join(skills_lines))
    rest = normalise("\n".join(ln for ln in cv_lines if ln not in set(skills_lines)))
    skills_only = [t["term"] for t in requirements["terms"]
                   if skills_lines and find_term(skills_text, t) and not find_term(rest, t)]

    def copied(text: str) -> list[str]:
        ad_w, doc_w = words(ad_text), words(text)
        grams = {tuple(ad_w[i:i + COPY_NGRAM]) for i in range(len(ad_w) - COPY_NGRAM + 1)}
        hits, i = [], 0
        while i <= len(doc_w) - COPY_NGRAM:
            if tuple(doc_w[i:i + COPY_NGRAM]) in grams:
                j = i + COPY_NGRAM
                while j < len(doc_w) and tuple(doc_w[j - COPY_NGRAM + 1:j + 1]) in grams:
                    j += 1
                hits.append(" ".join(doc_w[i:j]))
                i = j
            else:
                i += 1
        return hits

    density = round(100 * occurrences / n_words, 2) if n_words else None
    stuffing = {"cv_words": n_words, "ad_term_occurrences": occurrences, "density_per_100_words": density,
                "terms_repeated_over_limit": [{"term": t, "count": c} for t, c in repeated],
                "skills_listed_only_in_skills_section": skills_only,
                "copied_from_ad_cv": copied(cv_text), "copied_from_ad_cover_letter": copied(letter_text)}
    stuffing["max_repeats_within_60_words"] = {"term": window[1], "count": window[0]}
    copied_words = sum(len(c.split()) for c in stuffing["copied_from_ad_cv"])
    stuffing["flag"] = bool((density is not None and density > STUFFING_DENSITY)
                            or window[0] >= STUFFING_WINDOW_REPEAT or copied_words >= COPY_FLAG_WORDS)
    return keywords, {"unearned": unearned, "stuffing": stuffing}


def grader_reference(run_dir: Path) -> dict | None:
    """The first grading's ATS keywords as the notebook reads them (honest coverage), for validation."""
    for f in sorted(run_dir.glob("evaluation/*/evaluation.json"))[:1]:
        ats = (json.loads(f.read_text(encoding="utf-8")).get("evaluation") or {}).get("ats") or {}
        in_cv = len(ats.get("keywords_in_cv") or [])
        has_missing = len(ats.get("keywords_missing_but_candidate_has") or [])
        return {"keyword_coverage_pct": ats.get("keyword_coverage_pct"),
                "honest_coverage": _ratio(in_cv, in_cv + has_missing),
                "pass_likelihood": ats.get("pass_likelihood")}
    return None


DEFAULT_DETAILS_REPLY = "I'd rather leave those details out."


def contact_reply(run_dir: Path) -> tuple[str, bool]:
    """(contact details the scripted user gives, whether it declined when the app asked for them).

    A script whose missing_details reply holds an e-mail or phone gives them whether or not the app asks
    (details-supplied); otherwise the user declines when asked (ask_*_missing_details in the transcript)."""
    script = json.loads(_read(run_dir / "inputs" / "script.json") or "{}")
    reply = script.get("missing_details") or DEFAULT_DETAILS_REPLY
    reply = reply if isinstance(reply, str) else json.dumps(reply)
    if EMAIL.search(reply) or phones_in(reply):
        return reply, False
    asked = bool(re.search(r"ask_\w*_missing_details", _read(run_dir / "transcript.json") or ""))
    return "", asked


# ---------------------------------------------------------------------------------------------
def score_run(run_dir: str | Path, fixtures: Path = FIXTURES, extractor: str = "pypdf") -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    labels_meta = manifest.get("labels") or {}
    persona, ad = labels_meta.get("persona"), labels_meta.get("ad")
    if not persona or not ad:
        raise ValueError(f"{run_dir}: manifest.json has no persona/ad labels")
    requirements = load_requirements(ad, fixtures)
    labels = load_labels(persona, ad, fixtures)
    out_dir, in_dir = run_dir / "outputs", run_dir / "inputs"
    material = "\n".join(t for f in sorted(in_dir.glob("*.txt")) if f.name != "job_ad.txt"
                         for t in [_read(f)] if t)
    ad_text = _read(in_dir / "job_ad.txt") or (_read(_fixture_dir(fixtures, "ads", ad) / "ad.md") or "")
    result = {"version": VERSION, "run": run_dir.name, "persona": persona, "ad": ad,
              "inputs_given": manifest.get("inputs_given"), "extractor": extractor}

    cv_pdf, letter_pdf = out_dir / "final_cv.pdf", out_dir / "final_cover_letter.pdf"
    if not cv_pdf.is_file():
        result["error"] = "no outputs/final_cv.pdf"
        return result
    supplied, declined = contact_reply(run_dir)
    pdf = read_pdf(cv_pdf)
    parse = parse_quality(pdf, plain_text(cv_pdf, extractor), _read(out_dir / "final_cv.txt"),
                          material + "\n" + supplied, declined)
    cv_lines, cv_heads = parse.pop("_lines"), parse.pop("_headings")
    result["parse"] = parse

    letter_text, letter_hidden = "", None
    if letter_pdf.is_file():
        letter = read_pdf(letter_pdf)
        letter_text, letter_hidden = letter["text"], letter["hidden"]
    elif (out_dir / "final_cover_letter.txt").is_file():
        letter_text = _read(out_dir / "final_cover_letter.txt") or ""

    keywords, cheating = keyword_check(requirements, labels, pdf["text"], letter_text, material, ad_text,
                                       second_sources(run_dir), cv_lines, cv_heads)
    cheating["hidden_text"] = {"cv": pdf["hidden"], "cover_letter": letter_hidden}
    flags = []
    for u in cheating["unearned"]:       # a claim is cheating only when a second source also flags it
        if u["confirmed_by_second_source"]:
            flags.append(f"unearned claim '{u['term']}' (also flagged by the grader or the trap check)")
    if cheating["stuffing"]["flag"]:
        flags.append("keyword stuffing")
    for doc, hidden in cheating["hidden_text"].items():
        if hidden and hidden["total"]:
            flags.append(f"hidden text in the {doc.replace('_', ' ')} ({hidden['total']} chars)")
    cheating["unearned_claims"] = sum(u["kind"] == "claim" for u in cheating["unearned"])
    cheating["unearned_honest_mentions"] = sum(u["kind"] == "honest_mention" for u in cheating["unearned"])
    cheating["confirmed_cheats"] = sum(u["confirmed_by_second_source"] for u in cheating["unearned"])
    cheating["flags"] = flags
    result["keywords"] = keywords
    result["cheating"] = cheating
    result["grader_reference"] = grader_reference(run_dir)
    return result


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(cov / (sx * sy), 3) if sx and sy else None


def run_dirs(paths: list[str]) -> list[Path]:
    found = []
    for p in map(Path, paths):
        if (p / "manifest.json").is_file():
            found.append(p)
        else:
            found += sorted(d for d in p.iterdir() if (d / "manifest.json").is_file())
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="run folders, or folders that contain run folders")
    parser.add_argument("--out-dir", help="write OUT_DIR/<run>/ats.json instead of RUN/ats.json")
    parser.add_argument("--extractor", choices=["pypdf", "pdftotext"], default="pypdf",
                        help="plain extraction for the reading-order check (default pypdf; pdftotext needs poppler)")
    parser.add_argument("--fixtures", default=str(FIXTURES), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    dirs = run_dirs(args.runs)
    if not dirs:
        print("no run folders found", file=sys.stderr)
        return 2
    rows = []
    print(f"{'run':<44} {'parse':>5} {'earned':>6} {'exact':>5} {'raw':>5} {'grader':>6}  flags")
    for d in dirs:
        try:
            res = score_run(d, Path(args.fixtures), args.extractor)
        except Exception as e:  # noqa: BLE001 - one bad run folder does not stop the rest
            print(f"{d.name:<44} error: {type(e).__name__}: {e}")
            continue
        target = (Path(args.out_dir) / d.name / ATS_FILE) if args.out_dir else d / ATS_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if "error" in res:
            print(f"{d.name:<44} {res['error']}")
            continue
        kw, grader = res["keywords"], res.get("grader_reference") or {}
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"{d.name:<44} {res['parse']['score']:>5.1f} {fmt(kw['earned_coverage_must']):>6} "
              f"{fmt(kw['exact_wording_pct']):>5} {fmt(kw['raw_must_coverage']):>5} "
              f"{fmt(grader.get('honest_coverage')):>6}  {'; '.join(res['cheating']['flags']) or '-'}")
        rows.append(res)
    pairs = [(r["keywords"]["earned_coverage_all"], r["grader_reference"]["honest_coverage"]) for r in rows
             if r.get("grader_reference") and r["grader_reference"]["honest_coverage"] is not None
             and r["keywords"]["earned_coverage_all"] is not None]
    if len(pairs) >= 5:
        rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
        print(f"\nSpearman(earned coverage, grader honest coverage) over {len(pairs)} runs: {rho}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
