"""Evaluation fixtures tracked in this repository, and the cases of the evaluation matrix.

Every input an evaluation run uses comes from `evals/fixtures/`:

    fixtures/ads/<id>-<slug>/ad.md                anonymized or invented job ad
    fixtures/ads/<id>-<slug>/company.md           readable summary of the company pages
    fixtures/ads/<id>-<slug>/search_fixture.json  web-search results replayed instead of Tavily
    fixtures/personas/<id>-<slug>/{cv,portfolio,cover_letter,notes}.txt   fictional candidate
    fixtures/personas/<id>-<slug>/persona.json    grader's ground truth (never given to the app)
    scripts/<name>.json                           scripted-user replies that override the defaults

Ads and personas are referred to by their id prefix ("a3", "p07") or their full folder name.
`matrix.json` lists the cases: {"id", "persona", "ad", "input_set", "inputs", "model", "rep"}, plus an
optional "work_preference" and "script" (a name in scripts/, e.g. "revision"). A script written for
one persona (its "persona" key, e.g. a request to emphasise a skill only that candidate has) can
only be used with that persona.
"""
from __future__ import annotations

import json
from pathlib import Path

EVALS = Path(__file__).resolve().parent
FIXTURES = EVALS / "fixtures"
DEFAULT_MATRIX = EVALS / "matrix.json"
SCRIPTS = EVALS / "scripts"
INPUT_FILES = {"cv": "cv.txt", "portfolio": "portfolio.txt", "cover_letter": "cover_letter.txt", "notes": "notes.txt"}
INPUT_SETS = {
    "all4": ["cv", "portfolio", "cover_letter", "notes"],
    "cv_only": ["cv"],
    "no_cv": ["portfolio", "cover_letter", "notes"],
}


def _find(kind: str, ref: str) -> Path:
    folder = FIXTURES / kind
    matches = [p for p in sorted(folder.iterdir()) if p.is_dir() and (p.name == ref or p.name.split("-", 1)[0] == ref)]
    if len(matches) != 1:
        known = ", ".join(p.name for p in sorted(folder.iterdir()) if p.is_dir())
        raise ValueError(f"{kind[:-1]} '{ref}' matches {len(matches)} fixture folders (known: {known})")
    return matches[0]


def find_ad(ref: str) -> Path:
    return _find("ads", ref)


def find_persona(ref: str) -> Path:
    return _find("personas", ref)


def find_script(ref: str) -> Path:
    """A script file path, or the name of a tracked script in scripts/ ("revision")."""
    named = SCRIPTS / f"{ref}.json"
    if "/" not in ref and not ref.endswith(".json") and named.is_file():
        return named
    path = Path(ref)
    if path.is_file():
        return path
    known = ", ".join(sorted(p.stem for p in SCRIPTS.glob("*.json")))
    raise ValueError(f"script '{ref}' is neither a file nor a name in {SCRIPTS} (known: {known})")


def parse_inputs(value: str | list[str] | None) -> list[str]:
    """An input-set name (all4, cv_only, no_cv) or a comma-separated list of cv, portfolio, cover_letter, notes."""
    if value is None:
        return list(INPUT_SETS["all4"])
    items = value if isinstance(value, list) else ([*INPUT_SETS[value]] if value in INPUT_SETS else value.split(","))
    items = [i.strip().replace("-", "_") for i in items if i.strip()]
    unknown = [i for i in items if i not in INPUT_FILES]
    if unknown:
        raise ValueError(f"unknown inputs {unknown}; use {sorted(INPUT_SETS)} or a list of {list(INPUT_FILES)}")
    return items


def resolve_case(case: dict) -> dict:
    """Paths of everything a case uses: job ad, search fixture and the candidate files it gives."""
    ad = find_ad(case["ad"])
    persona = find_persona(case["persona"])
    paths = {
        "job_ad": ad / "ad.md",
        "search_fixture": ad / "search_fixture.json",
        "inputs": {k: persona / INPUT_FILES[k] for k in parse_inputs(case.get("inputs"))},
        "persona_truth": persona / "persona.json",
    }
    if case.get("script"):
        paths["script"] = find_script(case["script"])
        script = json.loads(paths["script"].read_text(encoding="utf-8"))
        if script.get("persona") and find_persona(script["persona"]) != persona:
            raise ValueError(f"case {case.get('id')}: script '{case['script']}' is written for persona "
                             f"{script['persona']}, not {case['persona']}")
    return paths


def load_matrix(path: str | Path = DEFAULT_MATRIX) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
