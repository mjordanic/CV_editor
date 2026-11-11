"""Build the comparison matrices of the CV-editor experiment plan (stages 1-2, and stage 3), deterministically.

The plan (cv-eval-plan, sections 1-3) compares the multi-agent app with one model call:

    A  MA-luna   the app as shipped (default script), gpt-6-luna writes           core, stages 1-2
    C  SP-luna   one gpt-6-luna call (evals/single_prompt.py)                     core, stages 1-2
    E  SP-sol    one gpt-6.1-sol call                                             core, stages 1-2
    F  SP-astra  one gpt-6-astra call                                             stage 2 only (added after the pilot)
    B  MA-terra  the app with gpt-6.1-terra as the writer                         stage 3, only if triggered
    D  SP-terra  one gpt-6.1-terra call                                           stage 3, only if triggered

on the 39 matched cases: personas p01-p13, each on its primary ad, with the input sets all4, cv_only and no_cv.

Stages 1-2 (evals/matrix_ab_stage12.json), 186 sessions:
    pilot  the 6 pilot cases (all4: p01/a1, p03/a3, p04/a6, p11/a1, p12/a7, p13/a8) x A, C, E x 2 writer runs = 36.
           Every session is graded once by Sonnet (eval1). The 18 first-run sessions are also graded by Sonnet a
           second time (eval2), by Opus once (opus1) and by Haiku twice (haiku1, haiku2).
    core   the other 33 cases x A, C, E, graded once by Sonnet; the 4 extra repeat cases get a second writer run
           (graded once) and their first-run sessions a second Sonnet grading; Opus grades the first-run sessions
           of the 6 check cases. 111 sessions. Plus F on all 39 cases (the pilot cases too), one writer run each,
           graded once by Sonnet: 39 sessions, 150 in all. Haiku was dropped as a re-grader after the pilot
           (option O1 off), so the core has no Haiku grading.
    The 6 check cases come from the 6 of the 7 non-pilot personas a seeded draw picks, two per input set; the 4
    extra repeat cases are the check cases with cv_only and no_cv (the 6 pilot repeats are all all4). Every session
    gets the ground-truth trap check (truth1). So: 10 repeat cases (30 re-graded first runs), 12 Opus cases (36
    Opus gradings), 18 sessions graded twice by Haiku.
Stage 3 (evals/matrix_ab_stage3.json, not run unless the core run calls for it): B and D on the 39 cases, graded once
by Sonnet; Opus grades the first runs of 10 of the 12 Opus cases (a seeded draw) in each arm; trap check on all.

Order: within a stage, matched cases come in a seeded random order and, within a case, its arms (and runs) in a
seeded random order, so a case's sessions are graded back to back in random arm order (plan section 5, item 5).

Freezing the app: `--app-commit SHA` writes the commit the sessions must run on into both files (run_matrix then
refuses to run sessions when the app's files differ from that commit); a rebuild keeps the files' current value.

Usage (no model calls; writes the two JSON files, or checks them):
    .venv/bin/python -m evals.build_ab_matrix [--app-commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from evals.cases import INPUT_SETS, find_persona

EVALS = Path(__file__).resolve().parent
STAGE12 = EVALS / "matrix_ab_stage12.json"
STAGE3 = EVALS / "matrix_ab_stage3.json"
SEED = 20261111  # the plan's date; changing it changes which cases are check, repeat and Opus cases

PERSONAS = [f"p{i:02d}" for i in range(1, 14)]
SETS = ("all4", "cv_only", "no_cv")
PILOT = (("p01", "a1"), ("p03", "a3"), ("p04", "a6"), ("p11", "a1"), ("p12", "a7"), ("p13", "a8"))  # all all4

ARMS = {
    "A": {"name": "MA-luna", "slug": "ma_luna", "method": "multi_agent", "model": "openai:gpt-6-luna"},
    "B": {"name": "MA-terra", "slug": "ma_terra", "method": "multi_agent", "model": "openai:gpt-6.1-terra"},
    "C": {"name": "SP-luna", "slug": "sp_luna", "method": "single_prompt", "model": "openai:gpt-6-luna"},
    "D": {"name": "SP-terra", "slug": "sp_terra", "method": "single_prompt", "model": "openai:gpt-6.1-terra"},
    "E": {"name": "SP-sol", "slug": "sp_sol", "method": "single_prompt", "model": "openai:gpt-6.1-sol"},
    "F": {"name": "SP-astra", "slug": "sp_astra", "method": "single_prompt", "model": "openai:gpt-6-astra"},
}
GRADERS = {
    "sonnet": "anthropic:claude-sonnet-5-5",  # the primary grader
    "opus": "anthropic:claude-opus-5-5",      # the second grader
    "haiku": "anthropic:claude-haiku-5-5",    # calibration only (pilot)
    "trap": "anthropic:claude-haiku-5-5",     # the ground-truth trap check (evals/score_truth.py)
}
# The plan's prices (section 3) and caps (+25%, rounded up to a dollar), per stage. Caps are on that stage's own
# spend in the ledger; "total" caps the whole ledger of the runs folder. The pilot keeps the figures it ran under
# (it spent 6.78 USD). The core is re-priced from the pilot's measured unit costs (run_matrix.AB_UNIT_USD) and
# includes arm F (2026-11-11; it was 29.88 / 17.85, cap 38 / 23, before). Its Batch figure has the sessions on
# direct OpenAI calls and the gradings through Anthropic's Batch API.
PLAN_USD_12 = {"pilot": {"standard": 15.74, "batch": 8.81}, "core": {"standard": 46.18, "batch": 35.40},
               "total": {"standard": 61.92, "batch": 44.21}}
CAPS_12 = {"pilot": {"standard": 20.0, "batch": 12.0}, "core": {"standard": 58.0, "batch": 45.0},
           "total": {"standard": 78.0, "batch": 56.0}}
PLAN_USD_3 = {"stage3": {"standard": 23.51, "batch": 14.58}, "total": {"standard": 23.51, "batch": 14.58}}
# Stage 3 runs into the same folder as stages 1-2 (its pairs are with arm A), so its total adds the stage 1-2 caps.
CAPS_3 = {"stage3": {"standard": 30.0, "batch": 19.0}, "total": {"standard": 87.0, "batch": 53.0}}


def primary_ad(persona: str) -> str:
    truth = json.loads((find_persona(persona) / "persona.json").read_text(encoding="utf-8"))
    return truth["primary_ad"].split("-", 1)[0]


def matched_cases() -> list[dict]:
    """The 39 matched cases: {"case", "persona", "ad", "input_set"}."""
    return [{"case": f"{p}_{primary_ad(p)}_{s}", "persona": p, "ad": primary_ad(p), "input_set": s}
            for p in PERSONAS for s in SETS]


def design(seed: int = SEED) -> dict:
    """Which cases are pilot, check, repeat and Opus cases (stages 1-2), and the stage-3 Opus cases."""
    rng = random.Random(seed)
    pilot = [f"{p}_{a}_all4" for p, a in PILOT]
    others = [p for p in PERSONAS if p not in {p for p, _ in PILOT}]
    drawn = rng.sample(others, 6)
    check = [f"{p}_{primary_ad(p)}_{s}" for p, s in zip(drawn, [s for s in SETS for _ in range(2)])]
    extra_repeats = [c for c in check if not c.endswith("_all4")]
    opus = pilot + check
    return {"seed": seed, "pilot": pilot, "check": check, "repeat": pilot + extra_repeats, "opus": opus,
            "stage3_opus": sorted(rng.sample(opus, 10), key=opus.index)}


def _grading(name: str, grader: str) -> dict:
    return {"name": name, "grader": grader, "model": GRADERS[grader]}


def _session(arm: str, mc: dict, rep: int, stage: str, gradings: list[dict]) -> dict:
    a = ARMS[arm]
    return {"id": f"{a['slug']}_{mc['case']}_r{rep}", "stage": stage, "arm": arm, "arm_name": a["name"],
            "method": a["method"], "case": mc["case"], "persona": mc["persona"], "ad": mc["ad"],
            "input_set": mc["input_set"], "inputs": list(INPUT_SETS[mc["input_set"]]), "model": a["model"],
            "writer_model": a["model"].split(":", 1)[-1], "rep": rep, "gradings": gradings,
            "truth": _grading("truth1", "trap")}


def _ordered(stage_cases: list[str], sessions_of, rng: random.Random) -> list[dict]:
    """Cases in random order; each case's sessions back to back, in random order."""
    order = list(stage_cases)
    rng.shuffle(order)
    out = []
    for case in order:
        block = sessions_of(case)
        rng.shuffle(block)
        out += block
    return out


def stage12(seed: int = SEED, app_commit: str | None = None) -> dict:
    d = design(seed)
    by_case = {mc["case"]: mc for mc in matched_cases()}
    rng = random.Random(seed + 12)

    def sessions_of(case: str) -> list[dict]:
        """Arms A, C, E of a case, in its stage."""
        stage = "pilot" if case in d["pilot"] else "core"
        out = []
        for arm in ("A", "C", "E"):
            first = [_grading("eval1", "sonnet")]
            if case in d["repeat"]:
                first.append(_grading("eval2", "sonnet"))
            if case in d["opus"]:
                first.append(_grading("opus1", "opus"))
            if case in d["pilot"]:
                first += [_grading("haiku1", "haiku"), _grading("haiku2", "haiku")]
            out.append(_session(arm, by_case[case], 1, stage, first))
            if case in d["repeat"]:
                out.append(_session(arm, by_case[case], 2, stage, [_grading("eval1", "sonnet")]))
        return out

    def core_sessions_of(case: str) -> list[dict]:
        """A pilot case has only its F session in the core; a core case has A, C, E and F."""
        astra = _session("F", by_case[case], 1, "core", [_grading("eval1", "sonnet")])
        return ([] if case in d["pilot"] else sessions_of(case)) + [astra]

    cases = _ordered(d["pilot"], sessions_of, rng) + _ordered(list(by_case), core_sessions_of, rng)
    return {
        "description": "Stages 1-2 of the CV-editor experiment plan (cv-eval-plan sections 1-3): arms A (MA-luna, the "
                       "app, default script), C (SP-luna) and E (SP-sol) on the 39 matched cases, the pilot as its own "
                       "stage, and F (SP-astra) on the 39 cases in the core. Generated by evals/build_ab_matrix.py; do not edit by hand. Run a stage with: python -m "
                       "evals.run_matrix --matrix evals/matrix_ab_stage12.json --stage pilot --out RUNS_DIR",
        "generated_by": "evals/build_ab_matrix.py", "plan_stages": [1, 2],
        "app_commit": app_commit,
        "arms": {k: ARMS[k] for k in ("A", "C", "E", "F")}, "graders": GRADERS,
        "evaluate_args": [], "design": d,
        "stages": {
            "pilot": {"description": "stage 1: 6 pilot cases x A, C, E x 2 writer runs; grader calibration",
                      "plan_usd": PLAN_USD_12["pilot"], "cap_usd": CAPS_12["pilot"]},
            "core": {"description": "stage 2: the other 33 cases x A, C, E; 4 extra repeat cases; Opus on 6 check cases; "
                                    "F on all 39 cases",
                     "plan_usd": PLAN_USD_12["core"], "cap_usd": CAPS_12["core"]},
        },
        "plan_usd": PLAN_USD_12["total"], "caps": {"total": CAPS_12["total"]},
        "cases": cases,
    }


def stage3(seed: int = SEED, app_commit: str | None = None) -> dict:
    d = design(seed)
    by_case = {mc["case"]: mc for mc in matched_cases()}
    rng = random.Random(seed + 3)

    def sessions_of(case: str) -> list[dict]:
        out = []
        for arm in ("B", "D"):
            gradings = [_grading("eval1", "sonnet")] + ([_grading("opus1", "opus")] if case in d["stage3_opus"] else [])
            out.append(_session(arm, by_case[case], 1, "stage3", gradings))
        return out

    return {
        "description": "Stage 3 of the CV-editor experiment plan, conditional (run only if the core run calls for it, "
                       "plan section 1): arms B (MA-terra) and D (SP-terra) on the 39 matched cases. Generated by "
                       "evals/build_ab_matrix.py; do not edit by hand. Run into the same folder as stages 1-2.",
        "generated_by": "evals/build_ab_matrix.py", "plan_stages": [3],
        "app_commit": app_commit,
        "arms": {k: ARMS[k] for k in ("B", "D")}, "graders": GRADERS,
        "evaluate_args": [], "design": {"seed": seed, "stage3_opus": d["stage3_opus"]},
        "stages": {"stage3": {"description": "stage 3: 39 cases x B, D; Opus on 10 of the 12 Opus cases",
                              "plan_usd": PLAN_USD_3["stage3"], "cap_usd": CAPS_3["stage3"]}},
        "plan_usd": PLAN_USD_3["total"], "caps": {"total": CAPS_3["total"]},
        "cases": _ordered(list(by_case), sessions_of, rng),
    }


def render(matrix: dict) -> str:
    return json.dumps(matrix, indent=2, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the files differ from what this builds")
    parser.add_argument("--app-commit", help="the frozen app commit the sessions must run on (kept on a rebuild)")
    opts = parser.parse_args(argv)
    stale = []
    for path, build in ((STAGE12, stage12), (STAGE3, stage3)):
        current = json.loads(path.read_text(encoding="utf-8")).get("app_commit") if path.is_file() else None
        matrix = build(app_commit=opts.app_commit or current)
        text = render(matrix)
        if opts.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(EVALS.parent)}: {len(matrix['cases'])} sessions")
    if stale:
        print(f"out of date: {', '.join(stale)}; run python -m evals.build_ab_matrix", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
