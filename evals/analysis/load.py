"""Load evaluation rounds and pair sessions across arms. Standard library only, no model calls.

A *round* is a folder as `evals/run_matrix.py` writes it and `evals/results/<date>/` publishes it: either the
runs folder itself or a folder with a `runs/` subfolder. Every run folder has `manifest.json`, `summary.json`
and `evaluation/<name>/evaluation.json` (one per grading: `eval1`, `eval2`, ... or a second grader's names).
The same layout holds for multi-agent sessions (`evals/run_case.py`) and the single-prompt baseline
(`evals/single_prompt.py`, whose manifest says `"method": "single_prompt"`).

Vocabulary used by the notebook:

    arm     system x writer model, e.g. "multi_agent/gpt-6-luna" or "single_prompt/gpt-6.1-sol"
    cell    what must be equal for two sessions to be compared: persona, ad, input set, script and
            stated work preference. Repeats of one arm in one cell are rep 1, 2, ...
    grader  the evaluator model of a grading (evaluation.json "evaluator_model"); one run can be graded
            several times by one grader (re-scores) and by several graders (calibration)

`read_rows()` gives one flat row per grading (a run without a grading gives one row with empty scores);
the pairing helpers average re-scores within a run, then repeats within a cell, so a cell counts once.
`truth_rows()` gives one row per trap check (`evaluation/<name>/truth_check.json`), with the same run fields.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

_REP_SUFFIX = re.compile(r"_r\d+$")
CELL_FIELDS = ("persona", "ad", "input_set", "script", "work_preference")

# Scores (0-10) from evaluation.json, by the column names of the published results.csv.
SCORES = {
    "overall_score": ("overall_score",),
    "invented_content_score": ("invented_content_score",),
    "completeness_score": ("completeness_score",),
    "fit": ("fit", "score"),
    "cv_standards": ("cv_standards", "score"),
    "cover_letter": ("cover_letter", "score"),
    "work_arrangement": ("work_arrangement", "score"),
    "visual_layout": ("visual_layout", "score"),
    "language_quality": ("language_quality", "score"),
    "additional_information": ("additional_information", "score"),
    "user_requests": ("user_requests", "score"),
    "ats": ("ats", "score"),  # the grader's one ATS score; rounds graded before it have none
}
COUNTS = ("claims", "cv_unsupported", "cv_partly", "cl_unsupported", "cl_partly", "omissions_high", "must_haves",
          "must_haves_shown", "gaps_overstated", "work_arrangement_invented", "requests", "requests_followed",
          "requests_partly", "requests_ignored", "requests_refused_wrongly", "untruthful_refused",
          "untruthful_followed", "layout_problems", "wrong_language", "language_errors")
# The composite of the plan's section 6: the mean of these sub-scores (the calibration checks use it, being steadier
# than the one overall grade). Every grading row gets it as "composite", None if any of them is missing.
COMPOSITE = ["invented_content_score", "completeness_score", "fit", "cv_standards", "cover_letter", "visual_layout",
             "language_quality"]


def composite(row: dict) -> float | None:
    """Mean of the COMPOSITE sub-scores of one grading row, None if any is missing."""
    values = [row.get(c) for c in COMPOSITE]
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in values):
        return None
    return statistics.fmean(values)


def runs_dir_of(path: str | Path) -> Path:
    """The folder holding the run folders: PATH/runs when it exists, else PATH."""
    path = Path(path)
    return path / "runs" if (path / "runs").is_dir() else path


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _dig(record: dict, keys: tuple) -> float | None:
    value = record
    for k in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(k)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def short_model(model: str | None) -> str:
    """'openai:gpt-6-luna' -> 'gpt-6-luna'."""
    return (model or "unknown").split(":", 1)[-1]


def method_of(manifest: dict) -> str:
    """single_prompt (evals/single_prompt.py) or multi_agent (evals/run_case.py, whose manifest has no method)."""
    return manifest.get("method") or (manifest.get("labels") or {}).get("method") or "multi_agent"


def arm_name(method: str, writer_model: str) -> str:
    return f"{method}/{writer_model}"


def _run_fields(run: Path) -> dict:
    manifest = _json(run / "manifest.json")
    summary = _json(run / "summary.json")
    labels = manifest.get("labels") or {}
    method = method_of(manifest)
    writer = short_model(labels.get("model") or manifest.get("model") or manifest.get("model_override"))
    status = str(manifest.get("status") or summary.get("status") or "")
    rep = labels.get("rep")
    m = re.search(r"_r(\d+)$", run.name)
    try:
        rep = int(rep) if rep not in (None, "") else (int(m.group(1)) if m else 1)
    except ValueError:
        rep = int(m.group(1)) if m else 1
    calls = summary.get("llm_calls")
    if calls is None:
        calls = manifest.get("calls", summary.get("calls"))
    return {
        "run_id": run.name,
        "case": labels.get("case") or _REP_SUFFIX.sub("", run.name),
        "persona": labels.get("persona") or "",
        "ad": labels.get("ad") or "",
        "input_set": labels.get("input_set") or "",
        "script": labels.get("script") or "default",
        "work_preference": manifest.get("work_preference") or "",
        "rep": rep,
        "method": method,
        "writer_model": writer,
        "arm": arm_name(method, writer),
        "status": status,
        # not finished: interrupted (out of credit, infrastructure, Ctrl+C / SIGTERM) or still "running" (killed)
        "interrupted": status == "running" or status.startswith("interrupted"),
        "session_usd": manifest.get("cost_usd", summary.get("cost_usd")),
        "session_minutes": manifest.get("minutes", summary.get("minutes")),
        "llm_calls": calls,
        "llm_errors": summary.get("llm_errors"),
        "fallback_pdfs": len(manifest.get("fallback_pdfs") or []),
        # single prompt: parse_error / error and the call's finish_reason (the pilot gate counts parse errors);
        # app: what a completed session did without (company research, a fact check)
        "failure": manifest.get("failure") or "",
        "finish_reason": manifest.get("finish_reason") or "",
        "degraded": ",".join(manifest.get("degraded") or []),
        "commit": (_json(run / "config.json").get("code") or {}).get("commit") or "",
    }


def _eval_fields(path: Path | None) -> dict:
    record = _json(path) if path else {}
    evaluation = record.get("evaluation") or {}
    counts = record.get("counts") or {}
    row = {
        "eval_name": path.parent.name if path else "",
        "grader": short_model(record.get("evaluator_model")) if record else "",
        "eval_usd": record.get("cost_usd"),
        # Anthropic gradings: did the full copy parse, and how many guaranteed fields does it state differently
        # (the scores always come from the guaranteed fields; empty for OpenAI gradings)
        "full_eval_parsed": (record.get("full_evaluation") or {}).get("parsed"),
        "n_full_disagreements": len((record.get("full_evaluation") or {}).get("disagreements") or [])
        if record.get("full_evaluation") else None,
    }
    for col, keys in SCORES.items():
        row[col] = _dig(evaluation, keys)
    row["composite"] = composite(row)
    for c in COUNTS:
        value = counts.get(c)
        row[f"n_{c}"] = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    unsupported = [row["n_cv_unsupported"], row["n_cl_unsupported"]]
    row["n_unsupported"] = sum(unsupported) if None not in unsupported else None
    partly = [row["n_cv_partly"], row["n_cl_partly"]]
    row["n_partly"] = sum(partly) if None not in partly else None
    row["any_unsupported"] = None if row["n_unsupported"] is None else int(row["n_unsupported"] > 0)
    return row


def read_rows(*paths: str | Path, include_interrupted: bool = False, round_name: str | None = None) -> list[dict]:
    """One row per grading of every run folder under the given rounds (a run without grading: one row
    with empty scores). Interrupted (out-of-credit) sessions are left out unless asked for."""
    rows = []
    for path in paths:
        runs_dir = runs_dir_of(path)
        name = round_name or (Path(path).name if Path(path).name != "runs" else Path(path).parent.name)
        for run in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "manifest.json").is_file()):
            base = _run_fields(run)
            if base["interrupted"] and not include_interrupted:
                continue
            evals = sorted(run.glob("evaluation/*/evaluation.json")) or [None]
            for ev in evals:
                rows.append({"round": name, **base, **_eval_fields(ev)})
    return rows


def truth_rows(*paths: str | Path, include_interrupted: bool = False, round_name: str | None = None) -> list[dict]:
    """One row per trap check (evals/score_truth.py) of every run folder: the run's fields, the check's name
    ("eval_name") and checker ("grader"), its counts, and "truth_rate": gaps claimed plus traps hit per item."""
    rows = []
    for path in paths:
        name = round_name or (Path(path).name if Path(path).name != "runs" else Path(path).parent.name)
        for run in sorted(p for p in runs_dir_of(path).iterdir() if p.is_dir() and (p / "manifest.json").is_file()):
            base = _run_fields(run)
            if base["interrupted"] and not include_interrupted:
                continue
            for f in sorted(run.glob("evaluation/*/truth_check.json")):
                record = _json(f)
                counts = record.get("counts") or {}
                items = (counts.get("gaps") or 0) + (counts.get("traps") or 0)
                hits = (counts.get("gaps_claimed") or 0) + (counts.get("traps_hit") or 0)
                rows.append({"round": name, **base, "eval_name": f.parent.name,
                             "grader": short_model(record.get("checker_model")),
                             **{k: counts.get(k) for k in ("gaps", "traps", "gaps_claimed", "traps_hit", "unclear",
                                                           "unverified_quotes")},
                             "truth_rate": hits / items if items else None})
    return rows


def read_results_csv(path: str | Path) -> list[dict]:
    """The published results.csv as dicts (strings), for cross-checks against read_rows()."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- pairing ---------------------------------------------------------------------------------------------
def cell_of(row: dict) -> tuple:
    return tuple(row.get(f) or "" for f in CELL_FIELDS)


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def run_values(rows: list[dict], metric: str, grader: str | None = None) -> dict[str, dict]:
    """{run_id: {"arm", "cell", "rep", "value" (mean over that grader's gradings), "k" (gradings)}}.
    With grader=None every grading of the run counts; pass a grader to keep graders apart."""
    acc: dict[str, dict] = {}
    for r in rows:
        if grader is not None and r.get("grader") != grader:
            continue
        v = _num(r.get(metric))
        entry = acc.setdefault(r["run_id"], {"arm": r["arm"], "cell": cell_of(r), "rep": r["rep"], "values": []})
        if v is not None:
            entry["values"].append(v)
    return {run: {"arm": e["arm"], "cell": e["cell"], "rep": e["rep"], "k": len(e["values"]),
                  "value": statistics.fmean(e["values"]) if e["values"] else None}
            for run, e in acc.items()}


def cell_values(rows: list[dict], metric: str, grader: str | None = None) -> dict[tuple, dict[str, dict]]:
    """{cell: {arm: {"value": mean over the arm's repeats, "reps": n, "run_values": [...]}}}."""
    out: dict[tuple, dict[str, dict]] = defaultdict(dict)
    by: dict[tuple, list[float]] = defaultdict(list)
    for e in run_values(rows, metric, grader).values():
        if e["value"] is not None:
            by[(e["cell"], e["arm"])].append(e["value"])
    for (cell, arm), values in by.items():
        out[cell][arm] = {"value": statistics.fmean(values), "reps": len(values), "run_values": values}
    return dict(out)


def pair_arms(rows: list[dict], arm_a: str, arm_b: str, metric: str, grader: str | None = None) -> dict:
    """Paired comparison B - A over the cells both arms have. Repeats in a cell are averaged first, so
    every cell counts once whatever its number of repeats (the pair records both repeat counts).
    Cells only one arm has are listed under only_a / only_b and left out of the pairs."""
    cells = cell_values(rows, metric, grader)
    pairs, only_a, only_b = [], [], []
    for cell in sorted(cells):
        arms = cells[cell]
        a, b = arms.get(arm_a), arms.get(arm_b)
        if a and b:
            pairs.append({**dict(zip(CELL_FIELDS, cell)), "cell": cell, "a": a["value"], "b": b["value"],
                          "diff": b["value"] - a["value"], "reps_a": a["reps"], "reps_b": b["reps"]})
        elif a:
            only_a.append(cell)
        elif b:
            only_b.append(cell)
    return {"arm_a": arm_a, "arm_b": arm_b, "metric": metric, "grader": grader, "pairs": pairs,
            "only_a": only_a, "only_b": only_b}


def repeat_groups(rows: list[dict], metric: str, grader: str | None = None) -> list[dict]:
    """Every (arm, cell) with two or more repeats: the run-level values (each the mean of its gradings) and "k",
    each run's number of gradings. The spread inside a group is writer variance plus grader variance / k
    (stats.writer_var takes the grader part out)."""
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for e in run_values(rows, metric, grader).values():
        if e["value"] is not None:
            groups[(e["arm"], e["cell"])].append((e["value"], e["k"]))
    return [{"arm": arm, "cell": cell, **dict(zip(CELL_FIELDS, cell)), "values": [x for x, _ in v],
             "k": [k for _, k in v], "n": len(v)}
            for (arm, cell), v in sorted(groups.items()) if len(v) >= 2]


def regrade_groups(rows: list[dict], metric: str, grader: str | None = None) -> list[dict]:
    """Every run graded two or more times by the same grader: those values (pure grader variance)."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if grader is not None and r.get("grader") != grader:
            continue
        v = _num(r.get(metric))
        if v is not None:
            groups[(r["run_id"], r.get("grader"))].append(v)
    return [{"run_id": run, "grader": g, "values": v, "n": len(v)} for (run, g), v in sorted(groups.items())
            if len(v) >= 2]


def grader_pairs(rows: list[dict], metric: str, grader_a: str, grader_b: str) -> list[dict]:
    """Runs graded by both graders: each grader's mean over its gradings of the run."""
    a = run_values(rows, metric, grader_a)
    b = run_values(rows, metric, grader_b)
    out = []
    for run in sorted(set(a) & set(b)):
        if a[run]["value"] is not None and b[run]["value"] is not None:
            out.append({"run_id": run, "arm": a[run]["arm"], "a": a[run]["value"], "b": b[run]["value"],
                        "k_a": a[run]["k"], "k_b": b[run]["k"]})
    return out


def token_usage(*paths: str | Path) -> list[dict]:
    """Tokens and cost per model from every session's and grading's summary.json:
    [{"kind": "session" | "grading", "run_id", "model", "uncached_in", "cached_in", "output", "cost_usd"}]."""
    out = []
    for path in paths:
        for run in sorted(p for p in runs_dir_of(path).iterdir() if p.is_dir()):
            sources = [("session", run / "summary.json")]
            sources += [("grading", f) for f in sorted(run.glob("evaluation/*/summary.json"))]
            for kind, f in sources:
                for model, u in (_json(f).get("by_model") or {}).items():
                    cached = u.get("cached_tokens") or 0
                    out.append({"kind": kind, "run_id": run.name, "model": short_model(model),
                                "uncached_in": (u.get("input_tokens") or 0) - cached, "cached_in": cached,
                                "output": u.get("output_tokens") or 0, "cost_usd": u.get("cost_usd") or 0.0})
    return out
