"""Run the evaluation matrix: one testing-mode session per case, then the blind evaluator on it.

Every input comes from the tracked fixtures (`evals/fixtures/`, resolved by `evals/cases.py`); the cases
are in `evals/matrix.json`. This makes paid model calls (never Tavily: every ad has a search fixture).

Usage (from the repo root):
    .venv/bin/python -m evals.run_matrix --out RUNS_DIR --env-file PATH/.env [--matrix evals/matrix.json]
        [--workers 4] [--only CASE_ID ...] [--total-cap 24] [--cost-cap 1.5] [--dry-run] [--estimate]
        [--repeats N] [--eval-name eval1 | --rescore K] [--truth-check]
        [--stage NAME ...] [--sessions-only] [--gradings-out FILE]
        [--sp-batch [--batch-wait] [--batch-resubmit-failed] [--batch-adopt KEY=ID] [--batch-forget-pending]]

Each case gets RUNS_DIR/<case id>/ (see evals/README.md) and RUNS_DIR/<case id>.out (console log).
RUNS_DIR/ledger.tsv records the cost of every session, evaluation and truth check; no new case starts
once it passes --total-cap. A case whose evaluations (and truth check, if asked) exist is skipped, and a
finished session is never run again, so an interrupted matrix can be resumed and an existing runs folder
re-scored. --dry-run prints the commands without running anything.

--repeats N runs every selected case N times: <id without _rN>_r1 .. _rN (rep 1 keeps the matrix id).
--rescore K scores each run K times (eval1 .. evalK); --eval-name names a single evaluation instead.
--truth-check also runs evals/score_truth.py (truth1) after the evaluations.
--estimate prints the expected cost of the selected cases (ESTIMATE_USD below) and runs nothing.

A case with "method": "single_prompt" runs evals/single_prompt.py (exactly one model call writes both documents as
text and LaTeX; pdflatex makes the PDFs) instead of the multi-agent session; the ledger, caps, resume, evaluation and
out-of-credit handling are the same. --total-cap defaults to the matrix file's "total_cap", else 24.

Infrastructure errors: a session that ended on a provider or network failure (connection error, timeout, HTTP 5xx,
rate limit; status "interrupted: infrastructure", evals/credit.py) is not the system's failure. It is never graded,
and it is run again at once, one time, for both methods alike (its manifest label "attempt" says 2). If it fails the
same way again it stays unfinished: run the same command later to redo it.

Staged comparison matrices (evals/matrix_ab_stage12.json, matrix_ab_stage3.json; built by evals/build_ab_matrix.py)
give every session a stage, an arm and its own gradings ({"name", "grader", "model"}: eval1/eval2 by Sonnet, opus1,
haiku1/haiku2) and a trap check ("truth"). On such a matrix:
    --stage pilot|core|stage3   runs (or --dry-run / --estimate / --gradings-out) only that stage; required for a run.
    --estimate                  prints the plan's cost table: every line (app sessions, Sonnet, re-gradings, Opus, Haiku,
                                trap check) per arm and per stage, at standard and at Batch prices, against the
                                plan's figures and the stage caps (AB_UNIT_USD below).
    --sessions-only             runs the sessions and no grading (for the Batch grader); --gradings-out FILE writes the
                                grading jobs of the selection, in the matrix's order (each case's arms back to back).
    Caps: each stage's own spend (its sessions' rows in the ledger) stops at the stage's cap, and the whole ledger at
    the matrix's total cap: the Batch caps with --sessions-only, the standard caps when the gradings run here.
    Freeze: every session's manifest gets the labels app_commit (the matrix's frozen "app_commit", else HEAD),
    app_head and app_dirty. When the matrix names a frozen commit, no session starts if the app's files (APP_PATHS)
    differ from it. Sessions of the app (multi_agent) run as before, through evals/run_case.py.

OpenAI Batch for the single-prompt arms (evals/batch_sp.py):
    --sp-batch                  the single-prompt sessions' one call goes through OpenAI's Batch API (half price): each
                                run submits the SP cases not yet submitted (within the caps, at Batch-price estimates)
                                and collects ended batches, running each collected case's session on its reply (same
                                PDFs, fallback layout, manifest; priced at the Batch rate). An SP case still in a batch
                                is reported and graded on a later run. Multi-agent sessions run as before. State:
                                RUNS_DIR/batches/sp_openai/. Synchronous stays the default.
    --batch-wait                poll until every batch has ended and its sessions are written, then go on to grading.
    --batch-resubmit-failed     an unanswered request (errored, expired, cancelled) goes out once more in a follow-up
                                batch by itself (the synchronous retry rule); this sends it again after that.
    --batch-adopt KEY=ID, --batch-forget-pending   settle a submission an interrupted run left unrecorded.
    With --estimate, --sp-batch prices the single-prompt sessions at the Batch rate.

Out of credit: when a session or an evaluation exits with code 75 (the provider reported no credit
left, evals/credit.py), no new case or step starts; cases already running finish their current step
(and stop at their own first credit error). The matrix then prints the finished / unfinished /
interrupted counts and the spend, and exits with code 75. Its partial cost stays in the ledger. Run
the same command again after a top-up: finished cases are skipped, an interrupted session is redone
from scratch, and a finished session whose evaluation was interrupted gets only the evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.cases import DEFAULT_MATRIX, load_matrix, matrix_total_cap, resolve_case
from evals.credit import EXIT_OUT_OF_CREDIT, INFRASTRUCTURE_STATUS, INTERRUPTED_STATUS
from provenance import BATCH_PRICE_FACTOR, cost_usd

REPO = Path(__file__).resolve().parent.parent
_LOCK = threading.Lock()
OUT_OF_CREDIT = threading.Event()  # set on the first exit code 75; nothing new starts after it

# Expected cost per step in USD for a matrix without stages (evals/matrix.json), from the measured partial runs
# (2026-11): a gpt-6-luna session about 0.013, a gpt-6.1-terra session about 0.06. A script with change requests
# redrafts more: its session is estimated at twice the default session. "eval" is one Claude Sonnet 5.5 grading and
# "truth" one Claude Haiku 5.5 trap check (older figures than AB_UNIT_USD below, which the staged matrices use).
# A single-prompt session (evals/single_prompt.py) is exactly one call and nothing else (PDFs by pdflatex): the prompt
# is the app's writer and LaTeX rules, both LaTeX templates and the case's inputs, job ad and search results (about
# 4-7k tokens on these fixtures); the answer is both documents as text and as LaTeX plus notes (about 4-5k tokens) and
# reasoning (up to about 5k): up to about 10k output tokens. sp_session_usd prices it from provenance's prices.
SP_TOKENS = (6_000, 10_000)  # input, output tokens of one single-prompt call


def sp_session_usd(model: str) -> float:
    """One single-prompt call at SP_TOKENS and provenance's list price, rounded up to a cent: luna 0.01,
    terra 0.06, sol 0.12 (the plan's section 3 figures)."""
    usd = cost_usd(model, *SP_TOKENS)
    if usd is None:
        raise ValueError(f"no price for {model} in provenance.PRICES_PER_1M")
    return math.ceil(round(usd * 100, 6)) / 100


ESTIMATE_USD = {"session": {"gpt-6-luna": 0.013, "gpt-6.1-terra": 0.06}, "eval": 0.15, "truth": 0.002,
                "scripted_session_factor": 2.0,
                "single_prompt": {m: sp_session_usd(m) for m in ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol")}}
DEFAULT_TOTAL_CAP = 24.0  # USD, when the matrix file sets no "total_cap"

# Staged comparison matrices: unit prices at standard rates, the stage-1 pilot's measured means
# (evals/results/2026-11-11: 12 sessions per arm, 54 Sonnet, 18 Opus and 36 Haiku gradings, 36 trap checks).
# The sessions were direct OpenAI calls. The gradings and trap checks went through Anthropic's Batch API, so their
# measured Batch means are divided by BATCH_PRICE_FACTOR here; the estimate's Batch column multiplies them back.
# The Batch API charges BATCH_PRICE_FACTOR x list price on every grading line; sessions run on OpenAI interactively,
# so Batch does not change them. gpt-6.1-terra was not in the pilot: its app session is the
# 2026-11-10 measured mean (20 sessions), and a single call is sp_session_usd (6k input and 10k output
# tokens, a ceiling: the pilot's single calls averaged 5.4k input and 3.8-5.0k output). Haiku grades only in the
# pilot (dropped as a re-grader).
PILOT_BATCH_USD = {"claude-sonnet-5-5": 0.0465, "claude-opus-5-5": 0.176, "claude-haiku-5-5": 0.0049,
                   "trap": 0.0005}
AB_UNIT_USD = {
    "multi_agent": {"gpt-6-luna": 0.0286, "gpt-6.1-terra": 0.085},
    "single_prompt": {"gpt-6-luna": 0.0028, "gpt-6.1-sol": 0.0439,
                      "gpt-6.1-terra": sp_session_usd("gpt-6.1-terra")},
    "grading": {m: PILOT_BATCH_USD[m] / BATCH_PRICE_FACTOR
                for m in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5")},
    "trap": {"claude-haiku-5-5": PILOT_BATCH_USD["trap"] / BATCH_PRICE_FACTOR},
}
# Paths whose content is the app (and the single-prompt arm) a frozen comparison runs on.
APP_PATHS = ("agents", "langgraph_agent.py", "config", "templates", "evals/run_case.py", "evals/single_prompt.py")


def spent(ledger: Path, case_ids: set | None = None) -> float:
    """USD in the ledger, or only in the rows of these case ids."""
    if not ledger.exists():
        return 0.0
    rows = [line.split("\t") for line in ledger.read_text().splitlines() if line.strip()]
    return sum(float(r[2]) for r in rows if case_ids is None or r[0] in case_ids)


def _record(ledger: Path, case_id: str, kind: str, cost: float) -> None:
    with _LOCK, open(ledger, "a") as f:
        f.write(f"{case_id}\t{kind}\t{cost:.6f}\t{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")


_REP_SUFFIX = re.compile(r"_r\d+$")


def base_id(case_id: str) -> str:
    """The case id without its trailing _rN (the repeat number)."""
    return _REP_SUFFIX.sub("", case_id)


def expand_repeats(cases: list[dict], repeats: int) -> list[dict]:
    """Each case as reps 1..repeats (ids <base>_r1.._rN; rep 1 keeps the matrix id). Duplicates are dropped,
    so a matrix that already has an _r2 of a case does not run it twice."""
    if repeats <= 1:
        return cases
    out, seen = [], set()
    for case in cases:
        base = base_id(case["id"])
        for k in range(1, repeats + 1):
            cid = f"{base}_r{k}" if k > 1 or base != case["id"] else case["id"]
            if cid not in seen:
                seen.add(cid)
                out.append({**case, "id": cid, "rep": k})
    return out


def eval_names(opts) -> list[str]:
    if getattr(opts, "rescore", None):
        return [f"eval{i}" for i in range(1, opts.rescore + 1)]
    return [getattr(opts, "eval_name", None) or "eval1"]


def session_command(case: dict, run_dir: Path, env_file: str | None, cost_cap: float,
                    labels: dict | None = None) -> list[str]:
    paths = resolve_case(case)  # fails early if a fixture is missing
    module = "evals.single_prompt" if case.get("method") == "single_prompt" else "evals.run_case"
    cmd = [sys.executable, "-m", module, "--run-dir", str(run_dir), "--ad", case["ad"],
           "--persona", case["persona"], "--inputs", ",".join(paths["inputs"]), "--model", case["model"],
           "--cost-cap", str(cost_cap)]
    if env_file:
        cmd += ["--env-file", env_file]
    if case.get("work_preference"):
        cmd += ["--work-preference", case["work_preference"]]
    if paths.get("script"):
        cmd += ["--script", str(paths["script"])]
    for key in ("persona", "ad", "model", "rep", "input_set"):
        cmd += ["--label", f"{key}={case.get(key)}"]
    # "case" is the matched case of a comparison matrix (the same for every arm), else the id without _rN.
    cmd += ["--label", f"case={case.get('case') or base_id(case['id'])}",
            "--label", f"script={case.get('script') or 'default'}"]
    for key in ("method", "arm", "stage", "writer_model"):
        if case.get(key):
            cmd += ["--label", f"{key}={case[key]}"]
    for key, value in (labels or {}).items():
        cmd += ["--label", f"{key}={value}"]
    return cmd


def evaluate_command(run_dir: Path, env_file: str | None, name: str = "eval1", model: str | None = None,
                     extra: list[str] | tuple = ()) -> list[str]:
    cmd = [sys.executable, "-m", "evals.evaluate", str(run_dir), "--eval-name", name]
    cmd += (["--grader-model", model] if model else []) + list(extra)
    return cmd + (["--env-file", env_file] if env_file else [])


def truth_command(run_dir: Path, env_file: str | None, name: str = "truth1", model: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "evals.score_truth", str(run_dir), "--name", name]
    cmd += ["--grader-model", model] if model else []
    return cmd + (["--env-file", env_file] if env_file else [])


def case_gradings(case: dict, opts) -> list[tuple[str, str | None]]:
    """(name, grader model) of each grading of a case: the matrix's own, else --eval-name / --rescore (default model)."""
    if case.get("gradings"):
        return [(g["name"], g.get("model")) for g in case["gradings"]]
    return [(n, None) for n in eval_names(opts)]


def case_truth(case: dict, opts) -> tuple[str, str | None] | None:
    """(name, model) of the case's trap check, or None."""
    if case.get("truth"):
        return case["truth"]["name"], case["truth"].get("model")
    return ("truth1", None) if getattr(opts, "truth_check", False) else None


def _manifest_status(manifest: Path) -> str | None:
    try:
        return json.loads(manifest.read_text()).get("status")
    except (OSError, json.JSONDecodeError):
        return None


def session_done(run_dir: Path) -> bool:
    """The session ended with a result to grade: its manifest has a final status (not running, not missing, not
    interrupted by an out-of-credit or infrastructure error)."""
    status = _manifest_status(run_dir / "manifest.json")
    return not (status is None or status == "running" or str(status).startswith("interrupted"))


def _partial_cost(folder: Path) -> float:
    """The cost an interrupted evaluation or truth check had made (its summary.json), else 0."""
    try:
        return float(json.loads((folder / "summary.json").read_text()).get("cost_usd") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return 0.0


def _mark_out_of_credit(cid: str, step: str) -> tuple[str, str]:
    OUT_OF_CREDIT.set()
    return cid, f"interrupted (out of credit) during {step}"


def run_one(case: dict, opts) -> tuple[str, str]:
    cid = case["id"]
    out = Path(opts.out)
    run_dir, log, ledger = out / cid, out / f"{cid}.out", out / "ledger.tsv"
    sessions_only = getattr(opts, "sessions_only", False)
    gradings = [] if sessions_only else case_gradings(case, opts)
    evals = {n: (run_dir / "evaluation" / n / "evaluation.json", model) for n, model in gradings}
    truth_spec = None if sessions_only else case_truth(case, opts)
    truth = run_dir / "evaluation" / (truth_spec or ("truth1",))[0] / "truth_check.json"
    manifest = run_dir / "manifest.json"
    done = session_done(run_dir)
    if sessions_only and done:
        return cid, "skip (session done)"
    if not sessions_only and all(f.exists() for f, _ in evals.values()) and (truth_spec is None or truth.exists()):
        return cid, "skip (done)"
    if OUT_OF_CREDIT.is_set():
        return cid, "not started (out of credit)"
    if spent(ledger) > opts.total_cap:
        return cid, "skip (total cap reached)"
    stage, stage_caps = case.get("stage"), getattr(opts, "stage_caps", None) or {}
    if stage in stage_caps and spent(ledger, opts.stage_ids[stage]) > stage_caps[stage]:
        return cid, f"skip (stage {stage} cap reached)"
    if not done and getattr(opts, "sp_batch", False) and case.get("method") == "single_prompt":
        from evals.batch_sp import pending_note
        return cid, pending_note((getattr(opts, "sp_batch_status", None) or {}).get(cid))
    if not done:
        for attempt in (1, 2):  # an infrastructure error is run again once, at once
            shutil.rmtree(run_dir, ignore_errors=True)  # an unfinished or interrupted session is redone from scratch
            labels = {**(getattr(opts, "app_labels", None) or {}), **({"attempt": attempt} if attempt > 1 else {})}
            with open(log, "w" if attempt == 1 else "a") as f:
                proc = subprocess.run(session_command(case, run_dir, opts.env_file, opts.cost_cap, labels),
                                      cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
            if not manifest.exists():
                if proc.returncode == EXIT_OUT_OF_CREDIT:
                    return _mark_out_of_credit(cid, "the session")
                return cid, "session failed before writing a manifest"
            _record(ledger, cid, "app", json.loads(manifest.read_text()).get("cost_usd") or 0)
            if proc.returncode == EXIT_OUT_OF_CREDIT or _manifest_status(manifest) == INTERRUPTED_STATUS:
                return _mark_out_of_credit(cid, "the session")  # not evaluated: redone after a top-up
            if _manifest_status(manifest) != INFRASTRUCTURE_STATUS:
                break
        else:
            return cid, "interrupted (infrastructure error, twice): not graded; run again later to redo it"
    if sessions_only:
        return cid, "session done (not graded)"
    failed = []
    for name, (result, model) in evals.items():
        if result.exists():
            continue
        if OUT_OF_CREDIT.is_set():
            return cid, "evaluation not started (out of credit)"
        if (result.parent).exists():
            shutil.rmtree(result.parent)  # a failed attempt leaves a folder the evaluator refuses to reuse
        ev = subprocess.run(evaluate_command(run_dir, opts.env_file, name, model, getattr(opts, "evaluate_args", ())),
                            cwd=str(REPO), capture_output=True, text=True)
        with open(log, "a") as f:
            f.write(f"\n--- evaluate {name} ---\n" + ev.stdout[-3000:] + ev.stderr[-3000:])
        if result.exists():
            _record(ledger, cid, f"eval:{name}" if model else "eval", json.loads(result.read_text())["cost_usd"])
        elif ev.returncode == EXIT_OUT_OF_CREDIT:
            _record(ledger, cid, "eval-interrupted", _partial_cost(result.parent))
            return _mark_out_of_credit(cid, f"evaluation {name}")
        else:
            failed.append(name)
    if truth_spec and not truth.exists():
        if OUT_OF_CREDIT.is_set():
            return cid, "truth check not started (out of credit)"
        if truth.parent.exists():
            shutil.rmtree(truth.parent)
        tc = subprocess.run(truth_command(run_dir, opts.env_file, *truth_spec), cwd=str(REPO), capture_output=True,
                            text=True)
        with open(log, "a") as f:
            f.write("\n--- truth check ---\n" + tc.stdout[-3000:] + tc.stderr[-3000:])
        if truth.exists():
            _record(ledger, cid, "truth", json.loads(truth.read_text())["cost_usd"])
        elif tc.returncode == EXIT_OUT_OF_CREDIT or OUT_OF_CREDIT.is_set():
            return _mark_out_of_credit(cid, "the truth check")
        else:
            failed.append(truth_spec[0])
    return cid, f"failed: {', '.join(failed)}" if failed else "ok"


def estimate(cases: list[dict], opts) -> dict:
    """Expected cost of the selected cases at ESTIMATE_USD: per step kind, per model and per script."""
    p = ESTIMATE_USD
    total = {"sessions": 0.0, "evaluations": 0.0, "truth_checks": 0.0}
    by_model, by_script, by_method = {}, {}, {}
    n_evals = len(eval_names(opts))
    for case in cases:
        model = case["model"].split(":", 1)[-1]
        method = case.get("method") or "multi_agent"
        if method == "single_prompt":
            session = p["single_prompt"].get(model, max(p["single_prompt"].values()))
            if getattr(opts, "sp_batch", False):
                session *= BATCH_PRICE_FACTOR
        else:
            session = p["session"].get(model, max(p["session"].values()))
        if case.get("script"):
            session *= p["scripted_session_factor"]
        cost = session + n_evals * p["eval"] + (p["truth"] if opts.truth_check else 0)
        total["sessions"] += session
        total["evaluations"] += n_evals * p["eval"]
        total["truth_checks"] += p["truth"] if opts.truth_check else 0
        by_model[model] = by_model.get(model, 0) + cost
        by_method[method] = by_method.get(method, 0) + cost
        key = case.get("script") or "default"
        by_script.setdefault(key, {"cases": 0, "usd": 0.0})
        by_script[key]["cases"] += 1
        by_script[key]["usd"] += cost
    r = lambda x: round(x, 2)  # noqa: E731
    return {"cases": len(cases), "total_usd": r(sum(total.values())), **{k: r(v) for k, v in total.items()},
            "by_model": {k: r(v) for k, v in sorted(by_model.items())},
            "by_script": {k: {"cases": v["cases"], "usd": r(v["usd"])} for k, v in sorted(by_script.items())},
            "by_method": {k: r(v) for k, v in sorted(by_method.items())},
            "total_cap": getattr(opts, "total_cap", None), "prices": p,
            "sp_batch": bool(getattr(opts, "sp_batch", False))}


LINES = ("app", "grading", "regrades", "opus", "haiku", "trap")
LINE_TITLES = {"app": "app cost", "grading": "grading x1", "regrades": "re-grades", "opus": "Opus",
               "haiku": "Haiku calibration", "trap": "trap check"}


def _model_name(model: str) -> str:
    return model.split(":", 1)[-1]


def case_lines(case: dict) -> dict:
    """The standard-rate cost of one session of a staged matrix, per estimate line (AB_UNIT_USD)."""
    method = case.get("method") or "multi_agent"
    lines = dict.fromkeys(LINES, 0.0)
    lines["app"] = AB_UNIT_USD[method][_model_name(case["model"])]
    for g in case.get("gradings") or []:
        unit = AB_UNIT_USD["grading"][_model_name(g["model"])]
        line = {"opus": "opus", "haiku": "haiku"}.get(g["grader"]) or ("grading" if g["name"] == "eval1" else "regrades")
        lines[line] += unit
    if case.get("truth"):
        lines["trap"] += AB_UNIT_USD["trap"][_model_name(case["truth"]["model"])]
    return lines


def _batch(lines: dict, app_batched: float = 0.0) -> float:
    """Total of a set of lines with Batch grading: every line but the app sessions at BATCH_PRICE_FACTOR x list price,
    and the part `app_batched` of the app line (single-prompt sessions with --sp-batch) too."""
    return (lines["app"] - app_batched + app_batched * BATCH_PRICE_FACTOR
            + sum(v for k, v in lines.items() if k != "app") * BATCH_PRICE_FACTOR)


def estimate_stages(cases: list[dict], meta: dict, sp_batch: bool = False) -> dict:
    """The plan's cost table for the selected cases of a staged matrix: per arm and per stage, every line at
    standard rates, the total and the total with Batch grading (and, with sp_batch, the single-prompt sessions
    through OpenAI's Batch API), beside the plan's figures and the caps."""
    def add(acc: dict, case: dict) -> None:
        acc["sessions"] = acc.get("sessions", 0) + 1
        lines = case_lines(case)
        for k, v in lines.items():
            acc[k] = acc.get(k, 0.0) + v
        if sp_batch and (case.get("method") or "multi_agent") == "single_prompt":
            acc["app_batched"] = acc.get("app_batched", 0.0) + lines["app"]

    by_arm, by_stage, total = {}, {}, {}
    for case in cases:
        add(by_arm.setdefault(case["arm"], {}), case)
        add(by_stage.setdefault(case["stage"], {}), case)
        add(total, case)

    def finish(acc: dict) -> dict:
        row = {"sessions": acc.get("sessions", 0), **{k: round(acc.get(k, 0.0), 4) for k in LINES}}
        row["standard"] = round(sum(acc.get(k, 0.0) for k in LINES), 2)
        row["batch"] = round(_batch({k: acc.get(k, 0.0) for k in LINES}, acc.get("app_batched", 0.0)), 2)
        return row

    stages = meta.get("stages") or {}
    out = {"arms": {a: {"name": (meta.get("arms") or {}).get(a, {}).get("name", a), **finish(v)}
                    for a, v in sorted(by_arm.items())},
           "stages": {st: {**finish(v), "plan_usd": stages.get(st, {}).get("plan_usd"),
                           "cap_usd": stages.get(st, {}).get("cap_usd")} for st, v in by_stage.items()},
           "total": finish(total), "unit_usd": AB_UNIT_USD, "batch_price_factor": BATCH_PRICE_FACTOR,
           "sp_batch": sp_batch}
    if set(by_stage) == set(stages):  # the whole matrix: compare with the plan's total and the total cap
        out["total"].update(plan_usd=meta.get("plan_usd"), cap_usd=(meta.get("caps") or {}).get("total"))
    for row in [*out["stages"].values(), out["total"]]:
        cap = row.get("cap_usd")
        if cap:
            row["within_cap"] = row["standard"] <= cap["standard"] and row["batch"] <= cap["batch"]
    return out


def format_estimate_table(est: dict) -> str:
    """The estimate as the plan prints it (cv-eval-plan section 3, "Recommended: stages 1-2")."""
    head = ["arm", "sessions", *(LINE_TITLES[k] for k in LINES), "total", "Batch"]
    usd = lambda x: f"${x:.2f}"  # noqa: E731
    rows = [[f"{a} {r['name']}", str(r["sessions"]), *(usd(r[k]) for k in LINES), usd(r["standard"]), usd(r["batch"])]
            for a, r in est["arms"].items()]
    t = est["total"]
    rows.append(["total", str(t["sessions"]), *(usd(t[k]) for k in LINES), usd(t["standard"]), usd(t["batch"])])
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    fmt = lambda r: "| " + " | ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))) + " |"  # noqa: E731
    lines = [fmt(head), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(fmt, rows), ""]

    def compare(label: str, row: dict) -> str:
        text = f"{label}: {row['sessions']} sessions, {usd(row['standard'])} standard / {usd(row['batch'])} Batch"
        if row.get("plan_usd"):
            text += f" (plan {usd(row['plan_usd']['standard'])} / {usd(row['plan_usd']['batch'])})"
        if row.get("cap_usd"):
            text += (f"; cap {usd(row['cap_usd']['standard'])} / {usd(row['cap_usd']['batch'])}: "
                     + ("within" if row["within_cap"] else "OVER"))
        return text

    lines += [compare(f"stage {st}", r) for st, r in est["stages"].items()]
    if t.get("plan_usd") or t.get("cap_usd"):
        lines.append(compare("total", t))
    if est.get("sp_batch"):
        lines.append("Standard: every grading and session at list price. Batch: the gradings, trap checks and "
                     f"single-prompt sessions (--sp-batch) at {1 - BATCH_PRICE_FACTOR:.0%} off (the app's sessions run "
                     "interactively, at the same price).")
    else:
        lines.append("Standard: every grading at list price. Batch: the gradings and trap checks at "
                     f"{1 - BATCH_PRICE_FACTOR:.0%} off (the sessions run interactively, at the same price; --sp-batch "
                     "prices the single-prompt sessions at the Batch rate too).")
    return "\n".join(lines)


def gradings_plan(cases: list[dict], out: Path, evaluate_args: list | tuple = ()) -> list[dict]:
    """Every grading and trap check of the selected cases, in the matrix's order (each case's sessions back to back):
    the jobs for a Batch grader."""
    jobs = []
    for case in cases:
        common = {"run_dir": str(out / case["id"]), "id": case["id"], "case": case.get("case"),
                  "arm": case.get("arm"), "stage": case.get("stage")}
        for g in case.get("gradings") or []:
            jobs.append({**common, "kind": "evaluation", "name": g["name"], "grader": g["grader"], "model": g["model"],
                         "evaluate_args": list(evaluate_args)})
        if case.get("truth"):
            jobs.append({**common, "kind": "truth_check", "name": case["truth"]["name"],
                         "grader": case["truth"]["grader"], "model": case["truth"]["model"]})
    for i, job in enumerate(jobs, 1):
        job["order"] = i
    return jobs


_GIT_RUN = subprocess.run  # bound at import: git is read-only, and tests fake subprocess.run for the sessions


def _git(*args: str):
    return _GIT_RUN(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def _git_ok(*args: str) -> bool:
    return _git(*args).returncode == 0


def app_version(frozen: str | None = None) -> dict:
    """The app version the sessions run on: HEAD, whether the app's files differ from it, and the frozen commit.
    With a frozen commit, "matches_frozen" says whether the app's files (APP_PATHS, working tree) equal it."""
    head = _git("rev-parse", "HEAD").stdout.strip()
    info = {"app_head": head or None, "app_dirty": not _git_ok("diff", "--quiet", "HEAD", "--", *APP_PATHS)}
    info["app_commit"] = frozen or info["app_head"]
    if frozen:
        info["matches_frozen"] = _git_ok("diff", "--quiet", frozen, "--", *APP_PATHS)
    return info


def select_cases(cases: list[dict], only: list[str] | None, repeats: int = 1) -> list[dict]:
    """The --only cases (all if none), expanded into repeats; every case must resolve to tracked fixtures."""
    selected = [c for c in cases if not only or c["id"] in only]
    if only and len(selected) < len(set(only)):
        missing = sorted(set(only) - {c["id"] for c in selected})
        raise SystemExit(f"unknown case ids: {', '.join(missing)}")
    for case in selected:
        resolve_case(case)  # before anything runs
    return expand_repeats(selected, repeats)


def default_total_cap(matrix: str | Path) -> float:
    """The matrix file's "total_cap", else DEFAULT_TOTAL_CAP (the 24 USD budget of the main round)."""
    cap = matrix_total_cap(matrix)
    return DEFAULT_TOTAL_CAP if cap is None else cap


def load_meta(matrix: str | Path) -> dict:
    """The matrix file without its cases."""
    data = json.loads(Path(matrix).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "cases"}


def select_stages(cases: list[dict], meta: dict, stages: list[str] | None) -> list[dict]:
    """The cases of the named stages (all if none); a matrix without stages takes no --stage."""
    known = list(meta.get("stages") or {})
    if not stages:
        return cases
    if not known:
        raise SystemExit("--stage: this matrix has no stages")
    unknown = sorted(set(stages) - set(known))
    if unknown:
        raise SystemExit(f"unknown stage(s) {', '.join(unknown)}; this matrix has {', '.join(known)}")
    return [c for c in cases if c.get("stage") in stages]


def configure_staged(opts, meta: dict, cases: list[dict]) -> None:
    """Caps, stage ids and grader arguments of a staged matrix on opts (Batch caps with --sessions-only)."""
    mode = "batch" if opts.sessions_only else "standard"
    if opts.total_cap is None:
        opts.total_cap = float(meta["caps"]["total"][mode])
    opts.stage_caps = {st: float(v["cap_usd"][mode]) for st, v in meta["stages"].items()}
    opts.stage_ids = {st: {c["id"] for c in cases if c.get("stage") == st} for st in meta["stages"]}
    opts.evaluate_args = list(meta.get("evaluate_args") or [])
    opts.cap_mode = mode


def _sp_batch_step(cases: list[dict], opts) -> int | None:
    """--sp-batch: submit / collect the single-prompt cases' calls before the pool runs; an exit code to stop with."""
    from evals.batch import PendingSubmission, format_summary
    from evals.batch_sp import sp_batch_round
    from evals.credit import OutOfCredit
    sp_cases = [c for c in cases if c.get("method") == "single_prompt"]
    if not sp_cases:
        opts.sp_batch_status = {}
        return None
    from dotenv import load_dotenv
    load_dotenv(opts.env_file or REPO / ".env")  # OPENAI_API_KEY for /v1/files and /v1/batches, as the sessions read it
    try:
        summary = sp_batch_round(sp_cases, opts, log=lambda m: print(f"sp batch: {m}", flush=True))
    except PendingSubmission as e:
        print(str(e).replace("--adopt", "--batch-adopt").replace("--forget-pending", "--batch-forget-pending"),
              file=sys.stderr)
        return 2
    except OutOfCredit as e:
        print(f"OUT OF CREDIT while submitting the OpenAI batch: {e}", flush=True)
        return EXIT_OUT_OF_CREDIT
    opts.sp_batch_status = summary["status"]
    print("sp batch: " + format_summary(summary).replace("--resubmit-failed", "--batch-resubmit-failed")
          .replace("\n", "\nsp batch: "), flush=True)
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--out", required=True, help="folder for run folders and the ledger (keep it outside the repo)")
    parser.add_argument("--env-file", help=".env file with API keys (read, never copied)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="+", help="run only these case ids")
    parser.add_argument("--total-cap", type=float, default=None,
                        help="start no new case past this many USD in total (default: the matrix's total_cap, else 24)")
    parser.add_argument("--cost-cap", type=float, default=1.5, help="per-session cap passed to run_case")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    parser.add_argument("--estimate", action="store_true", help="print the expected cost of the selected cases, run nothing")
    parser.add_argument("--repeats", type=int, default=1, help="run each selected case N times (ids _r1.._rN)")
    names = parser.add_mutually_exclusive_group()
    names.add_argument("--eval-name", default="eval1", help="name of the evaluation to make for each run (default eval1)")
    names.add_argument("--rescore", type=int, help="score each run K times: eval1 .. evalK (existing ones are kept)")
    parser.add_argument("--truth-check", action="store_true", help="also run evals.score_truth (truth1) on each run")
    parser.add_argument("--stage", nargs="+", help="staged matrix: run only these stages (e.g. pilot); required to run")
    parser.add_argument("--sessions-only", action="store_true",
                        help="run the sessions and no grading (grade them with the Batch grader); Batch caps apply")
    parser.add_argument("--gradings-out", help="staged matrix: write the grading jobs of the selection to this JSON file")
    parser.add_argument("--json", action="store_true", help="with --estimate on a staged matrix: print JSON, not the table")
    parser.add_argument("--sp-batch", action="store_true",
                        help="single-prompt sessions through OpenAI's Batch API (evals/batch_sp.py); synchronous otherwise")
    parser.add_argument("--batch-wait", action="store_true", help="with --sp-batch: poll until every batch is collected")
    parser.add_argument("--batch-poll-seconds", type=float, default=60.0)
    parser.add_argument("--batch-resubmit-failed", action="store_true",
                        help="with --sp-batch: send again the requests the API did not answer")
    parser.add_argument("--batch-adopt", action="append", help="with --sp-batch: KEY=BATCH_ID of an unrecorded submission")
    parser.add_argument("--batch-forget-pending", action="store_true",
                        help="with --sp-batch: drop an unrecorded submission that was never created")
    opts = parser.parse_args(argv)

    meta = load_meta(opts.matrix)
    staged = bool(meta.get("stages"))
    if staged and (opts.repeats > 1 or opts.rescore or opts.truth_check or opts.eval_name != "eval1"):
        raise SystemExit("a staged matrix names its own repeats, gradings and trap checks: "
                         "--repeats, --rescore, --eval-name and --truth-check do not apply")
    all_cases = load_matrix(opts.matrix)
    cases = select_cases(select_stages(all_cases, meta, opts.stage), opts.only, opts.repeats)
    if staged:
        configure_staged(opts, meta, all_cases)
    elif opts.total_cap is None:
        opts.total_cap = default_total_cap(opts.matrix)
    if opts.estimate:
        if staged:
            est = estimate_stages(cases, meta, opts.sp_batch)
            print(json.dumps(est, indent=2) if opts.json else format_estimate_table(est))
        else:
            print(json.dumps(estimate(cases, opts), indent=2))
        return 0
    if opts.gradings_out:
        if not staged:
            raise SystemExit("--gradings-out needs a staged matrix (its cases name their gradings)")
        jobs = gradings_plan(cases, Path(opts.out).resolve(), opts.evaluate_args)
        Path(opts.gradings_out).write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(jobs)} grading jobs to {opts.gradings_out}")
        return 0
    version = app_version(meta.get("app_commit"))
    opts.app_labels = {k: version[k] for k in ("app_commit", "app_head", "app_dirty")}
    if opts.dry_run:
        for case in cases:
            run_dir = Path(opts.out) / case["id"]
            command = " ".join(session_command(case, run_dir, opts.env_file, opts.cost_cap, opts.app_labels))
            if opts.sp_batch and case.get("method") == "single_prompt":
                command = f"[OpenAI batch {case['model']}] {command} --batch-result <the batch reply>"
            print(command)
            if opts.sessions_only:
                continue
            for name, model in case_gradings(case, opts):
                print(" ".join(evaluate_command(run_dir, opts.env_file, name, model, getattr(opts, "evaluate_args", ()))))
            truth = case_truth(case, opts)
            if truth:
                print(" ".join(truth_command(run_dir, opts.env_file, *truth)))
        return 0
    if staged and not opts.stage:
        raise SystemExit(f"choose the stage to run: --stage {' | '.join(meta['stages'])}")
    if version.get("matches_frozen") is False:
        raise SystemExit(f"the app's files ({', '.join(APP_PATHS)}) differ from the frozen commit "
                         f"{meta['app_commit']}: check out that commit, or rebuild the matrix with the new one")
    if staged:
        print(f"app {version['app_commit']} (HEAD {version['app_head']}{', app files changed' if version['app_dirty'] else ''}); "
              f"caps ({opts.cap_mode}): total ${opts.total_cap:.2f}, "
              + ", ".join(f"{st} ${c:.2f}" for st, c in opts.stage_caps.items()), flush=True)
    Path(opts.out).mkdir(parents=True, exist_ok=True)
    ledger = Path(opts.out) / "ledger.tsv"
    OUT_OF_CREDIT.clear()
    if opts.sp_batch:
        code = _sp_batch_step(cases, opts)
        if code is not None:
            return code
    results = {}
    with ThreadPoolExecutor(opts.workers) as pool:
        for cid, status in pool.map(lambda c: run_one(c, opts), cases):
            results[cid] = status
            print(f"{cid}: {status}  [spent ${spent(ledger):.3f}]", flush=True)
    if OUT_OF_CREDIT.is_set():
        finished = sum(1 for s in results.values()
                       if s == "ok" or s.startswith(("skip (done)", "skip (session done)", "session done")))
        interrupted = sorted(c for c, s in results.items() if s.startswith("interrupted"))
        unfinished = len(results) - finished - len(interrupted)
        print(f"\nOUT OF CREDIT: the model provider reported no credit left. Finished {finished}, "
              f"interrupted {len(interrupted)} ({', '.join(interrupted)}), not finished {unfinished}; "
              f"spent ${spent(ledger):.3f}. Top up, then run the same command again to resume.", flush=True)
        return EXIT_OUT_OF_CREDIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
