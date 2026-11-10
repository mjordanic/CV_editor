"""Run the evaluation matrix: one testing-mode session per case, then the blind evaluator on it.

Every input comes from the tracked fixtures (`evals/fixtures/`, resolved by `evals/cases.py`); the cases
are in `evals/matrix.json`. This makes paid model calls (never Tavily: every ad has a search fixture).

Usage (from the repo root):
    .venv/bin/python -m evals.run_matrix --out RUNS_DIR --env-file PATH/.env [--matrix evals/matrix.json]
        [--workers 4] [--only CASE_ID ...] [--total-cap 24] [--cost-cap 1.5] [--dry-run] [--estimate]
        [--repeats N] [--eval-name eval1 | --rescore K] [--truth-check]

Each case gets RUNS_DIR/<case id>/ (see evals/README.md) and RUNS_DIR/<case id>.out (console log).
RUNS_DIR/ledger.tsv records the cost of every session, evaluation and truth check; no new case starts
once it passes --total-cap. A case whose evaluations (and truth check, if asked) exist is skipped, and a
finished session is never run again, so an interrupted matrix can be resumed and an existing runs folder
re-scored. --dry-run prints the commands without running anything.

--repeats N runs every selected case N times: <id without _rN>_r1 .. _rN (rep 1 keeps the matrix id).
--rescore K scores each run K times (eval1 .. evalK); --eval-name names a single evaluation instead.
--truth-check also runs evals/score_truth.py (truth1) after the evaluations.
--estimate prints the expected cost of the selected cases (ESTIMATE_USD below) and runs nothing.

A case with "method": "single_prompt" (evals/matrix_single_prompt.json) runs evals/single_prompt.py
(one model call writes both documents, then the app's PDF path) instead of the multi-agent session; the
ledger, caps, resume, evaluation and out-of-credit handling are the same. --total-cap defaults to the
matrix file's "total_cap" (12 USD for the single-prompt matrix), else 24.

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
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.cases import DEFAULT_MATRIX, load_matrix, matrix_total_cap, resolve_case
from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS

REPO = Path(__file__).resolve().parent.parent
_LOCK = threading.Lock()
OUT_OF_CREDIT = threading.Event()  # set on the first exit code 75; nothing new starts after it

# Expected cost per step in USD, from the measured partial runs (2026-11): a gpt-6-luna session about
# 0.013, a gpt-6.1-terra session about 0.06, one gpt-6.1-sol evaluation 0.11-0.14 (text only). The
# evaluation now also reads the conversation, page images (about 1.5k tokens a page, up to 5 pages at
# $2/1M input) and answers four more rubric sections (about 2k more output tokens at $10/1M): about
# +0.04, so 0.18 at the high end. A script with change requests redrafts more: its session is
# estimated at twice the default session. The truth check is one short luna call.
# A single-prompt session (evals/single_prompt.py) is one generation call plus the app's PDF rendering.
# gpt-6.1-sol at $2 / $10 per 1M tokens: the prompt is the writers' rules plus the case's inputs, job ad and
# search results (about 4-7k tokens on these fixtures, ~0.01), the answer both documents and notes plus
# reasoning (up to about 7k tokens, ~0.07): 0.09 at the high end. Rendering is two LaTeX conversions on the
# template_customizer model (gpt-6-luna), with the possible fix and re-conversion: about 0.01.
ESTIMATE_USD = {"session": {"gpt-6-luna": 0.013, "gpt-6.1-terra": 0.06}, "eval": 0.18, "truth": 0.005,
                "scripted_session_factor": 2.0,
                "single_prompt": {"gpt-6.1-sol": 0.09}, "single_prompt_render": 0.01}
DEFAULT_TOTAL_CAP = 24.0  # USD, when the matrix file sets no "total_cap"


def spent(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    return sum(float(line.split("\t")[2]) for line in ledger.read_text().splitlines() if line.strip())


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


def session_command(case: dict, run_dir: Path, env_file: str | None, cost_cap: float) -> list[str]:
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
    cmd += ["--label", f"case={base_id(case['id'])}", "--label", f"script={case.get('script') or 'default'}"]
    if case.get("method"):
        cmd += ["--label", f"method={case['method']}"]
    if case.get("baseline"):
        cmd += ["--label", f"baseline={case['baseline']}"]
    return cmd


def evaluate_command(run_dir: Path, env_file: str | None, name: str = "eval1") -> list[str]:
    cmd = [sys.executable, "-m", "evals.evaluate", str(run_dir), "--name", name]
    return cmd + (["--env-file", env_file] if env_file else [])


def truth_command(run_dir: Path, env_file: str | None, name: str = "truth1") -> list[str]:
    cmd = [sys.executable, "-m", "evals.score_truth", str(run_dir), "--name", name]
    return cmd + (["--env-file", env_file] if env_file else [])


def _manifest_status(manifest: Path) -> str | None:
    try:
        return json.loads(manifest.read_text()).get("status")
    except (OSError, json.JSONDecodeError):
        return None


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
    evals = {n: run_dir / "evaluation" / n / "evaluation.json" for n in eval_names(opts)}
    truth = run_dir / "evaluation" / "truth1" / "truth_check.json"
    if all(f.exists() for f in evals.values()) and (not opts.truth_check or truth.exists()):
        return cid, "skip (done)"
    if OUT_OF_CREDIT.is_set():
        return cid, "not started (out of credit)"
    if spent(ledger) > opts.total_cap:
        return cid, "skip (total cap reached)"
    manifest = run_dir / "manifest.json"
    status = _manifest_status(manifest) if manifest.exists() else None
    if status is None or status == "running" or str(status).startswith("interrupted"):
        shutil.rmtree(run_dir, ignore_errors=True)  # an unfinished or interrupted session is redone from scratch
        with open(log, "w") as f:
            proc = subprocess.run(session_command(case, run_dir, opts.env_file, opts.cost_cap), cwd=str(REPO),
                                  stdout=f, stderr=subprocess.STDOUT)
        if not manifest.exists():
            if proc.returncode == EXIT_OUT_OF_CREDIT:
                return _mark_out_of_credit(cid, "the session")
            return cid, "session failed before writing a manifest"
        _record(ledger, cid, "app", json.loads(manifest.read_text()).get("cost_usd") or 0)
        if proc.returncode == EXIT_OUT_OF_CREDIT or _manifest_status(manifest) == INTERRUPTED_STATUS:
            return _mark_out_of_credit(cid, "the session")  # not evaluated: redone after a top-up
    failed = []
    for name, result in evals.items():
        if result.exists():
            continue
        if OUT_OF_CREDIT.is_set():
            return cid, "evaluation not started (out of credit)"
        if (result.parent).exists():
            shutil.rmtree(result.parent)  # a failed attempt leaves a folder the evaluator refuses to reuse
        ev = subprocess.run(evaluate_command(run_dir, opts.env_file, name), cwd=str(REPO), capture_output=True, text=True)
        with open(log, "a") as f:
            f.write(f"\n--- evaluate {name} ---\n" + ev.stdout[-3000:] + ev.stderr[-3000:])
        if result.exists():
            _record(ledger, cid, "eval", json.loads(result.read_text())["cost_usd"])
        elif ev.returncode == EXIT_OUT_OF_CREDIT:
            _record(ledger, cid, "eval-interrupted", _partial_cost(result.parent))
            return _mark_out_of_credit(cid, f"evaluation {name}")
        else:
            failed.append(name)
    if opts.truth_check and not truth.exists():
        if OUT_OF_CREDIT.is_set():
            return cid, "truth check not started (out of credit)"
        if truth.parent.exists():
            shutil.rmtree(truth.parent)
        tc = subprocess.run(truth_command(run_dir, opts.env_file), cwd=str(REPO), capture_output=True, text=True)
        with open(log, "a") as f:
            f.write("\n--- truth check ---\n" + tc.stdout[-3000:] + tc.stderr[-3000:])
        if truth.exists():
            _record(ledger, cid, "truth", json.loads(truth.read_text())["cost_usd"])
        elif tc.returncode == EXIT_OUT_OF_CREDIT or OUT_OF_CREDIT.is_set():
            return _mark_out_of_credit(cid, "the truth check")
        else:
            failed.append("truth1")
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
            session = p["single_prompt"].get(model, max(p["single_prompt"].values())) + p["single_prompt_render"]
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
            "total_cap": getattr(opts, "total_cap", None), "prices": p}


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
    opts = parser.parse_args(argv)

    if opts.total_cap is None:
        opts.total_cap = default_total_cap(opts.matrix)
    cases = select_cases(load_matrix(opts.matrix), opts.only, opts.repeats)
    if opts.estimate:
        print(json.dumps(estimate(cases, opts), indent=2))
        return 0
    if opts.dry_run:
        for case in cases:
            run_dir = Path(opts.out) / case["id"]
            print(" ".join(session_command(case, run_dir, opts.env_file, opts.cost_cap)))
            for name in eval_names(opts):
                print(" ".join(evaluate_command(run_dir, opts.env_file, name)))
            if opts.truth_check:
                print(" ".join(truth_command(run_dir, opts.env_file)))
        return 0
    Path(opts.out).mkdir(parents=True, exist_ok=True)
    ledger = Path(opts.out) / "ledger.tsv"
    OUT_OF_CREDIT.clear()
    results = {}
    with ThreadPoolExecutor(opts.workers) as pool:
        for cid, status in pool.map(lambda c: run_one(c, opts), cases):
            results[cid] = status
            print(f"{cid}: {status}  [spent ${spent(ledger):.3f}]", flush=True)
    if OUT_OF_CREDIT.is_set():
        finished = sum(1 for s in results.values() if s == "ok" or s.startswith("skip (done)"))
        interrupted = sorted(c for c, s in results.items() if s.startswith("interrupted"))
        unfinished = len(results) - finished - len(interrupted)
        print(f"\nOUT OF CREDIT: the model provider reported no credit left. Finished {finished}, "
              f"interrupted {len(interrupted)} ({', '.join(interrupted)}), not finished {unfinished}; "
              f"spent ${spent(ledger):.3f}. Top up, then run the same command again to resume.", flush=True)
        return EXIT_OUT_OF_CREDIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
