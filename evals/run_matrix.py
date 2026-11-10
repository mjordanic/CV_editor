"""Run the evaluation matrix: one testing-mode session per case, then the blind evaluator on it.

Every input comes from the tracked fixtures (`evals/fixtures/`, resolved by `evals/cases.py`); the cases
are in `evals/matrix.json`. This makes paid model calls (never Tavily: every ad has a search fixture).

Usage (from the repo root):
    .venv/bin/python -m evals.run_matrix --out RUNS_DIR --env-file PATH/.env [--matrix evals/matrix.json]
        [--workers 4] [--only CASE_ID ...] [--total-cap 23] [--cost-cap 1.5] [--dry-run]
        [--repeats N] [--eval-name eval1 | --rescore K] [--truth-check]

Each case gets RUNS_DIR/<case id>/ (see evals/README.md) and RUNS_DIR/<case id>.out (console log).
RUNS_DIR/ledger.tsv records the cost of every session, evaluation and truth check; no new case starts
once it passes --total-cap. A case whose evaluations (and truth check, if asked) exist is skipped, and a
finished session is never run again, so an interrupted matrix can be resumed and an existing runs folder
re-scored. --dry-run prints the commands without running anything.

--repeats N runs every selected case N times: <id without _rN>_r1 .. _rN (rep 1 keeps the matrix id).
--rescore K scores each run K times (eval1 .. evalK); --eval-name names a single evaluation instead.
--truth-check also runs evals/score_truth.py (truth1) after the evaluations.
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

from evals.cases import DEFAULT_MATRIX, load_matrix, resolve_case

REPO = Path(__file__).resolve().parent.parent
_LOCK = threading.Lock()


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
    cmd = [sys.executable, "-m", "evals.run_case", "--run-dir", str(run_dir), "--ad", case["ad"],
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
    return cmd


def evaluate_command(run_dir: Path, env_file: str | None, name: str = "eval1") -> list[str]:
    cmd = [sys.executable, "-m", "evals.evaluate", str(run_dir), "--name", name]
    return cmd + (["--env-file", env_file] if env_file else [])


def truth_command(run_dir: Path, env_file: str | None, name: str = "truth1") -> list[str]:
    cmd = [sys.executable, "-m", "evals.score_truth", str(run_dir), "--name", name]
    return cmd + (["--env-file", env_file] if env_file else [])


def run_one(case: dict, opts) -> tuple[str, str]:
    cid = case["id"]
    out = Path(opts.out)
    run_dir, log, ledger = out / cid, out / f"{cid}.out", out / "ledger.tsv"
    evals = {n: run_dir / "evaluation" / n / "evaluation.json" for n in eval_names(opts)}
    truth = run_dir / "evaluation" / "truth1" / "truth_check.json"
    if all(f.exists() for f in evals.values()) and (not opts.truth_check or truth.exists()):
        return cid, "skip (done)"
    if spent(ledger) > opts.total_cap:
        return cid, "skip (total cap reached)"
    manifest = run_dir / "manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("status") == "running":
        shutil.rmtree(run_dir, ignore_errors=True)  # an unfinished session is redone from scratch
        with open(log, "w") as f:
            subprocess.run(session_command(case, run_dir, opts.env_file, opts.cost_cap), cwd=str(REPO),
                           stdout=f, stderr=subprocess.STDOUT)
        if not manifest.exists():
            return cid, "session failed before writing a manifest"
        _record(ledger, cid, "app", json.loads(manifest.read_text()).get("cost_usd") or 0)
    failed = []
    for name, result in evals.items():
        if result.exists():
            continue
        if (result.parent).exists():
            shutil.rmtree(result.parent)  # a failed attempt leaves a folder the evaluator refuses to reuse
        ev = subprocess.run(evaluate_command(run_dir, opts.env_file, name), cwd=str(REPO), capture_output=True, text=True)
        with open(log, "a") as f:
            f.write(f"\n--- evaluate {name} ---\n" + ev.stdout[-3000:] + ev.stderr[-3000:])
        if result.exists():
            _record(ledger, cid, "eval", json.loads(result.read_text())["cost_usd"])
        else:
            failed.append(name)
    if opts.truth_check and not truth.exists():
        if truth.parent.exists():
            shutil.rmtree(truth.parent)
        tc = subprocess.run(truth_command(run_dir, opts.env_file), cwd=str(REPO), capture_output=True, text=True)
        with open(log, "a") as f:
            f.write("\n--- truth check ---\n" + tc.stdout[-3000:] + tc.stderr[-3000:])
        if truth.exists():
            _record(ledger, cid, "truth", json.loads(truth.read_text())["cost_usd"])
        else:
            failed.append("truth1")
    return cid, f"failed: {', '.join(failed)}" if failed else "ok"


def select_cases(cases: list[dict], only: list[str] | None, repeats: int = 1) -> list[dict]:
    """The --only cases (all if none), expanded into repeats; every case must resolve to tracked fixtures."""
    selected = [c for c in cases if not only or c["id"] in only]
    if only and len(selected) < len(set(only)):
        missing = sorted(set(only) - {c["id"] for c in selected})
        raise SystemExit(f"unknown case ids: {', '.join(missing)}")
    for case in selected:
        resolve_case(case)  # before anything runs
    return expand_repeats(selected, repeats)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--out", required=True, help="folder for run folders and the ledger (keep it outside the repo)")
    parser.add_argument("--env-file", help=".env file with API keys (read, never copied)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="+", help="run only these case ids")
    parser.add_argument("--total-cap", type=float, default=23.0, help="start no new case past this many USD in total")
    parser.add_argument("--cost-cap", type=float, default=1.5, help="per-session cap passed to run_case")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    parser.add_argument("--repeats", type=int, default=1, help="run each selected case N times (ids _r1.._rN)")
    names = parser.add_mutually_exclusive_group()
    names.add_argument("--eval-name", default="eval1", help="name of the evaluation to make for each run (default eval1)")
    names.add_argument("--rescore", type=int, help="score each run K times: eval1 .. evalK (existing ones are kept)")
    parser.add_argument("--truth-check", action="store_true", help="also run evals.score_truth (truth1) on each run")
    opts = parser.parse_args(argv)

    cases = select_cases(load_matrix(opts.matrix), opts.only, opts.repeats)
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
    with ThreadPoolExecutor(opts.workers) as pool:
        for cid, status in pool.map(lambda c: run_one(c, opts), cases):
            print(f"{cid}: {status}  [spent ${spent(ledger):.3f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
