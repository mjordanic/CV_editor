"""Run the evaluation matrix: one testing-mode session per case, then the blind evaluator on it.

Every input comes from the tracked fixtures (`evals/fixtures/`, resolved by `evals/cases.py`); the cases
are in `evals/matrix.json`. This makes paid model calls (never Tavily: every ad has a search fixture).

Usage (from the repo root):
    .venv/bin/python -m evals.run_matrix --out RUNS_DIR --env-file PATH/.env [--matrix evals/matrix.json]
        [--workers 4] [--only CASE_ID ...] [--total-cap 23] [--cost-cap 1.5] [--dry-run]

Each case gets RUNS_DIR/<case id>/ (see evals/README.md) and RUNS_DIR/<case id>.out (console log).
RUNS_DIR/ledger.tsv records the cost of every session and evaluation; no new case starts once it
passes --total-cap. A case whose evaluation exists is skipped, so an interrupted matrix can be resumed.
--dry-run prints the commands without running anything.
"""
from __future__ import annotations

import argparse
import json
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


def session_command(case: dict, run_dir: Path, env_file: str | None, cost_cap: float) -> list[str]:
    paths = resolve_case(case)  # fails early if a fixture is missing
    cmd = [sys.executable, "-m", "evals.run_case", "--run-dir", str(run_dir), "--ad", case["ad"],
           "--persona", case["persona"], "--inputs", ",".join(paths["inputs"]), "--model", case["model"],
           "--cost-cap", str(cost_cap)]
    if env_file:
        cmd += ["--env-file", env_file]
    if case.get("work_preference"):
        cmd += ["--work-preference", case["work_preference"]]
    for key in ("persona", "ad", "model", "rep", "input_set"):
        cmd += ["--label", f"{key}={case.get(key)}"]
    return cmd


def evaluate_command(run_dir: Path, env_file: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "evals.evaluate", str(run_dir), "--name", "eval1"]
    return cmd + (["--env-file", env_file] if env_file else [])


def run_one(case: dict, opts) -> tuple[str, str]:
    cid = case["id"]
    out = Path(opts.out)
    run_dir, log, ledger = out / cid, out / f"{cid}.out", out / "ledger.tsv"
    if (run_dir / "evaluation" / "eval1" / "evaluation.json").exists():
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
    ev = subprocess.run(evaluate_command(run_dir, opts.env_file), cwd=str(REPO), capture_output=True, text=True)
    with open(log, "a") as f:
        f.write("\n--- evaluate ---\n" + ev.stdout[-3000:] + ev.stderr[-3000:])
    result = run_dir / "evaluation" / "eval1" / "evaluation.json"
    if result.exists():
        _record(ledger, cid, "eval", json.loads(result.read_text())["cost_usd"])
        return cid, "ok"
    return cid, "evaluation failed"


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
    opts = parser.parse_args(argv)

    cases = [c for c in load_matrix(opts.matrix) if not opts.only or c["id"] in opts.only]
    for case in cases:
        resolve_case(case)  # every case must resolve to tracked fixtures before anything runs
    if opts.dry_run:
        for case in cases:
            print(" ".join(session_command(case, Path(opts.out) / case["id"], opts.env_file, opts.cost_cap)))
        return 0
    Path(opts.out).mkdir(parents=True, exist_ok=True)
    ledger = Path(opts.out) / "ledger.tsv"
    with ThreadPoolExecutor(opts.workers) as pool:
        for cid, status in pool.map(lambda c: run_one(c, opts), cases):
            print(f"{cid}: {status}  [spent ${spent(ledger):.3f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
