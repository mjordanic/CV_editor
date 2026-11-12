"""Run the multi-turn scenarios (evals/scenarios/scenarios.json) for the app and the chat baseline, score and report.

Each session is a subprocess: evals/run_case.py for the app ("app"; "app_frozen" runs the run_case of another tree,
--frozen-tree, e.g. an export of an earlier commit: `git archive <sha> | tar -x -C DIR`), evals/single_prompt_chat.py
for the chat baseline ("b2", and "b2_ask" with --ask-hint), on the scenario case's persona, ad, input set and script, every
writer on --model. Every finished session is scored by evals/score_checks.py (checks.json in its folder). The report
(report.md, results.json) has, per scenario, each check's pass count per system, cost, time and model calls per
user turn, and the paired comparison app vs each baseline (same case and rep), read by the pre-registered rule:
a difference is claimed only with at least 6 pairs, at least 4 discordant pairs in one direction and none the
other way (with the two-sided sign-test p).

Money: a session's own cap is 3x its estimate; a session starts only while spent (the scored sessions and the earlier
unfinished attempts of the planned ones) + the estimates of the running sessions + its own estimate stay within
--total-cap. Finished sessions (checks.json present) are not run again, so a stopped run resumes where it left off.
A session that did not finish (killed, Ctrl+C or SIGTERM, out of credit, a provider or network failure) is not scored;
its folder is set aside as <name>.unfinished-<time> and it runs again on the next pass. A session that runs out of
credit (exit code 75) stops the pool: nothing new starts, and the run exits with code 75 after the report.

app_frozen sessions need --frozen-tree and --frozen-commit SHA: the commit must be in this repository and the tree's
app files (run_matrix.APP_PATHS) must equal it byte for byte; each session's manifest gets the label app_commit=<sha>.

Usage (from the repo root):
    .venv/bin/python -m evals.run_scenarios --out DIR --dry-run                  # plan and cost estimate, no call
    .venv/bin/python -m evals.run_scenarios --out DIR --smoke --env-file .env     # each scenario's first case once
    .venv/bin/python -m evals.run_scenarios --out DIR --env-file .env [--only S2,S3] [--systems app,b2] [--reps 3]
        [--frozen-tree DIR --frozen-commit SHA]     # app_frozen: the tree is `git archive SHA | tar -x -C DIR`
    .venv/bin/python -m evals.run_scenarios --out DIR --report-only   # or --rescore: re-score with today's checks
    .venv/bin/python -m evals.run_scenarios --out DIR --report-only --publish evals/results/<date>-scenarios

--publish copies each scored session's records (manifest, config, summary, transcript, checks.json, inputs/, outputs/
without final_state.json; no events, model-call logs or workspace) with local paths made relative, plus the report
(as tables.md) and results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCENARIO_DIR = REPO / "evals" / "scenarios"
SCENARIOS = SCENARIO_DIR / "scenarios.json"
CAP_FACTOR = 3
RUNNERS = {"app": "evals.run_case", "app_frozen": "evals.run_case", "b2": "evals.single_prompt_chat",
           "b2_ask": "evals.single_prompt_chat"}
MIN_PAIRS, MIN_ONE_WAY = 6, 4


def load_spec(path: str | Path = SCENARIOS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan(spec: dict, only: list[str] | None = None, systems: list[str] | None = None, reps: int | None = None,
         smoke: bool = False) -> list[dict]:
    """The sessions to run: every scenario case x system x rep (smoke: each scenario's first case, once)."""
    sessions = []
    for sc in spec["scenarios"]:
        if only and sc["id"] not in only:
            continue
        for case in sc["cases"][:1] if smoke else sc["cases"]:
            for system, est in sc["est_usd"].items():
                if systems and system not in systems:
                    continue
                for rep in range(1, (1 if smoke else (reps or sc.get("reps", 1))) + 1):
                    sessions.append({"name": f"{sc['id']}-{case['script']}-{case['persona']}-{system}-r{rep}",
                                     "scenario": sc["id"], "system": system, "rep": rep, "est_usd": est,
                                     "case": f"{case['persona']}/{case['script']}", **case})
    return sessions


def estimate(sessions: list[dict]) -> dict:
    """{scenario: {system: USD}} and "total"."""
    out: dict = {"total": 0.0}
    for s in sessions:
        by = out.setdefault(s["scenario"], {})
        by[s["system"]] = round(by.get(s["system"], 0.0) + s["est_usd"], 4)
        out["total"] = round(out["total"] + s["est_usd"], 4)
    return out


def command(session: dict, run_dir: Path, model: str, env_file: str | None, labels: dict | None = None) -> list[str]:
    cmd = [sys.executable, "-m", RUNNERS[session["system"]], "--run-dir", str(run_dir), "--ad", session["ad"],
           "--persona", session["persona"], "--inputs", session["inputs"],
           "--script", str(SCENARIO_DIR / f"{session['script']}.json"), "--model", model,
           "--cost-cap", str(round(CAP_FACTOR * session["est_usd"], 4))]
    for key in ("scenario", "system", "case", "rep"):
        cmd += ["--label", f"{key}={session[key]}"]
    for key, value in (labels or {}).items():
        cmd += ["--label", f"{key}={value}"]
    if session["system"] == "b2_ask":
        cmd.append("--ask-hint")
    if env_file:
        cmd += ["--env-file", env_file]
    return cmd


def _cost(run_dir: Path) -> float:
    try:
        return float(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("cost_usd") or 0.0)
    except (OSError, ValueError):
        return 0.0


def _status(run_dir: Path) -> str | None:
    try:
        return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None


_GIT_RUN = subprocess.run  # bound at import: git is read-only, and tests fake subprocess.run for the sessions


def _blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_frozen_tree(tree: Path, commit: str, repo: Path = REPO) -> str:
    """The full id of `commit`, after checking that it is in `repo` and that every app file (run_matrix.APP_PATHS) it
    holds is in `tree` byte for byte (exits otherwise). Files the tree has beyond those (e.g. __pycache__) are ignored."""
    from evals.run_matrix import APP_PATHS
    git = lambda *a: _GIT_RUN(["git", "-C", str(repo), *a], capture_output=True, text=True)  # noqa: E731
    found = git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if found.returncode != 0 or not found.stdout.strip():
        sys.exit(f"--frozen-commit {commit} is not a commit in {repo}")
    sha = found.stdout.strip()
    listing = git("ls-tree", "-r", "-z", sha, "--", *APP_PATHS)
    differ = []
    for entry in filter(None, listing.stdout.split("\0")):
        meta, path = entry.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind != "blob" or mode == "120000":  # a symlink's target is not compared
            continue
        local = tree / path
        if not local.is_file() or _blob_sha(local) != blob:
            differ.append(path)
    if listing.returncode != 0 or not listing.stdout:
        sys.exit(f"--frozen-commit {sha[:12]}: no app files ({', '.join(APP_PATHS)}) found at that commit")
    if differ:
        sys.exit(f"--frozen-tree {tree} is not an export of {sha[:12]}: {len(differ)} app files differ or are missing, "
                 f"e.g. {', '.join(differ[:5])}")
    return sha


def run(sessions: list[dict], out: Path, model: str, env_file: str | None, total_cap: float, workers: int,
        frozen_tree: Path | None = None, frozen_commit: str | None = None) -> bool:
    """Run the sessions not yet scored. Returns True when a session ran out of credit (the pool then started nothing
    new)."""
    frozen_labels = {}
    if any(s["system"] == "app_frozen" for s in sessions):
        if not (frozen_tree and (frozen_tree / "evals" / "run_case.py").is_file()) or not frozen_commit:
            sys.exit("app_frozen sessions need --frozen-tree (a tree with evals/run_case.py: an export of the frozen "
                     "commit) and --frozen-commit SHA")
        frozen_labels = {"app_commit": check_frozen_tree(frozen_tree, frozen_commit)}
    from evals.credit import EXIT_OUT_OF_CREDIT, INTERRUPTED_STATUS
    from evals.score_checks import NotScorable, score_run
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    done = [s for s in sessions if (runs / s["name"] / "checks.json").is_file()]
    todo = [s for s in sessions if s not in done]
    attempts = [p for s in todo for p in (runs / s["name"], *runs.glob(f"{s['name']}.unfinished-*"))]
    money = {"spent": sum(_cost(runs / s["name"]) for s in done) + sum(_cost(p) for p in attempts), "reserved": 0.0}
    lock = threading.Lock()
    out_of_credit = threading.Event()

    def one(s: dict) -> None:
        run_dir = runs / s["name"]
        if out_of_credit.is_set():
            print(f"not started {s['name']}: out of credit", flush=True)
            return
        with lock:
            if money["spent"] + money["reserved"] + s["est_usd"] > total_cap:
                print(f"skip {s['name']}: ${money['spent']:.3f} spent + ${money['reserved']:.3f} running "
                      f"+ ${s['est_usd']} would pass the ${total_cap} cap", flush=True)
                return
            money["reserved"] += s["est_usd"]
        if run_dir.exists():  # an unfinished earlier attempt: kept aside, never deleted
            run_dir.rename(run_dir.with_name(f"{run_dir.name}.unfinished-{int(time.time())}"))
        started = time.time()
        frozen = s["system"] == "app_frozen"
        with open(runs / f"{s['name']}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(command(s, run_dir, model, env_file, frozen_labels if frozen else None),
                                  cwd=frozen_tree if frozen else REPO, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        cost = _cost(run_dir)
        with lock:
            money["reserved"] -= s["est_usd"]
            money["spent"] += cost
        if code == EXIT_OUT_OF_CREDIT or _status(run_dir) == INTERRUPTED_STATUS:
            out_of_credit.set()  # nothing new starts; the running sessions end at their own first credit error
        try:
            r = score_run(run_dir)
            result = f"{r['passed']}/{r['total']} checks, status {r['status']}"
        except NotScorable as e:  # unfinished: no checks.json, so the next pass runs it again
            result = f"not scored, runs again on the next pass ({e})"
        except Exception as e:  # noqa: BLE001 - a session that left no records is reported, not fatal
            result = f"not scored ({type(e).__name__}: {e})"
        print(f"{s['name']}: exit {code}, ${cost:.4f}, {(time.time() - started) / 60:.1f} min, {result} "
              f"(spent ${money['spent']:.3f})", flush=True)

    print(f"{len(done)} sessions already scored (${money['spent']:.3f} spent); running {len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    if out_of_credit.is_set():
        print("OUT OF CREDIT: no new session was started after it; run the same command after a top-up", flush=True)
    return out_of_credit.is_set()


def rescore(out: Path) -> None:
    """Score every session with a manifest again, with its scenario script as it is now (a corrected check). An app
    session without events.jsonl (a published one) is skipped and keeps its checks.json."""
    from evals.score_checks import NotScorable, score_run
    for manifest in sorted((out / "runs").glob("*/manifest.json")):
        if "." in manifest.parent.name:
            continue
        case = json.loads(manifest.read_text(encoding="utf-8")).get("labels", {}).get("case", "")
        script = SCENARIO_DIR / f"{case.split('/', 1)[-1]}.json"
        if script.is_file():
            try:
                score_run(manifest.parent, script)
            except NotScorable as e:
                print(f"skipped {e}", flush=True)


# ------------------------------------------------------------------------------------------- report
def collect(out: Path) -> list[dict]:
    """Every scored session in out/runs: its labels and checks.json."""
    rows = []
    for checks in sorted((out / "runs").glob("*/checks.json")):
        if "." in checks.parent.name:  # an attempt set aside (".unfinished-...", or renamed by hand)
            continue
        manifest = json.loads((checks.parent / "manifest.json").read_text(encoding="utf-8"))
        rows.append({**manifest.get("labels", {}), "name": checks.parent.name,
                     **json.loads(checks.read_text(encoding="utf-8"))})
    return rows


def paired(rows: list[dict], base: str, other: str, primary: list[str] | None = None) -> dict:
    """Pairs of sessions with the same case and rep: who passed (all `primary` checks, else all checks) alone."""
    def passed(r):
        checks = [c for c in r["checks"] if not primary or c["id"] in primary]
        return bool(checks) and all(c["pass"] for c in checks)
    by = {}
    for r in rows:
        by.setdefault((r.get("case"), str(r.get("rep"))), {})[r["system"]] = passed(r)
    pairs = [v for v in by.values() if base in v and other in v]
    a = sum(v[base] and not v[other] for v in pairs)
    b = sum(v[other] and not v[base] for v in pairs)
    if len(pairs) < MIN_PAIRS:
        verdict = f"too few pairs ({len(pairs)} < {MIN_PAIRS}) to claim a difference"
    elif a >= MIN_ONE_WAY and b == 0:
        verdict = f"{base} better"
    elif b >= MIN_ONE_WAY and a == 0:
        verdict = f"{other} better"
    else:
        verdict = "no difference detected"
    from evals.analysis.stats import sign_test
    p = sign_test([1] * a + [-1] * b)["p"]  # two-sided exact sign test on the discordant pairs
    return {"pairs": len(pairs), f"{base}_only": a, f"{other}_only": b, "sign_p": round(p, 3),
            "verdict": verdict}


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def report(out: Path, spec: dict) -> str:
    rows = collect(out)
    results, lines = {}, ["# Multi-turn scenarios: app vs chat baseline", "",
                          f"Sessions scored: {len(rows)}. Spent: ${sum(r.get('cost_usd') or 0 for r in rows):.3f}.", ""]
    for sc in spec["scenarios"]:
        mine = [r for r in rows if r.get("scenario") == sc["id"]]
        if not mine:
            continue
        systems = [s for s in sc["est_usd"] if any(r["system"] == s for r in mine)]
        lines += [f"## {sc['id']}: {sc['title']}", "", f"Expected: {sc.get('expected', '')}", "",
                  "| check | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
        check_ids = list(dict.fromkeys(c["id"] for r in mine for c in r["checks"]))
        for cid in check_ids:
            cells = []
            for s in systems:
                got = [c["pass"] for r in mine if r["system"] == s for c in r["checks"] if c["id"] == cid]
                cells.append(f"{sum(got)}/{len(got)}" if got else "-")
            lines.append(f"| {cid} | " + " | ".join(cells) + " |")
        metrics = {}
        for s in systems:
            rs = [r for r in mine if r["system"] == s]
            metrics[s] = {"sessions": len(rs), "all_passed": sum(r["all_passed"] for r in rs),
                          "not_completed": sum(r["status"] != "completed" for r in rs),
                          "usd_session": _mean([r.get("cost_usd") for r in rs]),
                          "usd_turn": _mean([r.get("usd_per_turn") for r in rs]),
                          "seconds_turn": _mean([r.get("seconds_per_turn") for r in rs]),
                          "calls_turn": _mean([r.get("calls_per_turn") for r in rs]),
                          "latex_fixes": sum(r.get("latex_fixes") or 0 for r in rs)}
        for key, label in (("all_passed", "**all checks passed**"), ("not_completed", "sessions not completed"),
                           ("usd_session", "USD per session"), ("usd_turn", "USD per user turn"),
                           ("seconds_turn", "seconds per user turn"), ("calls_turn", "model calls per user turn"),
                           ("latex_fixes", "LaTeX fixes (app repair calls)")):
            cells = [f"{metrics[s][key]}/{metrics[s]['sessions']}" if key == "all_passed" else str(metrics[s][key])
                     for s in systems]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        comparisons = {o: paired(mine, "app", o, sc.get("primary_checks")) for o in systems if o != "app"} \
            if "app" in systems else {}
        lines.append("")
        for o, c in comparisons.items():
            lines.append(f"Paired app vs {o}{' (primary checks)' if sc.get('primary_checks') else ''}: {c['pairs']} "
                         f"pairs, app alone {c['app_only']}, {o} alone {c[f'{o}_only']}, sign p={c['sign_p']}: "
                         f"**{c['verdict']}**.")
        failures = [f"- {r['name']}: " + "; ".join(f"{c['id']} ({c['detail'][:120]})" for c in r["checks"]
                                                    if not c["pass"]) for r in mine if not r["all_passed"]]
        if failures:
            lines += ["", "<details><summary>failed checks</summary>", "", *failures, "", "</details>"]
        lines.append("")
        results[sc["id"]] = {"metrics": metrics, "paired": comparisons}
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


PUBLISHED_FILES = ("manifest.json", "config.json", "summary.json", "transcript.json", "checks.json")


def _relative_paths(text: str, run_dir: Path, published: str, frozen_tree: Path | None = None) -> str:
    """Local absolute paths in a record made relative: the run folder, the repository, the frozen tree, the home folder."""
    pairs = [(str(run_dir), published), (str(REPO), "<repo>")]
    pairs += [(str(frozen_tree), "<frozen-tree>")] if frozen_tree else []
    for local, shown in (*pairs, (str(Path.home()), "~")):
        text = text.replace(local, shown)
    return text


def publish(out: Path, dest: Path, frozen_tree: Path | None = None) -> int:
    """Copy every scored session's shareable records to dest/runs/<name>/, and report.md and results.json to dest/."""
    from provenance import scrub
    rows = collect(out)
    for r in rows:
        src, target = out / "runs" / r["name"], dest / "runs" / r["name"]
        for path in [*(src / f for f in PUBLISHED_FILES), *sorted((src / "inputs").glob("*")),
                     *sorted(p for p in (src / "outputs").glob("*") if p.name != "final_state.json")]:
            if not path.is_file():
                continue
            to = target / path.relative_to(src)
            to.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in (".json", ".txt", ".tex", ".md"):
                text = _relative_paths(path.read_text(encoding="utf-8"), src, f"runs/{r['name']}", frozen_tree)
                if path.suffix == ".json":
                    text = json.dumps(scrub(json.loads(text)), indent=2, ensure_ascii=False)
                to.write_text(text, encoding="utf-8")
            else:
                shutil.copy(path, to)
    # report.md is published as tables.md: a round's own REPORT.md sits beside it (one file on case-insensitive disks)
    for name, published in (("report.md", "tables.md"), ("results.json", "results.json")):
        if (out / name).is_file():
            shutil.copy(out / name, dest / published)
    return len(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output folder (runs/, report.md, results.json)")
    parser.add_argument("--spec", default=str(SCENARIOS), help="scenario table (default evals/scenarios/scenarios.json)")
    parser.add_argument("--only", help="comma-separated scenario ids, e.g. S2,S3")
    parser.add_argument("--systems", help="comma-separated systems: app, b2, b2_ask (default: each scenario's own)")
    parser.add_argument("--reps", type=int, help="reps per case and system (default: each scenario's own)")
    parser.add_argument("--smoke", action="store_true", help="each scenario's first case, once per system")
    parser.add_argument("--dry-run", action="store_true", help="print the sessions and the cost estimate; run nothing")
    parser.add_argument("--report-only", action="store_true", help="only (re)write the report from scored sessions")
    parser.add_argument("--rescore", action="store_true",
                        help="score every finished session again with the current scenario scripts, then report")
    parser.add_argument("--model", help="model for every writer (default: the table's)")
    parser.add_argument("--env-file", help=".env file with API keys (read by each session, never copied)")
    parser.add_argument("--total-cap", type=float, help="USD for all sessions together (default: the table's)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frozen-tree", help="the tree app_frozen sessions run from (an export of the frozen commit)")
    parser.add_argument("--frozen-commit", help="the commit --frozen-tree is an export of (checked; recorded as the "
                                                "label app_commit)")
    parser.add_argument("--publish", help="after the report, copy the scored sessions' records here (see above)")
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    out = Path(args.out).resolve()
    sessions = plan(spec, args.only.split(",") if args.only else None,
                    args.systems.split(",") if args.systems else None, args.reps, args.smoke)
    if args.dry_run:
        for s in sessions:
            print(f"{s['name']}: ~${s['est_usd']}")
        print(json.dumps(estimate(sessions), indent=2))
        return 0
    stopped = False
    if args.rescore:
        rescore(out)
    elif not args.report_only:
        if shutil.which("pdflatex") is None:
            sys.exit("pdflatex is not installed: every session needs it")
        # absolute: app_frozen sessions run with the frozen tree as their working folder
        env_file = str(Path(args.env_file).resolve()) if args.env_file else None
        stopped = run(sessions, out, args.model or spec["model"], env_file,
                      args.total_cap if args.total_cap is not None else spec["total_cap"], args.workers,
                      Path(args.frozen_tree).resolve() if args.frozen_tree else None, args.frozen_commit)
    print(report(out, spec))
    if args.publish:
        frozen = Path(args.frozen_tree).resolve() if args.frozen_tree else None
        print(f"published {publish(out, Path(args.publish), frozen)} sessions to {args.publish}")
    if stopped:
        from evals.credit import EXIT_OUT_OF_CREDIT
        return EXIT_OUT_OF_CREDIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
