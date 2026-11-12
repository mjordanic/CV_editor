"""Run the multi-turn scenarios (evals/scenarios/scenarios.json) for the app and the chat baseline, score and report.

Each session is a subprocess: evals/run_case.py for the app ("app"; "app_frozen" runs the run_case of another tree,
--frozen-tree, e.g. an export of an earlier commit: `git archive <sha> | tar -x -C DIR`), evals/single_prompt_chat.py
for the chat baseline ("b2", and "b2_ask" with --ask-hint), on the scenario case's persona, ad, input set and script, every
writer on --model. Every finished session is scored by evals/score_checks.py (checks.json in its folder). The report
(report.md, results.json) has, per scenario, each check's pass count per system, cost, time and model calls per
user turn, and the paired comparison app vs each baseline (same case and rep), read by the pre-registered rule:
a difference is claimed only with at least 6 pairs, at least 4 discordant pairs in one direction and none the
other way (with the two-sided sign-test p).

Money: a session's own cap is 3x its estimate; a session starts only while spent + the estimates of the running
sessions + its own estimate stay within --total-cap. Finished sessions (checks.json present) are not run again,
so a stopped run resumes where it left off.

Usage (from the repo root):
    .venv/bin/python -m evals.run_scenarios --out DIR --dry-run                  # plan and cost estimate, no call
    .venv/bin/python -m evals.run_scenarios --out DIR --smoke --env-file .env     # each scenario's first case once
    .venv/bin/python -m evals.run_scenarios --out DIR --env-file .env [--only S2,S3] [--systems app,b2] [--reps 3]
    .venv/bin/python -m evals.run_scenarios --out DIR --report-only   # or --rescore: re-score with today's checks
    .venv/bin/python -m evals.run_scenarios --out DIR --report-only --publish evals/results/<date>-scenarios

--publish copies each scored session's records (manifest, config, summary, transcript, checks.json, inputs/, outputs/
without final_state.json; no events, model-call logs or workspace) with local paths made relative, plus the report.
"""
from __future__ import annotations

import argparse
import json
import math
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


def command(session: dict, run_dir: Path, model: str, env_file: str | None) -> list[str]:
    cmd = [sys.executable, "-m", RUNNERS[session["system"]], "--run-dir", str(run_dir), "--ad", session["ad"],
           "--persona", session["persona"], "--inputs", session["inputs"],
           "--script", str(SCENARIO_DIR / f"{session['script']}.json"), "--model", model,
           "--cost-cap", str(round(CAP_FACTOR * session["est_usd"], 4))]
    for key in ("scenario", "system", "case", "rep"):
        cmd += ["--label", f"{key}={session[key]}"]
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


def run(sessions: list[dict], out: Path, model: str, env_file: str | None, total_cap: float, workers: int,
        frozen_tree: Path | None = None) -> None:
    if any(s["system"] == "app_frozen" for s in sessions) and not (frozen_tree and (frozen_tree / "evals" / "run_case.py").is_file()):
        sys.exit("app_frozen sessions need --frozen-tree: a tree with evals/run_case.py (an export of the frozen commit)")
    from evals.score_checks import score_run
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    done = [s for s in sessions if (runs / s["name"] / "checks.json").is_file()]
    money = {"spent": sum(_cost(runs / s["name"]) for s in done), "reserved": 0.0}
    lock = threading.Lock()

    def one(s: dict) -> None:
        run_dir = runs / s["name"]
        with lock:
            if money["spent"] + money["reserved"] + s["est_usd"] > total_cap:
                print(f"skip {s['name']}: ${money['spent']:.3f} spent + ${money['reserved']:.3f} running "
                      f"+ ${s['est_usd']} would pass the ${total_cap} cap", flush=True)
                return
            money["reserved"] += s["est_usd"]
        if run_dir.exists():  # an unfinished earlier attempt: kept aside, never deleted
            run_dir.rename(run_dir.with_name(f"{run_dir.name}.unfinished-{int(time.time())}"))
        started = time.time()
        with open(runs / f"{s['name']}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(command(s, run_dir, model, env_file),
                                  cwd=frozen_tree if s["system"] == "app_frozen" else REPO, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        cost = _cost(run_dir)
        with lock:
            money["reserved"] -= s["est_usd"]
            money["spent"] += cost
        try:
            r = score_run(run_dir)
            result = f"{r['passed']}/{r['total']} checks, status {r['status']}"
        except Exception as e:  # noqa: BLE001 - a session that left no records is reported, not fatal
            result = f"not scored ({type(e).__name__}: {e})"
        print(f"{s['name']}: exit {code}, ${cost:.4f}, {(time.time() - started) / 60:.1f} min, {result} "
              f"(spent ${money['spent']:.3f})", flush=True)

    todo = [s for s in sessions if s not in done]
    print(f"{len(done)} sessions already scored (${money['spent']:.3f}); running {len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))


def rescore(out: Path) -> None:
    """Score every session with a manifest again, with its scenario script as it is now (a corrected check)."""
    from evals.score_checks import score_run
    for manifest in sorted((out / "runs").glob("*/manifest.json")):
        if "." in manifest.parent.name:
            continue
        case = json.loads(manifest.read_text(encoding="utf-8")).get("labels", {}).get("case", "")
        script = SCENARIO_DIR / f"{case.split('/', 1)[-1]}.json"
        if script.is_file():
            score_run(manifest.parent, script)


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


def sign_test(a: int, b: int) -> float:
    """Two-sided exact sign test on a and b discordant pairs."""
    n = a + b
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)


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
    return {"pairs": len(pairs), f"{base}_only": a, f"{other}_only": b, "sign_p": round(sign_test(a, b), 3),
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
    for name in ("report.md", "results.json"):
        if (out / name).is_file():
            shutil.copy(out / name, dest / name)
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
    if args.rescore:
        rescore(out)
    elif not args.report_only:
        if shutil.which("pdflatex") is None:
            sys.exit("pdflatex is not installed: every session needs it")
        run(sessions, out, args.model or spec["model"], args.env_file,
            args.total_cap if args.total_cap is not None else spec["total_cap"], args.workers,
            Path(args.frozen_tree).resolve() if args.frozen_tree else None)
    print(report(out, spec))
    if args.publish:
        frozen = Path(args.frozen_tree).resolve() if args.frozen_tree else None
        print(f"published {publish(out, Path(args.publish), frozen)} sessions to {args.publish}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
