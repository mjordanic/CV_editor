"""Batch grading: the evaluations and trap checks of many run folders through Anthropic's Message Batches API.

The same gradings as evals/evaluate.py and evals/score_truth.py, at the Batch price (half of list). Each request is
built by the synchronous code (build_grading_request / build_truth_check -> GradingRequest.anthropic_params()), so
a batch request's `params` are exactly the body the synchronous path streams (minus `stream`). Each reply is read
with evals.grader.result_from_message(..., batch_id=) and written by the synchronous code (finish_evaluation /
finish_truth_check) into the same RUN/evaluation/<name>/ files: prompt.json and pages/ when submitted;
raw_response.json, evaluation.json or truth_check.json, llm_calls.jsonl and summary.json when collected; the grader
record says `batch_id` and the cost is the Batch price.

Jobs come from a plan written by `run_matrix --gradings-out FILE` (each case's gradings: eval1/eval2 by Sonnet,
opus1 by Opus, haiku1/haiku2 by Haiku, truth1 the Haiku trap check) or from run folders on the command line.
Only Anthropic graders go through here; an openai:* grader is refused (grade it with evals/evaluate.py: its
LangChain path re-asks an unparseable answer inside the call loop, which a batch cannot replay).

Durable and resumable (evals/batch.py): STATE_DIR/state.json records every request's custom_id, request hash and
batch id before waiting. Re-running the same command collects ended batches and submits only what is due; nothing is
submitted twice. A grading that finished on disk (evaluation.json / truth_check.json) is never redone. All gradings
of one run go in one submission, a case's arms and graders (Sonnet, Opus, Haiku, trap check) always in the same batch.

The synchronous retry rule (evals.grader.MAX_ATTEMPTS = 2): an answer that does not parse, and a request the API did
not answer (errored, expired, canceled; not billed), go out once more in a follow-up batch on the next run (or at
once with --wait). The folder is written once, at the end, with every answered attempt in llm_calls.jsonl and
raw_response.json, as send() writes a re-asked grading; the grader record adds "batch" (custom_id, and each attempt's
batch id, result type, submitted and ended times) and "cost_list_usd" (the list price; "cost_usd" is billed). A
refusal is recorded and final, never re-routed. What still fails after two attempts is listed; --resubmit-failed sends
the unanswered ones again (never an answered one: that would be a third attempt the synchronous path does not make).

Usage (from the repo root; submitting and collecting make paid calls, nothing else does):
    .venv/bin/python -m evals.batch_grade --jobs GRADINGS.json [--state DIR] [--wait] [--resubmit-failed]
    .venv/bin/python -m evals.batch_grade RUN_DIR ... --eval-name eval1 [--grader-model anthropic:claude-sonnet-5-5]
        [--grading-mode document-only] [--effort medium]
    .venv/bin/python -m evals.batch_grade RUN_DIR ... --truth-check [--eval-name truth1] [--persona p12]
    common: [--env-file .env] [--state DIR] [--wait] [--poll-seconds 60] [--resubmit-failed] [--status]
            [--adopt KEY=BATCH_ID] [--forget-pending] [--ledger FILE] [--dry-run]
The state folder defaults to <the run folders' parent>/batches/grading. --status only reads state.json (no call).
--dry-run builds every request and prints the counts and sizes, writing nothing and calling nothing.
Costs of collected gradings are added to the ledger (default <the run folders' parent>/ledger.tsv when it exists,
as run_matrix writes it): eval:<name>, truth, *-failed rows for paid failures,
*-attempt1 for a re-asked first answer and *-discarded for a billed reply that was not written.
Out of credit while submitting: nothing more is submitted and the exit code is 75 (evals/credit.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evals.batch import (RETRY, Adapter, BatchStore, Job, NotSubmitted, Outcome, PendingSubmission, Prepared,  # noqa: E402
                         canonical_sha256, format_summary, make_custom_id, parse_adopt, run_round, summary)
from provenance import BATCH_PRICE_FACTOR  # noqa: E402

EVALUATION, TRUTH = "evaluation", "truth_check"
RESULT_FILE = {EVALUATION: "evaluation.json", TRUTH: "truth_check.json"}
NODE = {EVALUATION: "evaluator", TRUTH: "truth_check"}
STATE_KIND = "grading"


# --------------------------------------------------------------------------------------------- the provider
class AnthropicBatches(Adapter):
    """Message Batches through the official SDK (`client` replaces anthropic.Anthropic() in tests)."""
    provider = "anthropic"
    max_requests = 10_000  # the API allows 100,000 requests or 256 MB per batch
    max_bytes = 200_000_000

    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from evals.grader import anthropic_client
            self._client = anthropic_client()
        return self._client

    def submit(self, key: str, store_id: str, prepared: list[Prepared]) -> str:
        from evals.credit import OutOfCredit, is_out_of_credit
        try:
            client = self.client
        except Exception as e:  # no key, no SDK: nothing was sent
            raise NotSubmitted(f"{type(e).__name__}: {e}"[:500]) from e
        try:
            batch = client.messages.batches.create(requests=[p.request for p in prepared])
        except Exception as e:
            if is_out_of_credit(e):
                raise OutOfCredit(f"{type(e).__name__}: {e}"[:500]) from e
            raise
        return batch.id

    def status(self, batch_id: str) -> tuple[bool, dict]:
        batch = self.client.messages.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        counts = counts.model_dump() if hasattr(counts, "model_dump") else counts
        ended_at = getattr(batch, "ended_at", None)
        return batch.processing_status == "ended", {"processing_status": batch.processing_status,
                                                    "request_counts": counts,
                                                    "ended_at": ended_at.isoformat() if hasattr(ended_at, "isoformat")
                                                    else ended_at}

    def results(self, batch_id: str):
        for r in self.client.messages.batches.results(batch_id):
            result = r.result
            kind = result.type
            error = getattr(result, "error", None)
            if hasattr(error, "model_dump"):
                error = error.model_dump(mode="json")
            yield Outcome(r.custom_id, kind, getattr(result, "message", None), error if kind == "errored" else
                          (kind if kind in ("expired", "canceled") else None))


# --------------------------------------------------------------------------------------------- the jobs
def _evaluate_options(args: list[str]) -> dict:
    """--grading-mode / --effort from a plan's evaluate_args."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--grading-mode", dest="mode")
    p.add_argument("--effort")
    known, rest = p.parse_known_args(list(args or []))
    if rest:
        raise ValueError(f"evaluate_args {rest}: Batch grading takes only --grading-mode and --effort")
    return {k: v for k, v in vars(known).items() if v}


def normalise_job(raw: dict) -> dict:
    """A grading job: {"kind", "run_dir", "name", "model", "mode", "effort", "persona", "case_id"}."""
    from evals.evaluate import DEFAULT_MODE, DEFAULT_MODEL as EVAL_MODEL
    from evals.grader import parse_grader
    from evals.score_truth import DEFAULT_MODEL as TRUTH_MODEL
    kind = raw.get("kind") or EVALUATION
    if kind not in (EVALUATION, TRUTH):
        raise ValueError(f"job kind '{kind}': one of {EVALUATION}, {TRUTH}")
    opts = _evaluate_options(raw.get("evaluate_args") or [])
    job = {"kind": kind, "run_dir": str(Path(raw["run_dir"]).resolve()), "name": raw["name"],
           "model": raw.get("model") or (EVAL_MODEL if kind == EVALUATION else TRUTH_MODEL),
           "mode": raw.get("mode") or opts.get("mode") or (DEFAULT_MODE if kind == EVALUATION else None),
           "effort": raw.get("effort") or opts.get("effort"), "persona": raw.get("persona"),
           "case_id": raw.get("id") or Path(raw["run_dir"]).name,
           # the case (all its arms and graders) is never split across two batches
           "unit": raw.get("case") or raw.get("id") or Path(raw["run_dir"]).name}
    grader = parse_grader(job["model"], job["effort"])
    if grader.provider != "anthropic":
        raise ValueError(f"{job['case_id']} {job['name']}: grader {job['model']} is not Anthropic; Batch grading "
                         "takes anthropic:* graders only (grade it with evals/evaluate.py)")
    job["effort"] = grader.effort
    return job


def grading_custom_id(job: dict) -> str:
    """Stable per (kind, run folder name, grading name): the same grading always gets the same custom_id."""
    run = Path(job["run_dir"]).name
    return make_custom_id(job["name"], run, key=f"{job['kind']}|{run}|{job['name']}")


def _out_dir(job: dict) -> Path:
    return Path(job["run_dir"]) / "evaluation" / job["name"]


def is_done(job: dict) -> bool:
    return (_out_dir(job) / RESULT_FILE[job["kind"]]).exists()


def build(job: dict, reuse_pages: bool = False):
    """(GradingRequest, context) of a job, built by the synchronous code."""
    from evals.evaluate import build_grading_request
    from evals.grader import parse_grader
    from evals.score_truth import build_truth_check
    grader = parse_grader(job["model"], job["effort"])
    run_dir = Path(job["run_dir"])
    if job["kind"] == EVALUATION:
        request, checks, images = build_grading_request(run_dir, grader, _out_dir(job), job["mode"], reuse_pages)
        return request, {"checks": checks, "images": images}
    return build_truth_check(run_dir, grader, job.get("persona"))


def batch_request(custom_id: str, request) -> dict:
    """One Message Batches request: the synchronous call's params under a custom_id."""
    return {"custom_id": custom_id, "params": request.anthropic_params()}


def prepare(job: dict) -> Prepared:
    """Build the request and write the folder's prompt files as the synchronous path does (prompt.json, pages/).
    A folder without a result (an unsent or failed attempt) is cleared first, as run_matrix clears it."""
    from evals.evaluate import write_prompt
    from evals.score_truth import write_check_prompt
    from provenance import RunRecorder
    out_dir = _out_dir(job)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    request, ctx = build(job)
    recorder = RunRecorder(out_dir)
    if job["kind"] == EVALUATION:
        write_prompt(recorder, request, job["mode"], ctx["images"])
    else:
        write_check_prompt(recorder, request)
    line = batch_request(job["custom_id"], request)
    return Prepared(job["custom_id"], job, line, canonical_sha256(line["params"]),
                    len(json.dumps(line, ensure_ascii=False).encode("utf-8")), unit=job["unit"])


def _attempts_record(item: dict, outcome: Outcome, batch: dict) -> list[dict]:
    """The batch attempts of a grading for its record: earlier ones from the attempt log, then this one."""
    keys = ("attempt", "batch_id", "custom_id", "result_type", "submitted_at", "ended_at")
    earlier = [{k: a.get(k) for k in keys} for a in item.get("attempt_log", [])]
    return earlier + [{"attempt": item.get("attempt", 1), "batch_id": batch["batch_id"], "custom_id": outcome.custom_id,
                       "result_type": outcome.kind, "submitted_at": batch.get("submitted_at"),
                       "ended_at": batch.get("ended_at")}]


def _write(job: dict, request, ctx: dict, result, extra: dict, ledger: Path | None, recorded: float) -> dict:
    """Write a grading's folder as the synchronous path does (all its answered attempts in llm_calls.jsonl and
    raw_response.json; the record, or the failure as send() leaves it) and its ledger row."""
    from evals.evaluate import GradingFailed, finish_evaluation
    from evals.grader import log_attempts
    from evals.score_truth import TruthCheckFailed, finish_truth_check
    from provenance import RunRecorder
    recorder = RunRecorder(_out_dir(job))
    log_attempts(recorder, result, request, node=NODE[job["kind"]])
    label = _label(job)
    cost = {"cost_usd": result.cost_usd, "cost_list_usd": round(result.cost_usd / BATCH_PRICE_FACTOR, 6)}
    extra = {**extra, "cost_list_usd": cost["cost_list_usd"]}  # the grader record: billed and list price
    try:
        if job["kind"] == EVALUATION:
            finish_evaluation(Path(job["run_dir"]), job["name"], request, result, ctx["checks"], job["mode"], recorder,
                              extra)
        else:
            finish_truth_check(job["name"], request, result, ctx, recorder, extra)
    except (GradingFailed, TruthCheckFailed) as e:
        _ledger(ledger, job, f"{label}-failed", result.cost_usd - recorded)
        return {"status": result.status, "error": str(e)[:500], **cost}
    _ledger(ledger, job, label, result.cost_usd - recorded)
    return {"status": "ok", **cost, "message_id": result.attempts[-1].get("message_id")}


def apply(job: dict, item: dict, outcome: Outcome, batch: dict, ledger: Path | None = None) -> dict:
    """One batch result, under the synchronous retry rule (evals.grader.MAX_ATTEMPTS): an answer that does not parse
    is kept (in state.json, not yet in the folder) and asked again once in a follow-up batch; the folder is then written
    once, with both attempts, exactly as send() leaves it. A refusal is final. Unanswered results (errored, expired,
    canceled) are sent again by evals.batch's rule; nothing is written for them."""
    from evals.grader import MAX_ATTEMPTS, PARSE_ERROR, GradingResult, result_from_message
    attempt_no = item.get("attempt", 1)
    carry = item.get("carry") or {}
    earlier, recorded = carry.get("attempts", []), carry.get("ledger_usd", 0.0)
    extra = {"batch": {"custom_id": outcome.custom_id, "attempts": _attempts_record(item, outcome, batch)}}
    if outcome.kind != "succeeded":
        if earlier and attempt_no >= MAX_ATTEMPTS:  # the follow-up went unanswered: the answered attempt is final
            request, ctx = build(job, reuse_pages=True)
            last = GradingResult(PARSE_ERROR, None, earlier, request.grader.describe())
            written = _write(job, request, ctx, last, extra, ledger, recorded)
            return {**written, "error": f"{outcome.kind} on the follow-up; first attempt: {written.get('error')}"}
        return {"status": outcome.kind, "error": outcome.error}
    request, ctx = build(job, reuse_pages=True)
    if canonical_sha256(request.anthropic_params()) != item.get("request_sha256"):
        return {"status": "request_changed", **discard(job, outcome, ledger),
                "error": "the request rebuilt now differs from the one submitted (run folder or rubric changed)"}
    result = result_from_message(request, outcome.payload, batch_id=batch["batch_id"])
    for a in result.attempts:
        a.update(custom_id=outcome.custom_id, result_type=outcome.kind, attempt=attempt_no)
    if result.status == PARSE_ERROR and attempt_no < MAX_ATTEMPTS:
        _ledger(ledger, job, f"{_label(job)}-attempt{attempt_no}", result.cost_usd)
        return {"status": RETRY, "error": result.attempts[-1].get("parse_error"), "cost_usd": result.cost_usd,
                "carry": {"attempts": earlier + result.attempts, "ledger_usd": recorded + result.cost_usd}}
    combined = GradingResult(result.status, result.answer, earlier + result.attempts, result.grader)
    return _write(job, request, ctx, combined, extra, ledger, recorded)


def discard(job: dict, outcome: Outcome, ledger: Path | None = None) -> dict:
    """The billed cost of an answer that is not written (Job.discard), priced from its usage at the Batch rate."""
    from evals.grader import _usage_dict, anthropic_cost
    message = outcome.payload
    cost = anthropic_cost(getattr(message, "model", None) or job["model"],
                          _usage_dict(getattr(message, "usage", None)), batch=True) or 0.0
    _ledger(ledger, job, f"{_label(job)}-discarded", cost)
    return {"cost_usd": cost, "cost_list_usd": round(cost / BATCH_PRICE_FACTOR, 6)}


def _label(job: dict) -> str:
    return f"eval:{job['name']}" if job["kind"] == EVALUATION else "truth"


def _ledger(ledger: Path | None, job: dict, kind: str, cost: float) -> None:
    if ledger is not None:
        from evals.run_matrix import _record
        _record(ledger, job["case_id"], kind, cost)


def grading_job(ledger: Path | None = None) -> Job:
    return Job(custom_id=grading_custom_id, is_done=is_done, prepare=prepare,
               apply=lambda job, item, outcome, batch: apply(job, item, outcome, batch, ledger),
               discard=lambda job, outcome: discard(job, outcome, ledger))


# --------------------------------------------------------------------------------------------- the command
def jobs_from_args(args) -> list[dict]:
    if args.jobs:
        raw = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
        return [normalise_job(r) for r in raw]
    if not args.run_dirs:
        raise ValueError("give --jobs FILE or one or more run folders")
    kind = TRUTH if args.truth_check else EVALUATION
    name = args.name or ("truth1" if args.truth_check else "eval1")
    return [normalise_job({"kind": kind, "run_dir": d, "name": name, "model": args.model, "mode": args.mode,
                           "effort": args.effort, "persona": args.persona}) for d in args.run_dirs]


def default_parent(jobs: list[dict]) -> Path:
    parents = {Path(j["run_dir"]).parent for j in jobs}
    if len(parents) != 1:
        raise ValueError("the run folders have different parents: give --state DIR")
    return parents.pop()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="*", type=Path)
    p.add_argument("--jobs", help="grading plan from run_matrix --gradings-out")
    p.add_argument("--grader-model", "--model", dest="model", help="provider:model (anthropic:* only)")
    p.add_argument("--eval-name", "--name", dest="name", help="eval1 / truth1 (run folders on the command line)")
    p.add_argument("--grading-mode", dest="mode", help="document-only (default) or conversation")
    p.add_argument("--effort", help="Anthropic effort (default medium)")
    p.add_argument("--truth-check", action="store_true", help="the trap check (evals/score_truth.py) instead")
    p.add_argument("--persona", help="trap check: persona id/folder/file (default: the run's manifest label)")
    p.add_argument("--state", help="state folder (default <run folders' parent>/batches/grading)")
    p.add_argument("--ledger",
                   help="ledger.tsv to add costs to (default <run folders' parent>/ledger.tsv if it exists)")
    p.add_argument("--env-file", help=".env file with ANTHROPIC_API_KEY (read only when calling the API)")
    p.add_argument("--wait", action="store_true", help="poll until every batch has ended and is collected")
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--resubmit-failed", action="store_true", help="submit failed requests again")
    p.add_argument("--no-submit", action="store_true", help="only collect ended batches")
    p.add_argument("--status", action="store_true", help="print state.json's summary; no API call")
    p.add_argument("--dry-run", action="store_true", help="build the requests and print their counts; nothing written")
    p.add_argument("--adopt", action="append", help="KEY=BATCH_ID: a batch an interrupted run created")
    p.add_argument("--forget-pending", action="store_true", help="drop a batch an interrupted run did not create")
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    return p.parse_args(argv)


def dry_run(jobs: list[dict]) -> dict:
    """Every request built (page images rendered into a temporary folder): counts and sizes; nothing is written
    to the run folders and nothing is sent."""
    import tempfile
    from evals.evaluate import build_grading_request
    from evals.grader import parse_grader
    out = {"requests": 0, "done": 0, "bytes": 0, "by_model": {}}
    with tempfile.TemporaryDirectory(prefix="batch_grade_dry_") as tmp:
        for i, job in enumerate(jobs):
            if is_done(job):
                out["done"] += 1
                continue
            if job["kind"] == EVALUATION:
                request, _, _ = build_grading_request(Path(job["run_dir"]), parse_grader(job["model"], job["effort"]),
                                                      Path(tmp) / str(i), job["mode"])
            else:
                request, _ = build(job)
            size = len(json.dumps(batch_request(job["custom_id"], request), ensure_ascii=False).encode("utf-8"))
            out["requests"] += 1
            out["bytes"] += size
            out["by_model"][job["model"]] = out["by_model"].get(job["model"], 0) + 1
    return out


def main(argv=None, client=None, sleep=None) -> int:
    args = parse_args(argv)
    try:
        jobs = jobs_from_args(args)
        for j in jobs:
            j["custom_id"] = grading_custom_id(j)
        try:
            parent = default_parent(jobs)
        except ValueError:
            if not args.state:
                raise
            parent = None
        state_dir = Path(args.state) if args.state else parent / "batches" / "grading"
        ledger = Path(args.ledger) if args.ledger else (
            parent / "ledger.tsv" if parent is not None and (parent / "ledger.tsv").exists() else None)
        adopt = parse_adopt(args.adopt)
    except ValueError as e:
        sys.exit(str(e))
    from evals.run_matrix import session_done
    unfinished = sorted({str(j["run_dir"]) for j in jobs if not session_done(Path(j["run_dir"]))})
    if unfinished:  # nothing to grade yet: running, interrupted (credit, infrastructure) or never run
        print(f"not graded, the session is not finished: {', '.join(unfinished)}", file=sys.stderr)
        jobs = [j for j in jobs if str(j["run_dir"]) not in unfinished]
    if args.dry_run:
        print(json.dumps(dry_run(jobs), indent=2))
        return 0
    store = BatchStore(state_dir, STATE_KIND)
    if args.status:
        s = summary(store, [j["custom_id"] for j in jobs])
        print(json.dumps(s, indent=2) if args.json else format_summary(s))
        return 0
    if client is None:
        from dotenv import load_dotenv
        load_dotenv(args.env_file or REPO / ".env")
    from evals.credit import EXIT_OUT_OF_CREDIT, OutOfCredit
    try:
        s = run_round(store, AnthropicBatches(client), jobs, grading_job(ledger), submit=not args.no_submit,
                      resubmit_failed=args.resubmit_failed, wait=args.wait, poll_seconds=args.poll_seconds,
                      adopt=adopt, forget_pending=args.forget_pending,
                      log=lambda m: print(m, file=sys.stderr, flush=True), **({"sleep": sleep} if sleep else {}))
    except PendingSubmission as e:
        print(str(e), file=sys.stderr)
        return 2
    except OutOfCredit as e:
        print(f"out of credit: nothing more was submitted ({e})", file=sys.stderr)
        print(format_summary(summary(store, [j["custom_id"] for j in jobs])))
        return EXIT_OUT_OF_CREDIT
    print(json.dumps(s, indent=2) if args.json else format_summary(s))
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
