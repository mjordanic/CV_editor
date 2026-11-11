"""The single-prompt (SP) arms' writer calls through OpenAI's Batch API (/v1/batches), for run_matrix --sp-batch.

A single-prompt session is one model call (evals/single_prompt.py). Here that call goes into a JSONL batch file at
half price instead of being made synchronously:

  submit    for each SP case whose session is not finished: single_prompt.batch_request() builds the case's prompt
            and captures, offline, the exact HTTP request the synchronous call sends (POST /v1/chat/completions, the
            same model, messages, structured-output schema and settings). The batch line is
            {"custom_id", "method", "url", "body"} with that body text byte for byte. One input file per model
            (the API takes one model per file), uploaded with purpose "batch" and created with a 24h window and
            metadata naming this state folder and batch key (so a run that died mid-submit finds it again).
  collect   when a batch has ended, each answered line is saved to STATE_DIR/results/<custom_id>.json and the case's
            session is run as usual (`python -m evals.single_prompt <the same arguments> --batch-result FILE`, its
            console log in RUNS_DIR/<case id>.out): the reply is replayed through the same LangChain call on a mock
            transport (nothing is sent), so parsing, PDFs, the no-model fallback layout, manifest, transcript and
            records are those of a synchronous session. The manifest says "batch" (batch id, custom id); the call is
            priced at the Batch rate; the ledger gets the session's "app" row as before.

State: RUNS_DIR/batches/sp_openai/state.json and the uploaded input files (evals/batch.py: written before every
submission, so re-running run_matrix collects and never submits a case twice). A request the API did not answer
(failed line, expired, cancelled, batch validation failure) is sent once more in a follow-up batch (evals/batch.py's
retry rule), then listed; --batch-resubmit-failed sends it again after that (only such unanswered requests). A request
whose inputs changed between submission and collection is not run (request_changed; its billed reply is in the ledger,
settled by hand). An answer that does not parse is a finished session with
failure "parse_error", never re-asked, as in the synchronous path (one call per case).

Times and cost: the manifest's "batch" block has the batch id, custom id, attempt, and submitted / ended / collected
times (run_matrix submits the SP batch in the same run, before the app sessions of the stage start); "cost_usd" is
the billed (Batch) cost and "cost_list_usd" the list price of the same call, for the cost comparison with the app.

Caps: a case is submitted only while the ledger's spend plus the Batch-price estimate of every request in flight
(run_matrix.sp_session_usd x BATCH_PRICE_FACTOR) stays within --total-cap and the stage's cap.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from evals.batch import (FAILED, IN_FLIGHT, RETRY, SUBMITTING, UNANSWERED, Adapter, BatchStore, Job, Outcome, Prepared,
                         make_custom_id, now, run_round)
from provenance import BATCH_PRICE_FACTOR

REPO = Path(__file__).resolve().parent.parent
ENDPOINT = "/v1/chat/completions"
STATE_KIND = "single_prompt"
ENDED = ("completed", "expired", "cancelled", "failed")


class OpenAIBatches(Adapter):
    """/v1/files + /v1/batches through the official SDK (`client` replaces openai.OpenAI() in tests)."""
    provider = "openai"
    max_requests = 10_000  # the API allows 50,000 requests or 200 MB per input file
    max_bytes = 180_000_000

    def __init__(self, state_dir: Path, client=None):
        self.state_dir = Path(state_dir)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def submit(self, key: str, store_id: str, prepared: list[Prepared]) -> str:
        from evals.credit import OutOfCredit, is_out_of_credit
        path = self.state_dir / f"{key}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(p.request + "\n" for p in prepared), encoding="utf-8")
        try:
            with open(path, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="batch")
            batch = self.client.batches.create(input_file_id=uploaded.id, endpoint=ENDPOINT, completion_window="24h",
                                               metadata={"cv_eval_store": store_id, "cv_eval_batch": key})
        except Exception as e:
            if is_out_of_credit(e):
                raise OutOfCredit(f"{type(e).__name__}: {e}"[:500]) from e
            raise
        return batch.id

    def find(self, key: str, store_id: str) -> str | None:
        for batch in self.client.batches.list(limit=100):
            meta = getattr(batch, "metadata", None) or {}
            if meta.get("cv_eval_store") == store_id and meta.get("cv_eval_batch") == key:
                return batch.id
        return None

    def status(self, batch_id: str) -> tuple[bool, dict]:
        batch = self.client.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        counts = counts.model_dump() if hasattr(counts, "model_dump") else counts
        errors = getattr(batch, "errors", None)
        errors = errors.model_dump(mode="json") if hasattr(errors, "model_dump") else errors
        ended_at = next((getattr(batch, k, None) for k in ("completed_at", "expired_at", "cancelled_at", "failed_at")
                         if getattr(batch, k, None)), None)
        if isinstance(ended_at, (int, float)):
            from datetime import datetime, timezone
            ended_at = datetime.fromtimestamp(ended_at, timezone.utc).isoformat(timespec="seconds")
        return batch.status in ENDED, {"status": batch.status, "request_counts": counts,
                                       **({"ended_at": ended_at} if ended_at else {}),
                                       **({"errors": errors} if errors else {})}

    def _lines(self, file_id: str | None) -> list[dict]:
        if not file_id:
            return []
        return [json.loads(line) for line in self.client.files.content(file_id).text.splitlines() if line.strip()]

    def results(self, batch_id: str):
        batch = self.client.batches.retrieve(batch_id)
        for line in self._lines(getattr(batch, "output_file_id", None)):
            response, error = line.get("response") or {}, line.get("error")
            if response.get("status_code") == 200 and not error:
                yield Outcome(line["custom_id"], "succeeded", response)
            else:
                yield Outcome(line["custom_id"], "errored", error=error or (response.get("body") or {}).get("error")
                              or f"HTTP {response.get('status_code')}")
        for line in self._lines(getattr(batch, "error_file_id", None)):
            error = line.get("error") or (((line.get("response") or {}).get("body") or {}).get("error"))
            code = str((error or {}).get("code") if isinstance(error, dict) else error or "")
            kind = "expired" if "expired" in code else "canceled" if "cancel" in code else "errored"
            yield Outcome(line["custom_id"], kind, error=error)


# --------------------------------------------------------------------------------------------- the SP sessions
def _session_done(run_dir: Path) -> bool:
    """As run_matrix: a session is finished once its manifest has a final status (not running, not interrupted)."""
    from evals.run_matrix import _manifest_status
    manifest = run_dir / "manifest.json"
    status = _manifest_status(manifest) if manifest.exists() else None
    return not (status is None or status == "running" or str(status).startswith("interrupted"))


def sp_custom_id(job: dict) -> str:
    return make_custom_id("sp", job["case_id"], key=f"sp|{job['case_id']}")


def sp_jobs(cases: list[dict], opts) -> list[dict]:
    """One job per SP case: its run folder and the exact arguments run_matrix gives evals/single_prompt.py."""
    from evals.run_matrix import session_command
    out = Path(opts.out).resolve()
    jobs = []
    for case in cases:
        run_dir = out / case["id"]
        cmd = session_command(case, run_dir, opts.env_file, opts.cost_cap, getattr(opts, "app_labels", None))
        jobs.append({"case_id": case["id"], "run_dir": str(run_dir), "argv": cmd[3:], "model": case["model"],
                     "stage": case.get("stage")})
    return jobs


def prepare(job: dict) -> Prepared:
    from evals.single_prompt import batch_request
    req = batch_request(job["argv"])
    if req["url"] != ENDPOINT or req["method"] != "POST":
        raise RuntimeError(f"the single-prompt call goes to {req['method']} {req['url']}, not POST {ENDPOINT}")
    line = (f'{{"custom_id": {json.dumps(job["custom_id"])}, "method": "POST", "url": {json.dumps(ENDPOINT)}, '
            f'"body": {req["body"]}}}')
    return Prepared(job["custom_id"], job, line, req["request_sha256"], len(line.encode("utf-8")), group=req["model"],
                    unit=job["case_id"])


def apply(job: dict, item: dict, outcome: Outcome, batch: dict, state_dir: Path, ledger: Path) -> dict:
    """Run the case's session on the batch reply (a subprocess, as run_matrix runs sessions)."""
    from evals.run_matrix import _record
    from evals.single_prompt import batch_request
    if outcome.kind != "succeeded":
        return {"status": outcome.kind, "error": outcome.error}
    if batch_request(job["argv"])["request_sha256"] != item.get("request_sha256"):
        return {"status": "request_changed", **discard(job, outcome, ledger),
                "error": "the case's request rebuilt now differs from the one submitted (inputs or code changed)"}
    result_file = state_dir / "results" / f"{job['custom_id']}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps({
        "provider": "openai", "batch_id": batch["batch_id"], "custom_id": job["custom_id"], "url": ENDPOINT,
        "request_sha256": item["request_sha256"], "attempt": item.get("attempt", 1),
        "submitted_at": batch.get("submitted_at"), "ended_at": batch.get("ended_at"), "collected_at": now(),
        "response": outcome.payload}, indent=2, ensure_ascii=False), encoding="utf-8")
    run_dir = Path(job["run_dir"])
    shutil.rmtree(run_dir, ignore_errors=True)  # an unfinished session is redone from scratch, as in run_matrix
    log = run_dir.parent / f"{job['case_id']}.out"
    with open(log, "w") as f:
        subprocess.run([sys.executable, "-m", "evals.single_prompt", *job["argv"], "--batch-result", str(result_file)],
                       cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return {"status": "session_error", **discard(job, outcome, ledger),
                "error": f"the session wrote no manifest (see {log})"}
    data = json.loads(manifest.read_text())
    _record(ledger, job["case_id"], "app", data.get("cost_usd") or 0)
    if not _session_done(run_dir):
        return {"status": "session_error", "error": f"session status {data.get('status')} (see {log})",
                "cost_usd": data.get("cost_usd")}
    return {"status": "ok", "cost_usd": data.get("cost_usd"), "cost_list_usd": data.get("cost_list_usd"),
            "session_status": data.get("status")}


def discard(job: dict, outcome: Outcome, ledger: Path) -> dict:
    """The billed cost of an answer whose session is not written (Job.discard), priced from the reply's usage at the
    Batch rate; its ledger row is "app-discarded"."""
    from evals.run_matrix import _record
    from provenance import cost_usd
    body = (outcome.payload or {}).get("body") or {}
    usage = body.get("usage") or {}
    list_cost = cost_usd(body.get("model") or job["model"], usage.get("prompt_tokens", 0),
                         usage.get("completion_tokens", 0),
                         (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0.0
    cost = list_cost * BATCH_PRICE_FACTOR
    _record(ledger, job["case_id"], "app-discarded", cost)
    return {"cost_usd": round(cost, 6), "cost_list_usd": round(list_cost, 6)}


def batch_estimate_usd(model: str) -> float:
    from evals.run_matrix import sp_session_usd
    return sp_session_usd(model) * BATCH_PRICE_FACTOR


def cap_check(store: BatchStore, opts, ledger: Path):
    """allow(job, pending): submit only while spend + the estimates of the requests in flight now (state.json) and of
    those prepared in this pass stay within the total and stage caps."""
    from evals.run_matrix import spent
    stage_caps = getattr(opts, "stage_caps", None) or {}

    def allow(job: dict, pending: list[dict]) -> bool:
        committed = [*(i["job"] for i in store.items.values() if i["status"] in (IN_FLIGHT, SUBMITTING)), *pending,
                     job]
        if spent(ledger) + sum(batch_estimate_usd(j["model"]) for j in committed) > opts.total_cap:
            return False
        stage = job.get("stage")
        return stage not in stage_caps or (
            spent(ledger, opts.stage_ids[stage])
            + sum(batch_estimate_usd(j["model"]) for j in committed if j.get("stage") == stage) <= stage_caps[stage])
    return allow


def state_dir_for(opts) -> Path:
    return Path(opts.out).resolve() / "batches" / "sp_openai"


def sp_batch_round(cases: list[dict], opts, client=None, sleep=None, log=print) -> dict:
    """Submit / collect the SP cases' writer calls (one round; with opts.batch_wait until all are collected).
    Returns evals.batch.summary() plus {"status": {case id: item status}}."""
    from evals.batch import parse_adopt
    state_dir = state_dir_for(opts)
    ledger = Path(opts.out) / "ledger.tsv"
    store = BatchStore(state_dir, STATE_KIND)
    jobs = sp_jobs(cases, opts)
    job = Job(custom_id=sp_custom_id, is_done=lambda j: _session_done(Path(j["run_dir"])), prepare=prepare,
              apply=lambda j, item, outcome, batch: apply(j, item, outcome, batch, state_dir, ledger),
              allow=cap_check(store, opts, ledger),
              discard=lambda j, outcome: discard(j, outcome, ledger))
    s = run_round(store, OpenAIBatches(state_dir, client), jobs, job,
                  resubmit_failed=getattr(opts, "batch_resubmit_failed", False),
                  wait=getattr(opts, "batch_wait", False), poll_seconds=getattr(opts, "batch_poll_seconds", 60.0),
                  adopt=parse_adopt(getattr(opts, "batch_adopt", None)),
                  forget_pending=getattr(opts, "batch_forget_pending", False), log=log,
                  **({"sleep": sleep} if sleep else {}))
    s["status"] = {j["case_id"]: (store.items.get(j["custom_id"]) or {}).get("status", "not submitted") for j in jobs}
    return s


def pending_note(status: str | None) -> str:
    """run_matrix's line for an SP case whose session the batch has not produced yet."""
    if status in UNANSWERED:
        return f"OpenAI batch request {status} (re-submit with --batch-resubmit-failed)"
    if status in FAILED:
        return f"OpenAI batch request {status} (answered; settle by hand, see the batch state)"
    if status == IN_FLIGHT:
        return "session in the OpenAI batch (run again to collect, or add --batch-wait)"
    if status == RETRY:
        return "OpenAI batch request unanswered, sent again on the next run (one follow-up attempt)"
    return "session not submitted to the OpenAI batch (cap reached)"
