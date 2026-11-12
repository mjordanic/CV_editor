"""Durable submit / resume / collect for the Batch APIs (Anthropic Message Batches, OpenAI /v1/batches).

Shared by evals/batch_grade.py (the gradings, on Anthropic) and evals/batch_sp.py (the single-prompt writer calls,
on OpenAI). One folder holds one batch job's state:

    STATE_DIR/state.json   every request ("item", keyed by its custom_id) and every submitted batch
    STATE_DIR/*.jsonl      OpenAI input files as uploaded (batch_sp.py)

A round (run_round) does, in order:
  1. stop if an earlier run died while submitting (a batch entry without a batch id): it may or may not exist at the
     provider. The adapter looks for it (OpenAI: by the metadata it was created with); otherwise the run says how to
     go on (--adopt KEY=BATCH_ID if the provider shows it, --forget-pending if not).
  2. collect: every submitted batch that has ended has its results applied to its items (written to their run
     folders) exactly once; a batch still running is left alone.
  3. prepare: every job that is neither finished on disk nor in flight has its request built, with its hash; so has
     every item due for its follow-up attempt (below).
  4. submit: the requests go out in batches within the provider's limits, a unit (a case: all arms and all graders of
     it) never split across two batches. Each batch is written to state.json *before* the API call (status
     "submitting") and again with its id right after, so a re-run collects and never submits a request twice.
  5. with wait: poll, collect and submit follow-ups until nothing is in flight.

The retry rule is the synchronous path's (MAX_ATTEMPTS = 2 submissions per item, both recorded): a request the
provider did not answer (errored, expired, canceled; not billed) and, where the job asks for it (gradings), an answer
that did not parse go out once more in a follow-up batch automatically. A refusal is final and never re-routed.
Every attempt is in the item's attempt_log (batch id, custom id, result type, submitted / ended / collected times,
error, cost).

Item statuses: submitted (in flight); retry (waiting for its follow-up batch); ok (written to its run folder); done
(already finished on disk, e.g. by the synchronous path; never overwritten); and the final failures, each listed by
the round: errored, expired, canceled (still unanswered after both attempts; --resubmit-failed sends them again,
the only statuses it sends), parse_error and refused (answered and written as the synchronous path writes a failed
grading), request_changed (the request rebuilt at collection is not the one submitted: inputs or code changed in
between; nothing written), session_error (batch_sp: the session on the reply failed), collect_error (writing the
result raised; see its error). An answered request is never sent again: request_changed, session_error and
collect_error are settled by hand (see the error), and the cost of every answered reply is in the attempt log and,
through the job's apply, in the ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

IN_FLIGHT = "submitted"
SUBMITTING = "submitting"
RETRY = "retry"
UNANSWERED = ("errored", "expired", "canceled")  # the provider sent nothing to the model: not billed
FAILED = (*UNANSWERED, "parse_error", "refused", "request_changed", "session_error", "collect_error")
RESUBMITTABLE = UNANSWERED  # by --resubmit-failed (never an answered request)
MAX_ATTEMPTS = 2  # as evals.grader.MAX_ATTEMPTS: one follow-up attempt
CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")  # Anthropic's rule; OpenAI accepts any unique string


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_sha256(data: Any) -> str:
    """Hash of a JSON request (sorted keys): the same request always hashes the same."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def make_custom_id(*parts: str, key: str) -> str:
    """A stable custom_id: the readable parts (sanitised, shortened) and a hash of `key`, within
    ^[a-zA-Z0-9_-]{1,64}$."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", "-".join(p for p in parts if p)).strip("_-")[:64 - 11]
    return f"{readable}-{digest}" if readable else digest


class PendingSubmission(RuntimeError):
    """An earlier run stopped while submitting a batch: it may exist at the provider."""


class NotSubmitted(Exception):
    """Raised by an adapter's submit for an error known to leave no batch at the provider: one raised before the
    request was sent, or the provider's own refusal (an HTTP 4xx answer). Any other error, or an interrupt, may come
    after the provider created the batch."""


def provider_refused(error: BaseException | None) -> bool:
    """Whether the provider answered the request with an HTTP 4xx error (it created nothing): the error or one of its
    causes carries such a status_code."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        code = getattr(error, "status_code", None)
        if isinstance(code, int) and 400 <= code < 500:
            return True
        error = error.__cause__ or error.__context__
    return False


class BatchStore:
    """STATE_DIR/state.json, rewritten atomically (temp file + rename) after every change."""

    def __init__(self, state_dir: str | Path, kind: str):
        self.dir = Path(state_dir)
        self.path = self.dir / "state.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("kind") != kind:
                raise ValueError(f"{self.path} holds a '{self.data.get('kind')}' batch job, not '{kind}'")
        else:
            self.data = {"kind": kind, "created_at": now(), "store_id": hashlib.sha256(
                f"{self.dir.resolve()}{time.time()}".encode()).hexdigest()[:12], "items": {}, "batches": []}

    @property
    def items(self) -> dict:
        return self.data["items"]

    @property
    def batches(self) -> list:
        return self.data["batches"]

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def batch(self, key: str) -> dict:
        return next(b for b in self.batches if b["key"] == key)

    def next_key(self) -> str:
        return f"b{len(self.batches) + 1:03d}"

    def set_status(self, cid: str, status: str, **extra) -> None:
        item = self.items[cid]
        item.update(status=status, updated_at=now(), **extra)


@dataclass
class Prepared:
    """One request ready to submit: `request` is what the adapter sends for it (Anthropic: {"custom_id", "params"};
    OpenAI: the JSONL line text), `sha256` the hash checked again at collection, `group` the batch it may share
    (OpenAI: one model per input file), `size` its bytes, `unit` what must stay in one batch (a case)."""
    custom_id: str
    job: dict
    request: Any
    sha256: str
    size: int
    group: str = ""
    unit: str = ""


@dataclass
class Outcome:
    """One result of an ended batch: kind succeeded | errored | expired | canceled; `payload` the provider's reply
    (Anthropic: the Message; OpenAI: the output line's "response"), `error` what went wrong."""
    custom_id: str
    kind: str
    payload: Any = None
    error: Any = None


class Adapter:
    """What run_round needs from a provider."""
    provider = ""
    max_requests = 10_000
    max_bytes = 200_000_000

    def submit(self, key: str, store_id: str, prepared: list[Prepared]) -> str:  # returns the batch id
        raise NotImplementedError

    def find(self, key: str, store_id: str) -> str | None:  # a batch created by a run that died before recording it
        return None

    def status(self, batch_id: str) -> tuple[bool, dict]:  # (ended, provider status for the log)
        raise NotImplementedError

    def results(self, batch_id: str) -> Iterable[Outcome]:
        raise NotImplementedError


@dataclass
class Job:
    """What a round does with one kind of request (a grading, a single-prompt session)."""
    custom_id: Callable[[dict], str]
    is_done: Callable[[dict], bool]  # finished on disk (never overwritten)
    prepare: Callable[[dict], Prepared]  # build the request (may write the folder's prompt files)
    # (job, item, outcome, batch entry) -> {"status", ...}; item["attempt"] is this attempt's number (1, 2, ...) and
    # item.get("carry") what an earlier attempt of it returned under "carry"
    apply: Callable[[dict, dict, Outcome, dict], dict]
    # e.g. a cost cap; checked before preparing a new request, with the jobs prepared in this pass but not yet submitted
    allow: Callable[[dict, list[dict]], bool] = lambda job, pending: True
    # (job, succeeded outcome) -> {"cost_usd", "cost_list_usd"}: records in the ledger the billed cost of an
    # answer that is not written (its item already finished on disk, its request changed, its session failed)
    discard: Callable[[dict, Outcome], dict] = lambda job, outcome: {}
    max_attempts: int = MAX_ATTEMPTS


def resolve_pending(store: BatchStore, adapter: Adapter, adopt: dict | None = None, forget: bool = False,
                    log: Callable = print) -> None:
    """Settle batch entries an earlier run left in "submitting" (step 1)."""
    adopt = adopt or {}
    for b in [b for b in store.batches if b["status"] == SUBMITTING]:
        found = adopt.get(b["key"]) or adapter.find(b["key"], store.data["store_id"])
        if found:
            b.update(status=IN_FLIGHT, batch_id=found, submitted_at=b.get("submitted_at") or now(), adopted=True)
            for cid in b["custom_ids"]:
                store.set_status(cid, IN_FLIGHT, batch_id=found)
            log(f"batch {b['key']}: adopted {found}")
        elif forget:
            b["status"] = "forgotten"
            for cid in b["custom_ids"]:
                store.items.pop(cid, None)
            log(f"batch {b['key']}: forgotten; its {len(b['custom_ids'])} requests will be prepared again")
        else:
            raise PendingSubmission(
                f"batch {b['key']} ({len(b['custom_ids'])} requests) was being submitted to {adapter.provider} when "
                f"an earlier run stopped, and it is not known whether it was created. If the provider's console shows "
                f"it, re-run with --adopt {b['key']}=<batch id>; if not, re-run with --forget-pending.")
        store.save()


def collect(store: BatchStore, adapter: Adapter, jobs_by_id: dict, job: Job, log: Callable = print) -> None:
    """Apply the results of every ended batch (step 2). An item is applied once: re-collecting a batch (after a
    run died half-way) skips the items already settled."""
    for b in store.batches:
        if b["status"] != IN_FLIGHT:
            continue
        ended, info = adapter.status(b["batch_id"])
        b["provider_status"] = info
        if not ended:
            store.save()
            log(f"batch {b['key']} ({b['batch_id']}): in progress {info}")
            continue
        b.setdefault("ended_seen_at", now())
        b.setdefault("ended_at", info.get("ended_at") or b["ended_seen_at"])
        store.save()
        for outcome in adapter.results(b["batch_id"]):
            item = store.items.get(outcome.custom_id)
            if item is None or item["status"] != IN_FLIGHT or item.get("batch_id") != b["batch_id"]:
                continue
            spec = jobs_by_id.get(outcome.custom_id) or item["job"]
            if job.is_done(spec):
                cost = job.discard(spec, outcome) if outcome.kind == "succeeded" else {}
                _log_attempt(item, outcome, b, "done", cost)
                store.set_status(outcome.custom_id, "done", note="finished on disk before the batch was collected",
                                 **cost)
            else:
                _settle(store, job, spec, item, outcome, b)
            store.save()
        missing = [c for c in b["custom_ids"] if store.items.get(c, {}).get("status") == IN_FLIGHT
                   and store.items[c].get("batch_id") == b["batch_id"]]
        for cid in missing:  # an ended batch with no result for a request: not answered
            item = store.items[cid]
            _settle(store, job, jobs_by_id.get(cid) or item["job"], item,
                    Outcome(cid, "errored", error="no result in the ended batch"), b)
        b.update(status="collected", collected_at=now())
        store.save()
        log(f"batch {b['key']} ({b['batch_id']}): collected")


def _settle(store: BatchStore, job: Job, spec: dict, item: dict, outcome: Outcome, b: dict) -> None:
    """Apply one outcome, log the attempt, and decide: final status, or one more attempt (the retry rule)."""
    try:
        result = job.apply(spec, item, outcome, b)
    except Exception as e:  # noqa: BLE001 - one result that cannot be written must not block the rest
        result = {"status": "collect_error", "error": f"{type(e).__name__}: {e}"[:500]}
        if outcome.kind == "succeeded":
            try:
                result.update(job.discard(spec, outcome))
            except Exception as d:  # noqa: BLE001
                result["error"] += f"; not priced: {type(d).__name__}: {d}"[:200]
    status = result.pop("status")
    _log_attempt(item, outcome, b, status, result)
    if status == RETRY or (status in UNANSWERED and item.get("attempt", 1) < job.max_attempts):
        status = RETRY  # a follow-up batch sends it once more
    store.set_status(outcome.custom_id, status, **result)


def _log_attempt(item: dict, outcome: Outcome, b: dict, status: str, result: dict) -> None:
    item.setdefault("attempt_log", []).append({
        "attempt": item.get("attempt", 1), "batch_id": b["batch_id"], "custom_id": outcome.custom_id,
        "result_type": outcome.kind, "status": status, "submitted_at": b.get("submitted_at"),
        "ended_at": b.get("ended_at"), "collected_at": now(),
        **{k: result[k] for k in ("error", "cost_usd", "cost_list_usd") if result.get(k) is not None}})


def _submit(store: BatchStore, adapter: Adapter, group: list[Prepared], log: Callable) -> None:
    """Submit one batch (step 4): recorded as "submitting" before the call and with its id right after."""
    key = store.next_key()
    entry = {"key": key, "provider": adapter.provider, "status": SUBMITTING, "batch_id": None,
             "custom_ids": [p.custom_id for p in group], "group": group[0].group, "created_at": now()}
    store.batches.append(entry)
    previous = {p.custom_id: store.items.get(p.custom_id) for p in group}
    for p in group:
        before = previous[p.custom_id] or {}
        carried = {k: before[k] for k in ("attempt_log", "carry") if k in before}
        store.items[p.custom_id] = {"job": p.job, "status": SUBMITTING, "request_sha256": p.sha256, "batch_id": None,
                                    "attempt": len(before.get("attempt_log", [])) + 1, **carried,
                                    "updated_at": now()}
    store.save()  # before the call: a run that dies inside it leaves a "submitting" entry, never a duplicate
    try:
        batch_id = adapter.submit(key, store.data["store_id"], group)
    except BaseException as e:
        if not (isinstance(e, NotSubmitted) or provider_refused(e)):
            raise  # an interrupt, a timeout or a lost connection: the batch may exist; the entry stays "submitting"
        store.batches.remove(entry)  # the request never reached the provider, or it refused it: nothing was created
        for p in group:
            if previous[p.custom_id] is None:
                store.items.pop(p.custom_id)
            else:
                store.items[p.custom_id] = previous[p.custom_id]
        store.save()
        raise
    entry.update(status=IN_FLIGHT, batch_id=batch_id, submitted_at=now())
    for p in group:
        store.set_status(p.custom_id, IN_FLIGHT, batch_id=batch_id)
    store.save()
    log(f"batch {key}: submitted {len(group)} requests as {batch_id}")


def run_round(store: BatchStore, adapter: Adapter, jobs: list[dict], job: Job, *, submit: bool = True,
              resubmit_failed: bool = False, wait: bool = False, poll_seconds: float = 60.0,
              adopt: dict | None = None, forget_pending: bool = False, log: Callable = print,
              sleep: Callable = time.sleep) -> dict:
    """One submit / collect round over `jobs` (see the module docstring). Returns the summary (summary())."""
    for j in jobs:
        j.setdefault("custom_id", job.custom_id(j))
        if not CUSTOM_ID.match(j["custom_id"]):
            raise ValueError(f"custom_id {j['custom_id']!r} does not match {CUSTOM_ID.pattern}")
    by_id = {j["custom_id"]: j for j in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("two jobs share a custom_id")
    resolve_pending(store, adapter, adopt, forget_pending, log)
    while True:
        collect(store, adapter, by_id, job, log)
        if submit:
            _submit_new(store, adapter, jobs, job, resubmit_failed, log)
        if not wait or not any(b["status"] == IN_FLIGHT for b in store.batches):
            break
        sleep(poll_seconds)
    return summary(store, [j["custom_id"] for j in jobs])


def _due(store: BatchStore, j: dict, resubmit_failed: bool) -> bool:
    """Whether a job's request goes into this round's submission."""
    item = store.items.get(j["custom_id"])
    if item is None:
        return True
    return item["status"] == RETRY or (resubmit_failed and item["status"] in RESUBMITTABLE)


def _submit_new(store: BatchStore, adapter: Adapter, jobs: list[dict], job: Job, resubmit_failed: bool,
                log: Callable) -> None:
    """Steps 3-4. Requests are submitted as soon as a batch is full (a grading with page images is about 1 MB), so only
    one batch per group is held in memory; a unit (a case) always goes into one batch whole."""
    chunks: dict[str, list[Prepared]] = {}
    unit: list[Prepared] = []

    def place(members: list[Prepared]) -> None:
        if not members:
            return
        current = chunks.setdefault(members[0].group, [])
        if current and (len(current) + len(members) > adapter.max_requests
                        or sum(p.size for p in current) + sum(p.size for p in members) > adapter.max_bytes):
            _submit(store, adapter, current, log)
            current.clear()
        current.extend(members)

    for j in jobs:
        if not _due(store, j, resubmit_failed):
            continue
        if job.is_done(j):
            store.items[j["custom_id"]] = {**store.items.get(j["custom_id"], {}), "job": j, "status": "done",
                                           "updated_at": now()}
            store.save()
            continue
        if not job.allow(j, [p.job for p in [*unit, *(q for c in chunks.values() for q in c)]]):
            continue
        p = job.prepare(j)
        if unit and (p.unit != unit[0].unit or p.group != unit[0].group):
            place(unit)
            unit = []
        unit.append(p)
    place(unit)
    for current in chunks.values():
        if current:
            _submit(store, adapter, current, log)


def summary(store: BatchStore, custom_ids: list[str] | None = None) -> dict:
    """Counts by status, the failed items (with their error) and the batches still in flight."""
    ids = custom_ids if custom_ids is not None else list(store.items)
    counts: dict[str, int] = {}
    failed = []
    for cid in ids:
        item = store.items.get(cid)
        status = item["status"] if item else "not submitted"
        counts[status] = counts.get(status, 0) + 1
        if status in FAILED:
            failed.append({"custom_id": cid, "status": status, "error": item.get("error"),
                           **{k: item["job"].get(k) for k in ("run_dir", "name", "kind", "model") if k in item["job"]}})
    in_flight = [{"key": b["key"], "batch_id": b["batch_id"], "requests": len(b["custom_ids"]),
                  "provider_status": b.get("provider_status")} for b in store.batches if b["status"] == IN_FLIGHT]
    return {"state": str(store.path), "counts": dict(sorted(counts.items())), "failed": failed, "in_flight": in_flight}


def parse_adopt(values: list[str] | None) -> dict:
    """--adopt KEY=BATCH_ID (repeatable) -> {key: batch_id}."""
    out = {}
    for v in values or []:
        key, sep, batch_id = v.partition("=")
        if not sep or not key or not batch_id:
            raise ValueError(f"--adopt {v!r}: use KEY=BATCH_ID (the key is in state.json, e.g. b003)")
        out[key] = batch_id
    return out


def format_summary(s: dict) -> str:
    lines = [f"state: {s['state']}", "items: " + (", ".join(f"{k} {v}" for k, v in s["counts"].items()) or "none")]
    if s["counts"].get(RETRY):
        lines.append(f"{s['counts'][RETRY]} waiting for their follow-up attempt: run the same command again")
    for b in s["in_flight"]:
        lines.append(f"in flight: {b['key']} {b['batch_id']} ({b['requests']} requests) "
                     f"{b.get('provider_status') or ''}")
    if s["failed"]:
        lines.append("failed (unanswered ones can be sent again with --resubmit-failed; the rest by hand):")
        lines += [f"  {f['custom_id']}: {f['status']}: {str(f.get('error') or '')[:200]}" for f in s["failed"]]
    if s["in_flight"]:
        lines.append("run the same command again to collect (or add --wait)")
    return "\n".join(lines)
