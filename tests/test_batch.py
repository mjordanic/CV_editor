"""Batch grading (evals/batch_grade.py, Anthropic Message Batches) and batched single-prompt sessions
(evals/batch_sp.py, OpenAI /v1/batches), offline.

Nothing leaves the machine: the Anthropic and OpenAI clients are fakes (or the real SDKs on a mock HTTP transport),
no .env is read (clients are injected, an empty --env-file is passed), pdflatex is stubbed.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_eval_diversity import _evaluation
from tests.test_grader import _case_run, _Client, _message

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# --- fakes -------------------------------------------------------------------------------------------------
class FakeAnthropic:
    """messages.batches.create / retrieve / results; finish() ends a batch with the given results."""

    def __init__(self):
        self.created, self.ended, self.results_by_id = [], set(), {}
        fake = self

        class _Batches:
            def create(self, requests):
                fake.created.append(json.loads(json.dumps(requests)))
                return SimpleNamespace(id=f"msgbatch_{len(fake.created)}")

            def retrieve(self, batch_id):
                return SimpleNamespace(processing_status="ended" if batch_id in fake.ended else "in_progress",
                                       request_counts=None)

            def results(self, batch_id):
                return iter(fake.results_by_id[batch_id])

        self.messages = SimpleNamespace(batches=_Batches())

    def finish(self, batch_id: str, outcomes: dict, order=None):
        """outcomes: {custom_id: Message | "errored" | "expired" | "canceled"}."""
        rows = []
        for cid in order or outcomes:
            o = outcomes[cid]
            if isinstance(o, str):
                rows.append(SimpleNamespace(custom_id=cid, result=SimpleNamespace(
                    type=o, error={"type": "error", "error": {"type": "api_error", "message": "boom"}} if o == "errored" else None)))
            else:
                rows.append(SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=o)))
        self.results_by_id[batch_id] = rows
        self.ended.add(batch_id)


def _ids(fake, n=-1):
    return [r["custom_id"] for r in fake.created[n]]


def _grade(argv, fake):
    from evals import batch_grade
    return batch_grade.main(argv, client=fake, sleep=lambda s: None)


def _state(path: Path) -> dict:
    return json.loads(path.read_text())


# --- Anthropic: request equivalence --------------------------------------------------------------------------
def test_a_batch_request_is_the_synchronous_request_body(tmp_path):
    """Both through the real Anthropic SDK on a mock transport: the batch's `params` are the JSON body the
    synchronous path streams (without `stream`), page images included."""
    import anthropic
    import httpx2
    from anthropic import DefaultHttpxClient

    from evals.batch_grade import AnthropicBatches, Prepared, batch_request
    from evals.evaluate import build_grading_request
    from evals.grader import send
    run = _case_run(tmp_path, "sp")
    request, _, _ = build_grading_request(run, "anthropic:claude-sonnet-5-5", tmp_path / "g", "document-only")
    assert request.images  # the CV and letter pages
    seen = {}
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in [
        {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                              "model": "claude-sonnet-5-5", "content": [], "stop_reason": None,
                                              "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}},
        {"type": "message_stop"}]).encode()

    def handler(req):
        body = req.content.decode()
        if req.url.path.endswith("/batches"):
            seen["batch"] = body
            return httpx2.Response(200, json={"id": "msgbatch_x", "type": "message_batch",
                                              "processing_status": "in_progress", "request_counts": {
                                                  "processing": 1, "succeeded": 0, "errored": 0, "canceled": 0,
                                                  "expired": 0}, "ended_at": None,
                                              "created_at": "2026-11-11T00:00:00Z", "expires_at": "2026-11-12T00:00:00Z",
                                              "cancel_initiated_at": None, "results_url": None, "archived_at": None})
        seen["sync"] = body
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    client = anthropic.Anthropic(api_key="test-key", base_url="http://mock.invalid", max_retries=0,
                                 http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    send(request, client=client)
    line = batch_request("eval1-run-0123456789", request)
    assert AnthropicBatches(client).submit("b001", "s", [Prepared("eval1-run-0123456789", {}, line, "h", 1)]) == "msgbatch_x"
    sync = json.loads(seen["sync"])
    assert sync.pop("stream") is True
    batched = json.loads(seen["batch"])["requests"]
    assert [r["custom_id"] for r in batched] == ["eval1-run-0123456789"]
    # the same JSON, value for value (the SDK writes the streamed body's top-level keys in its own order)
    assert json.dumps(batched[0]["params"], sort_keys=True) == json.dumps(sync, sort_keys=True)
    assert batched[0]["params"] == request.anthropic_params()


# --- Anthropic: custom ids, resume, results written as the synchronous path writes them ---------------------
def test_custom_ids_are_stable_valid_and_distinct(tmp_path):
    from evals.batch import CUSTOM_ID
    from evals.batch_grade import grading_custom_id, normalise_job
    run = tmp_path / ("x" * 90 + ".weird name")
    jobs = [normalise_job({"kind": k, "run_dir": run, "name": n}) for k, n in
            (("evaluation", "eval1"), ("evaluation", "eval2"), ("truth_check", "truth1"))]
    ids = [grading_custom_id(j) for j in jobs]
    assert all(CUSTOM_ID.match(i) for i in ids) and len(set(ids)) == 3
    assert ids == [grading_custom_id(normalise_job({"kind": j["kind"], "run_dir": run, "name": j["name"]})) for j in jobs]
    assert ids[0].startswith("eval1-x")


def test_an_openai_grader_is_refused():
    from evals.batch_grade import normalise_job
    with pytest.raises(ValueError, match="anthropic"):
        normalise_job({"run_dir": "/tmp/r", "name": "eval1", "model": "openai:gpt-6.1-sol"})


def test_submit_then_resume_collects_without_submitting_again(tmp_path):
    run = _case_run(tmp_path, "sp")
    fake = FakeAnthropic()
    argv = [str(run), "--eval-name", "eval1"]
    assert _grade(argv, fake) == 0
    state_file = tmp_path / "batches" / "grading" / "state.json"
    state = _state(state_file)
    (cid, item), = state["items"].items()
    assert item["status"] == "submitted" and item["batch_id"] == "msgbatch_1" and len(item["request_sha256"]) == 64
    assert state["batches"][0]["custom_ids"] == [cid] and _ids(fake) == [cid]
    out = run / "evaluation" / "eval1"
    assert sorted(p.name for p in out.iterdir()) == ["pages", "prompt.json"]  # written at submission, as evaluate()

    assert _grade(argv, fake) == 0  # still running: nothing submitted again
    assert len(fake.created) == 1
    fake.finish("msgbatch_1", {cid: _message(_evaluation())})
    assert _grade(argv, fake) == 0  # collected
    assert len(fake.created) == 1 and _state(state_file)["items"][cid]["status"] == "ok"
    record = json.loads((out / "evaluation.json").read_text())
    assert record["grader"]["batch_id"] == "msgbatch_1"
    assert record["cost_usd"] == pytest.approx((1000 * 2 + 5000 * 2.5 + 10000 * 10) / 1e6 / 2)
    assert _grade(argv, fake) == 0 and len(fake.created) == 1  # nothing left to do
    assert (out / "evaluation.json").read_text() == json.dumps(record, indent=2, ensure_ascii=False)


def test_a_batch_grading_writes_what_the_synchronous_grading_writes(tmp_path):
    from evals.evaluate import evaluate
    sync_run = _case_run(tmp_path / "sync", "sp")
    batch_run = _case_run(tmp_path / "batch", "sp")
    evaluate(sync_run, "anthropic:claude-sonnet-5-5", "eval1", client=_Client(_message(_evaluation())))
    fake = FakeAnthropic()
    _grade([str(batch_run), "--eval-name", "eval1"], fake)
    fake.finish("msgbatch_1", {_ids(fake)[0]: _message(_evaluation())})
    _grade([str(batch_run), "--eval-name", "eval1"], fake)
    a, b = sync_run / "evaluation" / "eval1", batch_run / "evaluation" / "eval1"
    files = lambda d: sorted(str(p.relative_to(d)) for p in d.rglob("*"))  # noqa: E731
    assert files(a) == files(b)
    assert (a / "prompt.json").read_text() == (b / "prompt.json").read_text()
    for page in (a / "pages").iterdir():
        assert page.read_bytes() == (b / "pages" / page.name).read_bytes()
    ea, eb = (json.loads((d / "evaluation.json").read_text()) for d in (a, b))
    assert ea["grader"].pop("batch_id") is None and eb["grader"].pop("batch_id") == "msgbatch_1"
    assert eb["cost_usd"] == pytest.approx(ea["cost_usd"] / 2)
    assert eb["grader"].pop("cost_list_usd") == pytest.approx(ea["cost_usd"])
    batch = eb["grader"].pop("batch")
    assert batch["custom_id"] and [x["result_type"] for x in batch["attempts"]] == ["succeeded"]
    assert batch["attempts"][0]["batch_id"] == "msgbatch_1" and batch["attempts"][0]["submitted_at"]
    assert ea["grader"]["prompt_sha256"] == eb["grader"]["prompt_sha256"]
    for e in (ea, eb):
        e.pop("cost_usd"), e["grader"].pop("cost_usd"), e["grader"].pop("request_ids")
    assert ea == eb
    sa, sb = (json.loads((d / "summary.json").read_text()) for d in (a, b))
    assert set(sa) == set(sb) and sa["status"] == sb["status"] == "ok" and sb["llm_calls"] == 1
    la, lb = (json.loads((d / "llm_calls.jsonl").read_text()) for d in (a, b))
    assert set(la) == set(lb) and lb["batch_id"] == "msgbatch_1" and la["node"] == lb["node"] == "evaluator"


def _runs(tmp_path: Path, n: int) -> list[Path]:
    """n run folders under one parent, as run_matrix lays them out."""
    parent = tmp_path / "runs"
    parent.mkdir()
    return [Path(shutil.move(str(_case_run(tmp_path / str(i), "sp")), parent / f"case{i}")) for i in range(n)]


def test_the_synchronous_retry_rule_holds_in_batches(tmp_path, capsys):
    """MAX_ATTEMPTS = 2 as in send(): an unparseable answer and an unanswered request go out once more in a follow-up
    batch automatically, both attempts recorded; a refusal is final; then failures are listed, and only unanswered
    ones can be sent again by hand."""
    runs = _runs(tmp_path, 6)
    fake = FakeAnthropic()
    argv = [str(r) for r in runs] + ["--eval-name", "eval1"]
    state_file = tmp_path / "runs" / "batches" / "grading" / "state.json"
    status = lambda: {i: _state(state_file)["items"][c]["status"] for i, c in enumerate(ids)}  # noqa: E731
    out = lambda i: runs[i] / "evaluation" / "eval1"  # noqa: E731
    _grade(argv, fake)
    ids = _ids(fake)
    refusal = _message("", stop_reason="refusal")
    fake.finish("msgbatch_1", {ids[0]: _message(_evaluation()), ids[1]: "errored", ids[2]: "expired",
                               ids[3]: _message("not json"), ids[4]: _message("not json"), ids[5]: refusal},
                order=list(reversed(ids)))
    assert _grade(argv + ["--no-submit"], fake) == 1  # the refusal is a final failure
    assert status() == {0: "ok", 1: "retry", 2: "retry", 3: "retry", 4: "retry", 5: "refused"}
    assert sorted(p.name for p in out(3).iterdir()) == ["pages", "prompt.json"]  # nothing written until the end
    assert json.loads((out(5) / "summary.json").read_text())["status"] == "refused"

    assert _grade(argv, fake) == 1  # the follow-up batch: the four, not the ok one, not the refusal
    assert len(fake.created) == 2 and sorted(_ids(fake)) == sorted(ids[1:5])
    assert fake.created[1][0]["params"] == fake.created[0][1]["params"]  # the same request again
    fake.finish("msgbatch_2", {ids[1]: _message(_evaluation()), ids[2]: "expired", ids[3]: _message(_evaluation()),
                               ids[4]: _message("still not json")})
    capsys.readouterr()
    assert _grade(argv, fake) == 1
    assert status() == {0: "ok", 1: "ok", 2: "expired", 3: "ok", 4: "parse_error", 5: "refused"}
    assert len(fake.created) == 2  # at most two attempts each
    printed = capsys.readouterr().out
    assert all(ids[i] in printed for i in (2, 4, 5)) and "--resubmit-failed" in printed

    # an answer re-asked once: both attempts recorded, as send() records them
    raw = json.loads((out(3) / "raw_response.json").read_text())["attempts"]
    assert [a["parse_error"] is None for a in raw] == [False, True]
    assert [a["batch_id"] for a in raw] == ["msgbatch_1", "msgbatch_2"]
    record = json.loads((out(3) / "evaluation.json").read_text())
    assert record["grader"]["attempts"] == 2 and record["grader"]["batch_id"] == "msgbatch_2"
    assert [a["batch_id"] for a in record["grader"]["batch"]["attempts"]] == ["msgbatch_1", "msgbatch_2"]
    assert record["cost_usd"] == pytest.approx(2 * (1000 * 2 + 5000 * 2.5 + 10000 * 10) / 1e6 / 2)
    assert len((out(3) / "llm_calls.jsonl").read_text().splitlines()) == 2
    assert json.loads((out(3) / "summary.json").read_text())["llm_calls"] == 2
    # two unparseable answers: the failure as send() leaves it
    assert json.loads((out(4) / "summary.json").read_text())["status"] == "parse_error"
    assert len(json.loads((out(4) / "raw_response.json").read_text())["attempts"]) == 2
    assert not (out(4) / "evaluation.json").exists()
    # an errored request then an answer: one answered attempt, both batch attempts in the record
    record = json.loads((out(1) / "evaluation.json").read_text())
    assert record["grader"]["attempts"] == 1
    assert [a["result_type"] for a in record["grader"]["batch"]["attempts"]] == ["errored", "succeeded"]

    assert _grade(argv + ["--resubmit-failed"], fake) == 1  # by hand: only the unanswered one
    assert len(fake.created) == 3 and _ids(fake) == [ids[2]]
    fake.finish("msgbatch_3", {ids[2]: _message(_evaluation())})
    assert _grade(argv, fake) == 1
    assert status() == {0: "ok", 1: "ok", 2: "ok", 3: "ok", 4: "parse_error", 5: "refused"}
    assert len(fake.created) == 3


def test_a_case_is_never_split_across_two_batches(tmp_path, monkeypatch):
    from evals import batch_grade
    runs = _runs(tmp_path, 3)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"kind": "evaluation", "run_dir": str(r), "name": n, "case": f"case{i // 2}",
                                 "model": "anthropic:claude-sonnet-5-5"}
                                for i, r in enumerate(runs) for n in ("eval1", "eval2")]))
    monkeypatch.setattr(batch_grade.AnthropicBatches, "max_requests", 5)
    fake = FakeAnthropic()
    assert _grade(["--jobs", str(plan)], fake) == 0
    # case0 = runs 0-1 (4 gradings), case1 = run 2 (2 gradings): 4 + 2 > 5, so two batches, never 5 + 1
    assert [len(b) for b in fake.created] == [4, 2]


def test_a_run_that_died_while_submitting_is_never_submitted_twice(tmp_path, capsys):
    run = _case_run(tmp_path, "sp")
    fake = FakeAnthropic()

    class Dies(Exception):
        pass

    from evals import batch
    real = batch.BatchStore.save
    calls = {"n": 0}

    def save_then_die(self):  # the process dies inside the API call: state.json says "submitting", no id
        real(self)
        if any(b["status"] == "submitting" for b in self.batches):
            calls["n"] += 1
            raise Dies()
    batch.BatchStore.save = save_then_die
    try:
        with pytest.raises(Dies):
            _grade([str(run), "--eval-name", "eval1"], fake)
    finally:
        batch.BatchStore.save = real
    assert fake.created == []
    capsys.readouterr()
    assert _grade([str(run), "--eval-name", "eval1"], fake) == 2  # refuses: it may exist at the provider
    assert "--adopt b001=" in capsys.readouterr().err and fake.created == []
    assert _grade([str(run), "--eval-name", "eval1", "--adopt", "b001=msgbatch_9"], fake) == 0
    state = _state(tmp_path / "batches" / "grading" / "state.json")
    assert [i["batch_id"] for i in state["items"].values()] == ["msgbatch_9"] and fake.created == []
    cid = next(iter(state["items"]))
    fake.finish("msgbatch_9", {cid: _message(_evaluation())})
    assert _grade([str(run), "--eval-name", "eval1"], fake) == 0
    assert (run / "evaluation" / "eval1" / "evaluation.json").exists()


def test_a_grading_finished_synchronously_is_never_redone(tmp_path):
    from evals.evaluate import evaluate
    run = _case_run(tmp_path, "sp")
    evaluate(run, "anthropic:claude-sonnet-5-5", "eval1", client=_Client(_message(_evaluation())))
    before = (run / "evaluation" / "eval1" / "evaluation.json").read_text()
    fake = FakeAnthropic()
    assert _grade([str(run), "--eval-name", "eval1"], fake) == 0
    assert fake.created == [] and (run / "evaluation" / "eval1" / "evaluation.json").read_text() == before


def test_a_changed_run_folder_is_not_written_with_a_stale_reply(tmp_path):
    run = _case_run(tmp_path, "sp")
    fake = FakeAnthropic()
    _grade([str(run), "--eval-name", "eval1"], fake)
    (run / "outputs" / "final_cv.txt").write_text("a different CV")
    fake.finish("msgbatch_1", {_ids(fake)[0]: _message(_evaluation())})
    (tmp_path / "ledger.tsv").write_text("")
    assert _grade([str(run), "--eval-name", "eval1", "--no-submit"], fake) == 1
    item = next(iter(_state(tmp_path / "batches" / "grading" / "state.json")["items"].values()))
    assert item["status"] == "request_changed" and not (run / "evaluation" / "eval1" / "evaluation.json").exists()
    billed = (1000 * 2 + 5000 * 2.5 + 10000 * 10) / 1e6 / 2  # the answer was billed: in the ledger and the log
    (row,) = [line.split("\t") for line in (tmp_path / "ledger.tsv").read_text().splitlines()]
    assert row[1] == "eval:eval1-discarded" and float(row[2]) == pytest.approx(billed)
    assert item["attempt_log"][-1]["cost_usd"] == pytest.approx(billed)
    assert item["attempt_log"][-1]["cost_list_usd"] == pytest.approx(2 * billed)
    assert _grade([str(run), "--eval-name", "eval1", "--resubmit-failed"], fake) == 1
    assert len(fake.created) == 1  # an answered request is never sent again


def test_a_billed_reply_that_cannot_be_written_is_still_costed(tmp_path, monkeypatch):
    from evals import batch_grade
    run = _case_run(tmp_path, "sp")
    (tmp_path / "ledger.tsv").write_text("")
    fake = FakeAnthropic()
    _grade([str(run), "--eval-name", "eval1"], fake)
    fake.finish("msgbatch_1", {_ids(fake)[0]: _message(_evaluation())})

    def unreadable(job, reuse_pages=False):
        raise OSError("run folder unreadable")
    monkeypatch.setattr(batch_grade, "build", unreadable)
    assert _grade([str(run), "--eval-name", "eval1"], fake) == 1
    item = next(iter(_state(tmp_path / "batches" / "grading" / "state.json")["items"].values()))
    billed = (1000 * 2 + 5000 * 2.5 + 10000 * 10) / 1e6 / 2
    assert item["status"] == "collect_error" and "run folder unreadable" in item["error"]
    assert item["attempt_log"][-1]["cost_usd"] == pytest.approx(billed)
    (row,) = [line.split("\t") for line in (tmp_path / "ledger.tsv").read_text().splitlines()]
    assert row[1] == "eval:eval1-discarded" and float(row[2]) == pytest.approx(billed)
    assert _grade([str(run), "--eval-name", "eval1", "--resubmit-failed"], fake) == 1
    assert len(fake.created) == 1  # answered: never sent again


def test_a_reply_for_a_grading_finished_meanwhile_is_billed_not_written(tmp_path):
    run = _case_run(tmp_path, "sp")
    (tmp_path / "ledger.tsv").write_text("")
    fake = FakeAnthropic()
    _grade([str(run), "--eval-name", "eval1"], fake)
    before = '{"finished": "elsewhere"}'
    (run / "evaluation" / "eval1" / "evaluation.json").write_text(before)  # e.g. by the synchronous path
    fake.finish("msgbatch_1", {_ids(fake)[0]: _message(_evaluation())})
    assert _grade([str(run), "--eval-name", "eval1"], fake) == 0
    item = next(iter(_state(tmp_path / "batches" / "grading" / "state.json")["items"].values()))
    billed = (1000 * 2 + 5000 * 2.5 + 10000 * 10) / 1e6 / 2
    assert item["status"] == "done" and item["attempt_log"][-1]["cost_usd"] == pytest.approx(billed)
    assert [line.split("\t")[1] for line in (tmp_path / "ledger.tsv").read_text().splitlines()] == [
        "eval:eval1-discarded"]
    assert (run / "evaluation" / "eval1" / "evaluation.json").read_text() == before


def test_a_matrix_plan_batches_sonnet_opus_haiku_and_the_trap_check_together(tmp_path):
    """The jobs file run_matrix --gradings-out writes: every grader of a session in one batch; each result lands in
    its own folder with its own grader record and Batch price."""
    from evals.run_matrix import gradings_plan
    out = tmp_path / "runs"
    out.mkdir()
    run = _case_run(out, "sp")
    case = {"id": run.name, "case": "p03_a1", "arm": "E", "stage": "pilot",
            "gradings": [{"name": "eval1", "grader": "sonnet", "model": "anthropic:claude-sonnet-5-5"},
                         {"name": "eval2", "grader": "sonnet", "model": "anthropic:claude-sonnet-5-5"},
                         {"name": "opus1", "grader": "opus", "model": "anthropic:claude-opus-5-5"},
                         {"name": "haiku1", "grader": "haiku", "model": "anthropic:claude-haiku-5-5"}],
            "truth": {"name": "truth1", "grader": "trap", "model": "anthropic:claude-haiku-5-5"}}
    (run / "manifest.json").write_text(json.dumps({"status": "completed", "labels": {"persona": "p12"}}))
    plan = tmp_path / "gradings.json"
    plan.write_text(json.dumps(gradings_plan([case], out, ["--grading-mode", "document-only"])))
    (out / "ledger.tsv").write_text("")
    fake = FakeAnthropic()
    assert _grade(["--jobs", str(plan)], fake) == 0
    sent = fake.created[0]
    assert [r["params"]["model"] for r in sent] == ["claude-sonnet-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
                                                   "claude-haiku-5-5", "claude-haiku-5-5"]
    assert sent[0]["params"] == sent[1]["params"] and sent[0]["custom_id"] != sent[1]["custom_id"]
    truth_items = [{"id": "G1", "verdict": "not_claimed", "document": "none", "quote": "", "note": ""}]
    replies = {}
    for r in sent:
        model = r["params"]["model"]
        answer = {"items": truth_items} if r["custom_id"].startswith("truth1") else _evaluation()
        replies[r["custom_id"]] = _message(answer, model=model)
    fake.finish("msgbatch_1", replies)
    assert _grade(["--jobs", str(plan)], fake) == 0
    for name, model in (("eval1", "sonnet"), ("eval2", "sonnet"), ("opus1", "opus"), ("haiku1", "haiku")):
        record = json.loads((run / "evaluation" / name / "evaluation.json").read_text())
        assert record["grader"]["model_returned"] == f"claude-{model}-5-5" and record["grader"]["batch_id"] == "msgbatch_1"
    truth = json.loads((run / "evaluation" / "truth1" / "truth_check.json").read_text())
    assert truth["grader"]["batch_id"] == "msgbatch_1" and truth["persona"] == "p12"
    ledger = [line.split("\t")[:2] for line in (out / "ledger.tsv").read_text().splitlines()]
    assert sorted(k for _, k in ledger) == ["eval:eval1", "eval:eval2", "eval:haiku1", "eval:opus1", "truth"]


def test_status_and_dry_run_call_nothing(tmp_path, capsys):
    run = _case_run(tmp_path, "sp")
    fake = FakeAnthropic()
    assert _grade([str(run), "--eval-name", "eval1", "--dry-run"], fake) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["requests"] == 1 and dry["bytes"] > 0 and not (run / "evaluation").exists()
    assert _grade([str(run), "--eval-name", "eval1", "--status"], fake) == 0
    assert "not submitted 1" in capsys.readouterr().out and fake.created == []


# --- OpenAI single-prompt sessions --------------------------------------------------------------------------
CV_TEXT = "Marta Example\nDevOps Engineer\nCI pipelines with GitHub Actions."
LETTER_TEXT = "Dear hiring team,\nI would like to apply.\nKind regards, Marta"
ANSWER = {"cv_content": CV_TEXT, "cv_latex": "\\documentclass{article}\\begin{document}CV\\end{document}",
          "cover_letter_content": LETTER_TEXT,
          "cover_letter_latex": "\\documentclass{article}\\begin{document}L\\end{document}", "notes": "n"}


def _completion(model="gpt-6.1-sol"):
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": model, "service_tier": "default",
            "choices": [{"index": 0, "finish_reason": "stop", "logprobs": None,
                         "message": {"role": "assistant", "content": json.dumps(ANSWER), "refusal": None}}],
            "usage": {"prompt_tokens": 6000, "completion_tokens": 9000, "total_tokens": 15000,
                      "prompt_tokens_details": {"cached_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": 4000}}}


@pytest.fixture
def sp(monkeypatch, tmp_path):
    """The real ChatOpenAI on a mock transport for the synchronous call (every request recorded and answered with
    _completion()); pdflatex stubbed; an empty .env."""
    import httpx
    from openai import DefaultHttpxClient

    from evals import single_prompt
    from tests.test_single_prompt import _blank_pdf
    sent = []

    def handler(req):
        sent.append(req)
        return httpx.Response(200, json=_completion(json.loads(req.content)["model"]))

    real_init = single_prompt.init_model

    def init_model(model, **kwargs):
        if kwargs:  # the Batch path's own offline transports
            return real_init(model, **kwargs)
        return real_init(model, api_key="test-key", max_retries=0,
                         http_client=DefaultHttpxClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(single_prompt, "init_model", init_model)

    def fake_pdflatex(tex: Path):
        _blank_pdf(tex.with_suffix(".pdf"))
        return True, ""
    monkeypatch.setattr(single_prompt, "pdflatex", fake_pdflatex)
    env = tmp_path / "empty.env"
    env.write_text("")
    return {"sent": sent, "env": str(env)}


def _sp_argv(run_dir: Path, env: str, model="openai:gpt-6.1-sol"):
    return ["--run-dir", str(run_dir), "--ad", "a3", "--persona", "p03", "--inputs", "cv_only", "--model", model,
            "--env-file", env, "--label", "method=single_prompt"]


def test_the_batch_line_carries_the_synchronous_body_byte_for_byte(tmp_path, sp):
    from evals import single_prompt
    from evals.batch_sp import ENDPOINT, prepare
    assert single_prompt.main(_sp_argv(tmp_path / "sync", sp["env"])) == 0
    (sync_request,) = sp["sent"]
    req = single_prompt.batch_request(_sp_argv(tmp_path / "unused", sp["env"]))
    assert req["url"] == sync_request.url.path == ENDPOINT and req["method"] == "POST"
    assert req["body"].encode("utf-8") == sync_request.content  # the same bytes
    assert not (tmp_path / "unused").exists() and len(sp["sent"]) == 1  # built offline, nothing sent
    body = json.loads(req["body"])
    assert body["model"] == "gpt-6.1-sol" and body["response_format"]["type"] == "json_schema"
    assert "temperature" not in body
    job = {"custom_id": "sp-case-1", "case_id": "case-1", "argv": _sp_argv(tmp_path / "x", sp["env"])}
    line = prepare(job).request
    assert line.endswith(f'"body": {req["body"]}}}')
    parsed = json.loads(line)
    assert (parsed["custom_id"], parsed["method"], parsed["url"]) == ("sp-case-1", "POST", ENDPOINT)


def test_a_replayed_batch_reply_makes_the_synchronous_session(tmp_path, sp):
    from evals import single_prompt
    assert single_prompt.main(_sp_argv(tmp_path / "sync", sp["env"])) == 0
    sha = single_prompt.body_sha256(sp["sent"][0].content)
    result = tmp_path / "reply.json"
    result.write_text(json.dumps({"provider": "openai", "batch_id": "batch_1", "custom_id": "sp-c", "url": "/v1/chat/completions",
                                  "request_sha256": sha, "response": {"status_code": 200, "request_id": "req_1",
                                                                      "body": _completion()}}))
    assert single_prompt.main(_sp_argv(tmp_path / "batch", sp["env"]) + ["--batch-result", str(result)]) == 0
    assert len(sp["sent"]) == 1  # the replay sent nothing
    a, b = tmp_path / "sync", tmp_path / "batch"
    files = lambda d: sorted(str(p.relative_to(d)) for p in d.rglob("*") if "workspace" not in p.parts)  # noqa: E731
    assert files(a) == files(b)
    for name in ("outputs/final_cv.txt", "outputs/final_cover_letter.txt", "outputs/final_cv.tex",
                 "outputs/writer_notes.txt", "prompt.json", "transcript.json", "outputs/pdf_checks.json"):
        assert (a / name).read_text() == (b / name).read_text(), name
    ma, mb = (json.loads((d / "manifest.json").read_text()) for d in (a, b))
    assert set(mb) - set(ma) == {"batch"} and set(ma) - set(mb) == set()
    assert mb["batch"]["batch_id"] == "batch_1" and mb["status"] == ma["status"] == "completed"
    assert mb["cost_usd"] == pytest.approx(ma["cost_usd"] / 2) and ma["cost_usd"] == pytest.approx(0.102)
    assert mb["cost_list_usd"] == ma["cost_list_usd"] == pytest.approx(ma["cost_usd"])
    (la,), (lb,) = ([json.loads(x) for x in (d / "llm_calls.jsonl").read_text().splitlines()] for d in (a, b))
    assert lb["batch_id"] == "batch_1" and "batch_id" not in la and la["usage"] == lb["usage"]
    assert lb["cost_list_usd"] == pytest.approx(la["cost_usd"]) and "cost_list_usd" not in la
    for rec in (la, lb):  # what the pilot gates read, the same in both paths
        meta = rec["response_metadata"][0]
        assert (meta["finish_reason"], meta["service_tier"], meta["model_name"]) == ("stop", "default", "gpt-6.1-sol")


def test_a_reply_to_a_different_request_is_refused(tmp_path, sp):
    from evals import single_prompt
    result = tmp_path / "reply.json"
    result.write_text(json.dumps({"batch_id": "batch_1", "custom_id": "c", "request_sha256": "0" * 64,
                                  "response": {"status_code": 200, "body": _completion()}}))
    assert single_prompt.main(_sp_argv(tmp_path / "run", sp["env"]) + ["--batch-result", str(result)]) == 1
    assert "RequestChanged" in json.loads((tmp_path / "run" / "manifest.json").read_text())["status"]


class FakeOpenAI:
    """files.create / files.content / batches.create / retrieve / list; finish() ends a batch."""

    def __init__(self):
        self.uploads, self.created, self.batches_by_id = {}, [], {}
        fake = self

        class _Files:
            def create(self, file, purpose):
                assert purpose == "batch"
                fid = f"file-{len(fake.uploads) + 1}"
                fake.uploads[fid] = file.read().decode("utf-8")
                return SimpleNamespace(id=fid)

            def content(self, fid):
                return SimpleNamespace(text=fake.uploads[fid])

        class _Batches:
            def create(self, input_file_id, endpoint, completion_window, metadata):
                bid = f"batch_{len(fake.created) + 1}"
                fake.created.append({"id": bid, "input_file_id": input_file_id, "endpoint": endpoint,
                                     "completion_window": completion_window, "metadata": metadata})
                fake.batches_by_id[bid] = SimpleNamespace(id=bid, status="in_progress", output_file_id=None,
                                                          error_file_id=None, request_counts=None, errors=None,
                                                          metadata=metadata)
                return fake.batches_by_id[bid]

            def retrieve(self, bid):
                return fake.batches_by_id[bid]

            def list(self, limit=100):
                return list(fake.batches_by_id.values())

        self.files, self.batches = _Files(), _Batches()

    def lines(self, n=-1) -> list[dict]:
        return [json.loads(x) for x in self.uploads[self.created[n]["input_file_id"]].splitlines()]

    def finish(self, bid: str, ok: list[str], errors: dict | None = None):
        out, err = [], []
        for cid in ok:
            out.append({"id": "r", "custom_id": cid, "error": None,
                        "response": {"status_code": 200, "request_id": "req", "body": _completion()}})
        for cid, code in (errors or {}).items():
            err.append({"id": "r", "custom_id": cid, "response": None, "error": {"code": code, "message": code}})
        b = self.batches_by_id[bid]
        b.status = "completed"
        if out:
            b.output_file_id = f"file-out-{bid}"
            self.uploads[b.output_file_id] = "\n".join(json.dumps(x) for x in out) + "\n"
        if err:
            b.error_file_id = f"file-err-{bid}"
            self.uploads[b.error_file_id] = "\n".join(json.dumps(x) for x in err) + "\n"


def _single_prompt_matrix(folder: Path) -> Path:
    """A matrix without stages of two single-prompt (gpt-6.1-sol) cases."""
    cases = [{"id": f"sp_{p}_{a}_{s}_sol_r1", "method": "single_prompt", "persona": p, "ad": a, "input_set": s,
              "inputs": inputs, "model": "openai:gpt-6.1-sol", "rep": 1}
             for p, a, s, inputs in (("p01", "a1", "all4", "all4"), ("p01", "a1", "cv_only", ["cv"]))]
    path = folder / "matrix_single_prompt.json"
    path.write_text(json.dumps({"total_cap": 12.0, "cases": cases}))
    return path


@pytest.fixture
def sp_matrix(sp, monkeypatch, tmp_path):
    """run_matrix --sp-batch on the single-prompt matrix: OpenAI is FakeOpenAI, the collected sessions run in-process."""
    from evals import batch_sp, single_prompt
    fake = FakeOpenAI()
    monkeypatch.setattr(batch_sp.OpenAIBatches, "client", property(lambda self: fake))

    real_run = batch_sp.subprocess.run

    def run(cmd, cwd=None, stdout=None, stderr=None, **kw):
        if cmd[0] in ("git", "pdflatex"):  # the session's own version records
            return real_run(cmd, cwd=cwd, stdout=stdout, stderr=stderr, **kw)
        assert cmd[1:3] == ["-m", "evals.single_prompt"]
        return SimpleNamespace(returncode=single_prompt.main(cmd[3:]))
    monkeypatch.setattr(batch_sp.subprocess, "run", run)
    out = tmp_path / "runs"
    matrix = _single_prompt_matrix(tmp_path)

    def main(*extra, only=None):
        from evals import run_matrix
        from evals.cases import load_matrix
        ids = only or [c["id"] for c in load_matrix(matrix)]
        return run_matrix.main(["--matrix", str(matrix), "--out", str(out), "--env-file", sp["env"],
                                "--sessions-only", "--sp-batch", "--workers", "1", "--only", *ids, *extra])
    return {"fake": fake, "out": out, "main": main, "sent": sp["sent"], "matrix": matrix}


def test_run_matrix_submits_resumes_and_collects_the_single_prompt_sessions(sp_matrix, capsys):
    from evals.cases import load_matrix
    fake, out, main = sp_matrix["fake"], sp_matrix["out"], sp_matrix["main"]
    cases = load_matrix(sp_matrix["matrix"])
    assert main() == 0
    assert len(fake.created) == len({c["model"] for c in cases})  # one input file per model
    assert fake.created[0]["endpoint"] == "/v1/chat/completions" and fake.created[0]["completion_window"] == "24h"
    lines = [line for n in range(len(fake.created)) for line in fake.lines(n)]
    assert len(lines) == 2 and all(line["url"] == "/v1/chat/completions" for line in lines)
    assert all(not (out / c["id"]).exists() for c in cases) and sp_matrix["sent"] == []  # nothing ran, nothing sent
    assert "session in the OpenAI batch" in capsys.readouterr().out
    state = _state(out / "batches" / "sp_openai" / "state.json")
    assert {i["status"] for i in state["items"].values()} == {"submitted"}

    assert main() == 0 and len(fake.created) == len({c["model"] for c in cases})  # resumed: nothing submitted again
    for b in fake.created:
        fake.finish(b["id"], [line["custom_id"] for line in fake.lines(fake.created.index(b))])
    assert main() == 0
    for case in cases:
        manifest = json.loads((out / case["id"] / "manifest.json").read_text())
        assert manifest["status"] == "completed" and manifest["batch"]["batch_id"].startswith("batch_")
        assert (out / case["id"] / "outputs" / "final_cv.pdf").exists()
    ledger = [line.split("\t") for line in (out / "ledger.tsv").read_text().splitlines()]
    assert sorted(r[0] for r in ledger) == sorted(c["id"] for c in cases) and all(r[1] == "app" for r in ledger)
    assert sp_matrix["sent"] == []  # the sessions replayed the batch replies
    created = len(fake.created)
    assert main() == 0 and len(fake.created) == created  # done: nothing more
    assert "skip (session done)" in capsys.readouterr().out


def test_unanswered_requests_get_one_follow_up_then_are_listed(sp_matrix, capsys):
    from evals.cases import load_matrix
    fake, out, main = sp_matrix["fake"], sp_matrix["out"], sp_matrix["main"]
    sol = [c["id"] for c in load_matrix(sp_matrix["matrix"])]
    assert main(only=sol) == 0
    ids = [line["custom_id"] for line in fake.lines()]
    fake.finish("batch_1", [ids[0]], {ids[1]: "batch_expired"})
    capsys.readouterr()
    assert main(only=sol) == 0  # collects the answer, sends the expired one again
    assert (out / sol[0] / "manifest.json").exists() and not (out / sol[1]).exists()
    assert len(fake.created) == 2 and [line["custom_id"] for line in fake.lines()] == [ids[1]]
    manifest = json.loads((out / sol[0] / "manifest.json").read_text())
    assert manifest["batch"]["attempt"] == 1 and manifest["batch"]["submitted_at"] and manifest["batch"]["ended_at"]
    fake.finish("batch_2", [], {ids[1]: "batch_expired"})
    capsys.readouterr()
    assert main(only=sol) == 0
    printed = capsys.readouterr().out
    assert "expired" in printed and "--batch-resubmit-failed" in printed and len(fake.created) == 2
    assert main(only=sol) == 0 and len(fake.created) == 2  # no third attempt unasked
    assert main("--batch-resubmit-failed", only=sol) == 0
    assert len(fake.created) == 3 and [line["custom_id"] for line in fake.lines()] == [ids[1]]
    fake.finish("batch_3", [ids[1]])
    assert main(only=sol) == 0
    assert json.loads((out / sol[1] / "manifest.json").read_text())["batch"]["attempt"] == 3


def test_the_caps_hold_back_submissions(sp_matrix, capsys):
    fake, main = sp_matrix["fake"], sp_matrix["main"]
    assert main("--total-cap", "0.001") == 0
    assert fake.created == [] and "cap reached" in capsys.readouterr().out


def test_the_estimate_prices_batched_single_prompt_sessions_at_half(capsys, tmp_path):
    from evals import run_matrix
    matrix = str(REPO / "evals" / "matrix_ab_stage12.json")
    run_matrix.main(["--matrix", matrix, "--out", "unused", "--estimate", "--json"])
    plain = json.loads(capsys.readouterr().out)
    run_matrix.main(["--matrix", matrix, "--out", "unused", "--estimate", "--json", "--sp-batch"])
    batched = json.loads(capsys.readouterr().out)
    assert plain["total"]["standard"] == batched["total"]["standard"]
    for arm in ("C", "E"):  # SP-luna, SP-sol: the app line halves too
        a, b = plain["arms"][arm], batched["arms"][arm]
        assert b["batch"] == pytest.approx(a["batch"] - a["app"] / 2, abs=0.01)
    assert batched["arms"]["A"]["batch"] == plain["arms"]["A"]["batch"]  # the app's sessions are not batched
    single = str(_single_prompt_matrix(tmp_path))
    run_matrix.main(["--matrix", single, "--out", "unused", "--estimate"])
    a = json.loads(capsys.readouterr().out)
    run_matrix.main(["--matrix", single, "--out", "unused", "--estimate", "--sp-batch"])
    b = json.loads(capsys.readouterr().out)
    assert b["sessions"] == pytest.approx(a["sessions"] / 2, abs=0.01)


def test_an_unwritten_single_prompt_reply_is_priced_at_the_batch_rate(tmp_path):
    from evals.batch import Outcome
    from evals.batch_sp import discard
    ledger = tmp_path / "ledger.tsv"
    job = {"case_id": "case-1", "model": "openai:gpt-6.1-sol"}
    cost = discard(job, Outcome("sp-c", "succeeded", {"status_code": 200, "body": _completion()}), ledger)
    assert cost == {"cost_usd": pytest.approx(0.051), "cost_list_usd": pytest.approx(0.102)}
    (row,) = [line.split("\t") for line in ledger.read_text().splitlines()]
    assert row[:2] == ["case-1", "app-discarded"] and float(row[2]) == pytest.approx(0.051)


def test_the_cap_counts_what_is_in_flight_at_each_check(tmp_path):
    """The in-flight estimate is read from state.json at every check: a batch adopted after the check was set up
    counts, and a request collected (its cost now in the ledger) no longer counts twice."""
    from evals.batch import BatchStore
    from evals.batch_sp import batch_estimate_usd, cap_check
    model = "openai:gpt-6.1-sol"
    est = batch_estimate_usd(model)
    ledger = tmp_path / "ledger.tsv"
    store = BatchStore(tmp_path / "state", "single_prompt")
    store.items["c1"] = {"job": {"case_id": "c1", "model": model}, "status": "submitted"}
    store.items["c2"] = {"job": {"case_id": "c2", "model": model}, "status": "submitted"}
    allow = cap_check(store, SimpleNamespace(total_cap=2.5 * est), ledger)
    job = {"case_id": "c3", "model": model}
    assert not allow(job, [])  # 3 estimates > 2.5
    store.items["c1"]["status"] = "expired"
    store.items["c2"]["status"] = "ok"
    ledger.write_text(f"c2\tapp\t{est:.6f}\t-\n")
    assert allow(job, [])  # c2 counted once, from the ledger
    assert not allow(job, [{"case_id": "c4", "model": model}])  # and the requests prepared in this pass
    store.items["c5"] = {"job": {"case_id": "c5", "model": model}, "status": "submitted"}
    assert not allow(job, [])  # adopted after cap_check was set up
