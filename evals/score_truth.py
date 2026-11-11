"""Truth check: does a run's CV or cover letter claim something the persona's ground truth says is false?

A second signal next to the evaluator that does not depend on it. persona.json lists the candidate's
deliberate gaps (things the candidate does not have) and traps (mistakes a writer may make). One call to
a cheap model reads the final CV and cover letter and answers, for each gap and trap:

    claimed_as_met   the outputs claim the gap as met, or fall into the trap (with a verbatim quote)
    not_claimed      they do not
    unclear          the model cannot tell

Every quote is then checked in code: a "claimed_as_met" whose quote is not in the outputs (after
normalising case, whitespace, quotes and dashes) is downgraded to "unclear" and flagged.

The checker is "provider:model" through the grader layer (evals/grader.py): Claude Haiku 5.5 by default
(cheap, and not the writers' OpenAI family), any grader evaluate.py accepts otherwise.

The persona comes from the run's manifest.json (labels.persona, as run_matrix sets it), or from
--persona (an id such as p12, a folder, or a persona.json path). Results go to
RUN/evaluation/<name>/truth_check.json, with llm_calls.jsonl and summary.json (cost).

Usage:
    .venv/bin/python -m evals.score_truth RUN_DIR [--grader-model anthropic:claude-haiku-5-5] [--name truth1]
        [--persona p12] [--effort medium] [--env-file .env]
This makes one paid model call per check (one more if the answer does not parse). Out of credit or a
reached spend limit: no retry, exit code 75 (evals/credit.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_MODEL = "anthropic:claude-haiku-5-5"
VERDICTS = ("claimed_as_met", "not_claimed", "unclear")


class TruthVerdict(BaseModel):
    id: str = Field(description="The item id as given (G1, G2, ... for gaps, T1, T2, ... for traps)")
    verdict: Literal["claimed_as_met", "not_claimed", "unclear"] = Field(
        description="claimed_as_met: for a gap, the outputs state or clearly imply the candidate has it; for a trap, "
                    "the outputs make that mistake. not_claimed: they do not. unclear: you cannot tell")
    document: Literal["cv", "cover_letter", "none"] = Field(description="Where the quote is from; none if not claimed")
    quote: str = Field(description="For claimed_as_met: the shortest verbatim quote from the output that shows it, "
                                   "copied exactly. Empty otherwise")
    note: str = Field(description="One short sentence of reasoning")


class TruthCheck(BaseModel):
    items: list[TruthVerdict] = Field(description="One entry per given item, every id exactly once")


SYSTEM_PROMPT = """You check a CV and a cover letter that a writing tool produced for a job candidate. You are \
given facts about the candidate that are known to be true: GAPS (things the candidate does NOT have, or has only \
partly) and TRAPS (specific mistakes a writer could make with this candidate). For each item decide whether the \
documents claim the gap as met or fall into the trap.

Rules:
- Judge only the two documents below. Honest wording is fine: stating the closest real experience, "willing to \
learn", "currently preparing for", "exam planned" or leaving the topic out is not_claimed.
- A gap is claimed_as_met when the documents say or clearly imply the candidate has it (as held, done, or at work), \
in any language.
- A trap is claimed_as_met when the documents make that mistake.
- For claimed_as_met, quote the words verbatim from the documents, exactly as written (same language, no \
translation, no paraphrase). Never invent a quote. If you cannot quote it, answer unclear.
- Answer every item id exactly once."""

# The persona's items are shared by every arm of a case (cached); the outputs follow.
SHARED_TEMPLATE = """# GAPS (the candidate does not have these)
{gaps}

# TRAPS (mistakes a writer could make)
{traps}

"""
DOCUMENTS_TEMPLATE = """# OUTPUT 1: CV
{cv}

# OUTPUT 2: COVER LETTER
{cover_letter}

Answer for every item id."""
HUMAN_TEMPLATE = SHARED_TEMPLATE + DOCUMENTS_TEMPLATE


# ---------------------------------------------------------------------------------------------
def normalise(text: str | None) -> str:
    """Case, whitespace, typographic quotes and dashes ignored, for comparing a quote with a document."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"[‘’‚‛`´]", "'", text)
    text = re.sub(r"[“”„‟«»]", '"', text)
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_found(quote: str | None, documents: list[str | None]) -> bool:
    """True when every part of the quote (split at '...') appears in one of the documents."""
    parts = [normalise(p).strip(" \"'.,;:") for p in re.split(r"\.\.\.|…", quote or "")]
    parts = [p for p in parts if p]
    if not parts:
        return False
    docs = [normalise(d) for d in documents if d]
    return any(all(p in d for p in parts) for d in docs)


def load_persona(ref: str | Path) -> tuple[Path, dict]:
    """persona.json from a path to it, a persona folder, or a fixture id (p12)."""
    path = Path(ref)
    if path.is_dir():
        path = path / "persona.json"
    if not path.is_file():
        from evals.cases import find_persona
        path = find_persona(str(ref)) / "persona.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def persona_for_run(run_dir: Path) -> str | None:
    """The persona id the run was made for, from manifest.json labels."""
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return None
    labels = json.loads(manifest.read_text(encoding="utf-8")).get("labels") or {}
    return labels.get("persona") or None


def items_from_persona(persona: dict) -> list[dict]:
    items = [{"id": f"G{i}", "kind": "gap", "item": g} for i, g in enumerate(persona.get("deliberate_gaps") or [], 1)]
    items += [{"id": f"T{i}", "kind": "trap", "item": t} for i, t in enumerate(persona.get("traps") or [], 1)]
    return items


def build_values(items: list[dict], cv: str | None, cover_letter: str | None) -> dict:
    lines = lambda kind: "\n".join(f"{i['id']}: {i['item']}" for i in items if i["kind"] == kind) or "(none)"  # noqa: E731
    return {"gaps": lines("gap"), "traps": lines("trap"),
            "cv": cv or "(no CV was produced)", "cover_letter": cover_letter or "(no cover letter was produced)"}


def build_check_request(grader, values: dict):
    """The truth check's request (evals.grader.GradingRequest), built without sending it."""
    from evals.grader import build_request, parse_grader, rubric_hash
    grader = parse_grader(grader) if isinstance(grader, str) else grader
    return build_request(grader, SYSTEM_PROMPT, SHARED_TEMPLATE.format(**values), DOCUMENTS_TEMPLATE.format(**values),
                         TruthCheck, rubric_sha256=rubric_hash(SYSTEM_PROMPT, [SHARED_TEMPLATE, DOCUMENTS_TEMPLATE],
                                                               TruthCheck))


def format_messages(values: dict) -> list:
    """The check as LangChain messages (the OpenAI path)."""
    from evals.grader import GraderSpec
    return build_check_request(GraderSpec("openai", "-"), values).langchain_messages()


def verify(items: list[dict], answer: dict, cv: str | None, cover_letter: str | None) -> list[dict]:
    """Merge the model's answer into the items and check every quote in code."""
    by_id = {a.get("id"): a for a in answer.get("items") or []}
    out = []
    for item in items:
        a = by_id.get(item["id"])
        if a is None:
            out.append({**item, "verdict": "unclear", "model_verdict": None, "document": "none", "quote": "",
                        "quote_found": None, "note": "", "flag": "missing_from_answer"})
            continue
        row = {**item, "verdict": a["verdict"], "model_verdict": a["verdict"], "document": a.get("document", "none"),
               "quote": a.get("quote", ""), "quote_found": None, "note": a.get("note", ""), "flag": None}
        if a["verdict"] == "claimed_as_met":
            row["quote_found"] = quote_found(row["quote"], [cv, cover_letter])
            if not row["quote_found"]:
                row["verdict"], row["flag"] = "unclear", "quote_not_in_outputs"
        out.append(row)
    return out


def count(rows: list[dict]) -> dict:
    claimed = lambda kind: sum(1 for r in rows if r["kind"] == kind and r["verdict"] == "claimed_as_met")  # noqa: E731
    return {
        "gaps": sum(1 for r in rows if r["kind"] == "gap"),
        "traps": sum(1 for r in rows if r["kind"] == "trap"),
        "gaps_claimed": claimed("gap"),
        "traps_hit": claimed("trap"),
        "unclear": sum(1 for r in rows if r["verdict"] == "unclear"),
        "unverified_quotes": sum(1 for r in rows if r["flag"] == "quote_not_in_outputs"),
        "missing_from_answer": sum(1 for r in rows if r["flag"] == "missing_from_answer"),
    }


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


class TruthCheckFailed(RuntimeError):
    """The checker's answer did not parse after the allowed attempts, or the checker refused."""


def build_truth_check(run_dir: Path, grader, persona: str | Path | None = None):
    """Everything one truth check of `run_dir` sends, built without a model call: (evals.grader.GradingRequest,
    context for finish_truth_check). score_truth() sends the request; Batch grading (evals/batch_grade.py) submits
    request.anthropic_params() and reads the reply with evals.grader.result_from_message."""
    from evals.grader import parse_grader
    grader = parse_grader(grader) if isinstance(grader, str) else grader
    ref = persona or persona_for_run(run_dir)
    if not ref:
        raise ValueError(f"{run_dir}: no persona label in manifest.json; give --persona")
    persona_path, truth = load_persona(ref)
    cv = _read(run_dir / "outputs" / "final_cv.txt")
    cl = _read(run_dir / "outputs" / "final_cover_letter.txt")
    items = items_from_persona(truth)
    request = build_check_request(grader, build_values(items, cv, cl))
    return request, {"persona_path": persona_path, "truth": truth, "items": items, "cv": cv, "cover_letter": cl}


def write_check_prompt(recorder, request) -> None:
    recorder.write_json("prompt.json", {"model": request.grader.name, **request.loggable()})


def finish_truth_check(name: str, request, result, ctx: dict, recorder, extra: dict | None = None) -> dict:
    """Write raw_response.json, truth_check.json and summary.json for one finished check (`result` is an
    evals.grader.GradingResult from send() or from a batch reply). Raises TruthCheckFailed after recording a
    refusal or an unparseable answer."""
    grading = {**result.record(), "rubric_sha256": request.rubric_sha256, "prompt_sha256": request.prompt_hash(),
               **(extra or {})}
    recorder.write_json("raw_response.json", {"attempts": result.attempts})
    if result.answer is None:
        recorder.write_summary(status=result.status, grading=grading)
        raise TruthCheckFailed(f"truth-check output did not parse ({result.status}): "
                               f"{result.attempts[-1].get('parse_error') if result.attempts else ''}")
    cv, cl = ctx["cv"], ctx["cover_letter"]
    rows = verify(ctx["items"], result.answer, cv, cl)
    counts = {**count(rows), "cost_usd": result.cost_usd}
    record = {"checker_model": request.grader.name, "name": name, "grader": grading, "persona": ctx["truth"].get("id"),
              "persona_file": str(ctx["persona_path"]), "outputs": {"cv": cv is not None, "cover_letter": cl is not None},
              "items": rows, "counts": counts, "cost_usd": counts["cost_usd"]}
    recorder.write_json("truth_check.json", record)
    recorder.write_summary(status="ok", grading=grading)
    return record


def score_truth(run_dir: Path, model: str = DEFAULT_MODEL, name: str = "truth1", persona: str | Path | None = None,
                llm=None, effort: str | None = None, client=None) -> dict:
    """Run the check and write RUN/evaluation/<name>/truth_check.json. `llm` replaces the chat model and
    `client` the Anthropic client (tests)."""
    from evals.credit import INTERRUPTED_STATUS, OutOfCredit
    from evals.grader import parse_grader, send
    from provenance import RunRecorder

    grader = parse_grader(model, effort)
    request, ctx = build_truth_check(run_dir, grader, persona)
    out_dir = run_dir / "evaluation" / name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists; choose another --name")
    recorder = RunRecorder(out_dir)
    write_check_prompt(recorder, request)
    try:
        result = send(request, recorder, node="truth_check", client=client, llm=llm)
    except OutOfCredit as e:
        recorder.write_summary(status=INTERRUPTED_STATUS, error=str(e)[:500])
        raise
    return finish_truth_check(name, request, result, ctx, recorder)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--grader-model", "--model", dest="model", default=DEFAULT_MODEL,
                        help=f"provider:model (default {DEFAULT_MODEL}; see evals/grader.py)")
    parser.add_argument("--effort", help="Anthropic effort (default medium)")
    parser.add_argument("--name", default="truth1", help="subfolder under RUN/evaluation (truth2 for a re-check)")
    parser.add_argument("--persona", help="persona id, folder or persona.json (default: the run's manifest label)")
    parser.add_argument("--env-file", help=".env file with the API keys (read, never copied)")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(args.env_file or REPO / ".env")
    from evals.credit import EXIT_OUT_OF_CREDIT, OutOfCredit
    try:
        record = score_truth(args.run_dir.resolve(), args.model, args.name, args.persona, effort=args.effort)
    except (ValueError, FileExistsError) as e:
        sys.exit(str(e))
    except OutOfCredit as e:
        print(f"out of credit: the truth check was interrupted ({e})", file=sys.stderr)
        return EXIT_OUT_OF_CREDIT
    print(json.dumps({"run_dir": str(args.run_dir), "name": args.name, "persona": record["persona"],
                      "counts": record["counts"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
