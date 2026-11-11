"""Deterministic checks for scripted multi-turn sessions, the same for the app and the chat baseline.

A scenario script (evals/scenarios/*.json) is a run_case script with an extra "checks" list. This module reads
a finished run folder of either system, rebuilds what both documents looked like after every user turn, runs
the script's checks on it and writes `checks.json` next to the run's other records. No model call.

    app (evals/run_case.py)              turns rebuilt from events.jsonl: the documents and PDFs as the state held
                                         them when the user sent the next message; the assistant's message is the
                                         app's next question
    chat baseline (single_prompt_chat)   outputs/turns.json, written by the baseline itself in the same shape

A normalised session is {"system", "turns": [{"id", "user", "assistant", "docs": {"cv", "cover_letter"},
"pdf": {kind: {"compiled", "fallback", "fixes"}}}], "final": {"cv", "cover_letter", "cv_tex", "cover_letter_tex",
"cv_pdf", "cover_letter_pdf"}, "status", "cost_usd", "minutes", "calls"}. A turn's "id" is the script request id
the user sent, or the reply key ("first", "cover_letter", "missing_details", "exit", ...); `pdf` lists only the
documents whose PDF was made in that turn.

Check types (every check has "id", "type" and "what"; `docs` defaults to both documents):
    present / absent        {pattern, docs?, section?, exempt?}: the regex is found in every document (present) or
                            in none (absent); `section` limits a CV search to that section; for absent, a match
                            whose sentence also matches `exempt` (e.g. a negation) does not count
    heading                 {pattern, present?: true}: a section heading of the CV matches (or, false, none does)
    heading_order           {first, then, adjacent?}: the first heading comes before the second (right before it)
    max_sentences           {section, n}: that CV section has at most n sentences
    max_bullets_per_entry   {section, n}: no entry of that CV section has more than n bullet points
    max_words               {doc, n}: the document has at most n words
    max_pages               {doc, n}: the final PDF has at most n pages
    unchanged_by            {request, docs?}: that request's turn left the documents' text byte for byte as it was
    style_in_tex            {doc, pattern}: the final LaTeX source matches
    compiled_every_turn     {docs?}: every PDF made in the session compiled from the system's own LaTeX (an
                            app LaTeX fix counts as compiled; the baseline's plain fallback layout does not)
    told_user               {request, pattern}: the message the user saw after that request matches
    reply_sent              {reply}: the scripted user sent that reply (e.g. "missing_details": the system asked)

Usage: .venv/bin/python -m evals.score_checks RUN_DIR [RUN_DIR ...]   (prints a line per run, writes checks.json)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KINDS = ("cv", "cover_letter")
APP_TEX = {"cv": "generated_CV", "cover_letter": "generated_cover_letter"}
JOB_AD_TURN = "<path to job ad>"


# ------------------------------------------------------------------------------------------- text helpers
def headings(text: str) -> list[tuple[int, str]]:
    """(line number, heading) for each section heading: a short all-caps line, without | or @ (not a contact line)."""
    out = []
    for i, line in enumerate((text or "").splitlines()):
        s = line.strip().strip("#").strip().strip("*").strip().rstrip(":")
        if s and len(s) < 45 and s.upper() == s and re.search("[A-Z]{3}", s) and "|" not in s and "@" not in s:
            out.append((i, s))
    return out


def section(text: str, pattern: str) -> str | None:
    """The body of the first section whose heading matches `pattern`, up to the next heading; None if no such heading."""
    lines, hs = (text or "").splitlines(), headings(text)
    for k, (i, s) in enumerate(hs):
        if re.search(pattern, s, re.I):
            end = hs[k + 1][0] if k + 1 < len(hs) else len(lines)
            return "\n".join(lines[i + 1:end]).strip()
    return None


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text or ""))


BULLET = re.compile(r"^[-•*–·]\s")


def bullets_per_entry(body: str) -> list[int]:
    """Bullet count of each entry in a section: an entry starts at a non-bullet line after a bullet (or at the top)."""
    counts: list[int] = []
    in_bullets = False
    for line in (body or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if BULLET.match(s):
            if not counts:
                counts.append(0)
            counts[-1] += 1
            in_bullets = True
        elif in_bullets or not counts:
            counts.append(0)
            in_bullets = False
    return [c for c in counts if c]


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    right_dot, right_nl = text.find(".", end), text.find("\n", end)
    right = min([r for r in (right_dot, right_nl) if r >= 0], default=len(text))
    return text[left:right]


def find_unexempted(text: str, pattern: str, exempt: str | None = None) -> list[str]:
    """Matches of `pattern` (case-insensitive) whose sentence does not match `exempt`, each with a little context."""
    out = []
    for m in re.finditer(pattern, text or "", re.I | re.M):
        sentence = _sentence_around(text, m.start(), m.end())
        if exempt and re.search(exempt, sentence, re.I):
            continue
        out.append(sentence.strip()[:160])
    return out


# ------------------------------------------------------------------------------------------- sessions
def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reply_ids(script: dict) -> dict[str, str]:
    """User text -> what it is: a request id or a reply key. The job-ad hand-over is "job_ad"."""
    from evals.run_case import REPLY_KEYS, script_requests
    ids = {script[k]: k for k in REPLY_KEYS if isinstance(script.get(k), str)}
    ids.update({r["text"]: r["id"] for r in script_requests(script)})
    ids[JOB_AD_TURN] = "job_ad"
    return ids


def app_turns(run_dir: Path, script: dict) -> list[dict]:
    """The app's session as user turns, rebuilt from events.jsonl (state updates are logged in full)."""
    ids = _reply_ids(script)
    docs = {k: None for k in KINDS}
    turns: list[dict] = []
    current = None
    for e in _jsonl(run_dir / "events.jsonl"):
        if e.get("kind") == "user_reply":
            current = {"id": ids.get(e.get("text"), "unscripted"), "user": e.get("text"), "assistant": None,
                       "docs": dict(docs), "pdf": {}}
            turns.append(current)
        elif e.get("kind") == "interrupt":
            if current is not None and current["assistant"] is None:
                current["assistant"] = e.get("message")
        elif e.get("kind") == "node" and isinstance(e.get("update"), dict):
            update = e["update"]
            for kind in KINDS:
                if update.get(f"generated_{kind}"):
                    docs[kind] = update[f"generated_{kind}"]
                    if current is not None:
                        current["docs"][kind] = docs[kind]
                if current is not None and e.get("node") == f"generate_{kind}_pdf":
                    made = bool(update.get(f"{kind}_pdf_path")) and not update.get(f"{kind}_latex_last_error")
                    prev = current["pdf"].get(kind, {"compiled": True, "fallback": False, "fixes": 0})
                    current["pdf"][kind] = {"compiled": prev["compiled"] and made, "fallback": False,
                                            "fixes": prev["fixes"] + int(update.get(f"{kind}_latex_fix_attempts") or 0)}
    return turns


def _newest_tex(workspace: Path, stem: str) -> str | None:
    found = sorted((workspace / "data" / "generated_CVs" / "temp_latex").glob(f"{stem}_draft_*.tex"),
                   key=lambda p: p.stat().st_mtime)
    return found[-1].read_text(encoding="utf-8") if found else None


def load_session(run_dir: str | Path) -> dict:
    """A finished run folder of either system, normalised (see the module docstring)."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    script = json.loads((run_dir / "inputs" / "script.json").read_text(encoding="utf-8"))
    out = run_dir / "outputs"
    chat = manifest.get("method") == "single_prompt_chat"
    if chat:
        turns = json.loads((out / "turns.json").read_text(encoding="utf-8")) if (out / "turns.json").is_file() else []
        tex = {k: _read(out / f"final_{k}.tex") for k in KINDS}
    else:
        turns = app_turns(run_dir, script)
        tex = {k: _newest_tex(run_dir / "workspace", APP_TEX[k]) for k in KINDS}
    final = {k: _read(out / f"final_{k}.txt") for k in KINDS}
    final.update({f"{k}_tex": tex[k] for k in KINDS})
    final.update({f"{k}_pdf": str(out / f"final_{k}.pdf") if (out / f"final_{k}.pdf").is_file() else None
                  for k in KINDS})
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").is_file() else {}
    return {"system": manifest.get("labels", {}).get("system") or ("b2" if chat else "app"),
            "method": manifest.get("method", "multi_agent"), "script": script, "turns": turns, "final": final,
            "status": manifest.get("status"), "cost_usd": manifest.get("cost_usd"),
            "minutes": manifest.get("minutes"), "calls": summary.get("llm_calls")}


# ------------------------------------------------------------------------------------------- checks
def _docs(check: dict) -> list[str]:
    docs = check.get("docs") or list(KINDS)
    return [docs] if isinstance(docs, str) else docs


def _turn(session: dict, request: str) -> tuple[int, dict] | tuple[None, None]:
    for i, t in enumerate(session["turns"]):
        if t["id"] == request:
            return i, t
    return None, None


def _region(session: dict, doc: str, sect: str | None) -> str | None:
    text = session["final"].get(doc)
    return section(text, sect) if (sect and text is not None) else text


def _pages(pdf: str | None) -> int | None:
    if not pdf:
        return None
    from pypdf import PdfReader
    return len(PdfReader(pdf).pages)


def run_check(check: dict, session: dict) -> tuple[bool, str]:
    """(passed, detail) for one check on a normalised session."""
    kind, final = check["type"], session["final"]
    if kind in ("present", "absent"):
        details, ok = [], True
        for doc in _docs(check):
            text = _region(session, doc, check.get("section"))
            if text is None:
                if kind == "present":
                    ok = False
                details.append(f"{doc}: no {'section' if final.get(doc) is not None else 'document'}")
                continue
            hits = find_unexempted(text, check["pattern"], check.get("exempt") if kind == "absent" else None)
            if kind == "present" and not hits:
                ok = False
                details.append(f"{doc}: not found")
            if kind == "absent" and hits:
                ok = False
                details.append(f"{doc}: {hits[0]!r}")
        return ok, "; ".join(details)
    if kind == "heading":
        hs = [s for _, s in headings(final.get("cv"))]
        found = any(re.search(check["pattern"], s, re.I) for s in hs)
        return found == check.get("present", True), f"headings: {hs}"
    if kind == "heading_order":
        hs = [s for _, s in headings(final.get("cv"))]
        a = next((i for i, s in enumerate(hs) if re.search(check["first"], s, re.I)), None)
        b = next((i for i, s in enumerate(hs) if re.search(check["then"], s, re.I)), None)
        ok = a is not None and b is not None and (b == a + 1 if check.get("adjacent") else a < b)
        return ok, f"headings: {hs}"
    if kind == "max_sentences":
        body = section(final.get("cv"), check["section"])
        n = len(sentences(body)) if body is not None else None
        return n is not None and n <= check["n"], f"{n} sentences"
    if kind == "max_bullets_per_entry":
        counts = bullets_per_entry(section(final.get("cv"), check["section"]) or "")
        return bool(counts) and max(counts) <= check["n"], f"bullets per entry: {counts}"
    if kind == "max_words":
        n = words(final.get(check["doc"])) if final.get(check["doc"]) is not None else None
        return n is not None and n <= check["n"], f"{n} words"
    if kind == "max_pages":
        n = _pages(final.get(f"{check['doc']}_pdf"))
        return n is not None and n <= check["n"], f"{n} pages"
    if kind == "unchanged_by":
        i, turn = _turn(session, check["request"])
        if turn is None:
            return False, "request not sent"
        before = session["turns"][i - 1]["docs"] if i > 0 else {}
        changed = [d for d in _docs(check) if turn["docs"].get(d) != before.get(d)]
        return not changed, f"changed: {changed}" if changed else "unchanged"
    if kind == "style_in_tex":
        tex = final.get(f"{check['doc']}_tex")
        return bool(tex and re.search(check["pattern"], tex, re.I)), "no LaTeX" if not tex else ""
    if kind == "compiled_every_turn":
        made = [(t["id"], d, t["pdf"][d]) for t in session["turns"] for d in _docs(check) if d in t.get("pdf", {})]
        bad = [f"{tid}:{d}" for tid, d, p in made if not p["compiled"] or p["fallback"]]
        fixes = sum(p.get("fixes", 0) for _, _, p in made)
        return bool(made) and not bad, f"{len(made)} PDFs, failed: {bad}, LaTeX fixes: {fixes}"
    if kind == "told_user":
        _, turn = _turn(session, check["request"])
        if turn is None:
            return False, "request not sent"
        said = turn.get("assistant") or ""
        return bool(re.search(check["pattern"], said, re.I)), said[:200]
    if kind == "reply_sent":
        sent = any(t["id"] == check["reply"] for t in session["turns"])
        return sent, "sent" if sent else "never asked"
    raise ValueError(f"unknown check type {kind!r}")


def turn_metrics(session: dict) -> dict:
    """Cost and time per user turn that asked for something (the job-ad hand-over and the exit are not counted)."""
    asked = [t for t in session["turns"] if t["id"] not in ("job_ad", "exit")]
    n = len(asked) or None
    return {"user_turns": len(asked),
            "usd_per_turn": round(session["cost_usd"] / n, 5) if n and session.get("cost_usd") is not None else None,
            "seconds_per_turn": round(session["minutes"] * 60 / n, 1) if n and session.get("minutes") is not None else None,
            "calls_per_turn": round(session["calls"] / n, 2) if n and session.get("calls") is not None else None,
            "latex_fixes": sum(p.get("fixes", 0) for t in session["turns"] for p in t.get("pdf", {}).values())}


def score_session(session: dict, checks: list[dict]) -> dict:
    results = []
    for check in checks:
        try:
            ok, detail = run_check(check, session)
        except Exception as e:  # noqa: BLE001 - a broken check is reported, not fatal
            ok, detail = False, f"check error: {type(e).__name__}: {e}"
        results.append({"id": check["id"], "type": check["type"], "what": check.get("what", ""), "pass": ok,
                        "detail": detail})
    return {"system": session["system"], "status": session["status"],
            "passed": sum(r["pass"] for r in results), "total": len(results),
            "all_passed": bool(results) and all(r["pass"] for r in results),
            "cost_usd": session["cost_usd"], "minutes": session["minutes"], "calls": session["calls"],
            **turn_metrics(session), "turns": [t["id"] for t in session["turns"]], "checks": results}


def score_run(run_dir: str | Path) -> dict:
    """Score a run folder with its own script's checks; writes and returns checks.json."""
    session = load_session(run_dir)
    result = score_session(session, session["script"].get("checks") or [])
    (Path(run_dir) / "checks.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv=None) -> int:
    for run_dir in (argv if argv is not None else sys.argv[1:]):
        r = score_run(run_dir)
        failed = [c["id"] for c in r["checks"] if not c["pass"]]
        print(f"{run_dir}: {r['system']} {r['passed']}/{r['total']} status={r['status']} failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
