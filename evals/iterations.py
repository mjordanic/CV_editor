"""How often a recorded session actually redrafted its documents, read from its events.jsonl.

Every `draft_cv` / `draft_cover_letter` step in the event log is one draft. The step before it says
why it was written:

    critique_<kind>              critique-driven redraft (the critique found a defect; bounded by
                                 workflow.max_refinements / max_refinements_hard_cap)
    router or ask_<kind>_...     a request: the first draft, then each later one is user-driven (a
                                 revision request, a request to change language or tone, or the
                                 contact details the user supplied)

A critique-driven step whose message says the refinement was skipped (every suggestion clashed with
the user's earlier requests) is counted as `critique_skipped`, not as a redraft. The finalize step's
removal-only pass (unsupported claims deleted) is counted from the history as `removal_passes`.
"""
from __future__ import annotations

import json
from pathlib import Path

KINDS = ("cv", "cover_letter")


def _events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _skipped(update) -> bool:
    msgs = (update or {}).get("messages") or []
    return any("refinement skipped" in str((m or {}).get("content", "")).lower() for m in msgs if isinstance(m, dict))


def redrafts(run_dir: str | Path) -> dict:
    """{kind: {drafts, first, critique_redrafts, critique_skipped, user_redrafts, removal_passes}}."""
    run_dir = Path(run_dir)
    nodes = [e for e in _events(run_dir / "events.jsonl") if e.get("kind") == "node"]
    out = {k: {"drafts": 0, "first": 0, "critique_redrafts": 0, "critique_skipped": 0, "user_redrafts": 0,
               "removal_passes": 0} for k in KINDS}
    previous = None
    for event in nodes:
        node = event.get("node")
        for kind in KINDS:
            if node != f"draft_{kind}":
                continue
            counts = out[kind]
            if previous == f"critique_{kind}":
                if _skipped(event.get("update")):
                    counts["critique_skipped"] += 1
                else:
                    counts["critique_redrafts"] += 1
                    counts["drafts"] += 1
            elif counts["drafts"] == 0:
                counts["first"] += 1
                counts["drafts"] += 1
            else:
                counts["user_redrafts"] += 1
                counts["drafts"] += 1
        previous = node
    for kind in KINDS:
        history = run_dir / "outputs" / f"{kind}_history.json"
        if history.is_file():
            try:
                entries = json.loads(history.read_text(encoding="utf-8")) or []
                out[kind]["removal_passes"] = sum(1 for e in entries if isinstance(e, dict) and e.get("removal_pass_of"))
            except json.JSONDecodeError:
                pass
    return out
