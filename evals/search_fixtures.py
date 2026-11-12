"""Make web-search fixtures for testing mode: prepared search results that replace live Tavily calls.

Two ways to get one:

  record   Run the app's own job-ad extraction and company query on a job ad, call Tavily once
           (live, paid) and save the raw answer. Later runs replay it with no web access.
               .venv/bin/python -m evals.search_fixtures record --job-ad AD.md --out FIXTURE.json

  pages    Build a fixture from text/markdown pages you prepared (an invented company's about,
           news and careers pages, or an anonymized company profile). Each page becomes one search
           result, shaped like a Tavily result.
               .venv/bin/python -m evals.search_fixtures pages --company "Name" --out FIXTURE.json PAGE.md ...

The fixture format is documented in agents/search.py (load_search_fixture).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# A Tavily result's "content" is a snippet, not the whole page; keep prepared pages similar.
SNIPPET_CHARS = 1500


def _title_and_url(path: Path, text: str) -> tuple[str, str]:
    title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), path.stem)
    url = (re.search(r"Source(?: URL)?:\s*(\S+)", text)
           or re.search(r"\b((?:https?://)?[\w-]+(?:\.[\w-]+)+/[\w./-]*)", text))
    return title, (url.group(1).rstrip(").,") if url else f"https://example.invalid/{path.stem}")


def fixture_from_pages(company: str, pages: list[Path]) -> dict:
    results = []
    for page in pages:
        text = page.read_text(encoding="utf-8").strip()
        title, url = _title_and_url(page, text)
        results.append({"url": url, "title": title, "content": text[:SNIPPET_CHARS], "raw_content": text, "score": 0.9})
    return {
        "fixture_version": 1,
        "company": company,
        "source": "prepared pages",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "searches": [{"query": "*", "response": {"query": "*", "results": results}}],
    }


def record(job_ad: Path, out: Path, model: str | None, env_file: str | None = None) -> dict:
    from dotenv import load_dotenv
    load_dotenv(env_file or REPO / ".env")  # keys are read, never written
    from agents.job_description import JobDescriptionAgent
    from agents.search import SearchAgent, append_to_search_fixture, build_search_query
    from config.config_loader import get_agent_config

    cfg = get_agent_config("job_description")
    info = JobDescriptionAgent(model=model or cfg["model"], temperature=cfg["temperature"]).extract_info(
        job_ad.read_text(encoding="utf-8")).model_dump()
    query = build_search_query(info)
    scfg = get_agent_config("search")
    agent = SearchAgent(model=model or scfg["model"], temperature=scfg["temperature"])
    agent.fixture, agent.record_path = None, None  # live search even if a fixture is set in the env
    response = agent.search_tavily(query)
    append_to_search_fixture(str(out), query, response, company=info.get("company_name"))
    return {"company": info.get("company_name"), "query": query, "results": len(response.get("results", []))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="live Tavily search for the company in a job ad, saved as a fixture")
    r.add_argument("--job-ad", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--model", help="model for the job-ad extraction (default: config)")
    r.add_argument("--env-file", help=".env file with API keys (default: the repo's .env)")
    p = sub.add_parser("pages", help="fixture from prepared text pages")
    p.add_argument("--company", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("pages", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "record":
        if args.out.exists():
            sys.exit(f"{args.out} exists; delete it to record again")
        print(json.dumps(record(args.job_ad, args.out, args.model, args.env_file)))
    else:
        fixture = fixture_from_pages(args.company, args.pages)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"out": str(args.out), "results": len(fixture["searches"][0]["response"]["results"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
