# Testing mode and evaluation

Tools for running the CV editor end to end from prepared inputs, recording everything it did, and
scoring the result with an independent model. They make paid OpenAI calls (and Tavily calls only
when you record search results). Run them from the repository root with the project's environment
(`uv run python -m ...` or `.venv/bin/python -m ...`).

## 1. Run one session: `evals/run_case.py`

```bash
uv run python -m evals.run_case --run-dir ../runs/p03_a3_all4 --ad a3 --persona p03 --model openai:gpt-6-luna
```

- `--ad a3` uses `fixtures/ads/a3-*/ad.md` and its `search_fixture.json`; `--persona p03` gives the
  candidate's files from `fixtures/personas/p03-*/`. `--inputs` picks which: `all4` (default),
  `cv_only`, `no_cv`, or a list such as `cv,notes`. Without a CV and without notes the app asks for a
  CV, and the run stops there.
- Any file can be given explicitly instead, or to override one: `--job-ad`, `--cv`, `--portfolio`,
  `--cover-letter`, `--notes`, `--search-fixture FILE` (replays prepared results, no web access).
  `--record-search FILE` searches live and saves the answers; `--no-search` finds nothing (also the
  default when neither `--ad` nor `--search-fixture` is given).
- `--work-preference remote|hybrid|on-site|no preference` is the candidate's own stated preference.
  Without it the candidate counts as open to the job's arrangement.
- `--model` sets `CV_EDITOR_MODEL`, so every agent uses that model. Without it the models in
  `config/agent_models.yaml` are used.
- A scripted user answers the app: it gives the job ad, asks for the CV and then the cover letter,
  and exits. `--script FILE` overrides its replies (keys `first`, `cover_letter`, `exit`,
  `clarify_job_ad`, `max_turns`; see `DEFAULT_SCRIPT` in `run_case.py`).
- `--cost-cap USD` (default 1.0) stops a run that gets too expensive.
- Put run folders outside the repository if they hold real data.

The run folder:

| path | content |
|---|---|
| `manifest.json` | inputs given and omitted, model, status, cost, minutes, labels |
| `config.json` | model and settings of every agent as used, workflow settings, commit, package versions |
| `inputs/` | copies of the inputs given (only those), `job_ad.txt`, `search_fixture.json`, `script.json` |
| `events.jsonl` | every graph step with its state update, router decisions, questions and replies, errors |
| `llm_calls.jsonl` | every model call: node, model, prompt messages, response, tokens, cost, seconds |
| `transcript.json` | the conversation, plus observations (e.g. the app kept asking after "exit") |
| `outputs/` | `final_cv.txt/.pdf`, `final_cover_letter.txt/.pdf`, every draft with its critique and invented-claims result (`cv_history.json`, `cover_letter_history.json`), the extracted ad fields, company research, `final_state.json` |
| `summary.json` | tokens and cost per model and per node, errors |
| `workspace/` | the app's own working folder: logs, debug file, drafts, LaTeX sources and PDFs |

The same recording (without the scripted user) is available in normal use:
`python langgraph_agent.py --trace-dir DIR`. API keys are never written to these files.

## 2. Run the matrix: `evals/run_matrix.py`

```bash
uv run python -m evals.run_matrix --out ../runs --env-file PATH/.env [--workers 4] [--only CASE_ID ...] [--dry-run]
```

`matrix.json` lists the cases (`id`, `persona`, `ad`, `input_set`, `inputs`, `model`, `rep`, optional
`work_preference`); every case must resolve to tracked files in `fixtures/` (checked before anything
runs, and by the tests). Each case runs one session and then the evaluator (`eval1`). `OUT/ledger.tsv`
records every cost; no new case starts past `--total-cap` (default 23 USD), and finished cases are
skipped, so an interrupted matrix resumes. `--dry-run` prints the commands without running anything.

## 3. Prepare web-search results: `evals/search_fixtures.py`

```bash
# a real company: one live Tavily search, saved for replay
uv run python -m evals.search_fixtures record --job-ad AD.md --out fixture.json
# an invented or anonymized company: build the results from pages you wrote
uv run python -m evals.search_fixtures pages --company "Tessel Cargo Software" --out fixture.json about.md news.md
```

A fixture is `{"fixture_version": 1, "company": ..., "searches": [{"query": ..., "response": <Tavily answer>}]}`.
In replay the answer for the same query is used, else the first one.

## 4. Score a run: `evals/evaluate.py`

```bash
uv run python -m evals.evaluate runs/p03_a3_all4 [--model openai:gpt-6.1-sol] [--name eval1]
```

The evaluator sees the job ad, the company information from the search fixture, the candidate
documents that were given (omitted ones are absent), and the final CV and cover letter as text and
as extracted PDF text. It does not see the model, the input-set name or the run id. It returns one
JSON answer with quoted evidence:

- **claims**: every skill, tool, employer, title, date, number, degree, certificate in both outputs,
  each `supported`, `partly_supported` (stretched) or `unsupported`, with a quote from the material
- **omissions**: relevant facts in the material that the outputs leave out
- **fit**: every must-have and nice-to-have requirement: does the candidate have it, does the CV show it,
  and is a gap handled honestly
- **work_arrangement**: what the ad and company pages say (remote / hybrid / on-site), every statement
  about arrangement, location or the candidate's preference in the outputs, anything invented or wrong,
  culture actually used, and whether a preference was stated only if the candidate gave one and a
  conflict with the job was not hidden. Without a stated preference the candidate counts as open:
  neutral wording is fine, an invented commitment or presenting the arrangement as a weakness is not
- **cv_standards**, **ats** (PDF text parseable, headings, keyword coverage, pass likelihood low/medium/high
  with reasons) and **cover_letter**
- scores 0-10 for invented content, completeness and overall

`RUN/evaluation/<name>/` keeps `prompt.json`, `raw_response.json`, `evaluation.json` (the answer plus
counts and deterministic PDF checks: pages, extractable text, how much of the text version is in
the PDF) and the cost. Score the same run again with `--name eval2` to measure evaluator consistency.

## Fixtures

`fixtures/ads/` holds six job ads (`ad.md`), a readable company summary (`company.md`) and a
search fixture (`search_fixture.json`). All companies are fictional: two ads were invented, and their
search fixtures are built from written company pages; four are anonymized rewrites of real IT ads
(company renamed, details changed, wording rewritten), and their search fixtures keep the shape and
kind of content of one recorded web search (careers page values, review-site snippets, company
profile, remote or hybrid policy) with the company, people, numbers and URLs invented and the text
written fresh. No real ad text, search output or URL is in the repository; URLs use the reserved
`.example` domain. `fixtures/personas/` holds ten fully fictional candidates, each with `cv.txt`, `portfolio.txt`, `cover_letter.txt`
(an old letter for an earlier job), `notes.txt` (rough free-form notes) and `persona.json` (the
grader's ground truth: true skills, deliberate gaps, facts that appear in only one file, traps a
writer may fall into, and the ads it targets). Never give `persona.json` to the CV editor.

| persona | case | target ad |
|---|---|---|
| p01 Lena | strong match, senior data engineer | a1 (also a5) |
| p02 Daniel | partial match: strong backend engineer on a different language stack | a5 |
| p03 Marta | career changer, QA automation to DevOps; homelab tools, exam booked but not passed | a3 (also a4) |
| p04 Aiyana | junior, recent graduate | a6 (also a4) |
| p05 Rafael | over-qualified senior | a5 |
| p06 Petra | Croatian CV and cover letter, PhD in progress | a2 |
| p07 Samuel | employment gaps, Azure only | a3 |
| p08 Oskar | long, messy CV with duplicates and qualified skills | a6 (also a5) |
| p09 Nadia | strong match, SRE / platform | a3 |
| p10 Tobias | partial match: modelling strong, production and MLOps weak | a2 (also a1) |
