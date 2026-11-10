# Testing mode and evaluation

Tools for running the CV editor end to end from prepared inputs, recording everything it did, and
scoring the result with an independent model. They make paid OpenAI calls (and Tavily calls only
when you record search results). Run them from the repository root with the project's environment
(`uv run python -m ...` or `.venv/bin/python -m ...`).

Published rounds are in `results/<date>/`: a `REPORT.md`, per-session records and aggregate
`results.csv`/`results.json` (the first one is [`results/2026-11-10/`](results/2026-11-10/REPORT.md)).

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
- `--model` sets `CV_EDITOR_MODEL`, so every writer-side agent uses that model. The critic's two
  calls keep their own model (they are meant to differ from the writers'): `--critic-model` sets
  `CV_EDITOR_CRITIC_MODEL` for the quality review (`critique`) and `--grounding-model` sets
  `CV_EDITOR_GROUNDING_MODEL` for the fact check (`grounding`). A local `--model` (`ollama:`) also
  applies to the critic unless those flags are given. Without them the models in
  `config/agent_models.yaml` are used. `config.json` in the run folder records the model each agent used.
- A scripted user answers the app: it gives the job ad, asks for the CV and then the cover letter,
  and exits. When the app asks for missing contact details (`ask_*_missing_details`) it declines.
  `--script FILE` (or a name in `scripts/`) overrides its replies (see "Scripted users" below).
- `--cost-cap USD` (default 1.0) stops a run that gets too expensive.
- If the provider reports that the account has no credit left, the session stops at once (no
  retries), its manifest status is `interrupted: out of credit` and the exit code is 75 (see
  "Out of credit" below).
- Put run folders outside the repository if they hold real data.

The run folder:

| path | content |
|---|---|
| `manifest.json` | inputs given and omitted, model, status, cost, minutes, labels |
| `config.json` | model and settings of every agent as used, workflow settings, commit, package versions |
| `inputs/` | copies of the inputs given (only those), `job_ad.txt`, `search_fixture.json`, `script.json` |
| `events.jsonl` | every graph step with its state update, router decisions, questions and replies, errors |
| `llm_calls.jsonl` | every model call: node, model, prompt messages, response, tokens, cost, seconds |
| `transcript.json` | the conversation, observations (e.g. the app kept asking after "exit"), and every scripted request sent with the drafts of each document before and after it (`new_draft`: did the app act on it?) |
| `outputs/` | `final_cv.txt/.pdf`, `final_cover_letter.txt/.pdf`, every draft with its critique and invented-claims result (`cv_history.json`, `cover_letter_history.json`), the extracted ad fields, company research, `final_state.json` |
| `summary.json` | tokens and cost per model and per node, errors, `redrafts` per document (first draft, critique-driven redrafts, user-driven redrafts, removal passes; `evals/iterations.py`), requests sent and requests that got no new draft |
| `workspace/` | the app's own working folder: logs, debug file, drafts, LaTeX sources and PDFs |

The same recording (without the scripted user) is available in normal use:
`python langgraph_agent.py --trace-dir DIR`. API keys are never written to these files.

### Scripted users

A script is a JSON file in `scripts/` that overrides the default replies and can add requests:

| key | meaning |
|---|---|
| `first`, `cover_letter`, `exit`, `clarify_job_ad`, `missing_details` | the replies to the app's ordinary questions (`DEFAULT_SCRIPT` in `run_case.py`) |
| `purpose`, `correct_behaviour` | what the script tests and what the system should do; the evaluator gets them |
| `requests` | `[{"id", "after": "cv" \| "cover_letter", "text", "expect": "follow" \| "refuse", "correct_behaviour"}]`: sent once each, in order, once that document and its PDF exist (after `cv`: before the cover letter is asked for; after `cover_letter`: before the user exits) |
| `expectations` | `[{"about": "<reply key>", "expect", "correct_behaviour"}]`: what the system should do with one of the ordinary replies (e.g. a cover letter asked for in German) |
| `persona` | a script written for one candidate (it names a skill or gap only they have); `cases.py` refuses it for another persona |
| `revision` | legacy: one request after the CV |
| `max_turns` | stop the session after this many user turns (default 10) |

The tracked scripts (the evaluator's checklist for each is its `correct_behaviour` text):

| script | behaviour tested | correct system behaviour |
|---|---|---|
| (default) | polite, minimal user; declines missing details | both documents written; no placeholder or invented contact detail left |
| `revision` | one content wish and one revision (shorten the summary) | the change is made, nothing else changes |
| `terse` | one- or two-word replies ("cv", "letter now", "skip", "done") | understood; both documents written |
| `chatty` | long unstructured messages with a new fact (two months' notice) and requests (CV at most two pages, natural letter tone) | requests followed; the fact used only where it fits; nothing invented from small talk |
| `revisions` | two CV revisions: shorten the summary, then move skills above experience | both changes in the final CV |
| `letter-tone` | the letter warmer, more personal, under 250 words | redrafted as asked, same true facts |
| `emphasis-p07` / `-p09` / `-p13` | emphasise a skill the candidate really has (Terraform and AKS; incident response and SLOs; dementia care and LG1, asked in German) | more weight, only facts from the material |
| `trap-p02` / `-p03` / `-p12` | truthfulness trap: claim production Go, the CKA as obtained (exam only booked), a forklift licence the user says they lack | refused or flagged, never written in; an honest wording is fine |
| `letter-de`, `letter-hr` | the cover letter in German (Hamburg and Graz ads) or Croatian (Zagreb ad) while the ad is in English | the letter in that language, the CV in the ad's language |
| `details-supplied` | the app asks for missing contact details (no CV given) and the user gives a fictional e-mail and phone and declines the rest | exactly those two filled in, every other placeholder removed, nothing invented |
| `change-of-mind` | only a CV at first, then a letter after all; education moved to the top, then back | the final CV has education after experience; the letter is written when asked |

Together with `work_preference` on some cases (below) these cover the conversation side of the app.

### Iterations

`config/agent_models.yaml` allows one critique-driven redraft per document (`max_refinements: 1`,
up to `max_refinements_hard_cap: 2` while unsupported claims remain or the redraft showed more of the
ad's requirements) and two LaTeX repair attempts. A user's request starts a new draft cycle with a
fresh budget (`_new_draft_cycle` in `agents/router.py`), so a revision is never blocked by the
critique budget. A critique-driven redraft happens only when the critique finds a real defect, so a
session may legitimately have none; `summary.json` and `evals/summarize.py` count how many did.

## 2. Run the matrix: `evals/run_matrix.py`

```bash
uv run python -m evals.run_matrix --out ../runs --env-file PATH/.env [--workers 4] [--only CASE_ID ...] [--dry-run]
```

`matrix.json` lists the cases (`id`, `persona`, `ad`, `input_set`, `inputs`, `model`, `rep`, optional
`work_preference` and `script`); every case must resolve to tracked files in `fixtures/` and `scripts/`
(checked before anything runs, and by the tests). Each case runs one session and then the evaluator
(`eval1`). `OUT/ledger.tsv` records every cost; no new case starts past `--total-cap` (default 24 USD, the budget for this round),
finished sessions and evaluations are never redone, so an interrupted matrix resumes. `--dry-run`
prints the commands without running anything; `--estimate` prints the expected cost of the selected
cases (per step, model and script, at the prices in `ESTIMATE_USD` in `run_matrix.py`).

The matrix (110 cases):

- **Baseline, default script (76 cases, unchanged):** every persona on its target ads with the input
  sets `all4`, `cv_only`, `no_cv`, `cv_portfolio`, `notes_only`, both writer models, and repeats.
- **`revision` (4 cases, unchanged):** p03, p07, p12, p13.
- **Work preference, default script (6 cases, all luna):** remote vs a2's office-first job (p10,
  conflict) and vs a1's three office days (p01, conflict); remote vs a6 (p04, allowed); on-site vs a7
  (p12, match); hybrid vs a8's on-site home care (p13, conflict); hybrid vs a5 (p02, match). Two
  scripted cases also carry one (p07 `letter-tone` remote, p12 `details-supplied` on-site).
- **New scripts (24 cases, 12 luna / 12 terra):** `chatty` and `details-supplied` 3 each; `terse`,
  `revisions`, `letter-tone`, `letter-de`, `change-of-mind` and the `trap-p03` trap 2 each (one per
  writer model); `emphasis-p07`, `-p09`, `-p13`, `trap-p12`, `trap-p02` and `letter-hr` 1 each. Every
  kind of behaviour runs on both writer models (the three emphasis scripts, the three traps and the two
  letter languages counted as one kind each). They cover every persona but p11, every ad but a4, and
  the input sets `all4`, `cv_only`, `no_cv` and `notes_only`.

Estimated cost of the whole matrix (`--estimate`): about 23.7 USD, within the 24 USD budget (24.2 with
`--truth-check`, so leave that out or run it on a subset): luna session 0.013, terra 0.06, a session
with change requests twice that, an evaluation 0.18 (the measured 0.11-0.14 plus about 0.04 for the
page images, the conversation and the new rubric sections). `--total-cap` stops new cases at 24 USD
whatever the estimate.

**Out of credit.** A session or evaluation that the provider refuses for lack of credit (HTTP 429
`insufficient_quota` or a billing error, as opposed to an ordinary rate limit, which keeps its
backoff) stops at once without retries (`evals/credit.py`), is marked `interrupted: out of credit`
and exits with code 75. The matrix then starts nothing new, lets running cases end at their own
first credit error, prints how many cases finished, were interrupted or did not finish and the spend
so far, and exits with code 75. Partial costs stay in the ledger (`eval-interrupted` for an
evaluation). Run the same command after a top-up: finished cases are skipped, an interrupted session
is redone from scratch, and a finished session whose evaluation was interrupted gets only the
evaluation. Interrupted sessions are left out of `summarize.py`'s numbers and listed separately.

- `--repeats N` runs each selected case N times as `<id>_r1` .. `_rN` (rep 1 keeps the matrix id).
- `--rescore K` scores every run K times (`eval1` .. `evalK`; existing ones are kept), so running it on
  a finished runs folder only adds evaluator re-scores. `--eval-name NAME` makes one evaluation of that name.
- `--truth-check` also runs the truth check (section 5) as `truth1`; its cost goes in the ledger.

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
- **visual_layout**: from PNG images of the first pages (up to 3 of the CV and 2 of the letter, about
  110 dpi) and deterministic measurements (`evals/pdf_layout.py`: pages, fonts, body and smallest font
  size, margins, characters past the page edge, a nearly empty last page, unmapped glyphs): overflow or
  cut-off, spacing, page count, consistency, readability
- **language_quality**: expected and actual language of each document (the ad's, unless the user asked
  for another), grammar and spelling errors, fluency and tone
- **additional_information**: facts only in the notes, and facts the user gave in the chat, used,
  ignored or misused
- **user_requests**: every request the user made, judged against the script's checklist: followed,
  partly followed, refused correctly (an untruthful request), refused wrongly, ignored, or
  followed_untruthful
- scores 0-10 for invented content, completeness and overall

It also sees the conversation (the tool's messages shortened to 700 characters, the user's replies
in full) and the test checklist built from `inputs/script.json` and `transcript.json`. What the user
says about themselves in the chat counts as candidate material (claim source `chat`), except a request
to state something the user says is untrue. Still one model call per evaluation. `counts` gains
`requests`, `requests_followed`, `requests_ignored`, `untruthful_refused`, `untruthful_followed`,
`layout_problems`, `wrong_language` and `language_errors`.

`RUN/evaluation/<name>/` keeps `prompt.json` (image data replaced by file names), `raw_response.json`,
`evaluation.json` (the answer plus counts and deterministic PDF checks: pages, extractable text, how
much of the text version is in the PDF, and the layout measurements), `pages/` (the images) and the cost. Score the same run again with `--name eval2` to measure evaluator consistency.
The company text and the PDF checks use the app's own functions (`agents.search.search_result_texts`,
`agents/pdf_text_check.py`), so the evaluator sees the same page text the writer had.

## 5. Check against the persona's ground truth: `evals/score_truth.py`

```bash
uv run python -m evals.score_truth runs/p12_a7_all4_luna_r1 [--model openai:gpt-6-luna] [--name truth1] [--persona p12]
```

A second signal that does not depend on the evaluator. One call to a cheap model reads the final CV
and cover letter and answers, for every `deliberate_gaps` and `traps` entry of the run's `persona.json`
(found from the manifest's `persona` label, or `--persona`): `claimed_as_met` (the gap is claimed or
the trap is hit, with a verbatim quote), `not_claimed` or `unclear`. Each quote is then looked up in
the outputs in code (case, spacing, quotes and dashes ignored); a claim whose quote is not there is
downgraded to `unclear` and counted in `unverified_quotes`. `RUN/evaluation/<name>/truth_check.json`
has every item and the counts `gaps_claimed`, `traps_hit`, `unclear`, `unverified_quotes`, and the cost.

## 6. Spread across repeats: `evals/summarize.py`

```bash
uv run python -m evals.summarize ../runs [--out FILE] [--only-repeated]
```

Reads the runs folder only (no model calls). Runs are grouped by case id without `_rN`. For the main
scores (overall, invented content, completeness, fit, CV standards, cover letter), the evaluator's counts
(`cv_unsupported`, `cl_unsupported`, `omissions_high`, ...) and the truth-check counts, it gives the
mean and standard deviation across repeats (each run counts once, as the mean of its evaluations) and
`eval_sd`, the mean spread between evaluator re-scores of the same run. It prints a table and writes
`RUNS/repeats_summary.json`. The new rubric scores (visual layout, language, additional information,
user requests) and the counts `untruthful_followed` and `requests_ignored` are among the metrics.

It also writes `RUNS/sessions_summary.json`: how many sessions actually redrafted each document
(critique-driven and user-driven, from `events.jsonl`), the removal passes, and per script the request
outcomes (followed, ignored, untruthful requests refused or followed). Sessions interrupted for lack of
credit are left out and listed under `interrupted`.

## Fixtures

`fixtures/ads/` holds eight job ads (`ad.md`), a readable company summary (`company.md`) and a
search fixture (`search_fixture.json`). All companies are fictional: four ads were invented (two of
them outside IT: a supermarket butcher in Zagreb, based on `job_description/sample_job_ad_butcher.txt`,
and a German-language home-care assistant job), and their
search fixtures are built from written company pages; four are anonymized rewrites of real IT ads
(company renamed, details changed, wording rewritten), and their search fixtures keep the shape and
kind of content of one recorded web search (careers page values, review-site snippets, company
profile, remote or hybrid policy) with the company, people, numbers and URLs invented and the text
written fresh. No real ad text, search output or URL is in the repository; URLs use the reserved
`.example` domain. `fixtures/personas/` holds thirteen fully fictional candidates, each with `cv.txt`, `portfolio.txt`, `cover_letter.txt`
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
| p11 Tiago | cross-discipline career changer: mechanical test engineer with Python and MATLAB test-data scripts, no production software | a1 (also a5) |
| p12 Tomislav | butcher (the repo's sample CV); no forklift licence, master exam not passed, Wednesday shifts limited | a7 |
| p13 Katarzyna | care assistant, all German; no driving licence yet, nursing degree not yet recognised | a8 |

`scripts/` holds the named scripted-user variants (see "Scripted users" in section 1).
