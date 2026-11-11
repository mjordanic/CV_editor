# Testing mode and evaluation

Tools for running the CV editor end to end from prepared inputs, recording everything it did, and
scoring the result with an independent model. They make paid OpenAI calls, paid Anthropic calls when
a Claude model grades (`ANTHROPIC_API_KEY` in `.env`), and Tavily calls only when you record search
results. Run them from the repository root with the project's environment
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
| `summary.json` | tokens and cost per model and per node, errors, `redrafts` per document (first draft, critique-driven redrafts, user-driven redrafts, removal passes; `evals/iterations.py`), requests sent, requests never sent (`requests_not_sent`) and requests that got no new draft |
| `workspace/` | the app's own working folder: logs, debug file, drafts, LaTeX sources and PDFs |

The same recording (without the scripted user) is available in normal use:
`python langgraph_agent.py --trace-dir DIR`. API keys are never written to these files.

### Scripted users

A script is a JSON file in `scripts/` that overrides the default replies and can add requests:

| key | meaning |
|---|---|
| `first`, `cover_letter`, `exit`, `clarify_job_ad`, `missing_details` | the replies to the app's ordinary questions (`DEFAULT_SCRIPT` in `run_case.py`) |
| `purpose`, `correct_behaviour` | what the script tests and what the system should do; the evaluator gets them |
| `requests` | `[{"id", "after": "cv" \| "cover_letter", "text", "expect": "follow" \| "refuse", "correct_behaviour"}]`: sent once each, in order, once that document and its PDF exist (after `cv`: before the cover letter is asked for; after `cover_letter`: before the user exits). With `"unless_sent": "<reply key>"` a request is the fallback for a wish that ordinary reply carries: it is sent only if that reply never was (the app wrote the letter unasked, or never asked for missing details) |
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
| `letter-de`, `letter-hr` | the cover letter in German (Hamburg and Graz ads) or Croatian (Zagreb ad) while the ad is in English; the first reply asks for the CV only, and if the app writes the letter anyway the wish follows as a change request | the letter in that language, the CV in the ad's language |
| `details-supplied` | no contact details in the inputs (no CV given); the user gives a fictional e-mail and phone when the app asks for missing details, or volunteers them once the CV exists if it never asks, and declines the rest | exactly those two filled in, every other placeholder removed, nothing invented |
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
`work_preference`, `script`, and `method`/`baseline` for single-prompt cases, section 7); every case must resolve to tracked files in `fixtures/` and `scripts/`
(checked before anything runs, and by the tests). Each case runs one session and then the evaluator
(`eval1`). `OUT/ledger.tsv` records every cost; no new case starts past `--total-cap` (default: the matrix file's
`total_cap`, else 24 USD, the budget for this round),
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
uv run python -m evals.evaluate runs/p03_a3_all4 [--grader-model anthropic:claude-sonnet-5-5] [--eval-name eval1] \
    [--grading-mode document-only|conversation] [--effort medium]
```

(`--model` and `--name` still work as the old spellings of `--grader-model` and `--eval-name`.)

The evaluator sees the job ad, the company information from the search fixture, the candidate
documents that were given (omitted ones are absent), and the final CV and cover letter as text and
as plain text extracted from the PDFs (pdfplumber, no model). The CV's extracted text is labelled *what
an ATS would read*. It does not see the model, the input-set name or the run id. It returns one
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
- **cv_standards** and **cover_letter**
- **ats**: one ATS score 0-10 with a short `reason`, judged inside the same grading call from what an ATS
  would read (the CV PDF's extracted text), the ad and the candidate's material. Behind it, three
  judgements: the text is readable and in order with clear sections (`text_readable_in_order`,
  `clear_sections`, `headings_found`, `parse_issues`); the ad's important requirements that the candidate
  has are visible in wording a recruiter's search would find (`important_requirements`,
  `covered_in_searchable_wording`, `missed_but_candidate_has`); keyword stuffing and skills the candidate
  does not have are flagged (`keyword_stuffing`, `unsupported_keywords`) and lower the score.
  Requirements the candidate lacks cost nothing when left out. No extra model call and no extra input
  files; `counts.ats_score` carries the score, and it is the only ATS measure the analysis reads
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

**Grading modes.** `--grading-mode document-only` (the default, and the mode of the method
comparison) shows the grader the documents and the inputs above plus a fixed block, *the user's
instructions to the writing tool*, in place of the chat: the script's replies quoted verbatim (for the
default script: "Please write my CV first, then the cover letter.", the reply declining missing contact
details, the reply about the job ad) and "The user made no other requests." It never shows the app's
messages, the transcript, the manifest, the method or the writer model, so a multi-agent session and a
single-prompt session of the same case get the same block, and their requests differ only in the output
sections (`tests/test_grader.py` checks this). The block is built by `user_instructions()` from
`inputs/script.json`; a scripted change request is added, with its expected behaviour, only if it was
sent (the rule below). `--grading-mode conversation` is the 2026-11-10 mode, described next.

In conversation mode the evaluator also sees the conversation (the tool's messages shortened to 700
characters, the user's replies in full) and the test checklist built from `inputs/script.json` and `transcript.json`. Only requests
the user actually sent are in the checklist: whether each scripted reply or request was sent is decided
in code from the transcript (`scripted_request_status`), never by the model. One that was never sent
(the app took another path) is "not exercised": it is not named to the evaluator and so changes no
score, and the script's overall expectation is withheld when none of its requests was sent. A wish
that reached the app through its fallback request counts the ordinary reply as "replaced".
`evaluation.json` lists every scripted request with its status under `scripted_requests`. What the user
says about themselves in the chat counts as candidate material (claim source `chat`), except a request
to state something the user says is untrue. Still one model call per evaluation. `counts` gains
`requests`, `requests_followed`, `requests_ignored`, `untruthful_refused`, `untruthful_followed`,
`requests_not_exercised`, `layout_problems`, `wrong_language` and `language_errors`.

`RUN/evaluation/<name>/` keeps `prompt.json` (the exact request, image data replaced by file names),
`raw_response.json` (every attempt: answer text, usage, stop reason), `evaluation.json` (the answer plus
counts and deterministic PDF checks: pages, extractable text, how much of the text version is in the
PDF, and the layout measurements), `llm_calls.jsonl`, `summary.json`, `pages/` (the images) and the cost.
`evaluation.json` and `summary.json` carry a `grader` record: provider, model requested and model
returned, effort, temperature (`not settable` for Claude), grading mode, `rubric_sha256` (system prompt,
templates and answer schema), `prompt_sha256` (this exact prompt with its images), tokens (uncached input,
cache writes, cache reads, output), cost, attempts, stop reason and refusal details, request ids and
`batch_id` (set when the grading came from a batch).

Grade the same session several times under different names, side by side: `eval1` and `eval2` (the
primary grader twice, for its noise), `opus1` (`--grader-model anthropic:claude-opus-5-5`), `haiku1`
(`--grader-model anthropic:claude-haiku-5-5`). An existing `RUN/evaluation/<name>/` is never overwritten.
The company text and the PDF checks use the app's own functions (`agents.search.search_result_texts`,
`agents/pdf_text_check.py`), so the evaluator sees the same page text the writer had.

### Graders: `evals/grader.py`

| grader | role | prices per 1M tokens (input / cache write / cache hit / output) |
|---|---|---|
| `anthropic:claude-sonnet-5-5` | primary grader (default of `evaluate.py`) | $2 / $2.50 / $0.10 / $10 |
| `anthropic:claude-opus-5-5` | second grader on a fixed subset | $4 / $5 / $0.20 / $20 |
| `anthropic:claude-haiku-5-5` | trap check (default of `score_truth.py`), cheap re-grader | $0.10 / $0.125 / $0.01 / $0.50 |
| `openai:<model>` | the earlier LangChain path, e.g. `openai:gpt-6.1-sol` | `provenance.PRICES_PER_1M` |

Batch halves every price. The Anthropic prices are in `provenance.PRICES_PER_1M` and
`provenance.CACHE_WRITE_PER_1M` (Haiku's apply to prompts up to 100k tokens). Claude graders go through
the official `anthropic` SDK:

- **Pinned model.** Only the three ids above are accepted; the id the API returns is recorded per call.
- **Structured output.** `output_config.format` with a JSON schema made from the pydantic answer model
  (`json_schema_for`: every object closed and every field required; numeric limits such as the 0-10
  score ranges are not part of the API's schema subset, so the answer is validated against the pydantic
  model in code). Forced tool use is not used: it returns a 400 on Sonnet 5.5 and Opus 5.5.
- **Bounded retry.** An answer that is not valid JSON, fails validation or was cut off at `max_tokens`
  (64k) is asked for once more (`MAX_ATTEMPTS = 2`); both attempts are logged and paid. A refusal is
  not retried.
- **Thinking and effort.** Thinking is adaptive (it cannot be turned off on these models); the effort is
  set explicitly, `medium` by default (`--effort low|medium|high|xhigh|max`).
- **Temperature.** Not settable: Opus 5.5 rejects any `temperature`, and Sonnet 5.5 and Haiku 5.5 reject
  non-default values, so nothing is sent and the record says `temperature: not settable`. Repeat
  gradings (`eval1`/`eval2`) measure the noise instead. OpenAI graders run at the provider default.
- **Images.** PNG page images as Anthropic `image` blocks (base64).
- **Caching.** Breakpoints after the system prompt and after the shared block (job ad, company text,
  candidate material, work preference, user's instructions), which is identical for every arm of a
  case: grade a case's arms back to back.
- **No silent fallback.** No server-side `fallbacks`: a refusal is recorded (`stop_reason`,
  `stop_details`), `summary.json` says `refused`, nothing is written to `evaluation.json` and the exit
  code is 1. Re-grade it or report it.
- **Out of credit.** A used-up Anthropic balance ("credit balance is too low", `billing_error`) or a
  reached spend cap (`enforced_spend_limit_reached`) stops the grading like an OpenAI quota error:
  `interrupted: out of credit`, exit code 75 (`evals/credit.py`).

### Building a grading without sending it (for Batch)

Every grading is built first and sent second, so a batch submitter can take the request and leave the
sending out:

```python
from evals.evaluate import build_grading_request, finish_evaluation, write_prompt
from evals.grader import result_from_message, send
from provenance import RunRecorder

out_dir = run_dir / "evaluation" / "eval1"
request, checks, images = build_grading_request(run_dir, "anthropic:claude-sonnet-5-5", out_dir, "document-only")
params = request.anthropic_params()   # the Messages API params of one batch request (no "stream")
recorder = RunRecorder(out_dir)
write_prompt(recorder, request, "document-only", images)

# synchronous: result = send(request, recorder, node="evaluator")
# from a batch: one succeeded result's message, keyed by your custom_id
result = result_from_message(request, message, batch_id=batch.id)   # prices at the Batch rate
from evals.grader import log_attempts
log_attempts(recorder, result, request, node="evaluator")            # llm_calls.jsonl and summary totals
record = finish_evaluation(run_dir, "eval1", request, result, checks, "document-only", recorder)
```

- `build_grading_request(run_dir, grader, out_dir, mode)` renders the page images into `out_dir/pages/`
  and returns `(GradingRequest, checks, images)`; no model call. `grader` is a `"provider:model"` string
  or a `GraderSpec` (`parse_grader(name, effort)`).
- `GradingRequest` holds `grader`, `system`, `shared_text`, `document_text`, `images`, `schema`,
  `max_tokens` and `rubric_sha256`. `anthropic_params()` is the Messages API params dict;
  `langchain_messages()` the OpenAI form; `loggable()` the request for `prompt.json`; `prompt_hash()`.
- `result_from_message(request, message, seconds, batch_id)` reads one Anthropic `Message` (streamed or
  from a batch) into a `GradingResult` (`status` ok / parse_error / refused, `answer`, `attempts`,
  `cost_usd`, `record()`). `evals/batch_grade.py` (below) applies the synchronous retry rule across batches.
- `finish_evaluation(...)` writes `raw_response.json`, `evaluation.json` and `summary.json`, exactly as a
  synchronous grading does, and raises `GradingFailed` after recording a refusal or a parse error.
- The trap check has the same shape: `score_truth.build_truth_check(run_dir, grader)` returns the request and
  its context, `finish_truth_check(...)` writes `truth_check.json` (`build_check_request(grader, values)` builds
  the request alone).

### Batch grading: `evals/batch_grade.py`

The gradings and trap checks of many run folders through Anthropic's Message Batches API, at half price,
written into the same `evaluation/<name>/` folders as the synchronous path. Nothing is sent by anything but
submitting and collecting; `--dry-run` and `--status` call nothing.

```bash
# the gradings of a stage, as run_matrix lists them (section 9): submit, then run again to collect
uv run python -m evals.batch_grade --jobs ../runs-ab/pilot_gradings.json --env-file PATH/.env [--wait]
# or run folders directly
uv run python -m evals.batch_grade ../runs/*_r1 --eval-name eval1 [--grader-model anthropic:claude-opus-5-5]
uv run python -m evals.batch_grade ../runs/*_r1 --truth-check            # truth1 on Haiku
uv run python -m evals.batch_grade --jobs FILE --status | --dry-run      # no call
```

- **The same request.** Each batch request's `params` is `GradingRequest.anthropic_params()`, built by the
  synchronous code (`build_grading_request`, `build_truth_check`): same model id, effort, `max_tokens` 64k,
  adaptive thinking, JSON schema, images and cache breakpoints; no `fallbacks`. A test sends both through the
  real SDK on a mock transport and compares the bodies.
- **The same folder.** `prompt.json` and `pages/` are written at submission (as `evaluate()` writes them before its
  call); at collection the reply is read with `result_from_message(..., batch_id=)` and written by `log_attempts`
  and `finish_evaluation` / `finish_truth_check`, with the same `prompt_sha256`. The grader record adds `batch_id`,
  `batch` (`custom_id`, and per attempt the batch id, `result.type`, submitted and ended times) and
  `cost_list_usd`; `cost_usd` is the billed Batch price. Collection rebuilds the request from the folder (page
  images reused) and refuses a reply whose request hash differs from the one submitted (`request_changed`).
- **The same retry rule** (`MAX_ATTEMPTS = 2`). An answer that does not parse, and a request the API did not
  answer (`errored`, `expired`, `canceled`: not billed), go out once more in a follow-up batch; every answered
  attempt ends up in `llm_calls.jsonl` and `raw_response.json`, as after `send()`'s re-ask. A refusal is recorded
  and final, never re-routed. What still fails is listed; `--resubmit-failed` sends only unanswered ones again.
- **One window.** One run submits every due grading of its jobs; the batches split only between cases, so a case's
  arms and its Sonnet, Opus, Haiku and trap-check gradings are always in the same batch.
- **Durable.** `<runs>/batches/grading/state.json` records each request's `custom_id` (stable: grading name, run
  folder, hash), request hash and batch id before the run waits. Running the same command again collects ended
  batches and submits only what is due; a grading already on disk is never redone. A run that died inside a
  submission stops the next one until it is settled: `--adopt b003=msgbatch_...` if the console shows the batch,
  `--forget-pending` if not.
- **Ledger.** Collected costs go to `<runs>/ledger.tsv` when it exists (`eval:<name>`, `truth`, `*-failed`, and
  `*-attempt1` for a re-asked first answer), so run_matrix's caps see them.
- **Anthropic graders only.** An `openai:*` grader is refused: its LangChain path re-asks inside the call loop,
  which a batch cannot replay without a second structure; grade those with `evaluate.py`.

### Batch API support (checked 2026-11-11)

Read on 2026-11-11 before building the batch paths:

- Anthropic Message Batches
  ([batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing)): the supported models
  include `claude-sonnet-5-5`, `claude-opus-5-5` and `claude-haiku-5-5`. "Almost any request you can make to the
  Messages API can be included in a batch", including vision, system messages and extended thinking. Only `stream`,
  `speed`, `fallbacks` and `max_tokens: 0` are rejected, and the grader sends none of them. A batch holds 100,000
  requests or 256 MB. `custom_id` must match `^[a-zA-Z0-9_-]{1,64}$`. Results are `succeeded`, `errored`, `canceled`
  or `expired`, and only `succeeded` is billed. Results stay available for 29 days, at 50% of every price.
  [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) lists "Batch
  processing" under feature compatibility and is available on every model.
  [Effort](https://platform.claude.com/docs/en/build-with-claude/effort) covers Sonnet, Opus and Haiku 5.5, with
  adaptive thinking always on for Opus 5.5. The batch docs suggest the 1-hour cache for shared context, because
  batches can outlast the 5-minute cache. The grader keeps its 5-minute breakpoints, since changing them would change
  the request. Cache hits in a batch are best effort, and the recorded usage shows them.
- OpenAI Batch ([batch guide](https://developers.openai.com/api/docs/guides/batch)): it supports
  `/v1/chat/completions` and `/v1/responses`, and the line's `body` takes the endpoint's own parameters. A file holds
  one model and up to 50,000 requests or 200 MB. The window is `24h`, the discount 50%, and failed or expired
  requests go to the error file. The model pages for
  [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
  [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra) and
  [gpt-6.1-sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) each list Batch, Chat Completions,
  structured outputs and image input as supported.
  [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs) names the Batch API among
  the APIs that support them. The single-prompt writer's call, captured from `langchain-openai` 1.0.2 /
  `openai` 2.8.0, is `POST /v1/chat/completions` with `model`, `messages`, `response_format` (`json_schema`,
  strict) and `stream: false`. It sends no images, no `temperature`, no `max_tokens` and no reasoning effort, so the
  provider default applies.

## 5. Check against the persona's ground truth: `evals/score_truth.py`

```bash
uv run python -m evals.score_truth runs/p12_a7_all4_luna_r1 [--grader-model anthropic:claude-haiku-5-5] [--name truth1] [--persona p12]
```

A second signal that does not depend on the evaluator. One call to a cheap model (Claude Haiku 5.5 by
default, through the same grader layer; any `provider:model` above works) reads the final CV
and cover letter and answers, for every `deliberate_gaps` and `traps` entry of the run's `persona.json`
(found from the manifest's `persona` label, or `--persona`): `claimed_as_met` (the gap is claimed or
the trap is hit, with a verbatim quote), `not_claimed` or `unclear`. Each quote is then looked up in
the outputs in code (case, spacing, quotes and dashes ignored); a claim whose quote is not there is
downgraded to `unclear` and counted in `unverified_quotes`. `RUN/evaluation/<name>/truth_check.json`
has every item, the counts `gaps_claimed`, `traps_hit`, `unclear`, `unverified_quotes`, the cost and the
same `grader` record as an evaluation.

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
user requests, ATS) and the counts `untruthful_followed` and `requests_ignored` are among the metrics.

It also writes `RUNS/sessions_summary.json`: how many sessions actually redrafted each document
(critique-driven and user-driven, from `events.jsonl`), the removal passes, and per script the request
outcomes (followed, ignored, untruthful requests refused or followed), with the scripted requests
sent, replaced and not exercised counted separately (`scripted_sent`, `scripted_replaced`,
`scripted_not_exercised`, from each session's script and transcript). Sessions interrupted for lack of
credit are left out and listed under `interrupted`.

## 7. Single-prompt baseline: `evals/single_prompt.py`

Does the multi-agent app beat one strong model given everything at once? **Topology: exactly one model
call per case** (default `openai:gpt-6.1-sol`) and nothing else: no router, critic, grounding check,
redraft, LaTeX conversion, LaTeX repair or re-conversion call. The multi-agent default-script session it
is compared with makes a median of 22 model calls on three models.

```bash
# one case
uv run python -m evals.single_prompt --run-dir ../runs-sp/p03_a3_all4 --ad a3 --persona p03 --inputs all4 \
    [--model openai:gpt-6.1-sol] --env-file PATH/.env
# the 39-case matrix: cost first, then the run (same ledger, caps, resume and out-of-credit stop as section 2)
uv run python -m evals.run_matrix --matrix evals/matrix_single_prompt.json --out ../runs-sp --estimate
uv run python -m evals.run_matrix --matrix evals/matrix_single_prompt.json --out ../runs-sp --env-file PATH/.env
# side by side with the main round's runs
uv run python -m evals.summarize ../runs-sp --with ../runs
```

**The prompt** holds what the multi-agent session gets: the candidate files of the input set (omitted
ones are absent, exactly as in `run_case`), the job ad as published, and the company search results of
the ad's search fixture, read with the app's `agents.search.search_result_texts` (the same text the
app's company research and the evaluator read). It also holds the app's two LaTeX templates
(`templates/cv_template.tex`, `cover_letter_template.tex`). The instructions are the app's own, imported,
not copied: the goal sentences of both writer system prompts, `TRUTHFULNESS_PROTOCOL`, the writers'
fresh-draft ("creative liberty") instructions and general guidelines, the output-language and
work-arrangement rules, and the LaTeX formatters' rules (`CV_LATEX_SYSTEM_PROMPT`,
`COVER_LETTER_LATEX_SYSTEM_PROMPT`: copy the text word for word, escape specials). It carries the default
scripted user's own words verbatim (`DEFAULT_SCRIPT` in `run_case.py`): the request "Please write my CV
first, then the cover letter." and the answer to the app's question for missing contact details ("I'd
rather leave those details out."), so the prompt also asks for no placeholders. The answer has five fields: the CV as plain text and as full LaTeX, the same for the
letter, and notes. The prompt as sent is in `RUN/prompt.json`.

**PDFs, no model:** each document's LaTeX goes through the app's deterministic helpers
(`ensure_text_extractable`, `escape_text_specials`) and pdflatex (twice, as in the app). If it does not
compile, the manifest records it (`compile_failures`, `pdfs.<kind>.compile_error`) and the PDF is made
from the plain text by a fixed built-in layout (`fallback_latex`: escaped text, paragraphs as in the
text, characters the fonts lack left out and listed), marked `"source": "fallback_layout"` and listed in
`fallback_pdfs`, so the content can still be graded and the report can separate fallbacks from PDFs the
model's LaTeX made. The model's LaTeX is kept as `outputs/final_cv.tex` and `final_cover_letter.tex`.

**The run folder** has `run_case`'s layout, so `evals/evaluate.py` grades it unchanged; the plain text
versions are `final_*.txt`, so its text-vs-PDF checks work as for the app. The grader stays blind: it reads
`inputs/`, `outputs/`, `transcript.json` and `inputs/script.json` (the default script), never the manifest,
so it sees neither the method nor the model. `transcript.json` records the user's words the prompt carries as
the run's user turns, as `run_case` records the default user's (the job ad, shown as `<path to job ad>`, then
the request and the missing-details answer), with no tool messages, and lists them under `user_instructions`.
So both arms' run folders give the grader's input builder the same user wording. The manifest has `"method": "single_prompt"`, the labels
`method` and `baseline`, and `"calls": 1` (every model call made, from `llm_calls.jsonl`; a test fails if
any other call happens). `summarize.py` writes `methods_summary.json` with every metric split by method
(runs without a method are `multi_agent`) and, per single-prompt run, the pair with its baseline run and
the difference (`paired_mean_diff`).

**Models.** `--model` takes any model with a price in `provenance.PRICES_PER_1M` (or a local `ollama:` one);
the comparison uses `openai:gpt-6-luna`, `openai:gpt-6.1-terra` and `openai:gpt-6.1-sol`. The model is built
as the app builds its writers (`temperature=None`: gpt-6-luna accepts only its default). An answer that does
not parse is a failed session (`"failure": "parse_error"`, status `error: ...`): never retried, and graded
as it is ("no CV was produced").

**`matrix_single_prompt.json`** (39 cases, ids `sp_<persona>_<ad>_<input set>_sol_r1`): every persona
p01-p13 on its primary ad with the input sets `all4`, `cv_only` and `no_cv`. Each case's `baseline` is
the main matrix's default-script gpt-6-luna session with the same persona, ad and inputs (rep 1, no work
preference). Scripted-user cases are out of scope: a single prompt cannot hold a conversation, and
`cases.py` refuses a single-prompt case with a script. The file's `total_cap` (12 USD) is run_matrix's
default `--total-cap` for it.

Estimated cost (`--estimate`): **11.70 USD** for the 39 cases: 0.12 per call (a 4-7k token prompt at
$2/1M, and up to about 10k output tokens at $10/1M: both documents as text and LaTeX, notes and
reasoning) plus 0.18 per evaluation.

### Through OpenAI's Batch API: `run_matrix --sp-batch`

`--sp-batch` sends the single-prompt sessions' one call through `/v1/batches` at half price (`evals/batch_sp.py`).
`single_prompt.batch_request()` captures the exact request offline: the same LangChain `ChatOpenAI` on a mock
transport. The batch line's `body` is that request body byte for byte. On collection,
`single_prompt.py ... --batch-result FILE` runs the session on the reply, replayed through the same LangChain call
offline. It uses the same `SinglePromptResponse` parsing, `make_pdfs`, fallback layout and run-folder files.
The manifest adds `batch` (batch id, custom id, attempt, submitted, ended and collected times). `cost_usd` is
billed, and `cost_list_usd` is the list price, which every single-prompt manifest records. `llm_calls.jsonl` records
`batch_id`, `cost_list_usd` and, in both paths, the provider's `response_metadata` (`finish_reason`, `service_tier`,
`system_fingerprint`, model). An answer that does not parse is a finished `parse_error` session, never re-asked,
as without Batch. A request the API did not answer goes out once more in a follow-up batch, then is listed
(`--batch-resubmit-failed`).

## 8. Analysis notebook: `evals/analysis/analysis.ipynb`

The notebook reads one or more rounds and compares the arms. A round is a `results/<date>/` folder or a
runs folder, and several can be combined, e.g. the main round and the single-prompt runs. An arm is a
system with a writer model: multi-agent luna or terra, or single prompt with luna, terra or sol. The
notebook makes no model calls and runs offline.

```bash
uv run --group analysis jupyter execute --inplace evals/analysis/analysis.ipynb
```

- Sessions are paired within *cells* (persona, ad, input set, script and work preference). Re-gradings
  of a session are averaged first, then repeats within a cell, so each cell counts once. Cells that only
  one arm has are listed and left out.
- It reports:
  - per-dimension score distributions and invented-claim rates;
  - paired differences: bootstrap CIs, Wilcoxon and sign tests, Holm over the primary contrasts;
  - a noise floor from repeats and re-gradings;
  - breakdowns by input set, ad and persona, plus cost and latency;
  - the grader's ATS score (`ats`, 0-10; rounds graded before it, such as 2026-11-10, have none)
    next to the claims the grader flags, so a high ATS score bought by cheating shows;
  - grader calibration, when two graders graded the same sessions;
  - power, sample sizes and the cost of candidate designs.
- Figures and tables go to `evals/analysis/output/<label>/`. Set `ROUNDS` and `LABEL` in the parameter
  cell to choose the data.
- The loading and pairing code is in `evals/analysis/load.py` (standard library only). The statistics
  are in `evals/analysis/stats.py` (numpy, plus scipy when it is installed). Both are tested in
  `tests/test_analysis_load.py`.

## 9. The staged comparison (experiment plan, stages 1-3)

`evals/build_ab_matrix.py` writes two matrices (no model calls; `--check` fails if they are out of date,
and a test runs it):

- **`matrix_ab_stage12.json`**, 147 sessions: arms **A** MA-luna (the app, default script, gpt-6-luna),
  **C** SP-luna and **E** SP-sol on the 39 matched cases (p01-p13 on their primary ad x `all4`,
  `cv_only`, `no_cv`), in two stages. **pilot** (36): the 6 pilot cases (p01/a1, p03/a3, p04/a6, p11/a1,
  p12/a7, p13/a8, `all4`) x 3 arms x 2 writer runs. **core** (111): the other 33 cases x 3 arms, plus a
  second writer run of the 4 extra repeat cases.
- **`matrix_ab_stage3.json`**, 78 sessions, run only if the core run calls for it: **B** MA-terra and
  **D** SP-terra on the 39 cases.

Every session has the labels `arm`, `method`, `writer_model`, `rep`, `stage` and `case` (the matched case,
the same in every arm: `p03_a3_all4`), and its own gradings (`gradings`: `eval1` by Sonnet for every
session; `eval2` by Sonnet on the first runs of the 10 repeat cases; `opus1` on the first runs of the 12
Opus cases (6 pilot + 6 check cases); `haiku1`, `haiku2` on the 18 pilot first runs) and a trap check
(`truth1`). The check cases are 6 of the 7 non-pilot personas, two per input set, drawn with the file's
seed; the 4 extra repeat cases are the check cases with `cv_only` and `no_cv`. Stage 3's Opus grades the
first runs of 10 of the 12 Opus cases per arm (seeded draw). Within a stage, the cases come in a seeded
random order and each case's sessions back to back in random arm order: the order the gradings run in.

```bash
uv run python -m evals.run_matrix --matrix evals/matrix_ab_stage12.json --out ../runs-ab --estimate [--stage pilot] [--json]
uv run python -m evals.run_matrix --matrix evals/matrix_ab_stage12.json --out ../runs-ab --stage pilot --dry-run
uv run python -m evals.run_matrix --matrix evals/matrix_ab_stage12.json --out ../runs-ab --stage pilot --env-file PATH/.env
# or: sessions here (the single-prompt ones through OpenAI's Batch API), gradings through the Batch grader
uv run python -m evals.run_matrix --matrix evals/matrix_ab_stage12.json --out ../runs-ab --stage pilot --sessions-only --sp-batch --env-file PATH/.env
#     ... run it again until every SP session is collected (or add --batch-wait), then:
uv run python -m evals.run_matrix --matrix evals/matrix_ab_stage12.json --out ../runs-ab --stage pilot --gradings-out ../runs-ab/pilot_gradings.json
uv run python -m evals.batch_grade --jobs ../runs-ab/pilot_gradings.json --env-file PATH/.env   # again to collect
```

- **`--sp-batch`** (section 7) submits the stage's single-prompt calls in the same run that starts the app's
  sessions, before them, so both arms meet the provider in the same window. The manifest records the submitted
  and ended times. Each run collects ended batches and submits what is new. An SP case still in a batch is reported
  and graded on a later run. A case is submitted only while the ledger's spend plus the Batch-price estimates of the
  requests in flight stay within the total and stage caps. Collected sessions add their billed cost to
  `ledger.tsv` (`app`). State: `RUNS_DIR/batches/sp_openai/`.

- **`--estimate`** prints the plan's cost table: per arm, app sessions, Sonnet grading, re-gradings,
  Opus, Haiku and trap check at standard rates, the total and the total with Batch grading (50% off
  every grading line; the sessions run interactively on OpenAI), then each stage and the total against
  the plan's figures and caps. Unit prices (`AB_UNIT_USD` in `run_matrix.py`): app sessions the measured
  2026-11-10 means (luna 0.027, terra 0.085); single-prompt sessions 6k input and 10k output tokens at
  provenance's prices, rounded up to a cent (luna 0.01, terra 0.06, sol 0.12); a grading 16k input and
  12k output tokens (Sonnet 0.15, Opus 0.30, Haiku 0.008; the ATS score needs no extra input, the CV's PDF
  text was already in the prompt, and its answer replaces the earlier ATS section of about the same
  length); the trap check 0.002. Stages 1-2: **45.62 USD
  standard, 26.66 USD with Batch** (pilot 15.74 / 8.81, core 29.88 / 17.85); stage 3: 23.51 / 14.58.
  With `--sp-batch` the Batch column also halves the single-prompt sessions: stages 1-2 **23.47 USD**
  (pilot 8.03, core 15.44), stage 3 **13.41 USD**. The app's sessions stay interactive, and the cost comparison between arms uses list
  prices (`cost_list_usd`).
- **Caps** (the plan's +25%): each stage's own spend (its sessions' rows in `ledger.tsv`) and the whole
  ledger. Stages 1-2: pilot 20 / 12, core 38 / 23, total **57 / 34** (standard / Batch); stage 3: 30 / 19,
  run into the same folder, so its total cap (87 / 53) includes stages 1-2. The standard caps apply when
  the gradings run here, the Batch caps with `--sessions-only`. `--total-cap` overrides the total.
- **`--stage`** is required to run a staged matrix; `--only` narrows it further. Resume, the ledger and
  the out-of-credit stop work as in section 2; each grading is recorded as `eval:<name>`. A staged
  matrix names its own repeats and gradings, so `--repeats`, `--rescore`, `--eval-name` and
  `--truth-check` are refused. The matrix's `evaluate_args` are added to every evaluation command (for
  the grader's options, such as a document-only mode).
- **Freeze.** Every session's manifest gets the labels `app_commit`, `app_head` and `app_dirty`.
  `python -m evals.build_ab_matrix --app-commit SHA` writes the frozen commit into both files; from then on
  no session starts if the app's files (`APP_PATHS`: `agents/`, `langgraph_agent.py`, `config/`,
  `templates/`, `evals/run_case.py`, `evals/single_prompt.py`) differ from that commit.

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
