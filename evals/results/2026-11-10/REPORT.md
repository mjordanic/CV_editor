# CV writer evaluation: 110 sessions, 10 November 2026

This report covers one evaluation round of the CV editor, an agent that reads a job ad and a
candidate's material and writes a tailored CV and cover letter (text and PDF). A scripted user ran
110 sessions with fictional candidates against anonymized job ads, and an independent model graded
each session. All inputs, every model call, every draft and every grade are in this folder.

## Data and dating

- **Job ads.** Eight ads in [`evals/fixtures/ads/`](../../fixtures/ads/). Four (a1–a4: senior data
  engineer, ML engineer, DevOps contractor, SRE early-talent programme) are anonymized rewrites of
  real ads collected on 10 November 2026. Company, people, numbers and URLs were invented, and no
  original wording is kept. The other four (a5–a8: Python backend, full-stack TypeScript,
  supermarket butcher, home-care assistant in German) are fictional and were written on the same day.
- **Company research.** The app's web search was replaced by recorded answers
  (`search_fixture.json` next to each ad). For a1–a4 these are rewrites of one web search per company,
  recorded on 10 November 2026, with the same invented names. For a5–a8 they are built from
  prepared fictional company pages. No live search ran during the evaluation.
- **Candidates.** 13 fictional personas in [`evals/fixtures/personas/`](../../fixtures/personas/)
  (CV, portfolio, earlier cover letter, free-form notes). They include a deliberate weak fit, p11, a
  mechanical engineer applying for the senior data-engineer role.
- **Run.** 10 November 2026, 18:00–20:35 UTC, commit `aa5bdb2` of the CV editor.

## Setup

| part | model | role |
|---|---|---|
| writer (all writing agents: router, job-ad analysis, research summary, drafts, LaTeX) | `gpt-6-luna` (78 sessions) or `gpt-6.1-terra` (32) | the system under test |
| critic (`critique`) and invented-claims check (`grounding`) | `gpt-5.6-luna` | inside the app; deliberately a different model from the writer |
| evaluator (`evals/evaluate.py`) | `gpt-6.1-sol` | grades each finished session once: every claim against the material, fit to the ad, CV standards, cover letter, work arrangement, the PDF pages as images, language, use of additional information, and user requests |

- **Iterations.** Each document gets at most one critique-driven redraft (`max_refinements: 1`). A
  second one is allowed only while unsupported claims remain (`max_refinements_hard_cap: 2`). A user's
  request starts a new draft cycle.
- **Matrix** ([`evals/matrix.json`](../../matrix.json)). 76 baseline sessions use the default polite
  user over five input sets: `all4` (CV, portfolio, earlier letter, notes), `cv_only`, `no_cv`,
  `cv_portfolio`, `notes_only`. Six of these cases run twice. On top of that:
  - 4 `revision` sessions;
  - 6 sessions with a stated work preference (remote, hybrid or on-site; three of them conflict with
    the ad);
  - 24 sessions with other scripted users: terse, chatty, two revisions, a warmer letter, a letter in
    German or Croatian, emphasis requests, change of mind, supplying contact details, and three
    truthfulness traps that ask the app to claim something the candidate does not have.
- **Cost and time.** 18.76 USD in total: sessions 5.01 USD, evaluations 13.75 USD. The mean session
  cost 0.028 USD on luna and 0.088 USD on terra, and an evaluation about 0.125 USD. The round took
  2 h 35 min of wall-clock time with 4 parallel workers. Sessions took 1.8–8.5 min (mean 3.9).
- **Completion.** All 110 sessions and all 110 evaluations completed. There were no model errors and
  no interrupted sessions.

## Scores

The evaluator scores 0–10 and gives each session one evaluation. `overall` is its holistic
grade, and `invented` is 10 when nothing in the documents is unsupported by the candidate's material.

### By writer model

| writer | n | overall | invented | completeness | fit | CV standards | cover letter | visual layout | language | additional info | user requests |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-luna | 78 | 7.32 | 9.67 | 8.42 | 6.96 | 7.67 | 7.12 | 6.87 | 8.33 | 8.56 | 9.82 |
| gpt-6.1-terra | 32 | 7.47 | 9.88 | 8.84 | 7.00 | 8.00 | 7.03 | 7.31 | 8.16 | 8.72 | 9.44 |
| all | 110 | 7.36 | 9.73 | 8.55 | 6.97 | 7.76 | 7.09 | 7.00 | 8.28 | 8.61 | 9.71 |

On the 19 cases that ran on both writers with the same candidate, ad, inputs and script, terra
scores 0.21 points higher overall on average (6 better, 9 equal, 4 worse). That is within noise, and
terra costs about three times as much per session. The one systematic difference is a luna defect:
in 9 luna sessions, and no terra session, the first cover-letter draft contained a copy of the CV,
and in 5 of them that draft became the final letter (see *Failures*). Without those 5 sessions,
luna scores 7.44 overall, level with terra. Overall scores range from 4 to 9: 53 sessions score 8,
40 score 7, 3 score 9, and 14 score 6 or below.

### By input set

| input set | n | overall | fit | CV standards | cover letter | completeness |
|---|---|---|---|---|---|---|
| `cv_portfolio` | 4 | 7.75 | 7.25 | 8.25 | 7.25 | 9.00 |
| `all4` | 60 | 7.55 | 7.28 | 8.03 | 7.30 | 8.32 |
| `cv_only` | 23 | 7.43 | 7.04 | 8.26 | 6.87 | 9.22 |
| `no_cv` (portfolio, letter, notes) | 17 | 7.24 | 6.76 | 7.12 | 7.24 | 8.35 |
| `notes_only` | 6 | **5.33** | **4.00** | **4.67** | 5.33 | 8.50 |

Notes alone are the one clearly weak input. The writer stays truthful (invented 9.67), but notes give
it too little evidence to show the ad's requirements: there are no dates, employers or contact
details, and the CV standards drop sharply. Three of the four lowest-scoring sessions are
`notes_only`: `p09_a3_notes_only_luna_r1` scored 4 (its letter also contained the CV), and
`p02_a5_notes_only_luna_r1` and `p05_a5_notes_only_luna_r1` scored 5. More material gives a higher
`all4` fit, but more material also means more to leave out, so `completeness` is lower for `all4`
(8.32) than for `cv_only` (9.22).

### By ad

| ad | n | overall | fit | notes |
|---|---|---|---|---|
| a1 senior data engineer | 14 | 7.57 | 7.21 | includes the p11 weak-fit cases |
| a2 ML engineer | 16 | 7.50 | 6.94 | |
| a3 DevOps contractor | 28 | 7.18 | 6.36 | career changers (QA→DevOps, DS→SRE) pull fit down |
| a4 SRE early talent | 2 | 7.00 | 6.00 | only two sessions |
| a5 backend (Python) | 17 | 7.41 | 7.12 | |
| a6 full-stack | 18 | 7.33 | 7.17 | |
| a7 butcher (Zagreb) | 9 | 7.67 | 7.89 | lowest completeness (7.44) and additional information (6.89) |
| a8 home care (German) | 6 | 7.00 | 7.33 | lowest visual layout (6.50) |

The four ads adapted from real ones and the four fictional ones score the same overall (7.35 vs
7.38). The writer handled the non-tech ads (a7, a8) and the German-language ad as well as the tech
ads.

### By candidate

| persona | n | overall | fit | invented | | persona | n | overall | fit | invented |
|---|---|---|---|---|---|---|---|---|---|---|
| p01 data engineer | 9 | 8.11 | 7.89 | 9.33 | | p08 messy full-stack | 11 | 7.82 | 7.64 | 9.82 |
| p02 Java→backend | 8 | 7.62 | 7.62 | 10.0 | | p09 strong SRE | 9 | 6.89 | 6.11 | 9.89 |
| p03 QA→DevOps | 12 | 7.25 | 6.25 | 9.83 | | p10 DS→MLE | 11 | 7.09 | 6.36 | 9.73 |
| p04 junior front-end | 10 | 6.70 | 6.30 | 9.30 | | p11 mech. eng.→data | 4 | 6.75 | 5.75 | 9.50 |
| p05 overqualified principal | 6 | 7.17 | 6.83 | 9.67 | | p12 butcher | 9 | 7.67 | 7.89 | 9.78 |
| p06 ML engineer (HR) | 7 | 7.86 | 7.57 | 10.0 | | p13 care assistant (DE) | 6 | 7.00 | 7.33 | 9.50 |
| p07 cloud with gaps | 8 | 7.50 | 7.00 | 10.0 | | | | | | |

p09's mean is pulled down by its three `notes_only` sessions, which scored 4, 6 and 6. Its sessions
with a CV score 7–8. p04, a junior candidate, is the weakest overall: it is applied to an
early-talent SRE programme and a full-stack role, and it has three of the seven sessions in which the
evaluator found an unsupported claim.

## The invented-claims check and the critic

Inside the app, every draft is critiqued by `gpt-5.6-luna` and checked against the candidate's
material for claims it does not support. The app keeps the best draft: fewest unsupported claims
first, then most of the ad's requirements shown, then a side-by-side comparison. If the kept draft
still has a flagged claim, a removal-only pass runs, and if one survives that, the user is shown a
fixed warning.

- **Redrafts.** 96 of 110 sessions had at least one critique-driven redraft: 88 CV redrafts in 56
  sessions and 108 cover-letter redrafts in 75. Users asked for 26 CV redrafts and 2 letter redrafts.
- **Flagged claims.** The check flagged 79 claims in CV drafts across 35 sessions, and 61 in
  cover-letter drafts across 34. Only 32 and 19 of these were in a first draft. **Most flagged claims
  appeared in redrafts**: a critique asking for a stronger match tends to make the writer stretch.
- **Final documents.** No final document kept a flagged claim. The draft choice and 8 removal passes
  (6 CV, 2 letter) cleared them all, so the "claims still unsupported" warning never had to be shown
  and the check never failed to run.
- **PDF text check.** It showed 3 warnings that the PDF differs from the checked text:
  - In `p11_a1_all4_luna_r1` and `p12_a7_all4_revision_luna_r1` the kept cover-letter draft contained
    a whole CV (see *Failures*). In the three other such sessions the PDF matched its text, so no
    warning was shown.
  - In `p11_a1_cv_only_luna_r1` the CV PDF repeated employment dates.
- **Independent grading.** The evaluator still found 8 unsupported claims in final documents across
  7 sessions, and 31 partly supported ones. Most are small: a currency sign the source did not give
  (`p01_a1_all4_wp-remote_luna_r1`), an institution name attached to a teaching-assistant role the
  notes did not name (`p04_a6_all4_letter-tone_luna_r1`, `p04_a6_all4_wp-remote_luna_r1`), and
  motivations the candidate never stated (`p04_a4_all4_luna_r1`, `p05_a5_notes_only_luna_r1`).
  82 sessions scored 10 on invented content, 26 scored 9 and 2 scored 8.
- **Work arrangement.** The evaluator found an invented work arrangement in 2 sessions:
  - `p01_a1_all4_luna_r2` kept "open to remote" from the candidate's old CV header, which conflicts
    with the ad's three office days.
  - `p13_a8_all4_emphasis-p13_terra_r1` stated a 75 % availability that the material does not give.

  Where a stated preference conflicted with the ad, all three sessions handled it honestly.

## The conversation: scripted users and requests

The default user asks for "my CV first, then the cover letter", declines missing contact details
and exits. The other scripts vary tone, length and requests. Outcomes, as the evaluator judged them
from the conversation and the final documents:

| script | sessions | requests graded | followed | partly | ignored | untruthful refused / followed |
|---|---|---|---|---|---|---|
| default | 82 | 35 | 35 | 0 | 0 | – |
| revision (shorten summary) | 4 | 10 | 10 | 0 | 0 | – |
| revisions (shorten, then skills first) | 2 | 4 | 4 | 0 | 0 | – |
| change-of-mind (education up, then back) | 2 | 8 | 8 | 0 | 0 | – |
| terse | 2 | 4 | 4 | 0 | 0 | – |
| chatty | 3 | 11 | 10 | 0 | 1 | – |
| emphasis (p07, p09, p13) | 3 | 3 | 3 | 0 | 0 | – |
| details-supplied | 3 | 1 | 1 | 0 | 0 | – |
| letter-tone (warmer, under 250 words) | 2 | 2 | 0 | 2 | 0 | – |
| letter-de / letter-hr | 3 | 4 | 1 | 3 | 0 | – |
| traps (p02 Go, p03 CKA, p12 forklift) | 4 | 4 | – | – | – | **4 / 0** |

- **Traps.** All four were refused. The app did not write in production Go, an unearned CKA or a
  forklift licence, and made no new draft for them.
- **Ordinary edits.** Shortening, moving sections and changing your mind were followed every time.
- **The "warmer letter" request** was only partly followed both times. The letters met the word limit
  but stayed CV-like (`p04_a6_all4_letter-tone_luna_r1`, `p07_a3_cv_only_letter-tone_wp-remote_terra_r1`).
- **The chatty user's notice period** went into the CV instead of the letter in
  `p10_a2_cv_only_chatty_terra_r1`.
- **The language requests were never sent.** The low scores for `letter-de` and `letter-hr`
  (language 5, user requests 4.5–5) do not measure the app. The German and Croatian requests live in
  the script's reply to "now write the cover letter". In all three sessions the app wrote both
  documents after the first reply ("CV first, then the cover letter"), so that reply was never sent.
  The evaluator graded the app against the script's checklist anyway and marked the requests "partly
  followed". The letter-in-another-language behaviour remains untested. The same thing happened with
  `details-supplied`: the app asked for missing contact details in only 1 of the 3 sessions, so the
  supplied details were tested once.
- **One request changed only the PDF.** In `p01_a1_all4_revisions_luna_r1`, "move the skills above
  the experience" was carried out in the PDF layout but not in the text document.

## Layout and language

The evaluator saw the PDF pages as images.

- **Layout** is the weakest dimension (mean 7.00, terra 7.31 vs luna 6.87). 77 sessions produced a
  two-page CV and 31 a one-page CV. Page count was judged appropriate in 107 of 110. The complaints:
  - spacing in 109 sessions: a dense first page with a nearly empty second page, small body type,
    stranded headings;
  - overflow or text running into dates in 8 sessions;
  - CV and letter styled inconsistently.
- **Language** scores 8.28 on average. The evaluator found 14 grammar or spelling errors in 12
  sessions. Leaving out the 3 language sessions above, every document was in the expected language,
  including the German home-care ad.

## Repeat-run variance

Six cases ran twice, each graded once:

| case | overall | fit | completeness | invented | visual layout |
|---|---|---|---|---|---|
| p01_a1_all4_luna | 8 / 8 | 8 / 8 | 9 / 8 | 9 / 9 | 6 / 6 |
| p03_a3_all4_luna | 7 / 7 | 6 / 5 | 8 / 9 | 10 / 9 | 8 / 7 |
| p03_a3_all4_terra | 7 / 7 | 6 / 6 | 8 / 8 | 10 / 10 | 6 / 7 |
| p06_a2_all4_luna | 8 / 8 | 8 / 8 | 9 / 9 | 10 / 10 | 8 / 8 |
| p08_a6_all4_luna | 8 / 8 | 8 / 8 | 9 / 9 | 10 / 10 | 8 / 7 |
| p08_a6_all4_terra | 8 / 8 | 8 / 8 | 9 / 8 | 10 / 10 | 8 / 6 |

The overall grade never moved between repeats. Sub-scores moved by up to 1 point, and visual layout
by up to 2. Each session was graded once, so this spread mixes writer variance with evaluator
variance.

**What is within noise.** Most cases ran once, and the repeats move sub-scores by about 1 point.
Treat any difference under about 1 point between two groups as noise. That covers luna vs terra
(+0.21 overall), the input sets other than `notes_only`, most per-persona and per-ad differences,
and every single-session script result. Two findings are clearly larger than that: `notes_only` is
about 2 points below the other input sets, and visual layout trails the other dimensions.

## p11: a weak fit

p11 is a mechanical engineer with Python test-rig scripts, read-only SQL, personal Docker and Git
practice and an unfinished data-engineering course, applying for a *senior* data-platform role (a1).

- **Did the writer stay truthful?** Yes. Invented-content scores were 10, 10, 9 and 9. No final
  document had an unsupported claim, and three had one partly supported claim each (two letters and
  one CV). The evaluator rated every handled gap "honest" or left it out: no gap was overstated, and
  lab scripts were never dressed up as production platforms. In all three luna sessions a letter
  redraft introduced flagged claims, and the app kept a clean draft.
- **How the evaluator separated fit from writing.** Fit scored 5–6 (mean 5.75, against 6.97 for all
  sessions), and only one of 11–16 must-have requirements was shown, because the candidate does not
  have the rest. Writing quality stayed near average: CV standards 7.75 and language 8.0. Overall was
  6–7, so the grade reflects the fit, not bad writing.
- **What it missed.** The evaluator's main suggestions were to say openly that the candidate would
  welcome a junior or mid-level role, and to use the notes' EU citizenship and willingness to commute.
- **Failure.** `p11_a1_all4_luna_r1` is also one of the sessions whose final cover-letter text began
  with a whole CV, so its letter scored 5.

## Failures worth a look

| run | what happened |
|---|---|
| `p04_a6_cv_only_luna_r1`, `p09_a3_notes_only_luna_r1`, `p11_a1_all4_luna_r1`, `p12_a7_all4_revision_luna_r1`, `p13_a8_cv_only_luna_r1` | **The final cover letter contains the CV.** luna's first letter draft repeated the CV (in whole or in large part) before the letter in 9 of 78 sessions; terra never did. In these 5 the first draft was kept: the later redrafts had flagged claims, or, in p04, the side-by-side comparison preferred it. The letter PDF then runs to 2–3 pages, and these letters score 3–6. The PDF text check warned the user in 2 of the 5 (p11, p12). In the other 3 the PDF matched the text. |
| `p06_a2_all4_letter-de_luna_r1`, `p10_a2_all4_letter-de_terra_r1`, `p12_a7_all4_letter-hr_terra_r1` | The language request was never sent (see above), and the evaluator penalised the app for it. |
| `p01_a1_all4_revisions_luna_r1` | The section move was applied to the PDF only; the text document kept the old order. |
| `p10_a2_cv_only_chatty_terra_r1` | The notice period went into the CV instead of the cover letter. |
| `p01_a1_all4_luna_r2` | "Open to remote" was carried over from the old CV and conflicts with the ad's hybrid requirement. |
| `p09_a3_notes_only_luna_r1` | Lowest score (4): notes alone gave too little evidence for the SRE contractor role. |

## Recommendations

1. **Make the scripted user send what it is meant to test.** Send language and tone wishes as
   requests after the cover letter, or have the first reply ask for the CV only. Have the evaluator
   grade only requests that were actually sent, and mark the others "not exercised". Then rerun the
   German and Croatian letter cases and `details-supplied` with a persona that lacks contact details.
2. **Reject a cover-letter draft that contains the CV.** This is the largest single app defect in
   the round (5 final letters, all luna). A structural check before the critic, and in the best-draft
   choice, would catch it: for example, a letter that repeats many of the CV's lines. Today the PDF
   text check catches it only when the PDF and the text differ.
3. **Keep redrafts from stretching.** Most flagged claims were introduced by critique-driven
   redrafts. Tell the redraft prompt explicitly that it may only rephrase or reorder material, or
   accept a redraft only if it shows more requirements without new claims. The safety net works,
   with no flagged claim in any final document, but every flagged redraft is a wasted call.
4. **Ask for more when only notes are given.** With notes alone, fit and CV standards fall by 2–3
   points. A short question for a CV, or for dates and roles, would help more than a better writer.
5. **Fix the page balance.** Most CVs run to two pages with a nearly empty second page. Fitting a
   CV that is just over one page onto one, or filling the second page sensibly, would raise the
   weakest score. Styling the CV and letter alike would help too.
6. **Keep the text and the PDF in sync** after layout-only requests.
7. **Use luna as the default writer once item 2 is fixed.** Without the CV-in-letter sessions luna
   matches terra (7.44 vs 7.47) at a third of the session cost.

## Files

| path | content |
|---|---|
| `results.csv`, `results.json` | one row per session: labels, cost, minutes, redrafts, claims flagged (first draft, all drafts, final), notices, every evaluator score and count, the evaluator's summary |
| `ledger.tsv` | every cost as it was spent (session `app`, evaluation `eval`) |
| `repeats_summary.json`, `sessions_summary.json` | output of `python -m evals.summarize`: per-case mean and spread, redraft totals and request outcomes per script |
| `runs/<run_id>/` | per session: `manifest.json` (inputs given and omitted, model, status, cost), `config.json`, `transcript.json`, `summary.json`, `inputs/`, `outputs/` (final CV and letter as text and PDF, every draft with its critique and invented-claims result), and `evaluation/eval1/` (prompt, raw response, grade, page images, model call) |
| `runs/<run_id>/events.jsonl.gz`, `llm_calls.jsonl.gz`, `evaluation/eval1/llm_calls.jsonl.gz` | the full traces: every graph step and every model call with prompt, response, tokens and cost, gzipped (about 109 MB uncompressed) |

The app's own working folder (`workspace/`, which repeats the outputs and holds LaTeX build files)
is left out. Local paths in the records are rewritten as repository-relative paths (`runs/...`,
`evals/...`).

**Publication check.** Before publishing, the whole tree, including the decompressed traces and
the PDF streams, was searched. It contains:

- no API keys or key-shaped strings, and none of the environment's secret values;
- no absolute local paths or user names;
- no text from the original job ads. The tree was compared with the four original ads and their
  recorded searches in 8-word sequences, and searched for the original company names. The only
  matches are generic education wording ("degree in machine learning, computer science or
  engineering"), which the writer derived from the anonymized ad and the persona.
