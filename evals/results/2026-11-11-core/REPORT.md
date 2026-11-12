# CV writer evaluation, stage 2 (core): 111 sessions, 11 November 2026

This round is stage 2 of the staged comparison (`evals/README.md`, section 9). With the stage 1 pilot
([`../2026-11-11/`](../2026-11-11/REPORT.md)) it completes the comparison of three systems on 39 matched cases:

| arm | system | writer |
|---|---|---|
| A MA-luna | the app (multi-agent, default script) | gpt-6-luna |
| C SP-luna | one call writes CV and letter (`evals/single_prompt.py`) | gpt-6-luna |
| E SP-sol | one call | gpt-6.1-sol |

The 39 cases are the 13 personas, each on its primary ad, with three input sets: `all4`, `cv_only` and `no_cv`. The
pilot ran 6 of them (`all4`). This round ran the other 33 in every arm. It also ran a second writer run of 4 repeat
cases: p05/a5 `cv_only`, p08/a6 `no_cv`, p09/a3 `cv_only` and p10/a2 `no_cv`. Every result below combines the two
rounds unless it says otherwise.

## Data and dating

- **Inputs.** Only the tracked fixtures: the job ads in [`evals/fixtures/ads/`](../../fixtures/ads/) and the
  fictional personas in [`evals/fixtures/personas/`](../../fixtures/personas/). Four ads are anonymized rewrites of
  real ads collected on 10 November 2026; the others are fictional, written on the same day.
- **Company research.** Replayed from each ad's `search_fixture.json`, recorded or written on 10 November 2026. No
  live search ran.
- **Writing.** 11 November 2026, 21:20 to 22:09 UTC. OpenAI, direct calls (no Batch).
- **Grading.** 11 November 2026, collected 22:16 to 22:48 UTC. Anthropic Message Batches (half price),
  document-only mode, effort `medium`.
- **App version.** Commit `25b7876`, frozen in `matrix_ab_stage12.json` (`app_commit`). Every session's manifest
  records `app_commit`, `app_head` and `app_dirty: False`.

## What ran

- **111 sessions**, 37 per arm, all completed:
  - **Single-prompt:** every answer parsed (`finish_reason: stop` in 74 of 74).
  - **App:** no session was degraded and no model call failed. Critique-driven redrafts touched the CV in 22 of 37
    sessions and the letter in 30.
- **LaTeX fallbacks.** The single-prompt arms get no LaTeX repair, by design. When the model's LaTeX does not compile,
  the built-in layout is used:
  - SP-luna: 6 of 37 sessions. Over the 39 first runs it is 6 of 39 (15%).
  - SP-sol: 1 of 37 (a second run). Over the 39 first runs it is 0 of 39.
  - Over every single-prompt session of both rounds: 7 of 98.
- **252 gradings in the Batch grader, all answered:**
  - `eval1`: Sonnet 5.5 on all 111 sessions, the primary grade.
  - `eval2`: Sonnet again on the 12 first-run sessions of the repeat cases.
  - `opus1`: Opus 5.5 on the 18 first-run sessions of the six check cases (the 4 repeat cases plus p06/a2 and p07/a3,
    `all4`).
  - `truth1`: the Haiku trap check on all 111.

  One case went out first, to measure the cost, then the other 234.

### Re-asked gradings

Eight Sonnet `eval1` gradings, seven here and one of the pilot's, were asked a second time, once each, through the same
Batch grader:
- **Why.** The first answer put the whole grading inside the full copy (`full_evaluation_json`), which was invalid
  JSON, and filled the guaranteed fields with placeholders: every score 0 and the summary "..." (3 gradings), or no
  claims at all (5 gradings).
- **Why the retry rule missed them.** The guaranteed fields parsed, so the grader's own retry (one re-ask for an answer
  that does not parse) did not run.
- **Which.**
  - Placeholder scores: `ma_luna_p03_a3_no_cv_r1`, `sp_luna_p12_a7_cv_only_r1`, `sp_luna_p04_a6_no_cv_r1`.
  - No claims: `sp_sol_p06_a2_cv_only_r1`, `sp_sol_p13_a8_no_cv_r1`, `ma_luna_p08_a6_no_cv_r2`,
    `sp_luna_p05_a5_cv_only_r2`, and the pilot's `ma_luna_p12_a7_all4_r2`.
- **What is published.** The second answers, all well formed (every score between 6 and 10, claims listed). The first
  answers are kept outside the repository, and both answers' costs are in `ledger.tsv`.
- **Effect.** The three placeholder grades had decided the truthfulness test of SP-luna. With them, C−A invented content
  had a persona-cluster interval of −0.69 to +0.97, so non-inferiority was not shown. With the re-asked grades it is
  −0.08 to +0.46. E−A overall was +0.59 with them (a placeholder 0 for A on one case) and is +0.41 now. Its
  Holm-adjusted p is 0.022.

## Results (stages 1 + 2, 39 cases, primary grade: Sonnet `eval1` on the first writer run)

| arm | overall | composite* | invented content | fit | visual layout | ATS | sessions with a flagged claim | partly-supported claims per 100 | unsupported claims |
|---|---|---|---|---|---|---|---|---|---|
| A MA-luna | 7.46 | 7.59 | 9.05 | 6.79 | 7.31 | 7.41 | 25 of 39 | 8.2 | 0 |
| C SP-luna | 7.56 | 7.65 | 9.26 | 6.74 | 7.28 | 7.31 | 19 of 39 | 4.5 | 0 |
| E SP-sol | 7.87 | 7.97 | 9.38 | 6.92 | 7.69 | 7.33 | 15 of 39 | 3.3 | 0 |

\* The mean of invented content, completeness, fit, CV standards, cover letter, visual layout and language.

### Primary comparisons

Differences are B − A, paired by case (39 pairs). The interval is the 95% bootstrap interval resampling whole
personas. A claim needs both the Wilcoxon test on cases and the sign-flip test on personas; their larger p is
Holm-adjusted over the fixed family of four primary comparisons. A−B (MA-terra) and B−D are not run, so each counts as
not rejected (p = 1).

| measure | C − A | Holm p | E − A | Holm p |
|---|---|---|---|---|
| invented content | +0.21 (−0.08 to +0.46) | 0.78 | +0.33 (+0.10 to +0.56) | 0.23 |
| fit | −0.05 (−0.28 to +0.18) | 1 | +0.13 (−0.05 to +0.31) | 1 |
| visual layout | −0.03 (−0.31 to +0.28) | 1 | +0.38 (0.00 to +0.79) | 0.50 |
| **overall** | +0.10 (−0.13 to +0.33) | 1 | **+0.41 (+0.26 to +0.56)** | **0.022** |

- **E is better than A on overall.** It wins 16 cases, ties 19 and loses 4; Wilcoxon p = 0.005, persona sign-flip
  p = 0.002. The difference, about 0.4 points, is below the plan's pre-registered minimum effect of 1 point.
- **Truthfulness, non-inferiority (margin −0.5).** Both single-prompt arms pass. The lower bound of C − A is −0.08 and
  of E − A is +0.10.
- **C and A do not differ** on any primary measure, and every interval lies inside ±1 point.
- **Fixed-sequence reading.** If one verdict per comparison is wanted, the plan's fixed order is invented content, fit,
  layout, overall. E − A then makes no claim, because invented content (Holm 0.23) comes first.

**Secondary measures** (not adjusted):
- **Partly-supported claims per session:** C − A −0.49 (persona interval −0.80 to −0.15); E − A −0.67 (−0.97 to −0.36).
  Both single-prompt arms stretch fewer claims than the app.
- **Fully unsupported claims:** none in any arm.
- **Cover letter:** E − A +0.87 (+0.64 to +1.10; 25 better, 14 tied, 0 worse).
- **Language:** C − A +0.46 and E − A +0.54.
- **ATS score:** no difference (C − A −0.10, E − A −0.08).

**By input set** (overall, mean difference over 13 cases each):

| | `all4` | `cv_only` | `no_cv` |
|---|---|---|---|
| C − A | +0.46 | −0.23 | +0.08 |
| E − A | +0.46 | +0.31 | +0.46 |

**Without the LaTeX fallbacks.** On the 33 cases where SP-luna's first run compiled, C − A is:
- invented content +0.15 (−0.09 to +0.39);
- fit −0.15;
- visual layout +0.15;
- overall +0.09.

None of them differs. The six fallback sessions average 6.5 on visual layout, against 7.4 for SP-luna's other sessions.

### Trap check (`truth1`, first runs, 39 sessions per arm)

For each persona's deliberate gaps and traps, the check asks whether the delivered PDFs claim the gap as met or fall
into the trap. Traps that do not apply to a session's input set are skipped (`not_applicable`; p01 T6 and p04 T6
here).

| arm | sessions with a hit (95% CI) | gaps claimed | traps hit | items checked |
|---|---|---|---|---|
| A MA-luna | 8 (21%, 11–36%) | 5 | 4 | 471 |
| C SP-luna | 8 (21%, 11–36%) | 2 | 7 | 471 |
| E SP-sol | 5 (13%, 6–27%) | 2 | 3 | 471 |

The paired difference of the hit rate is +0.000 for C − A (persona interval −0.009 to +0.010) and −0.007 for E − A
(−0.024 to +0.009). No arm claims clearly more traps.

Every hit quotes the PDF text verbatim (0 unverified quotes). Several of the traps are about presentation rather than
invented facts:
- keeping a director-level CV for an individual-contributor ad: p05, 1 in A and 3 in C;
- dropping the qualifiers on weak skills ("Vue (a little)"): p08, 2 in A and 2 in C;
- not condensing duplicated sections: p08, 1 in A.

The gaps claimed:
- **A:** p11's "PLC programming" twice and "production software" once; p05's "recent hands-on coding"; p07's "beyond
  Terratest".
- **C:** p11's "production software"; p06's "LLMs in production".
- **E:** the same two as C.

## Grader calibration (plan section 6; 36 sessions graded by Sonnet and Opus, 30 graded twice by Sonnet)

| check | metric | result | threshold | verdict |
|---|---|---|---|---|
| Sonnet noise | composite SD between `eval1` and `eval2` (30 sessions) | 0.193 | ≤ 0.3 | pass |
| | overall within 1 point | 30 of 30 (23 exact) | ≥ 90% | pass |
| Sonnet vs Opus | Spearman ρ of the composite (36 sessions) | 0.74 | ≥ 0.7 | pass |
| no bias by arm, composite | C−A: Sonnet +0.20, Opus +0.08; E−A: Sonnet +0.33, Opus +0.49 | gaps 0.12 and 0.16; same sign in 8 of 9 and 9 of 9 non-tied cases | ≤ 0.3, ≥ 80% | pass |
| no bias by arm, overall | C−A: Sonnet +0.42, Opus 0.00; E−A: Sonnet +0.50, Opus +0.58 | gaps 0.42 and 0.08; same sign in 3 of 4 and 3 of 3 | ≤ 0.3, ≥ 80% | **C−A fails** |
| claim audit | Opus's flagged claims that Sonnet `eval1` also flags | 4 of 12 (33%) | ≥ 80% | fails (automatic match) |

- **Levels.** Opus grades the composite 0.12 higher than Sonnet (95% CI +0.02 to +0.21) and overall 0.11 lower. The
  Spearman ρ of the integer overall grade is 0.43.
- **Bias by arm on the other primary measures** (12 cases each). Invented content: Opus sees larger single-prompt
  advantages than Sonnet (C−A +0.58 against +0.25; E−A +0.67 against +0.33). Fit and visual layout: gaps of 0.08 to
  0.25. Fit has only 2 non-tied cases.
- **Overall, C−A.** The failure rests on 4 non-tied cases: Sonnet +0.42 (interval −0.08 to +1.00), Opus 0.00
  (−0.42 to +0.42). On the composite, the agreement measure, both graders agree in sign and size.
- **Claim audit.** Opus flagged 12 claims in 10 of the 36 sessions, all partly supported and none unsupported. Sonnet
  flagged 24 and flagged something in 8 of Opus's 10 sessions. Of Opus's eight claims without an automatic match:
  - three are headlines or titles: "Backend Engineer (Python) | Principal Software Engineer", "Senior Full-Stack /
    Frontend Engineer", "DevOps / Platform Engineering | QA Automation Engineer";
  - one is a skill named more strongly than the inputs support: "systems-design experience";
  - two are years or dates: "seven years in QA engineering", "in butchery since 2015";
  - one is a vague collaboration claim: "collaboration with engineering and BI users";
  - one is a claim Sonnet saw and judged supported: "Designed and built hub-spoke landing-zone networking in Terraform".

  Every one is a minor stretch. The plan asks a person to read them (about 30 minutes); the quotes are in each
  session's `evaluation/opus1/evaluation.json` and `evaluation/eval1/evaluation.json`.

## Noise (notebook, all gradings and runs; 30 repeat cells)

| measure | within-cell SD | grader SD | writer SD | SD of the C−A difference | arm × case |
|---|---|---|---|---|---|
| overall | 0.38 | 0.34 | 0.24 | 0.65 | 0.36 |
| invented content | 0.39 | 0.39 | 0.20 | 0.79 | 0.56 |
| fit | 0.41 | 0.45 | 0.13 | 0.64 | 0.27 |
| visual layout | 0.66 | 0.41 | 0.56 | 1.23 | 0.80 |

Grader noise is at least as large as writer noise, except on visual layout. SP-sol is the most stable arm: within-cell
SD 0.27 on overall, against 0.43 for MA-luna and 0.42 for SP-luna.

## Full-copy health (Claude graders, `full_evaluation`)

| grading | gradings | full copy did not parse | gradings with disagreements | disagreements |
|---|---|---|---|---|
| `eval1` (Sonnet) | 111 | 7 | 11 | 11 |
| `eval2` (Sonnet) | 12 | 0 | 5 | 6 |
| `opus1` (Opus) | 18 | 0 | 7 | 7 |

The seven unparsed copies are invalid JSON. Their gradings count on their guaranteed fields, which are complete (scores
and claims), but their omissions, layout and language counts are empty.

## Costs

| | sessions | mean per session (`AB_UNIT_USD`) | total |
|---|---|---|---|
| MA-luna (OpenAI) | 37 | $0.0275 ($0.0286, −4%) | $1.02 |
| SP-luna (OpenAI) | 37 | $0.0027 ($0.0028, −4%) | $0.10 |
| SP-sol (OpenAI) | 37 | $0.0436 ($0.0439, −1%) | $1.61 |

| grader (Batch price) | gradings | mean per grading (`AB_UNIT_USD` / 2) | total |
|---|---|---|---|
| Sonnet 5.5 | 123, plus 7 first answers re-asked | $0.0451 ($0.0465, −3%) | $5.75 |
| Opus 5.5 | 18 | $0.174 ($0.176, −1%) | $3.13 |
| trap check (Haiku) | 111 | $0.0004 ($0.0005) | $0.05 |

Total spend in `ledger.tsv`: **$11.66**.
- OpenAI $2.73.
- Anthropic $8.93.

The estimate was $11.73. By arm (writing / grading): A $1.02 / $2.93, C $0.10 / $3.02, E $1.61 / $2.97.

## Files

| path | content |
|---|---|
| `results.csv`, `results.json` | one row per session: labels, status, cost, minutes, calls, failure and fallbacks, and the primary grading (Sonnet `eval1`): every score and count, full-copy health, the grader's summary |
| `gradings.csv`, `gradings.json` | one row per grading (`eval1`, `eval2`, `opus1`), same columns |
| `truth_checks.csv`, `truth_checks.json` | one row per trap check (`truth1`) |
| `ledger.tsv` | every cost as it was spent (`app`, `eval:<name>`, `truth`) |
| `repeats_summary.json`, `sessions_summary.json` | output of `python -m evals.summarize` |
| `runs/<run_id>/` | per session: `manifest.json`, `config.json`, `transcript.json`, `summary.json`, `inputs/`, `outputs/`, and `evaluation/<name>/`: each grading's prompt, raw response, grade or trap check, summary and page images |

The notebook's tables and figures for stages 1 and 2 together are in two folders:
- [`evals/analysis/output/2026-11-11-stages12-primary/`](../../analysis/output/2026-11-11-stages12-primary/): the
  primary grade, `eval1` on the first run. The comparisons and the trap check above come from here.
- [`evals/analysis/output/2026-11-11-stages12/`](../../analysis/output/2026-11-11-stages12/): every grading and run,
  averaged as the notebook does. Noise and calibration come from here.

Not published:
- the app's working folder (`workspace/`);
- the full traces: `events.jsonl`, every `llm_calls.jsonl` and the single-prompt writer's `prompt.json`;
- the batch state;
- the first answers of the re-asked gradings.

These are kept outside the repository. Local paths in the records are rewritten as repository-relative paths
(`runs/...`, `evals/...`).

**Publication check.** The whole tree was searched, including the PDFs' extracted text and the page images. It contains
no API keys, key-shaped strings or secret values from the environment. It has no absolute local paths and no user
names. Every input is a tracked fixture, so no original ad text is in this folder.
