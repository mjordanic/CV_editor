# CV writer evaluation, stage 1 pilot: 36 sessions, 11 November 2026

This round is the pilot of the staged comparison (`evals/README.md`, section 9). It asks whether the multi-agent app
beats one model call writing both documents, and it calibrates the graders before the main run. Three arms ran on six
matched cases, two writer runs each:

| arm | system | writer |
|---|---|---|
| A MA-luna | the app (multi-agent, default script) | gpt-6-luna |
| C SP-luna | one call writes CV and letter (`evals/single_prompt.py`) | gpt-6-luna |
| E SP-sol | one call | gpt-6.1-sol |

The cases are p01/a1 (strong match), p03/a3 (career changer), p04/a6 (junior), p11/a1 (weak fit), p12/a7 (butcher,
Croatian ad in English) and p13/a8 (German home care), all with the `all4` input set. Six pilot cases are too few for
claims about the arms. The comparisons below describe the pilot only. The main comparison is stage 2.

## Data and dating

- **Inputs.** Only the tracked fixtures: the job ads in [`evals/fixtures/ads/`](../../fixtures/ads/) and the
  fictional personas in [`evals/fixtures/personas/`](../../fixtures/personas/). The ads a1 and a3 are anonymized
  rewrites of real ads collected on 10 November 2026. The others are fictional, written on the same day.
- **Company research.** Replayed from each ad's `search_fixture.json`, recorded or written on 10 November 2026. No
  live search ran.
- **Writing.** 11 November 2026, OpenAI, synchronous calls (no Batch).
- **Grading.** 11 November 2026, Anthropic Message Batches (half price), document-only mode, effort `medium`.
- **App version.** Commit `25b7876`, frozen in `matrix_ab_stage12.json` (`app_commit`). Every session's manifest
  records `app_commit`, `app_head` and `app_dirty: False`.

## What ran

- **36 sessions:** 12 per arm, all completed. Every single-prompt answer parsed (`finish_reason: stop` in 24 of 24).
  No single-prompt LaTeX fell back to the built-in layout (0 of 24). No app session was degraded and no model call
  failed. In the app, critique-driven redrafts touched the CV in 7 of 12 sessions and the letter in all 12.
- **144 batched gradings, all answered on the first attempt:**
  - `eval1`: Sonnet 5.5 on all 36 sessions, the primary grade.
  - On the 18 first-run sessions: `eval2` (Sonnet again), `opus1` (Opus 5.5), and `haiku1`, `haiku2` (Haiku 5.5).
  - `truth1`: the Haiku trap check on all 36.

  They went out in two batches: one case first, to measure the cost, then the remaining 120.

## Results by arm (primary grade, Sonnet `eval1`, 12 sessions each)

| arm | overall | composite* | invented content | fit | visual layout | ATS | sessions with a flagged claim | partly-supported claims | unsupported claims |
|---|---|---|---|---|---|---|---|---|---|
| MA-luna | 7.58 | 7.65 | 9.00 | 7.08 | 7.00 | 7.50 | 7 | 15 | 0 |
| SP-luna | 8.17 | 8.05 | 9.42 | 7.25 | 7.50 | 7.92 | 6 | 8 | 0 |
| SP-sol | 8.17 | 8.11 | 9.58 | 6.92 | 7.83 | 7.75 | 1 | 2 | 0 |

\* The mean of invented content, completeness, fit, CV standards, cover letter, visual layout and language.

Paired by case (6 pairs; the two runs averaged first):
- **Overall:** SP-luna minus MA-luna is +0.42 (5 better, 1 tied). SP-sol minus MA-luna is +0.58 (4 better, 2 tied).
- **Invented content:** +0.54 and +0.67.
- **Visual layout:** +0.50 and +1.00.

No contrast is significant at this size: Holm-adjusted p ≥ 0.25. The notebook's tables are in
[`evals/analysis/output/2026-11-11/`](../../analysis/output/2026-11-11/).

**Trap check (`truth1`).** The raw counts are:

| arm | gaps claimed | traps hit |
|---|---|---|
| MA-luna | 3 | 2 |
| SP-luna | 1 | 1 |
| SP-sol | 0 | 2 |

Every hit's quote was found in the PDF text. Five of these hits are p01's trap T6, "attributing Debezium/CDC when
portfolio and notes are omitted". In `all4` the portfolio and the notes do mention Debezium, so these hits are not
errors: the check does not know which input set a session had. Without T6 the real hits are:
- **MA-luna, 3:** the PLC gap of p11 twice (setpoint changes in TIA Portal written as PLC work), and p11's
  "production software" gap once.
- **SP-luna, 1:** p11's "production software" gap.
- **SP-sol, 0.**

## Grader calibration (plan section 6, pass thresholds fixed before the pilot)

All on the 18 first-run sessions. The composite is the mean of 7 sub-scores (above).

| check | metric | result | threshold | verdict |
|---|---|---|---|---|
| Sonnet noise | composite SD between `eval1` and `eval2` | 0.195 | ≤ 0.3 | pass |
| | overall within 1 point | 18 of 18 (12 exact) | ≥ 90% | pass |
| Sonnet vs Opus | Spearman ρ of the composite | 0.78 | ≥ 0.7 | pass |
| no bias by arm, composite | C−A: Sonnet +0.57, Opus +0.36; E−A: Sonnet +0.48, Opus +0.69 | gaps 0.21 and 0.21; same sign in 5 of 5 and 4 of 4 non-tied cases | ≤ 0.3, ≥ 80% | pass |
| no bias by arm, overall | C−A: Sonnet +1.00, Opus +0.17; E−A: +0.67 under both | gaps 0.83 and 0.00; same sign in all non-tied cases (2 each) | ≤ 0.3, ≥ 80% | C−A fails |
| claim audit | Opus's flagged claims that Sonnet `eval1` also flags | 3 of 7 (43%); with `eval2` as well, 5 of 7 | ≥ 80% | fails, see below |
| Haiku noise | composite SD `haiku1` vs `haiku2`; overall within 1 point | 0.245; 18 of 18 | as Sonnet | pass |
| Haiku vs Opus | Spearman ρ of the composite | 0.89 | ≥ 0.7 | pass |
| | bias by arm, composite / overall | gaps 0.21, 0.14 / 0.17, 0.33 | ≤ 0.3 | overall E−A fails |
| | claim audit | 2 of 7 (29%) | ≥ 80% | fails |

**Levels.**
- Opus grades the composite 0.12 higher than Sonnet. The 95% CI is −0.03 to +0.26.
- Haiku grades about 0.9 lower than Opus on the composite and 1.2 lower on overall. Against Sonnet, Haiku is 1.2
  lower on overall and 1.6 lower on fit.

A constant offset leaves paired differences unchanged, but Haiku's levels are not comparable to Sonnet's or Opus's.

**The claim audit.** Opus flagged 7 claims in 5 sessions, all partly supported and none unsupported. Sonnet `eval1`
flagged 16, also all partly supported. Sonnet's first grading missed four of Opus's claims:
- the letter's "seven years in QA engineering", where one year was customer support;
- "worked in butchery since 2015", which blurs a counter-assistant job (Sonnet's `eval2` flags it);
- a headline that puts "DevOps / Platform Engineering" before the current title;
- "collaboration with engineering and BI users" (Sonnet's `eval2` flags it).

Every miss is a minor stretch. Sonnet flagged something in 4 of the 5 sessions where Opus did. The plan asks a person
to read the quotes of the disagreements before the gate is settled.

**Overall vs composite.** The overall grade is an integer that is mostly 7 or 8. With 6 cases, a one-point change in
one session moves a contrast's mean by 0.17. The composite, which the threshold table names for agreement, passes.

## Full-copy health (Claude graders, `full_evaluation`)

| grading | gradings | full copy did not parse | gradings with disagreements | disagreements |
|---|---|---|---|---|
| `eval1` (Sonnet) | 36 | 1 | 6 | 6 |
| `eval2` (Sonnet) | 18 | 0 | 5 | 5 |
| `opus1` (Opus) | 18 | 0 | 2 | 2 |
| `haiku1` (Haiku) | 18 | 1 | 13 | 26 |
| `haiku2` (Haiku) | 18 | 1 | 13 | 26 |

The three unparsed copies were invalid JSON. One was Sonnet's copy for `ma_luna_p12_a7_all4_r2`, cut off at 3,649
characters. Those gradings still count on their guaranteed fields, but their omissions, layout and language counts are
empty.

## Noise (notebook, overall grade)

Within-cell SD across the two writer runs is 0.39. Its parts:
- **Grader**, from `eval1` and `eval2`: 0.41.
- **Writer:** 0.17.
- **Arm × case interaction:** estimated 0, from the SP-luna − MA-luna pairs (SD of the difference 0.30).

Per sub-score, the grader SD is 0.24 to 0.47 and the writer SD 0 to 0.47. Repeat runs of SP-sol were the most stable
(within-cell SD 0.20 on overall). The full table is `tables/noise_components.csv`.

## Costs

| | sessions | mean per session (plan) | total |
|---|---|---|---|
| MA-luna (OpenAI) | 12 | $0.0286 ($0.027, +6%) | $0.343 |
| SP-luna (OpenAI) | 12 | $0.0028 ($0.01, **−72%**) | $0.034 |
| SP-sol (OpenAI) | 12 | $0.0439 ($0.12, **−63%**) | $0.527 |

| grader (Batch price) | gradings | mean per grading (plan) | total |
|---|---|---|---|
| Sonnet 5.5 | 54 | $0.0465 ($0.075, **−38%**) | $2.51 |
| Opus 5.5 | 18 | $0.176 ($0.15, +18%) | $3.18 |
| Haiku 5.5 | 36 | $0.0049 ($0.004, +22%) | $0.176 |
| trap check (Haiku) | 36 | $0.0005 ($0.001, **−47%**) | $0.019 |

Total spend: **$6.78**.
- OpenAI $0.90.
- Anthropic $5.88.

By arm:

| arm | writing | grading |
|---|---|---|
| A | $0.34 | $2.04 |
| C | $0.03 | $1.93 |
| E | $0.53 | $1.90 |

The plan estimated $8.81 with Batch grading. Lines more than 30% off the plan need re-pricing before stage 2: the
single-prompt sessions, Sonnet and the trap check.
- **Single-prompt sessions:** about 5.4k input and 3.8k (sol) or 5.0k (luna) output tokens, not the 6k and 10k the
  plan assumed.
- **Sonnet gradings:** 6.9k output tokens on average.
- **Opus gradings:** 15.1k output tokens on average.

## Files

| path | content |
|---|---|
| `results.csv`, `results.json` | one row per session: labels, status, cost, minutes, calls, failure and fallbacks, and the primary grading (Sonnet `eval1`): every score and count, full-copy health, the grader's summary |
| `gradings.csv`, `gradings.json` | one row per grading (`eval1`, `eval2`, `opus1`, `haiku1`, `haiku2`), same columns |
| `truth_checks.csv`, `truth_checks.json` | one row per trap check (`truth1`) |
| `ledger.tsv` | every cost as it was spent (`app`, `eval:<name>`, `truth`) |
| `repeats_summary.json`, `sessions_summary.json` | output of `python -m evals.summarize` |
| `runs/<run_id>/` | per session: `manifest.json`, `config.json`, `transcript.json`, `summary.json`, `inputs/`, `outputs/`, and `evaluation/<name>/`: each grading's prompt, raw response, grade or trap check, summary and page images |

For the single-prompt arms, `outputs/` also has the model's LaTeX (`final_*.tex`) and its notes.

Not published:
- the app's working folder (`workspace/`);
- the full traces: `events.jsonl`, every `llm_calls.jsonl` and the single-prompt writer's `prompt.json`;
- the batch state.

These traces are kept outside the repository. `summary.json` gives each session's and grading's tokens and cost per
model. Local paths in the records are rewritten as repository-relative paths (`runs/...`, `evals/...`).

**Publication check.** The whole tree was searched, including the PDFs and page images. It contains no API keys,
key-shaped strings or secret values from the environment. It has no absolute local paths and no user names. The only
name match is the fictional persona Tomislav. Every input is a tracked fixture, so no original ad text is in this
folder.
