# Multi-turn scenarios: the app vs a one-call-per-turn baseline (2026-11-12)

**Question.** Once the user talks to the system (edits, corrections, appearance changes, statements about themselves, missing details), does the multi-agent app do better than a single model call per turn?

**Answer from this round:**
- **The app does better than the plain baseline in one place:** asking for missing contact details. A one-line prompt hint closes that gap.
- **The router fixes of #33 work:** the two editing failures of the frozen app (one change for both documents, PDF styles) are gone.
- **Elsewhere the fixed app ties the baseline, at 4–9× the cost and 3–5× the time per user turn.**
- **Two app weaknesses remain:**
  - its fact check removes a job title the user stated about themselves;
  - its cover letters overshoot a word limit the user set.

## What was run

- **Systems** (all writers on gpt-6-luna; the app's critic and grounding check on gpt-5.6-luna, as configured):
  - **app**: the app at `develop` `578b2ae`, with the router fixes of #33.
  - **app_frozen**: the app at `2320a98`, before #33. It is a reference arm on S2, S4 and S5, run from an exported tree of that commit.
  - **b2**: one call per user turn with the whole conversation (`evals/single_prompt_chat.py`). It gets the app's own writer, truthfulness and LaTeX rules, and the app writer's "apply only these changes" rule.
  - **b2_ask**: b2 plus one line telling it to ask for missing details. S3 only.
- **Scenarios** (`evals/scenarios/`): 13 scripted conversations on the tracked fictional personas and ads, 3 reps each, **138 sessions**.
- **Checks:** deterministic, with no grader model (`evals/score_checks.py`).
- **Cost:** **$4.16** in total (cap $6.00).
- **Truthfulness rule** (owner, 2026-11-12): what the user says about themselves in the chat is true, even if their documents do not say it. S2 follows it:
  - three requests the user backs with their own statement must be **applied**;
  - three requests the user's own words contradict must be **refused and explained**.
- **Pre-registered reading:** pairs share a case and a rep. A difference is claimed only with ≥6 pairs, ≥4 discordant pairs one way and none the other (exact sign test p shown).

The full tables are in [`report.md`](report.md) and [`results.json`](results.json). Each session's records are in `runs/<scenario>-<script>-<persona>-<system>-r<rep>/`: manifest, config, summary, transcript, `checks.json`, inputs and outputs. Event logs, model-call logs and workspaces are kept private.

## Results

"All checks" means sessions passing every check of their script.

| scenario | app (fixed) | app_frozen | b2 | b2_ask | pre-registered reading |
|---|---|---|---|---|---|
| **S1** 11 stacked edits, 12 turns (p03, p09) | 1/6 | – | 5/6 | – | no difference detected (app alone 1, b2 alone 5, p=0.22) |
| **S2** truthfulness under the ground-truth rule (6 cases) | 11/18 | 11/18 | 13/18 | – | no difference detected: app vs b2 0–2, app vs frozen 2–2 |
| **S3** contact details given only if asked (4 no-CV personas) | **12/12** | – | 7/12 | 10/12 | **app better than b2** (5–0 of 12, p=0.06); app vs b2_ask: no difference (2–0) |
| **S4** one change for both documents, then a correction everywhere | 6/6 | **0/6** | 6/6 | – | **fixed app better than frozen** (6–0, p=0.03); fixed app vs b2: tie |
| **S5** appearance and content edits interleaved | 6/6 | **0/6** | 6/6 | – | **fixed app better than frozen** (6–0, p=0.03); fixed app vs b2: tie |

### Per-check detail that matters

- **S1:**
  - **App letter length:** 5 of 6 app letters broke "keep the cover letter under 200 words" (198–228 words; the one that passed had 198). b2 kept 6/6 under the limit.
  - **App summary:** one app summary grew past two sentences.
  - **b2 location:** one b2 session dropped the city instead of respelling it "Wrocław".
  - **Ties:** every other edit held in both systems in every session: renamed headings, the removed job, Languages moved under the summary, the removed link, the 4-bullet limit, the figure in the summary, the salutation.
- **S2, the three applied cases.** The user states the fact, so it must be written.

  | case | app (fixed) | app_frozen | b2 |
  |---|---|---|---|
  | **Retitle** ("that's basically what I do there") | 1/3 | 2/3 | 3/3 |
  | **Homelab moved into experience** | 3/3 | 2/3 | 3/3 |
  | **"I run Kubernetes in production and handle on-call"** | 0/3 | 0/3 | 0/3 |

  - **Retitle:** every app failure (3 of 6 app runs across both versions) has the same trace. The writer applies "DevOps Engineer", the grounding check flags it as unsupported, and the redraft reverts it.
  - **Kubernetes / on-call:** every system declined it, citing the candidate's notes ("the platform team operates the cluster").
- **S2, the three traps.** The user's own words contradict the claim: CKA "exam booked", Go "only a side project", forklift "don't have it yet".
  - **Claim absent:** 9/9 in every system.
  - **Told the user why:** 7/9 in every system.
- **S3:** the app asked for the missing e-mail and phone number in 12/12 sessions and put them in the CV. b2 asked in 7/12; the other 5 left `[EMAIL]`/`[PHONE]` placeholders. b2_ask asked in 10/12.
- **S4 / S5 (frozen app):**
  - **S4:** the frozen app put the new link into the CV only, every time.
  - **S5:** it lost the earlier style requests (dark green headings, sans-serif) when a later one came, every time.
  - **The fixed app** did both right in 12/12. Text stayed byte-identical on every appearance-only turn, and every PDF compiled from the system's own LaTeX, for all systems.

### Cost and time per user turn (means per scenario)

| | app (fixed) | app_frozen | b2 |
|---|---|---|---|
| USD | 0.008–0.021 | 0.009–0.019 | 0.0022–0.0025 |
| seconds | 73–166 | 82–149 | 26–38 |
| model calls | 7.4–15.9 | 7.8–13.9 | 1 |

The fixed app costs a little more than the frozen one on S4 ($0.021 vs $0.019 per turn) because it now also redrafts the cover letter.

## Reading

1. **The structural advantage is real but narrow.** It holds where the graph enforces a step the model might skip.
   - The missing-details interrupt beat the plain baseline 12/12 vs 7/12.
   - A single prompt line (b2_ask) gets most of the way there (10/12), so the advantage over a well-prompted single call is not significant.
2. **The #33 fixes did what they were meant to.** Both editing scenarios went from 0/6 to 6/6, which ties the baseline.
3. **Two follow-ups for the app:**
   - **Grounding vs the ground-truth rule.** The fact check treats a user's statement about themselves as an unsupported claim and redraft removes it (retitle, 3 of 6 app runs). Pass the user's self-statements to the grounding check as candidate facts, and keep claims the user stated out of `claims_to_remove`.
   - **User-set limits.** The letter overshot "under 200 words" in 5 of 6 sessions. A deterministic word-count check in finalize, with one redraft on a breach, would hold it.
4. **Both systems' prompts fall short of the ground-truth rule when the chat contradicts the documents** (Kubernetes / on-call, 0/9). If the owner wants the chat to win in that case, the shared truthfulness protocol (`agents/document_writer.py`) should say so explicitly.
5. **Positioning:** at about 4–9× the cost and 3–5× the latency per turn, the app is a guarded pipeline (missing-details guard, text-invariant appearance edits), not a better interactive editor.

## Reproduce

```bash
git archive 2320a98 | tar -x -C FROZEN
.venv/bin/python -m evals.run_scenarios --out OUT --frozen-tree FROZEN --env-file .env --total-cap 6.0 --workers 6
.venv/bin/python -m evals.run_scenarios --out OUT --report-only --frozen-tree FROZEN --publish evals/results/<date>-scenarios
```
