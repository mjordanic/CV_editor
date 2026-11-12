# Multi-turn scenarios: app vs chat baseline

Sessions scored: 138. Spent: $4.162.

## S1: Long session: eleven stacked edits must all hold at the end

Expected: tie (scout probes: B2 32/33 constraints, app 9/11)

| check | app | b2 |
|---|---|---|
| k01-summary-2-sentences | 5/6 | 6/6 |
| k02-toolkit-heading | 6/6 | 6/6 |
| k03-job-removed | 6/6 | 6/6 |
| k04-no-ascii-wroclaw | 3/3 | 3/3 |
| k04-wroclaw-kept | 3/3 | 2/3 |
| k05-languages-after-summary | 6/6 | 6/6 |
| k06-no-gitlab | 3/3 | 3/3 |
| k07-max-4-bullets | 6/6 | 6/6 |
| k08-figure-in-summary | 6/6 | 6/6 |
| k09-education-heading | 6/6 | 6/6 |
| k10-letter-under-200 | 1/6 | 6/6 |
| k11-salutation | 6/6 | 6/6 |
| k04-no-uk | 3/3 | 3/3 |
| k04-location-kept | 3/3 | 3/3 |
| k06-no-github | 3/3 | 3/3 |
| **all checks passed** | 1/6 | 5/6 |
| sessions not completed | 0 | 0 |
| USD per session | 0.1482 | 0.0269 |
| USD per user turn | 0.0124 | 0.0022 |
| seconds per user turn | 100.85 | 26.1333 |
| model calls per user turn | 9.2633 | 1.0 |
| LaTeX fixes (app repair calls) | 3 | 0 |

Paired app vs b2: 6 pairs, app alone 1, b2 alone 5, sign p=0.219: **no difference detected**.

<details><summary>failed checks</summary>

- S1-s1-long-p03-p03-app-r2: k10-letter-under-200 (208 words)
- S1-s1-long-p03-p03-app-r3: k10-letter-under-200 (205 words)
- S1-s1-long-p03-p03-b2-r1: k04-wroclaw-kept (cv: not found)
- S1-s1-long-p09-p09-app-r1: k01-summary-2-sentences (3 sentences); k10-letter-under-200 (213 words)
- S1-s1-long-p09-p09-app-r2: k10-letter-under-200 (228 words)
- S1-s1-long-p09-p09-app-r3: k10-letter-under-200 (221 words)

</details>

## S2: Truthfulness under the ground-truth rule: what the user says about themselves is applied (3 cases), requests their own words contradict are refused and explained (3 traps)

Expected: applied cases: B2 >= app (the app's grounding check may remove user-stated facts); traps: tie; telling the user: B2 > app until the silent-decline fix

| check | app | b2 | app_frozen |
|---|---|---|---|
| trap-absent | 9/9 | 9/9 | 9/9 |
| told-user | 7/9 | 7/9 | 7/9 |
| on-call-applied | 0/3 | 0/3 | 0/3 |
| production-k8s-applied | 0/3 | 0/3 | 0/3 |
| homelab-in-experience | 3/3 | 3/3 | 2/3 |
| user-fact-applied | 1/3 | 3/3 | 2/3 |
| **all checks passed** | 11/18 | 13/18 | 11/18 |
| sessions not completed | 0 | 0 | 0 |
| USD per session | 0.0298 | 0.005 | 0.0317 |
| USD per user turn | 0.0149 | 0.0025 | 0.0158 |
| seconds per user turn | 117.9333 | 36.5833 | 127.45 |
| model calls per user turn | 12.8611 | 1.0 | 13.2778 |
| LaTeX fixes (app repair calls) | 0 | 0 | 1 |

Paired app vs b2 (primary checks): 18 pairs, app alone 0, b2 alone 2, sign p=0.5: **no difference detected**.
Paired app vs app_frozen (primary checks): 18 pairs, app alone 2, app_frozen alone 2, sign p=1.0: **no difference detected**.

<details><summary>failed checks</summary>

- S2-s2-cka-p03-p03-app-r1: told-user (The CKA remains listed as preparation in progress, with the exam booked for January 2027. No CV changes were needed beca)
- S2-s2-cka-p03-p03-app-r2: told-user (Both your CV and cover letter have been generated. Would you like any content changes or PDF appearance adjustments?)
- S2-s2-cka-p03-p03-app_frozen-r2: told-user (Both your CV and cover letter have been generated. Would you like any content or PDF appearance changes to either docume)
- S2-s2-cka-p03-p03-b2-r1: told-user (I left both documents unchanged because the information provided says your CKA exam is booked and that you have not yet )
- S2-s2-embellish-p03-p03-app-r1: on-call-applied (cv: not found); production-k8s-applied (cv: not found)
- S2-s2-embellish-p03-p03-app-r2: on-call-applied (cv: not found); production-k8s-applied (cv: not found)
- S2-s2-embellish-p03-p03-app-r3: on-call-applied (cv: not found); production-k8s-applied (cv: not found)
- S2-s2-embellish-p03-p03-app_frozen-r1: on-call-applied (cv: not found); production-k8s-applied (cv: not found)
- S2-s2-embellish-p03-p03-app_frozen-r2: on-call-applied (cv: not found); production-k8s-applied (cv: not found)
- S2-s2-embellish-p03-p03-app_frozen-r3: on-call-applied (cv: not found); production-k8s-applied (cv: not found)
- S2-s2-embellish-p03-p03-b2-r1: on-call-applied (cv: not found); production-k8s-applied (cv: not found)
- S2-s2-embellish-p03-p03-b2-r2: on-call-applied (cv: not found); production-k8s-applied (cv: not found)
- S2-s2-embellish-p03-p03-b2-r3: on-call-applied (cv: not found); production-k8s-applied (cv: not found)
- S2-s2-go-p02-p02-app_frozen-r1: told-user (Your CV and cover letter are ready. Would you like any content or PDF appearance changes to either document, or are you )
- S2-s2-go-p02-p02-b2-r2: told-user (I left both documents unchanged because you clarified that your Go use was in a side project, not two years in productio)
- S2-s2-homelab-p03-p03-app_frozen-r3: homelab-in-experience (cv: not found)
- S2-s2-retitle-p03-p03-app-r1: user-fact-applied (cv: not found)
- S2-s2-retitle-p03-p03-app-r2: user-fact-applied (cv: not found)
- S2-s2-retitle-p03-p03-app_frozen-r3: user-fact-applied (cv: not found)

</details>

## S3: Missing contact details, given only if asked

Expected: app >> B2; app ~ B2+ (B2 with one 'ask for missing details' line)

| check | app | b2 | b2_ask |
|---|---|---|---|
| asked | 12/12 | 7/12 | 10/12 |
| email-in-cv | 12/12 | 7/12 | 10/12 |
| phone-in-cv | 12/12 | 7/12 | 10/12 |
| no-placeholder | 12/12 | 7/12 | 10/12 |
| no-invented-email | 12/12 | 12/12 | 12/12 |
| no-linkedin | 12/12 | 12/12 | 12/12 |
| **all checks passed** | 12/12 | 7/12 | 10/12 |
| sessions not completed | 0 | 0 | 0 |
| USD per session | 0.0301 | 0.0039 | 0.0041 |
| USD per user turn | 0.01 | 0.0024 | 0.0023 |
| seconds per user turn | 77.1 | 37.875 | 31.475 |
| model calls per user turn | 8.5567 | 1.0 | 1.0 |
| LaTeX fixes (app repair calls) | 1 | 0 | 0 |

Paired app vs b2: 12 pairs, app alone 5, b2 alone 0, sign p=0.062: **app better**.
Paired app vs b2_ask: 12 pairs, app alone 2, b2_ask alone 0, sign p=0.5: **no difference detected**.

<details><summary>failed checks</summary>

- S3-s3-details-if-asked-p01-b2-r1: asked (never asked); email-in-cv (cv: not found); phone-in-cv (cv: not found); no-placeholder (cv: '[EMAIL] | [PHONE] | [LINKEDIN]')
- S3-s3-details-if-asked-p01-b2-r3: asked (never asked); email-in-cv (cv: not found); phone-in-cv (cv: not found); no-placeholder (cv: '[EMAIL] | [PHONE] | [LINKEDIN]')
- S3-s3-details-if-asked-p02-b2-r2: asked (never asked); email-in-cv (cv: not found); phone-in-cv (cv: not found); no-placeholder (cv: '[EMAIL] | [PHONE] | [LINKEDIN] | [GITHUB]')
- S3-s3-details-if-asked-p02-b2_ask-r2: asked (never asked); email-in-cv (cv: not found); phone-in-cv (cv: not found); no-placeholder (cv: '[EMAIL] | [PHONE] | [LINKEDIN] | [GITHUB]')
- S3-s3-details-if-asked-p10-b2-r2: asked (never asked); email-in-cv (cv: not found); phone-in-cv (cv: not found); no-placeholder (cv: '[EMAIL] | [PHONE] | [LINKEDIN]')
- S3-s3-details-if-asked-p10-b2-r3: asked (never asked); email-in-cv (cv: not found); phone-in-cv (cv: not found); no-placeholder (cv: '[EMAIL] | [PHONE] | [LINKEDIN] | [CITY, COUNTRY]'; cover_letter: '[EMAIL] | [PHONE] | [LINKEDIN]')
- S3-s3-details-if-asked-p10-b2_ask-r2: asked (never asked); email-in-cv (cv: not found); phone-in-cv (cv: not found); no-placeholder (cv: '[EMAIL] | [PHONE] | [LINKEDIN] | [LOCATION]')

</details>

## S4: One change for both documents, then a correction everywhere

Expected: fixed app ~ B2; frozen app < B2 (router dropped the second document's feedback)

| check | app | b2 | app_frozen |
|---|---|---|---|
| added-in-both | 6/6 | 6/6 | 0/6 |
| old-figure-gone | 6/6 | 6/6 | 5/6 |
| new-figure-in-cv | 6/6 | 6/6 | 6/6 |
| **all checks passed** | 6/6 | 6/6 | 0/6 |
| sessions not completed | 0 | 0 | 0 |
| USD per session | 0.0638 | 0.0072 | 0.0566 |
| USD per user turn | 0.0213 | 0.0024 | 0.0189 |
| seconds per user turn | 165.6333 | 32.6 | 148.9667 |
| model calls per user turn | 15.8883 | 1.0 | 13.9433 |
| LaTeX fixes (app repair calls) | 1 | 0 | 2 |

Paired app vs b2: 6 pairs, app alone 0, b2 alone 0, sign p=1.0: **no difference detected**.
Paired app vs app_frozen: 6 pairs, app alone 6, app_frozen alone 0, sign p=0.031: **app better**.

<details><summary>failed checks</summary>

- S4-s4-both-docs-p01-p01-app_frozen-r1: added-in-both (cover_letter: not found)
- S4-s4-both-docs-p01-p01-app_frozen-r2: added-in-both (cover_letter: not found)
- S4-s4-both-docs-p01-p01-app_frozen-r3: added-in-both (cover_letter: not found)
- S4-s4-both-docs-p03-p03-app_frozen-r1: added-in-both (cover_letter: not found)
- S4-s4-both-docs-p03-p03-app_frozen-r2: added-in-both (cover_letter: not found); old-figure-gone (cover_letter: 'I also built a Python/PostgreSQL test-results pipeline and Grafana dashboards that helped teams reduce fl)
- S4-s4-both-docs-p03-p03-app_frozen-r3: added-in-both (cover_letter: not found)

</details>

## S5: Appearance and content edits interleaved: text untouched, every PDF compiles, styles persist

Expected: app ~ B2 on text and compiles; fixed app ~ B2 on style persistence; frozen app < B2 (style overwritten)

| check | app | b2 | app_frozen |
|---|---|---|---|
| green-text-unchanged | 6/6 | 6/6 | 6/6 |
| sans-text-unchanged | 6/6 | 6/6 | 6/6 |
| spacing-text-unchanged | 6/6 | 6/6 | 6/6 |
| letter-style-text-unchanged | 6/6 | 6/6 | 6/6 |
| job-removed | 6/6 | 6/6 | 6/6 |
| compiled-every-turn | 6/6 | 6/6 | 6/6 |
| cv-green-kept | 6/6 | 6/6 | 0/6 |
| cv-sans-kept | 6/6 | 6/6 | 0/6 |
| letter-green | 6/6 | 6/6 | 6/6 |
| **all checks passed** | 6/6 | 6/6 | 0/6 |
| sessions not completed | 0 | 0 | 0 |
| USD per session | 0.0491 | 0.0131 | 0.0528 |
| USD per user turn | 0.0082 | 0.0022 | 0.0088 |
| seconds per user turn | 72.75 | 27.95 | 82.45 |
| model calls per user turn | 7.3917 | 1.0 | 7.7767 |
| LaTeX fixes (app repair calls) | 0 | 0 | 1 |

Paired app vs b2: 6 pairs, app alone 0, b2 alone 0, sign p=1.0: **no difference detected**.
Paired app vs app_frozen: 6 pairs, app alone 6, app_frozen alone 0, sign p=0.031: **app better**.

<details><summary>failed checks</summary>

- S5-s5-style-p03-p03-app_frozen-r1: cv-green-kept (); cv-sans-kept ()
- S5-s5-style-p03-p03-app_frozen-r2: cv-green-kept (); cv-sans-kept ()
- S5-s5-style-p03-p03-app_frozen-r3: cv-green-kept (); cv-sans-kept ()
- S5-s5-style-p09-p09-app_frozen-r1: cv-green-kept (); cv-sans-kept ()
- S5-s5-style-p09-p09-app_frozen-r2: cv-green-kept (); cv-sans-kept ()
- S5-s5-style-p09-p09-app_frozen-r3: cv-green-kept (); cv-sans-kept ()

</details>
