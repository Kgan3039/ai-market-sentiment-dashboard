---
Status: Approved
Approved by: Kartik Gangwar
Approved date: July 10, 2026
---

# Phase 0 Specification — Vertical Slice Proof

## Ticker Narratives

**Document owner:** Kartik (Project Manager / Head of Product)

**Version:** 1.0 — Phase 0 (supersedes PRD v2.0 Sections 23–25)

**Date:** July 10, 2026

**Duration:** 2 weeks (10 working days) + go/no-go review the following Monday

**Team:** Kartik (PM), Isaac (Data), Matthew (NLP), Abhi (AI), Mihir (Backend/Frontend)

---

## 1. Context and Relationship to the PRD

PRD v2.0 (Ticker Narratives) remains the approved product direction and is preserved unchanged as the **post-validation roadmap**. What changes is the commitment model. The PRD's two-sprint MVP plan (Sections 23–25) assumed four weeks of build before we learn whether the core intelligence loop produces trustworthy output. That is the same mistake we almost made with sentiment: building breadth before validating the core.

Phase 0 replaces that plan with a **two-week vertical slice**: the thinnest possible path through every stage that carries product risk — ingestion → relevance filtering → deduplication → theme clustering → cited summarization → simple UI — on five tickers, with no product surface area that doesn't test that path. Auth, watchlists, Reddit, Redis, Celery, ops dashboards, analytics, and day-over-day diffing are all deferred. None of them answers the only question that matters right now:

> **Can we reliably turn a day of raw headlines into a small set of correctly grouped, faithfully summarized, properly cited themes that a swing trader finds useful?**

If Phase 0 passes its gates (Section 8), the PRD's MVP scope is re-planned from a position of validated confidence. If it fails, we will have spent two weeks, not four, and we will know precisely which stage failed and why — the same discipline that caught the sentiment model.

**Wording policy (binding):** the product must not claim to explain the causality of price movement. We present "key narratives around today's move" and "themes dominating current coverage" — descriptions of coverage, not explanations of price. This constraint is enforced in copy review (Kartik), the summary linter (Abhi), and UI text (Mihir). Banned framings include: "the stock fell because…", "this move was driven by…", "explains today's decline". Permitted framings include: "coverage today is dominated by…", "the most-covered storyline is…", "key narratives around today's move".

## 2. Phase 0 Scope

**In scope:**

- **Tickers:** TSLA, NVDA, AMD, AAPL, META. Hardcoded; no watchlists, no universe management.
- **Sources:** Yahoo Finance headlines plus at most three RSS feeds (curated by Kartik, Day 1; candidates: Reuters Business, CNBC Markets, MarketWatch). No Reddit, no X.
- **Pipeline:** single scheduled Python process (cron/systemd timer, every 30 minutes during market hours, hourly otherwise) running stages sequentially: fetch → relevance filter → dedup → cluster → summarize → write. No Celery, no Redis, no queues.
- **Storage:** SQLite (or the existing persistence layer if it is already working — Isaac's call on Day 1; SQLite is the default). Raw items stored immutably before processing, exactly as in the PRD — replayability is non-negotiable even in Phase 0.
- **Relevance filtering:** RSS items mapped to the five tickers via symbol/company-name/alias matching (small hand-built alias table). Items matching no ticker are stored raw but excluded from processing. Ambiguous matches flagged, not guessed.
- **Intelligence:** embeddings (local sentence-transformers), three-stage dedup (URL canonicalization, MinHash near-exact, semantic merge), per-ticker daily clustering with a small-n fallback (n < 4 stories → no clustering, stories listed individually), LLM theme labels and 2–4 sentence cited summaries with the full guardrail chain (citation schema validation, banned-language linter, degrade-to-headlines fallback).
- **UI:** one React page (reusing the existing app shell): five ticker tabs, theme cards (label, summary with clickable citations, story list, outlet counts), "Other coverage" section, "Data as of HH:MM" stamp, standing disclosure, and the approved wording only. No auth — the page is served openly at a private URL.
- **API:** three read endpoints on the existing FastAPI app (Section 4, issue B1). No auth middleware, no rate limiting, no caching layer — SQLite reads at this scale are milliseconds.
- **Evaluation:** a labeled dedup pair set (~150 pairs), weekly-cadence human review compressed into a Phase 0 review during the soak window (theme correctness + summary faithfulness), and 5 moderated user sessions.

**Explicitly out of scope (deferred to post-validation MVP):** authentication, user accounts, watchlists, Reddit ingestion, Redis, Celery, separate ops dashboard (structured logs + one status endpoint instead), PostHog/analytics, portfolio tracking, day-over-day diffing, one-line ticker narratives, quiet-ticker logic, alerts, mobile polish, prediction or numeric sentiment of any kind (permanent non-goal, not a deferral).

## 3. Phase 0 Architecture

```
cron (30 min) ──► pipeline.py
                   ├─ 1. fetch: yahoo(5 tickers) + rss(≤3 feeds)  ──► raw_items
                   ├─ 2. relevance: ticker-map RSS items (alias table)
                   ├─ 3. dedup: canonicalize → MinHash → semantic merge ──► stories
                   ├─ 4. cluster: per ticker/day, HDBSCAN, n<4 fallback ──► themes
                   ├─ 5. summarize: LLM label + cited summary (changed themes only,
                   │      content-hash cached) + guardrail chain
                   └─ 6. write: themes + run_log (stage counts, latency, errors)

SQLite (single file, WAL mode)
   raw_items · stories · themes · run_log · eval_labels

FastAPI (existing app)          React (existing app)
   GET /api/v1/tickers             one page: 5 tabs → theme cards
   GET /api/v1/tickers/{sym}/themes        citations · Other coverage
   GET /api/v1/meta/status                 freshness stamp · disclosure
```

Design rules carried over from the PRD: raw-before-processing, idempotent stages keyed by (ticker, trading_day, pipeline_version), degrade to less information rather than unverified information, and every user-visible sentence traceable to a cited source. Deliberately dropped for Phase 0: horizontal scale, concurrency, caching, and operational tooling beyond structured logs and `/meta/status`.

## 4. Acceptance Criteria

Phase 0 is **feature-complete** when every criterion below passes. These are demo-able, binary checks — they gate the start of the validation soak (Section 7), not the go/no-go decision itself (Section 8).

**AC-1 Ingestion.** For each of the 5 tickers, the pipeline fetches Yahoo Finance headlines and the ≤3 RSS feeds on schedule for 3 consecutive days with zero manual intervention; every fetched item exists in `raw_items` with source, URL, timestamps, and raw payload.

**AC-2 Relevance.** ≥90% of RSS items assigned to a ticker actually concern that ticker (spot-check of 50 sampled assignments); zero ambiguous matches silently guessed (all flagged in `run_log`).

**AC-3 Dedup.** On the 150-pair labeled set: precision ≥85%, recall ≥75%. A story syndicated by multiple outlets appears as one canonical story with an outlet count ≥2 and all source links retained.

**AC-4 Clustering.** Each ticker-day with ≥4 canonical stories produces 2–6 themes plus "Other coverage"; no story is dropped (every canonical story is in exactly one theme or in Other); ticker-days with <4 stories skip clustering and list stories individually; re-running the pipeline within a day does not rename an unchanged theme.

**AC-5 Summaries.** Every rendered summary is 2–4 sentences; every sentence carries ≥1 citation resolving to a real member story; the linter passes 100% of shipped summaries (zero advisory, predictive, or causal-claim language — banned-phrase list per Section 1); a theme whose summary fails guardrails twice renders as label + story list with no summary, and the UI handles this state cleanly.

**AC-6 UI.** The page loads in <2 s, shows all five tickers, renders theme cards with working citation links (open publisher in new tab), shows "Data as of HH:MM", displays a stale-data banner if the last successful run is >60 minutes old during market hours, and carries the standing disclosure. All user-visible copy uses only approved wording.

**AC-7 End-to-end latency.** A headline published during market hours is visible on the page within 45 minutes (p95, measured over the soak window from `published_at` to run completion).

**AC-8 Replayability.** Deleting all derived tables and re-running the pipeline over stored `raw_items` reproduces themes for any past day (pipeline_version discipline works).

## 5. GitHub Issues

Conventions: repo `ai-market-sentiment-dashboard`, milestone **Phase 0**, labels `phase-0`, plus `data` / `nlp` / `ai` / `api` / `ui` / `eval` / `pm`. Estimates in ideal days (d). Issue IDs below (K/I/M/A/B prefixes) are for cross-referencing in this document; GitHub will assign its own numbers.

### Kartik (PM) — 4 issues

**K1 — Finalize RSS source list and ticker alias table** `pm` `data` — 0.5d

> Select ≤3 RSS feeds (criteria: relevant to our 5 tickers, stable feeds, headline+description available, permissive snippet use). Deliver `config/feeds.yaml` and `config/aliases.yaml` (e.g., "Alphabet"→GOOGL pattern applied to our five: "Facebook"/"Instagram"→META, "Apple Inc"→AAPL, etc., including common false-positive exclusions — "apple pie"). Definition of done: both files merged; Isaac unblocked.

**K2 — Approved-wording copy deck and banned-phrase list** `pm` — 0.5d

> One-page copy deck: page title, tab labels, theme-card framing ("Themes dominating current coverage"), move framing ("Key narratives around today's move"), disclosure text, empty/degraded/stale states. Banned-phrase list (causal, advisory, predictive) delivered as `config/banned_phrases.txt` consumed by Abhi's linter (A2) and used in Mihir's copy pass (B3). DoD: reviewed with Abhi and Mihir; file merged.

**K3 — Reviewer guidelines and labeling sessions** `pm` `eval` — 1.5d

> Write 2-page reviewer guidelines (adapted from the sentiment-experiment trust-gate methodology): how to judge (a) dedup pair correctness, (b) theme-assignment correctness, (c) summary-sentence faithfulness. Co-label the 150 dedup pairs with Matthew (M4); run the theme/faithfulness review during the soak window with one other reviewer; adjudicate disagreements. DoD: guidelines merged; all Phase 0 review sheets completed.

**K4 — User sessions and go/no-go report** `pm` — 2d

> Recruit 5 target users (swing traders, tech/growth focus). 30-minute moderated sessions during soak days 8–10 using the live page. Script: morning-triage task on 2 tickers, comprehension probes, trust probes, "would you open this tomorrow?" Compile the go/no-go report (Section 8 scorecard + qualitative findings) for the review meeting. DoD: 5 sessions done; report circulated 24h before the meeting.

### Isaac (Data Engineer) — 4 issues

**I1 — Phase 0 schema and persistence module** `data` — 1d

> SQLite (WAL mode) with tables: `raw_items` (id, source, ticker?, title, description, url, canonical_url, published_at, fetched_at, raw_json), `stories` (id, ticker, trading_day, canonical_title, embedding BLOB, outlet_count, member_ids JSON), `themes` (id, ticker, trading_day, label, summary, citations JSON, salience_rank, status, centroid BLOB, content_hash, pipeline_version), `run_log` (run_id, stage, counts, duration_ms, errors JSON, started_at). Thin repository module reused by all stages; migration script; decision recorded if existing persistence layer is used instead. DoD: module merged with unit tests; Matthew and Mihir unblocked. **Blocks: I2, I3, M1, B1.**

**I2 — Yahoo Finance fetcher for 5 tickers** `data` — 1d

> Adapt the existing Yahoo ingestion code to the Phase 0 schema. Fetch headlines per ticker, dedupe on (source, canonical_url) at insert (idempotent re-runs), record fetch metadata. Handle per-ticker failure without aborting the run. DoD: 3 consecutive scheduled runs store items for all 5 tickers; failure of one ticker logged, others unaffected. **Blocked by: I1.**

**I3 — RSS fetcher and relevance filter** `data` — 1.5d

> Fetch the ≤3 feeds from `config/feeds.yaml`; parse title/description/link/pubDate; canonicalize URLs (strip tracking params). Relevance filter: assign items to tickers via `config/aliases.yaml` (symbol, cashtag, company-name, alias match on title+description); non-matching items stored with ticker NULL and excluded from processing; ambiguous matches flagged in `run_log`. DoD: AC-2 spot-check passes on a 50-item sample. **Blocked by: I1, K1.**

**I4 — Pipeline runner, scheduling, and status endpoint data** `data` — 1.5d

> `pipeline.py` orchestrating stages sequentially with per-stage try/except, structured logging to `run_log`, idempotency keys (ticker, trading_day, pipeline_version), and a `--replay --date` mode (AC-8). Cron/systemd timer: every 30 min 9:00–16:30 ET weekdays, hourly otherwise. Writes the freshness data consumed by `/meta/status`. DoD: 3 days unattended operation; replay reproduces a past day. **Blocked by: I2, I3; stages plug in as M3/M5/A1 land.**

### Matthew (NLP Engineer) — 5 issues

**M1 — Embedding module** `nlp` — 1d

> Local sentence-transformers (all-MiniLM-L6-v2 baseline). Embed title + description. Batch API with simple disk model cache; vectors stored as BLOBs via I1's module. Cosine-similarity helper used by dedup, clustering, and theme stability. DoD: embeds a day's items for 5 tickers in <60 s on the dev box; unit tests. **Blocked by: I1.**

**M2 — Exact and near-exact dedup** `nlp` — 1d

> Stage 1: canonical-URL and normalized-title exact match. Stage 2: MinHash over title shingles for near-exact. Merge policy: earliest-published item is canonical; all members retained with source links; outlet_count maintained. DoD: unit tests on synthetic syndication cases; runs inside pipeline. **Blocked by: I1 (runs on I2/I3 data).**

**M3 — Semantic dedup merge** `nlp` — 1.5d

> Stage 3: embedding cosine similarity within (ticker, ±36h) window above threshold → merge into canonical story. Threshold tuned on the labeled set from M4. Precision favored over recall (never merge distinct stories). DoD: AC-3 numbers met on the labeled set; integrated in pipeline. **Blocked by: M1, M2, M4 (labels needed for tuning).**

**M4 — Dedup label set and evaluation script** `nlp` `eval` — 1d

> Sample ~150 candidate pairs from real ingested data across similarity bands (easy dupes, hard near-dupes, hard negatives). Co-label with Kartik per K3 guidelines. `eval_dedup.py` reporting precision/recall at a given threshold; results committed to the repo. DoD: labeled set + script merged; used by M3. **Blocked by: I2, I3 (needs real data), K3 (guidelines).**

**M5 — Theme clustering with small-n fallback** `nlp` — 2d

> Per (ticker, trading_day): HDBSCAN over canonical-story embeddings (agglomerative fallback if unstable at small n); noise → "Other coverage"; n<4 → skip clustering. Salience rank = f(story count, outlet diversity, recency). Intra-day theme stability: match to previous run's themes by centroid similarity so unchanged themes keep identity (feeds A3's cache and AC-4). DoD: AC-4 demonstrable on 3 real ticker-days of varying volume. **Blocked by: M3.**

### Abhi (AI Engineer) — 4 issues

**A1 — LLM summarization with structured cited output** `ai` — 2d

> Provider integration (decision from PRD OQ-3 stands). Prompt takes a theme's member stories (title, description, outlet, time) and returns strict JSON: `{label: ≤8 words, sentences: [{text, citation_ids[]}]}`. System prompt forbids advice, prediction, causal claims about price, and uncited statements; low temperature. Fixture-based development from Day 1 (sample story sets committed) so this doesn't block on M5. DoD: valid JSON on 20 fixture themes; integrated behind a `summarize(theme)` interface. **Blocked by: I1 (schema); integrates after M5.**

**A2 — Guardrail chain and banned-language linter** `ai` — 1.5d

> Post-generation checks in order: (1) JSON schema + citation-id validation (every sentence ≥1 valid member citation); (2) linter over `config/banned_phrases.txt` (K2) plus regex families for advisory/predictive/causal-price language; (3) fail → one regeneration with error feedback → fail again → theme ships as label + story list (degraded state, flagged in run_log). DoD: adversarial fixture suite (20 bad outputs) all caught; zero banned phrases reach the API layer. **Blocked by: A1, K2.**

**A3 — Summary caching and cost/latency logging** `ai` — 0.5d

> Cache summaries by content hash of theme membership (stable under M5's theme-identity matching); skip regeneration for unchanged themes; log tokens, cost, and latency per call to `run_log`. DoD: a re-run with unchanged data makes zero LLM calls. **Blocked by: A1, M5.**

**A4 — Evaluation sampling tooling** `ai` `eval` — 1d

> `make_review_sheets.py`: samples from live data (a) 40 story→theme assignments per review, (b) every summary sentence from 2 sampled days with its cited sources resolved, into CSV/Sheet form matching K3's guidelines. Computes agreement and faithfulness numbers from completed sheets for the Section 8 scorecard. DoD: sheets generated from soak-window data; scorecard numbers reproducible. **Blocked by: M5, A1 (live outputs to sample).**

### Mihir (Backend/Frontend Engineer) — 4 issues

**B1 — Phase 0 read API on existing FastAPI app** `api` — 1d

> Three endpoints, no auth: `GET /api/v1/tickers` (the 5, with freshness + today's theme count); `GET /api/v1/tickers/{sym}/themes?date=` (ranked themes: label, summary sentences with citation ids, citations resolved to {headline, outlet, url, published_at}, story lists, Other coverage, degraded flag); `GET /api/v1/meta/status` (last run per stage, data_as_of). Contract fixed Day 1 with committed fixture JSON so UI work proceeds in parallel; strip/disable auth middleware, Redis, and unused routes from the existing app. DoD: endpoints serve fixtures Day 2, real data when pipeline lands; p95 <300 ms. **Blocked by: I1 (schema); fixture-first from Day 1.**

**B2 — Ticker page UI** `ui` — 2.5d

> One page on the existing React app: header (product name, data-as-of stamp, stale banner >60 min), five ticker tabs, theme cards (label, ranked order, summary with superscript citations opening publisher in new tab, outlet/story counts, expandable story list), "Other coverage" section, degraded-state rendering (label + stories, "summary unavailable" note), empty state (market closed / no coverage yet), standing disclosure footer. Desktop-first, usable at 375 px. DoD: AC-6 against fixtures, then against live API. **Blocked by: B1 (contract); starts against fixtures Day 2.**

**B3 — Copy compliance and state audit** `ui` `pm` — 0.5d

> Sweep all user-visible strings against K2's copy deck; verify banned framings absent; verify every UI state (loading, empty, degraded, stale, error) uses approved copy. Joint sign-off with Kartik. DoD: checklist in the issue completed; screenshots attached. **Blocked by: B2, K2.**

**B4 — Deployment** `api` `ui` — 1d

> Single small VM: FastAPI + built React assets behind nginx (or the existing serve setup), SQLite on disk with nightly file backup, cron installed for I4, `.env` for LLM key, private URL (basic-auth at nginx level is acceptable and is not product auth). DoD: live URL shared with team; survives VM reboot; backup verified. **Blocked by: I4, B1, B2 (deployable slice).**

Total: 21 issues, ~24 ideal days across 5 people over 10 working days — realistic at ~60% allocation with review, integration, and soak-window fixes absorbing the rest.

---

## 6. Dependency Order

Issues grouped into levels; an issue may start once everything it "needs" is merged.

```
Level 0 (Day 1, no blockers)
  K1 feeds/aliases . K2 copy deck . I1 schema
  + 1-hour API contract session (Mihir + Kartik) fixing B1's response shapes

Level 1
  I2 yahoo fetcher      needs I1
  I3 rss + relevance    needs I1, K1
  M1 embeddings         needs I1
  M2 exact/minhash      needs I1
  A1 LLM summaries      needs I1        (fixture-first: real integration waits on M5)
  B1 read API           needs I1        (fixture-first: real data waits on pipeline)

Level 2
  M4 dedup label set    needs I2, I3 (real data) and K3 (guidelines)
  I4 pipeline runner    needs I2, I3    (stages plug in as they land)
  B2 ticker page UI     needs B1 contract (builds against fixtures)

Level 3
  M3 semantic dedup     needs M1, M2, M4
  A2 guardrail chain    needs A1, K2

Level 4
  M5 theme clustering   needs M3

Level 5
  A3 summary cache      needs A1, M5
  A4 eval tooling       needs M5, A1 (live outputs)
  B3 copy audit         needs B2, K2

Level 6
  B4 deploy             needs I4, B1, B2  ->  SOAK (live, unattended)
  K3 reviews            needs A4 sheets   ->  K4 user sessions + go/no-go report
```

**Critical path:** I1 → I2/I3 → M4 → M3 → M5 → A1-integration/A2 → I4-complete → B4 → soak → reviews/user sessions → go/no-go. Slack exists on the UI branch (fixture-first) and on A1 (fixture-first). The riskiest handoff is M3→M5 mid-week-1; if semantic dedup tuning slips, M5 proceeds on stage-1/2 dedup output and M3 lands during early soak (acceptable: clustering quality is measured at review time, not landing time).

## 7. Two-Week Schedule

Week 1 = Mon Jul 13 – Fri Jul 17; Week 2 = Mon Jul 20 – Fri Jul 24; go/no-go Mon Jul 27. Daily 15-min standup; demos Friday both weeks.

| Day | Isaac | Matthew | Abhi | Mihir | Kartik |
|---|---|---|---|---|---|
| 1 (Mon) | I1 schema | prep M1 (model selection) | A1 prompt + fixtures | API contract w/ Kartik; strip existing app (B1 start) | K1 feeds/aliases; contract review; K2 start |
| 2 (Tue) | I1 done; I2 start | M1 embeddings | A1 structured output | B1 fixture-served endpoints | K2 copy deck done; K3 guidelines start |
| 3 (Wed) | I2 done; I3 start | M2 minhash dedup | A1 done (fixtures); A2 start | B2 UI vs fixtures | K3 guidelines done |
| 4 (Thu) | I3 done; I4 start | M4 labeling w/ Kartik (real data from I2/I3) | A2 guardrails | B2 UI | M4 co-labeling |
| 5 (Fri) | I4 runner w/ dedup stages | M3 semantic dedup tuning | A2 done; adversarial suite | B2 UI core done | **Demo 1: raw→stories live; UI on fixtures** |
| 6 (Mon) | I4: wire M3; ops watch | M3 done (AC-3); M5 start | integrate A1 behind M5 interface | B1 on real data; B2 live wiring | recruit users (K4) |
| 7 (Tue) | replay mode (AC-8) | M5 clustering | A1/A2 in pipeline; A3 cache | B2 polish; degraded/empty states | user scheduling; review prep |
| 8 (Wed) | ops watch; fixes | M5 done (AC-4) | A4 eval tooling | B4 deploy → **soak begins** (private URL) | K3 review round 1 (from A4 sheets) |
| 9 (Thu) | soak support | threshold fixes from review | A4 scorecard numbers | B3 copy audit w/ Kartik | user sessions 1–2 |
| 10 (Fri) | soak support; backup verify | fixes | fixes; final scorecard draft | fixes | **Demo 2: full slice.** Sessions 3–5 |
| +1 (Mon) | — | — | — | — | **Go/no-go review** (report circulated Sun) |

Honest caveats built into the plan: the soak window inside the two weeks is 3 trading days (Wed–Fri of week 2), not 5. The pipeline keeps running unattended over the weekend and the following Monday's go/no-go reviews **5 trading days of data** (Wed Jul 22 – Tue Jul 28 would be ideal; we accept Wed–Mon = 4 trading days + weekend gap handling as sufficient evidence, since weekend behavior itself is a thing we need to observe). If deploy (B4) slips past Day 8, the go/no-go moves day-for-day — we do not compress the soak.

## 8. Go/No-Go Validation Plan

**Decision meeting:** Monday after week 2. Attendees: full team. Input: the scorecard below (computed by A4 tooling from K3 review sheets), the user-session findings (K4), and the ops record (`run_log`). Decision recorded in the repo.

**Quantitative gates (from soak-window data):**

| # | Gate | Threshold | Source |
|---|---|---|---|
| G1 | Theme-assignment human agreement | ≥75% | K3 review, 2 reviewers, adjudicated, ≥80 sampled assignments |
| G2 | Summary-sentence faithfulness (supported by cited source) | ≥95% | K3 review, every sentence from 2 sampled days |
| G3 | Banned-language violations reaching UI | 0 | A2 linter logs + B3 audit + review sweep |
| G4 | Dedup precision / recall on labeled set | ≥85% / ≥75% | M4 eval script |
| G5 | Pipeline reliability | ≥95% scheduled runs succeed over soak; no >2h gap during market hours | run_log |
| G6 | Freshness | p95 publish→visible ≤45 min (market hours) | run_log |
| G7 | Degraded-summary rate | ≤15% of themes shipped without summary | run_log |

**Qualitative gates (from 5 user sessions):**

| # | Gate | Threshold |
|---|---|---|
| Q1 | Comprehension: user correctly states the day's dominant narratives for 2 tickers using only the page | ≥4 of 5 users |
| Q2 | Trust: user does not flag any summary as wrong/misleading after checking citations | ≥4 of 5 users |
| Q3 | Intent: "would you open this tomorrow morning?" | ≥3 of 5 yes, with stated reason |

**Decision rules:**

- **GO** — all G-gates and Q-gates pass. Action: re-plan the PRD MVP (auth, watchlists, Reddit, diffing, ops hardening) as the next phase, using Phase 0 measurements to re-estimate; the PRD's Sections 23–25 are rewritten against validated velocity, not assumed.
- **CONDITIONAL GO** — at most one G-gate misses by a small margin (e.g., G1 at 70–75%) and the failing stage has an identified, scoped fix. Action: one focused fix week on that stage only, re-run the affected review, then decide. No scope expansion during the fix week.
- **NO-GO** — G2 or G3 fails at any margin (faithfulness and wording are trust-fatal, exactly like the sentiment trust gate), or two or more G-gates fail, or Q-gates show users don't comprehend or don't trust the output. Action: stop-and-rethink review; options include changing the clustering/summarization approach, narrowing to an even thinner product (e.g., dedup-only "collapsed feed"), or revisiting the thesis. We do not proceed to MVP build on a NO-GO.

**Anti-gaming rules (lessons from the sentiment experiment):** reviewers never review their own stage's outputs where avoidable; samples are drawn randomly by A4's tooling, not hand-picked; thresholds were fixed in this document before soak data existed; a gate marginally passed by measuring differently is a fail.

## 9. Post-Validation Roadmap (unchanged, re-sequenced)

PRD v2.0 remains the destination. On a GO, the expected sequence (to be re-planned, not committed): (1) hardening — Postgres migration if scale demands, Celery/Redis only when concurrency requires it, ops dashboard; (2) product breadth — auth, watchlists, full ticker universe; (3) intelligence depth — day-over-day diffing and "Since yesterday" (the PRD's Stage D), one-line narratives, quiet-ticker logic; (4) sources — Reddit, then X behind its own trust gate; (5) growth — alerts, analytics, beta widening per the PRD's Section 20 launch gates, which remain the MVP bar. Nothing in the PRD is a commitment until Phase 0 says the core loop deserves it.

---

*End of Phase 0 specification. Change requests to Kartik. On go/no-go day, this document gets a one-page appendix recording the scorecard and the decision.*
