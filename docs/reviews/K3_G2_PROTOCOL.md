# K3 G2 sentence-faithfulness protocol — `k3-g2-v1`

**Status:** ratified for G2 (K3a). This is the G2 portion of K3 (#60) only.
It does **not** close K3: the theme-assignment guidelines for G1 (K3b) and
the dedup/M4 pair guidelines (K3c) remain open, and #60 stays open until
they land.

**Identity.** Protocol id `k3-g2-v1`, registered in
`nlp.eval.faithfulness.RATIFIED_G2_PROTOCOLS`. The code pins this file's
SHA-256 (`K3_G2_V1_DOCUMENT_SHA256`) and records the protocol fingerprint
on every G2 scorecard. Any edit to this file fails the test suite until the
pin is deliberately updated. A *material* change — anything that could
turn one verdict into another — is a new id, `k3-g2-v2`, never an edit of
this one. Rounds drawn under an id keep its semantics forever; a round
drawn under `unratified` is never eligible, and is never re-labelled.

## 1. What the spec fixes, and what this protocol decides

**Spec-mandated** (`docs/PHASE_0_SPEC.md` §5 K3, §8): the unit is the
summary sentence; the question is whether it is *supported by its cited
source*; the gate is ≥ 95%; the population is every sentence from 2
randomly sampled days of soak-window data; two reviewers, disagreements
adjudicated; samples are drawn by tooling, not hand-picked; reviewers do
not review their own stage's output where avoidable; thresholds are fixed
before data; a gate marginally passed by measuring differently is a fail.

**`k3-g2-v1` policy:** everything in sections 2–7.

## 2. Evidence boundary

The review unit is **one generated sentence with its complete, ordered set
of cited evidence** — one sheet row. A reviewer judges it against only
what the row's `cited_evidence` shows for each cited story: **title,
description, outlet, publication time**. That is what the summary system
was given.

Not evidence: stories of the theme the sentence does not cite; the
publisher's page; anything at a URL; web search; the reviewer's own
knowledge. G2 measures faithfulness to the persisted cited evidence, not
real-world truth: a sentence that matches its evidence is supported even if
the world later proved the evidence wrong, and a true sentence its evidence
does not establish is unsupported.

Neighbouring sentences of the same summary (adjacent rows with the same
`artifact_id`) may be read **only** to resolve a referent — "it", "the
company", "the deal". They never support a claim.

## 3. The decision rule

A **material claim** is any of: an entity or its role; a number, amount,
unit or currency; a date, time or relative time ("today", "this week"); an
event's status (planned, announced, scheduled, occurred, completed); a
direction (rose, fell, raised, cut); an attribution (who said it); a
certainty level (reportedly, could, plans, according to); specificity or
scope; a comparison; a causal relationship; a temporal relationship
(before, after, while); a coverage-volume or prevalence claim (dominant,
widespread, most, several outlets, consensus).

**`supported`** — every material claim in the sentence is established by
the cited evidence, taken together, without adding anything from the list
above. Meaning-preserving paraphrase is allowed. Strict entailment that
drops irrelevant detail without changing the proposition is allowed.

**`unsupported`** — anything else. In particular:

- **Partial support** is unsupported. One unsupported material claim makes
  the whole sentence unsupported. There is no partial label.
- **Relevant but insufficient** evidence is unsupported.
- **Ambiguous or unverifiable** is unsupported: the burden is on the
  sentence, and no row leaves the denominator.
- **Plausible inference** beyond what the evidence states is unsupported,
  however reasonable.
- **Contradiction** of any cited story on a material claim is unsupported.
  If the cited stories conflict with each other, the sentence must keep
  the conflict or attribute each side; stating one side as settled fact is
  unsupported.

## 4. Claim-type rules

| claim | supported only when |
|---|---|
| **number** | it agrees with the evidence. Rounding only under explicitly approximate wording ("about", "nearly", "more than") that stays true. Reviewer arithmetic (a percentage from two figures, a sum) is not evidence. No unit or currency conversion. |
| **entity** | same entity in the same role. Company ↔ its ticker or common name (Meta / META, Alphabet / Google) is fine. Person ↔ role swaps, parent ↔ subsidiary, one organisation for another: unsupported. |
| **time** | tense and event status match: "plans to", "will", "is scheduled to" are not "has". A relative time is supported only by the evidence text or its publication time. |
| **causality** | a cited story itself asserts the relationship. Sequence is not cause: "after" is never "because", "on", "driven by" or "due to". Keep any attribution or hedge the evidence attaches. |
| **certainty** | every hedge and attribution survives: dropping "reportedly", "could", "plans", "sources say", "according to" is unsupported. |
| **paraphrase** | the meaning, strength and scope are unchanged. |
| **inference** | it is strict entailment; anything more is unsupported. |

## 5. Multiple citations

- The **union** of the cited stories may support the sentence; support may
  be distributed across them.
- **Every** citation on the sentence must support at least one of its
  material claims. A decorative or misleading citation makes the sentence
  unsupported (`MISCITATION`).
- Two separately supported facts may be stated together ("A, and B"). They
  may not be joined by a relationship no cited story states ("A because of
  B", "A after B", "A led to B"): that is `SYNTHESIS`, or `CAUSAL` when the
  relationship is causal.
- "Several", "multiple" or "both" reports or outlets needs that many
  distinct cited stories (or outlets) saying it.

## 6. Coverage framing — the only exemption

`ai.summarization.SYSTEM_PROMPT` (rule 9) tells the model to frame
summaries as descriptions of coverage. These **exact** sentence openings,
and nothing else, are read as product framing rather than as an
empirical claim about the day's coverage volume:

- `"Coverage today is dominated by "`
- `"The most-covered storyline is "`

(each quoted string ends in one space; registered as `K3_G2_V1_EXEMPT_FRAMINGS`, matched
by `nlp.eval.review.split_exempt_framing`). The match is exact and
sentence-initial — case, hyphen, tense and spacing as written — and only
those words are exempt. Whatever follows is judged under every rule above,
including volume words ("widespread", "multiple outlets"). Every other
dominance, prevalence, consensus, frequency, volume or outlet-count claim —
"Most coverage today…", "Coverage is dominated by…", "Coverage today was
dominated by…", "The most covered storyline is…", "Coverage today is
dominated by:", a framing in mid-sentence — is a material claim the cited
evidence must establish.

## 7. Reviewers, adjudication, reason codes

**Reviewers.** Exactly two, each identified by a stable handle (GitHub
username preferred). Ids are compared case-insensitively: `Alice` and
`alice` are one reviewer and are refused. Neither should own the
summarisation stage (A1–A3) where avoidable; if unavoidable, record who and
why with the round (process control: identity is recorded, not verified).

**Independence.** Each reviewer fills their own copy of the blank sheet,
alone. No discussion of rows, and no access to the other sheet, until both
completed sheets are committed (or their SHA-256s recorded in the round's
record).

**Verdicts.** `supported` or `unsupported`. Every `unsupported` verdict
opens `reviewer_notes` with exactly one reason code — the first failure
found — as `CODE` or `CODE: explanation`:

| code | the sentence… |
|---|---|
| `ADDITION` | adds a claim, detail or inference the evidence does not state |
| `CONTRADICTION` | contradicts a cited story, or settles a conflict between them |
| `NUMBER` | misstates a number, amount, unit or currency |
| `ENTITY` | names the wrong entity, person or role |
| `TEMPORAL` | misstates time, tense or event status |
| `CAUSAL` | asserts a causal relationship no cited story asserts |
| `CERTAINTY` | drops a hedge or attribution, or overstates certainty |
| `SYNTHESIS` | joins separately cited facts by a relationship none states |
| `MISCITATION` | carries a citation that supports none of its claims |
| `UNVERIFIABLE` | cannot be judged from the evidence (ambiguous, truncated, unresolvable referent) |

Codes are upper-case and from this list only. `supported` needs no notes.
`reviewed_at` and `adjudicated_at`, when filled, are ISO-8601
(`2026-07-27` or `2026-07-27T14:05:00Z`).

**Adjudication.** Agreement is the final verdict (`unanimous`). A
disagreement — two marked, different verdicts — is decided by a **third**
identity (never either reviewer, compared case-insensitively) in the
adjudication sheet, under this protocol, with `adjudication_notes`
required, and a reason code when the final verdict is `unsupported`. If
support cannot be established, the final verdict is `unsupported`. The
adjudication sheet holds disagreements only: a row the reviewers agreed on,
or one a reviewer left blank, is refused.

## 8. Score, completeness, eligibility

Numerator: census sentences whose final verdict is `supported`.
Denominator: every census sentence — every sentence of every reviewable
current artifact, all five tickers, on the two drawn days. PASS needs
numerator / denominator ≥ 0.95, compared exactly (19/20 meets it, 18/20 does
not). Adjudication states counted: `unanimous`, `resolved`. A blank
verdict, an open disagreement, a withheld artifact or a partition that
changed during sampling makes the review `INCOMPLETE`; unverified origin or
theme build, fewer than two reviewers, a development draw or threshold, or
any protocol but a ratified one makes it `NOT_ELIGIBLE`. Precedence:
`NOT_ELIGIBLE`, then `INCOMPLETE`, then `PASS` / `FAIL`. An open
disagreement is unfinished adjudication: under `k3-g2-v1` it scores
`INCOMPLETE`, not `NOT_ELIGIBLE`, and can never `PASS`. Sampling, the
denominator and the threshold are not changed by `k3-g2-v1`.

## 9. Pre-registration (process control)

Before running `sample-sentences`, commit a record of the round: the full
B4 soak window (start and end day), the seed, the round id, protocol
`k3-g2-v1`, and the two reviewers and the adjudicator. Only then sample,
with exactly those values. A round whose manifest disagrees with its
record, or whose record was committed after the draw, is not a release
round. Do not re-draw with another seed or window after seeing the days.

```
python -m tools.make_review_sheets sample-sentences --protocol k3-g2-v1 \
    --database "$PHASE0_DATABASE_PATH" \
    --window-start <soak start> --window-end <soak end> \
    --seed <registered seed> --round-id <id> --out docs/reviews/g2/<id>/g2.csv
```

## 10. Worked examples (hypothetical)

| # | cited evidence | sentence | verdict |
|---|---|---|---|
| 1 | [1] "Nvidia reports Q2 revenue of $30.0 billion" | "Nvidia reported second-quarter revenue of $30.0 billion [1]." | supported |
| 2 | [1] "Apple delays launch of smart-home display to 2026" | "Apple has pushed its smart-home display launch to 2026 [1]." | supported (paraphrase) |
| 3 | [1] "AMD wins Microsoft data-center chip order" | "AMD is gaining share in data-center chips [1]." | `ADDITION` (inference) |
| 4 | [1] "Tesla recalls 2,000 Cybertrucks over pedal issue" | "Tesla recalled 2,000 Cybertrucks over a pedal issue, its third recall this year [1]." | `ADDITION` (partial) |
| 5 | [1] "Meta unveils new Llama model" | "Meta unveiled a new Llama model, which analysts praised [1]." | `ADDITION` |
| 6 | [1] "AMD shares slip 2% in premarket" | "AMD shares rose in early trading [1]." | `CONTRADICTION` |
| 7 | [1] "Apple to buy back $90 billion of stock" | "Apple announced a $100 billion buyback [1]." | `NUMBER` |
| 8 | [1] "Nvidia CFO Colette Kress says supply improving" | "Nvidia CEO Jensen Huang said supply is improving [1]." | `ENTITY` |
| 9 | [1] "Tesla plans to open Mumbai showroom next month" | "Tesla opened a Mumbai showroom [1]." | `TEMPORAL` |
| 10 | [1] "Meta raises capex forecast to $40B"; [2] "Meta shares fall after earnings" | "Meta raised its capex forecast to $40 billion, and its shares fell after earnings [1][2]." | supported (distributed) |
| 11 | same [1], [2] | "Meta shares fell on its higher capex forecast [1][2]." | `CAUSAL` |
| 12 | [1] "Apple in talks to acquire startup X, sources say" | "Apple is acquiring startup X [1]." | `CERTAINTY` |
| 13 | [1] "AMD launches MI400 accelerator"; [2] "AMD CFO to speak at conference" | "AMD launched its MI400 accelerator [1][2]." | `MISCITATION` |
| 14 | [1] "Chipmaker shares slide on new export curbs" (no description, no company named) | "AMD shares slid on new export curbs [1]." | `UNVERIFIABLE` |
| 15 | [1] "Microsoft deal for AMD chips valued at $10B"; [2] "…valued at $12B" | "The deal is valued at $10 billion [1][2]." | `CONTRADICTION` (settles a conflict) |
| 16 | same [1], [2] | "Reports value the deal at between $10 billion and $12 billion [1][2]." | supported |
| 17 | [1] "Nvidia reports Q2 revenue of $30.0 billion" | "Coverage today is dominated by Nvidia's $30.0 billion second-quarter revenue [1]." | supported (exempt framing; remainder supported) |
| 18 | [1] "Tesla recalls 2,000 Cybertrucks" (one story) | "Coverage today is dominated by widespread reports of a Tesla recall [1]." | `ADDITION` ("widespread reports" is judged) |
| 19 | [1] "Apple to buy back $90 billion of stock" | "Most coverage today focuses on Apple's $90 billion buyback [1]." | `ADDITION` (not an exempt framing) |
| 20 | previous sentence: "Apple reported quarterly results [1]."; this row: [2] "Apple to buy back $90 billion of stock" | "It also announced a $90 billion buyback [2]." | supported ("it" resolved from the neighbour; support from [2]) |
