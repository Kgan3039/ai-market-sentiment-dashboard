# I5 provider observation

Observational only. No ingestion behavior was changed, nothing was written to the Phase 0 database, and no publisher mapping is implemented here.

## Window

- Generated at: `2026-10-14T23:00:16.187398+00:00`
- Observed from `2026-10-14T17:35:41.902185+00:00` to `2026-10-14T23:00:03.587686+00:00` across 5 attempts 2400s apart, a real span of 19461.7s
- Code commit: `ea24186bc15d14d61202e0169571022ee91747f0`
- Working tree dirty: `False`
- yfinance `0.2.65`, Python `3.11.15`
- Tickers: `AAPL`, `AMD`, `META`, `NVDA`, `TSLA`
- Feeds:  (`config/feeds.yaml` sha256 `3c12ec954a7a4372`)

## Yahoo `external_id` verdict

**SAFE TO IMPLEMENT** — field `uuid`

'uuid' was present on every valid item, unchanged across attempts, positions, and tickers, never shared by two canonical URLs, and held by 12 articles across observations at least 7200s apart.

Decision G's bar is met by article, not by run: 12 of 60 repeated articles carried the identifier across observations at least 2h apart. The longest was 5.41h, the median repeated article 1.55h, the shortest 0.00h.

The observation window itself spanned 5.41h. That is context, not evidence: a run's length says nothing about an identifier, and it does not enter the verdict.

### Candidate fields

| field | present | coverage | distinct ids | articles | repeated | repeats over the bar | longest repeat | cross-ticker | unstable | collisions | semantics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `id` | 0/233 | 0% | 0 | 102 | 59 | 0/0 | n/a | 14 | 0 | 0 | `absent` |
| `uuid` | 233/233 | 100% | 102 | 102 | 59 | 12/60 | 5.41h | 14 | 0 | 0 | `article_scoped` |
| `content.id` | 0/233 | 0% | 0 | 102 | 59 | 0/0 | n/a | 14 | 0 | 0 | `absent` |

### Which candidate the verdict rests on

Ranked on the evidence the verdict is read off: scope first, then whether the candidate cleared decision G's per-article bar, then coverage, then the field phase0 already reads.

1. `uuid` — `article_scoped`, meets decision G (12 of 60 repeated articles over the bar, longest 5.41h), 100% coverage, 0 collisions, 0 unstable articles
2. `id` — `absent`, unqualified (0 of 0 repeated articles over the bar, longest n/a), 0% coverage, 0 collisions, 0 unstable articles
3. `content.id` — `absent`, unqualified (0 of 0 repeated articles over the bar, longest n/a), 0% coverage, 0 collisions, 0 unstable articles

Selected `uuid`, decided on semantics: 'uuid' is article_scoped, which ranks safer than the absent of the next candidate 'id'.

### Do the candidates agree?

- `id` and `uuid` were never both present.
- `id` and `content.id` were never both present.
- `uuid` and `content.id` were never both present.

**`id`** — no valid item carried the field.

- 0/233 valid items carried id
- 102 distinct articles observed
- 59 articles observed in more than one attempt
- 0 articles observed more than once carrying id, 0 of them at least 7200s apart
- 50 articles observed at more than one response position
- 14 articles observed under more than one ticker

**`uuid`** — one identifier per article, unchanged across attempts, positions, and tickers, and never shared by two articles.

- 233/233 valid items carried uuid
- 102 distinct articles observed
- 59 articles observed in more than one attempt
- 60 articles observed more than once carrying uuid, 12 of them at least 7200s apart
- 50 articles observed at more than one response position
- 14 articles observed under more than one ticker

**`content.id`** — no valid item carried the field.

- 0/233 valid items carried content.id
- 102 distinct articles observed
- 59 articles observed in more than one attempt
- 0 articles observed more than once carrying content.id, 0 of them at least 7200s apart
- 50 articles observed at more than one response position
- 14 articles observed under more than one ticker

## Yahoo source strings

| stored source | raw publisher | field | provider sourceId | article hosts | observations | tickers |
| --- | --- | --- | --- | --- | --- | --- |
| `yahoo:24/7 Wall St.` | 24/7 Wall St. | `publisher` | — | `finance.yahoo.com` | 13 | AAPL, AMD, META |
| `yahoo:Barrons.com` | Barrons.com | `publisher` | — | `finance.yahoo.com` | 33 | AAPL, AMD, META, NVDA, TSLA |
| `yahoo:Benzinga` | Benzinga | `publisher` | — | `finance.yahoo.com` | 7 | META, TSLA |
| `yahoo:Blockspace` | Blockspace | `publisher` | — | `finance.yahoo.com` | 1 | AMD |
| `yahoo:Business Wire` | Business Wire | `publisher` | — | `finance.yahoo.com` | 6 | AAPL, META |
| `yahoo:GuruFocus.com` | GuruFocus.com | `publisher` | — | `finance.yahoo.com` | 21 | AAPL, AMD, META, NVDA, TSLA |
| `yahoo:Insider Monkey` | Insider Monkey | `publisher` | — | `finance.yahoo.com` | 1 | AMD |
| `yahoo:International Business Times` | International Business Times | `publisher` | — | `finance.yahoo.com` | 3 | AAPL |
| `yahoo:Investing.com` | Investing.com | `publisher` | — | `finance.yahoo.com` | 1 | META |
| `yahoo:Investor's Business Daily` | Investor's Business Daily | `publisher` | — | `finance.yahoo.com` | 6 | AMD, NVDA |
| `yahoo:MT Newswires` | MT Newswires | `publisher` | — | `finance.yahoo.com` | 10 | AAPL, META, NVDA |
| `yahoo:MarketBeat` | MarketBeat | `publisher` | — | `finance.yahoo.com` | 2 | AAPL |
| `yahoo:Moby` | Moby | `publisher` | — | `finance.yahoo.com` | 7 | NVDA |
| `yahoo:Moneywise` | Moneywise | `publisher` | — | `finance.yahoo.com` | 2 | AAPL |
| `yahoo:Motley Fool` | Motley Fool | `publisher` | — | `finance.yahoo.com` | 35 | AAPL, META, NVDA, TSLA |
| `yahoo:PR Newswire` | PR Newswire | `publisher` | — | `finance.yahoo.com` | 1 | META |
| `yahoo:Proactive` | Proactive | `publisher` | — | `finance.yahoo.com` | 1 | META |
| `yahoo:Simply Wall St.` | Simply Wall St. | `publisher` | — | `finance.yahoo.com` | 12 | AMD, META, NVDA, TSLA |
| `yahoo:Stocktwits` | Stocktwits | `publisher` | — | `finance.yahoo.com` | 10 | AMD, META, TSLA |
| `yahoo:TMX Newsfile` | TMX Newsfile | `publisher` | — | `finance.yahoo.com` | 8 | AAPL, NVDA, TSLA |
| `yahoo:TechCrunch` | TechCrunch | `publisher` | — | `finance.yahoo.com` | 5 | META, TSLA |
| `yahoo:The Wall Street Journal` | The Wall Street Journal | `publisher` | — | `finance.yahoo.com` | 15 | AAPL, AMD, NVDA |
| `yahoo:TheStreet` | TheStreet | `publisher` | — | `finance.yahoo.com` | 7 | AAPL, META |
| `yahoo:Trefis` | Trefis | `publisher` | — | `finance.yahoo.com` | 17 | AMD, META, NVDA, TSLA |
| `yahoo:Yahoo Finance` | Yahoo Finance | `publisher` | — | `finance.yahoo.com` | 7 | AAPL, AMD, TSLA |
| `yahoo:etf.com` | etf.com | `publisher` | — | `finance.yahoo.com` | 2 | META, NVDA |

No Yahoo record failed normalization in this window.

## RSS source strings

| stored source | resolved host | feeds | observations | articles | statuses |
| --- | --- | --- | --- | --- | --- |

Example entries, showing that a feed's host is not the article's:


## Cross-source publisher equivalence

| Yahoo source | RSS source | verdict | evidence |
| --- | --- | --- | --- |
| — | — | — | no pair carried any observed signal |

0 further pairs are `UNKNOWN`: nothing observed relates them, and they are counted rather than listed.

**No mapping is implemented from this table.** It is the input to I5's explicit reviewed publisher map, not the map.

## Limitations

- Stability is only as strong as the span each repeated article was watched over, which is what the verdict is gated on. How long the run lasted is reported beside it as context and decides nothing: a long run of closely spaced repeats tests a short claim.
- The verdict rests on one candidate, ranked on scope, then decision G's per-article bar, then coverage. A candidate below it is not thereby unsafe -- it may simply be less well evidenced in this window -- so the whole ranking is reported and not only its winner.
- Observation issues an unconditional GET, because reading a feed's stored ETag would mean reading source_state; a scheduled fetch sends conditional headers and can receive 304, which never appears here.
- Article identity is proxied by the canonical URL phase0 stores. Two URLs for one article would read as two articles, and one URL reused for two articles would read as one. One identifier on two canonical URLs is therefore counted as a collision, headlines included: suppressing it would need a URL-alias rule defined on its own evidence, and none is defined here.
- Only the five approved tickers and the enabled feeds were observed. A publisher that never appeared in this window is not evidence of absence.
- Every number here describes one observation window. Provider payload shapes have changed before -- the legacy uuid shape is why the code reads two -- so a verdict is a statement about now, not forever.
