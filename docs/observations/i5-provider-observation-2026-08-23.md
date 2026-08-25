# I5 provider observation

Observational only. No ingestion behavior was changed, nothing was written to the Phase 0 database, and no publisher mapping is implemented here.

## Window

- Generated at: `2026-08-23T23:06:44.140822+00:00`
- Observed from `2026-08-23T20:46:39.185889+00:00` to `2026-08-23T23:06:43.195915+00:00` across 4 attempts 2800s apart, a real span of 8404.01s
- Code commit: `1d71cc1d213ce78c56dd1a1dd174c620ada106c7`
- Working tree dirty: `True`
- yfinance `0.2.65`, Python `3.11.15`
- Tickers: `AAPL`, `AMD`, `META`, `NVDA`, `TSLA`
- Feeds: `marketwatch-top-stories`, `techcrunch-latest` (`config/feeds.yaml` sha256 `3c12ec954a7a4372`)

## Yahoo `external_id` verdict

**SAFE TO IMPLEMENT** — field `id`

'id' was present on every valid item, unchanged across attempts, positions, and tickers, never shared by two canonical URLs, and held by 37 articles across observations at least 7200s apart.

Decision G's bar is met by article, not by run: 37 of 48 repeated articles carried the identifier across observations at least 2h apart. The longest was 2.33h, the median repeated article 2.33h, the shortest 0.78h.

The observation window itself spanned 2.33h. That is context, not evidence: a run's length says nothing about an identifier, and it does not enter the verdict.

### Candidate fields

| field | present | coverage | distinct ids | articles | repeated | repeats over the bar | longest repeat | cross-ticker | unstable | collisions | semantics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `id` | 200/200 | 100% | 56 | 56 | 48 | 37/48 | 2.33h | 4 | 0 | 0 | `article_scoped` |
| `uuid` | 0/200 | 0% | 0 | 56 | 48 | 0/0 | n/a | 4 | 0 | 0 | `absent` |
| `content.id` | 200/200 | 100% | 56 | 56 | 48 | 37/48 | 2.33h | 4 | 0 | 0 | `article_scoped` |

### Which candidate the verdict rests on

Ranked on the evidence the verdict is read off: scope first, then whether the candidate cleared decision G's per-article bar, then coverage, then the field phase0 already reads.

1. `id` — `article_scoped`, meets decision G (37 of 48 repeated articles over the bar, longest 2.33h), 100% coverage, 0 collisions, 0 unstable articles
2. `content.id` — `article_scoped`, meets decision G (37 of 48 repeated articles over the bar, longest 2.33h), 100% coverage, 0 collisions, 0 unstable articles
3. `uuid` — `absent`, unqualified (0 of 0 repeated articles over the bar, longest n/a), 0% coverage, 0 collisions, 0 unstable articles

Selected `id`: 'id' and 'content.id' are indistinguishable on this evidence, and 'id' takes the documented tie-break as the field phase0 already reads.

### Do the candidates agree?

- `id` and `uuid` were never both present.
- `id` and `content.id` carried the same value on all 200 items that had both.
- `uuid` and `content.id` were never both present.

**`id`** — one identifier per article, unchanged across attempts, positions, and tickers, and never shared by two articles.

- 200/200 valid items carried id
- 56 distinct articles observed
- 48 articles observed in more than one attempt
- 48 articles observed more than once carrying id, 37 of them at least 7200s apart
- 19 articles observed at more than one response position
- 4 articles observed under more than one ticker

**`uuid`** — no valid item carried the field.

- 0/200 valid items carried uuid
- 56 distinct articles observed
- 48 articles observed in more than one attempt
- 0 articles observed more than once carrying uuid, 0 of them at least 7200s apart
- 19 articles observed at more than one response position
- 4 articles observed under more than one ticker

**`content.id`** — one identifier per article, unchanged across attempts, positions, and tickers, and never shared by two articles.

- 200/200 valid items carried content.id
- 56 distinct articles observed
- 48 articles observed in more than one attempt
- 48 articles observed more than once carrying content.id, 37 of them at least 7200s apart
- 19 articles observed at more than one response position
- 4 articles observed under more than one ticker

## Yahoo source strings

| stored source | raw publisher | field | provider sourceId | article hosts | observations | tickers |
| --- | --- | --- | --- | --- | --- | --- |
| `yahoo:24/7 Wall St.` | 24/7 Wall St. | `content.provider.displayName` | `24_7_wall_st__718` | `247wallst.com` | 31 | AAPL, AMD, META, NVDA, TSLA |
| `yahoo:Barchart` | Barchart | `content.provider.displayName` | `barchart_com_477` | `www.barchart.com` | 4 | TSLA |
| `yahoo:Benzinga` | Benzinga | `content.provider.displayName` | `benzinga_79` | `finance.yahoo.com` | 12 | TSLA |
| `yahoo:Fortune` | Fortune | `content.provider.displayName` | `fortune_175` | `finance.yahoo.com` | 4 | META |
| `yahoo:Insider Monkey` | Insider Monkey | `content.provider.displayName` | `insidermonkey.com` | `finance.yahoo.com` | 12 | AAPL, AMD, META |
| `yahoo:Investing.com` | Investing.com | `content.provider.displayName` | `investing_com_631` | `finance.yahoo.com` | 8 | AAPL, AMD |
| `yahoo:Investor's Business Daily` | Investor's Business Daily | `content.provider.displayName` | `ibd.com` | `www.investors.com` | 3 | NVDA |
| `yahoo:MarketBeat` | MarketBeat | `content.provider.displayName` | `marketbeat_955` | `www.marketbeat.com` | 4 | TSLA |
| `yahoo:Motley Fool` | Motley Fool | `content.provider.displayName` | `motleyfool.com` | `www.fool.com` | 90 | AAPL, AMD, META, NVDA, TSLA |
| `yahoo:Reuters` | Reuters | `content.provider.displayName` | `reuters-finance.com` | `finance.yahoo.com` | 1 | NVDA |
| `yahoo:Simply Wall St.` | Simply Wall St. | `content.provider.displayName` | `simply_wall_st__316` | `finance.yahoo.com` | 12 | AMD, META, TSLA |
| `yahoo:The Wall Street Journal` | The Wall Street Journal | `content.provider.displayName` | `wsj.com` | `www.wsj.com` | 2 | TSLA |
| `yahoo:TheStreet` | TheStreet | `content.provider.displayName` | `thestreet_881` | `www.thestreet.com` | 1 | NVDA |
| `yahoo:Trefis` | Trefis | `content.provider.displayName` | `trefis_142` | `www.trefis.com` | 4 | AMD |
| `yahoo:Yahoo Finance` | Yahoo Finance | `content.provider.displayName` | `yahoofinance.com` | `finance.yahoo.com` | 12 | META, NVDA |

No Yahoo record failed normalization in this window.

## RSS source strings

| stored source | resolved host | feeds | observations | articles | statuses |
| --- | --- | --- | --- | --- | --- |
| `rss:techcrunch.com` | `techcrunch.com` | techcrunch-latest | 80 | 20 | valid |
| `rss:www.marketwatch.com` | `www.marketwatch.com` | marketwatch-top-stories | 40 | 11 | valid |

Example entries, showing that a feed's host is not the article's:

- feed `techcrunch-latest` → `rss:techcrunch.com`  
  link: `https://techcrunch.com/2026/08/21/anthropics-opus-4-6-is-a-smut-machine/`  
  stored canonical URL: `https://techcrunch.com/2026/08/21/anthropics-opus-4-6-is-a-smut-machine`  
  external_id: `https://techcrunch.com/?p=3155131`
- feed `marketwatch-top-stories` → `rss:www.marketwatch.com`  
  link: `https://www.marketwatch.com/story/canada-announces-retaliatory-tariffs-on-u-s-goods-after-trade-talks-break-down-45081c2f?mod=mw_rss_topstories`  
  stored canonical URL: `https://www.marketwatch.com/story/canada-announces-retaliatory-tariffs-on-u-s-goods-after-trade-talks-break-down-45081c2f?mod=mw_rss_topstories`  
  external_id: `WP-MKTW-0005194199`

## Cross-source publisher equivalence

| Yahoo source | RSS source | verdict | evidence |
| --- | --- | --- | --- |
| — | — | — | no pair carried any observed signal |

30 further pairs are `UNKNOWN`: nothing observed relates them, and they are counted rather than listed.

**No mapping is implemented from this table.** It is the input to I5's explicit reviewed publisher map, not the map.

## Limitations

- Stability is only as strong as the span each repeated article was watched over, which is what the verdict is gated on. How long the run lasted is reported beside it as context and decides nothing: a long run of closely spaced repeats tests a short claim.
- The verdict rests on one candidate, ranked on scope, then decision G's per-article bar, then coverage. A candidate below it is not thereby unsafe -- it may simply be less well evidenced in this window -- so the whole ranking is reported and not only its winner.
- Observation issues an unconditional GET, because reading a feed's stored ETag would mean reading source_state; a scheduled fetch sends conditional headers and can receive 304, which never appears here.
- Article identity is proxied by the canonical URL phase0 stores. Two URLs for one article would read as two articles, and one URL reused for two articles would read as one. One identifier on two canonical URLs is therefore counted as a collision, headlines included: suppressing it would need a URL-alias rule defined on its own evidence, and none is defined here.
- Only the five approved tickers and the enabled feeds were observed. A publisher that never appeared in this window is not evidence of absence.
- Every number here describes one observation window. Provider payload shapes have changed before -- the legacy uuid shape is why the code reads two -- so a verdict is a statement about now, not forever.
