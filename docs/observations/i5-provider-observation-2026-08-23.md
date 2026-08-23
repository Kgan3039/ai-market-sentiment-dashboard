# I5 provider observation

Observational only. No ingestion behavior was changed, nothing was written to the Phase 0 database, and no publisher mapping is implemented here.

## Window

- Generated at: `2026-08-23T19:53:56.875545+00:00`
- Observed from `2026-08-22T23:14:51.090361+00:00` to `2026-08-23T19:53:55.809094+00:00` across 4 attempts 2500s apart, a real span of 74344.7s
- Code commit: `654ec87c4aa81a8b64966089e193627af9b2041d`
- Working tree dirty: `True`
- yfinance `0.2.65`, Python `3.11.15`
- Tickers: `AAPL`, `AMD`, `META`, `NVDA`, `TSLA`
- Feeds: `marketwatch-top-stories`, `techcrunch-latest` (`config/feeds.yaml` sha256 `3c12ec954a7a4372`)

## Yahoo `external_id` verdict

**SAFE TO IMPLEMENT** — field `id`

'id' was present on every valid item, unchanged across attempts, positions, and tickers, and never shared.

Stability was watched over 20.65h, meeting decision G's bar of 2h apart.

### Candidate fields

| field | present | coverage | distinct ids | articles | repeated | cross-ticker | unstable | collisions | semantics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `id` | 200/200 | 100% | 86 | 86 | 51 | 10 | 0 | 0 | `article_scoped` |
| `uuid` | 0/200 | 0% | 0 | 86 | 51 | 10 | 0 | 0 | `absent` |
| `content.id` | 200/200 | 100% | 86 | 86 | 51 | 10 | 0 | 0 | `article_scoped` |

### Do the candidates agree?

- `id` and `uuid` were never both present.
- `id` and `content.id` carried the same value on all 200 items that had both.
- `uuid` and `content.id` were never both present.

**`id`** — one identifier per article, unchanged across attempts, positions, and tickers, and never shared by two articles.

- 200/200 valid items carried id
- 86 distinct articles observed
- 51 articles observed in more than one attempt
- 43 articles observed at more than one response position
- 10 articles observed under more than one ticker

**`uuid`** — no valid item carried the field.

- 0/200 valid items carried uuid
- 86 distinct articles observed
- 51 articles observed in more than one attempt
- 43 articles observed at more than one response position
- 10 articles observed under more than one ticker

**`content.id`** — one identifier per article, unchanged across attempts, positions, and tickers, and never shared by two articles.

- 200/200 valid items carried content.id
- 86 distinct articles observed
- 51 articles observed in more than one attempt
- 43 articles observed at more than one response position
- 10 articles observed under more than one ticker

## Yahoo source strings

| stored source | raw publisher | field | provider sourceId | article hosts | observations | tickers |
| --- | --- | --- | --- | --- | --- | --- |
| `yahoo:24/7 Wall St.` | 24/7 Wall St. | `content.provider.displayName` | `24_7_wall_st__718` | `247wallst.com` | 24 | AAPL, AMD, META, NVDA, TSLA |
| `yahoo:AFP` | AFP | `content.provider.displayName` | `afp.com` | `finance.yahoo.com` | 2 | AAPL |
| `yahoo:Barchart` | Barchart | `content.provider.displayName` | `barchart_com_477` | `www.barchart.com` | 3 | AMD, TSLA |
| `yahoo:Barrons.com` | Barrons.com | `content.provider.displayName` | `Barrons.com` | `www.barrons.com` | 4 | AAPL, META |
| `yahoo:Benzinga` | Benzinga | `content.provider.displayName` | `benzinga_79` | `finance.yahoo.com` | 13 | TSLA |
| `yahoo:Fortune` | Fortune | `content.provider.displayName` | `fortune_175` | `finance.yahoo.com` | 4 | META |
| `yahoo:GuruFocus.com` | GuruFocus.com | `content.provider.displayName` | `us.finance.gurufocus` | `finance.yahoo.com` | 5 | AMD, META |
| `yahoo:Insider Monkey` | Insider Monkey | `content.provider.displayName` | `insidermonkey.com` | `finance.yahoo.com` | 13 | AAPL, AMD, META |
| `yahoo:Investing.com` | Investing.com | `content.provider.displayName` | `investing_com_631` | `finance.yahoo.com` | 9 | AAPL, AMD, NVDA |
| `yahoo:Investor's Business Daily` | Investor's Business Daily | `content.provider.displayName` | `ibd.com` | `www.investors.com` | 3 | NVDA |
| `yahoo:MarketBeat` | MarketBeat | `content.provider.displayName` | `marketbeat_955` | `www.marketbeat.com` | 2 | TSLA |
| `yahoo:Motley Fool` | Motley Fool | `content.provider.displayName` | `motleyfool.com` | `www.fool.com` | 86 | AAPL, AMD, META, NVDA, TSLA |
| `yahoo:Simply Wall St.` | Simply Wall St. | `content.provider.displayName` | `simply_wall_st__316` | `finance.yahoo.com` | 9 | AMD, META, TSLA |
| `yahoo:Stocktwits` | Stocktwits | `content.provider.displayName` | `stocktwits_383` | `stocktwits.com` | 4 | AAPL, TSLA |
| `yahoo:The Wall Street Journal` | The Wall Street Journal | `content.provider.displayName` | `wsj.com` | `www.wsj.com` | 4 | TSLA |
| `yahoo:TheStreet` | TheStreet | `content.provider.displayName` | `thestreet_881` | `www.thestreet.com` | 5 | AAPL, META, TSLA |
| `yahoo:Trefis` | Trefis | `content.provider.displayName` | `trefis_142` | `www.trefis.com` | 4 | AMD |
| `yahoo:Yahoo Finance` | Yahoo Finance | `content.provider.displayName` | `yahoofinance.com` | `finance.yahoo.com` | 6 | META, NVDA |

No Yahoo record failed normalization in this window.

## RSS source strings

| stored source | resolved host | feeds | observations | articles | statuses |
| --- | --- | --- | --- | --- | --- |
| `rss:techcrunch.com` | `techcrunch.com` | techcrunch-latest | 80 | 26 | valid |
| `rss:www.marketwatch.com` | `www.marketwatch.com` | marketwatch-top-stories | 40 | 16 | valid |

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

36 further pairs are `UNKNOWN`: nothing observed relates them, and they are counted rather than listed.

**No mapping is implemented from this table.** It is the input to I5's explicit reviewed publisher map, not the map.

## Limitations

- Stability is only as strong as the span it was watched over. The artifact reports that span next to the verdict; a verdict marked provisional was not watched long enough to meet decision G.
- Observation issues an unconditional GET, because reading a feed's stored ETag would mean reading source_state; a scheduled fetch sends conditional headers and can receive 304, which never appears here.
- Article identity is proxied by the canonical URL phase0 stores. Two URLs for one article would read as two articles, and one URL reused for two articles would read as one.
- Only the five approved tickers and the enabled feeds were observed. A publisher that never appeared in this window is not evidence of absence.
- Every number here describes one observation window. Provider payload shapes have changed before -- the legacy uuid shape is why the code reads two -- so a verdict is a statement about now, not forever.
