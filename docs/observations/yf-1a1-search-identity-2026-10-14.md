# YF-1A.1: Yahoo Search identity and contract validation

Companion to `i5-provider-observation-2026-10-14.{json,md}`. That pair is
the observer's own artifact and holds the Decision G verdict. This note
records how it was collected and the cross-checks the observer does not
make. Like the artifact, it describes one window and is evidence, not a
test.

## Why the 2026-10-14 pair exists

By October 2026 Yahoo answered `POST finance.yahoo.com/xhr/ncp` (the
endpoint behind `yf.Ticker(t).news` in yfinance 0.2.65 through 1.7.0) with
`404 {"message":"Not Found"}`. yfinance turns that into `[]`, so every AWS-0
run (EC2 and the Mac control) settled five empty tickers. The Search
endpoint, `GET query2.finance.yahoo.com/v1/finance/search`, served the same
five tickers normally (YF-1A, 2026-10-14).

## How it was collected

- The unmodified observer at commit `ea24186`, run from a wrapper that
  swapped its Yahoo fetch for Search with `newsCount=10`. The tool's
  default fetch is Search since YF-1B, so a plain re-run now asks the same
  endpoint.
- yfinance's `YfData.cache_get` keeps the first Search answer per query for
  the life of the process. The wrapper cleared it before every request, so
  each attempt was a real fetch. YF-1B avoids that cache altogether
  (`phase0/yahoo_search.py`).
- 5 attempts between 17:35 and 23:00 UTC (5.41 h real span; the gaps were
  uneven, consistent with the host sleeping). `--skip-rss`. One-second pause
  between tickers. Out of 25 Search requests, none failed or was
  rate-limited.

## Verdict (from the artifact)

**SAFE TO IMPLEMENT, field `uuid`.**

| Finding | Result |
| --- | --- |
| Presence | 233/233 valid items; `id` and `content.id` 0/233 |
| Stability | 59 articles seen more than once, none with two values; 12 held one `uuid` across observations at least 2 h apart (longest 5.41 h) |
| Semantics | `article_scoped`: 14 cross-ticker articles and 50 seen at more than one position each kept one value |
| Collisions | 0: 102 identifiers for 102 canonical URLs |

Fewer articles cleared the 2 h bar than in August (12 vs 37), because Search
results turn over faster: 43 of 102 articles were seen only once. The bar is
met per article, but the evidence is thinner.

## Cross-checks outside the artifact

**`uuid` is Yahoo's article identifier, not a Search-local one.**
- Yahoo's per-ticker RSS (`feeds.finance.yahoo.com/rss/2.0/headline`) was
  snapshotted at the first and last attempts. Every article present in both
  sources carried the same value: Search `uuid` equals RSS `<guid>`, 34/34
  and 33/33.
- Barron's links take the form `finance.yahoo.com/m/<uuid>/...`; the UUID in
  the path equalled the item's `uuid` on 7/7.
- Seven August headlines (from the 2026-08-23 artifact) were searched by
  title. Six had aged out of Search. The one still returned,
  "4 Low-Cost ETFs Every Investor Should Own", carried
  `158484a0-bfd5-3828-b965-441ae440747e` in both places: the retired
  endpoint's `id` and today's Search `uuid`. That is one match and no
  contradiction.

**`relatedTickers` attribution.**
- The requested ticker was present on 233/233 items, and Search's top quote
  match was the requested symbol on every response.
- The lists are not ownership evidence:
  - the requested ticker came first on only 75/233;
  - 96/233 items named more than one of the five tickers;
  - lists ran to 13 symbols and included out-of-universe instruments (`SPCX`,
    `BRK-B`, `ANTH.PVT`, `^IXIC`);
  - 3 lists repeated a symbol;
  - one article's list grew between attempts.

**Field shape and stability.**
- `uuid`, `title`, `publisher`, `link`, `providerPublishTime`, `type`
  (`STORY`) and `relatedTickers` were present on every item, and every item
  normalized through the production `normalize_yahoo_item`.
- `link` was always a `finance.yahoo.com` URL and never changed for an
  article. `publisher` never changed, and 9 of 15 publisher names matched
  August's exactly.
- Two live-updated articles kept their `uuid` and `link` while
  `providerPublishTime` moved by 2.6 h and 1.5 h, and one of them was
  retitled. The `(source, canonical_url)` insert conflict keeps the first
  version, so dedup never sees both.

## Not retained

Following this directory's rule, whole raw responses (`raw.jsonl`) and the
RSS snapshots (`rss.jsonl`) are not committed. They were collected in a
session scratch directory, and the conclusions drawn from them are the ones
above.
