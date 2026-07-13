# Phase 0 RSS sources and ticker aliases

Issue #58 supplies configuration only; issue #62 owns the RSS fetcher and
relevance filter. The approved list intentionally has two sources, leaving
headroom under the three-feed Phase 0 cap while keeping legal and operational
risk small. Yahoo Finance remains the primary ticker-specific source; both
RSS sources are supplemental.

## Selected sources

| Source | Why selected | Feed metadata | Limitations |
| --- | --- | --- | --- |
| MarketWatch Top Stories | Supplemental general financial and market coverage of earnings, macro events, and the five target companies. Its live Phase 0 check returned 10 RSS items. | `title`, `link`, `description`, `pubDate` | It is a general feed, so most items will not map to a target ticker. Its feed-specific reuse terms were not located during selection; keep attribution and links, store only feed metadata, and review before public/commercial deployment. |
| TechCrunch Latest | Supplemental technology, product, and company coverage that complements MarketWatch. Its live Phase 0 check returned 20 RSS items. | `title`, `link`, `description`, `pubDate` | It is not comprehensive financial-market coverage and not every item is market-relevant. TechCrunch permits display of feed-provided content with attribution and a link to the full article, prohibits modifying feed content or adding advertising, and may discontinue the feed. |

## Rejected candidates

| Candidate | Decision | Reason |
| --- | --- | --- |
| Reuters Business (`https://feeds.reuters.com/reuters/businessNews`) | Rejected | The legacy host did not resolve during the July 13, 2026 live check, so it could not provide XML or required metadata. |
| CNBC Markets (`https://www.cnbc.com/id/15839069/device/rss/rss.xml`) | Rejected | The endpoint returned `403` HTML (Akamai access denied), not an RSS document, during the live check. |

## Consumption contract for issue #62

Read enabled entries from `config/feeds.yaml`; use the listed field names to
normalize title, URL, description, and publication time. Poll every 30 minutes
with conditional requests where supported, and isolate source failures so one
feed does not abort a run. Store the raw item before relevance filtering.

Read `config/aliases.yaml` case-insensitively over title plus description,
using phrase/word boundaries. Exclusions override a potential match. A
context-required alias (bare `Tesla`, `Apple`, `Meta`, and `AMD`) needs a listed
same-item context term. If more than one ticker matches, flag it in `run_log`
and leave the ticker unassigned; never choose one heuristically. Bare Elon Musk
coverage is not TSLA coverage unless Tesla is also mentioned.

Run the local configuration check with:

```sh
python3 -m pip install -r requirements-dev.txt
python3 tools/validate_phase0_config.py
python3 -m unittest discover -s tests -p 'test_phase0_config.py'
```

`requirements-dev.txt` contains only PyYAML, which the standalone validation
utility and its unit tests need to parse the YAML configuration. It introduces
no runtime fetching or ingestion dependency.
