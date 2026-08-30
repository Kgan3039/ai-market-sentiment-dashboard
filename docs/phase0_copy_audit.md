# Phase 0 Copy And UI-State Audit

**Issue:** B3 / #75
**Owner:** Mihir
**Audit status:** Automated checks complete; Kartik sign-off and issue
attachments remain required.

This audit covers the rendered Phase 0 page in `frontend/src/App.jsx`. The
legacy dashboard components are not mounted by `frontend/src/main.jsx` and are
outside this page-level audit.

## Automated Evidence

- `npm test` covers successful, loading, error, empty, stale, degraded, and
  ticker-switch states.
- `npm run lint` and `npm run build` validate the frontend source and build.
- `backend/tests/test_phase0_api.py` checks the fixture content and rendered
  product copy against `config/banned_phrases.txt`.
- The API fixture endpoints return the five tabs, cited themes, Other coverage,
  and freshness data used by the page.

## Copy Checklist

| Surface | Approved source | Implementation status |
| --- | --- | --- |
| Page title and browser title | `Ticker Narratives` | Pass |
| Five ticker tabs | Copy-deck ticker pattern | Pass; supplied by API metadata |
| Coverage framing | `Key narratives around today’s move` | Pass |
| Theme heading | `Themes dominating current coverage` | Pass |
| Theme, Other coverage, Stories, Cited sources, Open source | Copy deck | Pass |
| Fresh and stale stamps | Copy deck freshness/stale templates | Pass |
| Loading, error, empty, and degraded states | Copy deck state messages | Pass |
| Standing disclosure | Copy deck disclosure | Pass |
| Generated labels and summaries | Banned-phrase rules and citations | Pass in fixture contract tests |

The story/outlet count grammar and `Stories (count)` are data-derived labels
required by B2. Kartik should explicitly confirm this formatting during the
human sign-off because the copy deck names the labels but not their count
placeholders.

## Remaining Human Evidence

1. Capture desktop and 375px screenshots against the fixture page, then attach
   them to issue #75.
2. Have Kartik review the count placeholders above and record sign-off on the
   issue.
3. Repeat the same checklist against the real API after the B1 SQLite adapter
   is enabled.
