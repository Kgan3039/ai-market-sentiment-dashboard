# Backend Notes

## Dashboard Data Providers

The dashboard data service uses yfinance/Yahoo Finance for headline items and
company fundamentals because the backend already depends on yfinance for market
data. These provider calls are cached in memory for `DATA_SERVICE_CACHE_TTL_SECONDS`
seconds, defaulting to 900 seconds.

If a fresh provider request fails, the service falls back to stale cached data
when available and marks the dashboard availability status as `fallback`. If no
usable data is available, the service returns an empty headlines array or `null`
fundamentals and marks that component as `unavailable`. The service does not
fabricate demo headlines or fundamentals.
