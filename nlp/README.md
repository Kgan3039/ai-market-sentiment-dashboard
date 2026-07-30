## Phase 0 embeddings

`nlp.embeddings` provides the shared local embedding API used by the Phase 0
deduplication, clustering, and theme-stability stages. It uses
`sentence-transformers/all-MiniLM-L6-v2` by default, loads it lazily, and
pins a model revision for replayability. It L2-normalizes float32 vectors.
Title and description whitespace is collapsed; when both are present, the
exact model input is `title + "\n\n" + description`.

Model files use the sentence-transformers/Hugging Face cache. Set
`TICKER_NARRATIVES_MODEL_CACHE` or pass `cache_location` to select a cache
outside the repository.

Persistence is intentionally behind the `EmbeddingRepository` protocol. Main
does not yet contain issue #57's write repository, so this module does not
create a parallel SQLite layer. The eventual adapter must atomically implement
`get_embedding()` and `upsert_embedding()` and persist all fields in
`PersistedEmbedding`. Unit tests recreate repository adapters over shared
in-memory backing state; they do not claim durable SQLite restart coverage.

Run the hardware-sensitive, fixture-derived warm-model benchmark separately:

```bash
python -m tools.benchmark_embeddings
```

Black-compatible Flake8 policy is committed in `.flake8`. Lint the M1 files
with:

```bash
python -m flake8 nlp/embeddings.py tests/test_embeddings.py tools/benchmark_embeddings.py
```
