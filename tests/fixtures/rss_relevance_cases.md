# RSS relevance fixture labeling

The JSON fixture is a deterministic policy test, not a sample of measured
production precision. Each case is labeled from the title and description
alone using the Phase 0 rule that a match must contain an approved symbol,
company name, strong alias, or a context-required alias plus an approved
context term.

Labels use these categories:

- `positive`: direct symbols, company names, or approved product aliases.
- `hard_positive`: valid references that require context or less-obvious
  approved aliases.
- `ambiguous`: explicit references to two supported companies; these must be
  retained as candidates and never assigned to one ticker.
- `exclusion` and `hard_exclusion`: phrases explicitly excluded by policy,
  even if they contain a ticker-like token.
- `hard_negative`: sector, person, comparison, or bare-token text that lacks
  enough evidence for a company assignment.

The cases are reviewed as exact text. Negated relationship phrases such as
“no relationship with NVIDIA” and “without semiconductor context” are explicit
exclusions, and hyphenated comparisons such as “Nvidia-like” do not satisfy a
company-name boundary.
