# Reporting and unknown outcomes

**Status:** accepted design; implementation pending.

This decision supersedes ADR 0003's known-trial annual probability denominator,
complete-year-only event-rate exposure, strict rejection of omitted leap days,
and reciprocal reporting guidance; other pattern-correctness decisions remain.
Unavailable calculations retain unknown outcomes, with three-valued conditions
settled only when available information determines the answer. Annual frequency
conditions assess possible fractions over all observed trials in a complete
water year, whereas descriptive summary fractions use known outcomes and disclose
coverage: these answer different questions. Daily records accept omitted
February 29 without distinguishing synthetic omission from missing real data;
exposure uses 366 only when that water year's recorded rows contain February 29,
otherwise 365, including partial years. Monthly exposure is observations/12.

Event rates use whole-record exposure and matched component event counts,
attributed annually to their first successful observed timestep. The MVP reports
potentially over-wide bounds from final unknown outcomes rather than preserving
their dependencies; this limits scope but can include impossible combinations.
An optional final TDD stage will tighten those bounds. Scalar event APIs reject
ambiguous counts/rates instead of choosing an endpoint. Remove `return_period`:
reciprocal portion adds no information and invites unsupported recurrence claims.
Response surfaces retain portion/percentage, default to at least 90% known
coverage, preserve withheld cells as gaps, and reject conflicting global filling.
Default red indicates lower final success/non-failure fraction for both pattern
types. The [TDD plan](../plans/2026-10-06-reporting-metrics-tdd.md) records tests,
trade-offs, migration requirements, and mandatory plain-language user examples.
