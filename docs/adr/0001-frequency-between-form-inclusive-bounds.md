# Frequency between-form (`[min_n, max_n, N]`) uses inclusive bounds

**Status:** accepted

Every other `between_parser` call site (magnitude, rate_of_change, legacy frequency)
uses `inclusive=False`. For the new frequency characteristic's `[min_n, max_n, N]`
form, we deliberately use `inclusive=True` instead, because inclusive counts read more
naturally for "how many times did this occur" style bounds than for continuous
magnitude/rate-of-change comparisons.

This is scoped to frequency only for now — magnitude and rate_of_change keep
`inclusive=False`. A follow-up task should revisit changing the shared
`between_parser` default to `inclusive=True` everywhere once the frequency
enhancement ships, to remove the inconsistency.
