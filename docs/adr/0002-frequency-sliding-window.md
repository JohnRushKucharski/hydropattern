# Historical: Frequency trial windows are sliding, not fixed buckets

**Status:** superseded by [ADR 0003](0003-pattern-correctness-contract.md)

This ADR records the historical decision. Its trailing-window rule is no longer
current: frequency now evaluates forward, source-anchored windows as described
in ADR 0003. Retain the rationale below for the decision that was in effect
before the pattern-correctness work.

The frequency characteristic's `N`-based forms (`[op, n, N]`, `[min_n, max_n, N]`,
and the interannual N-in-years form) counted occurrences within a trailing window of
`N` timesteps/years ending at each point in time, evaluated every step (a sliding /
moving window) — not fixed, non-overlapping buckets (e.g. steps 1-30, 31-60, ...).

This matched the `moving_average` implementation used by the then-existing
interannual frequency code, and avoided a boundary artifact: two closely-spaced
qualifying events that straddle an arbitrary fixed-bucket edge would otherwise never
be counted together, producing a false negative purely due to bucket alignment. The
tradeoff is that sliding windows "smear" a short cluster of events into a longer
marked-success stretch (up to `N` steps), which was accepted as consistent with
the behavior at the time.

Fixed, non-overlapping windows were considered and rejected for this reason. If a
fixed-window need arises later, it should be expressed via one or more `timing`
characteristics combined with frequency, not a new frequency parameter.
