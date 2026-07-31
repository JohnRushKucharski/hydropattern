# Frequency trial windows are sliding, not fixed buckets

**Status:** accepted

The frequency characteristic's `N`-based forms (`[op, n, N]`, `[min_n, max_n, N]`,
and the interannual N-in-years form) count occurrences within a trailing window of
`N` timesteps/years ending at each point in time, evaluated every step (a sliding /
moving window) — not fixed, non-overlapping buckets (e.g. steps 1-30, 31-60, ...).

This matches the existing `moving_average` implementation already used by the
legacy interannual frequency code, and avoids a boundary artifact: two closely-spaced
qualifying events that straddle an arbitrary fixed-bucket edge would otherwise never
be counted together, producing a false negative purely due to bucket alignment. The
tradeoff is that sliding windows "smear" a short cluster of events into a longer
marked-success stretch (up to `N` steps), which is accepted as consistent with
current behavior.

Fixed, non-overlapping windows were considered and rejected for this reason. If a
fixed-window need arises later, it should be expressed via one or more `timing`
characteristics combined with frequency, not a new frequency parameter.
