# Frequency characteristic

Un-nested frequency assesses how many qualifying timesteps occur in a
forward window. Nested frequency first evaluates an intra-annual pattern
within each complete water year, then an interannual pattern across
qualifying water years. A qualifying window can mark observations where the
preceding conditions are not met. Frequency counts qualifying timesteps or
water years, not component events.

## Un-nested frequency

An un-nested frequency pattern counts qualifying **timesteps** in forward
windows across the record. Its window length is a timestep count: for daily
data, `N = 5` means five daily observations; for monthly data, it means five
monthly observations. Windows may cross water-year boundaries.

```toml
frequency = [">=", 1, 5]
# Alternatives:
# frequency = [1, 3, 5]
# frequency = [">=", 1, 5, true]
```

| Parameter | Type | Valid values | Meaning |
|---|---|---|---|
| `operator` | string | `<`, `<=`, `>`, `>=`, `=`, `!=` | Comparison with a qualifying-timestep count. |
| `count` | integer | 0 through `window_length` | Count threshold. |
| `minimum_count`, `maximum_count` | integers | Inclusive count bounds within 0 through `window_length`; minimum less than maximum | Acceptable count range. |
| `window_length` | integer | 1 or greater | Maximum number of timesteps in a forward window. The record can shorten a window at its end. |
| `exclusive_windows` | boolean | `false` by default | When `true`, a qualifying window suppresses later anchors within its span. |

Count bounds include their endpoints. The predicate must be attainable within
the window: for example, `< 0` and `> window_length` cannot be satisfied and
are rejected, while a condition that accepts zero is valid.

Without `exclusive_windows`, every eligible anchor is assessed and successful
overlapping windows are combined. With exclusive windows, a qualifying
window claims its span. A window that does not meet its count condition does
not claim or suppress later anchors. The default `false` setting therefore
combines overlap; `true` prevents successful windows from overlapping.

An anchor is a timestep where all preceding characteristic conditions are
met. If the count condition can accept zero (for example `= 0`, `< 1`, or
`!= 1`), every timestep can anchor a window; this allows a condition to
assess absence. Otherwise, only qualifying timesteps anchor windows. At the
record end, a shortened window is assessed using the observations available.

### Overlapping and exclusive windows

The examples use qualifying timesteps `[0,1,0,0,1,0,0,0,0,0]`, a
five-timestep window, and a requirement for at least one qualifying
timestep. Frequency is terminal in these components, so the final component
outcome matches the frequency diagnostic:

| Window rule | Frequency diagnostic | Component outcome |
|---|---|---|
| Overlapping windows (`exclusive_windows = false`) | `[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]` | `[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]` |
| Exclusive windows (`exclusive_windows = true`) | `[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]` | `[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]` |

Expected diagnostic: `[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]`

Expected diagnostic: `[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]`

Expected component outcome: `[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]`

Expected component outcome: `[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]`

With overlapping windows, the second qualifying timestep at position 5
starts another window and extends the marked span. With exclusive windows,
the first qualifying window claims positions 2–6, suppressing the anchor at
position 5. An anchor at the final timestep of a claimed span is suppressed;
the next timestep can anchor another window.

### Zero-accepting and record-end examples

For qualifying timesteps `[0,0,0,1,0]`, the condition `= 0` with a
three-timestep window can anchor at every timestep. The first window has no
qualifying timesteps, while the final one-timestep window is also assessed:

Expected diagnostic: `[1, 1, 1, 0, 1]`

Expected component outcome: `[1, 1, 1, 0, 1]`

For `[0,0,0,0,1]` with `>= 1` in a five-timestep window, the last qualifying
timestep anchors a shortened window containing only itself:

Expected diagnostic: `[0, 0, 0, 0, 1]`

Expected component outcome: `[0, 0, 0, 0, 1]`

## Unknown qualifying outcomes

An unknown preceding outcome may be either a qualifying timestep or a
non-qualifying timestep. Frequency assesses every count allowed by those
possibilities. For a definite anchor, a window succeeds only when its count
condition passes for every possible count; it fails when none pass; otherwise
its verdict is unknown. If the anchor itself is unknown, a passing count is
only a possible successful window. An unknown timestep is a possible anchor
unless the condition accepts zero, in which case every timestep is an anchor
as usual.

With overlapping windows, definite successful coverage takes priority over
uncertain coverage from another window. For a three-timestep window and
`frequency = [">=", 1, 3]`:

| Preceding outcomes | Frequency diagnostic |
|---|---|
| `[unknown, 0, 0, 0]` | `[unknown, unknown, unknown, 0]` |
| `[unknown, 1, 0, 0]` | `[unknown, 1, 1, 1]` |

In the first case, the only possible qualifying anchor is the unknown first
timestep. If it qualifies, its window succeeds; otherwise no window succeeds.
In the second, the known qualifying timestep at position 2 always starts a
successful window covering positions 2–4, so it settles those outcomes even
though the overlapping first window is uncertain.

Exclusive windows preserve uncertainty about which anchors are suppressed.
For `[unknown, 1, 0, 0]` with the same condition, if the first timestep
qualifies, its window claims positions 1–3; otherwise the known second
timestep starts a window claiming positions 2–4. Both schedules agree on the
middle outcomes:

| Possible schedule | Frequency diagnostic |
|---|---|
| First timestep qualifies | `[1, 1, 1, 0]` |
| First timestep does not qualify | `[0, 1, 1, 1]` |
| Combined diagnostic | `[unknown, 1, 1, unknown]` |

Unknown counts can still produce definite results. In a three-timestep
window with two known qualifying timesteps and one unknown, `>= 2` succeeds
for every possible count, `>= 3` is unknown, and `>= 4` fails. A shortened
window at the record end uses only the available timesteps in the same
possible-count assessment.

Windows share the same unknown trials; their possible successes are not
independent. For `[1, unknown]`, `= 1`, and a two-timestep window, both
overlapping and exclusive rules produce `[unknown, 1]`. If the unknown is
zero, the first window covers both timesteps; if it is one, the shortened
final window covers the second timestep instead. That shared coverage is
definite even though neither candidate window has a definite verdict.

## Nested frequency: intra-annual and interannual patterns

Nested frequency first assesses a pattern within each complete water year
(the **intra-annual pattern**), then assesses qualifying water years in
forward windows (the **interannual pattern**). Intra-annual count windows
use timesteps; interannual count windows use water years. Each count pattern
can set `exclusive_windows` independently.

An intra-annual fraction condition compares the fraction of qualifying
timesteps among **all observed timesteps** in each complete water year with
a threshold. It is evaluated once for that water year, then the verdict is
repeated across its timesteps. This annual fraction form is not a frequency
window, so its `exclusive_windows` setting has no effect. An intra-annual
count pattern instead evaluates timestep windows within each complete water
year. The interannual pattern counts water years that meet the intra-annual
condition and broadcasts each successful window across the water years it
spans. A shortened interannual window at the end uses the complete water
years available; a partial water year does not count as a qualifying water
year.

For nested count patterns, each level can specify its own window length and
`exclusive_windows` setting:

```toml
[components.nested_counts]

[[components.nested_counts.characteristics]]
type = "magnitude"
parameters = [">", 0]

[[components.nested_counts.characteristics]]
type = "frequency"
parameters = [[">=", 1, 5, true], [">=", 1, 2, false]]
```

Here is a valid ordered configuration with a fraction condition followed by
an interannual count window:

```toml
[components.seasonal_pulse]

[[components.seasonal_pulse.characteristics]]
type = "magnitude"
parameters = [">=", 1]

[[components.seasonal_pulse.characteristics]]
type = "frequency"
parameters = [[">=", 0.5], [">=", 1, 2]]
```

For a fully known, three-year monthly illustration, the magnitude condition
qualifies 9 of 12 timesteps in the first year, none in the second, and 6 in
the third. The intra-annual condition is `>= 0.5`; the interannual condition
is `>= 1` qualifying water year in a forward two-water-year window.

| Water year | Qualifying timesteps | Fraction | Intra-annual outcome | Interannual and component outcome |
|---|---:|---:|---:|---:|
| 2018 | 9 of 12 | 0.75 | 1 | 1 |
| 2019 | 0 of 12 | 0.00 | 0 | 1 |
| 2020 | 6 of 12 | 0.50 | 1 | 1 |

Expected intra-annual diagnostic: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`

Expected interannual diagnostic: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`

Expected component outcome: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`

The first interannual window contains one qualifying water year and marks the
first two years. The third year qualifies under the intra-annual condition
and is marked by its own shortened final window. These examples use fully
known outcomes.

### Unknown annual outcomes

Unknown trials remain in the annual fraction denominator. If a complete
12-month water year contains one qualifying month, one failed month, and ten
unknown months, its possible fractions are `1/12`, `2/12`, through `11/12`,
not `1/2`. The annual verdict is known only when the condition agrees for
every attainable fraction, including equality and inequality comparisons.

For an intra-annual condition of `>= 0.5`:

| Monthly qualifying outcomes | Possible fractions | Intra-annual verdict |
|---|---|---|
| One success, one failure, ten unknowns | `1/12` through `11/12` | unknown |
| Seven successes, five unknowns | `7/12` through `12/12` | 1 |
| Seven failures, five unknowns | `0/12` through `5/12` | 0 |
| Twelve unknowns | `0/12` through `12/12` | unknown |

The verdict is broadcast across every timestep of that complete water year.
These fractions assess the annual **condition**; they are not summary
fractions calculated over known outcomes.

Intra-annual count and between patterns use the same unknown-aware timestep
windows as un-nested frequency, resetting at water-year boundaries. For
interannual counting, a year qualifies if any intra-annual diagnostic is
definitely successful. It fails only if every diagnostic is definitely zero;
zeros mixed with unknowns yield an unknown annual verdict.

An unknown annual verdict remains a possible qualifying anchor; it is not
converted to failure. Interannual windows preserve shared-trial correlations
and possible exclusive schedules just like timestep windows. For annual
verdicts `[unknown, 1, 0, 0]` with `>= 1` in three-water-year windows:

| Window rule | Interannual verdicts |
|---|---|
| Overlapping | `[unknown, 1, 1, 1]` |
| Exclusive | `[unknown, 1, 1, unknown]` |

Each interannual verdict broadcasts over its entire complete water year.
Partial years remain unknown and are excluded from the interannual trial
sequence; they neither anchor nor occupy a trial in a window. This exclusion
does not affect un-nested frequency, which can cross water-year boundaries.
