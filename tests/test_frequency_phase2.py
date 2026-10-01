'''Phase 2 (frequency) tests, per docs/plans/2026-10-01-pattern-correctness-tdd.md.

Supersedes the phase 0 baseline (tests/test_golden_frequency_baseline.py,
removed): the same six golden rows now pass for real (forward, event-
anchored windows), plus the plan's required edge-case oracle tests, and
coverage of the `event_bool` -> `exclusive_event_window` rename (new default
False).

Un-nested frequency contract:
  - Candidate windows are forward, anchored at a source-success timestep
    (or at every timestep for a zero-admitting predicate, e.g. `= 0`), of up
    to N timesteps, truncated at record end.
  - A window qualifies when `f(count)` is true, where count is the number of
    source-success timesteps actually inside the (possibly truncated)
    window.
  - exclusive_event_window=False (default): qualifying windows are unioned
    (OR'd) across all anchors.
  - exclusive_event_window=True: the first qualifying anchor fixes an
    N-timestep marked span; anchors inside that span are suppressed
    (skipped, not evaluated) until the span ends. A failed candidate never
    suppresses a later candidate.
'''
import numpy as np
import pandas as pd
import pytest

from hydropattern.patterns import comparison_fx, frequency_fx

# ---------------------------------------------------------------------------
# Un-nested frequency: six golden rows from the plan's table (nine cases,
# counting both exclusivity flags where the plan lists them separately).
# ---------------------------------------------------------------------------

GOLDEN_FREQUENCY_ROWS = [
    # (source diagnostic, [op, n, N], exclusive_event_window, expected)
    ([1, 0, 0, 0, 0, 0], ['>=', 1, 5], True, [1, 1, 1, 1, 1, 0]),
    ([1, 0, 0, 0, 0, 0], ['>=', 1, 5], False, [1, 1, 1, 1, 1, 0]),
    ([0, 0, 0, 0, 1, 0], ['>=', 1, 5], True, [0, 0, 0, 0, 1, 1]),
    ([0, 0, 0, 0, 1, 0], ['>=', 1, 5], False, [0, 0, 0, 0, 1, 1]),
    ([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], ['>=', 1, 5], True,
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
    ([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], ['>=', 1, 5], False,
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
    ([0, 1, 0, 0, 1, 0, 0, 0, 0, 0], ['>=', 1, 5], False,
     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
    ([0, 1, 0, 0, 1, 0, 0, 0, 0, 0], ['>=', 1, 5], True,
     [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
    ([1, 0, 1, 0, 0, 0, 0], ['>=', 2, 5], True, [1, 1, 1, 1, 1, 0, 0]),
]


def _run_frequency(source: list[int], metrics: list, exclusive_event_window: bool) -> list[float]:
    df = pd.DataFrame({'flow': source, 'dowy': np.arange(1, len(source) + 1)})
    output = np.array(source, dtype=float).reshape(-1, 1)
    op, n, big_n = metrics
    fx = frequency_fx(
        comparison_fx(op, n), order=2, big_n=big_n,
        exclusive_event_window=exclusive_event_window,
    )
    return fx(df, output).tolist()


@pytest.mark.parametrize('source,metrics,exclusive_event_window,expected', GOLDEN_FREQUENCY_ROWS)
def test_golden_frequency_row(source, metrics, exclusive_event_window, expected):
    assert _run_frequency(source, metrics, exclusive_event_window) == expected


# ---------------------------------------------------------------------------
# Default: exclusive_event_window now defaults to False (union), not True.
# ---------------------------------------------------------------------------

def test_default_exclusive_event_window_is_false():
    # Row 7/8 above show union (False) and exclusive (True) diverge for this
    # source; the no-kwarg default must match the union (False) result.
    source = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    df = pd.DataFrame({'flow': source, 'dowy': np.arange(1, len(source) + 1)})
    output = np.array(source, dtype=float).reshape(-1, 1)
    fx = frequency_fx(comparison_fx('>=', 1), order=2, big_n=5)
    assert fx(df, output).tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]


# ---------------------------------------------------------------------------
# Plan-required edge-case oracle tests (independent of the production
# algorithm: each one is hand-derived, not a re-implementation of it).
# ---------------------------------------------------------------------------

def test_zero_qualifying_operator_anchors_every_timestep():
    # '= 0 in 3': every timestep can anchor (absence is observable), not just
    # source-success timesteps. Source: one isolated success at t=2.
    # Windows are forward-only, so a position can only be marked by an
    # anchor at or before it whose window qualifies.
    source = [0, 0, 1, 0, 0, 0]
    expected = [
        0,  # t=0: window [0,2] count=1 -> fails (count != 0)
        0,  # t=1: window [1,3] count=1 -> fails
        0,  # t=2: window [2,4] count=1 -> fails
        1,  # t=3: window [3,5] count=0 -> qualifies
        1,  # t=4: window [4,5] (truncated) count=0 -> qualifies
        1,  # t=5: window [5,5] (truncated) count=0 -> qualifies
    ]
    assert _run_frequency(source, ['=', 0, 3], False) == expected


def test_failed_candidate_does_not_suppress_later_candidate_exclusive():
    # Exclusive mode: an anchor whose window fails to qualify must not start
    # a suppression span -- the very next anchor remains a live candidate.
    # n=2 in N=3; successes at t=0 (isolated, window [0,2] count=1 -> fails)
    # and t=3,4 (window [3,5] count=2 -> qualifies).
    source = [1, 0, 0, 1, 1, 0]
    expected = [0, 0, 0, 1, 1, 1]
    assert _run_frequency(source, ['>=', 2, 3], True) == expected


def test_inclusive_between_bounds():
    # [min_n, max_n, N] is inclusive on both ends; count==min_n and
    # count==max_n both qualify.
    source = [1, 1, 0, 0, 0]
    df = pd.DataFrame({'flow': source, 'dowy': np.arange(1, len(source) + 1)})
    output = np.array(source, dtype=float).reshape(-1, 1)
    f = lambda c: 1 <= c <= 2  # noqa: E731 inclusive [min_n, max_n] = [1, 2]
    fx = frequency_fx(f, order=2, big_n=5, exclusive_event_window=False)
    result = fx(df, output).tolist()
    # anchors t=0 (window [0,4], count=2, qualifies: min_n<=2<=max_n) and
    # t=1 (window [1,4], count=1, qualifies) -> union marks 0..4.
    assert result == [1, 1, 1, 1, 1]


def test_positive_n_greater_than_one_requires_multiple_successes():
    # n=2 in N=4: a single isolated success never qualifies on its own.
    source = [1, 0, 0, 0, 0, 0]
    expected = [0, 0, 0, 0, 0, 0]
    assert _run_frequency(source, ['>=', 2, 4], False) == expected


def test_window_size_one():
    # N=1: a window is just the anchor timestep itself.
    source = [0, 1, 0, 1, 0]
    expected = [0, 1, 0, 1, 0]
    assert _run_frequency(source, ['>=', 1, 1], False) == expected


def test_contiguous_successes_single_window_covers_all():
    source = [1, 1, 1, 0, 0]
    expected = [1, 1, 1, 0, 0]
    assert _run_frequency(source, ['>=', 3, 3], False) == expected


def test_end_of_record_truncation_cannot_reach_exact_count():
    # '= 3 in 5' anchored at the last two timesteps can never see a full
    # 5-timestep window (record ends), so an exact count of 3 is unreachable
    # there even though 3 total successes exist earlier in the record.
    source = [0, 1, 1, 1, 0]
    # anchor t=1: window [1,4] (truncated to 4 elements), count=3 -> qualifies
    # anchor t=2: window [2,4] (truncated to 3), count=2 -> fails
    # anchor t=3: window [3,4] (truncated to 2), count=1 -> fails
    expected = [0, 1, 1, 1, 1]
    assert _run_frequency(source, ['=', 3, 5], False) == expected


def test_suppression_boundary_next_anchor_exactly_at_span_end():
    # Exclusive mode: an anchor landing exactly on the last marked timestep
    # of an active span is still suppressed (span end is inclusive).
    source = [1, 0, 0, 0, 1, 0]
    # anchor t=0: window [0,4] (N=5), count=2 -> qualifies, span=[0,4].
    # anchor t=4 lands exactly at span_end=4 -> suppressed.
    expected = [1, 1, 1, 1, 1, 0]
    assert _run_frequency(source, ['>=', 1, 5], True) == expected

