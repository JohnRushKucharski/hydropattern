'''Phase 0 baseline: captures the plan's golden frequency/nested-frequency
arrays from docs/plans/2026-10-01-pattern-correctness-tdd.md against the
*current* implementation.

These are intentionally expected to fail (xfail, strict=True) right now:
frequency_fx currently computes a *trailing* sliding-window count (NaN for
the first big_n - 1 timesteps, no event-anchored forward window), not the
forward/event-anchored window described in the plan. Phase 2 implements the
real behavior; when that lands, these xfail markers must be removed (strict
mode turns an unexpected pass into a failure, forcing that cleanup).

Do not add new behavior here -- this file only documents the gap.
'''
import numpy as np
import pandas as pd
import pytest

from hydropattern.patterns import comparison_fx, frequency_fx

# ---------------------------------------------------------------------------
# Un-nested frequency: six golden rows from the plan's table.
# ---------------------------------------------------------------------------

GOLDEN_FREQUENCY_ROWS = [
    # (source diagnostic, [op, n, N], event_bool, expected)
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


def _run_frequency(source: list[int], metrics: list, event_bool: bool) -> list[float]:
    df = pd.DataFrame({'flow': source, 'dowy': np.arange(1, len(source) + 1)})
    output = np.array(source, dtype=float).reshape(-1, 1)
    op, n, big_n = metrics
    fx = frequency_fx(comparison_fx(op, n), order=2, big_n=big_n, event_bool=event_bool)
    return fx(df, output).tolist()


@pytest.mark.xfail(
    strict=True,
    reason='frequency_fx uses a trailing sliding window (NaN warm-up), not the '
           'forward event-anchored window from the plan; phase 2 implements this.',
)
@pytest.mark.parametrize('source,metrics,event_bool,expected', GOLDEN_FREQUENCY_ROWS)
def test_golden_frequency_row_baseline(source, metrics, event_bool, expected):
    assert _run_frequency(source, metrics, event_bool) == expected
