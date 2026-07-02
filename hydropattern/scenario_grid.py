'''Parse and build precipitation/temperature scenario grids for response-surface plots.

Scenario columns in a timeseries encode a two-axis "scenario grid" via a naming
convention: ``_<precip_delta>_<temp_delta>`` (e.g. ``_0_1.5`` -> precip_delta=0.0,
temp_delta=1.5). See CONTEXT.md for the canonical terms.
'''
import numpy as np

from hydropattern.errors import PlotErrorCode, raise_plot_error


def parse_scenario_name(name: str) -> tuple[float, float] | None:
    '''Parse a scenario column name into (precip_delta, temp_delta).

    Expects the ``_<precip_delta>_<temp_delta>`` naming convention, e.g. ``_0_1.5``.
    Returns None if the name does not match this convention.
    '''
    parts = name.split('_')
    if len(parts) != 3 or parts[0] != '':
        return None
    try:
        return (float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def is_scenario_grid(names: list[str]) -> bool:
    '''Return True if names form a real precip/temp scenario grid.

    Requires every name to match the `_<precip_delta>_<temp_delta>` convention, and
    at least two distinct values on each axis (otherwise there's nothing to plot as a
    2D response surface).
    '''
    parsed = [parse_scenario_name(name) for name in names]
    if any(p is None for p in parsed):
        return False
    precip_deltas = {p[0] for p in parsed}
    temp_deltas = {p[1] for p in parsed}
    return len(precip_deltas) >= 2 and len(temp_deltas) >= 2


def build_grid(scenario_names: list[str],
               metric_values: dict[str, float]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Build a rectangular (precip_delta, temp_delta, metric) grid from scenario names.

    xs = sorted unique precip deltas, ys = sorted unique temp deltas.
    zs[i, j] = metric_values[scenario] for the scenario at (temp=ys[i], precip=xs[j]),
    or NaN where no scenario exists for that combo (missing grid cell).
    '''
    coords = {name: parse_scenario_name(name) for name in scenario_names}
    xs = np.array(sorted({c[0] for c in coords.values()}))
    ys = np.array(sorted({c[1] for c in coords.values()}))
    zs = np.full((len(ys), len(xs)), np.nan)
    x_index = {x: j for j, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}
    for name, (precip, temp) in coords.items():
        zs[y_index[temp], x_index[precip]] = metric_values[name]
    return xs, ys, zs


def require_scenario_grid(names: list[str]) -> None:
    '''Raise a PLOT_INVALID_SCENARIO_GRID error if names don't form a scenario grid.'''
    if not is_scenario_grid(names):
        raise_plot_error(
            PlotErrorCode.INVALID_SCENARIO_GRID,
            'Scenario names do not form a valid precip/temp scenario grid. '
            'Expected `_<precip_delta>_<temp_delta>` names with >= 2 distinct values '
            'on each axis.',
            scenario_names=names,
        )
