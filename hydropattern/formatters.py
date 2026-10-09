"""Output formatting and file-writing helpers for CLI results."""

from __future__ import annotations

import re
import warnings
from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from climate_canvas.plots_utilities import plot_response_surface  # type: ignore[import-untyped]
from matplotlib.colors import Normalize

from hydropattern.errors import PlotErrorCode, raise_plot_error
from hydropattern.parsers import ClimateCanvasPlotOptions, MetricMode, MetricOptions
from hydropattern.parsing.specs import is_valid_minimum_coverage
from hydropattern.patterns import (
    Component,
    EventCountBounds,
    Result,
    identify_full_water_years,
    water_year_label,
)
from hydropattern.patterns.water_year import water_year_exposure_by_year
from hydropattern.scenario_grid import (
    build_grid,
    parse_scenario_name,
    require_scenario_grid,
)
from hydropattern.timeseries import to_day_of_water_year


def _water_year_label(date: pd.Timestamp, first_day_of_wy: int) -> int:
    '''Returns the water year label (ending-year convention) for a given date.

    WY starting Jan 1 -> label = calendar year.
    WY starting Oct 1 (doy 274): Oct 1 1970 -> WY 1971; Jan 1 1971 -> WY 1971.
    '''
    return water_year_label(date, first_day_of_wy)


def _group_by_water_year(
    result: Result, column: str, first_day_of_wy: int
) -> pd.DataFrame:
    '''Group result column by water year, returning successes, known, and total counts.

    Returns a DataFrame with index ['total', wy1, wy2, ...] and columns
    ['n', 'known', 'T'].
    '''
    df = result.df[[column]].copy()
    df['_wy'] = [_water_year_label(ts, first_day_of_wy) for ts in df.index]

    total_n = float(df[column].sum())
    total_t = float(len(df))
    rows: dict[str | int, dict[str, float]] = {
        'total': {
            'n': total_n,
            'known': float(df[column].notna().sum()),
            'T': total_t,
        }
    }
    for wy, group in df.groupby('_wy'):
        if not isinstance(wy, Integral):
            raise ValueError(f'Invalid water year label: {wy!r}.')
        rows[int(wy)] = {
            'n': float(group[column].sum()),
            'known': float(group[column].notna().sum()),
            'T': float(len(group)),
        }

    return pd.DataFrame(rows).T


def compute_portion_series(
    result: Result, column: str, first_day_of_wy: int = 1
) -> pd.Series:
    '''Compute successes divided by known outcomes for each water-year group.

    Returns a Series indexed by 'total' and water-year labels. A group with no
    known outcomes is pd.NA; known groups with no successes return 0.0.

    Water-year labels use the configured ending-year convention.
    '''
    groups = _group_by_water_year(result, column, first_day_of_wy)
    known = groups['known']
    n = groups['n']
    portion = (n / known).astype('Float64')
    return portion.mask(known <= 0, pd.NA)


def compute_metric_series(
    result: Result,
    column: str,
    mode: MetricMode = MetricMode.PORTION,
    first_day_of_wy: int = 1,
) -> pd.Series:
    '''Compute the configured summary metric for one column of a Result.

    Always starts from the underlying portion series (see compute_portion_series)
    and applies the metric mode transform. Zero-success summaries are zero;
    groups with no known outcomes are undefined.
    '''
    portion = compute_portion_series(result, column, first_day_of_wy)
    match mode:
        case MetricMode.PORTION:
            return portion
        case MetricMode.PERCENTAGE:
            return portion * 100
    raise ValueError(f'Unsupported metric mode: {mode!r}.')


def build_summary_sheet(
    scenario_results: dict[str, list[Result]],
    component_name: str,
    column: str,
    first_day_of_wy: int = 1,
    mode: MetricMode = MetricMode.PORTION,
) -> pd.DataFrame:
    '''Build one summary sheet for a single column (characteristic or component).

    Columns = scenario names (from scenario_results keys, in insertion order).
    Index   = ['total', wy1, wy2, ...] (water year labels, ending-year convention).
    Values  = configured metric (see compute_metric_series); default portion (0.0-1.0).

    Args:
        scenario_results: {scenario_name: [Result, ...]} for all scenarios.
        component_name:   name of the component whose Results to use.
        column:           characteristic or component column to compute metric for.
        first_day_of_wy:  first day of water year (1–365, default 1 = Jan 1).
        mode:             metric mode to compute (default MetricMode.PORTION).
    '''
    series: dict[str, pd.Series] = {}
    for scenario_name, results in scenario_results.items():
        result = next(r for r in results if r.component.name == component_name)
        series[scenario_name] = compute_metric_series(result, column, mode, first_day_of_wy)
    return pd.DataFrame(series)


# Public formatter API keeps this signature for CLI and library callers.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def write_results(
    scenario_results: dict[str, list[Result]],
    input_path: str,
    output_directory: str | None,
    write_to_excel: bool,
    overwrite: bool = True,
    first_day_of_wy: int = 1,
    metric_mode: MetricMode = MetricMode.PORTION,
) -> Path:
    """Write per-scenario results to csv files or a single Excel file,
    and always write a per-component summary Excel file.

    Each key in scenario_results is a scenario name (timeseries column header).
    Each value is the list of within-scenario component Results for that scenario.

    Args:
        overwrite:        When True (default), existing files are replaced.
            When False, a numeric suffix (__1, __2, …) is appended.
        first_day_of_wy: First day of water year (1–365). Used for summary WY grouping.
        metric_mode:     Summary metric mode (portion/percentage).

    Returns the directory (or file parent) that received output files.
    """
    output_path = _resolve_output_path(input_path, output_directory)
    if write_to_excel:
        _write_results_excel(scenario_results, output_path, input_path, overwrite)
    else:
        _write_results_csv(scenario_results, output_path, overwrite)

    write_summary(scenario_results, output_path, first_day_of_wy, overwrite, metric_mode)
    return output_path


def write_summary(
    scenario_results: dict[str, list[Result]],
    output_path: Path,
    first_day_of_wy: int = 1,
    overwrite: bool = True,
    metric_mode: MetricMode = MetricMode.PORTION,
) -> None:
    """Write one {component}_summary.xlsx per component to output_path.

    Each file has one sheet per characteristic + one sheet for the component itself.
    Columns = scenario names; index = ['total', wy1, wy2, ...].
    Always written as Excel regardless of raw-data format.
    """
    first_scenario_results = next(iter(scenario_results.values()))
    for result in first_scenario_results:
        _write_component_summary(
            scenario_results=scenario_results,
            output_path=output_path,
            component_result=result,
            first_day_of_wy=first_day_of_wy,
            overwrite=overwrite,
            metric_mode=metric_mode,
        )



def _build_all_filenames(
    scenario_results: dict[str, list[Result]],
) -> dict[tuple[str, str], str]:
    """Return {(scenario_name, component_name): base_filename} for all outputs.

    Numeric suffix (_{i}) is appended only when two pairs produce the same
    cleaned base name, keeping filenames readable in the common case.
    """
    pairs: list[tuple[str, str]] = [
        (scenario_name, result.component.name)
        for scenario_name, results in scenario_results.items()
        for result in results
    ]
    clean_bases = [
        f"{_clean_variable_name(s)}_{_clean_variable_name(c)}" for s, c in pairs
    ]
    counts = Counter(clean_bases)
    occurrence: dict[str, int] = {}
    filename_map: dict[tuple[str, str], str] = {}
    for pair, clean in zip(pairs, clean_bases):
        if counts[clean] > 1:
            idx = occurrence.get(clean, 0)
            occurrence[clean] = idx + 1
            filename_map[pair] = f"{clean}_{idx}"
        else:
            filename_map[pair] = clean
    return filename_map


def _write_results_excel(scenario_results: dict[str, list[Result]],
                         output_path: Path,
                         input_path: str,
                         overwrite: bool) -> None:
    '''Write raw scenario outputs as one workbook with one sheet per scenario/component pair.'''
    output_filename = Path(input_path).stem + "_output.xlsx"
    target = output_path / output_filename
    if not overwrite:
        target = _next_available_path(target)
    with pd.ExcelWriter(target) as writer:
        for scenario_name, results in scenario_results.items():
            for result in results:
                sheet = _build_sheet_name(scenario_name, result.component.name)
                result.df.to_excel(writer, sheet_name=sheet)


def _write_results_csv(scenario_results: dict[str, list[Result]],
                       output_path: Path,
                       overwrite: bool) -> None:
    '''Write raw scenario outputs as per-scenario/component csv files.'''
    filename_map = _build_all_filenames(scenario_results)
    for (scenario_name, component_name), base_name in filename_map.items():
        csv_path = output_path / (base_name + ".csv")
        if not overwrite:
            csv_path = _next_available_path(csv_path)
        results = scenario_results[scenario_name]
        result = next(r for r in results if r.component.name == component_name)
        result.df.to_csv(csv_path)


def _write_component_summary(scenario_results: dict[str, list[Result]],
                             output_path: Path,
                             component_result: Result,
                             first_day_of_wy: int,
                             overwrite: bool,
                             metric_mode: MetricMode) -> None:
    '''Write one {component}_summary.xlsx workbook for a single component.'''
    component = component_result.component
    filename = _clean_variable_name(component.name) + '_summary.xlsx'
    target = output_path / filename
    if not overwrite:
        target = _next_available_path(target)
    with pd.ExcelWriter(target) as writer:
        for char in component.characteristics:
            sheet_name = _clean_variable_name(char.name)[:31]
            sheet_df = build_summary_sheet(
                scenario_results, component.name, char.name, first_day_of_wy, metric_mode
            )
            sheet_df.to_excel(writer, sheet_name=sheet_name)
        comp_sheet = _clean_variable_name(component.name)[:31]
        comp_df = build_summary_sheet(
            scenario_results, component.name, component.name, first_day_of_wy, metric_mode
        )
        comp_df.to_excel(writer, sheet_name=comp_sheet)
        used_sheet_names = {
            _clean_variable_name(char.name)[:31].casefold()
            for char in component.characteristics
        }
        used_sheet_names.add(comp_sheet.casefold())
        details_sheet = _unique_sheet_name('reporting_details', used_sheet_names)
        details = _build_reporting_details_sheet(
            scenario_results, component.name, first_day_of_wy
        )
        details.to_excel(writer, sheet_name=details_sheet, index=False)


def _build_reporting_details_sheet(
    scenario_results: dict[str, list[Result]],
    component_name: str,
    first_day_of_wy: int,
) -> pd.DataFrame:
    '''Build per-scenario, per-outcome reporting counts and component event details.'''
    rows: list[dict[str, object]] = []
    for scenario_name, results in scenario_results.items():
        result = next(r for r in results if r.component.name == component_name)
        rows.extend(_reporting_rows_for_scenario(
            scenario_name, result, component_name, first_day_of_wy
        ))
    return pd.DataFrame(rows)


def _reporting_rows_for_scenario(
    scenario_name: str,
    result: Result,
    component_name: str,
    first_day_of_wy: int,
) -> list[dict[str, object]]:
    attributed_result = _result_with_water_year_days(result, first_day_of_wy)
    intervals, completeness_reason = _reporting_intervals(
        attributed_result, first_day_of_wy
    )
    count_by_year, count_error = _event_counts_by_water_year(attributed_result)
    exposure_by_year, exposure_error = _observed_exposure_by_water_year(
        result, first_day_of_wy
    )
    total_exposure = sum(exposure_by_year.values()) if exposure_by_year else None
    total_count = result.event_count_bounds()
    outcome_columns = [
        *(char.name for char in result.component.characteristics),
        component_name,
    ]
    rows = []
    for interval, group, interval_status in intervals:
        for outcome_column in outcome_columns:
            outcomes = group[outcome_column]
            known = int(outcomes.notna().sum())
            row = {
                'scenario': scenario_name,
                'outcome_column': outcome_column,
                'interval': interval,
                'successful_timesteps': int(outcomes.sum()),
                'known_timesteps': known,
                'total_timesteps': len(outcomes),
                'known_outcome_coverage': known / len(outcomes) if len(outcomes) else np.nan,
                'completeness': interval_status,
                'completeness_reason': completeness_reason,
            }
            if outcome_column == component_name:
                row.update(_event_detail_values(
                    interval, total_count, total_exposure,
                    count_by_year, count_error, exposure_by_year, exposure_error,
                ))
            else:
                row.update(_empty_event_detail_values())
            rows.append(row)
    return rows


def _event_counts_by_water_year(
    result: Result,
) -> tuple[dict[int, EventCountBounds], str | None]:
    try:
        return result.event_count_bounds_by_water_year(), None
    except (KeyError, ValueError) as error:
        if 'conflicts' in str(error):
            raise
        return {}, str(error)


def _result_with_water_year_days(
    result: Result,
    first_day_of_wy: int,
) -> Result:
    if not isinstance(result.df.index, pd.DatetimeIndex):
        return result
    if (
        result.first_day_of_water_year is not None
        and result.first_day_of_water_year != first_day_of_wy
    ):
        raise ValueError(
            'first_day_of_water_year conflicts with Result boundary metadata.'
        )
    if 'dowy' in result.df.columns:
        return result
    df = result.df.copy()
    df['dowy'] = [
        to_day_of_water_year(timestamp, first_day_of_wy)
        for timestamp in df.index
    ]
    return Result(
        df=df,
        component=result.component,
        dv_name=result.dv_name,
        first_day_of_water_year=first_day_of_wy,
    )


def _observed_exposure_by_water_year(
    result: Result,
    first_day_of_wy: int,
) -> tuple[dict[int, float], str | None]:
    if not isinstance(result.df.index, pd.DatetimeIndex):
        return {}, 'DatetimeIndex is required to determine observed water-year exposure.'
    try:
        return water_year_exposure_by_year(result.df.index, first_day_of_wy), None
    except ValueError as error:
        return {}, str(error)


def _event_detail_values(
    interval: str | int,
    total_count: EventCountBounds,
    total_exposure: float | None,
    count_by_year: dict[int, EventCountBounds],
    count_error: str | None,
    exposure_by_year: dict[int, float],
    exposure_error: str | None,
) -> dict[str, object]:
    if interval == 'total':
        count_bounds: EventCountBounds | None = total_count
        exposure = total_exposure
    else:
        year = int(interval)
        count_bounds = count_by_year.get(year)
        exposure = exposure_by_year.get(year)

    reasons = []
    if count_bounds is None:
        reasons.append(
            'Event-count bounds unavailable: '
            f'{count_error or "water-year attribution unavailable."}'
        )
    if exposure is None or exposure <= 0:
        reasons.append(
            'Observed exposure unavailable: '
            f'{exposure_error or "no positive exposure for interval."}'
        )
    rate_lower = None
    rate_upper = None
    rate_available = (
        count_bounds is not None and exposure is not None and exposure > 0
    )
    if rate_available:
        assert count_bounds is not None and exposure is not None
        rate_lower = count_bounds.lower / exposure
        rate_upper = count_bounds.upper / exposure
    if count_bounds is None:
        availability_status = 'event_count_unavailable'
    elif not rate_available:
        availability_status = 'event_rate_unavailable'
    else:
        availability_status = 'available'
    return {
        'event_count_lower': count_bounds.lower if count_bounds is not None else None,
        'event_count_upper': count_bounds.upper if count_bounds is not None else None,
        'observed_exposure_water_years': exposure,
        'event_rate_lower': rate_lower,
        'event_rate_upper': rate_upper,
        'availability_status': availability_status,
        'availability_reason': '; '.join(reasons) or None,
    }


def _empty_event_detail_values() -> dict[str, object]:
    return {
        'event_count_lower': None,
        'event_count_upper': None,
        'observed_exposure_water_years': None,
        'event_rate_lower': None,
        'event_rate_upper': None,
        'availability_status': 'not_applicable',
        'availability_reason': None,
    }


def _reporting_intervals(
    result: Result,
    first_day_of_wy: int,
) -> tuple[list[tuple[str | int, pd.DataFrame, str]], str | None]:
    intervals: list[tuple[str | int, pd.DataFrame, str]] = [
        ('total', result.df, 'whole_record')
    ]
    if not isinstance(result.df.index, pd.DatetimeIndex) or result.df.empty:
        return intervals, None

    years = np.array([
        water_year_label(timestamp, first_day_of_wy)
        for timestamp in result.df.index
    ])
    annual_groups = [
        (int(year), result.df.iloc[years == year])
        for year in dict.fromkeys(years)
    ]
    try:
        full_years = identify_full_water_years(
            result.df['dowy'].to_numpy(),
            result.df.index,
            first_day_of_wy,
        )
    except ValueError as error:
        if not any(
            reason in str(error)
            for reason in ('cadence', 'data gap', 'timestamps must be valid')
        ):
            raise
        reason = str(error)
        intervals.extend((year, group, 'undetermined') for year, group in annual_groups)
        return intervals, reason

    complete_labels = {
        water_year_label(result.df.index[end], first_day_of_wy)
        for _, end in full_years
    }
    intervals.extend(
        (year, group, 'complete' if year in complete_labels else 'partial')
        for year, group in annual_groups
    )
    return intervals, None


def _unique_sheet_name(name: str, used_names: set[str]) -> str:
    candidate = name[:31]
    suffix = 1
    while candidate.casefold() in used_names:
        ending = f'_{suffix}'
        candidate = f'{name[:31 - len(ending)]}{ending}'
        suffix += 1
    return candidate


def _resolve_output_path(
    input_path: str,
    output_directory: str | None,
) -> Path:
    if output_directory:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    input_parent = Path(input_path).parent
    output_path = input_parent / (Path(input_path).stem + "_output")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _build_sheet_name(scenario_name: str, component_name: str) -> str:
    # Excel sheet names are limited to 31 characters.
    raw = f"{_clean_variable_name(scenario_name)}_{_clean_variable_name(component_name)}"
    return raw[:31]


def _clean_variable_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "result"


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = 1
    while True:
        candidate = path.with_name(f"{path.stem}__{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def write_grid_csv(xs, ys, zs, path: Path) -> None:
    '''Write a (precip_delta x temp_delta) grid to csv: rows=temp deltas, columns=precip deltas.'''
    pd.DataFrame(zs, index=ys, columns=xs).to_csv(path, index_label='temp_delta\\precip_delta')


# Signature mirrors plotting options surface.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def plot_component_response_surface(
        scenario_results: dict[str, list[Result]],
        component: Component,
        metric_options: MetricOptions,
        first_day_of_wy: int,
        climate_canvas: ClimateCanvasPlotOptions = ClimateCanvasPlotOptions(),
        output_path: Path | None = None,
        minimum_coverage: float = 0.9) -> None:
    '''Build and plot one component's response-surface grid.

    Requires scenario names to form a valid precip/temp scenario grid (see
    hydropattern.scenario_grid). Raises HydropatternError otherwise.

    minimum_coverage: required known-outcome fraction for a scenario to appear.
    output_path: directory for the eligible summary grid, coverage CSV, and PNG.
    None (default) skips file writes and shows the plot interactively instead
    (forces show=True regardless of climate_canvas.show). When output_path is
    given, show follows climate_canvas.show as usual.

    title defaults to the component name; zlabel names the selected summary mode.
    Both defaults apply when climate_canvas.title/zlabel are unset. The title
    also reports the cutoff and number of withheld scenarios.
    '''
    scenario_names = list(scenario_results.keys())
    require_scenario_grid(scenario_names)
    if not is_valid_minimum_coverage(minimum_coverage):
        raise ValueError('minimum_coverage must be a finite number in [0, 1].')
    raw_values, coverage_rows = _plot_metric_coverage(
        scenario_results, component.name, scenario_names, first_day_of_wy,
        metric_options.mode, minimum_coverage,
    )
    metric_values = {
        name: raw_values[name] if coverage_rows[name]['eligible'] else np.nan
        for name in scenario_names
    }
    xs, ys, zs = build_grid(scenario_names, metric_values)
    withheld = [row for row in coverage_rows.values() if not row['eligible']]
    title = component.name if climate_canvas.title is None else climate_canvas.title
    title = (
        f'{title}\nMinimum coverage: {minimum_coverage:.0%}; '
        f'withheld: {len(withheld)} scenario(s)'
    )
    default_zlabel = (
        'Portion of known outcomes' if metric_options.mode is MetricMode.PORTION
        else 'Percentage of known outcomes (%)'
    )
    zlabel = default_zlabel if climate_canvas.zlabel is None else climate_canvas.zlabel
    color_map = climate_canvas.color_map
    if output_path is not None:
        write_grid_csv(xs, ys, zs, output_path / f'{component.name}_grid.csv')
        pd.DataFrame(coverage_rows.values()).to_csv(
            output_path / f'{component.name}_grid_coverage.csv', index=False
        )
        save_path = output_path / f'{component.name}_plot.png'
        show = climate_canvas.show
    else:
        save_path = None
        show = True
    if withheld:
        details = ', '.join(
            f"{row['scenario']} ({row['exclusion_reason']})" for row in withheld
        )
        warnings.warn(
            f'Response surface for {component.name!r} excluded scenarios at '
            f'{minimum_coverage:.0%} minimum coverage: {details}.',
            RuntimeWarning,
            stacklevel=2,
        )
        if climate_canvas.fillin:
            raise_plot_error(
                PlotErrorCode.FILLIN_WITHHELD_SCENARIOS,
                'fillin cannot be enabled when scenarios are withheld from the '
                'response surface; set fillin = false to preserve coverage gaps.',
                component=component.name,
                scenarios=[row['scenario'] for row in withheld],
            )
    if not _has_renderable_surface(scenario_names, metric_values):
        raise_plot_error(
            PlotErrorCode.NO_RENDERABLE_SURFACE,
            f'No renderable response surface remains for component {component.name!r}; '
            'at least three non-collinear scenarios with defined metrics are required.',
            component=component.name,
            eligible_scenarios=int(np.isfinite(zs).sum()),
        )
    norm, levels, widths = _degenerate_range_norm(zs)
    plot_response_surface(
        xs, ys, zs, interpolate=climate_canvas.interpolate,
        labels=(climate_canvas.xlabel, climate_canvas.ylabel, zlabel),
        title=title,
        save_path=save_path,
        show=show,
        threshold=climate_canvas.threshold,
        color_map=color_map,
        color_map_ticks=climate_canvas.color_map_ticks,
        fillin=climate_canvas.fillin,
        norm=norm,
        levels=levels,
        widths=widths,
    )


def _plot_metric_coverage(
    scenario_results: dict[str, list[Result]],
    component_name: str,
    scenario_names: list[str],
    first_day_of_wy: int,
    mode: MetricMode,
    minimum_coverage: float,
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    '''Return raw whole-record metrics and per-scenario coverage eligibility.'''
    summary = build_summary_sheet(
        scenario_results, component_name, component_name, first_day_of_wy, mode
    )
    metrics: dict[str, float] = {}
    rows: dict[str, dict[str, object]] = {}
    for name in scenario_names:
        result = next(r for r in scenario_results[name] if r.component.name == component_name)
        outcomes = result.df[component_name]
        total_count = len(outcomes)
        known_count = int(outcomes.notna().sum())
        coverage = known_count / total_count if total_count else 0.0
        raw_value = summary.at['total', name]
        metric = float(raw_value) if isinstance(raw_value, Real) else np.nan
        metrics[name] = metric
        coordinates = parse_scenario_name(name)
        assert coordinates is not None
        precip_delta, temp_delta = coordinates
        reason: str | None = None
        if total_count == 0:
            reason = 'no_recorded_timesteps'
        elif known_count == 0 or not np.isfinite(metric):
            reason = 'no_known_outcomes'
        elif coverage < minimum_coverage:
            reason = 'below_minimum_coverage'
        rows[name] = {
            'scenario': name,
            'precip_delta': precip_delta,
            'temp_delta': temp_delta,
            'raw_summary': metric,
            'known_count': known_count,
            'total_count': total_count,
            'coverage': coverage,
            'minimum_coverage': minimum_coverage,
            'eligible': reason is None,
            'exclusion_reason': reason,
        }
    return metrics, rows


def _has_renderable_surface(
    scenario_names: list[str], metric_values: dict[str, float]
) -> bool:
    '''Require three non-collinear known points to form a meaningful 2D region.'''
    points = [
        parse_scenario_name(name)
        for name in scenario_names
        if np.isfinite(metric_values[name])
    ]
    if len(points) < 3:
        return False
    coordinates = np.asarray(points, dtype=float)
    return bool(np.linalg.matrix_rank(coordinates[1:] - coordinates[0]) == 2)


def _degenerate_range_norm(
        zs: np.ndarray
) -> tuple[Normalize, tuple[float, ...], tuple[float, ...]] | tuple[None, None, None]:
    '''Escape hatch for an all-equal (or single-value) zs grid, e.g. a pattern that
    never/always succeeds across every scenario.

    climate_canvas builds TwoSlopeNorm(vmin=z_min, vcenter=threshold, vmax=z_max)
    internally, which raises ValueError when z_min == z_max (degenerate range,
    vmin == vmax). Returns a plain Normalize + single-level contour instead, which
    climate_canvas.plot_response_surface accepts as caller-supplied norm/levels/widths
    (used as-is, bypassing its own TwoSlopeNorm computation). Returns (None, None, None)
    when the range isn't degenerate, so the default TwoSlopeNorm behavior is unchanged.
    '''
    z_min, z_max = float(np.nanmin(zs)), float(np.nanmax(zs))
    if z_min != z_max:
        return None, None, None
    pad = max(abs(z_min) * 1e-3, 1e-3)
    return Normalize(vmin=z_min - pad, vmax=z_max + pad), (z_min,), (1.0,)


# Signature mirrors plotting options surface.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def plot_components(scenario_results: dict[str, list[Result]],
                    output_path: Path, metric_options: MetricOptions,
                    first_day_of_wy: int,
                    climate_canvas: ClimateCanvasPlotOptions = ClimateCanvasPlotOptions(),
                    minimum_coverage: float = 0.9) -> None:
    '''Save one response-surface grid csv + plot png per component to output_path.

    Requires scenario names to form a valid precip/temp scenario grid (see
    hydropattern.scenario_grid). Raises HydropatternError otherwise.
    '''
    first_scenario_results = next(iter(scenario_results.values()))
    for result in first_scenario_results:
        plot_component_response_surface(
            scenario_results, result.component, metric_options, first_day_of_wy,
            climate_canvas, output_path, minimum_coverage,
        )
