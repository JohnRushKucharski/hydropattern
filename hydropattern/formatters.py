"""Output formatting and file-writing helpers for CLI results."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from hydropattern.parsers import MetricMode
from hydropattern.patterns import Result


def _water_year_label(date: pd.Timestamp, first_day_of_wy: int) -> int:
    '''Returns the water year label (ending-year convention) for a given date.

    WY starting Jan 1 -> label = calendar year.
    WY starting Oct 1 (doy 274): Oct 1 1970 -> WY 1971; Jan 1 1971 -> WY 1971.
    '''
    if first_day_of_wy == 1:
        return date.year
    doy = date.dayofyear - 1 if date.is_leap_year and date.dayofyear > 59 else date.dayofyear
    return date.year + 1 if doy >= first_day_of_wy else date.year


def _group_by_water_year(
    result: Result, column: str, first_day_of_wy: int
) -> pd.DataFrame:
    '''Group result column by water year, returning n (successes) and T (count).

    Returns a DataFrame with index ['total', wy1, wy2, ...] and columns ['n', 'T'].
    All values are floats; zero-success years have n=0.0 (not NA).
    '''
    df = result.df[[column]].copy()
    df['_wy'] = [_water_year_label(ts, first_day_of_wy) for ts in df.index]

    total_n = float(df[column].sum())
    total_t = float(len(df))
    rows: dict[str | int, dict[str, float]] = {
        'total': {'n': total_n, 'T': total_t}
    }
    for wy, group in df.groupby('_wy'):
        rows[int(wy)] = {'n': float(group[column].sum()), 'T': float(len(group))}

    return pd.DataFrame(rows).T


def compute_portion_series(
    result: Result, column: str, first_day_of_wy: int = 1
) -> pd.Series:
    '''Compute portion (n/T) for one column of a Result, broken down by water year.

    Returns a Series with index = ['total', wy1, wy2, ...] and float values
    representing the fraction of time steps where the column value is 1.

    Zero successes -> 0.0. NA (T=0 for a group) -> pd.NA.
    WY label uses ending-year convention (US standard).
    '''
    groups = _group_by_water_year(result, column, first_day_of_wy)
    t = groups['T']
    n = groups['n']
    return (n / t).where(t > 0, other=pd.NA)


def compute_metric_series(
    result: Result,
    column: str,
    mode: MetricMode = MetricMode.PORTION,
    first_day_of_wy: int = 1,
) -> pd.Series:
    '''Compute the configured summary metric for one column of a Result.

    Always starts from the underlying portion series (see compute_portion_series)
    and applies the metric mode transform. NA/zero policy:
        - PORTION:        unchanged; zero-success -> 0.0, NA (no timesteps) -> pd.NA.
        - PERCENTAGE:      portion * 100; same NA/zero policy as portion.
        - RETURN_PERIOD:   1 / portion; zero-success (undefined, i.e. infinite) and
                           NA portions both -> pd.NA (never inf).
    '''
    portion = compute_portion_series(result, column, first_day_of_wy)
    match mode:
        case MetricMode.PORTION:
            return portion
        case MetricMode.PERCENTAGE:
            return portion * 100
        case MetricMode.RETURN_PERIOD:
            return (1 / portion).where(portion > 0, other=pd.NA)  # type: ignore[call-overload]
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
        metric_mode:     Summary metric mode (portion/percentage/return_period).

    Returns the directory (or file parent) that received output files.
    """
    output_path = _resolve_output_path(input_path, output_directory, write_to_excel)
    filename_map = _build_all_filenames(scenario_results)

    if write_to_excel:
        output_filename = Path(input_path).stem + "_output.xlsx"
        target = output_path / output_filename
        if not overwrite:
            target = _next_available_path(target)
        with pd.ExcelWriter(target) as writer:
            for scenario_name, results in scenario_results.items():
                for result in results:
                    sheet = _build_sheet_name(scenario_name, result.component.name)
                    result.df.to_excel(writer, sheet_name=sheet)
    else:
        for (scenario_name, component_name), base_name in filename_map.items():
            csv_path = output_path / (base_name + ".csv")
            if not overwrite:
                csv_path = _next_available_path(csv_path)
            results = scenario_results[scenario_name]
            result = next(r for r in results if r.component.name == component_name)
            result.df.to_csv(csv_path)

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
        component = result.component
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


def _resolve_output_path(
    input_path: str,
    output_directory: str | None,
    write_to_excel: bool,
) -> Path:
    if output_directory:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    input_parent = Path(input_path).parent
    if write_to_excel:
        return input_parent

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
