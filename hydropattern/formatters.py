"""Output formatting and file-writing helpers for CLI results."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from hydropattern.patterns import Result


def write_results(
    scenario_results: dict[str, list[Result]],
    input_path: str,
    output_directory: str | None,
    write_to_excel: bool,
    overwrite: bool = True,
) -> Path:
    """Write per-scenario results to csv files or a single Excel file.

    Each key in scenario_results is a scenario name (timeseries column header).
    Each value is the list of within-scenario component Results for that scenario.

    Args:
        overwrite: When True (default), existing output files are replaced.
            When False, a numeric suffix (__1, __2, …) is appended to avoid
            overwriting existing files.

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
        return output_path

    for (scenario_name, component_name), base_name in filename_map.items():
        csv_path = output_path / (base_name + ".csv")
        if not overwrite:
            csv_path = _next_available_path(csv_path)
        # find the matching result
        results = scenario_results[scenario_name]
        result = next(r for r in results if r.component.name == component_name)
        result.df.to_csv(csv_path)
    return output_path


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
