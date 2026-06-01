"""Output formatting and file-writing helpers for CLI results."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from hydropattern.patterns import Result


def write_results(
    results: list[Result],
    input_path: str,
    output_directory: str | None,
    write_to_excel: bool,
) -> Path:
    """Write results to csv files or a single Excel file.

    Returns the directory that received output files.
    """
    output_path = _resolve_output_path(input_path, output_directory, write_to_excel)
    if write_to_excel:
        output_filename = Path(input_path).stem + "_output.xlsx"
        with pd.ExcelWriter(output_path / output_filename) as writer:
            for result in results:
                result.df.to_excel(writer, sheet_name=result.component.name)
        return output_path

    for index, result in enumerate(results):
        base_name = _build_result_filename(result, index)
        target = _next_available_path(output_path / base_name)
        result.df.to_csv(target)
    return output_path


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


def _build_result_filename(result: Result, index: int) -> str:
    dv_name = _clean_variable_name(result.dv_name)
    component_name = _clean_variable_name(result.component.name)
    return f"{index:03d}_{dv_name}_{component_name}.csv"


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
