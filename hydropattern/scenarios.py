'''Library entrypoint for scenario-based evaluation.

Mirrors the evaluate_component -> evaluate_components progression in
hydropattern.patterns with one more level: evaluate_scenarios splits a
multi-column Timeseries into one DataFrame per scenario column and runs
evaluate_components on each, returning a ScenarioResults wrapper that keeps
CLI and library callers on the same code path (see hydropattern.cli).
'''
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hydropattern.formatters import build_summary_sheet, plot_component_response_surface, \
    write_results
from hydropattern.parsers import ClimateCanvasPlotOptions, MetricMode, MetricOptions
from hydropattern.patterns import Component, Result, evaluate_components
from hydropattern.timeseries import Timeseries


def split_scenarios(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    '''Split a multi-column timeseries into one DataFrame per scenario.

    The last column is always 'dowy' and is included in every scenario slice.
    Each returned DataFrame has exactly two columns: the scenario data column
    and 'dowy', matching the shape expected by evaluate_component.

    A single-column timeseries (one data column + dowy) returns a dict with
    one entry -- the degenerate single-scenario case.
    '''
    dowy_col = data.columns[-1]
    return {col: data[[col, dowy_col]] for col in data.columns[:-1]}


@dataclass(frozen=True)
class ScenarioResults:
    '''In-memory result of evaluating one or more components across every scenario
    column of a Timeseries.

    Built by evaluate_scenarios; callers should not construct this directly.
    '''
    scenario_results: dict[str, list[Result]]
    first_day_of_water_year: int = 1
    source_path: str | None = None

    def by_component(self, component_name: str) -> dict[str, Result]:
        '''Filters to {scenario_name: Result} for a single named component.

        Raises ValueError if no scenario has a Result for component_name (fails fast
        instead of silently returning an empty/partial dict).
        '''
        filtered: dict[str, Result] = {}
        for scenario_name, results in self.scenario_results.items():
            match = next((r for r in results if r.component.name == component_name), None)
            if match is not None:
                filtered[scenario_name] = match
        if not filtered:
            raise ValueError(
                f'No component named {component_name!r} found in any scenario results.'
            )
        return filtered

    def summary(self, component_name: str | None = None,
               mode: MetricMode = MetricMode.PORTION) -> pd.DataFrame | dict[str, pd.DataFrame]:
        '''Computes the configured summary metric (default: portion) for one component,
        or for every component when component_name is omitted.

        Opt-in, in-memory only -- delegates to formatters.build_summary_sheet, the same
        function the CLI's file-writing summary uses, so results match the CLI exactly.
        Columns = scenario names; index = ['total', wy1, wy2, ...].
        '''
        if component_name is not None:
            return build_summary_sheet(
                self.scenario_results, component_name, component_name,
                self.first_day_of_water_year, mode,
            )
        first_results = next(iter(self.scenario_results.values()))
        return {
            result.component.name: build_summary_sheet(
                self.scenario_results, result.component.name, result.component.name,
                self.first_day_of_water_year, mode,
            )
            for result in first_results
        }

    def to_excel(self, output_directory: str | None = None, overwrite: bool = True,
                metric_mode: MetricMode = MetricMode.PORTION) -> Path:
        '''Writes a single Excel workbook (one sheet per scenario/component pair) plus
        one {component}_summary.xlsx per component.

        Delegates to formatters.write_results (write_to_excel=True) -- the same writer
        the CLI uses -- so file layout/naming matches CLI output exactly.
        '''
        return self._write(output_directory, True, overwrite, metric_mode)

    def to_csv(self, output_directory: str | None = None, overwrite: bool = True,
              metric_mode: MetricMode = MetricMode.PORTION) -> Path:
        '''Writes one csv file per scenario/component pair, plus one
        {component}_summary.xlsx per component (always written, for CLI parity).

        Delegates to formatters.write_results (write_to_excel=False).
        '''
        return self._write(output_directory, False, overwrite, metric_mode)

    def _write(self, output_directory: str | None, write_to_excel: bool, overwrite: bool,
              metric_mode: MetricMode) -> Path:
        # No source csv/xlsx (in-memory Timeseries) -> fall back to a fixed name so the
        # default output dir/file are still sensibly named 'hydropattern_output'
        # (write_results/_resolve_output_path derive names from this path's stem).
        input_path = self.source_path or 'hydropattern.csv'
        return write_results(self.scenario_results, input_path, output_directory,
                             write_to_excel, overwrite, self.first_day_of_water_year,
                             metric_mode)

    def plot_response_surface(
            self, component_name: str, output_path: str | Path | None = None,
            metric_options: MetricOptions = MetricOptions(),
            climate_canvas: ClimateCanvasPlotOptions = ClimateCanvasPlotOptions()) -> None:
        '''Builds and plots component_name's response-surface grid.

        Raises ValueError if component_name isn't found (see by_component) or
        HydropatternError if the scenario names don't form a valid precip/temp
        grid (see hydropattern.scenario_grid).

        output_path=None (default): shows the plot interactively, writes no files.
        output_path given: treated as a directory (created if needed) and follows
        the CLI's file layout -- writes both '{component}_grid.csv' and
        '{component}_plot.png' into it.
        '''
        component = next(iter(self.by_component(component_name).values())).component
        resolved_output_path = Path(output_path) if output_path is not None else None
        if resolved_output_path is not None:
            resolved_output_path.mkdir(parents=True, exist_ok=True)
        plot_component_response_surface(
            self.scenario_results, component, metric_options, self.first_day_of_water_year,
            climate_canvas, resolved_output_path,
        )


def evaluate_scenarios(timeseries: Timeseries, components: list[Component]) -> ScenarioResults:
    '''Evaluates one or more components on every scenario column of a Timeseries.

    Splits timeseries.data into one [scenario_col, dowy] DataFrame per scenario
    (see split_scenarios) and runs evaluate_components on each, mirroring the CLI's
    per-scenario evaluation so library and CLI callers share the same results shape.
    '''
    scenarios = split_scenarios(timeseries.data)
    scenario_results = {
        name: evaluate_components(df, components) for name, df in scenarios.items()
    }
    return ScenarioResults(scenario_results, timeseries.first_day_of_water_year,
                          timeseries.file_path)


__all__ = ['ScenarioResults', 'evaluate_scenarios', 'split_scenarios']

