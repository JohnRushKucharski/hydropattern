import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydropattern.patterns import evaluate_component
from hydropattern.patterns import Component
from hydropattern.parsers import duration_parser, magnitude_parser, timing_parser
from hydropattern.scenario_grid import parse_scenario_name
from hydropattern.scenarios import evaluate_scenarios
from hydropattern.timeseries import Timeseries


@pytest.fixture(scope="module")
def notebook():
    path = Path(__file__).parents[1] / "examples" / "longtailpoint" / "musky_575.ipynb"
    document = json.loads(path.read_text(encoding="utf-8"))
    namespace = {
        "Component": Component,
        "magnitude_parser": magnitude_parser,
        "timing_parser": timing_parser,
        "duration_parser": duration_parser,
        "pd": pd,
        "np": np,
        "Timeseries": Timeseries,
        "parse_scenario_name": parse_scenario_name,
        "evaluate_scenarios": evaluate_scenarios,
    }
    for cell in document["cells"]:
        if "musky-definitions" in cell.get("metadata", {}).get("tags", []):
            exec("".join(cell["source"]), namespace)
    return namespace


def monthly_frame(levels, year=2000):
    dates = pd.date_range(f"{year}-01-01", periods=12, freq="MS", name="time")
    return Timeseries.from_dataframe(pd.DataFrame({"level": levels}, index=dates)).data


def test_igld85_bottom_conversion_and_depth_thresholds(notebook):
    assert notebook["bottom_m"](575) == pytest.approx(175.26)
    assert notebook["level_at_depth"](575, 0.3) == pytest.approx(175.56)
    components = notebook["build_components"](575)
    bottom = notebook["bottom_m"](575)

    juvenile = evaluate_component(monthly_frame(np.full(12, bottom + 0.30)),
                                  components["juvenile_habitat"])
    assert juvenile.df["juvenile_habitat"].loc["2000-05":"2000-08"].eq(0).all()
    juvenile = evaluate_component(monthly_frame(np.full(12, bottom + 0.31)),
                                  components["juvenile_habitat"])
    assert juvenile.df["juvenile_habitat"].loc["2000-05":"2000-08"].eq(1).all()

    for key, depths in [
        ("spawning_zones", (0.3, 1.5)),
        ("recruitment_zones", (0.3, 0.6)),
    ]:
        for depth in depths:
            frame = monthly_frame(np.full(12, bottom + depth))
            result = evaluate_component(frame, components[key])
            assert result.df[key].loc["2000-05":"2000-06"].eq(1).all()


@pytest.mark.parametrize(
    ("key", "months", "depth"),
    [
        ("juvenile_habitat", (5, 6, 7, 8), 0.31),
        ("spawning_zones", (5, 6), 0.3),
        ("recruitment_zones", (5, 6), 0.3),
    ],
)
def test_duration_requires_all_consecutive_season_months(notebook, key, months, depth):
    component = notebook["build_components"](575)[key]
    duration = next(
        characteristic for characteristic in component.characteristics
        if characteristic.type.name == "DURATION"
    )
    assert duration.name == "season_duration"

    bottom = notebook["bottom_m"](575)
    valid = np.full(12, bottom + depth)
    valid_frame = monthly_frame(valid)
    valid_result = evaluate_component(valid_frame, component).df
    assert valid_result.loc[valid_frame.index.month.isin(months), key].eq(1).all()

    for failed_month in months:
        invalid = valid.copy()
        invalid[failed_month - 1] = bottom
        invalid_frame = monthly_frame(invalid)
        result = evaluate_component(invalid_frame, component).df
        assert result.loc[invalid_frame.index.month.isin(months), key].eq(0).all()
        assert result.loc[invalid_frame.index.month.isin(months), "season_duration"].eq(0).all()


def test_recruitment_uses_rolling_five_year_windows(notebook):
    annual = pd.Series([1, 1, 0, 1, 0, 0, 1], index=range(2000, 2007), dtype=float)
    windows = notebook["window_verdicts"](annual)
    assert windows.index.tolist() == [2000, 2001, 2002]
    np.testing.assert_array_equal(windows, [1, 0, 0])
    summary = notebook["verdict_summary"](windows)
    assert summary["Evaluated units"] == 3
    assert summary["Successes"] == 1
    assert summary["Percentage"] == pytest.approx(100 / 3)


def test_unknown_window_uses_midpoint_and_sets_flag(notebook):
    annual = pd.Series([1, 1, np.nan, 0, 0], index=range(2000, 2005), dtype=float)
    summary = notebook["verdict_summary"](notebook["window_verdicts"](annual))
    assert summary["Lower percentage"] == 0
    assert summary["Upper percentage"] == 100
    assert summary["Percentage"] == 50
    assert summary["Midpoint used"]


def test_rolling_window_requires_consecutive_complete_years(notebook):
    annual = pd.Series([1, 1, 1, 1, 1], index=[2000, 2001, 2003, 2004, 2005])
    with pytest.raises(ValueError, match="consecutive"):
        notebook["window_verdicts"](annual)


def test_all_habitat_surfaces_use_climate_canvas_red_blue_ramp():
    root = Path(__file__).parents[1] / "examples" / "longtailpoint"
    depth_paths = [root / f"musky_{depth}.ipynb" for depth in range(573, 580)]
    for path in depth_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        surface_code = next(
            "".join(cell["source"]) for cell in document["cells"]
            if cell["cell_type"] == "code" and "def baseline_scale(" in "".join(cell["source"])
        )
        assert "'RdBu'" in surface_code
        assert "'Blues'" not in surface_code

    matrix_path = root / "musky_matrix.ipynb"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    plot_code = next(
        "".join(cell["source"]) for cell in matrix["cells"]
        if cell["cell_type"] == "code" and "def plot_depth_component_matrix" in "".join(cell["source"])
    )
    assert "cmap='RdBu'" in plot_code
    assert "ax.plot([0], [0]" not in plot_code
    assert "def plot_recruitment_matrix" in plot_code
    assert "summary_all, depths_ft, ('recruitment_zones',), 20" in plot_code
