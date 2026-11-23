import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydropattern.patterns import evaluate_component
from hydropattern.timeseries import Timeseries


@pytest.fixture(scope="module", params=[574, 575, 576])
def notebook(request):
    path = (
        Path(__file__).parents[1] / "examples" / "longtailpoint" / f"pondweed_{request.param}.ipynb"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    namespace = {}
    for cell in document["cells"]:
        if "pondweed-definitions" in cell.get("metadata", {}).get("tags", []):
            exec("".join(cell["source"]), namespace)
    namespace["notebook_path"] = path
    namespace["expected_bottom_ft"] = request.param
    return namespace


def monthly_frame(levels, year=2000):
    dates = pd.DatetimeIndex(
        np.array([f"{year}-{month:02d}-01" for month in range(1, 13)], dtype="datetime64[s]"),
        name="time",
    )
    return Timeseries.from_dataframe(pd.DataFrame({"level": levels}, index=dates)).data


def test_depth_to_elevation_and_inclusive_winter_threshold(notebook):
    bottom = notebook["expected_bottom_ft"] * 0.3048
    assert notebook["BOTTOM_FT"] == notebook["expected_bottom_ft"]
    assert notebook["BOTTOM_M"] == pytest.approx(bottom)
    assert notebook["WINTER_LEVEL_M"] == pytest.approx(bottom + 0.5)
    levels = np.full(12, round(bottom + 0.5, 4))
    frame = monthly_frame(levels)
    component = notebook["winter"]
    result = evaluate_component(frame, component)
    assert result.df[component.name].eq(1).all()
    frame.iloc[11, 0] -= 0.001
    result = evaluate_component(frame, component)
    annual = notebook["annual_verdicts"](result.df[component.name], tuple(range(1, 13)))
    assert annual.loc[2000] == 0


@pytest.mark.parametrize("year", [2000, 2001, 2400])
def test_growing_timing_includes_may_through_august_in_leap_and_future_years(notebook, year):
    bottom = notebook["expected_bottom_ft"] * 0.3048
    assert notebook["GROWING_LEVELS_M"] == pytest.approx((bottom + 1, bottom + 3))
    frame = monthly_frame(np.full(12, round(bottom + 1, 4)), year)
    result = evaluate_component(frame, notebook["growing"])
    np.testing.assert_array_equal(result.df["growing_season"], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    assert (
        notebook["annual_verdicts"](result.df[notebook["growing"].name], (5, 6, 7, 8)).loc[year]
        == 1
    )
    frame.iloc[7, 0] = round(bottom + 3, 4)
    result = evaluate_component(frame, notebook["growing"])
    assert result.df["magnitude"].iloc[7] == 1
    frame.iloc[7, 0] += 0.001
    result = evaluate_component(frame, notebook["growing"])
    assert result.df["magnitude"].iloc[7] == 0


def test_signed_monthly_rise_may_reset_and_exact_limit(notebook):
    frame = monthly_frame([175, 175, 175, 175, 177, 177.15, 176.7, 176.851, 177, 177, 177, 177])
    rises = notebook["seasonal_rises"](frame.iloc[:, 0])
    assert np.isnan(rises.iloc[4])
    assert rises.iloc[5] == pytest.approx(0.15)
    assert rises.iloc[6] == pytest.approx(-0.45)
    result = evaluate_component(frame, notebook["growing"])
    np.testing.assert_array_equal(result.df["monthly_rise"].iloc[4:8], [1, 1, 1, 0])
    second = monthly_frame(np.full(12, 177.0), 2001)
    combined = pd.concat([frame, second])
    assert np.isnan(notebook["seasonal_rises"](combined.iloc[:, 0]).loc["2001-05-01"])


def test_nan_rate_passes_without_converting_depth_to_success(notebook):
    frame = monthly_frame(np.full(12, 177.0))
    frame.iloc[5, 0] = np.nan
    np.testing.assert_array_equal(notebook["rise_outcomes"](frame).iloc[4:8], [1, 1, 1, 1])
    depth_outcomes = pd.Series(np.ones(12), index=frame.index)
    depth_outcomes.iloc[5] = np.nan
    assert np.isnan(notebook["annual_verdicts"](depth_outcomes, (5, 6, 7, 8)).iloc[0])


def test_annual_success_is_all_months_not_monthly_fraction(notebook):
    frame = pd.concat([monthly_frame(np.ones(12)), monthly_frame(np.ones(12), 2001)])
    outcomes = frame.iloc[:, 0].copy()
    outcomes.iloc[5] = 0
    annual = notebook["annual_verdicts"](outcomes, (5, 6, 7, 8))
    np.testing.assert_array_equal(annual, [0, 1])
    summary = notebook["annual_summary"](annual)
    assert summary["Percentage"] == 50
    assert summary["Complete years"] == 2
    assert summary["Qualifying years"] == 1


def test_partial_years_excluded_and_unknown_years_disclosed(notebook):
    frame = pd.concat([monthly_frame(np.ones(12)), monthly_frame(np.ones(12), 2001)])
    outcomes = frame.iloc[2:-2, 0]
    with pytest.raises(ValueError, match="complete"):
        notebook["annual_verdicts"](outcomes, (5, 6, 7, 8))
    outcomes = frame.iloc[:, 0].copy()
    outcomes.iloc[5] = np.nan
    annual = notebook["annual_verdicts"](outcomes, (5, 6, 7, 8))
    summary = notebook["annual_summary"](annual)
    assert np.isnan(summary["Percentage"])
    assert summary["Unknown years"] == 1
    assert summary["Lower percentage"] == 50
    assert summary["Upper percentage"] == 100
    outcomes.iloc[6] = 0
    assert notebook["annual_verdicts"](outcomes, (5, 6, 7, 8)).iloc[0] == 0
    complete_and_partial = notebook["annual_verdicts"](outcomes.iloc[:-2], (5, 6, 7, 8))
    assert complete_and_partial.index.tolist() == [2000]


def test_missing_month_is_rejected(notebook):
    frame = monthly_frame(np.ones(12)).drop(index="2000-06-01")
    with pytest.raises(ValueError):
        notebook["annual_verdicts"](frame.iloc[:, 0], (5, 6, 7, 8))


def test_all_scenarios_match_independent_annual_calculation(notebook):
    path = Path(__file__).parents[1] / "examples" / "longtailpoint" / "michiganhuron_avg.csv"
    timeseries = Timeseries.from_csv(str(path), first_dowy=1, date_format="%Y-%m-%d")
    evaluated = notebook["evaluate_scenarios"](
        timeseries, [notebook["winter"], notebook["growing"]]
    )
    seasonal = timeseries.data.index.month.isin((5, 6, 7, 8))
    for name in timeseries.data.columns[:-1]:
        levels = timeseries.data[name]
        bottom = round(notebook["expected_bottom_ft"] * 0.3048, 4)
        winter_expected = levels.ge(bottom + 0.5).groupby(levels.index.year).all()
        growing_levels = levels[seasonal]
        rises = growing_levels.groupby(growing_levels.index.year).diff().round(10)
        growing_expected = (
            (growing_levels.between(bottom + 1, bottom + 3) & (rises.isna() | rises.le(0.15)))
            .groupby(growing_levels.index.year)
            .all()
        )
        for component, expected, months in [
            (notebook["winter"], winter_expected, tuple(range(1, 13))),
            (notebook["growing"], growing_expected, (5, 6, 7, 8)),
        ]:
            result = evaluated.by_component(component.name)[name]
            annual = notebook["annual_verdicts"](result.df[component.name], months)
            np.testing.assert_array_equal(annual.index, expected.index)
            np.testing.assert_array_equal(annual, expected.astype(float))
            assert notebook["annual_summary"](annual)["Percentage"] == pytest.approx(
                100 * expected.mean()
            )


@pytest.mark.parametrize("baseline_percentage", [0, 30, 78.7378640776699, 100])
def test_surfaces_center_baseline_and_show_both_variants(notebook, baseline_percentage):
    path = notebook["notebook_path"]
    document = json.loads(path.read_text(encoding="utf-8"))
    surface_cell = next(cell for cell in document["cells"] if cell["id"] == "response-surfaces")
    names = ["_0_0", "_0_3", "_5_0", "_5_3"]
    rows = [
        {
            "Component": component.name,
            "Scenario": name,
            "Percentage": percentage,
            "Unknown years": 0,
        }
        for component in (notebook["winter"], notebook["growing"])
        for name, percentage in zip(names, (baseline_percentage, 0, 60, 100))
    ]
    captured = []
    namespace = {
        **notebook,
        "scenario_names": names,
        "summary": pd.DataFrame(rows).set_index(["Component", "Scenario"]),
        "display": lambda *args: None,
        "plot_response_surface": lambda *args, **kwargs: captured.append((args, kwargs)),
    }
    exec("".join(surface_cell["source"]), namespace)
    assert len(captured) == 4
    for index, (args, kwargs) in enumerate(captured):
        assert (kwargs["interpolate"], kwargs["fillin"]) == (
            bool(index % 2),
            bool(index % 2),
        )
        norm = kwargs["norm"]
        assert norm(baseline_percentage) == pytest.approx(0.5)
        assert kwargs["threshold"] == baseline_percentage
        assert kwargs["color_map_ticks"][3] == pytest.approx(baseline_percentage)
        assert kwargs["levels"][5] == pytest.approx(baseline_percentage)
        assert kwargs["widths"][5] > kwargs["widths"][4]
        assert f"baseline white = {baseline_percentage:.2f}%" in kwargs["title"]
        assert f"{notebook['expected_bottom_ft']} ft" in kwargs["title"]
        if 0 < baseline_percentage < 100:
            assert (norm.vmin, norm.vmax) == (0, 100)
        np.testing.assert_allclose(norm(kwargs["color_map_ticks"]), np.linspace(0, 1, 7))
        assert len(kwargs["widths"]) == len(kwargs["levels"]) == 11
        assert np.nanmin(args[2]) == 0 and np.nanmax(args[2]) == 100


def test_notebook_filename_and_site_text_match_parameters(notebook):
    document = json.loads(notebook["notebook_path"].read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in document["cells"])
    assert f"**{notebook['expected_bottom_ft']} ft**" in text
    assert f"{notebook['BOTTOM_M']:.4f} m" in text
    assert f"{notebook['WINTER_LEVEL_M']:.4f} m" in text
    assert f"{notebook['GROWING_LEVELS_M'][0]:.4f}-{notebook['GROWING_LEVELS_M'][1]:.4f} m" in text
    assert "baseline" in text and "midpoint" in text
