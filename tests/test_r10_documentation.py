"""Executable acceptance checks for R10 user documentation and examples."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from hydropattern.cli import app
from hydropattern.formatters import compute_portion_series
from hydropattern.patterns import (
    Component,
    Result,
    comparison_fx,
    duration_fx,
    frequency_fx,
    nested_frequency_intra_annual_fx,
    water_year_exposure,
)
from hydropattern.timeseries import to_day_of_water_year

ROOT = Path(__file__).resolve().parents[1]
USER_DOCS = ROOT / "docs" / "user"


def test_unknown_outcome_worked_examples_match_evaluator():
    page = (USER_DOCS / "concepts" / "unknown-outcomes.md").read_text(
        encoding="utf-8"
    )
    frame = pd.DataFrame(index=range(6))
    duration_input = np.array([0, 1, 1, np.nan, 1, 0], dtype=float)
    duration = duration_fx(comparison_fx(">=", 3), order=2)(
        frame, duration_input.reshape(-1, 1)
    )
    np.testing.assert_equal(duration, [0, np.nan, np.nan, np.nan, np.nan, 0])
    assert "[0, unknown, unknown, unknown, unknown, 0]" in page

    bounded_duration = duration_fx(
        comparison_fx("<=", 2, "<=", 3), order=2
    )(
        pd.DataFrame(index=range(3)),
        np.array([[1], [np.nan], [1]], dtype=float),
    )
    np.testing.assert_equal(bounded_duration, [np.nan, np.nan, np.nan])
    assert "[1, unknown, 1]" in page

    frequency_examples = (
        (
            np.array([[np.nan], [0], [0], [0]], dtype=float),
            comparison_fx(">=", 1),
            False,
            [np.nan, np.nan, np.nan, 0],
        ),
        (
            np.array([[np.nan], [1], [0], [0]], dtype=float),
            comparison_fx(">=", 1),
            False,
            [np.nan, 1, 1, 1],
        ),
        (
            np.array([[np.nan], [1], [0], [0]], dtype=float),
            comparison_fx(">=", 1),
            True,
            [np.nan, 1, 1, np.nan],
        ),
    )
    for source, predicate, exclusive, expected in frequency_examples:
        frequency = frequency_fx(
            predicate, order=2, big_n=3, exclusive_windows=exclusive
        )
        actual = frequency(pd.DataFrame(index=range(4)), source)
        np.testing.assert_equal(actual, expected)
    for exclusive in (False, True):
        correlated = frequency_fx(
            comparison_fx("=", 1), order=2, big_n=2,
            exclusive_windows=exclusive,
        )(
            pd.DataFrame(index=range(2)),
            np.array([[1], [np.nan]], dtype=float),
        )
        np.testing.assert_equal(correlated, [np.nan, 1])
    assert "[unknown, 1, 1, unknown]" in page
    assert "[unknown, 1]" in page

    counts = np.array([[1], [1], [np.nan]], dtype=float)
    for threshold, expected in ((2, 1), (3, np.nan), (4, 0)):
        actual = frequency_fx(
            comparison_fx(">=", threshold), order=2, big_n=3
        )(pd.DataFrame(index=range(3)), counts)
        np.testing.assert_equal(actual, np.full(3, expected))
    assert "`>= 2` succeeds, `>= 3` is unknown, and" in page
    assert "`>= 4` fails" in page


def test_annual_frequency_examples_match_evaluator():
    page = (USER_DOCS / "concepts" / "unknown-outcomes.md").read_text(
        encoding="utf-8"
    )
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    dowy = dates.dayofyear - (dates.is_leap_year & (dates.month > 2)).astype(int)
    frame = pd.DataFrame({"flow": np.ones(12), "dowy": dowy}, index=dates)
    examples = (
        ([1, 0] + [np.nan] * 10, np.nan),
        ([1] * 7 + [np.nan] * 5, 1),
        ([0] * 7 + [np.nan] * 5, 0),
        ([np.nan] * 12, np.nan),
    )
    for source, verdict in examples:
        actual = nested_frequency_intra_annual_fx(
            comparison_fx(">=", 0.5), order=2
        )(frame, np.asarray(source, dtype=float).reshape(-1, 1))
        np.testing.assert_equal(actual, np.full(12, verdict))
    assert "1 / 1 / 10" in page
    assert "7 / 0 / 5" in page
    assert "0 / 7 / 5" in page
    assert "0 / 0 / 12" in page


def test_unknown_summary_and_event_count_example_match_results():
    page = (USER_DOCS / "concepts" / "unknown-outcomes.md").read_text(
        encoding="utf-8"
    )
    dates = pd.date_range("2020-01-01", periods=3, freq="D", name="time")
    result_frame = pd.DataFrame(
        {"flow": [1, 1, 1], "dowy": [1, 2, 3], "component": [1, np.nan, 1]},
        index=dates,
    )
    result = Result(
        result_frame,
        Component(
            name="component", characteristics=[], is_success_pattern=True
        ),
        first_day_of_water_year=1,
    )

    assert compute_portion_series(result, "component")["total"] == 1
    assert result.df["component"].notna().mean() == pytest.approx(2 / 3)
    assert result.event_count_bounds().lower == 1
    assert result.event_count_bounds().upper == 2
    assert "[1, unknown, 1]" in page


def test_mvp_event_bounds_caveat_matches_frequency_possibilities():
    page = (USER_DOCS / "concepts" / "unknown-outcomes.md").read_text(
        encoding="utf-8"
    )
    frame = pd.DataFrame(index=range(3))
    frequency = frequency_fx(comparison_fx(">=", 1), order=2, big_n=3)
    uncertain = frequency(frame, np.array([[np.nan], [0], [0]], dtype=float))
    failed = frequency(frame, np.zeros((3, 1)))
    successful = frequency(frame, np.array([[1], [0], [0]], dtype=float))

    np.testing.assert_equal(uncertain, [np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(failed, [0, 0, 0])
    np.testing.assert_array_equal(successful, [1, 1, 1])
    result = Result(
        pd.DataFrame(
            {"flow": np.ones(3), "dowy": [1, 2, 3], "component": uncertain}
        ),
        Component(name="component", characteristics=[], is_success_pattern=True),
    )
    assert (result.event_count_bounds().lower, result.event_count_bounds().upper) == (
        0, 2
    )
    assert "source-dependent event count is 0 or 1" in page
    assert "conservative bounds of 0-2" in page


def test_leap_day_exposure_examples_match_calendar_helper():
    dates_2019 = pd.date_range("2019-01-01", "2019-12-31", freq="D")
    dates_2020 = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    dates_2020_without_leap_day = dates_2020.drop(pd.Timestamp("2020-02-29"))
    partial_2020 = pd.date_range("2020-01-01", "2020-02-28", freq="D")

    assert water_year_exposure(dates_2019, 1) == pytest.approx(1)
    assert water_year_exposure(dates_2020, 1) == pytest.approx(1)
    assert water_year_exposure(dates_2020_without_leap_day, 1) == pytest.approx(1)
    assert water_year_exposure(partial_2020, 1) == pytest.approx(59 / 365)

    page = (USER_DOCS / "concepts" / "unknown-outcomes.md").read_text(
        encoding="utf-8"
    )
    assert "365/365" in page
    assert "366/366" in page
    assert "January 1-February 28, 2020" in page


def test_event_attribution_and_partial_exposure_examples_match_result_api():
    event_page = (USER_DOCS / "guide" / "outputs.md").read_text(encoding="utf-8")
    crossing_dates = pd.date_range("2020-09-29", "2020-10-03", freq="D", name="time")
    crossing = Result(
        pd.DataFrame(
            {
                "flow": np.ones(5),
                "dowy": [
                    to_day_of_water_year(date, first_day_of_wy=274)
                    for date in crossing_dates
                ],
                "component": np.ones(5),
            },
            index=crossing_dates,
        ),
        Component(name="component", characteristics=[], is_success_pattern=True),
        first_day_of_water_year=274,
    )
    annual_counts = crossing.event_count_bounds_by_water_year()
    assert annual_counts[2020].lower == annual_counts[2020].upper == 1
    assert annual_counts[2021].lower == annual_counts[2021].upper == 0
    assert "September 29-October 3" in event_page

    monthly_dates = pd.date_range("2020-01-01", periods=18, freq="MS", name="time")
    monthly_outcomes = [1, 0, 1, 0, 1] + [0] * 13
    monthly = Result(
        pd.DataFrame(
            {
                "flow": np.ones(18),
                "dowy": [to_day_of_water_year(date) for date in monthly_dates],
                "component": monthly_outcomes,
            },
            index=monthly_dates,
        ),
        Component(name="component", characteristics=[], is_success_pattern=True),
        first_day_of_water_year=1,
    )
    rates = monthly.event_rate_bounds()
    assert rates.lower == pytest.approx(2)
    assert rates.upper == pytest.approx(2)
    assert "Eighteen monthly observations" in event_page

def test_unknown_outcomes_are_linked_from_user_guides_and_references():
    linked_pages = (
        USER_DOCS / "index.md",
        USER_DOCS / "concepts" / "evaluation-order.md",
        USER_DOCS / "reference" / "characteristics" / "duration.md",
        USER_DOCS / "reference" / "characteristics" / "frequency.md",
        USER_DOCS / "reference" / "characteristics" / "magnitude.md",
        USER_DOCS / "reference" / "characteristics" / "rate-of-change.md",
        USER_DOCS / "guide" / "outputs.md",
        USER_DOCS / "guide" / "plotting.md",
        USER_DOCS / "api" / "index.md",
    )
    for path in linked_pages:
        assert "unknown-outcomes.md" in path.read_text(encoding="utf-8"), path.name

    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Unknown outcomes: concepts/unknown-outcomes.md" in navigation
    page = (USER_DOCS / "concepts" / "unknown-outcomes.md").read_text(
        encoding="utf-8"
    )
    for claim in (
        "[1, unknown, 1]", "66.7%", "365/365", "366/366",
    ):
        assert claim in page
    outputs = (USER_DOCS / "guide" / "outputs.md").read_text(encoding="utf-8")
    assert "September 29-October 3" in outputs
    examples = (USER_DOCS / "examples" / "index.md").read_text(encoding="utf-8")
    assert "below_minimum_coverage" in examples


def test_response_surface_pack_cli_matches_documented_coverage(tmp_path, monkeypatch):
    source = ROOT / "examples" / "response-surface"
    for name in ("flow.csv", "response-surface.toml"):
        shutil.copyfile(source / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)

    with pytest.warns(RuntimeWarning, match="below_minimum_coverage"):
        result = CliRunner().invoke(
            app, ["run", "response-surface.toml", "--plot", "--no-excel"]
        )

    assert result.exit_code == 0, result.output
    output = tmp_path / "response-surface_output"
    outcomes = {
        name: pd.read_csv(output / f"{name[1:]}_increasing_flow.csv")
        for name in ("_0_0", "_0_1", "_1_0", "_1_1")
    }
    expected_outcomes = {
        "_0_0": [np.nan, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "_0_1": [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "_1_0": [np.nan, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "_1_1": [np.nan, 1, 0, np.nan, 1, 0, 1, 0, 1, 0],
    }
    for name, expected in expected_outcomes.items():
        actual = outcomes[name]["increasing_flow"].astype(float).to_numpy()
        np.testing.assert_equal(actual, expected)

    coverage = pd.read_csv(output / "increasing_flow_grid_coverage.csv")
    coverage = coverage.set_index("scenario")
    assert coverage.loc["_0_0", "known_count"] == 9
    assert coverage.loc["_0_0", "total_count"] == 10
    assert coverage.loc["_0_0", "eligible"]
    assert coverage.loc["_1_1", "known_count"] == 8
    assert coverage.loc["_1_1", "total_count"] == 10
    assert not coverage.loc["_1_1", "eligible"]
    assert coverage.loc["_1_1", "exclusion_reason"] == "below_minimum_coverage"
    assert coverage["raw_summary"].tolist() == pytest.approx(
        [5 / 9, 1, 4 / 9, 1 / 2]
    )
    grid = pd.read_csv(output / "increasing_flow_grid.csv", index_col=0)
    assert pd.isna(grid.loc[1.0, "1.0"])
    assert (output / "increasing_flow_plot.png").is_file()
    expected_files = {
        "0_0_increasing_flow.csv",
        "0_1_increasing_flow.csv",
        "1_0_increasing_flow.csv",
        "1_1_increasing_flow.csv",
        "increasing_flow_summary.xlsx",
        "increasing_flow_grid.csv",
        "increasing_flow_grid_coverage.csv",
        "increasing_flow_plot.png",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    summary = pd.read_excel(
        output / "increasing_flow_summary.xlsx",
        sheet_name="increasing_flow",
        index_col=0,
    )
    assert summary.loc["total"].to_dict() == pytest.approx(
        {"_0_0": 5 / 9, "_0_1": 1, "_1_0": 4 / 9, "_1_1": 1 / 2}
    )

    page = (USER_DOCS / "examples" / "index.md").read_text(encoding="utf-8")
    assert "response-surface coverage" in page.lower()
    unknown_page = (USER_DOCS / "concepts" / "unknown-outcomes.md").read_text(
        encoding="utf-8"
    )
    assert "`[1, unknown, 1]`" in unknown_page
    assert "response-surface" in page
