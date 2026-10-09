"""Executable checks for worked examples added during documentation phase S4."""

import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydropattern.parsers import build_components, parse_request
from hydropattern.patterns import evaluate_component
from hydropattern.timeseries import to_day_of_water_year

ROOT = Path(__file__).resolve().parents[1]
USER_DOCS = ROOT / "docs" / "user"


def _evaluate(flow, characteristics, *, timestamps=None, dowy=None):
    if dowy is None:
        dowy = np.arange(1, len(flow) + 1)
    if timestamps is not None:
        timestamps = pd.DatetimeIndex(timestamps, name="time")
    data = pd.DataFrame({"flow": flow, "dowy": dowy}, index=timestamps)
    component = build_components(parse_request({
        "worked_example": {"characteristics": characteristics},
    }))[0]
    return evaluate_component(data, component).df


@pytest.mark.parametrize(
    ("relative_path", "characteristics", "flow", "dowy", "timestamps", "column", "expected"),
    [
        (
            "reference/characteristics/magnitude.md",
            [{"type": "magnitude", "parameters": [">=", 1]}],
            [0, 1, 2, 1, 0],
            None,
            None,
            "magnitude_ge1",
            [0, 1, 1, 1, 0],
        ),
        (
            "reference/characteristics/timing.md",
            [{"type": "timing", "parameters": [335, 60]}],
            [1, 1, 1, 1, 1],
            [334, 335, 59, 60, 61],
            pd.to_datetime([
                "2021-11-30", "2021-12-01", "2022-02-28",
                "2022-03-01", "2022-03-02",
            ]),
            "timing_335-60",
            [0, 1, 1, 1, 0],
        ),
        (
            "reference/characteristics/duration.md",
            [
                {"type": "magnitude", "parameters": [">", 0]},
                {"type": "duration", "parameters": [2, 3]},
            ],
            [1, 1, 1, 1, 0, 1, 1, 0, 0],
            None,
            None,
            "duration_2-3",
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
        ),
        (
            "reference/characteristics/duration.md",
            [
                {"type": "magnitude", "parameters": [">", 0]},
                {"type": "duration", "parameters": [">=", 7]},
            ],
            [1] * 10 + [0],
            None,
            None,
            "duration_ge7",
            [1] * 10 + [0],
        ),
        (
            "reference/characteristics/frequency.md",
            [
                {"type": "magnitude", "parameters": [">", 0]},
                {"type": "frequency", "parameters": [">=", 1, 5]},
            ],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            None,
            None,
            "frequency_ge1in5(union)",
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ),
        (
            "reference/characteristics/frequency.md",
            [
                {"type": "magnitude", "parameters": [">", 0]},
                {"type": "frequency", "parameters": [">=", 1, 5, True]},
            ],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            None,
            None,
            "frequency_ge1in5(exclusive)",
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ),
        (
            "reference/characteristics/frequency.md",
            [
                {"type": "magnitude", "parameters": [">", 0]},
                {"type": "frequency", "parameters": ["=", 0, 3]},
            ],
            [0, 0, 0, 1, 0],
            None,
            None,
            "frequency_eq0in3(union)",
            [1, 1, 1, 0, 1],
        ),
        (
            "reference/characteristics/frequency.md",
            [
                {"type": "magnitude", "parameters": [">", 0]},
                {"type": "frequency", "parameters": [">=", 1, 5]},
            ],
            [0, 0, 0, 0, 1],
            None,
            None,
            "frequency_ge1in5(union)",
            [0, 0, 0, 0, 1],
        ),
    ],
)
def test_worked_characteristic_examples_match_evaluator(
    relative_path, characteristics, flow, dowy, timestamps, column, expected
):
    result = _evaluate(flow, characteristics, dowy=dowy, timestamps=timestamps)

    np.testing.assert_array_equal(result[column], expected)
    np.testing.assert_array_equal(result["worked_example"], expected)
    documented = (USER_DOCS / relative_path).read_text(encoding="utf-8")
    assert f"Expected diagnostic: `{expected}`" in documented
    assert f"Expected component outcome: `{expected}`" in documented


def test_every_toml_code_block_is_syntactically_valid():
    for path in USER_DOCS.rglob("*.md"):
        markdown = path.read_text(encoding="utf-8")
        for number, block in enumerate(
            re.findall(r"```toml[^\n]*\n(.*?)```", markdown, re.DOTALL), start=1
        ):
            try:
                tomllib.loads(block)
            except tomllib.TOMLDecodeError as exc:
                pytest.fail(f"{path.relative_to(ROOT)} TOML block {number}: {exc}")


def test_nested_frequency_example_matches_complete_year_results():
    timestamps = pd.date_range("2018-01-01", periods=36, freq="MS", name="time")
    flow = [1] * 9 + [0] * 3 + [0] * 12 + [1] * 6 + [0] * 6
    result = _evaluate(
        flow,
        [
            {"type": "magnitude", "parameters": [">=", 1]},
            {"type": "frequency", "parameters": [[">=", 0.5], [">=", 1, 2]]},
        ],
        timestamps=timestamps,
        dowy=[to_day_of_water_year(date) for date in timestamps],
    )

    np.testing.assert_array_equal(
        result["frequency_ge0.5(union)"], [1] * 12 + [0] * 12 + [1] * 12
    )
    np.testing.assert_array_equal(result["frequency_ge1in2(interannual_union)"], [1] * 36)
    np.testing.assert_array_equal(result["worked_example"], [1] * 36)
    documented = (
        USER_DOCS / "reference" / "characteristics" / "frequency.md"
    ).read_text(encoding="utf-8")
    annual = ", ".join(["1"] * 12 + ["0"] * 12 + ["1"] * 12)
    all_success = ", ".join(["1"] * 36)
    assert f"Expected intra-annual diagnostic: `[{annual}]`" in documented
    assert f"Expected interannual diagnostic: `[{all_success}]`" in documented
    assert f"Expected component outcome: `[{all_success}]`" in documented


def test_rate_of_change_example_uses_only_defined_ratios():
    result = _evaluate(
        [1, 2, 4, 8, 4],
        [{"type": "rate_of_change", "parameters": [">=", 2]}],
    )

    expected = [1, 1, 1, 0]
    np.testing.assert_array_equal(result["rate_of_change_ge2"].iloc[1:], expected)
    np.testing.assert_array_equal(result["worked_example"].iloc[1:], expected)
    documented = (
        USER_DOCS / "reference" / "characteristics" / "rate-of-change.md"
    ).read_text(encoding="utf-8")
    assert f"Expected diagnostic from the second observation onward: `{expected}`" in documented
    assert (
        f"Expected component outcome from the second observation onward: `{expected}`"
        in documented
    )


def test_evaluation_order_example_matches_characteristic_and_component_results():
    result = _evaluate(
        [0, 2, 3, 0, 4, 0],
        [
            {"type": "magnitude", "parameters": [">", 1]},
            {"type": "duration", "parameters": [">=", 2]},
        ],
    )

    np.testing.assert_array_equal(result["magnitude_gt1"], [0, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(result["duration_ge2"], [0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(result["worked_example"], [0, 1, 1, 0, 0, 0])
    documented = (USER_DOCS / "concepts" / "evaluation-order.md").read_text(encoding="utf-8")
    assert "Expected magnitude diagnostic: `[0, 1, 1, 0, 1, 0]`" in documented
    assert "Expected duration diagnostic: `[0, 1, 1, 0, 0, 0]`" in documented
    assert "Expected component outcome: `[0, 1, 1, 0, 0, 0]`" in documented
