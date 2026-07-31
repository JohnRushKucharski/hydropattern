"""Tests for the Great Lakes known-scenario directionality analysis script.

analyze_known_scenarios.py lives outside the hydropattern package
(examples/great_lakes/), and (unlike fillin_twl.py / common_twl.py) uses bare
`import common_twl` / `from fillin_twl import ...` script-style imports rather than a
package-relative import, so it only works with examples/great_lakes/ itself on
sys.path -- the same way it would run as `python analyze_known_scenarios.py`. This
test file replicates that by inserting examples/great_lakes/ onto sys.path before
importing it via importlib.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

GREAT_LAKES_DIR = Path(__file__).parent.parent / "examples" / "great_lakes"
if str(GREAT_LAKES_DIR) not in sys.path:
    sys.path.insert(0, str(GREAT_LAKES_DIR))

_spec = importlib.util.spec_from_file_location(
    "analyze_known_scenarios", GREAT_LAKES_DIR / "analyze_known_scenarios.py"
)
assert _spec is not None and _spec.loader is not None
analyze_known_scenarios = importlib.util.module_from_spec(_spec)
sys.modules["analyze_known_scenarios"] = analyze_known_scenarios
_spec.loader.exec_module(analyze_known_scenarios)


# ---- sheet_name_for_suffix ----------------------------------------------------------

def test_sheet_name_for_suffix_finds_matching_sheet():
    coords = {"_0_0": (0.0, 0.0), "_0_7": (0.0, 7.0)}
    names = ["baseline-_0_0", "extreme_low-_0_7"]
    assert analyze_known_scenarios.sheet_name_for_suffix("_0_7", coords, names) == "extreme_low-_0_7"


def test_sheet_name_for_suffix_raises_when_suffix_not_known():
    with pytest.raises(ValueError, match="not a known scenario"):
        analyze_known_scenarios.sheet_name_for_suffix("_99_9", {}, [])


def test_sheet_name_for_suffix_raises_when_no_sheet_matches():
    coords = {"_0_0": (0.0, 0.0)}
    with pytest.raises(ValueError, match="No sheet found"):
        analyze_known_scenarios.sheet_name_for_suffix("_0_0", coords, ["unrelated-_5_5"])


# ---- summarize ------------------------------------------------------------------------

def test_summarize_computes_expected_stats_and_sign_shares():
    # Two save points, one ARI, one lake: precip_pct is -10% (falls, matches
    # expectation) for point A and +10% (rises, does not match) for point B; temp_pct
    # mirrors this (rises for A, falls for B).
    results = pd.DataFrame({
        "lake": ["superior", "superior"],
        "ID": [1, 2],
        "lat": [46.0, 46.1],
        "lon": [-84.0, -84.1],
        "ari": [10, 10],
        "twl_p10_dt5": [100.0, 100.0],
        "twl_p20_dt5": [90.0, 110.0],
        "precip_delta": [-10.0, 10.0],
        "precip_pct": [-10.0, 10.0],
        "twl_dt0_p0": [100.0, 100.0],
        "twl_dt7_p0": [110.0, 90.0],
        "temp_delta": [10.0, -10.0],
        "temp_pct": [10.0, -10.0],
    })

    summaries = analyze_known_scenarios.summarize(results)

    overall = summaries["overall"].iloc[0]
    assert overall["n_save_points"] == 2
    assert overall["precip_pct_mean"] == pytest.approx(0.0)
    assert overall["pct_negative_precip"] == pytest.approx(50.0)
    assert overall["temp_pct_mean"] == pytest.approx(0.0)
    assert overall["pct_positive_temp"] == pytest.approx(50.0)

    by_ari = summaries["by_ari"]
    assert by_ari["ari"].tolist() == [10]

    by_lake_and_ari = summaries["by_lake_and_ari"]
    assert by_lake_and_ari[["lake", "ari"]].iloc[0].tolist() == ["superior", 10]


# ---- analyze_lake ---------------------------------------------------------------------

def _write_known_scenario_workbook(path: Path) -> None:
    """Minimal <lake>_twl.xlsx with the 4 sheets analyze_lake actually reads."""
    sheets = {
        "baseline-_0_0": pd.DataFrame({"ID": [1], "lat": [46.0], "lon": [-84.0], 10: [100.0]}),
        "extreme_low-_0_7": pd.DataFrame({"ID": [1], "lat": [46.0], "lon": [-84.0], 10: [110.0]}),
        "moderate_low-_10_5": pd.DataFrame({"ID": [1], "lat": [46.0], "lon": [-84.0], 10: [100.0]}),
        "extreme_high-_20_5": pd.DataFrame({"ID": [1], "lat": [46.0], "lon": [-84.0], 10: [95.0]}),
    }
    with pd.ExcelWriter(path) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def test_analyze_lake_computes_precip_and_temp_directionality(tmp_path):
    from examples.great_lakes import common_twl
    workbook_path = tmp_path / common_twl.LAKE_TWL_FILENAMES["superior"]
    _write_known_scenario_workbook(workbook_path)

    result = analyze_known_scenarios.analyze_lake("superior", tmp_path)

    assert len(result) == 1  # 1 save point x 1 ARI column
    row = result.iloc[0]
    assert row["lake"] == "superior"
    assert row["ari"] == 10
    # precip_delta = twl_p20_dt5 (95.0) - twl_p10_dt5 (100.0) = -5.0
    assert row["precip_delta"] == pytest.approx(-5.0)
    assert row["precip_pct"] == pytest.approx(-5.0)
    # temp_delta = twl_dt7_p0 (110.0) - twl_dt0_p0 (100.0) = 10.0
    assert row["temp_delta"] == pytest.approx(10.0)
    assert row["temp_pct"] == pytest.approx(10.0)
