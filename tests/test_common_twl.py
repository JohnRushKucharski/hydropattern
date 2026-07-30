"""Tests for the Great Lakes common_twl.py shared helpers.

The script lives outside the hydropattern package, in examples/great_lakes/ -- which is
itself a regular Python package (see examples/great_lakes/__init__.py) -- so it's
imported normally rather than loaded by file path.
"""
import pytest

from examples.great_lakes import common_twl

# ---- m_igld85_to_ft_navg88 -------------------------------------------------------------

# Known conversion pairs from examples/great_lakes/longtailpoint/longtail_waterlevel.xlsx
# Sheet2's "NAVG88 to IGLD85" table: NAVG88 587ft -> IGLD85 586.56ft, NAVG88 582ft ->
# IGLD85 581.56ft -- i.e. a flat -0.44ft offset (IGLD85_ft = NAVG88_ft - 0.44), the
# "USACE team estimate" noted in that workbook's conversion_ex sheet (not the slightly
# different computed ~0.4093ft offset, which the workbook explicitly says was not used).
# Meters here are exact (offset ft) * 0.3048, not the workbook's own separately-sourced
# "Full Set of Elevations" ft->m column, which carries its own independent rounding
# noise (~0.01ft) from an external survey-based conversion and so isn't a valid oracle
# for this module's simple, exact offset+scale formula.
@pytest.mark.parametrize("meters,expected_ft_navg88", [
    (586.56 * 0.3048, 587.0),
    (581.56 * 0.3048, 582.0),
])
def test_m_igld85_to_ft_navg88_known_pairs(meters, expected_ft_navg88):
    assert common_twl.m_igld85_to_ft_navg88(meters) == pytest.approx(expected_ft_navg88, abs=1e-6)


def test_m_igld85_to_ft_navg88_zero():
    # 0m IGLD85 -> 0ft IGLD85 -> +0.44ft NAVG88 (offset only, no scale error).
    assert common_twl.m_igld85_to_ft_navg88(0.0) == pytest.approx(0.44)


def test_m_igld85_to_ft_navg88_negative_value():
    assert common_twl.m_igld85_to_ft_navg88(-30.48) == pytest.approx(-99.56)


def test_m_igld85_to_ft_navg88_round_trips_with_ft_navg88_to_m_igld85():
    original = 178.478688
    converted = common_twl.m_igld85_to_ft_navg88(original)
    assert common_twl.ft_navg88_to_m_igld85(converted) == pytest.approx(original)


def test_ft_navg88_to_m_igld85_known_value():
    # 586ft NAVG88 -> 585.56ft IGLD85 -> *0.3048 -> 178.478688m IGLD85.
    assert common_twl.ft_navg88_to_m_igld85(586.0) == pytest.approx(178.478688)


# ---- resolve_lake_avg_path -----------------------------------------------------------


def test_resolve_lake_avg_path_michigan_and_huron_share_same_file(tmp_path):
    michigan_path = common_twl.resolve_lake_avg_path("michigan", tmp_path)
    huron_path = common_twl.resolve_lake_avg_path("huron", tmp_path)
    assert michigan_path == huron_path == tmp_path / "michiganhuron_avg.csv"


def test_resolve_lake_avg_path_superior(tmp_path):
    assert common_twl.resolve_lake_avg_path("superior", tmp_path) == tmp_path / "superior_avg.csv"


def test_resolve_lake_avg_path_ontario(tmp_path):
    assert common_twl.resolve_lake_avg_path("ontario", tmp_path) == tmp_path / "ontario_avg.csv"


# ---- read_avg_scenario_means ----------------------------------------------------------

# A tiny, hand-computable avg csv: 3 monthly rows, 2 scenario columns.
#   _0_0:  183.0, 183.2, 183.4  -> mean = 183.2
#   _0_7:  182.0, 182.5, 182.5  -> mean = 182.333...
_TINY_AVG_ROWS = (
    "time,_0_0,_0_7\n"
    "1970-01-01,183.0,182.0\n"
    "1970-02-01,183.2,182.5\n"
    "1970-03-01,183.4,182.5\n"
)


def _write_tiny_avg_csv(tmp_path, filename="superior_avg.csv"):
    path = tmp_path / filename
    path.write_text(_TINY_AVG_ROWS)
    return path


def test_read_avg_scenario_means_computes_mean_per_scenario_column(tmp_path):
    _write_tiny_avg_csv(tmp_path)
    means = common_twl.read_avg_scenario_means("superior", tmp_path)
    assert means["_0_0"] == pytest.approx(183.2)
    assert means["_0_7"] == pytest.approx(182.333333, rel=1e-6)


def test_read_avg_scenario_means_excludes_time_column(tmp_path):
    _write_tiny_avg_csv(tmp_path)
    means = common_twl.read_avg_scenario_means("superior", tmp_path)
    assert "time" not in means


def test_read_avg_scenario_means_michigan_and_huron_read_same_file(tmp_path):
    _write_tiny_avg_csv(tmp_path, filename="michiganhuron_avg.csv")
    michigan_means = common_twl.read_avg_scenario_means("michigan", tmp_path)
    huron_means = common_twl.read_avg_scenario_means("huron", tmp_path)
    assert michigan_means == huron_means == {"_0_0": pytest.approx(183.2),
                                              "_0_7": pytest.approx(182.333333, rel=1e-6)}


def test_read_avg_scenario_means_uses_full_record_not_a_subset(tmp_path):
    # 12 rows instead of 3 -- mean must reflect all of them, not e.g. just the first 3.
    rows = "time,_0_0\n" + "\n".join(f"1970-{m:02d}-01,{100.0 + m}" for m in range(1, 13)) + "\n"
    (tmp_path / "superior_avg.csv").write_text(rows)
    means = common_twl.read_avg_scenario_means("superior", tmp_path)
    expected = sum(100.0 + m for m in range(1, 13)) / 12
    assert means["_0_0"] == pytest.approx(expected)
