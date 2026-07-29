# ruff: noqa
"""Plot TWL response surfaces (like data/analysis/twl/plot_center_save_points.py)
for the `data/filled/` and `data/extrapolated/` workbooks, using their own actual
values (no climate_canvas interpolation/fillin -- these directories already hold
hydropattern's own fillin_twl.py fill/extrapolation for every scenario cell they
have).

Same centroid save point per lake, same 13 ARI columns present in every TWL
workbook (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000), same
threshold (baseline `_0_0` average lake level from data/clean/<lake>_avg.csv,
in feet) as data/analysis/twl/plot_center_save_points.py. Title reads
"baseline average lake level=" (not "threshold=").

- data/filled/: 12 of 17 scenarios present (5 known + 7 in-hull filled); the 5
  out-of-hull cells are left blank (NaN).
- data/extrapolated/: all 17 scenarios present (also includes the 5 out-of-hull
  row-shift extrapolated cells) -- full grid, no blanks.

Writes `<lake>_<ari>.png` (raw actual values, interpolate=False) and
`<lake>_<ari>_interpolated.png` (climate_canvas's own interpolate=True,
fillin=True over the same actual cell values -- resampled to a finer grid,
Delaunay-fillin for any remaining NaN, e.g. data/filled/'s 5 missing
out-of-hull cells) to a new `img/` subdirectory under each of those two data
directories.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

GREAT_LAKES_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(GREAT_LAKES_DIR))

import common_twl  # noqa: E402
from fillin_twl import ALL_SCENARIO_SUFFIXES  # noqa: E402
from hydropattern.scenario_grid import parse_scenario_name  # noqa: E402
from climate_canvas.plots_utilities import plot_response_surface  # noqa: E402
from ari_constants import TWL_ARIS, format_ari  # noqa: E402

CLEAN_DATA_DIR = GREAT_LAKES_DIR / "data" / "clean"

ARIS = TWL_ARIS
METERS_TO_FEET = 1 / 0.3048

# All 17 scenarios' (precip, temp) coords, keyed by bare suffix, e.g. "_0_0" -> (0., 0.)
ALL_COORDS = {suffix: parse_scenario_name(suffix) for suffix in ALL_SCENARIO_SUFFIXES}
ALL_PRECIPS = sorted({p for p, _ in ALL_COORDS.values()})
ALL_TEMPS = sorted({t for _, t in ALL_COORDS.values()})


def center_save_point_id(sheets: dict[str, pd.DataFrame]) -> int:
    """Return the ID of the save point nearest the lake's lat/lon centroid."""
    baseline = next(iter(sheets.values()))
    centroid_lat, centroid_lon = baseline["lat"].mean(), baseline["lon"].mean()
    dist = np.hypot(baseline["lat"] - centroid_lat, baseline["lon"] - centroid_lon)
    return int(baseline.loc[dist.idxmin(), "ID"])


def build_grid(sheets: dict[str, pd.DataFrame], save_point_id: int,
               ari: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (precip x, temp y, TWL-in-feet z) over the full 17-scenario coordinate
    space. z is NaN for any of the 17 coordinates whose sheet isn't present in
    `sheets` (i.e. data/filled/'s 5 missing out-of-hull scenarios).
    """
    z = np.full((len(ALL_TEMPS), len(ALL_PRECIPS)), np.nan)
    for sheet_name, df in sheets.items():
        suffix = common_twl.parse_scenario_sheet_name(sheet_name)
        if suffix not in ALL_COORDS:
            continue
        precip, temp = ALL_COORDS[suffix]
        row = df.loc[df["ID"] == save_point_id, ari]
        if row.empty:
            continue
        z[ALL_TEMPS.index(temp), ALL_PRECIPS.index(precip)] = row.iloc[0] * METERS_TO_FEET
    return np.array(ALL_PRECIPS, dtype=float), np.array(ALL_TEMPS, dtype=float), z


def plot_data_dir(data_dir: Path) -> None:
    output_dir = data_dir / "img"
    output_dir.mkdir(parents=True, exist_ok=True)

    for lake in common_twl.LAKE_TWL_FILENAMES:
        path = common_twl.resolve_lake_twl_path(lake, data_dir)
        sheets = pd.read_excel(path, sheet_name=None)
        save_point_id = center_save_point_id(sheets)

        avg_means = common_twl.read_avg_scenario_means(lake, CLEAN_DATA_DIR)
        baseline_avg_ft = avg_means["_0_0"] * METERS_TO_FEET

        for ari in ARIS:
            ari_label = format_ari(ari)
            xs, ys, zs = build_grid(sheets, save_point_id, ari)
            title = (f"{lake.capitalize()} save point {save_point_id} -- ARI={ari_label} "
                      f"(baseline average lake level={baseline_avg_ft:.2f} ft)")

            save_path = output_dir / f"{lake}_{ari_label}.png"
            plot_response_surface(
                xs, ys, zs,
                interpolate=False,
                labels=("precip_delta", "temp_delta", "TWL (ft)"),
                title=title,
                save_path=save_path,
                show=False,
                threshold=baseline_avg_ft,
            )
            print(f"Wrote {save_path} (save point {save_point_id}, "
                  f"threshold={baseline_avg_ft:.2f} ft)")

            interpolated_path = output_dir / f"{lake}_{ari_label}_interpolated.png"
            plot_response_surface(
                xs, ys, zs,
                interpolate=True,
                fillin=True,
                labels=("precip_delta", "temp_delta", "TWL (ft)"),
                title=title,
                save_path=interpolated_path,
                show=False,
                threshold=baseline_avg_ft,
            )
            print(f"Wrote {interpolated_path} (save point {save_point_id}, "
                  f"threshold={baseline_avg_ft:.2f} ft)")


def main() -> None:
    plot_data_dir(GREAT_LAKES_DIR / "data" / "filled")
    plot_data_dir(GREAT_LAKES_DIR / "data" / "extrapolated")


if __name__ == "__main__":
    main()
