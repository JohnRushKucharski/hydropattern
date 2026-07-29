# ruff: noqa
"""Plot TWL response surfaces for a center save point on each lake, in feet.

For each of the 4 Great Lakes twl workbooks in `data/clean/` (5 **known** scenarios
only -- no filled/interpolated/extrapolated sheets), picks the save point nearest the
lake's centroid (mean lat/lon over all its save points) and plots climate_canvas's
response surface -- precip_delta (x) vs temp_delta (y), colored by TWL in **feet** --
for the 1, 10, 50, and 100 ARI columns. Two plots per lake/ARI:

- `<lake>_<ari>.png`: interpolation OFF -- only the 5 known precip/temp cells are
  colored, everything else left blank.
- `<lake>_<ari>_interpolated.png`: climate_canvas's own `--interp`/`--fillin`
  features on -- the known cells are linearly resampled to a finer grid, and the
  cells outside the known scenarios' convex hull (which linear resampling alone
  leaves blank) are estimated via climate_canvas's own Delaunay-linear fillin. This
  is independent of (and does not use) hydropattern's own fillin_twl.py row-shift
  extrapolation in data/filled or data/extrapolated -- it is climate_canvas
  resampling the same 5 known-scenario points shown in the first plot.

The colormap threshold (its centered/diverging color) is that lake's baseline
scenario (`_0_0`) average lake level -- i.e. today's average water level under that
same scenario -- read from `<avg-lake>_avg.csv` in `data/clean/`, converted to feet,
same threshold for both plots and all 4 ARIs of a given lake. The threshold value is
also shown in the plot title.

Writes all PNGs to this directory (`data/analysis/twl/`).

This is a one-off example-tooling script, not part of the hydropattern package, so it
is excluded from linting (see the `# ruff: noqa` above).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# examples/great_lakes/ holds common_twl.py and fillin_twl.py; add it to sys.path so
# this script can be run directly regardless of cwd. This script lives 3 directories
# below it: data/analysis/twl/.
GREAT_LAKES_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(GREAT_LAKES_DIR))

import common_twl  # noqa: E402
from fillin_twl import known_scenario_coords  # noqa: E402
from hydropattern.scenario_grid import parse_scenario_name  # noqa: E402
from climate_canvas.plots_utilities import plot_response_surface  # noqa: E402
from ari_constants import TWL_ARIS, format_ari  # noqa: E402

CLEAN_DATA_DIR = GREAT_LAKES_DIR / "data" / "clean"
OUTPUT_DIR = Path(__file__).parent

ARIS = TWL_ARIS
METERS_TO_FEET = 1 / 0.3048


def center_save_point_id(sheets: dict[str, pd.DataFrame]) -> int:
    """Return the ID of the save point nearest the lake's lat/lon centroid.

    Uses the baseline (`_0_0`) sheet's lat/lon columns -- identical across every
    known sheet in the workbook, so any one of them would do.
    """
    baseline = next(iter(sheets.values()))
    centroid_lat, centroid_lon = baseline["lat"].mean(), baseline["lon"].mean()
    dist = np.hypot(baseline["lat"] - centroid_lat, baseline["lon"] - centroid_lon)
    return int(baseline.loc[dist.idxmin(), "ID"])


def build_known_grid(sheets: dict[str, pd.DataFrame], save_point_id: int,
                      ari: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (precip x, temp y, TWL-in-feet z) grid for one save point/ARI.

    z is shaped (len(y), len(x)) with NaN everywhere except the 5 known
    (precip_delta, temp_delta) cells, matching climate_canvas.cli.response's own
    csv-driven grid layout (interpolation off leaves the rest blank).
    """
    coords = known_scenario_coords(list(sheets.keys()))
    precips = sorted({p for p, _ in coords.values()})
    temps = sorted({t for _, t in coords.values()})

    z = np.full((len(temps), len(precips)), np.nan)
    for sheet_name, df in sheets.items():
        suffix = common_twl.parse_scenario_sheet_name(sheet_name)
        if suffix not in coords:
            continue
        precip, temp = parse_scenario_name(suffix)
        row = df.loc[df["ID"] == save_point_id, ari]
        if row.empty:
            continue
        z[temps.index(temp), precips.index(precip)] = row.iloc[0] * METERS_TO_FEET

    return np.array(precips, dtype=float), np.array(temps, dtype=float), z


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for lake in common_twl.LAKE_TWL_FILENAMES:
        path = common_twl.resolve_lake_twl_path(lake, CLEAN_DATA_DIR)
        sheets = pd.read_excel(path, sheet_name=None)
        save_point_id = center_save_point_id(sheets)

        avg_means = common_twl.read_avg_scenario_means(lake, CLEAN_DATA_DIR)
        baseline_avg_ft = avg_means["_0_0"] * METERS_TO_FEET

        for ari in ARIS:
            ari_label = format_ari(ari)
            xs, ys, zs = build_known_grid(sheets, save_point_id, ari)
            title = (f"{lake.capitalize()} save point {save_point_id} -- ARI={ari_label} "
                      f"(baseline average lake level={baseline_avg_ft:.2f} ft)")

            known_path = OUTPUT_DIR / f"{lake}_{ari_label}.png"
            plot_response_surface(
                xs, ys, zs,
                interpolate=False,
                labels=("precip_delta", "temp_delta", "TWL (ft)"),
                title=title,
                save_path=known_path,
                show=False,
                threshold=baseline_avg_ft,
            )
            print(f"Wrote {known_path} (save point {save_point_id}, "
                  f"threshold={baseline_avg_ft:.2f} ft)")

            interpolated_path = OUTPUT_DIR / f"{lake}_{ari_label}_interpolated.png"
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


if __name__ == "__main__":
    main()
