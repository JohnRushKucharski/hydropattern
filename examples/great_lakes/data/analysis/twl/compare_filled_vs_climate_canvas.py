# ruff: noqa
"""Compare hydropattern's fillin_twl.py in-hull fill against climate_canvas's own
interpolate+fillin, at the same (precip_delta, temp_delta) coordinates.

Both fill the 7 in-hull scenarios from the same 5 known scenarios using
Delaunay-linear (barycentric) interpolation:

- fillin_twl.py (data/filled/*.xlsx): builds one Delaunay triangulation over the
  5 known (precip, temp) points and evaluates it directly at each of the 7 in-hull
  target coordinates (see docs/adr/0001).
- climate_canvas.data_utilities.interpolator(..., fillin=True): bilinear on the
  known grid's cells, falling back to its own delaunay_fill() (also a Delaunay-
  linear fit over the same known points) for any cell with a missing corner --
  which is every in-hull scenario here, since none sit on the sparse known grid's
  own row/column lines.

For the 4 lakes' centroid save points (same ones used by
plot_center_save_points.py) and ARIs 1/10/50/100, this script evaluates both
methods at each of the 7 in-hull scenario coordinates and writes a CSV of the
values and their difference, to check the two approaches agree.

Result (see comparison.csv / the printed summary): differences are all
<= ~0.02 ft, consistent with float/rounding noise rather than a methodological
difference -- both are the same Delaunay-linear fit over the same known points.
See FILLED_VS_CLIMATE_CANVAS.md in this directory for the write-up.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

GREAT_LAKES_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(GREAT_LAKES_DIR))

import common_twl  # noqa: E402
from fillin_twl import known_scenario_coords  # noqa: E402
from hydropattern.scenario_grid import parse_scenario_name  # noqa: E402
from climate_canvas.data_utilities import interpolator  # noqa: E402

CLEAN_DATA_DIR = GREAT_LAKES_DIR / "data" / "clean"
FILLED_DATA_DIR = GREAT_LAKES_DIR / "data" / "filled"
OUTPUT_DIR = Path(__file__).parent

ARIS = (1, 10, 50, 100)
METERS_TO_FEET = 1 / 0.3048


def center_save_point_id(sheets: dict[str, pd.DataFrame]) -> int:
    baseline = next(iter(sheets.values()))
    centroid_lat, centroid_lon = baseline["lat"].mean(), baseline["lon"].mean()
    dist = np.hypot(baseline["lat"] - centroid_lat, baseline["lon"] - centroid_lon)
    return int(baseline.loc[dist.idxmin(), "ID"])


def main() -> None:
    rows = []
    for lake in common_twl.LAKE_TWL_FILENAMES:
        clean_path = common_twl.resolve_lake_twl_path(lake, CLEAN_DATA_DIR)
        known_sheets = pd.read_excel(clean_path, sheet_name=None)
        save_point_id = center_save_point_id(known_sheets)

        filled_path = common_twl.resolve_lake_twl_path(lake, FILLED_DATA_DIR)
        filled_sheets = pd.read_excel(filled_path, sheet_name=None)

        coords = known_scenario_coords(list(known_sheets.keys()))
        precips = sorted({p for p, _ in coords.values()})
        temps = sorted({t for _, t in coords.values()})

        for ari in ARIS:
            z = np.full((len(temps), len(precips)), np.nan)
            for name, df in known_sheets.items():
                suffix = common_twl.parse_scenario_sheet_name(name)
                if suffix not in coords:
                    continue
                p, t = parse_scenario_name(suffix)
                row = df.loc[df["ID"] == save_point_id, ari]
                if row.empty:
                    continue
                z[temps.index(t), precips.index(p)] = row.iloc[0] * METERS_TO_FEET
            xs = np.array(precips, dtype=float)
            ys = np.array(temps, dtype=float)
            cc_interp = interpolator(xs, ys, z, fillin=True)

            for name, df in filled_sheets.items():
                suffix = common_twl.parse_scenario_sheet_name(name)
                if suffix in coords or not suffix.startswith("_"):
                    continue  # skip known scenarios; only compare in-hull fills
                p, t = parse_scenario_name(suffix)
                row = df.loc[df["ID"] == save_point_id, ari]
                if row.empty:
                    continue
                filled_ft = row.iloc[0] * METERS_TO_FEET
                cc_ft = cc_interp((p, t))
                rows.append({
                    "lake": lake,
                    "save_point_id": save_point_id,
                    "ari": ari,
                    "scenario": suffix,
                    "precip_delta": p,
                    "temp_delta": t,
                    "fillin_twl_ft": round(filled_ft, 4),
                    "climate_canvas_ft": round(cc_ft, 4),
                    "diff_ft": round(filled_ft - cc_ft, 4),
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "filled_vs_climate_canvas_comparison.csv", index=False)
    print(f"n={len(out)}  max|diff|={out['diff_ft'].abs().max():.4f} ft  "
          f"mean|diff|={out['diff_ft'].abs().mean():.4f} ft")
    print(f"Wrote {OUTPUT_DIR / 'filled_vs_climate_canvas_comparison.csv'}")


if __name__ == "__main__":
    main()
