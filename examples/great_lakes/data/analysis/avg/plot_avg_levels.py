# ruff: noqa
"""Plot mean and Poisson-method ARI response surfaces for average lake levels.

For each of the 3 distinct average-lake-level datasets in `data/clean/` --
superior, michiganhuron (Michigan and Huron are one hydraulically-connected lake
and share a single avg-level dataset), ontario -- and each of the 17 precip/temp
scenarios in that dataset's monthly time series (1970-2999, 1030 years), this
computes:

1. The scenario's overall mean average lake level -> `<lake>_mean.png` (one
   response surface per lake, in `data/analysis/avg/`).
2. The average lake level associated with the TWL workbooks' ARI values that
   are >= 1 year (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 -- see
   ari_constants.TWL_ARIS; the <1-year TWL ARI values, 0.1/0.2/0.5, are dropped
   here -- see "Why ARI < 1 is dropped" below), for both the high-water tail
   (level exceeded on average once every ARI years) and the low-water tail
   (level undershot on average once every ARI years), using the
   Poisson/peaks-over-threshold method described in `declustered/METHOD.md` --
   NOT the annual-maximum-series (block maxima) method. Two variants:
   - non-declustered: every monthly value is treated as a Poisson-arrival
     candidate -> `high/<lake>_<ari>.png`, `low/<lake>_<ari>.png`.
   - declustered: only local turning points (peaks for the high tail, troughs
     for the low tail) of the monthly series are treated as candidates, so a
     multi-year high/low-water episode isn't counted as many separate "events"
     -> `declustered/high/<lake>_<ari>.png`, `declustered/low/<lake>_<ari>.png`.

Each of the plots above also gets a companion `*_interpolated.png` using
climate_canvas's own `interpolate=True, fillin=True` (matching the TWL plots'
convention) -- resampled to a finer grid; since the avg data already has every
one of the 17 scenarios (no NaN cells to begin with), this only resamples, it
never needs the Delaunay fillin fallback.

All plots are response surfaces over the full 17-scenario precip/temp grid,
values converted from meters to feet, with the baseline (`_0_0`) scenario's
mean average lake level (in feet) as both the diverging-colormap threshold and
shown in the title as "baseline average lake level=".

## Why ARI < 1 is dropped, and why ARI=1 is replaced

The Poisson method's target annual exceedance probability is `aep = 1/ari`.
`aep` is a probability, so it must be <= 1, which requires `ari >= 1` -- ARI
values below 1 year (like the TWL data's 0.1/0.2/0.5) are mathematically
undefined here, for *either* population (non-declustered or declustered): they
don't reflect data limitations, they reflect `aep` no longer being a valid
probability. (TWL's own <1-year ARI columns are fine -- they're precomputed by
a different, already-tabulated method upstream, not derived through this
Poisson inversion.)

Even ARI=1 itself is only a hair inside the valid range: it requires
`aep = 1.0` exactly, which needs an infinite Poisson rate, which no finite
population can supply. This implementation caps the required rate at the
population's own size, giving the population's raw extreme value as the
answer whenever the target is at or beyond what the population can achieve.
For the non-declustered population (fixed at 12 values/year for every
scenario), that ceiling is at ARI~=1.000006 years -- so close to 1 that it barely
matters. For the declustered population (roughly one peak/trough per year from
the lakes' seasonal cycle), that ceiling is much further out, more like
ARI~=1.7-1.9 years -- so declustered ARI=1 is a real, not just theoretical,
degenerate case (see `declustered/METHOD.md`).

Rather than plot a possibly-meaningless capped ARI=1 result, this instead
computes, per (lake, tail, declustered-or-not), the single highest (i.e.
worst-case, most restrictive) minimum-achievable ARI across all 17 scenarios,
and uses that -- instead of the literal value 1 -- as the smallest ARI in the
plotted set, applied uniformly to every scenario cell in that response surface
(so the whole plot stays one consistent, single-ARI comparison instead of each
cell secretly showing a different return period).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

GREAT_LAKES_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(GREAT_LAKES_DIR))

import common_twl  # noqa: E402
from fillin_twl import ALL_SCENARIO_SUFFIXES  # noqa: E402
from hydropattern.scenario_grid import parse_scenario_name  # noqa: E402
from climate_canvas.plots_utilities import plot_response_surface  # noqa: E402
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(GREAT_LAKES_DIR / "data" / "analysis" / "twl"))  # noqa: E402
from ari_constants import TWL_ARIS, format_ari  # noqa: E402

CLEAN_DATA_DIR = GREAT_LAKES_DIR / "data" / "clean"
OUTPUT_DIR = Path(__file__).parent

# ARI=1's target aep=1/1=1.0 is only reachable in the theoretical limit (needs
# infinite Poisson rate); ARI<1 needs aep>1, which isn't a valid probability at
# all. Drop ARI<1; ARI=1 itself is replaced per-lake/tail/decluster below with
# the actual minimum achievable ARI (see module docstring).
AVG_ARIS = tuple(ari for ari in TWL_ARIS if ari >= 1.0)
METERS_TO_FEET = 1 / 0.3048

# The 3 distinct avg-level datasets (Michigan and Huron share one; see
# common_twl.LAKE_AVG_FILENAMES).
AVG_LAKES = ("superior", "michiganhuron", "ontario")

ALL_COORDS = {suffix: parse_scenario_name(suffix) for suffix in ALL_SCENARIO_SUFFIXES}
ALL_PRECIPS = sorted({p for p, _ in ALL_COORDS.values()})
ALL_TEMPS = sorted({t for _, t in ALL_COORDS.values()})


def local_turning_points(values: np.ndarray, kind: str) -> np.ndarray:
    """Indices of local peaks (kind="peak") or troughs (kind="trough") in `values`.

    A peak is a point strictly greater than both its neighbors (or its one
    neighbor, at the series' boundary); a trough is strictly less. These are
    threshold-independent "events" -- identified from the series' shape alone,
    not from where a candidate ARI level happens to fall -- so counting how many
    of them exceed/undershoot a candidate level doesn't double-count a single
    multi-month excursion. See METHOD.md.
    """
    v = -values if kind == "trough" else values
    n = len(v)
    is_turning = np.zeros(n, dtype=bool)
    is_turning[1:-1] = (v[1:-1] > v[:-2]) & (v[1:-1] > v[2:])
    is_turning[0] = v[0] > v[1]
    is_turning[-1] = v[-1] > v[-2]
    return np.nonzero(is_turning)[0]


def population_for(values_ft: np.ndarray, tail: str, decluster: bool) -> np.ndarray:
    """The candidate population used for the Poisson rate: every monthly value
    (decluster=False), or just its local peaks/troughs (decluster=True).
    """
    if not decluster:
        return values_ft
    idx = local_turning_points(values_ft, "peak" if tail == "high" else "trough")
    return values_ft[idx]


def min_achievable_ari(columns: dict[str, np.ndarray], tail: str, decluster: bool) -> float:
    """Highest (worst-case/most-restrictive) minimum achievable ARI across all
    17 scenarios for this (tail, decluster) combination -- i.e. 1/max_aep for
    whichever scenario's population has the fewest members (the sparsest
    events), so a single ARI value is valid to apply uniformly across the
    whole response-surface grid. See module docstring.
    """
    years = None
    worst_ari = 1.0
    for suffix, values in columns.items():
        if suffix not in ALL_COORDS:
            continue
        values_ft = values * METERS_TO_FEET
        years = len(values_ft) / 12.0
        population = population_for(values_ft, tail, decluster)
        n = len(population)
        max_aep = 1.0 - np.exp(-n / years)
        worst_ari = max(worst_ari, 1.0 / max_aep)
    return worst_ari


def poisson_ari_level(population: np.ndarray, ari: float, years: float, tail: str) -> float:
    """Empirical Poisson-process ARI level for `tail` ("high" or "low").

    See METHOD.md for the derivation. Summary: target annual exceedance
    probability aep = 1/ari; solve aep = 1-exp(-k/years) for k (the expected
    number of population members exceeding/undershooting the answer over the
    whole record); read the level off the population's k-th order statistic
    (interpolating between order statistics for fractional k). If the required
    k exceeds the population size (aep >= the population's max achievable rate),
    the result is capped at the population's extreme value -- see METHOD.md's
    "ARI=1 edge case" section.
    """
    n = len(population)
    aep = 1.0 / ari
    if aep >= 1.0:
        k = float(n)
    else:
        k = min(years * (-np.log(1.0 - aep)), float(n))
    sorted_pop = np.sort(population)[::-1] if tail == "high" else np.sort(population)
    k_floor = max(1, min(int(np.floor(k)), n))
    k_ceil = max(1, min(int(np.ceil(k)), n))
    if k_floor == k_ceil:
        return float(sorted_pop[k_floor - 1])
    frac = k - k_floor
    return float(sorted_pop[k_floor - 1] * (1 - frac) + sorted_pop[k_ceil - 1] * frac)


def build_grid(columns: dict[str, np.ndarray],
               ari: float | None, tail: str | None, decluster: bool) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """Build (precip x, temp y, z) grid, z = mean (ari is None) or Poisson ARI
    level (ari given), over the full 17-scenario coordinate space, in feet.
    """
    z = np.full((len(ALL_TEMPS), len(ALL_PRECIPS)), np.nan)
    for suffix, values in columns.items():
        if suffix not in ALL_COORDS:
            continue
        precip, temp = ALL_COORDS[suffix]
        values_ft = values * METERS_TO_FEET
        if ari is None:
            level = values_ft.mean()
        else:
            years = len(values_ft) / 12.0
            population = population_for(values_ft, tail, decluster)
            level = poisson_ari_level(population, ari, years, tail)
        z[ALL_TEMPS.index(temp), ALL_PRECIPS.index(precip)] = level
    return np.array(ALL_PRECIPS, dtype=float), np.array(ALL_TEMPS, dtype=float), z


def one_sided_color_style(zs: np.ndarray, threshold: float, tail: str | None) -> tuple | None:
    """When every grid cell falls on one side of `threshold`, climate_canvas's
    default RdBu/TwoSlopeNorm can't place `threshold` at its usual colorbar
    center (TwoSlopeNorm requires vmin < vcenter < vmax) -- it silently falls
    back to the *data range's own midpoint* instead (see
    climate_canvas.data_utilities.check_threshold), which breaks the
    "colorbar center = baseline lake level" convention this whole module
    relies on: e.g. a low-ARI grid (every cell's level is, by definition,
    below the baseline average) would render with red/blue split around some
    arbitrary in-range value instead of showing "every cell is below
    baseline".

    Returns None for the normal (threshold falls strictly inside the
    grid's z-range) case, so the caller keeps climate_canvas's default RdBu
    diverging behavior. Otherwise returns (color_map, norm, levels, widths)
    for a one-sided sequential scale anchored at `threshold`:
    - low tail, all levels <= threshold: 'Reds_r' colormap, lightest (white)
      at `threshold`, darkest red at the grid's most extreme (lowest) level.
    - high tail, all levels >= threshold: 'Blues' colormap, lightest at
      `threshold`, darkest blue at the grid's most extreme (highest) level.
    `levels`/`widths` mark `threshold` plus 5 evenly-spaced points between it
    and the extreme (bold at threshold, matching the normal RdBu contour
    convention) -- these become the colorbar ticks; several (including
    threshold's own bold line) may fall outside the actual [min, max] of the
    plotted grid and simply won't have a visible contour line, but still
    label the colorbar so its scale stays legible.

    Tail-agnostic (mean) plots always pass tail=None and keep the default
    RdBu behavior, since the mean's threshold is not expected to fall outside
    its own grid's range the way an extreme-tail ARI level does.
    """
    if tail is None:
        return None
    z_min, z_max = float(np.nanmin(zs)), float(np.nanmax(zs))
    if z_min < threshold < z_max:
        return None
    if tail == "low" and z_max <= threshold:
        extreme, color_map = z_min, "Reds_r"
    elif tail == "high" and z_min >= threshold:
        extreme, color_map = z_max, "Blues"
    else:
        return None
    mids = [threshold + i * (extreme - threshold) / 6 for i in range(1, 6)]
    ascending = sorted([extreme, threshold] + mids)
    widths = tuple(2.0 if lvl == threshold else 1.0 for lvl in ascending)
    return color_map, Normalize(vmin=min(threshold, extreme), vmax=max(threshold, extreme)), \
        tuple(ascending), widths


def plot_pair(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, title: str,
              save_path: Path, threshold: float, labels: tuple[str, str, str],
              tail: str | None = None) -> None:
    """Write both the raw (interpolate=False) plot and its interpolated
    (interpolate=True, fillin=True) companion, `<name>.png` / `<name>_interpolated.png`.
    """
    style = one_sided_color_style(zs, threshold, tail)
    color_map = style[0] if style else "RdBu"
    extra = dict(norm=style[1], levels=style[2], widths=style[3]) if style else {}

    plot_response_surface(
        xs, ys, zs, interpolate=False, labels=labels, title=title,
        save_path=save_path, show=False, threshold=threshold, color_map=color_map, **extra,
    )
    print(f"Wrote {save_path}")

    interpolated_path = save_path.with_name(save_path.stem + "_interpolated.png")
    plot_response_surface(
        xs, ys, zs, interpolate=True, fillin=True, labels=labels, title=title,
        save_path=interpolated_path, show=False, threshold=threshold, color_map=color_map, **extra,
    )
    print(f"Wrote {interpolated_path}")


def main() -> None:
    for lake in AVG_LAKES:
        path = CLEAN_DATA_DIR / f"{lake}_avg.csv"
        df = pd.read_csv(path)
        columns = {col: df[col].to_numpy(dtype=float) for col in df.columns if col != "time"}
        baseline_avg_ft = columns["_0_0"].mean() * METERS_TO_FEET

        # 1. Mean plot (tail-agnostic).
        xs, ys, zs = build_grid(columns, ari=None, tail=None, decluster=False)
        title = f"{lake.capitalize()} mean level (baseline avg={baseline_avg_ft:.2f} ft)"
        plot_pair(xs, ys, zs, title, OUTPUT_DIR / f"{lake}_mean.png", baseline_avg_ft,
                  ("precip_delta", "temp_delta", "mean lake level (ft)"))

        # 2. ARI plots: high/low tail x declustered/non-declustered.
        for tail in ("high", "low"):
            for decluster in (False, True):
                subdir = (OUTPUT_DIR / "declustered" / tail) if decluster else (OUTPUT_DIR / tail)

                # Replace the dropped ARI=1 with this (tail, decluster)
                # combination's actual worst-case minimum achievable ARI,
                # applied uniformly across every scenario in the grid.
                floor_ari = min_achievable_ari(columns, tail, decluster)
                aris = sorted({round(floor_ari, 2)} | {a for a in AVG_ARIS if a > 1.0})

                for ari in aris:
                    xs, ys, zs = build_grid(columns, ari=ari, tail=tail, decluster=decluster)
                    decl_note = " decl" if decluster else ""
                    ari_label = format_ari(ari)
                    floor_note = " [min]" if ari == round(floor_ari, 2) else ""
                    # Kept short (single line) -- a longer title clips at the
                    # figure edge since plot_response_surface doesn't use a
                    # tight savefig bbox.
                    title = (f"{lake.capitalize()} {tail}-water ARI={ari_label}{floor_note}"
                             f"{decl_note} (baseline avg={baseline_avg_ft:.2f} ft)")
                    plot_pair(xs, ys, zs, title, subdir / f"{lake}_{ari_label}.png",
                              baseline_avg_ft,
                              ("precip_delta", "temp_delta", "lake level (ft)"), tail=tail)


if __name__ == "__main__":
    main()
