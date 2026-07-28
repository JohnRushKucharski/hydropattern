# ruff: noqa
"""Shared helpers for reading `<lake>_twl.xlsx` workbooks and `<avg-lake>_avg.csv` files.

Used by both batch_run_twl.py (evaluates twl frequency curves against a resource's
magnitude threshold) and fillin_twl.py (estimates missing scenario sheets via
Delaunay-linear interpolation and row-shift extrapolation). Factored out here rather
than duplicated, or imported one from the other, since neither script is a natural
"owner" of the other's logic.

This is a one-off example-tooling module, not part of the hydropattern package, so it
is excluded from linting (see the `# ruff: noqa` above).
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

# lake code -> twl xlsx filename, shared by data/clean/ and data/filled/ (directory
# alone disambiguates the two). Only 4 lakes have twl data (michigan and huron are
# separate files here, unlike batch_run_avg.py's combined "michiganhuron"; stclair/erie
# have no twl data at all).
LAKE_TWL_FILENAMES = {
    "superior": "superior_twl.xlsx",
    "michigan": "michigan_twl.xlsx",
    "huron": "huron_twl.xlsx",
    "ontario": "ontario_twl.xlsx",
}

# twl-lake code -> avg-lake key. Michigan and Huron are one hydraulically-connected
# lake body sharing a single average-lake-level record, despite having separate twl
# workbooks/codes above (see docs/adr/0001-row-shift-extrapolation-for-out-of-hull-
# scenarios.md for why this matters: extrapolation shifts a twl sheet by a delta in
# *average lake level*, which must come from the correct shared record for Michigan
# and Huron).
TWL_LAKE_TO_AVG_LAKE = {
    "superior": "superior",
    "michigan": "michiganhuron",
    "huron": "michiganhuron",
    "ontario": "ontario",
}

# avg-lake key -> average-lake-level csv filename (see
# data/raw/clean_lake_levels_all_scenarios.py, which produces these files; stclair and
# erie also exist there but have no corresponding twl data, so are omitted here).
LAKE_AVG_FILENAMES = {
    "superior": "superior_avg.csv",
    "michiganhuron": "michiganhuron_avg.csv",
    "ontario": "ontario_avg.csv",
}

# Sheet columns that are not ARI (Average Return Interval) values.
NON_ARI_COLUMNS = frozenset({"ID", "lat", "lon"})


def resolve_lake_twl_path(lake: str, data_dir: Path) -> Path:
    """Resolve a lake code to its twl xlsx path under data_dir."""
    return data_dir / LAKE_TWL_FILENAMES[lake]


def resolve_lake_avg_path(lake: str, data_dir: Path) -> Path:
    """Resolve a twl-lake code (e.g. "michigan") to its average-lake-level csv path.

    Looks up the shared avg-lake key first (TWL_LAKE_TO_AVG_LAKE), since michigan and
    huron share one avg csv (michiganhuron_avg.csv) despite having separate twl
    workbooks.
    """
    avg_lake = TWL_LAKE_TO_AVG_LAKE[lake]
    return data_dir / LAKE_AVG_FILENAMES[avg_lake]


def parse_scenario_sheet_name(sheet_name: str) -> str | None:
    """Parse a twl workbook sheet name into its bare scenario-grid suffix.

    Sheet names follow the `<label>-_<precip_delta>_<temp_delta>` convention (e.g.
    `baseline-_0_0`, `moderate_low-_10_5` -- note the label itself may contain
    underscores, so splitting is done on the first `-` only, not `_`). Returns the
    `_<precip_delta>_<temp_delta>` suffix (e.g. `_0_0`), suitable for
    hydropattern.scenario_grid's naming convention. Returns None if sheet_name does not
    match this convention (no `-` separator, blank label, or blank/missing suffix).
    """
    if "-" not in sheet_name:
        return None
    label, _, suffix = sheet_name.partition("-")
    if not label or not suffix:
        return None
    return suffix


@lru_cache(maxsize=None)
def load_lake_sheets(path: Path) -> dict[str, pd.DataFrame]:
    """Read every sheet of a `<lake>_twl.xlsx` workbook into {sheet_name: DataFrame}.

    Cached per path: twl workbooks are large (~20k save-point rows), and a batch run
    commonly needs the same lake's data more than once, so re-reading per call would
    be wasteful.
    """
    return pd.read_excel(path, sheet_name=None, engine="calamine")


@lru_cache(maxsize=None)
def read_avg_scenario_means(lake: str, data_dir: Path) -> dict[str, float]:
    """Read a lake's average-lake-level csv and return each scenario's mean level.

    Returns {scenario_suffix: mean}, one entry per non-"time" column (e.g.
    {"_0_0": 183.3, "_5_7": 182.9, ...}). The mean is taken over the *entire* synthetic
    scenario record (~12,360 monthly rows spanning 1970-2999), not a calendar
    sub-period -- see docs/adr/0001-row-shift-extrapolation-for-out-of-hull-
    scenarios.md, which uses this as AVG(l, s) in its shift-extrapolation math.

    Cached per (lake, data_dir): the avg csvs are ~12,360 rows x 17 columns, and a
    fill run needs the same lake's means only once but may be looked up from more than
    one place (e.g. once per out-of-hull target scenario).
    """
    path = resolve_lake_avg_path(lake, data_dir)
    frame = pd.read_csv(path)
    return {column: float(frame[column].mean()) for column in frame.columns if column != "time"}
