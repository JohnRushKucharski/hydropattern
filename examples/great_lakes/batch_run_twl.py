# ruff: noqa
"""Batch-generate twl (total water level) response-surface plots for Great Lakes resources.

Reads a resources.xlsx template (config sheet + resources sheet) and, for each row of the
resources sheet, evaluates a magnitude-only characteristic against a water-level/ARI
(Average Return Interval) frequency curve -- one curve per save point, per scenario sheet,
in the corresponding lake's `<lake>_twl.xlsx` data file (see
examples/great_lakes/data/clean/).

Unlike batch_run_avg.py, this script does NOT generate a hydropattern .toml or invoke
hydropattern.cli: there is no timeseries here (each scenario sheet holds one static
water-level-vs-ARI curve per save point), so hydropattern's Component/Characteristic
evaluation engine -- which operates on per-timestep pandas series -- does not apply. Instead,
each resource row's magnitude threshold is resolved directly against its save point's curve
via linear interpolation across ARI values, and only a response-surface grid csv + plot png
are produced per resource (no raw results, no summary xlsx, no .toml).

This is a one-off example-tooling script, not part of the hydropattern package, so it is
excluded from linting (see the `# ruff: noqa` above).
"""

import warnings
from dataclasses import dataclass
from math import exp, isnan
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from climate_canvas.plots_utilities import plot_response_surface  # type: ignore[import-untyped]

from hydropattern.cli import _resolve_color_map, write_grid_csv
from hydropattern.parsers import MetricMode
from hydropattern.scenario_grid import build_grid, require_scenario_grid

# common_twl.py is a sibling module within this package (examples/great_lakes/, not
# part of the hydropattern package). The try/except supports both normal package import
# (e.g. `from examples.great_lakes import batch_run_twl`, used by tests) and running
# this file directly as a script (`python batch_run_twl.py`), where there is no parent
# package and Python instead auto-adds this file's own directory to sys.path.
try:
    from . import common_twl
except ImportError:
    import common_twl  # type: ignore[import-not-found,no-redef]

LAKE_TWL_FILENAMES = common_twl.LAKE_TWL_FILENAMES
resolve_lake_twl_path = common_twl.resolve_lake_twl_path
parse_scenario_sheet_name = common_twl.parse_scenario_sheet_name
_NON_ARI_COLUMNS = common_twl.NON_ARI_COLUMNS
_load_lake_sheets = common_twl.load_lake_sheets


def _is_blank(value: Any) -> bool:
    """True for None, NaN, empty string, or whitespace-only string (a blank cell)."""
    if value is None:
        return True
    if isinstance(value, float) and isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _to_bool(value: Any, default: bool) -> bool:
    """Coerce a spreadsheet cell to bool; blank -> default."""
    if _is_blank(value):
        return default
    return bool(value)


def _require_str(row: dict[str, Any], field: str, errors: list[str]) -> str:
    value = row.get(field)
    if _is_blank(value):
        errors.append(f"{field!r} is required but missing or blank.")
        return ""
    return str(value).strip()


def interpolate_ari(levels: Sequence[float], aris: Sequence[float], threshold: float) -> float:
    """Linearly interpolate the ARI (Average Return Interval) at a water-level threshold.

    levels/aris are one save point's water-level-vs-ARI curve (levels ascending with
    ARI, per the twl data files -- one value per ARI column D:P). Interpolates linearly
    across ARI (per the fixed-space convention used throughout this script), i.e.
    ARI is treated as linear in water level between the two bracketing curve points.

    If threshold falls outside the curve's [min, max] water-level range, clamps to the
    nearest end ARI (0.1 or 1000-yr) and emits a UserWarning rather than extrapolating
    or failing -- thresholds a bit beyond the modeled range are still usable, just less
    precise.
    """
    if len(levels) != len(aris):
        raise ValueError(
            f"levels and aris must be the same length; got {len(levels)} and {len(aris)}."
        )
    if len(levels) < 2:
        raise ValueError("levels/aris must contain at least two points to interpolate.")

    if threshold < levels[0]:
        warnings.warn(
            f"Threshold {threshold!r} is below curve minimum {levels[0]!r}; "
            f"clamping to ARI={aris[0]!r}."
        )
        return float(aris[0])
    if threshold > levels[-1]:
        warnings.warn(
            f"Threshold {threshold!r} is above curve maximum {levels[-1]!r}; "
            f"clamping to ARI={aris[-1]!r}."
        )
        return float(aris[-1])
    return float(np.interp(threshold, levels, aris))


def interpolate_level(aris: Sequence[float], levels: Sequence[float], target_ari: float) -> float:
    """Linearly interpolate the water level at a target ARI (reverse of interpolate_ari).

    aris/levels are one save point's water-level-vs-ARI curve (levels ascending with
    ARI, per the twl data files -- one value per ARI column D:P). Interpolates linearly
    across ARI (per the fixed-space convention used throughout this script), i.e. water
    level is treated as linear in ARI between the two bracketing curve points.

    Used to compute equivalent elevation: given a baseline-scenario ARI, find the water
    level under a different scenario's curve that corresponds to that same ARI.

    If target_ari falls outside the curve's [min, max] ARI range, clamps to the nearest
    end level (the level at ARI=0.1 or ARI=1000) and emits a UserWarning rather than
    extrapolating or failing -- ARIs a bit beyond the modeled range are still usable,
    just less precise.
    """
    if len(aris) != len(levels):
        raise ValueError(
            f"aris and levels must be the same length; got {len(aris)} and {len(levels)}."
        )
    if len(aris) < 2:
        raise ValueError("aris/levels must contain at least two points to interpolate.")

    if target_ari < aris[0]:
        warnings.warn(
            f"Target ARI {target_ari!r} is below curve minimum {aris[0]!r}; "
            f"clamping to level={levels[0]!r}."
        )
        return float(levels[0])
    if target_ari > aris[-1]:
        warnings.warn(
            f"Target ARI {target_ari!r} is above curve maximum {aris[-1]!r}; "
            f"clamping to level={levels[-1]!r}."
        )
        return float(levels[-1])
    return float(np.interp(target_ari, aris, levels))


# Magnitude operators supported here. '=' and '!=' are deliberately excluded: against a
# continuous exceedance curve they have no sensible probability (always ~0 or ~1), unlike
# hydropattern's full operator set used for discrete per-timestep comparisons.
_VALID_OPERATORS = frozenset({">", ">=", "<", "<="})


def exceedance_probability(
    levels: Sequence[float], aris: Sequence[float], threshold: float, operator: str
) -> float:
    """Probability that the magnitude condition (`value <operator> threshold`) holds.

    Interpolates the ARI at threshold (see interpolate_ari) and converts to an annual
    exceedance probability via the Poisson relation `p = 1 - exp(-1/ARI)` (ARI is treated
    as the mean number of years between independent exceedance events, i.e. `1/ARI` is the
    mean annual exceedance rate). This stays correctly bounded in (0, 1) for any ARI > 0,
    unlike the naive `p = 1/ARI` approximation, which exceeds 1 whenever ARI < 1 year --
    this dataset's ARI columns go as low as 0.1 years, so the naive approximation is not
    safe to use here. For large ARI the two formulas converge (`1 - exp(-x) ~= x` for
    small x), so this only meaningfully changes results for ARI below roughly 10 years.
    For '>' / '>=' this probability *is* the exceedance probability; for '<' / '<=' it's
    the complement (1 - p), since the condition holds everywhere except in the exceedance
    tail.
    """
    if operator not in _VALID_OPERATORS:
        raise ValueError(
            f"Unsupported comparison operator {operator!r}; must be one of "
            f"{sorted(_VALID_OPERATORS)} ('=' and '!=' are not supported against a "
            "continuous exceedance curve)."
        )
    ari = interpolate_ari(levels, aris, threshold)
    p_exceed = 1.0 - exp(-1.0 / ari)
    if operator in (">", ">="):
        return p_exceed
    return 1.0 - p_exceed


def compute_metric(p_exceed: float, success_pattern: bool, mode: str) -> float:
    """Compute the configured summary metric from a magnitude condition's probability.

    Mirrors hydropattern.formatters.compute_metric_series's semantics exactly, just
    starting from a single probability instead of a per-timestep portion series:
        - portion:        p_exceed if success_pattern else (1 - p_exceed).
        - percentage:      portion * 100.
        - return_period:   1 / portion; NaN (undefined) when portion <= 0.
    """
    portion = p_exceed if success_pattern else (1.0 - p_exceed)
    if mode == "portion":
        return portion
    if mode == "percentage":
        return portion * 100.0
    if mode == "return_period":
        return float("nan") if portion <= 0 else 1.0 / portion
    raise ValueError(
        f"Unsupported metric mode {mode!r}; must be one of "
        "'portion', 'percentage', 'return_period'."
    )


def select_save_point(
    save_points: pd.DataFrame,
    save_point_id: Any = None,
    lat: float | None = None,
    lon: float | None = None,
) -> pd.Series:
    """Select one save-point row from a twl sheet's DataFrame (columns ID, lat, lon, ...).

    save_point_id, if given, always wins (even when lat/lon are also given). Otherwise
    lat and lon must both be given, and the nearest save point is picked by simple
    euclidean distance on (lat, lon) degrees -- good enough for picking the closest
    modeled point, not a navigational distance.

    Raises ValueError if neither save_point_id nor a complete (lat, lon) pair is given,
    or if save_point_id does not match any row.
    """
    if save_point_id is not None:
        matches = save_points[save_points["ID"] == save_point_id]
        if matches.empty:
            raise ValueError(f"No save point found with ID {save_point_id!r}.")
        return matches.iloc[0]

    if lat is None or lon is None:
        raise ValueError(
            "Must provide either save_point_id or both lat and lon to select a save point."
        )
    # Plain pandas arithmetic (rather than np.hypot) keeps this a Series, so .idxmin()
    # is available and the return type stays a Series row (not a bare numpy scalar).
    distances = ((save_points["lat"] - lat) ** 2 + (save_points["lon"] - lon) ** 2) ** 0.5
    # A single integer position always selects one row (a Series), never a DataFrame,
    # but pandas-stubs' .loc/.iloc-by-label overloads can't express that -- looking up
    # the plain positional argmin sidesteps the ambiguity.
    nearest_position = int(np.asarray(distances).argmin())
    return save_points.iloc[nearest_position]


# ---- resources sheet: row parsing ------------------------------------------------------

_DEFAULT_COMPONENT_NAME = "twl"


class RowValidationError(ValueError):
    """Raised when a resources-sheet row fails validation.

    Carries all validation problems found for the row (not just the first), so callers
    can report a complete picture per row.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class ResourceSpec:
    """Pure-data specification for one resources-sheet row.

    One ResourceSpec = one save-point magnitude threshold = one scenario-grid metric
    per scenario = one generated grid csv + plot png.
    """

    resource_name: str
    component_name: str
    lake: str
    magnitude_operator: str
    magnitude_value: float
    save_point_id: Any = None
    lat: float | None = None
    lon: float | None = None
    success_pattern: bool = False
    # Climate-canvas plot reference line (colormap center), in the config's metric_mode
    # units (portion 0-1 / percentage 0-100 / return_period years) -- NOT a water level,
    # unlike magnitude_value. See docs/user/reference.md's climate-canvas `threshold`.
    threshold: float | None = None

    @property
    def qualified_name(self) -> str:
        """resource_name+component_name, used as the prefix of every generated output
        filename. Needed because multiple rows commonly share one output folder (flat
        mode, the default), so component_name alone would collide across resources."""
        return f"{self.resource_name}_{self.component_name}"


def _validate_lake(lake: str, errors: list[str]) -> None:
    if lake and lake not in LAKE_TWL_FILENAMES:
        errors.append(
            f"Unknown lake {lake!r}; must be one of {sorted(LAKE_TWL_FILENAMES)}."
        )


def _parse_magnitude_operator_value(
    row: dict[str, Any], errors: list[str]
) -> tuple[str | None, float | None]:
    operator: str | None = _require_str(row, "magnitude_operator", errors)
    value = row.get("magnitude_value")
    if value is None or _is_blank(value):
        errors.append("'magnitude_value' is required but missing or blank.")
        value = None
    else:
        value = float(value)
    if operator and operator not in _VALID_OPERATORS:
        errors.append(
            f"magnitude_operator {operator!r} is not a valid comparison symbol; "
            f"must be one of {sorted(_VALID_OPERATORS)}."
        )
        operator = None
    return (operator or None, value)


def _parse_save_point_selector(
    row: dict[str, Any], errors: list[str]
) -> tuple[Any, float | None, float | None]:
    save_point_id = row.get("save_point_id")
    save_point_id = None if _is_blank(save_point_id) else save_point_id
    lat = row.get("lat")
    lat = None if lat is None or _is_blank(lat) else float(lat)
    lon = row.get("lon")
    lon = None if lon is None or _is_blank(lon) else float(lon)

    if save_point_id is None and (lat is None or lon is None):
        errors.append(
            "Row must provide either 'save_point_id', or both 'lat' and 'lon', to "
            "select a save point."
        )
    elif save_point_id is None and (lat is None) != (lon is None):
        errors.append("Row provides only one of 'lat'/'lon'; both are required together.")
    return (save_point_id, lat, lon)


def _parse_threshold(row: dict[str, Any]) -> float | None:
    value = row.get("threshold")
    if value is None or _is_blank(value):
        return None
    return float(value)


def parse_resource_row(row: dict[str, Any]) -> ResourceSpec:
    """Validate and parse one resources-sheet row into a ResourceSpec.

    Raises RowValidationError (carrying all problems found, not just the first) if the
    row is invalid.
    """
    errors: list[str] = []

    resource_name = _require_str(row, "resource_name", errors)
    lake = _require_str(row, "lake", errors)
    _validate_lake(lake, errors)

    component_name = row.get("component_name")
    component_name = (
        _DEFAULT_COMPONENT_NAME if _is_blank(component_name) else str(component_name).strip()
    )

    magnitude_operator, magnitude_value = _parse_magnitude_operator_value(row, errors)
    save_point_id, lat, lon = _parse_save_point_selector(row, errors)
    success_pattern = _to_bool(row.get("success_pattern"), default=False)
    threshold = _parse_threshold(row)

    if errors:
        raise RowValidationError(errors)

    return ResourceSpec(
        resource_name=resource_name,
        component_name=component_name,
        lake=lake,
        magnitude_operator=magnitude_operator,  # type: ignore[arg-type]
        magnitude_value=magnitude_value,  # type: ignore[arg-type]
        save_point_id=save_point_id,
        lat=lat,
        lon=lon,
        success_pattern=success_pattern,
        threshold=threshold,
    )


# ---- config sheet -----------------------------------------------------------------

_VALID_SUBDIRECTORY_STRUCTURES = ("flat", "resource", "row")
_VALID_METRIC_MODES = frozenset({"portion", "percentage", "return_period"})


@dataclass(frozen=True)
class BatchConfig:
    """Pure-data specification for the config-sheet options shared across all rows."""

    output_directory: str = ""  # base output directory; required, no sensible default
    subdirectory_structure: str = "flat"  # "flat" | "resource" | "row"
    metric_mode: str = "return_period"  # portion | percentage | return_period
    overwrite: bool = False
    plot_interpolate: bool = True
    plot_color_map: str = "RdBu"
    plot_color_map_ticks: tuple[float, ...] | None = None
    # If true, also compute+write a second grid csv + plot png per row: the water level
    # under every scenario equivalent to the baseline (_0_0) scenario's ARI at this
    # row's magnitude_value. See compute_equivalent_elevation_metrics.
    compute_equivalent_elevation: bool = False
    # Forwarded as climate-canvas's plot_response_surface(fillin=...) on every plot_fn
    # call this script makes (primary + equivalent-elevation). Estimates missing (NaN)
    # grid cells via Delaunay triangulation; see climate-canvas's --fillin option.
    fillin: bool = False


def resolve_output_folder(resource: ResourceSpec, config: BatchConfig) -> Path:
    """Compute the output folder for one resource row.

    Pure/stateless: depends only on this row's resource_name/component_name and the
    config's output_directory/subdirectory_structure. Does NOT detect collisions
    between rows -- that's run_batch's job.

    subdirectory_structure:
        "flat"     -> output_directory/
        "resource" -> output_directory/<resource_name>/
        "row"      -> output_directory/<resource_name>/<component_name>/
    """
    if config.subdirectory_structure not in _VALID_SUBDIRECTORY_STRUCTURES:
        raise ValueError(
            f"Invalid subdirectory_structure {config.subdirectory_structure!r}; "
            f"must be one of {_VALID_SUBDIRECTORY_STRUCTURES!r}."
        )
    base = Path(config.output_directory)
    if config.subdirectory_structure == "flat":
        return base
    if config.subdirectory_structure == "resource":
        return base / resource.resource_name
    return base / resource.resource_name / resource.component_name


class SheetValidationError(ValueError):
    """Raised when a template sheet (config or resources) fails header/structural
    validation -- i.e. problems with the sheet's shape itself, not with one row's data
    (see RowValidationError for that).

    Carries all validation problems found (not just the first).
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


_RESOURCE_REQUIRED_COLUMNS = frozenset(
    {"resource_name", "lake", "magnitude_operator", "magnitude_value"}
)
_RESOURCE_OPTIONAL_COLUMNS = frozenset({
    "component_name", "save_point_id", "lat", "lon", "success_pattern", "threshold",
})
_RESOURCE_ALL_COLUMNS = _RESOURCE_REQUIRED_COLUMNS | _RESOURCE_OPTIONAL_COLUMNS


def read_resources_sheet(path: Path, sheet_name: str = "resources") -> list[dict[str, Any]]:
    """Read the resources sheet into a list of raw row dicts (column name -> cell value).

    One dict per row. NaN/blank cells become None (not float('nan')) so downstream
    _is_blank()-based checks (e.g. in parse_resource_row) behave correctly.

    Only validates the sheet's *shape* (required columns present, no unrecognized
    columns -- likely a typo). Per-row data validation is parse_resource_row's job.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, engine="calamine")
    columns = set(df.columns)
    errors: list[str] = []
    missing = _RESOURCE_REQUIRED_COLUMNS - columns
    if missing:
        errors.append(
            f"Resources sheet is missing required column(s): {sorted(missing)}."
        )
    unknown = columns - _RESOURCE_ALL_COLUMNS
    if unknown:
        errors.append(
            f"Resources sheet has unrecognized column(s): {sorted(unknown)}."
        )
    if errors:
        raise SheetValidationError(errors)

    records = df.to_dict(orient="records")
    # df.to_dict()'s keys are typed Hashable (pandas' general column-label type), but
    # they're always the sheet's str column names here, matching this function's
    # declared list[dict[str, Any]] return type.
    return [{str(k): (None if _is_blank(v) else v) for k, v in row.items()} for row in records]


_CONFIG_BOOL_KEYS = frozenset(
    {"overwrite", "plot_interpolate", "compute_equivalent_elevation", "fillin"}
)
_CONFIG_STR_KEYS = frozenset(
    {"metric_mode", "output_directory", "subdirectory_structure", "plot_color_map"}
)
_CONFIG_TICKS_KEY = "plot_color_map_ticks"
_CONFIG_KNOWN_KEYS = _CONFIG_BOOL_KEYS | _CONFIG_STR_KEYS | {_CONFIG_TICKS_KEY}


def _parse_color_map_ticks(raw: Any, errors: list[str]) -> tuple[float, ...] | None:
    if _is_blank(raw):
        return None
    try:
        return tuple(float(v.strip()) for v in str(raw).split(","))
    except ValueError:
        errors.append(
            f"Config sheet option 'plot_color_map_ticks' {raw!r} must be a "
            "comma-separated list of numbers."
        )
        return None


def read_config_sheet(path: Path, sheet_name: str = "config") -> BatchConfig:
    """Read the config sheet (key/value pairs: 'option' column, 'value' column) into a
    BatchConfig. Options left blank or absent fall back to BatchConfig's own defaults,
    except 'output_directory', which is required.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, engine="calamine")
    if not {"option", "value"} <= set(df.columns):
        raise SheetValidationError(
            [f"Config sheet must have 'option' and 'value' columns; got {list(df.columns)}."]
        )

    errors: list[str] = []
    values: dict[str, Any] = {}
    for _, row in df.iterrows():
        key = row["option"]
        if _is_blank(key):
            continue
        key = str(key).strip()
        if key in values:
            errors.append(f"Config sheet has duplicate option {key!r}.")
            continue
        value = row["value"]
        values[key] = None if _is_blank(value) else value

    unknown = set(values) - _CONFIG_KNOWN_KEYS
    if unknown:
        errors.append(f"Config sheet has unrecognized option(s): {sorted(unknown)}.")

    output_directory = values.get("output_directory")
    if _is_blank(output_directory):
        errors.append(
            "Config sheet option 'output_directory' is required but missing or blank."
        )

    subdirectory_structure = values.get("subdirectory_structure")
    if not _is_blank(subdirectory_structure) \
            and str(subdirectory_structure).strip() not in _VALID_SUBDIRECTORY_STRUCTURES:
        errors.append(
            f"Config sheet option 'subdirectory_structure' {subdirectory_structure!r} "
            f"must be one of {_VALID_SUBDIRECTORY_STRUCTURES!r}."
        )

    metric_mode = values.get("metric_mode")
    if not _is_blank(metric_mode) and str(metric_mode).strip() not in _VALID_METRIC_MODES:
        errors.append(
            f"Config sheet option 'metric_mode' {metric_mode!r} "
            f"must be one of {sorted(_VALID_METRIC_MODES)}."
        )

    ticks = _parse_color_map_ticks(values.get(_CONFIG_TICKS_KEY), errors)

    if errors:
        raise SheetValidationError(errors)

    defaults = BatchConfig()
    kwargs: dict[str, Any] = {"output_directory": str(output_directory).strip()}
    if not _is_blank(metric_mode):
        kwargs["metric_mode"] = str(metric_mode).strip()
    if not _is_blank(values.get("overwrite")):
        kwargs["overwrite"] = _to_bool(values.get("overwrite"), defaults.overwrite)
    if not _is_blank(subdirectory_structure):
        kwargs["subdirectory_structure"] = str(subdirectory_structure).strip()
    if not _is_blank(values.get("plot_interpolate")):
        kwargs["plot_interpolate"] = _to_bool(
            values.get("plot_interpolate"), defaults.plot_interpolate
        )
    if not _is_blank(values.get("plot_color_map")):
        kwargs["plot_color_map"] = str(values["plot_color_map"]).strip()
    if ticks is not None:
        kwargs["plot_color_map_ticks"] = ticks
    if not _is_blank(values.get("compute_equivalent_elevation")):
        kwargs["compute_equivalent_elevation"] = _to_bool(
            values.get("compute_equivalent_elevation"), defaults.compute_equivalent_elevation
        )
    if not _is_blank(values.get("fillin")):
        kwargs["fillin"] = _to_bool(values.get("fillin"), defaults.fillin)

    return BatchConfig(**kwargs)


# ---- per-resource grid + plot building --------------------------------------------


def compute_scenario_metrics(resource: ResourceSpec, twl_path: Path, metric_mode: str
                             ) -> dict[str, float]:
    """Compute one metric value per scenario sheet for one resource's save point.

    Reads every sheet of the lake's twl workbook, selects the resource's save point in
    each, interpolates its magnitude condition's exceedance probability, and converts
    to metric_mode's units. Keys are the bare `_<precip>_<temp>` scenario-grid suffix
    (see parse_scenario_sheet_name) -- sheets that don't match that naming convention
    are skipped.
    """
    sheets = _load_lake_sheets(twl_path)
    metric_values: dict[str, float] = {}
    for sheet_name, df in sheets.items():
        suffix = parse_scenario_sheet_name(sheet_name)
        if suffix is None:
            continue
        save_point = select_save_point(df, resource.save_point_id, resource.lat, resource.lon)
        ari_columns = [c for c in df.columns if c not in _NON_ARI_COLUMNS]
        aris = [float(c) for c in ari_columns]
        levels = [float(save_point[c]) for c in ari_columns]
        p_exceed = exceedance_probability(
            levels, aris, resource.magnitude_value, resource.magnitude_operator
        )
        metric_values[suffix] = compute_metric(p_exceed, resource.success_pattern, metric_mode)
    return metric_values


# The known/filled/extrapolated scenario-grid suffix for the (0% precip delta, 0C temp
# delta) baseline scenario -- always present in every lake's twl workbook (see
# CONTEXT.md's "Known scenario" entry).
_BASELINE_SCENARIO_SUFFIX = "_0_0"


def compute_equivalent_elevation_metrics(resource: ResourceSpec, twl_path: Path
                                          ) -> dict[str, float]:
    """Compute one equivalent-elevation value per scenario sheet for one resource.

    First finds the baseline (_0_0) scenario's ARI at resource.magnitude_value (via
    interpolate_ari), then, for every scenario sheet (including baseline itself), finds
    the water level at that same ARI under that scenario's curve (via interpolate_level)
    -- i.e. "what water level, under this scenario, is equally likely (same ARI) as
    resource.magnitude_value is under the baseline scenario". Keys are the bare
    `_<precip>_<temp>` scenario-grid suffix (see parse_scenario_sheet_name) -- sheets
    that don't match that naming convention are skipped.

    Raises ValueError if the lake's twl workbook has no baseline (_0_0) scenario sheet.
    """
    sheets = _load_lake_sheets(twl_path)

    def _save_point_ari_columns_levels(df: pd.DataFrame) -> tuple[list[float], list[float]]:
        save_point = select_save_point(df, resource.save_point_id, resource.lat, resource.lon)
        ari_columns = [c for c in df.columns if c not in _NON_ARI_COLUMNS]
        aris = [float(c) for c in ari_columns]
        levels = [float(save_point[c]) for c in ari_columns]
        return aris, levels

    baseline_aris: list[float] | None = None
    baseline_levels: list[float] | None = None
    for sheet_name, df in sheets.items():
        if parse_scenario_sheet_name(sheet_name) == _BASELINE_SCENARIO_SUFFIX:
            baseline_aris, baseline_levels = _save_point_ari_columns_levels(df)
            break
    if baseline_aris is None or baseline_levels is None:
        raise ValueError(
            f"No baseline ({_BASELINE_SCENARIO_SUFFIX!r}) scenario sheet found in "
            f"{twl_path}; cannot compute equivalent elevation."
        )
    baseline_ari = interpolate_ari(baseline_levels, baseline_aris, resource.magnitude_value)

    equivalent_elevations: dict[str, float] = {}
    for sheet_name, df in sheets.items():
        suffix = parse_scenario_sheet_name(sheet_name)
        if suffix is None:
            continue
        aris, levels = _save_point_ari_columns_levels(df)
        equivalent_elevations[suffix] = interpolate_level(aris, levels, baseline_ari)
    return equivalent_elevations


def build_resource_outputs(
    resource: ResourceSpec,
    data_dir: Path,
    config: BatchConfig,
    plot_fn: Callable[..., None] = plot_response_surface,
) -> tuple[Path, Path]:
    """Compute one resource's scenario-grid metrics and write its grid csv + plot png.

    If config.compute_equivalent_elevation is True, also computes and writes a second
    grid csv + plot png: the water level under every scenario equivalent (same ARI) to
    the baseline scenario's ARI at resource.magnitude_value (see
    compute_equivalent_elevation_metrics). This second plot's z-axis is labeled
    "Equivalent Elevation", its threshold is resource.magnitude_value (a water level,
    unlike the primary plot's metric-mode-units threshold), and it uses
    config.plot_color_map as given -- unlike the primary plot, no RdBu-direction
    auto-reversal or color_map_ticks are applied, since success_pattern/metric_mode
    semantics don't apply to a raw elevation value.

    plot_fn is injectable so tests can substitute a recording stub instead of the real
    climate_canvas plotting call (which opens a matplotlib figure).

    Raises FileExistsError if any target output file already exists and
    config.overwrite is False. Raises HydropatternError (via require_scenario_grid) if
    the lake's sheet names don't form a valid scenario grid.
    """
    twl_path = resolve_lake_twl_path(resource.lake, data_dir)
    metric_values = compute_scenario_metrics(resource, twl_path, config.metric_mode)
    scenario_names = list(metric_values.keys())
    require_scenario_grid(scenario_names)
    xs, ys, zs = build_grid(scenario_names, metric_values)

    output_folder = resolve_output_folder(resource, config)
    grid_path = output_folder / f"{resource.qualified_name}_grid.csv"
    plot_path = output_folder / f"{resource.qualified_name}_plot.png"
    elev_grid_path = output_folder / f"{resource.qualified_name}_equivalent_elevation_grid.csv"
    elev_plot_path = output_folder / f"{resource.qualified_name}_equivalent_elevation_plot.png"

    targets = [grid_path, plot_path]
    if config.compute_equivalent_elevation:
        targets += [elev_grid_path, elev_plot_path]
    if not config.overwrite:
        existing = [p for p in targets if p.exists()]
        if existing:
            raise FileExistsError(
                f"Output file(s) already exist and overwrite=False: "
                f"{', '.join(str(p) for p in existing)}."
            )

    output_folder.mkdir(parents=True, exist_ok=True)
    write_grid_csv(xs, ys, zs, grid_path)
    color_map = _resolve_color_map(
        config.plot_color_map, resource.success_pattern, MetricMode(config.metric_mode)
    )
    plot_fn(
        xs, ys, zs,
        interpolate=config.plot_interpolate,
        labels=("Precipitation Delta (%)", "Temperature Delta (C)", config.metric_mode),
        title=resource.qualified_name,
        save_path=plot_path,
        show=False,
        threshold=resource.threshold,
        color_map=color_map,
        color_map_ticks=config.plot_color_map_ticks,
        fillin=config.fillin,
    )

    if config.compute_equivalent_elevation:
        elev_values = compute_equivalent_elevation_metrics(resource, twl_path)
        elev_xs, elev_ys, elev_zs = build_grid(list(elev_values.keys()), elev_values)
        write_grid_csv(elev_xs, elev_ys, elev_zs, elev_grid_path)
        plot_fn(
            elev_xs, elev_ys, elev_zs,
            interpolate=config.plot_interpolate,
            labels=("Precipitation Delta (%)", "Temperature Delta (C)", "Equivalent Elevation"),
            title=f"{resource.qualified_name}_equivalent_elevation",
            save_path=elev_plot_path,
            show=False,
            threshold=resource.magnitude_value,
            color_map=config.plot_color_map,
            color_map_ticks=None,
            fillin=config.fillin,
        )

    return (grid_path, plot_path)


# ---- run_batch orchestration -----------------------------------------------------------

@dataclass(frozen=True)
class RowResult:
    """Outcome of processing one resources-sheet row."""

    row_index: int  # 1-based position in the resources sheet's data rows
    resource_name: str | None
    component_name: str | None
    status: str  # "succeeded" | "failed"
    message: str | None = None


@dataclass(frozen=True)
class BatchSummary:
    """End-of-run report for a full run_batch call."""

    results: list[RowResult]

    @property
    def succeeded(self) -> list[RowResult]:
        return [r for r in self.results if r.status == "succeeded"]

    @property
    def failed(self) -> list[RowResult]:
        return [r for r in self.results if r.status == "failed"]


def format_summary(summary: BatchSummary) -> str:
    """Human-readable end-of-run report: counts + one line per failed row."""
    lines = [
        f"{len(summary.succeeded)} succeeded, {len(summary.failed)} failed "
        f"out of {len(summary.results)} row(s)."
    ]
    for result in summary.failed:
        lines.append(
            f"  Row {result.row_index} ({result.resource_name!r}, "
            f"{result.component_name!r}): {result.message}"
        )
    return "\n".join(lines)


def _default_progress(row_index: int, total_rows: int, resource_name: str | None,
                      component_name: str | None, status: str) -> None:
    """Prints one line per row as it starts and finishes."""
    label = f"{resource_name or '?'}/{component_name or '?'}"
    print(f"[{row_index}/{total_rows}] {label}: {status}", flush=True)


def run_batch(
    resources_path: Path,
    data_dir: Path,
    config: BatchConfig,
    build_outputs: Callable[[ResourceSpec, Path, BatchConfig], tuple[Path, Path]]
        = build_resource_outputs,
    progress: Callable[[int, int, str | None, str | None, str], None] = _default_progress,
) -> BatchSummary:
    """Generate one scenario-grid csv + plot png per hydropattern resources-sheet row.

    Continues past a failing row (bad row data, duplicate output target, pre-existing
    output with overwrite=False, or a build_outputs error) rather than aborting the
    whole batch; every row's outcome is captured in the returned BatchSummary.

    build_outputs is injectable so tests can mock the actual grid/plot generation
    without touching the filesystem or matplotlib. progress is injectable for the same
    reason (and to let callers redirect/silence the real-time per-row status output).
    """
    raw_rows = read_resources_sheet(resources_path)
    total_rows = len(raw_rows)
    results: list[RowResult] = []
    seen_targets: set[tuple[Path, str]] = set()

    def _finish(row_index: int, resource_name: str | None, component_name: str | None,
               status: str, message: str | None = None) -> None:
        results.append(RowResult(row_index, resource_name, component_name, status, message))
        progress(row_index, total_rows, resource_name, component_name, status)

    for row_index, raw_row in enumerate(raw_rows, start=1):
        resource_name = raw_row.get("resource_name")
        component_name = raw_row.get("component_name")
        progress(row_index, total_rows, resource_name, component_name, "running")
        try:
            resource = parse_resource_row(raw_row)
        except RowValidationError as exc:
            _finish(row_index, resource_name, component_name, "failed", str(exc))
            continue

        output_folder = resolve_output_folder(resource, config)
        target = (output_folder, resource.qualified_name)
        if target in seen_targets:
            _finish(
                row_index, resource.resource_name, resource.component_name, "failed",
                f"Duplicate output target within this batch run: "
                f"{output_folder / f'{resource.qualified_name}_grid.csv'}",
            )
            continue
        seen_targets.add(target)

        try:
            build_outputs(resource, data_dir, config)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # continue-on-failure: any error in this row (bad data, a missing file, a
            # plotting error, ...) is reported and the batch moves on.
            _finish(row_index, resource.resource_name, resource.component_name,
                   "failed", str(exc))
            continue

        _finish(row_index, resource.resource_name, resource.component_name, "succeeded")

    return BatchSummary(results=results)


# ---- CLI entry point --------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: read a resources workbook, run every row, print a
    summary. Returns a process exit code (0 if every row succeeded, 1 otherwise).

    Usage:
        uv run python examples/great_lakes/batch_run_twl.py <resources.xlsx> <data_dir>
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "resources_path", type=Path,
        help="Path to the resources+config .xlsx workbook (see templates/template_twl.xlsx).",
    )
    parser.add_argument(
        "data_dir", type=Path,
        help="Directory containing the lake twl xlsx files (see LAKE_TWL_FILENAMES).",
    )
    args = parser.parse_args(argv)

    config = read_config_sheet(args.resources_path)
    summary = run_batch(args.resources_path, args.data_dir, config)
    print(format_summary(summary))
    return 0 if not summary.failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
