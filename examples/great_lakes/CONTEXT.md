# Great Lakes example

Example tooling (not part of the hydropattern package) for batch-generating and running
hydropattern `.toml` configs against Great Lakes lake-level timeseries, one `.toml`/run
per row of a spreadsheet (`batch_run_avg.py`) — plus a second, standalone batch tool
(`batch_run_twl.py`) that evaluates total-water-level frequency curves instead.

## Language

**Resource**:
A named grouping label for one or more rows in the resources sheet, used only to name
output subfolders when the config sheet's subdirectory structure is `resource` or `row`.
Sharing a resource name across rows has no effect on `.toml` generation — each row still
produces its own independent `.toml` and hydropattern run.
_Avoid_: Project (too generic — a resource is the Great-Lakes-specific unit of grouping).

**Resources sheet**:
The `templates/template_avg.xlsx` sheet where each row fully specifies one hydropattern
component (name, lake, characteristics, plot threshold) to generate into its own `.toml`
file and run independently.

**Config sheet**:
The `templates/template_avg.xlsx` sheet holding options shared across every row of the
resources sheet (output directory, subdirectory structure, overwrite, first day of water
year, metric mode, excel/plot output toggles).

**Subdirectory structure**:
The config sheet option controlling where each row's generated `.toml` and hydropattern
output land, relative to the config's base output directory:
- `"flat"`: directly in the base output directory.
- `"resource"`: one subfolder per resource name.
- `"row"`: one subfolder per resource name, nested with one subfolder per component name.

**Save point**:
A single row in a `batch_run_twl.py` lake workbook sheet — a fixed lat/lon location with an
`ID`, whose water-level-vs-ARI curve is looked up by a resources-sheet row via
`save_point_id` (exact match, wins if given) or nearest `lat`/`lon` (Euclidean distance,
fallback).
_Avoid_: Location, site (too generic — save point is the twl-data-specific unit each
sheet row represents).

**ARI (Average Return Interval)**:
The twl workbook's column headers (in years) — the inverse of exceedance probability for
a given water level at a save point. `batch_run_twl.py` linearly interpolates in ARI-space
against a save point's (water level, ARI) pairs to find the ARI at a resource row's
magnitude threshold. Out-of-range thresholds are clamped to the nearest end ARI (`0.1` or
`1000`) with a warning, never extrapolated.

**Equivalent elevation**:
An optional (resources-sheet `equivalent_elevation` column, blank by default) second
scenario-grid metric alongside a resource row's primary exceedance-probability/ARI
metric. The column is blank (skip), `"baseline_magnitude"` (case-insensitive; use the
row's `magnitude_value`), or a number (override `magnitude_value` for this analysis
only — the primary metric always uses `magnitude_value`). Computed by first finding
the baseline (`_0_0`) scenario's ARI at the resolved baseline-ARI lookup value, then
interpolating the water level at that same ARI under every other scenario's own curve
— the reverse direction of the ARI lookup used for the primary metric. Answers "what
water level is equally likely, under this scenario, as the lookup value is under the
baseline scenario?" Written to its own grid csv + plot png
(`..._equivalent_elevation_grid.csv` / `_equivalent_elevation_plot.png`), with plot
z-axis label `"Equivalent Elevation"` and plot threshold equal to the resolved
baseline-ARI lookup value itself (a water level) rather than a metric-mode-units value.
_Avoid_: Equivalent ARI, equivalent magnitude (the output is a water level, not an ARI
or the metric-mode value the primary grid produces).

**Known scenario**:
One of the 5 (of 17 total) precip/temp scenarios a `<lake>_twl.xlsx` workbook already has
a sheet for, per save point (`baseline-_0_0`, `nearterm-_5_1.5`, `moderate_low-_10_5`,
`extreme_low-_20_5`, `extreme_high-_0_7`) — identical set across all 4 lake twl workbooks.
_Avoid_: Source scenario, observed scenario.

**Target scenario**:
One of the 12 remaining precip/temp scenarios a `<lake>_twl.xlsx` workbook has no sheet
for, whose water-level-vs-ARI curve is being estimated per save point. Splits into
**in-hull** (7 scenarios, within the 5 known scenarios' convex hull — estimated via
Delaunay-linear interpolation, sheet name `filled-_<precip>_<temp>`) and **out-of-hull**
(5 scenarios, outside it — Delaunay-linear can't extrapolate there; estimated instead via
**row-shift extrapolation**, sheet name `extrapolated-_<precip>_<temp>`; see
`docs/adr/0001-row-shift-extrapolation-for-out-of-hull-scenarios.md`).
_Avoid_: Missing scenario, estimated scenario.

**Warming row (dT row)**:
The set of scenarios (known, filled, or target) that share the same temp_delta, varying
only by precip_delta — e.g. the dT=7 row is `_0_7, _5_7, _10_7, _15_7, _20_7`. Row-shift
extrapolation always shifts within a single warming row; it never crosses rows.

**Anchor scenario**:
For an out-of-hull target scenario, the known-or-already-filled scenario on the *same
warming row* nearest to it by precip_delta distance. Its full water-level-vs-ARI sheet
is the starting point that row-shift extrapolation shifts by a lake-average delta to
estimate the target. Requires the in-hull interpolation stage to run first, since some
anchors (e.g. `_10_3` anchoring `_15_3`) are themselves Delaunay-filled scenarios, not
known ones.
_Avoid_: Base scenario, reference scenario (reference scenario collides with the
`reference_timeseries` vocabulary used elsewhere in hydropattern core).

**Average lake level (AVG(l,s))**:
The mean of the full synthetic scenario record (all ~12,360 monthly rows spanning
1970-2999) in `data/clean/<avg-lake>_avg.csv` for lake `l`'s scenario `s` column — a
single scalar per lake+scenario, independent of save point or ARI. `michigan` and
`huron` (separate `<lake>_twl.xlsx` workbooks) both read the same `michiganhuron_avg.csv`
column, since Michigan-Huron is one hydraulically-connected lake with one water level.
_Avoid_: Mean lake level (AVG is the established short form used in row-shift
extrapolation's math; keep both docs and code consistent).

## Example dialogue

> **Dev**: Two rows in the resources sheet share the name "Duluth Harbor" but have
> different component names. Do they get merged into one `.toml`?
> **Domain expert**: No — every row is independent. "Duluth Harbor" here is just a
> **resource** label; it only affects output folder naming when the **subdirectory
> structure** is `resource` or `row`. Each row still gets its own `.toml` and its own
> hydropattern run.
