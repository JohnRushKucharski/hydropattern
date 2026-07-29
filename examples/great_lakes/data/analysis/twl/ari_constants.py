# ruff: noqa
"""Shared constants for the TWL response-surface plotting scripts
(plot_center_save_points.py, plot_filled_and_extrapolated.py) and, for the
ARI-filename formatting helper, the avg-level script (plot_avg_levels.py).
"""

# All 13 ARI columns present in the TWL xlsx workbooks (data/clean, data/filled,
# data/extrapolated) -- see the "ID"/lat/lon + these columns in any sheet.
TWL_ARIS: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
                                100.0, 200.0, 500.0, 1000.0)


def format_ari(ari: float) -> str:
    """Format an ARI value for use in a filename/title, e.g. 0.1 -> "0.1",
    1.0 -> "1", 1000.0 -> "1000" (no superfluous ".0" on whole numbers).
    """
    return f"{ari:g}"
