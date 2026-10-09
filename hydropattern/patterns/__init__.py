'''
Public API for pattern evaluation: Component/Result orchestration and shared
utilities (hydropattern.patterns.core), characteristic-fx factories and
comparison-fx helpers (hydropattern.patterns.characteristics), and water-year
utilities (hydropattern.patterns.water_year). Re-exported here so existing
callers can keep using `from hydropattern.patterns import ...`.
'''
from hydropattern.patterns.core import (
    Characteristic,
    CharacteristicFx,
    CharacteristicType,
    Component,
    EventCountBounds,
    EventRateBounds,
    Result,
    count_event_bounds,
    count_events,
    event_rate,
    evaluate_component,
    evaluate_components,
    find_exceeding_events,
    find_runs,
    is_dowy_timeseries,
    mark_windows,
    moving_average,
    sliding_window_count,
    validate_order,
    validate_timeseries,
)
from hydropattern.patterns.characteristics import (
    comparison_fx,
    duration_fx,
    eq,
    frequency_fx,
    ge,
    gt,
    le,
    lt,
    magnitude_fx,
    ne,
    nested_frequency_intra_annual_fx,
    nested_frequency_interannual_fx,
    rate_of_change_fx,
    timing_fx,
)
from hydropattern.patterns.water_year import (
    identify_full_water_years,
    infer_first_day_of_water_year,
    or_reduce_per_water_year,
    record_length_years,
    water_year_label,
    water_year_exposure,
    water_year_exposure_by_year,
    water_year_probability_ratio,
    windowed_count_per_water_year,
)
