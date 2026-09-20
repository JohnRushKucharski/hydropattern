'''
Public API for pattern evaluation: characteristic-fx factories, Component/
Result orchestration (hydropattern.patterns.core), and water-year utilities
(hydropattern.patterns.water_year). Re-exported here so existing callers can
keep using `from hydropattern.patterns import ...`.
'''
from hydropattern.patterns.core import (
    Characteristic,
    CharacteristicFx,
    CharacteristicType,
    Component,
    Result,
    comparison_fx,
    duration_fx,
    eq,
    evaluate_component,
    evaluate_components,
    frequency_fx,
    ge,
    gt,
    is_dowy_timeseries,
    le,
    lt,
    magnitude_fx,
    mark_events,
    moving_average,
    ne,
    nested_frequency_intra_annual_fx,
    nested_frequency_interannual_fx,
    rate_of_change_fx,
    sliding_window_count,
    timing_fx,
    validate_order,
    validate_timeseries,
)
from hydropattern.patterns.water_year import (
    identify_full_water_years,
    or_reduce_per_water_year,
    water_year_probability_ratio,
    windowed_count_per_water_year,
)
