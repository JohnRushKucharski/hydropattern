# Glossary

These terms distinguish observations, characteristic assessments, and final
component classifications. They follow the project's domain terminology.
Some terms are needed to discuss planned reporting improvements; their
definition here does not mean those improvements are implemented.

## Observations and configuration

**Scenario** is one input-data column, evaluated independently against all
configured components. A file can contain one scenario or many.

**Timestep interval** is the spacing between successive observations. Cadence
describes its regularity, such as daily or monthly sampling. A timestep count
is not automatically a calendar length: months have different numbers of days.

**Component** is a configured flow pattern assessed using one or more
characteristics.

**Characteristic** is an aspect of the pattern: timing, magnitude, duration,
frequency, or rate of change.

**Characteristic parameters** define a characteristic's condition, such as
a magnitude threshold or duration bounds.

**Condition** is a criterion used to decide whether a characteristic is
satisfied.

**Water year** is a year-long interval starting at a configured calendar
boundary and labelled by its ending year. A partial water year contains only
part of that interval in the observed record.

**Calendar length** expresses an interval in calendar units, such as days
or months, rather than in a count of timesteps.

## Outcomes

**Characteristic outcome** is a characteristic's verdict at one timestep,
before its effect on the final component is assessed.

**Component outcome** is the final verdict at one timestep: success, failure,
or unknown. These verdicts describe configured conditions, not guaranteed
ecological benefit. A failure-pattern configuration describes conditions to
avoid, so its final known success means non-failure.

**Unknown outcome** means available information cannot determine success or
failure. It is neither one. Unavailable independent comparisons and duration
run-boundary uncertainty preserve unknowns. Unknown-aware frequency and annual
evaluation, plus reporting summaries and explicit coverage, remain planned.

**Known-outcome coverage** is the fraction of recorded timesteps whose outcome
is known, for the column under discussion. Coverage differs from how many known
outcomes are successful and from whether the water year is complete.
Explicit coverage reporting is planned, not implemented in this step.

**Summary metric** summarizes the final component outcomes for one scenario.
**Characteristic summary** summarizes a characteristic's diagnostic column
instead. The first-result example has fully known outcomes; rules for
summarizing unknowns will be documented with their reporting implementation.

## Duration and rate of change

**Qualifying timestep** is an observation interval satisfying the conditions
being assessed. For duration or frequency, these are the combined conditions
of the preceding characteristics.

**Qualifying run** is a maximal uninterrupted sequence of qualifying timesteps.
Its **qualifying run length** counts those timesteps.

**Duration threshold** is a timestep count compared with the qualifying run
length. **Duration bounds** define an inclusive acceptable range of such
counts. The duration characteristic assesses the whole qualifying run.

**Look-back interval** separates the current timestep from the earlier
timestep used in a rate-of-change comparison. It does not require intervening
observations to qualify.

## Frequency

**Evaluation interval** is the span considered when assessing a condition;
a frequency window, qualifying run, or look-back interval is a specific kind.

**Frequency window** is a forward interval over which qualifying timesteps
or qualifying water years are counted.

**Frequency window anchor** is the qualifying timestep or water year where a
window starts. If the count condition accepts zero, any timestep or water year
can start one. **Frequency window length** is its configured maximum number of
timesteps or water years; the observed record can truncate it.

**Frequency count** counts qualifying timesteps or qualifying water years,
not component events.

**Exclusive windows** is the optional frequency setting that suppresses later
anchors inside a qualifying window. Without it, the results of qualifying
overlapping windows are combined. Each part of nested frequency sets it
independently.
See the [frequency reference](../reference/characteristics/frequency.md) for
exact existing rules.

**Un-nested frequency** assesses a single frequency condition across the
record's timesteps; its windows can cross water-year boundaries.

**Intra-annual pattern** is the part of nested frequency that assesses
timesteps within complete water years and identifies qualifying water years.
**Interannual pattern** assesses those qualifying water years across years.

**Qualifying water year** meets its intra-annual condition.

**Trial** generalizes over the units counted by frequency: timesteps or water
years. It does not imply independence or random sampling. Guides name the
concrete unit wherever possible.

## Component events and scenario grids

**Component event** is a maximal uninterrupted occurrence of final component
success. It is also called a component-success run when discussing consecutive
successful timesteps. It is not necessarily a distinct physical hydrologic
occurrence such as a flood.

**Component success period** is the time interval occupied by that component
event. **Component-success run length** counts its timesteps. It can differ
from the qualifying run length assessed by duration.

**Scenario grid** is a set of scenarios whose names encode two numeric axes,
for example `_0_1.5`. It supports a response-surface plot.

**Precipitation delta** is the first number in a scenario-grid name, expressed
in percent. **Temperature delta** is the second number, expressed in degrees
Celsius. In `_0_1.5`, these are 0% and 1.5 degrees Celsius respectively.
