# Scientific foundations

Streamflow varies in amount, timing, duration, frequency, and rate of change.
These parts of a flow regime describe different aspects of how water moves
through a river system over time. Ecological processes can depend on both the
usual pattern of flow and departures from it.

hydropattern lets you express selected parts of a flow pattern as conditions
and assess observations against them. A configured component is a software
classification: it says whether the observations meet the conditions you
specified. It is not, by itself, a measure of ecosystem health or proof of
ecological benefit.

The terminology used on this site follows the project's
[glossary](glossary.md). For broader scientific background, see
[Poff et al. (1997)](https://doi.org/10.2307/1313099) and
[Yarnell et al. (2020)](https://doi.org/10.1002/rra.3575).

## What a configured condition means

A characteristic describes one aspect of a component. For example, a
magnitude characteristic can require flow to exceed a threshold; a duration
characteristic can require a sequence of qualifying timesteps to last for an
acceptable number of observations. Conditions and their combinations are
chosen for the analysis question and reference data. The values in examples
are illustrative software inputs, not universal ecological criteria.

Use timing for the part of the calendar year when a condition is evaluated;
magnitude for the observed flow level; duration for the length of a
consecutive qualifying run; frequency for qualifying timesteps in forward
windows or water years in interannual windows; and rate of change for the
ratio between flow values separated by a configured number of timesteps.
These characteristics answer different questions and use different units.

See [evaluation order and interpretation](evaluation-order.md) for how
characteristic outcomes combine, and the
[characteristic reference](../reference.md#characteristic-reference) for
configuration and evaluation details.
