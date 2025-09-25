# Code notes
Technical notes about the code and its structure.

## General notes on the evaluation of characteristics
Context:
1. User provides a .toml file with components and characteristics.
2. The characteristics are defined in the .toml file under the component (to which they belong).
3. The characteristic definition includes the characteristic type (i.e., duration, magnitude, etc.) and the parameters required to evaluate that characteristic (i.e., operator, threshold value, order, etc.).
4. This information is used to build a function that evaluates that characteristic over a timeseries (dataframe).

Bulding the characteristic function:
1. **Entry Point**: The construction of characteristic functions usually starts inside a loop (characteristics) within a loop (components) inside the `hydropattern.parsers.parse_components()` function which is called when reading the .toml file. The `parse_components()` function calls specific characteristic parsers (i.e., the `hydropattern.parsers.duration_parser()` function). This list of notes generalizes the process, which is similiar (but not the same) for all characteristics.
 2. **Comparision function**: In the characterstic parser (i.e. `duration_parser()`), operator symbols and operater threshold values (i.e. >, 5) are input into the `hydropattern.patterns.comparison_fx()` function. Thes inputs create a partially parameterized version of a built-in python operator (i.e., `operator.gt(a, b)`) function, using the `functools.partial()` utility. By default, the python operator functions "b" parameter is set (i.e., `functools.partial(operator.gt, b=5)` in the example). Therefore what is returned is a closure function that takes a single input (the "a" parameter by default) and compares it to the set parameter value (the "b" parameter by default). For example, if `f = functools.partial(operator.gt, b=5)` then `f(4) = False` and `f(6) = True`. [Note: If a "between" comparision is specified (i.e., x between 5 and 10) as is always the case for timing characteristics, two partially parameterized functions are created, one for each bound (i.e., `partial(gt, b=5)` and `partial(lt, b=10)`)].
3. **Characteristic function**: The comparsion funcion and other parameters (i.e., order and other characteristic specific parameters) are passed as inputs into a characteristic function builder (i.e., `hydropattern.patterns.duration_fx()` function). This function returns a closure function that takes two inputs: (1) a pandas dataframe (the timeseries data) and (2) an optional numpy array (of 0s and 1s), which is the output of previously evaluated characteristics in the same component. The order parameter of the characteristic building function determines the order in which multiple characteistics belonging to the same component are evaluated.

