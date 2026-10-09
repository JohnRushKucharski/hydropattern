'''Correlated forward-window outcomes, represented as reduced decision graphs.'''
from typing import Callable

import numpy as np


class _DecisionGraph:
    '''Share equivalent Boolean subexpressions without enumerating assignments.'''

    def __init__(self) -> None:
        self.nodes: list[tuple[int, int, int]] = [(-1, 0, 0), (-1, 1, 1)]
        self.unique: dict[tuple[int, int, int], int] = {}
        self.combinations: dict[tuple[bool, int, int], int] = {}
        self.complements = {0: 1, 1: 0}
        self.forgotten: dict[tuple[int, int], int] = {}

    def node(self, variable: int, low: int, high: int) -> int:
        if low == high:
            return low
        key = (variable, low, high)
        existing = self.unique.get(key)
        if existing is not None:
            return existing
        result = len(self.nodes)
        self.nodes.append(key)
        self.unique[key] = result
        return result

    def negate(self, root: int) -> int:
        pending = [root]
        while pending:
            current = pending[-1]
            if current in self.complements:
                pending.pop()
                continue
            variable, low, high = self.nodes[current]
            missing = [
                child for child in (low, high) if child not in self.complements
            ]
            if missing:
                pending.extend(missing)
                continue
            inverse = self.node(
                variable, self.complements[low], self.complements[high]
            )
            self.complements[current] = inverse
            self.complements[inverse] = current
            pending.pop()
        return self.complements[root]

    def combine(self, left: int, right: int, conjunction: bool = False) -> int:
        def key_for(a: int, b: int) -> tuple[bool, int, int]:
            return conjunction, min(a, b), max(a, b)

        root = key_for(left, right)
        pending = [root]
        while pending:
            key = pending[-1]
            if key in self.combinations:
                pending.pop()
                continue
            _, a, b = key
            if a == b:
                result = a
            elif conjunction and a == 0:
                result = 0
            elif conjunction and a == 1:
                result = b
            elif not conjunction and a == 0:
                result = b
            elif not conjunction and a == 1:
                result = 1
            else:
                variable = min(self.nodes[a][0], self.nodes[b][0])
                _, a_low, a_high = self.nodes[a]
                _, b_low, b_high = self.nodes[b]
                if self.nodes[a][0] != variable:
                    a_low = a_high = a
                if self.nodes[b][0] != variable:
                    b_low = b_high = b
                low_key = key_for(a_low, b_low)
                high_key = key_for(a_high, b_high)
                missing = [
                    child for child in (low_key, high_key)
                    if child not in self.combinations
                ]
                if missing:
                    pending.extend(missing)
                    continue
                result = self.node(
                    variable,
                    self.combinations[low_key],
                    self.combinations[high_key],
                )
            self.combinations[key] = result
            pending.pop()
        return self.combinations[root]

    def forget(self, root: int, variable: int) -> int:
        '''Existentially eliminate a consumed trial, retaining future correlations.'''
        pending = [root]
        while pending:
            current = pending[-1]
            key = current, variable
            if key in self.forgotten:
                pending.pop()
                continue
            current_variable, low, high = self.nodes[current]
            if current <= 1 or current_variable > variable:
                result = current
            elif current_variable == variable:
                result = self.combine(low, high)
            else:
                missing = [
                    child for child in (low, high)
                    if (child, variable) not in self.forgotten
                ]
                if missing:
                    pending.extend(missing)
                    continue
                result = self.node(
                    current_variable,
                    self.forgotten[low, variable],
                    self.forgotten[high, variable],
                )
            self.forgotten[key] = result
            pending.pop()
        return self.forgotten[root, variable]


def correlated_forward_windows(
    eligible: np.ndarray,
    predicate: Callable[[float], bool],
    window_length: int,
    exclusive: bool,
) -> np.ndarray:
    '''Combine window truth and suppression using the same unknown variables.'''
    graph = _DecisionGraph()
    zero_admitting = predicate(0)
    result = np.zeros(len(eligible))
    schedules = {-1: 1}
    for start in range(len(eligible)):
        end = min(start + window_length, len(eligible))
        window = eligible[start:end]
        ones = int(np.count_nonzero(window == 1))
        unknown_indices = np.flatnonzero(np.isnan(window)) + start
        verdicts = [
            int(bool(predicate(ones + count)))
            for count in range(len(unknown_indices) + 1)
        ]
        for variable in reversed(unknown_indices):
            verdicts = [
                graph.node(int(variable), low, high)
                for low, high in zip(verdicts, verdicts[1:])
            ]
        succeeds = verdicts[0]
        if not zero_admitting:
            if eligible[start] == 0:
                succeeds = 0
            elif np.isnan(eligible[start]):
                succeeds = graph.combine(
                    succeeds, graph.node(start, 0, 1), conjunction=True
                )
        available = 0
        continuing = {}
        fails = graph.negate(succeeds)
        for claimed_until, condition in schedules.items():
            if claimed_until < start:
                available = graph.combine(available, condition)
            else:
                remaining = (
                    condition if exclusive
                    else graph.combine(condition, fails, conjunction=True)
                )
                if remaining:
                    continuing[claimed_until] = remaining
        claim = (
            graph.combine(available, succeeds, conjunction=True)
            if exclusive else succeeds
        )
        idle = graph.combine(available, fails, conjunction=True)
        if claim:
            continuing[end - 1] = graph.combine(
                continuing.get(end - 1, 0), claim
            )
        if idle:
            continuing[-1] = idle
        can_succeed = any(until >= start for until in continuing)
        result[start] = (
            np.nan if can_succeed and idle
            else 1 if can_succeed else 0
        )
        # Consumed trials may select different schedules, but their values
        # cannot enter another window. Keep only future-trial constraints.
        schedules = (
            {
                until: graph.forget(condition, start)
                for until, condition in continuing.items()
            }
            if np.isnan(eligible[start]) else continuing
        )
    return result
