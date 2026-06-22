# Behavior Delta Report — Issue #5: Operator/Value Format Normalization

## Summary

Whitespace-padded comparison operators (e.g. `" > "`, `" <= "`) now normalize to their
stripped form before validation. Previously they raised `UNKNOWN_COMPARISON_SYMBOL`;
now they succeed and are treated identically to the exact symbol.

---

## Conflict Entry

- **Conflict ID:** 5-01
- **Area:** Parser — operator/value format normalization (`hydropattern/parsers.py`)
- **Baseline behavior (hydropattern):** Only exact-match symbols accepted. `" > "` →
  `KeyError` → `HydropatternError(UNKNOWN_COMPARISON_SYMBOL)`.
- **Candidate behavior (fork/slice):** Strip leading/trailing whitespace before symbol
  lookup, accept the result as the normalized symbol.
- **Winner:** Candidate (whitespace stripping)
- **Rationale:** User intent is unambiguous. Whitespace is not meaningful in an operator
  token and rejecting it produces a confusing, hard-to-debug error with no
  recoverability. Normalizing is strictly more permissive and does not change the
  semantics of valid inputs.
- **Risks:** None identified. Whitespace-only strings (`"   "`) still fail with
  `UNKNOWN_COMPARISON_SYMBOL` because they strip to `""` which is not a valid symbol.
- **Evidence test:** `tests/test_operator_normalization.py`
  - `TestNormalizeOperator::test_surrounding_whitespace_stripped`
  - `TestNormalizeOperator::test_whitespace_only_string_raises`
  - `TestParserNormalizationIntegration::test_magnitude_parser_accepts_all_padded_symbols`
  - `TestParserNormalizationIntegration::test_regression_exact_symbols_unchanged`
- **Affected code references:**
  - `hydropattern/parsers.py` — added `_VALID_SYMBOLS`, `normalize_operator()`;
    updated `validate_symbol()` to delegate to `normalize_operator`;
    updated `validate_simple_comparision_pair()` to assign normalized value back to
    `metrics[0]` so all downstream consumers (name generation, `comparison_fx`) receive
    the stripped symbol.

---

## Before / After

### Before

```toml
# This TOML config would raise UNKNOWN_COMPARISON_SYMBOL at parse time:
[components.high_flow]
magnitude = [" > ", 100.0]
```

```
HydropatternError: Invalid comparision symbol:  > .
  code: PARSER_UNKNOWN_COMPARISON_SYMBOL
  context: {symbol: ' > '}
```

### After

```toml
# Same config now parses successfully; " > " normalizes to ">"
[components.high_flow]
magnitude = [" > ", 100.0]
```

Produces a `Characteristic` with name `magnitude_gt100.0`, identical to having written `">"`.

---

## Decision date

2026-06-22

## Human approval

☐ Pending — required before merge per umbrella PRD #1 HITL policy.
