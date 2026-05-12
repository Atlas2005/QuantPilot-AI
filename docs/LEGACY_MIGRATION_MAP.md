# Legacy Migration Map

## Classification Labels

Legacy components should be classified as:

- `keep_as_core_idea`: Preserve the idea, but reimplement cleanly in 2.0.
- `keep_as_guardrail`: Preserve as a validation or safety reference.
- `keep_as_fixture_or_regression_reference`: Preserve outputs or behavior as
  fixtures for future compatibility checks.
- `reference_only`: Read for context, but do not extend or copy as core.
- `replace_with_open_source`: Prefer mature external tools or adapters.
- `borrow_architecture_only`: Use the shape of the idea, not the code.
- `deprecate_do_not_extend`: Freeze and avoid future investment.

## Explicit Migration Decisions

| Legacy area | Classification | 2.0 direction |
| --- | --- | --- |
| Old backtester | `replace_with_open_source`, `reference_only` | Compare Qlib, LEAN, vectorbt, Backtrader, RQAlpha, and Zipline-reloaded behind engine contracts. |
| Old ML trainer | `replace_with_open_source`, `reference_only` | Replace with open-source ML/research workflows and clear model evaluation contracts. |
| Old factor builder | `replace_with_open_source`, `reference_only` | Rebuild around data contracts, factor validation, and tools such as Qlib, Alphalens, Polars, DuckDB, and Parquet. |
| Old `semi_auto_order_generator` | `deprecate_do_not_extend` | Do not extend. Any future execution path requires a new readiness policy, broker adapter review, and human approval gates. |
| V6/V7 guardrails | `keep_as_guardrail`, `reference_only` | Preserve validation, reproducibility, schema, and closure concepts as guardrail references. |
| Capital constraint engine | `keep_as_core_idea` | Reimplement cleanly in 2.0 contracts and A-share adapters. |
| Tradable universe filter | `keep_as_core_idea` | Reimplement as A-share market-rule and small-capital universe contracts. |
| Position sizing engine | `keep_as_core_idea` | Reimplement as portfolio and risk contracts with small-capital constraints. |
| Paper ledger concepts | `borrow_architecture_only`, `keep_as_guardrail` | Rebuild later as a paper feedback loop after market realism contracts exist. |
| Output schema validators | `keep_as_guardrail` | Recreate as contract tests with Pandera, Great Expectations, or equivalent validation tooling. |
| Reproducibility checks | `keep_as_guardrail` | Preserve the discipline and create deterministic fixture-based CI checks. |
| Broker integration research | `reference_only` | Use only as context. Do not treat as a broker path or readiness evidence. |
| Existing outputs | `keep_as_fixture_or_regression_reference` | Preserve as historical evidence. Do not mutate or treat as production results. |
| Dashboard/app code | `reference_only` | Use as UX context only. Build future product UI cleanly, likely with TypeScript/React. |

## Migration Rule

The default migration action is not copying code. The preferred path is:

1. Identify the useful concept.
2. Define a 2.0 contract.
3. Evaluate open-source providers.
4. Build the smallest A-share adapter or fixture needed.
5. Add validation before expanding behavior.

## Things To Avoid Carrying Forward

- Custom framework shape as the default architecture.
- Hidden assumptions from v2 command scripts.
- Any implied live-trading or broker readiness.
- Historical outputs as proof of profitability.
- Strategy-specific patches that bypass contract-level validation.
