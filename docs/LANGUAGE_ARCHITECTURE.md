# Language Architecture

## Principle

QuantPilot-AI 2.0 is Python-first, not Python-only.

The platform should use the simplest language mix that supports research,
adapter integration, data quality, and future product workflows. Avoid
premature polyglot complexity.

## Python

Python should be the default language for:

- Orchestration.
- Research workflows.
- Data-source adapters.
- Engine adapters.
- A-share market-rule adapters.
- Agent tools.
- Contract tests and smoke tests.
- Local CLI workflows.

Python remains the center of gravity because the quant, data science, and agent
ecosystems are strongest there.

## SQL, DuckDB, Parquet, And Arrow

SQL and DuckDB should be used for local analytics when tabular questions are
better expressed declaratively.

Parquet and Arrow should be the preferred storage and interchange formats for
local columnar datasets and fixtures. They make data contracts, reproducibility,
and cross-language access easier.

## Polars

Polars is a candidate for fast local data handling when pandas-style workflows
become too slow or memory-heavy. It should be adopted through data contracts,
not scattered ad hoc throughout the codebase.

## C# And LEAN

C#/LEAN should be treated as an external engine candidate, not something to
rewrite.

If LEAN is selected for any prototype, QuantPilot-AI 2.0 should integrate with
it through an adapter boundary and preserve Python-facing contracts for the
rest of the platform.

## TypeScript And React

TypeScript/React is the likely future product UI path. It should remain outside
the core research contracts until product workflows are clear.

The UI should consume platform outputs and APIs rather than owning quant logic.

## Rust And C++

Rust or C++ should be considered only after a proven bottleneck exists and
Python, DuckDB, Polars, vectorized libraries, or engine-level optimization are
insufficient.

## Guiding Rule

Start with Python, contracts, fixtures, and adapters. Add other languages only
when they clearly reduce risk, improve capability, or integrate an external
engine that should not be rewritten.
