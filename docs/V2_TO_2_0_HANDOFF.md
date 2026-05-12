# QuantPilot-AI v2 to 2.0 Handoff

## What v2 Achieved

The `v2-real-data` line created a broad educational and research workflow for
A-share experimentation. It connected simple data loading, technical signals,
factor dataset generation, baseline model training, diagnostic backtests,
capital-aware planning, paper-ledger ideas, validation manifests, and smoke-test
style checks.

It also produced useful project memory:

- A-share-specific constraints are central, not optional.
- Small-capital behavior changes the tradable universe and position sizing.
- Data quality and output schema checks must be explicit.
- Validation evidence matters as much as strategy code.
- Guardrails around reproducibility, simulation assumptions, and broker
  boundaries are necessary before any product claim.

## What v2 Failed To Prove

v2 did not prove:

- A profitable or durable alpha engine.
- A production-grade backtesting engine.
- A reliable ML research platform.
- A safe broker integration path.
- A scalable market data architecture.
- A maintainable custom framework worth extending as the core of 2.0.
- Trading readiness for live or semi-automated execution.

## Why v2 Should Not Continue As Main Architecture

The legacy line grew as a custom Codex-generated framework. That made it useful
for exploration, but not ideal as the long-term core. Continuing the same path
would preserve too much bespoke infrastructure where mature open-source systems
already exist.

The future project should avoid sunk-cost attachment. It should compare,
integrate, wrap, and adapt established tools before building major modules from
scratch.

## What Must Be Carried Into 2.0

QuantPilot-AI 2.0 should carry forward:

- A-share-first assumptions and market-rule awareness.
- Small-capital constraints, including tradable universe and position sizing.
- Explicit data quality contracts.
- Validation manifests, reproducibility checks, and smoke-test discipline.
- Separation between research, paper feedback, and any future broker path.
- Clear non-trading-ready language until a formal readiness review exists.
- Lessons from failure cases and diagnostic outputs.

## What Must Not Be Carried Into 2.0

QuantPilot-AI 2.0 should not carry forward by default:

- The old custom backtester as the core engine.
- The old ML trainer as the core ML platform.
- The old factor builder as the core factor platform.
- The semi-auto order generator.
- Any broker-facing implication from v2.
- Bespoke modules that are inferior to mature open-source alternatives.
- Output folders as authoritative production evidence.

## Handoff Decision

Freeze this repository as a legacy research archive. Start QuantPilot-AI-Next
as a clean project for QuantPilot-AI 2.0, using v2 only as reference material,
guardrail evidence, and migration input.
