# QuantPilot-AI Legacy Closure

## Status

The current `v2-real-data` line is now closed as a legacy research archive.

This repository should not be used as the future core architecture for
QuantPilot-AI 2.0 or QuantPilot-AI-Next. It remains valuable, but its value is
archival and evidentiary rather than architectural.

## What This Archive Is For

This line should be preserved as:

- A record of A-share research constraints encountered during the v2 work.
- A guardrail reference for validation, smoke tests, reproducibility checks,
  and failure triage.
- A CI and smoke-test reference for future projects.
- A migration input for identifying useful concepts, bad assumptions, and
  design decisions that should not be repeated.
- A historical record of both failed and useful Codex-generated framework
  decisions.

## What This Archive Is Not For

This line is not:

- The target architecture for QuantPilot-AI 2.0.
- A trading-ready system.
- A broker-connected product.
- A production data, model, portfolio, or execution platform.
- A framework that should continue accumulating custom modules by default.

## Trading Readiness

No trading-ready claim exists for this repository.

Any future QuantPilot-AI 2.0 work must treat this branch as research evidence
only. It must not inherit implied production status, brokerage readiness,
profitability claims, or execution safety from the legacy line.

## Closure Principle

The legacy branch is closed because it reached a useful strategic endpoint:
it exposed constraints, generated fixtures, created validation artifacts, and
clarified what a future platform must do differently. Future work should move
to QuantPilot-AI 2.0 / QuantPilot-AI-Next with an open-source-first,
adapter-first, contract-first architecture.
