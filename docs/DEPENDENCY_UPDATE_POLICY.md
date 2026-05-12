# Dependency Update Policy

## Future Automation

QuantPilot-AI 2.0 should eventually use Renovate or Dependabot to detect
dependency updates and open automated pull requests.

Automated dependency management should:

- Detect Python, JavaScript/TypeScript, GitHub Actions, and container updates
  when those stacks exist.
- Open small, reviewable PRs.
- Run compatibility tests before review.
- Generate dependency risk reports.
- Separate patch, minor, and major updates.
- Highlight license, security, and quant-behavior risks.

## Required Test Gates

Dependency update PRs should run:

- Unit tests.
- Contract tests.
- Fixture-based data validation tests.
- Smoke tests.
- Engine adapter compatibility tests when relevant.
- Reproducibility checks for deterministic fixtures.

## High-Risk Dependencies

Never auto-merge high-risk quant, data, agent, or broker-related dependencies.

Human approval is required for updates involving:

- Qlib.
- LEAN.
- vectorbt.
- Backtrader, RQAlpha, Zipline-reloaded, vn.py / VeighNa.
- AkShare, Baostock, Tushare, OpenBB, or other data-source libraries.
- Pandera, Great Expectations, Polars, DuckDB, PyArrow, or storage contract
  dependencies when they affect data semantics.
- TradingAgents, FinRobot, FinGPT, RD-Agent, LangGraph, AutoGen, CrewAI,
  OpenAI Agents SDK, OpenAI Skills, Scrapling, MCP-related packages, or other
  agent frameworks.
- Broker-related packages, SDKs, gateways, or execution integrations.

## Risk Report Contents

Each dependency risk report should include:

- Package name and version change.
- Update type: patch, minor, major, security, or breaking.
- License change, if any.
- Known breaking changes.
- Affected adapters or contracts.
- Required fixture revalidation.
- Human-review reason for high-risk packages.

## Merge Policy

Low-risk patch updates may be eligible for normal review after tests pass.

High-risk updates require explicit human approval and should not be merged until
the reviewer confirms that quant semantics, data semantics, and adapter behavior
remain acceptable.
