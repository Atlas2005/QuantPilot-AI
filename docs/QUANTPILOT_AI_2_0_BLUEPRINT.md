# QuantPilot-AI 2.0 Blueprint

## Product Definition

QuantPilot-AI 2.0 is an A-share AI Quant Integration Platform.

Its purpose is not to replace mature quant infrastructure with custom code. Its
purpose is to integrate strong open-source capabilities and own the A-share
adaptation layer, validation contracts, market realism, research workflow, and
eventual agent orchestration around them.

## Core Principle

QuantPilot-AI 2.0 is open-source-first, adapter-first, contract-first,
A-share-first, Python-first, and polyglot-ready.

Open-source tools are external capability providers. QuantPilot-AI 2.0 should
evaluate and integrate mature projects when they are clearly better than custom
code, including:

- Qlib, LEAN, vectorbt, vn.py / VeighNa, Backtrader, RQAlpha, and
  Zipline-reloaded for research, backtesting, or engine comparison.
- AkShare, Baostock, Tushare, and OpenBB for data access prototypes.
- Pandera, Great Expectations, Polars, DuckDB, PyArrow, and Parquet for data
  contracts, analytics, and storage.
- Alphalens, quantstats, pyfolio, and empyrical for factor and performance
  analysis.
- PyPortfolioOpt and riskfolio-lib for portfolio and risk allocation.
- TradingAgents, FinRobot, FinGPT, RD-Agent, LangGraph, AutoGen, CrewAI,
  OpenAI Agents SDK, OpenAI Skills, Scrapling, and the MCP ecosystem for later
  agent and research automation layers.
- Warp and Ruflo as tooling references only.

## What QuantPilot-AI 2.0 Owns

QuantPilot-AI 2.0 should own:

- A-share adaptation and market conventions.
- Small-capital constraints.
- Data quality contracts and fixture governance.
- Market realism, including trading calendars, suspensions, limit-up/limit-down
  behavior, lot size, transaction costs, slippage, and liquidity constraints.
- Alpha validation and factor evidence.
- Strategy tournament design.
- Paper trading feedback loops.
- Later multi-agent orchestration.
- Product-facing workflow, reporting, and readiness review.

## Proposed Modular Architecture

The platform should be organized around stable contracts and replaceable
adapters:

- `contracts`: shared interfaces, schemas, and validation boundaries.
- `registry`: capability registry for data sources, engines, factors,
  strategies, evaluators, and agents.
- `adapters`: wrappers around open-source tools and external services.
- `ashare`: A-share calendars, symbol rules, trading rules, lot rules, fees,
  and constraints.
- `data`: local fixtures, data quality checks, lineage, and storage interfaces.
- `research`: factor research, alpha validation, experiments, and reports.
- `engines`: backtest and research engine adapters.
- `tournament`: controlled strategy comparison and promotion rules.
- `portfolio`: allocation, risk budgeting, and constraints.
- `paper`: simulated execution feedback, paper ledger, and replay.
- `agents`: later orchestration for research, diagnostics, and maintenance.
- `ui`: future product UI.
- `ops`: dependency monitoring, CI, smoke tests, and release checks.

## Proposed Directory Structure

```text
QuantPilot-AI-Next/
  README.md
  docs/
    ARCHITECTURE.md
    OPEN_SOURCE_FIRST_POLICY.md
    ROADMAP.md
    TRADING_READINESS_POLICY.md
  pyproject.toml
  src/
    quantpilot_next/
      contracts/
      registry/
      adapters/
        data_sources/
        engines/
        analytics/
        agents/
      ashare/
      data/
      research/
      engines/
      tournament/
      portfolio/
      paper/
      ops/
  tests/
    fixtures/
    contract/
    smoke/
    regression/
  examples/
  notebooks/
  ui/
  scripts/
```

## Non-Goals For The First Skeleton

The first QuantPilot-AI-Next skeleton should not add trading logic, train
models, fetch live market data, connect brokers, run backtests, or claim
readiness. It should establish documentation, contracts, fixtures, and adapter
boundaries first.
