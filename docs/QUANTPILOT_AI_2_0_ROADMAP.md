# QuantPilot-AI 2.0 Roadmap

## Phase 0: Create New Repo / Project Skeleton

Create QuantPilot-AI-Next with documentation, basic repository structure,
license review placeholders, CI placeholders, and no trading logic.

## Phase 1: Core Contracts And Registry

Define contracts for data sources, datasets, factors, engines, strategies,
evaluators, portfolios, paper feedback, and agents. Add a registry for
capability discovery and adapter selection.

## Phase 2: Data Contracts And Local Fixtures

Create local fixture datasets and validation checks. Establish schemas,
lineage, reproducibility rules, and fixture-only smoke tests.

## Phase 3: Controlled Data-Source Prototypes

Prototype AkShare, Baostock, Tushare, and OpenBB adapters in controlled mode.
Do not fetch data in CI unless explicitly approved. Capture source limitations,
licenses, and reliability risks.

## Phase 4: A-Share Market Rule Engine

Implement A-share calendars, symbol rules, lot size, price limits, suspensions,
transaction costs, slippage assumptions, and small-capital constraints behind
contracts.

## Phase 5: Backtest / Research Engine Prototype Comparison

Compare Qlib, LEAN, vectorbt, Backtrader, RQAlpha, and Zipline-reloaded through
adapters. Select engines by evidence, not preference.

## Phase 6: Alpha Engine And Factor Validation

Build factor validation workflows using contract-driven datasets and analytics
tools such as Alphalens, quantstats, pyfolio, empyrical, Polars, DuckDB, and
Parquet where appropriate.

## Phase 7: Strategy Tournament

Create controlled strategy comparison rules, promotion criteria, failure
reports, and tournament evidence. Keep this separate from broker or execution
claims.

## Phase 8: Walk-Forward / OOS Validation

Add walk-forward, out-of-sample, regime, and robustness validation. Require
clear evidence before any strategy promotion.

## Phase 9: Portfolio / Risk Allocation

Prototype portfolio and risk allocation with PyPortfolioOpt, riskfolio-lib, or
equivalent tools. Preserve A-share and small-capital constraints.

## Phase 10: Paper Trading Feedback Loop

Create a paper-only feedback loop with replay, simulated fills, drift reports,
and decision review. Keep broker paths disabled unless a separate readiness
review approves them.

## Phase 11: Multi-Agent Orchestration

Evaluate TradingAgents, FinRobot, FinGPT, RD-Agent, LangGraph, AutoGen, CrewAI,
OpenAI Agents SDK, OpenAI Skills, Scrapling, and the MCP ecosystem for research
automation, diagnostics, and maintenance workflows.

## Phase 12: Productization And Commercial Readiness Review

Review product UX, licensing, data rights, security, deployment, compliance,
broker safety, and commercial viability. Only after this phase may the project
consider any trading-readiness language, and only if supported by evidence.
