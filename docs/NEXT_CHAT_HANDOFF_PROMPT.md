# Next Chat Handoff Prompt

Copy and paste this prompt into a new ChatGPT/Codex conversation to begin
QuantPilot-AI 2.0 / QuantPilot-AI-Next from scratch.

```text
You are helping start QuantPilot-AI-Next, the clean QuantPilot-AI 2.0 project.

Project goal:
Build QuantPilot-AI 2.0 as an A-share AI Quant Integration Platform. The
platform should be open-source-first, adapter-first, contract-first,
A-share-first, Python-first, and polyglot-ready.

Why v2 is frozen:
The old QuantPilot-AI v2-real-data branch is now a legacy research archive. It
created useful research artifacts, constraints, guardrails, validation ideas,
and failure lessons, but it should not continue as the core architecture. It was
a custom Codex-generated framework and did not prove production-grade
backtesting, durable alpha, reliable ML infrastructure, safe broker integration,
or trading readiness. Treat it only as migration input, guardrail reference,
smoke-test reference, and historical evidence.

2.0 principles:
- Do not copy old v2 code as core by default.
- Evaluate mature open-source alternatives before building major modules.
- Prefer adapters and contracts over custom framework lock-in.
- Own A-share adaptation, small-capital constraints, data quality, market
  realism, alpha validation, strategy tournament design, paper feedback, and
  later multi-agent orchestration.
- Preserve clear boundaries between research, paper trading, and any future
  broker path.
- Do not mark anything trading-ready without a formal readiness review.

Open-source-first requirement:
Before building any major module from scratch, evaluate relevant projects such
as Qlib, LEAN, vectorbt, vn.py / VeighNa, Backtrader, RQAlpha,
Zipline-reloaded, AkShare, Baostock, Tushare, OpenBB, Pandera, Great
Expectations, Polars, DuckDB, PyArrow / Parquet, Alphalens, quantstats,
pyfolio, empyrical, PyPortfolioOpt, riskfolio-lib, TradingAgents, FinRobot,
FinGPT, RD-Agent, LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, OpenAI Skills,
Scrapling, and the MCP ecosystem. Use Warp and Ruflo as tooling references
only. Evaluate license, maintenance, Windows compatibility, A-share fit,
commercial risk, integration cost, reliability, and testability.

Language architecture:
Python-first, not Python-only. Use Python for orchestration, research, adapters,
contracts, and agent tools. Use SQL/DuckDB for local analytics. Use
Parquet/Arrow for storage and interchange. Use Polars where performance
justifies it. Treat C#/LEAN as an external engine candidate, not something to
rewrite. Use TypeScript/React later for product UI. Use Rust/C++ only after a
proven bottleneck. Avoid premature polyglot complexity.

Roadmap:
Phase 0: create new repo/project skeleton.
Phase 1: core contracts and registry.
Phase 2: data contracts and local fixtures.
Phase 3: controlled data-source prototypes.
Phase 4: A-share market rule engine.
Phase 5: backtest/research engine prototype comparison.
Phase 6: alpha engine and factor validation.
Phase 7: strategy tournament.
Phase 8: walk-forward/OOS validation.
Phase 9: portfolio/risk allocation.
Phase 10: paper trading feedback loop.
Phase 11: multi-agent orchestration.
Phase 12: productization and commercial readiness review.

Forbidden actions for the first task:
- Do not add trading logic.
- Do not fetch market data.
- Do not call external APIs.
- Do not train models.
- Do not run backtests.
- Do not connect brokers.
- Do not execute or submit orders.
- Do not mark anything trading-ready.
- Do not install packages unless explicitly approved.

First task:
Create the QuantPilot-AI-Next skeleton and documentation only. Establish the
project structure, README, architecture docs, open-source-first policy, roadmap,
trading-readiness policy, and placeholder contract directories. Do not implement
trading logic.
```
