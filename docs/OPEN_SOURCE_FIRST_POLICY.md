# Open-Source-First Policy

## Policy

Before building any major QuantPilot-AI 2.0 module from scratch, evaluate
mature GitHub and open-source alternatives.

The default decision should not be "write custom code." The decision should be
one of:

- Adopt the external project directly.
- Wrap it behind a QuantPilot-AI contract.
- Integrate it as one adapter among several.
- Borrow its architecture while implementing only the A-share-specific layer.
- Defer the module until there is a real need.
- Avoid the dependency because the fit, risk, or maintenance cost is poor.

Do not keep inferior custom modules because of sunk cost.

## Evaluation Criteria

Every major candidate should be evaluated against:

- License and commercial-use compatibility.
- Maintenance activity, issue health, release cadence, and maintainer risk.
- Windows compatibility.
- Python version and dependency compatibility.
- A-share fit and ability to model China market rules.
- Commercial risk, including data redistribution limits and broker terms.
- Integration cost and adapter complexity.
- Reliability under local fixture tests.
- Testability and deterministic behavior.
- Documentation quality and community usage.
- Extensibility for contract-first adapters.
- Operational risk in CI, local development, and future deployment.

## Candidate Categories

QuantPilot-AI 2.0 should actively evaluate:

- Research and backtesting: Qlib, LEAN, vectorbt, Backtrader, RQAlpha,
  Zipline-reloaded.
- China trading and broker ecosystem: vn.py / VeighNa.
- Data sources: AkShare, Baostock, Tushare, OpenBB.
- Data validation and storage: Pandera, Great Expectations, Polars, DuckDB,
  PyArrow, Parquet.
- Factor and performance analytics: Alphalens, quantstats, pyfolio, empyrical.
- Portfolio and risk: PyPortfolioOpt, riskfolio-lib.
- Agent and research automation: TradingAgents, FinRobot, FinGPT, RD-Agent,
  LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, OpenAI Skills, Scrapling, MCP.

## Decision Record Requirement

For each major module, QuantPilot-AI 2.0 should keep a short decision record:

- Problem being solved.
- Candidates evaluated.
- Decision.
- Why the chosen path fits A-share and small-capital constraints.
- Risks and rollback path.
- Contract boundary used to avoid lock-in.

## Guardrail

Open-source-first does not mean dependency-maximal. It means mature external
capabilities should be evaluated honestly before custom work begins, and custom
work should focus on the platform's real ownership areas.
