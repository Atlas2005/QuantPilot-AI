import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/open_source_quant_stack_audit_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "inventory": "open_source_candidate_inventory.csv",
    "evaluation": "open_source_evaluation_matrix.csv",
    "a_share_fit": "open_source_a_share_fit_matrix.csv",
    "recommendations": "open_source_integration_recommendations.csv",
    "risk_register": "open_source_risk_register.csv",
    "architecture": "open_source_v7_architecture_decision.csv",
    "guardrails": "open_source_guardrails.csv",
    "summary": "open_source_stack_audit_summary.csv",
    "report": "open_source_stack_audit_report.md",
}

FORBIDDEN_FLAGS = [
    "market_data_fetch",
    "broker_connected",
    "execution_allowed",
    "live_trading",
    "real_order_submission",
    "trading_ready",
]

GUARDRAILS = [
    ("no_package_install", "confirmed", "No pip, conda, poetry, uv, or package manager command is executed."),
    ("no_external_framework_import", "confirmed", "Candidate frameworks are recorded as metadata only; no framework package is imported."),
    ("no_market_data_fetch", "confirmed", "The audit does not call AkShare, Baostock, Tushare, broker APIs, or any market data endpoint."),
    ("no_live_data", "confirmed", "No live quote, streaming, or broker data path is enabled."),
    ("no_backtest_execution", "confirmed", "No historical or synthetic backtest is run by this step."),
    ("no_model_training", "confirmed", "No ML model, factor model, optimizer, or agent is trained."),
    ("no_threshold_change", "confirmed", "No strategy threshold or selection threshold is read or modified."),
    ("no_feature_engineering_change", "confirmed", "No factor or feature engineering code is modified or executed."),
    ("no_broker_sdk_import", "confirmed", "No broker SDK or live trading package is imported."),
    ("no_broker_credentials", "confirmed", "No credential, token, account id, or secret argument is accepted."),
    ("no_broker_connection", "confirmed", "No broker connection object or network call is created."),
    ("no_order_execution", "confirmed", "No simulated, paper, sandbox, or live order executor is invoked."),
    ("no_real_order_submission", "confirmed", "No real order submission path exists in this audit."),
    ("no_trading_ready_upgrade", "confirmed", "All outputs explicitly keep trading_ready=False."),
    ("framework_selection_only", "confirmed", "This step is an architecture decision layer only."),
    ("open_source_first_policy_applied", "confirmed", "Major V7 modules are routed through mature tool evaluation before custom implementation."),
    ("educational_research_only", "confirmed", "The report is research-only and is not financial advice."),
]

CANDIDATES = [
    {
        "candidate": "Qlib",
        "project_category": "data_platform; factor_analysis; ML_AI_research; workflow_orchestration",
        "primary_language": "Python",
        "license_summary": "MIT license reported by repository metadata; verify dependency and data licenses before adoption.",
        "source_url": "https://github.com/microsoft/qlib",
        "maturity_level": "high",
        "maintenance_risk": "medium",
        "windows_local_compatibility": "medium",
        "python_compatibility": "medium",
        "a_share_market_fit": "medium",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "medium",
        "walk_forward_oos_validation_support": "high",
        "factor_research_support": "high",
        "ml_ai_research_support": "high",
        "live_trading_broker_risk": "medium",
        "commercial_license_usage_risk": "low",
        "integration_complexity": "high",
        "recommended_action": "evaluate_with_prototype",
        "rationale": "Strong AI quant research and workflow candidate, but integration, data format adaptation, and A-share execution assumptions must be proven locally.",
        "quantpilot_ai_role": "Evaluate as the main AI/factor/model workflow candidate, not as a trading-ready engine.",
    },
    {
        "candidate": "LEAN",
        "project_category": "backtest_engine; data_platform; workflow_orchestration",
        "primary_language": "C# with Python algorithm support",
        "license_summary": "Apache-2.0; live trading and cloud/data integrations require strict isolation.",
        "source_url": "https://github.com/QuantConnect/Lean",
        "maturity_level": "high",
        "maintenance_risk": "low",
        "windows_local_compatibility": "medium",
        "python_compatibility": "medium",
        "a_share_market_fit": "low",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "high",
        "walk_forward_oos_validation_support": "medium",
        "factor_research_support": "medium",
        "ml_ai_research_support": "medium",
        "live_trading_broker_risk": "high",
        "commercial_license_usage_risk": "low",
        "integration_complexity": "high",
        "recommended_action": "evaluate_with_prototype",
        "rationale": "Robust event-driven architecture and fill/cost modeling ideas are valuable, but direct live features must remain disabled and A-share support is not native.",
        "quantpilot_ai_role": "Prototype offline architecture lessons for event-driven backtest design and risk modeling only.",
    },
    {
        "candidate": "vectorbt",
        "project_category": "backtest_engine; performance_analysis; strategy_research",
        "primary_language": "Python",
        "license_summary": "Apache-2.0 with Commons Clause; high commercial usage risk.",
        "source_url": "https://github.com/polakowo/vectorbt",
        "maturity_level": "high",
        "maintenance_risk": "medium",
        "windows_local_compatibility": "high",
        "python_compatibility": "medium",
        "a_share_market_fit": "medium",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "medium",
        "walk_forward_oos_validation_support": "medium",
        "factor_research_support": "medium",
        "ml_ai_research_support": "medium",
        "live_trading_broker_risk": "low",
        "commercial_license_usage_risk": "high",
        "integration_complexity": "medium",
        "recommended_action": "evaluate_with_prototype",
        "rationale": "Excellent for fast vectorized research and strategy tournaments, but commercial restrictions may block direct product use.",
        "quantpilot_ai_role": "Use only in a license-reviewed research prototype for fast candidate exploration.",
    },
    {
        "candidate": "RQAlpha",
        "project_category": "backtest_engine; workflow_orchestration",
        "primary_language": "Python",
        "license_summary": "Repository states non-commercial use only; commercial use requires contacting Ricequant.",
        "source_url": "https://github.com/ricequant/rqalpha",
        "maturity_level": "high",
        "maintenance_risk": "low",
        "windows_local_compatibility": "medium",
        "python_compatibility": "medium",
        "a_share_market_fit": "high",
        "a_share_trading_rule_support": "high",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "medium",
        "walk_forward_oos_validation_support": "medium",
        "factor_research_support": "medium",
        "ml_ai_research_support": "low",
        "live_trading_broker_risk": "high",
        "commercial_license_usage_risk": "high",
        "integration_complexity": "high",
        "recommended_action": "borrow_architecture_only",
        "rationale": "A-share domain fit is strong, but non-commercial language makes direct product adoption unsafe without explicit permission.",
        "quantpilot_ai_role": "Study A-share market-rule and adapter architecture; avoid direct commercial integration for now.",
    },
    {
        "candidate": "Backtrader",
        "project_category": "backtest_engine",
        "primary_language": "Python",
        "license_summary": "Open-source; verify exact package license and maintenance status before commercial use.",
        "source_url": "https://www.backtrader.com/",
        "maturity_level": "medium",
        "maintenance_risk": "medium",
        "windows_local_compatibility": "high",
        "python_compatibility": "medium",
        "a_share_market_fit": "medium",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "medium",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "low",
        "ml_ai_research_support": "low",
        "live_trading_broker_risk": "medium",
        "commercial_license_usage_risk": "medium",
        "integration_complexity": "medium",
        "recommended_action": "evaluate_with_prototype",
        "rationale": "Simple Python backtesting architecture may be useful for adapters, but A-share rules and modern validation tooling are not enough out of the box.",
        "quantpilot_ai_role": "Evaluate as a lightweight A-share adapter prototype candidate, not as final infrastructure.",
    },
    {
        "candidate": "Alphalens",
        "project_category": "factor_analysis",
        "primary_language": "Python",
        "license_summary": "Apache-2.0, but project is tied to the older Quantopian ecosystem.",
        "source_url": "https://github.com/quantopian/alphalens",
        "maturity_level": "medium",
        "maintenance_risk": "high",
        "windows_local_compatibility": "medium",
        "python_compatibility": "low",
        "a_share_market_fit": "medium",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "low",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "medium",
        "factor_research_support": "high",
        "ml_ai_research_support": "low",
        "live_trading_broker_risk": "low",
        "commercial_license_usage_risk": "low",
        "integration_complexity": "medium",
        "recommended_action": "borrow_architecture_only",
        "rationale": "Factor IC, turnover, quantile, and grouped analysis design is valuable, but direct dependency may be stale.",
        "quantpilot_ai_role": "Borrow factor tear-sheet concepts for a future A-share factor evaluation layer.",
    },
    {
        "candidate": "quantstats",
        "project_category": "performance_analysis",
        "primary_language": "Python",
        "license_summary": "Open-source; verify exact license and dependency stack before product use.",
        "source_url": "https://github.com/ranaroussi/quantstats",
        "maturity_level": "medium",
        "maintenance_risk": "medium",
        "windows_local_compatibility": "high",
        "python_compatibility": "medium",
        "a_share_market_fit": "medium",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "low",
        "ml_ai_research_support": "low",
        "live_trading_broker_risk": "low",
        "commercial_license_usage_risk": "medium",
        "integration_complexity": "low",
        "recommended_action": "wrap_and_integrate",
        "rationale": "Useful performance reporting style with low integration burden, but should remain downstream analytics after trustworthy backtest outputs exist.",
        "quantpilot_ai_role": "Wrap later for performance tear sheets if license verification passes.",
    },
    {
        "candidate": "PyPortfolioOpt",
        "project_category": "portfolio_optimization",
        "primary_language": "Python",
        "license_summary": "MIT license on PyPI metadata.",
        "source_url": "https://github.com/PyPortfolio/PyPortfolioOpt",
        "maturity_level": "high",
        "maintenance_risk": "low",
        "windows_local_compatibility": "medium",
        "python_compatibility": "high",
        "a_share_market_fit": "medium",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "low",
        "ml_ai_research_support": "low",
        "live_trading_broker_risk": "low",
        "commercial_license_usage_risk": "low",
        "integration_complexity": "medium",
        "recommended_action": "evaluate_with_prototype",
        "rationale": "Good future portfolio allocation candidate, but optimization before alpha and realistic cost validation would be premature.",
        "quantpilot_ai_role": "Defer actual use until V7 data/backtest/factor evidence exists; prototype portfolio allocation later.",
    },
    {
        "candidate": "riskfolio-lib",
        "project_category": "portfolio_optimization; risk_management",
        "primary_language": "Python",
        "license_summary": "BSD-3-Clause reported by conda-forge; dependency stack is heavy.",
        "source_url": "https://github.com/dcajasn/Riskfolio-Lib",
        "maturity_level": "high",
        "maintenance_risk": "low",
        "windows_local_compatibility": "medium",
        "python_compatibility": "medium",
        "a_share_market_fit": "medium",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "low",
        "ml_ai_research_support": "low",
        "live_trading_broker_risk": "low",
        "commercial_license_usage_risk": "low",
        "integration_complexity": "high",
        "recommended_action": "defer_until_later",
        "rationale": "Powerful risk/portfolio toolkit, but heavy optimization dependencies are over-engineering before real alpha and portfolio inputs exist.",
        "quantpilot_ai_role": "Revisit for advanced risk allocation after portfolio research becomes necessary.",
    },
    {
        "candidate": "AkShare",
        "project_category": "data_source",
        "primary_language": "Python",
        "license_summary": "MIT license; underlying data-source terms must be reviewed endpoint by endpoint.",
        "source_url": "https://github.com/akfamily/akshare",
        "maturity_level": "high",
        "maintenance_risk": "medium",
        "windows_local_compatibility": "high",
        "python_compatibility": "high",
        "a_share_market_fit": "high",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "medium",
        "ml_ai_research_support": "low",
        "live_trading_broker_risk": "low",
        "commercial_license_usage_risk": "medium",
        "integration_complexity": "medium",
        "recommended_action": "evaluate_with_prototype",
        "rationale": "Broad A-share data access is attractive, but endpoint stability, provenance, rate limits, and commercial data rights need validation.",
        "quantpilot_ai_role": "Candidate data adapter source for V7 Step 2 data asset mapping; no fetch in this step.",
    },
    {
        "candidate": "Baostock",
        "project_category": "data_source",
        "primary_language": "Python",
        "license_summary": "BSD license reported by PyPI; service terms and data quality still require review.",
        "source_url": "https://pypi.org/project/baostock/",
        "maturity_level": "medium",
        "maintenance_risk": "medium",
        "windows_local_compatibility": "high",
        "python_compatibility": "medium",
        "a_share_market_fit": "high",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "low",
        "ml_ai_research_support": "low",
        "live_trading_broker_risk": "low",
        "commercial_license_usage_risk": "medium",
        "integration_complexity": "low",
        "recommended_action": "evaluate_with_prototype",
        "rationale": "Simple historical A-share daily data candidate, but coverage, adjustments, suspensions, and stability must be tested.",
        "quantpilot_ai_role": "Candidate baseline daily data source for V7 Step 2 data selection; no fetch in this step.",
    },
    {
        "candidate": "Tushare",
        "project_category": "data_source",
        "primary_language": "Python",
        "license_summary": "Usage, token, points, and commercial terms require explicit review before adoption.",
        "source_url": "https://tushare.pro/",
        "maturity_level": "high",
        "maintenance_risk": "medium",
        "windows_local_compatibility": "high",
        "python_compatibility": "high",
        "a_share_market_fit": "high",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "medium",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "medium",
        "ml_ai_research_support": "low",
        "live_trading_broker_risk": "low",
        "commercial_license_usage_risk": "high",
        "integration_complexity": "medium",
        "recommended_action": "defer_until_later",
        "rationale": "Potentially valuable A-share data source, but token/points/commercial constraints are material for a future product.",
        "quantpilot_ai_role": "Review only during V7 Step 2 data-source selection and legal/commercial screening.",
    },
    {
        "candidate": "RD-Agent",
        "project_category": "agent_orchestration; ML_AI_research",
        "primary_language": "Python",
        "license_summary": "MIT license; repository states current Linux-only support and financial disclaimer.",
        "source_url": "https://github.com/microsoft/RD-Agent",
        "maturity_level": "medium",
        "maintenance_risk": "medium",
        "windows_local_compatibility": "low",
        "python_compatibility": "medium",
        "a_share_market_fit": "medium",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "low",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "medium",
        "factor_research_support": "medium",
        "ml_ai_research_support": "high",
        "live_trading_broker_risk": "medium",
        "commercial_license_usage_risk": "low",
        "integration_complexity": "high",
        "recommended_action": "defer_until_later",
        "rationale": "Interesting DeepSeek-style R&D automation direction, but unstable foundation risk is too high before data/backtest/factor layers exist.",
        "quantpilot_ai_role": "Defer agentic quant R&D automation until V7 foundations are trustworthy.",
    },
    {
        "candidate": "LangGraph",
        "project_category": "agent_orchestration; workflow_orchestration",
        "primary_language": "Python",
        "license_summary": "MIT license.",
        "source_url": "https://github.com/langchain-ai/langgraph",
        "maturity_level": "high",
        "maintenance_risk": "low",
        "windows_local_compatibility": "high",
        "python_compatibility": "high",
        "a_share_market_fit": "low",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "low",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "low",
        "ml_ai_research_support": "medium",
        "live_trading_broker_risk": "medium",
        "commercial_license_usage_risk": "low",
        "integration_complexity": "high",
        "recommended_action": "defer_until_later",
        "rationale": "Good general stateful agent/workflow graph, but using it now would over-engineer before quantitative validation foundations are stable.",
        "quantpilot_ai_role": "Potential later orchestration layer for research agents, not part of V7 Step 1 implementation.",
    },
    {
        "candidate": "AutoGen",
        "project_category": "agent_orchestration",
        "primary_language": "Python",
        "license_summary": "Code license is MIT; repository also includes non-code content under CC-BY, so packaging boundaries need review.",
        "source_url": "https://github.com/microsoft/autogen",
        "maturity_level": "high",
        "maintenance_risk": "medium",
        "windows_local_compatibility": "high",
        "python_compatibility": "high",
        "a_share_market_fit": "low",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "low",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "low",
        "ml_ai_research_support": "medium",
        "live_trading_broker_risk": "medium",
        "commercial_license_usage_risk": "medium",
        "integration_complexity": "high",
        "recommended_action": "defer_until_later",
        "rationale": "Mature general multi-agent toolkit, but it does not solve A-share data, backtest realism, or alpha validation.",
        "quantpilot_ai_role": "Defer until research governance and deterministic evaluation harnesses exist.",
    },
    {
        "candidate": "CrewAI",
        "project_category": "agent_orchestration",
        "primary_language": "Python",
        "license_summary": "MIT license; verify telemetry/default data-sharing settings before enterprise use.",
        "source_url": "https://github.com/crewAIInc/crewAI",
        "maturity_level": "high",
        "maintenance_risk": "low",
        "windows_local_compatibility": "high",
        "python_compatibility": "high",
        "a_share_market_fit": "low",
        "a_share_trading_rule_support": "low",
        "small_capital_constraint_fit": "low",
        "transaction_cost_slippage_support": "low",
        "walk_forward_oos_validation_support": "low",
        "factor_research_support": "low",
        "ml_ai_research_support": "medium",
        "live_trading_broker_risk": "medium",
        "commercial_license_usage_risk": "low",
        "integration_complexity": "high",
        "recommended_action": "defer_until_later",
        "rationale": "Popular general agent framework, but agent orchestration before clean data and validation would add noise rather than alpha evidence.",
        "quantpilot_ai_role": "Defer as a possible later workflow/agent layer after quant foundations are stable.",
    },
]


def _add_forbidden_flags(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    for flag in FORBIDDEN_FLAGS:
        updated[flag] = False
    return updated


def _level_score(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(str(value).strip().lower(), 0)


def _is_high(value: Any) -> bool:
    return str(value).strip().lower() == "high"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def count_forbidden_true_flags(frames: list[pd.DataFrame], config: dict[str, Any]) -> int:
    count = 0
    for frame in frames:
        for flag in FORBIDDEN_FLAGS:
            if flag in frame:
                count += int(frame[flag].map(_is_true).sum())
    for flag in FORBIDDEN_FLAGS:
        count += int(_is_true(config.get(flag)))
    return count


def build_candidate_inventory() -> pd.DataFrame:
    rows = []
    for index, candidate in enumerate(CANDIDATES, start=1):
        rows.append(
            _add_forbidden_flags(
                {
                    "candidate_id": f"OSS-{index:03d}",
                    "candidate": candidate["candidate"],
                    "project_category": candidate["project_category"],
                    "primary_language": candidate["primary_language"],
                    "license_summary": candidate["license_summary"],
                    "source_url": candidate["source_url"],
                    "inventory_status": "reviewed_metadata_only",
                    "installed": False,
                    "imported": False,
                    "market_data_fetched": False,
                    "prototype_ran": False,
                    "notes": "Research-only inventory row; no external package was installed, imported, or executed.",
                }
            )
        )
    return pd.DataFrame(rows)


def build_evaluation_matrix() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        rows.append(
            _add_forbidden_flags(
                {
                    "candidate": candidate["candidate"],
                    "project_category": candidate["project_category"],
                    "maturity_level": candidate["maturity_level"],
                    "maintenance_risk": candidate["maintenance_risk"],
                    "windows_local_compatibility": candidate["windows_local_compatibility"],
                    "python_compatibility": candidate["python_compatibility"],
                    "a_share_market_fit": candidate["a_share_market_fit"],
                    "a_share_trading_rule_support": candidate["a_share_trading_rule_support"],
                    "small_capital_constraint_fit": candidate["small_capital_constraint_fit"],
                    "transaction_cost_slippage_support": candidate["transaction_cost_slippage_support"],
                    "walk_forward_oos_validation_support": candidate["walk_forward_oos_validation_support"],
                    "factor_research_support": candidate["factor_research_support"],
                    "ml_ai_research_support": candidate["ml_ai_research_support"],
                    "live_trading_broker_risk": candidate["live_trading_broker_risk"],
                    "commercial_license_usage_risk": candidate["commercial_license_usage_risk"],
                    "integration_complexity": candidate["integration_complexity"],
                    "recommended_action": candidate["recommended_action"],
                    "rationale": candidate["rationale"],
                    "quantpilot_ai_role": candidate["quantpilot_ai_role"],
                }
            )
        )
    return pd.DataFrame(rows)


def build_a_share_fit_matrix() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        readiness_score = (
            _level_score(candidate["a_share_market_fit"])
            + _level_score(candidate["a_share_trading_rule_support"])
            + _level_score(candidate["small_capital_constraint_fit"])
            + _level_score(candidate["transaction_cost_slippage_support"])
        )
        if readiness_score >= 10:
            fit_verdict = "strong_but_verify"
        elif readiness_score >= 7:
            fit_verdict = "partial_fit_requires_adapter"
        else:
            fit_verdict = "weak_direct_fit"
        rows.append(
            _add_forbidden_flags(
                {
                    "candidate": candidate["candidate"],
                    "a_share_market_fit": candidate["a_share_market_fit"],
                    "a_share_trading_rule_support": candidate["a_share_trading_rule_support"],
                    "small_capital_constraint_fit": candidate["small_capital_constraint_fit"],
                    "transaction_cost_slippage_support": candidate["transaction_cost_slippage_support"],
                    "required_a_share_adaptations": _required_a_share_adaptations(candidate),
                    "fit_verdict": fit_verdict,
                    "research_note": "A-share fit is architectural only; no A-share data was fetched or replayed.",
                }
            )
        )
    return pd.DataFrame(rows)


def _required_a_share_adaptations(candidate: dict[str, str]) -> str:
    needs = [
        "calendar_and_suspension_handling",
        "T_plus_1_sell_rule",
        "lot_size_100_shares",
        "limit_up_down_handling",
        "fees_stamp_tax_commission",
        "small_capital_position_constraints",
    ]
    if candidate["candidate"] in {"AkShare", "Baostock", "Tushare"}:
        return "data_quality_validation; corporate_action_adjustment; " + "; ".join(needs)
    if candidate["candidate"] == "RQAlpha":
        return "license_safe_reimplementation_or_permission; small_capital_constraints; data_adapter_validation"
    if candidate["candidate"] == "LEAN":
        return "custom_a_share_market_adapter; " + "; ".join(needs)
    return "; ".join(needs)


def build_integration_recommendations(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    priority_map = {
        "Qlib": "P1",
        "LEAN": "P1",
        "vectorbt": "P1",
        "RQAlpha": "P1",
        "Backtrader": "P2",
        "Alphalens": "P2",
        "quantstats": "P2",
        "PyPortfolioOpt": "P3",
        "riskfolio-lib": "P4",
        "AkShare": "P1",
        "Baostock": "P1",
        "Tushare": "P3",
        "RD-Agent": "P4",
        "LangGraph": "P4",
        "AutoGen": "P4",
        "CrewAI": "P4",
    }
    next_step_map = {
        "Qlib": "Create an offline metadata-only prototype plan for factor/model workflow after data source selection.",
        "LEAN": "Review event-driven architecture and disable all live/broker paths in any future prototype.",
        "vectorbt": "Run legal review before any prototype; use only for internal research if allowed.",
        "RQAlpha": "Do not integrate directly unless commercial permission is obtained; study A-share rule design.",
        "Backtrader": "Evaluate thin A-share adapter cost against LEAN/RQAlpha architecture lessons.",
        "Alphalens": "Port factor tear-sheet concepts instead of depending on stale package versions.",
        "quantstats": "Wrap only after trustworthy return series and benchmark series are produced.",
        "PyPortfolioOpt": "Prototype after candidate strategies produce credible OOS return streams.",
        "riskfolio-lib": "Revisit after simple portfolio/risk constraints are insufficient.",
        "AkShare": "Move to V7 Step 2 data asset map with endpoint-level provenance checks.",
        "Baostock": "Move to V7 Step 2 baseline daily data quality comparison.",
        "Tushare": "Review terms, token constraints, costs, and commercial data rights before use.",
        "RD-Agent": "Defer until reproducible research pipelines and validation gates exist.",
        "LangGraph": "Defer until orchestration needs are concrete and deterministic.",
        "AutoGen": "Defer until agent evaluation and governance controls exist.",
        "CrewAI": "Defer until agent telemetry, governance, and reproducibility policies exist.",
    }
    for _, row in evaluation.iterrows():
        rows.append(
            _add_forbidden_flags(
                {
                    "candidate": row["candidate"],
                    "recommended_action": row["recommended_action"],
                    "integration_priority": priority_map[row["candidate"]],
                    "prototype_allowed_now": False,
                    "direct_dependency_allowed_now": False,
                    "requires_license_review": _is_high(row["commercial_license_usage_risk"]),
                    "requires_live_feature_isolation": _is_high(row["live_trading_broker_risk"]),
                    "next_evaluation_step": next_step_map[row["candidate"]],
                    "quantpilot_custom_code_boundary": "Focus custom code on integration, A-share adaptation, small-capital constraints, proprietary alpha logic, and orchestration.",
                }
            )
        )
    return pd.DataFrame(rows)


def build_risk_register(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    risk_id = 1
    for _, row in evaluation.iterrows():
        candidate = row["candidate"]
        if _is_high(row["commercial_license_usage_risk"]):
            rows.append(
                _risk_row(
                    risk_id,
                    candidate,
                    "commercial_license_usage_risk",
                    "high",
                    f"{candidate} has high commercial or license uncertainty for product use.",
                    "Require legal/commercial review before install, import, integration, or distribution.",
                )
            )
            risk_id += 1
        if _is_high(row["integration_complexity"]):
            rows.append(
                _risk_row(
                    risk_id,
                    candidate,
                    "integration_complexity_risk",
                    "high",
                    f"{candidate} would add substantial integration and maintenance surface area.",
                    "Prototype only after narrower V7 data/backtest/factor requirements are explicit.",
                )
            )
            risk_id += 1
        if _is_high(row["live_trading_broker_risk"]):
            rows.append(
                _risk_row(
                    risk_id,
                    candidate,
                    "live_trading_broker_risk",
                    "high",
                    f"{candidate} includes or is adjacent to live trading and broker pathways.",
                    "Keep live/broker modules disabled and out of scope until a future live-readiness phase.",
                )
            )
            risk_id += 1
        if row["maintenance_risk"] == "high":
            rows.append(
                _risk_row(
                    risk_id,
                    candidate,
                    "maintenance_risk",
                    "medium",
                    f"{candidate} may have stale dependencies or ecosystem drift.",
                    "Borrow architecture or concepts unless direct dependency compatibility is proven.",
                )
            )
            risk_id += 1
    rows.extend(
        [
            _risk_row(
                risk_id,
                "V7 custom development",
                "custom_engine_reinvention_risk",
                "high",
                "Continuing pure custom development without mature tool evaluation risks low-quality infrastructure.",
                "Apply open-source-first policy before building data, backtest, factor, portfolio, or agent systems.",
            ),
            _risk_row(
                risk_id + 1,
                "Agent frameworks",
                "over_engineering_risk",
                "high",
                "Agent orchestration could distract from data quality, A-share realism, and alpha validation.",
                "Defer RD-Agent, LangGraph, AutoGen, and CrewAI until foundations are stable.",
            ),
        ]
    )
    return pd.DataFrame(rows)


def _risk_row(
    risk_id: int,
    candidate: str,
    risk_type: str,
    severity: str,
    risk_description: str,
    mitigation: str,
) -> dict[str, Any]:
    return _add_forbidden_flags(
        {
            "risk_id": f"OSR-{risk_id:03d}",
            "candidate": candidate,
            "risk_type": risk_type,
            "severity": severity,
            "risk_description": risk_description,
            "mitigation": mitigation,
            "status": "open_for_future_review",
        }
    )


def build_architecture_decision(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (
            "ADR-V7-001",
            "V7 should not continue pure custom development before mature tool evaluation.",
            "accepted",
            "Open-source-first policy reduces reinvention risk and exposes proven design patterns.",
        ),
        (
            "ADR-V7-002",
            "Evaluate Qlib for AI/ML quant research and factor/model workflow.",
            "accepted",
            "Qlib has the strongest fit for AI quant research, but needs local A-share adaptation.",
        ),
        (
            "ADR-V7-003",
            "Evaluate LEAN for robust event-driven architecture while disabling live trading.",
            "accepted",
            "LEAN is mature and architecturally rich, but live/broker pathways are out of scope.",
        ),
        (
            "ADR-V7-004",
            "Evaluate vectorbt only after commercial license review.",
            "accepted",
            "Fast research value is high, but Commons Clause creates product risk.",
        ),
        (
            "ADR-V7-005",
            "Use RQAlpha as architecture study material unless commercial permission is obtained.",
            "accepted",
            "A-share fit is strong, but non-commercial language blocks direct product adoption.",
        ),
        (
            "ADR-V7-006",
            "Evaluate Backtrader as a lightweight adapter prototype candidate.",
            "accepted",
            "It may be useful for a local Python prototype, but not sufficient without A-share extensions.",
        ),
        (
            "ADR-V7-007",
            "Borrow Alphalens-style factor analytics and evaluate quantstats-style performance reporting.",
            "accepted",
            "Factor/performance analytics are needed, but should consume validated data and return series.",
        ),
        (
            "ADR-V7-008",
            "Defer PyPortfolioOpt/riskfolio-lib until credible strategy returns exist.",
            "accepted",
            "Portfolio optimization before alpha validation would optimize noise.",
        ),
        (
            "ADR-V7-009",
            "Evaluate AkShare, Baostock, and Tushare in V7 Step 2 data source selection.",
            "accepted",
            "A-share data quality, cost, stability, and rights are foundational and must be tested separately.",
        ),
        (
            "ADR-V7-010",
            "Defer RD-Agent, LangGraph, AutoGen, and CrewAI until data/backtest/factor foundations are stable.",
            "accepted",
            "Agent systems should orchestrate validated workflows, not compensate for missing market reality.",
        ),
    ]
    return pd.DataFrame(
        [
            _add_forbidden_flags(
                {
                    "decision_id": decision_id,
                    "decision": decision,
                    "decision_status": status,
                    "rationale": rationale,
                    "v7_stack_layer": _architecture_layer(decision_id),
                    "candidate_dependencies_added_now": False,
                    "implementation_allowed_now": False,
                }
            )
            for decision_id, decision, status, rationale in rows
        ]
    )


def _architecture_layer(decision_id: str) -> str:
    return {
        "ADR-V7-001": "policy",
        "ADR-V7-002": "factor_model_workflow",
        "ADR-V7-003": "backtest_architecture",
        "ADR-V7-004": "fast_research",
        "ADR-V7-005": "a_share_backtest_rules",
        "ADR-V7-006": "backtest_adapter",
        "ADR-V7-007": "analytics",
        "ADR-V7-008": "portfolio_risk",
        "ADR-V7-009": "data_source",
        "ADR-V7-010": "agent_orchestration",
    }.get(decision_id, "unknown")


def build_guardrails() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _add_forbidden_flags(
                {
                    "guardrail": guardrail,
                    "status": status,
                    "evidence": evidence,
                }
            )
            for guardrail, status, evidence in GUARDRAILS
        ]
    )


def build_summary(
    inventory: pd.DataFrame,
    evaluation: pd.DataFrame,
    risk_register: pd.DataFrame,
    architecture: pd.DataFrame,
    guardrails: pd.DataFrame,
    run_config: dict[str, Any],
) -> pd.DataFrame:
    forbidden_count = count_forbidden_true_flags(
        [inventory, evaluation, risk_register, architecture, guardrails],
        run_config,
    )
    action_counts = evaluation["recommended_action"].value_counts().to_dict()
    high_license = int(evaluation["commercial_license_usage_risk"].map(_is_high).sum())
    high_integration = int(evaluation["integration_complexity"].map(_is_high).sum())
    high_live = int(evaluation["live_trading_broker_risk"].map(_is_high).sum())
    v7_stack_count = int(
        evaluation["recommended_action"].isin(
            ["wrap_and_integrate", "evaluate_with_prototype", "borrow_architecture_only"]
        ).sum()
    )
    validation_status = "pass" if forbidden_count == 0 and len(inventory) == len(CANDIDATES) else "fail"
    return pd.DataFrame(
        [
            {
                "summary_item": "v7_step1_open_source_quant_stack_audit",
                "reviewed_candidate_count": int(len(inventory)),
                "recommended_adopt_directly_count": int(action_counts.get("adopt_directly", 0)),
                "recommended_wrap_and_integrate_count": int(action_counts.get("wrap_and_integrate", 0)),
                "recommended_evaluate_with_prototype_count": int(action_counts.get("evaluate_with_prototype", 0)),
                "recommended_borrow_architecture_only_count": int(action_counts.get("borrow_architecture_only", 0)),
                "recommended_defer_until_later_count": int(action_counts.get("defer_until_later", 0)),
                "recommended_avoid_for_now_count": int(action_counts.get("avoid_for_now", 0)),
                "high_license_risk_count": high_license,
                "high_integration_complexity_count": high_integration,
                "high_live_trading_risk_count": high_live,
                "v7_recommended_stack_count": v7_stack_count,
                "market_data_fetch_count": 0,
                "broker_connected_count": 0,
                "execution_allowed_count": 0,
                "live_trading_count": 0,
                "real_order_submission_count": 0,
                "forbidden_true_flag_count": forbidden_count,
                "trading_ready": False,
                "validation_status": validation_status,
                "conclusion": "open_source_quant_stack_audit_completed_research_only",
                "recommended_next_step": "V7 Step 2 A-share Data Asset Map / Data Source Selection",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    inventory: pd.DataFrame,
    evaluation: pd.DataFrame,
    a_share_fit: pd.DataFrame,
    recommendations: pd.DataFrame,
    risk_register: pd.DataFrame,
    architecture: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0]
    conclusion_lines = [
        "V7 should not continue pure custom development without evaluating mature tools.",
        "Qlib should be evaluated for AI/ML quant research and factor/model workflow.",
        "LEAN should be evaluated for robust event-driven architecture, with live trading disabled.",
        "vectorbt should be evaluated only under license-reviewed research constraints.",
        "RQAlpha and Backtrader should be evaluated for A-share/backtest adapter lessons, with RQAlpha treated as architecture-only unless commercial permission is obtained.",
        "Alphalens and quantstats-style analytics should inform factor and performance reporting.",
        "PyPortfolioOpt and riskfolio-lib should wait until credible return streams and risk inputs exist.",
        "AkShare, Baostock, and Tushare belong in V7 Step 2 data-source selection and rights review.",
        "RD-Agent, LangGraph, AutoGen, and CrewAI should be deferred until data/backtest/factor foundations are stable.",
        "Custom code should focus on integration, A-share adaptation, small-capital constraints, proprietary alpha logic, and orchestration.",
    ]
    return "\n".join(
        [
            "# V7 Step 1 Open-source Quant Stack Audit / Framework Selection",
            "",
            "## Executive Summary",
            "This is a research-only architecture decision layer for QuantPilot-AI V7.",
            "It evaluates candidate open-source tools before building data, backtesting, factor, portfolio, risk, workflow, or later agent infrastructure.",
            "No package was installed, no candidate framework was imported, no market data was fetched, no backtest was run, no model was trained, no broker was connected, and no trading-ready claim is made.",
            "",
            "## Summary",
            f"- Reviewed candidates: {row['reviewed_candidate_count']}",
            f"- Adopt directly: {row['recommended_adopt_directly_count']}",
            f"- Wrap and integrate: {row['recommended_wrap_and_integrate_count']}",
            f"- Evaluate with prototype: {row['recommended_evaluate_with_prototype_count']}",
            f"- Borrow architecture only: {row['recommended_borrow_architecture_only_count']}",
            f"- Defer until later: {row['recommended_defer_until_later_count']}",
            f"- Avoid for now: {row['recommended_avoid_for_now_count']}",
            f"- High license risk: {row['high_license_risk_count']}",
            f"- High integration complexity: {row['high_integration_complexity_count']}",
            f"- High live trading risk: {row['high_live_trading_risk_count']}",
            f"- V7 recommended stack count: {row['v7_recommended_stack_count']}",
            f"- Market data fetch count: {row['market_data_fetch_count']}",
            f"- Broker connected count: {row['broker_connected_count']}",
            f"- Execution allowed count: {row['execution_allowed_count']}",
            f"- Live trading count: {row['live_trading_count']}",
            f"- Real order submission count: {row['real_order_submission_count']}",
            f"- Forbidden true flag count: {row['forbidden_true_flag_count']}",
            f"- Trading ready: {row['trading_ready']}",
            f"- Validation status: {row['validation_status']}",
            f"- Conclusion: {row['conclusion']}",
            f"- Recommended next step: {row['recommended_next_step']}",
            "",
            "## Core Conclusions",
            "\n".join(f"- {line}" for line in conclusion_lines),
            "",
            "## Candidate Inventory",
            _table(inventory, "No candidate inventory rows were generated."),
            "",
            "## Evaluation Matrix",
            _table(evaluation, "No evaluation rows were generated."),
            "",
            "## A-share Fit Matrix",
            _table(a_share_fit, "No A-share fit rows were generated."),
            "",
            "## Integration Recommendations",
            _table(recommendations, "No recommendation rows were generated."),
            "",
            "## Risk Register",
            _table(risk_register, "No risk rows were generated."),
            "",
            "## Architecture Decisions",
            _table(architecture, "No architecture decision rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-only Boundary",
            "This audit is not financial advice and does not establish real alpha evidence, market replay validity, realistic execution quality, broker readiness, or trading readiness.",
            "",
        ]
    )


def generate_open_source_quant_stack_audit_outputs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    inventory = build_candidate_inventory()
    evaluation = build_evaluation_matrix()
    a_share_fit = build_a_share_fit_matrix()
    recommendations = build_integration_recommendations(evaluation)
    risk_register = build_risk_register(evaluation)
    architecture = build_architecture_decision(evaluation)
    guardrails = build_guardrails()
    run_config = {
        "output_dir": str(output_path),
        "scope": "V7 Step 1 framework selection and architecture decision layer only",
        "candidate_count": int(len(CANDIDATES)),
        "market_data_fetch": False,
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "no_package_install": True,
        "no_external_framework_import": True,
        "framework_selection_only": True,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    summary = build_summary(inventory, evaluation, risk_register, architecture, guardrails, run_config)
    report = build_report(
        summary,
        inventory,
        evaluation,
        a_share_fit,
        recommendations,
        risk_register,
        architecture,
        guardrails,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    output_files["run_config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    inventory.to_csv(output_files["inventory"], index=False)
    evaluation.to_csv(output_files["evaluation"], index=False)
    a_share_fit.to_csv(output_files["a_share_fit"], index=False)
    recommendations.to_csv(output_files["recommendations"], index=False)
    risk_register.to_csv(output_files["risk_register"], index=False)
    architecture.to_csv(output_files["architecture"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    summary.to_csv(output_files["summary"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    return {
        "summary": summary,
        "inventory": inventory,
        "evaluation": evaluation,
        "a_share_fit": a_share_fit,
        "recommendations": recommendations,
        "risk_register": risk_register,
        "architecture": architecture,
        "guardrails": guardrails,
        "report": report,
        "run_config": run_config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
