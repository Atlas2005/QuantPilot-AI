import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/global_open_source_replacement_audit_real_v1")
DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_STEP1_DIR = Path("outputs/open_source_quant_stack_audit_real_v1")
DEFAULT_STEP2_DIR = Path("outputs/a_share_data_asset_map_real_v1")
DEFAULT_V6_CLOSURE_DIR = Path("outputs/simulation_hardening_closure_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "project_inventory": "project_module_inventory.csv",
    "candidate_inventory": "open_source_candidate_expanded_inventory.csv",
    "replacement_matrix": "module_replacement_matrix.csv",
    "retention": "past_work_retention_decision.csv",
    "summary": "architecture_reassessment_summary.csv",
    "roadmap": "open_source_integration_roadmap.csv",
    "agent_audit": "agent_tooling_ecosystem_audit.csv",
    "profitability": "profitability_alignment_review.csv",
    "risk_register": "replacement_risk_register.csv",
    "guardrails": "global_reassessment_guardrails.csv",
    "report": "global_open_source_replacement_audit_report.md",
}

FORBIDDEN_FLAGS = [
    "market_data_fetch",
    "external_api_call",
    "package_install",
    "broker_connected",
    "execution_allowed",
    "live_trading",
    "real_order_submission",
    "trading_ready",
]

GUARDRAILS = [
    ("no_package_install", "confirmed", "No package manager command is executed and no dependency file is changed."),
    ("no_external_framework_import", "confirmed", "No open-source framework candidate is imported."),
    ("no_market_data_fetch", "confirmed", "No market data fetch, downloader, scraper, or data API is called."),
    ("no_external_api_call", "confirmed", "The audit reads local repository metadata only and performs no external API calls."),
    ("no_live_data", "confirmed", "No live quote, stream, broker feed, or real-time endpoint is used."),
    ("no_backtest_execution", "confirmed", "No backtest, replay, simulation, or strategy engine is executed."),
    ("no_model_training", "confirmed", "No model training, fitting, optimization, or agent experiment is run."),
    ("no_threshold_change", "confirmed", "No threshold, gate, or strategy parameter is changed."),
    ("no_feature_engineering_change", "confirmed", "No factor or feature builder is modified or executed."),
    ("no_broker_sdk_import", "confirmed", "No broker SDK or live trading package is imported."),
    ("no_broker_connection", "confirmed", "No broker connection path exists in this step."),
    ("no_order_execution", "confirmed", "No order executor, order generator, or order submission path is invoked."),
    ("no_real_order_submission", "confirmed", "No real order submission path exists."),
    ("no_trading_ready_upgrade", "confirmed", "All outputs explicitly keep trading_ready=False."),
    ("architecture_reassessment_only", "confirmed", "This step produces architecture decisions only."),
    ("no_sunk_cost_bias", "confirmed", "Past effort is not treated as a reason to keep inferior custom infrastructure."),
    ("open_source_first_policy_applied", "confirmed", "Mature tools are evaluated before custom infrastructure is promoted."),
    ("educational_research_only", "confirmed", "The report is research-only and is not financial advice."),
]

OPEN_SOURCE_CANDIDATES = [
    ("Qlib", "quant_data_backtest", "factor/model workflow", "MIT", "high", "medium", "medium", "high", "high", "Evaluate for AI quant workflow after data contract."),
    ("LEAN", "quant_data_backtest", "event-driven backtest/live architecture", "Apache-2.0", "high", "low", "high", "high", "medium", "Borrow/adapt architecture; keep live disabled."),
    ("vectorbt", "quant_data_backtest", "fast vectorized research", "Apache-2.0 plus Commons Clause risk", "high", "high", "medium", "medium", "medium", "Use only after license review for research tournaments."),
    ("vn.py / VeighNa", "quant_data_backtest", "trading system and gateway architecture", "open-source with gateway/commercial review needed", "high", "medium", "high", "high", "medium", "Borrow broker/gateway architecture later; do not enable trading."),
    ("Backtrader", "quant_data_backtest", "Python backtest engine", "open-source, verify current status", "medium", "medium", "medium", "medium", "low", "Prototype adapter only if simpler than LEAN/Qlib."),
    ("RQAlpha", "quant_data_backtest", "A-share-oriented backtest architecture", "non-commercial language risk", "high", "high", "high", "high", "medium", "Borrow architecture unless commercial permission exists."),
    ("Zipline / zipline-reloaded", "quant_data_backtest", "pipeline/backtest architecture", "open-source, ecosystem drift risk", "medium", "medium", "medium", "medium", "low", "Borrow pipeline ideas, not core now."),
    ("Alphalens / Alphalens Reloaded", "quant_data_backtest", "factor tear sheets", "open-source, compatibility varies", "medium", "low", "medium", "medium", "low", "Borrow factor analytics concepts."),
    ("quantstats", "quant_data_backtest", "performance analytics", "open-source, verify dependencies", "medium", "low", "low", "medium", "low", "Wrap later for reports after validated returns."),
    ("PyPortfolioOpt", "quant_data_backtest", "portfolio optimization", "MIT", "high", "low", "medium", "medium", "low", "Evaluate after credible OOS return streams exist."),
    ("riskfolio-lib", "quant_data_backtest", "portfolio risk optimization", "BSD-style, heavy dependencies", "high", "low", "high", "medium", "low", "Defer advanced risk allocation."),
    ("OpenBB", "quant_data_backtest", "financial data/research platform", "open-source with data-provider terms", "high", "medium", "high", "medium", "medium", "Evaluate as research/data UX reference, not core now."),
    ("AkShare", "quant_data_backtest", "A-share data access", "MIT plus endpoint rights risk", "high", "medium", "medium", "high", "low", "Primary future data prototype candidate."),
    ("Baostock", "quant_data_backtest", "A-share daily data access", "BSD reported; service terms review", "medium", "medium", "low", "high", "low", "Secondary future data prototype candidate."),
    ("Tushare", "quant_data_backtest", "token-based A-share data", "commercial/token risk", "high", "high", "medium", "high", "low", "Prototype only after legal/commercial review."),
    ("Pandera", "quant_data_backtest", "dataframe schema validation", "MIT", "high", "low", "low", "high", "low", "Strong candidate for V7 Step 3 local data contract."),
    ("Great Expectations", "quant_data_backtest", "data quality framework", "Apache-2.0", "high", "low", "high", "medium", "low", "Borrow checks or evaluate if Pandera is insufficient."),
    ("Polars", "quant_data_backtest", "fast dataframe engine", "MIT", "high", "low", "medium", "high", "low", "Evaluate for scalable data processing after contracts."),
    ("DuckDB", "quant_data_backtest", "embedded analytics database", "MIT", "high", "low", "medium", "high", "low", "Optional query layer over Parquet."),
    ("PyArrow / Parquet", "quant_data_backtest", "columnar storage", "Apache-2.0", "high", "low", "medium", "high", "low", "Likely future canonical storage foundation."),
    ("TradingAgents", "ai_agent_workflow", "multi-agent trading research pattern", "verify license and claims", "medium", "medium", "high", "low", "high", "Study later; do not drive foundation."),
    ("FinRobot", "ai_agent_workflow", "financial AI agent workflow", "verify license and model/data terms", "medium", "medium", "high", "low", "high", "Study later after deterministic research harness."),
    ("FinGPT", "ai_agent_workflow", "financial LLM models/data recipes", "license/model/data restrictions vary", "medium", "high", "high", "low", "high", "Defer; possible sentiment/model research reference."),
    ("RD-Agent", "ai_agent_workflow", "research automation", "MIT; platform constraints", "medium", "low", "high", "medium", "high", "Defer until data/backtest/factor foundations are stable."),
    ("LangGraph", "ai_agent_workflow", "stateful agent graph", "MIT", "high", "low", "medium", "medium", "medium", "Good future orchestrator candidate after foundations."),
    ("AutoGen", "ai_agent_workflow", "multi-agent framework", "MIT code plus content license review", "high", "medium", "medium", "medium", "high", "Defer; strong but overengineering risk now."),
    ("CrewAI", "ai_agent_workflow", "agent workflow framework", "MIT; telemetry/config review", "high", "low", "medium", "medium", "high", "Defer; compare with LangGraph later."),
    ("OpenAI Agents SDK", "ai_agent_workflow", "agent runtime/tool orchestration", "commercial API terms", "high", "medium", "medium", "high", "medium", "Evaluate later for product-grade controlled agents."),
    ("OpenAI Skills", "ai_agent_workflow", "repeatable task packaging", "platform/product terms", "high", "medium", "low", "high", "low", "Useful for codified workflows and human-reviewed tools."),
    ("Ruflo", "ai_agent_workflow", "workflow/tooling candidate", "verify maturity/license", "low", "medium", "medium", "low", "medium", "Avoid core dependency until maturity is proven."),
    ("Warp", "ai_agent_workflow", "developer/agent workflow tooling", "commercial product terms", "high", "medium", "medium", "low", "medium", "Developer UX reference only, not architecture foundation."),
    ("Scrapling", "ai_agent_workflow", "web extraction/scraping", "verify license/site terms", "medium", "medium", "medium", "medium", "medium", "Consider later for news/announcements instead of custom scraping."),
    ("MCP ecosystem placeholder", "ai_agent_workflow", "tool/resource connector standard", "varies by server", "high", "medium", "medium", "high", "medium", "Future connector boundary for agents and tools."),
]

MODULE_AREAS = [
    ("data_loading", "src/real_data_loader.py; src/data_loader.py", "Load demo/local/real A-share data.", "medium", "high", "AkShare; Baostock; Tushare; OpenBB; PyArrow / Parquet", "AkShare", "wrap_open_source", "Existing custom loaders should become thin adapters; source reliability and rights need validation.", "high", "high", "yes", "medium", "medium", "P1", "Keep local fixtures, design source adapters after V7 Step 3 contract."),
    ("factor_building", "src/factor_builder.py; src/build_factor_dataset.py", "Create technical and simple feature datasets.", "medium", "high", "Qlib; Alphalens / Alphalens Reloaded; Polars; Pandera", "Qlib", "wrap_open_source", "Qlib-style workflow and factor analytics are stronger foundations than ad hoc feature scripts.", "high", "medium", "yes", "low", "high", "P2", "Retain factor ideas as fixtures while designing Qlib-compatible factor contracts."),
    ("model_training_prediction_evaluation", "src/model_trainer.py; src/model_predictor.py; src/model_evaluator.py", "Train/evaluate baseline ML models.", "medium", "high", "Qlib; RD-Agent; OpenAI Skills", "Qlib", "replace_with_open_source", "Custom sklearn scripts are useful demos but not enough for robust alpha research workflow tracking.", "high", "high", "yes", "low", "high", "P2", "Move toward Qlib experiment workflow after data quality foundation."),
    ("backtesting", "src/backtester.py; src/run_stock_backtest.py", "Custom long-only backtest and experiment runners.", "medium", "high", "LEAN; vectorbt; Backtrader; RQAlpha; Zipline / zipline-reloaded; Qlib", "LEAN", "replace_with_open_source", "Custom engine must not become long-term core; mature engines have stronger event/fill architecture.", "high", "high", "yes", "medium", "high", "P1", "Evaluate LEAN/Qlib/vectorbt/RQAlpha architecture after data contract."),
    ("transaction_cost_handling", "src/backtester.py; src/trade_metrics.py", "Model commission, tax, slippage, and trade metrics.", "low", "high", "LEAN; RQAlpha; vn.py / VeighNa", "LEAN", "borrow_architecture_only", "A-share cost/fill realism is too important for simplistic custom logic.", "high", "high", "yes", "medium", "high", "P1", "Borrow cost/fill model architecture; implement A-share-specific adapter tests."),
    ("capital_constraint", "src/capital_constraint_engine.py", "Small-capital feasibility constraints.", "medium", "medium", "custom A-share rules; PyPortfolioOpt", "custom A-share rules", "keep_as_core", "Small-capital constraints are project-specific and should remain custom, but validated against engine outputs.", "high", "medium", "yes", "low", "medium", "P1", "Keep as core project-specific layer with stronger tests."),
    ("tradable_universe", "src/tradable_universe_filter.py", "Filter tradable symbols for small capital and A-share constraints.", "medium", "medium", "AkShare; Tushare; Pandera; Qlib", "Pandera", "keep_as_core", "Universe rules are A-share/product-specific, but need validated data contracts.", "high", "medium", "yes", "medium", "medium", "P1", "Keep rules custom, validate inputs through Step 3 data contract."),
    ("position_sizing", "src/position_sizing_engine.py", "Position sizing under cash and lot constraints.", "medium", "medium", "PyPortfolioOpt; riskfolio-lib; custom A-share rules", "custom A-share rules", "keep_as_core", "Lot size, minimum fees, cash limits, and small account behavior are product-specific.", "high", "medium", "yes", "low", "medium", "P2", "Keep as core after integration with realistic backtest engine."),
    ("exit_engine", "src/exit_engine.py", "Stop-loss/take-profit/max-holding planning.", "medium", "medium", "LEAN; Backtrader; vectorbt", "LEAN", "borrow_architecture_only", "Exit policy can remain proprietary, but execution semantics should follow mature engine patterns.", "medium", "medium", "yes", "low", "medium", "P2", "Keep rules as alpha logic; align execution semantics with chosen engine."),
    ("paper_ledger", "src/paper_trading_ledger.py", "Local paper ledger scaffold.", "low", "high", "LEAN; vn.py / VeighNa; custom audit ledger", "custom audit ledger", "keep_as_fixture_or_regression_test", "Useful audit record, not a production paper trading feedback system.", "medium", "medium", "yes", "medium", "medium", "P3", "Downgrade to regression fixture until broker-neutral feedback design exists."),
    ("order_generator", "src/semi_auto_order_generator.py", "Semi-auto local order plan generation.", "low", "high", "vn.py / VeighNa; LEAN", "vn.py / VeighNa", "deprecate_later", "Order generation before validated data/backtest/live readiness is dangerous and not core now.", "low", "high", "yes", "high", "high", "P4", "Freeze as historical artifact; do not extend before live-readiness phase."),
    ("broker_research", "src/broker_integration_research.py", "Broker integration research notes.", "medium", "medium", "vn.py / VeighNa; LEAN; MCP ecosystem placeholder", "vn.py / VeighNa", "borrow_architecture_only", "Broker gateways should borrow mature gateway architecture but remain disabled.", "low", "medium", "yes", "high", "high", "P4", "Keep research-only notes; no broker SDK work."),
    ("monitoring_reporting", "src/monitoring_reporting_layer.py; src/report_generator.py", "Output monitoring, alerts, and reports.", "medium", "medium", "quantstats; OpenBB; Great Expectations", "quantstats", "wrap_open_source", "Performance reporting should wrap mature analytics once validated returns exist.", "medium", "medium", "partial", "medium", "medium", "P3", "Keep current reports as audit UI; later wrap quantstats-style outputs."),
    ("v6_validation_simulation_hardening", "src/output_schema_validator.py; src/cross_step_dependency_validator.py; src/simulation_hardening_closure.py", "Validation, evidence, reproducibility, and synthetic hardening records.", "high", "low", "Pandera; Great Expectations", "Pandera", "keep_as_guardrail", "V6 work is valuable as governance and regression evidence, not as alpha proof.", "medium", "high", "partial", "low", "medium", "P1", "Preserve as guardrail suite and migrate schema checks toward Pandera where useful."),
    ("v7_open_source_audit", "src/open_source_quant_stack_audit.py", "Open-source stack selection audit.", "high", "low", "custom architecture decision records", "custom architecture decision records", "keep_as_guardrail", "This is a decision record and should guide future replacement choices.", "medium", "high", "partial", "low", "low", "P1", "Keep as architecture guardrail; update when prototypes produce evidence."),
    ("v7_data_asset_map", "src/a_share_data_asset_map.py", "A-share data source and asset map.", "high", "low", "Pandera; DuckDB; PyArrow / Parquet", "Pandera", "keep_as_guardrail", "This is a foundation decision record, not a data engine.", "high", "high", "yes", "low", "low", "P1", "Keep as guardrail feeding Step 3 contract."),
    ("future_data_quality_validator", "planned", "Validate local data contracts and market reality fields.", "not_started", "high", "Pandera; Great Expectations; Polars", "Pandera", "wrap_open_source", "Data quality should not be custom-only; mature schema validators reduce risk.", "high", "high", "yes", "low", "medium", "P1", "Implement V7 Step 3 around Pandera-style local contracts without installing yet."),
    ("future_realistic_a_share_rule_engine", "planned", "A-share T+1, lots, limit, suspension, fees, fills.", "not_started", "high", "LEAN; RQAlpha; vn.py / VeighNa", "LEAN", "borrow_architecture_only", "Execution realism should borrow mature event/fill architecture while adapting A-share rules.", "high", "high", "yes", "medium", "high", "P1", "Design adapter after data validator."),
    ("future_strategy_tournament", "planned", "Compare many strategies fairly under OOS and costs.", "not_started", "high", "Qlib; vectorbt; OpenBB", "Qlib", "wrap_open_source", "Strategy tournament should use mature experiment workflow and fast research where license permits.", "high", "high", "yes", "medium", "high", "P2", "Prototype Qlib/vectorbt-style tournament after validated data."),
    ("future_walk_forward_oos_validation", "planned", "Locked walk-forward and OOS validation protocol.", "not_started", "high", "Qlib; sklearn model_selection; MLflow-style records", "Qlib", "replace_with_open_source", "OOS validation must be reproducible and hard to game; Qlib workflow is stronger than ad hoc scripts.", "high", "high", "yes", "low", "high", "P2", "Specify protocol before any strategy promotion."),
    ("future_portfolio_risk_engine", "planned", "Portfolio allocation and risk controls.", "not_started", "medium", "PyPortfolioOpt; riskfolio-lib; LEAN", "PyPortfolioOpt", "defer_until_foundation_ready", "Portfolio optimization before credible alpha returns would optimize noise.", "medium", "medium", "yes", "low", "medium", "P3", "Defer until OOS return streams exist."),
    ("future_paper_trading_feedback_loop", "planned", "Forward paper feedback collection.", "not_started", "high", "LEAN; vn.py / VeighNa; custom audit ledger", "LEAN", "defer_until_foundation_ready", "Paper feedback is useful only after validated data and backtest realism.", "medium", "medium", "yes", "high", "high", "P4", "Design later without enabling broker/live paths."),
    ("future_multi_agent_system", "planned", "DeepSeek-style research/validation/orchestration agents.", "not_started", "high", "RD-Agent; LangGraph; AutoGen; CrewAI; OpenAI Agents SDK; TradingAgents; FinRobot", "LangGraph", "defer_until_foundation_ready", "Agents should orchestrate validated workflows, not compensate for weak foundations.", "medium", "medium", "partial", "medium", "high", "P4", "Defer; evaluate RD-Agent/LangGraph/Agents SDK later."),
    ("future_web_news_sentiment_collection", "planned", "News, announcements, and sentiment collection.", "not_started", "high", "Scrapling; FinGPT; OpenBB", "Scrapling", "defer_until_foundation_ready", "Do not custom scrape now; structured market data and legal source policy come first.", "medium", "medium", "partial", "high", "medium", "P4", "Consider Scrapling later under site terms and source governance."),
    ("future_agent_skills_tooling", "planned", "Reusable agent skills, MCP tools, and workflow packaging.", "not_started", "medium", "OpenAI Skills; MCP ecosystem placeholder; OpenAI Agents SDK; Ruflo; Warp", "OpenAI Skills", "defer_until_foundation_ready", "Tooling is valuable only after stable deterministic tasks exist.", "medium", "medium", "partial", "medium", "medium", "P4", "Codify validated workflows as skills later."),
]


def _add_forbidden_flags(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    for flag in FORBIDDEN_FLAGS:
        updated[flag] = False
    return updated


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


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


def _is_high(value: Any) -> bool:
    return str(value).strip().lower() == "high"


def count_forbidden_true_flags(frames: list[pd.DataFrame], config: dict[str, Any]) -> int:
    count = 0
    for frame in frames:
        for flag in FORBIDDEN_FLAGS:
            if flag in frame:
                count += int(frame[flag].map(_is_true).sum())
    for flag in FORBIDDEN_FLAGS:
        count += int(_is_true(config.get(flag)))
    return count


def build_repo_context(
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    step1_dir: str | Path = DEFAULT_STEP1_DIR,
    step2_dir: str | Path = DEFAULT_STEP2_DIR,
    v6_closure_dir: str | Path = DEFAULT_V6_CLOSURE_DIR,
) -> dict[str, Any]:
    root = Path(project_root)
    src_dir = root / "src"
    src_files = sorted(path.name for path in src_dir.glob("*.py")) if src_dir.exists() else []
    readme_exists = (root / "README.md").exists()
    app_exists = (root / "app.py").exists()
    return {
        "project_root": str(root),
        "readme_exists": readme_exists,
        "app_exists": app_exists,
        "src_py_file_count": len(src_files),
        "src_py_files": "; ".join(src_files),
        "step1_summary_rows": len(_read_csv(Path(step1_dir) / "open_source_stack_audit_summary.csv")),
        "step2_summary_rows": len(_read_csv(Path(step2_dir) / "a_share_data_asset_map_summary.csv")),
        "v6_gap_rows": len(_read_csv(Path(v6_closure_dir) / "v6_remaining_gap_register.csv")),
    }


def _example_status(examples: str, src_files: set[str]) -> str:
    if examples == "planned":
        return "future_planned"
    names = [Path(item.strip()).name for item in examples.split(";") if item.strip().startswith("src/")]
    if not names:
        return "metadata_only"
    found = [name for name in names if name in src_files]
    if len(found) == len(names):
        return "all_examples_present"
    if found:
        return "some_examples_present"
    return "examples_missing"


def build_project_module_inventory(context: dict[str, Any]) -> pd.DataFrame:
    src_files = set(context["src_py_files"].split("; ")) if context["src_py_files"] else set()
    rows = []
    for index, area in enumerate(MODULE_AREAS, start=1):
        (
            module_area,
            examples,
            purpose,
            maturity,
            risk,
            candidates,
            best,
            mode,
            reason,
            profit,
            risk_reduction,
            a_share,
            license_risk,
            complexity,
            priority,
            next_action,
        ) = area
        rows.append(
            _add_forbidden_flags(
                {
                    "module_area_id": f"QPM-{index:03d}",
                    "current_quantpilot_module_area": module_area,
                    "current_custom_module_examples": examples,
                    "current_module_purpose": purpose,
                    "repo_metadata_status": _example_status(examples, src_files),
                    "maturity_of_existing_custom_work": maturity,
                    "risk_of_continuing_custom": risk,
                    "recommended_priority": priority,
                }
            )
        )
    return pd.DataFrame(rows)


def build_open_source_candidate_expanded_inventory() -> pd.DataFrame:
    rows = []
    for index, candidate in enumerate(OPEN_SOURCE_CANDIDATES, start=1):
        (
            name,
            category,
            purpose,
            license_note,
            maturity,
            license_risk,
            integration_complexity,
            a_share_fit,
            agent_overengineering_risk,
            role,
        ) = candidate
        rows.append(
            _add_forbidden_flags(
                {
                    "candidate_id": f"OSC-{index:03d}",
                    "candidate_name": name,
                    "candidate_category": category,
                    "candidate_primary_purpose": purpose,
                    "static_license_or_commercial_note": license_note,
                    "maturity_level": maturity,
                    "commercial_or_license_risk": license_risk,
                    "integration_complexity": integration_complexity,
                    "a_share_fit": a_share_fit,
                    "agent_overengineering_risk": agent_overengineering_risk,
                    "recommended_quantpilot_role": role,
                    "installed_now": False,
                    "imported_now": False,
                }
            )
        )
    return pd.DataFrame(rows)


def build_module_replacement_matrix() -> pd.DataFrame:
    rows = []
    for area in MODULE_AREAS:
        (
            module_area,
            examples,
            purpose,
            maturity,
            risk,
            candidates,
            best,
            mode,
            reason,
            profit,
            risk_reduction,
            a_share,
            license_risk,
            complexity,
            priority,
            next_action,
        ) = area
        rows.append(
            _add_forbidden_flags(
                {
                    "current_quantpilot_module_area": module_area,
                    "current_custom_module_examples": examples,
                    "current_module_purpose": purpose,
                    "maturity_of_existing_custom_work": maturity,
                    "risk_of_continuing_custom": risk,
                    "open_source_replacement_candidates": candidates,
                    "best_candidate": best,
                    "replacement_mode": mode,
                    "reason": reason,
                    "expected_profitability_impact": profit,
                    "expected_engineering_risk_reduction": risk_reduction,
                    "a_share_adaptation_needed": a_share,
                    "commercial_or_license_risk": license_risk,
                    "integration_complexity": complexity,
                    "recommended_priority": priority,
                    "next_action": next_action,
                }
            )
        )
    return pd.DataFrame(rows)


def build_past_work_retention_decision(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in matrix.iterrows():
        mode = row["replacement_mode"]
        if mode in {"keep_as_guardrail", "keep_as_fixture_or_regression_test"}:
            value = "valuable_as_regression_or_governance"
        elif mode in {"replace_with_open_source", "wrap_open_source", "borrow_architecture_only"}:
            value = "valuable_as_design_record_not_long_term_core"
        elif mode == "deprecate_later":
            value = "historical_artifact_do_not_extend"
        elif mode == "keep_as_core":
            value = "project_specific_core_after_validation"
        else:
            value = "defer_until_foundation_ready"
        rows.append(
            _add_forbidden_flags(
                {
                    "module_area": row["current_quantpilot_module_area"],
                    "retention_decision": value,
                    "should_be_long_term_core": mode == "keep_as_core",
                    "should_be_regression_test_or_fixture": mode in {"keep_as_guardrail", "keep_as_fixture_or_regression_test"},
                    "should_not_be_long_term_core": mode in {"replace_with_open_source", "wrap_open_source", "borrow_architecture_only", "deprecate_later"},
                    "sunk_cost_warning": "Past effort does not justify keeping inferior architecture.",
                    "retention_reason": row["reason"],
                }
            )
        )
    return pd.DataFrame(rows)


def build_open_source_integration_roadmap() -> pd.DataFrame:
    rows = [
        ("ROAD-001", "V7 Step 3", "Data Quality Validator / Local Data Contract", "Pandera; PyArrow / Parquet; DuckDB optional", "Implement schema and market-reality checks before any fetch/backtest expansion.", "P1"),
        ("ROAD-002", "V7 Step 4", "Controlled data source prototypes", "AkShare; Baostock; Tushare only after review", "Use explicit non-CI commands and compare source quality; no source trusted by default.", "P1"),
        ("ROAD-003", "V7 Step 5", "Realistic A-share rule engine design", "LEAN; RQAlpha; vn.py / VeighNa", "Borrow event/fill/gateway architecture while implementing A-share adaptations.", "P1"),
        ("ROAD-004", "V7 Step 6", "Research workflow prototype", "Qlib; vectorbt after license review", "Evaluate factor/model workflow and fast strategy tournament mechanics.", "P2"),
        ("ROAD-005", "V7 Step 7", "Walk-forward/OOS validation protocol", "Qlib workflow concepts", "Lock promotion rules before any profitability claim.", "P2"),
        ("ROAD-006", "V7 Step 8", "Performance and portfolio layer", "quantstats; PyPortfolioOpt; riskfolio-lib", "Add only after credible OOS return streams exist.", "P3"),
        ("ROAD-007", "Later", "Paper feedback and broker-neutral logging", "LEAN; vn.py / VeighNa; custom audit ledger", "Design later with broker/live paths disabled until live-readiness phase.", "P4"),
        ("ROAD-008", "Later", "Multi-agent research orchestration", "LangGraph; RD-Agent; OpenAI Agents SDK; OpenAI Skills; MCP", "Agents orchestrate validated workflows after foundations are stable.", "P4"),
        ("ROAD-009", "Later", "News/sentiment collection", "Scrapling; FinGPT; OpenBB", "Use legal source governance; avoid direct custom scraping until needed.", "P4"),
    ]
    return pd.DataFrame([_add_forbidden_flags({
        "roadmap_id": rid,
        "phase": phase,
        "roadmap_item": item,
        "open_source_candidates": candidates,
        "roadmap_decision": decision,
        "recommended_priority": priority,
    }) for rid, phase, item, candidates, decision, priority in rows])


def build_agent_tooling_ecosystem_audit() -> pd.DataFrame:
    rows = [
        ("TradingAgents", "defer_until_foundation_ready", "Useful pattern for multi-agent trading research, but it cannot replace data quality, execution realism, or OOS proof."),
        ("FinRobot", "defer_until_foundation_ready", "Potential financial-agent reference, but deterministic validation must come first."),
        ("FinGPT", "defer_until_foundation_ready", "May influence future news/sentiment/model work; model/data rights and evaluation risk are high."),
        ("RD-Agent", "defer_until_foundation_ready", "Relevant for research automation after Qlib/data/backtest workflows exist."),
        ("LangGraph", "borrow_architecture_only", "Strong candidate for later stateful orchestration, but not before stable foundations."),
        ("AutoGen", "defer_until_foundation_ready", "Powerful but high overengineering risk for current stage."),
        ("CrewAI", "defer_until_foundation_ready", "Easy agent workflows, but quantify reproducibility and governance before use."),
        ("OpenAI Agents SDK", "defer_until_foundation_ready", "Potential product-grade agent runtime after tool boundaries and evals exist."),
        ("OpenAI Skills", "borrow_architecture_only", "Good packaging model for repeatable validated research tasks."),
        ("Ruflo", "avoid_for_now", "Insufficient proven fit for this quant foundation stage."),
        ("Warp", "borrow_architecture_only", "Developer UX inspiration only, not quant architecture."),
        ("Scrapling", "defer_until_foundation_ready", "Consider for later news/announcement collection instead of custom scraping."),
        ("MCP ecosystem placeholder", "borrow_architecture_only", "Good connector boundary for future tools/resources, with server-specific risk review."),
    ]
    return pd.DataFrame([_add_forbidden_flags({
        "tooling_candidate": name,
        "recommended_action": action,
        "agent_overengineering_risk": "high" if action == "defer_until_foundation_ready" else "medium",
        "roadmap_impact": impact,
    }) for name, action, impact in rows])


def build_profitability_alignment_review(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in matrix.iterrows():
        issue = row["expected_profitability_impact"] in {"low", "medium"} and row["recommended_priority"] in {"P1", "P2"}
        rows.append(
            _add_forbidden_flags(
                {
                    "module_area": row["current_quantpilot_module_area"],
                    "profitability_alignment": row["expected_profitability_impact"],
                    "alignment_issue": bool(issue),
                    "issue_reason": "Priority work must improve alpha validation, market realism, or risk control." if issue else "Aligned with staged profitability evidence building.",
                    "must_not_claim_profitability": True,
                    "recommended_action": row["next_action"],
                }
            )
        )
    rows.append(
        _add_forbidden_flags(
            {
                "module_area": "global_sunk_cost_warning",
                "profitability_alignment": "high",
                "alignment_issue": False,
                "issue_reason": "Past effort is not evidence of alpha, market realism, or product readiness.",
                "must_not_claim_profitability": True,
                "recommended_action": "Replace, wrap, downgrade, or deprecate custom work whenever mature tools are stronger.",
            }
        )
    )
    return pd.DataFrame(rows)


def build_replacement_risk_register(matrix: pd.DataFrame, candidates: pd.DataFrame, agent_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    risk_id = 1
    for _, row in matrix.iterrows():
        if row["recommended_priority"] == "P1" and row["replacement_mode"] in {"replace_with_open_source", "wrap_open_source", "borrow_architecture_only"}:
            rows.append(_risk_row(risk_id, row["current_quantpilot_module_area"], "high_replacement_priority", "high", row["reason"], row["next_action"]))
            risk_id += 1
        if row["commercial_or_license_risk"] == "high":
            rows.append(_risk_row(risk_id, row["current_quantpilot_module_area"], "license_or_commercial_risk", "high", "Candidate or module touches live, broker, data rights, or commercial restrictions.", "Legal/commercial review before adoption."))
            risk_id += 1
        if row["integration_complexity"] == "high":
            rows.append(_risk_row(risk_id, row["current_quantpilot_module_area"], "integration_complexity", "high", "Replacement may add heavy interfaces and migration cost.", "Prototype narrowly after data contract."))
            risk_id += 1
    high_agent = int((agent_audit["agent_overengineering_risk"] == "high").sum())
    rows.append(_risk_row(risk_id, "agent_tooling", "agent_overengineering_risk", "high", f"{high_agent} agent candidates are high overengineering risk now.", "Defer agents until deterministic data/backtest/factor workflows exist."))
    return pd.DataFrame(rows)


def _risk_row(risk_id: int, area: str, risk_type: str, severity: str, description: str, mitigation: str) -> dict[str, Any]:
    return _add_forbidden_flags({
        "risk_id": f"GRR-{risk_id:03d}",
        "module_area_or_candidate": area,
        "risk_type": risk_type,
        "severity": severity,
        "risk_description": description,
        "mitigation": mitigation,
        "status": "open_for_architecture_review",
    })


def build_guardrails() -> pd.DataFrame:
    return pd.DataFrame([_add_forbidden_flags({
        "guardrail": guardrail,
        "status": status,
        "evidence": evidence,
    }) for guardrail, status, evidence in GUARDRAILS])


def build_summary(
    matrix: pd.DataFrame,
    candidates: pd.DataFrame,
    profitability: pd.DataFrame,
    risk_register: pd.DataFrame,
    agent_audit: pd.DataFrame,
    guardrails: pd.DataFrame,
    run_config: dict[str, Any],
) -> pd.DataFrame:
    mode_counts = matrix["replacement_mode"].value_counts().to_dict()
    forbidden_count = count_forbidden_true_flags(
        [matrix, candidates, profitability, risk_register, agent_audit, guardrails],
        run_config,
    )
    validation_status = "pass" if forbidden_count == 0 and len(matrix) == len(MODULE_AREAS) else "fail"
    return pd.DataFrame([{
        "summary_item": "v7_step2_5_global_open_source_replacement_audit",
        "reviewed_module_area_count": int(len(matrix)),
        "reviewed_open_source_candidate_count": int(len(candidates)),
        "keep_as_core_count": int(mode_counts.get("keep_as_core", 0)),
        "keep_as_guardrail_count": int(mode_counts.get("keep_as_guardrail", 0)),
        "keep_as_fixture_or_regression_test_count": int(mode_counts.get("keep_as_fixture_or_regression_test", 0)),
        "wrap_open_source_count": int(mode_counts.get("wrap_open_source", 0)),
        "replace_with_open_source_count": int(mode_counts.get("replace_with_open_source", 0)),
        "borrow_architecture_only_count": int(mode_counts.get("borrow_architecture_only", 0)),
        "defer_until_foundation_ready_count": int(mode_counts.get("defer_until_foundation_ready", 0)),
        "deprecate_later_count": int(mode_counts.get("deprecate_later", 0)),
        "high_replacement_priority_count": int((matrix["recommended_priority"] == "P1").sum()),
        "high_license_risk_count": int((matrix["commercial_or_license_risk"] == "high").sum() + (candidates["commercial_or_license_risk"] == "high").sum()),
        "high_integration_complexity_count": int((matrix["integration_complexity"] == "high").sum() + (candidates["integration_complexity"] == "high").sum()),
        "high_agent_overengineering_risk_count": int((agent_audit["agent_overengineering_risk"] == "high").sum()),
        "profitability_alignment_issue_count": int(profitability["alignment_issue"].map(_is_true).sum()),
        "market_data_fetch_count": 0,
        "external_api_call_count": 0,
        "package_install_count": 0,
        "broker_connected_count": 0,
        "execution_allowed_count": 0,
        "live_trading_count": 0,
        "real_order_submission_count": 0,
        "forbidden_true_flag_count": forbidden_count,
        "trading_ready": False,
        "validation_status": validation_status,
        "conclusion": "global_open_source_replacement_audit_completed_research_only",
        "recommended_next_step": "V7 Step 3 Data Quality Validator / Local Data Contract, revised by replacement audit",
    }])


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    project_inventory: pd.DataFrame,
    candidates: pd.DataFrame,
    matrix: pd.DataFrame,
    retention: pd.DataFrame,
    roadmap: pd.DataFrame,
    agent_audit: pd.DataFrame,
    profitability: pd.DataFrame,
    risk_register: pd.DataFrame,
    guardrails: pd.DataFrame,
    context: dict[str, Any],
) -> str:
    row = summary.iloc[0]
    conclusions = [
        "Sunk cost warning: past effort should not justify keeping inferior custom architecture.",
        "Custom backtesting and ML workflow should not become long-term core if Qlib, LEAN, vectorbt, or similar tools prove stronger.",
        "V6 validation/simulation hardening is valuable as guardrails, regression evidence, and design records, not as alpha proof.",
        "TradingAgents, FinRobot, FinGPT, OpenBB, and RD-Agent are useful references, but they do not replace data quality or realistic execution foundations now.",
        "Scrapling should be considered later for web/news/announcement collection instead of custom scraping, subject to source terms.",
        "OpenAI Skills, Agents SDK, LangGraph, MCP, CrewAI, and AutoGen should influence later agent architecture only after deterministic foundations exist.",
        "The revised roadmap keeps V7 Step 3 as the next step, but biases it toward Pandera-style contracts and Parquet/DuckDB-ready local storage.",
    ]
    return "\n".join([
        "# V7 Step 2.5 Global Open-source Replacement & Architecture Reassessment",
        "",
        "## Executive Summary",
        "This research-only audit reassesses the entire QuantPilot-AI project with an open-source-first, profitability-oriented lens.",
        "It does not defend past work. It identifies what should be kept, wrapped, replaced, downgraded to fixtures/tests, deferred, or deprecated.",
        "No package was installed, no framework was imported, no market data was fetched, no API was called, no backtest was run, no model was trained, no broker was connected, and no trading-ready claim is made.",
        "",
        "## Local Metadata Context",
        f"- README exists: {context['readme_exists']}",
        f"- app.py exists: {context['app_exists']}",
        f"- src/*.py file count: {context['src_py_file_count']}",
        f"- Step 1 summary rows detected: {context['step1_summary_rows']}",
        f"- Step 2 summary rows detected: {context['step2_summary_rows']}",
        f"- V6 gap rows detected: {context['v6_gap_rows']}",
        "",
        "## Summary",
        f"- Reviewed module areas: {row['reviewed_module_area_count']}",
        f"- Reviewed open-source candidates: {row['reviewed_open_source_candidate_count']}",
        f"- Keep as core: {row['keep_as_core_count']}",
        f"- Keep as guardrail: {row['keep_as_guardrail_count']}",
        f"- Keep as fixture/regression test: {row['keep_as_fixture_or_regression_test_count']}",
        f"- Wrap open source: {row['wrap_open_source_count']}",
        f"- Replace with open source: {row['replace_with_open_source_count']}",
        f"- Borrow architecture only: {row['borrow_architecture_only_count']}",
        f"- Defer until foundation ready: {row['defer_until_foundation_ready_count']}",
        f"- Deprecate later: {row['deprecate_later_count']}",
        f"- High replacement priority: {row['high_replacement_priority_count']}",
        f"- High license risk: {row['high_license_risk_count']}",
        f"- High integration complexity: {row['high_integration_complexity_count']}",
        f"- High agent overengineering risk: {row['high_agent_overengineering_risk_count']}",
        f"- Profitability alignment issues: {row['profitability_alignment_issue_count']}",
        f"- Forbidden true flag count: {row['forbidden_true_flag_count']}",
        f"- Trading ready: {row['trading_ready']}",
        f"- Validation status: {row['validation_status']}",
        f"- Conclusion: {row['conclusion']}",
        f"- Recommended next step: {row['recommended_next_step']}",
        "",
        "## Required Conclusions",
        "\n".join(f"- {item}" for item in conclusions),
        "",
        "## Project Module Inventory",
        _table(project_inventory, "No module inventory rows were generated."),
        "",
        "## Expanded Open-source Candidate Inventory",
        _table(candidates, "No candidate rows were generated."),
        "",
        "## Module Replacement Matrix",
        _table(matrix, "No replacement matrix rows were generated."),
        "",
        "## Past Work Retention Decisions",
        _table(retention, "No retention rows were generated."),
        "",
        "## Revised Integration Roadmap",
        _table(roadmap, "No roadmap rows were generated."),
        "",
        "## Agent Tooling Ecosystem Audit",
        _table(agent_audit, "No agent audit rows were generated."),
        "",
        "## Profitability Alignment Review",
        _table(profitability, "No profitability rows were generated."),
        "",
        "## Replacement Risk Register",
        _table(risk_register, "No risk rows were generated."),
        "",
        "## Guardrails",
        _table(guardrails, "No guardrail rows were generated."),
        "",
        "## Research-only Boundary",
        "This audit is not financial advice and does not establish real alpha evidence, market replay quality, execution quality, broker readiness, or trading readiness.",
        "",
    ])


def generate_global_open_source_replacement_audit_outputs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    step1_dir: str | Path = DEFAULT_STEP1_DIR,
    step2_dir: str | Path = DEFAULT_STEP2_DIR,
    v6_closure_dir: str | Path = DEFAULT_V6_CLOSURE_DIR,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    context = build_repo_context(project_root, step1_dir, step2_dir, v6_closure_dir)
    project_inventory = build_project_module_inventory(context)
    candidates = build_open_source_candidate_expanded_inventory()
    matrix = build_module_replacement_matrix()
    retention = build_past_work_retention_decision(matrix)
    roadmap = build_open_source_integration_roadmap()
    agent_audit = build_agent_tooling_ecosystem_audit()
    profitability = build_profitability_alignment_review(matrix)
    risk_register = build_replacement_risk_register(matrix, candidates, agent_audit)
    guardrails = build_guardrails()
    run_config = {
        "output_dir": str(output_path),
        "project_root": str(project_root),
        "step1_dir": str(step1_dir),
        "step2_dir": str(step2_dir),
        "v6_closure_dir": str(v6_closure_dir),
        "scope": "V7 Step 2.5 architecture reassessment only",
        "market_data_fetch": False,
        "external_api_call": False,
        "package_install": False,
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "architecture_reassessment_only": True,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    summary = build_summary(matrix, candidates, profitability, risk_register, agent_audit, guardrails, run_config)
    report = build_report(summary, project_inventory, candidates, matrix, retention, roadmap, agent_audit, profitability, risk_register, guardrails, context)

    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    output_files["run_config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    project_inventory.to_csv(output_files["project_inventory"], index=False)
    candidates.to_csv(output_files["candidate_inventory"], index=False)
    matrix.to_csv(output_files["replacement_matrix"], index=False)
    retention.to_csv(output_files["retention"], index=False)
    summary.to_csv(output_files["summary"], index=False)
    roadmap.to_csv(output_files["roadmap"], index=False)
    agent_audit.to_csv(output_files["agent_audit"], index=False)
    profitability.to_csv(output_files["profitability"], index=False)
    risk_register.to_csv(output_files["risk_register"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    return {
        "summary": summary,
        "project_inventory": project_inventory,
        "candidates": candidates,
        "matrix": matrix,
        "retention": retention,
        "roadmap": roadmap,
        "agent_audit": agent_audit,
        "profitability": profitability,
        "risk_register": risk_register,
        "guardrails": guardrails,
        "report": report,
        "run_config": run_config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
