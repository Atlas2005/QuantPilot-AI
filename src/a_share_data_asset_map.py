import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/a_share_data_asset_map_real_v1")
DEFAULT_STEP1_DIR = Path("outputs/open_source_quant_stack_audit_real_v1")
DEFAULT_V6_CLOSURE_DIR = Path("outputs/simulation_hardening_closure_real_v1")
DEFAULT_DATA_DIR = Path("data")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "inventory": "a_share_data_source_inventory.csv",
    "coverage": "a_share_data_coverage_matrix.csv",
    "requirements": "a_share_market_reality_data_requirements.csv",
    "gaps": "a_share_data_gap_register.csv",
    "selection": "a_share_data_source_selection.csv",
    "storage": "a_share_data_storage_recommendations.csv",
    "quality": "a_share_data_quality_check_plan.csv",
    "guardrails": "a_share_data_guardrails.csv",
    "summary": "a_share_data_asset_map_summary.csv",
    "report": "a_share_data_asset_map_report.md",
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
    ("no_package_install", "confirmed", "No package manager command is executed and no data dependency is added."),
    ("no_external_data_framework_import", "confirmed", "AkShare, Baostock, Tushare, Qlib, and other external data frameworks are not imported."),
    ("no_market_data_fetch", "confirmed", "No market data endpoint, library fetcher, or downloader is called."),
    ("no_external_api_call", "confirmed", "The generator reads local metadata only and performs no external API calls."),
    ("no_token_or_credentials", "confirmed", "No token, key, credential, account, or secret argument is accepted."),
    ("no_live_data", "confirmed", "No live quote, stream, broker feed, or real-time data source is used."),
    ("no_backtest_execution", "confirmed", "No strategy, replay, or backtest engine is invoked."),
    ("no_model_training", "confirmed", "No model training, fitting, or optimization is performed."),
    ("no_threshold_change", "confirmed", "No strategy threshold, model threshold, or gate threshold is changed."),
    ("no_feature_engineering_change", "confirmed", "No feature builder is modified or executed."),
    ("no_broker_sdk_import", "confirmed", "No broker SDK or order-routing package is imported."),
    ("no_broker_connection", "confirmed", "No broker connection path exists in this step."),
    ("no_order_execution", "confirmed", "No simulated, paper, sandbox, or live order executor is invoked."),
    ("no_real_order_submission", "confirmed", "No real order submission path exists."),
    ("no_trading_ready_upgrade", "confirmed", "All outputs explicitly keep trading_ready=False."),
    ("data_source_selection_only", "confirmed", "This step is a data-source selection and local asset map only."),
    ("open_source_first_policy_applied", "confirmed", "AkShare, Baostock, Tushare, Qlib data workflow, and local files are evaluated before custom data infrastructure."),
    ("educational_research_only", "confirmed", "The report is research-only and is not financial advice."),
]

DATA_SOURCES = [
    {
        "source_name": "AkShare",
        "source_type": "open_source_library",
        "source_url": "https://github.com/akfamily/akshare",
        "a_share_ohlcv_coverage": "high",
        "adjusted_price_qfq_hfq_support": "medium",
        "index_data_coverage": "high",
        "index_constituent_coverage": "medium",
        "st_suspension_delisting_coverage": "medium",
        "limit_up_limit_down_support": "medium",
        "corporate_actions_support": "medium",
        "fundamental_data_support": "medium",
        "valuation_data_support": "medium",
        "money_flow_northbound_fund_flow_support": "high",
        "news_sentiment_support": "medium",
        "intraday_minute_data_support": "medium",
        "data_freshness": "medium",
        "historical_depth": "medium",
        "access_cost": "free_public_endpoint_dependent",
        "token_credential_requirement": "none_expected_for_common_public_endpoints",
        "commercial_license_risk": "medium",
        "stability_risk": "high",
        "rate_limit_blocking_risk": "high",
        "windows_local_compatibility": "high",
        "python_compatibility": "high",
        "future_qlib_vectorbt_backtest_integration_fit": "medium",
        "future_multi_agent_research_fit": "medium",
        "recommended_action": "primary_candidate",
        "rationale": "Broad A-share data surface and useful factor-adjacent endpoints, but public endpoint stability and data rights must be validated before trust.",
        "quantpilot_ai_role": "Primary candidate for a later explicit non-CI prototype fetch command, not used by this step.",
    },
    {
        "source_name": "Baostock",
        "source_type": "open_source_library",
        "source_url": "https://pypi.org/project/baostock/",
        "a_share_ohlcv_coverage": "high",
        "adjusted_price_qfq_hfq_support": "medium",
        "index_data_coverage": "medium",
        "index_constituent_coverage": "low",
        "st_suspension_delisting_coverage": "low",
        "limit_up_limit_down_support": "low",
        "corporate_actions_support": "medium",
        "fundamental_data_support": "low",
        "valuation_data_support": "low",
        "money_flow_northbound_fund_flow_support": "low",
        "news_sentiment_support": "low",
        "intraday_minute_data_support": "low",
        "data_freshness": "medium",
        "historical_depth": "medium",
        "access_cost": "free",
        "token_credential_requirement": "none_expected",
        "commercial_license_risk": "medium",
        "stability_risk": "medium",
        "rate_limit_blocking_risk": "medium",
        "windows_local_compatibility": "high",
        "python_compatibility": "medium",
        "future_qlib_vectorbt_backtest_integration_fit": "medium",
        "future_multi_agent_research_fit": "low",
        "recommended_action": "secondary_candidate",
        "rationale": "Useful simple baseline for daily A-share OHLCV, but insufficient alone for market-reality replay.",
        "quantpilot_ai_role": "Secondary baseline candidate for daily price cross-checks and fixture comparison.",
    },
    {
        "source_name": "Tushare",
        "source_type": "token_based_api",
        "source_url": "https://tushare.pro/",
        "a_share_ohlcv_coverage": "high",
        "adjusted_price_qfq_hfq_support": "high",
        "index_data_coverage": "high",
        "index_constituent_coverage": "high",
        "st_suspension_delisting_coverage": "high",
        "limit_up_limit_down_support": "medium",
        "corporate_actions_support": "high",
        "fundamental_data_support": "high",
        "valuation_data_support": "high",
        "money_flow_northbound_fund_flow_support": "high",
        "news_sentiment_support": "medium",
        "intraday_minute_data_support": "medium",
        "data_freshness": "high",
        "historical_depth": "high",
        "access_cost": "token_points_or_paid_tier",
        "token_credential_requirement": "required",
        "commercial_license_risk": "high",
        "stability_risk": "medium",
        "rate_limit_blocking_risk": "medium",
        "windows_local_compatibility": "high",
        "python_compatibility": "high",
        "future_qlib_vectorbt_backtest_integration_fit": "high",
        "future_multi_agent_research_fit": "high",
        "recommended_action": "prototype_required",
        "rationale": "Potentially strong A-share coverage, but token, points, cost, and commercial terms must be reviewed before product use.",
        "quantpilot_ai_role": "License-reviewed candidate for later controlled data prototype; no token use now.",
    },
    {
        "source_name": "Qlib China Data Workflow",
        "source_type": "platform_dataset",
        "source_url": "https://github.com/microsoft/qlib",
        "a_share_ohlcv_coverage": "medium",
        "adjusted_price_qfq_hfq_support": "medium",
        "index_data_coverage": "medium",
        "index_constituent_coverage": "medium",
        "st_suspension_delisting_coverage": "low",
        "limit_up_limit_down_support": "low",
        "corporate_actions_support": "medium",
        "fundamental_data_support": "low",
        "valuation_data_support": "low",
        "money_flow_northbound_fund_flow_support": "low",
        "news_sentiment_support": "low",
        "intraday_minute_data_support": "low",
        "data_freshness": "low",
        "historical_depth": "medium",
        "access_cost": "free_or_source_dependent",
        "token_credential_requirement": "none_expected_for_public_workflow",
        "commercial_license_risk": "medium",
        "stability_risk": "medium",
        "rate_limit_blocking_risk": "medium",
        "windows_local_compatibility": "medium",
        "python_compatibility": "medium",
        "future_qlib_vectorbt_backtest_integration_fit": "high",
        "future_multi_agent_research_fit": "high",
        "recommended_action": "prototype_required",
        "rationale": "Useful for Qlib factor/model workflow compatibility, but its bundled/download workflow must not replace validated canonical local data.",
        "quantpilot_ai_role": "Future adapter target after stable local A-share data exists.",
    },
    {
        "source_name": "Existing Local CSV/sample data",
        "source_type": "local_file",
        "source_url": "local:data/",
        "a_share_ohlcv_coverage": "low",
        "adjusted_price_qfq_hfq_support": "unknown",
        "index_data_coverage": "low",
        "index_constituent_coverage": "low",
        "st_suspension_delisting_coverage": "low",
        "limit_up_limit_down_support": "low",
        "corporate_actions_support": "low",
        "fundamental_data_support": "low",
        "valuation_data_support": "low",
        "money_flow_northbound_fund_flow_support": "low",
        "news_sentiment_support": "low",
        "intraday_minute_data_support": "low",
        "data_freshness": "unknown",
        "historical_depth": "low",
        "access_cost": "already_local",
        "token_credential_requirement": "none",
        "commercial_license_risk": "low",
        "stability_risk": "low",
        "rate_limit_blocking_risk": "low",
        "windows_local_compatibility": "high",
        "python_compatibility": "high",
        "future_qlib_vectorbt_backtest_integration_fit": "medium",
        "future_multi_agent_research_fit": "low",
        "recommended_action": "use_as_fixture_only",
        "rationale": "Local files are valuable for CI fixtures and contract tests, but they are not sufficient evidence for real alpha or market replay.",
        "quantpilot_ai_role": "Fixture baseline for schema and validator development only.",
    },
    {
        "source_name": "Future Paid/Commercial Data Vendor",
        "source_type": "future_paid_vendor",
        "source_url": "placeholder:future_vendor",
        "a_share_ohlcv_coverage": "high",
        "adjusted_price_qfq_hfq_support": "high",
        "index_data_coverage": "high",
        "index_constituent_coverage": "high",
        "st_suspension_delisting_coverage": "high",
        "limit_up_limit_down_support": "high",
        "corporate_actions_support": "high",
        "fundamental_data_support": "high",
        "valuation_data_support": "high",
        "money_flow_northbound_fund_flow_support": "medium",
        "news_sentiment_support": "medium",
        "intraday_minute_data_support": "high",
        "data_freshness": "high",
        "historical_depth": "high",
        "access_cost": "paid_contract_required",
        "token_credential_requirement": "likely_required",
        "commercial_license_risk": "high",
        "stability_risk": "low",
        "rate_limit_blocking_risk": "low",
        "windows_local_compatibility": "medium",
        "python_compatibility": "medium",
        "future_qlib_vectorbt_backtest_integration_fit": "high",
        "future_multi_agent_research_fit": "high",
        "recommended_action": "defer_until_paid_stage",
        "rationale": "Likely needed for production-quality reliability, rights, and depth, but premature before free-source prototypes define the required contract.",
        "quantpilot_ai_role": "Paid-stage replacement or validation benchmark after V7 data requirements mature.",
    },
    {
        "source_name": "Eastmoney/Sina/163-style Public Endpoints",
        "source_type": "open_source_library",
        "source_url": "indirect_only_through_mature_libraries",
        "a_share_ohlcv_coverage": "medium",
        "adjusted_price_qfq_hfq_support": "unknown",
        "index_data_coverage": "medium",
        "index_constituent_coverage": "unknown",
        "st_suspension_delisting_coverage": "unknown",
        "limit_up_limit_down_support": "unknown",
        "corporate_actions_support": "unknown",
        "fundamental_data_support": "medium",
        "valuation_data_support": "medium",
        "money_flow_northbound_fund_flow_support": "medium",
        "news_sentiment_support": "medium",
        "intraday_minute_data_support": "medium",
        "data_freshness": "medium",
        "historical_depth": "unknown",
        "access_cost": "free_public_endpoint_dependent",
        "token_credential_requirement": "none_expected",
        "commercial_license_risk": "high",
        "stability_risk": "high",
        "rate_limit_blocking_risk": "high",
        "windows_local_compatibility": "medium",
        "python_compatibility": "medium",
        "future_qlib_vectorbt_backtest_integration_fit": "low",
        "future_multi_agent_research_fit": "low",
        "recommended_action": "avoid_for_now",
        "rationale": "Do not build direct scraping. Use mature libraries if this data is needed and validate provenance, stability, and rights.",
        "quantpilot_ai_role": "Avoid direct integration; indirect fallback only through maintained libraries after source review.",
    },
    {
        "source_name": "Local Parquet/DuckDB Data Lake",
        "source_type": "storage_layer",
        "source_url": "local:future_storage",
        "a_share_ohlcv_coverage": "depends_on_ingested_sources",
        "adjusted_price_qfq_hfq_support": "depends_on_schema",
        "index_data_coverage": "depends_on_ingested_sources",
        "index_constituent_coverage": "depends_on_ingested_sources",
        "st_suspension_delisting_coverage": "depends_on_ingested_sources",
        "limit_up_limit_down_support": "depends_on_schema",
        "corporate_actions_support": "depends_on_schema",
        "fundamental_data_support": "depends_on_ingested_sources",
        "valuation_data_support": "depends_on_ingested_sources",
        "money_flow_northbound_fund_flow_support": "depends_on_ingested_sources",
        "news_sentiment_support": "depends_on_ingested_sources",
        "intraday_minute_data_support": "depends_on_storage_design",
        "data_freshness": "depends_on_refresh_pipeline",
        "historical_depth": "depends_on_ingested_sources",
        "access_cost": "local_storage_cost",
        "token_credential_requirement": "none_for_storage",
        "commercial_license_risk": "low",
        "stability_risk": "low",
        "rate_limit_blocking_risk": "low",
        "windows_local_compatibility": "high",
        "python_compatibility": "high",
        "future_qlib_vectorbt_backtest_integration_fit": "high",
        "future_multi_agent_research_fit": "high",
        "recommended_action": "secondary_candidate",
        "rationale": "A local canonical storage layer is necessary, but it should be designed after the data contract is specified.",
        "quantpilot_ai_role": "Future canonical storage layer; CSV fixtures first, Parquet primary later, DuckDB optional for queries.",
    },
]

REQUIREMENTS = [
    ("REQ-MR-001", "minimum_market_reality", "daily_ohlcv", "blocking", "Open/high/low/close/volume/amount by symbol and trading date."),
    ("REQ-MR-002", "minimum_market_reality", "adjusted_prices", "blocking", "qfq/hfq or corporate-action-adjusted close fields for consistent return calculation."),
    ("REQ-MR-003", "minimum_market_reality", "trading_calendar", "blocking", "Exchange trading days and holidays for alignment and walk-forward splits."),
    ("REQ-MR-004", "minimum_market_reality", "suspension_flags", "blocking", "Suspended dates must be known to prevent impossible fills."),
    ("REQ-MR-005", "minimum_market_reality", "st_flags", "blocking", "ST and special treatment flags affect tradability and risk policy."),
    ("REQ-MR-006", "minimum_market_reality", "limit_up_limit_down_prices", "blocking", "Daily limit prices or enough fields to infer them are needed for A-share fills."),
    ("REQ-MR-007", "minimum_market_reality", "listing_delisting_dates", "blocking", "Listing and delisting dates avoid survivorship and impossible trades."),
    ("REQ-MR-008", "minimum_market_reality", "index_benchmark_prices", "blocking", "Benchmark prices are required for excess return and regime analysis."),
    ("REQ-MR-009", "minimum_market_reality", "index_constituents", "blocking", "Historical constituents reduce universe and benchmark membership bias."),
    ("REQ-MR-010", "minimum_market_reality", "transaction_cost_assumptions", "blocking", "Commission, stamp tax, transfer fee, slippage, and min fee assumptions must be explicit."),
    ("REQ-MR-011", "minimum_market_reality", "liquidity_volume_constraints", "blocking", "Volume, turnover, amount, and lot-size constraints are needed for small-capital realism."),
    ("REQ-AR-001", "alpha_research", "technical_price_volume_features", "high", "Price-volume features, turnover proxies, volatility, momentum, and liquidity features."),
    ("REQ-AR-002", "alpha_research", "fundamentals", "medium", "Financial statements and quality/profitability features for broader alpha research."),
    ("REQ-AR-003", "alpha_research", "valuations", "medium", "PE/PB/PS/dividend yield and valuation percentile fields."),
    ("REQ-AR-004", "alpha_research", "money_flow_northbound_flow", "medium", "Money flow, northbound, fund flow, and participant activity when available."),
    ("REQ-AR-005", "alpha_research", "sector_industry_classification", "high", "Sector/industry classification for neutralization, grouping, and risk exposure."),
    ("REQ-AR-006", "alpha_research", "news_sentiment_later", "low", "News and sentiment should be delayed until structured market data is stable."),
]

STAGED_PLAN = [
    ("STAGE-001", "Stage 1", "local_data_inventory_and_fixture_consistency", "Use local CSV/sample files only as fixtures; record schemas, sizes, and gaps."),
    ("STAGE-002", "Stage 2", "controlled_akshare_baostock_fetch_prototype_later", "Later add explicit non-CI commands for AkShare/Baostock fetch prototypes after user approval."),
    ("STAGE-003", "Stage 3", "canonical_local_storage_format", "Choose canonical CSV/Parquet schema; keep DuckDB optional for query acceleration later."),
    ("STAGE-004", "Stage 4", "data_quality_validator", "Build validator for OHLCV, calendar, suspensions, ST, limits, adjustments, and benchmark alignment."),
    ("STAGE-005", "Stage 5", "qlib_vectorbt_backtest_adapter_evaluation", "Evaluate adapters only after stable validated local data exists."),
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


def _is_token_required(value: Any) -> bool:
    return "required" in str(value).strip().lower()


def count_forbidden_true_flags(frames: list[pd.DataFrame], config: dict[str, Any]) -> int:
    count = 0
    for frame in frames:
        for flag in FORBIDDEN_FLAGS:
            if flag in frame:
                count += int(frame[flag].map(_is_true).sum())
    for flag in FORBIDDEN_FLAGS:
        count += int(_is_true(config.get(flag)))
    return count


def collect_local_data_metadata(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    base = Path(data_dir)
    files = [path for path in base.rglob("*") if path.is_file()] if base.exists() else []
    csv_files = [path for path in files if path.suffix.lower() == ".csv"]
    return {
        "data_dir": str(base),
        "data_dir_exists": base.exists(),
        "local_file_count": len(files),
        "local_csv_count": len(csv_files),
        "local_csv_paths": "; ".join(str(path.as_posix()) for path in csv_files[:20]),
        "local_total_bytes": int(sum(path.stat().st_size for path in files)),
    }


def build_input_context(
    step1_dir: str | Path = DEFAULT_STEP1_DIR,
    v6_closure_dir: str | Path = DEFAULT_V6_CLOSURE_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    step1 = Path(step1_dir)
    v6 = Path(v6_closure_dir)
    local_meta = collect_local_data_metadata(data_dir)
    step1_summary = _read_csv(step1 / "open_source_stack_audit_summary.csv")
    step1_inventory = _read_csv(step1 / "open_source_candidate_inventory.csv")
    step1_a_share = _read_csv(step1 / "open_source_a_share_fit_matrix.csv")
    step1_recs = _read_csv(step1 / "open_source_integration_recommendations.csv")
    v6_gaps = _read_csv(v6 / "v6_remaining_gap_register.csv")
    return {
        "step1_summary_rows": int(len(step1_summary)),
        "step1_inventory_rows": int(len(step1_inventory)),
        "step1_a_share_fit_rows": int(len(step1_a_share)),
        "step1_recommendation_rows": int(len(step1_recs)),
        "v6_gap_rows": int(len(v6_gaps)),
        **local_meta,
    }


def build_data_source_inventory(context: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for index, source in enumerate(DATA_SOURCES, start=1):
        local_note = ""
        if source["source_name"] == "Existing Local CSV/sample data":
            local_note = (
                f"Detected {context['local_csv_count']} local CSV files and "
                f"{context['local_total_bytes']} bytes under {context['data_dir']}."
            )
        rows.append(
            _add_forbidden_flags(
                {
                    "source_id": f"ADS-{index:03d}",
                    "source_name": source["source_name"],
                    "source_type": source["source_type"],
                    "source_url": source["source_url"],
                    "inventory_status": "reviewed_metadata_only",
                    "package_installed_now": False,
                    "external_framework_imported_now": False,
                    "api_called_now": False,
                    "token_used_now": False,
                    "local_metadata_note": local_note,
                    "quantpilot_ai_role": source["quantpilot_ai_role"],
                }
            )
        )
    return pd.DataFrame(rows)


def build_coverage_matrix() -> pd.DataFrame:
    rows = []
    for source in DATA_SOURCES:
        rows.append(
            _add_forbidden_flags(
                {
                    "source_name": source["source_name"],
                    "source_type": source["source_type"],
                    "a_share_ohlcv_coverage": source["a_share_ohlcv_coverage"],
                    "adjusted_price_qfq_hfq_support": source["adjusted_price_qfq_hfq_support"],
                    "index_data_coverage": source["index_data_coverage"],
                    "index_constituent_coverage": source["index_constituent_coverage"],
                    "st_suspension_delisting_coverage": source["st_suspension_delisting_coverage"],
                    "limit_up_limit_down_support": source["limit_up_limit_down_support"],
                    "corporate_actions_support": source["corporate_actions_support"],
                    "fundamental_data_support": source["fundamental_data_support"],
                    "valuation_data_support": source["valuation_data_support"],
                    "money_flow_northbound_fund_flow_support": source["money_flow_northbound_fund_flow_support"],
                    "news_sentiment_support": source["news_sentiment_support"],
                    "intraday_minute_data_support": source["intraday_minute_data_support"],
                    "data_freshness": source["data_freshness"],
                    "historical_depth": source["historical_depth"],
                    "access_cost": source["access_cost"],
                    "token_credential_requirement": source["token_credential_requirement"],
                    "commercial_license_risk": source["commercial_license_risk"],
                    "stability_risk": source["stability_risk"],
                    "rate_limit_blocking_risk": source["rate_limit_blocking_risk"],
                    "windows_local_compatibility": source["windows_local_compatibility"],
                    "python_compatibility": source["python_compatibility"],
                    "future_qlib_vectorbt_backtest_integration_fit": source["future_qlib_vectorbt_backtest_integration_fit"],
                    "future_multi_agent_research_fit": source["future_multi_agent_research_fit"],
                    "recommended_action": source["recommended_action"],
                    "rationale": source["rationale"],
                    "quantpilot_ai_role": source["quantpilot_ai_role"],
                }
            )
        )
    return pd.DataFrame(rows)


def build_market_reality_requirements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _add_forbidden_flags(
                {
                    "requirement_id": req_id,
                    "requirement_category": category,
                    "requirement_name": name,
                    "priority": priority,
                    "requirement_description": description,
                    "required_before_realistic_backtest": category == "minimum_market_reality",
                    "required_before_alpha_research_scaleup": category in {"minimum_market_reality", "alpha_research"},
                    "validated_now": False,
                    "validation_boundary": "Requirement identified only; no source is trusted until validated.",
                }
            )
            for req_id, category, name, priority, description in REQUIREMENTS
        ]
    )


def build_gap_register(requirements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in requirements.iterrows():
        severity = "blocking" if row["requirement_category"] == "minimum_market_reality" else "medium"
        rows.append(
            _add_forbidden_flags(
                {
                    "gap_id": f"ADG-{index + 1:03d}",
                    "gap_name": f"unvalidated_{row['requirement_name']}",
                    "related_requirement_id": row["requirement_id"],
                    "severity": severity,
                    "current_status": "not_validated",
                    "risk": "Data source coverage, correctness, rights, and local schema are not proven.",
                    "required_resolution": "Add source-specific contract tests and data quality checks in V7 Step 3+ before trusting this field.",
                    "blocks_trading_ready": True,
                }
            )
        )
    rows.append(
        _add_forbidden_flags(
            {
                "gap_id": f"ADG-{len(rows) + 1:03d}",
                "gap_name": "public_free_data_instability",
                "related_requirement_id": "SOURCE-RISK",
                "severity": "blocking",
                "current_status": "not_validated",
                "risk": "Public/free data can change format, become unavailable, block requests, or carry unclear rights.",
                "required_resolution": "Validate stability, provenance, rate limits, and license constraints before relying on it.",
                "blocks_trading_ready": True,
            }
        )
    )
    return pd.DataFrame(rows)


def build_data_source_selection(coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stage_map = {
        "AkShare": "Stage 2 controlled explicit prototype later",
        "Baostock": "Stage 2 controlled explicit prototype later",
        "Tushare": "Commercial/legal review before prototype",
        "Qlib China Data Workflow": "Stage 5 adapter evaluation after stable local data",
        "Existing Local CSV/sample data": "Stage 1 fixture and schema consistency only",
        "Future Paid/Commercial Data Vendor": "Paid-stage review after requirements mature",
        "Eastmoney/Sina/163-style Public Endpoints": "Avoid direct scraping; indirect library-only fallback",
        "Local Parquet/DuckDB Data Lake": "Stage 3 local canonical storage design",
    }
    for _, row in coverage.iterrows():
        rows.append(
            _add_forbidden_flags(
                {
                    "source_name": row["source_name"],
                    "recommended_action": row["recommended_action"],
                    "selection_stage": stage_map[row["source_name"]],
                    "trusted_now": False,
                    "prototype_allowed_in_current_step": False,
                    "requires_explicit_future_command": row["recommended_action"] in {"primary_candidate", "secondary_candidate", "prototype_required"},
                    "requires_license_review": _is_high(row["commercial_license_risk"]),
                    "requires_quality_validation": True,
                    "selection_rationale": row["rationale"],
                }
            )
        )
    return pd.DataFrame(rows)


def build_storage_recommendations() -> pd.DataFrame:
    rows = [
        ("STORE-001", "CSV fixtures", "use_now_for_tests_only", "Keep small local CSVs for CI fixtures and schema contract examples."),
        ("STORE-002", "Partitioned Parquet", "recommended_future_primary", "Use symbol/date partitioned Parquet for validated daily market data after source selection."),
        ("STORE-003", "DuckDB", "optional_later", "Use DuckDB as a query layer over Parquet when local analytics and joins become heavy."),
        ("STORE-004", "Qlib binary/cache format", "adapter_target_later", "Generate Qlib-compatible format only after canonical local data is validated."),
    ]
    return pd.DataFrame(
        [
            _add_forbidden_flags(
                {
                    "storage_id": storage_id,
                    "storage_option": option,
                    "recommendation": recommendation,
                    "rationale": rationale,
                    "implemented_now": False,
                    "requires_validated_source_data": option != "CSV fixtures",
                }
            )
            for storage_id, option, recommendation, rationale in rows
        ]
    )


def build_quality_check_plan(requirements: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("DQC-001", "schema_contract", "Verify required columns, dtypes, symbol format, date format, timezone assumptions, and no duplicate keys."),
        ("DQC-002", "ohlcv_integrity", "Check high/low/open/close consistency, nonnegative volume/amount, and missing or zero-price rows."),
        ("DQC-003", "calendar_alignment", "Validate dates against exchange trading calendar and detect missing trading days."),
        ("DQC-004", "adjustment_consistency", "Compare raw, qfq, and hfq fields and detect unexplained jumps around corporate actions."),
        ("DQC-005", "suspension_and_st_flags", "Confirm suspended and ST dates align with price availability and tradability rules."),
        ("DQC-006", "limit_price_validation", "Validate limit-up/down fields or derived limits before execution modeling."),
        ("DQC-007", "listing_delisting_survivorship", "Ensure universe construction respects listing and delisting dates."),
        ("DQC-008", "benchmark_and_constituents", "Validate benchmark prices and historical constituent membership for excess returns."),
        ("DQC-009", "liquidity_cost_fields", "Check amount, turnover, lot-size constraints, commission, tax, and slippage assumptions."),
        ("DQC-010", "cross_source_reconciliation", "Compare AkShare/Baostock/Tushare/local samples where permitted and log disagreements."),
    ]
    requirement_count = int(len(requirements))
    return pd.DataFrame(
        [
            _add_forbidden_flags(
                {
                    "check_id": check_id,
                    "check_name": name,
                    "check_description": description,
                    "planned_for_step": "V7 Step 3 Data Quality Validator / Local Data Contract",
                    "implemented_now": False,
                    "related_requirement_count": requirement_count,
                }
            )
            for check_id, name, description in checks
        ]
    )


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
    coverage: pd.DataFrame,
    requirements: pd.DataFrame,
    gaps: pd.DataFrame,
    storage: pd.DataFrame,
    guardrails: pd.DataFrame,
    run_config: dict[str, Any],
) -> pd.DataFrame:
    action_counts = coverage["recommended_action"].value_counts().to_dict()
    forbidden_count = count_forbidden_true_flags(
        [inventory, coverage, requirements, gaps, storage, guardrails],
        run_config,
    )
    validation_status = "pass" if forbidden_count == 0 and len(coverage) == len(DATA_SOURCES) else "fail"
    return pd.DataFrame(
        [
            {
                "summary_item": "v7_step2_a_share_data_asset_map",
                "reviewed_data_source_count": int(len(coverage)),
                "primary_candidate_count": int(action_counts.get("primary_candidate", 0)),
                "secondary_candidate_count": int(action_counts.get("secondary_candidate", 0)),
                "prototype_required_count": int(action_counts.get("prototype_required", 0)),
                "fixture_only_count": int(action_counts.get("use_as_fixture_only", 0)),
                "defer_until_paid_stage_count": int(action_counts.get("defer_until_paid_stage", 0)),
                "avoid_for_now_count": int(action_counts.get("avoid_for_now", 0)),
                "minimum_market_reality_requirement_count": int((requirements["requirement_category"] == "minimum_market_reality").sum()),
                "alpha_research_requirement_count": int((requirements["requirement_category"] == "alpha_research").sum()),
                "blocking_data_gap_count": int((gaps["severity"] == "blocking").sum()),
                "high_commercial_license_risk_count": int(coverage["commercial_license_risk"].map(_is_high).sum()),
                "high_stability_risk_count": int(coverage["stability_risk"].map(_is_high).sum()),
                "token_required_source_count": int(coverage["token_credential_requirement"].map(_is_token_required).sum()),
                "local_storage_recommendation_count": int(len(storage)),
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
                "conclusion": "a_share_data_asset_map_completed_research_only",
                "recommended_next_step": "V7 Step 3 Data Quality Validator / Local Data Contract",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    requirements: pd.DataFrame,
    gaps: pd.DataFrame,
    selection: pd.DataFrame,
    storage: pd.DataFrame,
    quality: pd.DataFrame,
    guardrails: pd.DataFrame,
    context: dict[str, Any],
) -> str:
    row = summary.iloc[0]
    conclusions = [
        "No data source is trusted until validated.",
        "Public/free data can be unstable and needs quality, provenance, rights, and rate-limit checks.",
        "Direct scraping is avoided when mature libraries can provide the same data.",
        "Agent frameworks remain deferred; V7 Step 2 is a data foundation decision layer.",
        "Stage 1 uses local data inventory and fixture consistency only.",
        "Stage 2 should prototype AkShare/Baostock fetches later through explicit non-CI commands.",
        "Stage 3 should choose canonical local CSV/Parquet storage, with DuckDB optional later.",
        "Stage 4 should implement a data quality validator and local data contract.",
        "Stage 5 should evaluate Qlib/vectorbt/backtest adapters only after stable local data exists.",
    ]
    return "\n".join(
        [
            "# V7 Step 2 A-share Data Asset Map / Data Source Selection",
            "",
            "## Executive Summary",
            "This research-only step maps A-share data assets and source candidates before building data infrastructure.",
            "It is designed to support future realistic A-share backtesting, alpha research, walk-forward validation, portfolio/risk allocation, and later multi-agent research.",
            "No data is fetched, no data package is installed, no external data framework is imported, no API is called, no credential is used, no backtest is run, no model is trained, no broker is connected, and no trading-ready claim is made.",
            "",
            "## Local Context",
            f"- Step 1 summary rows detected: {context['step1_summary_rows']}",
            f"- Step 1 inventory rows detected: {context['step1_inventory_rows']}",
            f"- Step 1 A-share fit rows detected: {context['step1_a_share_fit_rows']}",
            f"- Step 1 recommendation rows detected: {context['step1_recommendation_rows']}",
            f"- V6 gap rows detected: {context['v6_gap_rows']}",
            f"- Local CSV files detected under data/: {context['local_csv_count']}",
            "",
            "## Summary",
            f"- Reviewed data sources: {row['reviewed_data_source_count']}",
            f"- Primary candidates: {row['primary_candidate_count']}",
            f"- Secondary candidates: {row['secondary_candidate_count']}",
            f"- Prototype required: {row['prototype_required_count']}",
            f"- Fixture only: {row['fixture_only_count']}",
            f"- Defer until paid stage: {row['defer_until_paid_stage_count']}",
            f"- Avoid for now: {row['avoid_for_now_count']}",
            f"- Minimum market-reality requirements: {row['minimum_market_reality_requirement_count']}",
            f"- Alpha research requirements: {row['alpha_research_requirement_count']}",
            f"- Blocking data gaps: {row['blocking_data_gap_count']}",
            f"- High commercial/license risk: {row['high_commercial_license_risk_count']}",
            f"- High stability risk: {row['high_stability_risk_count']}",
            f"- Token-required sources: {row['token_required_source_count']}",
            f"- Local storage recommendations: {row['local_storage_recommendation_count']}",
            f"- Market data fetch count: {row['market_data_fetch_count']}",
            f"- External API call count: {row['external_api_call_count']}",
            f"- Package install count: {row['package_install_count']}",
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
            "## Required Conclusions",
            "\n".join(f"- {item}" for item in conclusions),
            "",
            "## Data Source Inventory",
            _table(inventory, "No inventory rows were generated."),
            "",
            "## Data Coverage Matrix",
            _table(coverage, "No coverage rows were generated."),
            "",
            "## Market Reality And Alpha Requirements",
            _table(requirements, "No requirement rows were generated."),
            "",
            "## Data Gap Register",
            _table(gaps, "No data gap rows were generated."),
            "",
            "## Data Source Selection",
            _table(selection, "No selection rows were generated."),
            "",
            "## Storage Recommendations",
            _table(storage, "No storage rows were generated."),
            "",
            "## Data Quality Check Plan",
            _table(quality, "No quality check rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-only Boundary",
            "This output is not financial advice and does not establish real alpha evidence, market replay quality, validated data quality, broker readiness, or trading readiness.",
            "",
        ]
    )


def generate_a_share_data_asset_map_outputs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    step1_dir: str | Path = DEFAULT_STEP1_DIR,
    v6_closure_dir: str | Path = DEFAULT_V6_CLOSURE_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    context = build_input_context(step1_dir, v6_closure_dir, data_dir)
    inventory = build_data_source_inventory(context)
    coverage = build_coverage_matrix()
    requirements = build_market_reality_requirements()
    gaps = build_gap_register(requirements)
    selection = build_data_source_selection(coverage)
    storage = build_storage_recommendations()
    quality = build_quality_check_plan(requirements)
    guardrails = build_guardrails()
    run_config = {
        "output_dir": str(output_path),
        "step1_dir": str(step1_dir),
        "v6_closure_dir": str(v6_closure_dir),
        "data_dir": str(data_dir),
        "scope": "V7 Step 2 A-share data asset map and source selection only",
        "market_data_fetch": False,
        "external_api_call": False,
        "package_install": False,
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "data_source_selection_only": True,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    summary = build_summary(inventory, coverage, requirements, gaps, storage, guardrails, run_config)
    report = build_report(summary, inventory, coverage, requirements, gaps, selection, storage, quality, guardrails, context)

    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    output_files["run_config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    inventory.to_csv(output_files["inventory"], index=False)
    coverage.to_csv(output_files["coverage"], index=False)
    requirements.to_csv(output_files["requirements"], index=False)
    gaps.to_csv(output_files["gaps"], index=False)
    selection.to_csv(output_files["selection"], index=False)
    storage.to_csv(output_files["storage"], index=False)
    quality.to_csv(output_files["quality"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    summary.to_csv(output_files["summary"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    return {
        "summary": summary,
        "inventory": inventory,
        "coverage": coverage,
        "requirements": requirements,
        "gaps": gaps,
        "selection": selection,
        "storage": storage,
        "quality": quality,
        "guardrails": guardrails,
        "report": report,
        "run_config": run_config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
