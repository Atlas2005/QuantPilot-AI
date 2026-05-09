import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CAPITAL_DIR = Path("outputs/capital_constraint_engine_real_v1")
DEFAULT_UNIVERSE_DIR = Path("outputs/tradable_universe_filter_real_v1")
DEFAULT_POSITION_DIR = Path("outputs/position_sizing_engine_real_v1")
DEFAULT_EXIT_DIR = Path("outputs/exit_engine_real_v1")
DEFAULT_DAILY_PLAN_DIR = Path("outputs/daily_trading_plan_real_v1")
DEFAULT_PAPER_LEDGER_DIR = Path("outputs/paper_trading_ledger_real_v1")
DEFAULT_SEMI_AUTO_DIR = Path("outputs/semi_auto_order_generator_real_v1")
DEFAULT_BROKER_RESEARCH_DIR = Path("outputs/broker_integration_research_real_v1")
DEFAULT_MONITORING_DIR = Path("outputs/monitoring_reporting_layer_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/capital_aware_infrastructure_review_real_v1")

OUTPUT_FILENAMES = {
    "summary": "v5_infrastructure_closure_summary.csv",
    "capability_matrix": "v5_step_capability_matrix.csv",
    "guardrail_audit": "v5_guardrail_audit.csv",
    "limitations": "v5_limitations_register.csv",
    "blockers": "v5_readiness_blockers.csv",
    "recommendations": "v5_next_phase_recommendations.csv",
    "report": "v5_capital_aware_closure_report.md",
    "run_config": "run_config.json",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
]

STEP_DEFINITIONS = [
    ("V5 Step 1", "capital feasibility", "capital", "Capital feasibility checks for candidate orders."),
    ("V5 Step 2", "tradable universe filtering", "universe", "Tradable universe eligibility filters."),
    ("V5 Step 3", "position sizing", "position", "Bounded research-only position sizing."),
    ("V5 Step 4", "exit planning", "exit", "Static exit planning for sized positions."),
    ("V5 Step 5", "daily plan generation", "daily", "Human-reviewable daily trading plan."),
    ("V5 Step 6", "paper ledger", "paper", "Deterministic paper ledger and cash ledger."),
    ("V5 Step 7", "semi-auto draft order generation", "semi", "Broker-neutral draft tickets requiring human review."),
    ("V5 Step 8", "broker integration research", "broker", "Research-only broker integration constraints and risks."),
    ("V5 Step 9", "monitoring/reporting", "monitoring", "Unified local monitoring/reporting rollup."),
]


def _read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype or {"symbol": str})
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


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


def _bool_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df:
        return 0
    return int(df[column].map(_is_true).sum())


def _json_flag_count(value: Any, flag: str) -> int:
    if isinstance(value, dict):
        total = 1 if flag in value and _is_true(value[flag]) else 0
        return total + sum(_json_flag_count(child, flag) for child in value.values())
    if isinstance(value, list):
        return sum(_json_flag_count(child, flag) for child in value)
    return 0


def _safe_file_count(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return len([item for item in path.iterdir() if item.is_file()])


def build_input_paths(
    capital_dir: str | Path = DEFAULT_CAPITAL_DIR,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
    position_dir: str | Path = DEFAULT_POSITION_DIR,
    exit_dir: str | Path = DEFAULT_EXIT_DIR,
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    semi_auto_dir: str | Path = DEFAULT_SEMI_AUTO_DIR,
    broker_research_dir: str | Path = DEFAULT_BROKER_RESEARCH_DIR,
    monitoring_dir: str | Path = DEFAULT_MONITORING_DIR,
) -> dict[str, Path]:
    return {
        "capital": Path(capital_dir),
        "universe": Path(universe_dir),
        "position": Path(position_dir),
        "exit": Path(exit_dir),
        "daily": Path(daily_plan_dir),
        "paper": Path(paper_ledger_dir),
        "semi": Path(semi_auto_dir),
        "broker": Path(broker_research_dir),
        "monitoring": Path(monitoring_dir),
    }


def scan_safety_flags(paths: dict[str, Path]) -> dict[str, int]:
    counts = {flag: 0 for flag in SAFETY_FLAGS}
    for output_dir in paths.values():
        if not output_dir.exists() or not output_dir.is_dir():
            continue
        for csv_path in output_dir.glob("*.csv"):
            frame = _read_csv(csv_path)
            for flag in SAFETY_FLAGS:
                counts[flag] += _bool_count(frame, flag)
        for json_path in output_dir.glob("*.json"):
            payload = _read_json(json_path)
            for flag in SAFETY_FLAGS:
                counts[flag] += _json_flag_count(payload, flag)
    return counts


def build_capability_matrix(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for step, capability, key, notes in STEP_DEFINITIONS:
        output_dir = paths[key]
        present = output_dir.exists()
        rows.append(
            {
                "step": step,
                "output_dir": str(output_dir),
                "status": "present" if present else "missing",
                "capability_added": capability,
                "execution_capability": "none_research_only",
                "trading_ready": False,
                "notes": (
                    f"{notes} Local file count: {_safe_file_count(output_dir)}."
                    if present
                    else f"{notes} Expected output directory is missing."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_guardrail_audit(flag_counts: dict[str, int]) -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The review reads existing V5 Step 1-9 local outputs only.", "No historical backtest is run."),
        ("no_market_data_fetch", "confirmed", "The review has no data loader calls and no market-data source arguments.", "No market data is fetched."),
        ("no_threshold_change", "confirmed", "No strategy threshold module or value is changed.", "Existing thresholds are only referenced as prior context when present."),
        ("no_model_retraining", "confirmed", "No training module is imported or called.", "Model artifacts are unchanged."),
        ("no_feature_engineering_change", "confirmed", "No factor builder or feature engineering module is imported or called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local output directories are read.", "No new data source is added."),
        ("no_broker_sdk_import", "confirmed", "The module imports only standard library modules and pandas.", "No broker SDK is imported."),
        ("no_broker_credentials", "confirmed", "The CLI does not accept credentials and the module does not request credentials.", "No account login or credential storage exists."),
        ("no_broker_connection", "confirmed", "The closure outputs write broker_connected as false.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "The closure outputs write live_trading as false.", "No live trading is performed."),
        ("no_real_order_submission", "confirmed", "The closure outputs write real_order_submission as false.", "No real orders are submitted."),
        ("no_order_execution", "confirmed", "The closure outputs write execution_allowed as false.", "No order execution is enabled."),
        ("no_trading_ready_upgrade", "confirmed", "The closure summary writes trading_ready as false.", "No deployable status is claimed."),
        ("review_only_closure", "confirmed", "The step produces closure summary, capability matrix, guardrail audit, limitations, blockers, recommendations, and report.", "No new trading capability is added."),
        ("educational_research_only", "confirmed", "The report states the project remains educational/research-only.", "Not financial advice."),
    ]
    rows.extend(
        [
            (f"{flag}_true_count_zero", "confirmed" if count == 0 else "blocking", f"Input V5 Step 1-9 scan found {count} true values for {flag}.", "A nonzero count blocks readiness.")
            for flag, count in flag_counts.items()
        ]
    )
    return pd.DataFrame(
        [
            {
                "guardrail": guardrail,
                "status": status,
                "evidence": evidence,
                "notes": notes,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for guardrail, status, evidence, notes in rows
        ]
    )


def build_limitations_register() -> pd.DataFrame:
    limitations = [
        ("no_validated_profitable_strategy", "No validated profitable strategy exists.", "blocking"),
        ("bull_remediation_unresolved_from_v4", "Bull remediation remains unresolved from V4.", "blocking"),
        ("no_live_data_pipeline", "No live data pipeline exists.", "blocking"),
        ("no_broker_execution", "No broker execution integration exists.", "blocking"),
        ("no_automated_order_routing", "No automated order routing exists.", "blocking"),
        ("no_slippage_commission_tax_realistic_execution_model", "No slippage, commission, and tax model has been validated for production-like execution.", "warning"),
        ("no_portfolio_optimizer", "No portfolio optimizer exists.", "warning"),
        ("no_risk_adjusted_production_validation", "No risk-adjusted production validation exists.", "blocking"),
        ("no_paper_trading_over_real_time", "No multi-day real-time paper trading evidence exists.", "blocking"),
        ("no_monitoring_daemon", "No monitoring daemon exists.", "blocking"),
        ("no_autonomous_self_research_agent", "No autonomous self-research agent exists.", "warning"),
        ("no_trading_ready_certification", "No trading-ready certification exists.", "blocking"),
    ]
    return pd.DataFrame(
        [
            {
                "limitation_id": f"V5-LIMIT-{idx:03d}",
                "limitation": limitation,
                "description": description,
                "severity": severity,
                "status": "open",
                "trading_ready": False,
                "notes": "Closure limitation only; future work must address this before any readiness claim.",
            }
            for idx, (limitation, description, severity) in enumerate(limitations, start=1)
        ]
    )


def build_readiness_blockers() -> pd.DataFrame:
    blockers = [
        ("no_profitable_validated_candidate", "No profitable validated candidate exists."),
        ("no_robust_out_of_sample_live_or_paper_evidence", "No robust out-of-sample live or real-time paper evidence exists."),
        ("no_broker_sandbox_or_live_integration", "No broker sandbox or live integration exists."),
        ("no_compliance_risk_approval_layer", "No compliance and risk approval layer exists."),
        ("no_production_monitoring", "No production monitoring process exists."),
        ("no_kill_switch", "No kill switch exists."),
        ("no_real_time_capital_account_reconciliation", "No real-time capital/account reconciliation exists."),
    ]
    return pd.DataFrame(
        [
            {
                "blocker_id": f"V5-BLOCKER-{idx:03d}",
                "blocker": blocker,
                "severity": "blocking",
                "description": description,
                "required_resolution": "future_v6_or_later_work",
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for idx, (blocker, description) in enumerate(blockers, start=1)
        ]
    )


def build_next_phase_recommendations() -> pd.DataFrame:
    rows = [
        ("V6 Step 1", "Multi-day Paper Trading Simulation Harness", "Replay local plans across multiple days with deterministic state transitions before any live consideration."),
        ("V6 Step 2", "Realistic Cost/Slippage/Tax Model", "Add realistic execution-cost assumptions for research validation only."),
        ("V6 Step 3", "Portfolio Risk Exposure Review", "Review concentration, cash use, sector exposure, and drawdown risk across planned positions."),
        ("V6 Step 4", "Strategy Validation Hardening", "Harden out-of-sample, regime, and risk-adjusted validation before any readiness discussion."),
        ("V6 Step 5", "Monitoring Daemon Design", "Design a monitoring daemon and alerting model without implementing live trading."),
    ]
    return pd.DataFrame(
        [
            {
                "phase_step": step,
                "recommendation": recommendation,
                "scope": "future_work_only",
                "rationale": rationale,
                "implementation_status": "not_implemented_in_v5_step10",
                "trading_ready": False,
            }
            for step, recommendation, rationale in rows
        ]
    )


def build_summary(
    capability_matrix: pd.DataFrame,
    guardrail_audit: pd.DataFrame,
    limitations: pd.DataFrame,
    blockers: pd.DataFrame,
    flag_counts: dict[str, int],
) -> pd.DataFrame:
    completed_count = int((capability_matrix["status"] == "present").sum())
    missing_count = int((capability_matrix["status"] == "missing").sum())
    blocking_guardrails = int((guardrail_audit["status"] == "blocking").sum())
    blocking_limitations = int((limitations["severity"] == "blocking").sum())
    blocking_count = int(len(blockers) + blocking_guardrails + blocking_limitations)
    warning_count = int((limitations["severity"] == "warning").sum())
    return pd.DataFrame(
        [
            {
                "summary_item": "v5_capital_aware_infrastructure_closure",
                "reviewed_step_count": int(len(capability_matrix)),
                "completed_step_count": completed_count,
                "missing_step_count": missing_count,
                "trading_ready_true_count": flag_counts["trading_ready"],
                "execution_allowed_true_count": flag_counts["execution_allowed"],
                "broker_connected_true_count": flag_counts["broker_connected"],
                "live_trading_true_count": flag_counts["live_trading"],
                "real_order_submission_true_count": flag_counts["real_order_submission"],
                "blocking_readiness_issue_count": blocking_count,
                "warning_issue_count": warning_count,
                "final_v5_status": "capital_aware_infrastructure_closed_research_only",
                "recommended_next_phase": "V6 validation_and_simulation_hardening",
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    capability_matrix: pd.DataFrame,
    guardrail_audit: pd.DataFrame,
    limitations: pd.DataFrame,
    blockers: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V5 Step 10 Capital-Aware Infrastructure Review / Closure",
            "",
            "## Executive Summary",
            "V5 Step 10 is a review-only closure layer for V5 Steps 1-9.",
            "It reads existing local V5 outputs and summarizes capabilities, limitations, guardrails, readiness blockers, and recommended future V6 work.",
            "It does not add trading capability, run historical backtests, fetch market data, change thresholds, retrain models, change features, add data sources, import broker SDKs, connect to brokers, submit orders, enable live trading, or make a trading-ready claim.",
            "The project remains educational/research-only.",
            "",
            "## Closure Summary",
            f"- Reviewed steps: {row.get('reviewed_step_count', 0)}",
            f"- Completed steps: {row.get('completed_step_count', 0)}",
            f"- Missing steps: {row.get('missing_step_count', 0)}",
            f"- Safety flag true counts: trading_ready={row.get('trading_ready_true_count', 0)}, execution_allowed={row.get('execution_allowed_true_count', 0)}, broker_connected={row.get('broker_connected_true_count', 0)}, live_trading={row.get('live_trading_true_count', 0)}, real_order_submission={row.get('real_order_submission_true_count', 0)}",
            f"- Blocking readiness issues: {row.get('blocking_readiness_issue_count', 0)}",
            f"- Warning issues: {row.get('warning_issue_count', 0)}",
            f"- Final V5 status: {row.get('final_v5_status', '')}",
            f"- Recommended next phase: {row.get('recommended_next_phase', '')}",
            "",
            "## Capability Matrix",
            _table(capability_matrix, "No capability rows were generated."),
            "",
            "## Guardrail Audit",
            _table(guardrail_audit, "No guardrail rows were generated."),
            "",
            "## Limitations Register",
            _table(limitations, "No limitation rows were generated."),
            "",
            "## Readiness Blockers",
            _table(blockers, "No readiness blocker rows were generated."),
            "",
            "## Next Phase Recommendations",
            _table(recommendations, "No recommendation rows were generated."),
            "",
            "## Closure Statement",
            "V5 closes as capital-aware research infrastructure, not as a deployable trading system.",
            "Any future execution, broker connection, credential handling, live data, or readiness certification must be treated as separate future work with explicit controls and validation.",
            "",
        ]
    )


def generate_capital_aware_infrastructure_review_outputs(
    capital_dir: str | Path = DEFAULT_CAPITAL_DIR,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
    position_dir: str | Path = DEFAULT_POSITION_DIR,
    exit_dir: str | Path = DEFAULT_EXIT_DIR,
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    semi_auto_dir: str | Path = DEFAULT_SEMI_AUTO_DIR,
    broker_research_dir: str | Path = DEFAULT_BROKER_RESEARCH_DIR,
    monitoring_dir: str | Path = DEFAULT_MONITORING_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        capital_dir=capital_dir,
        universe_dir=universe_dir,
        position_dir=position_dir,
        exit_dir=exit_dir,
        daily_plan_dir=daily_plan_dir,
        paper_ledger_dir=paper_ledger_dir,
        semi_auto_dir=semi_auto_dir,
        broker_research_dir=broker_research_dir,
        monitoring_dir=monitoring_dir,
    )
    output_path = Path(output_dir)
    flag_counts = scan_safety_flags(paths)
    capability_matrix = build_capability_matrix(paths)
    guardrail_audit = build_guardrail_audit(flag_counts)
    limitations = build_limitations_register()
    blockers = build_readiness_blockers()
    recommendations = build_next_phase_recommendations()
    summary = build_summary(
        capability_matrix,
        guardrail_audit,
        limitations,
        blockers,
        flag_counts,
    )
    report = build_report(
        summary,
        capability_matrix,
        guardrail_audit,
        limitations,
        blockers,
        recommendations,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    out_paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(out_paths["summary"], index=False)
    capability_matrix.to_csv(out_paths["capability_matrix"], index=False)
    guardrail_audit.to_csv(out_paths["guardrail_audit"], index=False)
    limitations.to_csv(out_paths["limitations"], index=False)
    blockers.to_csv(out_paths["blockers"], index=False)
    recommendations.to_csv(out_paths["recommendations"], index=False)
    out_paths["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "reviewed_step_count": int(len(capability_matrix)),
        "completed_step_count": int((capability_matrix["status"] == "present").sum()),
        "missing_step_count": int((capability_matrix["status"] == "missing").sum()),
        "scope": "V5 Step 10 review-only infrastructure closure",
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "closure_summary": summary,
        "capability_matrix": capability_matrix,
        "guardrail_audit": guardrail_audit,
        "limitations_register": limitations,
        "readiness_blockers": blockers,
        "next_phase_recommendations": recommendations,
        "closure_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in out_paths.items()},
    }
