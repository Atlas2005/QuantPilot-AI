import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("outputs/semi_auto_order_generator_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/broker_integration_research_real_v1")

OUTPUT_FILENAMES = {
    "summary": "broker_integration_summary.csv",
    "modes": "broker_integration_modes.csv",
    "constraints": "broker_integration_constraints.csv",
    "risks": "broker_integration_risk_register.csv",
    "guardrails": "broker_integration_guardrails.csv",
    "report": "broker_integration_research_report.md",
    "run_config": "run_config.json",
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _format_symbol(value: Any) -> str:
    text = _clean_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"symbol": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "symbol" in df:
        df["symbol"] = df["symbol"].map(_format_symbol)
    for column in [
        "broker_connected",
        "execution_allowed",
        "trading_ready",
        "live_trading",
        "real_order_submission",
    ]:
        if column in df:
            df[column] = False
    return df.reset_index(drop=True)


def build_modes() -> pd.DataFrame:
    rows = [
        (
            "manual_review_only",
            "Human reads local research outputs and manually decides whether any separate action is appropriate.",
            "current_allowed_research_mode",
            "lowest",
        ),
        (
            "broker_neutral_ticket_export",
            "Export broker-neutral draft tickets without broker-specific routing fields or credentials.",
            "current_allowed_research_mode",
            "low",
        ),
        (
            "paper_trading_only",
            "Maintain deterministic paper ledgers without live quotes, broker APIs, or real cash movement.",
            "current_allowed_research_mode",
            "low",
        ),
        (
            "broker_api_research_only",
            "Document possible broker API requirements, risks, permissions, and operational controls.",
            "research_only_not_implemented",
            "high",
        ),
        (
            "future_human_approved_broker_bridge",
            "Potential future bridge that would require explicit human approval, credentials handling, legal review, and separate safeguards.",
            "future_research_only_not_implemented",
            "highest",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "integration_mode": mode,
                "description": description,
                "implementation_status": status,
                "risk_level": risk,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
                "notes": "Research classification only. No broker integration is implemented.",
            }
            for mode, description, status, risk in rows
        ]
    )


def build_constraints(order_drafts: pd.DataFrame, checklist: pd.DataFrame) -> pd.DataFrame:
    draft_count = int(len(order_drafts))
    checklist_count = int(len(checklist))
    rows = [
        ("account_login_credential_risk", "high", "Credentials must never be requested or stored by this research step.", "No credential collection or broker login flow exists."),
        ("broker_api_availability", "medium", "Broker APIs vary by broker, account type, region, and product.", "Only research documentation is produced."),
        ("region_market_restrictions", "high", "Market access rules can vary by jurisdiction and instrument.", "No market access is assumed."),
        ("two_factor_authentication", "high", "2FA flows are interactive and sensitive.", "No authentication flow is implemented."),
        ("trading_permissions", "high", "Account permissions may block order types, markets, or products.", "No account permissions are queried."),
        ("order_type_support", "medium", "Supported order types differ across brokers.", "Step 7 tickets remain broker-neutral drafts."),
        ("lot_size_compatibility", "medium", "Broker and exchange lot rules must be validated before any real order.", f"Current draft order count: {draft_count}."),
        ("minimum_cash_capital_constraints", "medium", "Cash and margin rules require broker/account validation.", "Only prior local capital planning outputs are summarized."),
        ("rate_limits", "medium", "Broker APIs may throttle requests or reject bursts.", "No API calls are made."),
        ("market_hours", "medium", "Order handling depends on session calendars and local market hours.", "No live session state is queried."),
        ("quote_latency", "high", "Live quotes can be delayed, stale, or unavailable.", "No live market data is fetched."),
        ("failure_handling", "high", "Execution systems require robust retry, cancel, partial-fill, and outage handling.", "No execution system exists in this step."),
        ("audit_logging", "medium", "Any future bridge would need immutable audit trails.", "Research outputs are local CSV/Markdown only."),
        ("manual_confirmation", "high", "Human review must remain explicit before any external action.", f"Current checklist rows: {checklist_count}."),
        ("legal_compliance_limitations", "high", "Broker automation may require legal, compliance, tax, and suitability review.", "No trading-ready or deployable claim is made."),
    ]
    return pd.DataFrame(
        [
            {
                "constraint": constraint,
                "risk_level": risk_level,
                "description": description,
                "current_step_handling": handling,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
                "notes": "Research-only constraint row.",
            }
            for constraint, risk_level, description, handling in rows
        ]
    )


def build_risk_register(constraints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in constraints.reset_index(drop=True).iterrows():
        risk_level = _clean_text(row.get("risk_level"))
        priority = "P1" if risk_level == "high" else "P2" if risk_level == "medium" else "P3"
        rows.append(
            {
                "risk_id": f"BROKER-RISK-{idx + 1:03d}",
                "risk_name": row.get("constraint"),
                "risk_level": risk_level,
                "priority": priority,
                "risk_description": row.get("description"),
                "mitigation_status": "documented_not_implemented",
                "required_control": "human_review_and_separate_approval_before_any_future_broker_work",
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
                "notes": "Risk register entry only. No broker bridge is created.",
            }
        )
    return pd.DataFrame(rows)


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The step reads Step 7 local outputs and writes research CSV/Markdown files only.", "No historical backtest is run."),
        ("no_threshold_change", "confirmed", "No signal threshold module or value is changed.", "Signal thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local Step 7 CSV outputs are read when available.", "No market data source is added and no live data is fetched."),
        ("no_broker_credentials", "confirmed", "The CLI does not accept credentials and the module does not request credentials.", "No account login or credential storage is implemented."),
        ("no_broker_sdk_import", "confirmed", "The module imports only standard library modules and pandas.", "No broker SDK is imported."),
        ("no_broker_connection", "confirmed", "broker_connected=False is written to research outputs.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "live_trading=False is written to research outputs.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "execution_allowed=False and real_order_submission=False are written to research outputs.", "No orders are executed or submitted."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False is written to Step 8 outputs.", "No deployable status is claimed."),
        ("broker_research_only", "confirmed", "The output classifies possible broker integration modes and constraints only.", "No broker bridge is created."),
        ("human_review_required", "confirmed", "Human review remains a required future control for any external broker action.", "No automated execution approval exists."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def build_summary(
    order_drafts: pd.DataFrame,
    modes: pd.DataFrame,
    constraints: pd.DataFrame,
    risks: pd.DataFrame,
) -> pd.DataFrame:
    high_risk_count = int((constraints["risk_level"] == "high").sum()) if not constraints.empty else 0
    return pd.DataFrame(
        [
            {
                "summary_item": "broker_integration_research_run",
                "input_draft_order_count": int(len(order_drafts)),
                "researched_mode_count": int(len(modes)),
                "constraint_count": int(len(constraints)),
                "high_risk_constraint_count": high_risk_count,
                "risk_register_count": int(len(risks)),
                "broker_connected_count": 0,
                "execution_allowed_count": 0,
                "trading_ready_count": 0,
                "live_trading_count": 0,
                "real_order_submission_count": 0,
                "conclusion": "broker_integration_research_only_no_execution",
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
    modes: pd.DataFrame,
    constraints: pd.DataFrame,
    risks: pd.DataFrame,
    guardrails: pd.DataFrame,
    input_dir: str | Path,
    output_dir: str | Path,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V5 Step 8 Broker Integration Research Report",
            "",
            "## Executive Summary",
            "V5 Step 8 researches broker integration constraints and risks only.",
            "This step does not connect to any broker.",
            "This step does not request or store credentials.",
            "This step does not place orders.",
            "This step does not simulate real broker routing.",
            "This step does not fetch live market data.",
            "All outputs preserve broker_connected=False, execution_allowed=False, live_trading=False, real_order_submission=False, and trading_ready=False.",
            "The project remains educational/research-only and not trading-ready.",
            "",
            "## Inputs",
            f"- Input directory: {input_dir}",
            f"- Output directory: {output_dir}",
            f"- Input draft order count: {row.get('input_draft_order_count', 0)}",
            "",
            "## Summary",
            f"- Researched integration modes: {row.get('researched_mode_count', 0)}",
            f"- Constraint count: {row.get('constraint_count', 0)}",
            f"- High-risk constraints: {row.get('high_risk_constraint_count', 0)}",
            f"- Broker connected count: {row.get('broker_connected_count', 0)}",
            f"- Execution allowed count: {row.get('execution_allowed_count', 0)}",
            f"- Trading ready count: {row.get('trading_ready_count', 0)}",
            f"- Live trading count: {row.get('live_trading_count', 0)}",
            f"- Real order submission count: {row.get('real_order_submission_count', 0)}",
            f"- Conclusion: {row.get('conclusion', '')}",
            "",
            "## Integration Modes",
            _table(modes, "No integration modes were generated."),
            "",
            "## Constraints",
            _table(constraints, "No constraints were generated."),
            "",
            "## Risk Register",
            _table(risks, "No risk register rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This is research and educational documentation only. It is not financial advice.",
            "No broker SDK is imported, no broker connection is made, no credentials are requested, and no order execution code is created.",
            "",
        ]
    )


def generate_broker_integration_research_outputs(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    order_drafts = _read_csv(input_path / "order_drafts.csv")
    checklist = _read_csv(input_path / "manual_review_checklist.csv")
    semi_summary = _read_csv(input_path / "semi_auto_order_summary.csv")

    modes = build_modes()
    constraints = build_constraints(order_drafts, checklist)
    risks = build_risk_register(constraints)
    guardrails = build_guardrails()
    summary = build_summary(order_drafts, modes, constraints, risks)
    report = build_report(summary, modes, constraints, risks, guardrails, input_path, output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(paths["summary"], index=False)
    modes.to_csv(paths["modes"], index=False)
    constraints.to_csv(paths["constraints"], index=False)
    risks.to_csv(paths["risks"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    paths["report"].write_text(report, encoding="utf-8")
    config = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "order_drafts_path": str(input_path / "order_drafts.csv"),
        "manual_review_checklist_path": str(input_path / "manual_review_checklist.csv"),
        "semi_auto_order_summary_path": str(input_path / "semi_auto_order_summary.csv"),
        "input_draft_order_count": int(len(order_drafts)),
        "input_checklist_count": int(len(checklist)),
        "input_summary_rows": int(len(semi_summary)),
        "researched_mode_count": int(len(modes)),
        "constraint_count": int(len(constraints)),
        "scope": "V5 Step 8 broker integration research only",
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "broker_credentials_requested": False,
        "broker_sdk_imported": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "broker_integration_summary": summary,
        "broker_integration_modes": modes,
        "broker_integration_constraints": constraints,
        "broker_integration_risk_register": risks,
        "broker_integration_guardrails": guardrails,
        "broker_integration_research_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
