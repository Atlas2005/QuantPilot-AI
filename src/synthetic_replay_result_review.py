import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PRICE_PATH_DIR = Path("outputs/replay_price_path_simulator_real_v1")
DEFAULT_REVIEW_DIR = Path("outputs/simulation_hardening_review_real_v1")
DEFAULT_REPLAY_DIR = Path("outputs/multi_day_paper_replay_harness_real_v1")
DEFAULT_DESIGN_DIR = Path("outputs/simulation_hardening_design_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/synthetic_replay_result_review_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "summary": "synthetic_replay_result_summary.csv",
    "classification": "synthetic_replay_scenario_classification.csv",
    "risk_register": "synthetic_replay_risk_register.csv",
    "guardrails": "synthetic_replay_guardrails.csv",
    "report": "synthetic_replay_review_report.md",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
    "market_data_fetch",
]

INPUT_DEFINITIONS = [
    ("DEP-001", "price_path", "replay_price_path_summary.csv", "Step 11 synthetic price path summary"),
    ("DEP-002", "price_path", "replay_price_path_position_results.csv", "Step 11 scenario-position outcomes"),
    ("DEP-003", "price_path", "synthetic_price_scenarios.csv", "Step 11 synthetic scenario definitions"),
    ("DEP-004", "price_path", "replay_price_paths.csv", "Step 11 synthetic price path rows"),
    ("DEP-005", "price_path", "replay_price_path_event_log.csv", "Step 11 simulator event log"),
    ("DEP-006", "price_path", "replay_price_path_guardrails.csv", "Step 11 simulator guardrails"),
    ("DEP-007", "review", "simulation_hardening_review_summary.csv", "Step 10 closure review summary"),
    ("DEP-008", "replay", "multi_day_replay_summary.csv", "Step 9 replay scaffold summary"),
    ("DEP-009", "design", "simulation_hardening_design_summary.csv", "Step 8 design summary"),
]


def build_input_paths(
    price_path_dir: str | Path = DEFAULT_PRICE_PATH_DIR,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    replay_dir: str | Path = DEFAULT_REPLAY_DIR,
    design_dir: str | Path = DEFAULT_DESIGN_DIR,
) -> dict[str, Path]:
    return {
        "price_path": Path(price_path_dir),
        "review": Path(review_dir),
        "replay": Path(replay_dir),
        "design": Path(design_dir),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"symbol": str})
    except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError):
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
        count = 1 if flag in value and _is_true(value[flag]) else 0
        return count + sum(_json_flag_count(child, flag) for child in value.values())
    if isinstance(value, list):
        return sum(_json_flag_count(child, flag) for child in value)
    return 0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_input_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for dependency_id, key, filename, purpose in INPUT_DEFINITIONS:
        source_dir = paths[key]
        source_file = source_dir / filename
        frame = _read_csv(source_file)
        run_config = _read_json(source_dir / "run_config.json")
        forbidden = sum(_bool_count(frame, flag) for flag in SAFETY_FLAGS)
        forbidden += sum(_json_flag_count(run_config, flag) for flag in SAFETY_FLAGS)
        rows.append(
            {
                "dependency_id": dependency_id,
                "dependency_name": key,
                "source_dir": str(source_dir),
                "expected_file": filename,
                "source_file": str(source_file),
                "source_file_exists": source_file.exists(),
                "row_count": int(len(frame)),
                "purpose": purpose,
                "input_use": "local_read_only_synthetic_replay_review_input",
                "forbidden_true_flag_count": int(forbidden),
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def classify_scenarios(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if results.empty:
        return pd.DataFrame()
    for _, row in results.iterrows():
        outcome = str(row.get("scenario_exit_reason", "unresolved_or_needs_more_replay_evidence"))
        stop = outcome == "synthetic_stop_loss_touch"
        take = outcome == "synthetic_take_profit_touch"
        max_or_no_exit = outcome == "synthetic_max_holding_reached_or_no_exit"
        if stop:
            risk_level = "high"
            note = "Synthetic path touched the stop-loss level; downside scenario remains a risk-control focus."
        elif max_or_no_exit:
            risk_level = "medium"
            note = "Synthetic path did not trigger an exit before max-holding/no-exit classification; more scenario evidence is needed."
        elif take:
            risk_level = "low"
            note = "Synthetic path touched take-profit, but this is not profitability evidence."
        else:
            outcome = "unresolved_or_needs_more_replay_evidence"
            risk_level = "high"
            note = "Outcome could not be classified from local synthetic result evidence."
        rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "symbol": row.get("symbol", ""),
                "scenario_name": row.get("scenario_name", ""),
                "outcome_type": outcome,
                "exit_event_detected": bool(stop or take),
                "stop_loss_touched": bool(stop),
                "take_profit_touched": bool(take),
                "max_holding_reached_or_no_exit": bool(max_or_no_exit),
                "risk_level": risk_level,
                "review_note": note,
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_risk_register(classification: pd.DataFrame) -> pd.DataFrame:
    stop_count = int(classification["stop_loss_touched"].sum()) if not classification.empty else 0
    no_exit_count = int(classification["max_holding_reached_or_no_exit"].sum()) if not classification.empty else 0
    unresolved_count = int((classification["outcome_type"] == "unresolved_or_needs_more_replay_evidence").sum()) if not classification.empty else 0
    rows = [
        (
            "RISK-001",
            "synthetic_stop_loss_sensitivity",
            "high" if stop_count else "medium",
            f"{stop_count} synthetic scenario(s) touched stop-loss.",
            "Expand local synthetic downside paths before any stronger simulation interpretation.",
        ),
        (
            "RISK-002",
            "synthetic_no_exit_or_max_holding_concentration",
            "medium" if no_exit_count else "low",
            f"{no_exit_count} scenario(s) reached max-holding/no-exit classification.",
            "Review whether additional scenario families are needed for unresolved holding behavior.",
        ),
        (
            "RISK-003",
            "not_real_market_validation",
            "high",
            "Synthetic replay is not real market validation and does not use historical or live replay prices.",
            "Keep real-market validation blocker open.",
        ),
        (
            "RISK-004",
            "not_broker_paper_or_live_evidence",
            "high",
            "Synthetic replay is not broker paper trading, not live trading, and not broker integration.",
            "Keep broker reconciliation, monitoring, and execution blockers open.",
        ),
        (
            "RISK-005",
            "unresolved_or_needs_more_replay_evidence",
            "high" if unresolved_count else "medium",
            f"{unresolved_count} scenario(s) could not be classified from current local evidence.",
            "Carry forward scenario expansion and interpretation review.",
        ),
        (
            "RISK-006",
            "no_trading_ready_evidence",
            "high",
            "Synthetic scenario outcomes are not trading-ready evidence.",
            "Keep trading_ready=False until separate future validation and governance evidence exists.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "risk_id": risk_id,
                "risk_name": name,
                "risk_level": level,
                "evidence": evidence,
                "recommended_resolution": resolution,
                "current_status": "open_research_risk",
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for risk_id, name, level, evidence, resolution in rows
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Step 12 reads Step 11 outputs and classifies results only."),
        ("no_market_data_fetch", "confirmed", "No market-data loader is imported or called."),
        ("no_live_data", "confirmed", "No live data path is accepted or used."),
        ("no_model_retraining", "confirmed", "No training module is imported or called."),
        ("no_threshold_change", "confirmed", "No strategy threshold is modified."),
        ("no_feature_engineering_change", "confirmed", "No feature engineering module is modified or called."),
        ("no_new_external_data_sources", "confirmed", "Only existing local V6 outputs are read."),
        ("no_broker_sdk_import", "confirmed", "No broker SDK is imported."),
        ("no_broker_credentials", "confirmed", "No credential argument, token, account id, or secret is accepted."),
        ("no_broker_connection", "confirmed", "No broker connection path exists."),
        ("no_order_execution", "confirmed", "No order execution function exists."),
        ("no_real_order_submission", "confirmed", "No real order submission path exists."),
        ("no_trading_ready_upgrade", "confirmed", "All outputs preserve trading_ready=False."),
        ("synthetic_replay_review_only", "confirmed", "Outputs are classification, risk review, guardrail, summary, and report artifacts."),
        ("educational_research_only", "confirmed", "The report states synthetic replay is research-only."),
    ]
    return pd.DataFrame(
        [
            {
                "guardrail": guardrail,
                "status": status,
                "evidence": evidence,
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for guardrail, status, evidence in rows
        ]
    )


def build_summary(manifest: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    missing = int((~manifest["source_file_exists"].astype(bool)).sum()) if not manifest.empty else 0
    forbidden = int(manifest["forbidden_true_flag_count"].sum()) if not manifest.empty else 0
    reviewed = int(len(classification))
    stop = int(classification["stop_loss_touched"].sum()) if not classification.empty else 0
    take = int(classification["take_profit_touched"].sum()) if not classification.empty else 0
    max_or_no_exit = int(classification["max_holding_reached_or_no_exit"].sum()) if not classification.empty else 0
    unresolved = int((classification["outcome_type"] == "unresolved_or_needs_more_replay_evidence").sum()) if not classification.empty else 0
    high_risk = int((classification["risk_level"] == "high").sum()) if not classification.empty else 0
    validation_status = "pass" if reviewed > 0 and missing == 0 and forbidden == 0 else "warning"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step12_synthetic_replay_result_review",
                "reviewed_input_count": int(manifest["source_file_exists"].astype(bool).sum()) if not manifest.empty else 0,
                "missing_input_count": missing,
                "reviewed_scenario_count": reviewed,
                "stop_loss_touch_count": stop,
                "take_profit_touch_count": take,
                "max_holding_or_no_exit_count": max_or_no_exit,
                "unresolved_scenario_count": unresolved,
                "high_risk_scenario_count": high_risk,
                "synthetic_exit_event_count": stop + take,
                "market_data_fetch_count": 0,
                "broker_connected_count": 0,
                "execution_allowed_count": 0,
                "live_trading_count": 0,
                "real_order_submission_count": 0,
                "forbidden_true_flag_count": forbidden,
                "trading_ready": False,
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
                "validation_status": validation_status,
                "conclusion": "synthetic_replay_result_review_completed_research_only",
                "recommended_next_step": "V6 Step 13 Synthetic Replay Stress Matrix / Scenario Expansion Plan",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    classification: pd.DataFrame,
    risk_register: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 12 Synthetic Replay Result Review / Scenario Risk Classification",
            "",
            "## Executive Summary",
            "Step 12 reviews local synthetic replay outcomes from Step 11 and classifies scenario-level risk.",
            "Synthetic replay is not real market validation, not broker paper trading, and not live evidence.",
            "Remaining simulation-hardening blockers stay open and trading_ready remains False.",
            "",
            "## Summary",
            f"- Reviewed scenarios: {row.get('reviewed_scenario_count', 0)}",
            f"- Stop-loss touches: {row.get('stop_loss_touch_count', 0)}",
            f"- Take-profit touches: {row.get('take_profit_touch_count', 0)}",
            f"- Max-holding/no-exit outcomes: {row.get('max_holding_or_no_exit_count', 0)}",
            f"- Unresolved scenarios: {row.get('unresolved_scenario_count', 0)}",
            f"- High-risk scenarios: {row.get('high_risk_scenario_count', 0)}",
            f"- Synthetic exit events: {row.get('synthetic_exit_event_count', 0)}",
            f"- Market data fetches: {row.get('market_data_fetch_count', 0)}",
            f"- Broker connected count: {row.get('broker_connected_count', 0)}",
            f"- Execution allowed count: {row.get('execution_allowed_count', 0)}",
            f"- Live trading count: {row.get('live_trading_count', 0)}",
            f"- Real order submission count: {row.get('real_order_submission_count', 0)}",
            f"- Trading ready: {row.get('trading_ready', False)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            f"- Recommended next step: {row.get('recommended_next_step', '')}",
            "",
            "## Scenario Classification",
            _table(classification, "No scenario classifications were generated."),
            "",
            "## Risk Register",
            _table(risk_register, "No risk register rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This review does not fetch market data, run backtests, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or claim trading-ready status.",
            "",
        ]
    )


def generate_synthetic_replay_result_review_outputs(
    price_path_dir: str | Path = DEFAULT_PRICE_PATH_DIR,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    replay_dir: str | Path = DEFAULT_REPLAY_DIR,
    design_dir: str | Path = DEFAULT_DESIGN_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        price_path_dir=price_path_dir,
        review_dir=review_dir,
        replay_dir=replay_dir,
        design_dir=design_dir,
    )
    manifest = build_input_manifest(paths)
    results = _read_csv(paths["price_path"] / "replay_price_path_position_results.csv")
    classification = classify_scenarios(results)
    risk_register = build_risk_register(classification)
    guardrails = build_guardrails()
    summary = build_summary(manifest, classification)
    report = build_report(summary, classification, risk_register, guardrails)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(output_files["summary"], index=False)
    classification.to_csv(output_files["classification"], index=False)
    risk_register.to_csv(output_files["risk_register"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "reviewed_scenario_count": int(summary.iloc[0]["reviewed_scenario_count"]),
        "scope": "V6 Step 12 synthetic replay result review and risk classification only",
        "market_data_fetch": False,
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    output_files["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "synthetic_replay_result_summary": summary,
        "synthetic_replay_scenario_classification": classification,
        "synthetic_replay_risk_register": risk_register,
        "synthetic_replay_guardrails": guardrails,
        "synthetic_replay_review_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
