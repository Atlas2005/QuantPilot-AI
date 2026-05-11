import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PRICE_PATH_DIR = Path("outputs/replay_price_path_simulator_real_v1")
DEFAULT_RESULT_REVIEW_DIR = Path("outputs/synthetic_replay_result_review_real_v1")
DEFAULT_HARDENING_REVIEW_DIR = Path("outputs/simulation_hardening_review_real_v1")
DEFAULT_REPLAY_DIR = Path("outputs/multi_day_paper_replay_harness_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/synthetic_replay_stress_matrix_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "summary": "synthetic_replay_stress_matrix_summary.csv",
    "dimensions": "synthetic_replay_stress_dimensions.csv",
    "matrix": "synthetic_replay_stress_matrix.csv",
    "expansion_plan": "synthetic_replay_expansion_plan.csv",
    "risk_register": "synthetic_replay_stress_risk_register.csv",
    "guardrails": "synthetic_replay_stress_guardrails.csv",
    "report": "synthetic_replay_stress_report.md",
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
    ("DEP-001", "price_path", "replay_price_path_summary.csv", "Step 11 synthetic simulator summary"),
    ("DEP-002", "price_path", "synthetic_price_scenarios.csv", "Step 11 current scenario definitions"),
    ("DEP-003", "price_path", "replay_price_path_position_results.csv", "Step 11 current scenario outcomes"),
    ("DEP-004", "result_review", "synthetic_replay_result_summary.csv", "Step 12 result review summary"),
    ("DEP-005", "result_review", "synthetic_replay_scenario_classification.csv", "Step 12 scenario classifications"),
    ("DEP-006", "result_review", "synthetic_replay_risk_register.csv", "Step 12 risk register"),
    ("DEP-007", "hardening_review", "simulation_hardening_review_summary.csv", "Step 10 hardening review summary"),
    ("DEP-008", "replay", "multi_day_replay_summary.csv", "Step 9 replay scaffold summary"),
]


def build_input_paths(
    price_path_dir: str | Path = DEFAULT_PRICE_PATH_DIR,
    result_review_dir: str | Path = DEFAULT_RESULT_REVIEW_DIR,
    hardening_review_dir: str | Path = DEFAULT_HARDENING_REVIEW_DIR,
    replay_dir: str | Path = DEFAULT_REPLAY_DIR,
) -> dict[str, Path]:
    return {
        "price_path": Path(price_path_dir),
        "result_review": Path(result_review_dir),
        "hardening_review": Path(hardening_review_dir),
        "replay": Path(replay_dir),
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


def _first_row_value(df: pd.DataFrame, column: str, default: Any = 0) -> Any:
    if df.empty or column not in df:
        return default
    value = df.iloc[0][column]
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


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
                "input_use": "local_read_only_stress_matrix_design_input",
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


def build_stress_dimensions() -> pd.DataFrame:
    rows = [
        ("DIM-001", "larger_gap_down", "downside_gap", "high", "Open below stop-loss reference to test overnight-style synthetic downside sensitivity."),
        ("DIM-002", "slow_grind_down", "downside_trend", "high", "Decline gradually toward stop-loss without an immediate touch."),
        ("DIM-003", "fast_rebound", "reversal", "medium", "Drop early and rebound quickly to test path-order sensitivity."),
        ("DIM-004", "volatile_stop_loss_whipsaw", "volatility", "high", "Touch stop-loss then rebound to expose whipsaw behavior."),
        ("DIM-005", "near_take_profit_reversal", "upside_reversal", "medium", "Approach take-profit then reverse before exit."),
        ("DIM-006", "flat_illiquid_path", "liquidity_placeholder", "medium", "Remain flat with low synthetic movement and unresolved exit behavior."),
        ("DIM-007", "delayed_stop_loss_touch", "delayed_downside", "high", "Touch stop-loss late in the max-holding horizon."),
        ("DIM-008", "delayed_take_profit_touch", "delayed_upside", "medium", "Touch take-profit late in the max-holding horizon."),
        ("DIM-009", "max_holding_unresolved_path", "holding_period", "high", "Never touch stop-loss or take-profit before max-holding review."),
        ("DIM-010", "benchmark_lag_stress_path", "benchmark_placeholder", "high", "Design placeholder for benchmark-lag underperformance stress without fetching benchmark data."),
    ]
    return pd.DataFrame(
        [
            {
                "dimension_id": dimension_id,
                "stress_dimension": dimension,
                "stress_family": family,
                "priority": priority,
                "design_rationale": rationale,
                "design_only": True,
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for dimension_id, dimension, family, priority, rationale in rows
        ]
    )


def build_stress_matrix(dimensions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in dimensions.iterrows():
        scenario_id = f"STRESS-{index + 1:03d}"
        rows.append(
            {
                "stress_scenario_id": scenario_id,
                "dimension_id": row["dimension_id"],
                "stress_dimension": row["stress_dimension"],
                "proposed_scenario_name": f"{row['stress_dimension']}_design",
                "path_shape": row["stress_family"],
                "expected_exit_focus": _expected_exit_focus(str(row["stress_dimension"])),
                "coverage_gap_addressed": _coverage_gap(str(row["stress_dimension"])),
                "priority": row["priority"],
                "implementation_status": "planned_not_executed",
                "design_only": True,
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _expected_exit_focus(dimension: str) -> str:
    if "stop_loss" in dimension or "gap_down" in dimension or "grind_down" in dimension:
        return "stop_loss_and_downside_path_order"
    if "take_profit" in dimension or "rebound" in dimension:
        return "take_profit_and_reversal_path_order"
    if "max_holding" in dimension or "flat" in dimension:
        return "max_holding_or_no_exit_behavior"
    if "benchmark" in dimension:
        return "benchmark_lag_placeholder_behavior"
    return "scenario_expansion_coverage"


def _coverage_gap(dimension: str) -> str:
    if "benchmark" in dimension:
        return "benchmark_lag_rule_placeholder_not_stressed"
    if "flat" in dimension or "max_holding" in dimension:
        return "max_holding_no_exit_concentration"
    if "take_profit" in dimension or "rebound" in dimension:
        return "upside_reversal_and_profit_capture_sensitivity"
    return "downside_stop_loss_sensitivity"


def build_expansion_plan(stress_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in stress_matrix.iterrows():
        priority = "high" if row["priority"] == "high" else "medium"
        rows.append(
            {
                "plan_id": f"EXPAND-{index + 1:03d}",
                "stress_scenario_id": row["stress_scenario_id"],
                "planned_addition": row["proposed_scenario_name"],
                "priority": priority,
                "planned_evidence": "future_local_synthetic_path_rows_only",
                "acceptance_note": "Future implementation must remain local synthetic scenario analysis and preserve false safety flags.",
                "current_step_status": "design_only_not_implemented",
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_risk_register(result_summary: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    high_risk = _safe_int(_first_row_value(result_summary, "high_risk_scenario_count"))
    no_exit = _safe_int(_first_row_value(result_summary, "max_holding_or_no_exit_count"))
    risk_rows = [
        ("RISK-001", "current_scenario_set_too_small", "high", "Current Step 11 scenario set has 6 scenarios.", "Expand synthetic scenario families before interpreting robustness."),
        ("RISK-002", "downside_path_coverage_incomplete", "high", f"Step 12 high-risk scenario count is {high_risk}.", "Add gap-down, grind-down, whipsaw, and delayed stop-loss paths."),
        ("RISK-003", "max_holding_no_exit_concentration", "high" if no_exit else "medium", f"Step 12 max-holding/no-exit count is {no_exit}.", "Add unresolved and flat illiquid stress paths."),
        ("RISK-004", "benchmark_lag_not_stressed", "high", "Benchmark-lag rule remains a placeholder without benchmark data.", "Design local placeholder stress before any benchmark-aware simulation."),
        ("RISK-005", "not_real_market_validation", "high", "Stress matrix is planning-only and does not use real market replay prices.", "Keep real-market validation blocker open."),
        ("RISK-006", "not_trading_ready_evidence", "high", "Stress matrix design rows are not trading-ready evidence.", "Keep trading_ready=False and require future validation layers."),
    ]
    if not classification.empty and "risk_level" in classification:
        medium_count = int((classification["risk_level"] == "medium").sum())
        risk_rows.append(("RISK-007", "medium_risk_scenario_followup", "medium", f"Step 12 medium-risk scenario count is {medium_count}.", "Prioritize medium-risk no-exit paths after high-risk downside expansion."))
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
            for risk_id, name, level, evidence, resolution in risk_rows
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Step 13 produces stress matrix design rows only."),
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
        ("stress_matrix_design_only", "confirmed", "Outputs are scenario expansion planning artifacts, not simulations."),
        ("educational_research_only", "confirmed", "The report states educational/research-only scope."),
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


def build_summary(
    manifest: pd.DataFrame,
    scenarios: pd.DataFrame,
    result_summary: pd.DataFrame,
    dimensions: pd.DataFrame,
    stress_matrix: pd.DataFrame,
    expansion_plan: pd.DataFrame,
) -> pd.DataFrame:
    missing = int((~manifest["source_file_exists"].astype(bool)).sum()) if not manifest.empty else 0
    forbidden = int(manifest["forbidden_true_flag_count"].sum()) if not manifest.empty else 0
    high_priority = int((expansion_plan["priority"] == "high").sum()) if not expansion_plan.empty else 0
    validation_status = "pass" if missing == 0 and forbidden == 0 and len(stress_matrix) > 0 else "warning"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step13_synthetic_replay_stress_matrix",
                "reviewed_input_count": int(manifest["source_file_exists"].astype(bool).sum()) if not manifest.empty else 0,
                "missing_input_count": missing,
                "existing_scenario_count": int(len(scenarios)),
                "existing_high_risk_scenario_count": _safe_int(_first_row_value(result_summary, "high_risk_scenario_count")),
                "proposed_stress_dimension_count": int(len(dimensions)),
                "proposed_stress_scenario_count": int(len(stress_matrix)),
                "expansion_plan_row_count": int(len(expansion_plan)),
                "high_priority_expansion_count": high_priority,
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
                "conclusion": "synthetic_replay_stress_matrix_completed_research_only",
                "recommended_next_step": "V6 Step 14 Synthetic Stress Matrix Implementation / Local Scenario Generator",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    dimensions: pd.DataFrame,
    stress_matrix: pd.DataFrame,
    expansion_plan: pd.DataFrame,
    risk_register: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 13 Synthetic Replay Stress Matrix / Scenario Expansion Plan",
            "",
            "## Executive Summary",
            "Step 13 creates a research-only stress matrix design for expanding local synthetic replay scenarios.",
            "It is a design/planning layer only and does not execute new price simulations.",
            "It does not run backtests, fetch market or live data, retrain models, change thresholds or features, connect to brokers, execute orders, submit orders, or upgrade readiness.",
            "",
            "## Summary",
            f"- Reviewed inputs: {row.get('reviewed_input_count', 0)}",
            f"- Missing inputs: {row.get('missing_input_count', 0)}",
            f"- Existing scenarios: {row.get('existing_scenario_count', 0)}",
            f"- Existing high-risk scenarios: {row.get('existing_high_risk_scenario_count', 0)}",
            f"- Proposed stress dimensions: {row.get('proposed_stress_dimension_count', 0)}",
            f"- Proposed stress scenarios: {row.get('proposed_stress_scenario_count', 0)}",
            f"- Expansion plan rows: {row.get('expansion_plan_row_count', 0)}",
            f"- High-priority expansions: {row.get('high_priority_expansion_count', 0)}",
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
            "## Stress Dimensions",
            _table(dimensions, "No stress dimensions were generated."),
            "",
            "## Stress Matrix",
            _table(stress_matrix, "No stress matrix rows were generated."),
            "",
            "## Expansion Plan",
            _table(expansion_plan, "No expansion plan rows were generated."),
            "",
            "## Risk Register",
            _table(risk_register, "No risk register rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This stress matrix is a planning layer only. It is not a historical backtest, not real market validation, not broker paper trading, not live trading, and not trading-ready evidence.",
            "",
        ]
    )


def generate_synthetic_replay_stress_matrix_outputs(
    price_path_dir: str | Path = DEFAULT_PRICE_PATH_DIR,
    result_review_dir: str | Path = DEFAULT_RESULT_REVIEW_DIR,
    hardening_review_dir: str | Path = DEFAULT_HARDENING_REVIEW_DIR,
    replay_dir: str | Path = DEFAULT_REPLAY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        price_path_dir=price_path_dir,
        result_review_dir=result_review_dir,
        hardening_review_dir=hardening_review_dir,
        replay_dir=replay_dir,
    )
    manifest = build_input_manifest(paths)
    scenarios = _read_csv(paths["price_path"] / "synthetic_price_scenarios.csv")
    result_summary = _read_csv(paths["result_review"] / "synthetic_replay_result_summary.csv")
    classification = _read_csv(paths["result_review"] / "synthetic_replay_scenario_classification.csv")
    dimensions = build_stress_dimensions()
    stress_matrix = build_stress_matrix(dimensions)
    expansion_plan = build_expansion_plan(stress_matrix)
    risk_register = build_risk_register(result_summary, classification)
    guardrails = build_guardrails()
    summary = build_summary(manifest, scenarios, result_summary, dimensions, stress_matrix, expansion_plan)
    report = build_report(summary, dimensions, stress_matrix, expansion_plan, risk_register, guardrails)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(output_files["summary"], index=False)
    dimensions.to_csv(output_files["dimensions"], index=False)
    stress_matrix.to_csv(output_files["matrix"], index=False)
    expansion_plan.to_csv(output_files["expansion_plan"], index=False)
    risk_register.to_csv(output_files["risk_register"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "proposed_stress_dimension_count": int(len(dimensions)),
        "proposed_stress_scenario_count": int(len(stress_matrix)),
        "scope": "V6 Step 13 synthetic replay stress matrix design only",
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
        "synthetic_replay_stress_matrix_summary": summary,
        "synthetic_replay_stress_dimensions": dimensions,
        "synthetic_replay_stress_matrix": stress_matrix,
        "synthetic_replay_expansion_plan": expansion_plan,
        "synthetic_replay_stress_risk_register": risk_register,
        "synthetic_replay_stress_guardrails": guardrails,
        "synthetic_replay_stress_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
