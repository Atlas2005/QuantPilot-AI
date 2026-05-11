import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PRICE_PATH_DIR = Path("outputs/replay_price_path_simulator_real_v1")
DEFAULT_RESULT_REVIEW_DIR = Path("outputs/synthetic_replay_result_review_real_v1")
DEFAULT_STRESS_MATRIX_DIR = Path("outputs/synthetic_replay_stress_matrix_real_v1")
DEFAULT_HARDENING_REVIEW_DIR = Path("outputs/simulation_hardening_review_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/synthetic_stress_scenario_generator_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "manifest": "synthetic_stress_input_manifest.csv",
    "definitions": "synthetic_stress_scenario_definitions.csv",
    "assumptions": "synthetic_stress_price_path_assumptions.csv",
    "execution_plan": "synthetic_stress_execution_plan.csv",
    "guardrails": "synthetic_stress_guardrails.csv",
    "summary": "synthetic_stress_summary.csv",
    "report": "synthetic_stress_report.md",
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
    ("DEP-001", "price_path", "replay_price_path_summary.csv", "Step 11 simulator summary"),
    ("DEP-002", "price_path", "synthetic_price_scenarios.csv", "Step 11 baseline synthetic scenarios"),
    ("DEP-003", "result_review", "synthetic_replay_result_summary.csv", "Step 12 result review summary"),
    ("DEP-004", "result_review", "synthetic_replay_scenario_classification.csv", "Step 12 scenario risk classification"),
    ("DEP-005", "stress_matrix", "synthetic_replay_stress_matrix_summary.csv", "Step 13 stress matrix summary"),
    ("DEP-006", "stress_matrix", "synthetic_replay_stress_dimensions.csv", "Step 13 stress dimensions"),
    ("DEP-007", "stress_matrix", "synthetic_replay_stress_matrix.csv", "Step 13 proposed stress scenarios"),
    ("DEP-008", "stress_matrix", "synthetic_replay_expansion_plan.csv", "Step 13 expansion plan"),
    ("DEP-009", "hardening_review", "simulation_hardening_review_summary.csv", "Step 10 hardening review summary"),
]


def build_input_paths(
    price_path_dir: str | Path = DEFAULT_PRICE_PATH_DIR,
    result_review_dir: str | Path = DEFAULT_RESULT_REVIEW_DIR,
    stress_matrix_dir: str | Path = DEFAULT_STRESS_MATRIX_DIR,
    hardening_review_dir: str | Path = DEFAULT_HARDENING_REVIEW_DIR,
) -> dict[str, Path]:
    return {
        "price_path": Path(price_path_dir),
        "result_review": Path(result_review_dir),
        "stress_matrix": Path(stress_matrix_dir),
        "hardening_review": Path(hardening_review_dir),
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
                "input_use": "local_read_only_synthetic_stress_definition_input",
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


def _assumptions_for_dimension(dimension: str) -> dict[str, str]:
    mapping = {
        "larger_gap_down": {
            "shape": "single_step_gap_below_reference_then_hold",
            "stop": "designed_to_start_below_or_at_stop_loss_reference",
            "take": "take_profit_not_expected_to_interact",
            "holding": "exit_focus_before_max_holding",
            "benchmark": "not_applicable",
            "exit": "expected_synthetic_stop_loss_focus",
            "limits": "Gap size is synthetic and not calibrated to real opening auction behavior.",
        },
        "slow_grind_down": {
            "shape": "monotonic_decline_toward_stop_loss",
            "stop": "designed_to_test_delayed_or_near_stop_loss_pressure",
            "take": "take_profit_not_expected_to_interact",
            "holding": "may_reach_max_holding_if_stop_not_touched",
            "benchmark": "not_applicable",
            "exit": "expected_stop_loss_or_max_holding_review",
            "limits": "No volatility clustering or liquidity behavior is modeled.",
        },
        "fast_rebound": {
            "shape": "early_drop_then_fast_rebound",
            "stop": "tests_whether_early_drawdown_touches_stop_before_rebound",
            "take": "may_approach_take_profit_after_rebound",
            "holding": "path_order_sensitive_before_max_holding",
            "benchmark": "not_applicable",
            "exit": "expected_path_order_sensitivity",
            "limits": "Rebound path is deterministic and not market-derived.",
        },
        "volatile_stop_loss_whipsaw": {
            "shape": "stop_loss_touch_then_rebound",
            "stop": "explicit_stop_loss_whipsaw_focus",
            "take": "post_stop_rebound_not_interpreted_as_profit_evidence",
            "holding": "exit_focus_before_max_holding",
            "benchmark": "not_applicable",
            "exit": "expected_synthetic_stop_loss_touch",
            "limits": "Does not model real fill quality after stop trigger.",
        },
        "near_take_profit_reversal": {
            "shape": "approach_take_profit_then_reverse",
            "stop": "late_reversal_may_move_toward_stop_but_not_primary_focus",
            "take": "tests_near_take_profit_without_touch_or_with_late_reversal",
            "holding": "may_end_as_no_exit_or_late_exit",
            "benchmark": "not_applicable",
            "exit": "expected_take_profit_reversal_sensitivity",
            "limits": "Does not prove profit capture quality.",
        },
        "flat_illiquid_path": {
            "shape": "low_movement_flat_path",
            "stop": "stop_loss_not_expected_to_interact",
            "take": "take_profit_not_expected_to_interact",
            "holding": "expected_max_holding_or_no_exit_focus",
            "benchmark": "not_applicable",
            "exit": "expected_synthetic_max_holding_or_no_exit",
            "limits": "Illiquidity is a label only; no volume, spread, or queue model exists.",
        },
        "delayed_stop_loss_touch": {
            "shape": "late_horizon_decline_to_stop_loss",
            "stop": "designed_to_touch_stop_loss_late",
            "take": "take_profit_not_expected_to_interact",
            "holding": "tests_late_exit_before_or_at_max_holding",
            "benchmark": "not_applicable",
            "exit": "expected_delayed_synthetic_stop_loss_touch",
            "limits": "Late touch timing is deterministic and not market-derived.",
        },
        "delayed_take_profit_touch": {
            "shape": "late_horizon_rise_to_take_profit",
            "stop": "stop_loss_not_expected_to_interact",
            "take": "designed_to_touch_take_profit_late",
            "holding": "tests_late_profit_exit_before_or_at_max_holding",
            "benchmark": "not_applicable",
            "exit": "expected_delayed_synthetic_take_profit_touch",
            "limits": "Late touch timing is deterministic and not market-derived.",
        },
        "max_holding_unresolved_path": {
            "shape": "bounded_path_avoids_both_exits",
            "stop": "designed_to_avoid_stop_loss",
            "take": "designed_to_avoid_take_profit",
            "holding": "explicit_max_holding_unresolved_focus",
            "benchmark": "not_applicable",
            "exit": "expected_synthetic_max_holding_or_no_exit",
            "limits": "Does not resolve whether holding period is economically useful.",
        },
        "benchmark_lag_stress_path": {
            "shape": "local_placeholder_underperformance_path",
            "stop": "may_or_may_not_interact_with_stop_loss",
            "take": "take_profit_not_primary_focus",
            "holding": "may_remain_open_until_benchmark_lag_placeholder_review",
            "benchmark": "benchmark_lag_placeholder_without_external_benchmark_data",
            "exit": "expected_benchmark_lag_placeholder_review",
            "limits": "No benchmark prices are fetched or modeled.",
        },
    }
    return mapping.get(
        dimension,
        {
            "shape": "local_synthetic_design_placeholder",
            "stop": "not_specified",
            "take": "not_specified",
            "holding": "not_specified",
            "benchmark": "not_applicable",
            "exit": "needs_future_definition",
            "limits": "Scenario needs future local specification.",
        },
    )


def build_scenario_definitions(stress_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in stress_matrix.iterrows():
        dimension = str(row.get("stress_dimension", ""))
        assumptions = _assumptions_for_dimension(dimension)
        rows.append(
            {
                "scenario_id": f"GEN-{index + 1:03d}",
                "scenario_name": dimension,
                "source_step13_dimension": dimension,
                "source_stress_scenario_id": row.get("stress_scenario_id", ""),
                "local_synthetic_only": True,
                "not_real_market_evidence": True,
                "execution_status": "definition_created_not_market_executed",
                "expected_risk_focus": row.get("expected_exit_focus", assumptions["exit"]),
                "price_path_shape": assumptions["shape"],
                "entry_price_assumption": "use_existing_step9_entry_or_reference_price_only",
                "stop_loss_interaction_assumption": assumptions["stop"],
                "take_profit_interaction_assumption": assumptions["take"],
                "max_holding_interaction_assumption": assumptions["holding"],
                "benchmark_interaction_assumption": assumptions["benchmark"],
                "expected_exit_behavior": assumptions["exit"],
                "priority": row.get("priority", "medium"),
                "limitations": assumptions["limits"],
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_price_path_assumptions(definitions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "assumption_id": f"ASSUME-{index + 1:03d}",
                "scenario_id": row["scenario_id"],
                "scenario_name": row["scenario_name"],
                "price_path_shape": row["price_path_shape"],
                "local_price_basis": "existing_step9_reference_price_only",
                "price_generation_status": "assumption_defined_not_generated",
                "path_execution_allowed": False,
                "not_real_market_evidence": True,
                "market_data_source": "none",
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for index, row in definitions.iterrows()
        ]
    )


def build_execution_plan(definitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in definitions.iterrows():
        rows.append(
            {
                "plan_id": f"PLAN-{index + 1:03d}",
                "scenario_id": row["scenario_id"],
                "scenario_name": row["scenario_name"],
                "planned_step": "future_local_synthetic_path_generation",
                "current_status": "definition_created_not_executed",
                "required_future_controls": "schema_check_false_safety_flags_no_market_data_no_broker",
                "priority": row["priority"],
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Step 14 creates definitions and assumptions only."),
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
        ("local_synthetic_scenario_definitions_only", "confirmed", "Outputs define local synthetic scenarios and assumptions only."),
        ("not_real_market_evidence", "confirmed", "Definitions are explicitly not real market validation evidence."),
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
    stress_matrix: pd.DataFrame,
    definitions: pd.DataFrame,
    assumptions: pd.DataFrame,
    execution_plan: pd.DataFrame,
) -> pd.DataFrame:
    missing = int((~manifest["source_file_exists"].astype(bool)).sum()) if not manifest.empty else 0
    forbidden = int(manifest["forbidden_true_flag_count"].sum()) if not manifest.empty else 0
    local_count = int(definitions["local_synthetic_only"].astype(bool).sum()) if not definitions.empty else 0
    not_real_count = int(definitions["not_real_market_evidence"].astype(bool).sum()) if not definitions.empty else 0
    high_priority = int((definitions["priority"] == "high").sum()) if not definitions.empty else 0
    validation_status = "pass" if missing == 0 and forbidden == 0 and len(definitions) == len(stress_matrix) else "warning"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step14_synthetic_stress_scenario_generator",
                "reviewed_input_count": int(manifest["source_file_exists"].astype(bool).sum()) if not manifest.empty else 0,
                "missing_input_count": missing,
                "source_stress_dimension_count": int(len(stress_matrix)),
                "generated_scenario_definition_count": int(len(definitions)),
                "generated_price_path_assumption_count": int(len(assumptions)),
                "execution_plan_row_count": int(len(execution_plan)),
                "high_priority_scenario_count": high_priority,
                "local_synthetic_only_count": local_count,
                "not_real_market_evidence_count": not_real_count,
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
                "conclusion": "synthetic_stress_scenario_generator_completed_research_only",
                "recommended_next_step": "V6 Step 15 Simulation Hardening Closure / Transition to Data & Market Reality Foundation",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    definitions: pd.DataFrame,
    assumptions: pd.DataFrame,
    execution_plan: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 14 Synthetic Stress Matrix Implementation / Local Scenario Generator",
            "",
            "## Executive Summary",
            "Step 14 converts Step 13 stress matrix design rows into local deterministic synthetic scenario definitions and price-path assumptions.",
            "It creates definitions only; it does not execute new price simulations, run real backtests, fetch market or live data, retrain models, change thresholds or features, connect to brokers, execute orders, submit orders, or mark anything trading-ready.",
            "These outputs are not alpha discovery, not market-data replay, not broker execution, and not trading-ready evidence.",
            "",
            "## Summary",
            f"- Reviewed inputs: {row.get('reviewed_input_count', 0)}",
            f"- Missing inputs: {row.get('missing_input_count', 0)}",
            f"- Source stress dimensions: {row.get('source_stress_dimension_count', 0)}",
            f"- Scenario definitions: {row.get('generated_scenario_definition_count', 0)}",
            f"- Price path assumptions: {row.get('generated_price_path_assumption_count', 0)}",
            f"- Execution plan rows: {row.get('execution_plan_row_count', 0)}",
            f"- High-priority scenarios: {row.get('high_priority_scenario_count', 0)}",
            f"- Local synthetic only rows: {row.get('local_synthetic_only_count', 0)}",
            f"- Not-real-market evidence rows: {row.get('not_real_market_evidence_count', 0)}",
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
            "## Scenario Definitions",
            _table(definitions, "No scenario definitions were generated."),
            "",
            "## Price Path Assumptions",
            _table(assumptions, "No price path assumptions were generated."),
            "",
            "## Execution Plan",
            _table(execution_plan, "No execution plan rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This generator creates local synthetic scenario definitions only. It is not real market evidence, not a historical backtest, not live trading, not broker paper trading, and not trading-ready evidence.",
            "",
        ]
    )


def generate_synthetic_stress_scenario_generator_outputs(
    price_path_dir: str | Path = DEFAULT_PRICE_PATH_DIR,
    result_review_dir: str | Path = DEFAULT_RESULT_REVIEW_DIR,
    stress_matrix_dir: str | Path = DEFAULT_STRESS_MATRIX_DIR,
    hardening_review_dir: str | Path = DEFAULT_HARDENING_REVIEW_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        price_path_dir=price_path_dir,
        result_review_dir=result_review_dir,
        stress_matrix_dir=stress_matrix_dir,
        hardening_review_dir=hardening_review_dir,
    )
    manifest = build_input_manifest(paths)
    stress_matrix = _read_csv(paths["stress_matrix"] / "synthetic_replay_stress_matrix.csv")
    definitions = build_scenario_definitions(stress_matrix)
    assumptions = build_price_path_assumptions(definitions)
    execution_plan = build_execution_plan(definitions)
    guardrails = build_guardrails()
    summary = build_summary(manifest, stress_matrix, definitions, assumptions, execution_plan)
    report = build_report(summary, definitions, assumptions, execution_plan, guardrails)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    manifest.to_csv(output_files["manifest"], index=False)
    definitions.to_csv(output_files["definitions"], index=False)
    assumptions.to_csv(output_files["assumptions"], index=False)
    execution_plan.to_csv(output_files["execution_plan"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    summary.to_csv(output_files["summary"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "generated_scenario_definition_count": int(len(definitions)),
        "generated_price_path_assumption_count": int(len(assumptions)),
        "scope": "V6 Step 14 local synthetic stress scenario definitions only",
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
        "synthetic_stress_summary": summary,
        "synthetic_stress_input_manifest": manifest,
        "synthetic_stress_scenario_definitions": definitions,
        "synthetic_stress_price_path_assumptions": assumptions,
        "synthetic_stress_execution_plan": execution_plan,
        "synthetic_stress_guardrails": guardrails,
        "synthetic_stress_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
