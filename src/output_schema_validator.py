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
DEFAULT_V5_CLOSURE_DIR = Path("outputs/capital_aware_infrastructure_review_real_v1")
DEFAULT_BASELINE_DIR = Path("outputs/validation_baseline_manifest_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/output_schema_validator_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "summary": "output_schema_validation_summary.csv",
    "results": "output_schema_validation_results.csv",
    "guardrails": "output_schema_validation_guardrails.csv",
    "report": "output_schema_validation_report.md",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
]

EXPECTED_SCHEMAS = [
    ("V5 Step 1 Capital Constraint Engine", "capital", "capital_constraint_summary.csv", ["summary_item", "candidate_count", "approved_order_count", "rejected_order_count", "trading_ready", "conclusion"]),
    ("V5 Step 1 Capital Constraint Engine", "capital", "capital_feasibility.csv", ["symbol", "side", "price", "order_allowed", "trading_ready"]),
    ("V5 Step 1 Capital Constraint Engine", "capital", "capital_constraint_guardrails.csv", ["guardrail", "status"]),
    ("V5 Step 2 Tradable Universe Filter", "universe", "universe_filter_summary.csv", ["summary_item", "candidate_count", "tradable_count", "excluded_count", "trading_ready", "conclusion"]),
    ("V5 Step 2 Tradable Universe Filter", "universe", "tradable_universe.csv", ["symbol", "side", "price", "tradable", "trading_ready"]),
    ("V5 Step 2 Tradable Universe Filter", "universe", "universe_filter_guardrails.csv", ["guardrail", "status"]),
    ("V5 Step 3 Position Sizing Engine", "position", "position_sizing_summary.csv", ["summary_item", "candidate_count", "sized_position_count", "deferred_position_count", "trading_ready", "conclusion"]),
    ("V5 Step 3 Position Sizing Engine", "position", "sized_positions.csv", ["symbol", "side", "price", "quantity", "approved_notional", "trading_ready"]),
    ("V5 Step 3 Position Sizing Engine", "position", "position_sizing_guardrails.csv", ["guardrail", "status"]),
    ("V5 Step 4 Exit Engine", "exit", "exit_summary.csv", ["summary_item", "sized_position_count", "planned_exit_count", "invalid_exit_plan_count", "trading_ready", "conclusion"]),
    ("V5 Step 4 Exit Engine", "exit", "exit_plan.csv", ["symbol", "entry_price", "quantity", "stop_loss_price", "take_profit_price", "trading_ready"]),
    ("V5 Step 4 Exit Engine", "exit", "exit_guardrails.csv", ["guardrail", "status"]),
    ("V5 Step 5 Daily Trading Plan", "daily", "daily_trading_plan_summary.csv", ["summary_item", "daily_plan_row_count", "trading_ready", "conclusion"]),
    ("V5 Step 5 Daily Trading Plan", "daily", "daily_trading_plan.csv", ["plan_section", "symbol", "trading_ready"]),
    ("V5 Step 5 Daily Trading Plan", "daily", "daily_trading_plan_guardrails.csv", ["guardrail", "status"]),
    ("V5 Step 6 Paper Trading Ledger", "paper", "paper_trading_summary.csv", ["summary_item", "paper_order_count", "paper_filled_order_count", "trading_ready", "conclusion"]),
    ("V5 Step 6 Paper Trading Ledger", "paper", "paper_orders.csv", ["paper_order_id", "symbol", "side", "order_quantity", "trading_ready"]),
    ("V5 Step 6 Paper Trading Ledger", "paper", "paper_trading_guardrails.csv", ["guardrail", "status"]),
    ("V5 Step 7 Semi-Auto Order Generator", "semi", "semi_auto_order_summary.csv", ["summary_item", "draft_order_count", "execution_allowed_count", "broker_connected_count", "trading_ready_count", "conclusion", "trading_ready"]),
    ("V5 Step 7 Semi-Auto Order Generator", "semi", "order_drafts.csv", ["draft_order_id", "symbol", "side", "quantity", "execution_allowed", "broker_connected", "trading_ready"]),
    ("V5 Step 7 Semi-Auto Order Generator", "semi", "semi_auto_order_guardrails.csv", ["guardrail", "status"]),
    ("V5 Step 8 Broker Integration Research", "broker", "broker_integration_summary.csv", ["summary_item", "broker_connected_count", "execution_allowed_count", "live_trading_count", "real_order_submission_count", "trading_ready"]),
    ("V5 Step 8 Broker Integration Research", "broker", "broker_integration_modes.csv", ["integration_mode", "implementation_status", "broker_connected", "execution_allowed", "live_trading", "real_order_submission", "trading_ready"]),
    ("V5 Step 8 Broker Integration Research", "broker", "broker_integration_guardrails.csv", ["guardrail", "status"]),
    ("V5 Step 9 Monitoring / Reporting Layer", "monitoring", "monitoring_summary.csv", ["summary_item", "monitored_step_count", "trading_ready_true_count", "execution_allowed_true_count", "broker_connected_true_count", "live_trading_true_count", "real_order_submission_true_count", "trading_ready"]),
    ("V5 Step 9 Monitoring / Reporting Layer", "monitoring", "monitoring_status_dashboard.csv", ["step", "output_dir", "status", "metric", "trading_ready"]),
    ("V5 Step 9 Monitoring / Reporting Layer", "monitoring", "monitoring_guardrails.csv", ["guardrail", "status"]),
    ("V5 Step 10 Capital-Aware Infrastructure Review / Closure", "v5_closure", "v5_infrastructure_closure_summary.csv", ["summary_item", "reviewed_step_count", "completed_step_count", "missing_step_count", "trading_ready_true_count", "execution_allowed_true_count", "broker_connected_true_count", "live_trading_true_count", "real_order_submission_true_count", "trading_ready"]),
    ("V5 Step 10 Capital-Aware Infrastructure Review / Closure", "v5_closure", "v5_step_capability_matrix.csv", ["step", "output_dir", "status", "capability_added", "execution_capability", "trading_ready"]),
    ("V5 Step 10 Capital-Aware Infrastructure Review / Closure", "v5_closure", "v5_guardrail_audit.csv", ["guardrail", "status", "trading_ready"]),
    ("V6 Step 1 Validation Baseline Manifest", "baseline", "validation_baseline_summary.csv", ["summary_item", "baseline_step_count", "present_output_dir_count", "missing_output_dir_count", "trading_ready_true_count", "execution_allowed_true_count", "broker_connected_true_count", "live_trading_true_count", "real_order_submission_true_count", "trading_ready"]),
    ("V6 Step 1 Validation Baseline Manifest", "baseline", "validation_baseline_manifest.csv", ["step_name", "output_directory_path", "directory_exists", "file_count", "trading_ready_true_count", "execution_allowed_true_count", "broker_connected_true_count", "live_trading_true_count", "real_order_submission_true_count"]),
    ("V6 Step 1 Validation Baseline Manifest", "baseline", "validation_baseline_guardrails.csv", ["guardrail", "status", "trading_ready"]),
]


def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), "missing_file"
    try:
        return pd.read_csv(path, dtype={"symbol": str}), ""
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "empty_csv_no_header"
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        return pd.DataFrame(), f"read_error:{type(exc).__name__}"


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
    v5_closure_dir: str | Path = DEFAULT_V5_CLOSURE_DIR,
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
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
        "v5_closure": Path(v5_closure_dir),
        "baseline": Path(baseline_dir),
    }


def count_directory_flags(output_dir: Path) -> dict[str, int]:
    counts = {flag: 0 for flag in SAFETY_FLAGS}
    if not output_dir.exists() or not output_dir.is_dir():
        return counts
    for csv_path in output_dir.glob("*.csv"):
        frame, _ = _read_csv(csv_path)
        for flag in SAFETY_FLAGS:
            counts[flag] += _bool_count(frame, flag)
    for json_path in output_dir.glob("*.json"):
        payload = _read_json(json_path)
        for flag in SAFETY_FLAGS:
            counts[flag] += _json_flag_count(payload, flag)
    return counts


def validate_expected_file(step_name: str, path_key: str, filename: str, required_columns: list[str], paths: dict[str, Path]) -> dict[str, Any]:
    output_dir = paths[path_key]
    file_path = output_dir / filename
    directory_exists = output_dir.exists() and output_dir.is_dir()
    file_exists = file_path.exists()
    df, read_error = _read_csv(file_path)
    columns = list(df.columns)
    missing_columns = [column for column in required_columns if column not in columns]
    malformed_row_count = int(df.isna().all(axis=1).sum()) if not df.empty else 0
    flag_counts = {flag: _bool_count(df, flag) for flag in SAFETY_FLAGS}

    if not directory_exists or not file_exists or read_error.startswith("read_error") or missing_columns:
        validation_status = "fail"
    elif read_error or df.empty or malformed_row_count:
        validation_status = "warning"
    else:
        validation_status = "pass"

    notes = []
    if not directory_exists:
        notes.append("output_directory_missing")
    if not file_exists:
        notes.append("expected_file_missing")
    if read_error and read_error != "missing_file":
        notes.append(read_error)
    if missing_columns:
        notes.append("missing_columns:" + "|".join(missing_columns))
    if df.empty and file_exists and not read_error.startswith("read_error"):
        notes.append("empty_csv")
    if malformed_row_count:
        notes.append(f"all_null_rows:{malformed_row_count}")
    if not notes:
        notes.append("schema_validated")

    return {
        "step_name": step_name,
        "output_directory": str(output_dir),
        "file_name": filename,
        "file_path": str(file_path),
        "directory_exists": directory_exists,
        "file_exists": file_exists,
        "row_count": int(len(df)),
        "column_count": int(len(columns)),
        "required_column_count": int(len(required_columns)),
        "missing_column_count": int(len(missing_columns)),
        "missing_columns": "|".join(missing_columns),
        "malformed_row_count": malformed_row_count,
        "trading_ready_true_count": flag_counts["trading_ready"],
        "execution_allowed_true_count": flag_counts["execution_allowed"],
        "broker_connected_true_count": flag_counts["broker_connected"],
        "live_trading_true_count": flag_counts["live_trading"],
        "real_order_submission_true_count": flag_counts["real_order_submission"],
        "validation_status": validation_status,
        "notes": "; ".join(notes),
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
    }


def build_validation_results(paths: dict[str, Path]) -> pd.DataFrame:
    rows = [
        validate_expected_file(step_name, path_key, filename, columns, paths)
        for step_name, path_key, filename, columns in EXPECTED_SCHEMAS
    ]
    return pd.DataFrame(rows)


def build_summary(paths: dict[str, Path], results: pd.DataFrame) -> pd.DataFrame:
    directory_count = int(len(paths))
    checked_file_count = int(len(results))
    present_file_count = int(results["file_exists"].sum()) if not results.empty else 0
    missing_file_count = int((~results["file_exists"]).sum()) if not results.empty else 0
    pass_count = int((results["validation_status"] == "pass").sum()) if not results.empty else 0
    warning_count = int((results["validation_status"] == "warning").sum()) if not results.empty else 0
    fail_count = int((results["validation_status"] == "fail").sum()) if not results.empty else 0
    forbidden_true_count = int(
        results[
            [
                "trading_ready_true_count",
                "execution_allowed_true_count",
                "broker_connected_true_count",
                "live_trading_true_count",
                "real_order_submission_true_count",
            ]
        ].sum().sum()
    ) if not results.empty else 0
    if fail_count:
        status = "fail"
        conclusion = "schema_validation_failed_research_only"
    elif warning_count:
        status = "warning"
        conclusion = "schema_validation_passed_with_warnings_research_only"
    else:
        status = "pass"
        conclusion = "schema_validation_passed_research_only"

    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step2_output_schema_validation",
                "checked_directory_count": directory_count,
                "checked_file_count": checked_file_count,
                "present_file_count": present_file_count,
                "missing_file_count": missing_file_count,
                "schema_pass_count": pass_count,
                "schema_warning_count": warning_count,
                "schema_fail_count": fail_count,
                "forbidden_true_flag_count": forbidden_true_count,
                "trading_ready": False,
                "validation_status": status,
                "conclusion": conclusion,
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The validator reads existing local output files only.", "No historical backtest is run."),
        ("no_market_data_fetch", "confirmed", "The validator has no market-data source arguments and no data loader calls.", "No market data is fetched."),
        ("no_threshold_change", "confirmed", "No threshold module or value is changed.", "Thresholds are not modified."),
        ("no_model_retraining", "confirmed", "No training module is imported or called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is imported or called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local V5/V6 output directories are inspected.", "No new data source is added."),
        ("no_broker_credentials", "confirmed", "The CLI does not accept credentials and the module does not request credentials.", "No account login or credential storage exists."),
        ("no_broker_sdk_import", "confirmed", "The module imports only standard library modules and pandas.", "No broker SDK is imported."),
        ("no_broker_connection", "confirmed", "The validator reads files only.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "The validator has no live trading path.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "The validator has no execution path.", "No orders are executed."),
        ("no_real_order_submission", "confirmed", "The validator has no order submission path.", "No real orders are submitted."),
        ("no_trading_ready_upgrade", "confirmed", "The summary writes trading_ready as false.", "No deployable status is claimed."),
        ("schema_validation_only", "confirmed", "The outputs are schema validation CSV/Markdown reports only.", "No trading capability is added."),
        ("educational_research_only", "confirmed", "The report states this is educational/research-only.", "Not financial advice."),
    ]
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


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(summary: pd.DataFrame, results: pd.DataFrame, guardrails: pd.DataFrame) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 2 Output Consistency / Schema Validation Layer",
            "",
            "## Executive Summary",
            "V6 Step 2 validates required files and required columns across existing local V5/V6 research outputs.",
            "It detects missing files, missing columns, empty CSVs, malformed all-null rows, and forbidden safety flag violations.",
            "It does not run backtests, fetch market data, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or upgrade trading readiness.",
            "",
            "## Summary",
            f"- Checked directories: {row.get('checked_directory_count', 0)}",
            f"- Checked files: {row.get('checked_file_count', 0)}",
            f"- Present files: {row.get('present_file_count', 0)}",
            f"- Missing files: {row.get('missing_file_count', 0)}",
            f"- Schema passes: {row.get('schema_pass_count', 0)}",
            f"- Schema warnings: {row.get('schema_warning_count', 0)}",
            f"- Schema failures: {row.get('schema_fail_count', 0)}",
            f"- Forbidden true flags: {row.get('forbidden_true_flag_count', 0)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            "",
            "## Validation Results",
            _table(results, "No validation rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This schema validation report is educational/research-only. It is not financial advice and is not a trading-ready certification.",
            "",
        ]
    )


def generate_output_schema_validation_outputs(
    capital_dir: str | Path = DEFAULT_CAPITAL_DIR,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
    position_dir: str | Path = DEFAULT_POSITION_DIR,
    exit_dir: str | Path = DEFAULT_EXIT_DIR,
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    semi_auto_dir: str | Path = DEFAULT_SEMI_AUTO_DIR,
    broker_research_dir: str | Path = DEFAULT_BROKER_RESEARCH_DIR,
    monitoring_dir: str | Path = DEFAULT_MONITORING_DIR,
    v5_closure_dir: str | Path = DEFAULT_V5_CLOSURE_DIR,
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
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
        v5_closure_dir=v5_closure_dir,
        baseline_dir=baseline_dir,
    )
    output_path = Path(output_dir)
    results = build_validation_results(paths)
    summary = build_summary(paths, results)
    guardrails = build_guardrails()
    report = build_report(summary, results, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    out_paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(out_paths["summary"], index=False)
    results.to_csv(out_paths["results"], index=False)
    guardrails.to_csv(out_paths["guardrails"], index=False)
    out_paths["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "checked_directory_count": int(len(paths)),
        "checked_file_count": int(len(results)),
        "scope": "V6 Step 2 output consistency and schema validation only",
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
        "output_schema_validation_summary": summary,
        "output_schema_validation_results": results,
        "output_schema_validation_guardrails": guardrails,
        "output_schema_validation_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in out_paths.items()},
    }
