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
DEFAULT_SCHEMA_VALIDATOR_DIR = Path("outputs/output_schema_validator_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/cross_step_dependency_validator_real_v1")

OUTPUT_FILENAMES = {
    "results": "cross_step_dependency_results.csv",
    "summary": "cross_step_dependency_summary.csv",
    "guardrails": "cross_step_dependency_guardrails.csv",
    "report": "cross_step_dependency_report.md",
    "run_config": "run_config.json",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"symbol": str})
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


def _norm(value: str | Path | None) -> str:
    if value is None:
        return ""
    return Path(str(value)).as_posix().strip().lower()


def _same_path(left: str | Path | None, right: str | Path | None) -> bool:
    return _norm(left) == _norm(right)


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
    schema_validator_dir: str | Path = DEFAULT_SCHEMA_VALIDATOR_DIR,
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
        "schema_validator": Path(schema_validator_dir),
    }


def _dependency_row(
    dependency_id: str,
    source_step: str,
    target_step: str,
    dependency_type: str,
    expected_reference: str | Path,
    actual_reference: str | Path | None,
    required: bool,
    evidence: str,
) -> dict[str, Any]:
    expected_path = Path(expected_reference)
    actual_matches = _same_path(actual_reference, expected_reference)
    expected_exists = expected_path.exists()
    if actual_matches and expected_exists:
        status = "pass"
        notes = "dependency_reference_matches_and_target_exists"
    elif required:
        status = "fail"
        notes = "required_dependency_broken"
    else:
        status = "warning"
        notes = "optional_dependency_missing_or_mismatched"
    return {
        "dependency_id": dependency_id,
        "source_step": source_step,
        "target_step": target_step,
        "dependency_type": dependency_type,
        "required": required,
        "expected_reference": str(expected_reference),
        "actual_reference": "" if actual_reference is None else str(actual_reference),
        "expected_exists": expected_exists,
        "actual_matches_expected": actual_matches,
        "validation_status": status,
        "evidence": evidence,
        "notes": notes,
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
    }


def _config(paths: dict[str, Path], key: str) -> dict[str, Any]:
    return _read_json(paths[key] / "run_config.json")


def build_dependency_results(paths: dict[str, Path]) -> pd.DataFrame:
    configs = {key: _config(paths, key) for key in paths}
    rows: list[dict[str, Any]] = []

    rows.append(_dependency_row("DEP-001", "V5 Step 3 Position Sizing", "V5 Step 2 Tradable Universe", "input_file", paths["universe"] / "tradable_universe.csv", configs["position"].get("input_path"), True, "Step 3 run_config.input_path should point to Step 2 tradable_universe.csv."))
    rows.append(_dependency_row("DEP-002", "V5 Step 4 Exit Engine", "V5 Step 3 Position Sizing", "input_file", paths["position"] / "sized_positions.csv", configs["exit"].get("input_path"), True, "Step 4 run_config.input_path should point to Step 3 sized_positions.csv."))
    rows.append(_dependency_row("DEP-003", "V5 Step 5 Daily Trading Plan", "V5 Step 2 Tradable Universe", "input_file", paths["universe"] / "tradable_universe.csv", configs["daily"].get("tradable_path"), True, "Step 5 should include Step 2 tradable universe output."))
    rows.append(_dependency_row("DEP-004", "V5 Step 5 Daily Trading Plan", "V5 Step 3 Position Sizing", "input_file", paths["position"] / "sized_positions.csv", configs["daily"].get("sized_path"), True, "Step 5 should include Step 3 sized positions."))
    rows.append(_dependency_row("DEP-005", "V5 Step 5 Daily Trading Plan", "V5 Step 3 Position Sizing", "input_file", paths["position"] / "deferred_positions.csv", configs["daily"].get("deferred_path"), True, "Step 5 should include Step 3 deferred positions."))
    rows.append(_dependency_row("DEP-006", "V5 Step 5 Daily Trading Plan", "V5 Step 4 Exit Engine", "input_file", paths["exit"] / "exit_plan.csv", configs["daily"].get("exit_plan_path"), True, "Step 5 should include Step 4 exit plan."))
    rows.append(_dependency_row("DEP-007", "V5 Step 6 Paper Trading Ledger", "V5 Step 5 Daily Trading Plan", "input_dir", paths["daily"], configs["paper"].get("input_dir"), True, "Step 6 should read the Step 5 daily trading plan directory."))
    rows.append(_dependency_row("DEP-008", "V5 Step 7 Semi-Auto Order Generator", "V5 Step 5 Daily Trading Plan", "input_file", paths["daily"] / "daily_trading_plan.csv", configs["semi"].get("daily_plan_path"), True, "Step 7 should read the Step 5 daily plan CSV."))
    rows.append(_dependency_row("DEP-009", "V5 Step 7 Semi-Auto Order Generator", "V5 Step 4 Exit Engine", "input_file", paths["exit"] / "exit_plan.csv", configs["semi"].get("exit_plan_path"), True, "Step 7 should read the Step 4 exit plan CSV."))
    rows.append(_dependency_row("DEP-010", "V5 Step 8 Broker Integration Research", "V5 Step 7 Semi-Auto Order Generator", "input_dir", paths["semi"], configs["broker"].get("input_dir"), True, "Step 8 should read the Step 7 semi-auto output directory."))
    rows.append(_dependency_row("DEP-011", "V5 Step 8 Broker Integration Research", "V5 Step 7 Semi-Auto Order Generator", "input_file", paths["semi"] / "order_drafts.csv", configs["broker"].get("order_drafts_path"), True, "Step 8 should read Step 7 order_drafts.csv."))

    for idx, key in enumerate(["capital", "universe", "position", "exit", "daily", "paper", "semi", "broker"], start=12):
        rows.append(_dependency_row(f"DEP-{idx:03d}", "V5 Step 9 Monitoring / Reporting Layer", f"V5 upstream {key}", "input_dir", paths[key], configs["monitoring"].get(f"{key}_dir"), True, "Step 9 should monitor V5 Step 1 through Step 8 outputs."))

    for idx, key in enumerate(["capital", "universe", "position", "exit", "daily", "paper", "semi", "broker", "monitoring"], start=20):
        rows.append(_dependency_row(f"DEP-{idx:03d}", "V5 Step 10 Capital-Aware Infrastructure Review / Closure", f"V5 upstream {key}", "input_dir", paths[key], configs["v5_closure"].get(f"{key}_dir"), True, "Step 10 should review V5 Step 1 through Step 9 outputs."))

    baseline_key_map = {
        "capital": "capital_dir",
        "universe": "universe_dir",
        "position": "position_dir",
        "exit": "exit_dir",
        "daily": "daily_dir",
        "paper": "paper_dir",
        "semi": "semi_dir",
        "broker": "broker_dir",
        "monitoring": "monitoring_dir",
        "v5_closure": "v5_closure_dir",
    }
    for idx, (key, config_key) in enumerate(baseline_key_map.items(), start=29):
        rows.append(_dependency_row(f"DEP-{idx:03d}", "V6 Step 1 Validation Baseline Manifest", f"baseline reference {key}", "input_dir", paths[key], configs["baseline"].get(config_key), True, "V6 Step 1 should preserve baseline references to V5 output directories."))

    schema_key_map = {
        "capital": "capital_dir",
        "universe": "universe_dir",
        "position": "position_dir",
        "exit": "exit_dir",
        "daily": "daily_dir",
        "paper": "paper_dir",
        "semi": "semi_dir",
        "broker": "broker_dir",
        "monitoring": "monitoring_dir",
        "v5_closure": "v5_closure_dir",
        "baseline": "baseline_dir",
    }
    for idx, (key, config_key) in enumerate(schema_key_map.items(), start=39):
        rows.append(_dependency_row(f"DEP-{idx:03d}", "V6 Step 2 Output Schema Validator", f"schema validation reference {key}", "input_dir", paths[key], configs["schema_validator"].get(config_key), True, "V6 Step 2 should validate expected output schema files from prior V5/V6 outputs."))

    schema_summary = _read_csv(paths["schema_validator"] / "output_schema_validation_summary.csv")
    schema_status = schema_summary["validation_status"].iloc[0] if not schema_summary.empty and "validation_status" in schema_summary else None
    rows.append(
        {
            "dependency_id": "DEP-050",
            "source_step": "V6 Step 2 Output Schema Validator",
            "target_step": "V5/V6 expected schemas",
            "dependency_type": "schema_validation_status",
            "required": True,
            "expected_reference": "pass_or_warning",
            "actual_reference": "" if schema_status is None else str(schema_status),
            "expected_exists": not schema_summary.empty,
            "actual_matches_expected": schema_status in {"pass", "warning"},
            "validation_status": "pass" if schema_status in {"pass", "warning"} else "fail",
            "evidence": "V6 Step 2 summary should confirm schema validation completed without fail status.",
            "notes": "schema_validation_status_checked",
            "broker_connected": False,
            "execution_allowed": False,
            "live_trading": False,
            "real_order_submission": False,
            "trading_ready": False,
        }
    )
    return pd.DataFrame(rows)


def scan_forbidden_flags(paths: dict[str, Path]) -> int:
    total = 0
    for output_dir in paths.values():
        if not output_dir.exists() or not output_dir.is_dir():
            continue
        for csv_path in output_dir.glob("*.csv"):
            frame = _read_csv(csv_path)
            total += sum(_bool_count(frame, flag) for flag in SAFETY_FLAGS)
        for json_path in output_dir.glob("*.json"):
            payload = _read_json(json_path)
            total += sum(_json_flag_count(payload, flag) for flag in SAFETY_FLAGS)
    return int(total)


def build_summary(paths: dict[str, Path], results: pd.DataFrame, forbidden_true_flag_count: int) -> pd.DataFrame:
    fail_count = int((results["validation_status"] == "fail").sum()) if not results.empty else 0
    warning_count = int((results["validation_status"] == "warning").sum()) if not results.empty else 0
    pass_count = int((results["validation_status"] == "pass").sum()) if not results.empty else 0
    if fail_count:
        status = "fail"
        conclusion = "cross_step_dependency_validation_failed_research_only"
    elif warning_count:
        status = "warning"
        conclusion = "cross_step_dependency_validation_passed_with_warnings_research_only"
    else:
        status = "pass"
        conclusion = "cross_step_dependency_validation_passed_research_only"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step3_cross_step_dependency_validation",
                "checked_dependency_count": int(len(results)),
                "dependency_pass_count": pass_count,
                "dependency_warning_count": warning_count,
                "dependency_fail_count": fail_count,
                "checked_output_dir_count": int(len(paths)),
                "missing_output_dir_count": int(sum(1 for path in paths.values() if not path.exists())),
                "forbidden_true_flag_count": forbidden_true_flag_count,
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
        ("no_broker_connection", "confirmed", "The validator reads files only.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "The validator has no live trading path.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "The validator has no execution path.", "No orders are executed."),
        ("no_real_order_submission", "confirmed", "The validator has no order submission path.", "No real orders are submitted."),
        ("no_trading_ready_upgrade", "confirmed", "The summary writes trading_ready as false.", "No deployable status is claimed."),
        ("dependency_validation_only", "confirmed", "The outputs are dependency validation CSV/Markdown reports only.", "No trading capability is added."),
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
            "# V6 Step 3 Cross-Step Dependency Integrity Validator",
            "",
            "## Executive Summary",
            "V6 Step 3 validates dependency links across existing local V5/V6 research outputs.",
            "It checks that downstream steps reference the expected upstream directories or files and that V6 schema validation completed without failure.",
            "It does not run backtests, fetch market data, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or upgrade trading readiness.",
            "",
            "## Summary",
            f"- Checked dependencies: {row.get('checked_dependency_count', 0)}",
            f"- Dependency passes: {row.get('dependency_pass_count', 0)}",
            f"- Dependency warnings: {row.get('dependency_warning_count', 0)}",
            f"- Dependency failures: {row.get('dependency_fail_count', 0)}",
            f"- Checked output directories: {row.get('checked_output_dir_count', 0)}",
            f"- Missing output directories: {row.get('missing_output_dir_count', 0)}",
            f"- Forbidden true flags: {row.get('forbidden_true_flag_count', 0)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            "",
            "## Dependency Results",
            _table(results, "No dependency rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This dependency validation report is educational/research-only. It is not financial advice and is not a trading-ready certification.",
            "",
        ]
    )


def generate_cross_step_dependency_validation_outputs(
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
    schema_validator_dir: str | Path = DEFAULT_SCHEMA_VALIDATOR_DIR,
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
        schema_validator_dir=schema_validator_dir,
    )
    output_path = Path(output_dir)
    results = build_dependency_results(paths)
    forbidden_true_flag_count = scan_forbidden_flags(paths)
    summary = build_summary(paths, results, forbidden_true_flag_count)
    guardrails = build_guardrails()
    report = build_report(summary, results, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    out_paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    results.to_csv(out_paths["results"], index=False)
    summary.to_csv(out_paths["summary"], index=False)
    guardrails.to_csv(out_paths["guardrails"], index=False)
    out_paths["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "checked_dependency_count": int(len(results)),
        "checked_output_dir_count": int(len(paths)),
        "scope": "V6 Step 3 cross-step dependency validation only",
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
        "cross_step_dependency_results": results,
        "cross_step_dependency_summary": summary,
        "cross_step_dependency_guardrails": guardrails,
        "cross_step_dependency_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in out_paths.items()},
    }
