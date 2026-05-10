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
DEFAULT_OUTPUT_DIR = Path("outputs/validation_baseline_manifest_real_v1")

OUTPUT_FILENAMES = {
    "summary": "validation_baseline_summary.csv",
    "manifest": "validation_baseline_manifest.csv",
    "guardrails": "validation_baseline_guardrails.csv",
    "report": "validation_baseline_report.md",
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
    (
        "V5 Step 1 Capital Constraint Engine",
        "capital",
        "capital_constraint_summary.csv",
        "capital_constraint_guardrails.csv",
        "capital_constraint_report.md",
    ),
    (
        "V5 Step 2 Tradable Universe Filter",
        "universe",
        "universe_filter_summary.csv",
        "tradable_universe_guardrails.csv",
        "tradable_universe_report.md",
    ),
    (
        "V5 Step 3 Position Sizing Engine",
        "position",
        "position_sizing_summary.csv",
        "position_sizing_guardrails.csv",
        "position_sizing_report.md",
    ),
    (
        "V5 Step 4 Exit Engine",
        "exit",
        "exit_summary.csv",
        "exit_guardrails.csv",
        "exit_engine_report.md",
    ),
    (
        "V5 Step 5 Daily Trading Plan",
        "daily",
        "daily_trading_plan_summary.csv",
        "daily_trading_plan_guardrails.csv",
        "daily_trading_plan.md",
    ),
    (
        "V5 Step 6 Paper Trading Ledger",
        "paper",
        "paper_trading_summary.csv",
        "paper_trading_guardrails.csv",
        "paper_trading_report.md",
    ),
    (
        "V5 Step 7 Semi-Auto Order Generator",
        "semi",
        "semi_auto_order_summary.csv",
        "semi_auto_order_guardrails.csv",
        "broker_neutral_order_tickets.md",
    ),
    (
        "V5 Step 8 Broker Integration Research",
        "broker",
        "broker_integration_summary.csv",
        "broker_integration_guardrails.csv",
        "broker_integration_research_report.md",
    ),
    (
        "V5 Step 9 Monitoring / Reporting Layer",
        "monitoring",
        "monitoring_summary.csv",
        "monitoring_guardrails.csv",
        "monitoring_report.md",
    ),
    (
        "V5 Step 10 Capital-Aware Infrastructure Review / Closure",
        "v5_closure",
        "v5_infrastructure_closure_summary.csv",
        "v5_guardrail_audit.csv",
        "v5_capital_aware_closure_report.md",
    ),
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


def _file_stats(output_dir: Path) -> dict[str, int]:
    if not output_dir.exists() or not output_dir.is_dir():
        return {
            "file_count": 0,
            "csv_file_count": 0,
            "markdown_file_count": 0,
            "json_file_count": 0,
            "total_byte_size": 0,
        }
    files = [item for item in output_dir.iterdir() if item.is_file()]
    return {
        "file_count": len(files),
        "csv_file_count": len([item for item in files if item.suffix.lower() == ".csv"]),
        "markdown_file_count": len([item for item in files if item.suffix.lower() == ".md"]),
        "json_file_count": len([item for item in files if item.suffix.lower() == ".json"]),
        "total_byte_size": sum(item.stat().st_size for item in files),
    }


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
    }


def scan_directory_flags(output_dir: Path) -> dict[str, int]:
    counts = {flag: 0 for flag in SAFETY_FLAGS}
    if not output_dir.exists() or not output_dir.is_dir():
        return counts
    for csv_path in output_dir.glob("*.csv"):
        frame = _read_csv(csv_path)
        for flag in SAFETY_FLAGS:
            counts[flag] += _bool_count(frame, flag)
    for json_path in output_dir.glob("*.json"):
        payload = _read_json(json_path)
        for flag in SAFETY_FLAGS:
            counts[flag] += _json_flag_count(payload, flag)
    return counts


def build_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step_name, path_key, summary_file, guardrail_file, report_file in STEP_DEFINITIONS:
        output_dir = paths[path_key]
        stats = _file_stats(output_dir)
        flags = scan_directory_flags(output_dir)
        rows.append(
            {
                "step_name": step_name,
                "output_directory_path": str(output_dir),
                "directory_exists": output_dir.exists() and output_dir.is_dir(),
                "file_count": stats["file_count"],
                "csv_file_count": stats["csv_file_count"],
                "markdown_file_count": stats["markdown_file_count"],
                "json_file_count": stats["json_file_count"],
                "total_byte_size": stats["total_byte_size"],
                "important_summary_file": summary_file if (output_dir / summary_file).exists() else "",
                "important_guardrail_file": guardrail_file if (output_dir / guardrail_file).exists() else "",
                "important_report_file": report_file if (output_dir / report_file).exists() else "",
                "trading_ready_true_count": flags["trading_ready"],
                "execution_allowed_true_count": flags["execution_allowed"],
                "broker_connected_true_count": flags["broker_connected"],
                "live_trading_true_count": flags["live_trading"],
                "real_order_submission_true_count": flags["real_order_submission"],
            }
        )
    return pd.DataFrame(rows)


def build_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_validation_baseline_manifest",
                "baseline_step_count": int(len(manifest)),
                "present_output_dir_count": int(manifest["directory_exists"].sum()) if not manifest.empty else 0,
                "missing_output_dir_count": int((~manifest["directory_exists"]).sum()) if not manifest.empty else 0,
                "total_file_count": int(manifest["file_count"].sum()) if not manifest.empty else 0,
                "total_csv_file_count": int(manifest["csv_file_count"].sum()) if not manifest.empty else 0,
                "total_markdown_file_count": int(manifest["markdown_file_count"].sum()) if not manifest.empty else 0,
                "total_json_file_count": int(manifest["json_file_count"].sum()) if not manifest.empty else 0,
                "trading_ready_true_count": int(manifest["trading_ready_true_count"].sum()) if not manifest.empty else 0,
                "execution_allowed_true_count": int(manifest["execution_allowed_true_count"].sum()) if not manifest.empty else 0,
                "broker_connected_true_count": int(manifest["broker_connected_true_count"].sum()) if not manifest.empty else 0,
                "live_trading_true_count": int(manifest["live_trading_true_count"].sum()) if not manifest.empty else 0,
                "real_order_submission_true_count": int(manifest["real_order_submission_true_count"].sum()) if not manifest.empty else 0,
                "baseline_status": "v6_validation_baseline_manifest_created_research_only",
                "trading_ready": False,
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The manifest reads existing V5 output directories only.", "No historical backtest is run."),
        ("no_market_data_fetch", "confirmed", "The manifest has no market-data source arguments and no data loader calls.", "No market data is fetched."),
        ("no_threshold_change", "confirmed", "No threshold module or value is changed.", "Thresholds are not modified."),
        ("no_model_retraining", "confirmed", "No training module is imported or called.", "Model artifacts are unchanged."),
        ("no_feature_engineering_change", "confirmed", "No factor builder or feature engineering module is imported or called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local V5 outputs are inspected.", "No new data source is added."),
        ("no_broker_credentials", "confirmed", "The CLI does not accept credentials and the module does not request credentials.", "No account login or credential storage exists."),
        ("no_broker_sdk_import", "confirmed", "The module imports only standard library modules and pandas.", "No broker SDK is imported."),
        ("no_broker_connection", "confirmed", "The manifest records local files only.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "The manifest has no live trading path.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "The manifest has no execution path.", "No orders are executed."),
        ("no_real_order_submission", "confirmed", "The manifest has no order submission path.", "No real orders are submitted."),
        ("no_trading_ready_upgrade", "confirmed", "The summary writes trading_ready as false.", "No deployable status is claimed."),
        ("manifest_only", "confirmed", "The outputs are a summary, manifest, guardrails, report, and run config.", "No new trading capability is added."),
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


def build_report(summary: pd.DataFrame, manifest: pd.DataFrame, guardrails: pd.DataFrame) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 1 Validation Baseline Manifest",
            "",
            "## Executive Summary",
            "V6 Step 1 records the current V5 output state as a stable research baseline for future V6 simulation, replay, and hardening work.",
            "It reads existing local V5 output directories only and does not create any trading capability.",
            "It does not run backtests, fetch market data, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or upgrade trading readiness.",
            "",
            "## Baseline Summary",
            f"- Baseline steps: {row.get('baseline_step_count', 0)}",
            f"- Present output directories: {row.get('present_output_dir_count', 0)}",
            f"- Missing output directories: {row.get('missing_output_dir_count', 0)}",
            f"- Total files: {row.get('total_file_count', 0)}",
            f"- CSV files: {row.get('total_csv_file_count', 0)}",
            f"- Markdown files: {row.get('total_markdown_file_count', 0)}",
            f"- JSON files: {row.get('total_json_file_count', 0)}",
            f"- trading_ready true count: {row.get('trading_ready_true_count', 0)}",
            f"- execution_allowed true count: {row.get('execution_allowed_true_count', 0)}",
            f"- broker_connected true count: {row.get('broker_connected_true_count', 0)}",
            f"- live_trading true count: {row.get('live_trading_true_count', 0)}",
            f"- real_order_submission true count: {row.get('real_order_submission_true_count', 0)}",
            f"- Baseline status: {row.get('baseline_status', '')}",
            "",
            "## Manifest",
            _table(manifest, "No manifest rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This manifest is educational/research-only. It is not financial advice and is not a trading-ready certification.",
            "",
        ]
    )


def generate_validation_baseline_manifest_outputs(
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
    )
    output_path = Path(output_dir)
    manifest = build_manifest(paths)
    summary = build_summary(manifest)
    guardrails = build_guardrails()
    report = build_report(summary, manifest, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    out_paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(out_paths["summary"], index=False)
    manifest.to_csv(out_paths["manifest"], index=False)
    guardrails.to_csv(out_paths["guardrails"], index=False)
    out_paths["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "baseline_step_count": int(len(manifest)),
        "present_output_dir_count": int(manifest["directory_exists"].sum()) if not manifest.empty else 0,
        "missing_output_dir_count": int((~manifest["directory_exists"]).sum()) if not manifest.empty else 0,
        "scope": "V6 Step 1 validation baseline manifest only",
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
        "validation_baseline_summary": summary,
        "validation_baseline_manifest": manifest,
        "validation_baseline_guardrails": guardrails,
        "validation_baseline_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in out_paths.items()},
    }
