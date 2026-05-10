import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("outputs/reproducibility_rerun_validator_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/reproducibility_warning_triage_real_v1")

OUTPUT_FILENAMES = {
    "summary": "reproducibility_warning_triage_summary.csv",
    "results": "reproducibility_warning_triage_results.csv",
    "guardrails": "reproducibility_warning_triage_guardrails.csv",
    "report": "reproducibility_warning_triage_report.md",
    "run_config": "run_config.json",
}

SAFETY_FLAGS = [
    "broker_connected",
    "execution_allowed",
    "live_trading",
    "real_order_submission",
    "trading_ready",
]


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


def _int_value(value: Any) -> int:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(number) if pd.notna(number) else 0


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def load_inputs(input_dir: str | Path = DEFAULT_INPUT_DIR) -> dict[str, pd.DataFrame]:
    base = Path(input_dir)
    return {
        "summary": _read_csv(base / "reproducibility_rerun_summary.csv"),
        "results": _read_csv(base / "reproducibility_rerun_results.csv"),
        "guardrails": _read_csv(base / "reproducibility_rerun_guardrails.csv"),
    }


def row_is_acceptable_warning(row: pd.Series) -> bool:
    if not bool(row.get("canonical_exists", False)):
        return False
    if not bool(row.get("rerun_exists", False)):
        return False
    if not bool(row.get("row_count_matches", False)):
        return False
    if not bool(row.get("summary_keys_match", False)):
        return False
    if _int_value(row.get("forbidden_true_flag_count")) != 0:
        return False
    for flag in SAFETY_FLAGS:
        if _is_true(row.get(flag)):
            return False
    notes = _text(row.get("notes")).lower()
    return (
        "normalized_fingerprint_differs_but_structure_matches" in notes
        or "fingerprint" in notes and "structure_matches" in notes
    )


def classify_warning(row: pd.Series) -> str:
    if not row_is_acceptable_warning(row):
        return "needs_investigation"
    file_name = _text(row.get("file_name")).lower()
    if file_name == "run_config.json":
        return "acceptable_run_config_path_difference"
    if file_name.endswith(".md") or "report" in file_name:
        return "acceptable_report_embedded_path_difference"
    return "acceptable_path_or_workspace_difference"


def build_triage_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "rerun_id",
                "step_name",
                "file_name",
                "validation_status",
                "triage_classification",
                "triage_status",
                "reason",
                "canonical_exists",
                "rerun_exists",
                "row_count_matches",
                "summary_keys_match",
                "forbidden_true_flag_count",
                *SAFETY_FLAGS,
            ]
        )
    selected = results[
        (results["validation_status"].fillna("").astype(str) != "pass")
        | results["notes"].fillna("").astype(str).str.contains("fingerprint", case=False)
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        classification = classify_warning(row)
        triage_status = "acceptable" if classification.startswith("acceptable_") else "needs_investigation"
        if triage_status == "acceptable":
            reason = "Structure, row counts, summary keys, and safety flags match; only normalized fingerprint differs."
        else:
            reason = "Structure, row counts, summary keys, forbidden flags, or safety flags require investigation."
        rows.append(
            {
                "rerun_id": row.get("rerun_id", ""),
                "step_name": row.get("step_name", ""),
                "file_name": row.get("file_name", ""),
                "validation_status": row.get("validation_status", ""),
                "triage_classification": classification,
                "triage_status": triage_status,
                "reason": reason,
                "notes": row.get("notes", ""),
                "canonical_exists": bool(row.get("canonical_exists", False)),
                "rerun_exists": bool(row.get("rerun_exists", False)),
                "row_count_matches": bool(row.get("row_count_matches", False)),
                "summary_keys_match": bool(row.get("summary_keys_match", False)),
                "forbidden_true_flag_count": _int_value(row.get("forbidden_true_flag_count")),
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_summary(triage: pd.DataFrame) -> pd.DataFrame:
    total = int(len(triage))
    acceptable = int((triage["triage_status"] == "acceptable").sum()) if not triage.empty else 0
    needs = int((triage["triage_status"] == "needs_investigation").sum()) if not triage.empty else 0
    path_count = int((triage["triage_classification"] == "acceptable_path_or_workspace_difference").sum()) if not triage.empty else 0
    run_config_count = int((triage["triage_classification"] == "acceptable_run_config_path_difference").sum()) if not triage.empty else 0
    report_count = int((triage["triage_classification"] == "acceptable_report_embedded_path_difference").sum()) if not triage.empty else 0
    forbidden = int(triage["forbidden_true_flag_count"].sum()) if not triage.empty else 0
    if needs:
        status = "fail"
        conclusion = "reproducibility_warnings_need_investigation_research_only"
    else:
        status = "pass"
        conclusion = "reproducibility_warnings_triaged_as_acceptable_research_only"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step5_reproducibility_warning_triage",
                "total_warning_row_count": total,
                "acceptable_warning_count": acceptable,
                "needs_investigation_count": needs,
                "path_or_workspace_warning_count": path_count,
                "run_config_warning_count": run_config_count,
                "report_embedded_path_warning_count": report_count,
                "forbidden_true_flag_count": forbidden,
                "trading_ready": False,
                "validation_status": status,
                "conclusion": conclusion,
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The triage layer reads Step 4 CSV outputs only.", "No historical backtest is run."),
        ("no_market_data_fetch", "confirmed", "No market-data source arguments or data loader calls exist.", "No market data is fetched."),
        ("no_threshold_change", "confirmed", "No threshold module or value is changed.", "Thresholds are not modified."),
        ("no_model_retraining", "confirmed", "No training module is imported or called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is imported or called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local Step 4 outputs are read.", "No new data source is added."),
        ("no_broker_credentials", "confirmed", "The CLI does not accept credentials and the module does not request credentials.", "No credential handling exists."),
        ("no_broker_sdk_import", "confirmed", "The module imports only standard library modules and pandas.", "No broker SDK is imported."),
        ("no_broker_connection", "confirmed", "The triage layer reads files only.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "The triage layer has no live trading path.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "The triage layer has no execution path.", "No orders are executed."),
        ("no_real_order_submission", "confirmed", "The triage layer has no order submission path.", "No real orders are submitted."),
        ("no_trading_ready_upgrade", "confirmed", "The summary writes trading_ready as false.", "No deployable status is claimed."),
        ("warning_triage_only", "confirmed", "The outputs classify Step 4 warnings only.", "No new trading capability is added."),
        ("previous_outputs_not_overwritten", "confirmed", "Step 4 inputs are read-only.", "Prior outputs are not overwritten."),
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


def build_report(summary: pd.DataFrame, triage: pd.DataFrame, guardrails: pd.DataFrame) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 5 Validation Warning Triage / Normalization Report",
            "",
            "## Executive Summary",
            "V6 Step 5 triages V6 Step 4 reproducibility warnings into acceptable normalized path/workspace differences or genuine reproducibility issues.",
            "It reads Step 4 outputs only and does not rerun any command.",
            "It does not run backtests, fetch market data, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or upgrade trading readiness.",
            "",
            "## Summary",
            f"- Warning rows: {row.get('total_warning_row_count', 0)}",
            f"- Acceptable warnings: {row.get('acceptable_warning_count', 0)}",
            f"- Needs investigation: {row.get('needs_investigation_count', 0)}",
            f"- Path/workspace warnings: {row.get('path_or_workspace_warning_count', 0)}",
            f"- Run config warnings: {row.get('run_config_warning_count', 0)}",
            f"- Report embedded path warnings: {row.get('report_embedded_path_warning_count', 0)}",
            f"- Forbidden true flags: {row.get('forbidden_true_flag_count', 0)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            "",
            "## Triage Results",
            _table(triage, "No warning rows required triage."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This warning triage report is educational/research-only. It is not financial advice and is not a trading-ready certification.",
            "",
        ]
    )


def generate_reproducibility_warning_triage_outputs(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    inputs = load_inputs(input_path)
    triage = build_triage_results(inputs["results"])
    summary = build_summary(triage)
    guardrails = build_guardrails()
    report = build_report(summary, triage, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    out_paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(out_paths["summary"], index=False)
    triage.to_csv(out_paths["results"], index=False)
    guardrails.to_csv(out_paths["guardrails"], index=False)
    out_paths["report"].write_text(report, encoding="utf-8")
    config = {
        "input_dir": str(input_path),
        "summary_path": str(input_path / "reproducibility_rerun_summary.csv"),
        "results_path": str(input_path / "reproducibility_rerun_results.csv"),
        "guardrails_path": str(input_path / "reproducibility_rerun_guardrails.csv"),
        "output_dir": str(output_path),
        "triaged_warning_rows": int(len(triage)),
        "scope": "V6 Step 5 reproducibility warning triage only",
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
        "reproducibility_warning_triage_summary": summary,
        "reproducibility_warning_triage_results": triage,
        "reproducibility_warning_triage_guardrails": guardrails,
        "reproducibility_warning_triage_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in out_paths.items()},
    }
