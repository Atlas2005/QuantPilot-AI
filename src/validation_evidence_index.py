import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BASELINE_DIR = Path("outputs/validation_baseline_manifest_real_v1")
DEFAULT_SCHEMA_VALIDATOR_DIR = Path("outputs/output_schema_validator_real_v1")
DEFAULT_DEPENDENCY_VALIDATOR_DIR = Path("outputs/cross_step_dependency_validator_real_v1")
DEFAULT_RERUN_VALIDATOR_DIR = Path("outputs/reproducibility_rerun_validator_real_v1")
DEFAULT_WARNING_TRIAGE_DIR = Path("outputs/reproducibility_warning_triage_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/validation_evidence_index_real_v1")

OUTPUT_FILENAMES = {
    "catalog": "validation_evidence_catalog.csv",
    "traceability": "validation_evidence_traceability_matrix.csv",
    "guardrails": "validation_evidence_guardrails.csv",
    "summary": "validation_evidence_summary.csv",
    "report": "validation_evidence_report.md",
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
    {
        "step": "V6 Step 1 Validation Baseline Manifest",
        "key": "baseline",
        "required_files": [
            "validation_baseline_summary.csv",
            "validation_baseline_manifest.csv",
            "validation_baseline_guardrails.csv",
            "validation_baseline_report.md",
            "run_config.json",
        ],
        "summary_file": "validation_baseline_summary.csv",
    },
    {
        "step": "V6 Step 2 Output Schema Validator",
        "key": "schema_validator",
        "required_files": [
            "output_schema_validation_summary.csv",
            "output_schema_validation_results.csv",
            "output_schema_validation_guardrails.csv",
            "output_schema_validation_report.md",
            "run_config.json",
        ],
        "summary_file": "output_schema_validation_summary.csv",
    },
    {
        "step": "V6 Step 3 Cross-Step Dependency Validator",
        "key": "dependency_validator",
        "required_files": [
            "cross_step_dependency_summary.csv",
            "cross_step_dependency_results.csv",
            "cross_step_dependency_guardrails.csv",
            "cross_step_dependency_report.md",
            "run_config.json",
        ],
        "summary_file": "cross_step_dependency_summary.csv",
    },
    {
        "step": "V6 Step 4 Reproducibility Rerun Validator",
        "key": "rerun_validator",
        "required_files": [
            "reproducibility_rerun_summary.csv",
            "reproducibility_rerun_results.csv",
            "reproducibility_rerun_guardrails.csv",
            "reproducibility_rerun_report.md",
            "run_config.json",
        ],
        "summary_file": "reproducibility_rerun_summary.csv",
    },
    {
        "step": "V6 Step 5 Reproducibility Warning Triage",
        "key": "warning_triage",
        "required_files": [
            "reproducibility_warning_triage_summary.csv",
            "reproducibility_warning_triage_results.csv",
            "reproducibility_warning_triage_guardrails.csv",
            "reproducibility_warning_triage_report.md",
            "run_config.json",
        ],
        "summary_file": "reproducibility_warning_triage_summary.csv",
    },
]


def build_input_paths(
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
    schema_validator_dir: str | Path = DEFAULT_SCHEMA_VALIDATOR_DIR,
    dependency_validator_dir: str | Path = DEFAULT_DEPENDENCY_VALIDATOR_DIR,
    rerun_validator_dir: str | Path = DEFAULT_RERUN_VALIDATOR_DIR,
    warning_triage_dir: str | Path = DEFAULT_WARNING_TRIAGE_DIR,
) -> dict[str, Path]:
    return {
        "baseline": Path(baseline_dir),
        "schema_validator": Path(schema_validator_dir),
        "dependency_validator": Path(dependency_validator_dir),
        "rerun_validator": Path(rerun_validator_dir),
        "warning_triage": Path(warning_triage_dir),
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
        total = 1 if flag in value and _is_true(value[flag]) else 0
        return total + sum(_json_flag_count(child, flag) for child in value.values())
    if isinstance(value, list):
        return sum(_json_flag_count(child, flag) for child in value)
    return 0


def _file_category(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".md":
        return "markdown"
    if suffix == ".json":
        return "json"
    return "other"


def _summary_fields(summary_path: Path) -> dict[str, str]:
    frame = _read_csv(summary_path)
    if frame.empty:
        return {"validation_status": "", "conclusion": ""}
    row = frame.iloc[0]
    return {
        "validation_status": str(row.get("validation_status", "")),
        "conclusion": str(row.get("conclusion", row.get("baseline_status", ""))),
    }


def _file_flag_counts(path: Path) -> dict[str, int]:
    counts = {flag: 0 for flag in SAFETY_FLAGS}
    category = _file_category(path)
    if category == "csv":
        frame = _read_csv(path)
        for flag in SAFETY_FLAGS:
            counts[flag] = _bool_count(frame, flag)
    elif category == "json":
        payload = _read_json(path)
        for flag in SAFETY_FLAGS:
            counts[flag] = _json_flag_count(payload, flag)
    return counts


def build_catalog(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition in STEP_DEFINITIONS:
        step = str(definition["step"])
        base = paths[str(definition["key"])]
        summary_info = _summary_fields(base / str(definition["summary_file"]))
        files = sorted([item for item in base.iterdir() if item.is_file()]) if base.exists() else []
        required = set(definition["required_files"])
        for file_path in files:
            category = _file_category(file_path)
            row_count = len(_read_csv(file_path)) if category == "csv" else 0
            flags = _file_flag_counts(file_path)
            rows.append(
                {
                    "source_step": step,
                    "evidence_file_path": str(file_path),
                    "evidence_file_name": file_path.name,
                    "file_category": category,
                    "required_evidence": file_path.name in required,
                    "file_exists": file_path.exists(),
                    "byte_size": file_path.stat().st_size if file_path.exists() else 0,
                    "row_count": row_count,
                    "source_validation_status": summary_info["validation_status"],
                    "source_conclusion": summary_info["conclusion"],
                    "trading_ready_true_count": flags["trading_ready"],
                    "execution_allowed_true_count": flags["execution_allowed"],
                    "broker_connected_true_count": flags["broker_connected"],
                    "live_trading_true_count": flags["live_trading"],
                    "real_order_submission_true_count": flags["real_order_submission"],
                    "trading_ready": False,
                }
            )
    return pd.DataFrame(rows)


def build_traceability(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition in STEP_DEFINITIONS:
        step = str(definition["step"])
        base = paths[str(definition["key"])]
        summary_info = _summary_fields(base / str(definition["summary_file"]))
        for file_name in definition["required_files"]:
            path = base / str(file_name)
            rows.append(
                {
                    "source_step": step,
                    "required_evidence_file": file_name,
                    "evidence_file_path": str(path),
                    "file_category": _file_category(path),
                    "file_exists": path.exists(),
                    "source_validation_status": summary_info["validation_status"],
                    "source_conclusion": summary_info["conclusion"],
                    "traceability_status": "present" if path.exists() else "missing",
                    "trading_ready": False,
                }
            )
    return pd.DataFrame(rows)


def build_summary(paths: dict[str, Path], catalog: pd.DataFrame, traceability: pd.DataFrame) -> pd.DataFrame:
    indexed_steps = int(sum(1 for path in paths.values() if path.exists()))
    expected_steps = len(STEP_DEFINITIONS)
    missing_steps = expected_steps - indexed_steps
    missing_required = int((traceability["file_exists"] == False).sum()) if not traceability.empty else expected_steps
    forbidden_true = int(
        catalog[
            [
                "trading_ready_true_count",
                "execution_allowed_true_count",
                "broker_connected_true_count",
                "live_trading_true_count",
                "real_order_submission_true_count",
            ]
        ].sum().sum()
    ) if not catalog.empty else 0
    status = "pass" if missing_steps == 0 and missing_required == 0 and forbidden_true == 0 else "fail"
    conclusion = (
        "validation_evidence_index_audit_trail_created_research_only"
        if status == "pass"
        else "validation_evidence_index_requires_investigation_research_only"
    )
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step6_validation_evidence_index",
                "indexed_step_count": indexed_steps,
                "expected_step_count": expected_steps,
                "missing_step_count": missing_steps,
                "indexed_evidence_file_count": int(len(catalog)),
                "csv_evidence_file_count": int((catalog["file_category"] == "csv").sum()) if not catalog.empty else 0,
                "markdown_evidence_file_count": int((catalog["file_category"] == "markdown").sum()) if not catalog.empty else 0,
                "json_evidence_file_count": int((catalog["file_category"] == "json").sum()) if not catalog.empty else 0,
                "total_catalog_row_count": int(len(catalog)),
                "traceability_row_count": int(len(traceability)),
                "missing_required_evidence_count": missing_required,
                "forbidden_true_flag_count": forbidden_true,
                "trading_ready": False,
                "validation_status": status,
                "conclusion": conclusion,
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The evidence index reads existing V6 outputs only.", "No historical backtest is run."),
        ("no_market_data_fetch", "confirmed", "No market-data source arguments or data loader calls exist.", "No market data is fetched."),
        ("no_threshold_change", "confirmed", "No threshold module or value is changed.", "Thresholds are not modified."),
        ("no_model_retraining", "confirmed", "No training module is imported or called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is imported or called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local V6 output directories are read.", "No new data source is added."),
        ("no_broker_connection", "confirmed", "The evidence index reads files only.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "The evidence index has no live trading path.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "The evidence index has no execution path.", "No orders are executed."),
        ("no_real_order_submission", "confirmed", "The evidence index has no order submission path.", "No real orders are submitted."),
        ("no_trading_ready_upgrade", "confirmed", "The summary writes trading_ready as false.", "No deployable status is claimed."),
        ("evidence_index_only", "confirmed", "The outputs catalog and cross-reference validation evidence only.", "No trading capability is added."),
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


def build_report(summary: pd.DataFrame, catalog: pd.DataFrame, traceability: pd.DataFrame, guardrails: pd.DataFrame) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 6 Validation Evidence Index / Audit Trail Catalog",
            "",
            "## Executive Summary",
            "V6 Step 6 catalogs and cross-references validation evidence files from V6 Step 1-5.",
            "It reads existing local V6 outputs only and creates a unified research audit trail.",
            "It does not run backtests, fetch market data, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or upgrade trading readiness.",
            "",
            "## Summary",
            f"- Indexed steps: {row.get('indexed_step_count', 0)} / {row.get('expected_step_count', 0)}",
            f"- Indexed evidence files: {row.get('indexed_evidence_file_count', 0)}",
            f"- CSV evidence files: {row.get('csv_evidence_file_count', 0)}",
            f"- Markdown evidence files: {row.get('markdown_evidence_file_count', 0)}",
            f"- JSON evidence files: {row.get('json_evidence_file_count', 0)}",
            f"- Missing required evidence: {row.get('missing_required_evidence_count', 0)}",
            f"- Forbidden true flags: {row.get('forbidden_true_flag_count', 0)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            "",
            "## Evidence Catalog",
            _table(catalog, "No evidence catalog rows were generated."),
            "",
            "## Traceability Matrix",
            _table(traceability, "No traceability rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This evidence index is educational/research-only. It is not financial advice and is not a trading-ready certification.",
            "",
        ]
    )


def generate_validation_evidence_index_outputs(
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
    schema_validator_dir: str | Path = DEFAULT_SCHEMA_VALIDATOR_DIR,
    dependency_validator_dir: str | Path = DEFAULT_DEPENDENCY_VALIDATOR_DIR,
    rerun_validator_dir: str | Path = DEFAULT_RERUN_VALIDATOR_DIR,
    warning_triage_dir: str | Path = DEFAULT_WARNING_TRIAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        baseline_dir=baseline_dir,
        schema_validator_dir=schema_validator_dir,
        dependency_validator_dir=dependency_validator_dir,
        rerun_validator_dir=rerun_validator_dir,
        warning_triage_dir=warning_triage_dir,
    )
    output_path = Path(output_dir)
    catalog = build_catalog(paths)
    traceability = build_traceability(paths)
    summary = build_summary(paths, catalog, traceability)
    guardrails = build_guardrails()
    report = build_report(summary, catalog, traceability, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    out_paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    catalog.to_csv(out_paths["catalog"], index=False)
    traceability.to_csv(out_paths["traceability"], index=False)
    guardrails.to_csv(out_paths["guardrails"], index=False)
    summary.to_csv(out_paths["summary"], index=False)
    out_paths["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "indexed_step_count": int(summary.iloc[0]["indexed_step_count"]),
        "indexed_evidence_file_count": int(len(catalog)),
        "scope": "V6 Step 6 validation evidence index and audit trail only",
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
        "validation_evidence_catalog": catalog,
        "validation_evidence_traceability_matrix": traceability,
        "validation_evidence_guardrails": guardrails,
        "validation_evidence_summary": summary,
        "validation_evidence_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in out_paths.items()},
    }
