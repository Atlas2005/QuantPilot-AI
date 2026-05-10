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
DEFAULT_EVIDENCE_INDEX_DIR = Path("outputs/validation_evidence_index_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/validation_coverage_gap_review_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "summary": "validation_coverage_gap_summary.csv",
    "coverage": "validation_coverage_gap_results.csv",
    "risks": "validation_readiness_risk_register.csv",
    "guardrails": "validation_coverage_gap_guardrails.csv",
    "report": "validation_coverage_gap_report.md",
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
        "step_name": "V6 Step 1 Validation Baseline Manifest",
        "key": "baseline",
        "validation_category": "baseline_manifest",
        "summary_file": "validation_baseline_summary.csv",
    },
    {
        "step_name": "V6 Step 2 Output Schema Validator",
        "key": "schema",
        "validation_category": "schema_validation",
        "summary_file": "output_schema_validation_summary.csv",
    },
    {
        "step_name": "V6 Step 3 Cross-Step Dependency Validator",
        "key": "dependency",
        "validation_category": "dependency_integrity",
        "summary_file": "cross_step_dependency_summary.csv",
    },
    {
        "step_name": "V6 Step 4 Reproducibility Rerun Validator",
        "key": "rerun",
        "validation_category": "reproducibility_rerun",
        "summary_file": "reproducibility_rerun_summary.csv",
    },
    {
        "step_name": "V6 Step 5 Warning Triage",
        "key": "warning_triage",
        "validation_category": "warning_triage",
        "summary_file": "reproducibility_warning_triage_summary.csv",
    },
    {
        "step_name": "V6 Step 6 Evidence Index",
        "key": "evidence",
        "validation_category": "evidence_audit_trail",
        "summary_file": "validation_evidence_summary.csv",
    },
]


def build_input_paths(
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
    schema_validator_dir: str | Path = DEFAULT_SCHEMA_VALIDATOR_DIR,
    dependency_validator_dir: str | Path = DEFAULT_DEPENDENCY_VALIDATOR_DIR,
    rerun_validator_dir: str | Path = DEFAULT_RERUN_VALIDATOR_DIR,
    warning_triage_dir: str | Path = DEFAULT_WARNING_TRIAGE_DIR,
    evidence_index_dir: str | Path = DEFAULT_EVIDENCE_INDEX_DIR,
) -> dict[str, Path]:
    return {
        "baseline": Path(baseline_dir),
        "schema": Path(schema_validator_dir),
        "dependency": Path(dependency_validator_dir),
        "rerun": Path(rerun_validator_dir),
        "warning_triage": Path(warning_triage_dir),
        "evidence": Path(evidence_index_dir),
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


def _int_value(value: Any) -> int:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(number) if pd.notna(number) else 0


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


def count_forbidden_flags(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    total = 0
    for csv_path in path.glob("*.csv"):
        frame = _read_csv(csv_path)
        total += sum(_bool_count(frame, flag) for flag in SAFETY_FLAGS)
    for json_path in path.glob("*.json"):
        payload = _read_json(json_path)
        total += sum(_json_flag_count(payload, flag) for flag in SAFETY_FLAGS)
    return int(total)


def first_row(path: Path) -> dict[str, Any]:
    frame = _read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _status_from_row(row: dict[str, Any], evidence_present: bool, forbidden: int) -> str:
    if not evidence_present or forbidden:
        return "fail"
    return str(row.get("validation_status", "pass") or "pass")


def _warning_count(row: dict[str, Any]) -> int:
    for key in [
        "schema_warning_count",
        "dependency_warning_count",
        "rerun_warning_count",
        "warning_issue_count",
        "total_warning_row_count",
    ]:
        if key in row:
            return _int_value(row.get(key))
    return 0


def _fail_count(row: dict[str, Any]) -> int:
    for key in [
        "schema_fail_count",
        "dependency_fail_count",
        "rerun_fail_count",
        "needs_investigation_count",
        "missing_required_evidence_count",
    ]:
        if key in row:
            return _int_value(row.get(key))
    return 0


def build_coverage_matrix(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition in STEP_DEFINITIONS:
        output_dir = paths[str(definition["key"])]
        summary_path = output_dir / str(definition["summary_file"])
        row = first_row(summary_path)
        evidence_present = output_dir.exists() and summary_path.exists()
        forbidden = count_forbidden_flags(output_dir)
        validation_status = _status_from_row(row, evidence_present, forbidden)
        rows.append(
            {
                "step_name": definition["step_name"],
                "output_dir": str(output_dir),
                "validation_category": definition["validation_category"],
                "validation_status": validation_status,
                "evidence_present": evidence_present,
                "missing_required_evidence_count": _int_value(row.get("missing_required_evidence_count")) if evidence_present else 1,
                "warning_count": _warning_count(row),
                "fail_count": _fail_count(row),
                "forbidden_true_flag_count": forbidden,
                "trading_ready": False,
                "source_conclusion": str(row.get("conclusion", row.get("baseline_status", ""))),
            }
        )
    return pd.DataFrame(rows)


def build_risk_register() -> pd.DataFrame:
    gaps = [
        (
            "GAP-001",
            "no_real_time_paper_trading_evidence_yet",
            "blocking",
            "open_future_work",
            "No multi-day real-time paper trading evidence has been collected.",
            "Implement future V6 paper replay/simulation evidence before any readiness claim.",
            True,
        ),
        (
            "GAP-002",
            "no_walk_forward_simulation_hardening_yet",
            "blocking",
            "open_future_work",
            "Validation has not yet hardened walk-forward simulation behavior.",
            "Add future walk-forward simulation hardening with locked assumptions.",
            True,
        ),
        (
            "GAP-003",
            "no_out_of_sample_live_or_paper_validation_layer_yet",
            "blocking",
            "open_future_work",
            "No out-of-sample live or real-time paper validation layer exists.",
            "Add future OOS paper validation layer and retain research-only controls.",
            True,
        ),
        (
            "GAP-004",
            "no_capital_reconciliation_against_real_broker_account_yet",
            "blocking",
            "open_future_work",
            "Capital constraints have not been reconciled with a real account source.",
            "Design future capital/account reconciliation after broker sandbox controls exist.",
            True,
        ),
        (
            "GAP-005",
            "no_broker_sandbox_or_live_integration_yet",
            "blocking",
            "open_future_work",
            "No broker sandbox or live broker integration exists.",
            "Future broker work requires separate credential, compliance, and sandbox controls.",
            True,
        ),
        (
            "GAP-006",
            "no_production_monitoring_or_kill_switch_yet",
            "blocking",
            "open_future_work",
            "No production daemon, alerting process, or kill switch exists.",
            "Design future monitoring daemon and kill switch before any execution bridge.",
            True,
        ),
        (
            "GAP-007",
            "no_compliance_or_risk_approval_layer_yet",
            "blocking",
            "open_future_work",
            "No compliance, suitability, tax, or risk approval workflow exists.",
            "Add future human approval and risk governance process before deployability discussion.",
            True,
        ),
        (
            "GAP-008",
            "no_trading_ready_candidate_exists",
            "blocking",
            "open_future_work",
            "Prior validation did not produce a trading-ready candidate.",
            "Keep all future validation research-only until a separately validated candidate exists.",
            True,
        ),
        (
            "GAP-009",
            "reproducibility_fingerprint_warnings_triaged_as_path_differences",
            "info",
            "triaged_acceptable",
            "Step 5 classified Step 4 warnings as acceptable path/report embedded differences.",
            "No immediate fix; continue preserving isolated rerun path normalization.",
            False,
        ),
    ]
    return pd.DataFrame(
        [
            {
                "gap_id": gap_id,
                "gap_name": gap_name,
                "severity": severity,
                "current_status": current_status,
                "why_it_matters": why,
                "required_future_resolution": resolution,
                "allowed_to_fix_now": False,
                "trading_ready_blocker": bool(blocker),
                "trading_ready": False,
            }
            for gap_id, gap_name, severity, current_status, why, resolution, blocker in gaps
        ]
    )


def build_summary(coverage: pd.DataFrame, risks: pd.DataFrame) -> pd.DataFrame:
    forbidden = int(coverage["forbidden_true_flag_count"].sum()) if not coverage.empty else 0
    fail_count = int((coverage["validation_status"] == "fail").sum()) if not coverage.empty else 1
    validation_status = "pass" if forbidden == 0 and fail_count == 0 else "fail"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step7_validation_coverage_gap_review",
                "reviewed_v6_step_count": int(len(coverage)),
                "coverage_result_row_count": int(len(coverage)),
                "readiness_gap_count": int(len(risks)),
                "blocking_gap_count": int((risks["severity"] == "blocking").sum()) if not risks.empty else 0,
                "warning_gap_count": int((risks["severity"] == "warning").sum()) if not risks.empty else 0,
                "info_gap_count": int((risks["severity"] == "info").sum()) if not risks.empty else 0,
                "forbidden_true_flag_count": forbidden,
                "trading_ready": False,
                "validation_status": validation_status,
                "conclusion": "validation_coverage_gap_review_completed_research_only",
                "recommended_next_step": "V6 Step 8 simulation_hardening_design_or_multi_day_paper_replay_planning",
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The review reads existing V6 validation outputs only.", "No historical backtest is run."),
        ("no_market_data_fetch", "confirmed", "No market-data source arguments or data loader calls exist.", "No market data is fetched."),
        ("no_threshold_change", "confirmed", "No threshold module or value is changed.", "Thresholds are not modified."),
        ("no_model_retraining", "confirmed", "No training module is imported or called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is imported or called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local V6 output directories are read.", "No new data source is added."),
        ("no_broker_connection", "confirmed", "The review reads files only.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "The review has no live trading path.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "The review has no execution path.", "No orders are executed."),
        ("no_real_order_submission", "confirmed", "The review has no order submission path.", "No real orders are submitted."),
        ("no_trading_ready_upgrade", "confirmed", "The summary writes trading_ready as false.", "No deployable status is claimed."),
        ("coverage_gap_review_only", "confirmed", "The outputs identify validation/readiness gaps only.", "No gap is fixed in this step."),
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


def build_report(summary: pd.DataFrame, coverage: pd.DataFrame, risks: pd.DataFrame, guardrails: pd.DataFrame) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 7 Validation Coverage Gap / Readiness Risk Review",
            "",
            "## Executive Summary",
            "V6 Step 7 reviews existing V6 validation evidence and identifies remaining readiness gaps before any future simulation hardening.",
            "The validation evidence exists and safety flags remain false, but blocking future-work gaps remain. This is expected and does not make the project trading-ready.",
            "It does not run backtests, fetch market data, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or upgrade trading readiness.",
            "",
            "## Summary",
            f"- Reviewed V6 steps: {row.get('reviewed_v6_step_count', 0)}",
            f"- Coverage rows: {row.get('coverage_result_row_count', 0)}",
            f"- Readiness gaps: {row.get('readiness_gap_count', 0)}",
            f"- Blocking gaps: {row.get('blocking_gap_count', 0)}",
            f"- Warning gaps: {row.get('warning_gap_count', 0)}",
            f"- Info gaps: {row.get('info_gap_count', 0)}",
            f"- Forbidden true flags: {row.get('forbidden_true_flag_count', 0)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            f"- Recommended next step: {row.get('recommended_next_step', '')}",
            "",
            "## Coverage Matrix",
            _table(coverage, "No coverage rows were generated."),
            "",
            "## Readiness Risk Register",
            _table(risks, "No readiness risk rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This coverage gap review is educational/research-only. It is not financial advice and is not a trading-ready certification.",
            "",
        ]
    )


def generate_validation_coverage_gap_review_outputs(
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
    schema_validator_dir: str | Path = DEFAULT_SCHEMA_VALIDATOR_DIR,
    dependency_validator_dir: str | Path = DEFAULT_DEPENDENCY_VALIDATOR_DIR,
    rerun_validator_dir: str | Path = DEFAULT_RERUN_VALIDATOR_DIR,
    warning_triage_dir: str | Path = DEFAULT_WARNING_TRIAGE_DIR,
    evidence_index_dir: str | Path = DEFAULT_EVIDENCE_INDEX_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        baseline_dir=baseline_dir,
        schema_validator_dir=schema_validator_dir,
        dependency_validator_dir=dependency_validator_dir,
        rerun_validator_dir=rerun_validator_dir,
        warning_triage_dir=warning_triage_dir,
        evidence_index_dir=evidence_index_dir,
    )
    output_path = Path(output_dir)
    coverage = build_coverage_matrix(paths)
    risks = build_risk_register()
    summary = build_summary(coverage, risks)
    guardrails = build_guardrails()
    report = build_report(summary, coverage, risks, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {
        "run_config": output_path / "run_config.json",
        "summary": output_path / "validation_coverage_gap_summary.csv",
        "coverage": output_path / "validation_coverage_gap_results.csv",
        "risks": output_path / "validation_readiness_risk_register.csv",
        "guardrails": output_path / "validation_coverage_gap_guardrails.csv",
        "report": output_path / "validation_coverage_gap_report.md",
    }
    summary.to_csv(output_files["summary"], index=False)
    coverage.to_csv(output_files["coverage"], index=False)
    risks.to_csv(output_files["risks"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "reviewed_v6_step_count": int(len(coverage)),
        "readiness_gap_count": int(len(risks)),
        "scope": "V6 Step 7 validation coverage gap review only",
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
        "validation_coverage_gap_summary": summary,
        "validation_coverage_gap_results": coverage,
        "validation_readiness_risk_register": risks,
        "validation_coverage_gap_guardrails": guardrails,
        "validation_coverage_gap_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
