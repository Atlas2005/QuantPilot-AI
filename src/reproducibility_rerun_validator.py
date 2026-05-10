import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SEMI_AUTO_DIR = Path("outputs/semi_auto_order_generator_real_v1")
DEFAULT_BROKER_RESEARCH_DIR = Path("outputs/broker_integration_research_real_v1")
DEFAULT_CAPITAL_DIR = Path("outputs/capital_constraint_engine_real_v1")
DEFAULT_UNIVERSE_DIR = Path("outputs/tradable_universe_filter_real_v1")
DEFAULT_POSITION_DIR = Path("outputs/position_sizing_engine_real_v1")
DEFAULT_EXIT_DIR = Path("outputs/exit_engine_real_v1")
DEFAULT_DAILY_PLAN_DIR = Path("outputs/daily_trading_plan_real_v1")
DEFAULT_PAPER_LEDGER_DIR = Path("outputs/paper_trading_ledger_real_v1")
DEFAULT_MONITORING_DIR = Path("outputs/monitoring_reporting_layer_real_v1")
DEFAULT_V5_CLOSURE_DIR = Path("outputs/capital_aware_infrastructure_review_real_v1")
DEFAULT_BASELINE_DIR = Path("outputs/validation_baseline_manifest_real_v1")
DEFAULT_SCHEMA_VALIDATOR_DIR = Path("outputs/output_schema_validator_real_v1")
DEFAULT_DEPENDENCY_VALIDATOR_DIR = Path("outputs/cross_step_dependency_validator_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/reproducibility_rerun_validator_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "summary": "reproducibility_rerun_summary.csv",
    "results": "reproducibility_rerun_results.csv",
    "guardrails": "reproducibility_rerun_guardrails.csv",
    "report": "reproducibility_rerun_report.md",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
]

RERUN_SPECS = [
    {
        "rerun_id": "RERUN-001",
        "step_name": "V5 Step 8 Broker Integration Research",
        "script": "src/run_broker_integration_research.py",
        "canonical_key": "broker",
        "rerun_subdir": "v5_step8_broker_integration_research",
        "expected_files": [
            "broker_integration_summary.csv",
            "broker_integration_modes.csv",
            "broker_integration_constraints.csv",
            "broker_integration_risk_register.csv",
            "broker_integration_guardrails.csv",
            "broker_integration_research_report.md",
            "run_config.json",
        ],
        "summary_file": "broker_integration_summary.csv",
        "summary_keys": [
            "input_draft_order_count",
            "researched_mode_count",
            "constraint_count",
            "high_risk_constraint_count",
            "risk_register_count",
            "broker_connected_count",
            "execution_allowed_count",
            "trading_ready_count",
            "live_trading_count",
            "real_order_submission_count",
            "conclusion",
        ],
    },
    {
        "rerun_id": "RERUN-002",
        "step_name": "V5 Step 9 Monitoring / Reporting Layer",
        "script": "src/run_monitoring_reporting_layer.py",
        "canonical_key": "monitoring",
        "rerun_subdir": "v5_step9_monitoring_reporting_layer",
        "expected_files": [
            "monitoring_summary.csv",
            "monitoring_status_dashboard.csv",
            "monitoring_alerts.csv",
            "monitoring_guardrails.csv",
            "monitoring_report.md",
            "run_config.json",
        ],
        "summary_file": "monitoring_summary.csv",
        "summary_keys": [
            "monitored_step_count",
            "dashboard_row_count",
            "alert_count",
            "blocking_alert_count",
            "warning_alert_count",
            "trading_ready_true_count",
            "execution_allowed_true_count",
            "broker_connected_true_count",
            "live_trading_true_count",
            "real_order_submission_true_count",
            "conclusion",
        ],
    },
    {
        "rerun_id": "RERUN-003",
        "step_name": "V5 Step 10 Capital-Aware Infrastructure Review / Closure",
        "script": "src/run_capital_aware_infrastructure_review.py",
        "canonical_key": "v5_closure",
        "rerun_subdir": "v5_step10_capital_aware_closure",
        "expected_files": [
            "v5_infrastructure_closure_summary.csv",
            "v5_step_capability_matrix.csv",
            "v5_guardrail_audit.csv",
            "v5_limitations_register.csv",
            "v5_readiness_blockers.csv",
            "v5_next_phase_recommendations.csv",
            "v5_capital_aware_closure_report.md",
            "run_config.json",
        ],
        "summary_file": "v5_infrastructure_closure_summary.csv",
        "summary_keys": [
            "reviewed_step_count",
            "completed_step_count",
            "missing_step_count",
            "trading_ready_true_count",
            "execution_allowed_true_count",
            "broker_connected_true_count",
            "live_trading_true_count",
            "real_order_submission_true_count",
            "final_v5_status",
            "recommended_next_phase",
        ],
    },
    {
        "rerun_id": "RERUN-004",
        "step_name": "V6 Step 1 Validation Baseline Manifest",
        "script": "src/run_validation_baseline_manifest.py",
        "canonical_key": "baseline",
        "rerun_subdir": "v6_step1_validation_baseline_manifest",
        "expected_files": [
            "validation_baseline_summary.csv",
            "validation_baseline_manifest.csv",
            "validation_baseline_guardrails.csv",
            "validation_baseline_report.md",
            "run_config.json",
        ],
        "summary_file": "validation_baseline_summary.csv",
        "summary_keys": [
            "baseline_step_count",
            "present_output_dir_count",
            "missing_output_dir_count",
            "trading_ready_true_count",
            "execution_allowed_true_count",
            "broker_connected_true_count",
            "live_trading_true_count",
            "real_order_submission_true_count",
            "baseline_status",
        ],
    },
    {
        "rerun_id": "RERUN-005",
        "step_name": "V6 Step 2 Output Schema Validator",
        "script": "src/run_output_schema_validator.py",
        "canonical_key": "schema_validator",
        "rerun_subdir": "v6_step2_output_schema_validator",
        "expected_files": [
            "run_config.json",
            "output_schema_validation_summary.csv",
            "output_schema_validation_results.csv",
            "output_schema_validation_guardrails.csv",
            "output_schema_validation_report.md",
        ],
        "summary_file": "output_schema_validation_summary.csv",
        "summary_keys": [
            "checked_directory_count",
            "checked_file_count",
            "present_file_count",
            "missing_file_count",
            "schema_fail_count",
            "forbidden_true_flag_count",
            "validation_status",
            "conclusion",
        ],
    },
    {
        "rerun_id": "RERUN-006",
        "step_name": "V6 Step 3 Cross-Step Dependency Integrity Validator",
        "script": "src/run_cross_step_dependency_validator.py",
        "canonical_key": "dependency_validator",
        "rerun_subdir": "v6_step3_cross_step_dependency_validator",
        "expected_files": [
            "cross_step_dependency_results.csv",
            "cross_step_dependency_summary.csv",
            "cross_step_dependency_guardrails.csv",
            "cross_step_dependency_report.md",
            "run_config.json",
        ],
        "summary_file": "cross_step_dependency_summary.csv",
        "summary_keys": [
            "checked_dependency_count",
            "dependency_fail_count",
            "checked_output_dir_count",
            "missing_output_dir_count",
            "forbidden_true_flag_count",
            "validation_status",
            "conclusion",
        ],
    },
]


def build_input_paths(
    semi_auto_dir: str | Path = DEFAULT_SEMI_AUTO_DIR,
    broker_research_dir: str | Path = DEFAULT_BROKER_RESEARCH_DIR,
    capital_dir: str | Path = DEFAULT_CAPITAL_DIR,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
    position_dir: str | Path = DEFAULT_POSITION_DIR,
    exit_dir: str | Path = DEFAULT_EXIT_DIR,
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    monitoring_dir: str | Path = DEFAULT_MONITORING_DIR,
    v5_closure_dir: str | Path = DEFAULT_V5_CLOSURE_DIR,
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
    schema_validator_dir: str | Path = DEFAULT_SCHEMA_VALIDATOR_DIR,
    dependency_validator_dir: str | Path = DEFAULT_DEPENDENCY_VALIDATOR_DIR,
) -> dict[str, Path]:
    return {
        "semi": Path(semi_auto_dir),
        "broker": Path(broker_research_dir),
        "capital": Path(capital_dir),
        "universe": Path(universe_dir),
        "position": Path(position_dir),
        "exit": Path(exit_dir),
        "daily": Path(daily_plan_dir),
        "paper": Path(paper_ledger_dir),
        "monitoring": Path(monitoring_dir),
        "v5_closure": Path(v5_closure_dir),
        "baseline": Path(baseline_dir),
        "schema_validator": Path(schema_validator_dir),
        "dependency_validator": Path(dependency_validator_dir),
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


def count_forbidden_flags(output_dir: Path) -> int:
    total = 0
    if not output_dir.exists() or not output_dir.is_dir():
        return total
    for csv_path in output_dir.glob("*.csv"):
        frame = _read_csv(csv_path)
        total += sum(_bool_count(frame, flag) for flag in SAFETY_FLAGS)
    for json_path in output_dir.glob("*.json"):
        payload = _read_json(json_path)
        total += sum(_json_flag_count(payload, flag) for flag in SAFETY_FLAGS)
    return int(total)


def normalize_text(text: str, canonical_dir: Path, rerun_dir: Path) -> str:
    normalized = text.replace("\\", "/")
    replacements = [
        canonical_dir.as_posix(),
        str(canonical_dir).replace("\\", "/"),
        rerun_dir.as_posix(),
        str(rerun_dir).replace("\\", "/"),
    ]
    for value in replacements:
        normalized = normalized.replace(value, "<OUTPUT_DIR>")
    lines = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if '"timestamp":' in stripped or stripped.startswith('"output_dir":'):
            continue
        if stripped.startswith("- Output directory:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def fingerprint(path: Path, canonical_dir: Path, rerun_dir: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    normalized = normalize_text(text, canonical_dir, rerun_dir)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def first_summary_row(path: Path) -> dict[str, Any]:
    frame = _read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def compare_summary_keys(canonical_path: Path, rerun_path: Path, keys: list[str]) -> tuple[bool, str]:
    canonical = first_summary_row(canonical_path)
    rerun = first_summary_row(rerun_path)
    mismatches = []
    for key in keys:
        if str(canonical.get(key, "")) != str(rerun.get(key, "")):
            mismatches.append(key)
    return not mismatches, "|".join(mismatches)


def compare_file(
    spec: dict[str, Any],
    canonical_dir: Path,
    rerun_dir: Path,
    file_name: str,
    command_return_code: int,
) -> dict[str, Any]:
    canonical_path = canonical_dir / file_name
    rerun_path = rerun_dir / file_name
    canonical_exists = canonical_path.exists()
    rerun_exists = rerun_path.exists()
    canonical_rows = len(_read_csv(canonical_path)) if file_name.endswith(".csv") else 0
    rerun_rows = len(_read_csv(rerun_path)) if file_name.endswith(".csv") else 0
    row_count_matches = canonical_rows == rerun_rows
    canonical_hash = fingerprint(canonical_path, canonical_dir, rerun_dir)
    rerun_hash = fingerprint(rerun_path, canonical_dir, rerun_dir)
    fingerprint_matches = bool(canonical_hash and canonical_hash == rerun_hash)
    summary_keys_match = True
    summary_mismatches = ""
    if file_name == spec["summary_file"]:
        summary_keys_match, summary_mismatches = compare_summary_keys(
            canonical_path,
            rerun_path,
            list(spec["summary_keys"]),
        )
    forbidden_true_count = count_forbidden_flags(rerun_dir)

    if command_return_code != 0 or not canonical_exists or not rerun_exists or not row_count_matches or not summary_keys_match or forbidden_true_count:
        status = "fail"
    elif file_name != "run_config.json" and canonical_hash and rerun_hash and not fingerprint_matches:
        status = "warning"
    else:
        status = "pass"

    notes = []
    if command_return_code != 0:
        notes.append("rerun_command_failed")
    if not canonical_exists:
        notes.append("canonical_file_missing")
    if not rerun_exists:
        notes.append("rerun_file_missing")
    if not row_count_matches:
        notes.append("row_count_mismatch")
    if summary_mismatches:
        notes.append("summary_mismatch:" + summary_mismatches)
    if forbidden_true_count:
        notes.append("forbidden_true_flag_detected")
    if status == "warning":
        notes.append("normalized_fingerprint_differs_but_structure_matches")
    if not notes:
        notes.append("rerun_output_matches")

    return {
        "rerun_id": spec["rerun_id"],
        "step_name": spec["step_name"],
        "file_name": file_name,
        "canonical_path": str(canonical_path),
        "rerun_path": str(rerun_path),
        "canonical_exists": canonical_exists,
        "rerun_exists": rerun_exists,
        "canonical_row_count": canonical_rows,
        "rerun_row_count": rerun_rows,
        "row_count_matches": row_count_matches,
        "summary_keys_match": summary_keys_match,
        "summary_mismatched_keys": summary_mismatches,
        "normalized_fingerprint_matches": fingerprint_matches,
        "canonical_fingerprint": canonical_hash,
        "rerun_fingerprint": rerun_hash,
        "forbidden_true_flag_count": forbidden_true_count,
        "validation_status": status,
        "notes": "; ".join(notes),
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
    }


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def command_for_spec(spec: dict[str, Any], paths: dict[str, Path], rerun_dirs: dict[str, Path], rerun_dir: Path) -> list[str]:
    script = str(spec["script"])
    if spec["rerun_id"] == "RERUN-001":
        return [sys.executable, script, "--input-dir", str(paths["semi"]), "--output-dir", str(rerun_dir)]
    if spec["rerun_id"] == "RERUN-002":
        return [
            sys.executable,
            script,
            "--capital-dir",
            str(paths["capital"]),
            "--universe-dir",
            str(paths["universe"]),
            "--position-dir",
            str(paths["position"]),
            "--exit-dir",
            str(paths["exit"]),
            "--daily-plan-dir",
            str(paths["daily"]),
            "--paper-ledger-dir",
            str(paths["paper"]),
            "--semi-auto-dir",
            str(paths["semi"]),
            "--broker-research-dir",
            str(rerun_dirs.get("broker", paths["broker"])),
            "--output-dir",
            str(rerun_dir),
        ]
    if spec["rerun_id"] == "RERUN-003":
        return [
            sys.executable,
            script,
            "--capital-dir",
            str(paths["capital"]),
            "--universe-dir",
            str(paths["universe"]),
            "--position-dir",
            str(paths["position"]),
            "--exit-dir",
            str(paths["exit"]),
            "--daily-plan-dir",
            str(paths["daily"]),
            "--paper-ledger-dir",
            str(paths["paper"]),
            "--semi-auto-dir",
            str(paths["semi"]),
            "--broker-research-dir",
            str(rerun_dirs.get("broker", paths["broker"])),
            "--monitoring-dir",
            str(rerun_dirs.get("monitoring", paths["monitoring"])),
            "--output-dir",
            str(rerun_dir),
        ]
    if spec["rerun_id"] == "RERUN-004":
        return [
            sys.executable,
            script,
            "--capital-dir",
            str(paths["capital"]),
            "--universe-dir",
            str(paths["universe"]),
            "--position-dir",
            str(paths["position"]),
            "--exit-dir",
            str(paths["exit"]),
            "--daily-plan-dir",
            str(paths["daily"]),
            "--paper-ledger-dir",
            str(paths["paper"]),
            "--semi-auto-dir",
            str(paths["semi"]),
            "--broker-research-dir",
            str(rerun_dirs.get("broker", paths["broker"])),
            "--monitoring-dir",
            str(rerun_dirs.get("monitoring", paths["monitoring"])),
            "--v5-closure-dir",
            str(rerun_dirs.get("v5_closure", paths["v5_closure"])),
            "--output-dir",
            str(rerun_dir),
        ]
    if spec["rerun_id"] == "RERUN-005":
        return [
            sys.executable,
            script,
            "--capital-dir",
            str(paths["capital"]),
            "--universe-dir",
            str(paths["universe"]),
            "--position-dir",
            str(paths["position"]),
            "--exit-dir",
            str(paths["exit"]),
            "--daily-plan-dir",
            str(paths["daily"]),
            "--paper-ledger-dir",
            str(paths["paper"]),
            "--semi-auto-dir",
            str(paths["semi"]),
            "--broker-research-dir",
            str(rerun_dirs.get("broker", paths["broker"])),
            "--monitoring-dir",
            str(rerun_dirs.get("monitoring", paths["monitoring"])),
            "--v5-closure-dir",
            str(rerun_dirs.get("v5_closure", paths["v5_closure"])),
            "--baseline-dir",
            str(rerun_dirs.get("baseline", paths["baseline"])),
            "--output-dir",
            str(rerun_dir),
        ]
    return [
        sys.executable,
        script,
        "--capital-dir",
        str(paths["capital"]),
        "--universe-dir",
        str(paths["universe"]),
        "--position-dir",
        str(paths["position"]),
        "--exit-dir",
        str(paths["exit"]),
        "--daily-plan-dir",
        str(paths["daily"]),
        "--paper-ledger-dir",
        str(paths["paper"]),
        "--semi-auto-dir",
        str(paths["semi"]),
        "--broker-research-dir",
        str(rerun_dirs.get("broker", paths["broker"])),
        "--monitoring-dir",
        str(rerun_dirs.get("monitoring", paths["monitoring"])),
        "--v5-closure-dir",
        str(rerun_dirs.get("v5_closure", paths["v5_closure"])),
        "--baseline-dir",
        str(rerun_dirs.get("baseline", paths["baseline"])),
        "--schema-validator-dir",
        str(rerun_dirs.get("schema_validator", paths["schema_validator"])),
        "--output-dir",
        str(rerun_dir),
    ]


def prepare_rerun_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for item in workspace.iterdir():
        if item.is_dir():
            shutil.rmtree(item)


def run_reruns(paths: dict[str, Path], output_dir: Path) -> pd.DataFrame:
    workspace = output_dir / "rerun_workspace"
    prepare_rerun_workspace(workspace)
    rows: list[dict[str, Any]] = []
    rerun_dirs: dict[str, Path] = {}
    key_by_rerun = {
        "RERUN-001": "broker",
        "RERUN-002": "monitoring",
        "RERUN-003": "v5_closure",
        "RERUN-004": "baseline",
        "RERUN-005": "schema_validator",
        "RERUN-006": "dependency_validator",
    }
    for spec in RERUN_SPECS:
        rerun_dir = workspace / str(spec["rerun_subdir"])
        rerun_dir.mkdir(parents=True, exist_ok=True)
        command = command_for_spec(spec, paths, rerun_dirs, rerun_dir)
        result = _run_command(command)
        rerun_dirs[key_by_rerun[str(spec["rerun_id"])]] = rerun_dir
        canonical_dir = paths[str(spec["canonical_key"])]
        for file_name in spec["expected_files"]:
            row = compare_file(spec, canonical_dir, rerun_dir, file_name, result.returncode)
            row["command_return_code"] = result.returncode
            row["command"] = " ".join(command)
            row["stdout_tail"] = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            row["stderr_tail"] = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
            rows.append(row)
    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        fail_count = 0
        warning_count = 0
        pass_count = 0
    else:
        grouped = results.groupby("rerun_id")["validation_status"].apply(list)
        fail_count = int(grouped.map(lambda values: "fail" in values).sum())
        warning_count = int(grouped.map(lambda values: "fail" not in values and "warning" in values).sum())
        pass_count = int(grouped.map(lambda values: set(values) == {"pass"}).sum())
    checked_file_count = int(len(results))
    matched_file_count = int(
        (
            results["rerun_exists"].fillna(False).astype(bool)
            & results["row_count_matches"].fillna(False).astype(bool)
            & results["summary_keys_match"].fillna(False).astype(bool)
            & (results["forbidden_true_flag_count"].fillna(0).astype(int) == 0)
        ).sum()
    ) if not results.empty else 0
    mismatched_file_count = checked_file_count - matched_file_count
    forbidden_true_flag_count = int(results["forbidden_true_flag_count"].sum()) if not results.empty else 0
    if fail_count:
        status = "fail"
        conclusion = "reproducibility_rerun_failed_research_only"
    elif warning_count:
        status = "warning"
        conclusion = "reproducibility_rerun_passed_with_warnings_research_only"
    else:
        status = "pass"
        conclusion = "reproducibility_rerun_passed_research_only"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step4_reproducibility_rerun_validation",
                "checked_rerun_count": len(RERUN_SPECS),
                "rerun_pass_count": pass_count,
                "rerun_warning_count": warning_count,
                "rerun_fail_count": fail_count,
                "checked_file_count": checked_file_count,
                "matched_file_count": matched_file_count,
                "mismatched_file_count": mismatched_file_count,
                "forbidden_true_flag_count": forbidden_true_flag_count,
                "trading_ready": False,
                "validation_status": status,
                "conclusion": conclusion,
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Only selected local output-reporting scripts are rerun.", "No historical backtest is run."),
        ("no_market_data_fetch", "confirmed", "Rerun commands use existing local output directories only.", "No market data is fetched."),
        ("no_threshold_change", "confirmed", "No threshold arguments or modules are changed.", "Thresholds are not modified."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local V5/V6 outputs are read.", "No new data source is added."),
        ("no_broker_credentials", "confirmed", "The CLI does not accept credentials and no rerun command requests credentials.", "No credential handling exists."),
        ("no_broker_sdk_import", "confirmed", "The validator imports only standard library modules and pandas.", "No broker SDK is imported."),
        ("no_broker_connection", "confirmed", "Selected rerun steps are research/reporting layers with broker_connected false.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "Selected rerun steps have no live trading path.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "Selected rerun steps have no execution path.", "No orders are executed."),
        ("no_real_order_submission", "confirmed", "Selected rerun steps have no order submission path.", "No real orders are submitted."),
        ("no_trading_ready_upgrade", "confirmed", "The summary writes trading_ready as false.", "No deployable status is claimed."),
        ("isolated_rerun_only", "confirmed", "All reruns write under the Step 4 rerun_workspace.", "Reruns are isolated from canonical outputs."),
        ("previous_outputs_not_overwritten", "confirmed", "Canonical output directories are read-only inputs.", "Historical outputs are not overwritten."),
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
            "# V6 Step 4 Historical Output Reproducibility / Rerun Consistency Check",
            "",
            "## Executive Summary",
            "V6 Step 4 reruns selected deterministic local-only V5/V6 output-producing commands into an isolated rerun workspace and compares their structural outputs against canonical outputs.",
            "It checks expected files, row counts, key summary values, normalized fingerprints, and forbidden safety flags.",
            "It does not run backtests, fetch market data, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or upgrade trading readiness.",
            "",
            "## Summary",
            f"- Checked reruns: {row.get('checked_rerun_count', 0)}",
            f"- Rerun passes: {row.get('rerun_pass_count', 0)}",
            f"- Rerun warnings: {row.get('rerun_warning_count', 0)}",
            f"- Rerun failures: {row.get('rerun_fail_count', 0)}",
            f"- Checked files: {row.get('checked_file_count', 0)}",
            f"- Matched files: {row.get('matched_file_count', 0)}",
            f"- Mismatched files: {row.get('mismatched_file_count', 0)}",
            f"- Forbidden true flags: {row.get('forbidden_true_flag_count', 0)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            "",
            "## Rerun Results",
            _table(results, "No rerun result rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This reproducibility validation report is educational/research-only. It is not financial advice and is not a trading-ready certification.",
            "",
        ]
    )


def generate_reproducibility_rerun_validation_outputs(
    semi_auto_dir: str | Path = DEFAULT_SEMI_AUTO_DIR,
    broker_research_dir: str | Path = DEFAULT_BROKER_RESEARCH_DIR,
    capital_dir: str | Path = DEFAULT_CAPITAL_DIR,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
    position_dir: str | Path = DEFAULT_POSITION_DIR,
    exit_dir: str | Path = DEFAULT_EXIT_DIR,
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    monitoring_dir: str | Path = DEFAULT_MONITORING_DIR,
    v5_closure_dir: str | Path = DEFAULT_V5_CLOSURE_DIR,
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
    schema_validator_dir: str | Path = DEFAULT_SCHEMA_VALIDATOR_DIR,
    dependency_validator_dir: str | Path = DEFAULT_DEPENDENCY_VALIDATOR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        semi_auto_dir=semi_auto_dir,
        broker_research_dir=broker_research_dir,
        capital_dir=capital_dir,
        universe_dir=universe_dir,
        position_dir=position_dir,
        exit_dir=exit_dir,
        daily_plan_dir=daily_plan_dir,
        paper_ledger_dir=paper_ledger_dir,
        monitoring_dir=monitoring_dir,
        v5_closure_dir=v5_closure_dir,
        baseline_dir=baseline_dir,
        schema_validator_dir=schema_validator_dir,
        dependency_validator_dir=dependency_validator_dir,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = run_reruns(paths, output_path)
    summary = build_summary(results)
    guardrails = build_guardrails()
    report = build_report(summary, results, guardrails)

    out_paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(out_paths["summary"], index=False)
    results.to_csv(out_paths["results"], index=False)
    guardrails.to_csv(out_paths["guardrails"], index=False)
    out_paths["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "rerun_workspace": str(output_path / "rerun_workspace"),
        "checked_rerun_count": len(RERUN_SPECS),
        "checked_file_count": int(len(results)),
        "scope": "V6 Step 4 reproducibility rerun validation only",
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
        "reproducibility_rerun_summary": summary,
        "reproducibility_rerun_results": results,
        "reproducibility_rerun_guardrails": guardrails,
        "reproducibility_rerun_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in out_paths.items()},
    }
