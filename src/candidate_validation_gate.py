import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


MISSING_MODE_VALUES = {"", "n/a", "na", "nan", "none", "null", "unknown"}
KNOWN_CANONICAL_MODES = {"full", "canonical_reduced_40", "keep_core_only"}
STRICT_GATE_COLUMNS = [
    "mode_known",
    "role_allowed",
    "expanded_validation_passed",
    "stress_validation_passed",
    "validation_excess_positive",
    "stress_excess_positive",
    "stress_beat_benchmark_rate_passed",
    "stress_sufficient_trade_rate_passed",
    "no_regime_failure",
    "risk_flags_acceptable",
]
BLOCKING_RISK_CATEGORIES = {
    "benchmark_underperformance",
    "low_trade_or_low_confidence",
}


def _read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _is_missing_text(value: Any) -> bool:
    return _clean_text(value).lower() in MISSING_MODE_VALUES


def _numeric_value(row: pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    return pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]


def _markdown_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "_No rows available._"
    table_df = df.head(max_rows)
    headers = [str(column) for column in table_df.columns]

    def clean(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            if pd.isna(value):
                return "n/a"
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table_df.iterrows():
        rows.append("| " + " | ".join(clean(row[column]) for column in headers) + " |")
    if len(df) > max_rows:
        rows.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(rows)


def load_validation_gate_inputs(revalidation_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(revalidation_dir)
    return {
        "summary": _read_csv(base / "canonical_candidate_revalidation_summary.csv"),
        "risk_flags": _read_csv(base / "candidate_risk_flags.csv", dtype={"symbol": str}),
    }


def _risk_counts(risk_flags: pd.DataFrame) -> pd.DataFrame:
    if risk_flags.empty or "canonical_mode" not in risk_flags:
        return pd.DataFrame(
            columns=[
                "canonical_mode",
                "risk_flag_count",
                "benchmark_underperformance_count",
                "low_trade_or_low_confidence_count",
                "research_warning_count",
            ]
        )
    flags = risk_flags.copy()
    flags["canonical_mode"] = flags["canonical_mode"].map(_clean_text)
    if "risk_category" not in flags:
        flags["risk_category"] = "research_warning"
    rows = []
    for mode, group in flags.groupby("canonical_mode", dropna=False):
        categories = group["risk_category"].fillna("research_warning").astype(str)
        rows.append(
            {
                "canonical_mode": mode,
                "risk_flag_count": len(group),
                "benchmark_underperformance_count": int(
                    (categories == "benchmark_underperformance").sum()
                ),
                "low_trade_or_low_confidence_count": int(
                    (categories == "low_trade_or_low_confidence").sum()
                ),
                "research_warning_count": int((categories == "research_warning").sum()),
            }
        )
    return pd.DataFrame(rows)


def _has_regime_failure(reason: Any) -> bool:
    text = _clean_text(reason).lower()
    return any(
        pattern in text
        for pattern in [
            "failed regimes",
            "bull",
            "bear",
            "sideways",
            "regime weakness",
            "stress validation fail due to regime weakness",
        ]
    )


def _decision_for_row(row: dict[str, Any], missing_key_data: bool) -> str:
    mode = row["canonical_mode"]
    if mode == "full":
        return "baseline_only"
    if missing_key_data:
        return "rejected_or_not_tested"
    if all(bool(row[column]) for column in STRICT_GATE_COLUMNS):
        return "trading_ready"
    if mode == "keep_core_only":
        return "low_confidence_challenger"
    if bool(row["expanded_validation_passed"]):
        return "research_only_not_trading_ready"
    return "rejected_or_not_tested"


def _decision_reason(row: dict[str, Any], failed_checks: list[str]) -> str:
    mode = row["canonical_mode"]
    if row["final_gate_decision"] == "trading_ready":
        return "All strict validation gate checks passed."
    if mode == "full":
        return "full is retained only as baseline and cannot become the default trading candidate."
    if mode == "canonical_reduced_40" and not row["stress_validation_passed"]:
        return "canonical_reduced_40 remains the primary research candidate but is blocked by stress validation failure."
    if mode == "keep_core_only":
        return "keep_core_only remains a low-feature challenger with instability or low-trade-count risk."
    if row["final_gate_decision"] == "rejected_or_not_tested":
        return "Key validation or stress data is missing or did not pass."
    return "Strict validation gate checks failed: " + ", ".join(failed_checks)


def build_validation_gate_results(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = inputs["summary"]
    risk_counts = _risk_counts(inputs["risk_flags"])
    if summary.empty:
        return pd.DataFrame()
    risk_by_mode = (
        risk_counts.set_index("canonical_mode").to_dict("index")
        if not risk_counts.empty
        else {}
    )
    rows = []
    for _, source_row in summary.iterrows():
        mode = _clean_text(source_row.get("canonical_mode"))
        risk = risk_by_mode.get(mode, {})
        validation_decision = _clean_text(source_row.get("validation_decision")).lower()
        stress_decision = _clean_text(source_row.get("stress_decision")).lower()
        validation_excess = _numeric_value(source_row, "avg_validation_excess_pct")
        stress_excess = _numeric_value(source_row, "avg_stress_excess_pct")
        beat_rate = _numeric_value(source_row, "stress_beat_benchmark_rate")
        trade_rate = _numeric_value(source_row, "stress_sufficient_trade_rate")
        role = _clean_text(source_row.get("role"))
        blocking_risk_count = int(risk.get("benchmark_underperformance_count", 0)) + int(
            risk.get("low_trade_or_low_confidence_count", 0)
        )
        row = {
            "canonical_mode": mode,
            "role": role,
            "validation_decision": validation_decision or "not_tested",
            "stress_decision": stress_decision or "not_tested",
            "avg_validation_excess_pct": validation_excess,
            "avg_stress_excess_pct": stress_excess,
            "stress_beat_benchmark_rate": beat_rate,
            "stress_sufficient_trade_rate": trade_rate,
            "risk_flag_count": int(risk.get("risk_flag_count", 0)),
            "benchmark_underperformance_count": int(
                risk.get("benchmark_underperformance_count", 0)
            ),
            "low_trade_or_low_confidence_count": int(
                risk.get("low_trade_or_low_confidence_count", 0)
            ),
            "research_warning_count": int(risk.get("research_warning_count", 0)),
            "mode_known": bool(
                mode in KNOWN_CANONICAL_MODES and not _is_missing_text(mode)
            ),
            "role_allowed": mode != "full",
            "expanded_validation_passed": validation_decision == "pass",
            "stress_validation_passed": stress_decision == "pass",
            "validation_excess_positive": bool(pd.notna(validation_excess) and validation_excess > 0),
            "stress_excess_positive": bool(pd.notna(stress_excess) and stress_excess > 0),
            "stress_beat_benchmark_rate_passed": bool(pd.notna(beat_rate) and beat_rate >= 0.60),
            "stress_sufficient_trade_rate_passed": bool(pd.notna(trade_rate) and trade_rate >= 0.80),
            "no_regime_failure": not _has_regime_failure(source_row.get("decision_reason")),
            "risk_flags_acceptable": blocking_risk_count == 0,
        }
        missing_key_data = (
            row["validation_decision"] in MISSING_MODE_VALUES
            or row["validation_decision"] == "not_tested"
            or row["stress_decision"] in MISSING_MODE_VALUES
            or row["stress_decision"] == "not_tested"
            or pd.isna(validation_excess)
            or pd.isna(stress_excess)
            or pd.isna(beat_rate)
            or pd.isna(trade_rate)
        )
        failed_checks = [
            column for column in STRICT_GATE_COLUMNS if not bool(row[column])
        ]
        row["strict_gates_passed"] = len(failed_checks) == 0
        row["final_gate_decision"] = _decision_for_row(row, missing_key_data)
        row["trading_ready"] = bool(
            row["final_gate_decision"] == "trading_ready"
            and row["strict_gates_passed"]
        )
        row["gate_reason"] = _decision_reason(row, failed_checks)
        rows.append(row)
    return pd.DataFrame(rows)


def build_validation_gate_failures(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for _, row in results.iterrows():
        for check_name in STRICT_GATE_COLUMNS:
            if bool(row[check_name]):
                continue
            rows.append(
                {
                    "canonical_mode": row["canonical_mode"],
                    "check_name": check_name,
                    "severity": "blocking",
                    "message": _failure_message(row, check_name),
                }
            )
    return pd.DataFrame(rows)


def _failure_message(row: pd.Series, check_name: str) -> str:
    mode = row["canonical_mode"]
    messages = {
        "mode_known": f"{mode} is blank, n/a, unknown, or not in the canonical mode set.",
        "role_allowed": f"{mode} is not allowed to become the default trading candidate.",
        "expanded_validation_passed": f"{mode} expanded validation decision is {row['validation_decision']}.",
        "stress_validation_passed": f"{mode} stress validation decision is {row['stress_decision']}.",
        "validation_excess_positive": f"{mode} validation excess return is not positive.",
        "stress_excess_positive": f"{mode} stress excess return is not positive.",
        "stress_beat_benchmark_rate_passed": f"{mode} stress beat benchmark rate is below 0.60.",
        "stress_sufficient_trade_rate_passed": f"{mode} sufficient trade rate is below 0.80.",
        "no_regime_failure": f"{mode} decision reason references regime weakness.",
        "risk_flags_acceptable": f"{mode} has benchmark underperformance or low-trade/low-confidence risk flags.",
    }
    return messages[check_name]


def generate_validation_gate_report(
    results: pd.DataFrame,
    failures: pd.DataFrame,
    risk_flags: pd.DataFrame,
) -> str:
    trading_ready_count = (
        int((results["final_gate_decision"] == "trading_ready").sum())
        if not results.empty
        else 0
    )
    readiness_line = (
        "No candidate is currently trading-ready; all current candidates remain not trading-ready unless every strict gate passes."
        if trading_ready_count == 0
        else f"{trading_ready_count} candidate(s) passed every strict gate."
    )
    sections = [
        "# Candidate Validation Gate Report",
        "",
        "## Validation Gate Summary",
        "This is educational/research diagnostics only. It is not financial advice.",
        readiness_line,
        (
            "canonical_reduced_40 remains the primary research candidate but is blocked "
            "by stress validation failure when stress validation does not pass."
        ),
        "full is retained only as baseline.",
        (
            "keep_core_only remains a low-feature challenger with instability / "
            "low-trade-count risk unless every strict rule passes."
        ),
        (
            "Do not add new features or agents until the validation gate is stable "
            "and the candidate passes stricter validation."
        ),
        "",
        "## Gate Results",
        _markdown_table(results),
        "",
        "## Gate Failures",
        _markdown_table(failures),
        "",
        "## Candidate Risk Flags",
        _markdown_table(risk_flags),
        "",
    ]
    return "\n".join(sections)


def build_candidate_validation_gate(revalidation_dir: str | Path) -> dict[str, Any]:
    inputs = load_validation_gate_inputs(revalidation_dir)
    results = build_validation_gate_results(inputs)
    failures = build_validation_gate_failures(results)
    report = generate_validation_gate_report(results, failures, inputs["risk_flags"])
    return {
        "validation_gate_results": results,
        "validation_gate_failures": failures,
        "candidate_validation_gate_report": report,
    }


def save_candidate_validation_gate(
    revalidation_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    result = build_candidate_validation_gate(revalidation_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_path / "validation_gate_results.csv",
        "failures": output_path / "validation_gate_failures.csv",
        "report": output_path / "candidate_validation_gate_report.md",
        "run_config": output_path / "run_config.json",
    }
    result["validation_gate_results"].to_csv(paths["results"], index=False)
    result["validation_gate_failures"].to_csv(paths["failures"], index=False)
    paths["report"].write_text(
        result["candidate_validation_gate_report"],
        encoding="utf-8",
    )
    run_config = {
        "revalidation_dir": str(revalidation_dir),
        "output_dir": str(output_path),
        "strict_gate_columns": STRICT_GATE_COLUMNS,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["run_config"] = run_config
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result
