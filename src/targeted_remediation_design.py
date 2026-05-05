import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PRIMARY_CANDIDATE = "canonical_reduced_40"
CANONICAL_MODES = [PRIMARY_CANDIDATE, "full", "keep_core_only"]


def _read_csv(
    path: Path,
    warnings: list[str],
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing optional input file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        warnings.append(f"Empty optional input file: {path}")
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


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean_text(value).lower() in {"true", "1", "yes", "y"}


def _numeric_value(row: pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    return pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]


def _safe_int(value: Any, default: int = 0) -> int:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return default
    return int(number)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _format_symbol(value: Any) -> str:
    text = _clean_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


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


def load_targeted_remediation_inputs(
    failure_analysis_dir: str | Path,
    gate_dir: str | Path,
    revalidation_dir: str | Path,
) -> dict[str, Any]:
    warnings: list[str] = []
    failure_base = Path(failure_analysis_dir)
    gate_base = Path(gate_dir)
    revalidation_base = Path(revalidation_dir)
    inputs = {
        "failure_by_regime": _read_csv(failure_base / "failure_by_regime.csv", warnings),
        "failure_by_candidate": _read_csv(failure_base / "failure_by_candidate.csv", warnings),
        "failure_by_check": _read_csv(failure_base / "failure_by_check.csv", warnings),
        "failure_by_symbol": _read_csv(
            failure_base / "failure_by_symbol.csv",
            warnings,
            dtype={"symbol": str},
        ),
        "risk_flag_summary": _read_csv(
            failure_base / "risk_flag_summary.csv",
            warnings,
            dtype={"symbol": str},
        ),
        "remediation_plan": _read_csv(failure_base / "remediation_plan.csv", warnings),
        "validation_gate_results": _read_csv(gate_base / "validation_gate_results.csv", warnings),
        "validation_gate_failures": _read_csv(gate_base / "validation_gate_failures.csv", warnings),
        "revalidation_summary": _read_csv(
            revalidation_base / "canonical_candidate_revalidation_summary.csv",
            warnings,
        ),
        "candidate_risk_flags": _read_csv(
            revalidation_base / "candidate_risk_flags.csv",
            warnings,
            dtype={"symbol": str},
        ),
        "input_warnings": warnings,
    }
    for key in ["failure_by_symbol", "risk_flag_summary", "candidate_risk_flags"]:
        df = inputs[key]
        if not df.empty and "symbol" in df:
            result = df.copy()
            result["symbol"] = result["symbol"].map(_format_symbol)
            inputs[key] = result
    return inputs


def build_regime_remediation_plan(failure_by_regime: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "canonical_mode",
        "regime",
        "avg_strategy_vs_benchmark_pct",
        "beat_benchmark_rate",
        "sufficient_trade_rate",
        "regime_gate_failed",
        "has_regime_warnings",
        "remediation_priority",
        "recommended_action",
    ]
    if failure_by_regime.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in failure_by_regime.iterrows():
        mode = _clean_text(row.get("canonical_mode"))
        regime = _clean_text(row.get("regime")).lower()
        avg_excess = _numeric_value(row, "avg_strategy_vs_benchmark_pct")
        beat_rate = _numeric_value(row, "beat_benchmark_rate")
        trade_rate = _numeric_value(row, "sufficient_trade_rate")
        gate_failed = _to_bool(row.get("regime_gate_failed"))
        has_warnings = _to_bool(row.get("has_regime_warnings"))
        if mode == PRIMARY_CANDIDATE and regime == "bull" and gate_failed:
            priority = "P0"
            action = "Prioritize bull regime threshold sensitivity because average strategy-vs-benchmark excess return is negative or below gate."
        elif mode == PRIMARY_CANDIDATE and regime == "sideways" and gate_failed:
            priority = "P1"
            action = "Prioritize sideways trade-frequency remediation because beat benchmark rate or sufficient trade rate is weak."
        elif regime == "bear" and not gate_failed and has_warnings:
            priority = "monitor"
            action = "Monitor bear warnings; do not classify bear as a primary failed aggregate regime gate."
        elif gate_failed:
            priority = "P1"
            action = "Remediate aggregate regime gate weakness before any trading-ready claim."
        elif has_warnings:
            priority = "P2"
            action = "Track warning rows while preserving aggregate pass/fail interpretation."
        else:
            priority = "monitor"
            action = "No immediate aggregate regime remediation required."
        rows.append(
            {
                "canonical_mode": mode,
                "regime": regime,
                "avg_strategy_vs_benchmark_pct": avg_excess,
                "beat_benchmark_rate": beat_rate,
                "sufficient_trade_rate": trade_rate,
                "regime_gate_failed": gate_failed,
                "has_regime_warnings": has_warnings,
                "remediation_priority": priority,
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_candidate_remediation_plan(
    failure_by_candidate: pd.DataFrame,
    validation_gate_results: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "canonical_mode",
        "role",
        "final_gate_decision",
        "trading_ready",
        "failure_count",
        "failed_checks",
        "remediation_role",
        "recommended_action",
    ]
    gate_by_mode = (
        validation_gate_results.set_index("canonical_mode").to_dict("index")
        if not validation_gate_results.empty and "canonical_mode" in validation_gate_results
        else {}
    )
    failure_by_mode = (
        failure_by_candidate.set_index("canonical_mode").to_dict("index")
        if not failure_by_candidate.empty and "canonical_mode" in failure_by_candidate
        else {}
    )
    modes = list(dict.fromkeys(CANONICAL_MODES + list(gate_by_mode) + list(failure_by_mode)))
    rows = []
    for mode in modes:
        gate = gate_by_mode.get(mode, {})
        failure = failure_by_mode.get(mode, {})
        decision = gate.get("final_gate_decision", failure.get("final_gate_decision", "not_available"))
        trading_ready = bool(_to_bool(gate.get("trading_ready", failure.get("trading_ready", False))))
        if mode == PRIMARY_CANDIDATE:
            role = "primary_research_candidate"
            remediation_role = "primary_remediation_candidate"
            action = "Target bull/sideways robustness, benchmark beat rate, sufficient trade rate, and risk-flag reduction."
        elif mode == "full":
            role = "baseline_only"
            remediation_role = "baseline_monitoring"
            action = "Keep full as baseline comparison only; do not remediate it as the default trading candidate."
        elif mode == "keep_core_only":
            role = "low_feature_challenger"
            remediation_role = "challenger_only"
            action = "Run low-trade-count challenger validation only after primary remediation gates stabilize."
        else:
            role = _clean_text(gate.get("role"))
            remediation_role = "diagnostic_only"
            action = "Review candidate before assigning remediation priority."
        rows.append(
            {
                "canonical_mode": mode,
                "role": role,
                "final_gate_decision": decision,
                "trading_ready": trading_ready,
                "failure_count": _safe_int(failure.get("failure_count", 0)),
                "failed_checks": failure.get("failed_checks", ""),
                "remediation_role": remediation_role,
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_symbol_remediation_priority(
    failure_by_symbol: pd.DataFrame,
    risk_flag_summary: pd.DataFrame,
    candidate_risk_flags: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "canonical_mode",
        "symbol",
        "stress_failure_count",
        "warning_count",
        "benchmark_underperformance_count",
        "low_trade_warning_count",
        "regime_warning_count",
        "priority_score",
        "recommended_action",
    ]
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    def ensure(mode: str, symbol: str) -> dict[str, Any]:
        key = (mode, _format_symbol(symbol))
        if key not in rows:
            rows[key] = {
                "canonical_mode": key[0],
                "symbol": key[1],
                "stress_failure_count": 0,
                "warning_count": 0,
                "benchmark_underperformance_count": 0,
                "low_trade_warning_count": 0,
                "regime_warning_count": 0,
            }
        return rows[key]

    if not failure_by_symbol.empty and {"canonical_mode", "symbol"}.issubset(failure_by_symbol.columns):
        for _, row in failure_by_symbol.iterrows():
            item = ensure(_clean_text(row.get("canonical_mode")), row.get("symbol"))
            item["stress_failure_count"] += _safe_int(row.get("stress_failure_count", 0))
            item["warning_count"] += _safe_int(row.get("warning_count", 0))

    if not risk_flag_summary.empty and {"canonical_mode", "symbol"}.issubset(risk_flag_summary.columns):
        for _, row in risk_flag_summary.iterrows():
            item = ensure(_clean_text(row.get("canonical_mode")), row.get("symbol"))
            issue_count = _safe_int(row.get("issue_count", 0))
            risk_category = _clean_text(row.get("risk_category")).lower()
            warning_type = _clean_text(row.get("warning_type")).lower()
            item["warning_count"] += issue_count
            if "benchmark" in risk_category or "benchmark" in warning_type:
                item["benchmark_underperformance_count"] += issue_count
            if "low_trade" in risk_category or "low_confidence" in risk_category or "low" in warning_type:
                item["low_trade_warning_count"] += issue_count

    if not candidate_risk_flags.empty and {"canonical_mode", "symbol"}.issubset(candidate_risk_flags.columns):
        for _, row in candidate_risk_flags.iterrows():
            item = ensure(_clean_text(row.get("canonical_mode")), row.get("symbol"))
            text = " ".join(
                _clean_text(row.get(column)).lower()
                for column in ["risk_category", "warning_type", "message", "reason"]
            )
            if "benchmark" in text or "underperform" in text:
                item["benchmark_underperformance_count"] += 1
            if "low_trade" in text or "low-confidence" in text or "low_confidence" in text:
                item["low_trade_warning_count"] += 1

    result_rows = []
    for item in rows.values():
        item["regime_warning_count"] = item["stress_failure_count"]
        score = (
            item["stress_failure_count"] * 3
            + item["benchmark_underperformance_count"] * 2
            + item["low_trade_warning_count"] * 2
            + item["warning_count"]
        )
        item["priority_score"] = score
        item["recommended_action"] = (
            "Prioritize symbol in bull/sideways remediation diagnostics."
            if score > 0
            else "Monitor only."
        )
        result_rows.append(item)
    if not result_rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(result_rows, columns=columns)
        .sort_values(["priority_score", "stress_failure_count", "symbol"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def build_targeted_remediation_experiments(
    regime_plan: pd.DataFrame,
    candidate_plan: pd.DataFrame,
    symbol_priority: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "priority",
        "target_candidate",
        "target_regime",
        "target_failure",
        "hypothesis",
        "experiment_type",
        "suggested_config",
        "expected_improvement",
        "required_success_criteria",
        "risk",
        "should_run_next",
    ]
    rows = [
        {
            "experiment_id": "TRD-001",
            "priority": "P0",
            "target_candidate": PRIMARY_CANDIDATE,
            "target_regime": "bull",
            "target_failure": "negative average strategy-vs-benchmark excess return",
            "hypothesis": "A tighter threshold grid can reduce bull-regime underperformance without adding features.",
            "experiment_type": "regime_specific_threshold_test",
            "suggested_config": "Reuse existing canonical_reduced_40 factors and models; evaluate existing buy/sell threshold grid by bull regime.",
            "expected_improvement": "Bull avg_strategy_vs_benchmark_pct becomes positive while preserving stress gates.",
            "required_success_criteria": "Bull regime_gate_failed=False and total stress validation passes.",
            "risk": "Threshold tuning can overfit historical regimes.",
            "should_run_next": True,
        },
        {
            "experiment_id": "TRD-002",
            "priority": "P0",
            "target_candidate": PRIMARY_CANDIDATE,
            "target_regime": "sideways",
            "target_failure": "low beat benchmark rate and low sufficient trade rate",
            "hypothesis": "Sideways-specific threshold refinement may improve trade sufficiency without adding new signals.",
            "experiment_type": "trade_count_sufficiency_test",
            "suggested_config": "Reuse current candidate configuration; compare threshold combinations against sideways sufficient_trade_rate >= 0.80.",
            "expected_improvement": "Sideways beat_benchmark_rate >= 0.60 and sufficient_trade_rate >= 0.80.",
            "required_success_criteria": "Sideways regime_gate_failed=False with sufficient trades.",
            "risk": "Increasing trade count can worsen benchmark underperformance.",
            "should_run_next": True,
        },
        {
            "experiment_id": "TRD-003",
            "priority": "P1",
            "target_candidate": PRIMARY_CANDIDATE,
            "target_regime": "all",
            "target_failure": "benchmark underperformance",
            "hypothesis": "Existing threshold choices may be too permissive for symbols with repeated benchmark underperformance.",
            "experiment_type": "benchmark_comparison_test",
            "suggested_config": "Rank existing symbols by failure count and retest canonical_reduced_40 thresholds on the highest-priority symbols.",
            "expected_improvement": "Benchmark underperformance warnings materially reduced.",
            "required_success_criteria": "Stress avg excess > 0 and beat benchmark rate >= 0.60.",
            "risk": "Symbol-level improvements may not generalize.",
            "should_run_next": True,
        },
        {
            "experiment_id": "TRD-004",
            "priority": "P1",
            "target_candidate": PRIMARY_CANDIDATE,
            "target_regime": "all",
            "target_failure": "risk flags acceptable failed",
            "hypothesis": "Concentrating on recurring warning categories can reduce gate-blocking risk flags.",
            "experiment_type": "risk_flag_reduction",
            "suggested_config": "Audit benchmark_underperformance and low_trade_or_low_confidence rows from existing reports; no new features or agents.",
            "expected_improvement": "Fewer severe risk flags in candidate_risk_flags.csv.",
            "required_success_criteria": "risk_flags_acceptable=True or materially fewer blocking risk flags.",
            "risk": "Suppressing warnings would hide risk; warnings must remain visible.",
            "should_run_next": True,
        },
        {
            "experiment_id": "TRD-005",
            "priority": "P2",
            "target_candidate": "keep_core_only",
            "target_regime": "all",
            "target_failure": "low-trade-count challenger instability",
            "hypothesis": "The challenger can be evaluated for trade sufficiency without becoming the default candidate.",
            "experiment_type": "challenger_validation",
            "suggested_config": "Run existing keep_core_only validation only after primary candidate remediation stabilizes.",
            "expected_improvement": "Clear challenger status with sufficient-trade evidence or rejection.",
            "required_success_criteria": "keep_core_only cannot become default unless trade count sufficiency passes every strict gate.",
            "risk": "Low feature count may produce too few trades.",
            "should_run_next": False,
        },
        {
            "experiment_id": "TRD-006",
            "priority": "monitor",
            "target_candidate": "full",
            "target_regime": "all",
            "target_failure": "baseline only",
            "hypothesis": "Full remains useful as a comparison baseline, not a default trading candidate.",
            "experiment_type": "baseline_monitoring",
            "suggested_config": "Retain existing full baseline comparison in reports.",
            "expected_improvement": "Stable reference point for reduced candidate diagnostics.",
            "required_success_criteria": "full remains baseline_only.",
            "risk": "Treating baseline as default would violate the validation gate role constraint.",
            "should_run_next": False,
        },
    ]
    if not regime_plan.empty:
        bear_rows = regime_plan[
            (regime_plan["target_regime"] if "target_regime" in regime_plan else regime_plan["regime"]) == "bear"
        ] if "regime" in regime_plan else pd.DataFrame()
        if not bear_rows.empty:
            bear_failed = bear_rows["regime_gate_failed"].map(_to_bool).any()
            bear_warning = bear_rows["has_regime_warnings"].map(_to_bool).any()
            if bear_warning and not bear_failed:
                rows.append(
                    {
                        "experiment_id": "TRD-007",
                        "priority": "monitor",
                        "target_candidate": PRIMARY_CANDIDATE,
                        "target_regime": "bear",
                        "target_failure": "warnings without aggregate regime gate failure",
                        "hypothesis": "Bear warnings should be monitored without labeling bear as a failed aggregate regime.",
                        "experiment_type": "risk_flag_reduction",
                        "suggested_config": "Review warning rows only; do not prioritize bear aggregate remediation unless thresholds fail.",
                        "expected_improvement": "Clear separation between warning presence and regime gate failure.",
                        "required_success_criteria": "bear regime_gate_failed remains False while warnings are tracked.",
                        "risk": "Overreacting to warnings could distract from bull/sideways blockers.",
                        "should_run_next": False,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def build_remediation_success_criteria() -> pd.DataFrame:
    rows = [
        ("stress_validation_passed", "Stress validation must pass.", "stress_decision == pass"),
        ("avg_stress_excess_positive", "Average stress excess return must be positive.", "avg_stress_excess_pct > 0"),
        ("beat_benchmark_rate", "Beat benchmark rate must meet the strict gate.", "stress_beat_benchmark_rate >= 0.60"),
        ("sufficient_trade_rate", "Sufficient trade rate must meet the strict gate.", "stress_sufficient_trade_rate >= 0.80"),
        ("no_failed_bull_sideways", "No failed bull or sideways aggregate regime gate.", "bull/sideways regime_gate_failed == False"),
        ("trading_ready_guardrail", "No trading_ready=True unless all strict gates pass.", "final_gate_decision == trading_ready and strict_gates_passed == True"),
        ("benchmark_warning_reduction", "Benchmark underperformance warnings materially reduced.", "benchmark_underperformance_count decreases"),
        ("low_trade_warning_reduction", "Low-trade warnings materially reduced.", "low_trade_warning_count decreases"),
        ("symbol_coverage", "At least 5 symbols tested.", "tested_symbol_count >= 5"),
        ("keep_core_only_guardrail", "keep_core_only cannot become default unless trade count sufficiency passes.", "keep_core_only sufficient_trade_rate >= 0.80 and all strict gates pass"),
    ]
    return pd.DataFrame(
        [
            {
                "criterion_id": criterion_id,
                "required_success_criteria": description,
                "measurement": measurement,
            }
            for criterion_id, description, measurement in rows
        ]
    )


def generate_targeted_remediation_design_report(
    experiments: pd.DataFrame,
    regime_plan: pd.DataFrame,
    candidate_plan: pd.DataFrame,
    symbol_priority: pd.DataFrame,
    success_criteria: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> str:
    sections = [
        "# Targeted Remediation Experiment Design",
        "",
        "## Executive Summary",
        "This is educational/research diagnostics only, not financial advice.",
        "No candidate is trading-ready; the current strategy remains not trading-ready.",
        "canonical_reduced_40 remains the primary research candidate, not a trading candidate.",
        "full remains baseline only.",
        "keep_core_only remains challenger only with instability / low-trade-count risk.",
        "do not add new features or agents yet.",
        "Next experiments should target bull and sideways robustness and sufficient trade rate.",
        "",
        "## Targeted Remediation Experiments",
        _markdown_table(experiments),
        "",
        "## Regime Remediation Plan",
        _markdown_table(regime_plan),
        "",
        "## Candidate Remediation Plan",
        _markdown_table(candidate_plan),
        "",
        "## Symbol Remediation Priority",
        _markdown_table(symbol_priority),
        "",
        "## Remediation Success Criteria",
        _markdown_table(success_criteria),
        "",
        "## Warnings",
        _markdown_table(warnings_df),
        "",
    ]
    return "\n".join(sections)


def build_targeted_remediation_design(
    failure_analysis_dir: str | Path,
    gate_dir: str | Path,
    revalidation_dir: str | Path,
) -> dict[str, Any]:
    inputs = load_targeted_remediation_inputs(
        failure_analysis_dir,
        gate_dir,
        revalidation_dir,
    )
    warnings_df = pd.DataFrame(
        [{"warning": warning} for warning in inputs["input_warnings"]]
    )
    if warnings_df.empty:
        warnings_df = pd.DataFrame(columns=["warning"])
    regime_plan = build_regime_remediation_plan(inputs["failure_by_regime"])
    candidate_plan = build_candidate_remediation_plan(
        inputs["failure_by_candidate"],
        inputs["validation_gate_results"],
    )
    symbol_priority = build_symbol_remediation_priority(
        inputs["failure_by_symbol"],
        inputs["risk_flag_summary"],
        inputs["candidate_risk_flags"],
    )
    experiments = build_targeted_remediation_experiments(
        regime_plan,
        candidate_plan,
        symbol_priority,
    )
    success_criteria = build_remediation_success_criteria()
    report = generate_targeted_remediation_design_report(
        experiments,
        regime_plan,
        candidate_plan,
        symbol_priority,
        success_criteria,
        warnings_df,
    )
    return {
        "targeted_remediation_experiments": experiments,
        "regime_remediation_plan": regime_plan,
        "candidate_remediation_plan": candidate_plan,
        "symbol_remediation_priority": symbol_priority,
        "remediation_success_criteria": success_criteria,
        "targeted_remediation_design_report": report,
        "warnings": warnings_df,
    }


def save_targeted_remediation_design(
    failure_analysis_dir: str | Path,
    gate_dir: str | Path,
    revalidation_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    result = build_targeted_remediation_design(
        failure_analysis_dir,
        gate_dir,
        revalidation_dir,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "targeted_remediation_experiments": output_path / "targeted_remediation_experiments.csv",
        "regime_remediation_plan": output_path / "regime_remediation_plan.csv",
        "candidate_remediation_plan": output_path / "candidate_remediation_plan.csv",
        "symbol_remediation_priority": output_path / "symbol_remediation_priority.csv",
        "remediation_success_criteria": output_path / "remediation_success_criteria.csv",
        "report": output_path / "targeted_remediation_design_report.md",
        "warnings": output_path / "warnings.csv",
        "run_config": output_path / "run_config.json",
    }
    for key in [
        "targeted_remediation_experiments",
        "regime_remediation_plan",
        "candidate_remediation_plan",
        "symbol_remediation_priority",
        "remediation_success_criteria",
        "warnings",
    ]:
        result[key].to_csv(paths[key], index=False)
    paths["report"].write_text(
        result["targeted_remediation_design_report"],
        encoding="utf-8",
    )
    run_config = {
        "failure_analysis_dir": str(failure_analysis_dir),
        "gate_dir": str(gate_dir),
        "revalidation_dir": str(revalidation_dir),
        "output_dir": str(output_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["run_config"] = run_config
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result
