import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_MODE = "canonical_reduced_40"
MODEL_TYPE = "logistic_regression"
MIN_TRADES = 3
OUTPUT_FILENAMES = {
    "report": "integrated_remediation_revalidation_report.md",
    "summary": "integrated_remediation_summary.csv",
    "regime_status": "regime_remediation_status.csv",
    "per_symbol_risk": "per_symbol_remediation_risk.csv",
    "gate_results": "integrated_gate_results.csv",
    "risk_flags": "integrated_risk_flags.csv",
    "run_config": "run_config.json",
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _format_symbol(value: Any) -> str:
    text = _clean_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _read_required_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Required input file is empty: {path}") from exc


def _read_optional_csv(
    path: Path,
    warnings: list[dict[str, Any]],
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        warnings.append(
            {
                "source": str(path),
                "warning_type": "missing_optional_input",
                "message": f"Optional input file not found: {path}",
            }
        )
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        warnings.append(
            {
                "source": str(path),
                "warning_type": "empty_optional_input",
                "message": f"Optional input file is empty: {path}",
            }
        )
        return pd.DataFrame()


def _read_optional_text(path: Path, warnings: list[dict[str, Any]]) -> str:
    if not path.exists():
        warnings.append(
            {
                "source": str(path),
                "warning_type": "missing_optional_input",
                "message": f"Optional input file not found: {path}",
            }
        )
        return ""
    return path.read_text(encoding="utf-8")


def _numeric(row: pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    return pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(value).lower()
    return text in {"true", "1", "yes", "y"}


def _normalize_symbols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "symbol" not in df:
        return df.copy()
    result = df.copy()
    result["symbol"] = result["symbol"].map(_format_symbol)
    return result


def _first_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    return df.iloc[0]


def _warning_has(row: pd.Series, pattern: str) -> bool:
    return pattern in _clean_text(row.get("warning")).lower()


def load_integrated_inputs(
    bull_dir: str | Path,
    sideways_dir: str | Path,
    validation_gate_dir: str | Path | None = None,
    failure_analysis_dir: str | Path | None = None,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    bull = Path(bull_dir)
    sideways = Path(sideways_dir)
    inputs = {
        "bull_summary": _read_required_csv(bull / "bull_threshold_summary.csv"),
        "best_bull": _read_required_csv(bull / "best_bull_thresholds.csv"),
        "per_symbol_bull": _normalize_symbols(
            _read_required_csv(
                bull / "per_symbol_bull_results.csv",
                dtype={"symbol": str},
            )
        ),
        "bull_warnings": _normalize_symbols(
            _read_optional_csv(bull / "warnings.csv", warnings, dtype={"symbol": str})
        ),
        "sideways_summary": _read_required_csv(sideways / "sideways_trade_summary.csv"),
        "best_sideways": _read_required_csv(sideways / "best_sideways_thresholds.csv"),
        "per_symbol_sideways": _normalize_symbols(
            _read_required_csv(
                sideways / "per_symbol_sideways_results.csv",
                dtype={"symbol": str},
            )
        ),
        "sideways_warnings": _normalize_symbols(
            _read_optional_csv(sideways / "warnings.csv", warnings, dtype={"symbol": str})
        ),
        "validation_gate_results": pd.DataFrame(),
        "validation_gate_failures": pd.DataFrame(),
        "candidate_validation_gate_report": "",
        "failure_by_regime": pd.DataFrame(),
        "validation_gate_failure_analysis_report": "",
        "input_warnings": warnings,
    }
    if validation_gate_dir:
        gate = Path(validation_gate_dir)
        inputs["validation_gate_results"] = _read_optional_csv(
            gate / "validation_gate_results.csv",
            warnings,
        )
        inputs["validation_gate_failures"] = _read_optional_csv(
            gate / "validation_gate_failures.csv",
            warnings,
        )
        inputs["candidate_validation_gate_report"] = _read_optional_text(
            gate / "candidate_validation_gate_report.md",
            warnings,
        )
    if failure_analysis_dir:
        failure = Path(failure_analysis_dir)
        inputs["failure_by_regime"] = _read_optional_csv(
            failure / "failure_by_regime.csv",
            warnings,
        )
        inputs["validation_gate_failure_analysis_report"] = _read_optional_text(
            failure / "validation_gate_failure_analysis_report.md",
            warnings,
        )
    return inputs


def build_regime_remediation_status(inputs: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for regime, best_key, pass_column in [
        ("bull", "best_bull", "bull_gate_passed"),
        ("sideways", "best_sideways", "sideways_gate_passed"),
    ]:
        row = _first_row(inputs[best_key])
        passed = _bool_value(row.get(pass_column))
        final_decision = _clean_text(row.get("final_decision"))
        if regime == "bull":
            unresolved = True
            interpretation = (
                "close_to_passing_but_failed_due_to_negative_average_excess_return"
                if final_decision == "bull_remediation_failed"
                else "bull_research_gate_passed_but_still_not_trading_ready"
            )
        else:
            unresolved = False
            interpretation = (
                "aggregate_sideways_gate_passed_but_remaining_per_symbol_weaknesses_must_be_reviewed"
                if passed
                else "sideways_research_gate_failed"
            )
        rows.append(
            {
                "regime": regime,
                "candidate": _clean_text(row.get("canonical_mode")) or CANONICAL_MODE,
                "model": _clean_text(row.get("model_type")) or MODEL_TYPE,
                "buy_threshold": _numeric(row, "buy_threshold"),
                "sell_threshold": _numeric(row, "sell_threshold"),
                "avg_total_return_pct": _numeric(row, "avg_total_return_pct"),
                "avg_benchmark_return_pct": _numeric(row, "avg_benchmark_return_pct"),
                "avg_strategy_vs_benchmark_pct": _numeric(
                    row,
                    "avg_strategy_vs_benchmark_pct",
                ),
                "beat_benchmark_rate": _numeric(row, "beat_benchmark_rate"),
                "sufficient_trade_rate": _numeric(row, "sufficient_trade_rate"),
                "tested_symbol_count": _numeric(row, "tested_symbol_count"),
                "final_decision": final_decision,
                "passed_configured_research_gate": passed,
                "unresolved_blocker": unresolved,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def _risk_flags_for_row(row: pd.Series, min_trades: int = MIN_TRADES) -> tuple[list[str], bool, bool]:
    excess = _numeric(row, "strategy_vs_benchmark_pct")
    total_return = _numeric(row, "total_return_pct")
    trade_count = _numeric(row, "trade_count")
    beat_benchmark = (
        _bool_value(row.get("beat_benchmark"))
        if "beat_benchmark" in row and _clean_text(row.get("beat_benchmark"))
        else bool(pd.notna(excess) and excess > 0)
    )
    sufficient_trade = (
        _bool_value(row.get("sufficient_trade"))
        if "sufficient_trade" in row and _clean_text(row.get("sufficient_trade"))
        else bool(pd.notna(trade_count) and trade_count >= min_trades)
    )
    flags = []
    if pd.notna(excess) and excess < 0:
        flags.append("underperformed_benchmark")
    if pd.notna(total_return) and total_return < 0:
        flags.append("negative_total_return")
    if _warning_has(row, "low_trade_count") or not sufficient_trade:
        flags.append("low_trade_count" if _warning_has(row, "low_trade_count") else "insufficient_trade")
    return flags, beat_benchmark, sufficient_trade


def build_per_symbol_remediation_risk(inputs: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for regime, key in [
        ("bull", "per_symbol_bull"),
        ("sideways", "per_symbol_sideways"),
    ]:
        source = inputs[key]
        if source.empty:
            continue
        for _, row in source.iterrows():
            flags, beat_benchmark, sufficient_trade = _risk_flags_for_row(row)
            risk_level = "low"
            if len(flags) >= 2:
                risk_level = "high"
            elif len(flags) == 1:
                risk_level = "medium"
            rows.append(
                {
                    "symbol": _format_symbol(row.get("symbol")),
                    "regime": regime,
                    "candidate": _clean_text(row.get("canonical_mode")) or CANONICAL_MODE,
                    "model": _clean_text(row.get("model_type")) or MODEL_TYPE,
                    "buy_threshold": _numeric(row, "buy_threshold"),
                    "sell_threshold": _numeric(row, "sell_threshold"),
                    "total_return_pct": _numeric(row, "total_return_pct"),
                    "benchmark_return_pct": _numeric(row, "benchmark_return_pct"),
                    "strategy_vs_benchmark_pct": _numeric(row, "strategy_vs_benchmark_pct"),
                    "trade_count": _numeric(row, "trade_count"),
                    "beat_benchmark": beat_benchmark,
                    "sufficient_trade": sufficient_trade,
                    "risk_flags": ",".join(flags) if flags else "",
                    "risk_level": risk_level,
                    "interpretation": (
                        "symbol_has_remediation_risk_flags"
                        if flags
                        else "no_symbol_level_flag_in_selected_result"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "regime",
                "candidate",
                "model",
                "buy_threshold",
                "sell_threshold",
                "total_return_pct",
                "benchmark_return_pct",
                "strategy_vs_benchmark_pct",
                "trade_count",
                "beat_benchmark",
                "sufficient_trade",
                "risk_flags",
                "risk_level",
                "interpretation",
            ]
        )
    return result.sort_values(["risk_level", "regime", "symbol"], ascending=[True, True, True]).reset_index(drop=True)


def build_integrated_summary(regime_status: pd.DataFrame) -> pd.DataFrame:
    bull = regime_status[regime_status["regime"] == "bull"].head(1)
    sideways = regime_status[regime_status["regime"] == "sideways"].head(1)
    bull_row = _first_row(bull)
    sideways_row = _first_row(sideways)
    return pd.DataFrame(
        [
            {
                "canonical_mode": CANONICAL_MODE,
                "model": MODEL_TYPE,
                "overall_decision": "research_only_not_trading_ready",
                "trading_ready": False,
                "research_status": "partial_remediation_progress_but_not_trading_ready",
                "bull_final_decision": _clean_text(bull_row.get("final_decision")),
                "bull_avg_strategy_vs_benchmark_pct": _numeric(
                    bull_row,
                    "avg_strategy_vs_benchmark_pct",
                ),
                "bull_beat_benchmark_rate": _numeric(bull_row, "beat_benchmark_rate"),
                "bull_sufficient_trade_rate": _numeric(bull_row, "sufficient_trade_rate"),
                "bull_buy_threshold": _numeric(bull_row, "buy_threshold"),
                "bull_sell_threshold": _numeric(bull_row, "sell_threshold"),
                "sideways_final_decision": _clean_text(sideways_row.get("final_decision")),
                "sideways_avg_strategy_vs_benchmark_pct": _numeric(
                    sideways_row,
                    "avg_strategy_vs_benchmark_pct",
                ),
                "sideways_beat_benchmark_rate": _numeric(sideways_row, "beat_benchmark_rate"),
                "sideways_sufficient_trade_rate": _numeric(sideways_row, "sufficient_trade_rate"),
                "sideways_buy_threshold": _numeric(sideways_row, "buy_threshold"),
                "sideways_sell_threshold": _numeric(sideways_row, "sell_threshold"),
                "main_blocker": "bull_remediation_failed",
                "notes": (
                    "Sideways aggregate remediation passed, but bull remediation failed "
                    "and per-symbol weaknesses remain. No candidate is trading-ready."
                ),
            }
        ]
    )


def build_integrated_gate_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "canonical_mode": CANONICAL_MODE,
                "trading_ready": False,
                "gate_decision": "research_only_not_trading_ready",
                "reason": (
                    "Bull remediation failed; sideways passed configured aggregate research gates "
                    "but that is not sufficient for trading-ready status, and per-symbol weaknesses remain."
                ),
            },
            {
                "canonical_mode": "full",
                "trading_ready": False,
                "gate_decision": "baseline_only",
                "reason": "full remains a baseline comparison mode only.",
            },
            {
                "canonical_mode": "keep_core_only",
                "trading_ready": False,
                "gate_decision": "low_feature_challenger_only",
                "reason": (
                    "keep_core_only remains a low-feature challenger only because low-trade "
                    "and instability risk remain unresolved."
                ),
            },
        ]
    )


def _severity_for_warning(warning_type: str) -> str:
    if warning_type in {"bull_remediation_failed", "not_trading_ready"}:
        return "blocking"
    if warning_type in {"low_trade_count", "benchmark_underperformance", "negative_total_return"}:
        return "medium"
    return "info"


def _risk_type_from_gate_check(check_name: Any) -> str:
    text = _clean_text(check_name)
    if not text:
        return "validation_gate_failure"
    if text == "role_allowed":
        return "role_not_allowed"
    if text == "risk_flags_acceptable":
        return "risk_flags_not_acceptable"
    if text.endswith("_passed"):
        return text[: -len("_passed")] + "_failed"
    if text.endswith("_positive"):
        return text[: -len("_positive")] + "_nonpositive"
    return text


def build_integrated_risk_flags(
    inputs: dict[str, Any],
    per_symbol_risk: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "source": "integrated_gate",
            "canonical_mode": CANONICAL_MODE,
            "regime": "bull",
            "symbol": "",
            "risk_type": "bull_remediation_failed",
            "severity": "blocking",
            "message": "Bull remediation failed and remains the main unresolved blocker.",
        },
        {
            "source": "integrated_gate",
            "canonical_mode": CANONICAL_MODE,
            "regime": "",
            "symbol": "",
            "risk_type": "not_trading_ready",
            "severity": "blocking",
            "message": "No candidate is trading-ready in Step 36.",
        },
        {
            "source": "integrated_gate",
            "canonical_mode": "full",
            "regime": "",
            "symbol": "",
            "risk_type": "baseline_only",
            "severity": "info",
            "message": "full remains baseline only.",
        },
        {
            "source": "integrated_gate",
            "canonical_mode": "keep_core_only",
            "regime": "",
            "symbol": "",
            "risk_type": "low_feature_challenger_risk",
            "severity": "medium",
            "message": "keep_core_only remains a low-feature challenger only.",
        },
    ]
    warning_sources = [
        ("bull_warnings", "bull_warnings"),
        ("sideways_warnings", "sideways_warnings"),
    ]
    for input_key, source_name in warning_sources:
        warnings = inputs[input_key]
        if warnings.empty:
            continue
        for _, row in warnings.iterrows():
            warning_type = _clean_text(row.get("warning_type"))
            message = _clean_text(row.get("message"))
            message_lower = message.lower()
            risk_types = []
            if warning_type == "underperformed_benchmark" or "underperformed_benchmark" in message_lower:
                risk_types.append("benchmark_underperformance")
            if warning_type == "negative_total_return" or "negative_total_return" in message_lower:
                risk_types.append("negative_total_return")
            if warning_type == "low_trade_count" or "low_trade_count" in message_lower:
                risk_types.append("low_trade_count")
            if not risk_types:
                risk_types.append(warning_type or "research_warning")
            for risk_type in risk_types:
                rows.append(
                    {
                        "source": source_name,
                        "canonical_mode": _clean_text(row.get("canonical_mode")) or CANONICAL_MODE,
                        "regime": _clean_text(row.get("regime")),
                        "symbol": _format_symbol(row.get("symbol")),
                        "risk_type": risk_type,
                        "severity": _severity_for_warning(risk_type),
                        "message": message or risk_type,
                    }
                )
    if not per_symbol_risk.empty:
        for _, row in per_symbol_risk.iterrows():
            flags = [
                flag.strip()
                for flag in _clean_text(row.get("risk_flags")).split(",")
                if flag.strip()
            ]
            for flag in flags:
                risk_type = (
                    "sideways_per_symbol_underperformance"
                    if row.get("regime") == "sideways" and flag == "underperformed_benchmark"
                    else flag
                )
                rows.append(
                    {
                        "source": "per_symbol_remediation_risk",
                        "canonical_mode": _clean_text(row.get("candidate")) or CANONICAL_MODE,
                        "regime": _clean_text(row.get("regime")),
                        "symbol": _format_symbol(row.get("symbol")),
                        "risk_type": risk_type,
                        "severity": "medium" if row.get("risk_level") == "medium" else "high",
                        "message": f"{row.get('regime')} {row.get('symbol')} risk flag: {flag}",
                    }
                )
    gate_failures = inputs.get("validation_gate_failures", pd.DataFrame())
    if not gate_failures.empty:
        for _, row in gate_failures.iterrows():
            rows.append(
                {
                    "source": "validation_gate_failures",
                    "canonical_mode": _clean_text(row.get("canonical_mode")),
                    "regime": "",
                    "symbol": "",
                    "risk_type": _risk_type_from_gate_check(row.get("check_name")),
                    "severity": _clean_text(row.get("severity")) or "blocking",
                    "message": _clean_text(row.get("message")),
                }
            )
    optional_warnings = inputs.get("input_warnings", [])
    for warning in optional_warnings:
        rows.append(
            {
                "source": "input_warning",
                "canonical_mode": "",
                "regime": "",
                "symbol": "",
                "risk_type": _clean_text(warning.get("warning_type")),
                "severity": "info",
                "message": _clean_text(warning.get("message")),
            }
        )
    result = pd.DataFrame(rows)
    if "symbol" in result:
        result["symbol"] = result["symbol"].map(_format_symbol)
    return result


def generate_report(
    integrated_summary: pd.DataFrame,
    regime_status: pd.DataFrame,
    per_symbol_risk: pd.DataFrame,
    gate_results: pd.DataFrame,
    risk_flags: pd.DataFrame,
    bull_dir: str | Path,
    sideways_dir: str | Path,
    validation_gate_dir: str | Path | None,
    failure_analysis_dir: str | Path | None,
) -> str:
    summary = _first_row(integrated_summary)
    bull = _first_row(regime_status[regime_status["regime"] == "bull"])
    sideways = _first_row(regime_status[regime_status["regime"] == "sideways"])
    high_count = int((per_symbol_risk.get("risk_level", pd.Series(dtype=str)) == "high").sum())
    medium_count = int((per_symbol_risk.get("risk_level", pd.Series(dtype=str)) == "medium").sum())
    risk_counts = (
        risk_flags.groupby("risk_type").size().sort_values(ascending=False).to_dict()
        if not risk_flags.empty and "risk_type" in risk_flags
        else {}
    )
    risk_lines = "\n".join(f"- {key}: {value}" for key, value in risk_counts.items())
    sections = [
        "# V4 Step 36 Integrated Remediation Revalidation Report",
        "",
        "## Executive Summary",
        "No candidate is trading-ready.",
        "canonical_reduced_40 remains research-only, not trading-ready.",
        "Sideways aggregate remediation passed the configured research gates.",
        "Bull remediation failed and remains the main unresolved blocker.",
        "Per-symbol weaknesses remain and must be reviewed before any future gate upgrade.",
        "full remains baseline only. keep_core_only remains low-feature challenger only.",
        "",
        "## Inputs Used",
        f"- Bull remediation directory: {bull_dir}",
        f"- Sideways remediation directory: {sideways_dir}",
        f"- Validation gate directory: {validation_gate_dir or 'not provided'}",
        f"- Failure analysis directory: {failure_analysis_dir or 'not provided'}",
        "",
        "## Integrated Decision",
        f"- Candidate: {summary.get('canonical_mode')} + {summary.get('model')}",
        f"- Overall decision: {summary.get('overall_decision')}",
        f"- Trading ready: {summary.get('trading_ready')}",
        f"- Research status: {summary.get('research_status')}",
        f"- Main blocker: {summary.get('main_blocker')}",
        "",
        "## Bull Remediation Review",
        f"- Best threshold: buy {bull.get('buy_threshold')}, sell {bull.get('sell_threshold')}",
        f"- Average strategy vs benchmark pct: {bull.get('avg_strategy_vs_benchmark_pct')}",
        f"- Beat benchmark rate: {bull.get('beat_benchmark_rate')}",
        f"- Sufficient trade rate: {bull.get('sufficient_trade_rate')}",
        f"- Final decision: {bull.get('final_decision')}",
        "Bull is close to passing but has not passed because average excess return is still slightly negative.",
        "",
        "## Sideways Remediation Review",
        f"- Best threshold: buy {sideways.get('buy_threshold')}, sell {sideways.get('sell_threshold')}",
        f"- Average strategy vs benchmark pct: {sideways.get('avg_strategy_vs_benchmark_pct')}",
        f"- Beat benchmark rate: {sideways.get('beat_benchmark_rate')}",
        f"- Sufficient trade rate: {sideways.get('sufficient_trade_rate')}",
        f"- Final decision: {sideways.get('final_decision')}",
        "The sideways aggregate pass is not enough to upgrade the candidate because per-symbol weakness remains.",
        "",
        "## Per-Symbol Risk Review",
        f"- High-risk selected symbol/regime rows: {high_count}",
        f"- Medium-risk selected symbol/regime rows: {medium_count}",
        "- Review per_symbol_remediation_risk.csv for symbol-level details.",
        "",
        "## Updated Gate Status",
        "- canonical_reduced_40: research_only_not_trading_ready",
        "- full: baseline_only",
        "- keep_core_only: low_feature_challenger_only",
        "- No integrated gate row has trading_ready=True.",
        "",
        "## Why This Is Still Not Trading-Ready",
        "Bull remediation failed the configured aggregate research gate.",
        "Sideways passed aggregate gates, but selected per-symbol results still include benchmark underperformance, negative returns, and low-trade warnings.",
        "The integrated result is a diagnostic update only and does not override the existing strict validation gate.",
        "",
        "## Recommended Next Step",
        "Recommended next step: V4 Step 37 Bull Regime Failure Drilldown / Symbol-Level Bull Diagnosis.",
        "Step 37 should identify which symbols or sub-periods cause bull underperformance without changing the feature set, adding data sources, adding agents, or tuning more thresholds.",
        "",
        "## Risk Flag Counts",
        risk_lines if risk_lines else "_No risk flags were recorded._",
        "",
        "## Educational / Research Disclaimer",
        "This report is educational/research diagnostics only. It is not financial advice.",
        "No strategy, model, threshold, or candidate in this report should be treated as deployable or trading-ready.",
        "",
    ]
    return "\n".join(sections)


def build_integrated_remediation_revalidation(
    bull_dir: str | Path,
    sideways_dir: str | Path,
    validation_gate_dir: str | Path | None = None,
    failure_analysis_dir: str | Path | None = None,
) -> dict[str, Any]:
    inputs = load_integrated_inputs(
        bull_dir,
        sideways_dir,
        validation_gate_dir=validation_gate_dir,
        failure_analysis_dir=failure_analysis_dir,
    )
    regime_status = build_regime_remediation_status(inputs)
    per_symbol_risk = build_per_symbol_remediation_risk(inputs)
    integrated_summary = build_integrated_summary(regime_status)
    gate_results = build_integrated_gate_results()
    risk_flags = build_integrated_risk_flags(inputs, per_symbol_risk)
    report = generate_report(
        integrated_summary,
        regime_status,
        per_symbol_risk,
        gate_results,
        risk_flags,
        bull_dir,
        sideways_dir,
        validation_gate_dir,
        failure_analysis_dir,
    )
    return {
        "integrated_remediation_summary": integrated_summary,
        "regime_remediation_status": regime_status,
        "per_symbol_remediation_risk": per_symbol_risk,
        "integrated_gate_results": gate_results,
        "integrated_risk_flags": risk_flags,
        "integrated_remediation_revalidation_report": report,
        "input_warnings": inputs["input_warnings"],
    }


def generate_integrated_remediation_revalidation(
    bull_dir: str | Path,
    sideways_dir: str | Path,
    output_dir: str | Path,
    validation_gate_dir: str | Path | None = None,
    failure_analysis_dir: str | Path | None = None,
) -> dict[str, Any]:
    result = build_integrated_remediation_revalidation(
        bull_dir,
        sideways_dir,
        validation_gate_dir=validation_gate_dir,
        failure_analysis_dir=failure_analysis_dir,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    result["integrated_remediation_summary"].to_csv(paths["summary"], index=False)
    result["regime_remediation_status"].to_csv(paths["regime_status"], index=False)
    result["per_symbol_remediation_risk"].to_csv(paths["per_symbol_risk"], index=False)
    result["integrated_gate_results"].to_csv(paths["gate_results"], index=False)
    result["integrated_risk_flags"].to_csv(paths["risk_flags"], index=False)
    paths["report"].write_text(
        result["integrated_remediation_revalidation_report"],
        encoding="utf-8",
    )
    run_config = {
        "bull_dir": str(bull_dir),
        "sideways_dir": str(sideways_dir),
        "output_dir": str(output_path),
        "validation_gate_dir": str(validation_gate_dir) if validation_gate_dir else None,
        "failure_analysis_dir": str(failure_analysis_dir) if failure_analysis_dir else None,
        "input_warnings": result["input_warnings"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["run_config"] = run_config
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result
