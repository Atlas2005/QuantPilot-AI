import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_MODES = ["canonical_reduced_40", "full", "keep_core_only"]
REGIMES = ["bull", "bear", "sideways"]
BLOCKER_CHECKS = [
    "stress_validation_passed",
    "stress_beat_benchmark_rate_passed",
    "stress_sufficient_trade_rate_passed",
    "risk_flags_acceptable",
]


def _read_csv(
    path: Path,
    warnings: list[str],
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing input file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        warnings.append(f"Empty input file: {path}")
        return pd.DataFrame()


def _read_text(path: Path, warnings: list[str]) -> str:
    if not path.exists():
        warnings.append(f"Missing input file: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


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


def _normalize_symbol_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "symbol" not in df:
        return df.copy()
    result = df.copy()
    result["symbol"] = result["symbol"].map(_format_symbol)
    return result


def load_failure_analysis_inputs(
    gate_dir: str | Path,
    revalidation_dir: str | Path,
    stress_dir: str | Path,
) -> dict[str, Any]:
    warnings: list[str] = []
    gate = Path(gate_dir)
    revalidation = Path(revalidation_dir)
    stress = Path(stress_dir)
    inputs = {
        "validation_gate_results": _read_csv(gate / "validation_gate_results.csv", warnings),
        "validation_gate_failures": _read_csv(gate / "validation_gate_failures.csv", warnings),
        "candidate_validation_gate_report": _read_text(
            gate / "candidate_validation_gate_report.md",
            warnings,
        ),
        "revalidation_summary": _read_csv(
            revalidation / "canonical_candidate_revalidation_summary.csv",
            warnings,
        ),
        "candidate_risk_flags": _read_csv(
            revalidation / "candidate_risk_flags.csv",
            warnings,
            dtype={"symbol": str},
        ),
        "candidate_stress_summary": _read_csv(
            stress / "candidate_stress_summary.csv",
            warnings,
        ),
        "regime_summary": _read_csv(stress / "regime_summary.csv", warnings),
        "per_symbol_stress_results": _read_csv(
            stress / "per_symbol_stress_results.csv",
            warnings,
            dtype={"symbol": str},
        ),
        "stress_warnings": _read_csv(
            stress / "stress_warnings.csv",
            warnings,
            dtype={"symbol": str},
        ),
        "input_warnings": warnings,
    }
    for key in ["candidate_risk_flags", "per_symbol_stress_results", "stress_warnings"]:
        inputs[key] = _normalize_symbol_columns(inputs[key])
    return inputs


def build_gate_failure_summary(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "canonical_mode",
        "role",
        "final_gate_decision",
        "trading_ready",
        "strict_gates_passed",
        "gate_reason",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    result = results.copy()
    if "trading_ready" not in result:
        result["trading_ready"] = False
    result["trading_ready"] = result["trading_ready"].fillna(False).astype(bool)
    for column in columns:
        if column not in result:
            result[column] = ""
    return result[columns]


def build_failure_by_check(failures: pd.DataFrame) -> pd.DataFrame:
    columns = ["check_name", "canonical_mode", "failure_count"]
    if failures.empty:
        return pd.DataFrame(columns=columns)
    result = (
        failures.groupby(["check_name", "canonical_mode"], dropna=False)
        .size()
        .reset_index(name="failure_count")
        .sort_values(["failure_count", "check_name", "canonical_mode"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return result[columns]


def build_failure_by_candidate(failures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "canonical_mode",
        "final_gate_decision",
        "trading_ready",
        "failure_count",
        "failed_checks",
        "main_blockers",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    failure_groups = (
        failures.groupby("canonical_mode", dropna=False)["check_name"]
        .apply(lambda values: ",".join(sorted(values.dropna().astype(str).unique())))
        if not failures.empty and "canonical_mode" in failures
        else pd.Series(dtype=str)
    )
    failure_counts = (
        failures.groupby("canonical_mode", dropna=False).size()
        if not failures.empty and "canonical_mode" in failures
        else pd.Series(dtype=int)
    )
    rows = []
    for _, row in results.iterrows():
        mode = _clean_text(row.get("canonical_mode"))
        failed_checks = failure_groups.get(mode, "")
        blockers = ",".join(
            check for check in BLOCKER_CHECKS if check in failed_checks.split(",")
        )
        rows.append(
            {
                "canonical_mode": mode,
                "final_gate_decision": row.get("final_gate_decision", ""),
                "trading_ready": bool(row.get("trading_ready", False)),
                "failure_count": int(failure_counts.get(mode, 0)),
                "failed_checks": failed_checks,
                "main_blockers": blockers,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_risk_flag_summary(risk_flags: pd.DataFrame, stress_warnings: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source",
        "risk_category",
        "canonical_mode",
        "symbol",
        "warning_type",
        "issue_count",
    ]
    frames = []
    if not risk_flags.empty:
        flags = risk_flags.copy()
        for column, default in [
            ("source", "candidate_risk_flags"),
            ("risk_category", "research_warning"),
            ("canonical_mode", ""),
            ("symbol", ""),
            ("warning_type", ""),
        ]:
            if column not in flags:
                flags[column] = default
        frames.append(flags[columns[:-1]])
    if not stress_warnings.empty:
        warnings = stress_warnings.copy()
        if "source" not in warnings:
            warnings["source"] = "stress_warnings"
        if "risk_category" not in warnings:
            warnings["risk_category"] = "stress_warning"
        for column in ["canonical_mode", "symbol", "warning_type"]:
            if column not in warnings:
                warnings[column] = ""
        frames.append(warnings[columns[:-1]])
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "symbol" in combined:
        combined["symbol"] = combined["symbol"].map(_format_symbol)
    result = (
        combined.groupby(columns[:-1], dropna=False)
        .size()
        .reset_index(name="issue_count")
        .sort_values(["issue_count", "canonical_mode", "symbol"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return result[columns]


def build_failure_by_symbol(
    per_symbol: pd.DataFrame,
    risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "canonical_mode",
        "symbol",
        "stress_row_count",
        "stress_failure_count",
        "avg_strategy_vs_benchmark_pct",
        "avg_trade_count",
        "warning_count",
    ]
    rows = []
    if not per_symbol.empty and {"canonical_mode", "symbol"}.issubset(per_symbol.columns):
        stress = per_symbol.copy()
        stress["symbol"] = stress["symbol"].map(_format_symbol)
        stress["stress_failed"] = _numeric(stress, "strategy_vs_benchmark_pct") <= 0
        for (mode, symbol), group in stress.groupby(["canonical_mode", "symbol"], dropna=False):
            rows.append(
                {
                    "canonical_mode": mode,
                    "symbol": symbol,
                    "stress_row_count": len(group),
                    "stress_failure_count": int(group["stress_failed"].sum()),
                    "avg_strategy_vs_benchmark_pct": _numeric(
                        group,
                        "strategy_vs_benchmark_pct",
                    ).mean(),
                    "avg_trade_count": _numeric(group, "trade_count").mean(),
                    "warning_count": 0,
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    if not risk_summary.empty and {"canonical_mode", "symbol"}.issubset(risk_summary.columns):
        warning_counts = (
            risk_summary.groupby(["canonical_mode", "symbol"], dropna=False)["issue_count"]
            .sum()
            .reset_index(name="warning_count")
        )
        warning_counts["symbol"] = warning_counts["symbol"].map(_format_symbol)
        if result.empty:
            result = warning_counts.copy()
            result["stress_row_count"] = 0
            result["stress_failure_count"] = 0
            result["avg_strategy_vs_benchmark_pct"] = pd.NA
            result["avg_trade_count"] = pd.NA
            result = result[columns]
        else:
            result = result.merge(
                warning_counts,
                on=["canonical_mode", "symbol"],
                how="outer",
                suffixes=("", "_from_warnings"),
            )
            result["warning_count"] = (
                result["warning_count_from_warnings"].fillna(result["warning_count"]).fillna(0).astype(int)
            )
            result = result.drop(columns=["warning_count_from_warnings"])
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["symbol"] = result["symbol"].map(_format_symbol)
    return (
        result[columns]
        .sort_values(["stress_failure_count", "warning_count", "canonical_mode", "symbol"], ascending=[False, False, True, True])
        .reset_index(drop=True)
    )


def build_failure_by_regime(regime_summary: pd.DataFrame, per_symbol: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "canonical_mode",
        "regime",
        "tested_symbol_count",
        "avg_strategy_vs_benchmark_pct",
        "beat_benchmark_rate",
        "sufficient_trade_rate",
        "regime_gate_failed",
        "has_regime_warnings",
        "stress_failure_count",
    ]
    rows = []
    if not regime_summary.empty and {"canonical_mode", "regime"}.issubset(regime_summary.columns):
        for _, row in regime_summary.iterrows():
            mode = _clean_text(row.get("canonical_mode"))
            regime = _clean_text(row.get("regime"))
            avg_excess = pd.to_numeric(pd.Series([row.get("avg_strategy_vs_benchmark_pct")]), errors="coerce").iloc[0]
            beat_rate = pd.to_numeric(pd.Series([row.get("beat_benchmark_rate")]), errors="coerce").iloc[0]
            trade_rate = pd.to_numeric(pd.Series([row.get("sufficient_trade_rate")]), errors="coerce").iloc[0]
            rows.append(
                {
                    "canonical_mode": mode,
                    "regime": regime,
                    "tested_symbol_count": row.get("tested_symbol_count", 0),
                    "avg_strategy_vs_benchmark_pct": avg_excess,
                    "beat_benchmark_rate": beat_rate,
                    "sufficient_trade_rate": trade_rate,
                    "regime_gate_failed": bool(
                        (pd.notna(avg_excess) and avg_excess <= 0)
                        or (pd.notna(beat_rate) and beat_rate < 0.60)
                        or (pd.notna(trade_rate) and trade_rate < 0.80)
                    ),
                    "has_regime_warnings": False,
                    "stress_failure_count": 0,
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    if not per_symbol.empty and {"canonical_mode", "regime"}.issubset(per_symbol.columns):
        stress = per_symbol.copy()
        stress["stress_failed"] = _numeric(stress, "strategy_vs_benchmark_pct") <= 0
        counts = (
            stress.groupby(["canonical_mode", "regime"], dropna=False)["stress_failed"]
            .sum()
            .reset_index(name="stress_failure_count")
        )
        if result.empty:
            result = counts.copy()
            result["tested_symbol_count"] = 0
            result["avg_strategy_vs_benchmark_pct"] = pd.NA
            result["beat_benchmark_rate"] = pd.NA
            result["sufficient_trade_rate"] = pd.NA
            result["regime_gate_failed"] = False
            result["has_regime_warnings"] = result["stress_failure_count"] > 0
            result = result[columns]
        else:
            result = result.merge(
                counts,
                on=["canonical_mode", "regime"],
                how="outer",
                suffixes=("", "_from_symbols"),
            )
            result["stress_failure_count"] = (
                result["stress_failure_count_from_symbols"]
                .fillna(result["stress_failure_count"])
                .fillna(0)
                .astype(int)
            )
            result = result.drop(columns=["stress_failure_count_from_symbols"])
            result["regime_gate_failed"] = result["regime_gate_failed"].fillna(False).astype(bool)
            result["has_regime_warnings"] = (
                result["stress_failure_count"] > 0
            )
    if result.empty:
        return pd.DataFrame(columns=columns)
    return (
        result[result["regime"].isin(REGIMES) | result["regime"].notna()]
        .sort_values(["canonical_mode", "regime"])
        .reset_index(drop=True)
    )


def build_remediation_plan(
    results: pd.DataFrame,
    failures: pd.DataFrame,
    risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "priority": "P0",
            "focus_area": "validation weakness",
            "applies_to": "canonical_reduced_40,keep_core_only",
            "action": "Fix validation weakness before any trading-ready claim.",
            "evidence": "Gate decisions are not trading_ready or required validation/stress checks failed.",
        },
        {
            "priority": "P1",
            "focus_area": "stress robustness and sufficient trade rate",
            "applies_to": "canonical_reduced_40",
            "action": "Improve stress robustness and sufficient trade rate across bull, bear, and sideways regimes.",
            "evidence": "Stress validation, beat-benchmark, or sufficient-trade gates failed.",
        },
        {
            "priority": "P2",
            "focus_area": "risk warnings",
            "applies_to": "all non-baseline candidates",
            "action": "Reduce benchmark underperformance and low-confidence warnings.",
            "evidence": "Risk flag categories remain visible and block readiness when severe.",
        },
        {
            "priority": "P3",
            "focus_area": "future expansion",
            "applies_to": "research backlog",
            "action": "Only then consider new features or agents.",
            "evidence": "Feature or agent expansion should wait until validation gate failure causes are remediated.",
        },
    ]
    if not failures.empty:
        reduced_failures = failures[
            failures.get("canonical_mode", pd.Series(dtype=str)) == "canonical_reduced_40"
        ]
        if not reduced_failures.empty:
            rows[0]["evidence"] = "canonical_reduced_40 failed: " + ",".join(
                sorted(reduced_failures["check_name"].dropna().astype(str).unique())
            )
    if not risk_summary.empty:
        rows[2]["evidence"] = f"{int(risk_summary['issue_count'].sum())} aggregated risk/warning issue(s)."
    return pd.DataFrame(rows)


def generate_failure_analysis_report(
    gate_summary: pd.DataFrame,
    failure_by_candidate: pd.DataFrame,
    failure_by_check: pd.DataFrame,
    failure_by_symbol: pd.DataFrame,
    failure_by_regime: pd.DataFrame,
    risk_summary: pd.DataFrame,
    remediation_plan: pd.DataFrame,
    input_warnings: list[str],
) -> str:
    no_ready = (
        gate_summary.empty
        or not gate_summary.get("trading_ready", pd.Series(dtype=bool)).fillna(False).astype(bool).any()
    )
    readiness_line = (
        "No candidate is trading-ready; all candidates are not trading-ready under the current validation gate."
        if no_ready
        else "At least one candidate is marked trading-ready; review the gate results before using this report."
    )
    sections = [
        "# Validation Gate Failure Analysis / Remediation Plan",
        "",
        "## Executive Summary",
        "This is reporting-only research diagnostics. It does not change trading, backtest, model, or strategy logic.",
        readiness_line,
        "canonical_reduced_40 remains the primary research candidate, not a trading candidate.",
        "canonical_reduced_40 is blocked by stress validation failure and related validation gate failures when those checks fail.",
        "full remains baseline only.",
        "keep_core_only remains a low-feature challenger with instability / low-trade-count risk.",
        "The remediation plan prioritizes validation weakness, stress robustness, warning reduction, and delayed expansion.",
        "Do not add new features or agents until the validation gate failure causes are remediated.",
        "",
        "## Gate Failure Summary",
        _markdown_table(gate_summary),
        "",
        "## Failure by Candidate",
        _markdown_table(failure_by_candidate),
        "",
        "## Failure by Check",
        _markdown_table(failure_by_check),
        "",
        "## Failure by Regime",
        (
            "Regime gate failure is based only on aggregate thresholds: "
            "avg_strategy_vs_benchmark_pct <= 0, beat_benchmark_rate < 0.60, "
            "or sufficient_trade_rate < 0.80. A regime with warning rows is "
            "shown with has_regime_warnings and is not labeled as a failed "
            "regime unless aggregate thresholds fail."
        ),
        _markdown_table(failure_by_regime),
        "",
        "## Failure by Symbol",
        _markdown_table(failure_by_symbol),
        "",
        "## Risk Flag Summary",
        _markdown_table(risk_summary),
        "",
        "## Remediation Plan",
        _markdown_table(remediation_plan),
        "",
        "## Input Warnings",
        "\n".join(f"- {warning}" for warning in input_warnings) if input_warnings else "_No input warnings._",
        "",
    ]
    return "\n".join(sections)


def build_validation_gate_failure_analysis(
    gate_dir: str | Path,
    revalidation_dir: str | Path,
    stress_dir: str | Path,
) -> dict[str, Any]:
    inputs = load_failure_analysis_inputs(gate_dir, revalidation_dir, stress_dir)
    gate_summary = build_gate_failure_summary(inputs["validation_gate_results"])
    failure_by_check = build_failure_by_check(inputs["validation_gate_failures"])
    failure_by_candidate = build_failure_by_candidate(
        inputs["validation_gate_failures"],
        inputs["validation_gate_results"],
    )
    risk_summary = build_risk_flag_summary(
        inputs["candidate_risk_flags"],
        inputs["stress_warnings"],
    )
    failure_by_symbol = build_failure_by_symbol(
        inputs["per_symbol_stress_results"],
        risk_summary,
    )
    failure_by_regime = build_failure_by_regime(
        inputs["regime_summary"],
        inputs["per_symbol_stress_results"],
    )
    remediation_plan = build_remediation_plan(
        inputs["validation_gate_results"],
        inputs["validation_gate_failures"],
        risk_summary,
    )
    report = generate_failure_analysis_report(
        gate_summary,
        failure_by_candidate,
        failure_by_check,
        failure_by_symbol,
        failure_by_regime,
        risk_summary,
        remediation_plan,
        inputs["input_warnings"],
    )
    return {
        "gate_failure_summary": gate_summary,
        "failure_by_check": failure_by_check,
        "failure_by_candidate": failure_by_candidate,
        "failure_by_symbol": failure_by_symbol,
        "failure_by_regime": failure_by_regime,
        "risk_flag_summary": risk_summary,
        "remediation_plan": remediation_plan,
        "validation_gate_failure_analysis_report": report,
        "input_warnings": inputs["input_warnings"],
    }


def save_validation_gate_failure_analysis(
    gate_dir: str | Path,
    revalidation_dir: str | Path,
    stress_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    result = build_validation_gate_failure_analysis(gate_dir, revalidation_dir, stress_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "gate_failure_summary": output_path / "gate_failure_summary.csv",
        "failure_by_check": output_path / "failure_by_check.csv",
        "failure_by_candidate": output_path / "failure_by_candidate.csv",
        "failure_by_symbol": output_path / "failure_by_symbol.csv",
        "failure_by_regime": output_path / "failure_by_regime.csv",
        "risk_flag_summary": output_path / "risk_flag_summary.csv",
        "remediation_plan": output_path / "remediation_plan.csv",
        "report": output_path / "validation_gate_failure_analysis_report.md",
        "run_config": output_path / "run_config.json",
    }
    for key in [
        "gate_failure_summary",
        "failure_by_check",
        "failure_by_candidate",
        "failure_by_symbol",
        "failure_by_regime",
        "risk_flag_summary",
        "remediation_plan",
    ]:
        result[key].to_csv(paths[key], index=False)
    paths["report"].write_text(
        result["validation_gate_failure_analysis_report"],
        encoding="utf-8",
    )
    run_config = {
        "gate_dir": str(gate_dir),
        "revalidation_dir": str(revalidation_dir),
        "stress_dir": str(stress_dir),
        "output_dir": str(output_path),
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
