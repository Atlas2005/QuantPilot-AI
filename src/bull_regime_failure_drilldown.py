import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_MODE = "canonical_reduced_40"
MODEL_TYPE = "logistic_regression"
MIN_TRADES = 3
NEAR_NEUTRAL_EXCESS_ABS = 0.25
STRONGLY_POSITIVE_BENCHMARK = 5.0
OUTPUT_FILENAMES = {
    "report": "bull_regime_failure_drilldown_report.md",
    "symbol_summary": "bull_symbol_failure_summary.csv",
    "contribution": "bull_failure_contribution.csv",
    "reasons": "bull_failure_reasons.csv",
    "threshold_context": "bull_threshold_context.csv",
    "limitations": "bull_drilldown_limitations.csv",
    "trade_level": "bull_trade_level_diagnostics.csv",
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


def _numeric(row: pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    return pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean_text(value).lower() in {"true", "1", "yes", "y"}


def _filter_best_threshold_rows(
    threshold_results: pd.DataFrame,
    per_symbol_results: pd.DataFrame,
    best_row: pd.Series,
) -> tuple[pd.DataFrame, str]:
    if threshold_results.empty:
        return per_symbol_results.copy(), "per_symbol_bull_results"
    buy = _numeric(best_row, "buy_threshold")
    sell = _numeric(best_row, "sell_threshold")
    result = threshold_results.copy()
    mask = (
        pd.to_numeric(result.get("buy_threshold"), errors="coerce").round(10) == round(buy, 10)
    ) & (
        pd.to_numeric(result.get("sell_threshold"), errors="coerce").round(10) == round(sell, 10)
    )
    if "canonical_mode" in result:
        mask &= result["canonical_mode"].astype(str) == _clean_text(best_row.get("canonical_mode"))
    if "model_type" in result:
        mask &= result["model_type"].astype(str) == _clean_text(best_row.get("model_type"))
    selected = result[mask].copy()
    if selected.empty:
        return per_symbol_results.copy(), "per_symbol_bull_results"
    return selected, "bull_threshold_results_selected_threshold"


def load_bull_drilldown_inputs(
    bull_dir: str | Path,
    integrated_dir: str | Path | None,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    bull = Path(bull_dir)
    inputs = {
        "bull_summary": _read_required_csv(bull / "bull_threshold_summary.csv"),
        "best_bull": _read_required_csv(bull / "best_bull_thresholds.csv"),
        "per_symbol_bull": _normalize_symbols(
            _read_required_csv(bull / "per_symbol_bull_results.csv", dtype={"symbol": str})
        ),
        "bull_threshold_results": _normalize_symbols(
            _read_optional_csv(bull / "bull_threshold_results.csv", warnings, dtype={"symbol": str})
        ),
        "bull_warnings": _normalize_symbols(
            _read_optional_csv(bull / "warnings.csv", warnings, dtype={"symbol": str})
        ),
        "integrated_summary": pd.DataFrame(),
        "regime_status": pd.DataFrame(),
        "integrated_gate_results": pd.DataFrame(),
        "integrated_risk_flags": pd.DataFrame(),
        "integrated_report": "",
        "input_warnings": warnings,
    }
    if integrated_dir:
        integrated = Path(integrated_dir)
        inputs["integrated_summary"] = _read_optional_csv(
            integrated / "integrated_remediation_summary.csv",
            warnings,
        )
        inputs["regime_status"] = _read_optional_csv(
            integrated / "regime_remediation_status.csv",
            warnings,
        )
        inputs["integrated_gate_results"] = _read_optional_csv(
            integrated / "integrated_gate_results.csv",
            warnings,
        )
        inputs["integrated_risk_flags"] = _normalize_symbols(
            _read_optional_csv(
                integrated / "integrated_risk_flags.csv",
                warnings,
                dtype={"symbol": str},
            )
        )
        inputs["integrated_report"] = _read_optional_text(
            integrated / "integrated_remediation_revalidation_report.md",
            warnings,
        )
    best_row = _first_row(inputs["best_bull"])
    selected_rows, selected_source = _filter_best_threshold_rows(
        inputs["bull_threshold_results"],
        inputs["per_symbol_bull"],
        best_row,
    )
    inputs["selected_symbol_rows"] = _normalize_symbols(selected_rows)
    inputs["selected_symbol_source"] = selected_source
    return inputs


def build_bull_threshold_context(inputs: dict[str, Any]) -> pd.DataFrame:
    row = _first_row(inputs["best_bull"])
    return pd.DataFrame(
        [
            {
                "candidate": _clean_text(row.get("canonical_mode")) or CANONICAL_MODE,
                "model": _clean_text(row.get("model_type")) or MODEL_TYPE,
                "buy_threshold": _numeric(row, "buy_threshold"),
                "sell_threshold": _numeric(row, "sell_threshold"),
                "source_step": "V4 Step 34 Bull Regime Threshold Remediation",
                "final_decision": _clean_text(row.get("final_decision")),
                "avg_total_return_pct": _numeric(row, "avg_total_return_pct"),
                "avg_benchmark_return_pct": _numeric(row, "avg_benchmark_return_pct"),
                "avg_strategy_vs_benchmark_pct": _numeric(
                    row,
                    "avg_strategy_vs_benchmark_pct",
                ),
                "beat_benchmark_rate": _numeric(row, "beat_benchmark_rate"),
                "sufficient_trade_rate": _numeric(row, "sufficient_trade_rate"),
                "tested_symbol_count": _numeric(row, "tested_symbol_count"),
                "threshold_action": "reused_for_diagnosis_only",
                "notes": (
                    "Step 37 reuses the selected Step 34 bull threshold for diagnosis only. "
                    "No threshold tuning or model change is performed."
                ),
            }
        ]
    )


def _symbol_reason(row: pd.Series, sufficient_trade: bool) -> str:
    total_return = _numeric(row, "total_return_pct")
    benchmark_return = _numeric(row, "benchmark_return_pct")
    excess = _numeric(row, "strategy_vs_benchmark_pct")
    trade_count = _numeric(row, "trade_count")
    if pd.notna(total_return) and pd.notna(excess) and total_return < 0 and excess < 0:
        return "negative_strategy_and_underperformed_benchmark"
    if not sufficient_trade:
        return "insufficient_trade"
    if pd.notna(trade_count) and trade_count < MIN_TRADES:
        return "low_trade_count"
    if (
        pd.notna(benchmark_return)
        and pd.notna(excess)
        and benchmark_return >= STRONGLY_POSITIVE_BENCHMARK
        and excess < 0
    ):
        return "benchmark_outpaced_strategy"
    if pd.notna(total_return) and pd.notna(excess) and total_return >= 0 and excess < 0:
        return "positive_return_but_lagged_benchmark"
    return "no_symbol_level_failure"


def _failure_role(excess: float) -> str:
    if pd.isna(excess):
        return "unknown"
    if abs(excess) <= NEAR_NEUTRAL_EXCESS_ABS:
        return "near_neutral"
    if excess < 0:
        return "drag_on_bull_average"
    return "support_for_bull_average"


def _risk_level(row: pd.Series, beat_benchmark: bool, sufficient_trade: bool) -> str:
    total_return = _numeric(row, "total_return_pct")
    excess = _numeric(row, "strategy_vs_benchmark_pct")
    underperformed = pd.notna(excess) and excess < 0
    negative = pd.notna(total_return) and total_return < 0
    if (negative and underperformed) or (not sufficient_trade and underperformed):
        return "high"
    if underperformed or not sufficient_trade:
        return "medium"
    if beat_benchmark and sufficient_trade:
        return "low"
    return "medium"


def build_bull_symbol_failure_summary(inputs: dict[str, Any]) -> pd.DataFrame:
    source = inputs["selected_symbol_rows"]
    tested_count = max(int(source["symbol"].nunique()), 1) if not source.empty and "symbol" in source else 1
    rows = []
    for _, row in source.iterrows():
        excess = _numeric(row, "strategy_vs_benchmark_pct")
        trade_count = _numeric(row, "trade_count")
        beat_benchmark = bool(pd.notna(excess) and excess > 0)
        sufficient_trade = bool(pd.notna(trade_count) and trade_count >= MIN_TRADES)
        primary_reason = _symbol_reason(row, sufficient_trade)
        rows.append(
            {
                "symbol": _format_symbol(row.get("symbol")),
                "candidate": _clean_text(row.get("canonical_mode")) or CANONICAL_MODE,
                "model": _clean_text(row.get("model_type")) or MODEL_TYPE,
                "buy_threshold": _numeric(row, "buy_threshold"),
                "sell_threshold": _numeric(row, "sell_threshold"),
                "total_return_pct": _numeric(row, "total_return_pct"),
                "benchmark_return_pct": _numeric(row, "benchmark_return_pct"),
                "strategy_vs_benchmark_pct": excess,
                "trade_count": trade_count,
                "beat_benchmark": beat_benchmark,
                "sufficient_trade": sufficient_trade,
                "contribution_to_avg_excess_pct": (
                    excess / tested_count if pd.notna(excess) else float("nan")
                ),
                "failure_role": _failure_role(excess),
                "primary_failure_reason": primary_reason,
                "risk_level": _risk_level(row, beat_benchmark, sufficient_trade),
                "notes": (
                    "Selected Step 34 threshold row from bull_threshold_results."
                    if inputs["selected_symbol_source"] == "bull_threshold_results_selected_threshold"
                    else "Fallback row from per_symbol_bull_results; exact threshold context may differ."
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
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
                "contribution_to_avg_excess_pct",
                "failure_role",
                "primary_failure_reason",
                "risk_level",
                "notes",
            ]
        )
    return result.sort_values(
        ["strategy_vs_benchmark_pct", "symbol"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_bull_failure_contribution(symbol_summary: pd.DataFrame) -> pd.DataFrame:
    if symbol_summary.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "strategy_vs_benchmark_pct",
                "contribution_to_avg_excess_pct",
                "rank_by_negative_contribution",
                "cumulative_negative_drag_pct",
                "would_aggregate_pass_without_this_symbol",
                "interpretation",
            ]
        )
    rows = []
    excess = pd.to_numeric(symbol_summary["strategy_vs_benchmark_pct"], errors="coerce")
    total_count = int(excess.notna().sum())
    current_sum = excess.sum()
    ranked = symbol_summary.copy()
    ranked["negative_sort"] = pd.to_numeric(
        ranked["strategy_vs_benchmark_pct"],
        errors="coerce",
    ).fillna(0)
    ranked = ranked.sort_values(["negative_sort", "symbol"], ascending=[True, True])
    cumulative_drag = 0.0
    rank = 0
    for _, row in ranked.iterrows():
        value = _numeric(row, "strategy_vs_benchmark_pct")
        if pd.notna(value) and value < 0:
            rank += 1
            cumulative_drag += value / max(total_count, 1)
            remaining_mean = (current_sum - value) / max(total_count - 1, 1)
            interpretation = "negative_contributor_to_bull_average"
        else:
            remaining_mean = (current_sum - value) / max(total_count - 1, 1) if pd.notna(value) else float("nan")
            interpretation = "positive_or_neutral_contributor"
        rows.append(
            {
                "symbol": _format_symbol(row.get("symbol")),
                "strategy_vs_benchmark_pct": value,
                "contribution_to_avg_excess_pct": (
                    value / max(total_count, 1) if pd.notna(value) else float("nan")
                ),
                "rank_by_negative_contribution": rank if pd.notna(value) and value < 0 else pd.NA,
                "cumulative_negative_drag_pct": cumulative_drag if pd.notna(value) and value < 0 else pd.NA,
                "would_aggregate_pass_without_this_symbol": bool(
                    pd.notna(remaining_mean) and remaining_mean > 0
                ),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def build_bull_failure_reasons(
    threshold_context: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    trade_level_available: bool,
    subperiod_available: bool,
) -> pd.DataFrame:
    context = _first_row(threshold_context)
    avg_excess = _numeric(context, "avg_strategy_vs_benchmark_pct")
    rows = [
        {
            "level": "aggregate",
            "symbol": "",
            "reason_type": "bull_average_excess_slightly_negative",
            "severity": "blocking",
            "evidence": f"avg_strategy_vs_benchmark_pct={avg_excess}",
            "interpretation": "Bull average excess stayed below the strict > 0 gate.",
        },
        {
            "level": "aggregate",
            "symbol": "",
            "reason_type": "bull_remediation_failed",
            "severity": "blocking",
            "evidence": f"final_decision={context.get('final_decision')}",
            "interpretation": "Step 34 bull remediation remains failed.",
        },
        {
            "level": "aggregate",
            "symbol": "",
            "reason_type": "near_pass_but_not_passed",
            "severity": "medium",
            "evidence": "beat_benchmark_rate=0.60 and sufficient_trade_rate=0.80, but avg excess <= 0",
            "interpretation": "The result is close to passing but still a strict failure.",
        },
        {
            "level": "aggregate",
            "symbol": "",
            "reason_type": "not_trading_ready",
            "severity": "blocking",
            "evidence": "canonical_reduced_40 remains research_only_not_trading_ready",
            "interpretation": "No candidate is trading-ready.",
        },
    ]
    for _, row in symbol_summary.iterrows():
        symbol = _format_symbol(row.get("symbol"))
        excess = _numeric(row, "strategy_vs_benchmark_pct")
        total_return = _numeric(row, "total_return_pct")
        benchmark_return = _numeric(row, "benchmark_return_pct")
        trade_count = _numeric(row, "trade_count")
        sufficient = _bool_from_value(row.get("sufficient_trade"))
        if pd.notna(benchmark_return) and benchmark_return >= STRONGLY_POSITIVE_BENCHMARK and pd.notna(excess) and excess < 0:
            rows.append(
                {
                    "level": "symbol",
                    "symbol": symbol,
                    "reason_type": "benchmark_outpaced_strategy",
                    "severity": "medium",
                    "evidence": f"benchmark_return_pct={benchmark_return}, excess={excess}",
                    "interpretation": "The benchmark rose more than the strategy in this bull symbol row.",
                }
            )
        if pd.notna(total_return) and total_return < 0:
            rows.append(
                {
                    "level": "symbol",
                    "symbol": symbol,
                    "reason_type": "negative_strategy_return",
                    "severity": "medium",
                    "evidence": f"total_return_pct={total_return}",
                    "interpretation": "The strategy return was negative in the selected bull row.",
                }
            )
        if pd.notna(excess) and excess < 0:
            rows.append(
                {
                    "level": "symbol",
                    "symbol": symbol,
                    "reason_type": "underperformed_benchmark",
                    "severity": "medium",
                    "evidence": f"strategy_vs_benchmark_pct={excess}",
                    "interpretation": "This symbol dragged the bull average below the benchmark.",
                }
            )
        if pd.notna(trade_count) and trade_count < MIN_TRADES:
            rows.append(
                {
                    "level": "symbol",
                    "symbol": symbol,
                    "reason_type": "low_trade_count",
                    "severity": "medium",
                    "evidence": f"trade_count={trade_count}",
                    "interpretation": "Trade count is below the configured sufficiency threshold.",
                }
            )
        if not sufficient:
            rows.append(
                {
                    "level": "symbol",
                    "symbol": symbol,
                    "reason_type": "insufficient_trade",
                    "severity": "medium",
                    "evidence": f"sufficient_trade={sufficient}",
                    "interpretation": "This symbol does not meet trade sufficiency.",
                }
            )
    if not trade_level_available:
        rows.append(
            {
                "level": "data_limitation",
                "symbol": "",
                "reason_type": "trade_level_data_unavailable",
                "severity": "medium",
                "evidence": "No Step 34 trade-level bull output file was found.",
                "interpretation": "Trade-by-trade diagnosis requires a future output enhancement.",
            }
        )
    if not subperiod_available:
        rows.append(
            {
                "level": "data_limitation",
                "symbol": "",
                "reason_type": "subperiod_diagnostics_limited",
                "severity": "medium",
                "evidence": "Selected Step 34 bull rows do not include date or subperiod columns.",
                "interpretation": "Subperiod diagnosis requires date-level or window-level output.",
            }
        )
    return pd.DataFrame(rows)


def build_bull_drilldown_limitations(
    trade_level_available: bool,
    subperiod_available: bool,
) -> pd.DataFrame:
    rows = [
        {
            "limitation_type": "no_new_data_sources_used",
            "severity": "info",
            "description": "Step 37 uses only existing Step 34 and optional Step 36 outputs.",
            "consequence": "No external or new market data is introduced.",
            "recommended_followup": "Keep future diagnostics on existing outputs unless a later step explicitly expands scope.",
        },
        {
            "limitation_type": "no_new_threshold_search",
            "severity": "info",
            "description": "The Step 34 selected bull threshold is reused for diagnosis only.",
            "consequence": "This report does not optimize or tune thresholds.",
            "recommended_followup": "Do not tune thresholds in Step 37.",
        },
        {
            "limitation_type": "trade_level_data_may_be_unavailable",
            "severity": "medium" if not trade_level_available else "info",
            "description": "Trade-level bull output is not present in current Step 34 outputs." if not trade_level_available else "Trade-level bull output was detected.",
            "consequence": "Trade-level entry/exit pattern diagnosis is limited." if not trade_level_available else "Trade-level rows can be inspected separately.",
            "recommended_followup": "V4 Step 38: Bull Trade/Window Diagnostics Output Enhancement." if not trade_level_available else "V4 Step 38: Bull Error Pattern Classification.",
        },
        {
            "limitation_type": "subperiod_analysis_requires_trade_or_date_level_outputs",
            "severity": "medium" if not subperiod_available else "info",
            "description": "Selected bull rows do not expose enough date/window fields for subperiod drilldown." if not subperiod_available else "Date/window fields are available.",
            "consequence": "The current report remains symbol-level and aggregate-level." if not subperiod_available else "Subperiod grouping can be explored in a later diagnostic.",
            "recommended_followup": "Add date/window diagnostics output before subperiod attribution." if not subperiod_available else "Classify bull error patterns by available windows.",
        },
        {
            "limitation_type": "symbol_count_is_small",
            "severity": "medium",
            "description": "The bull aggregate uses five tested symbols.",
            "consequence": "Single-symbol drag can materially change aggregate excess.",
            "recommended_followup": "Treat leave-one-symbol-out results as diagnostics, not symbol removal recommendations.",
        },
        {
            "limitation_type": "research_only_not_trading_ready",
            "severity": "blocking",
            "description": "canonical_reduced_40 remains research-only and not trading-ready.",
            "consequence": "This drilldown does not upgrade any candidate.",
            "recommended_followup": "Continue conservative diagnostics before any future gate update.",
        },
    ]
    return pd.DataFrame(rows)


def build_trade_level_diagnostics_placeholder(trade_level_available: bool) -> pd.DataFrame:
    if trade_level_available:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "diagnostic_status": "trade_level_data_unavailable",
                "source": "outputs/bull_regime_threshold_remediation_real_v1",
                "note": (
                    "Current Step 34 bull remediation outputs do not include trade-level rows. "
                    "Step 37 therefore performs aggregate and symbol-level diagnosis only."
                ),
            }
        ]
    )


def generate_report(
    threshold_context: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    contribution: pd.DataFrame,
    reasons: pd.DataFrame,
    limitations: pd.DataFrame,
    bull_dir: str | Path,
    integrated_dir: str | Path | None,
    trade_level_available: bool,
) -> str:
    context = _first_row(threshold_context)
    top_drags = contribution[
        pd.to_numeric(contribution.get("strategy_vs_benchmark_pct"), errors="coerce") < 0
    ].head(3)
    top_drag_lines = [
        f"- {row['symbol']}: excess={row['strategy_vs_benchmark_pct']}, contribution={row['contribution_to_avg_excess_pct']}"
        for _, row in top_drags.iterrows()
    ]
    next_step = (
        "V4 Step 38: Bull Error Pattern Classification."
        if trade_level_available
        else "V4 Step 38: Bull Trade/Window Diagnostics Output Enhancement."
    )
    sections = [
        "# V4 Step 37 Bull Regime Failure Drilldown Report",
        "",
        "## Executive Summary",
        "This is educational/research diagnostics only, not financial advice.",
        "No candidate is trading-ready.",
        "canonical_reduced_40 remains research-only and not trading-ready.",
        "Bull remediation failed because average strategy-vs-benchmark excess return remained slightly negative.",
        "The failure is close to pass but still a strict failure.",
        "This step does not tune thresholds, change models, add data sources, add agents, or change feature logic.",
        "Sideways remediation pass does not offset unresolved bull weakness.",
        "",
        "## Inputs Used",
        f"- Bull remediation directory: {bull_dir}",
        f"- Integrated remediation directory: {integrated_dir or 'not provided'}",
        f"- Symbol source: selected Step 34 threshold rows when available",
        "",
        "## Bull Threshold Context",
        f"- Candidate: {context.get('candidate')} + {context.get('model')}",
        f"- Reused threshold: buy {context.get('buy_threshold')}, sell {context.get('sell_threshold')}",
        f"- Final decision: {context.get('final_decision')}",
        f"- Average strategy vs benchmark pct: {context.get('avg_strategy_vs_benchmark_pct')}",
        f"- Threshold action: {context.get('threshold_action')}",
        "",
        "## Aggregate Bull Failure Diagnosis",
        "The configured bull gate requires average strategy-vs-benchmark excess return to be positive.",
        "The selected Step 34 bull threshold remains slightly negative on average, so the strict gate is still failed.",
        "",
        "## Symbol-Level Failure Contribution",
        "Symbols with negative excess return drag the bull average. This is diagnostic only and is not a recommendation to remove symbols.",
        "\n".join(top_drag_lines) if top_drag_lines else "_No negative symbol contributors were found._",
        "",
        "## Failure Reason Breakdown",
        f"- Failure reason rows: {len(reasons)}",
        "- See bull_failure_reasons.csv for aggregate, symbol, and data limitation reasons.",
        "",
        "## Data and Diagnostic Limitations",
        "- See bull_drilldown_limitations.csv for explicit limitations.",
        (
            "- Trade-level data is unavailable from current Step 34 outputs; deeper subperiod diagnosis requires trade-level or date-level output."
            if not trade_level_available
            else "- Trade-level data was detected for further diagnostics."
        ),
        "",
        "## Why This Does Not Change Trading-Ready Status",
        "This step is diagnostic only and does not upgrade any candidate.",
        "canonical_reduced_40 remains research_only_not_trading_ready because bull remediation remains unresolved.",
        "full remains baseline only and keep_core_only remains a low-feature challenger only.",
        "",
        "## Recommended Next Step",
        f"Recommended next step: {next_step}",
        "Keep the next step diagnostic-only unless explicitly scoped otherwise.",
        "",
        "## Educational / Research Disclaimer",
        "This report is educational/research diagnostics only. It is not financial advice.",
        "No strategy, model, threshold, symbol, or candidate in this report should be treated as deployable or trading-ready.",
        "",
    ]
    return "\n".join(sections)


def build_bull_regime_failure_drilldown(
    bull_dir: str | Path,
    integrated_dir: str | Path | None = None,
) -> dict[str, Any]:
    inputs = load_bull_drilldown_inputs(bull_dir, integrated_dir)
    selected = inputs["selected_symbol_rows"]
    trade_level_available = False
    subperiod_available = bool(
        not selected.empty
        and any(column in selected.columns for column in ["date", "window_id", "test_start", "test_end"])
    )
    threshold_context = build_bull_threshold_context(inputs)
    symbol_summary = build_bull_symbol_failure_summary(inputs)
    contribution = build_bull_failure_contribution(symbol_summary)
    reasons = build_bull_failure_reasons(
        threshold_context,
        symbol_summary,
        trade_level_available,
        subperiod_available,
    )
    limitations = build_bull_drilldown_limitations(
        trade_level_available,
        subperiod_available,
    )
    trade_level = build_trade_level_diagnostics_placeholder(trade_level_available)
    report = generate_report(
        threshold_context,
        symbol_summary,
        contribution,
        reasons,
        limitations,
        bull_dir,
        integrated_dir,
        trade_level_available,
    )
    return {
        "bull_threshold_context": threshold_context,
        "bull_symbol_failure_summary": symbol_summary,
        "bull_failure_contribution": contribution,
        "bull_failure_reasons": reasons,
        "bull_drilldown_limitations": limitations,
        "bull_trade_level_diagnostics": trade_level,
        "bull_regime_failure_drilldown_report": report,
        "trade_level_available": trade_level_available,
        "subperiod_available": subperiod_available,
        "input_warnings": inputs["input_warnings"],
    }


def generate_bull_regime_failure_drilldown(
    bull_dir: str | Path,
    integrated_dir: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    result = build_bull_regime_failure_drilldown(
        bull_dir=bull_dir,
        integrated_dir=integrated_dir,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    result["bull_symbol_failure_summary"].to_csv(paths["symbol_summary"], index=False)
    result["bull_failure_contribution"].to_csv(paths["contribution"], index=False)
    result["bull_failure_reasons"].to_csv(paths["reasons"], index=False)
    result["bull_threshold_context"].to_csv(paths["threshold_context"], index=False)
    result["bull_drilldown_limitations"].to_csv(paths["limitations"], index=False)
    result["bull_trade_level_diagnostics"].to_csv(paths["trade_level"], index=False)
    paths["report"].write_text(
        result["bull_regime_failure_drilldown_report"],
        encoding="utf-8",
    )
    run_config = {
        "bull_dir": str(bull_dir),
        "integrated_dir": str(integrated_dir) if integrated_dir else None,
        "output_dir": str(output_path),
        "trade_level_available": result["trade_level_available"],
        "subperiod_available": result["subperiod_available"],
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
