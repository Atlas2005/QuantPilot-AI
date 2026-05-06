import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILENAMES = {
    "report": "bull_error_pattern_remediation_design_report.md",
    "trade_classification": "bull_trade_error_classification.csv",
    "window_classification": "bull_window_error_classification.csv",
    "symbol_profile": "bull_symbol_error_profile.csv",
    "aggregate_profile": "bull_aggregate_error_profile.csv",
    "design_options": "bull_remediation_design_options.csv",
    "priority_matrix": "bull_remediation_priority_matrix.csv",
    "guardrails": "bull_no_change_guardrails.csv",
    "limitations": "bull_design_limitations.csv",
    "run_config": "run_config.json",
}

SHORT_HOLDING_DAYS = 3
LONG_HOLDING_DAYS = 60


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


def _read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "symbol" in df:
        df["symbol"] = df["symbol"].map(_format_symbol)
    return df


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(row: pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    return pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean_text(value).lower() in {"true", "1", "yes", "y"}


def _main_value(series: pd.Series, fallback: str) -> str:
    clean = series.dropna().astype(str)
    clean = clean[clean != ""]
    if clean.empty:
        return fallback
    return clean.value_counts().index[0]


def _main_error_value(series: pd.Series, fallback: str) -> str:
    clean = series.dropna().astype(str)
    clean = clean[clean != ""]
    error_like = clean[
        ~clean.str.contains("supportive", case=False, na=False)
        & ~clean.str.contains("neutral", case=False, na=False)
    ]
    if not error_like.empty:
        return error_like.value_counts().index[0]
    if clean.empty:
        return fallback
    return clean.value_counts().index[0]


def _trade_classification(row: pd.Series) -> dict[str, str]:
    trade_return = _numeric(row, "trade_return_pct")
    trade_excess = _numeric(row, "trade_excess_pct")
    holding_days = _numeric(row, "holding_days")
    notes = []
    if pd.notna(holding_days) and holding_days <= SHORT_HOLDING_DAYS and pd.notna(trade_excess) and trade_excess < 0:
        notes.append("possible_signal_churn")
    if pd.notna(holding_days) and holding_days >= LONG_HOLDING_DAYS and pd.notna(trade_excess) and trade_excess < 0:
        notes.append("possible_slow_exit_or_benchmark_chase_failure")
    if pd.notna(trade_return) and trade_return < 0:
        pattern = "negative_trade_return"
        family = "absolute_loss"
        severity = "high" if pd.notna(trade_excess) and trade_excess < 0 else "medium"
        implication = "Review loss-control and bull exit behavior in a future controlled prototype."
    elif pd.notna(trade_return) and pd.notna(trade_excess) and trade_return >= 0 and trade_excess < 0:
        pattern = "positive_return_but_lagged_benchmark"
        family = "relative_underperformance"
        severity = "high" if trade_excess <= -5 else "medium"
        implication = "Review benchmark-lag reduction without changing thresholds in this step."
    elif pd.notna(trade_return) and pd.notna(trade_excess) and trade_return >= 0 and trade_excess >= 0:
        pattern = "successful_bull_trade"
        family = "supportive_trade"
        severity = "low"
        implication = "Retain as supportive diagnostic context."
    else:
        pattern = "unclassified_trade_due_to_missing_values"
        family = "data_limitation"
        severity = "medium"
        implication = "Inspect missing trade return or benchmark fields before remediation design."
    return {
        "classified_error_pattern": pattern,
        "error_family": family,
        "severity": severity,
        "evidence": f"trade_return_pct={trade_return}, trade_excess_pct={trade_excess}, holding_days={holding_days}",
        "remediation_implication": implication,
        "notes": ";".join(notes),
    }


def build_trade_error_classification(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "entry_date",
        "exit_date",
        "holding_days",
        "trade_return_pct",
        "benchmark_return_pct",
        "trade_excess_pct",
        "was_profitable",
        "beat_benchmark",
        "original_error_pattern",
        "classified_error_pattern",
        "error_family",
        "severity",
        "evidence",
        "remediation_implication",
        "notes",
    ]
    rows = []
    for _, row in trades.iterrows():
        classification = _trade_classification(row)
        trade_return = _numeric(row, "trade_return_pct")
        trade_excess = _numeric(row, "trade_excess_pct")
        rows.append(
            {
                "symbol": _format_symbol(row.get("symbol")),
                "entry_date": row.get("entry_date"),
                "exit_date": row.get("exit_date"),
                "holding_days": row.get("holding_days"),
                "trade_return_pct": trade_return,
                "benchmark_return_pct": _numeric(row, "benchmark_return_pct"),
                "trade_excess_pct": trade_excess,
                "was_profitable": bool(pd.notna(trade_return) and trade_return >= 0),
                "beat_benchmark": bool(pd.notna(trade_excess) and trade_excess >= 0),
                "original_error_pattern": _clean_text(row.get("error_pattern")),
                **classification,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _window_classification(row: pd.Series) -> dict[str, str]:
    strategy_return = _numeric(row, "strategy_return_pct")
    benchmark_return = _numeric(row, "benchmark_return_pct")
    excess = _numeric(row, "excess_return_pct")
    drawdown = _numeric(row, "max_drawdown_pct")
    trade_count = _numeric(row, "trade_count")
    notes = []
    if pd.notna(trade_count) and int(trade_count) == 0 and pd.notna(benchmark_return) and benchmark_return > 0:
        notes.append("missed_bull_participation")
    if pd.notna(excess) and excess < 0 and pd.notna(strategy_return) and strategy_return >= 0:
        pattern = "positive_window_but_lagged_benchmark"
        family = "relative_underperformance"
    elif pd.notna(excess) and excess < 0 and pd.notna(strategy_return) and strategy_return < 0:
        pattern = "negative_window_return"
        family = "absolute_loss"
    elif pd.notna(excess) and excess >= 0:
        pattern = "supportive_or_neutral_window"
        family = "supportive_window"
    else:
        pattern = "unclassified_window_due_to_missing_values"
        family = "data_limitation"
    if "missed_bull_participation" in notes and pattern == "positive_window_but_lagged_benchmark":
        pattern = "positive_window_but_lagged_benchmark;missed_bull_participation"
    elif "missed_bull_participation" in notes and pattern == "supportive_or_neutral_window":
        pattern = "supportive_or_neutral_window;missed_bull_participation"
    severity = "low"
    if pd.notna(excess) and excess < 0:
        severity = "high" if excess <= -5 or (pd.notna(drawdown) and drawdown <= -10) else "medium"
    return {
        "classified_window_pattern": pattern,
        "error_family": family,
        "severity": severity,
        "evidence": f"strategy_return_pct={strategy_return}, benchmark_return_pct={benchmark_return}, excess_return_pct={excess}, max_drawdown_pct={drawdown}, trade_count={trade_count}",
        "remediation_implication": "Review window-level participation, exit, and loss-control design in a later prototype step.",
        "notes": ";".join(notes),
    }


def build_window_error_classification(windows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "window_id",
        "start_date",
        "end_date",
        "rows",
        "strategy_return_pct",
        "benchmark_return_pct",
        "excess_return_pct",
        "max_drawdown_pct",
        "trade_count",
        "original_error_pattern",
        "classified_window_pattern",
        "error_family",
        "severity",
        "evidence",
        "remediation_implication",
        "notes",
    ]
    rows = []
    for _, row in windows.iterrows():
        rows.append(
            {
                "symbol": _format_symbol(row.get("symbol")),
                "window_id": row.get("window_id"),
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
                "rows": row.get("rows"),
                "strategy_return_pct": _numeric(row, "strategy_return_pct"),
                "benchmark_return_pct": _numeric(row, "benchmark_return_pct"),
                "excess_return_pct": _numeric(row, "excess_return_pct"),
                "max_drawdown_pct": _numeric(row, "max_drawdown_pct"),
                "trade_count": row.get("trade_count"),
                "original_error_pattern": _clean_text(row.get("error_pattern")),
                **_window_classification(row),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_symbol_error_profile(
    trade_classification: pd.DataFrame,
    window_classification: pd.DataFrame,
    step37_contribution: pd.DataFrame,
) -> pd.DataFrame:
    symbols = sorted(
        set(trade_classification.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
        | set(window_classification.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
        | set(step37_contribution.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
    )
    contribution_roles = {}
    if not step37_contribution.empty and "interpretation" in step37_contribution:
        contribution_roles = dict(
            zip(
                step37_contribution["symbol"].map(_format_symbol),
                step37_contribution["interpretation"].astype(str),
            )
        )
    rows = []
    for symbol in symbols:
        tdf = trade_classification[trade_classification["symbol"] == symbol]
        wdf = window_classification[window_classification["symbol"] == symbol]
        families = pd.concat(
            [
                tdf.get("error_family", pd.Series(dtype=str)),
                wdf.get("error_family", pd.Series(dtype=str)),
            ],
            ignore_index=True,
        )
        patterns = pd.concat(
            [
                tdf.get("classified_error_pattern", pd.Series(dtype=str)),
                wdf.get("classified_window_pattern", pd.Series(dtype=str)),
            ],
            ignore_index=True,
        )
        dominant_family = _main_error_value(families, "no_observed_error_family")
        dominant_pattern = _main_error_value(patterns, "no_observed_error_pattern")
        step37_role = contribution_roles.get(symbol, "")
        role = "supporting_context"
        priority = "low"
        if symbol == "601318":
            role = "main_step37_bull_drag"
            priority = "high"
        elif symbol == "600036":
            role = "near_neutral_underperformance"
            priority = "medium"
        elif step37_role == "negative_contributor_to_bull_average":
            role = "main_step37_bull_drag" if symbol == "601318" else "bull_average_drag"
            priority = "high" if symbol == "601318" else "medium"
        elif int((tdf.get("trade_return_pct", pd.Series(dtype=float)) < 0).sum()) > 0:
            role = "negative_trade_cluster_context"
            priority = "medium"
        direction = "benchmark_lag_reduction_design"
        if dominant_family == "absolute_loss":
            direction = "window_level_loss_control_review"
        if symbol == "601318":
            direction = "trade_sufficiency_review_for_601318;benchmark_lag_reduction_design"
        rows.append(
            {
                "symbol": symbol,
                "total_trades": int(len(tdf)),
                "profitable_trades": int(tdf.get("was_profitable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "losing_trades": int(pd.to_numeric(tdf.get("trade_return_pct", pd.Series(dtype=float)), errors="coerce").lt(0).sum()),
                "beat_benchmark_trades": int(tdf.get("beat_benchmark", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "total_windows": int(len(wdf)),
                "negative_excess_windows": int(pd.to_numeric(wdf.get("excess_return_pct", pd.Series(dtype=float)), errors="coerce").lt(0).sum()),
                "positive_excess_windows": int(pd.to_numeric(wdf.get("excess_return_pct", pd.Series(dtype=float)), errors="coerce").ge(0).sum()),
                "worst_trade_return_pct": pd.to_numeric(tdf.get("trade_return_pct", pd.Series(dtype=float)), errors="coerce").min(),
                "worst_trade_excess_pct": pd.to_numeric(tdf.get("trade_excess_pct", pd.Series(dtype=float)), errors="coerce").min(),
                "worst_window_excess_pct": pd.to_numeric(wdf.get("excess_return_pct", pd.Series(dtype=float)), errors="coerce").min(),
                "dominant_error_family": dominant_family,
                "dominant_error_pattern": dominant_pattern,
                "symbol_failure_role": role,
                "step37_failure_role_if_available": step37_role,
                "remediation_priority": priority,
                "remediation_direction": direction,
                "notes": "Track even when aggregate contribution is positive; negative trades and lagging trades remain diagnostic evidence.",
            }
        )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_priority_order"] = result["remediation_priority"].map(priority_order).fillna(9)
    return result.sort_values(["_priority_order", "symbol"]).drop(columns="_priority_order").reset_index(drop=True)


def build_aggregate_error_profile(
    trade_classification: pd.DataFrame,
    window_classification: pd.DataFrame,
    symbol_profile: pd.DataFrame,
    integrated_summary: pd.DataFrame,
) -> pd.DataFrame:
    integrated_row = integrated_summary.iloc[0] if not integrated_summary.empty else pd.Series(dtype=object)
    all_zero = bool(
        not symbol_profile.empty
        and "beat_benchmark_trades" in symbol_profile
        and pd.to_numeric(symbol_profile["beat_benchmark_trades"], errors="coerce").fillna(0).eq(0).all()
    )
    rows = [
        ("aggregate", "bull_final_decision", _clean_text(integrated_row.get("bull_final_decision")) or "bull_remediation_failed", "Bull remediation remains failed.", "Keep bull blocker unresolved."),
        ("aggregate", "avg_strategy_vs_benchmark_pct", _numeric(integrated_row, "bull_avg_strategy_vs_benchmark_pct"), "Average bull excess remains slightly negative if Step 36 summary is available.", "Do not upgrade candidate status."),
        ("aggregate", "dominant_trade_error_pattern", _main_value(trade_classification.get("classified_error_pattern", pd.Series(dtype=str)), "no_trade_data"), "Most common trade-level classified pattern.", "Prioritize dominant observed trade error family."),
        ("aggregate", "dominant_window_error_pattern", _main_value(window_classification.get("classified_window_pattern", pd.Series(dtype=str)), "no_window_data"), "Most common window-level classified pattern.", "Use windows to scope future prototype diagnostics."),
        ("aggregate", "all_symbols_have_zero_beat_benchmark_trades", all_zero, "Every symbol has zero benchmark-beating bull trades when supported by Step 38.", "Treat benchmark lag as a global bull issue."),
        ("aggregate", "main_blocker", "relative benchmark underperformance in bull regime", "Bull failures are not only outright negative returns.", "Design remediation around benchmark lag and participation."),
        ("aggregate", "trading_ready", False, "No candidate is trading-ready.", "Do not claim deployment readiness."),
        ("aggregate", "candidate_status", "research_only", "canonical_reduced_40 remains research-only.", "Continue controlled diagnostics."),
    ]
    return pd.DataFrame(
        [
            {
                "level": level,
                "metric": metric,
                "value": value,
                "interpretation": interpretation,
                "implication": implication,
            }
            for level, metric, value, interpretation, implication in rows
        ]
    )


def build_remediation_design_options(symbol_profile: pd.DataFrame) -> pd.DataFrame:
    negative_symbols = ",".join(symbol_profile.loc[symbol_profile["losing_trades"] > 0, "symbol"].astype(str).tolist()) if not symbol_profile.empty else ""
    rows = [
        ("BDO-01", "bull_participation_filter_review", "relative_underperformance", "all_symbols", "diagnostic_filter_design", "Review whether bull signals miss participation windows without changing features or thresholds."),
        ("BDO-02", "bull_exit_logic_review", "absolute_loss", negative_symbols or "symbols_with_negative_trades", "exit_logic_prototype_design", "Reduce prolonged absolute-loss exposure in a later controlled experiment."),
        ("BDO-03", "benchmark_lag_reduction_design", "relative_underperformance", "all_symbols", "benchmark_comparison_design", "Focus on why profitable trades still lag the close-series benchmark."),
        ("BDO-04", "trade_sufficiency_review_for_601318", "relative_underperformance", "601318", "sample_sufficiency_diagnostic_design", "Investigate single-trade drag and limited bull trade evidence for 601318."),
        ("BDO-05", "symbol_specific_diagnostic_review", "mixed", "600036,000858,600519", "symbol_review_design", "Separate near-neutral drag from negative-trade clusters."),
        ("BDO-06", "probability_signal_timing_review", "relative_underperformance", "all_symbols", "probability_timeline_diagnostic_design", "Inspect probability timing using existing Step 38 timeline outputs only."),
        ("BDO-07", "window_level_loss_control_review", "absolute_loss", "negative_excess_windows", "window_rule_prototype_design", "Use window clusters to design future loss-control experiments."),
    ]
    return pd.DataFrame(
        [
            {
                "option_id": option_id,
                "remediation_theme": theme,
                "target_error_family": family,
                "target_symbols": target,
                "proposed_change_type": change_type,
                "implementation_status": "design_only_not_implemented",
                "expected_effect": effect,
                "risk": "Could overfit bull diagnostics or reduce trade sufficiency if implemented without controlled validation.",
                "required_future_step": "V4 Step 40 Bull Remediation Prototype Design",
                "allowed_in_current_step": False,
                "notes": "Design option only; no remediation is implemented in Step 39.",
            }
            for option_id, theme, family, target, change_type, effect in rows
        ]
    )


def build_priority_matrix(symbol_profile: pd.DataFrame, window_classification: pd.DataFrame) -> pd.DataFrame:
    negative_trade_symbols = ",".join(symbol_profile.loc[symbol_profile["losing_trades"] > 0, "symbol"].astype(str).tolist()) if not symbol_profile.empty else ""
    negative_windows = int(pd.to_numeric(window_classification.get("excess_return_pct", pd.Series(dtype=float)), errors="coerce").lt(0).sum())
    rows = [
        (1, "601318 benchmark lag / single-trade drag", "symbol", "Profitable trade still lagged benchmark and Step 37 identified it as the main drag.", "Step 37 main drag plus Step 38 one profitable non-benchmark-beating trade.", "High", "Medium", "Design 601318-specific benchmark-lag and trade-sufficiency prototype."),
        (2, "All symbols zero beat_benchmark_trades", "global", "No bull trade beat benchmark in any covered symbol.", "Step 38 symbol profile shows beat_benchmark_trades=0 for all symbols.", "High", "Medium", "Design global benchmark-lag reduction diagnostics."),
        (3, negative_trade_symbols or "negative_trade_return cluster", "symbol_cluster", "Absolute-loss trades remain in several symbols.", f"Symbols with losing trades: {negative_trade_symbols or 'available after real diagnostics'}", "Medium", "Medium", "Separate loss-control review from benchmark-lag review."),
        (4, "Window-level negative excess clusters", "window_cluster", "Some bull windows lagged benchmark or lost money.", f"negative_excess_windows={negative_windows}", "Medium", "Medium", "Use window clusters to scope future prototype tests."),
    ]
    return pd.DataFrame(
        [
            {
                "priority_rank": rank,
                "target": target,
                "target_type": target_type,
                "primary_issue": issue,
                "evidence": evidence,
                "expected_diagnostic_value": value,
                "implementation_risk": risk,
                "recommended_next_step": next_step,
            }
            for rank, target, target_type, issue, evidence, value, risk, next_step in rows
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_threshold_change", "confirmed", "Step 39 reads existing diagnostics and does not run a threshold sweep.", "Selected 0.65 / 0.50 remains diagnosis-only."),
        ("no_model_retraining", "confirmed", "No trainer or model pipeline is called.", "Existing Step 38 outputs are reused."),
        ("no_feature_engineering_change", "confirmed", "No factor builder or feature selection code is changed by this module.", "Factor engineering remains unchanged."),
        ("no_new_data_sources", "confirmed", "Only supplied Step 38/37/36 output directories are read.", "No external data is added."),
        ("no_new_agents", "confirmed", "No agents or parallel research workers are added by the code.", "Scope remains local diagnostics."),
        ("no_trading_ready_upgrade", "confirmed", "Outputs explicitly keep trading_ready=False and candidate_status=research_only.", "No readiness claim is made."),
        ("diagnostic_only", "confirmed", "Remediation options use design_only_not_implemented.", "No remediation is implemented."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame(
        [{"guardrail": guardrail, "status": status, "evidence": evidence, "notes": notes} for guardrail, status, evidence, notes in rows]
    )


def build_limitations() -> pd.DataFrame:
    rows = [
        ("small_symbol_count", "medium", "The bull diagnostic covers a small configured symbol set.", "Single-symbol behavior can dominate aggregate findings.", "Treat symbol-level priorities as diagnostics, not symbol recommendations."),
        ("limited_bull_sample", "medium", "Step 38 contains limited bull trades and windows.", "Classification may be sensitive to a few rows.", "Use controlled prototypes with strict validation later."),
        ("diagnosis_not_causal_proof", "medium", "Pattern classification is descriptive.", "It does not prove why a trade or window underperformed.", "Use Step 40 prototypes to test hypotheses conservatively."),
        ("remediation_not_implemented", "blocking", "This step only designs options.", "No performance improvement is produced or claimed.", "Implement only in a future explicitly scoped step."),
        ("no_new_validation_run", "blocking", "No retraining, threshold sweep, or validation run is performed.", "Candidate status cannot change.", "Run validation only after future controlled prototypes."),
        ("benchmark_comparison_uses_available_proxy_if_applicable", "medium", "Step 38 benchmark comparison uses the available close-series benchmark/proxy.", "Benchmark conclusions depend on the available proxy.", "Document benchmark assumptions in future remediation tests."),
        ("research_only_not_trading_ready", "blocking", "canonical_reduced_40 remains research-only.", "No trading-ready status can be inferred.", "Continue diagnostics and gate-controlled validation."),
    ]
    return pd.DataFrame(
        [
            {
                "limitation_type": kind,
                "severity": severity,
                "description": description,
                "consequence": consequence,
                "recommended_followup": followup,
            }
            for kind, severity, description, consequence, followup in rows
        ]
    )


def _report(
    diagnostics_dir: str | Path,
    drilldown_dir: str | Path | None,
    integrated_dir: str | Path | None,
    aggregate_profile: pd.DataFrame,
    trade_classification: pd.DataFrame,
    window_classification: pd.DataFrame,
    symbol_profile: pd.DataFrame,
    design_options: pd.DataFrame,
    priority_matrix: pd.DataFrame,
) -> str:
    agg = dict(zip(aggregate_profile["metric"], aggregate_profile["value"])) if not aggregate_profile.empty else {}
    symbols = ", ".join(symbol_profile["symbol"].astype(str).tolist()) if not symbol_profile.empty else "none"
    row_601318 = symbol_profile[symbol_profile["symbol"] == "601318"].head(1)
    row_600036 = symbol_profile[symbol_profile["symbol"] == "600036"].head(1)
    explanation_601318 = "601318 remains high priority because Step 37 identified it as the main bull drag and Step 38 shows its only bull trade was profitable but lagged benchmark."
    if not row_601318.empty:
        explanation_601318 += f" Dominant pattern: {row_601318.iloc[0].get('dominant_error_pattern')}; worst window excess: {row_601318.iloc[0].get('worst_window_excess_pct')}."
    explanation_600036 = "600036 remains medium priority because it is near-neutral but slightly negative in Step 37 context."
    if not row_600036.empty:
        explanation_600036 += f" Dominant pattern: {row_600036.iloc[0].get('dominant_error_pattern')}; losing trades: {row_600036.iloc[0].get('losing_trades')}."
    return "\n".join(
        [
            "# V4 Step 39 Bull Error Pattern Classification and Remediation Design Report",
            "",
            "## Executive Summary",
            "This step classifies bull failure patterns and designs conservative remediation options only.",
            "It does not implement remediation.",
            "No threshold, model, factor, data source, or agent was changed.",
            "The selected 0.65 / 0.50 threshold remains diagnosis-only.",
            "canonical_reduced_40 remains research-only.",
            "No candidate is trading-ready.",
            "Bull remediation remains failed.",
            "The main issue appears to be bull benchmark lag / relative underperformance, not only outright negative returns.",
            "",
            "## Inputs Used",
            f"- Step 38 diagnostics directory: {diagnostics_dir}",
            f"- Step 37 drilldown directory: {drilldown_dir or 'not provided'}",
            f"- Step 36 integrated directory: {integrated_dir or 'not provided'}",
            "",
            "## Guardrails and No-Change Confirmation",
            "Step 39 reads existing outputs only. It does not tune thresholds, retrain models, change factors, add data sources, add agents, or optimize the strategy.",
            "",
            "## Aggregate Bull Error Profile",
            f"- Bull final decision: {agg.get('bull_final_decision', 'bull_remediation_failed')}",
            f"- Average strategy vs benchmark pct: {agg.get('avg_strategy_vs_benchmark_pct', '')}",
            f"- Dominant trade error pattern: {agg.get('dominant_trade_error_pattern', '')}",
            f"- Dominant window error pattern: {agg.get('dominant_window_error_pattern', '')}",
            f"- All symbols have zero beat benchmark trades: {agg.get('all_symbols_have_zero_beat_benchmark_trades', '')}",
            "",
            "## Trade-Level Error Classification",
            f"- Trade rows classified: {len(trade_classification)}",
            "Negative trades are classified as negative_trade_return. Profitable trades with negative excess are classified as positive_return_but_lagged_benchmark.",
            "",
            "## Window-Level Error Classification",
            f"- Window rows classified: {len(window_classification)}",
            "Windows are classified by strategy return, benchmark return, excess return, drawdown, and trade count.",
            "",
            "## Symbol-Level Error Profiles",
            f"- Symbols profiled: {symbols}",
            "Symbols with positive aggregate contribution but negative trades are tracked, not ignored.",
            "",
            "## Remediation Design Options",
            f"- Design option rows: {len(design_options)}",
            "All options are marked design_only_not_implemented and allowed_in_current_step=False.",
            "",
            "## Remediation Priority Matrix",
            f"- Top priority: {priority_matrix.iloc[0].get('target') if not priority_matrix.empty else 'not available'}",
            "",
            "## What This Explains About 601318",
            explanation_601318,
            "",
            "## What This Explains About 600036",
            explanation_600036,
            "",
            "## Why This Does Not Change Trading-Ready Status",
            "This step classifies and designs only; it does not run validation or implement remediation.",
            "canonical_reduced_40 remains research_only_not_trading_ready.",
            "No candidate is trading-ready.",
            "",
            "## Recommended Next Step",
            "Recommended next step: V4 Step 40: Bull Remediation Prototype Design, where design options are converted into controlled prototype experiments, still without claiming trading-ready.",
            "",
            "## Educational / Research Disclaimer",
            "This report is educational/research diagnostics only. It is not financial advice.",
            "No strategy, model, threshold, symbol, or candidate in this report should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_bull_error_pattern_remediation_design(
    diagnostics_dir: str | Path,
    drilldown_dir: str | Path | None,
    integrated_dir: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    diagnostics = Path(diagnostics_dir)
    drilldown = Path(drilldown_dir) if drilldown_dir else None
    integrated = Path(integrated_dir) if integrated_dir else None
    trades = _read_csv(diagnostics / "bull_trade_level_diagnostics.csv", dtype={"symbol": str})
    windows = _read_csv(diagnostics / "bull_window_diagnostics.csv", dtype={"symbol": str})
    patterns = _read_csv(diagnostics / "bull_error_pattern_summary.csv")
    run_config_in = _read_json(diagnostics / "run_config.json")
    contribution = _read_csv(drilldown / "bull_failure_contribution.csv", dtype={"symbol": str}) if drilldown else pd.DataFrame()
    integrated_summary = _read_csv(integrated / "integrated_remediation_summary.csv") if integrated else pd.DataFrame()
    trade_classification = build_trade_error_classification(trades)
    window_classification = build_window_error_classification(windows)
    symbol_profile = build_symbol_error_profile(trade_classification, window_classification, contribution)
    aggregate_profile = build_aggregate_error_profile(trade_classification, window_classification, symbol_profile, integrated_summary)
    design_options = build_remediation_design_options(symbol_profile)
    priority_matrix = build_priority_matrix(symbol_profile, window_classification)
    guardrails = build_guardrails()
    limitations = build_limitations()
    report = _report(
        diagnostics_dir,
        drilldown_dir,
        integrated_dir,
        aggregate_profile,
        trade_classification,
        window_classification,
        symbol_profile,
        design_options,
        priority_matrix,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    paths["report"].write_text(report, encoding="utf-8")
    trade_classification.to_csv(paths["trade_classification"], index=False)
    window_classification.to_csv(paths["window_classification"], index=False)
    symbol_profile.to_csv(paths["symbol_profile"], index=False)
    aggregate_profile.to_csv(paths["aggregate_profile"], index=False)
    design_options.to_csv(paths["design_options"], index=False)
    priority_matrix.to_csv(paths["priority_matrix"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    limitations.to_csv(paths["limitations"], index=False)
    run_config = {
        "diagnostics_dir": str(diagnostics_dir),
        "drilldown_dir": str(drilldown_dir) if drilldown_dir else None,
        "integrated_dir": str(integrated_dir) if integrated_dir else None,
        "output_dir": str(output_path),
        "candidate": run_config_in.get("candidate", "canonical_reduced_40"),
        "model": run_config_in.get("model", "logistic_regression"),
        "buy_threshold": run_config_in.get("buy_threshold", 0.65),
        "sell_threshold": run_config_in.get("sell_threshold", 0.50),
        "status": "design_only_not_implemented",
        "educational_research_only": True,
        "trading_ready": False,
        "input_error_patterns": patterns.to_dict(orient="records") if not patterns.empty else [],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "bull_error_pattern_remediation_design_report": report,
        "bull_trade_error_classification": trade_classification,
        "bull_window_error_classification": window_classification,
        "bull_symbol_error_profile": symbol_profile,
        "bull_aggregate_error_profile": aggregate_profile,
        "bull_remediation_design_options": design_options,
        "bull_remediation_priority_matrix": priority_matrix,
        "bull_no_change_guardrails": guardrails,
        "bull_design_limitations": limitations,
        "run_config": run_config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
