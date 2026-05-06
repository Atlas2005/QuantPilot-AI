import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILENAMES = {
    "report": "bull_remediation_prototype_design_report.md",
    "specs": "bull_prototype_experiment_specs.csv",
    "metric_plan": "bull_prototype_metric_plan.csv",
    "guardrails": "bull_prototype_guardrails.csv",
    "risk_assessment": "bull_prototype_risk_assessment.csv",
    "execution_plan": "bull_prototype_execution_plan.csv",
    "not_implemented_log": "bull_prototype_not_implemented_log.csv",
    "priority_ranking": "bull_prototype_priority_ranking.csv",
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


def _first_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    return df.iloc[0]


def _numeric(row: pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    return pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]


def _metric_value(metrics: pd.DataFrame, name: str) -> str:
    if metrics.empty or "metric" not in metrics or "value" not in metrics:
        return "unavailable"
    matches = metrics[metrics["metric"].astype(str) == name]
    if matches.empty:
        return "unavailable"
    value = matches.iloc[0].get("value")
    return _clean_text(value) or "unavailable"


def _symbol_list(symbol_profile: pd.DataFrame, condition_column: str | None = None) -> str:
    if symbol_profile.empty or "symbol" not in symbol_profile:
        return "unavailable"
    df = symbol_profile.copy()
    if condition_column and condition_column in df:
        df = df[pd.to_numeric(df[condition_column], errors="coerce").fillna(0) > 0]
    values = df["symbol"].map(_format_symbol).dropna().astype(str).tolist()
    return ",".join(values) if values else "none"


def build_prototype_experiment_specs(
    design_options: pd.DataFrame,
    symbol_profile: pd.DataFrame,
) -> pd.DataFrame:
    negative_symbols = _symbol_list(symbol_profile, "losing_trades")
    option_by_theme = {}
    if not design_options.empty and "remediation_theme" in design_options:
        option_by_theme = {
            _clean_text(row.get("remediation_theme")): row
            for _, row in design_options.iterrows()
        }

    def source(theme: str, column: str, fallback: str) -> str:
        row = option_by_theme.get(theme)
        if row is None:
            return fallback
        return _clean_text(row.get(column)) or fallback

    rows = [
        {
            "prototype_id": "BP-001",
            "prototype_name": "benchmark_lag_reduction_for_601318",
            "source_step39_theme": "benchmark_lag_reduction_design",
            "target_error_family": "relative_underperformance",
            "target_symbols": "601318",
            "hypothesis": "601318 bull failure is mainly benchmark lag rather than absolute loss, so a future prototype should test whether participation or exit timing can reduce excess-return drag without changing the selected threshold.",
            "proposed_change_summary": "Design an isolated 601318 benchmark-lag prototype using existing Step 38 trade/window diagnostics as baseline context.",
            "change_type": source("benchmark_lag_reduction_design", "proposed_change_type", "benchmark_comparison_design"),
            "expected_effect": "Reduce 601318 relative underperformance in a controlled future experiment.",
            "key_risks": "Small sample and single-trade evidence can create false confidence.",
            "required_inputs": "Step 38 trade/window diagnostics; Step 39 symbol profile; unchanged Step 34 threshold context.",
        },
        {
            "prototype_id": "BP-002",
            "prototype_name": "bull_exit_logic_review_prototype",
            "source_step39_theme": "bull_exit_logic_review",
            "target_error_family": "absolute_loss",
            "target_symbols": negative_symbols,
            "hypothesis": "Negative bull trades in 000858, 600519, and 600036 may need a future exit-logic review, but any change must preserve trade sufficiency and benchmark comparison discipline.",
            "proposed_change_summary": "Specify future exit-rule prototype candidates for negative-trade clusters without adding implementation logic in Step 40.",
            "change_type": source("bull_exit_logic_review", "proposed_change_type", "exit_logic_prototype_design"),
            "expected_effect": "Reduce negative_trade_return count in future controlled prototypes.",
            "key_risks": "Exit changes can reduce participation or improve absolute returns while still lagging benchmark.",
            "required_inputs": "Step 38 trade diagnostics; Step 39 negative-trade cluster profile.",
        },
        {
            "prototype_id": "BP-003",
            "prototype_name": "bull_participation_filter_review_prototype",
            "source_step39_theme": "bull_participation_filter_review",
            "target_error_family": "relative_underperformance",
            "target_symbols": "all_symbols",
            "hypothesis": "The global zero beat_benchmark_trades issue may reflect participation timing weakness during bull windows.",
            "proposed_change_summary": "Design participation-filter review specs for future isolated experiments while keeping the selected 0.65 / 0.50 threshold fixed.",
            "change_type": source("bull_participation_filter_review", "proposed_change_type", "diagnostic_filter_design"),
            "expected_effect": "Increase benchmark participation quality in future prototype tests.",
            "key_risks": "Filters can overfit five symbols or reduce trade count below sufficiency.",
            "required_inputs": "Step 38 signal timeline and window diagnostics; Step 39 aggregate profile.",
        },
        {
            "prototype_id": "BP-004",
            "prototype_name": "trade_sufficiency_review_for_601318",
            "source_step39_theme": "trade_sufficiency_review_for_601318",
            "target_error_family": "relative_underperformance",
            "target_symbols": "601318",
            "hypothesis": "601318 has limited bull trade evidence, so a future prototype should first test whether any remediation remains valid under trade-sufficiency constraints.",
            "proposed_change_summary": "Define 601318 trade-sufficiency diagnostics before any performance-oriented prototype is considered.",
            "change_type": source("trade_sufficiency_review_for_601318", "proposed_change_type", "sample_sufficiency_diagnostic_design"),
            "expected_effect": "Prevent overinterpreting a single profitable but benchmark-lagging trade.",
            "key_risks": "A narrow 601318-specific design could fail to generalize.",
            "required_inputs": "Step 38 trade count; Step 39 601318 profile.",
        },
        {
            "prototype_id": "BP-005",
            "prototype_name": "negative_trade_cluster_review_for_000858_600519_600036",
            "source_step39_theme": "symbol_specific_diagnostic_review",
            "target_error_family": "absolute_loss",
            "target_symbols": "000858,600519,600036",
            "hypothesis": "The negative-trade cluster should be reviewed separately from benchmark-lag trades so future prototypes do not mix absolute-loss and relative-underperformance problems.",
            "proposed_change_summary": "Define symbol-cluster experiment specs for absolute-loss diagnostics only.",
            "change_type": source("symbol_specific_diagnostic_review", "proposed_change_type", "symbol_review_design"),
            "expected_effect": "Separate loss-control diagnostics from benchmark-lag diagnostics in future prototypes.",
            "key_risks": "Cluster-specific changes can improve one group while hurting the full symbol set.",
            "required_inputs": "Step 38 trade diagnostics; Step 39 symbol error profile.",
        },
        {
            "prototype_id": "BP-006",
            "prototype_name": "probability_signal_timing_review_prototype",
            "source_step39_theme": "probability_signal_timing_review",
            "target_error_family": "relative_underperformance",
            "target_symbols": "all_symbols",
            "hypothesis": "Existing probability timelines may explain signal timing weaknesses without changing model training or thresholds.",
            "proposed_change_summary": "Specify future probability-timing diagnostics using existing Step 38 probability outputs only.",
            "change_type": source("probability_signal_timing_review", "proposed_change_type", "probability_timeline_diagnostic_design"),
            "expected_effect": "Identify whether probability timing should be instrumented in a later harness.",
            "key_risks": "Timing diagnostics can drift into threshold tuning if guardrails are not enforced.",
            "required_inputs": "Step 38 signal timeline diagnostics.",
        },
        {
            "prototype_id": "BP-007",
            "prototype_name": "window_level_loss_control_review_prototype",
            "source_step39_theme": "window_level_loss_control_review",
            "target_error_family": "absolute_loss",
            "target_symbols": "negative_excess_windows",
            "hypothesis": "Window-level negative excess clusters can guide future loss-control prototypes without changing factor engineering.",
            "proposed_change_summary": "Define future window-level loss-control review specs using fixed Step 38 windows as baseline context.",
            "change_type": source("window_level_loss_control_review", "proposed_change_type", "window_rule_prototype_design"),
            "expected_effect": "Reduce worst window excess in future controlled prototypes.",
            "key_risks": "Window rules can overfit fixed 20-row windows and miss out-of-window behavior.",
            "required_inputs": "Step 38 window diagnostics; Step 39 priority matrix.",
        },
    ]
    for row in rows:
        row.update(
            {
                "implementation_status": "prototype_design_only",
                "execution_status": "not_executed",
                "allowed_in_current_step": False,
                "success_metric_primary": "avg_strategy_vs_benchmark_pct",
                "failure_metric_primary": "beat_benchmark_trades",
                "notes": "Specification only. No prototype is executed in Step 40.",
            }
        )
    return pd.DataFrame(rows)


def build_metric_plan(
    aggregate_profile: pd.DataFrame,
    symbol_profile: pd.DataFrame,
    trades: pd.DataFrame,
    windows: pd.DataFrame,
    integrated_summary: pd.DataFrame,
) -> pd.DataFrame:
    integrated = _first_row(integrated_summary)
    total_trade_count = len(trades) if not trades.empty else "unavailable"
    negative_trade_count = (
        int(pd.to_numeric(trades.get("trade_return_pct", pd.Series(dtype=float)), errors="coerce").lt(0).sum())
        if not trades.empty
        else "unavailable"
    )
    lagged_count = (
        int((trades.get("error_pattern", pd.Series(dtype=str)).astype(str) == "positive_return_but_lagged_benchmark").sum())
        if not trades.empty and "error_pattern" in trades
        else _metric_value(aggregate_profile, "dominant_trade_error_pattern")
    )
    beat_trades = (
        int(trades.get("beat_benchmark", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if not trades.empty and "beat_benchmark" in trades
        else "unavailable"
    )
    worst_trade_excess = (
        pd.to_numeric(trades.get("trade_excess_pct", pd.Series(dtype=float)), errors="coerce").min()
        if not trades.empty
        else "unavailable"
    )
    worst_window_excess = (
        pd.to_numeric(windows.get("excess_return_pct", pd.Series(dtype=float)), errors="coerce").min()
        if not windows.empty
        else "unavailable"
    )
    rows = [
        ("avg_strategy_vs_benchmark_pct", "aggregate", "higher_is_better", _numeric(integrated, "bull_avg_strategy_vs_benchmark_pct"), "Future prototype must improve versus unchanged Step 34 baseline and pass strict aggregate gate only in a later validation step.", "Primary bull benchmark-excess blocker.", "Value remains <= baseline or improves absolute return without excess improvement.", "Baseline from Step 36 if available."),
        ("beat_benchmark_rate", "aggregate", "higher_is_better", _numeric(integrated, "bull_beat_benchmark_rate"), "Future prototype should not reduce the configured research gate context.", "Captures benchmark comparison breadth.", "Rate deteriorates or benchmark comparison remains weak.", "Baseline from Step 36 if available."),
        ("sufficient_trade_rate", "aggregate", "higher_or_equal_is_better", _numeric(integrated, "bull_sufficient_trade_rate"), "Future prototype should preserve trade sufficiency.", "Prevents over-filtering.", "Trade sufficiency falls below existing context.", "Baseline from Step 36 if available."),
        ("trade_count", "trade", "higher_or_equal_with_quality", total_trade_count, "Future prototype should not gain performance by eliminating too many trades.", "Tracks trade sufficiency and sample size.", "Trade count becomes too small to interpret.", "Baseline from Step 38 trade diagnostics if available."),
        ("beat_benchmark_trades", "trade", "higher_is_better", beat_trades, "Future prototype should increase benchmark-beating trades without overfitting.", "Directly addresses zero beat_benchmark_trades issue.", "No increase or improvement only in one symbol while aggregate weakens.", "Baseline from Step 38 trades if available."),
        ("negative_trade_count", "trade", "lower_is_better", negative_trade_count, "Future prototype should reduce negative_trade_return count without sacrificing benchmark excess.", "Tracks absolute-loss cluster.", "Negative trades persist or benchmark excess worsens.", "Baseline from Step 38 trades if available."),
        ("positive_return_but_lagged_benchmark_count", "trade", "lower_is_better", lagged_count, "Future prototype should reduce profitable benchmark-lagging trades.", "Tracks relative underperformance.", "Lagged profitable trades remain dominant.", "Baseline from Step 38 error_pattern values if available."),
        ("worst_trade_excess_pct", "trade", "higher_is_better", worst_trade_excess, "Future prototype should improve worst trade excess.", "Captures tail trade-level benchmark lag.", "Worst trade excess deteriorates.", "Baseline from Step 38 trades if available."),
        ("worst_window_excess_pct", "window", "higher_is_better", worst_window_excess, "Future prototype should improve worst window excess.", "Captures window-level cluster risk.", "Worst window excess deteriorates.", "Baseline from Step 38 windows if available."),
        ("symbol_level_excess_for_601318", "symbol", "higher_is_better", "unavailable", "Future prototype should explicitly report 601318 symbol-level excess against unchanged baseline.", "601318 is the Step 37 main bull drag.", "601318 improves by unsupported sample effects only.", "Step 40 does not invent this value when not present in Step 39 inputs."),
        ("symbol_level_excess_for_600036", "symbol", "higher_is_better", "unavailable", "Future prototype should explicitly report 600036 near-neutral excess.", "600036 is medium priority near-neutral underperformance.", "600036 weakens or masks aggregate deterioration.", "Step 40 does not invent this value when not present in Step 39 inputs."),
    ]
    return pd.DataFrame(
        [
            {
                "metric_name": name,
                "metric_level": level,
                "direction": direction,
                "baseline_value_if_available": baseline,
                "target_condition": target,
                "why_it_matters": why,
                "failure_condition": failure,
                "notes": notes,
            }
            for name, level, direction, baseline, target, why, failure, notes in rows
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_execution_in_step40", "confirmed", "Only specification CSVs and report text are generated.", "Would create unvalidated performance claims.", "No prototype backtests are run."),
        ("no_threshold_change", "confirmed", "Selected 0.65 / 0.50 threshold is reused for design context only.", "Would invalidate Step 34 selected-threshold context.", "threshold_action=reused_for_design_context_only."),
        ("no_model_retraining", "confirmed", "No trainer or model pipeline is called.", "Would create a new model experiment.", "Model remains logistic_regression from existing context."),
        ("no_feature_engineering_change", "confirmed", "No factor builder or feature code is modified by this module.", "Would change research surface.", "Feature engineering remains unchanged."),
        ("no_new_data_sources", "confirmed", "Only Step 39, optional Step 38, and optional Step 36 outputs are read.", "Would break scope and comparability.", "No external or new market data is added."),
        ("no_new_agents", "confirmed", "No agent configuration or agent outputs are created.", "Would alter research process scope.", "No new agents are added."),
        ("no_trading_ready_upgrade", "confirmed", "Outputs keep trading_ready=False and research-only language.", "Would overclaim unvalidated diagnostics.", "No candidate is trading-ready."),
        ("prototype_design_only", "confirmed", "Every prototype has implementation_status=prototype_design_only and execution_status=not_executed.", "Would blur design and execution steps.", "Implementation waits for a future explicit step."),
        ("educational_research_only", "confirmed", "Report and CLI warn that this is educational/research only.", "Would risk financial-advice framing.", "Not financial advice."),
    ]
    return pd.DataFrame(
        [
            {
                "guardrail": guardrail,
                "status": status,
                "evidence": evidence,
                "consequence_if_violated": consequence,
                "notes": notes,
            }
            for guardrail, status, evidence, consequence, notes in rows
        ]
    )


def build_risk_assessment(specs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    risk_templates = [
        ("overfitting_to_5_symbols", "high", "Prototype may fit the five-symbol diagnostic set too closely.", "Require unchanged-baseline comparison and later broader validation."),
        ("reducing_trade_count_too_much", "medium", "Prototype may appear better by suppressing trades.", "Track trade_count and sufficient_trade_rate."),
        ("improving_601318_but_hurting_other_symbols", "high", "601318-specific improvement may degrade the rest of the symbol set.", "Report both symbol-level and aggregate metrics."),
        ("improving_absolute_return_but_not_benchmark_excess", "high", "Prototype may reduce losses but still lag benchmark.", "Keep avg_strategy_vs_benchmark_pct as primary metric."),
        ("threshold_creep", "high", "Design may drift into threshold tuning.", "Keep threshold fixed at 0.65 / 0.50 unless a future step explicitly scopes threshold work."),
        ("hidden_data_leakage", "high", "Timing or filter designs may accidentally use future information.", "Require leakage review before any future execution."),
        ("false_confidence_from_small_sample", "high", "Limited bull trade/window rows can create unstable conclusions.", "Use conservative interpretation and reject weak evidence."),
        ("trading_ready_overclaim", "blocking", "Prototype design could be misread as deployable evidence.", "State no candidate is trading-ready in every output layer."),
    ]
    prototype_ids = specs["prototype_id"].tolist() if not specs.empty else ["BP-000"]
    for prototype_id in prototype_ids:
        for risk_type, severity, description, mitigation in risk_templates:
            rows.append(
                {
                    "prototype_id": prototype_id,
                    "risk_type": risk_type,
                    "severity": severity,
                    "description": description,
                    "mitigation": mitigation,
                    "notes": "Risk assessment is design-only; no execution evidence is produced.",
                }
            )
    return pd.DataFrame(rows)


def build_execution_plan(specs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    phases = [
        ("Phase 1", "prototype instrumentation and config scaffolding", "V4 Step 41 Bull Prototype Experiment Harness", "Step 40 specs and guardrails", "prototype_config_schema.csv; prototype_harness_plan.md", "Missing design specs or guardrail violations"),
        ("Phase 2", "isolated prototype backtests", "Future controlled prototype execution step", "Approved harness and unchanged baseline", "isolated_prototype_results.csv", "Threshold, model, feature, or data-source drift"),
        ("Phase 3", "compare against unchanged Step 34 baseline", "Future prototype comparison step", "Isolated results plus Step 34 baseline context", "prototype_vs_baseline_comparison.csv", "No unchanged baseline available"),
        ("Phase 4", "rejection / retention decision", "Future prototype decision step", "Comparison outputs and risk assessment", "prototype_decision_matrix.csv", "Insufficient trade count or benchmark weakness"),
        ("Phase 5", "only if robust, broader validation", "Future broader validation step", "Retained prototype with no blocking risks", "broader_validation_results.csv", "Any trading-ready claim before strict validation"),
    ]
    prototype_ids = specs["prototype_id"].tolist() if not specs.empty else ["all_prototypes"]
    for phase, action, future_step, prerequisites, outputs, blocking in phases:
        for prototype_id in prototype_ids:
            rows.append(
                {
                    "phase": phase,
                    "prototype_id": prototype_id,
                    "action": action,
                    "required_future_step": future_step,
                    "prerequisite_outputs": prerequisites,
                    "expected_output_files": outputs,
                    "blocking_conditions": blocking,
                    "notes": "Future sequencing only; Step 40 does not execute this action.",
                }
            )
    return pd.DataFrame(rows)


def build_not_implemented_log() -> pd.DataFrame:
    rows = [
        ("thresholds_not_changed", "confirmed", "No threshold sweep or threshold write path exists in Step 40.", "0.65 / 0.50 remains design context only."),
        ("model_not_retrained", "confirmed", "No model training functions are called.", "Existing model context remains unchanged."),
        ("features_not_changed", "confirmed", "No factor engineering code is touched by this module.", "Feature set remains unchanged."),
        ("data_sources_not_added", "confirmed", "Only local Step 39/38/36 outputs are read.", "No new data source is added."),
        ("agents_not_added", "confirmed", "No agent files or configuration are produced.", "No new agents are added."),
        ("prototypes_not_executed", "confirmed", "execution_status=not_executed for all prototypes.", "No prototype results exist."),
        ("no_new_performance_claims", "confirmed", "Metric plan records baselines only when available.", "No future result is invented."),
        ("no_trading_ready_claim", "confirmed", "Report states no candidate is trading-ready.", "Research status remains unchanged."),
    ]
    return pd.DataFrame(
        [{"item": item, "status": status, "evidence": evidence, "notes": notes} for item, status, evidence, notes in rows]
    )


def build_priority_ranking() -> pd.DataFrame:
    rows = [
        (1, "BP-001", "601318 benchmark lag / relative underperformance", "601318 is the high-priority Step 37 drag and Step 39 benchmark-lag symbol.", "High", "Medium", "High", "V4 Step 41 should scaffold a controlled 601318 benchmark-lag prototype harness."),
        (2, "BP-003", "Global zero beat_benchmark_trades issue", "All symbols had zero Step 38 benchmark-beating trades.", "High", "Medium", "High", "Design global benchmark-lag instrumentation before any execution."),
        (3, "BP-005", "Negative trade cluster for 000858 / 600519 / 600036", "Step 39 identified absolute-loss cluster symbols.", "Medium", "Medium", "Medium", "Separate absolute-loss diagnostics from benchmark-lag diagnostics."),
        (4, "BP-007", "Window-level loss control review", "Step 39 found negative excess window clusters.", "Medium", "Medium", "Medium", "Define fixed-window comparison hooks in the future harness."),
        (5, "BP-006", "Probability signal timing review", "Step 38 timeline includes probability and signal timing fields.", "Medium", "Medium", "Medium", "Instrument timing review without threshold tuning."),
        (6, "BP-004", "601318 trade sufficiency review", "601318 has limited trade evidence.", "Medium", "Low", "Medium", "Use as prerequisite context for 601318-specific prototypes."),
        (7, "BP-002", "Bull exit logic review", "Negative trades exist but exit changes can easily overfit.", "Medium", "Medium", "High", "Keep behind benchmark-lag and cluster diagnostics."),
    ]
    return pd.DataFrame(
        [
            {
                "priority_rank": rank,
                "prototype_id": prototype_id,
                "target": target,
                "rationale": rationale,
                "expected_diagnostic_value": value,
                "implementation_complexity": complexity,
                "risk_level": risk,
                "recommended_next_step": next_step,
            }
            for rank, prototype_id, target, rationale, value, complexity, risk, next_step in rows
        ]
    )


def build_report(
    design_dir: str | Path,
    diagnostics_dir: str | Path | None,
    integrated_dir: str | Path | None,
    specs: pd.DataFrame,
    metric_plan: pd.DataFrame,
    guardrails: pd.DataFrame,
    priority_ranking: pd.DataFrame,
) -> str:
    top = priority_ranking.head(2)["target"].astype(str).tolist() if not priority_ranking.empty else []
    return "\n".join(
        [
            "# V4 Step 40 Bull Remediation Prototype Design Report",
            "",
            "## Executive Summary",
            "This step designs controlled prototype experiments only.",
            "No prototype was executed.",
            "No threshold was changed.",
            "No model was retrained.",
            "No features, data sources, or agents were added.",
            "canonical_reduced_40 remains research-only.",
            "Bull remediation remains failed.",
            "No candidate is trading-ready.",
            "The highest priority is addressing benchmark lag / relative underperformance for 601318 and the global zero beat-benchmark-trades issue.",
            "",
            "## Inputs Used",
            f"- Step 39 design directory: {design_dir}",
            f"- Step 38 diagnostics directory: {diagnostics_dir or 'not provided'}",
            f"- Step 36 integrated directory: {integrated_dir or 'not provided'}",
            "",
            "## No-Execution Guardrails",
            f"- Guardrails recorded: {len(guardrails)}",
            "- All prototypes are marked prototype_design_only, not_executed, and allowed_in_current_step=False.",
            "",
            "## Prototype Experiment Specifications",
            f"- Prototype count: {len(specs)}",
            f"- Highest priorities: {', '.join(top) if top else 'not available'}",
            "",
            "## Metric Plan",
            f"- Metric rows: {len(metric_plan)}",
            "- Baselines are populated only when available from Step 36, Step 38, or Step 39.",
            "",
            "## Risk Assessment",
            "The main risks are overfitting to five symbols, reducing trade count too much, improving one symbol while hurting others, threshold creep, hidden leakage, and trading-ready overclaim.",
            "",
            "## Priority Ranking",
            "The ranking places 601318 benchmark lag first, followed by the global zero beat-benchmark-trades issue.",
            "",
            "## Future Execution Plan",
            "Future sequencing should start with V4 Step 41: Bull Prototype Experiment Harness, then isolated prototype backtests, unchanged-baseline comparison, rejection or retention decisions, and only then broader validation if robust.",
            "",
            "## Why This Does Not Change Trading-Ready Status",
            "Step 40 does not execute prototypes or validate remediation.",
            "It does not change Step 34, Step 36, Step 38, or Step 39 conclusions.",
            "canonical_reduced_40 remains research_only_not_trading_ready.",
            "Bull remediation remains failed and no candidate is trading-ready.",
            "",
            "## Recommended Next Step",
            "Recommended next step: V4 Step 41: Bull Prototype Experiment Harness, which implements controlled experiment infrastructure but still should not claim trading-ready.",
            "",
            "## Educational / Research Disclaimer",
            "This report is educational/research diagnostics only. It is not financial advice.",
            "No strategy, model, threshold, symbol, prototype, or candidate in this report should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_bull_remediation_prototype_design(
    design_dir: str | Path,
    diagnostics_dir: str | Path | None,
    integrated_dir: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    design = Path(design_dir)
    diagnostics = Path(diagnostics_dir) if diagnostics_dir else None
    integrated = Path(integrated_dir) if integrated_dir else None
    design_options = _read_csv(design / "bull_remediation_design_options.csv")
    priority_matrix = _read_csv(design / "bull_remediation_priority_matrix.csv")
    symbol_profile = _read_csv(design / "bull_symbol_error_profile.csv", dtype={"symbol": str})
    aggregate_profile = _read_csv(design / "bull_aggregate_error_profile.csv")
    design_config = _read_json(design / "run_config.json")
    trades = _read_csv(diagnostics / "bull_trade_level_diagnostics.csv", dtype={"symbol": str}) if diagnostics else pd.DataFrame()
    windows = _read_csv(diagnostics / "bull_window_diagnostics.csv", dtype={"symbol": str}) if diagnostics else pd.DataFrame()
    integrated_summary = _read_csv(integrated / "integrated_remediation_summary.csv") if integrated else pd.DataFrame()
    specs = build_prototype_experiment_specs(design_options, symbol_profile)
    metric_plan = build_metric_plan(aggregate_profile, symbol_profile, trades, windows, integrated_summary)
    guardrails = build_guardrails()
    risk_assessment = build_risk_assessment(specs)
    execution_plan = build_execution_plan(specs)
    not_implemented_log = build_not_implemented_log()
    priority_ranking = build_priority_ranking()
    report = build_report(
        design_dir,
        diagnostics_dir,
        integrated_dir,
        specs,
        metric_plan,
        guardrails,
        priority_ranking,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    paths["report"].write_text(report, encoding="utf-8")
    specs.to_csv(paths["specs"], index=False)
    metric_plan.to_csv(paths["metric_plan"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    risk_assessment.to_csv(paths["risk_assessment"], index=False)
    execution_plan.to_csv(paths["execution_plan"], index=False)
    not_implemented_log.to_csv(paths["not_implemented_log"], index=False)
    priority_ranking.to_csv(paths["priority_ranking"], index=False)
    run_config = {
        "design_dir": str(design_dir),
        "diagnostics_dir": str(diagnostics_dir) if diagnostics_dir else None,
        "integrated_dir": str(integrated_dir) if integrated_dir else None,
        "output_dir": str(output_path),
        "candidate": design_config.get("candidate", "canonical_reduced_40"),
        "model": design_config.get("model", "logistic_regression"),
        "buy_threshold": design_config.get("buy_threshold", 0.65),
        "sell_threshold": design_config.get("sell_threshold", 0.50),
        "threshold_action": "reused_for_design_context_only",
        "implementation_status": "prototype_design_only",
        "execution_status": "not_executed",
        "trading_ready": False,
        "prototype_count": int(len(specs)),
        "source_step39_priority_rows": priority_matrix.to_dict(orient="records") if not priority_matrix.empty else [],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "bull_remediation_prototype_design_report": report,
        "bull_prototype_experiment_specs": specs,
        "bull_prototype_metric_plan": metric_plan,
        "bull_prototype_guardrails": guardrails,
        "bull_prototype_risk_assessment": risk_assessment,
        "bull_prototype_execution_plan": execution_plan,
        "bull_prototype_not_implemented_log": not_implemented_log,
        "bull_prototype_priority_ranking": priority_ranking,
        "run_config": run_config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
