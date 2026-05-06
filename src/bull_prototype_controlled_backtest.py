import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILENAMES = {
    "report": "bull_prototype_controlled_backtest_report.md",
    "execution_results": "bull_prototype_execution_results.csv",
    "metric_comparison": "bull_prototype_metric_comparison.csv",
    "symbol_comparison": "bull_prototype_symbol_comparison.csv",
    "trade_comparison": "bull_prototype_trade_comparison.csv",
    "window_comparison": "bull_prototype_window_comparison.csv",
    "decision_summary": "bull_prototype_decision_summary.csv",
    "execution_audit": "bull_prototype_execution_audit.csv",
    "guardrail_check": "bull_prototype_guardrail_check.csv",
    "limitations": "bull_prototype_limitations.csv",
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


def _format_symbol_list(value: Any) -> str:
    text = _clean_text(value)
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    return ",".join(_format_symbol(token) if token.isdigit() else token for token in tokens)


def _read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "symbol" in df:
        df["symbol"] = df["symbol"].map(_format_symbol)
    if "target_symbols" in df:
        df["target_symbols"] = df["target_symbols"].map(_format_symbol_list)
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


def _metric_baseline(contract: pd.DataFrame, metric_name: str) -> float:
    if contract.empty or "metric_name" not in contract:
        return float("nan")
    rows = contract[contract["metric_name"].astype(str) == metric_name]
    if rows.empty:
        return float("nan")
    return pd.to_numeric(
        pd.Series([rows.iloc[0].get("baseline_value_if_available")]),
        errors="coerce",
    ).iloc[0]


def _trade_metrics(trades: pd.DataFrame, included: pd.Series | None = None) -> dict[str, Any]:
    if trades.empty:
        return {
            "avg_excess_pct": float("nan"),
            "trade_count": 0,
            "negative_trade_count": 0,
            "beat_benchmark_trades": 0,
            "positive_return_but_lagged_benchmark_count": 0,
            "worst_trade_excess_pct": float("nan"),
        }
    mask = included if included is not None else pd.Series([True] * len(trades), index=trades.index)
    subset = trades[mask].copy()
    trade_return = pd.to_numeric(subset.get("trade_return_pct", pd.Series(dtype=float)), errors="coerce")
    trade_excess = pd.to_numeric(subset.get("trade_excess_pct", pd.Series(dtype=float)), errors="coerce")
    pattern = subset.get("error_pattern", pd.Series(dtype=str)).astype(str)
    beat = subset.get("beat_benchmark", pd.Series(dtype=bool)).fillna(False).astype(bool)
    return {
        "avg_excess_pct": float(trade_excess.mean()) if not subset.empty else float("nan"),
        "trade_count": int(len(subset)),
        "negative_trade_count": int(trade_return.lt(0).sum()),
        "beat_benchmark_trades": int(beat.sum()),
        "positive_return_but_lagged_benchmark_count": int((pattern == "positive_return_but_lagged_benchmark").sum()),
        "worst_trade_excess_pct": float(trade_excess.min()) if not subset.empty else float("nan"),
    }


def _window_metrics(windows: pd.DataFrame, included: pd.Series | None = None) -> dict[str, Any]:
    if windows.empty:
        return {"worst_window_excess_pct": float("nan")}
    mask = included if included is not None else pd.Series([True] * len(windows), index=windows.index)
    subset = windows[mask].copy()
    excess = pd.to_numeric(subset.get("excess_return_pct", pd.Series(dtype=float)), errors="coerce")
    return {"worst_window_excess_pct": float(excess.min()) if not subset.empty else float("nan")}


def _simulation_masks(prototype: pd.Series, trades: pd.DataFrame, windows: pd.DataFrame, timeline: pd.DataFrame) -> dict[str, Any]:
    pid = _clean_text(prototype.get("prototype_id"))
    trade_mask = pd.Series([True] * len(trades), index=trades.index)
    window_mask = pd.Series([True] * len(windows), index=windows.index)
    status = "executed"
    sim_type = "diagnostic_filter_simulation"
    reason = ""
    if pid == "BP-001":
        sim_type = "symbol_ablation_or_focus_simulation"
        trade_mask = trades["symbol"] != "601318" if "symbol" in trades else trade_mask
        window_mask = windows["symbol"] != "601318" if "symbol" in windows else window_mask
        reason = "601318 symbol excluded only for diagnostic focus comparison; not a removal recommendation."
    elif pid == "BP-002":
        status = "not_executable_with_current_data"
        sim_type = "not_executable_prototype"
        reason = "Exit logic review requires implementable exit rules and pipeline execution beyond Step 38 diagnostic rows."
    elif pid == "BP-003":
        sim_type = "diagnostic_filter_simulation"
        if not windows.empty and {"trade_count", "benchmark_return_pct"}.issubset(windows.columns):
            window_mask = ~(
                pd.to_numeric(windows["trade_count"], errors="coerce").fillna(0).eq(0)
                & pd.to_numeric(windows["benchmark_return_pct"], errors="coerce").fillna(0).gt(0)
            )
            reason = "Windows with zero trades and positive benchmark return are excluded for missed-participation diagnostic simulation."
        else:
            status = "not_executable_with_current_data"
            reason = "Window trade_count and benchmark_return_pct are unavailable."
    elif pid == "BP-004":
        sim_type = "symbol_ablation_or_focus_simulation"
        trade_mask = trades["symbol"] == "601318" if "symbol" in trades else trade_mask
        window_mask = windows["symbol"] == "601318" if "symbol" in windows else window_mask
        reason = "601318-only focus simulation to inspect limited trade sufficiency."
    elif pid == "BP-005":
        sim_type = "diagnostic_filter_simulation"
        targets = {"000858", "600519", "600036"}
        trade_return = pd.to_numeric(trades.get("trade_return_pct", pd.Series(dtype=float)), errors="coerce")
        trade_mask = ~((trades.get("symbol", pd.Series(dtype=str)).isin(targets)) & trade_return.lt(0))
        reason = "Negative trade rows for 000858, 600519, and 600036 are excluded for diagnostic loss-cluster simulation."
    elif pid == "BP-006":
        if not timeline.empty and "prediction_probability" in timeline:
            status = "not_executable_with_current_data"
            sim_type = "not_executable_prototype"
            reason = "Probability timeline exists, but causal timing changes require future rule implementation and cannot be faithfully executed from diagnostics alone."
        else:
            status = "not_executable_with_current_data"
            sim_type = "not_executable_prototype"
            reason = "Probability timeline diagnostics are unavailable."
    elif pid == "BP-007":
        sim_type = "window_filter_simulation"
        if not windows.empty and "excess_return_pct" in windows:
            excess = pd.to_numeric(windows["excess_return_pct"], errors="coerce")
            drawdown = pd.to_numeric(windows.get("max_drawdown_pct", pd.Series(dtype=float)), errors="coerce")
            window_mask = ~(excess.lt(0) | drawdown.le(-10))
            reason = "Negative-excess or high-drawdown windows are excluded for diagnostic window-risk simulation."
        else:
            status = "not_executable_with_current_data"
            reason = "Window excess diagnostics are unavailable."
    return {
        "execution_status": status,
        "simulation_type": sim_type,
        "trade_mask": trade_mask,
        "window_mask": window_mask,
        "reason": reason,
    }


def _conservative_result(status: str, baseline_avg: float, proto_avg: float, baseline_metrics: dict[str, Any], proto_metrics: dict[str, Any], reason: str) -> tuple[str, str, bool, str]:
    if status == "not_executable_with_current_data":
        return (
            "not_executable_with_current_data",
            "Do not advance until executable rule and data support exist.",
            False,
            reason,
        )
    delta = proto_avg - baseline_avg if pd.notna(proto_avg) and pd.notna(baseline_avg) else float("nan")
    neg_improved = proto_metrics["negative_trade_count"] < baseline_metrics["negative_trade_count"]
    avg_improved = pd.notna(delta) and delta > 0
    if avg_improved:
        return (
            "improved_but_not_validated",
            "May advance to further diagnostic testing only.",
            True,
            "The primary average-excess metric improves under a controlled what-if filter, but this is not validation and may be unimplementable without future rules.",
        )
    if pd.notna(delta) and delta <= 0 and neg_improved:
        return (
            "mixed_secondary_improvement_but_primary_worse",
            "Do not advance without redesign of the primary metric behavior.",
            False,
            "A secondary diagnostic metric improved, but the primary average-excess metric did not improve over the unchanged baseline.",
        )
    if pd.notna(delta) and delta <= 0:
        return (
            "no_improvement",
            "Do not advance without redesign.",
            False,
            "Controlled diagnostic simulation did not improve the primary comparison metric.",
        )
    return (
        "inconclusive",
        "Requires additional rule design before further testing.",
        False,
        "Available diagnostic rows are insufficient for a clear conservative decision.",
    )


def build_execution_outputs(
    registry: pd.DataFrame,
    specs: pd.DataFrame,
    metric_contract: pd.DataFrame,
    trades: pd.DataFrame,
    windows: pd.DataFrame,
    timeline: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    baseline_avg = _metric_baseline(metric_contract, "avg_strategy_vs_benchmark_pct")
    baseline_trade_metrics = _trade_metrics(trades)
    baseline_window_metrics = _window_metrics(windows)
    result_rows = []
    metric_rows = []
    symbol_rows = []
    trade_rows = []
    window_rows = []
    spec_by_id = {row["prototype_id"]: row for _, row in specs.iterrows()} if not specs.empty else {}
    for _, proto in registry.iterrows():
        pid = _clean_text(proto.get("prototype_id"))
        spec = spec_by_id.get(pid, proto)
        masks = _simulation_masks(proto, trades, windows, timeline)
        status = masks["execution_status"]
        trade_mask = masks["trade_mask"]
        window_mask = masks["window_mask"]
        proto_trade_metrics = _trade_metrics(trades, trade_mask if status == "executed" else None)
        proto_window_metrics = _window_metrics(windows, window_mask if status == "executed" else None)
        proto_avg = proto_trade_metrics["avg_excess_pct"] if status == "executed" else float("nan")
        conservative, recommendation, can_advance, decision_reason = _conservative_result(
            status,
            baseline_avg,
            proto_avg,
            baseline_trade_metrics,
            proto_trade_metrics,
            masks["reason"],
        )
        result_rows.append(
            {
                "prototype_id": pid,
                "prototype_name": proto.get("prototype_name"),
                "execution_status": status,
                "simulation_type": masks["simulation_type"],
                "target_symbols": proto.get("target_symbols"),
                "baseline_metric_source": "Step 36 integrated summary and Step 38 diagnostics",
                "prototype_metric_source": "Step 38 diagnostic rows controlled what-if simulation" if status == "executed" else "not executable from current diagnostics",
                "baseline_avg_excess_pct": baseline_avg,
                "prototype_avg_excess_pct": proto_avg,
                "delta_avg_excess_pct": proto_avg - baseline_avg if pd.notna(proto_avg) and pd.notna(baseline_avg) else float("nan"),
                "baseline_trade_count": baseline_trade_metrics["trade_count"],
                "prototype_trade_count": proto_trade_metrics["trade_count"] if status == "executed" else pd.NA,
                "baseline_negative_trade_count": baseline_trade_metrics["negative_trade_count"],
                "prototype_negative_trade_count": proto_trade_metrics["negative_trade_count"] if status == "executed" else pd.NA,
                "baseline_beat_benchmark_trades": baseline_trade_metrics["beat_benchmark_trades"],
                "prototype_beat_benchmark_trades": proto_trade_metrics["beat_benchmark_trades"] if status == "executed" else pd.NA,
                "conservative_result": conservative,
                "decision_reason": decision_reason,
                "trading_ready": False,
                "notes": "Research-only controlled diagnostic simulation. No production strategy logic is changed.",
            }
        )
        metric_values = {
            "avg_strategy_vs_benchmark_pct": (baseline_avg, proto_avg, "higher_is_better"),
            "trade_count": (baseline_trade_metrics["trade_count"], proto_trade_metrics["trade_count"] if status == "executed" else pd.NA, "higher_or_equal_with_quality"),
            "negative_trade_count": (baseline_trade_metrics["negative_trade_count"], proto_trade_metrics["negative_trade_count"] if status == "executed" else pd.NA, "lower_is_better"),
            "positive_return_but_lagged_benchmark_count": (baseline_trade_metrics["positive_return_but_lagged_benchmark_count"], proto_trade_metrics["positive_return_but_lagged_benchmark_count"] if status == "executed" else pd.NA, "lower_is_better"),
            "beat_benchmark_trades": (baseline_trade_metrics["beat_benchmark_trades"], proto_trade_metrics["beat_benchmark_trades"] if status == "executed" else pd.NA, "higher_is_better"),
            "worst_trade_excess_pct": (baseline_trade_metrics["worst_trade_excess_pct"], proto_trade_metrics["worst_trade_excess_pct"] if status == "executed" else pd.NA, "higher_is_better"),
            "worst_window_excess_pct": (baseline_window_metrics["worst_window_excess_pct"], proto_window_metrics["worst_window_excess_pct"] if status == "executed" else pd.NA, "higher_is_better"),
        }
        for metric, (base, value, direction) in metric_values.items():
            delta = value - base if pd.notna(value) and pd.notna(base) else float("nan")
            metric_rows.append(
                {
                    "prototype_id": pid,
                    "metric_name": metric,
                    "baseline_value": base,
                    "prototype_value": value,
                    "delta": delta,
                    "direction": direction,
                    "interpretation": "diagnostic comparison only; not validation of a deployable rule",
                    "validation_status": conservative,
                }
            )
        for symbol in sorted(trades.get("symbol", pd.Series(dtype=str)).dropna().unique()):
            base_sym = trades[trades["symbol"] == symbol]
            proto_sym = trades[(trades["symbol"] == symbol) & trade_mask] if status == "executed" else pd.DataFrame()
            base_excess = pd.to_numeric(base_sym.get("trade_excess_pct"), errors="coerce")
            proto_excess = pd.to_numeric(proto_sym.get("trade_excess_pct"), errors="coerce")
            symbol_rows.append(
                {
                    "prototype_id": pid,
                    "symbol": _format_symbol(symbol),
                    "baseline_trade_count": len(base_sym),
                    "prototype_trade_count": len(proto_sym) if status == "executed" else pd.NA,
                    "baseline_trade_excess_sum": base_excess.sum(),
                    "prototype_trade_excess_sum": proto_excess.sum() if status == "executed" else pd.NA,
                    "baseline_avg_trade_excess_pct": base_excess.mean(),
                    "prototype_avg_trade_excess_pct": proto_excess.mean() if status == "executed" and not proto_sym.empty else pd.NA,
                    "baseline_negative_trade_count": int(pd.to_numeric(base_sym.get("trade_return_pct"), errors="coerce").lt(0).sum()),
                    "prototype_negative_trade_count": int(pd.to_numeric(proto_sym.get("trade_return_pct"), errors="coerce").lt(0).sum()) if status == "executed" else pd.NA,
                    "interpretation": conservative,
                    "notes": "Symbol-level diagnostic comparison only.",
                }
            )
        for idx, trade in trades.iterrows():
            included = bool(trade_mask.loc[idx]) if status == "executed" and idx in trade_mask.index else False
            reason = "" if included else masks["reason"] or "prototype not executable"
            trade_rows.append(
                {
                    "prototype_id": pid,
                    "symbol": _format_symbol(trade.get("symbol")),
                    "entry_date": trade.get("entry_date"),
                    "exit_date": trade.get("exit_date"),
                    "baseline_trade_return_pct": trade.get("trade_return_pct"),
                    "baseline_trade_excess_pct": trade.get("trade_excess_pct"),
                    "included_in_prototype": included,
                    "exclusion_or_adjustment_reason": reason,
                    "prototype_trade_return_pct": trade.get("trade_return_pct") if included else pd.NA,
                    "prototype_trade_excess_pct": trade.get("trade_excess_pct") if included else pd.NA,
                    "interpretation": conservative,
                    "notes": "Original trade values are preserved; no adjusted returns are fabricated.",
                }
            )
        for idx, window in windows.iterrows():
            included = bool(window_mask.loc[idx]) if status == "executed" and idx in window_mask.index else False
            reason = "" if included else masks["reason"] or "prototype not executable"
            window_rows.append(
                {
                    "prototype_id": pid,
                    "symbol": _format_symbol(window.get("symbol")),
                    "window_id": window.get("window_id"),
                    "start_date": window.get("start_date"),
                    "end_date": window.get("end_date"),
                    "baseline_strategy_return_pct": window.get("strategy_return_pct"),
                    "baseline_benchmark_return_pct": window.get("benchmark_return_pct"),
                    "baseline_excess_return_pct": window.get("excess_return_pct"),
                    "included_in_prototype": included,
                    "exclusion_or_adjustment_reason": reason,
                    "interpretation": conservative,
                    "notes": "Window diagnostic comparison only; no implementable production rule is claimed.",
                }
            )
    results = pd.DataFrame(result_rows)
    decisions = pd.DataFrame(
        [
            {
                "prototype_id": row["prototype_id"],
                "conservative_result": row["conservative_result"],
                "recommendation": _conservative_result(
                    row["execution_status"],
                    row["baseline_avg_excess_pct"],
                    row["prototype_avg_excess_pct"],
                    {"negative_trade_count": row["baseline_negative_trade_count"]},
                    {"negative_trade_count": row["prototype_negative_trade_count"] if pd.notna(row["prototype_negative_trade_count"]) else row["baseline_negative_trade_count"]},
                    row["decision_reason"],
                )[1],
                "can_advance_to_further_testing": row["conservative_result"] == "improved_but_not_validated",
                "reason": row["decision_reason"],
                "required_next_validation": "V4 Step 43 review, then broader validation before any candidate status change.",
                "trading_ready": False,
            }
            for _, row in results.iterrows()
        ]
    )
    return {
        "results": results,
        "metric_comparison": pd.DataFrame(metric_rows),
        "symbol_comparison": pd.DataFrame(symbol_rows),
        "trade_comparison": pd.DataFrame(trade_rows),
        "window_comparison": pd.DataFrame(window_rows),
        "decision_summary": decisions,
    }


def build_execution_audit() -> pd.DataFrame:
    rows = [
        ("used_existing_diagnostics_only", "confirmed", "Inputs are Step 41/40/38/36 output files.", "No new rows are sourced externally."),
        ("no_new_data_sources", "confirmed", "No data loader is called.", "No new data source is added."),
        ("no_new_agents", "confirmed", "No agent configuration is created.", "No new agents are added."),
        ("no_model_retraining", "confirmed", "No trainer is called.", "Model remains unchanged."),
        ("no_feature_engineering_change", "confirmed", "No factor builder is called.", "Features remain unchanged."),
        ("no_threshold_change", "confirmed", "0.65 / 0.50 is reused as unchanged baseline.", "No threshold sweep is run."),
        ("no_previous_outputs_overwritten", "confirmed", "Step 42 writes only to its own output directory.", "Baseline outputs remain untouched."),
        ("research_only_execution", "confirmed", "Simulations are labelled diagnostic only.", "No production strategy claim."),
        ("no_trading_ready_claim", "confirmed", "trading_ready=False for all prototypes.", "No candidate is trading-ready."),
    ]
    return pd.DataFrame([{"audit_item": a, "status": s, "evidence": e, "notes": n} for a, s, e, n in rows])


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_data_sources", "confirmed", "Only existing output CSV files are read.", "Would break comparability.", "No new data source."),
        ("no_new_agents", "confirmed", "No agent files or configs are added.", "Would alter process scope.", "No new agents."),
        ("no_model_retraining", "confirmed", "No model training path is used.", "Would create a new model experiment.", "No retraining."),
        ("no_feature_engineering_change", "confirmed", "No feature pipeline is called.", "Would change research surface.", "No feature changes."),
        ("no_threshold_change", "confirmed", "threshold_action=reused_as_unchanged_baseline_for_controlled_execution.", "Would invalidate baseline.", "0.65 / 0.50 unchanged."),
        ("no_threshold_sweep", "confirmed", "No threshold grid is created.", "Would optimize strategy.", "No sweep."),
        ("no_previous_outputs_overwritten", "confirmed", "Output path is Step 42-only.", "Would corrupt baseline record.", "Prior outputs untouched."),
        ("no_trading_ready_upgrade", "confirmed", "All prototype rows have trading_ready=False.", "Would overclaim.", "No candidate is trading-ready."),
        ("educational_research_only", "confirmed", "Report and CLI use research-only warnings.", "Would risk financial-advice framing.", "Not financial advice."),
    ]
    return pd.DataFrame(
        [{"guardrail": g, "status": s, "evidence": e, "consequence_if_violated": c, "notes": n} for g, s, e, c, n in rows]
    )


def build_limitations() -> pd.DataFrame:
    rows = [
        ("prototype_results_are_diagnostic_not_production", "blocking", "Simulations use exported diagnostic rows.", "Results are not deployable strategy evidence.", "Require future validation."),
        ("filters_may_be_unimplementable_without_future_rules", "high", "Some simulations exclude bad trades/windows after observation.", "Improvements may be unrealistic.", "Design implementable rules before validation."),
        ("small_symbol_count", "medium", "Diagnostics cover five symbols.", "Results may be unstable.", "Use broader validation later."),
        ("risk_of_overfitting", "high", "Prototypes target known error patterns.", "May fit historical diagnostics.", "Reject weak evidence."),
        ("no_out_of_sample_confirmation", "blocking", "No new out-of-sample validation is run.", "Candidate status cannot change.", "Run future validation gates."),
        ("benchmark_proxy_limitations_if_applicable", "medium", "Benchmark comparison uses available Step 38 benchmark/proxy fields.", "Benchmark conclusions depend on that proxy.", "Document benchmark assumptions."),
        ("not_trading_ready", "blocking", "No candidate is trading-ready.", "No deployment conclusion.", "Continue research diagnostics."),
    ]
    return pd.DataFrame(
        [{"limitation_type": t, "severity": s, "description": d, "consequence": c, "recommended_followup": f} for t, s, d, c, f in rows]
    )


def summarize_best_avg_excess_candidate(results: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "best_diagnostic_candidate": "no_avg_excess_improvement",
        "best_prototype_id": "",
        "best_prototype_name": "",
        "baseline_avg_excess_pct": None,
        "prototype_avg_excess_pct": None,
        "delta_avg_excess_pct": None,
        "reason": "No executable prototype improved average excess over the unchanged baseline.",
    }
    if results.empty:
        summary["reason"] = "No prototype result rows are available."
        return summary
    executable = results[results["execution_status"].astype(str) == "executed"].copy()
    if executable.empty:
        summary["reason"] = "No executable prototype rows are available."
        return summary
    executable["baseline_avg_excess_pct"] = pd.to_numeric(executable["baseline_avg_excess_pct"], errors="coerce")
    executable["prototype_avg_excess_pct"] = pd.to_numeric(executable["prototype_avg_excess_pct"], errors="coerce")
    executable["delta_avg_excess_pct"] = pd.to_numeric(executable["delta_avg_excess_pct"], errors="coerce")
    executable = executable.dropna(subset=["baseline_avg_excess_pct", "prototype_avg_excess_pct"])
    if executable.empty:
        summary["reason"] = "No executable prototype has numeric baseline and prototype average excess values."
        return summary
    improved = executable[executable["prototype_avg_excess_pct"] > executable["baseline_avg_excess_pct"]]
    if improved.empty:
        baseline = executable["baseline_avg_excess_pct"].dropna()
        if not baseline.empty:
            summary["baseline_avg_excess_pct"] = float(baseline.iloc[0])
        return summary
    best = improved.sort_values("prototype_avg_excess_pct", ascending=False).iloc[0]
    summary.update(
        {
            "best_diagnostic_candidate": _clean_text(best.get("prototype_name")) or _clean_text(best.get("prototype_id")),
            "best_prototype_id": _clean_text(best.get("prototype_id")),
            "best_prototype_name": _clean_text(best.get("prototype_name")),
            "baseline_avg_excess_pct": float(best["baseline_avg_excess_pct"]),
            "prototype_avg_excess_pct": float(best["prototype_avg_excess_pct"]),
            "delta_avg_excess_pct": float(best["prototype_avg_excess_pct"] - best["baseline_avg_excess_pct"]),
            "reason": "Highest executable prototype average excess among prototypes that improved over the unchanged baseline. Diagnostic only, not validation.",
        }
    )
    return summary


def build_report(
    results: pd.DataFrame,
    best_avg_excess_summary: dict[str, Any],
    output_dir: str | Path,
    harness_dir: str | Path,
    prototype_design_dir: str | Path,
    diagnostics_dir: str | Path,
    integrated_dir: str | Path | None,
) -> str:
    executed = int((results["execution_status"] == "executed").sum()) if not results.empty else 0
    not_exec = int((results["execution_status"] == "not_executable_with_current_data").sum()) if not results.empty else 0
    best_candidate = best_avg_excess_summary.get("best_diagnostic_candidate", "no_avg_excess_improvement")
    return "\n".join(
        [
            "# V4 Step 42 Bull Prototype Controlled Backtest Execution Report",
            "",
            "## Executive Summary",
            "Step 42 executes controlled research-only prototype simulations/backtests using existing diagnostics and the unchanged baseline.",
            "No model was retrained. No feature engineering was changed. No new data sources or agents were added.",
            "The selected 0.65 / 0.50 threshold remains unchanged.",
            "Any improvement is diagnostic only, not validation of a deployable strategy.",
            "canonical_reduced_40 remains research-only. Bull remediation remains unresolved unless a future validation gate explicitly passes.",
            "No candidate is trading-ready.",
            "",
            "## Inputs Used",
            f"- Harness directory: {harness_dir}",
            f"- Prototype design directory: {prototype_design_dir}",
            f"- Diagnostics directory: {diagnostics_dir}",
            f"- Integrated directory: {integrated_dir or 'not provided'}",
            f"- Output directory: {output_dir}",
            "",
            "## Baseline Context",
            "The unchanged Step 34 / Step 38 baseline is reused for comparison. The selected buy/sell threshold remains 0.65 / 0.50.",
            "",
            "## Execution Boundary",
            "Simulations are diagnostic filters, symbol-focus what-if checks, or window-risk filters over existing Step 38 rows. They do not modify production strategy logic.",
            "",
            "## Prototype Execution Results",
            f"- Executed diagnostic simulations: {executed}",
            f"- Not executable with current data: {not_exec}",
            f"- Best diagnostic candidate by average excess: {best_candidate}",
            f"- Best candidate reason: {best_avg_excess_summary.get('reason', 'unavailable')}",
            "",
            "## Metric Comparison",
            "Metric comparisons are long-format diagnostic comparisons and do not validate deployment.",
            "",
            "## Symbol-Level Comparison",
            "Symbol-level comparisons preserve six-digit symbols and show included/excluded diagnostic rows.",
            "",
            "## Trade-Level Comparison",
            "Trade-level rows keep original return and excess values visible. No adjusted trade returns are fabricated.",
            "",
            "## Window-Level Comparison",
            "Window-level rows show diagnostic inclusion/exclusion only.",
            "",
            "## Conservative Decision Summary",
            "A prototype can only advance to further diagnostic testing if it improves diagnostic metrics without guardrail violations. trading_ready remains False in all cases.",
            "",
            "## Guardrail Check",
            "All guardrails are confirmed: no new data, agents, retraining, feature changes, threshold changes, or trading-ready upgrade.",
            "",
            "## Limitations",
            "Diagnostic filters may be unimplementable without future rules, the symbol set is small, and no out-of-sample confirmation is run.",
            "",
            "## Why This Does Not Change Trading-Ready Status",
            "This step does not validate a deployable strategy. canonical_reduced_40 remains research_only_not_trading_ready and bull remediation remains unresolved.",
            "",
            "## Recommended Next Step",
            "Recommended next step: V4 Step 43 Bull Prototype Result Review / Candidate Selection for Further Validation.",
            "",
            "## Educational / Research Disclaimer",
            "This report is educational/research diagnostics only. It is not financial advice.",
            "No strategy, model, threshold, symbol, prototype, or candidate in this report should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_bull_prototype_controlled_backtest(
    harness_dir: str | Path,
    prototype_design_dir: str | Path,
    diagnostics_dir: str | Path,
    integrated_dir: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    harness = Path(harness_dir)
    design = Path(prototype_design_dir)
    diagnostics = Path(diagnostics_dir)
    integrated = Path(integrated_dir) if integrated_dir else None
    registry = _read_csv(harness / "bull_prototype_registry.csv")
    metric_contract = _read_csv(harness / "bull_prototype_metric_contract.csv")
    specs = _read_csv(design / "bull_prototype_experiment_specs.csv")
    trades = _read_csv(diagnostics / "bull_trade_level_diagnostics.csv", dtype={"symbol": str})
    timeline = _read_csv(diagnostics / "bull_signal_timeline_diagnostics.csv", dtype={"symbol": str})
    windows = _read_csv(diagnostics / "bull_window_diagnostics.csv", dtype={"symbol": str})
    integrated_summary = _read_csv(integrated / "integrated_remediation_summary.csv") if integrated else pd.DataFrame()
    outputs = build_execution_outputs(registry, specs, metric_contract, trades, windows, timeline)
    audit = build_execution_audit()
    guardrails = build_guardrails()
    limitations = build_limitations()
    best_avg_excess_summary = summarize_best_avg_excess_candidate(outputs["results"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = build_report(outputs["results"], best_avg_excess_summary, output_path, harness_dir, prototype_design_dir, diagnostics_dir, integrated_dir)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    paths["report"].write_text(report, encoding="utf-8")
    outputs["results"].to_csv(paths["execution_results"], index=False)
    outputs["metric_comparison"].to_csv(paths["metric_comparison"], index=False)
    outputs["symbol_comparison"].to_csv(paths["symbol_comparison"], index=False)
    outputs["trade_comparison"].to_csv(paths["trade_comparison"], index=False)
    outputs["window_comparison"].to_csv(paths["window_comparison"], index=False)
    outputs["decision_summary"].to_csv(paths["decision_summary"], index=False)
    audit.to_csv(paths["execution_audit"], index=False)
    guardrails.to_csv(paths["guardrail_check"], index=False)
    limitations.to_csv(paths["limitations"], index=False)
    config = {
        "harness_dir": str(harness_dir),
        "prototype_design_dir": str(prototype_design_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "integrated_dir": str(integrated_dir) if integrated_dir else None,
        "output_dir": str(output_path),
        "candidate": "canonical_reduced_40",
        "model": "logistic_regression",
        "buy_threshold": 0.65,
        "sell_threshold": 0.50,
        "threshold_action": "reused_as_unchanged_baseline_for_controlled_execution",
        "executed_prototype_count": int((outputs["results"]["execution_status"] == "executed").sum()),
        "not_executable_prototype_count": int((outputs["results"]["execution_status"] == "not_executable_with_current_data").sum()),
        "best_avg_excess_summary": best_avg_excess_summary,
        "trading_ready": False,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "bull_prototype_controlled_backtest_report": report,
        "bull_prototype_execution_results": outputs["results"],
        "bull_prototype_metric_comparison": outputs["metric_comparison"],
        "bull_prototype_symbol_comparison": outputs["symbol_comparison"],
        "bull_prototype_trade_comparison": outputs["trade_comparison"],
        "bull_prototype_window_comparison": outputs["window_comparison"],
        "bull_prototype_decision_summary": outputs["decision_summary"],
        "bull_prototype_execution_audit": audit,
        "bull_prototype_guardrail_check": guardrails,
        "bull_prototype_limitations": limitations,
        "best_avg_excess_summary": best_avg_excess_summary,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
