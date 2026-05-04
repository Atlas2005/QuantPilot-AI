import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .candidate_mode_normalization import add_canonical_mode_columns
    from .reduced_feature_threshold_experiment import (
        run_reduced_feature_threshold_experiment,
        run_reduced_feature_walk_forward_experiment,
        summarize_threshold_results,
    )
except ImportError:
    from candidate_mode_normalization import add_canonical_mode_columns
    from reduced_feature_threshold_experiment import (
        run_reduced_feature_threshold_experiment,
        run_reduced_feature_walk_forward_experiment,
        summarize_threshold_results,
    )


DEFAULT_SYMBOLS = ["000001", "600519", "000858", "600036", "601318"]


def parse_symbols(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_SYMBOLS.copy()
    return [_format_symbol(item.strip()) for item in text.split(",") if item.strip()]


def _format_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _factor_path(factor_dir: str | Path, symbol: str) -> Path:
    return Path(factor_dir) / f"factors_{_format_symbol(symbol)}.csv"


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows available._"
    table_df = df.head(max_rows)
    headers = [str(column) for column in table_df.columns]

    def clean(value: Any) -> str:
        if value is None or pd.isna(value):
            return "n/a"
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


def _candidate_summary(
    results_df: pd.DataFrame,
    min_trades: int,
    candidate_label: str,
) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(
            [
                {
                    "candidate_label": candidate_label,
                    "symbol_count": 0,
                    "avg_strategy_vs_benchmark_pct": None,
                    "benchmark_win_rate": 0.0,
                    "avg_trade_count": None,
                    "sufficient_trade_rate": 0.0,
                    "final_decision": "fail",
                    "decision_reason": "No result rows were produced.",
                }
            ]
        )
    summary = summarize_threshold_results(
        results_df,
        ["canonical_mode", "model_type", "buy_threshold", "sell_threshold"],
        min_trades,
    )
    summary.insert(0, "candidate_label", candidate_label)
    summary["benchmark_win_rate"] = summary["beat_benchmark_rate"]
    symbol_count = int(results_df["symbol"].nunique()) if "symbol" in results_df else 0
    summary["tested_symbol_count"] = symbol_count

    enough_symbols = symbol_count >= 3
    avg_excess = _numeric(summary, "avg_strategy_vs_benchmark_pct").iloc[0]
    win_rate = _numeric(summary, "benchmark_win_rate").iloc[0]
    avg_trades = _numeric(summary, "avg_trade_count").iloc[0]
    low_trade_rows = int((_numeric(results_df, "trade_count") < min_trades).sum())
    severe_low_confidence = low_trade_rows > max(0, len(results_df) // 2)
    passed = (
        avg_excess > 0
        and win_rate >= 0.60
        and avg_trades >= min_trades
        and enough_symbols
        and not severe_low_confidence
    )
    reasons = []
    if avg_excess <= 0:
        reasons.append("average excess return is not positive")
    if win_rate < 0.60:
        reasons.append("benchmark win rate is below 0.60")
    if avg_trades < min_trades:
        reasons.append("average trade count is below min_trades")
    if not enough_symbols:
        reasons.append("fewer than three symbols were tested")
    if severe_low_confidence:
        reasons.append("low-trade warnings dominate the result rows")
    summary["low_trade_row_count"] = low_trade_rows
    summary["final_decision"] = "pass" if passed else "fail"
    summary["decision_reason"] = (
        "All strict validation conditions were met."
        if passed
        else "; ".join(reasons)
    )
    return summary


def _build_warnings(
    results_df: pd.DataFrame,
    source_warnings: pd.DataFrame,
    min_trades: int,
    candidate_label: str,
) -> pd.DataFrame:
    warnings = []
    if not source_warnings.empty:
        for row in source_warnings.to_dict("records"):
            row["candidate_label"] = candidate_label
            warnings.append(row)
    if not results_df.empty:
        low_trade = results_df[_numeric(results_df, "trade_count") < min_trades]
        for _, row in low_trade.iterrows():
            warnings.append(
                {
                    "candidate_label": candidate_label,
                    "symbol": _format_symbol(row.get("symbol")),
                    "model_type": row.get("model_type"),
                    "pruning_mode": row.get("pruning_mode"),
                    "legacy_pruning_mode": row.get("legacy_pruning_mode"),
                    "canonical_mode": row.get("canonical_mode"),
                    "buy_threshold": row.get("buy_threshold"),
                    "sell_threshold": row.get("sell_threshold"),
                    "warning_type": "low_trade_count",
                    "message": f"trade_count {row.get('trade_count')} is below min_trades {min_trades}.",
                }
            )
        underperform = results_df[_numeric(results_df, "strategy_vs_benchmark_pct") < 0]
        for _, row in underperform.iterrows():
            warnings.append(
                {
                    "candidate_label": candidate_label,
                    "symbol": _format_symbol(row.get("symbol")),
                    "model_type": row.get("model_type"),
                    "pruning_mode": row.get("pruning_mode"),
                    "legacy_pruning_mode": row.get("legacy_pruning_mode"),
                    "canonical_mode": row.get("canonical_mode"),
                    "buy_threshold": row.get("buy_threshold"),
                    "sell_threshold": row.get("sell_threshold"),
                    "warning_type": "underperformed_benchmark",
                    "message": "Candidate underperformed benchmark for this symbol/window.",
                }
            )
    warnings_df = pd.DataFrame(warnings)
    if "symbol" in warnings_df:
        warnings_df["symbol"] = warnings_df["symbol"].map(_format_symbol)
    return warnings_df


def run_candidate_expanded_validation(
    factor_dir: str | Path,
    symbols: list[str],
    recommendations_path: str | Path,
    target_col: str = "label_up_5d",
    candidate_pruning_mode: str = "keep_core_and_observe",
    candidate_model: str = "logistic_regression",
    candidate_buy_threshold: float = 0.50,
    candidate_sell_threshold: float = 0.40,
    walk_forward_pruning_mode: str = "drop_reduce_weight",
    walk_forward_model: str = "logistic_regression",
    walk_forward_buy_threshold: float = 0.50,
    walk_forward_sell_threshold: float = 0.40,
    initial_cash: float = 10000.0,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    min_commission: float = 5.0,
    min_trades: int = 3,
    enable_walk_forward: bool = False,
) -> dict[str, Any]:
    symbols = [_format_symbol(symbol) for symbol in symbols]
    result_frames = []
    warning_frames = []
    walk_forward_frames = []
    walk_forward_warning_frames = []

    for symbol in symbols:
        factor_csv = _factor_path(factor_dir, symbol)
        if not factor_csv.exists():
            warning_frames.append(
                pd.DataFrame(
                    [
                        {
                            "candidate_label": "historical_candidate",
                            "symbol": symbol,
                            "warning_type": "missing_factor_csv",
                            "message": f"Factor CSV not found: {factor_csv}",
                        }
                    ]
                )
            )
            continue

        historical = run_reduced_feature_threshold_experiment(
            factor_csv=factor_csv,
            recommendations_path=recommendations_path,
            model_types=[candidate_model],
            pruning_modes=[candidate_pruning_mode],
            target_col=target_col,
            buy_thresholds=[candidate_buy_threshold],
            sell_thresholds=[candidate_sell_threshold],
            initial_cash=initial_cash,
            execution_mode=execution_mode,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
            min_trades=min_trades,
        )
        historical_results = historical["threshold_results"].copy()
        if not historical_results.empty:
            historical_results["symbol"] = historical_results["symbol"].map(_format_symbol)
            historical_results["candidate_label"] = "historical_candidate"
            historical_results = add_canonical_mode_columns(historical_results)
            result_frames.append(historical_results)
        warning_frames.append(
            _build_warnings(
                historical_results,
                historical["warnings"],
                min_trades,
                "historical_candidate",
            )
        )

        if enable_walk_forward:
            wf_results, wf_source_warnings = run_reduced_feature_walk_forward_experiment(
                factor_csv=factor_csv,
                recommendations_path=recommendations_path,
                model_types=[walk_forward_model],
                pruning_modes=[walk_forward_pruning_mode],
                target_col=target_col,
                buy_thresholds=[walk_forward_buy_threshold],
                sell_thresholds=[walk_forward_sell_threshold],
                initial_cash=initial_cash,
                execution_mode=execution_mode,
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax_rate,
                slippage_pct=slippage_pct,
                min_commission=min_commission,
                min_trades=min_trades,
            )
            if not wf_results.empty:
                wf_results["symbol"] = wf_results["symbol"].map(_format_symbol)
                wf_results["candidate_label"] = "walk_forward_candidate"
                wf_results = add_canonical_mode_columns(wf_results)
                walk_forward_frames.append(wf_results)
            walk_forward_warning_frames.append(
                _build_warnings(
                    wf_results,
                    wf_source_warnings,
                    min_trades,
                    "walk_forward_candidate",
                )
            )

    results_df = (
        pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    )
    warnings_df = (
        pd.concat(warning_frames, ignore_index=True) if warning_frames else pd.DataFrame()
    )
    if "symbol" in warnings_df:
        warnings_df["symbol"] = warnings_df["symbol"].map(_format_symbol)
    summary_df = _candidate_summary(results_df, min_trades, "historical_candidate")
    per_symbol_df = results_df.copy()

    walk_forward_df = (
        pd.concat(walk_forward_frames, ignore_index=True)
        if walk_forward_frames
        else pd.DataFrame()
    )
    wf_warnings_df = (
        pd.concat(walk_forward_warning_frames, ignore_index=True)
        if walk_forward_warning_frames
        else pd.DataFrame()
    )
    if not wf_warnings_df.empty:
        warnings_df = pd.concat([warnings_df, wf_warnings_df], ignore_index=True)
    walk_forward_summary = (
        _candidate_summary(walk_forward_df, min_trades, "walk_forward_candidate")
        if enable_walk_forward
        else pd.DataFrame()
    )
    report = generate_candidate_validation_report(
        results_df,
        summary_df,
        per_symbol_df,
        warnings_df,
        walk_forward_df,
        walk_forward_summary,
        enable_walk_forward,
        min_trades,
    )
    return {
        "candidate_validation_results": results_df,
        "candidate_validation_summary": summary_df,
        "per_symbol_candidate_results": per_symbol_df,
        "candidate_validation_warnings": warnings_df,
        "walk_forward_candidate_results": walk_forward_df,
        "walk_forward_candidate_summary": walk_forward_summary,
        "candidate_validation_report": report,
    }


def generate_candidate_validation_report(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    per_symbol_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
    walk_forward_df: pd.DataFrame,
    walk_forward_summary: pd.DataFrame,
    enable_walk_forward: bool,
    min_trades: int,
) -> str:
    final_decision = (
        summary_df["final_decision"].iloc[0] if "final_decision" in summary_df else "fail"
    )
    decision_text = (
        "The candidate passes the strict research validation gate."
        if final_decision == "pass"
        else "The candidate remains research-only and should not be promoted."
    )
    sections = [
        "# Recommended Candidate Expanded Validation",
        "",
        "## Executive Summary",
        (
            "This validation is educational research diagnostics only. It is not "
            "trading-ready, not financially reliable, and not financial advice."
        ),
        decision_text,
        "",
        "## Tested Candidates",
        _markdown_table(
            summary_df[
                [
                    column
                    for column in [
                        "candidate_label",
                        "canonical_mode",
                        "legacy_pruning_mode",
                        "model_type",
                        "buy_threshold",
                        "sell_threshold",
                        "tested_symbol_count",
                    ]
                    if column in summary_df
                ]
            ]
        ),
        "",
        "## Historical Candidate Result",
        _markdown_table(summary_df),
        "",
        "## Benchmark Comparison",
        _markdown_table(
            per_symbol_df[
                [
                    column
                    for column in [
                        "symbol",
                        "total_return_pct",
                        "benchmark_return_pct",
                        "strategy_vs_benchmark_pct",
                        "trade_count",
                        "warning",
                    ]
                    if column in per_symbol_df
                ]
            ]
        ),
        "",
        "## Trade-Count Diagnostics",
        (
            f"Rows with trade_count below min_trades ({min_trades}) are "
            "low-confidence and cannot support a strong recommendation."
        ),
        _markdown_table(
            per_symbol_df[
                _numeric(per_symbol_df, "trade_count") < min_trades
            ]
            if not per_symbol_df.empty
            else pd.DataFrame()
        ),
        "",
        "## Per-Symbol Results",
        _markdown_table(per_symbol_df),
    ]
    if enable_walk_forward:
        sections.extend(
            [
                "",
                "## Walk-Forward Candidate Result",
                _markdown_table(walk_forward_summary),
                "",
                "## Walk-Forward Per-Window Results",
                _markdown_table(walk_forward_df),
            ]
        )
    sections.extend(
        [
            "",
            "## Warning Summary",
            _markdown_table(warnings_df),
            "",
            "## Final Decision",
            (
                "Pass requires average excess return > 0, benchmark win rate >= "
                "0.60, average trade count >= min_trades, enough tested symbols, "
                "and no severe low-confidence warning dominance."
            ),
            decision_text,
            "",
        ]
    )
    return "\n".join(sections)


def save_candidate_expanded_validation(
    factor_dir: str | Path,
    symbols: list[str],
    recommendations_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    result = run_candidate_expanded_validation(
        factor_dir=factor_dir,
        symbols=symbols,
        recommendations_path=recommendations_path,
        **kwargs,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_validation_results": output_path / "candidate_validation_results.csv",
        "candidate_validation_summary": output_path / "candidate_validation_summary.csv",
        "per_symbol_candidate_results": output_path / "per_symbol_candidate_results.csv",
        "candidate_validation_warnings": output_path
        / "candidate_validation_warnings.csv",
        "candidate_validation_report": output_path / "candidate_validation_report.md",
        "run_config": output_path / "run_config.json",
    }
    if kwargs.get("enable_walk_forward"):
        paths["walk_forward_candidate_results"] = (
            output_path / "walk_forward_candidate_results.csv"
        )
        paths["walk_forward_candidate_summary"] = (
            output_path / "walk_forward_candidate_summary.csv"
        )
    for key, path in paths.items():
        if key in {"candidate_validation_report", "run_config"}:
            continue
        result.get(key, pd.DataFrame()).to_csv(path, index=False)
    paths["candidate_validation_report"].write_text(
        result["candidate_validation_report"],
        encoding="utf-8",
    )
    run_config = {
        "factor_dir": str(factor_dir),
        "symbols": [_format_symbol(symbol) for symbol in symbols],
        "recommendations_path": str(recommendations_path),
        "output_dir": str(output_path),
        **kwargs,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["run_config"] = run_config
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result
