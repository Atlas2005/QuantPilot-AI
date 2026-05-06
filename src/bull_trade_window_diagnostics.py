import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .backtester import run_long_only_backtest_with_trades
    from .bull_regime_threshold_remediation import (
        CANONICAL_TO_PRUNING_MODE,
        _drop_non_factor_diagnostics,
        _factor_path,
    )
    from .candidate_stress_test import add_market_regime
    from .dataset_splitter import (
        chronological_split,
        clean_factor_dataset,
        load_factor_dataset,
        validate_required_columns,
    )
    from .factor_pruning_experiment import (
        build_feature_sets,
        identify_safe_feature_columns,
        load_pruning_recommendations,
    )
    from .metrics import summarize_performance
    from .ml_signal_backtester import predict_probabilities_for_rows, probabilities_to_signals
    from .model_trainer import train_baseline_model
    from .trade_metrics import summarize_trade_metrics
except ImportError:
    from backtester import run_long_only_backtest_with_trades
    from bull_regime_threshold_remediation import (
        CANONICAL_TO_PRUNING_MODE,
        _drop_non_factor_diagnostics,
        _factor_path,
    )
    from candidate_stress_test import add_market_regime
    from dataset_splitter import (
        chronological_split,
        clean_factor_dataset,
        load_factor_dataset,
        validate_required_columns,
    )
    from factor_pruning_experiment import (
        build_feature_sets,
        identify_safe_feature_columns,
        load_pruning_recommendations,
    )
    from metrics import summarize_performance
    from ml_signal_backtester import predict_probabilities_for_rows, probabilities_to_signals
    from model_trainer import train_baseline_model
    from trade_metrics import summarize_trade_metrics


DEFAULT_SYMBOLS = ["000001", "600519", "000858", "600036", "601318"]
DEFAULT_CANDIDATE = "canonical_reduced_40"
DEFAULT_MODEL = "logistic_regression"
DEFAULT_BUY_THRESHOLD = 0.65
DEFAULT_SELL_THRESHOLD = 0.50
WINDOW_SIZE = 20
INITIAL_CASH = 10000.0
OUTPUT_FILENAMES = {
    "report": "bull_trade_window_diagnostics_report.md",
    "trade_level": "bull_trade_level_diagnostics.csv",
    "timeline": "bull_signal_timeline_diagnostics.csv",
    "window": "bull_window_diagnostics.csv",
    "symbol_summary": "bull_symbol_window_summary.csv",
    "patterns": "bull_error_pattern_summary.csv",
    "availability": "bull_diagnostics_data_availability.csv",
    "limitations": "bull_diagnostics_limitations.csv",
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
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(row: pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    return pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]


def _load_context(
    bull_dir: str | Path,
    drilldown_dir: str | Path | None,
) -> dict[str, Any]:
    bull = Path(bull_dir)
    drilldown = Path(drilldown_dir) if drilldown_dir else None
    run_config = _read_json(bull / "run_config.json")
    best_bull = _read_csv(bull / "best_bull_thresholds.csv")
    threshold_context = (
        _read_csv(drilldown / "bull_threshold_context.csv") if drilldown else pd.DataFrame()
    )
    symbol_summary = (
        _read_csv(drilldown / "bull_symbol_failure_summary.csv", dtype={"symbol": str})
        if drilldown
        else pd.DataFrame()
    )
    return {
        "run_config": run_config,
        "best_bull": best_bull,
        "threshold_context": threshold_context,
        "drilldown_symbol_summary": symbol_summary,
    }


def _build_symbol_backtest(
    factor_csv: Path,
    symbol: str,
    recommendations_df: pd.DataFrame,
    target_col: str,
    candidate: str,
    model: str,
    buy_threshold: float,
    sell_threshold: float,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_pct: float,
    min_commission: float,
    regime_window: int = 60,
    purge_rows: int = 5,
) -> dict[str, Any]:
    raw_df = load_factor_dataset(factor_csv)
    validate_required_columns(raw_df, target_col)
    regime_df = add_market_regime(raw_df, regime_window=regime_window)
    bull_df = regime_df[regime_df["regime"] == "bull"].copy()
    if bull_df.empty:
        raise ValueError(f"No bull rows available for {symbol}.")
    pruning_mode = CANONICAL_TO_PRUNING_MODE.get(candidate, candidate)
    feature_source_df = _drop_non_factor_diagnostics(regime_df)
    feature_columns = identify_safe_feature_columns(feature_source_df, target_col)
    feature_sets = build_feature_sets(feature_columns, recommendations_df, [pruning_mode])
    features = feature_sets.get(pruning_mode, [])
    if not features:
        raise ValueError(f"No features selected for {symbol}.")
    model_input_df = _drop_non_factor_diagnostics(bull_df)
    cleaned_df, _ = clean_factor_dataset(model_input_df, features, target_col)
    train_df, _, test_df = chronological_split(
        cleaned_df,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        purge_rows=purge_rows,
        split_mode="global_date",
    )
    if train_df.empty or test_df.empty:
        raise ValueError(f"Train or test split is empty for {symbol}.")
    trained_model, training_info = train_baseline_model(
        train_df,
        features,
        target_col=target_col,
        model_name=model,
    )
    signal_df = test_df.copy().reset_index(drop=True)
    probabilities = predict_probabilities_for_rows(trained_model, signal_df, features)
    signal_df["prediction_probability"] = probabilities.values
    signal_df["signal"] = probabilities_to_signals(
        signal_df["prediction_probability"],
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).values
    backtest_df, trades_df = run_long_only_backtest_with_trades(
        signal_df,
        initial_cash=INITIAL_CASH,
        execution_mode="same_close",
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_pct=slippage_pct,
        min_commission=min_commission,
    )
    return {
        "symbol": symbol,
        "signal_df": signal_df,
        "backtest_df": backtest_df,
        "trades_df": trades_df,
        "performance": summarize_performance(backtest_df),
        "trade_metrics": summarize_trade_metrics(trades_df),
        "training_info": training_info,
        "feature_count": len(features),
    }


def _probability_at(signal_df: pd.DataFrame, date_value: Any) -> float:
    if date_value is None or pd.isna(date_value):
        return float("nan")
    dates = pd.to_datetime(signal_df["date"], errors="coerce")
    target = pd.to_datetime(date_value, errors="coerce")
    matches = signal_df[dates == target]
    if matches.empty:
        return float("nan")
    return pd.to_numeric(matches["prediction_probability"], errors="coerce").iloc[0]


def _close_at(signal_df: pd.DataFrame, date_value: Any) -> float:
    if date_value is None or pd.isna(date_value):
        return float("nan")
    dates = pd.to_datetime(signal_df["date"], errors="coerce")
    target = pd.to_datetime(date_value, errors="coerce")
    matches = signal_df[dates == target]
    if matches.empty:
        return float("nan")
    return pd.to_numeric(matches["close"], errors="coerce").iloc[0]


def _trade_error_pattern(return_pct: float, excess_pct: float) -> str:
    if pd.notna(return_pct) and return_pct < 0:
        return "negative_trade_return"
    if pd.notna(return_pct) and pd.notna(excess_pct) and return_pct >= 0 and excess_pct < 0:
        return "positive_return_but_lagged_benchmark"
    if pd.notna(excess_pct) and excess_pct < 0:
        return "benchmark_outpaced_strategy"
    return "no_trade_error_pattern"


def _build_trade_rows(
    result: dict[str, Any],
    candidate: str,
    model: str,
    buy_threshold: float,
    sell_threshold: float,
) -> list[dict[str, Any]]:
    symbol = result["symbol"]
    signal_df = result["signal_df"]
    trades = result["trades_df"]
    rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        entry_date = trade.get("entry_date")
        exit_date = trade.get("exit_date")
        effective_exit = exit_date if _clean_text(exit_date) else signal_df["date"].iloc[-1]
        entry_close = _close_at(signal_df, entry_date)
        exit_close = _close_at(signal_df, effective_exit)
        benchmark_return = (
            float((exit_close / entry_close - 1) * 100)
            if pd.notna(entry_close) and pd.notna(exit_close) and entry_close > 0
            else float("nan")
        )
        trade_return = pd.to_numeric(pd.Series([trade.get("return_pct")]), errors="coerce").iloc[0]
        if pd.isna(trade_return):
            trade_return = pd.to_numeric(
                pd.Series([trade.get("unrealized_return_pct")]),
                errors="coerce",
            ).iloc[0]
        excess = trade_return - benchmark_return if pd.notna(trade_return) and pd.notna(benchmark_return) else float("nan")
        rows.append(
            {
                "symbol": symbol,
                "candidate": candidate,
                "model": model,
                "regime": "bull",
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "holding_days": trade.get("holding_days"),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "trade_return_pct": trade_return,
                "benchmark_return_pct": benchmark_return,
                "trade_excess_pct": excess,
                "entry_signal_probability": _probability_at(signal_df, entry_date),
                "exit_signal_probability": _probability_at(signal_df, effective_exit),
                "exit_reason": trade.get("exit_reason"),
                "was_profitable": bool(pd.notna(trade_return) and trade_return > 0),
                "beat_benchmark": bool(pd.notna(excess) and excess > 0),
                "error_pattern": _trade_error_pattern(trade_return, excess),
                "notes": (
                    "open_trade_marked_to_final_bull_test_row"
                    if not _clean_text(exit_date)
                    else "closed_trade_from_reconstructed_step34_path"
                ),
            }
        )
    return rows


def _build_timeline_rows(
    result: dict[str, Any],
    candidate: str,
    model: str,
) -> pd.DataFrame:
    symbol = result["symbol"]
    signal = result["signal_df"].copy().reset_index(drop=True)
    backtest = result["backtest_df"].copy().reset_index(drop=True)
    df = backtest.merge(
        signal[["date", "prediction_probability"]],
        on="date",
        how="left",
    )
    df["symbol"] = symbol
    df["candidate"] = candidate
    df["model"] = model
    df["regime"] = "bull"
    close = pd.to_numeric(df["close"], errors="coerce")
    total_value = pd.to_numeric(df["total_value"], errors="coerce")
    df["benchmark_close_or_return_if_available"] = close
    df["position"] = (pd.to_numeric(df["shares"], errors="coerce") > 0).astype(int)
    df["signal_action"] = df["signal"].map({1: "buy", -1: "sell", 0: "hold"}).fillna("hold")
    df["strategy_daily_return_pct"] = total_value.pct_change().fillna(0) * 100
    df["benchmark_daily_return_pct"] = close.pct_change().fillna(0) * 100
    df["daily_excess_pct"] = df["strategy_daily_return_pct"] - df["benchmark_daily_return_pct"]
    df["cumulative_strategy_return_pct"] = (total_value / total_value.iloc[0] - 1) * 100
    df["cumulative_benchmark_return_pct"] = (close / close.iloc[0] - 1) * 100
    df["cumulative_excess_pct"] = (
        df["cumulative_strategy_return_pct"] - df["cumulative_benchmark_return_pct"]
    )
    df["drawdown_pct"] = (total_value / total_value.cummax() - 1) * 100
    df["notes"] = "reconstructed_from_step34_selected_bull_threshold"
    return df[
        [
            "symbol",
            "date",
            "candidate",
            "model",
            "regime",
            "close",
            "benchmark_close_or_return_if_available",
            "prediction_probability",
            "position",
            "signal_action",
            "strategy_daily_return_pct",
            "benchmark_daily_return_pct",
            "daily_excess_pct",
            "cumulative_strategy_return_pct",
            "cumulative_benchmark_return_pct",
            "cumulative_excess_pct",
            "drawdown_pct",
            "notes",
        ]
    ]


def _window_error_pattern(strategy_return: float, benchmark_return: float, excess: float) -> str:
    if pd.notna(strategy_return) and strategy_return < 0:
        return "negative_trade_return"
    if pd.notna(strategy_return) and pd.notna(excess) and strategy_return >= 0 and excess < 0:
        return "positive_return_but_lagged_benchmark"
    if pd.notna(benchmark_return) and pd.notna(excess) and benchmark_return > 0 and excess < 0:
        return "benchmark_outpaced_strategy"
    if pd.notna(excess) and -0.5 <= excess < 0:
        return "near_neutral_underperformance"
    return "no_window_error_pattern"


def _build_windows_for_symbol(
    timeline: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
) -> list[dict[str, Any]]:
    rows = []
    symbol_timeline = timeline[timeline["symbol"] == symbol].copy().reset_index(drop=True)
    symbol_trades = trades[trades["symbol"] == symbol].copy() if not trades.empty else pd.DataFrame()
    window_count = int((len(symbol_timeline) + WINDOW_SIZE - 1) // WINDOW_SIZE)
    for window_idx in range(window_count):
        window = symbol_timeline.iloc[window_idx * WINDOW_SIZE : (window_idx + 1) * WINDOW_SIZE].copy()
        if window.empty:
            continue
        total_value = pd.to_numeric(window["cumulative_strategy_return_pct"], errors="coerce")
        close = pd.to_numeric(window["close"], errors="coerce")
        strategy_return = float(total_value.iloc[-1] - total_value.iloc[0])
        benchmark_return = float((close.iloc[-1] / close.iloc[0] - 1) * 100) if close.iloc[0] > 0 else float("nan")
        excess = strategy_return - benchmark_return if pd.notna(benchmark_return) else float("nan")
        drawdown = pd.to_numeric(window["drawdown_pct"], errors="coerce").min()
        start_date = window["date"].iloc[0]
        end_date = window["date"].iloc[-1]
        in_window = pd.DataFrame()
        if not symbol_trades.empty:
            entry_dates = pd.to_datetime(symbol_trades["entry_date"], errors="coerce")
            in_window = symbol_trades[
                (entry_dates >= pd.to_datetime(start_date)) & (entry_dates <= pd.to_datetime(end_date))
            ]
        trade_count = len(in_window)
        win_rate = (
            float(pd.to_numeric(in_window["trade_return_pct"], errors="coerce").gt(0).mean() * 100)
            if trade_count > 0
            else float("nan")
        )
        rows.append(
            {
                "symbol": symbol,
                "window_id": window_idx + 1,
                "start_date": start_date,
                "end_date": end_date,
                "rows": len(window),
                "strategy_return_pct": strategy_return,
                "benchmark_return_pct": benchmark_return,
                "excess_return_pct": excess,
                "max_drawdown_pct": drawdown,
                "trade_count": trade_count,
                "win_rate": win_rate,
                "error_pattern": _window_error_pattern(strategy_return, benchmark_return, excess),
                "contribution_to_symbol_excess": excess / max(window_count, 1) if pd.notna(excess) else float("nan"),
                "notes": "fixed_20_row_bull_test_window",
            }
        )
    return rows


def _main_pattern(values: pd.Series, fallback: str) -> str:
    clean = values.dropna().astype(str)
    clean = clean[~clean.str.startswith("no_")]
    if clean.empty:
        return fallback
    return clean.value_counts().index[0]


def _build_symbol_window_summary(
    symbols: list[str],
    trades: pd.DataFrame,
    timeline: pd.DataFrame,
    windows: pd.DataFrame,
    drilldown_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    drag_symbols = set()
    if not drilldown_summary.empty and {"symbol", "failure_role"}.issubset(drilldown_summary.columns):
        drag_symbols = set(
            drilldown_summary.loc[
                drilldown_summary["failure_role"] == "drag_on_bull_average",
                "symbol",
            ].map(_format_symbol)
        )
    for symbol in symbols:
        symbol_trades = trades[trades["symbol"] == symbol].copy() if not trades.empty else pd.DataFrame()
        symbol_windows = windows[windows["symbol"] == symbol].copy() if not windows.empty else pd.DataFrame()
        returns = pd.to_numeric(symbol_trades.get("trade_return_pct"), errors="coerce") if not symbol_trades.empty else pd.Series(dtype=float)
        excess = pd.to_numeric(symbol_trades.get("trade_excess_pct"), errors="coerce") if not symbol_trades.empty else pd.Series(dtype=float)
        window_excess = pd.to_numeric(symbol_windows.get("excess_return_pct"), errors="coerce") if not symbol_windows.empty else pd.Series(dtype=float)
        pattern = _main_pattern(
            pd.concat(
                [
                    symbol_trades.get("error_pattern", pd.Series(dtype=str)),
                    symbol_windows.get("error_pattern", pd.Series(dtype=str)),
                ],
                ignore_index=True,
            ),
            "no_observed_error_pattern",
        )
        likely = (
            "main_bull_average_drag_identified_in_step37"
            if symbol in drag_symbols
            else "support_or_minor_drag_in_step37"
        )
        rows.append(
            {
                "symbol": symbol,
                "trade_level_available": not symbol_trades.empty,
                "timeline_available": not timeline[timeline["symbol"] == symbol].empty,
                "window_available": not symbol_windows.empty,
                "total_trades": int(len(symbol_trades)),
                "profitable_trades": int(returns.gt(0).sum()) if not returns.empty else 0,
                "losing_trades": int(returns.lt(0).sum()) if not returns.empty else 0,
                "beat_benchmark_trades": int(symbol_trades.get("beat_benchmark", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not symbol_trades.empty else 0,
                "worst_trade_return_pct": returns.min() if not returns.empty else float("nan"),
                "worst_trade_excess_pct": excess.min() if not excess.empty else float("nan"),
                "worst_window_excess_pct": window_excess.min() if not window_excess.empty else float("nan"),
                "main_error_pattern": pattern,
                "likely_failure_mechanism": likely,
                "recommended_followup": (
                    "Classify 601318 bull entry/holding behavior in Step 39."
                    if symbol == "601318"
                    else "Review only as supporting symbol-level context."
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_pattern_summary(trades: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    pattern_symbols: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for df, evidence in [(trades, "trade_level"), (windows, "window_level")]:
        if df.empty or "error_pattern" not in df:
            continue
        for _, row in df.iterrows():
            pattern = _clean_text(row.get("error_pattern"))
            if not pattern or pattern.startswith("no_"):
                continue
            counts[pattern] = counts.get(pattern, 0) + 1
            pattern_symbols.setdefault(pattern, set()).add(_format_symbol(row.get("symbol")))
    if not counts:
        counts = {"no_trade_level_data": 1 if trades.empty else 0, "no_window_data": 1 if windows.empty else 0}
    rows = []
    for pattern, count in counts.items():
        if count <= 0:
            continue
        affected = sorted(pattern_symbols.get(pattern, set()))
        rows.append(
            {
                "error_pattern": pattern,
                "count": count,
                "affected_symbols": ",".join(affected),
                "evidence_level": "observed_diagnostic_output" if affected else "data_availability",
                "interpretation": f"{pattern} observed in bull diagnostics.",
                "recommended_followup": "Use Step 39 for pattern classification / remediation design.",
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "error_pattern"], ascending=[False, True]).reset_index(drop=True)


def _build_availability(
    symbol_rows_available: bool,
    trade_available: bool,
    timeline_available: bool,
    window_available: bool,
    probability_available: bool,
    benchmark_available: bool,
) -> pd.DataFrame:
    rows = [
        ("symbol_level", symbol_rows_available, "Step 34 bull selected rows", "symbol rows reconstructed or loaded"),
        ("trade_level", trade_available, "existing backtester trade log", "trade rows reconstructed from selected threshold"),
        ("date_level_timeline", timeline_available, "existing backtester daily records", "date-level signal/equity rows reconstructed"),
        ("window_level", window_available, "fixed 20-row windows", "window rows built from date-level timeline"),
        ("probability_signal_level", probability_available, "model probability predictions", "prediction_probability available"),
        ("benchmark_comparison_level", benchmark_available, "same close series benchmark", "benchmark returns computed"),
    ]
    return pd.DataFrame(
        [
            {
                "diagnostic_layer": layer,
                "available": available,
                "source": source,
                "evidence": evidence if available else f"{layer} unavailable",
                "consequence": "diagnostic layer available" if available else "diagnostic layer limited",
                "required_followup": "Use in Step 39 pattern classification." if available else "Add instrumentation in a later diagnostic step.",
            }
            for layer, available, source, evidence in rows
        ]
    )


def _build_limitations(trade_available: bool, timeline_available: bool, window_available: bool) -> pd.DataFrame:
    rows = [
        ("no_new_data_sources_used", "info", "No new data sources were added.", "Diagnostics are limited to existing factor/backtest inputs.", "Keep scope unchanged."),
        ("no_threshold_search", "info", "The selected 0.65 / 0.50 threshold is reused.", "No optimization is performed.", "Do not tune in Step 38."),
        ("no_model_retraining", "info", "Model training logic is reused exactly for reconstruction.", "This is not a new model experiment.", "Keep training logic unchanged."),
        ("diagnostic_only_not_optimization", "blocking", "This step only exports diagnostics.", "No candidate is upgraded.", "Use results for later diagnostic design only."),
        ("small_symbol_count", "medium", "The bull diagnostic covers five configured symbols.", "Single-symbol behavior can dominate aggregate findings.", "Interpret symbol effects conservatively."),
        ("research_only_not_trading_ready", "blocking", "canonical_reduced_40 remains research-only and not trading-ready.", "No trading-ready status changes.", "Continue validation diagnostics."),
    ]
    if not trade_available:
        rows.append(("trade_level_data_unavailable", "medium", "Trade-level rows could not be produced.", "Trade pattern diagnosis is limited.", "Add trade instrumentation."))
    if not timeline_available:
        rows.append(("date_level_data_unavailable", "medium", "Date-level rows could not be produced.", "Timeline diagnosis is limited.", "Add date-level instrumentation."))
    if not window_available:
        rows.append(("window_level_data_unavailable", "medium", "Window rows could not be produced.", "Window diagnosis is limited.", "Add window-level instrumentation."))
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


def _placeholder(status: str, reason: str, required_followup: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_status": status,
                "reason": reason,
                "required_followup": required_followup,
                "notes": "No records were fabricated.",
            }
        ]
    )


def _threshold_context_row(
    context: dict[str, Any],
    candidate: str,
    model: str,
    buy_threshold: float,
    sell_threshold: float,
) -> dict[str, Any]:
    best = context["best_bull"]
    row = best.iloc[0] if not best.empty else pd.Series(dtype=object)
    return {
        "candidate": candidate,
        "model": model,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "final_decision": _clean_text(row.get("final_decision")) or "bull_remediation_failed",
        "avg_strategy_vs_benchmark_pct": _numeric(row, "avg_strategy_vs_benchmark_pct"),
        "threshold_action": "reused_for_diagnosis_only",
    }


def _generate_report(
    threshold_context: dict[str, Any],
    availability: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    patterns: pd.DataFrame,
    limitations: pd.DataFrame,
    trade_available: bool,
    timeline_available: bool,
    window_available: bool,
    bull_dir: str | Path,
    drilldown_dir: str | Path | None,
) -> str:
    main_601318 = ""
    if not symbol_summary.empty and "symbol" in symbol_summary:
        row_601318 = symbol_summary[symbol_summary["symbol"] == "601318"]
        if not row_601318.empty:
            main_601318 = (
                f"601318 main pattern: {row_601318.iloc[0].get('main_error_pattern')}; "
                f"worst window excess: {row_601318.iloc[0].get('worst_window_excess_pct')}"
            )
    next_step = (
        "V4 Step 39 Bull Error Pattern Classification / Remediation Design."
        if trade_available and timeline_available and window_available
        else "Instrumentation enhancement is still needed before detailed pattern classification."
    )
    sections = [
        "# V4 Step 38 Bull Trade/Window Diagnostics Output Enhancement Report",
        "",
        "## Executive Summary",
        "This step is diagnostics-output enhancement only.",
        "It reuses the selected Step 34 bull threshold 0.65 / 0.50.",
        "It does not tune thresholds, retrain models, change factor engineering, add data sources, or add agents.",
        "It does not upgrade any candidate to trading-ready.",
        "canonical_reduced_40 remains research-only.",
        "Bull remediation remains failed under the existing Step 34 aggregate status.",
        "",
        "## Inputs Used",
        f"- Bull remediation directory: {bull_dir}",
        f"- Drilldown directory: {drilldown_dir or 'not provided'}",
        "",
        "## Selected Bull Threshold Context",
        f"- Candidate: {threshold_context['candidate']} + {threshold_context['model']}",
        f"- Buy threshold: {threshold_context['buy_threshold']}",
        f"- Sell threshold: {threshold_context['sell_threshold']}",
        f"- Step 34 final decision: {threshold_context['final_decision']}",
        f"- Step 34 average excess: {threshold_context['avg_strategy_vs_benchmark_pct']}",
        "",
        "## Data Availability",
        f"- Trade-level diagnostics generated: {trade_available}",
        f"- Signal timeline diagnostics generated: {timeline_available}",
        f"- Window diagnostics generated: {window_available}",
        "",
        "## Trade-Level Diagnostics",
        "Trade-level rows are exported in bull_trade_level_diagnostics.csv." if trade_available else "Trade-level rows are unavailable; placeholder rows were written without fabricated trades.",
        "",
        "## Signal Timeline Diagnostics",
        "Date-level signal/equity rows are exported in bull_signal_timeline_diagnostics.csv." if timeline_available else "Date-level signal/equity rows are unavailable; placeholder rows were written.",
        "",
        "## Window-Level Diagnostics",
        "Fixed 20-row bull windows are exported in bull_window_diagnostics.csv." if window_available else "Window diagnostics are unavailable; placeholder rows were written.",
        "",
        "## Symbol-Level Diagnostic Summary",
        f"- Symbols summarized: {len(symbol_summary) if not symbol_summary.empty else 0}",
        "",
        "## Error Pattern Summary",
        f"- Error pattern rows: {len(patterns) if not patterns.empty else 0}",
        "",
        "## What This Explains About 601318 and 600036",
        main_601318 or "601318 was identified by Step 37 as the main bull drag; detailed claims require the exported diagnostics.",
        "600036 was identified by Step 37 as slightly negative / near-neutral; detailed claims require the exported diagnostics.",
        "",
        "## Limitations",
        f"- Limitation rows: {len(limitations)}",
        "",
        "## Why This Does Not Change Trading-Ready Status",
        "This step exports diagnostics only and does not change the Step 34 or Step 36 gate status.",
        "No candidate is trading-ready.",
        "",
        "## Recommended Next Step",
        f"Recommended next step: {next_step}",
        "",
        "## Educational / Research Disclaimer",
        "This report is educational/research diagnostics only. It is not financial advice.",
        "No strategy, model, threshold, symbol, or candidate in this report should be treated as deployable or trading-ready.",
        "",
    ]
    return "\n".join(sections)


def generate_bull_trade_window_diagnostics(
    bull_dir: str | Path,
    drilldown_dir: str | Path | None,
    output_dir: str | Path,
    candidate: str = DEFAULT_CANDIDATE,
    model: str = DEFAULT_MODEL,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
) -> dict[str, Any]:
    context = _load_context(bull_dir, drilldown_dir)
    config = context["run_config"]
    symbols = [_format_symbol(symbol) for symbol in config.get("symbols", DEFAULT_SYMBOLS)]
    factor_dir = config.get("factor_dir", "outputs/model_robustness_real_v2/factors")
    recommendations_path = config.get(
        "recommendations_path",
        "outputs/feature_ablation_real_v1/feature_pruning_recommendations.csv",
    )
    recommendations = load_pruning_recommendations(recommendations_path)
    target_col = config.get("target_col", "label_up_5d")
    trade_rows: list[dict[str, Any]] = []
    timeline_frames: list[pd.DataFrame] = []
    reconstruction_warnings: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            result = _build_symbol_backtest(
                factor_csv=_factor_path(factor_dir, symbol),
                symbol=symbol,
                recommendations_df=recommendations,
                target_col=target_col,
                candidate=candidate,
                model=model,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                commission_rate=float(config.get("commission_rate", 0.0003)),
                stamp_tax_rate=float(config.get("stamp_tax_rate", 0.001)),
                slippage_pct=float(config.get("slippage_pct", 0.0005)),
                min_commission=float(config.get("min_commission", 5.0)),
            )
            trade_rows.extend(
                _build_trade_rows(result, candidate, model, buy_threshold, sell_threshold)
            )
            timeline_frames.append(_build_timeline_rows(result, candidate, model))
        except Exception as exc:
            reconstruction_warnings.append(
                {
                    "symbol": symbol,
                    "warning_type": "reconstruction_failed",
                    "message": str(exc),
                }
            )
    trades = pd.DataFrame(trade_rows)
    timeline = pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame()
    window_rows: list[dict[str, Any]] = []
    if not timeline.empty:
        for symbol in symbols:
            window_rows.extend(_build_windows_for_symbol(timeline, trades, symbol))
    windows = pd.DataFrame(window_rows)
    trade_available = not trades.empty
    timeline_available = not timeline.empty
    window_available = not windows.empty
    if not trade_available:
        trades = _placeholder(
            "trade_level_data_unavailable",
            "Existing pipeline did not produce trade records for the selected threshold.",
            "Add trade instrumentation before trade pattern classification.",
        )
    if not timeline_available:
        timeline = _placeholder(
            "timeline_data_unavailable",
            "Existing pipeline did not produce date-level signal/equity records.",
            "Add date-level instrumentation before window diagnostics.",
        )
    if not window_available:
        windows = _placeholder(
            "window_data_unavailable",
            "Window diagnostics require date-level timeline rows.",
            "Add date-level instrumentation before window diagnostics.",
        )
    summary = _build_symbol_window_summary(
        symbols,
        trades if trade_available else pd.DataFrame(),
        timeline if timeline_available else pd.DataFrame(),
        windows if window_available else pd.DataFrame(),
        context["drilldown_symbol_summary"],
    )
    patterns = _build_pattern_summary(
        trades if trade_available else pd.DataFrame(),
        windows if window_available else pd.DataFrame(),
    )
    availability = _build_availability(
        symbol_rows_available=not context["drilldown_symbol_summary"].empty,
        trade_available=trade_available,
        timeline_available=timeline_available,
        window_available=window_available,
        probability_available=timeline_available and "prediction_probability" in timeline,
        benchmark_available=timeline_available and "benchmark_daily_return_pct" in timeline,
    )
    limitations = _build_limitations(trade_available, timeline_available, window_available)
    threshold_context = _threshold_context_row(
        context,
        candidate,
        model,
        buy_threshold,
        sell_threshold,
    )
    report = _generate_report(
        threshold_context,
        availability,
        summary,
        patterns,
        limitations,
        trade_available,
        timeline_available,
        window_available,
        bull_dir,
        drilldown_dir,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    paths["report"].write_text(report, encoding="utf-8")
    trades.to_csv(paths["trade_level"], index=False)
    timeline.to_csv(paths["timeline"], index=False)
    windows.to_csv(paths["window"], index=False)
    summary.to_csv(paths["symbol_summary"], index=False)
    patterns.to_csv(paths["patterns"], index=False)
    availability.to_csv(paths["availability"], index=False)
    limitations.to_csv(paths["limitations"], index=False)
    run_config = {
        "bull_dir": str(bull_dir),
        "drilldown_dir": str(drilldown_dir) if drilldown_dir else None,
        "output_dir": str(output_path),
        "candidate": candidate,
        "model": model,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "threshold_action": "reused_for_diagnosis_only",
        "trade_level_available": trade_available,
        "timeline_available": timeline_available,
        "window_available": window_available,
        "symbols": symbols,
        "reconstruction_warnings": reconstruction_warnings,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "bull_trade_window_diagnostics_report": report,
        "bull_trade_level_diagnostics": trades,
        "bull_signal_timeline_diagnostics": timeline,
        "bull_window_diagnostics": windows,
        "bull_symbol_window_summary": summary,
        "bull_error_pattern_summary": patterns,
        "bull_diagnostics_data_availability": availability,
        "bull_diagnostics_limitations": limitations,
        "run_config": run_config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
