from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .backtester import run_long_only_backtest_with_trades
    from .metrics import summarize_performance
    from .ml_signal_backtester import (
        calculate_buy_and_hold_summary,
        predict_probabilities_for_rows,
        probabilities_to_signals,
        prepare_ml_signal_data,
    )
    from .model_predictor import load_model_bundle, load_prediction_input
    from .model_trainer import train_baseline_model
    from .trade_metrics import summarize_trade_metrics
except ImportError:
    from backtester import run_long_only_backtest_with_trades
    from metrics import summarize_performance
    from ml_signal_backtester import (
        calculate_buy_and_hold_summary,
        predict_probabilities_for_rows,
        probabilities_to_signals,
        prepare_ml_signal_data,
    )
    from model_predictor import load_model_bundle, load_prediction_input
    from model_trainer import train_baseline_model
    from trade_metrics import summarize_trade_metrics


DEFAULT_BUY_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
DEFAULT_SELL_THRESHOLDS = [0.40, 0.45, 0.50, 0.55]


def parse_thresholds(text: str | None, defaults: list[float]) -> list[float]:
    """Parse a comma-separated threshold list."""
    if text is None or not text.strip():
        return defaults.copy()

    thresholds = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))

    if not thresholds:
        raise ValueError("Threshold list cannot be empty.")
    if any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("Thresholds must be between 0 and 1.")

    return sorted(set(thresholds))


def threshold_pairs(
    buy_thresholds: list[float],
    sell_thresholds: list[float],
) -> list[tuple[float, float]]:
    """Return valid threshold pairs where sell_threshold < buy_threshold."""
    return [
        (buy_threshold, sell_threshold)
        for buy_threshold in buy_thresholds
        for sell_threshold in sell_thresholds
        if sell_threshold < buy_threshold
    ]


def score_result(total_return_pct, max_drawdown_pct, profit_factor) -> float:
    """
    Simple educational ranking score.

    Higher return helps. Larger drawdown losses hurt. Profit factor adds a small
    bonus when available.
    """
    total_return = 0.0 if pd.isna(total_return_pct) else float(total_return_pct)
    drawdown = 0.0 if pd.isna(max_drawdown_pct) else float(max_drawdown_pct)
    profit_bonus = 0.0 if pd.isna(profit_factor) else float(profit_factor) * 2
    return total_return + drawdown * 0.3 + profit_bonus


def _result_row(
    buy_threshold: float,
    sell_threshold: float,
    backtest_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    benchmark: dict[str, Any],
    window_id: int | None = None,
    train_start=None,
    train_end=None,
    test_start=None,
    test_end=None,
) -> dict[str, Any]:
    performance = summarize_performance(backtest_df)
    trade_metrics = summarize_trade_metrics(trades_df)
    benchmark_return = benchmark.get("benchmark_return_pct")
    strategy_vs_benchmark = (
        None
        if benchmark_return is None
        else performance["total_return_pct"] - benchmark_return
    )
    profit_factor = trade_metrics.get("profit_factor")
    row = {
        "buy_threshold": float(buy_threshold),
        "sell_threshold": float(sell_threshold),
        "total_return_pct": performance["total_return_pct"],
        "max_drawdown_pct": performance["max_drawdown_pct"],
        "profit_factor": profit_factor,
        "win_rate_pct": trade_metrics.get("win_rate_pct"),
        "total_trades": trade_metrics.get("total_trades"),
        "final_value": performance["final_value"],
        "benchmark_return_pct": benchmark_return,
        "strategy_vs_benchmark_pct": strategy_vs_benchmark,
        "score": score_result(
            performance["total_return_pct"],
            performance["max_drawdown_pct"],
            profit_factor,
        ),
    }
    if window_id is not None:
        row.update(
            {
                "window_id": int(window_id),
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return row


def run_threshold_experiment_from_signal_data(
    signal_base_df: pd.DataFrame,
    probabilities: pd.Series,
    buy_thresholds: list[float],
    sell_thresholds: list[float],
    initial_cash: float = 10000.0,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0,
    stamp_tax_rate: float = 0.0,
    slippage_pct: float = 0.0,
    min_commission: float = 0.0,
    window_id: int | None = None,
    train_start=None,
    train_end=None,
    test_start=None,
    test_end=None,
) -> pd.DataFrame:
    """Run threshold combinations against already computed probabilities."""
    rows = []
    for buy_threshold, sell_threshold in threshold_pairs(
        buy_thresholds,
        sell_thresholds,
    ):
        backtest_input = signal_base_df.copy()
        backtest_input["prediction_probability"] = probabilities.values
        backtest_input["signal"] = probabilities_to_signals(
            probabilities,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        ).values
        backtest_df, trades_df = run_long_only_backtest_with_trades(
            backtest_input,
            initial_cash=initial_cash,
            execution_mode=execution_mode,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
        )
        benchmark = calculate_buy_and_hold_summary(backtest_input, initial_cash)
        rows.append(
            _result_row(
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                backtest_df=backtest_df,
                trades_df=trades_df,
                benchmark=benchmark,
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return pd.DataFrame(rows)


def run_threshold_experiment(
    model_dir: str | Path,
    input_path: str | Path,
    buy_thresholds: list[float] | None = None,
    sell_thresholds: list[float] | None = None,
    initial_cash: float = 10000.0,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0,
    stamp_tax_rate: float = 0.0,
    slippage_pct: float = 0.0,
    min_commission: float = 0.0,
) -> pd.DataFrame:
    """Run a grid of ML probability threshold backtests."""
    if buy_thresholds is None:
        buy_thresholds = DEFAULT_BUY_THRESHOLDS
    if sell_thresholds is None:
        sell_thresholds = DEFAULT_SELL_THRESHOLDS

    signal_df, _ = prepare_ml_signal_data(
        model_dir=model_dir,
        factor_csv=input_path,
        buy_threshold=max(buy_thresholds),
        sell_threshold=min(sell_thresholds),
    )
    probabilities = signal_df["prediction_probability"]
    return run_threshold_experiment_from_signal_data(
        signal_base_df=signal_df,
        probabilities=probabilities,
        buy_thresholds=buy_thresholds,
        sell_thresholds=sell_thresholds,
        initial_cash=initial_cash,
        execution_mode=execution_mode,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_pct=slippage_pct,
        min_commission=min_commission,
    )


def rank_threshold_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return results sorted by the educational score."""
    if results_df.empty:
        return results_df
    return results_df.sort_values(
        ["score", "total_return_pct", "max_drawdown_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def chronological_windows(
    df: pd.DataFrame,
    train_window: int,
    test_window: int,
    step_size: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Create rolling chronological train/test windows by row count."""
    if train_window <= 0 or test_window <= 0 or step_size <= 0:
        raise ValueError("train_window, test_window, and step_size must be positive.")

    sorted_df = df.copy()
    sorted_df["date"] = pd.to_datetime(sorted_df["date"], errors="coerce")
    sorted_df = sorted_df.dropna(subset=["date"]).sort_values("date").reset_index(
        drop=True
    )

    windows = []
    start = 0
    window_id = 1
    while start + train_window + test_window <= len(sorted_df):
        train_df = sorted_df.iloc[start : start + train_window].copy()
        test_df = sorted_df.iloc[
            start + train_window : start + train_window + test_window
        ].copy()
        windows.append((window_id, train_df, test_df))
        start += step_size
        window_id += 1
    return windows


def run_walk_forward_threshold_experiment(
    model_dir: str | Path,
    input_path: str | Path,
    target_col: str = "label_up_5d",
    model_name: str = "random_forest",
    buy_thresholds: list[float] | None = None,
    sell_thresholds: list[float] | None = None,
    train_window: int = 120,
    test_window: int = 40,
    step_size: int = 40,
    initial_cash: float = 10000.0,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0,
    stamp_tax_rate: float = 0.0,
    slippage_pct: float = 0.0,
    min_commission: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simple walk-forward threshold experiment.

    The model is retrained on each earlier window and tested on the following
    chronological window. This is intentionally small and educational.
    """
    if buy_thresholds is None:
        buy_thresholds = DEFAULT_BUY_THRESHOLDS
    if sell_thresholds is None:
        sell_thresholds = DEFAULT_SELL_THRESHOLDS

    model_dir_path = Path(model_dir)
    model_candidates = sorted(model_dir_path.glob("*.joblib"))
    if not model_candidates:
        raise FileNotFoundError(f"No .joblib model found in {model_dir_path}")
    _, _, feature_columns = load_model_bundle(model_candidates[0])

    df = load_prediction_input(input_path)
    windows = chronological_windows(df, train_window, test_window, step_size)
    rows = []

    for window_id, train_df, test_df in windows:
        required = feature_columns + [target_col]
        train_df = train_df.dropna(subset=[column for column in required if column in train_df])
        test_df = test_df.dropna(subset=[column for column in feature_columns if column in test_df])
        if train_df.empty or test_df.empty:
            continue

        model, _ = train_baseline_model(
            train_df=train_df,
            feature_columns=feature_columns,
            target_col=target_col,
            model_name=model_name,
            random_state=random_state,
        )
        test_df = test_df.copy().reset_index(drop=True)
        test_df["date"] = pd.to_datetime(test_df["date"], errors="coerce")
        test_df = test_df.dropna(subset=["date", "open", "close"]).sort_values("date")
        probabilities = predict_probabilities_for_rows(
            model,
            test_df,
            feature_columns,
        )
        window_results = run_threshold_experiment_from_signal_data(
            signal_base_df=test_df,
            probabilities=probabilities,
            buy_thresholds=buy_thresholds,
            sell_thresholds=sell_thresholds,
            initial_cash=initial_cash,
            execution_mode=execution_mode,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
            window_id=window_id,
            train_start=train_df["date"].min(),
            train_end=train_df["date"].max(),
            test_start=test_df["date"].min(),
            test_end=test_df["date"].max(),
        )
        rows.append(window_results)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    for column in ["train_start", "train_end", "test_start", "test_end"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column]).dt.strftime("%Y-%m-%d")
    return result
