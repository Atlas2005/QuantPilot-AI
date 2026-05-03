from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .backtester import run_long_only_backtest_with_trades
    from .indicators import add_all_indicators
    from .metrics import summarize_performance
    from .model_predictor import load_model_bundle, load_prediction_input, prepare_features
    from .strategy import generate_ma_crossover_signals
    from .trade_metrics import summarize_trade_metrics
except ImportError:
    from backtester import run_long_only_backtest_with_trades
    from indicators import add_all_indicators
    from metrics import summarize_performance
    from model_predictor import load_model_bundle, load_prediction_input, prepare_features
    from strategy import generate_ma_crossover_signals
    from trade_metrics import summarize_trade_metrics


REQUIRED_MARKET_COLUMNS = ["date", "open", "close"]


def validate_thresholds(buy_threshold: float, sell_threshold: float) -> None:
    if not 0 <= sell_threshold <= buy_threshold <= 1:
        raise ValueError(
            "Thresholds must satisfy 0 <= sell_threshold <= buy_threshold <= 1."
        )


def validate_market_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "ML signal backtest input is missing required market columns: "
            f"{missing}"
        )


def predict_probabilities_for_rows(
    model,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """Generate per-row class-1 probabilities for a factor or ML split CSV."""
    features = prepare_features(df, feature_columns)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classifier = getattr(model, "named_steps", {}).get("classifier")
        classes = getattr(classifier, "classes_", getattr(model, "classes_", []))
        if 1 in classes:
            return pd.Series(
                probabilities[:, list(classes).index(1)],
                index=df.index,
                dtype="float64",
            )
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return pd.Series(probabilities[:, 1], index=df.index, dtype="float64")

    predictions = model.predict(features)
    return pd.Series(predictions, index=df.index, dtype="float64")


def probabilities_to_signals(
    probabilities: pd.Series,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.50,
) -> pd.Series:
    """
    Convert probabilities to long/flat trade actions.

    Signal convention matches the existing backtester:
    - 1 means buy/open long when flat
    - -1 means sell/close long when holding
    - 0 means no action
    """
    validate_thresholds(buy_threshold, sell_threshold)

    holding = False
    signals = []
    for probability in probabilities:
        if pd.isna(probability):
            signals.append(0)
            continue

        if not holding and probability >= buy_threshold:
            signals.append(1)
            holding = True
        elif holding and probability < sell_threshold:
            signals.append(-1)
            holding = False
        else:
            signals.append(0)

    return pd.Series(signals, index=probabilities.index, dtype="int64")


def prepare_ml_signal_data(
    model_dir: str | Path,
    factor_csv: str | Path,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load model artifacts and factor data, then add probabilities and signals."""
    model_dir_path = Path(model_dir)
    model_candidates = sorted(model_dir_path.glob("*.joblib"))
    if not model_candidates:
        raise FileNotFoundError(f"No .joblib model found in {model_dir_path}")

    model_path = model_candidates[0]
    model, metrics, feature_columns = load_model_bundle(model_path)
    source_df = load_prediction_input(factor_csv)
    validate_market_columns(source_df)

    result = source_df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "open", "close"]).sort_values("date")
    result = result.reset_index(drop=True)

    probabilities = predict_probabilities_for_rows(model, result, feature_columns)
    result["prediction_probability"] = probabilities.values
    result["signal"] = probabilities_to_signals(
        result["prediction_probability"],
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).values

    metadata = {
        "model_path": str(model_path),
        "feature_count": len(feature_columns),
        "metrics": metrics,
    }
    return result, metadata


def calculate_buy_and_hold_summary(
    df: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    """Calculate a simple buy-and-hold benchmark over the same close series."""
    if df.empty:
        return {
            "benchmark_final_value": None,
            "benchmark_return_pct": None,
            "benchmark_max_drawdown_pct": None,
        }

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty or close.iloc[0] <= 0:
        return {
            "benchmark_final_value": None,
            "benchmark_return_pct": None,
            "benchmark_max_drawdown_pct": None,
        }

    shares = initial_cash / close.iloc[0]
    values = close * shares
    running_max = values.cummax()
    drawdown = values / running_max - 1
    final_value = float(values.iloc[-1])
    return {
        "benchmark_final_value": final_value,
        "benchmark_return_pct": float((final_value - initial_cash) / initial_cash * 100),
        "benchmark_max_drawdown_pct": float(drawdown.min() * 100),
    }


def run_rule_based_comparison(
    df: pd.DataFrame,
    initial_cash: float,
    execution_mode: str,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_pct: float,
    min_commission: float,
) -> dict[str, Any]:
    """Run the existing MA-crossover strategy for comparison only."""
    required = ["date", "open", "high", "low", "close", "volume"]
    if any(column not in df.columns for column in required):
        return {"available": False, "reason": "Input is missing OHLCV columns."}

    strategy_df = add_all_indicators(df[required].copy())
    strategy_df = generate_ma_crossover_signals(strategy_df)
    backtest_df, trades_df = run_long_only_backtest_with_trades(
        strategy_df,
        initial_cash=initial_cash,
        execution_mode=execution_mode,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_pct=slippage_pct,
        min_commission=min_commission,
    )
    return {
        "available": True,
        "performance": summarize_performance(backtest_df),
        "trade_metrics": summarize_trade_metrics(trades_df),
    }


def has_suspicious_perfect_metrics(metrics: dict[str, Any]) -> bool:
    values = []
    for split_name in ["validation_metrics", "test_metrics"]:
        split_metrics = metrics.get(split_name, {})
        values.extend(
            split_metrics.get(key)
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        )
    valid_values = [value for value in values if value is not None]
    return bool(valid_values) and all(value >= 0.98 for value in valid_values)


def run_ml_signal_backtest(
    model_dir: str | Path,
    factor_csv: str | Path,
    initial_cash: float = 10000.0,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.50,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0,
    stamp_tax_rate: float = 0.0,
    slippage_pct: float = 0.0,
    min_commission: float = 0.0,
    compare_rule_based: bool = True,
) -> dict[str, Any]:
    """Run an ML probability signal through the existing long-only backtester."""
    signal_df, metadata = prepare_ml_signal_data(
        model_dir=model_dir,
        factor_csv=factor_csv,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    backtest_df, trades_df = run_long_only_backtest_with_trades(
        signal_df,
        initial_cash=initial_cash,
        execution_mode=execution_mode,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_pct=slippage_pct,
        min_commission=min_commission,
    )
    performance = summarize_performance(backtest_df)
    trade_metrics = summarize_trade_metrics(trades_df)
    benchmark = calculate_buy_and_hold_summary(signal_df, initial_cash)
    benchmark_return = benchmark.get("benchmark_return_pct")
    performance["strategy_vs_benchmark_pct"] = (
        None
        if benchmark_return is None
        else performance["total_return_pct"] - benchmark_return
    )

    rule_based = None
    if compare_rule_based:
        rule_based = run_rule_based_comparison(
            signal_df,
            initial_cash=initial_cash,
            execution_mode=execution_mode,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
        )

    warnings = [
        "ML signal backtest is educational research only and is not a trading recommendation."
    ]
    if has_suspicious_perfect_metrics(metadata.get("metrics", {})):
        warnings.append(
            "Saved model metrics are suspiciously close to perfect. Treat this backtest "
            "as a diagnostic workflow, not evidence of reliable performance."
        )

    return {
        "signal_data": signal_df,
        "backtest": backtest_df,
        "trades": trades_df,
        "performance": performance,
        "trade_metrics": trade_metrics,
        "benchmark": benchmark,
        "rule_based_comparison": rule_based,
        "metadata": metadata,
        "warnings": warnings,
    }
