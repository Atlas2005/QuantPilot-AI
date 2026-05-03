import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .backtester import run_long_only_backtest_with_trades
    from .dataset_splitter import (
        check_for_leakage_columns,
        chronological_split,
        clean_factor_dataset,
        infer_feature_columns,
        load_factor_dataset,
        normalize_date_column,
        validate_required_columns,
    )
    from .factor_ablation import parse_csv_list, parse_model_types
    from .factor_pruning_experiment import (
        DEFAULT_PRUNING_MODES,
        build_feature_sets,
        load_pruning_recommendations,
        parse_pruning_modes,
    )
    from .metrics import summarize_performance
    from .ml_signal_backtester import (
        calculate_buy_and_hold_summary,
        predict_probabilities_for_rows,
        probabilities_to_signals,
    )
    from .model_trainer import train_baseline_model
    from .trade_metrics import summarize_trade_metrics
except ImportError:
    from backtester import run_long_only_backtest_with_trades
    from dataset_splitter import (
        check_for_leakage_columns,
        chronological_split,
        clean_factor_dataset,
        infer_feature_columns,
        load_factor_dataset,
        normalize_date_column,
        validate_required_columns,
    )
    from factor_ablation import parse_csv_list, parse_model_types
    from factor_pruning_experiment import (
        DEFAULT_PRUNING_MODES,
        build_feature_sets,
        load_pruning_recommendations,
        parse_pruning_modes,
    )
    from metrics import summarize_performance
    from ml_signal_backtester import (
        calculate_buy_and_hold_summary,
        predict_probabilities_for_rows,
        probabilities_to_signals,
    )
    from model_trainer import train_baseline_model
    from trade_metrics import summarize_trade_metrics


RESULT_COLUMNS = [
    "symbol",
    "model_type",
    "pruning_mode",
    "feature_count",
    "buy_threshold",
    "sell_threshold",
    "total_return_pct",
    "benchmark_return_pct",
    "strategy_vs_benchmark_pct",
    "max_drawdown_pct",
    "trade_count",
    "win_rate_pct",
    "final_value",
    "warning",
]


def identify_safe_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    feature_columns = infer_feature_columns(df, target_col)
    leakage_checks = check_for_leakage_columns(feature_columns, target_col)
    if not leakage_checks["passed"]:
        raise ValueError(
            "Leakage-safe feature inference failed: "
            f"{leakage_checks['leakage_columns']}"
        )
    return feature_columns


def _infer_symbol(df: pd.DataFrame, factor_csv: str | Path) -> str:
    if "symbol" in df.columns and not df["symbol"].dropna().empty:
        return str(df["symbol"].dropna().iloc[0])
    stem = Path(factor_csv).stem
    return stem.replace("factors_", "", 1)


def _extract_final_value(performance: dict[str, Any]) -> float | None:
    for key in ["final_value", "ending_value", "portfolio_value"]:
        if key in performance:
            return performance[key]
    return None


def run_one_reduced_feature_backtest(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
    pruning_mode: str,
    symbol: str,
    target_col: str,
    initial_cash: float,
    buy_threshold: float,
    sell_threshold: float,
    execution_mode: str,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_pct: float,
    min_commission: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    warning_parts = []
    if not feature_columns:
        warning = "No features selected for this pruning mode."
        return {
            "symbol": symbol,
            "model_type": model_type,
            "pruning_mode": pruning_mode,
            "feature_count": 0,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "total_return_pct": None,
            "benchmark_return_pct": None,
            "strategy_vs_benchmark_pct": None,
            "max_drawdown_pct": None,
            "trade_count": 0,
            "win_rate_pct": None,
            "final_value": None,
            "warning": warning,
        }, pd.DataFrame()

    cleaned_df, _ = clean_factor_dataset(df, feature_columns, target_col)
    train_df, validation_df, test_df = chronological_split(
        cleaned_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        purge_rows=purge_rows,
        split_mode="global_date",
    )
    if len(test_df) < 50:
        warning_parts.append(f"Small test sample: {len(test_df)} rows.")

    model, training_info = train_baseline_model(
        train_df=train_df,
        feature_columns=feature_columns,
        target_col=target_col,
        model_name=model_type,
    )
    if training_info.get("single_class_training"):
        warning_parts.append("Training split had one target class.")

    signal_df = test_df.copy().reset_index(drop=True)
    probabilities = predict_probabilities_for_rows(model, signal_df, feature_columns)
    signal_df["prediction_probability"] = probabilities.values
    signal_df["signal"] = probabilities_to_signals(
        signal_df["prediction_probability"],
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).values

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
    total_return = performance.get("total_return_pct")
    strategy_vs_benchmark = (
        None
        if benchmark_return is None or total_return is None
        else total_return - benchmark_return
    )
    trade_count = int(trade_metrics.get("total_trades") or 0)
    if trade_count == 0:
        warning_parts.append("No trades were executed.")

    result = {
        "symbol": symbol,
        "model_type": model_type,
        "pruning_mode": pruning_mode,
        "feature_count": len(feature_columns),
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "total_return_pct": total_return,
        "benchmark_return_pct": benchmark_return,
        "strategy_vs_benchmark_pct": strategy_vs_benchmark,
        "max_drawdown_pct": performance.get("max_drawdown_pct"),
        "trade_count": trade_count,
        "win_rate_pct": trade_metrics.get("win_rate_pct"),
        "final_value": _extract_final_value(performance),
        "warning": " | ".join(warning_parts) if warning_parts else None,
    }
    return result, trades_df


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    return (
        results_df.groupby(["pruning_mode", "model_type"], dropna=False)
        .agg(
            symbol_count=("symbol", "nunique"),
            avg_feature_count=("feature_count", "mean"),
            avg_total_return_pct=("total_return_pct", "mean"),
            avg_benchmark_return_pct=("benchmark_return_pct", "mean"),
            avg_strategy_vs_benchmark_pct=("strategy_vs_benchmark_pct", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
            avg_trade_count=("trade_count", "mean"),
            avg_win_rate_pct=("win_rate_pct", "mean"),
            avg_final_value=("final_value", "mean"),
        )
        .reset_index()
        .sort_values("avg_strategy_vs_benchmark_pct", ascending=False)
    )


def run_reduced_feature_backtest(
    factor_csv: str | Path,
    recommendations_path: str | Path,
    model_types: list[str] | None = None,
    pruning_modes: list[str] | None = None,
    target_col: str = "label_up_5d",
    initial_cash: float = 10000.0,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.50,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0,
    stamp_tax_rate: float = 0.0,
    slippage_pct: float = 0.0,
    min_commission: float = 0.0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    purge_rows: int = 5,
) -> dict[str, Any]:
    if model_types is None:
        model_types = ["logistic_regression", "random_forest"]
    if pruning_modes is None:
        pruning_modes = DEFAULT_PRUNING_MODES.copy()

    df = normalize_date_column(load_factor_dataset(factor_csv))
    validate_required_columns(df, target_col)
    symbol = _infer_symbol(df, factor_csv)
    safe_features = identify_safe_feature_columns(df, target_col)
    recommendations_df = load_pruning_recommendations(recommendations_path)
    feature_sets = build_feature_sets(safe_features, recommendations_df, pruning_modes)

    rows = []
    warning_rows = []
    for model_type in model_types:
        for mode in pruning_modes:
            row, _ = run_one_reduced_feature_backtest(
                df=df,
                feature_columns=feature_sets.get(mode, []),
                model_type=model_type,
                pruning_mode=mode,
                symbol=symbol,
                target_col=target_col,
                initial_cash=initial_cash,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                execution_mode=execution_mode,
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax_rate,
                slippage_pct=slippage_pct,
                min_commission=min_commission,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                purge_rows=purge_rows,
            )
            rows.append(row)
            if row.get("warning"):
                warning_rows.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "pruning_mode": mode,
                        "warning": row["warning"],
                    }
                )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    warnings_df = pd.DataFrame(
        warning_rows,
        columns=["symbol", "model_type", "pruning_mode", "warning"],
    )
    return {
        "results": results_df,
        "summary": build_summary(results_df),
        "warnings": warnings_df,
    }


def save_reduced_feature_backtest_outputs(
    output_dir: str | Path,
    result: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_path / "reduced_feature_backtest_results.csv",
        "summary": output_path / "reduced_feature_backtest_summary.csv",
        "warnings": output_path / "warnings.csv",
        "run_config": output_path / "run_config.json",
    }
    result["results"].to_csv(paths["results"], index=False)
    result["summary"].to_csv(paths["summary"], index=False)
    result["warnings"].to_csv(paths["warnings"], index=False)
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def run_and_save_reduced_feature_backtest(
    factor_csv: str | Path,
    recommendations_path: str | Path,
    output_dir: str | Path,
    model_types: list[str] | None = None,
    pruning_modes: list[str] | None = None,
    target_col: str = "label_up_5d",
    initial_cash: float = 10000.0,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.50,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0,
    stamp_tax_rate: float = 0.0,
    slippage_pct: float = 0.0,
    min_commission: float = 0.0,
) -> dict[str, Any]:
    result = run_reduced_feature_backtest(
        factor_csv=factor_csv,
        recommendations_path=recommendations_path,
        model_types=model_types,
        pruning_modes=pruning_modes,
        target_col=target_col,
        initial_cash=initial_cash,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        execution_mode=execution_mode,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_pct=slippage_pct,
        min_commission=min_commission,
    )
    run_config = {
        "factor_csv": str(factor_csv),
        "recommendations_path": str(recommendations_path),
        "output_dir": str(output_dir),
        "model_types": model_types,
        "pruning_modes": pruning_modes,
        "target_col": target_col,
        "initial_cash": initial_cash,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "execution_mode": execution_mode,
        "commission_rate": commission_rate,
        "stamp_tax_rate": stamp_tax_rate,
        "slippage_pct": slippage_pct,
        "min_commission": min_commission,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    output_files = save_reduced_feature_backtest_outputs(output_dir, result, run_config)
    result["run_config"] = run_config
    result["output_files"] = output_files
    return result
