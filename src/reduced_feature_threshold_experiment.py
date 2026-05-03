import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .backtester import run_long_only_backtest_with_trades
    from .dataset_splitter import (
        chronological_split,
        clean_factor_dataset,
        load_factor_dataset,
        normalize_date_column,
        validate_required_columns,
    )
    from .factor_ablation import parse_model_types
    from .factor_pruning_experiment import (
        DEFAULT_PRUNING_MODES,
        build_feature_sets,
        identify_safe_feature_columns,
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
        chronological_split,
        clean_factor_dataset,
        load_factor_dataset,
        normalize_date_column,
        validate_required_columns,
    )
    from factor_ablation import parse_model_types
    from factor_pruning_experiment import (
        DEFAULT_PRUNING_MODES,
        build_feature_sets,
        identify_safe_feature_columns,
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


DEFAULT_BUY_THRESHOLDS = [0.50, 0.55, 0.60, 0.65]
DEFAULT_SELL_THRESHOLDS = [0.35, 0.40, 0.45, 0.50]
DEFAULT_MODELS = ["logistic_regression", "random_forest"]


def parse_thresholds(text: str | None, defaults: list[float]) -> list[float]:
    if text is None or not text.strip():
        return defaults.copy()
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Threshold list cannot be empty.")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Thresholds must be between 0 and 1.")
    return sorted(set(values))


def threshold_pairs(
    buy_thresholds: list[float],
    sell_thresholds: list[float],
) -> list[tuple[float, float]]:
    return [
        (buy_threshold, sell_threshold)
        for buy_threshold in buy_thresholds
        for sell_threshold in sell_thresholds
        if buy_threshold > sell_threshold
    ]


def infer_symbol(df: pd.DataFrame, factor_csv: str | Path) -> str:
    if "symbol" in df.columns and not df["symbol"].dropna().empty:
        return str(df["symbol"].dropna().iloc[0])
    return Path(factor_csv).stem.replace("factors_", "", 1)


def _extract_final_value(performance: dict[str, Any]) -> float | None:
    for key in ["final_value", "ending_value", "portfolio_value"]:
        if key in performance:
            return performance[key]
    return None


def _date_min(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d")


def _date_max(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")


def run_threshold_grid_from_probabilities(
    signal_df: pd.DataFrame,
    probabilities: pd.Series,
    symbol: str,
    model_type: str,
    pruning_mode: str,
    feature_count: int,
    buy_thresholds: list[float],
    sell_thresholds: list[float],
    initial_cash: float,
    execution_mode: str,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_pct: float,
    min_commission: float,
    min_trades: int,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for buy_threshold, sell_threshold in threshold_pairs(buy_thresholds, sell_thresholds):
        backtest_input = signal_df.copy().reset_index(drop=True)
        backtest_input["prediction_probability"] = probabilities.values
        backtest_input["signal"] = probabilities_to_signals(
            backtest_input["prediction_probability"],
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
        performance = summarize_performance(backtest_df)
        trade_metrics = summarize_trade_metrics(trades_df)
        benchmark = calculate_buy_and_hold_summary(backtest_input, initial_cash)
        benchmark_return = benchmark.get("benchmark_return_pct")
        total_return = performance.get("total_return_pct")
        trade_count = int(trade_metrics.get("total_trades") or 0)
        warnings = []
        if trade_count <= min_trades:
            warnings.append(f"low_trade_count: {trade_count}")
        if total_return is not None and total_return < 0:
            warnings.append("negative_total_return")
        strategy_vs_benchmark = (
            None
            if benchmark_return is None or total_return is None
            else total_return - benchmark_return
        )
        if strategy_vs_benchmark is not None and strategy_vs_benchmark < 0:
            warnings.append("underperformed_benchmark")

        row = {
            "symbol": symbol,
            "model_type": model_type,
            "pruning_mode": pruning_mode,
            "feature_count": feature_count,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "total_return_pct": total_return,
            "benchmark_return_pct": benchmark_return,
            "strategy_vs_benchmark_pct": strategy_vs_benchmark,
            "max_drawdown_pct": performance.get("max_drawdown_pct"),
            "trade_count": trade_count,
            "win_rate_pct": trade_metrics.get("win_rate_pct"),
            "final_value": _extract_final_value(performance),
            "warning": " | ".join(warnings) if warnings else None,
        }
        if extra_fields:
            row.update(extra_fields)
        rows.append(row)
    return rows


def _score_group(group: pd.DataFrame, min_trades: int) -> dict[str, Any]:
    excess = pd.to_numeric(group["strategy_vs_benchmark_pct"], errors="coerce")
    drawdown = pd.to_numeric(group["max_drawdown_pct"], errors="coerce")
    trades = pd.to_numeric(group["trade_count"], errors="coerce")
    avg_excess = excess.mean()
    avg_drawdown = drawdown.mean()
    sufficient_trade_rate = float((trades > min_trades).mean()) if not trades.empty else 0.0
    beat_rate = float((excess > 0).mean()) if not excess.dropna().empty else 0.0
    stability_score = (
        (0.0 if pd.isna(avg_excess) else avg_excess / 100.0)
        + beat_rate * 0.40
        + sufficient_trade_rate * 0.20
        + (0.0 if pd.isna(avg_drawdown) else avg_drawdown / 200.0)
    )
    return {
        "avg_feature_count": pd.to_numeric(group["feature_count"], errors="coerce").mean(),
        "avg_total_return_pct": pd.to_numeric(group["total_return_pct"], errors="coerce").mean(),
        "avg_benchmark_return_pct": pd.to_numeric(group["benchmark_return_pct"], errors="coerce").mean(),
        "avg_strategy_vs_benchmark_pct": avg_excess,
        "avg_max_drawdown_pct": avg_drawdown,
        "avg_trade_count": trades.mean(),
        "avg_win_rate_pct": pd.to_numeric(group["win_rate_pct"], errors="coerce").mean(),
        "avg_final_value": pd.to_numeric(group["final_value"], errors="coerce").mean(),
        "threshold_count": len(group),
        "beat_benchmark_rate": beat_rate,
        "sufficient_trade_rate": sufficient_trade_rate,
        "stability_score": stability_score,
    }


def summarize_threshold_results(
    results_df: pd.DataFrame,
    group_columns: list[str],
    min_trades: int,
) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in results_df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(_score_group(group, min_trades))
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["stability_score", "avg_strategy_vs_benchmark_pct", "avg_max_drawdown_pct"],
            ascending=[False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_best_thresholds(results_df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["symbol", "model_type", "pruning_mode"]
    for keys, group in results_df.groupby(group_columns, dropna=False):
        ranked = group.copy()
        ranked["threshold_score"] = (
            pd.to_numeric(ranked["strategy_vs_benchmark_pct"], errors="coerce").fillna(0)
            / 100.0
            + (pd.to_numeric(ranked["trade_count"], errors="coerce") > min_trades).astype(float)
            * 0.20
            + pd.to_numeric(ranked["max_drawdown_pct"], errors="coerce").fillna(0)
            / 200.0
        )
        best = ranked.sort_values(
            ["threshold_score", "strategy_vs_benchmark_pct", "max_drawdown_pct"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows).reset_index(drop=True)


def _prepare_feature_sets(
    factor_csv: str | Path,
    recommendations_path: str | Path,
    target_col: str,
    pruning_modes: list[str],
) -> tuple[pd.DataFrame, str, dict[str, list[str]]]:
    df = normalize_date_column(load_factor_dataset(factor_csv))
    validate_required_columns(df, target_col)
    symbol = infer_symbol(df, factor_csv)
    safe_features = identify_safe_feature_columns(df, target_col)
    recommendations_df = load_pruning_recommendations(recommendations_path)
    feature_sets = build_feature_sets(safe_features, recommendations_df, pruning_modes)
    return df, symbol, feature_sets


def run_reduced_feature_threshold_experiment(
    factor_csv: str | Path,
    recommendations_path: str | Path,
    model_types: list[str] | None = None,
    pruning_modes: list[str] | None = None,
    target_col: str = "label_up_5d",
    buy_thresholds: list[float] | None = None,
    sell_thresholds: list[float] | None = None,
    initial_cash: float = 10000.0,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    min_commission: float = 5.0,
    min_trades: int = 3,
    purge_rows: int = 5,
) -> dict[str, Any]:
    model_types = model_types or DEFAULT_MODELS.copy()
    pruning_modes = pruning_modes or DEFAULT_PRUNING_MODES.copy()
    buy_thresholds = buy_thresholds or DEFAULT_BUY_THRESHOLDS.copy()
    sell_thresholds = sell_thresholds or DEFAULT_SELL_THRESHOLDS.copy()

    df, symbol, feature_sets = _prepare_feature_sets(
        factor_csv,
        recommendations_path,
        target_col,
        pruning_modes,
    )
    rows = []
    warning_rows = []
    for model_type in model_types:
        for pruning_mode in pruning_modes:
            features = feature_sets.get(pruning_mode, [])
            if not features:
                warning_rows.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "pruning_mode": pruning_mode,
                        "warning_type": "empty_feature_set",
                        "message": "No features selected for this pruning mode.",
                    }
                )
                continue

            cleaned_df, _ = clean_factor_dataset(df, features, target_col)
            train_df, _, test_df = chronological_split(
                cleaned_df,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                purge_rows=purge_rows,
                split_mode="global_date",
            )
            if train_df.empty or test_df.empty:
                warning_rows.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "pruning_mode": pruning_mode,
                        "warning_type": "empty_split",
                        "message": "Train or test split is empty.",
                    }
                )
                continue
            model, training_info = train_baseline_model(
                train_df,
                features,
                target_col=target_col,
                model_name=model_type,
            )
            if training_info.get("single_class_training"):
                warning_rows.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "pruning_mode": pruning_mode,
                        "warning_type": "single_class_training",
                        "message": "Training split had one target class.",
                    }
                )
            test_signal_df = test_df.copy().reset_index(drop=True)
            probabilities = predict_probabilities_for_rows(model, test_signal_df, features)
            rows.extend(
                run_threshold_grid_from_probabilities(
                    signal_df=test_signal_df,
                    probabilities=probabilities,
                    symbol=symbol,
                    model_type=model_type,
                    pruning_mode=pruning_mode,
                    feature_count=len(features),
                    buy_thresholds=buy_thresholds,
                    sell_thresholds=sell_thresholds,
                    initial_cash=initial_cash,
                    execution_mode=execution_mode,
                    commission_rate=commission_rate,
                    stamp_tax_rate=stamp_tax_rate,
                    slippage_pct=slippage_pct,
                    min_commission=min_commission,
                    min_trades=min_trades,
                )
            )

    results_df = pd.DataFrame(rows)
    warnings_df = pd.DataFrame(warning_rows)
    if not results_df.empty:
        result_warnings = results_df[results_df["warning"].notna()].copy()
        for _, row in result_warnings.iterrows():
            warning_rows.append(
                {
                    "symbol": row["symbol"],
                    "model_type": row["model_type"],
                    "pruning_mode": row["pruning_mode"],
                    "buy_threshold": row["buy_threshold"],
                    "sell_threshold": row["sell_threshold"],
                    "warning_type": "threshold_result_warning",
                    "message": row["warning"],
                }
            )
        warnings_df = pd.DataFrame(warning_rows)

    return {
        "threshold_results": results_df,
        "threshold_summary_by_mode": summarize_threshold_results(
            results_df,
            ["pruning_mode"],
            min_trades,
        ),
        "threshold_summary_by_model": summarize_threshold_results(
            results_df,
            ["model_type"],
            min_trades,
        ),
        "threshold_summary_by_mode_model": summarize_threshold_results(
            results_df,
            ["pruning_mode", "model_type"],
            min_trades,
        ),
        "best_thresholds": build_best_thresholds(results_df, min_trades),
        "warnings": warnings_df,
    }


def build_walk_forward_windows(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    step_ratio: float,
    purge_rows: int,
) -> list[dict[str, Any]]:
    sorted_df = normalize_date_column(df).reset_index(drop=True)
    row_count = len(sorted_df)
    train_rows = max(1, int(row_count * train_ratio))
    validation_rows = max(1, int(row_count * validation_ratio))
    test_rows = max(1, int(row_count * test_ratio))
    step_rows = max(1, int(row_count * step_ratio))
    windows = []
    start = 0
    window_id = 1
    while start + train_rows + validation_rows + test_rows + purge_rows * 2 <= row_count:
        train_start = start
        train_end = train_start + train_rows
        validation_start = train_end + purge_rows
        validation_end = validation_start + validation_rows
        test_start = validation_end + purge_rows
        test_end = test_start + test_rows
        windows.append(
            {
                "window_id": window_id,
                "train": sorted_df.iloc[train_start:train_end].copy(),
                "validation": sorted_df.iloc[validation_start:validation_end].copy(),
                "test": sorted_df.iloc[test_start:test_end].copy(),
            }
        )
        start += step_rows
        window_id += 1
    return windows


def run_reduced_feature_walk_forward_experiment(
    factor_csv: str | Path,
    recommendations_path: str | Path,
    model_types: list[str] | None = None,
    pruning_modes: list[str] | None = None,
    target_col: str = "label_up_5d",
    buy_thresholds: list[float] | None = None,
    sell_thresholds: list[float] | None = None,
    initial_cash: float = 10000.0,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    min_commission: float = 5.0,
    min_trades: int = 3,
    train_ratio: float = 0.50,
    validation_ratio: float = 0.20,
    test_ratio: float = 0.20,
    step_ratio: float = 0.10,
    purge_rows: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_types = model_types or DEFAULT_MODELS.copy()
    pruning_modes = pruning_modes or DEFAULT_PRUNING_MODES.copy()
    buy_thresholds = buy_thresholds or DEFAULT_BUY_THRESHOLDS.copy()
    sell_thresholds = sell_thresholds or DEFAULT_SELL_THRESHOLDS.copy()
    df, symbol, feature_sets = _prepare_feature_sets(
        factor_csv,
        recommendations_path,
        target_col,
        pruning_modes,
    )
    rows = []
    warnings = []
    for model_type in model_types:
        for pruning_mode in pruning_modes:
            features = feature_sets.get(pruning_mode, [])
            if not features:
                warnings.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "pruning_mode": pruning_mode,
                        "warning_type": "empty_feature_set",
                        "message": "No walk-forward features selected.",
                    }
                )
                continue
            cleaned_df, _ = clean_factor_dataset(df, features, target_col)
            windows = build_walk_forward_windows(
                cleaned_df,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                step_ratio=step_ratio,
                purge_rows=purge_rows,
            )
            if not windows:
                warnings.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "pruning_mode": pruning_mode,
                        "warning_type": "no_walk_forward_windows",
                        "message": "Not enough rows for walk-forward windows.",
                    }
                )
                continue
            for window in windows:
                train_df = window["train"]
                validation_df = window["validation"]
                test_df = window["test"]
                if train_df.empty or test_df.empty:
                    continue
                model, training_info = train_baseline_model(
                    train_df,
                    features,
                    target_col=target_col,
                    model_name=model_type,
                )
                if training_info.get("single_class_training"):
                    warnings.append(
                        {
                            "symbol": symbol,
                            "model_type": model_type,
                            "pruning_mode": pruning_mode,
                            "window_id": window["window_id"],
                            "warning_type": "single_class_training",
                            "message": "Walk-forward train window had one class.",
                        }
                    )
                test_signal_df = test_df.copy().reset_index(drop=True)
                probabilities = predict_probabilities_for_rows(
                    model,
                    test_signal_df,
                    features,
                )
                rows.extend(
                    run_threshold_grid_from_probabilities(
                        signal_df=test_signal_df,
                        probabilities=probabilities,
                        symbol=symbol,
                        model_type=model_type,
                        pruning_mode=pruning_mode,
                        feature_count=len(features),
                        buy_thresholds=buy_thresholds,
                        sell_thresholds=sell_thresholds,
                        initial_cash=initial_cash,
                        execution_mode=execution_mode,
                        commission_rate=commission_rate,
                        stamp_tax_rate=stamp_tax_rate,
                        slippage_pct=slippage_pct,
                        min_commission=min_commission,
                        min_trades=min_trades,
                        extra_fields={
                            "window_id": window["window_id"],
                            "train_start": _date_min(train_df),
                            "train_end": _date_max(train_df),
                            "validation_start": _date_min(validation_df),
                            "validation_end": _date_max(validation_df),
                            "test_start": _date_min(test_df),
                            "test_end": _date_max(test_df),
                            "train_rows": len(train_df),
                            "validation_rows": len(validation_df),
                            "test_rows": len(test_df),
                        },
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(warnings)


def save_threshold_experiment_outputs(
    output_dir: str | Path,
    result: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "threshold_results": output_path / "threshold_backtest_results.csv",
        "threshold_summary_by_mode": output_path / "threshold_summary_by_mode.csv",
        "threshold_summary_by_model": output_path / "threshold_summary_by_model.csv",
        "threshold_summary_by_mode_model": output_path
        / "threshold_summary_by_mode_model.csv",
        "best_thresholds": output_path / "best_thresholds.csv",
        "warnings": output_path / "warnings.csv",
        "run_config": output_path / "run_config.json",
    }
    for key, path in paths.items():
        if key == "run_config":
            continue
        result.get(key, pd.DataFrame()).to_csv(path, index=False)
    if "walk_forward_results" in result:
        paths["walk_forward_results"] = output_path / "walk_forward_results.csv"
        paths["walk_forward_summary"] = output_path / "walk_forward_summary.csv"
        result["walk_forward_results"].to_csv(paths["walk_forward_results"], index=False)
        result["walk_forward_summary"].to_csv(paths["walk_forward_summary"], index=False)
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def run_and_save_threshold_experiment(
    factor_csv: str | Path,
    recommendations_path: str | Path,
    output_dir: str | Path,
    model_types: list[str] | None = None,
    pruning_modes: list[str] | None = None,
    target_col: str = "label_up_5d",
    buy_thresholds: list[float] | None = None,
    sell_thresholds: list[float] | None = None,
    initial_cash: float = 10000.0,
    execution_mode: str = "same_close",
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    min_commission: float = 5.0,
    min_trades: int = 3,
    enable_walk_forward: bool = False,
    walk_forward_train_ratio: float = 0.50,
    walk_forward_validation_ratio: float = 0.20,
    walk_forward_test_ratio: float = 0.20,
    walk_forward_step_ratio: float = 0.10,
    purge_rows: int = 5,
) -> dict[str, Any]:
    result = run_reduced_feature_threshold_experiment(
        factor_csv=factor_csv,
        recommendations_path=recommendations_path,
        model_types=model_types,
        pruning_modes=pruning_modes,
        target_col=target_col,
        buy_thresholds=buy_thresholds,
        sell_thresholds=sell_thresholds,
        initial_cash=initial_cash,
        execution_mode=execution_mode,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_pct=slippage_pct,
        min_commission=min_commission,
        min_trades=min_trades,
        purge_rows=purge_rows,
    )
    if enable_walk_forward:
        walk_forward_results, walk_forward_warnings = (
            run_reduced_feature_walk_forward_experiment(
                factor_csv=factor_csv,
                recommendations_path=recommendations_path,
                model_types=model_types,
                pruning_modes=pruning_modes,
                target_col=target_col,
                buy_thresholds=buy_thresholds,
                sell_thresholds=sell_thresholds,
                initial_cash=initial_cash,
                execution_mode=execution_mode,
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax_rate,
                slippage_pct=slippage_pct,
                min_commission=min_commission,
                min_trades=min_trades,
                train_ratio=walk_forward_train_ratio,
                validation_ratio=walk_forward_validation_ratio,
                test_ratio=walk_forward_test_ratio,
                step_ratio=walk_forward_step_ratio,
                purge_rows=purge_rows,
            )
        )
        result["walk_forward_results"] = walk_forward_results
        result["walk_forward_summary"] = summarize_threshold_results(
            walk_forward_results,
            ["pruning_mode", "model_type", "buy_threshold", "sell_threshold"],
            min_trades,
        )
        if not walk_forward_warnings.empty:
            result["warnings"] = pd.concat(
                [result["warnings"], walk_forward_warnings],
                ignore_index=True,
            )

    run_config = {
        "factor_csv": str(factor_csv),
        "recommendations_path": str(recommendations_path),
        "output_dir": str(output_dir),
        "model_types": model_types or DEFAULT_MODELS.copy(),
        "pruning_modes": pruning_modes or DEFAULT_PRUNING_MODES.copy(),
        "target_col": target_col,
        "buy_thresholds": buy_thresholds or DEFAULT_BUY_THRESHOLDS.copy(),
        "sell_thresholds": sell_thresholds or DEFAULT_SELL_THRESHOLDS.copy(),
        "initial_cash": initial_cash,
        "execution_mode": execution_mode,
        "commission_rate": commission_rate,
        "stamp_tax_rate": stamp_tax_rate,
        "slippage_pct": slippage_pct,
        "min_commission": min_commission,
        "min_trades": min_trades,
        "enable_walk_forward": enable_walk_forward,
        "walk_forward_train_ratio": walk_forward_train_ratio,
        "walk_forward_validation_ratio": walk_forward_validation_ratio,
        "walk_forward_test_ratio": walk_forward_test_ratio,
        "walk_forward_step_ratio": walk_forward_step_ratio,
        "purge_rows": purge_rows,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    output_files = save_threshold_experiment_outputs(output_dir, result, run_config)
    result["run_config"] = run_config
    result["output_files"] = output_files
    return result
