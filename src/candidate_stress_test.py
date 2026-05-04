import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .candidate_mode_normalization import add_canonical_mode_columns
    from .dataset_splitter import (
        chronological_split,
        clean_factor_dataset,
        load_factor_dataset,
        normalize_date_column,
        validate_required_columns,
    )
    from .factor_pruning_experiment import (
        build_feature_sets,
        identify_safe_feature_columns,
        load_pruning_recommendations,
    )
    from .ml_signal_backtester import predict_probabilities_for_rows
    from .model_trainer import train_baseline_model
    from .reduced_feature_threshold_experiment import run_threshold_grid_from_probabilities
except ImportError:
    from candidate_mode_normalization import add_canonical_mode_columns
    from dataset_splitter import (
        chronological_split,
        clean_factor_dataset,
        load_factor_dataset,
        normalize_date_column,
        validate_required_columns,
    )
    from factor_pruning_experiment import (
        build_feature_sets,
        identify_safe_feature_columns,
        load_pruning_recommendations,
    )
    from ml_signal_backtester import predict_probabilities_for_rows
    from model_trainer import train_baseline_model
    from reduced_feature_threshold_experiment import run_threshold_grid_from_probabilities


DEFAULT_SYMBOLS = ["000001", "600519", "000858", "600036", "601318"]
REGIME_POSITIVE_THRESHOLD = 0.03
REGIME_NEGATIVE_THRESHOLD = -0.03
NON_FACTOR_DIAGNOSTIC_COLUMNS = {
    "regime",
    "regime_return",
    "candidate_label",
    "warning",
    "warning_type",
    "message",
    "window_id",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "train_rows",
    "validation_rows",
    "test_rows",
    "regime_rows",
    "input_dir",
    "report_min_trades",
    "selection_confidence",
    "selection_note",
    "case_type",
    "reason",
    "final_decision",
    "decision_reason",
}


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


def _drop_non_factor_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [
        column
        for column in df.columns
        if column in NON_FACTOR_DIAGNOSTIC_COLUMNS
        or column.startswith("candidate_")
        or column.startswith("walk_forward_")
    ]
    return df.drop(columns=drop_columns, errors="ignore")


def _markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
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


def add_market_regime(
    df: pd.DataFrame,
    regime_window: int,
    positive_threshold: float = REGIME_POSITIVE_THRESHOLD,
    negative_threshold: float = REGIME_NEGATIVE_THRESHOLD,
) -> pd.DataFrame:
    result = normalize_date_column(df).copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    rolling_return = close.pct_change(periods=max(1, regime_window))
    result["regime_return"] = rolling_return
    result["regime"] = "sideways"
    result.loc[rolling_return > positive_threshold, "regime"] = "bull"
    result.loc[rolling_return < negative_threshold, "regime"] = "bear"
    return result


def _run_candidate_on_regime(
    df: pd.DataFrame,
    symbol: str,
    recommendations_df: pd.DataFrame,
    target_col: str,
    candidate_label: str,
    pruning_mode: str,
    model_type: str,
    buy_threshold: float,
    sell_threshold: float,
    initial_cash: float,
    execution_mode: str,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_pct: float,
    min_commission: float,
    min_trades: int,
    purge_rows: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    warnings = []
    feature_source_df = _drop_non_factor_diagnostics(df)
    feature_columns = identify_safe_feature_columns(feature_source_df, target_col)
    feature_sets = build_feature_sets(feature_columns, recommendations_df, [pruning_mode])
    features = feature_sets.get(pruning_mode, [])
    if not features:
        return rows, [
            {
                "candidate_label": candidate_label,
                "symbol": symbol,
                "pruning_mode": pruning_mode,
                "model_type": model_type,
                "warning_type": "empty_feature_set",
                "message": "No features selected for this pruning mode.",
            }
        ]

    for regime, regime_df in df.groupby("regime", dropna=False):
        model_input_df = _drop_non_factor_diagnostics(regime_df)
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
            warnings.append(
                {
                    "candidate_label": candidate_label,
                    "symbol": symbol,
                    "pruning_mode": pruning_mode,
                    "model_type": model_type,
                    "regime": regime,
                    "warning_type": "empty_regime_split",
                    "message": "Train or test split is empty for this regime.",
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
            warnings.append(
                {
                    "candidate_label": candidate_label,
                    "symbol": symbol,
                    "pruning_mode": pruning_mode,
                    "model_type": model_type,
                    "regime": regime,
                    "warning_type": "single_class_training",
                    "message": "Training split had one target class.",
                }
            )
        test_signal_df = test_df.copy().reset_index(drop=True)
        probabilities = predict_probabilities_for_rows(model, test_signal_df, features)
        candidate_rows = run_threshold_grid_from_probabilities(
            signal_df=test_signal_df,
            probabilities=probabilities,
            symbol=symbol,
            model_type=model_type,
            pruning_mode=pruning_mode,
            feature_count=len(features),
            buy_thresholds=[buy_threshold],
            sell_thresholds=[sell_threshold],
            initial_cash=initial_cash,
            execution_mode=execution_mode,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
            min_trades=min_trades,
            extra_fields={
                "candidate_label": candidate_label,
                "regime": regime,
                "regime_rows": len(regime_df),
                "test_rows": len(test_signal_df),
            },
        )
        for row in candidate_rows:
            row["symbol"] = _format_symbol(row.get("symbol"))
            row["legacy_pruning_mode"] = row.get("pruning_mode")
            row["canonical_mode"] = add_canonical_mode_columns(
                pd.DataFrame([row])
            )["canonical_mode"].iloc[0]
            rows.append(row)
            if row.get("warning"):
                warnings.append(
                    {
                        "candidate_label": candidate_label,
                        "symbol": row["symbol"],
                        "pruning_mode": pruning_mode,
                        "legacy_pruning_mode": pruning_mode,
                        "canonical_mode": row["canonical_mode"],
                        "model_type": model_type,
                        "buy_threshold": buy_threshold,
                        "sell_threshold": sell_threshold,
                        "regime": regime,
                        "warning_type": "threshold_result_warning",
                        "message": row.get("warning"),
                    }
                )
    return rows, warnings


def _summary(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    group_columns = [
        "canonical_mode",
        "model_type",
        "buy_threshold",
        "sell_threshold",
    ]
    for keys, group in df.groupby(group_columns, dropna=False):
        row = {column: value for column, value in zip(group_columns, keys)}
        row["candidate_labels"] = ",".join(
            sorted(group["candidate_label"].dropna().astype(str).unique())
        )
        row["legacy_pruning_modes"] = ",".join(
            sorted(group["legacy_pruning_mode"].dropna().astype(str).unique())
        )
        excess = _numeric(group, "strategy_vs_benchmark_pct")
        trades = _numeric(group, "trade_count")
        sufficient_trade_rate = float((trades >= min_trades).mean()) if not trades.empty else 0.0
        beat_rate = float((excess > 0).mean()) if not excess.dropna().empty else 0.0
        tested_symbol_count = int(group["symbol"].nunique()) if "symbol" in group else 0
        regime_count = int(group["regime"].nunique()) if "regime" in group else 0
        row.update(
            {
                "avg_total_return_pct": _numeric(group, "total_return_pct").mean(),
                "avg_benchmark_return_pct": _numeric(group, "benchmark_return_pct").mean(),
                "avg_strategy_vs_benchmark_pct": excess.mean(),
                "avg_max_drawdown_pct": _numeric(group, "max_drawdown_pct").mean(),
                "avg_trade_count": trades.mean(),
                "beat_benchmark_rate": beat_rate,
                "sufficient_trade_rate": sufficient_trade_rate,
                "tested_symbol_count": tested_symbol_count,
                "regime_count": regime_count,
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    return _add_final_decision(summary, df)


def _regime_summary(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["canonical_mode", "regime"]
    for keys, group in df.groupby(group_columns, dropna=False):
        row = {column: value for column, value in zip(group_columns, keys)}
        row["candidate_labels"] = ",".join(
            sorted(group["candidate_label"].dropna().astype(str).unique())
        )
        excess = _numeric(group, "strategy_vs_benchmark_pct")
        trades = _numeric(group, "trade_count")
        row.update(
            {
                "tested_symbol_count": int(group["symbol"].nunique()),
                "avg_total_return_pct": _numeric(group, "total_return_pct").mean(),
                "avg_benchmark_return_pct": _numeric(group, "benchmark_return_pct").mean(),
                "avg_strategy_vs_benchmark_pct": excess.mean(),
                "avg_max_drawdown_pct": _numeric(group, "max_drawdown_pct").mean(),
                "avg_trade_count": trades.mean(),
                "beat_benchmark_rate": float((excess > 0).mean()),
                "sufficient_trade_rate": float((trades >= min_trades).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["canonical_mode", "regime"]).reset_index(drop=True)


def _add_final_decision(summary: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    regime_failures = _failed_regimes(results_df)
    decisions = []
    reasons = []
    for _, row in summary.iterrows():
        label = row["canonical_mode"]
        failures = regime_failures.get(label, [])
        passed = (
            row["avg_strategy_vs_benchmark_pct"] > 0
            and row["beat_benchmark_rate"] >= 0.60
            and row["sufficient_trade_rate"] >= 0.80
            and row["tested_symbol_count"] >= 5
            and not failures
        )
        if passed:
            decisions.append("pass")
            reasons.append("All stress-test conditions passed.")
            continue
        severe = failures or row["avg_strategy_vs_benchmark_pct"] <= 0
        decisions.append("fail" if severe else "warn")
        reason_parts = []
        if row["avg_strategy_vs_benchmark_pct"] <= 0:
            reason_parts.append("average excess return is not positive")
        if row["beat_benchmark_rate"] < 0.60:
            reason_parts.append("beat benchmark rate is below 0.60")
        if row["sufficient_trade_rate"] < 0.80:
            reason_parts.append("sufficient trade rate is below 0.80")
        if row["tested_symbol_count"] < 5:
            reason_parts.append("fewer than five symbols were tested")
        if failures:
            reason_parts.append(f"failed regimes: {', '.join(failures)}")
        reasons.append("; ".join(reason_parts))
    summary["final_decision"] = decisions
    summary["decision_reason"] = reasons
    return summary


def _failed_regimes(results_df: pd.DataFrame) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    if results_df.empty:
        return failures
    for (label, regime), group in results_df.groupby(["canonical_mode", "regime"], dropna=False):
        avg_excess = _numeric(group, "strategy_vs_benchmark_pct").mean()
        beat_rate = float((_numeric(group, "strategy_vs_benchmark_pct") > 0).mean())
        if avg_excess <= 0 or beat_rate < 0.50:
            failures.setdefault(label, []).append(str(regime))
    return failures


def _failed_symbols(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    mask = _numeric(results_df, "strategy_vs_benchmark_pct") <= 0
    return results_df[mask].copy()


def run_candidate_stress_test(
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
    regime_window: int = 60,
    enable_walk_forward: bool = False,
) -> dict[str, Any]:
    symbols = [_format_symbol(symbol) for symbol in symbols]
    recommendations_df = load_pruning_recommendations(recommendations_path)
    rows = []
    warnings = []
    for symbol in symbols:
        factor_csv = _factor_path(factor_dir, symbol)
        if not factor_csv.exists():
            warnings.append(
                {
                    "candidate_label": "all",
                    "symbol": symbol,
                    "warning_type": "missing_factor_csv",
                    "message": f"Factor CSV not found: {factor_csv}",
                }
            )
            continue
        df = add_market_regime(load_factor_dataset(factor_csv), regime_window)
        validate_required_columns(df, target_col)
        candidates = [
            (
                "historical_candidate",
                candidate_pruning_mode,
                candidate_model,
                candidate_buy_threshold,
                candidate_sell_threshold,
            )
        ]
        if enable_walk_forward:
            candidates.append(
                (
                    "walk_forward_candidate",
                    walk_forward_pruning_mode,
                    walk_forward_model,
                    walk_forward_buy_threshold,
                    walk_forward_sell_threshold,
                )
            )
        for label, pruning_mode, model_type, buy_threshold, sell_threshold in candidates:
            candidate_rows, candidate_warnings = _run_candidate_on_regime(
                df=df,
                symbol=symbol,
                recommendations_df=recommendations_df,
                target_col=target_col,
                candidate_label=label,
                pruning_mode=pruning_mode,
                model_type=model_type,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                initial_cash=initial_cash,
                execution_mode=execution_mode,
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax_rate,
                slippage_pct=slippage_pct,
                min_commission=min_commission,
                min_trades=min_trades,
            )
            rows.extend(candidate_rows)
            warnings.extend(candidate_warnings)

    results_df = pd.DataFrame(rows)
    if "symbol" in results_df:
        results_df["symbol"] = results_df["symbol"].map(_format_symbol)
    warnings_df = pd.DataFrame(warnings)
    if "symbol" in warnings_df:
        warnings_df["symbol"] = warnings_df["symbol"].map(_format_symbol)
    stress_summary = _summary(results_df, min_trades)
    regime_summary = _regime_summary(results_df, min_trades)
    report = generate_stress_report(
        stress_summary,
        regime_summary,
        results_df,
        warnings_df,
        min_trades,
    )
    return {
        "candidate_stress_results": results_df,
        "candidate_stress_summary": stress_summary,
        "per_symbol_stress_results": results_df.copy(),
        "regime_summary": regime_summary,
        "stress_warnings": warnings_df,
        "candidate_stress_report": report,
    }


def generate_stress_report(
    stress_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    results_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
    min_trades: int,
) -> str:
    failed_symbols = _failed_symbols(results_df)
    failed_regimes = _failed_regimes(results_df)
    robust_text = (
        "No candidate is robust across regimes unless it passes every stress-test gate."
        if stress_summary.empty or (stress_summary["final_decision"] != "pass").any()
        else "The candidate passed the configured regime stress-test gates."
    )
    regime_text = (
        json.dumps(failed_regimes, ensure_ascii=False)
        if failed_regimes
        else "No regime-specific failures were detected."
    )
    sections = [
        "# Candidate Stress Test / Market Regime Validation",
        "",
        "## Executive Summary",
        (
            "This stress test is educational research diagnostics only. It is not "
            "trading-ready and not financial advice."
        ),
        robust_text,
        "",
        "## Final Decision Rules",
        (
            "PASS requires average excess return > 0, beat benchmark rate >= 0.60, "
            f"sufficient trade rate >= 0.80 using min_trades {min_trades}, at "
            "least five tested symbols, and no severe regime-specific failure."
        ),
        "",
        "## Candidate Stress Summary",
        _markdown_table(stress_summary),
        "",
        "## Regime Diagnostics",
        "Regime failures: " + regime_text,
        _markdown_table(regime_summary),
        "",
        "## Per-Symbol and Per-Regime Results",
        _markdown_table(results_df),
        "",
        "## Symbols That Failed Benchmark Comparison",
        _markdown_table(failed_symbols),
        "",
        "## Warning Summary",
        _markdown_table(warnings_df),
        "",
        "## Feature Expansion Decision",
        (
            "Do not add more features just because one candidate has a strong "
            "historical result. Add features only after the reduced candidate is "
            "stable across bull, bear, and sideways regimes with sufficient trades."
        ),
        "",
    ]
    return "\n".join(sections)


def save_candidate_stress_test(
    factor_dir: str | Path,
    symbols: list[str],
    recommendations_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    result = run_candidate_stress_test(
        factor_dir=factor_dir,
        symbols=symbols,
        recommendations_path=recommendations_path,
        **kwargs,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_stress_results": output_path / "candidate_stress_results.csv",
        "candidate_stress_summary": output_path / "candidate_stress_summary.csv",
        "per_symbol_stress_results": output_path / "per_symbol_stress_results.csv",
        "regime_summary": output_path / "regime_summary.csv",
        "stress_warnings": output_path / "stress_warnings.csv",
        "candidate_stress_report": output_path / "candidate_stress_report.md",
        "run_config": output_path / "run_config.json",
    }
    for key, path in paths.items():
        if key in {"candidate_stress_report", "run_config"}:
            continue
        result.get(key, pd.DataFrame()).to_csv(path, index=False)
    paths["candidate_stress_report"].write_text(
        result["candidate_stress_report"],
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
