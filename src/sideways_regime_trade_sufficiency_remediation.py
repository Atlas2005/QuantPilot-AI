import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .candidate_mode_normalization import add_canonical_mode_columns
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
    from .ml_signal_backtester import predict_probabilities_for_rows
    from .model_trainer import train_baseline_model
    from .reduced_feature_threshold_experiment import (
        parse_thresholds,
        run_threshold_grid_from_probabilities,
        summarize_threshold_results,
    )
except ImportError:
    from candidate_mode_normalization import add_canonical_mode_columns
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
    from ml_signal_backtester import predict_probabilities_for_rows
    from model_trainer import train_baseline_model
    from reduced_feature_threshold_experiment import (
        parse_thresholds,
        run_threshold_grid_from_probabilities,
        summarize_threshold_results,
    )


DEFAULT_SYMBOLS = ["000001", "600519", "000858", "600036", "601318"]
DEFAULT_BUY_THRESHOLDS = [0.45, 0.50, 0.55, 0.60]
DEFAULT_SELL_THRESHOLDS = [0.40, 0.45, 0.50, 0.55]
CANONICAL_TO_PRUNING_MODE = {
    "canonical_reduced_40": "keep_core_and_observe",
    "full": "full",
    "keep_core_only": "keep_core_only",
}
NON_FACTOR_DIAGNOSTIC_COLUMNS = {
    "regime",
    "regime_return",
    "candidate_label",
    "warning",
    "warning_type",
    "message",
}


def parse_symbols(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_SYMBOLS.copy()
    return [_format_symbol(item.strip()) for item in text.split(",") if item.strip()]


def parse_threshold_list(text: str | None, defaults: list[float]) -> list[float]:
    return parse_thresholds(text, defaults)


def _format_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _factor_path(factor_dir: str | Path, symbol: str) -> Path:
    return Path(factor_dir) / f"factors_{_format_symbol(symbol)}.csv"


def _drop_non_factor_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[c for c in df.columns if c in NON_FACTOR_DIAGNOSTIC_COLUMNS],
        errors="ignore",
    )


def _markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
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


def _read_optional(path: Path, warnings: list[dict[str, Any]]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(
            {
                "source": str(path),
                "warning_type": "missing_optional_input",
                "message": f"Optional input file not found: {path}",
            }
        )
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"symbol": str})
    except pd.errors.EmptyDataError:
        warnings.append(
            {
                "source": str(path),
                "warning_type": "empty_optional_input",
                "message": f"Optional input file is empty: {path}",
            }
        )
        return pd.DataFrame()


def _passes_sideways_gate(row: pd.Series, min_trades: int) -> bool:
    return bool(
        row.get("avg_strategy_vs_benchmark_pct") > 0
        and row.get("beat_benchmark_rate") >= 0.60
        and row.get("sufficient_trade_rate") >= 0.80
        and row.get("tested_symbol_count") >= 5
    )


def _add_result_flags(rows: list[dict[str, Any]], min_trades: int) -> list[dict[str, Any]]:
    flagged_rows = []
    for row in rows:
        trade_count = pd.to_numeric(pd.Series([row.get("trade_count")]), errors="coerce").iloc[0]
        excess = pd.to_numeric(
            pd.Series([row.get("strategy_vs_benchmark_pct")]),
            errors="coerce",
        ).iloc[0]
        total_return = pd.to_numeric(
            pd.Series([row.get("total_return_pct")]),
            errors="coerce",
        ).iloc[0]
        row["sufficient_trade"] = bool(pd.notna(trade_count) and trade_count >= min_trades)
        row["beat_benchmark"] = bool(pd.notna(excess) and excess > 0)
        row["low_trade_count"] = not row["sufficient_trade"]
        row["negative_total_return"] = bool(pd.notna(total_return) and total_return < 0)
        row["underperformed_benchmark"] = bool(pd.notna(excess) and excess < 0)
        flagged_rows.append(row)
    return flagged_rows


def _run_symbol_sideways_grid(
    factor_csv: Path,
    symbol: str,
    recommendations_df: pd.DataFrame,
    target_col: str,
    canonical_mode: str,
    model_type: str,
    buy_thresholds: list[float],
    sell_thresholds: list[float],
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_pct: float,
    min_commission: float,
    min_trades: int,
    regime_window: int = 60,
    initial_cash: float = 10000.0,
    execution_mode: str = "same_close",
    purge_rows: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    pruning_mode = CANONICAL_TO_PRUNING_MODE.get(canonical_mode, canonical_mode)
    try:
        raw_df = load_factor_dataset(factor_csv)
        validate_required_columns(raw_df, target_col)
        regime_df = add_market_regime(raw_df, regime_window=regime_window)
    except Exception as exc:
        return rows, [
            {
                "symbol": symbol,
                "canonical_mode": canonical_mode,
                "model_type": model_type,
                "warning_type": "load_or_regime_error",
                "message": str(exc),
            }
        ]

    sideways_df = regime_df[regime_df["regime"] == "sideways"].copy()
    if sideways_df.empty:
        return rows, [
            {
                "symbol": symbol,
                "canonical_mode": canonical_mode,
                "model_type": model_type,
                "warning_type": "empty_sideways_regime",
                "message": "No sideways-regime rows were available for this symbol.",
            }
        ]

    try:
        feature_source_df = _drop_non_factor_diagnostics(regime_df)
        feature_columns = identify_safe_feature_columns(feature_source_df, target_col)
        feature_sets = build_feature_sets(feature_columns, recommendations_df, [pruning_mode])
        features = feature_sets.get(pruning_mode, [])
        if not features:
            raise ValueError(f"No features selected for pruning mode {pruning_mode}.")
        model_input_df = _drop_non_factor_diagnostics(sideways_df)
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
            raise ValueError("Train or test split is empty for sideways regime.")
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
                    "canonical_mode": canonical_mode,
                    "model_type": model_type,
                    "warning_type": "single_class_training",
                    "message": "Training split had one target class.",
                }
            )
        test_signal_df = test_df.copy().reset_index(drop=True)
        probabilities = predict_probabilities_for_rows(model, test_signal_df, features)
        grid_rows = run_threshold_grid_from_probabilities(
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
                "regime": "sideways",
                "sideways_rows": len(sideways_df),
                "test_rows": len(test_signal_df),
            },
        )
    except Exception as exc:
        return rows, warnings + [
            {
                "symbol": symbol,
                "canonical_mode": canonical_mode,
                "model_type": model_type,
                "warning_type": "sideways_grid_error",
                "message": str(exc),
            }
        ]

    for row in _add_result_flags(grid_rows, min_trades):
        row["symbol"] = _format_symbol(row.get("symbol"))
        row["legacy_pruning_mode"] = row.get("pruning_mode")
        row["canonical_mode"] = add_canonical_mode_columns(pd.DataFrame([row]))[
            "canonical_mode"
        ].iloc[0]
        rows.append(row)
        for warning_type in [
            "low_trade_count",
            "negative_total_return",
            "underperformed_benchmark",
        ]:
            if row.get(warning_type):
                warnings.append(
                    {
                        "symbol": row["symbol"],
                        "canonical_mode": row["canonical_mode"],
                        "legacy_pruning_mode": row["legacy_pruning_mode"],
                        "model_type": model_type,
                        "buy_threshold": row.get("buy_threshold"),
                        "sell_threshold": row.get("sell_threshold"),
                        "regime": "sideways",
                        "warning_type": warning_type,
                        "message": row.get("warning") or warning_type,
                    }
                )
    return rows, warnings


def build_sideways_trade_summary(results_df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    group_columns = ["canonical_mode", "model_type", "buy_threshold", "sell_threshold", "regime"]
    summary = summarize_threshold_results(results_df, group_columns, min_trades)
    tested = (
        results_df.groupby(group_columns, dropna=False)["symbol"]
        .nunique()
        .reset_index(name="tested_symbol_count")
    )
    summary = summary.merge(tested, on=group_columns, how="left")
    summary["sideways_gate_passed"] = summary.apply(
        lambda row: _passes_sideways_gate(row, min_trades),
        axis=1,
    )
    summary["final_decision"] = summary["sideways_gate_passed"].map(
        {True: "sideways_remediation_passed", False: "sideways_remediation_failed"}
    )
    return summary.sort_values(
        [
            "sideways_gate_passed",
            "sufficient_trade_rate",
            "beat_benchmark_rate",
            "avg_strategy_vs_benchmark_pct",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_per_symbol_sideways_results(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    ranked = results_df.sort_values(
        ["symbol", "sufficient_trade", "beat_benchmark", "strategy_vs_benchmark_pct", "trade_count"],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    return ranked.groupby("symbol", dropna=False).head(1).reset_index(drop=True)


def select_best_sideways_thresholds(summary_df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    result = summary_df.head(1).copy()
    passed = bool(result["sideways_gate_passed"].iloc[0])
    result["selected"] = passed
    result["selection_decision"] = (
        "selected_for_research_review" if passed else "no_passing_sideways_threshold"
    )
    result["selection_reason"] = (
        "Sideways threshold candidate passed strict research remediation gates."
        if passed
        else (
            "No sideways threshold candidate passed avg excess > 0, beat benchmark "
            "rate >= 0.60, sufficient trade rate >= 0.80, and at least five tested symbols."
        )
    )
    return result


def generate_sideways_remediation_report(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_df: pd.DataFrame,
    per_symbol_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
    canonical_mode: str,
    model_type: str,
) -> str:
    baseline = pd.DataFrame()
    if not summary_df.empty:
        baseline = summary_df[
            (summary_df["buy_threshold"] == 0.50)
            & (summary_df["sell_threshold"] == 0.40)
        ].copy()
    passed = bool(not best_df.empty and best_df.get("selected", pd.Series([False])).iloc[0])
    status = (
        "Sideways remediation passed the configured research diagnostic gates."
        if passed
        else "Sideways remediation failed; canonical_reduced_40 remains research-only."
    )
    improvement_text = "Baseline 0.50 / 0.40 was available for comparison." if not baseline.empty else (
        "Baseline 0.50 / 0.40 was not available in this threshold grid."
    )
    sections = [
        "# Sideways Regime Trade Sufficiency Remediation Experiment",
        "",
        "## Executive Summary",
        "This output is educational/research diagnostics only, not financial advice.",
        "This output is not trading-ready and does not make any trading-ready claim.",
        f"{canonical_mode} remains research-only unless strict validation gates pass.",
        f"Target: {canonical_mode} + {model_type}, sideways regime only.",
        status,
        "The diagnostic focuses on trade sufficiency, benchmark beat stability, and sideways regime gate repair.",
        "",
        "## Best Sideways Threshold Candidate",
        _markdown_table(best_df),
        "",
        "## Baseline 0.50 / 0.40 Comparison",
        improvement_text,
        _markdown_table(baseline),
        "",
        "## Sideways Trade Summary",
        "A candidate passes only when average excess is positive, beat_benchmark_rate >= 0.60, "
        "sufficient_trade_rate >= 0.80, and tested_symbol_count >= 5.",
        _markdown_table(summary_df),
        "",
        "## Trade Sufficiency and Benchmark Beat Diagnostics",
        "Trade sufficiency improved only if sufficient_trade_rate exceeds the previous sideways blocker level of 0.40.",
        "Benchmark beat stability improved only if beat_benchmark_rate exceeds the previous sideways blocker level of 0.40.",
        "",
        "## Per-Symbol Best Sideways Results",
        _markdown_table(per_symbol_df),
        "",
        "## Remaining Blockers",
        "Remaining blockers are any failed strict gate, low trade count warnings, benchmark underperformance warnings, "
        "or fewer than five tested symbols.",
        "",
        "## Warnings",
        _markdown_table(warnings_df),
        "",
        "## Result Rows",
        _markdown_table(results_df),
        "",
    ]
    return "\n".join(sections)


def run_sideways_regime_trade_sufficiency_remediation(
    factor_dir: str | Path,
    symbols: list[str],
    recommendations_path: str | Path,
    failure_analysis_dir: str | Path = "outputs/validation_gate_failure_analysis_real_v1",
    targeted_design_dir: str | Path = "outputs/targeted_remediation_design_real_v1",
    target_col: str = "label_up_5d",
    canonical_mode: str = "canonical_reduced_40",
    model_type: str = "logistic_regression",
    buy_thresholds: list[float] | None = None,
    sell_thresholds: list[float] | None = None,
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    min_commission: float = 5.0,
    min_trades: int = 3,
) -> dict[str, Any]:
    symbols = [_format_symbol(symbol) for symbol in symbols]
    buy_thresholds = buy_thresholds or DEFAULT_BUY_THRESHOLDS.copy()
    sell_thresholds = sell_thresholds or DEFAULT_SELL_THRESHOLDS.copy()
    recommendations_df = load_pruning_recommendations(recommendations_path)
    warnings: list[dict[str, Any]] = []
    _read_optional(Path(failure_analysis_dir) / "failure_by_regime.csv", warnings)
    _read_optional(Path(targeted_design_dir) / "targeted_remediation_experiments.csv", warnings)
    result_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        factor_csv = _factor_path(factor_dir, symbol)
        if not factor_csv.exists():
            warnings.append(
                {
                    "symbol": symbol,
                    "canonical_mode": canonical_mode,
                    "model_type": model_type,
                    "warning_type": "missing_factor_file",
                    "message": f"Missing factor file: {factor_csv}",
                }
            )
            continue
        rows, symbol_warnings = _run_symbol_sideways_grid(
            factor_csv=factor_csv,
            symbol=symbol,
            recommendations_df=recommendations_df,
            target_col=target_col,
            canonical_mode=canonical_mode,
            model_type=model_type,
            buy_thresholds=buy_thresholds,
            sell_thresholds=sell_thresholds,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
            min_trades=min_trades,
        )
        result_rows.extend(rows)
        warnings.extend(symbol_warnings)

    results_df = pd.DataFrame(result_rows)
    if not results_df.empty:
        keep_columns = [
            "symbol",
            "canonical_mode",
            "model_type",
            "buy_threshold",
            "sell_threshold",
            "regime",
            "total_return_pct",
            "benchmark_return_pct",
            "strategy_vs_benchmark_pct",
            "max_drawdown_pct",
            "trade_count",
            "win_rate_pct",
            "final_value",
            "sufficient_trade",
            "beat_benchmark",
            "low_trade_count",
            "negative_total_return",
            "underperformed_benchmark",
            "warning",
            "legacy_pruning_mode",
            "feature_count",
            "sideways_rows",
            "test_rows",
        ]
        results_df = results_df[[c for c in keep_columns if c in results_df.columns]]
    summary_df = build_sideways_trade_summary(results_df, min_trades)
    per_symbol_df = build_per_symbol_sideways_results(results_df)
    best_df = select_best_sideways_thresholds(summary_df, min_trades)
    warnings_df = pd.DataFrame(warnings)
    if "symbol" in warnings_df:
        warnings_df["symbol"] = warnings_df["symbol"].map(_format_symbol)
    report = generate_sideways_remediation_report(
        results_df,
        summary_df,
        best_df,
        per_symbol_df,
        warnings_df,
        canonical_mode,
        model_type,
    )
    return {
        "sideways_trade_results": results_df,
        "sideways_trade_summary": summary_df,
        "per_symbol_sideways_results": per_symbol_df,
        "best_sideways_thresholds": best_df,
        "sideways_remediation_report": report,
        "warnings": warnings_df,
    }


def save_sideways_regime_trade_sufficiency_remediation(
    factor_dir: str | Path,
    symbols: list[str],
    recommendations_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    result = run_sideways_regime_trade_sufficiency_remediation(
        factor_dir=factor_dir,
        symbols=symbols,
        recommendations_path=recommendations_path,
        **kwargs,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "sideways_trade_results": output_path / "sideways_trade_results.csv",
        "sideways_trade_summary": output_path / "sideways_trade_summary.csv",
        "per_symbol_sideways_results": output_path / "per_symbol_sideways_results.csv",
        "best_sideways_thresholds": output_path / "best_sideways_thresholds.csv",
        "report": output_path / "sideways_remediation_report.md",
        "warnings": output_path / "warnings.csv",
        "run_config": output_path / "run_config.json",
    }
    for key in [
        "sideways_trade_results",
        "sideways_trade_summary",
        "per_symbol_sideways_results",
        "best_sideways_thresholds",
        "warnings",
    ]:
        result[key].to_csv(paths[key], index=False)
    paths["report"].write_text(result["sideways_remediation_report"], encoding="utf-8")
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
