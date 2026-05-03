import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIRS = [
    "outputs/reduced_feature_backtest_real_000001",
    "outputs/reduced_feature_backtest_real_600519",
    "outputs/reduced_feature_backtest_real_000858",
    "outputs/reduced_feature_backtest_real_600036",
    "outputs/reduced_feature_backtest_real_601318",
]

SUMMARY_COLUMNS = [
    "symbol_count",
    "model_count",
    "avg_feature_count",
    "avg_total_return_pct",
    "avg_benchmark_return_pct",
    "avg_strategy_vs_benchmark_pct",
    "avg_max_drawdown_pct",
    "avg_trade_count",
    "avg_win_rate_pct",
    "avg_final_value",
    "win_rate_vs_benchmark",
    "positive_return_rate",
    "low_trade_count_rate",
    "stability_score",
]


def parse_input_dirs(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_INPUT_DIRS.copy()
    return [item.strip() for item in text.split(",") if item.strip()]


def load_json_if_available(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"Could not parse {path.name}"}


def infer_symbol(input_dir: Path, run_config: dict[str, Any]) -> str:
    factor_csv = str(run_config.get("factor_csv", ""))
    if factor_csv:
        name = Path(factor_csv).stem
        if name.startswith("factors_"):
            return name.replace("factors_", "", 1)

    name = input_dir.name
    for prefix in ["reduced_feature_backtest_real_", "reduced_feature_backtest_"]:
        if name.startswith(prefix):
            return name.replace(prefix, "", 1)
    return name


def _read_csv_if_available(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_reduced_feature_backtest_directory(
    input_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    path = Path(input_dir)
    run_config = load_json_if_available(path / "run_config.json")
    symbol = infer_symbol(path, run_config)
    warnings: list[dict[str, Any]] = []

    results_path = path / "reduced_feature_backtest_results.csv"
    summary_path = path / "reduced_feature_backtest_summary.csv"

    if not results_path.exists():
        warnings.append(
            {
                "warning_type": "missing_input_file",
                "symbol": symbol,
                "input_dir": str(path),
                "message": "Missing reduced_feature_backtest_results.csv.",
            }
        )
        results_df = pd.DataFrame()
    else:
        results_df = _read_csv_if_available(results_path)
        if results_df.empty:
            warnings.append(
                {
                    "warning_type": "empty_results",
                    "symbol": symbol,
                    "input_dir": str(path),
                    "message": "reduced_feature_backtest_results.csv is empty.",
                }
            )

    if not summary_path.exists():
        warnings.append(
            {
                "warning_type": "missing_input_file",
                "symbol": symbol,
                "input_dir": str(path),
                "message": "Missing reduced_feature_backtest_summary.csv.",
            }
        )
        summary_df = pd.DataFrame()
    else:
        summary_df = _read_csv_if_available(summary_path)
        if summary_df.empty:
            warnings.append(
                {
                    "warning_type": "empty_summary",
                    "symbol": symbol,
                    "input_dir": str(path),
                    "message": "reduced_feature_backtest_summary.csv is empty.",
                }
            )

    for df in [results_df, summary_df]:
        if not df.empty:
            df["symbol"] = symbol
            df["input_dir"] = str(path)

    existing_warnings = _read_csv_if_available(path / "warnings.csv")
    if not existing_warnings.empty:
        for _, row in existing_warnings.iterrows():
            warnings.append(
                {
                    "warning_type": "source_warning",
                    "symbol": row.get("symbol", symbol),
                    "input_dir": str(path),
                    "message": row.get("warning", row.to_dict()),
                }
            )

    return results_df, summary_df, warnings


def load_multiple_reduced_feature_outputs(
    input_dirs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_frames = []
    summary_frames = []
    warnings = []

    for input_dir in input_dirs:
        results_df, summary_df, directory_warnings = (
            load_reduced_feature_backtest_directory(input_dir)
        )
        if not results_df.empty:
            result_frames.append(results_df)
        if not summary_df.empty:
            summary_frames.append(summary_df)
        warnings.extend(directory_warnings)

    combined_results = (
        pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    )
    combined_summary = (
        pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    )
    warnings_df = pd.DataFrame(
        warnings,
        columns=["warning_type", "symbol", "input_dir", "message"],
    )
    return combined_results, combined_summary, warnings_df


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _rate(series: pd.Series, condition) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(condition(values).mean())


def _build_summary_for_group(group: pd.DataFrame, min_trades: int) -> dict[str, Any]:
    excess = _numeric(group, "strategy_vs_benchmark_pct")
    total_return = _numeric(group, "total_return_pct")
    trade_count = _numeric(group, "trade_count")

    avg_excess = float(excess.mean()) if not excess.dropna().empty else None
    avg_return = (
        float(total_return.mean()) if not total_return.dropna().empty else None
    )
    win_rate_vs_benchmark = _rate(excess, lambda values: values > 0)
    positive_return_rate = _rate(total_return, lambda values: values > 0)
    low_trade_count_rate = _rate(trade_count, lambda values: values <= min_trades)

    stability_score = None
    if avg_excess is not None and avg_return is not None:
        stability_score = (
            avg_excess / 100.0
            + avg_return / 200.0
            + (win_rate_vs_benchmark or 0.0) * 0.40
            + (positive_return_rate or 0.0) * 0.20
            - (low_trade_count_rate or 0.0) * 0.25
        )

    return {
        "symbol_count": int(group["symbol"].nunique()) if "symbol" in group else 0,
        "model_count": int(group["model_type"].nunique()) if "model_type" in group else 0,
        "avg_feature_count": _numeric(group, "feature_count").mean(),
        "avg_total_return_pct": avg_return,
        "avg_benchmark_return_pct": _numeric(group, "benchmark_return_pct").mean(),
        "avg_strategy_vs_benchmark_pct": avg_excess,
        "avg_max_drawdown_pct": _numeric(group, "max_drawdown_pct").mean(),
        "avg_trade_count": trade_count.mean(),
        "avg_win_rate_pct": _numeric(group, "win_rate_pct").mean(),
        "avg_final_value": _numeric(group, "final_value").mean(),
        "win_rate_vs_benchmark": win_rate_vs_benchmark,
        "positive_return_rate": positive_return_rate,
        "low_trade_count_rate": low_trade_count_rate,
        "stability_score": stability_score,
    }


def summarize_by(
    combined_results: pd.DataFrame,
    group_columns: list[str],
    min_trades: int,
) -> pd.DataFrame:
    if combined_results.empty:
        return pd.DataFrame(columns=[*group_columns, *SUMMARY_COLUMNS])

    rows = []
    for keys, group in combined_results.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(_build_summary_for_group(group, min_trades))
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("stability_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def build_per_symbol_best_modes(combined_results: pd.DataFrame) -> pd.DataFrame:
    if combined_results.empty:
        return pd.DataFrame()

    rows = []
    for symbol, group in combined_results.groupby("symbol", dropna=False):
        excess_group = group.dropna(subset=["strategy_vs_benchmark_pct"])
        return_group = group.dropna(subset=["total_return_pct"])
        drawdown_group = group.dropna(subset=["max_drawdown_pct"])

        best_excess = (
            excess_group.sort_values("strategy_vs_benchmark_pct", ascending=False).iloc[0]
            if not excess_group.empty
            else None
        )
        best_return = (
            return_group.sort_values("total_return_pct", ascending=False).iloc[0]
            if not return_group.empty
            else None
        )
        best_drawdown = (
            drawdown_group.sort_values("max_drawdown_pct", ascending=False).iloc[0]
            if not drawdown_group.empty
            else None
        )

        rows.append(
            {
                "symbol": symbol,
                "best_mode_by_excess_return": None
                if best_excess is None
                else best_excess["pruning_mode"],
                "best_model_by_excess_return": None
                if best_excess is None
                else best_excess["model_type"],
                "best_excess_return_pct": None
                if best_excess is None
                else best_excess["strategy_vs_benchmark_pct"],
                "best_total_return_pct": None
                if best_excess is None
                else best_excess["total_return_pct"],
                "benchmark_return_pct": None
                if best_excess is None
                else best_excess["benchmark_return_pct"],
                "best_max_drawdown_pct": None
                if best_excess is None
                else best_excess["max_drawdown_pct"],
                "best_trade_count": None
                if best_excess is None
                else best_excess["trade_count"],
                "best_win_rate_pct": None
                if best_excess is None
                else best_excess["win_rate_pct"],
                "best_mode_by_total_return": None
                if best_return is None
                else best_return["pruning_mode"],
                "best_model_by_total_return": None
                if best_return is None
                else best_return["model_type"],
                "best_mode_by_drawdown": None
                if best_drawdown is None
                else best_drawdown["pruning_mode"],
                "best_model_by_drawdown": None
                if best_drawdown is None
                else best_drawdown["model_type"],
            }
        )

    return pd.DataFrame(rows)


def build_underperformance_cases(
    combined_results: pd.DataFrame,
    min_trades: int,
) -> pd.DataFrame:
    if combined_results.empty:
        return pd.DataFrame()

    excess = _numeric(combined_results, "strategy_vs_benchmark_pct")
    total_return = _numeric(combined_results, "total_return_pct")
    trade_count = _numeric(combined_results, "trade_count")
    mask = (excess < 0) | (total_return < 0) | (trade_count <= min_trades)
    cases = combined_results.loc[mask].copy()
    if cases.empty:
        return cases

    reasons = []
    for _, row in cases.iterrows():
        row_reasons = []
        if pd.notna(row.get("strategy_vs_benchmark_pct")) and row.get(
            "strategy_vs_benchmark_pct"
        ) < 0:
            row_reasons.append("underperformed_benchmark")
        if pd.notna(row.get("total_return_pct")) and row.get("total_return_pct") < 0:
            row_reasons.append("negative_total_return")
        if pd.notna(row.get("trade_count")) and row.get("trade_count") <= min_trades:
            row_reasons.append("low_trade_count")
        reasons.append(",".join(row_reasons))
    cases["underperformance_reason"] = reasons
    return cases


def build_research_warnings(
    combined_results: pd.DataFrame,
    existing_warnings: pd.DataFrame,
    min_trades: int,
) -> pd.DataFrame:
    rows = []
    if not existing_warnings.empty:
        rows.extend(existing_warnings.to_dict("records"))

    if not combined_results.empty:
        for _, row in combined_results.iterrows():
            base = {
                "symbol": row.get("symbol"),
                "model_type": row.get("model_type"),
                "pruning_mode": row.get("pruning_mode"),
                "input_dir": row.get("input_dir"),
            }
            if pd.notna(row.get("trade_count")) and row.get("trade_count") <= min_trades:
                rows.append(
                    {
                        **base,
                        "warning_type": "low_trade_count",
                        "message": (
                            f"Trade count {row.get('trade_count')} is at or below "
                            f"the min_trades threshold {min_trades}."
                        ),
                    }
                )
            if pd.notna(row.get("strategy_vs_benchmark_pct")) and row.get(
                "strategy_vs_benchmark_pct"
            ) < 0:
                rows.append(
                    {
                        **base,
                        "warning_type": "underperformed_benchmark",
                        "message": "Strategy return was below buy-and-hold benchmark.",
                    }
                )
            if pd.notna(row.get("total_return_pct")) and row.get("total_return_pct") < 0:
                rows.append(
                    {
                        **base,
                        "warning_type": "negative_total_return",
                        "message": "Strategy total return was negative.",
                    }
                )

    columns = [
        "warning_type",
        "symbol",
        "model_type",
        "pruning_mode",
        "input_dir",
        "message",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows available._"

    table_df = df.head(max_rows).copy()
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


def _best_label(df: pd.DataFrame, metric: str, id_columns: list[str]) -> str:
    if df.empty or metric not in df.columns:
        return "n/a"
    metric_df = df.dropna(subset=[metric])
    if metric_df.empty:
        return "n/a"
    row = metric_df.sort_values(metric, ascending=False).iloc[0]
    label = " / ".join(str(row[column]) for column in id_columns)
    return f"{label} ({metric}: {row[metric]:.4f})"


def generate_markdown_report(
    mode_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    mode_model_summary: pd.DataFrame,
    per_symbol_best: pd.DataFrame,
    underperformance_cases: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> str:
    best_excess = _best_label(
        mode_summary,
        "avg_strategy_vs_benchmark_pct",
        ["pruning_mode"],
    )
    best_stability = _best_label(mode_summary, "stability_score", ["pruning_mode"])
    best_model = _best_label(
        model_summary,
        "avg_strategy_vs_benchmark_pct",
        ["model_type"],
    )

    sections = [
        "# Reduced Feature Backtest Summary Report",
        "",
        "## Overall Conclusion",
        (
            "This report aggregates reduced feature set ML signal backtests across "
            "symbols. It compares pruning modes and model types using trading "
            "backtest outputs, not only classification metrics. These results are "
            "not trading recommendations."
        ),
        "",
        "## Best Pruning Mode by Excess Return",
        best_excess,
        "",
        "## Best Pruning Mode by Stability",
        best_stability,
        "",
        "## Best Model Type",
        best_model,
        "",
        "## Pruning Mode Ranking",
        _markdown_table(mode_summary),
        "",
        "## Model Type Ranking",
        _markdown_table(model_summary),
        "",
        "## Pruning Mode + Model Ranking",
        _markdown_table(mode_model_summary),
        "",
        "## Per-Symbol Best Modes",
        _markdown_table(per_symbol_best),
        "",
        "## Underperformance Cases",
        _markdown_table(underperformance_cases),
        "",
        "## Research Warnings",
        _markdown_table(warnings_df),
        "",
        "## Recommended Next Step",
        (
            "A pruning mode should not become the default unless it is stable across "
            "symbols, model types, drawdown, and trade count. Strong returns with "
            "very low trade count may be unreliable. The next step should be "
            "walk-forward validation and threshold sensitivity testing, not adding "
            "more features immediately."
        ),
        "",
        (
            "High backtest return, high ROC AUC, or a strong single-symbol result "
            "does not guarantee future trading profit after costs, slippage, and "
            "market regime changes."
        ),
        "",
    ]
    return "\n".join(sections)


def build_reduced_feature_backtest_report(
    input_dirs: list[str],
    min_trades: int = 3,
) -> dict[str, Any]:
    combined_results, _, load_warnings = load_multiple_reduced_feature_outputs(input_dirs)
    mode_summary = summarize_by(combined_results, ["pruning_mode"], min_trades)
    model_summary = summarize_by(combined_results, ["model_type"], min_trades)
    mode_model_summary = summarize_by(
        combined_results,
        ["pruning_mode", "model_type"],
        min_trades,
    )
    per_symbol_best = build_per_symbol_best_modes(combined_results)
    underperformance_cases = build_underperformance_cases(combined_results, min_trades)
    warnings_df = build_research_warnings(
        combined_results,
        load_warnings,
        min_trades,
    )
    report = generate_markdown_report(
        mode_summary=mode_summary,
        model_summary=model_summary,
        mode_model_summary=mode_model_summary,
        per_symbol_best=per_symbol_best,
        underperformance_cases=underperformance_cases,
        warnings_df=warnings_df,
    )
    return {
        "combined_reduced_feature_backtest_results": combined_results,
        "reduced_feature_backtest_mode_summary": mode_summary,
        "reduced_feature_backtest_model_summary": model_summary,
        "reduced_feature_backtest_mode_model_summary": mode_model_summary,
        "per_symbol_best_backtest_modes": per_symbol_best,
        "underperformance_cases": underperformance_cases,
        "warnings": warnings_df,
        "markdown_report": report,
    }


def save_reduced_feature_backtest_report(
    input_dirs: list[str],
    output_dir: str | Path,
    min_trades: int = 3,
) -> dict[str, Any]:
    result = build_reduced_feature_backtest_report(
        input_dirs=input_dirs,
        min_trades=min_trades,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined_reduced_feature_backtest_results": output_path
        / "combined_reduced_feature_backtest_results.csv",
        "reduced_feature_backtest_mode_summary": output_path
        / "reduced_feature_backtest_mode_summary.csv",
        "reduced_feature_backtest_model_summary": output_path
        / "reduced_feature_backtest_model_summary.csv",
        "reduced_feature_backtest_mode_model_summary": output_path
        / "reduced_feature_backtest_mode_model_summary.csv",
        "per_symbol_best_backtest_modes": output_path
        / "per_symbol_best_backtest_modes.csv",
        "underperformance_cases": output_path / "underperformance_cases.csv",
        "warnings": output_path / "warnings.csv",
        "run_config": output_path / "run_config.json",
        "report": output_path / "reduced_feature_backtest_report.md",
    }
    for key, path in paths.items():
        if key in {"run_config", "report"}:
            continue
        result[key].to_csv(path, index=False)

    run_config = {
        "input_dirs": input_dirs,
        "output_dir": str(output_path),
        "min_trades": min_trades,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["report"].write_text(result["markdown_report"], encoding="utf-8")
    result["run_config"] = run_config
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result
