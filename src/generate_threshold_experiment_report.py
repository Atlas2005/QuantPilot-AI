import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_MIN_TRADES = 3


def parse_input_dirs(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def infer_symbol(input_dir: Path, run_config: dict[str, Any]) -> str:
    factor_csv = str(run_config.get("factor_csv", ""))
    if factor_csv:
        stem = Path(factor_csv).stem
        if stem.startswith("factors_"):
            return stem.replace("factors_", "", 1)
    return input_dir.name.replace("reduced_feature_threshold_", "")


def _format_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _min_trades_from_config(run_config: dict[str, Any]) -> int:
    try:
        return int(run_config.get("min_trades", DEFAULT_MIN_TRADES))
    except (TypeError, ValueError):
        return DEFAULT_MIN_TRADES


def load_threshold_experiment_outputs(
    input_dirs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_frames = []
    walk_forward_frames = []
    warnings = []
    for input_dir in input_dirs:
        path = Path(input_dir)
        run_config = _load_json(path / "run_config.json")
        symbol = _format_symbol(infer_symbol(path, run_config))
        min_trades = _min_trades_from_config(run_config)
        threshold_df = _read_csv(path / "threshold_backtest_results.csv", {"symbol": str})
        if threshold_df.empty:
            warnings.append(
                {
                    "warning_type": "missing_or_empty_threshold_results",
                    "symbol": symbol,
                    "input_dir": str(path),
                    "message": "threshold_backtest_results.csv missing or empty.",
                }
            )
        else:
            if "symbol" not in threshold_df:
                threshold_df["symbol"] = symbol
            threshold_df["symbol"] = threshold_df["symbol"].fillna(symbol).map(_format_symbol)
            threshold_df["input_dir"] = str(path)
            threshold_df["report_min_trades"] = min_trades
            threshold_frames.append(threshold_df)

        walk_forward_df = _read_csv(path / "walk_forward_results.csv", {"symbol": str})
        if not walk_forward_df.empty:
            if "symbol" not in walk_forward_df:
                walk_forward_df["symbol"] = symbol
            walk_forward_df["symbol"] = walk_forward_df["symbol"].fillna(symbol).map(_format_symbol)
            walk_forward_df["input_dir"] = str(path)
            walk_forward_df["report_min_trades"] = min_trades
            walk_forward_frames.append(walk_forward_df)

        source_warnings = _read_csv(path / "warnings.csv", {"symbol": str})
        if not source_warnings.empty:
            if "symbol" in source_warnings:
                source_warnings["symbol"] = source_warnings["symbol"].map(_format_symbol)
            source_warnings["input_dir"] = str(path)
            source_warnings["report_min_trades"] = min_trades
            warnings.extend(source_warnings.to_dict("records"))

    combined = (
        pd.concat(threshold_frames, ignore_index=True)
        if threshold_frames
        else pd.DataFrame()
    )
    walk_forward = (
        pd.concat(walk_forward_frames, ignore_index=True)
        if walk_forward_frames
        else pd.DataFrame()
    )
    warnings_df = pd.DataFrame(warnings)
    return combined, walk_forward, warnings_df


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _min_trades_series(df: pd.DataFrame) -> pd.Series:
    if "report_min_trades" not in df:
        return pd.Series(DEFAULT_MIN_TRADES, index=df.index, dtype="int64")
    values = pd.to_numeric(df["report_min_trades"], errors="coerce").fillna(
        DEFAULT_MIN_TRADES
    )
    return values.astype(int)


def _has_low_trade_warning(df: pd.DataFrame) -> pd.Series:
    text = _low_trade_text(df)
    has_low_trade_text = text.str.contains(
        "low_trade_count|low trades",
        case=False,
        regex=True,
    )
    parsed_count = pd.to_numeric(
        text.str.extract(r"(?i)low(?:_| )trade(?:_| )?count\D*(\d+)")[0],
        errors="coerce",
    )
    has_parsed_count = parsed_count.notna()
    parsed_low_trade = parsed_count < _min_trades_series(df)
    if "trade_count" in df:
        trade_low_trade = _numeric(df, "trade_count") < _min_trades_series(df)
        return has_low_trade_text & (
            (has_parsed_count & parsed_low_trade)
            | (~has_parsed_count & trade_low_trade)
        )
    return has_low_trade_text & (
        (has_parsed_count & parsed_low_trade) | ~has_parsed_count
    )


def _low_trade_text(df: pd.DataFrame) -> pd.Series:
    text = pd.Series("", index=df.index, dtype="object")
    for column in ["warning", "message", "warning_type"]:
        if column in df:
            text = text.str.cat(df[column].fillna("").astype(str), sep=" ")
    return text


def _drop_stale_low_trade_warning_text(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = df.copy()
    stale_low_trade = _low_trade_text(cleaned).str.contains(
        "low_trade_count|low trades",
        case=False,
        regex=True,
    ) & ~_has_low_trade_warning(cleaned)
    for column in ["warning", "message"]:
        if column in cleaned:
            cleaned.loc[stale_low_trade, column] = (
                cleaned.loc[stale_low_trade, column]
                .fillna("")
                .astype(str)
                .str.replace(
                    r"(?i)\s*\|?\s*low(?:_| )trade(?:_| )?count\D*\d+\s*\|?\s*",
                    " | ",
                    regex=True,
                )
                .str.replace(r"\s*\|\s*\|\s*", " | ", regex=True)
                .str.strip(" |")
                .replace("", pd.NA)
            )
    if "warning_type" in cleaned:
        cleaned.loc[
            stale_low_trade
            & cleaned["warning_type"].fillna("").astype(str).str.contains(
                "low_trade",
                case=False,
                regex=True,
            ),
            "warning_type",
        ] = pd.NA
    return cleaned


def _sufficient_trade_mask(df: pd.DataFrame) -> pd.Series:
    trade_count = _numeric(df, "trade_count")
    return trade_count >= _min_trades_series(df)


def summarize_results(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        excess = _numeric(group, "strategy_vs_benchmark_pct")
        trade_count = _numeric(group, "trade_count")
        drawdown = _numeric(group, "max_drawdown_pct")
        avg_excess = excess.mean()
        beat_rate = float((excess > 0).mean()) if not excess.dropna().empty else 0.0
        positive_rate = float((_numeric(group, "total_return_pct") > 0).mean())
        sufficient_trade_rate = (
            float(_sufficient_trade_mask(group).mean()) if not trade_count.empty else 0.0
        )
        stability_score = (
            (0.0 if pd.isna(avg_excess) else avg_excess / 100.0)
            + beat_rate * 0.40
            + positive_rate * 0.20
            + sufficient_trade_rate * 0.15
            + (0.0 if pd.isna(drawdown.mean()) else drawdown.mean() / 200.0)
        )
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(
            {
                "symbol_count": int(group["symbol"].nunique()) if "symbol" in group else 0,
                "threshold_count": len(group),
                "avg_feature_count": _numeric(group, "feature_count").mean(),
                "avg_total_return_pct": _numeric(group, "total_return_pct").mean(),
                "avg_benchmark_return_pct": _numeric(group, "benchmark_return_pct").mean(),
                "avg_strategy_vs_benchmark_pct": avg_excess,
                "avg_max_drawdown_pct": drawdown.mean(),
                "avg_trade_count": trade_count.mean(),
                "avg_win_rate_pct": _numeric(group, "win_rate_pct").mean(),
                "beat_benchmark_rate": beat_rate,
                "positive_return_rate": positive_rate,
                "sufficient_trade_rate": sufficient_trade_rate,
                "stability_score": stability_score,
            }
        )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("stability_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def build_per_symbol_best_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for symbol, group in df.groupby("symbol", dropna=False):
        ranked = group.copy()
        ranked["score"] = (
            _numeric(ranked, "strategy_vs_benchmark_pct").fillna(0) / 100.0
            + _sufficient_trade_mask(ranked).astype(float) * 0.20
            + _numeric(ranked, "max_drawdown_pct").fillna(0) / 200.0
        )
        ranked = ranked.sort_values(
            ["score", "strategy_vs_benchmark_pct", "max_drawdown_pct"],
            ascending=[False, False, False],
        )
        valid_ranked = ranked[
            _sufficient_trade_mask(ranked) & ~_has_low_trade_warning(ranked)
        ]
        best = valid_ranked.iloc[0] if not valid_ranked.empty else ranked.iloc[0]
        confidence = "normal" if not valid_ranked.empty else "low_confidence_low_trade_count"
        rows.append(
            {
                "symbol": _format_symbol(symbol),
                "best_pruning_mode": best.get("pruning_mode"),
                "best_model_type": best.get("model_type"),
                "best_buy_threshold": best.get("buy_threshold"),
                "best_sell_threshold": best.get("sell_threshold"),
                "best_total_return_pct": best.get("total_return_pct"),
                "best_strategy_vs_benchmark_pct": best.get("strategy_vs_benchmark_pct"),
                "best_max_drawdown_pct": best.get("max_drawdown_pct"),
                "best_trade_count": best.get("trade_count"),
                "selection_confidence": confidence,
                "selection_note": (
                    "Selected from rows meeting the minimum trade-count rule."
                    if confidence == "normal"
                    else "No row met the minimum trade-count rule without low-trade warnings; this is the historical best fallback."
                ),
            }
        )
    return pd.DataFrame(rows)


def select_recommended_walk_forward_candidate(
    walk_forward_summary: pd.DataFrame,
) -> pd.DataFrame:
    if walk_forward_summary.empty:
        return pd.DataFrame()
    sort_columns = [
        "stability_score",
        "sufficient_trade_rate",
        "beat_benchmark_rate",
        "avg_strategy_vs_benchmark_pct",
    ]
    existing_columns = [column for column in sort_columns if column in walk_forward_summary]
    if not existing_columns:
        return walk_forward_summary.head(1).copy()
    return walk_forward_summary.sort_values(
        existing_columns,
        ascending=[False] * len(existing_columns),
        na_position="last",
    ).head(1)


def build_low_confidence_and_low_trade_cases(
    combined_results: pd.DataFrame,
    warnings_df: pd.DataFrame,
    per_symbol_best: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not combined_results.empty:
        low_trade_results = combined_results[
            (~_sufficient_trade_mask(combined_results))
            | _has_low_trade_warning(combined_results)
        ].copy()
        if not low_trade_results.empty:
            low_trade_results["case_type"] = "threshold_result_low_trade"
            frames.append(low_trade_results)

    if not warnings_df.empty:
        warning_cases = warnings_df[_has_low_trade_warning(warnings_df)].copy()
        if not warning_cases.empty:
            if "symbol" in warning_cases:
                warning_cases["symbol"] = warning_cases["symbol"].map(_format_symbol)
            warning_cases["case_type"] = "warning_low_trade"
            frames.append(warning_cases)

    if not per_symbol_best.empty and "selection_confidence" in per_symbol_best:
        low_confidence = per_symbol_best[
            per_symbol_best["selection_confidence"] == "low_confidence_low_trade_count"
        ].copy()
        if not low_confidence.empty:
            low_confidence["case_type"] = "best_threshold_low_confidence"
            frames.append(low_confidence)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    if "symbol" in result:
        result["symbol"] = result["symbol"].map(_format_symbol)
    preferred_columns = [
        "case_type",
        "symbol",
        "model_type",
        "pruning_mode",
        "buy_threshold",
        "sell_threshold",
        "trade_count",
        "report_min_trades",
        "warning_type",
        "warning",
        "message",
        "selection_confidence",
        "selection_note",
        "strategy_vs_benchmark_pct",
        "max_drawdown_pct",
        "input_dir",
    ]
    ordered_columns = [
        column for column in preferred_columns if column in result.columns
    ] + [column for column in result.columns if column not in preferred_columns]
    return result[ordered_columns]


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows available._"
    table_df = df.head(max_rows)
    headers = [str(column) for column in table_df.columns]

    def clean(value):
        if value is None or pd.isna(value):
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table_df.iterrows():
        rows.append("| " + " | ".join(clean(row[column]) for column in headers) + " |")
    return "\n".join(rows)


def generate_markdown_report(
    mode_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    mode_model_summary: pd.DataFrame,
    per_symbol_best: pd.DataFrame,
    combined_results: pd.DataFrame,
    walk_forward_summary: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> str:
    best_mode_excess = (
        mode_summary.sort_values("avg_strategy_vs_benchmark_pct", ascending=False).iloc[0]
        if not mode_summary.empty
        else None
    )
    best_mode_stability = (
        mode_summary.sort_values("stability_score", ascending=False).iloc[0]
        if not mode_summary.empty
        else None
    )
    best_thresholds = (
        combined_results[
            _sufficient_trade_mask(combined_results)
            & ~_has_low_trade_warning(combined_results)
        ].sort_values(
            ["strategy_vs_benchmark_pct", "max_drawdown_pct"],
            ascending=[False, False],
        ).head(20)
        if not combined_results.empty
        else pd.DataFrame()
    )
    if best_thresholds.empty and not combined_results.empty:
        best_thresholds = combined_results.sort_values(
            ["strategy_vs_benchmark_pct", "max_drawdown_pct"],
            ascending=[False, False],
        ).head(20).copy()
        best_thresholds["selection_confidence"] = "low_confidence_low_trade_count"
    beats = (
        combined_results[_numeric(combined_results, "strategy_vs_benchmark_pct") > 0]
        if not combined_results.empty
        else pd.DataFrame()
    )
    underperforms = (
        combined_results[_numeric(combined_results, "strategy_vs_benchmark_pct") < 0]
        if not combined_results.empty
        else pd.DataFrame()
    )
    wf_text = (
        "Walk-forward results were available and are summarized below."
        if not walk_forward_summary.empty
        else "No walk-forward result files were available in the selected inputs."
    )
    walk_forward_candidate = select_recommended_walk_forward_candidate(
        walk_forward_summary
    )
    low_trade_cases = build_low_confidence_and_low_trade_cases(
        combined_results,
        warnings_df,
        per_symbol_best,
    )
    sections = [
        "# Reduced Feature Threshold Experiment Report",
        "",
        "## Overall Conclusion",
        (
            "This report aggregates threshold sensitivity experiments across symbols. "
            "The historical threshold tables show in-sample backtest winners. The "
            "walk-forward section is the robustness check to use for candidate "
            "selection. Threshold tuning can overfit historical data and does not "
            "imply real trading profit."
        ),
        "",
        "## Best Pruning Mode by Average Excess Return",
        "n/a"
        if best_mode_excess is None
        else f"`{best_mode_excess['pruning_mode']}` with average excess return {best_mode_excess['avg_strategy_vs_benchmark_pct']:.4f}%.",
        "",
        "## Best Pruning Mode by Stability",
        "n/a"
        if best_mode_stability is None
        else f"`{best_mode_stability['pruning_mode']}` with stability score {best_mode_stability['stability_score']:.4f}.",
        "",
        "## Best Historical Threshold Results",
        (
            "Rows in this section are historical threshold backtest results. When "
            "possible, low-trade rows are excluded from this ranking."
        ),
        _markdown_table(best_thresholds),
        "",
        "## Per-Symbol Best Historical Mode and Threshold Pair",
        (
            "Each symbol prefers rows meeting the minimum trade-count rule and "
            "without low-trade warnings. Fallback rows are marked low confidence."
        ),
        _markdown_table(per_symbol_best),
        "",
        "## Cases Where a Mode Beats Benchmark",
        _markdown_table(beats),
        "",
        "## Cases Where a Mode Underperforms Benchmark",
        _markdown_table(underperforms),
        "",
        "## Pruning Mode Ranking",
        _markdown_table(mode_summary),
        "",
        "## Model Ranking",
        _markdown_table(model_summary),
        "",
        "## Pruning Mode + Model Ranking",
        _markdown_table(mode_model_summary),
        "",
        "## Recommended Walk-Forward Candidate",
        (
            "This candidate is selected from walk-forward summaries by stability "
            "score, sufficient trade rate, benchmark beat rate, and average excess "
            "return, in that order."
        ),
        _markdown_table(walk_forward_candidate),
        "",
        "## Walk-Forward Robustness Ranking",
        wf_text,
        _markdown_table(walk_forward_summary),
        "",
        "## Low-Confidence and Low-Trade Cases",
        _markdown_table(low_trade_cases),
        "",
        "## Research Warnings",
        _markdown_table(warnings_df),
        "",
    ]
    return "\n".join(sections)


def build_threshold_experiment_report(input_dirs: list[str]) -> dict[str, Any]:
    combined, walk_forward, warnings_df = load_threshold_experiment_outputs(input_dirs)
    combined = _drop_stale_low_trade_warning_text(combined)
    walk_forward = _drop_stale_low_trade_warning_text(walk_forward)
    warnings_df = _drop_stale_low_trade_warning_text(warnings_df)
    mode_summary = summarize_results(combined, ["pruning_mode"])
    model_summary = summarize_results(combined, ["model_type"])
    mode_model_summary = summarize_results(combined, ["pruning_mode", "model_type"])
    per_symbol_best = build_per_symbol_best_thresholds(combined)
    walk_forward_summary = summarize_results(
        walk_forward,
        ["pruning_mode", "model_type", "buy_threshold", "sell_threshold"],
    )
    report = generate_markdown_report(
        mode_summary,
        model_summary,
        mode_model_summary,
        per_symbol_best,
        combined,
        walk_forward_summary,
        warnings_df,
    )
    return {
        "combined_threshold_results": combined,
        "threshold_mode_summary": mode_summary,
        "threshold_model_summary": model_summary,
        "threshold_mode_model_summary": mode_model_summary,
        "per_symbol_best_thresholds": per_symbol_best,
        "walk_forward_combined_results": walk_forward,
        "walk_forward_summary": walk_forward_summary,
        "warnings": warnings_df,
        "markdown_report": report,
    }


def save_threshold_experiment_report(
    input_dirs: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    result = build_threshold_experiment_report(input_dirs)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined_threshold_results": output_path / "combined_threshold_results.csv",
        "threshold_mode_summary": output_path / "threshold_mode_summary.csv",
        "threshold_model_summary": output_path / "threshold_model_summary.csv",
        "threshold_mode_model_summary": output_path
        / "threshold_mode_model_summary.csv",
        "per_symbol_best_thresholds": output_path / "per_symbol_best_thresholds.csv",
        "walk_forward_combined_results": output_path
        / "walk_forward_combined_results.csv",
        "walk_forward_summary": output_path / "walk_forward_summary.csv",
        "warnings": output_path / "warnings.csv",
        "report": output_path / "threshold_experiment_report.md",
        "run_config": output_path / "run_config.json",
    }
    for key, path in paths.items():
        if key in {"report", "run_config"}:
            continue
        result[key].to_csv(path, index=False)
    paths["report"].write_text(result["markdown_report"], encoding="utf-8")
    run_config = {
        "input_dirs": input_dirs,
        "output_dir": str(output_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["run_config"] = run_config
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate reduced feature threshold experiment outputs.",
    )
    parser.add_argument("--input-dirs", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/reduced_feature_threshold_summary_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        input_dirs = parse_input_dirs(args.input_dirs)
        result = save_threshold_experiment_report(input_dirs, args.output_dir)
    except Exception as exc:
        print(f"Error: failed to generate threshold experiment report: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Threshold Experiment Report")
    print("-----------------------------------------")
    print(f"Input directories: {input_dirs}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: Threshold tuning can overfit historical data. "
        "This is research only, not financial advice."
    )


if __name__ == "__main__":
    main()
