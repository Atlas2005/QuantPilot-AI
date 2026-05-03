import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_input_dirs(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
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


def load_threshold_experiment_outputs(
    input_dirs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_frames = []
    walk_forward_frames = []
    warnings = []
    for input_dir in input_dirs:
        path = Path(input_dir)
        run_config = _load_json(path / "run_config.json")
        symbol = infer_symbol(path, run_config)
        threshold_df = _read_csv(path / "threshold_backtest_results.csv")
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
            threshold_df["symbol"] = threshold_df.get("symbol", symbol)
            threshold_df["input_dir"] = str(path)
            threshold_frames.append(threshold_df)

        walk_forward_df = _read_csv(path / "walk_forward_results.csv")
        if not walk_forward_df.empty:
            walk_forward_df["symbol"] = walk_forward_df.get("symbol", symbol)
            walk_forward_df["input_dir"] = str(path)
            walk_forward_frames.append(walk_forward_df)

        source_warnings = _read_csv(path / "warnings.csv")
        if not source_warnings.empty:
            source_warnings["input_dir"] = str(path)
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
        sufficient_trade_rate = float((trade_count > 3).mean()) if not trade_count.empty else 0.0
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
            + (_numeric(ranked, "trade_count") > 3).astype(float) * 0.20
            + _numeric(ranked, "max_drawdown_pct").fillna(0) / 200.0
        )
        best = ranked.sort_values(
            ["score", "strategy_vs_benchmark_pct", "max_drawdown_pct"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(
            {
                "symbol": symbol,
                "best_pruning_mode": best.get("pruning_mode"),
                "best_model_type": best.get("model_type"),
                "best_buy_threshold": best.get("buy_threshold"),
                "best_sell_threshold": best.get("sell_threshold"),
                "best_total_return_pct": best.get("total_return_pct"),
                "best_strategy_vs_benchmark_pct": best.get("strategy_vs_benchmark_pct"),
                "best_max_drawdown_pct": best.get("max_drawdown_pct"),
                "best_trade_count": best.get("trade_count"),
            }
        )
    return pd.DataFrame(rows)


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
        combined_results.sort_values(
            ["strategy_vs_benchmark_pct", "max_drawdown_pct"],
            ascending=[False, False],
        ).head(20)
        if not combined_results.empty
        else pd.DataFrame()
    )
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
    sections = [
        "# Reduced Feature Threshold Experiment Report",
        "",
        "## Overall Conclusion",
        (
            "This report aggregates threshold sensitivity experiments across symbols. "
            "Threshold tuning can overfit historical data and does not imply real "
            "trading profit."
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
        "## Best Threshold Pair by Pruning Mode",
        _markdown_table(best_thresholds),
        "",
        "## Per-Symbol Best Mode and Threshold Pair",
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
        "## Walk-Forward Robustness Conclusion",
        wf_text,
        _markdown_table(walk_forward_summary),
        "",
        "## Low Trade Count and Research Warnings",
        _markdown_table(warnings_df),
        "",
    ]
    return "\n".join(sections)


def build_threshold_experiment_report(input_dirs: list[str]) -> dict[str, Any]:
    combined, walk_forward, warnings_df = load_threshold_experiment_outputs(input_dirs)
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
