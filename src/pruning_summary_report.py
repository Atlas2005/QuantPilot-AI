import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIRS = [
    "outputs/factor_pruning_real_000001",
    "outputs/factor_pruning_real_600519",
    "outputs/factor_pruning_real_000858",
    "outputs/factor_pruning_real_600036",
    "outputs/factor_pruning_real_601318",
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
    for prefix in ["factor_pruning_real_", "factor_pruning_", "pruning_"]:
        if name.startswith(prefix):
            return name.replace(prefix, "", 1)
    return name


def load_pruning_directory(input_dir: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    path = Path(input_dir)
    warnings = []
    summary_path = path / "pruning_summary.csv"
    run_config = load_json_if_available(path / "run_config.json")
    symbol = infer_symbol(path, run_config)

    if not summary_path.exists():
        warnings.append(
            {
                "input_dir": str(path),
                "symbol": symbol,
                "warning": "Missing pruning_summary.csv.",
            }
        )
        return pd.DataFrame(), warnings

    try:
        df = pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:
        warnings.append(
            {
                "input_dir": str(path),
                "symbol": symbol,
                "warning": "pruning_summary.csv is empty.",
            }
        )
        return pd.DataFrame(), warnings

    df["symbol"] = symbol
    df["input_dir"] = str(path)
    return df, warnings


def load_multiple_pruning_outputs(input_dirs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    warnings = []
    for input_dir in input_dirs:
        df, directory_warnings = load_pruning_directory(input_dir)
        if not df.empty:
            frames.append(df)
        warnings.extend(directory_warnings)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    warning_df = pd.DataFrame(warnings, columns=["input_dir", "symbol", "warning"])
    return combined, warning_df


def _positive_rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float((values > 0).mean())


def build_pruning_mode_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame()

    rows = []
    for mode, group in combined_df.groupby("pruning_mode", dropna=False):
        roc_delta = pd.to_numeric(
            group["avg_delta_test_roc_auc_vs_full"],
            errors="coerce",
        )
        f1_delta = pd.to_numeric(
            group["avg_delta_test_f1_vs_full"],
            errors="coerce",
        )
        avg_roc_delta = float(roc_delta.mean()) if not roc_delta.dropna().empty else None
        avg_f1_delta = float(f1_delta.mean()) if not f1_delta.dropna().empty else None
        roc_win_rate = _positive_rate(roc_delta)
        f1_win_rate = _positive_rate(f1_delta)
        stability_score = None
        if avg_roc_delta is not None and avg_f1_delta is not None:
            stability_score = (
                avg_roc_delta * 2.0
                + avg_f1_delta
                + (roc_win_rate or 0.0) * 0.25
                + (f1_win_rate or 0.0) * 0.15
            )

        rows.append(
            {
                "pruning_mode": mode,
                "symbol_count": int(group["symbol"].nunique()),
                "avg_model_count": group["model_count"].mean(),
                "avg_feature_count": group["avg_feature_count"].mean(),
                "avg_test_roc_auc": group["avg_test_roc_auc"].mean(),
                "avg_test_f1": group["avg_test_f1"].mean(),
                "avg_validation_roc_auc": group["avg_validation_roc_auc"].mean(),
                "avg_delta_test_roc_auc_vs_full": avg_roc_delta,
                "avg_delta_test_f1_vs_full": avg_f1_delta,
                "win_rate_roc_vs_full": roc_win_rate,
                "win_rate_f1_vs_full": f1_win_rate,
                "stability_score": stability_score,
            }
        )

    return pd.DataFrame(rows).sort_values("stability_score", ascending=False)


def build_per_symbol_best_modes(combined_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame()
    rows = []
    for symbol, group in combined_df.groupby("symbol", dropna=False):
        roc_group = group.dropna(subset=["avg_test_roc_auc"])
        f1_group = group.dropna(subset=["avg_test_f1"])
        best_roc = roc_group.sort_values("avg_test_roc_auc", ascending=False).iloc[0] if not roc_group.empty else None
        best_f1 = f1_group.sort_values("avg_test_f1", ascending=False).iloc[0] if not f1_group.empty else None
        rows.append(
            {
                "symbol": symbol,
                "best_mode_by_roc_auc": None if best_roc is None else best_roc["pruning_mode"],
                "best_roc_auc": None if best_roc is None else best_roc["avg_test_roc_auc"],
                "best_mode_by_f1": None if best_f1 is None else best_f1["pruning_mode"],
                "best_f1": None if best_f1 is None else best_f1["avg_test_f1"],
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows available._"
    table_df = df.head(max_rows).copy()
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
    if len(df) > max_rows:
        rows.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(rows)


def generate_markdown_report(
    combined_df: pd.DataFrame,
    mode_summary: pd.DataFrame,
    per_symbol_best: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> str:
    if mode_summary.empty:
        recommended = None
        best_roc = None
        best_f1 = None
        best_stability = None
    else:
        best_roc = mode_summary.sort_values("avg_test_roc_auc", ascending=False).iloc[0]
        best_f1 = mode_summary.sort_values("avg_test_f1", ascending=False).iloc[0]
        best_stability = mode_summary.sort_values("stability_score", ascending=False).iloc[0]
        recommended = best_stability["pruning_mode"]

    hurts_full = pd.DataFrame()
    if not combined_df.empty:
        hurts_full = combined_df[
            pd.to_numeric(combined_df["avg_delta_test_roc_auc_vs_full"], errors="coerce")
            < 0
        ]

    sections = [
        "# Multi-Symbol Factor Pruning Summary Report",
        "",
        "## Overall Conclusion",
    ]
    if recommended is None:
        sections.append("No recommendation can be made because no pruning summaries were loaded.")
    else:
        sections.append(
            f"The recommended default candidate for the next experiments is "
            f"`{recommended}` because it has the highest stability score in this "
            "aggregation. This is a research decision, not a trading recommendation."
        )

    sections.extend(
        [
            "",
            "## Best Modes",
            f"- Best by average ROC AUC: `{best_roc['pruning_mode'] if best_roc is not None else 'n/a'}`",
            f"- Best by average F1: `{best_f1['pruning_mode'] if best_f1 is not None else 'n/a'}`",
            f"- Best by stability score: `{best_stability['pruning_mode'] if best_stability is not None else 'n/a'}`",
            "",
            "## Pruning Mode Ranking",
            _markdown_table(mode_summary),
            "",
            "## Per-Symbol Best Modes",
            _markdown_table(per_symbol_best),
            "",
            "## Cases Where Pruning Hurts Full",
            _markdown_table(hurts_full),
            "",
            "## Recommended Default Mode for Next Experiments",
        ]
    )
    if recommended is None:
        sections.append("_No default mode selected._")
    else:
        sections.append(
            f"Use `{recommended}` as the next reduced feature set candidate, then "
            "retest it with walk-forward validation and more symbols before making "
            "any default model workflow changes."
        )

    sections.extend(
        [
            "",
            "## Warnings",
            _markdown_table(warnings_df),
            "",
            "## Research Warnings",
            (
                "Reduced feature sets can overfit the symbols and dates used in this "
                "summary. A better ROC AUC or F1 score does not guarantee profitable "
                "trading after transaction costs, slippage, execution timing, and "
                "regime changes."
            ),
            "",
        ]
    )
    return "\n".join(sections)


def build_pruning_summary_report(input_dirs: list[str]) -> dict[str, Any]:
    combined_df, warnings_df = load_multiple_pruning_outputs(input_dirs)
    mode_summary = build_pruning_mode_summary(combined_df)
    per_symbol_best = build_per_symbol_best_modes(combined_df)
    report = generate_markdown_report(
        combined_df=combined_df,
        mode_summary=mode_summary,
        per_symbol_best=per_symbol_best,
        warnings_df=warnings_df,
    )
    return {
        "combined_pruning_results": combined_df,
        "pruning_mode_summary": mode_summary,
        "per_symbol_best_modes": per_symbol_best,
        "warnings": warnings_df,
        "markdown_report": report,
    }


def save_pruning_summary_report(
    input_dirs: list[str],
    output_dir: str | Path,
    report_name: str = "pruning_summary_report.md",
) -> dict[str, Any]:
    result = build_pruning_summary_report(input_dirs)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined_pruning_results": output_path / "combined_pruning_results.csv",
        "pruning_mode_summary": output_path / "pruning_mode_summary.csv",
        "per_symbol_best_modes": output_path / "per_symbol_best_modes.csv",
        "warnings": output_path / "warnings.csv",
        "run_config": output_path / "run_config.json",
        "report": output_path / report_name,
    }
    result["combined_pruning_results"].to_csv(paths["combined_pruning_results"], index=False)
    result["pruning_mode_summary"].to_csv(paths["pruning_mode_summary"], index=False)
    result["per_symbol_best_modes"].to_csv(paths["per_symbol_best_modes"], index=False)
    result["warnings"].to_csv(paths["warnings"], index=False)
    paths["run_config"].write_text(
        json.dumps(
            {
                "input_dirs": input_dirs,
                "output_dir": str(output_path),
                "report_name": report_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths["report"].write_text(result["markdown_report"], encoding="utf-8")
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result
