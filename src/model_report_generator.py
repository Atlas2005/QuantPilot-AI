import json
from pathlib import Path
from typing import Any

import pandas as pd


ROBUSTNESS_FILES = {
    "model_summary": "model_summary.csv",
    "model_ranking": "model_ranking.csv",
    "training_results": "training_results.csv",
    "warnings": "warnings.csv",
    "run_config": "run_config.json",
}


def load_csv_if_available(path: Path) -> pd.DataFrame:
    """Load a CSV if it exists, returning an empty DataFrame otherwise."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_json_if_available(path: Path) -> dict[str, Any]:
    """Load a JSON file if it exists, returning an empty dict otherwise."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"Could not parse {path.name}"}


def load_robustness_outputs(input_dir: str | Path) -> dict[str, Any]:
    """Read robustness output files from an existing output directory."""
    base = Path(input_dir)
    return {
        "input_dir": base,
        "model_summary": load_csv_if_available(base / ROBUSTNESS_FILES["model_summary"]),
        "model_ranking": load_csv_if_available(base / ROBUSTNESS_FILES["model_ranking"]),
        "training_results": load_csv_if_available(
            base / ROBUSTNESS_FILES["training_results"]
        ),
        "warnings": load_csv_if_available(base / ROBUSTNESS_FILES["warnings"]),
        "run_config": load_json_if_available(base / ROBUSTNESS_FILES["run_config"]),
    }


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def format_value(value: Any, digits: int = 4) -> str:
    """Format report values without exposing pandas NaN strings to readers."""
    if _is_missing(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def describe_roc_auc(value: Any) -> str:
    """Return a plain-language description for a ROC AUC value."""
    if _is_missing(value):
        return "ROC AUC is unavailable."
    roc_auc = float(value)
    if roc_auc < 0.45:
        return "below random, which suggests the signal may be inverted or unstable."
    if roc_auc < 0.55:
        return "close to random."
    if roc_auc < 0.65:
        return "modestly above random, but still weak for research conclusions."
    if roc_auc < 0.80:
        return "meaningfully above random and worth further investigation."
    return "very high; verify leakage, sample size, and period stability carefully."


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a small DataFrame as a Markdown table without extra dependencies."""
    if df.empty:
        return "_No rows available._"

    table_df = df.head(max_rows).copy()
    headers = [str(column) for column in table_df.columns]

    def clean_cell(value: Any) -> str:
        text = format_value(value)
        return text.replace("|", "\\|").replace("\n", " ")

    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table_df.iterrows():
        rows.append("| " + " | ".join(clean_cell(row[column]) for column in headers) + " |")

    if len(df) > max_rows:
        rows.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(rows)


def get_best_model(summary_df: pd.DataFrame) -> dict[str, Any] | None:
    """Find the model type with the highest average test ROC AUC."""
    if summary_df.empty or "avg_test_roc_auc" not in summary_df.columns:
        return None
    ranked = summary_df.dropna(subset=["avg_test_roc_auc"]).sort_values(
        "avg_test_roc_auc",
        ascending=False,
    )
    if ranked.empty:
        return None
    return ranked.iloc[0].to_dict()


def get_weakest_symbol_model(results_df: pd.DataFrame) -> dict[str, Any] | None:
    """Find the weakest successful symbol/model row by test ROC AUC then F1."""
    if results_df.empty or "test_roc_auc" not in results_df.columns:
        return None
    success = results_df.copy()
    if "error" in success.columns:
        success = success[success["error"].isna()]
    success = success.dropna(subset=["test_roc_auc"])
    if success.empty:
        return None
    sort_columns = ["test_roc_auc"]
    if "test_f1" in success.columns:
        sort_columns.append("test_f1")
    return success.sort_values(sort_columns, ascending=True).iloc[0].to_dict()


def build_validation_test_gap_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact validation-vs-test ROC AUC stability table."""
    required = {"symbol", "model_type", "validation_roc_auc", "test_roc_auc"}
    if results_df.empty or not required.issubset(results_df.columns):
        return pd.DataFrame()

    gap_df = results_df.copy()
    if "error" in gap_df.columns:
        gap_df = gap_df[gap_df["error"].isna()]
    gap_df = gap_df.dropna(subset=["validation_roc_auc", "test_roc_auc"])
    if gap_df.empty:
        return pd.DataFrame()

    gap_df["validation_test_roc_auc_gap"] = (
        gap_df["validation_roc_auc"] - gap_df["test_roc_auc"]
    )
    columns = [
        "symbol",
        "model_type",
        "validation_roc_auc",
        "test_roc_auc",
        "validation_test_roc_auc_gap",
    ]
    return gap_df[columns].sort_values(
        "validation_test_roc_auc_gap",
        key=lambda values: values.abs(),
        ascending=False,
    )


def build_warning_summary(warnings_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize warnings by warning type."""
    if warnings_df.empty or "warning_type" not in warnings_df.columns:
        return pd.DataFrame()
    summary = (
        warnings_df.groupby("warning_type", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return summary


def _run_config_section(run_config: dict[str, Any]) -> str:
    if not run_config:
        return "_No run_config.json file was available._"
    lines = []
    for key, value in run_config.items():
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        lines.append(f"- **{key}**: {format_value(value)}")
    return "\n".join(lines)


def _build_executive_summary(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> list[str]:
    lines = []
    best_model = get_best_model(summary_df)
    weakest_pair = get_weakest_symbol_model(results_df)
    warning_summary = build_warning_summary(warnings_df)
    gap_df = build_validation_test_gap_table(results_df)

    if best_model is None:
        lines.append("- No model has an available average test ROC AUC.")
    else:
        model_type = best_model.get("model_type")
        roc_auc = best_model.get("avg_test_roc_auc")
        lines.append(
            f"- Best model by average test ROC AUC: **{model_type}** "
            f"({format_value(roc_auc)}), which is {describe_roc_auc(roc_auc)}"
        )

    if weakest_pair is None:
        lines.append("- No successful symbol/model pair was available for weakness analysis.")
    else:
        lines.append(
            "- Weakest successful pair: "
            f"**{weakest_pair.get('symbol')} / {weakest_pair.get('model_type')}** "
            f"with test ROC AUC {format_value(weakest_pair.get('test_roc_auc'))}."
        )

    if gap_df.empty:
        lines.append("- Validation-test ROC AUC stability could not be measured.")
    else:
        max_gap = gap_df["validation_test_roc_auc_gap"].abs().max()
        if max_gap > 0.20:
            lines.append(
                f"- Largest validation-test ROC AUC gap is {format_value(max_gap)}; "
                "this is large enough to question stability."
            )
        else:
            lines.append(
                f"- Largest validation-test ROC AUC gap is {format_value(max_gap)}; "
                "this looks reasonably controlled for a first robustness check."
            )

    if warning_summary.empty:
        lines.append("- No warnings were recorded in warnings.csv.")
    else:
        total_warnings = int(warning_summary["count"].sum())
        lines.append(f"- Warning rows recorded: **{total_warnings}**.")

    return lines


def generate_model_robustness_report(input_dir: str | Path) -> str:
    """Generate a Markdown research report from robustness output files."""
    outputs = load_robustness_outputs(input_dir)
    summary_df = outputs["model_summary"]
    ranking_df = outputs["model_ranking"]
    results_df = outputs["training_results"]
    warnings_df = outputs["warnings"]
    run_config = outputs["run_config"]

    best_model = get_best_model(summary_df)
    weakest_pair = get_weakest_symbol_model(results_df)
    gap_df = build_validation_test_gap_table(results_df)
    warning_summary = build_warning_summary(warnings_df)

    sections = [
        "# Model Robustness Research Report",
        "",
        f"Source directory: `{Path(input_dir)}`",
        "",
        "## Run Configuration",
        _run_config_section(run_config),
        "",
        "## Executive Summary",
        "\n".join(_build_executive_summary(summary_df, results_df, warnings_df)),
        "",
        "## Best Model Summary",
    ]

    if best_model is None:
        sections.append("_No best model could be selected from model_summary.csv._")
    else:
        best_roc_auc = best_model.get("avg_test_roc_auc")
        sections.extend(
            [
                (
                    f"The best model by average test ROC AUC is "
                    f"**{best_model.get('model_type')}**."
                ),
                "",
                markdown_table(pd.DataFrame([best_model])),
                "",
                (
                    f"Its average test ROC AUC is {format_value(best_roc_auc)}, "
                    f"which is {describe_roc_auc(best_roc_auc)}"
                ),
            ]
        )

    sections.extend(
        [
            "",
            "## Model Ranking Interpretation",
        ]
    )
    if ranking_df.empty:
        sections.append("_model_ranking.csv was missing or empty._")
    else:
        sections.extend(
            [
                markdown_table(ranking_df),
                "",
                (
                    "The ranking score is an educational comparison metric. It should "
                    "be read as a stability screen, not as evidence of a profitable "
                    "trading strategy."
                ),
            ]
        )

    sections.extend(
        [
            "",
            "## Symbol-Level Weakness Analysis",
        ]
    )
    if weakest_pair is None:
        sections.append("_No successful symbol/model row was available._")
    else:
        sample_note = ""
        test_rows = weakest_pair.get("test_rows")
        if not _is_missing(test_rows) and float(test_rows) < 50:
            sample_note = " The test sample is small, so this weakness may be noisy."
        sections.append(
            f"The weakest successful pair is **{weakest_pair.get('symbol')} / "
            f"{weakest_pair.get('model_type')}**, with test ROC AUC "
            f"{format_value(weakest_pair.get('test_roc_auc'))} and test F1 "
            f"{format_value(weakest_pair.get('test_f1'))}.{sample_note}"
        )
        detail_columns = [
            column
            for column in [
                "symbol",
                "model_type",
                "test_rows",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "test_positive_rate",
            ]
            if column in results_df.columns
        ]
        if detail_columns:
            sections.extend(["", markdown_table(results_df[detail_columns].head(20))])

    sections.extend(
        [
            "",
            "## Validation vs Test Stability Analysis",
        ]
    )
    if gap_df.empty:
        sections.append("_Validation-test ROC AUC gap could not be calculated._")
    else:
        max_gap = gap_df["validation_test_roc_auc_gap"].abs().max()
        gap_note = (
            "A gap above about 0.20 is a practical warning sign in this educational "
            "workflow because validation performance may not carry into test data."
        )
        sections.extend(
            [
                f"Largest absolute validation-test ROC AUC gap: **{format_value(max_gap)}**.",
                gap_note,
                "",
                markdown_table(gap_df),
            ]
        )

    sections.extend(
        [
            "",
            "## Warning Summary",
        ]
    )
    if warning_summary.empty:
        sections.append("_No warning rows were available._")
    else:
        sections.extend([markdown_table(warning_summary), ""])
        sections.append(markdown_table(warnings_df, max_rows=30))

    sections.extend(
        [
            "",
            "## Educational Interpretation",
            (
                "ROC AUC around 0.5 means the classifier is close to random. Values "
                "below 0.5 can mean the signal is inverted, unstable, or simply noisy. "
                "Higher values are worth investigating only after checking sample size, "
                "date splits, leakage warnings, and whether performance is consistent "
                "across symbols."
            ),
            "",
            (
                "High ML classification metrics do not guarantee trading profits. A "
                "classifier can predict the label reasonably well and still lose money "
                "after transaction costs, slippage, execution timing, position sizing, "
                "and market regime changes."
            ),
            "",
            "## Next-Step Recommendations",
            "- Re-run robustness training across more symbols and longer date ranges.",
            "- Check suspiciously high metrics for feature leakage or tiny test samples.",
            "- Compare validation and test ROC AUC gaps before trusting a model type.",
            "- Add richer non-leaking features only after the baseline pipeline is stable.",
            "- Test any promising model through the ML signal backtest with realistic costs.",
            "- Treat this report as educational research, not financial advice.",
            "",
        ]
    )

    return "\n".join(sections)


def write_model_robustness_report(
    input_dir: str | Path,
    output_path: str | Path,
) -> tuple[Path, str]:
    """Generate and save a Markdown robustness report."""
    report_text = generate_model_robustness_report(input_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    return path, report_text
