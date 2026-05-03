import json
from pathlib import Path
from typing import Any

import pandas as pd


DECISION_COLUMNS = [
    "factor_group",
    "decision",
    "avg_only_group_test_roc_auc",
    "avg_only_group_test_f1",
    "avg_drop_group_roc_auc_delta",
    "avg_drop_group_f1_delta",
    "experiment_count",
    "model_type_count",
    "consistency_score",
    "rationale",
]


def load_csv_if_available(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_json_if_available(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"Could not parse {path.name}"}


def load_factor_ablation_outputs(input_dir: str | Path) -> dict[str, Any]:
    base = Path(input_dir)
    return {
        "input_dir": base,
        "group_summary": load_csv_if_available(base / "group_summary.csv"),
        "feature_impact_ranking": load_csv_if_available(
            base / "feature_impact_ranking.csv"
        ),
        "feature_pruning_recommendations": load_csv_if_available(
            base / "feature_pruning_recommendations.csv"
        ),
        "ablation_results": load_csv_if_available(base / "ablation_results.csv"),
        "warnings": load_csv_if_available(base / "warnings.csv"),
        "run_config": load_json_if_available(base / "run_config.json"),
    }


def _mean_or_none(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _format_value(value: Any, digits: int = 4) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _decision_from_metrics(
    only_auc: float | None,
    drop_delta: float | None,
    experiment_count: int,
    consistency_score: float,
) -> tuple[str, str]:
    if experiment_count < 2:
        return (
            "needs_more_data",
            "Too few experiments are available for a stable decision.",
        )

    if only_auc is not None and only_auc >= 0.56 and drop_delta is not None and drop_delta <= -0.01 and consistency_score >= 0.5:
        return (
            "core_keep",
            "Only-group performance is above random and dropping the group tends to hurt the full model.",
        )

    if only_auc is not None and only_auc >= 0.52 and (drop_delta is None or drop_delta <= 0.01):
        return (
            "keep_observe",
            "The group shows some useful signal or does not clearly damage the full model.",
        )

    if drop_delta is not None and drop_delta > 0.01:
        return (
            "reduce_weight",
            "Dropping this group improves test ROC AUC on average, so it may be adding noise.",
        )

    if only_auc is not None and only_auc < 0.50:
        return (
            "weak_or_noisy",
            "Only-group ROC AUC is below random, suggesting weak or unstable signal.",
        )

    return (
        "needs_more_data",
        "The diagnostics are mixed or incomplete, so more symbols and periods are needed.",
    )


def build_decision_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    if group_summary.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)

    rows = []
    for factor_group, group_df in group_summary.groupby("factor_group", dropna=False):
        only_df = group_df[group_df["ablation_type"] == "only_group"]
        drop_df = group_df[group_df["ablation_type"] == "drop_group"]

        only_auc = _mean_or_none(only_df.get("avg_test_roc_auc", pd.Series(dtype=float)))
        only_f1 = _mean_or_none(only_df.get("avg_test_f1", pd.Series(dtype=float)))
        drop_delta = _mean_or_none(
            drop_df.get("avg_test_roc_auc_delta_vs_full", pd.Series(dtype=float))
        )
        drop_f1_delta = _mean_or_none(
            drop_df.get("avg_test_f1_delta_vs_full", pd.Series(dtype=float))
        )
        experiment_count = int(
            pd.to_numeric(
                group_df.get("experiment_count", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        )
        model_type_count = int(group_df["model_type"].nunique()) if "model_type" in group_df else 0

        consistency_parts = []
        if not only_df.empty and "avg_test_roc_auc" in only_df:
            consistency_parts.append(float((only_df["avg_test_roc_auc"] >= 0.5).mean()))
        if not drop_df.empty and "avg_test_roc_auc_delta_vs_full" in drop_df:
            consistency_parts.append(
                float((drop_df["avg_test_roc_auc_delta_vs_full"] <= 0).mean())
            )
        consistency_score = (
            float(sum(consistency_parts) / len(consistency_parts))
            if consistency_parts
            else 0.0
        )
        decision, rationale = _decision_from_metrics(
            only_auc=only_auc,
            drop_delta=drop_delta,
            experiment_count=experiment_count,
            consistency_score=consistency_score,
        )
        rows.append(
            {
                "factor_group": factor_group,
                "decision": decision,
                "avg_only_group_test_roc_auc": only_auc,
                "avg_only_group_test_f1": only_f1,
                "avg_drop_group_roc_auc_delta": drop_delta,
                "avg_drop_group_f1_delta": drop_f1_delta,
                "experiment_count": experiment_count,
                "model_type_count": model_type_count,
                "consistency_score": consistency_score,
                "rationale": rationale,
            }
        )

    decision_order = {
        "core_keep": 0,
        "keep_observe": 1,
        "reduce_weight": 2,
        "weak_or_noisy": 3,
        "needs_more_data": 4,
    }
    result = pd.DataFrame(rows, columns=DECISION_COLUMNS)
    result["_decision_order"] = result["decision"].map(decision_order).fillna(9)
    return result.sort_values(
        ["_decision_order", "avg_only_group_test_roc_auc"],
        ascending=[True, False],
    ).drop(columns=["_decision_order"])


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows available._"
    table_df = df.head(max_rows).copy()
    headers = [str(column) for column in table_df.columns]

    def clean(value: Any) -> str:
        return _format_value(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table_df.iterrows():
        rows.append("| " + " | ".join(clean(row[column]) for column in headers) + " |")
    if len(df) > max_rows:
        rows.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(rows)


def _run_config_text(run_config: dict[str, Any]) -> str:
    if not run_config:
        return "_No run_config.json was available._"
    return "\n".join(f"- **{key}**: {_format_value(value)}" for key, value in run_config.items())


def generate_factor_decision_report(input_dir: str | Path) -> dict[str, Any]:
    outputs = load_factor_ablation_outputs(input_dir)
    group_summary = outputs["group_summary"]
    feature_ranking = outputs["feature_impact_ranking"]
    pruning_recommendations = outputs["feature_pruning_recommendations"]
    warnings_df = outputs["warnings"]
    run_config = outputs["run_config"]
    decision_summary = build_decision_summary(group_summary)

    strongest = decision_summary[
        decision_summary["decision"].isin(["core_keep", "keep_observe"])
    ].head(10)
    weakest = decision_summary[
        decision_summary["decision"].isin(["reduce_weight", "weak_or_noisy"])
    ].head(10)

    sections = [
        "# Factor Selection and Retention Decision Report",
        "",
        f"Source directory: `{Path(input_dir)}`",
        "",
        "## Run Configuration",
        _run_config_text(run_config),
        "",
        "## Overall Conclusion",
    ]

    if decision_summary.empty:
        sections.append(
            "No factor decision could be produced because group_summary.csv was missing or empty."
        )
    else:
        counts = decision_summary["decision"].value_counts().to_dict()
        sections.append(
            "This report converts ablation diagnostics into research decisions. "
            f"Decision counts: {counts}. These are feature-engineering decisions, "
            "not trading advice."
        )

    sections.extend(
        [
            "",
            "## Decision Summary",
            markdown_table(decision_summary),
            "",
            "## Strongest Factor Groups",
            markdown_table(strongest),
            "",
            "## Weakest or Noisy Factor Groups",
            markdown_table(weakest),
            "",
            "## What to Keep",
        ]
    )
    keep_df = decision_summary[decision_summary["decision"].isin(["core_keep", "keep_observe"])]
    if keep_df.empty:
        sections.append("_No groups are currently classified as keep candidates._")
    else:
        sections.append(
            "Keep or observe groups that perform reasonably as only-group tests "
            "and do not improve the full model when removed."
        )
        sections.append(markdown_table(keep_df[["factor_group", "decision", "rationale"]]))

    sections.extend(["", "## What to Reduce"])
    reduce_df = decision_summary[
        decision_summary["decision"].isin(["reduce_weight", "weak_or_noisy"])
    ]
    if reduce_df.empty:
        sections.append("_No groups are currently classified as reduce candidates._")
    else:
        sections.append(
            "Groups in this section may be noisy for the current symbols, period, "
            "or model types. Reduce or re-test them before adding more related factors."
        )
        sections.append(markdown_table(reduce_df[["factor_group", "decision", "rationale"]]))

    sections.extend(
        [
            "",
            "## Individual Feature Signals",
        ]
    )
    if pruning_recommendations.empty and feature_ranking.empty:
        sections.append(
            "_feature_impact_ranking.csv is empty. Run ablation mode `drop_feature` for individual P0 feature diagnostics._"
        )
    else:
        if not pruning_recommendations.empty:
            sections.append("### Pruning Recommendations")
            sections.append(markdown_table(pruning_recommendations.head(30)))
            sections.append("")
        if not feature_ranking.empty:
            sections.append("### Feature Impact Ranking")
            sections.append(markdown_table(feature_ranking.head(20)))

    sections.extend(["", "## Warnings"])
    if warnings_df.empty:
        sections.append("_No warnings were recorded._")
    else:
        sections.append(markdown_table(warnings_df, max_rows=30))

    sections.extend(
        [
            "",
            "## What to Test Next",
            "- Re-run ablation on more symbols and longer periods.",
            "- Use walk-forward validation before retaining noisy factor groups.",
            "- Test reduced feature sets against the full feature set by model type.",
            "- Treat any group with small experiment count as provisional.",
            "- Never use future, label, target, symbol, or date columns as model inputs.",
            "",
            "## Research Warnings",
            (
                "Ablation decisions can overfit one sample, one market regime, or one "
                "model type. Positive ML deltas do not guarantee profitable trading "
                "after costs, slippage, execution timing, and market regime changes."
            ),
            "",
        ]
    )

    return {
        "decision_summary": decision_summary,
        "strongest_groups": strongest,
        "weakest_groups": weakest,
        "feature_pruning_recommendations": pruning_recommendations,
        "markdown_report": "\n".join(sections),
        "loaded_outputs": outputs,
    }


def write_factor_decision_report(
    input_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    result = generate_factor_decision_report(input_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result["markdown_report"], encoding="utf-8")
    result["report_path"] = str(path)
    return result
