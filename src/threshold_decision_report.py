import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .candidate_mode_normalization import add_canonical_mode_columns
except ImportError:
    from candidate_mode_normalization import add_canonical_mode_columns


REPORT_FILES = {
    "mode_summary": "threshold_mode_summary.csv",
    "model_summary": "threshold_model_summary.csv",
    "mode_model_summary": "threshold_mode_model_summary.csv",
    "per_symbol_best": "per_symbol_best_thresholds.csv",
    "walk_forward_summary": "walk_forward_summary.csv",
    "warnings": "warnings.csv",
}


def _read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _format_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _format_value(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
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


def load_threshold_summary_outputs(summary_dir: str | Path) -> dict[str, Any]:
    base = Path(summary_dir)
    outputs = {"summary_dir": base}
    for key, filename in REPORT_FILES.items():
        dtype = {"symbol": str} if key in {"per_symbol_best", "warnings"} else None
        outputs[key] = _read_csv(base / filename, dtype=dtype)
        if "symbol" in outputs[key]:
            outputs[key]["symbol"] = outputs[key]["symbol"].map(_format_symbol)
    return outputs


def _rank_candidates(df: pd.DataFrame, disallow_full: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ranked = df.copy()
    mode_column = "canonical_mode" if "canonical_mode" in ranked else "pruning_mode"
    if disallow_full and mode_column in ranked:
        non_full = ranked[ranked[mode_column] != "full"].copy()
        if not non_full.empty:
            ranked = non_full
    sort_columns = [
        column
        for column in [
            "stability_score",
            "avg_total_return_pct",
            "avg_strategy_vs_benchmark_pct",
            "sufficient_trade_rate",
            "avg_trade_count",
        ]
        if column in ranked
    ]
    if not sort_columns:
        return ranked.head(1)
    return ranked.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
        na_position="last",
    )


def _canonicalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "pruning_mode" not in df:
        return df
    with_modes = add_canonical_mode_columns(df)
    group_columns = ["canonical_mode"]
    if "model_type" in with_modes:
        group_columns.append("model_type")
    numeric_columns = [
        column
        for column in with_modes.columns
        if column not in group_columns
        and column not in {"pruning_mode", "legacy_pruning_mode"}
        and pd.api.types.is_numeric_dtype(with_modes[column])
    ]
    rows = []
    for keys, group in with_modes.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row["legacy_pruning_modes"] = ",".join(
            sorted(group["legacy_pruning_mode"].dropna().astype(str).unique())
        )
        for column in numeric_columns:
            row[column] = pd.to_numeric(group[column], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def select_pruning_mode_candidate(mode_summary: pd.DataFrame) -> dict[str, Any]:
    if mode_summary.empty:
        return {
            "recommended_pruning_mode": "n/a",
            "recommended_pruning_mode_rationale": "No pruning-mode summary was available.",
        }
    ranked = _rank_candidates(mode_summary, disallow_full=True)
    candidate = ranked.iloc[0]
    return {
        "recommended_pruning_mode": candidate.get(
            "canonical_mode",
            candidate.get("pruning_mode", "n/a"),
        ),
        "recommended_legacy_pruning_modes": candidate.get("legacy_pruning_modes", "n/a"),
        "recommended_pruning_mode_rationale": (
            "recommended research candidate based on non-full pruning-mode "
            "stability, average total return, and trade-count diagnostics."
        ),
        "recommended_pruning_mode_avg_total_return_pct": candidate.get(
            "avg_total_return_pct"
        ),
        "recommended_pruning_mode_avg_strategy_vs_benchmark_pct": candidate.get(
            "avg_strategy_vs_benchmark_pct"
        ),
        "recommended_pruning_mode_stability_score": candidate.get("stability_score"),
    }


def select_model_candidate(model_summary: pd.DataFrame) -> dict[str, Any]:
    if model_summary.empty:
        return {
            "recommended_model_type": "n/a",
            "recommended_model_rationale": "No model summary was available.",
        }
    candidate = _rank_candidates(model_summary).iloc[0]
    return {
        "recommended_model_type": candidate.get("model_type", "n/a"),
        "recommended_model_rationale": (
            "recommended research candidate based on aggregate return, benchmark "
            "comparison, stability, and trade-count diagnostics."
        ),
        "recommended_model_avg_trade_count": candidate.get("avg_trade_count"),
        "recommended_model_avg_strategy_vs_benchmark_pct": candidate.get(
            "avg_strategy_vs_benchmark_pct"
        ),
        "recommended_model_stability_score": candidate.get("stability_score"),
    }


def build_threshold_range_summary(per_symbol_best: pd.DataFrame) -> dict[str, Any]:
    if per_symbol_best.empty:
        return {
            "candidate_buy_threshold_range": "n/a",
            "candidate_sell_threshold_range": "n/a",
            "candidate_threshold_symbol_count": 0,
        }
    best = per_symbol_best.copy()
    confidence = best.get("selection_confidence", pd.Series("", index=best.index))
    valid = best[
        ~confidence.fillna("").astype(str).str.contains("low_confidence", case=False)
    ].copy()
    if valid.empty:
        return {
            "candidate_buy_threshold_range": "n/a",
            "candidate_sell_threshold_range": "n/a",
            "candidate_threshold_symbol_count": 0,
            "candidate_threshold_note": (
                "No non-low-confidence per-symbol threshold rows were available."
            ),
        }
    buy = _numeric(valid, "best_buy_threshold").dropna()
    sell = _numeric(valid, "best_sell_threshold").dropna()
    return {
        "candidate_buy_threshold_range": "n/a"
        if buy.empty
        else f"{buy.min():.2f}-{buy.max():.2f}",
        "candidate_sell_threshold_range": "n/a"
        if sell.empty
        else f"{sell.min():.2f}-{sell.max():.2f}",
        "candidate_threshold_symbol_count": int(valid["symbol"].nunique())
        if "symbol" in valid
        else len(valid),
        "candidate_threshold_note": (
            "Derived only from per-symbol best rows that are not marked "
            "low-confidence."
        ),
    }


def select_walk_forward_candidate(walk_forward_summary: pd.DataFrame) -> pd.DataFrame:
    if walk_forward_summary.empty:
        return pd.DataFrame()
    sort_columns = [
        column
        for column in [
            "stability_score",
            "sufficient_trade_rate",
            "beat_benchmark_rate",
            "avg_strategy_vs_benchmark_pct",
        ]
        if column in walk_forward_summary
    ]
    if not sort_columns:
        return walk_forward_summary.head(1).copy()
    return walk_forward_summary.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
        na_position="last",
    ).head(1)


def build_rejected_or_low_confidence_configs(
    mode_summary: pd.DataFrame,
    mode_model_summary: pd.DataFrame,
    per_symbol_best: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not mode_summary.empty and "canonical_mode" in mode_summary:
        rejected_modes = mode_summary[mode_summary["canonical_mode"] == "full"].copy()
        if not rejected_modes.empty:
            rejected_modes["case_type"] = "rejected_default_full_feature_set"
            rejected_modes["reason"] = (
                "Full feature set is the weakest default candidate in threshold diagnostics."
            )
            frames.append(rejected_modes)

    if not mode_model_summary.empty and "avg_strategy_vs_benchmark_pct" in mode_model_summary:
        underperform = mode_model_summary[
            _numeric(mode_model_summary, "avg_strategy_vs_benchmark_pct") < 0
        ].copy()
        if not underperform.empty:
            underperform["case_type"] = "benchmark_underperformance"
            underperform["reason"] = "Configuration underperforms benchmark on average."
            frames.append(underperform)

    if not per_symbol_best.empty and "selection_confidence" in per_symbol_best:
        low_confidence = per_symbol_best[
            per_symbol_best["selection_confidence"]
            .fillna("")
            .astype(str)
            .str.contains("low_confidence", case=False)
        ].copy()
        if not low_confidence.empty:
            low_confidence["case_type"] = "low-confidence_best_threshold"
            low_confidence["reason"] = (
                "High return or best row is low-confidence because trade count is too low."
            )
            frames.append(low_confidence)

    if not warnings_df.empty:
        warning_text = pd.Series("", index=warnings_df.index, dtype="object")
        for column in ["warning_type", "message", "warning"]:
            if column in warnings_df:
                warning_text = warning_text.str.cat(
                    warnings_df[column].fillna("").astype(str),
                    sep=" ",
                )
        warning_cases = warnings_df[
            warning_text.str.contains(
                "low_trade_count|underperformed_benchmark",
                case=False,
                regex=True,
            )
        ].copy()
        if not warning_cases.empty:
            warning_cases["case_type"] = "warning_flag"
            warning_cases["reason"] = "Warning row from threshold summary diagnostics."
            frames.append(warning_cases)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    if "symbol" in result:
        result["symbol"] = result["symbol"].map(_format_symbol)
    preferred = [
        "case_type",
        "reason",
        "symbol",
        "pruning_mode",
        "model_type",
        "best_buy_threshold",
        "best_sell_threshold",
        "best_trade_count",
        "selection_confidence",
        "avg_total_return_pct",
        "avg_strategy_vs_benchmark_pct",
        "stability_score",
        "warning_type",
        "message",
    ]
    columns = [column for column in preferred if column in result.columns] + [
        column for column in result.columns if column not in preferred
    ]
    return result[columns]


def _overall_underperformance_text(mode_summary: pd.DataFrame) -> str:
    if mode_summary.empty or "avg_strategy_vs_benchmark_pct" not in mode_summary:
        return "Benchmark comparison was not available."
    avg_excess = _numeric(mode_summary, "avg_strategy_vs_benchmark_pct").mean()
    if pd.isna(avg_excess):
        return "Benchmark comparison was not available."
    if avg_excess < 0:
        return (
            f"Average strategy returns still underperforms benchmark on average "
            f"by {avg_excess:.4f} percentage points across pruning modes."
        )
    return (
        "Average strategy returns do not show enough evidence to describe any "
        "configuration as financially reliable."
    )


def build_threshold_decision_report(summary_dir: str | Path) -> dict[str, Any]:
    outputs = load_threshold_summary_outputs(summary_dir)
    mode_summary = outputs["mode_summary"]
    model_summary = outputs["model_summary"]
    mode_model_summary = outputs["mode_model_summary"]
    per_symbol_best = outputs["per_symbol_best"]
    walk_forward_summary = outputs["walk_forward_summary"]
    warnings_df = outputs["warnings"]
    mode_summary = _canonicalize_summary(mode_summary)
    mode_model_summary = _canonicalize_summary(mode_model_summary)
    if not per_symbol_best.empty and "best_pruning_mode" in per_symbol_best:
        per_symbol_best = per_symbol_best.copy()
        per_symbol_best["legacy_pruning_mode"] = per_symbol_best["best_pruning_mode"]
        per_symbol_best["canonical_mode"] = per_symbol_best["best_pruning_mode"].map(
            lambda value: add_canonical_mode_columns(
                pd.DataFrame({"pruning_mode": [value]})
            )["canonical_mode"].iloc[0]
        )

    pruning_decision = select_pruning_mode_candidate(mode_summary)
    model_decision = select_model_candidate(model_summary)
    threshold_ranges = build_threshold_range_summary(per_symbol_best)
    walk_forward_candidate = select_walk_forward_candidate(walk_forward_summary)
    rejected = build_rejected_or_low_confidence_configs(
        mode_summary,
        mode_model_summary,
        per_symbol_best,
        warnings_df,
    )

    decision_summary = pd.DataFrame(
        [
            {"decision_item": "pruning_mode", **pruning_decision},
            {"decision_item": "model_type", **model_decision},
            {"decision_item": "threshold_ranges", **threshold_ranges},
            {
                "decision_item": "walk_forward_candidate_available",
                "value": not walk_forward_candidate.empty,
                "rationale": (
                    "Walk-forward diagnostics are caveats, not proof of future reliability."
                ),
            },
        ]
    )

    report = generate_markdown_report(
        summary_dir=summary_dir,
        mode_summary=mode_summary,
        model_summary=model_summary,
        mode_model_summary=mode_model_summary,
        per_symbol_best=per_symbol_best,
        walk_forward_summary=walk_forward_summary,
        walk_forward_candidate=walk_forward_candidate,
        warnings_df=warnings_df,
        rejected=rejected,
        decision_summary=decision_summary,
    )
    return {
        "decision_summary": decision_summary,
        "rejected_or_low_confidence_configs": rejected,
        "markdown_report": report,
        "loaded_outputs": outputs,
    }


def generate_markdown_report(
    summary_dir: str | Path,
    mode_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    mode_model_summary: pd.DataFrame,
    per_symbol_best: pd.DataFrame,
    walk_forward_summary: pd.DataFrame,
    walk_forward_candidate: pd.DataFrame,
    warnings_df: pd.DataFrame,
    rejected: pd.DataFrame,
    decision_summary: pd.DataFrame,
) -> str:
    sections = [
        "# Reduced Feature Threshold Decision Report",
        "",
        f"Source summary directory: `{Path(summary_dir)}`",
        "",
        "## Overall Decision",
        (
            "This is a conservative educational/research decision report. It is "
            "not trading-ready, not financially reliable, and not financial advice."
        ),
        _overall_underperformance_text(mode_summary),
        (
            "The recommended research candidate should be treated as the next "
            "diagnostic configuration to test, not as a production trading rule."
        ),
        "",
        "## Decision Summary",
        markdown_table(decision_summary),
        "",
        "## Recommended Pruning Mode Candidate",
        (
            "Prefer the non-full pruning candidate with the best combined stability, "
            "return, and trade-count profile. Full feature set remains rejected as "
            "a default candidate when reduced modes are available."
        ),
        markdown_table(mode_summary),
        "",
        "## Recommended Model Candidate",
        (
            "The model candidate is selected from aggregate model diagnostics. A "
            "higher average trade count improves interpretability but does not make "
            "the result reliable."
        ),
        markdown_table(model_summary),
        "",
        "## Candidate Threshold Ranges",
        (
            "Threshold ranges are derived from non-low-confidence per-symbol best "
            "rows only. Low-trade winners are excluded from this recommendation."
        ),
        markdown_table(per_symbol_best),
        "",
        "## Recommended Walk-Forward Candidate",
        (
            "Walk-forward ranking prioritizes stability score, sufficient trade "
            "rate, benchmark beat rate, and average strategy-vs-benchmark return. "
            "These results are caveats and diagnostics, not validation of a "
            "trading-ready strategy."
        ),
        markdown_table(walk_forward_candidate),
        "",
        "## Walk-Forward Caveats",
        markdown_table(walk_forward_summary),
        "",
        "## Rejected or Low-Confidence Configurations",
        (
            "This section includes full-feature defaults, benchmark "
            "underperformance, low-confidence best rows, and warning rows. "
            "High-return but low-trade-count configurations remain low-confidence."
        ),
        markdown_table(rejected),
        "",
        "## Mode + Model Diagnostics",
        markdown_table(mode_model_summary),
        "",
        "## Warnings",
        markdown_table(warnings_df),
        "",
        "## Next Experiment Recommendation",
        (
            "Next, rerun the recommended research candidate across more symbols, "
            "longer periods, stricter walk-forward windows, and realistic cost "
            "settings. Do not add more feature complexity until the reduced "
            "candidate can beat the benchmark with sufficient trade counts across "
            "multiple symbols."
        ),
        "",
    ]
    return "\n".join(sections)


def save_threshold_decision_report(
    summary_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    result = build_threshold_decision_report(summary_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_path / "threshold_decision_report.md",
        "decision_summary": output_path / "threshold_decision_summary.csv",
        "rejected_or_low_confidence_configs": output_path
        / "rejected_or_low_confidence_configs.csv",
        "run_config": output_path / "run_config.json",
    }
    result["decision_summary"].to_csv(paths["decision_summary"], index=False)
    result["rejected_or_low_confidence_configs"].to_csv(
        paths["rejected_or_low_confidence_configs"],
        index=False,
    )
    paths["report"].write_text(result["markdown_report"], encoding="utf-8")
    run_config = {
        "summary_dir": str(summary_dir),
        "output_dir": str(output_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "report_type": "threshold_decision_report",
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["run_config"] = run_config
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result
