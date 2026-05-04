import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .candidate_expanded_validation import parse_symbols
    from .dataset_splitter import load_factor_dataset, normalize_date_column, validate_required_columns
    from .factor_pruning_experiment import (
        DEFAULT_PRUNING_MODES,
        build_feature_sets,
        identify_safe_feature_columns,
        load_pruning_recommendations,
    )
except ImportError:
    from candidate_expanded_validation import parse_symbols
    from dataset_splitter import load_factor_dataset, normalize_date_column, validate_required_columns
    from factor_pruning_experiment import (
        DEFAULT_PRUNING_MODES,
        build_feature_sets,
        identify_safe_feature_columns,
        load_pruning_recommendations,
    )


STRONG_CANDIDATE_MODES = ["drop_reduce_weight", "keep_core_and_observe"]


def _format_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _factor_path(factor_dir: str | Path, symbol: str) -> Path:
    return Path(factor_dir) / f"factors_{_format_symbol(symbol)}.csv"


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


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _mode_feature_sets(selected_df: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    feature_sets = {}
    for (symbol, mode), group in selected_df.groupby(["symbol", "pruning_mode"], dropna=False):
        feature_sets[(symbol, mode)] = set(group["feature"].dropna().astype(str))
    return feature_sets


def build_selected_features(
    factor_dir: str | Path,
    symbols: list[str],
    recommendations_path: str | Path,
    target_col: str,
    pruning_modes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recommendations_df = load_pruning_recommendations(recommendations_path)
    rows = []
    warnings = []
    for symbol in [_format_symbol(symbol) for symbol in symbols]:
        factor_csv = _factor_path(factor_dir, symbol)
        if not factor_csv.exists():
            warnings.append(
                {
                    "symbol": symbol,
                    "warning_type": "missing_factor_csv",
                    "message": f"Factor CSV not found: {factor_csv}",
                }
            )
            continue
        df = normalize_date_column(load_factor_dataset(factor_csv))
        validate_required_columns(df, target_col)
        feature_columns = identify_safe_feature_columns(df, target_col)
        feature_sets = build_feature_sets(feature_columns, recommendations_df, pruning_modes)
        for mode in pruning_modes:
            features = feature_sets.get(mode, [])
            if not features:
                warnings.append(
                    {
                        "symbol": symbol,
                        "pruning_mode": mode,
                        "warning_type": "empty_feature_set",
                        "message": "No features selected for this pruning mode.",
                    }
                )
            for position, feature in enumerate(features, start=1):
                rows.append(
                    {
                        "symbol": symbol,
                        "pruning_mode": mode,
                        "feature": feature,
                        "feature_position": position,
                        "feature_count": len(features),
                    }
                )
    selected_df = pd.DataFrame(rows)
    warnings_df = pd.DataFrame(warnings)
    if "symbol" in selected_df:
        selected_df["symbol"] = selected_df["symbol"].map(_format_symbol)
    if "symbol" in warnings_df:
        warnings_df["symbol"] = warnings_df["symbol"].map(_format_symbol)
    return selected_df, warnings_df


def build_overlap_matrix(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()
    feature_sets = _mode_feature_sets(selected_df)
    rows = []
    for symbol in sorted({key[0] for key in feature_sets}):
        modes = sorted(mode for current_symbol, mode in feature_sets if current_symbol == symbol)
        for left_mode in modes:
            for right_mode in modes:
                left = feature_sets[(symbol, left_mode)]
                right = feature_sets[(symbol, right_mode)]
                rows.append(
                    {
                        "symbol": _format_symbol(symbol),
                        "left_mode": left_mode,
                        "right_mode": right_mode,
                        "left_feature_count": len(left),
                        "right_feature_count": len(right),
                        "shared_feature_count": len(left & right),
                        "union_feature_count": len(left | right),
                        "jaccard_similarity": _jaccard(left, right),
                        "identical": left == right,
                    }
                )
    return pd.DataFrame(rows)


def build_equivalence_summary(selected_df: pd.DataFrame, overlap_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()
    rows = []
    feature_sets = _mode_feature_sets(selected_df)
    symbols = sorted({key[0] for key in feature_sets})
    pairs = [
        ("drop_reduce_weight", "keep_core_and_observe"),
        ("full", "drop_reduce_weight"),
        ("full", "keep_core_only"),
        ("keep_core_only", "keep_core_and_observe"),
    ]
    for left_mode, right_mode in pairs:
        pair_rows = overlap_df[
            (overlap_df["left_mode"] == left_mode)
            & (overlap_df["right_mode"] == right_mode)
        ]
        if pair_rows.empty:
            continue
        avg_similarity = float(pair_rows["jaccard_similarity"].mean())
        identical_rate = float(pair_rows["identical"].mean())
        if identical_rate == 1.0:
            interpretation = "identical"
        elif avg_similarity >= 0.90:
            interpretation = "nearly identical"
        elif avg_similarity >= 0.50:
            interpretation = "partially overlapping"
        else:
            interpretation = "meaningfully different"
        rows.append(
            {
                "left_mode": left_mode,
                "right_mode": right_mode,
                "symbol_count": len(symbols),
                "avg_jaccard_similarity": avg_similarity,
                "identical_symbol_rate": identical_rate,
                "interpretation": interpretation,
                "comparison_meaningful": interpretation
                not in {"identical", "nearly identical"},
            }
        )
    return pd.DataFrame(rows)


def build_feature_frequency(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()
    rows = []
    symbol_count = selected_df["symbol"].nunique()
    for (mode, feature), group in selected_df.groupby(["pruning_mode", "feature"], dropna=False):
        rows.append(
            {
                "pruning_mode": mode,
                "feature": feature,
                "symbol_count": int(group["symbol"].nunique()),
                "symbol_rate": float(group["symbol"].nunique() / symbol_count)
                if symbol_count
                else 0.0,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["pruning_mode", "symbol_rate", "feature"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def _shared_strong_features(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()
    mode_sets = []
    for mode in STRONG_CANDIDATE_MODES:
        mode_sets.append(set(selected_df[selected_df["pruning_mode"] == mode]["feature"]))
    shared = set.intersection(*mode_sets) if mode_sets else set()
    return pd.DataFrame({"feature": sorted(shared)})


def _unique_mode_features(selected_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()
    mode_features = set(selected_df[selected_df["pruning_mode"] == mode]["feature"])
    other_features = set(selected_df[selected_df["pruning_mode"] != mode]["feature"])
    return pd.DataFrame({"feature": sorted(mode_features - other_features)})


def generate_markdown_report(
    selected_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    equivalence_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> str:
    strong_pair = equivalence_df[
        (equivalence_df["left_mode"] == "drop_reduce_weight")
        & (equivalence_df["right_mode"] == "keep_core_and_observe")
    ]
    if strong_pair.empty:
        strong_text = "drop_reduce_weight and keep_core_and_observe could not be compared."
    else:
        row = strong_pair.iloc[0]
        strong_text = (
            "drop_reduce_weight and keep_core_and_observe are "
            f"{row['interpretation']} with average Jaccard similarity "
            f"{row['avg_jaccard_similarity']:.4f}."
        )
    redundant = (
        not strong_pair.empty
        and strong_pair.iloc[0]["interpretation"] in {"identical", "nearly identical"}
    )
    sections = [
        "# Candidate Equivalence Audit + Feature Set Export",
        "",
        "## Overall Conclusion",
        (
            "This is an audit/reporting step only. It is not a trading "
            "recommendation and does not change model training or backtest logic."
        ),
        strong_text,
        (
            "Current candidate comparison may be redundant if the strong candidates "
            "are identical or nearly identical."
            if redundant
            else "Current candidate comparison appears meaningful because selected feature sets differ."
        ),
        "",
        "## Feature Set Equivalence Summary",
        _markdown_table(equivalence_df),
        "",
        "## Strong Candidate Shared Features",
        _markdown_table(_shared_strong_features(selected_df)),
        "",
        "## Features Unique to Full",
        _markdown_table(_unique_mode_features(selected_df, "full")),
        "",
        "## Features Unique to keep_core_only",
        _markdown_table(_unique_mode_features(selected_df, "keep_core_only")),
        "",
        "## Feature Frequency by Mode",
        _markdown_table(frequency_df),
        "",
        "## Feature Set Overlap Matrix",
        _markdown_table(overlap_df),
        "",
        "## Selected Features by Symbol and Mode",
        _markdown_table(selected_df),
        "",
        "## Warnings",
        _markdown_table(warnings_df),
        "",
    ]
    return "\n".join(sections)


def build_candidate_equivalence_audit(
    factor_dir: str | Path,
    symbols: list[str],
    recommendations_path: str | Path,
    target_col: str = "label_up_5d",
    pruning_modes: list[str] | None = None,
) -> dict[str, Any]:
    modes = pruning_modes or DEFAULT_PRUNING_MODES.copy()
    selected_df, warnings_df = build_selected_features(
        factor_dir=factor_dir,
        symbols=symbols,
        recommendations_path=recommendations_path,
        target_col=target_col,
        pruning_modes=modes,
    )
    overlap_df = build_overlap_matrix(selected_df)
    equivalence_df = build_equivalence_summary(selected_df, overlap_df)
    frequency_df = build_feature_frequency(selected_df)
    report = generate_markdown_report(
        selected_df,
        overlap_df,
        equivalence_df,
        frequency_df,
        warnings_df,
    )
    return {
        "selected_features_by_symbol_mode": selected_df,
        "feature_set_overlap_matrix": overlap_df,
        "feature_set_equivalence_summary": equivalence_df,
        "feature_frequency_by_mode": frequency_df,
        "warnings": warnings_df,
        "candidate_equivalence_report": report,
    }


def save_candidate_equivalence_audit(
    factor_dir: str | Path,
    symbols: list[str],
    recommendations_path: str | Path,
    output_dir: str | Path,
    target_col: str = "label_up_5d",
    pruning_modes: list[str] | None = None,
) -> dict[str, Any]:
    result = build_candidate_equivalence_audit(
        factor_dir=factor_dir,
        symbols=symbols,
        recommendations_path=recommendations_path,
        target_col=target_col,
        pruning_modes=pruning_modes,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected_features_by_symbol_mode": output_path
        / "selected_features_by_symbol_mode.csv",
        "feature_set_overlap_matrix": output_path / "feature_set_overlap_matrix.csv",
        "feature_set_equivalence_summary": output_path
        / "feature_set_equivalence_summary.csv",
        "feature_frequency_by_mode": output_path / "feature_frequency_by_mode.csv",
        "candidate_equivalence_report": output_path / "candidate_equivalence_report.md",
        "warnings": output_path / "warnings.csv",
        "run_config": output_path / "run_config.json",
    }
    for key, path in paths.items():
        if key in {"candidate_equivalence_report", "run_config"}:
            continue
        result.get(key, pd.DataFrame()).to_csv(path, index=False)
    paths["candidate_equivalence_report"].write_text(
        result["candidate_equivalence_report"],
        encoding="utf-8",
    )
    run_config = {
        "factor_dir": str(factor_dir),
        "symbols": [_format_symbol(symbol) for symbol in symbols],
        "recommendations_path": str(recommendations_path),
        "output_dir": str(output_path),
        "target_col": target_col,
        "pruning_modes": pruning_modes or DEFAULT_PRUNING_MODES.copy(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["run_config"] = run_config
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result
