import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_REDUCED_MODE = "canonical_reduced_40"
MODE_ALIASES = {
    "full": "full",
    "drop_reduce_weight": CANONICAL_REDUCED_MODE,
    "keep_core_and_observe": CANONICAL_REDUCED_MODE,
    "keep_core_only": "keep_core_only",
}


def normalize_candidate_mode(mode: str) -> str:
    return MODE_ALIASES.get(str(mode), str(mode))


def add_canonical_mode_columns(
    df: pd.DataFrame,
    mode_column: str = "pruning_mode",
) -> pd.DataFrame:
    if df.empty or mode_column not in df:
        return df.copy()
    result = df.copy()
    if "legacy_pruning_mode" not in result:
        result["legacy_pruning_mode"] = result[mode_column]
    result["canonical_mode"] = result[mode_column].map(normalize_candidate_mode)
    return result


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


def load_equivalence_outputs(equivalence_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(equivalence_dir)
    return {
        "selected_features": _read_csv(
            base / "selected_features_by_symbol_mode.csv",
            dtype={"symbol": str},
        ),
        "overlap": _read_csv(base / "feature_set_overlap_matrix.csv", dtype={"symbol": str}),
        "equivalence": _read_csv(base / "feature_set_equivalence_summary.csv"),
    }


def build_canonical_mode_summary(equivalence_dir: str | Path) -> dict[str, Any]:
    outputs = load_equivalence_outputs(equivalence_dir)
    selected = outputs["selected_features"]
    overlap = outputs["overlap"]
    equivalence = outputs["equivalence"]
    if "symbol" in selected:
        selected["symbol"] = selected["symbol"].map(_format_symbol)
    if "symbol" in overlap:
        overlap["symbol"] = overlap["symbol"].map(_format_symbol)

    feature_counts = (
        selected.groupby(["pruning_mode", "symbol"], dropna=False)["feature"]
        .nunique()
        .reset_index(name="feature_count")
        if not selected.empty
        else pd.DataFrame(columns=["pruning_mode", "symbol", "feature_count"])
    )
    avg_counts = (
        feature_counts.groupby("pruning_mode", dropna=False)["feature_count"].mean()
        if not feature_counts.empty
        else pd.Series(dtype=float)
    )
    strong_pair = equivalence[
        (equivalence.get("left_mode", pd.Series(dtype=str)) == "drop_reduce_weight")
        & (
            equivalence.get("right_mode", pd.Series(dtype=str))
            == "keep_core_and_observe"
        )
    ]
    if strong_pair.empty:
        strong_interpretation = "unknown"
        comparison_meaningful = False
    else:
        strong_interpretation = str(strong_pair.iloc[0].get("interpretation", "unknown"))
        comparison_meaningful = bool(strong_pair.iloc[0].get("comparison_meaningful", False))

    rows = [
        {
            "canonical_mode": "full",
            "legacy_aliases": "full",
            "role": "baseline",
            "avg_feature_count": avg_counts.get("full"),
            "comparison_meaning": "Full feature baseline; keep as reference, not default candidate.",
        },
        {
            "canonical_mode": CANONICAL_REDUCED_MODE,
            "legacy_aliases": "drop_reduce_weight,keep_core_and_observe",
            "role": "primary_reduced_candidate",
            "avg_feature_count": avg_counts.get("drop_reduce_weight"),
            "comparison_meaning": (
                "Legacy reduced modes are equivalent aliases; avoid redundant candidate comparisons."
                if not comparison_meaningful
                else "Legacy reduced modes still differ in the audit; review before collapsing."
            ),
        },
        {
            "canonical_mode": "keep_core_only",
            "legacy_aliases": "keep_core_only",
            "role": "low_feature_challenger",
            "avg_feature_count": avg_counts.get("keep_core_only"),
            "comparison_meaning": "Distinct low-feature challenger with low-trade-count risk.",
        },
    ]
    summary = pd.DataFrame(rows)

    alias_rows = []
    for legacy_mode, canonical_mode in MODE_ALIASES.items():
        alias_rows.append(
            {
                "legacy_mode": legacy_mode,
                "canonical_mode": canonical_mode,
                "is_alias": legacy_mode != canonical_mode,
                "avg_feature_count": avg_counts.get(legacy_mode),
            }
        )
    alias_map = pd.DataFrame(alias_rows)
    report = generate_markdown_report(
        summary,
        alias_map,
        strong_interpretation,
        comparison_meaningful,
    )
    return {
        "canonical_mode_summary": summary,
        "legacy_alias_map": alias_map,
        "canonical_mode_report": report,
    }


def generate_markdown_report(
    summary: pd.DataFrame,
    alias_map: pd.DataFrame,
    strong_interpretation: str,
    comparison_meaningful: bool,
) -> str:
    sections = [
        "# Candidate Mode Simplification / Canonical Feature Set Cleanup",
        "",
        "## Overall Conclusion",
        (
            f"`drop_reduce_weight` and `keep_core_and_observe` are {strong_interpretation} "
            f"and should map to `{CANONICAL_REDUCED_MODE}` when future reports "
            "need candidate-level comparisons."
        ),
        (
            "Future candidate comparison reports can avoid redundant comparisons "
            "between legacy aliases by using the canonical mode mapping."
            if not comparison_meaningful
            else (
                "The equivalence audit did not fully support collapsing these modes; "
                "review the inputs before using the alias map to avoid redundant comparisons."
            )
        ),
        "This is a reporting cleanup only. It adds no new data sources, models, or factors.",
        "",
        "## Canonical Candidate Modes",
        _markdown_table(summary),
        "",
        "## Legacy Alias Map",
        _markdown_table(alias_map),
        "",
    ]
    return "\n".join(sections)


def save_canonical_mode_report(
    equivalence_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    result = build_canonical_mode_summary(equivalence_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "canonical_mode_summary": output_path / "canonical_mode_summary.csv",
        "legacy_alias_map": output_path / "legacy_alias_map.csv",
        "canonical_mode_report": output_path / "canonical_mode_report.md",
        "run_config": output_path / "run_config.json",
    }
    result["canonical_mode_summary"].to_csv(paths["canonical_mode_summary"], index=False)
    result["legacy_alias_map"].to_csv(paths["legacy_alias_map"], index=False)
    paths["canonical_mode_report"].write_text(
        result["canonical_mode_report"],
        encoding="utf-8",
    )
    run_config = {
        "equivalence_dir": str(equivalence_dir),
        "output_dir": str(output_path),
        "canonical_reduced_mode": CANONICAL_REDUCED_MODE,
        "mode_aliases": MODE_ALIASES,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["run_config"] = run_config
    result["output_files"] = {key: str(path) for key, path in paths.items()}
    return result
