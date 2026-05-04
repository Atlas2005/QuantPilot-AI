import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .candidate_mode_normalization import (
        CANONICAL_REDUCED_MODE,
        add_canonical_mode_columns,
    )
except ImportError:
    from candidate_mode_normalization import (
        CANONICAL_REDUCED_MODE,
        add_canonical_mode_columns,
    )


CANONICAL_MODES = [CANONICAL_REDUCED_MODE, "full", "keep_core_only"]


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


def _ensure_canonical(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = df.copy()
    if "canonical_mode" in result:
        return result
    if "pruning_mode" in result:
        return add_canonical_mode_columns(result)
    if "best_pruning_mode" in result:
        result["legacy_pruning_mode"] = result["best_pruning_mode"]
        result["canonical_mode"] = add_canonical_mode_columns(
            pd.DataFrame({"pruning_mode": result["best_pruning_mode"]})
        )["canonical_mode"].values
    return result


def load_revalidation_inputs(
    expanded_validation_dir: str | Path,
    stress_dir: str | Path,
    threshold_decision_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    expanded = Path(expanded_validation_dir)
    stress = Path(stress_dir)
    decision = Path(threshold_decision_dir)
    inputs = {
        "validation_summary": _read_csv(expanded / "candidate_validation_summary.csv"),
        "validation_results": _read_csv(
            expanded / "candidate_validation_results.csv",
            dtype={"symbol": str},
        ),
        "validation_warnings": _read_csv(
            expanded / "candidate_validation_warnings.csv",
            dtype={"symbol": str},
        ),
        "stress_summary": _read_csv(stress / "candidate_stress_summary.csv"),
        "stress_results": _read_csv(
            stress / "candidate_stress_results.csv",
            dtype={"symbol": str},
        ),
        "stress_warnings": _read_csv(stress / "stress_warnings.csv", dtype={"symbol": str}),
        "threshold_decision_summary": _read_csv(
            decision / "threshold_decision_summary.csv"
        ),
        "threshold_rejected": _read_csv(
            decision / "rejected_or_low_confidence_configs.csv",
            dtype={"symbol": str},
        ),
    }
    for key, df in inputs.items():
        if "symbol" in df:
            df["symbol"] = df["symbol"].map(_format_symbol)
        inputs[key] = _ensure_canonical(df)
    return inputs


def build_revalidation_summary(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    validation_summary = inputs["validation_summary"]
    stress_summary = inputs["stress_summary"]
    for mode in CANONICAL_MODES:
        validation_rows = validation_summary[
            validation_summary.get("canonical_mode", pd.Series(dtype=str)) == mode
        ]
        stress_rows = stress_summary[
            stress_summary.get("canonical_mode", pd.Series(dtype=str)) == mode
        ]
        validation_decision = (
            ",".join(validation_rows.get("final_decision", pd.Series(dtype=str)).dropna().astype(str).unique())
            if not validation_rows.empty
            else "not_tested"
        )
        stress_decision = (
            ",".join(stress_rows.get("final_decision", pd.Series(dtype=str)).dropna().astype(str).unique())
            if not stress_rows.empty
            else "not_tested"
        )
        if mode == CANONICAL_REDUCED_MODE:
            role = "primary_research_candidate"
            final_decision = "research_only_not_trading_ready"
            reason = "canonical_reduced_40 is the current primary research candidate, but stress validation still fails or is not pass."
        elif mode == "full":
            role = "baseline_only"
            final_decision = "baseline_only"
            reason = "full is retained as baseline only, not as the default candidate."
        else:
            role = "low_feature_challenger"
            final_decision = "low_confidence_challenger"
            reason = "keep_core_only is a low-feature challenger with low-trade-count or instability risk."
        rows.append(
            {
                "canonical_mode": mode,
                "role": role,
                "validation_decision": validation_decision,
                "stress_decision": stress_decision,
                "avg_validation_excess_pct": _numeric(
                    validation_rows,
                    "avg_strategy_vs_benchmark_pct",
                ).mean(),
                "avg_stress_excess_pct": _numeric(
                    stress_rows,
                    "avg_strategy_vs_benchmark_pct",
                ).mean(),
                "stress_beat_benchmark_rate": _numeric(
                    stress_rows,
                    "beat_benchmark_rate",
                ).mean(),
                "stress_sufficient_trade_rate": _numeric(
                    stress_rows,
                    "sufficient_trade_rate",
                ).mean(),
                "final_research_decision": final_decision,
                "decision_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_risk_flags(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for source_name, key in [
        ("expanded_validation", "validation_warnings"),
        ("stress_validation", "stress_warnings"),
        ("threshold_decision", "threshold_rejected"),
    ]:
        df = inputs[key]
        if df.empty:
            continue
        flagged = df.copy()
        flagged["source"] = source_name
        text = pd.Series("", index=flagged.index, dtype="object")
        for column in ["warning_type", "message", "reason", "warning"]:
            if column in flagged:
                text = text.str.cat(flagged[column].fillna("").astype(str), sep=" ")
        flagged["risk_category"] = "research_warning"
        flagged.loc[
            text.str.contains("low_trade|low-confidence", case=False, regex=True),
            "risk_category",
        ] = "low_trade_or_low_confidence"
        flagged.loc[
            text.str.contains("underperformed|benchmark", case=False, regex=True),
            "risk_category",
        ] = "benchmark_underperformance"
        frames.append(flagged)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    if "symbol" in result:
        result["symbol"] = result["symbol"].map(_format_symbol)
    preferred = [
        "source",
        "risk_category",
        "canonical_mode",
        "legacy_pruning_mode",
        "symbol",
        "warning_type",
        "message",
        "reason",
    ]
    columns = [column for column in preferred if column in result.columns] + [
        column for column in result.columns if column not in preferred
    ]
    return result[columns]


def generate_markdown_report(
    summary: pd.DataFrame,
    risk_flags: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
) -> str:
    failed_symbols = pd.DataFrame()
    stress_results = inputs["stress_results"]
    if not stress_results.empty and "strategy_vs_benchmark_pct" in stress_results:
        failed_symbols = stress_results[
            _numeric(stress_results, "strategy_vs_benchmark_pct") <= 0
        ].copy()
    sections = [
        "# Canonical Candidate Revalidation Report",
        "",
        "## Executive Summary",
        (
            "canonical_reduced_40 is the current primary research candidate. "
            "canonical_reduced_40 is not trading-ready because stress validation still fails."
        ),
        "full is baseline only.",
        (
            "keep_core_only is a low-feature challenger with low-trade-count / "
            "instability risk."
        ),
        "This is a reporting/decision-control step only, not financial advice.",
        "",
        "## Canonical Candidate Comparison",
        _markdown_table(summary),
        "",
        "## Risk Flags",
        _markdown_table(risk_flags),
        "",
        "## Stress Failures by Symbol",
        _markdown_table(failed_symbols),
        "",
        "## Expanded Validation Summary Input",
        _markdown_table(inputs["validation_summary"]),
        "",
        "## Stress Summary Input",
        _markdown_table(inputs["stress_summary"]),
        "",
        "## Threshold Decision Summary Input",
        _markdown_table(inputs["threshold_decision_summary"]),
        "",
    ]
    return "\n".join(sections)


def build_canonical_candidate_revalidation_report(
    expanded_validation_dir: str | Path,
    stress_dir: str | Path,
    threshold_decision_dir: str | Path,
) -> dict[str, Any]:
    inputs = load_revalidation_inputs(
        expanded_validation_dir,
        stress_dir,
        threshold_decision_dir,
    )
    summary = build_revalidation_summary(inputs)
    risk_flags = build_risk_flags(inputs)
    report = generate_markdown_report(summary, risk_flags, inputs)
    return {
        "canonical_candidate_revalidation_summary": summary,
        "candidate_risk_flags": risk_flags,
        "canonical_candidate_revalidation_report": report,
    }


def save_canonical_candidate_revalidation_report(
    expanded_validation_dir: str | Path,
    stress_dir: str | Path,
    threshold_decision_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    result = build_canonical_candidate_revalidation_report(
        expanded_validation_dir,
        stress_dir,
        threshold_decision_dir,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_path / "canonical_candidate_revalidation_report.md",
        "summary": output_path / "canonical_candidate_revalidation_summary.csv",
        "risk_flags": output_path / "candidate_risk_flags.csv",
        "run_config": output_path / "run_config.json",
    }
    result["canonical_candidate_revalidation_summary"].to_csv(
        paths["summary"],
        index=False,
    )
    result["candidate_risk_flags"].to_csv(paths["risk_flags"], index=False)
    paths["report"].write_text(
        result["canonical_candidate_revalidation_report"],
        encoding="utf-8",
    )
    run_config = {
        "expanded_validation_dir": str(expanded_validation_dir),
        "stress_dir": str(stress_dir),
        "threshold_decision_dir": str(threshold_decision_dir),
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
