import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .dataset_splitter import (
        check_for_leakage_columns,
        chronological_split,
        clean_factor_dataset,
        infer_feature_columns,
        load_factor_dataset,
        normalize_date_column,
        validate_required_columns,
    )
    from .factor_ablation import parse_csv_list, parse_model_types
    from .model_trainer import evaluate_model, train_baseline_model
except ImportError:
    from dataset_splitter import (
        check_for_leakage_columns,
        chronological_split,
        clean_factor_dataset,
        infer_feature_columns,
        load_factor_dataset,
        normalize_date_column,
        validate_required_columns,
    )
    from factor_ablation import parse_csv_list, parse_model_types
    from model_trainer import evaluate_model, train_baseline_model


DEFAULT_PRUNING_MODES = [
    "full",
    "drop_reduce_weight",
    "keep_core_only",
    "keep_core_and_observe",
]

RESULT_COLUMNS = [
    "pruning_mode",
    "model_type",
    "feature_count",
    "train_rows",
    "validation_rows",
    "test_rows",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "validation_roc_auc",
    "delta_test_roc_auc_vs_full",
    "delta_test_f1_vs_full",
    "warning",
]


def parse_pruning_modes(text: str | None) -> list[str]:
    if not text:
        return DEFAULT_PRUNING_MODES.copy()
    modes = parse_csv_list(text)
    invalid = [mode for mode in modes if mode not in DEFAULT_PRUNING_MODES]
    if invalid:
        raise ValueError(
            f"Unsupported pruning modes: {invalid}. Supported: {DEFAULT_PRUNING_MODES}"
        )
    return modes


def load_pruning_recommendations(path: str | Path) -> pd.DataFrame:
    recommendation_path = Path(path)
    if not recommendation_path.exists():
        raise FileNotFoundError(
            f"Feature pruning recommendation CSV does not exist: {recommendation_path}"
        )
    return pd.read_csv(recommendation_path)


def identify_safe_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    feature_columns = infer_feature_columns(df, target_col)
    leakage_checks = check_for_leakage_columns(feature_columns, target_col)
    if not leakage_checks["passed"]:
        raise ValueError(
            "Leakage-safe feature inference failed: "
            f"{leakage_checks['leakage_columns']}"
        )
    return feature_columns


def _features_by_recommendation(
    recommendations_df: pd.DataFrame,
    recommendation: str,
    available_features: set[str],
) -> list[str]:
    if recommendations_df.empty or "recommendation" not in recommendations_df:
        return []
    feature_column = (
        "removed_feature"
        if "removed_feature" in recommendations_df.columns
        else "feature"
    )
    if feature_column not in recommendations_df:
        return []
    rows = recommendations_df[recommendations_df["recommendation"] == recommendation]
    return [
        feature
        for feature in rows[feature_column].dropna().astype(str).tolist()
        if feature in available_features
    ]


def build_feature_sets(
    feature_columns: list[str],
    recommendations_df: pd.DataFrame,
    pruning_modes: list[str],
) -> dict[str, list[str]]:
    available = set(feature_columns)
    reduce_weight_features = set(
        _features_by_recommendation(recommendations_df, "reduce_weight", available)
    )
    keep_core_features = _features_by_recommendation(
        recommendations_df,
        "keep_core",
        available,
    )
    keep_observe_features = _features_by_recommendation(
        recommendations_df,
        "keep_observe",
        available,
    )

    feature_sets = {}
    for mode in pruning_modes:
        if mode == "full":
            feature_sets[mode] = feature_columns.copy()
        elif mode == "drop_reduce_weight":
            feature_sets[mode] = [
                feature for feature in feature_columns if feature not in reduce_weight_features
            ]
        elif mode == "keep_core_only":
            feature_sets[mode] = keep_core_features
        elif mode == "keep_core_and_observe":
            ordered = []
            for feature in keep_core_features + keep_observe_features:
                if feature in available and feature not in ordered:
                    ordered.append(feature)
            feature_sets[mode] = ordered
    return feature_sets


def _metric(metrics: dict[str, Any], key: str):
    return metrics.get(key)


def _safe_delta(value, baseline):
    if pd.isna(value) or pd.isna(baseline):
        return None
    return value - baseline


def run_pruning_mode(
    df: pd.DataFrame,
    feature_columns: list[str],
    pruning_mode: str,
    model_type: str,
    target_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_rows: int,
    baseline_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    warning = None
    if not feature_columns:
        return {
            "pruning_mode": pruning_mode,
            "model_type": model_type,
            "feature_count": 0,
            "train_rows": 0,
            "validation_rows": 0,
            "test_rows": 0,
            "test_accuracy": None,
            "test_precision": None,
            "test_recall": None,
            "test_f1": None,
            "test_roc_auc": None,
            "validation_roc_auc": None,
            "delta_test_roc_auc_vs_full": None,
            "delta_test_f1_vs_full": None,
            "warning": "No feature columns selected for this pruning mode.",
        }

    cleaned_df, _ = clean_factor_dataset(df, feature_columns, target_col)
    train_df, validation_df, test_df = chronological_split(
        cleaned_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        purge_rows=purge_rows,
        split_mode="global_date",
    )
    if train_df.empty or validation_df.empty or test_df.empty:
        warning = "One or more split datasets are empty."

    model, _ = train_baseline_model(
        train_df=train_df,
        feature_columns=feature_columns,
        target_col=target_col,
        model_name=model_type,
    )
    validation_metrics, _ = evaluate_model(
        model,
        validation_df,
        feature_columns,
        target_col,
    )
    test_metrics, _ = evaluate_model(model, test_df, feature_columns, target_col)

    baseline_test_roc_auc = baseline_metrics.get("test_roc_auc") if baseline_metrics else None
    baseline_test_f1 = baseline_metrics.get("test_f1") if baseline_metrics else None

    return {
        "pruning_mode": pruning_mode,
        "model_type": model_type,
        "feature_count": len(feature_columns),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "test_accuracy": _metric(test_metrics, "accuracy"),
        "test_precision": _metric(test_metrics, "precision"),
        "test_recall": _metric(test_metrics, "recall"),
        "test_f1": _metric(test_metrics, "f1"),
        "test_roc_auc": _metric(test_metrics, "roc_auc"),
        "validation_roc_auc": _metric(validation_metrics, "roc_auc"),
        "delta_test_roc_auc_vs_full": _safe_delta(
            _metric(test_metrics, "roc_auc"),
            baseline_test_roc_auc,
        ),
        "delta_test_f1_vs_full": _safe_delta(
            _metric(test_metrics, "f1"),
            baseline_test_f1,
        ),
        "warning": warning,
    }


def build_pruning_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    return (
        results_df.groupby("pruning_mode", dropna=False)
        .agg(
            model_count=("model_type", "nunique"),
            avg_feature_count=("feature_count", "mean"),
            avg_test_roc_auc=("test_roc_auc", "mean"),
            avg_test_f1=("test_f1", "mean"),
            avg_validation_roc_auc=("validation_roc_auc", "mean"),
            avg_delta_test_roc_auc_vs_full=("delta_test_roc_auc_vs_full", "mean"),
            avg_delta_test_f1_vs_full=("delta_test_f1_vs_full", "mean"),
        )
        .reset_index()
        .sort_values("avg_delta_test_roc_auc_vs_full", ascending=False)
    )


def build_feature_set_details(feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pruning_mode": mode,
                "feature_count": len(features),
                "features": " | ".join(features),
            }
            for mode, features in feature_sets.items()
        ]
    )


def run_factor_pruning_experiment(
    factor_csv: str | Path,
    recommendations_path: str | Path,
    model_types: list[str] | None = None,
    pruning_modes: list[str] | None = None,
    target_col: str = "label_up_5d",
    purge_rows: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict[str, Any]:
    if model_types is None:
        model_types = ["logistic_regression", "random_forest"]
    if pruning_modes is None:
        pruning_modes = DEFAULT_PRUNING_MODES.copy()

    df = normalize_date_column(load_factor_dataset(factor_csv))
    validate_required_columns(df, target_col)
    safe_features = identify_safe_feature_columns(df, target_col)
    recommendations_df = load_pruning_recommendations(recommendations_path)
    feature_sets = build_feature_sets(safe_features, recommendations_df, pruning_modes)

    results = []
    warnings = []
    for model_type in model_types:
        baseline_row = run_pruning_mode(
            df=df,
            feature_columns=feature_sets.get("full", safe_features),
            pruning_mode="full",
            model_type=model_type,
            target_col=target_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_rows=purge_rows,
            baseline_metrics=None,
        )
        baseline_row["delta_test_roc_auc_vs_full"] = 0.0
        baseline_row["delta_test_f1_vs_full"] = 0.0
        if "full" in pruning_modes:
            results.append(baseline_row)
        baseline_metrics = {
            "test_roc_auc": baseline_row["test_roc_auc"],
            "test_f1": baseline_row["test_f1"],
        }

        for mode in pruning_modes:
            if mode == "full":
                continue
            row = run_pruning_mode(
                df=df,
                feature_columns=feature_sets.get(mode, []),
                pruning_mode=mode,
                model_type=model_type,
                target_col=target_col,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                purge_rows=purge_rows,
                baseline_metrics=baseline_metrics,
            )
            results.append(row)
            if row["warning"]:
                warnings.append(
                    {
                        "pruning_mode": mode,
                        "model_type": model_type,
                        "warning": row["warning"],
                    }
                )

    results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    warnings_df = pd.DataFrame(warnings, columns=["pruning_mode", "model_type", "warning"])
    return {
        "pruning_results": results_df,
        "pruning_summary": build_pruning_summary(results_df),
        "feature_set_details": build_feature_set_details(feature_sets),
        "warnings": warnings_df,
        "safe_feature_count": len(safe_features),
    }


def save_pruning_outputs(
    output_dir: str | Path,
    result: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "pruning_results": output_path / "pruning_results.csv",
        "pruning_summary": output_path / "pruning_summary.csv",
        "feature_set_details": output_path / "feature_set_details.csv",
        "warnings": output_path / "warnings.csv",
        "run_config": output_path / "run_config.json",
    }
    result["pruning_results"].to_csv(paths["pruning_results"], index=False)
    result["pruning_summary"].to_csv(paths["pruning_summary"], index=False)
    result["feature_set_details"].to_csv(paths["feature_set_details"], index=False)
    result["warnings"].to_csv(paths["warnings"], index=False)
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def run_and_save_factor_pruning_experiment(
    factor_csv: str | Path,
    recommendations_path: str | Path,
    output_dir: str | Path,
    model_types: list[str] | None = None,
    pruning_modes: list[str] | None = None,
    target_col: str = "label_up_5d",
    purge_rows: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict[str, Any]:
    result = run_factor_pruning_experiment(
        factor_csv=factor_csv,
        recommendations_path=recommendations_path,
        model_types=model_types,
        pruning_modes=pruning_modes,
        target_col=target_col,
        purge_rows=purge_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    run_config = {
        "factor_csv": str(factor_csv),
        "recommendations_path": str(recommendations_path),
        "output_dir": str(output_dir),
        "model_types": model_types,
        "pruning_modes": pruning_modes,
        "target_col": target_col,
        "purge_rows": purge_rows,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    output_files = save_pruning_outputs(output_dir, result, run_config)
    result["run_config"] = run_config
    result["output_files"] = output_files
    return result
