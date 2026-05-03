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
    from .model_trainer import MODEL_CHOICES, evaluate_model, train_baseline_model
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
    from model_trainer import MODEL_CHOICES, evaluate_model, train_baseline_model


FACTOR_GROUPS = {
    "baseline_price_return": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "return_5d",
        "return_20d",
        "volume_change_1d",
        "volume_change_5d",
        "dollar_volume",
        "high_low_range_pct",
        "close_open_return_pct",
    ],
    "baseline_ma_trend": [
        "ma5",
        "ma20",
        "ma60",
        "ma5_gap_pct",
        "ma20_gap_pct",
        "ma60_gap_pct",
        "ma5_ma20_gap_pct",
        "ma20_ma60_gap_pct",
    ],
    "baseline_risk_momentum": [
        "volatility_5d",
        "volatility_20d",
        "drawdown_20d",
        "drawdown_60d",
        "rolling_high_20d",
        "rolling_low_20d",
        "RSI",
        "CCI",
    ],
    "p0_intraday_candle": [
        "intraday_range_pct",
        "candle_body_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
    ],
    "p0_volume_liquidity": [
        "volume_ma5",
        "volume_ma20",
        "volume_ratio_5d",
        "volume_ratio_20d",
        "turnover_proxy",
    ],
    "p0_position_breakout": [
        "price_position_20d",
        "price_position_60d",
        "breakout_20d",
        "breakdown_20d",
        "trend_strength_20d",
        "volatility_ratio_5d_20d",
    ],
}

P0_FEATURES = (
    FACTOR_GROUPS["p0_intraday_candle"]
    + FACTOR_GROUPS["p0_volume_liquidity"]
    + FACTOR_GROUPS["p0_position_breakout"]
)

FEATURE_TO_GROUP = {
    feature: group_name
    for group_name, features in FACTOR_GROUPS.items()
    for feature in features
}

PRUNING_COLUMNS = [
    "removed_feature",
    "factor_group",
    "recommendation",
    "experiment_count",
    "model_type_count",
    "avg_test_roc_auc_delta_when_removed",
    "avg_test_f1_delta_when_removed",
    "roc_auc_improve_rate_when_removed",
    "roc_auc_hurt_rate_when_removed",
    "rationale",
]

RESULT_COLUMNS = [
    "experiment_name",
    "ablation_type",
    "removed_group",
    "kept_group",
    "removed_feature",
    "model_type",
    "symbol",
    "feature_count",
    "train_rows",
    "validation_rows",
    "test_rows",
    "validation_accuracy",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_roc_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_positive_rate",
    "baseline_test_roc_auc",
    "test_roc_auc_delta_vs_full",
    "test_f1_delta_vs_full",
    "warning",
]


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_model_types(models_text: str) -> list[str]:
    models = parse_csv_list(models_text)
    invalid = [model for model in models if model not in MODEL_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}. Supported: {MODEL_CHOICES}")
    return models


def parse_ablation_modes(modes_text: str) -> list[str]:
    allowed = {"full", "drop_group", "only_group", "drop_feature"}
    modes = parse_csv_list(modes_text)
    invalid = [mode for mode in modes if mode not in allowed]
    if invalid:
        raise ValueError(f"Unsupported ablation modes: {invalid}. Supported: {sorted(allowed)}")
    return modes


def get_group_columns(group_name: str, available_columns: list[str]) -> list[str]:
    available = set(available_columns)
    return [column for column in FACTOR_GROUPS[group_name] if column in available]


def identify_safe_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    feature_columns = infer_feature_columns(df, target_col)
    leakage_checks = check_for_leakage_columns(feature_columns, target_col)
    if not leakage_checks["passed"]:
        raise ValueError(
            "Leakage-safe feature inference failed: "
            f"{leakage_checks['leakage_columns']}"
        )
    return feature_columns


def build_ablation_specs(
    feature_columns: list[str],
    ablation_modes: list[str],
    max_drop_features: int | None = None,
) -> list[dict[str, Any]]:
    specs = []
    if "full" in ablation_modes or "drop_group" in ablation_modes or "only_group" in ablation_modes or "drop_feature" in ablation_modes:
        specs.append(
            {
                "experiment_name": "full",
                "ablation_type": "full",
                "removed_group": None,
                "kept_group": None,
                "removed_feature": None,
                "feature_columns": feature_columns,
            }
        )

    if "drop_group" in ablation_modes:
        for group_name in FACTOR_GROUPS:
            group_columns = get_group_columns(group_name, feature_columns)
            if not group_columns:
                continue
            specs.append(
                {
                    "experiment_name": f"drop_group__{group_name}",
                    "ablation_type": "drop_group",
                    "removed_group": group_name,
                    "kept_group": None,
                    "removed_feature": None,
                    "feature_columns": [
                        column for column in feature_columns if column not in group_columns
                    ],
                }
            )

    if "only_group" in ablation_modes:
        for group_name in FACTOR_GROUPS:
            group_columns = get_group_columns(group_name, feature_columns)
            if not group_columns:
                continue
            specs.append(
                {
                    "experiment_name": f"only_group__{group_name}",
                    "ablation_type": "only_group",
                    "removed_group": None,
                    "kept_group": group_name,
                    "removed_feature": None,
                    "feature_columns": group_columns,
                }
            )

    if "drop_feature" in ablation_modes:
        ordered_features = []
        for group_features in FACTOR_GROUPS.values():
            ordered_features.extend(
                feature
                for feature in group_features
                if feature in feature_columns and feature not in ordered_features
            )
        ordered_features.extend(
            feature for feature in feature_columns if feature not in ordered_features
        )
        drop_features = ordered_features
        if max_drop_features is not None:
            drop_features = drop_features[:max_drop_features]
        for feature in drop_features:
            specs.append(
                {
                    "experiment_name": f"drop_feature__{feature}",
                    "ablation_type": "drop_feature",
                    "removed_group": None,
                    "kept_group": None,
                    "removed_feature": feature,
                    "feature_columns": [
                        column for column in feature_columns if column != feature
                    ],
                }
            )

    return specs


def _metric(metrics: dict[str, Any], key: str):
    return metrics.get(key)


def _safe_delta(value, baseline):
    if pd.isna(value) or pd.isna(baseline):
        return None
    return value - baseline


def run_one_experiment(
    df: pd.DataFrame,
    spec: dict[str, Any],
    model_type: str,
    symbol: str,
    target_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_rows: int,
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warning = None
    feature_columns = spec["feature_columns"]
    if not feature_columns:
        raise ValueError(f"Experiment {spec['experiment_name']} has no feature columns.")

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

    baseline_test_roc_auc = None
    baseline_test_f1 = None
    if baseline_metrics:
        baseline_test_roc_auc = baseline_metrics.get("test_roc_auc")
        baseline_test_f1 = baseline_metrics.get("test_f1")

    return {
        "experiment_name": spec["experiment_name"],
        "ablation_type": spec["ablation_type"],
        "removed_group": spec["removed_group"],
        "kept_group": spec["kept_group"],
        "removed_feature": spec["removed_feature"],
        "model_type": model_type,
        "symbol": symbol,
        "feature_count": len(feature_columns),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "validation_accuracy": _metric(validation_metrics, "accuracy"),
        "validation_precision": _metric(validation_metrics, "precision"),
        "validation_recall": _metric(validation_metrics, "recall"),
        "validation_f1": _metric(validation_metrics, "f1"),
        "validation_roc_auc": _metric(validation_metrics, "roc_auc"),
        "test_accuracy": _metric(test_metrics, "accuracy"),
        "test_precision": _metric(test_metrics, "precision"),
        "test_recall": _metric(test_metrics, "recall"),
        "test_f1": _metric(test_metrics, "f1"),
        "test_roc_auc": _metric(test_metrics, "roc_auc"),
        "test_positive_rate": _metric(test_metrics, "positive_rate"),
        "baseline_test_roc_auc": baseline_test_roc_auc,
        "test_roc_auc_delta_vs_full": _safe_delta(
            _metric(test_metrics, "roc_auc"),
            baseline_test_roc_auc,
        ),
        "test_f1_delta_vs_full": _safe_delta(
            _metric(test_metrics, "f1"),
            baseline_test_f1,
        ),
        "warning": warning,
    }


def run_factor_ablation(
    input_path: str | Path,
    target_col: str = "label_up_5d",
    model_types: list[str] | None = None,
    ablation_modes: list[str] | None = None,
    purge_rows: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    symbol: str | None = None,
    max_drop_features: int | None = None,
) -> dict[str, Any]:
    if model_types is None:
        model_types = ["logistic_regression", "random_forest"]
    if ablation_modes is None:
        ablation_modes = ["drop_group", "only_group"]

    df = normalize_date_column(load_factor_dataset(input_path))
    validate_required_columns(df, target_col)
    if symbol is None:
        symbol = str(df["symbol"].dropna().iloc[0]) if "symbol" in df and not df["symbol"].dropna().empty else "UNKNOWN"
    feature_columns = identify_safe_feature_columns(df, target_col)
    specs = build_ablation_specs(feature_columns, ablation_modes, max_drop_features)

    results = []
    warnings = []
    for model_type in model_types:
        full_spec = next(spec for spec in specs if spec["ablation_type"] == "full")
        try:
            full_row = run_one_experiment(
                df=df,
                spec=full_spec,
                model_type=model_type,
                symbol=symbol,
                target_col=target_col,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                purge_rows=purge_rows,
                baseline_metrics=None,
            )
            full_row["baseline_test_roc_auc"] = full_row["test_roc_auc"]
            full_row["test_roc_auc_delta_vs_full"] = 0.0
            full_row["test_f1_delta_vs_full"] = 0.0
            results.append(full_row)
            baseline_metrics = {
                "test_roc_auc": full_row["test_roc_auc"],
                "test_f1": full_row["test_f1"],
            }
        except Exception as exc:
            warnings.append(
                {
                    "symbol": symbol,
                    "model_type": model_type,
                    "experiment_name": "full",
                    "warning": str(exc),
                }
            )
            continue

        for spec in specs:
            if spec["ablation_type"] == "full":
                continue
            try:
                results.append(
                    run_one_experiment(
                        df=df,
                        spec=spec,
                        model_type=model_type,
                        symbol=symbol,
                        target_col=target_col,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        purge_rows=purge_rows,
                        baseline_metrics=baseline_metrics,
                    )
                )
            except Exception as exc:
                warnings.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "experiment_name": spec["experiment_name"],
                        "warning": str(exc),
                    }
                )

    results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    warnings_df = pd.DataFrame(
        warnings,
        columns=["symbol", "model_type", "experiment_name", "warning"],
    )
    return {
        "ablation_results": results_df,
        "group_summary": build_group_summary(results_df),
        "feature_impact_ranking": build_feature_impact_ranking(results_df),
        "feature_pruning_recommendations": build_feature_pruning_recommendations(
            results_df
        ),
        "warnings": warnings_df,
        "feature_columns": feature_columns,
    }


def build_group_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    group_rows = results_df[
        results_df["ablation_type"].isin(["drop_group", "only_group"])
    ].copy()
    if group_rows.empty:
        return pd.DataFrame()
    group_rows["factor_group"] = group_rows["removed_group"].fillna(
        group_rows["kept_group"]
    )
    summary = (
        group_rows.groupby(["factor_group", "ablation_type", "model_type"], dropna=False)
        .agg(
            experiment_count=("experiment_name", "count"),
            avg_test_roc_auc=("test_roc_auc", "mean"),
            avg_test_f1=("test_f1", "mean"),
            avg_test_roc_auc_delta_vs_full=("test_roc_auc_delta_vs_full", "mean"),
            avg_test_f1_delta_vs_full=("test_f1_delta_vs_full", "mean"),
        )
        .reset_index()
        .sort_values("avg_test_roc_auc_delta_vs_full", ascending=False)
    )
    return summary


def build_feature_impact_ranking(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    feature_rows = results_df[results_df["ablation_type"] == "drop_feature"].copy()
    if feature_rows.empty:
        return pd.DataFrame()
    ranking = (
        feature_rows.groupby(["removed_feature", "model_type"], dropna=False)
        .agg(
            experiment_count=("experiment_name", "count"),
            avg_test_roc_auc_delta_when_removed=("test_roc_auc_delta_vs_full", "mean"),
            avg_test_f1_delta_when_removed=("test_f1_delta_vs_full", "mean"),
            avg_test_roc_auc=("test_roc_auc", "mean"),
            avg_test_f1=("test_f1", "mean"),
        )
        .reset_index()
        .sort_values("avg_test_roc_auc_delta_when_removed", ascending=False)
    )
    return ranking


def _feature_recommendation(
    avg_roc_delta: float | None,
    avg_f1_delta: float | None,
    improve_rate: float,
    hurt_rate: float,
    experiment_count: int,
) -> tuple[str, str]:
    if experiment_count < 2:
        return (
            "needs_more_data",
            "Too few feature-level ablation experiments are available.",
        )
    if (
        avg_roc_delta is not None
        and avg_f1_delta is not None
        and avg_roc_delta > 0.01
        and avg_f1_delta > 0
        and improve_rate >= 0.6
    ):
        return (
            "drop_candidate",
            "Removing the feature improves ROC AUC and F1 consistently.",
        )
    if avg_roc_delta is not None and avg_roc_delta > 0.005:
        return (
            "reduce_weight",
            "Removing the feature improves ROC AUC on average, so it may add noise.",
        )
    if (
        avg_roc_delta is not None
        and avg_f1_delta is not None
        and avg_roc_delta < -0.01
        and avg_f1_delta <= 0
        and hurt_rate >= 0.6
    ):
        return (
            "keep_core",
            "Removing the feature hurts ROC AUC and does not improve F1 consistently.",
        )
    if avg_roc_delta is not None and avg_roc_delta <= 0.005:
        return (
            "keep_observe",
            "The feature is mixed or mildly useful and is not clearly harmful.",
        )
    return (
        "needs_more_data",
        "Evidence is mixed or incomplete across models and symbols.",
    )


def build_feature_pruning_recommendations(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(columns=PRUNING_COLUMNS)
    feature_rows = results_df[results_df["ablation_type"] == "drop_feature"].copy()
    if feature_rows.empty:
        return pd.DataFrame(columns=PRUNING_COLUMNS)

    rows = []
    for feature, group in feature_rows.groupby("removed_feature", dropna=False):
        roc_delta = pd.to_numeric(
            group["test_roc_auc_delta_vs_full"],
            errors="coerce",
        )
        f1_delta = pd.to_numeric(group["test_f1_delta_vs_full"], errors="coerce")
        avg_roc_delta = None if roc_delta.dropna().empty else float(roc_delta.mean())
        avg_f1_delta = None if f1_delta.dropna().empty else float(f1_delta.mean())
        improve_rate = float((roc_delta.dropna() > 0).mean()) if not roc_delta.dropna().empty else 0.0
        hurt_rate = float((roc_delta.dropna() < 0).mean()) if not roc_delta.dropna().empty else 0.0
        experiment_count = int(len(group))
        recommendation, rationale = _feature_recommendation(
            avg_roc_delta=avg_roc_delta,
            avg_f1_delta=avg_f1_delta,
            improve_rate=improve_rate,
            hurt_rate=hurt_rate,
            experiment_count=experiment_count,
        )
        rows.append(
            {
                "removed_feature": feature,
                "factor_group": FEATURE_TO_GROUP.get(feature, "unmapped"),
                "recommendation": recommendation,
                "experiment_count": experiment_count,
                "model_type_count": int(group["model_type"].nunique()),
                "avg_test_roc_auc_delta_when_removed": avg_roc_delta,
                "avg_test_f1_delta_when_removed": avg_f1_delta,
                "roc_auc_improve_rate_when_removed": improve_rate,
                "roc_auc_hurt_rate_when_removed": hurt_rate,
                "rationale": rationale,
            }
        )

    order = {
        "keep_core": 0,
        "keep_observe": 1,
        "reduce_weight": 2,
        "drop_candidate": 3,
        "needs_more_data": 4,
    }
    result = pd.DataFrame(rows, columns=PRUNING_COLUMNS)
    result["_order"] = result["recommendation"].map(order).fillna(9)
    return result.sort_values(
        ["_order", "avg_test_roc_auc_delta_when_removed"],
        ascending=[True, True],
    ).drop(columns=["_order"])


def save_ablation_outputs(
    output_dir: str | Path,
    result: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "ablation_results": output_path / "ablation_results.csv",
        "group_summary": output_path / "group_summary.csv",
        "feature_impact_ranking": output_path / "feature_impact_ranking.csv",
        "feature_ablation_results": output_path / "feature_ablation_results.csv",
        "feature_pruning_recommendations": output_path / "feature_pruning_recommendations.csv",
        "warnings": output_path / "warnings.csv",
        "run_config": output_path / "run_config.json",
    }
    result["ablation_results"].to_csv(paths["ablation_results"], index=False)
    result["group_summary"].to_csv(paths["group_summary"], index=False)
    result["feature_impact_ranking"].to_csv(
        paths["feature_impact_ranking"],
        index=False,
    )
    if result["ablation_results"].empty or "ablation_type" not in result["ablation_results"]:
        feature_ablation_results = pd.DataFrame(columns=RESULT_COLUMNS)
    else:
        feature_ablation_results = result["ablation_results"][
            result["ablation_results"]["ablation_type"] == "drop_feature"
        ].copy()
    feature_ablation_results.to_csv(paths["feature_ablation_results"], index=False)
    result.get(
        "feature_pruning_recommendations",
        build_feature_pruning_recommendations(result["ablation_results"]),
    ).to_csv(paths["feature_pruning_recommendations"], index=False)
    result["warnings"].to_csv(paths["warnings"], index=False)
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def run_and_save_factor_ablation(
    input_path: str | Path,
    output_dir: str | Path,
    target_col: str = "label_up_5d",
    model_types: list[str] | None = None,
    ablation_modes: list[str] | None = None,
    purge_rows: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    symbol: str | None = None,
    max_drop_features: int | None = None,
) -> dict[str, Any]:
    result = run_factor_ablation(
        input_path=input_path,
        target_col=target_col,
        model_types=model_types,
        ablation_modes=ablation_modes,
        purge_rows=purge_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        symbol=symbol,
        max_drop_features=max_drop_features,
    )
    run_config = {
        "input_path": str(input_path),
        "target_col": target_col,
        "model_types": model_types,
        "ablation_modes": ablation_modes,
        "purge_rows": purge_rows,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "symbol": symbol,
        "max_drop_features": max_drop_features,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    result["feature_pruning_recommendations"] = build_feature_pruning_recommendations(
        result["ablation_results"]
    )
    output_files = save_ablation_outputs(output_dir, result, run_config)
    result["run_config"] = run_config
    result["output_files"] = output_files
    return result
