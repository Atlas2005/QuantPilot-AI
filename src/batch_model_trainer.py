import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .build_factor_dataset import build_demo_ohlcv
    from .dataset_splitter import (
        build_split_report,
        check_for_leakage_columns,
        chronological_split,
        clean_factor_dataset,
        duplicate_symbol_date_count,
        infer_feature_columns,
        normalize_date_column,
        save_split_outputs,
        validate_required_columns,
    )
    from .factor_builder import build_factor_dataset
    from .model_trainer import MODEL_CHOICES, run_training_workflow
    from .real_data_loader import fetch_a_share_daily_from_source
except ImportError:
    from build_factor_dataset import build_demo_ohlcv
    from dataset_splitter import (
        build_split_report,
        check_for_leakage_columns,
        chronological_split,
        clean_factor_dataset,
        duplicate_symbol_date_count,
        infer_feature_columns,
        normalize_date_column,
        save_split_outputs,
        validate_required_columns,
    )
    from factor_builder import build_factor_dataset
    from model_trainer import MODEL_CHOICES, run_training_workflow
    from real_data_loader import fetch_a_share_daily_from_source


RESULT_COLUMNS = [
    "symbol",
    "model_type",
    "source",
    "start",
    "end",
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
    "validation_positive_rate",
    "test_positive_rate",
    "model_dir",
    "error",
]


def parse_symbols(symbols_text: str) -> list[str]:
    """Parse comma-separated A-share symbols."""
    symbols = []
    for item in symbols_text.split(","):
        symbol = item.strip()
        if not symbol:
            continue
        if symbol.isdigit() and len(symbol) < 6:
            symbol = symbol.zfill(6)
        symbols.append(symbol)
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def parse_model_types(models_text: str) -> list[str]:
    """Parse comma-separated model type names."""
    models = [item.strip() for item in models_text.split(",") if item.strip()]
    if not models:
        raise ValueError("At least one model type is required.")

    invalid = [model for model in models if model not in MODEL_CHOICES]
    if invalid:
        raise ValueError(
            f"Unsupported model types: {invalid}. Supported: {list(MODEL_CHOICES)}"
        )
    return models


def fetch_symbol_ohlcv(
    symbol: str,
    source: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Load OHLCV data for one symbol from demo or Baostock."""
    if source == "demo":
        return build_demo_ohlcv(start, end)
    if source == "baostock":
        return fetch_a_share_daily_from_source(
            symbol=symbol,
            start_date=start,
            end_date=end,
            adjust="qfq",
            source="baostock",
        )
    raise ValueError("source must be demo or baostock.")


def prepare_symbol_split(
    symbol: str,
    source: str,
    start: str,
    end: str,
    target_col: str,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_rows: int,
    split_mode: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build factor data and save chronological train/validation/test splits."""
    raw_df = fetch_symbol_ohlcv(symbol, source, start, end)
    factor_df = build_factor_dataset(raw_df, symbol=symbol)

    factor_dir = output_dir / "factors"
    factor_dir.mkdir(parents=True, exist_ok=True)
    factor_path = factor_dir / f"factors_{symbol}.csv"
    factor_save_df = factor_df.copy()
    factor_save_df["date"] = pd.to_datetime(factor_save_df["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    factor_save_df.to_csv(factor_path, index=False)

    normalized_df = normalize_date_column(factor_df)
    validate_required_columns(normalized_df, target_col)
    feature_columns = infer_feature_columns(normalized_df, target_col)
    leakage_checks = check_for_leakage_columns(feature_columns, target_col)
    duplicate_count = duplicate_symbol_date_count(normalized_df)
    cleaned_df, missing_values = clean_factor_dataset(
        normalized_df,
        feature_columns,
        target_col,
    )
    train_df, validation_df, test_df = chronological_split(
        cleaned_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        purge_rows=purge_rows,
        split_mode=split_mode,
    )

    split_dir = output_dir / "splits" / symbol
    split_report = build_split_report(
        input_path=factor_path,
        target_col=target_col,
        feature_columns=feature_columns,
        original_rows=len(normalized_df),
        cleaned_rows=len(cleaned_df),
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        purge_rows=purge_rows,
        split_mode=split_mode,
        leakage_checks=leakage_checks,
        missing_values=missing_values,
        duplicate_count=duplicate_count,
    )
    save_split_outputs(
        output_dir=split_dir,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_columns=feature_columns,
        split_report=split_report,
    )

    split_info = {
        "factor_path": str(factor_path),
        "split_dir": str(split_dir),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
    }
    return factor_path, split_dir, split_info


def _metric(metrics: dict[str, Any], key: str):
    return metrics.get(key)


def suspicious_perfect_metrics(metrics: dict[str, Any]) -> bool:
    values = [
        metrics.get(key)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        if metrics.get(key) is not None
    ]
    return bool(values) and all(value >= 0.98 for value in values)


def make_error_result(
    symbol: str,
    model_type: str,
    source: str,
    start: str,
    end: str,
    error: str,
) -> dict[str, Any]:
    row = {column: None for column in RESULT_COLUMNS}
    row.update(
        {
            "symbol": symbol,
            "model_type": model_type,
            "source": source,
            "start": start,
            "end": end,
            "error": error,
        }
    )
    return row


def warnings_for_result(
    symbol: str,
    model_type: str,
    training_info: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    output_files: dict[str, Any],
) -> list[dict[str, str]]:
    warnings = []
    if training_info.get("single_class_training"):
        warnings.append(
            {
                "warning_type": "single_class_split",
                "symbol": symbol,
                "model_type": model_type,
                "message": "Training split had one target class; dummy fallback was used.",
            }
        )

    if (test_metrics.get("sample_count") or 0) < 50:
        warnings.append(
            {
                "warning_type": "small_test_sample",
                "symbol": symbol,
                "model_type": model_type,
                "message": f"Test sample is small: {test_metrics.get('sample_count')}.",
            }
        )

    if suspicious_perfect_metrics(validation_metrics) or suspicious_perfect_metrics(
        test_metrics
    ):
        warnings.append(
            {
                "warning_type": "suspicious_perfect_metrics",
                "symbol": symbol,
                "model_type": model_type,
                "message": "Validation or test metrics are suspiciously close to 1.0.",
            }
        )

    if not output_files.get("feature_importance"):
        warnings.append(
            {
                "warning_type": "missing_feature_importance",
                "symbol": symbol,
                "model_type": model_type,
                "message": "Feature importance output was not produced for this model.",
            }
        )

    return warnings


def train_symbol_models(
    symbol: str,
    model_types: list[str],
    source: str,
    start: str,
    end: str,
    target_col: str,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_rows: int,
    split_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build/split data once for a symbol, then train each requested model."""
    results = []
    warnings = []

    try:
        _, split_dir, split_info = prepare_symbol_split(
            symbol=symbol,
            source=source,
            start=start,
            end=end,
            target_col=target_col,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_rows=purge_rows,
            split_mode=split_mode,
        )
    except Exception as exc:
        for model_type in model_types:
            error = str(exc)
            results.append(
                make_error_result(symbol, model_type, source, start, end, error)
            )
            warnings.append(
                {
                    "warning_type": "training_failed",
                    "symbol": symbol,
                    "model_type": model_type,
                    "message": error,
                }
            )
        return results, warnings

    for model_type in model_types:
        model_dir = output_dir / "models" / symbol / model_type
        try:
            workflow = run_training_workflow(
                dataset_dir=split_dir,
                target_col=target_col,
                model_name=model_type,
                output_dir=model_dir,
            )
            validation_metrics = workflow["validation_metrics"]
            test_metrics = workflow["test_metrics"]
            row = {
                "symbol": symbol,
                "model_type": model_type,
                "source": source,
                "start": start,
                "end": end,
                "train_rows": split_info["train_rows"],
                "validation_rows": split_info["validation_rows"],
                "test_rows": split_info["test_rows"],
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
                "validation_positive_rate": _metric(
                    validation_metrics,
                    "positive_rate",
                ),
                "test_positive_rate": _metric(test_metrics, "positive_rate"),
                "model_dir": str(model_dir),
                "error": None,
            }
            results.append(row)
            warnings.extend(
                warnings_for_result(
                    symbol=symbol,
                    model_type=model_type,
                    training_info=workflow["training_info"],
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    output_files=workflow["output_files"],
                )
            )
        except Exception as exc:
            error = str(exc)
            results.append(
                make_error_result(symbol, model_type, source, start, end, error)
            )
            warnings.append(
                {
                    "warning_type": "training_failed",
                    "symbol": symbol,
                    "model_type": model_type,
                    "message": error,
                }
            )

    return results, warnings


def build_model_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level training results by model type."""
    rows = []
    for model_type, group in results_df.groupby("model_type", dropna=False):
        success = group[group["error"].isna()]
        failed = group[group["error"].notna()]
        rows.append(
            {
                "model_type": model_type,
                "symbols_tested": int(group["symbol"].nunique()),
                "successful_symbols": int(success["symbol"].nunique()),
                "failed_symbols": int(failed["symbol"].nunique()),
                "avg_test_accuracy": success["test_accuracy"].mean(),
                "avg_test_precision": success["test_precision"].mean(),
                "avg_test_recall": success["test_recall"].mean(),
                "avg_test_f1": success["test_f1"].mean(),
                "avg_test_roc_auc": success["test_roc_auc"].mean(),
                "avg_validation_roc_auc": success["validation_roc_auc"].mean(),
                "avg_test_positive_rate": success["test_positive_rate"].mean(),
                "avg_test_sample_count": success["test_rows"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_model_ranking(
    summary_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank model types with a simple educational robustness score."""
    rows = []
    for row in summary_df.to_dict(orient="records"):
        model_type = row["model_type"]
        penalty = 0.0
        if pd.notna(row.get("avg_test_sample_count")) and row["avg_test_sample_count"] < 50:
            penalty += 0.25
        if row.get("failed_symbols", 0) > 0:
            penalty += 0.10
        if not warnings_df.empty:
            warning_mask = (
                (warnings_df["model_type"] == model_type)
                & (
                    warnings_df["warning_type"]
                    == "suspicious_perfect_metrics"
                )
            )
            if warning_mask.any():
                penalty += 0.25

        roc_auc = 0.0 if pd.isna(row.get("avg_test_roc_auc")) else row["avg_test_roc_auc"]
        f1 = 0.0 if pd.isna(row.get("avg_test_f1")) else row["avg_test_f1"]
        rows.append(
            {
                "model_type": model_type,
                "avg_test_roc_auc": row.get("avg_test_roc_auc"),
                "avg_test_f1": row.get("avg_test_f1"),
                "penalty": penalty,
                "score": roc_auc + f1 - penalty,
                "failed_symbols": row.get("failed_symbols"),
                "avg_test_sample_count": row.get("avg_test_sample_count"),
            }
        )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(
        drop=True
    )


def save_outputs(
    output_dir: Path,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
    run_config: dict[str, Any],
) -> dict[str, str]:
    """Save robustness CSV and JSON outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_results": output_dir / "training_results.csv",
        "model_summary": output_dir / "model_summary.csv",
        "model_ranking": output_dir / "model_ranking.csv",
        "warnings": output_dir / "warnings.csv",
        "run_config": output_dir / "run_config.json",
    }
    results_df.to_csv(paths["training_results"], index=False)
    summary_df.to_csv(paths["model_summary"], index=False)
    ranking_df.to_csv(paths["model_ranking"], index=False)
    warnings_df.to_csv(paths["warnings"], index=False)
    paths["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


def run_batch_model_training(
    symbols: list[str],
    model_types: list[str],
    source: str = "demo",
    start: str = "20240101",
    end: str = "20241231",
    output_dir: str | Path = "outputs/model_robustness",
    target_col: str = "label_up_5d",
    purge_rows: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    split_mode: str = "global_date",
) -> dict[str, Any]:
    """Run multi-symbol, multi-model robustness training."""
    output_path = Path(output_dir)
    all_results = []
    all_warnings = []
    progress = []

    for symbol in symbols:
        progress.append(f"Starting {symbol}")
        results, warnings = train_symbol_models(
            symbol=symbol,
            model_types=model_types,
            source=source,
            start=start,
            end=end,
            target_col=target_col,
            output_dir=output_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_rows=purge_rows,
            split_mode=split_mode,
        )
        all_results.extend(results)
        all_warnings.extend(warnings)
        progress.append(f"Finished {symbol}")

    results_df = pd.DataFrame(all_results, columns=RESULT_COLUMNS)
    warnings_df = pd.DataFrame(
        all_warnings,
        columns=["warning_type", "symbol", "model_type", "message"],
    )
    summary_df = build_model_summary(results_df)
    ranking_df = build_model_ranking(summary_df, warnings_df)
    run_config = {
        "symbols": symbols,
        "source": source,
        "start": start,
        "end": end,
        "model_types": model_types,
        "target_col": target_col,
        "purge_rows": purge_rows,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_mode": split_mode,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    output_files = save_outputs(
        output_path,
        results_df,
        summary_df,
        ranking_df,
        warnings_df,
        run_config,
    )
    return {
        "training_results": results_df,
        "model_summary": summary_df,
        "model_ranking": ranking_df,
        "warnings": warnings_df,
        "run_config": run_config,
        "output_files": output_files,
        "progress": progress,
    }
