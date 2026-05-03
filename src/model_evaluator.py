import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_TARGET_COL = "label_up_5d"
DEFAULT_PROBABILITY_COL = "prediction_probability"
DEFAULT_PREDICTION_COL = "prediction"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_feature_columns(path: str | Path) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_predictions(model_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load validation and test prediction CSVs from a model output directory."""
    path = Path(model_dir)
    prediction_files = {
        "validation": path / "validation_predictions.csv",
        "test": path / "test_predictions.csv",
    }

    missing = [str(file_path) for file_path in prediction_files.values() if not file_path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction files: {missing}")

    return {
        name: pd.read_csv(file_path, dtype={"symbol": str})
        for name, file_path in prediction_files.items()
    }


def _target_series(df: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col not in df.columns:
        raise ValueError(f"Prediction file is missing target column: {target_col}")
    return pd.to_numeric(df[target_col], errors="coerce")


def _prediction_series(df: pd.DataFrame) -> pd.Series:
    if DEFAULT_PREDICTION_COL not in df.columns:
        raise ValueError(f"Prediction file is missing column: {DEFAULT_PREDICTION_COL}")
    return pd.to_numeric(df[DEFAULT_PREDICTION_COL], errors="coerce")


def _probability_series(df: pd.DataFrame) -> pd.Series | None:
    if DEFAULT_PROBABILITY_COL not in df.columns:
        return None
    return pd.to_numeric(df[DEFAULT_PROBABILITY_COL], errors="coerce")


def clean_prediction_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Keep rows with valid target/prediction and optional probability."""
    result = df.copy()
    result[target_col] = _target_series(result, target_col)
    result[DEFAULT_PREDICTION_COL] = _prediction_series(result)
    if DEFAULT_PROBABILITY_COL in result.columns:
        result[DEFAULT_PROBABILITY_COL] = _probability_series(result)

    required = [target_col, DEFAULT_PREDICTION_COL]
    return result.dropna(subset=required).reset_index(drop=True)


def classification_metrics(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    cleaned = clean_prediction_frame(df, target_col)
    sample_count = len(cleaned)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "positive_rate": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_true = cleaned[target_col].astype(int)
    y_pred = cleaned[DEFAULT_PREDICTION_COL].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "sample_count": int(sample_count),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    probability = _probability_series(cleaned)
    if probability is not None and y_true.nunique() == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probability))
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def probability_analysis(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    cleaned = clean_prediction_frame(df, target_col)
    probability = _probability_series(cleaned)
    if probability is None or probability.dropna().empty:
        return {
            "available": False,
            "min_probability": None,
            "max_probability": None,
            "mean_probability": None,
            "median_probability": None,
            "avg_probability_actual_positive": None,
            "avg_probability_actual_negative": None,
            "bucket_distribution": [],
        }

    valid = cleaned.dropna(subset=[DEFAULT_PROBABILITY_COL]).copy()
    y_true = valid[target_col].astype(int)
    probability = valid[DEFAULT_PROBABILITY_COL]

    buckets = []
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        if index == 9:
            mask = (probability >= low) & (probability <= high)
        else:
            mask = (probability >= low) & (probability < high)
        count = int(mask.sum())
        buckets.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "count": count,
                "pct": float(count / len(valid)) if len(valid) else 0.0,
            }
        )

    positive_probs = probability[y_true == 1]
    negative_probs = probability[y_true == 0]
    return {
        "available": True,
        "min_probability": float(probability.min()),
        "max_probability": float(probability.max()),
        "mean_probability": float(probability.mean()),
        "median_probability": float(probability.median()),
        "avg_probability_actual_positive": (
            float(positive_probs.mean()) if not positive_probs.empty else None
        ),
        "avg_probability_actual_negative": (
            float(negative_probs.mean()) if not negative_probs.empty else None
        ),
        "bucket_distribution": buckets,
    }


def threshold_analysis(df: pd.DataFrame, target_col: str) -> list[dict[str, Any]]:
    cleaned = clean_prediction_frame(df, target_col)
    probability = _probability_series(cleaned)
    if probability is None or probability.dropna().empty:
        return []

    valid = cleaned.dropna(subset=[DEFAULT_PROBABILITY_COL]).copy()
    y_true = valid[target_col].astype(int)
    probability = valid[DEFAULT_PROBABILITY_COL]

    rows = []
    for threshold in THRESHOLDS:
        y_pred = (probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "predicted_positive_rate": float(y_pred.mean()),
                "signal_count": int(y_pred.sum()),
            }
        )

    return rows


def detect_feature_leakage(feature_columns: list[str]) -> list[str]:
    leak_words = ("target", "label", "future")
    return [
        column
        for column in feature_columns
        if any(word in column.lower() for word in leak_words)
    ]


def warning_checks(
    split_name: str,
    metrics: dict[str, Any],
    probabilities: dict[str, Any],
    feature_columns: list[str],
) -> list[str]:
    warnings = []
    sample_count = metrics.get("sample_count", 0) or 0
    if sample_count < 50:
        warnings.append(
            f"{split_name}: sample_count is small ({sample_count}); metrics may be unstable."
        )

    metric_values = [
        metrics.get("accuracy"),
        metrics.get("precision"),
        metrics.get("recall"),
        metrics.get("f1"),
        metrics.get("roc_auc"),
    ]
    valid_values = [value for value in metric_values if value is not None]
    if valid_values and all(value >= 0.98 for value in valid_values):
        warnings.append(
            f"{split_name}: classification metrics are suspiciously close to 1.0. "
            "Check for leakage, synthetic data effects, or an overly easy demo target."
        )

    positive_rate = metrics.get("positive_rate")
    if positive_rate is not None and (positive_rate < 0.1 or positive_rate > 0.9):
        warnings.append(
            f"{split_name}: positive_rate is extreme ({positive_rate:.2%}); "
            "class imbalance can make metrics misleading."
        )

    if probabilities.get("available"):
        min_prob = probabilities.get("min_probability")
        max_prob = probabilities.get("max_probability")
        if min_prob is not None and max_prob is not None and (max_prob - min_prob) < 0.1:
            warnings.append(
                f"{split_name}: predicted probabilities are tightly concentrated; "
                "threshold signals may not be meaningful."
            )

        buckets = probabilities.get("bucket_distribution", [])
        if buckets:
            largest_bucket_pct = max(bucket["pct"] for bucket in buckets)
            if largest_bucket_pct >= 0.8:
                warnings.append(
                    f"{split_name}: at least 80% of probabilities fall in one bucket; "
                    "probability distribution is concentrated."
                )

    leakage_columns = detect_feature_leakage(feature_columns)
    if leakage_columns:
        warnings.append(
            "Feature leakage risk: model features contain target/label/future columns: "
            + ", ".join(leakage_columns)
        )

    warnings.append(
        "Good ML classification metrics do not guarantee profitable trading. "
        "Trading results also depend on execution, costs, risk controls, and market regime."
    )
    return warnings


def simple_signal_backtest(
    df: pd.DataFrame,
    threshold: float = 0.6,
    probability_col: str = DEFAULT_PROBABILITY_COL,
) -> dict[str, Any]:
    """
    Educational long/flat return check using close-to-close returns.

    This is not the existing strategy backtester and does not change trading
    rules. It is only a quick diagnostic when close prices are present.
    """
    if "close" not in df.columns or probability_col not in df.columns:
        return {
            "available": False,
            "reason": "Prediction CSV needs close and prediction_probability columns.",
        }

    result = df.copy()
    result["date"] = pd.to_datetime(result.get("date"), errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result[probability_col] = pd.to_numeric(result[probability_col], errors="coerce")
    result = result.dropna(subset=["close", probability_col]).sort_values("date")
    if len(result) < 2:
        return {"available": False, "reason": "Not enough rows with close prices."}

    result["next_return"] = result["close"].pct_change().shift(-1)
    result["signal"] = (result[probability_col] >= threshold).astype(int)
    result["strategy_return"] = result["signal"] * result["next_return"]
    valid = result.dropna(subset=["next_return", "strategy_return"])
    if valid.empty:
        return {"available": False, "reason": "No valid return rows after shifting."}

    strategy_total_return = (1 + valid["strategy_return"]).prod() - 1
    buy_hold_return = (1 + valid["next_return"]).prod() - 1
    return {
        "available": True,
        "threshold": float(threshold),
        "rows": int(len(valid)),
        "signal_days": int(valid["signal"].sum()),
        "signal_rate": float(valid["signal"].mean()),
        "model_signal_return_pct": float(strategy_total_return * 100),
        "buy_and_hold_return_pct": float(buy_hold_return * 100),
        "excess_return_pct": float((strategy_total_return - buy_hold_return) * 100),
    }


def evaluate_prediction_frame(
    df: pd.DataFrame,
    split_name: str,
    target_col: str,
    feature_columns: list[str],
    signal_threshold: float,
) -> dict[str, Any]:
    metrics = classification_metrics(df, target_col)
    probability = probability_analysis(df, target_col)
    thresholds = threshold_analysis(df, target_col)
    warnings = warning_checks(split_name, metrics, probability, feature_columns)
    signal_backtest = simple_signal_backtest(df, threshold=signal_threshold)
    return {
        "metrics": metrics,
        "probability_analysis": probability,
        "threshold_analysis": thresholds,
        "warnings": warnings,
        "signal_backtest": signal_backtest,
    }


def evaluate_model_directory(
    model_dir: str | Path,
    target_col: str = DEFAULT_TARGET_COL,
    signal_threshold: float = 0.6,
) -> dict[str, Any]:
    """Evaluate validation/test prediction artifacts from one model directory."""
    path = Path(model_dir)
    metrics_json = load_json(path / "metrics.json")
    feature_columns = load_feature_columns(path / "feature_columns.txt")
    predictions = load_predictions(path)
    importance_path = path / "feature_importance.csv"
    feature_importance = (
        pd.read_csv(importance_path).to_dict(orient="records")
        if importance_path.exists()
        else []
    )

    validation = evaluate_prediction_frame(
        predictions["validation"],
        "validation",
        target_col,
        feature_columns,
        signal_threshold,
    )
    test = evaluate_prediction_frame(
        predictions["test"],
        "test",
        target_col,
        feature_columns,
        signal_threshold,
    )

    return {
        "model_dir": str(model_dir),
        "target_col": target_col,
        "signal_threshold": float(signal_threshold),
        "saved_metrics": metrics_json,
        "feature_columns": feature_columns,
        "feature_leakage_columns": detect_feature_leakage(feature_columns),
        "feature_importance": feature_importance,
        "validation": validation,
        "test": test,
    }
