import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_feature_columns(feature_columns_path: str | Path) -> list[str]:
    """Load the exact model input columns used during training."""
    path = Path(feature_columns_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature columns file does not exist: {path}")

    columns = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not columns:
        raise ValueError(f"No feature columns found in {path}")
    return columns


def load_metrics(metrics_path: str | Path) -> dict[str, Any]:
    """Load saved model metrics JSON."""
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def load_model_bundle(
    model_path: str | Path,
    metrics_path: str | Path | None = None,
    feature_columns_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    """
    Load a trained model, metrics, and feature columns.

    If metrics_path or feature_columns_path are omitted, they are inferred from
    the model directory as metrics.json and feature_columns.txt.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")

    model_dir = model_file.parent
    metrics_file = Path(metrics_path) if metrics_path else model_dir / "metrics.json"
    feature_file = (
        Path(feature_columns_path)
        if feature_columns_path
        else model_dir / "feature_columns.txt"
    )

    model = joblib.load(model_file)
    metrics = load_metrics(metrics_file)
    feature_columns = load_feature_columns(feature_file)
    return model, metrics, feature_columns


def load_prediction_input(input_path: str | Path) -> pd.DataFrame:
    """Load a factor CSV or ML split CSV used for prediction."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction input CSV does not exist: {path}")

    return pd.read_csv(path, dtype={"symbol": str})


def select_latest_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest row by date when available, otherwise the final row."""
    if df.empty:
        raise ValueError("Prediction input has no rows.")

    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result = result.sort_values("date")

    return result.tail(1).reset_index(drop=True)


def prepare_features(row_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Prepare one-row model features using the saved feature column list."""
    missing_columns = [
        column for column in feature_columns if column not in row_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Prediction input is missing model feature columns: "
            f"{missing_columns}"
        )

    features = row_df[feature_columns].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def predict_positive_probability(model: Any, features: pd.DataFrame) -> float | None:
    """Return class-1 probability when the model supports predict_proba."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(features)
    if probabilities.ndim != 2:
        return None

    classifier = getattr(model, "named_steps", {}).get("classifier")
    classes = getattr(classifier, "classes_", getattr(model, "classes_", []))
    if 1 in classes:
        return float(probabilities[0, list(classes).index(1)])
    if probabilities.shape[1] >= 2:
        return float(probabilities[0, 1])
    return None


def classify_model_signal(
    predicted_class: int,
    probability: float | None,
    bullish_threshold: float = 0.6,
    bearish_threshold: float = 0.4,
) -> str:
    """Convert prediction output into a simple educational model signal."""
    if probability is None:
        return "bullish" if predicted_class == 1 else "bearish"
    if probability >= bullish_threshold:
        return "bullish"
    if probability <= bearish_threshold:
        return "bearish"
    return "neutral"


def load_feature_importance(
    model: Any,
    model_path: str | Path,
    feature_columns: list[str],
    feature_importance_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load saved feature importance CSV or infer it from the model when possible."""
    model_dir = Path(model_path).parent
    importance_file = (
        Path(feature_importance_path)
        if feature_importance_path
        else model_dir / "feature_importance.csv"
    )
    if importance_file.exists():
        return pd.read_csv(importance_file)

    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is not None and hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": classifier.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
    if classifier is not None and hasattr(classifier, "coef_"):
        return pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": np.abs(classifier.coef_).ravel(),
            }
        ).sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["feature", "importance"])


def run_model_prediction(
    model_path: str | Path,
    input_path: str | Path,
    metrics_path: str | Path | None = None,
    feature_columns_path: str | Path | None = None,
    feature_importance_path: str | Path | None = None,
    top_n: int = 10,
    bullish_threshold: float = 0.6,
    bearish_threshold: float = 0.4,
) -> dict[str, Any]:
    """Load artifacts, select the latest row, and make one prediction."""
    model, metrics, feature_columns = load_model_bundle(
        model_path=model_path,
        metrics_path=metrics_path,
        feature_columns_path=feature_columns_path,
    )
    input_df = load_prediction_input(input_path)
    latest_row = select_latest_row(input_df)
    features = prepare_features(latest_row, feature_columns)

    predicted_class = int(model.predict(features)[0])
    probability = predict_positive_probability(model, features)
    signal = classify_model_signal(
        predicted_class=predicted_class,
        probability=probability,
        bullish_threshold=bullish_threshold,
        bearish_threshold=bearish_threshold,
    )

    importance_df = load_feature_importance(
        model=model,
        model_path=model_path,
        feature_columns=feature_columns,
        feature_importance_path=feature_importance_path,
    )
    top_importance = importance_df.head(top_n).to_dict(orient="records")

    row_info = {}
    for column in ["symbol", "date", "close"]:
        if column in latest_row.columns:
            value = latest_row[column].iloc[0]
            if hasattr(value, "strftime"):
                value = value.strftime("%Y-%m-%d")
            elif hasattr(value, "item"):
                value = value.item()
            row_info[column] = value

    return {
        "model_path": str(model_path),
        "input_path": str(input_path),
        "row_info": row_info,
        "predicted_probability": probability,
        "predicted_class": predicted_class,
        "model_signal": signal,
        "feature_count": len(feature_columns),
        "metrics": metrics,
        "top_feature_importance": top_importance,
    }
