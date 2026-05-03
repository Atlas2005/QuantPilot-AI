import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_TARGET_COL = "label_up_5d"
MODEL_CHOICES = ("logistic_regression", "random_forest")


def load_split_datasets(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test CSV files from a split dataset directory."""
    path = Path(dataset_dir)
    train_path = path / "train.csv"
    validation_path = path / "validation.csv"
    test_path = path / "test.csv"

    missing = [
        str(csv_path)
        for csv_path in [train_path, validation_path, test_path]
        if not csv_path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing split dataset files: {missing}")

    read_kwargs = {"dtype": {"symbol": str}}
    return (
        pd.read_csv(train_path, **read_kwargs),
        pd.read_csv(validation_path, **read_kwargs),
        pd.read_csv(test_path, **read_kwargs),
    )


def load_feature_columns(dataset_dir: str | Path) -> list[str]:
    """Load model feature column names from feature_columns.txt."""
    feature_path = Path(dataset_dir) / "feature_columns.txt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature column file does not exist: {feature_path}")

    columns = [
        line.strip()
        for line in feature_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not columns:
        raise ValueError(f"No feature columns found in {feature_path}")

    return columns


def validate_training_inputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
) -> None:
    """Check that all required feature and target columns exist."""
    required_columns = feature_columns + [target_col]
    problems = {}
    for name, df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            problems[name] = missing

    if problems:
        raise ValueError(f"Missing required training columns: {problems}")


def _clean_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = df[feature_columns].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def _clean_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    target = pd.to_numeric(df[target_col], errors="coerce")
    return target.astype("Int64")


def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare features and target while dropping rows with missing targets.

    Feature missing and infinite values are left as NaN after conversion and are
    handled by the model pipeline's SimpleImputer.
    """
    x = _clean_features(df, feature_columns)
    y = _clean_target(df, target_col)
    keep_mask = y.notna()
    cleaned_df = df.loc[keep_mask].reset_index(drop=True)
    return x.loc[keep_mask].reset_index(drop=True), y.loc[keep_mask].astype(int).reset_index(drop=True), cleaned_df


def build_model(
    model_name: str,
    random_state: int = 42,
    single_class: bool = False,
) -> Pipeline:
    """Create a beginner-friendly sklearn classification pipeline."""
    if single_class:
        classifier = DummyClassifier(strategy="most_frequent")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", classifier),
            ]
        )

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    if model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", classifier),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}. Use one of {MODEL_CHOICES}.")


def train_baseline_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = DEFAULT_TARGET_COL,
    model_name: str = "random_forest",
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a baseline supervised classifier."""
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model: {model_name}. Use one of {MODEL_CHOICES}.")

    x_train, y_train, cleaned_train = prepare_xy(train_df, feature_columns, target_col)
    if len(cleaned_train) == 0:
        raise ValueError("Training split has no rows after target cleaning.")

    single_class = y_train.nunique(dropna=True) < 2
    model = build_model(model_name, random_state=random_state, single_class=single_class)
    model.fit(x_train, y_train)

    training_info = {
        "requested_model": model_name,
        "actual_model": "dummy_most_frequent" if single_class else model_name,
        "single_class_training": bool(single_class),
        "train_rows_used": int(len(cleaned_train)),
        "positive_rate_train": float(y_train.mean()) if len(y_train) else None,
    }
    return model, training_info


def _predict_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(x)
    if probabilities.ndim != 2:
        return None

    classifier = model.named_steps.get("classifier")
    classes = getattr(classifier, "classes_", [])
    if 1 in classes:
        positive_index = list(classes).index(1)
        return probabilities[:, positive_index]
    if probabilities.shape[1] >= 2:
        return probabilities[:, 1]
    return None


def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate a trained model and return metrics plus row-level predictions."""
    x, y_true, cleaned_df = prepare_xy(df, feature_columns, target_col)
    sample_count = len(cleaned_df)
    if sample_count == 0:
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "positive_rate": None,
            "sample_count": 0,
        }
        return metrics, pd.DataFrame()

    y_pred = model.predict(x)
    y_probability = _predict_probability(model, x)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "positive_rate": float(y_true.mean()),
        "sample_count": int(sample_count),
    }

    if y_probability is not None and y_true.nunique(dropna=True) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability))
        except ValueError:
            metrics["roc_auc"] = None

    prediction_df = pd.DataFrame()
    for column in ["symbol", "date"]:
        if column in cleaned_df.columns:
            prediction_df[column] = cleaned_df[column]
    prediction_df[target_col] = y_true
    prediction_df["prediction"] = y_pred
    if y_probability is not None:
        prediction_df["prediction_probability"] = y_probability

    return metrics, prediction_df


def extract_feature_importance(
    model: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return feature importance or coefficient magnitudes when available."""
    classifier = model.named_steps.get("classifier")
    if classifier is None:
        return pd.DataFrame()

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_).ravel()
    else:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)


def run_training_workflow(
    dataset_dir: str | Path,
    target_col: str = DEFAULT_TARGET_COL,
    model_name: str = "random_forest",
    output_dir: str | Path = "models/baseline",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and save a baseline ML model workflow."""
    train_df, validation_df, test_df = load_split_datasets(dataset_dir)
    feature_columns = load_feature_columns(dataset_dir)
    validate_training_inputs(
        train_df,
        validation_df,
        test_df,
        feature_columns,
        target_col,
    )

    model, training_info = train_baseline_model(
        train_df=train_df,
        feature_columns=feature_columns,
        target_col=target_col,
        model_name=model_name,
        random_state=random_state,
    )

    train_metrics, _ = evaluate_model(model, train_df, feature_columns, target_col)
    validation_metrics, validation_predictions = evaluate_model(
        model,
        validation_df,
        feature_columns,
        target_col,
    )
    test_metrics, test_predictions = evaluate_model(
        model,
        test_df,
        feature_columns,
        target_col,
    )

    output_files = save_model_outputs(
        model=model,
        output_dir=output_dir,
        model_name=model_name,
        feature_columns=feature_columns,
        metrics={
            "dataset_dir": str(dataset_dir),
            "target_col": target_col,
            "model": training_info,
            "feature_count": len(feature_columns),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        feature_importance=extract_feature_importance(model, feature_columns),
    )

    return {
        "feature_columns": feature_columns,
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_info": training_info,
        "output_files": output_files,
    }


def save_model_outputs(
    model: Pipeline,
    output_dir: str | Path,
    model_name: str,
    feature_columns: list[str],
    metrics: dict[str, Any],
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> dict[str, str | None]:
    """Save model artifact, metrics, predictions, and optional importance."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / f"{model_name}.joblib"
    metrics_path = output_path / "metrics.json"
    feature_columns_path = output_path / "feature_columns.txt"
    validation_path = output_path / "validation_predictions.csv"
    test_path = output_path / "test_predictions.csv"
    importance_path = output_path / "feature_importance.csv"

    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    feature_columns_path.write_text(
        "\n".join(feature_columns) + "\n",
        encoding="utf-8",
    )
    validation_predictions.to_csv(validation_path, index=False)
    test_predictions.to_csv(test_path, index=False)

    saved_importance_path = None
    if not feature_importance.empty:
        feature_importance.to_csv(importance_path, index=False)
        saved_importance_path = str(importance_path)

    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "feature_columns": str(feature_columns_path),
        "validation_predictions": str(validation_path),
        "test_predictions": str(test_path),
        "feature_importance": saved_importance_path,
    }
