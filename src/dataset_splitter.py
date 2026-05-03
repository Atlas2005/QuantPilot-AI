import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TARGET_COL = "label_up_5d"
EXCLUDED_FEATURE_WORDS = ("future", "label", "target")
IDENTIFIER_COLUMNS = ("date", "symbol")


def load_factor_dataset(input_path: str | Path) -> pd.DataFrame:
    """Load a factor CSV from disk."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Factor dataset does not exist: {path}")

    return pd.read_csv(path, dtype={"symbol": str})


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the date column to datetime and sort rows chronologically."""
    if "date" not in df.columns:
        raise ValueError("Factor dataset must contain a date column.")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        bad_count = int(result["date"].isna().sum())
        raise ValueError(f"Date column contains {bad_count} invalid values.")

    sort_columns = ["date"]
    if "symbol" in result.columns:
        sort_columns.append("symbol")

    return result.sort_values(sort_columns).reset_index(drop=True)


def validate_required_columns(df: pd.DataFrame, target_col: str) -> None:
    """Validate columns required for safe ML dataset splitting."""
    missing = [column for column in ["date", "symbol", target_col] if column not in df]
    if missing:
        raise ValueError(f"Factor dataset is missing required columns: {missing}")


def infer_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Infer model input columns by excluding identifiers and future-looking names.

    Any column containing future, label, or target is excluded to reduce leakage
    risk. The target column is also explicitly excluded.
    """
    feature_cols = []
    for column in df.columns:
        lower_column = column.lower()
        if column == target_col:
            continue
        if lower_column in IDENTIFIER_COLUMNS:
            continue
        if any(word in lower_column for word in EXCLUDED_FEATURE_WORDS):
            continue
        feature_cols.append(column)

    if not feature_cols:
        raise ValueError("No feature columns were inferred from the factor dataset.")

    return feature_cols


def check_for_leakage_columns(feature_cols: list[str], target_col: str) -> dict[str, Any]:
    """Check that selected feature columns do not contain target/leakage names."""
    bad_columns = []
    for column in feature_cols:
        lower_column = column.lower()
        if column == target_col or any(word in lower_column for word in EXCLUDED_FEATURE_WORDS):
            bad_columns.append(column)
        elif lower_column in IDENTIFIER_COLUMNS:
            bad_columns.append(column)

    passed = not bad_columns and target_col not in feature_cols
    return {
        "passed": passed,
        "target_not_in_features": target_col not in feature_cols,
        "leakage_columns": bad_columns,
        "excluded_name_fragments": list(EXCLUDED_FEATURE_WORDS),
        "identifier_columns_excluded": list(IDENTIFIER_COLUMNS),
    }


def _validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    invalid = {name: value for name, value in ratios.items() if value <= 0}
    if invalid:
        raise ValueError(f"Split ratios must be positive: {invalid}")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.000001:
        raise ValueError(
            "Split ratios must add up to 1.0. "
            f"Got train+validation+test={total:.6f}."
        )


def _split_one_frame(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    purge_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_count = len(df)
    train_end = int(row_count * train_ratio)
    validation_end = int(row_count * (train_ratio + val_ratio))

    train_keep_end = max(0, train_end - purge_rows)
    validation_keep_end = max(train_end, validation_end - purge_rows)

    train_df = df.iloc[:train_keep_end].copy()
    validation_df = df.iloc[train_end:validation_keep_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def _split_by_global_dates(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    purge_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = pd.Series(df["date"].drop_duplicates()).sort_values().reset_index(
        drop=True
    )
    date_count = len(unique_dates)
    train_end = int(date_count * train_ratio)
    validation_end = int(date_count * (train_ratio + val_ratio))

    train_keep_end = max(0, train_end - purge_rows)
    validation_keep_end = max(train_end, validation_end - purge_rows)

    train_dates = set(unique_dates.iloc[:train_keep_end])
    validation_dates = set(unique_dates.iloc[train_end:validation_keep_end])
    test_dates = set(unique_dates.iloc[validation_end:])

    train_df = df[df["date"].isin(train_dates)].copy()
    validation_df = df[df["date"].isin(validation_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    return train_df, validation_df, test_df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    purge_rows: int = 5,
    split_mode: str = "global_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split rows chronologically without shuffling.

    purge_rows removes the final N rows before each split boundary. This helps
    reduce label-window leakage for labels such as future_return_5d/label_up_5d.
    """
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    if purge_rows < 0:
        raise ValueError("purge_rows must be zero or greater.")
    if split_mode not in ("global_date", "per_symbol"):
        raise ValueError("split_mode must be either global_date or per_symbol.")

    sorted_df = normalize_date_column(df)

    if split_mode == "global_date":
        return _split_by_global_dates(sorted_df, train_ratio, val_ratio, purge_rows)

    train_parts = []
    validation_parts = []
    test_parts = []
    for _, symbol_df in sorted_df.groupby("symbol", sort=True):
        train_df, validation_df, test_df = _split_one_frame(
            symbol_df.reset_index(drop=True),
            train_ratio,
            val_ratio,
            purge_rows,
        )
        train_parts.append(train_df)
        validation_parts.append(validation_df)
        test_parts.append(test_df)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    validation_df = (
        pd.concat(validation_parts, ignore_index=True)
        if validation_parts
        else pd.DataFrame()
    )
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    return (
        normalize_date_column(train_df) if not train_df.empty else train_df,
        normalize_date_column(validation_df) if not validation_df.empty else validation_df,
        normalize_date_column(test_df) if not test_df.empty else test_df,
    )


def missing_value_report(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Count missing values for selected columns."""
    return {column: int(df[column].isna().sum()) for column in columns}


def duplicate_symbol_date_count(df: pd.DataFrame) -> int:
    """Count duplicate symbol/date rows."""
    if "symbol" not in df.columns or "date" not in df.columns:
        return 0
    return int(df.duplicated(subset=["symbol", "date"]).sum())


def clean_factor_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows with missing selected features or target values."""
    required_for_training = feature_cols + [target_col]
    report = missing_value_report(df, required_for_training)
    cleaned_df = df.dropna(subset=required_for_training).reset_index(drop=True)
    return cleaned_df, report


def _date_min(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d")


def _date_max(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")


def _date_ranges_do_not_overlap(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    ranges = [
        (_date_min(train_df), _date_max(train_df)),
        (_date_min(validation_df), _date_max(validation_df)),
        (_date_min(test_df), _date_max(test_df)),
    ]
    if any(start is None or end is None for start, end in ranges):
        return False

    train_end = pd.to_datetime(ranges[0][1])
    validation_start = pd.to_datetime(ranges[1][0])
    validation_end = pd.to_datetime(ranges[1][1])
    test_start = pd.to_datetime(ranges[2][0])

    return train_end < validation_start and validation_end < test_start


def build_split_report(
    input_path: str | Path,
    target_col: str,
    feature_columns: list[str],
    original_rows: int,
    cleaned_rows: int,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    purge_rows: int,
    split_mode: str,
    leakage_checks: dict[str, Any],
    missing_values: dict[str, int],
    duplicate_count: int,
) -> dict[str, Any]:
    """Build a JSON-serializable report for the split outputs."""
    date_ranges_ok = _date_ranges_do_not_overlap(train_df, validation_df, test_df)
    leakage_report = dict(leakage_checks)
    leakage_report["date_ranges_do_not_overlap"] = date_ranges_ok

    return {
        "input_path": str(input_path),
        "target_col": target_col,
        "feature_columns": feature_columns,
        "original_rows": int(original_rows),
        "cleaned_rows": int(cleaned_rows),
        "dropped_rows": int(original_rows - cleaned_rows),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": _date_min(train_df),
        "train_date_max": _date_max(train_df),
        "validation_date_min": _date_min(validation_df),
        "validation_date_max": _date_max(validation_df),
        "test_date_min": _date_min(test_df),
        "test_date_max": _date_max(test_df),
        "purge_rows": int(purge_rows),
        "split_mode": split_mode,
        "leakage_checks": leakage_report,
        "missing_value_report": missing_values,
        "duplicate_symbol_date_count": int(duplicate_count),
    }


def _prepare_csv_for_save(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    if "date" in output_df.columns:
        output_df["date"] = pd.to_datetime(output_df["date"]).dt.strftime("%Y-%m-%d")
    return output_df


def save_split_outputs(
    output_dir: str | Path,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    split_report: dict[str, Any],
) -> dict[str, str]:
    """Save train/validation/test CSVs, feature list, and split report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.csv"
    validation_path = output_path / "validation.csv"
    test_path = output_path / "test.csv"
    feature_path = output_path / "feature_columns.txt"
    report_path = output_path / "split_report.json"

    _prepare_csv_for_save(train_df).to_csv(train_path, index=False)
    _prepare_csv_for_save(validation_df).to_csv(validation_path, index=False)
    _prepare_csv_for_save(test_df).to_csv(test_path, index=False)
    feature_path.write_text("\n".join(feature_columns) + "\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(split_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "train": str(train_path),
        "validation": str(validation_path),
        "test": str(test_path),
        "feature_columns": str(feature_path),
        "split_report": str(report_path),
    }
