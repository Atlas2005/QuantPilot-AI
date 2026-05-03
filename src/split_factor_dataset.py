import argparse
import sys

from dataset_splitter import (
    DEFAULT_TARGET_COL,
    build_split_report,
    check_for_leakage_columns,
    chronological_split,
    clean_factor_dataset,
    duplicate_symbol_date_count,
    infer_feature_columns,
    load_factor_dataset,
    normalize_date_column,
    save_split_outputs,
    validate_required_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a factor dataset into chronological ML datasets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input factor CSV path, for example data/factors/factors_000001.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for train/validation/test CSVs.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help="Target label column to predict. Defaults to label_up_5d.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Chronological training ratio. Defaults to 0.6.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Chronological validation ratio. Defaults to 0.2.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Chronological test ratio. Defaults to 0.2.",
    )
    parser.add_argument(
        "--purge-rows",
        type=int,
        default=5,
        help="Rows removed before each split boundary to reduce label leakage.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["global_date", "per_symbol"],
        default="global_date",
        help="Split globally by date order or independently per symbol.",
    )
    return parser.parse_args()


def _format_date_range(split_report: dict, prefix: str) -> str:
    date_min = split_report[f"{prefix}_date_min"]
    date_max = split_report[f"{prefix}_date_max"]
    if date_min is None or date_max is None:
        return "empty"
    return f"{date_min} to {date_max}"


def main() -> None:
    args = parse_args()

    try:
        raw_df = load_factor_dataset(args.input)
        original_rows = len(raw_df)
        factor_df = normalize_date_column(raw_df)
        validate_required_columns(factor_df, args.target_col)

        duplicate_count = duplicate_symbol_date_count(factor_df)
        feature_columns = infer_feature_columns(factor_df, args.target_col)
        leakage_checks = check_for_leakage_columns(feature_columns, args.target_col)
        if not leakage_checks["passed"]:
            raise ValueError(
                "Leakage check failed. Bad feature columns: "
                f"{leakage_checks['leakage_columns']}"
            )

        cleaned_df, missing_values = clean_factor_dataset(
            factor_df,
            feature_columns,
            args.target_col,
        )

        train_df, validation_df, test_df = chronological_split(
            cleaned_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            purge_rows=args.purge_rows,
            split_mode=args.split_mode,
        )

        split_report = build_split_report(
            input_path=args.input,
            target_col=args.target_col,
            feature_columns=feature_columns,
            original_rows=original_rows,
            cleaned_rows=len(cleaned_df),
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            purge_rows=args.purge_rows,
            split_mode=args.split_mode,
            leakage_checks=leakage_checks,
            missing_values=missing_values,
            duplicate_count=duplicate_count,
        )

        output_files = save_split_outputs(
            output_dir=args.output_dir,
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            feature_columns=feature_columns,
            split_report=split_report,
        )
    except Exception as exc:
        print(f"Error: failed to split factor dataset: {exc}")
        sys.exit(1)

    print("QuantPilot-AI ML Dataset Splitter")
    print("---------------------------------")
    print(f"Input path: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target column: {args.target_col}")
    print(f"Feature columns ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f"Original row count: {original_rows}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train date range: {_format_date_range(split_report, 'train')}")
    print(f"Validation date range: {_format_date_range(split_report, 'validation')}")
    print(f"Test date range: {_format_date_range(split_report, 'test')}")
    print(f"Leakage check result: {split_report['leakage_checks']}")
    print(f"Duplicate symbol/date rows: {duplicate_count}")
    print("Output files created:")
    for label, path in output_files.items():
        print(f"- {label}: {path}")
    print()
    print("No model was trained.")


if __name__ == "__main__":
    main()
