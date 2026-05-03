import argparse
from pathlib import Path
import sys

import pandas as pd

from factor_builder import (
    build_factor_dataset,
    get_feature_columns,
    get_label_columns,
    validate_no_future_leakage,
)
from real_data_loader import fetch_a_share_daily_from_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a factor dataset for future ML research."
    )
    parser.add_argument(
        "--symbol",
        default="000001",
        help="A-share stock code without market prefix, for example 600519.",
    )
    parser.add_argument(
        "--source",
        choices=["akshare", "baostock", "auto", "csv"],
        default="csv",
        help="Data source to use. csv is offline and uses --input.",
    )
    parser.add_argument(
        "--input",
        default="data/real/000001.csv",
        help="Input CSV path used when --source csv.",
    )
    parser.add_argument(
        "--start",
        default="20240101",
        help="Start date in YYYYMMDD format for real data sources.",
    )
    parser.add_argument(
        "--end",
        default="20241231",
        help="End date in YYYYMMDD format for real data sources.",
    )
    parser.add_argument(
        "--adjust",
        default="qfq",
        help="Price adjustment mode for real data sources, for example qfq.",
    )
    parser.add_argument(
        "--output",
        default="data/factors/factors_000001.csv",
        help="Output CSV path for the factor dataset.",
    )
    return parser.parse_args()


def load_source_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "csv":
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
        return pd.read_csv(input_path)

    return fetch_a_share_daily_from_source(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
        source=args.source,
    )


def save_factor_dataset(factor_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = factor_df.copy()
    output_df["date"] = pd.to_datetime(output_df["date"]).dt.strftime("%Y-%m-%d")
    output_df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    try:
        source_df = load_source_data(args)
        factor_df = build_factor_dataset(source_df, symbol=args.symbol)
        feature_columns = get_feature_columns()
        label_columns = get_label_columns()
        validate_no_future_leakage(feature_columns)

        output_path = Path(args.output)
        save_factor_dataset(factor_df, output_path)
    except Exception as exc:
        print(f"Error: failed to build factor dataset: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Factor Dataset Builder")
    print("------------------------------------")
    print(f"Selected symbol: {args.symbol}")
    print(f"Selected source: {args.source}")
    if args.source == "csv":
        print(f"Input path: {args.input}")
    else:
        print(f"Date range: {args.start} to {args.end}")
        print(f"Adjust mode: {args.adjust}")
    print(f"Output path: {output_path}")
    print(f"Rows: {len(factor_df)}")
    print(f"Columns: {len(factor_df.columns)}")
    print(f"Feature column count: {len(feature_columns)}")
    print(f"Label column count: {len(label_columns)}")
    print()
    print(
        "Warning: This dataset is for research and future model training. "
        "It is not a trading recommendation."
    )
    print("No model was trained.")


if __name__ == "__main__":
    main()
