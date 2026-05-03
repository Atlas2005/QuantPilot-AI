import argparse
import sys

try:
    from .reduced_feature_backtest_report import (
        DEFAULT_INPUT_DIRS,
        parse_input_dirs,
        save_reduced_feature_backtest_report,
    )
except ImportError:
    from reduced_feature_backtest_report import (
        DEFAULT_INPUT_DIRS,
        parse_input_dirs,
        save_reduced_feature_backtest_report,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-symbol reduced feature backtest report.",
    )
    parser.add_argument(
        "--input-dirs",
        default=",".join(DEFAULT_INPUT_DIRS),
        help="Comma-separated Step 21 reduced feature backtest output directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reduced_feature_backtest_summary_real_v1",
        help="Output directory for aggregate CSVs and Markdown report.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=3,
        help="Trade count threshold used to flag low-trade-count cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        input_dirs = parse_input_dirs(args.input_dirs)
        result = save_reduced_feature_backtest_report(
            input_dirs=input_dirs,
            output_dir=args.output_dir,
            min_trades=args.min_trades,
        )
    except Exception as exc:
        print(f"Error: failed to generate reduced feature backtest report: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Reduced Feature Backtest Summary")
    print("----------------------------------------------")
    print(f"Input directories: {input_dirs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum trades warning threshold: {args.min_trades}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: This is educational research only. "
        "It is not a trading recommendation."
    )


if __name__ == "__main__":
    main()
