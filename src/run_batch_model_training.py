import argparse
import sys

from batch_model_trainer import (
    parse_model_types,
    parse_symbols,
    run_batch_model_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-symbol baseline model robustness training."
    )
    parser.add_argument("--symbols", default="000001,600519")
    parser.add_argument("--source", choices=["demo", "baostock"], default="demo")
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--output-dir", default="outputs/model_robustness_demo")
    parser.add_argument(
        "--models",
        default="logistic_regression,random_forest",
        help="Comma-separated model types.",
    )
    parser.add_argument("--target-col", default="label_up_5d")
    parser.add_argument("--purge-rows", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=["global_date", "per_symbol"],
        default="global_date",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        symbols = parse_symbols(args.symbols)
        model_types = parse_model_types(args.models)
    except Exception as exc:
        print(f"Error: invalid arguments: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Batch Model Training")
    print("----------------------------------")
    print(f"Selected symbols: {symbols}")
    print(f"Selected models: {model_types}")
    print(f"Source mode: {args.source}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Output directory: {args.output_dir}")
    print()

    for symbol in symbols:
        print(f"Processing symbol: {symbol}")

    try:
        result = run_batch_model_training(
            symbols=symbols,
            model_types=model_types,
            source=args.source,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            target_col=args.target_col,
            purge_rows=args.purge_rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_mode=args.split_mode,
        )
    except Exception as exc:
        print(f"Error: batch model training failed: {exc}")
        sys.exit(1)

    print()
    print("Final Summary")
    print("-------------")
    print(result["model_summary"].to_string(index=False))
    print()
    print("Model Ranking")
    print("-------------")
    print(result["model_ranking"].to_string(index=False))
    print()
    print("Warnings Summary")
    print("----------------")
    if result["warnings"].empty:
        print("No warnings.")
    else:
        print(result["warnings"].to_string(index=False))
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: High ML metrics do not guarantee profitable trading. "
        "This is educational robustness research only."
    )


if __name__ == "__main__":
    main()
