import argparse
import sys

try:
    from .factor_ablation import (
        parse_ablation_modes,
        parse_model_types,
        run_and_save_factor_ablation,
    )
except ImportError:
    from factor_ablation import (
        parse_ablation_modes,
        parse_model_types,
        run_and_save_factor_ablation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run factor group and feature ablation diagnostics on a factor CSV.",
    )
    parser.add_argument("--input", required=True, help="Input factor CSV path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--target-col", default="label_up_5d")
    parser.add_argument("--models", default="logistic_regression,random_forest")
    parser.add_argument("--ablation-modes", default="drop_group,only_group")
    parser.add_argument("--purge-rows", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--symbol", help="Optional symbol label for output rows.")
    parser.add_argument("--max-drop-features", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model_types = parse_model_types(args.models)
        ablation_modes = parse_ablation_modes(args.ablation_modes)
        result = run_and_save_factor_ablation(
            input_path=args.input,
            output_dir=args.output_dir,
            target_col=args.target_col,
            model_types=model_types,
            ablation_modes=ablation_modes,
            purge_rows=args.purge_rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            symbol=args.symbol,
            max_drop_features=args.max_drop_features,
        )
    except Exception as exc:
        print(f"Error: factor ablation failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Factor Ablation")
    print("-----------------------------")
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target column: {args.target_col}")
    print(f"Models: {model_types}")
    print(f"Ablation modes: {ablation_modes}")
    print(f"Rows: {len(result['ablation_results'])}")
    print()
    print("Group Summary")
    print("-------------")
    if result["group_summary"].empty:
        print("No group summary rows.")
    else:
        print(result["group_summary"].to_string(index=False))
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: Factor ablation diagnostics are research tools. Good ML "
        "metrics do not guarantee profitable trading."
    )


if __name__ == "__main__":
    main()
