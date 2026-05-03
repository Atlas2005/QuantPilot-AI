import argparse
import sys

try:
    from .factor_ablation import parse_model_types
    from .factor_pruning_experiment import (
        parse_pruning_modes,
        run_and_save_factor_pruning_experiment,
    )
except ImportError:
    from factor_ablation import parse_model_types
    from factor_pruning_experiment import (
        parse_pruning_modes,
        run_and_save_factor_pruning_experiment,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reduced feature set experiments from pruning recommendations.",
    )
    parser.add_argument("--factor-csv", required=True, help="Input factor CSV.")
    parser.add_argument(
        "--recommendations",
        required=True,
        help="feature_pruning_recommendations.csv path.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="logistic_regression,random_forest")
    parser.add_argument(
        "--pruning-modes",
        default="full,drop_reduce_weight,keep_core_only,keep_core_and_observe",
    )
    parser.add_argument("--target-col", default="label_up_5d")
    parser.add_argument("--purge-rows", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model_types = parse_model_types(args.models)
        pruning_modes = parse_pruning_modes(args.pruning_modes)
        result = run_and_save_factor_pruning_experiment(
            factor_csv=args.factor_csv,
            recommendations_path=args.recommendations,
            output_dir=args.output_dir,
            model_types=model_types,
            pruning_modes=pruning_modes,
            target_col=args.target_col,
            purge_rows=args.purge_rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    except Exception as exc:
        print(f"Error: factor pruning experiment failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Factor Pruning Experiment")
    print("---------------------------------------")
    print(f"Factor CSV: {args.factor_csv}")
    print(f"Recommendations: {args.recommendations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {model_types}")
    print(f"Pruning modes: {pruning_modes}")
    print()
    print("Pruning Summary")
    print("---------------")
    if result["pruning_summary"].empty:
        print("No pruning summary rows.")
    else:
        print(result["pruning_summary"].to_string(index=False))
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: Reduced feature experiments are research diagnostics only. "
        "They are not financial advice."
    )


if __name__ == "__main__":
    main()
