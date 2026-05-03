import argparse
import sys

try:
    from .factor_ablation import parse_model_types
    from .factor_pruning_experiment import parse_pruning_modes
    from .reduced_feature_backtest import run_and_save_reduced_feature_backtest
except ImportError:
    from factor_ablation import parse_model_types
    from factor_pruning_experiment import parse_pruning_modes
    from reduced_feature_backtest import run_and_save_reduced_feature_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ML signal backtests across reduced feature sets.",
    )
    parser.add_argument("--factor-csv", required=True)
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="logistic_regression,random_forest")
    parser.add_argument("--target-col", default="label_up_5d")
    parser.add_argument("--buy-threshold", type=float, default=0.60)
    parser.add_argument("--sell-threshold", type=float, default=0.50)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--commission-rate", type=float, default=0.0)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0)
    parser.add_argument("--minimum-commission", type=float, default=0.0)
    parser.add_argument(
        "--execution-mode",
        choices=["same_close", "next_open", "next_close"],
        default="same_close",
    )
    parser.add_argument(
        "--modes",
        default="full,drop_reduce_weight,keep_core_only,keep_core_and_observe",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model_types = parse_model_types(args.models)
        pruning_modes = parse_pruning_modes(args.modes)
        result = run_and_save_reduced_feature_backtest(
            factor_csv=args.factor_csv,
            recommendations_path=args.recommendations,
            output_dir=args.output_dir,
            model_types=model_types,
            pruning_modes=pruning_modes,
            target_col=args.target_col,
            initial_cash=args.initial_cash,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            execution_mode=args.execution_mode,
            commission_rate=args.commission_rate,
            stamp_tax_rate=args.stamp_tax_rate,
            slippage_pct=args.slippage_pct,
            min_commission=args.minimum_commission,
        )
    except Exception as exc:
        print(f"Error: reduced feature backtest failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Reduced Feature Backtest")
    print("--------------------------------------")
    print(f"Factor CSV: {args.factor_csv}")
    print(f"Recommendations: {args.recommendations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {model_types}")
    print(f"Modes: {pruning_modes}")
    print()
    print("Summary")
    print("-------")
    if result["summary"].empty:
        print("No summary rows.")
    else:
        print(result["summary"].to_string(index=False))
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: Reduced feature backtests are educational diagnostics only. "
        "They are not financial advice."
    )


if __name__ == "__main__":
    main()
