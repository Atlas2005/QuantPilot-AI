import argparse
import sys

try:
    from .factor_ablation import parse_model_types
    from .factor_pruning_experiment import parse_pruning_modes
    from .reduced_feature_threshold_experiment import (
        DEFAULT_BUY_THRESHOLDS,
        DEFAULT_MODELS,
        DEFAULT_SELL_THRESHOLDS,
        parse_thresholds,
        run_and_save_threshold_experiment,
    )
except ImportError:
    from factor_ablation import parse_model_types
    from factor_pruning_experiment import parse_pruning_modes
    from reduced_feature_threshold_experiment import (
        DEFAULT_BUY_THRESHOLDS,
        DEFAULT_MODELS,
        DEFAULT_SELL_THRESHOLDS,
        parse_thresholds,
        run_and_save_threshold_experiment,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reduced feature threshold sensitivity backtests.",
    )
    parser.add_argument("--factor-csv", required=True)
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--target-col", default="label_up_5d")
    parser.add_argument("--buy-thresholds", default="0.50,0.55,0.60,0.65")
    parser.add_argument("--sell-thresholds", default="0.35,0.40,0.45,0.50")
    parser.add_argument(
        "--pruning-modes",
        default="full,drop_reduce_weight,keep_core_only,keep_core_and_observe",
    )
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--execution-mode", default="same_close")
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.001)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--min-trades", type=int, default=3)
    parser.add_argument("--enable-walk-forward", action="store_true")
    parser.add_argument("--walk-forward-train-ratio", type=float, default=0.50)
    parser.add_argument("--walk-forward-validation-ratio", type=float, default=0.20)
    parser.add_argument("--walk-forward-test-ratio", type=float, default=0.20)
    parser.add_argument("--walk-forward-step-ratio", type=float, default=0.10)
    parser.add_argument("--purge-rows", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model_types = parse_model_types(args.models)
        pruning_modes = parse_pruning_modes(args.pruning_modes)
        buy_thresholds = parse_thresholds(args.buy_thresholds, DEFAULT_BUY_THRESHOLDS)
        sell_thresholds = parse_thresholds(args.sell_thresholds, DEFAULT_SELL_THRESHOLDS)
        result = run_and_save_threshold_experiment(
            factor_csv=args.factor_csv,
            recommendations_path=args.recommendations,
            output_dir=args.output_dir,
            model_types=model_types,
            pruning_modes=pruning_modes,
            target_col=args.target_col,
            buy_thresholds=buy_thresholds,
            sell_thresholds=sell_thresholds,
            initial_cash=args.initial_cash,
            execution_mode=args.execution_mode,
            commission_rate=args.commission_rate,
            stamp_tax_rate=args.stamp_tax_rate,
            slippage_pct=args.slippage_pct,
            min_commission=args.minimum_commission,
            min_trades=args.min_trades,
            enable_walk_forward=args.enable_walk_forward,
            walk_forward_train_ratio=args.walk_forward_train_ratio,
            walk_forward_validation_ratio=args.walk_forward_validation_ratio,
            walk_forward_test_ratio=args.walk_forward_test_ratio,
            walk_forward_step_ratio=args.walk_forward_step_ratio,
            purge_rows=args.purge_rows,
        )
    except Exception as exc:
        print(f"Error: reduced feature threshold experiment failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Reduced Feature Threshold Experiment")
    print("--------------------------------------------------")
    print(f"Factor CSV: {args.factor_csv}")
    print(f"Recommendations: {args.recommendations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {model_types}")
    print(f"Pruning modes: {pruning_modes}")
    print(f"Buy thresholds: {buy_thresholds}")
    print(f"Sell thresholds: {sell_thresholds}")
    print(f"Rows: {len(result['threshold_results'])}")
    print(f"Walk-forward enabled: {args.enable_walk_forward}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: Threshold tuning can overfit historical data. "
        "This is research only, not financial advice."
    )


if __name__ == "__main__":
    main()
