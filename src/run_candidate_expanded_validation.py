import argparse
import sys

try:
    from .candidate_expanded_validation import (
        parse_symbols,
        save_candidate_expanded_validation,
    )
except ImportError:
    from candidate_expanded_validation import (
        parse_symbols,
        save_candidate_expanded_validation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run expanded validation for recommended threshold candidates.",
    )
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-col", default="label_up_5d")
    parser.add_argument("--candidate-pruning-mode", default="keep_core_and_observe")
    parser.add_argument("--candidate-model", default="logistic_regression")
    parser.add_argument("--candidate-buy-threshold", type=float, default=0.50)
    parser.add_argument("--candidate-sell-threshold", type=float, default=0.40)
    parser.add_argument("--walk-forward-pruning-mode", default="drop_reduce_weight")
    parser.add_argument("--walk-forward-model", default="logistic_regression")
    parser.add_argument("--walk-forward-buy-threshold", type=float, default=0.50)
    parser.add_argument("--walk-forward-sell-threshold", type=float, default=0.40)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.001)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--min-trades", type=int, default=3)
    parser.add_argument("--enable-walk-forward", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        symbols = parse_symbols(args.symbols)
        result = save_candidate_expanded_validation(
            factor_dir=args.factor_dir,
            symbols=symbols,
            recommendations_path=args.recommendations,
            output_dir=args.output_dir,
            target_col=args.target_col,
            candidate_pruning_mode=args.candidate_pruning_mode,
            candidate_model=args.candidate_model,
            candidate_buy_threshold=args.candidate_buy_threshold,
            candidate_sell_threshold=args.candidate_sell_threshold,
            walk_forward_pruning_mode=args.walk_forward_pruning_mode,
            walk_forward_model=args.walk_forward_model,
            walk_forward_buy_threshold=args.walk_forward_buy_threshold,
            walk_forward_sell_threshold=args.walk_forward_sell_threshold,
            commission_rate=args.commission_rate,
            stamp_tax_rate=args.stamp_tax_rate,
            slippage_pct=args.slippage_pct,
            min_commission=args.minimum_commission,
            min_trades=args.min_trades,
            enable_walk_forward=args.enable_walk_forward,
        )
    except Exception as exc:
        print(f"Error: candidate expanded validation failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Candidate Expanded Validation")
    print("-------------------------------------------")
    print(f"Factor directory: {args.factor_dir}")
    print(f"Symbols: {symbols}")
    print(f"Recommendations: {args.recommendations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Walk-forward enabled: {args.enable_walk_forward}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: This is educational research only. "
        "It is not trading-ready and is not financial advice."
    )


if __name__ == "__main__":
    main()
