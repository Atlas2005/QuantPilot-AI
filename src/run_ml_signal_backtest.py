import argparse
import sys

from ml_signal_backtester import run_ml_signal_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a long/flat backtest from saved ML prediction probabilities."
    )
    parser.add_argument(
        "--model-dir",
        default="models/demo_000001",
        help="Directory containing a trained .joblib model, metrics.json, and feature_columns.txt.",
    )
    parser.add_argument(
        "--factor-csv",
        required=True,
        help="Factor CSV or ML split CSV containing OHLCV and feature columns.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10000.0,
        help="Starting cash for the backtest.",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.60,
        help="Open long position when probability is at or above this value.",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=0.50,
        help="Close long position when probability falls below this value.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["same_close", "next_open", "next_close"],
        default="same_close",
        help="Trade execution mode used by the existing backtester.",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.0,
        help="Commission rate per trade, for example 0.0003.",
    )
    parser.add_argument(
        "--stamp-tax-rate",
        type=float,
        default=0.0,
        help="Stamp tax rate applied to sell trades only.",
    )
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.0,
        help="Slippage percentage applied to execution prices.",
    )
    parser.add_argument(
        "--min-commission",
        type=float,
        default=0.0,
        help="Minimum commission charged per trade.",
    )
    parser.add_argument(
        "--no-rule-based-comparison",
        action="store_true",
        help="Skip comparison against the existing MA crossover strategy.",
    )
    return parser.parse_args()


def print_dict(title: str, values: dict) -> None:
    print(title)
    print("-" * len(title))
    for key, value in values.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print()


def main() -> None:
    args = parse_args()

    try:
        result = run_ml_signal_backtest(
            model_dir=args.model_dir,
            factor_csv=args.factor_csv,
            initial_cash=args.initial_cash,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            execution_mode=args.execution_mode,
            commission_rate=args.commission_rate,
            stamp_tax_rate=args.stamp_tax_rate,
            slippage_pct=args.slippage_pct,
            min_commission=args.min_commission,
            compare_rule_based=not args.no_rule_based_comparison,
        )
    except Exception as exc:
        print(f"Error: failed to run ML signal backtest: {exc}")
        sys.exit(1)

    print("QuantPilot-AI ML Signal Backtest")
    print("--------------------------------")
    print(f"Model directory: {args.model_dir}")
    print(f"Model path: {result['metadata']['model_path']}")
    print(f"Factor CSV: {args.factor_csv}")
    print(f"Feature count: {result['metadata']['feature_count']}")
    print(f"Initial cash: {args.initial_cash:.2f}")
    print(f"Buy threshold: {args.buy_threshold:.2f}")
    print(f"Sell threshold: {args.sell_threshold:.2f}")
    print(f"Execution mode: {args.execution_mode}")
    print()

    print_dict("ML Strategy Performance", result["performance"])
    print_dict("Trade Metrics", result["trade_metrics"])
    print_dict("Buy-and-Hold Benchmark", result["benchmark"])

    rule_based = result.get("rule_based_comparison")
    if rule_based and rule_based.get("available"):
        print_dict("Rule-Based Strategy Comparison", rule_based["performance"])
    elif rule_based:
        print(f"Rule-based comparison unavailable: {rule_based.get('reason')}")
        print()

    print("Trade Log")
    print("---------")
    if result["trades"].empty:
        print("No trades were executed.")
    else:
        print(result["trades"].to_string(index=False))
    print()

    print("Probability / Signal Preview")
    print("----------------------------")
    preview_cols = ["date", "close", "prediction_probability", "signal"]
    print(result["signal_data"][preview_cols].tail(10).to_string(index=False))
    print()

    for warning in result["warnings"]:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
