import argparse
from pathlib import Path
import sys

from ml_threshold_experiment import (
    DEFAULT_BUY_THRESHOLDS,
    DEFAULT_SELL_THRESHOLDS,
    parse_thresholds,
    rank_threshold_results,
    run_threshold_experiment,
    run_walk_forward_threshold_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ML threshold and walk-forward research experiments."
    )
    parser.add_argument(
        "--model-dir",
        default="models/demo_000001",
        help="Directory containing a trained .joblib model and feature_columns.txt.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Factor CSV or ML split CSV containing OHLCV and model feature columns.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path, for example outputs/ml_threshold_experiment.csv.",
    )
    parser.add_argument(
        "--buy-thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_BUY_THRESHOLDS),
        help="Comma-separated buy thresholds.",
    )
    parser.add_argument(
        "--sell-thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_SELL_THRESHOLDS),
        help="Comma-separated sell thresholds.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10000.0,
        help="Starting cash for each threshold backtest.",
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
        help="Commission rate per trade.",
    )
    parser.add_argument(
        "--stamp-tax-rate",
        type=float,
        default=0.0,
        help="Stamp tax rate applied to sell trades.",
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
        "--walk-forward",
        action="store_true",
        help="Retrain and test across rolling chronological windows.",
    )
    parser.add_argument(
        "--target-col",
        default="label_up_5d",
        help="Target column used for walk-forward retraining.",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest"],
        default="random_forest",
        help="Model type used for walk-forward retraining.",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=120,
        help="Walk-forward training window row count.",
    )
    parser.add_argument(
        "--test-window",
        type=int,
        default=40,
        help="Walk-forward test window row count.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=40,
        help="Walk-forward window step size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for walk-forward model retraining.",
    )
    return parser.parse_args()


def print_best_rows(results_df) -> None:
    if results_df.empty:
        print("No threshold results were produced.")
        return

    ranked = rank_threshold_results(results_df)
    best_score = ranked.iloc[0]
    best_return = results_df.sort_values(
        "total_return_pct",
        ascending=False,
    ).iloc[0]
    best_drawdown = results_df.sort_values(
        "max_drawdown_pct",
        ascending=False,
    ).iloc[0]

    def print_row(title: str, row) -> None:
        print(title)
        print("-" * len(title))
        print(f"buy_threshold: {row['buy_threshold']:.2f}")
        print(f"sell_threshold: {row['sell_threshold']:.2f}")
        print(f"total_return_pct: {row['total_return_pct']:.4f}")
        print(f"max_drawdown_pct: {row['max_drawdown_pct']:.4f}")
        print(f"strategy_vs_benchmark_pct: {row['strategy_vs_benchmark_pct']:.4f}")
        print(f"score: {row['score']:.4f}")
        print()

    print_row("Best By Score", best_score)
    print_row("Best Total Return", best_return)
    print_row("Best Drawdown Control", best_drawdown)


def main() -> None:
    args = parse_args()

    try:
        buy_thresholds = parse_thresholds(args.buy_thresholds, DEFAULT_BUY_THRESHOLDS)
        sell_thresholds = parse_thresholds(args.sell_thresholds, DEFAULT_SELL_THRESHOLDS)
        if args.walk_forward:
            results_df = run_walk_forward_threshold_experiment(
                model_dir=args.model_dir,
                input_path=args.input,
                target_col=args.target_col,
                model_name=args.model,
                buy_thresholds=buy_thresholds,
                sell_thresholds=sell_thresholds,
                train_window=args.train_window,
                test_window=args.test_window,
                step_size=args.step_size,
                initial_cash=args.initial_cash,
                execution_mode=args.execution_mode,
                commission_rate=args.commission_rate,
                stamp_tax_rate=args.stamp_tax_rate,
                slippage_pct=args.slippage_pct,
                min_commission=args.min_commission,
                random_state=args.random_state,
            )
        else:
            results_df = run_threshold_experiment(
                model_dir=args.model_dir,
                input_path=args.input,
                buy_thresholds=buy_thresholds,
                sell_thresholds=sell_thresholds,
                initial_cash=args.initial_cash,
                execution_mode=args.execution_mode,
                commission_rate=args.commission_rate,
                stamp_tax_rate=args.stamp_tax_rate,
                slippage_pct=args.slippage_pct,
                min_commission=args.min_commission,
            )
    except Exception as exc:
        print(f"Error: failed to run ML threshold experiment: {exc}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

    print("QuantPilot-AI ML Threshold Experiment")
    print("-------------------------------------")
    print(f"Model directory: {args.model_dir}")
    print(f"Input path: {args.input}")
    print(f"Walk-forward: {args.walk_forward}")
    print(f"Buy thresholds: {buy_thresholds}")
    print(f"Sell thresholds: {sell_thresholds}")
    print(f"Rows: {len(results_df)}")
    if args.output:
        print(f"Output path: {args.output}")
    print()
    print_best_rows(results_df)
    print("Top Results")
    print("-----------")
    if results_df.empty:
        print("No rows.")
    else:
        print(rank_threshold_results(results_df).head(10).to_string(index=False))
    print()
    print(
        "Warning: This is threshold research only. Optimizing thresholds on "
        "past data can overfit and does not guarantee future profitability."
    )


if __name__ == "__main__":
    main()
