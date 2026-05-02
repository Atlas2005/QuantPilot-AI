import argparse
import sys

from backtester import run_long_only_backtest
from indicators import add_all_indicators
from metrics import summarize_performance
from real_data_loader import fetch_a_share_daily_from_source, save_stock_csv
from report_generator import generate_rule_based_report
from strategy import generate_ma_crossover_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch real A-share data and run a simple backtest."
    )
    parser.add_argument(
        "--symbol",
        default="000001",
        help="A-share stock code without market prefix, for example 600519.",
    )
    parser.add_argument(
        "--source",
        choices=["akshare", "baostock", "auto"],
        default="auto",
        help="Data source to use: akshare, baostock, or auto.",
    )
    parser.add_argument(
        "--start",
        default="20240101",
        help="Start date in YYYYMMDD format, for example 20240101.",
    )
    parser.add_argument(
        "--end",
        default="20241231",
        help="End date in YYYYMMDD format, for example 20241231.",
    )
    parser.add_argument(
        "--adjust",
        default="qfq",
        help="Price adjustment mode, for example qfq.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10000,
        help="Starting cash for the backtest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("QuantPilot-AI Stock Backtest")
    print("----------------------------")
    print(f"Selected symbol: {args.symbol}")
    print(f"Selected source: {args.source}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Adjust mode: {args.adjust}")
    print(f"Initial cash: {args.initial_cash:.2f}")
    print()

    try:
        stock_data = fetch_a_share_daily_from_source(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            adjust=args.adjust,
            source=args.source,
        )
    except Exception as exc:
        print(f"Error: failed to fetch data for {args.symbol}: {exc}")
        sys.exit(1)

    saved_path = save_stock_csv(stock_data, args.symbol)
    print(f"Saved CSV path: {saved_path}")
    print()

    stock_data = add_all_indicators(stock_data)
    stock_data = generate_ma_crossover_signals(stock_data)
    backtest_result = run_long_only_backtest(
        stock_data,
        initial_cash=args.initial_cash,
    )
    performance_summary = summarize_performance(backtest_result)
    report = generate_rule_based_report(performance_summary)

    print("Performance Summary")
    print("-------------------")
    for key, value in performance_summary.items():
        print(f"{key}: {value}")

    print()
    print(report)

    print()
    print("Last 10 Backtest Rows")
    print("---------------------")
    print(backtest_result.tail(10))


if __name__ == "__main__":
    main()
