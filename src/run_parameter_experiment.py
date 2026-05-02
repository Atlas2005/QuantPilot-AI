import argparse
import sys

import pandas as pd

from backtester import run_long_only_backtest_with_trades
from indicators import add_all_indicators
from metrics import summarize_performance
from real_data_loader import fetch_a_share_daily_from_source
from strategy import generate_ma_crossover_signals
from trade_metrics import summarize_trade_metrics


SCENARIOS = [
    {
        "scenario": "baseline",
        "stop_loss_pct": None,
        "take_profit_pct": None,
        "max_holding_days": None,
    },
    {
        "scenario": "sl_3",
        "stop_loss_pct": 3,
        "take_profit_pct": None,
        "max_holding_days": None,
    },
    {
        "scenario": "sl_5",
        "stop_loss_pct": 5,
        "take_profit_pct": None,
        "max_holding_days": None,
    },
    {
        "scenario": "tp_10",
        "stop_loss_pct": None,
        "take_profit_pct": 10,
        "max_holding_days": None,
    },
    {
        "scenario": "max_30",
        "stop_loss_pct": None,
        "take_profit_pct": None,
        "max_holding_days": 30,
    },
    {
        "scenario": "sl_3_tp_10",
        "stop_loss_pct": 3,
        "take_profit_pct": 10,
        "max_holding_days": None,
    },
    {
        "scenario": "sl_3_max_30",
        "stop_loss_pct": 3,
        "take_profit_pct": None,
        "max_holding_days": 30,
    },
    {
        "scenario": "sl_3_tp_10_max_30",
        "stop_loss_pct": 3,
        "take_profit_pct": 10,
        "max_holding_days": 30,
    },
]


RESULT_COLUMNS = [
    "scenario",
    "stop_loss_pct",
    "take_profit_pct",
    "max_holding_days",
    "final_value",
    "total_return_pct",
    "max_drawdown_pct",
    "total_trades",
    "closed_trades",
    "open_trades",
    "win_rate_pct",
    "profit_factor",
    "average_return_pct",
    "best_trade_return_pct",
    "worst_trade_return_pct",
    "average_holding_days",
    "currently_holding",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare risk-control settings for one A-share backtest."
    )
    parser.add_argument(
        "--symbol",
        default="000001",
        help="A-share stock code without market prefix, for example 600519.",
    )
    parser.add_argument(
        "--source",
        choices=["akshare", "baostock", "auto"],
        default="baostock",
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
        default=10000.0,
        help="Starting cash for each backtest scenario.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for saving raw experiment results.",
    )
    return parser.parse_args()


def run_scenario(prepared_data: pd.DataFrame, initial_cash: float, scenario: dict) -> dict:
    backtest_df, trades_df = run_long_only_backtest_with_trades(
        prepared_data.copy(),
        initial_cash=initial_cash,
        stop_loss_pct=scenario["stop_loss_pct"],
        take_profit_pct=scenario["take_profit_pct"],
        max_holding_days=scenario["max_holding_days"],
    )
    performance = summarize_performance(backtest_df)
    trade_metrics = summarize_trade_metrics(trades_df)

    return {
        "scenario": scenario["scenario"],
        "stop_loss_pct": scenario["stop_loss_pct"],
        "take_profit_pct": scenario["take_profit_pct"],
        "max_holding_days": scenario["max_holding_days"],
        "final_value": performance["final_value"],
        "total_return_pct": performance["total_return_pct"],
        "max_drawdown_pct": performance["max_drawdown_pct"],
        "total_trades": trade_metrics["total_trades"],
        "closed_trades": trade_metrics["closed_trades"],
        "open_trades": trade_metrics["open_trades"],
        "win_rate_pct": trade_metrics["win_rate_pct"],
        "profit_factor": trade_metrics["profit_factor"],
        "average_return_pct": trade_metrics["average_return_pct"],
        "best_trade_return_pct": trade_metrics["best_trade_return_pct"],
        "worst_trade_return_pct": trade_metrics["worst_trade_return_pct"],
        "average_holding_days": trade_metrics["average_holding_days"],
        "currently_holding": performance["currently_holding"],
    }


def format_optional_pct(value) -> str:
    return "disabled" if pd.isna(value) else f"{value:.2f}%"


def format_optional_number(value) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def format_optional_int(value) -> str:
    return "disabled" if pd.isna(value) else str(int(value))


def print_results_table(results_df: pd.DataFrame) -> None:
    print("Parameter Experiment Results")
    print("----------------------------")

    display_df = results_df.copy()

    for column in [
        "stop_loss_pct",
        "take_profit_pct",
        "total_return_pct",
        "max_drawdown_pct",
        "win_rate_pct",
        "average_return_pct",
        "best_trade_return_pct",
        "worst_trade_return_pct",
    ]:
        display_df[column] = display_df[column].apply(format_optional_pct)

    display_df["max_holding_days"] = display_df["max_holding_days"].apply(
        format_optional_int
    )

    for column in ["final_value", "profit_factor", "average_holding_days"]:
        display_df[column] = display_df[column].apply(format_optional_number)

    print(display_df.to_string(index=False))
    print()
    print("Note: This report is generated from simple rules for educational research only.")
    print("It is not financial advice.")


def main() -> None:
    args = parse_args()

    print("QuantPilot-AI Parameter Experiment")
    print("----------------------------------")
    print(f"Selected symbol: {args.symbol}")
    print(f"Selected source: {args.source}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Adjust mode: {args.adjust}")
    print(f"Initial cash: {args.initial_cash:.2f}")
    print()
    print("Fetching data once for all scenarios...")

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

    print(f"Fetched {len(stock_data)} rows.")
    print("Preparing indicators and strategy signals once...")
    print()

    prepared_data = add_all_indicators(stock_data)
    prepared_data = generate_ma_crossover_signals(prepared_data)

    rows = [
        run_scenario(prepared_data, args.initial_cash, scenario)
        for scenario in SCENARIOS
    ]
    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)

    print_results_table(results_df)

    if args.output:
        results_df.to_csv(args.output, index=False)
        print()
        print(f"Saved raw experiment results to {args.output}")


if __name__ == "__main__":
    main()
