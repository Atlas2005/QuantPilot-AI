import pandas as pd

from backtester import run_long_only_backtest_with_trades
from indicators import add_all_indicators
from metrics import summarize_performance
from report_generator import generate_rule_based_report
from strategy import generate_ma_crossover_signals
from trade_metrics import summarize_trade_metrics


DEMO_DATA_PATH = "data/sample/demo_000001.csv"


def format_number(value) -> str:
    return "" if value is None or value != value else f"{value:.2f}"


def format_pct(value) -> str:
    return "" if value is None or value != value else f"{value:.2f}%"


def format_date(value) -> str:
    if value is None or value != value:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def print_trade_log(trades_df: pd.DataFrame) -> None:
    print()
    print("Trade Log")
    print("---------")

    if trades_df.empty:
        print("No trades were executed.")
        return

    display_df = trades_df.copy()
    for column in ["entry_date", "exit_date"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_date)

    for column in [
        "entry_price",
        "exit_price",
        "shares",
        "profit",
        "unrealized_profit",
    ]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_number)

    for column in ["return_pct", "unrealized_return_pct"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_pct)

    print(display_df.to_string(index=False))


def print_trade_metrics(metrics: dict) -> None:
    print()
    print("Trade Metrics")
    print("-------------")

    money_fields = {
        "total_realized_profit",
        "average_profit",
        "average_loss",
        "best_trade_profit",
        "worst_trade_profit",
        "open_unrealized_profit",
    }
    pct_fields = {
        "win_rate_pct",
        "average_return_pct",
        "best_trade_return_pct",
        "worst_trade_return_pct",
        "open_unrealized_return_pct",
    }
    float_fields = {"profit_factor", "average_holding_days"}

    for key, value in metrics.items():
        if value is None:
            display_value = "N/A"
        elif key in money_fields:
            display_value = f"{value:.2f}"
        elif key in pct_fields:
            display_value = f"{value:.2f}%"
        elif key in float_fields:
            display_value = f"{value:.2f}"
        else:
            display_value = value

        print(f"{key}: {display_value}")


def main() -> None:
    print("QuantPilot-AI Offline Demo")
    print("--------------------------")
    print(f"Using demo data file: {DEMO_DATA_PATH}")
    print()

    stock_data = pd.read_csv(DEMO_DATA_PATH)
    stock_data = add_all_indicators(stock_data)
    stock_data = generate_ma_crossover_signals(stock_data)
    backtest_result, trades = run_long_only_backtest_with_trades(stock_data)
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

    print_trade_log(trades)
    print_trade_metrics(summarize_trade_metrics(trades))


if __name__ == "__main__":
    main()
