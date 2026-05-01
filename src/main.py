import pandas as pd

from backtester import run_long_only_backtest
from indicators import add_all_indicators
from metrics import summarize_performance
from strategy import generate_ma_crossover_signals


def main() -> None:
    """
    Run the full V1 workflow from sample CSV data to backtest summary.
    """
    file_path = "data/sample/sample_stock.csv"

    # Step 1: Load the sample daily K-line data from CSV.
    stock_data = pd.read_csv(file_path)

    # Step 2: Add technical indicators used by the strategy.
    stock_data = add_all_indicators(stock_data)

    # Step 3: Generate MA crossover buy/sell/no-action signals.
    stock_data = generate_ma_crossover_signals(stock_data)

    # Step 4: Run a simple long-only backtest using those signals.
    backtest_result = run_long_only_backtest(stock_data)

    # Step 5: Summarize the main V1 performance metrics.
    performance_summary = summarize_performance(backtest_result)

    print("Performance Summary")
    print("-------------------")
    for key, value in performance_summary.items():
        print(f"{key}: {value}")

    print()
    print("Last 10 Backtest Rows")
    print("---------------------")
    print(backtest_result.tail(10))


if __name__ == "__main__":
    main()
