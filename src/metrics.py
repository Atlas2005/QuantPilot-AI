import pandas as pd


def calculate_total_return(backtest_df: pd.DataFrame) -> float:
    """
    Calculate total return as a percentage.

    Example:
    - Initial value: 10000
    - Final value: 11000
    - Total return: 10.0
    """
    initial_value = backtest_df["total_value"].iloc[0]
    final_value = backtest_df["total_value"].iloc[-1]

    return float(((final_value - initial_value) / initial_value) * 100)


def calculate_max_drawdown(backtest_df: pd.DataFrame) -> float:
    """
    Calculate the largest portfolio drop from a previous high.

    Max drawdown helps show the biggest loss an investor would have seen
    during the backtest before the portfolio recovered.
    """
    total_value = backtest_df["total_value"]

    # running_max stores the best portfolio value seen up to each day.
    running_max = total_value.cummax()

    # drawdown is negative or zero because it compares today to a past high.
    drawdown = (total_value - running_max) / running_max

    return float(drawdown.min() * 100)


def count_buy_signals(backtest_df: pd.DataFrame) -> int:
    """
    Count how many buy signals appeared in the backtest.
    """
    return int((backtest_df["signal"] == 1).sum())


def count_sell_signals(backtest_df: pd.DataFrame) -> int:
    """
    Count how many sell signals appeared in the backtest.
    """
    return int((backtest_df["signal"] == -1).sum())


def is_currently_holding(backtest_df: pd.DataFrame) -> bool:
    """
    Check whether the strategy is holding shares on the final day.
    """
    final_shares = backtest_df["shares"].iloc[-1]
    return bool(final_shares > 0)


def summarize_performance(backtest_df: pd.DataFrame) -> dict:
    """
    Return the main V1 performance summary as a dictionary.
    """
    initial_value = backtest_df["total_value"].iloc[0]
    final_value = backtest_df["total_value"].iloc[-1]

    summary = {
        "initial_value": float(initial_value),
        "final_value": float(final_value),
        "total_return_pct": calculate_total_return(backtest_df),
        "max_drawdown_pct": calculate_max_drawdown(backtest_df),
        "buy_signals": count_buy_signals(backtest_df),
        "sell_signals": count_sell_signals(backtest_df),
        "currently_holding": is_currently_holding(backtest_df),
    }

    for column, key in [
        ("total_commission", "total_commission"),
        ("total_stamp_tax", "total_stamp_tax"),
        ("total_slippage_cost", "total_slippage_cost"),
        ("total_transaction_cost", "total_transaction_cost"),
    ]:
        if column in backtest_df.columns:
            summary[key] = float(backtest_df[column].iloc[-1])
        else:
            summary[key] = 0.0

    return summary
