import pandas as pd


def generate_ma_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simple moving average crossover signals.

    This strategy uses two existing columns:
    - MA5: short-term moving average
    - MA20: longer-term moving average

    Signal meanings:
    - 1 means buy when MA5 crosses above MA20
    - -1 means sell when MA5 crosses below MA20
    - 0 means no action
    """
    result = df.copy()

    if "MA5" not in result.columns or "MA20" not in result.columns:
        raise ValueError("DataFrame must contain MA5 and MA20 columns.")

    # Start with no action for every row.
    result["signal"] = 0

    # Today's relationship between MA5 and MA20.
    ma5_above_ma20_today = result["MA5"] > result["MA20"]
    ma5_below_ma20_today = result["MA5"] < result["MA20"]

    # Yesterday's relationship between MA5 and MA20.
    ma5_above_ma20_yesterday = result["MA5"].shift(1) > result["MA20"].shift(1)
    ma5_below_ma20_yesterday = result["MA5"].shift(1) < result["MA20"].shift(1)

    # Buy when MA5 moves from below MA20 to above MA20.
    buy_signal = ma5_above_ma20_today & ma5_below_ma20_yesterday

    # Sell when MA5 moves from above MA20 to below MA20.
    sell_signal = ma5_below_ma20_today & ma5_above_ma20_yesterday

    result.loc[buy_signal, "signal"] = 1
    result.loc[sell_signal, "signal"] = -1

    return result
