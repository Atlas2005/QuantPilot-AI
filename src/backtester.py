import pandas as pd


def run_long_only_backtest(
    df: pd.DataFrame,
    initial_cash: float = 10000,
) -> pd.DataFrame:
    """
    Run a simple long-only backtest using the existing signal column.

    Long-only means:
    - The strategy can buy shares.
    - The strategy can sell shares it already owns.
    - The strategy cannot short sell.
    - The strategy cannot borrow money with margin.
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame must contain a signal column.")

    result = df.copy()

    cash = initial_cash
    shares = 0.0
    daily_records = []

    for _, row in result.iterrows():
        date = row["date"]
        close_price = row["close"]
        signal = row["signal"]

        # Buy with all available cash when there is a buy signal
        # and we are not already holding shares.
        if signal == 1 and shares == 0:
            shares = cash / close_price
            cash = 0.0

        # Sell all shares when there is a sell signal
        # and we currently hold a position.
        elif signal == -1 and shares > 0:
            cash = shares * close_price
            shares = 0.0

        position_value = shares * close_price
        total_value = cash + position_value

        daily_records.append(
            {
                "date": date,
                "close": close_price,
                "signal": signal,
                "cash": cash,
                "shares": shares,
                "position_value": position_value,
                "total_value": total_value,
            }
        )

    return pd.DataFrame(daily_records)
