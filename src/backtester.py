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


def run_long_only_backtest_with_trades(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the long-only backtest and return both daily results and a trade log.

    The daily backtest output follows the same logic and column format as
    run_long_only_backtest. The trade log adds one row per closed trade, plus
    one open row if the strategy is still holding shares at the end.
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame must contain a signal column.")

    result = df.copy()

    cash = initial_cash
    shares = 0.0
    daily_records = []
    trade_records = []
    entry_date = None
    entry_price = None

    for _, row in result.iterrows():
        date = row["date"]
        close_price = row["close"]
        signal = row["signal"]

        if signal == 1 and shares == 0:
            shares = cash / close_price
            cash = 0.0
            entry_date = date
            entry_price = close_price

        elif signal == -1 and shares > 0:
            cash = shares * close_price
            profit = (close_price - entry_price) * shares
            return_pct = ((close_price - entry_price) / entry_price) * 100
            holding_days = (
                pd.to_datetime(date) - pd.to_datetime(entry_date)
            ).days

            trade_records.append(
                {
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": date,
                    "exit_price": close_price,
                    "shares": shares,
                    "profit": profit,
                    "return_pct": return_pct,
                    "unrealized_profit": None,
                    "unrealized_return_pct": None,
                    "holding_days": holding_days,
                    "status": "closed",
                }
            )

            shares = 0.0
            entry_date = None
            entry_price = None

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

    if shares > 0:
        final_row = result.iloc[-1]
        final_date = final_row["date"]
        final_price = final_row["close"]
        unrealized_profit = (final_price - entry_price) * shares
        unrealized_return_pct = ((final_price - entry_price) / entry_price) * 100
        holding_days = (
            pd.to_datetime(final_date) - pd.to_datetime(entry_date)
        ).days

        trade_records.append(
            {
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": None,
                "exit_price": None,
                "shares": shares,
                "profit": None,
                "return_pct": None,
                "unrealized_profit": unrealized_profit,
                "unrealized_return_pct": unrealized_return_pct,
                "holding_days": holding_days,
                "status": "open",
            }
        )

    return pd.DataFrame(daily_records), pd.DataFrame(trade_records)
