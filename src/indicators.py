import pandas as pd


def calculate_moving_average(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate a simple moving average of the close price.

    A moving average smooths price movement by averaging the most recent
    closing prices. For example, a 5-day moving average uses the latest
    5 closing prices.
    """
    return df["close"].rolling(window=window).mean()


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MA5 and MA20 columns to a copy of the DataFrame.

    MA5 is a short-term moving average.
    MA20 is a longer-term moving average.
    """
    result = df.copy()
    result["MA5"] = calculate_moving_average(result, window=5)
    result["MA20"] = calculate_moving_average(result, window=20)
    return result


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).

    RSI compares recent average gains with recent average losses.
    A higher RSI usually means stronger recent upward momentum.
    A lower RSI usually means stronger recent downward momentum.
    """
    price_change = df["close"].diff()

    # Positive changes are gains. Negative changes are set to 0.
    gains = price_change.clip(lower=0)

    # Negative changes are losses. Convert them to positive numbers.
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(window=window).mean()
    average_loss = losses.rolling(window=window).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


def calculate_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate the Commodity Channel Index (CCI).

    CCI compares the current typical price with its recent average.
    The typical price is the average of high, low, and close.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    typical_price_ma = typical_price.rolling(window=window).mean()

    # Mean absolute deviation measures how far prices usually move
    # away from their recent average.
    mean_deviation = typical_price.rolling(window=window).apply(
        lambda values: abs(values - values.mean()).mean(),
        raw=False,
    )

    cci = (typical_price - typical_price_ma) / (0.015 * mean_deviation)

    return cci


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all V1 technical indicators to a copy of the DataFrame.

    The returned DataFrame includes:
    - MA5
    - MA20
    - RSI
    - CCI
    """
    result = add_moving_averages(df)
    result["RSI"] = calculate_rsi(result)
    result["CCI"] = calculate_cci(result)
    return result
