import numpy as np
import pandas as pd

try:
    from .indicators import calculate_cci, calculate_rsi
except ImportError:
    from indicators import calculate_cci, calculate_rsi


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

FEATURE_GROUPS = {
    "identifier": ["symbol", "date"],
    "raw_market": ["open", "high", "low", "close", "volume"],
    "price_return": [
        "return_1d",
        "return_5d",
        "return_20d",
        "volume_change_1d",
        "volume_change_5d",
        "dollar_volume",
        "turnover_proxy",
        "high_low_range_pct",
        "close_open_return_pct",
        "intraday_range_pct",
        "candle_body_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
    ],
    "trend": [
        "ma5",
        "ma20",
        "ma60",
        "ma5_gap_pct",
        "ma20_gap_pct",
        "ma60_gap_pct",
        "ma5_ma20_gap_pct",
        "ma20_ma60_gap_pct",
        "price_position_20d",
        "price_position_60d",
        "breakout_20d",
        "breakdown_20d",
        "trend_strength_20d",
    ],
    "volatility": [
        "volatility_5d",
        "volatility_20d",
        "volatility_ratio_5d_20d",
        "drawdown_20d",
        "drawdown_60d",
        "rolling_high_20d",
        "rolling_low_20d",
    ],
    "technical": [
        "volume_ma5",
        "volume_ma20",
        "volume_ratio_5d",
        "volume_ratio_20d",
        "RSI",
        "CCI",
    ],
    "future_labels": [
        "future_return_1d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "label_up_1d",
        "label_up_5d",
        "label_up_10d",
        "label_up_20d",
        "label_outperform_5d",
        "label_outperform_20d",
    ],
    "future_external_valuation": [],
    "future_external_fund_flow": [],
    "future_external_institutional_holding": [],
    "future_external_dividend": [],
    "future_external_sentiment": [],
    "future_external_news": [],
    "future_external_industry": [],
}


def get_feature_groups() -> dict[str, list[str]]:
    """
    Return the factor column groups used by the dataset builder.

    Empty future_external_* groups are reserved for later valuation, fund flow,
    institutional holding, dividend, sentiment, news, and industry features.
    """
    return {group: columns.copy() for group, columns in FEATURE_GROUPS.items()}


def get_label_columns() -> list[str]:
    """Return columns that are labels or future-return targets."""
    return FEATURE_GROUPS["future_labels"].copy()


def get_feature_columns(include_labels: bool = False) -> list[str]:
    """
    Return model feature columns.

    By default this excludes identifier columns and all future/label columns.
    """
    columns = []
    for group, group_columns in FEATURE_GROUPS.items():
        if group == "identifier":
            continue
        if group == "future_labels" and not include_labels:
            continue
        columns.extend(group_columns)

    if not include_labels:
        columns = [
            column
            for column in columns
            if not column.startswith(("future_", "label_"))
        ]

    return columns


def validate_no_future_leakage(feature_columns: list[str]) -> None:
    """
    Raise an error if future-looking columns are selected as model features.
    """
    leaked_columns = [
        column
        for column in feature_columns
        if column.startswith(("future_", "label_"))
    ]
    if leaked_columns:
        raise ValueError(
            "Future leakage detected. Model features must not include "
            f"future or label columns: {leaked_columns}"
        )


def merge_external_features(
    base_df: pd.DataFrame,
    external_df: pd.DataFrame,
    on: list[str] | None = None,
    how: str = "left",
) -> pd.DataFrame:
    """
    Merge optional future external factors into the base factor dataset.

    This placeholder keeps the project ready for valuation, fund-flow,
    sentiment, news, and industry data without requiring those sources today.
    """
    if on is None:
        on = ["symbol", "date"]

    missing_base = [column for column in on if column not in base_df.columns]
    missing_external = [column for column in on if column not in external_df.columns]
    if missing_base or missing_external:
        raise ValueError(
            "External feature merge keys are missing. "
            f"base missing={missing_base}, external missing={missing_external}"
        )

    return base_df.merge(external_df, on=on, how=how)


def _require_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Input dataframe is missing required columns: "
            f"{missing_columns}. Required columns are: {REQUIRED_COLUMNS}"
        )


def _future_up_label(future_return: pd.Series) -> pd.Series:
    label = (future_return > 0).astype("Int64")
    label = label.where(future_return.notna(), pd.NA)
    return label


def build_factor_dataset(df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    """
    Build an extensible feature-and-label dataset from OHLCV data.

    Leakage rule:
    - Feature columns use only current and past rows through rolling windows,
      pct_change, and current-day OHLCV values.
    - Only columns beginning with future_ or label_ may use shifted future data.
    """
    normalized_df = df.copy()
    normalized_df.columns = [column.strip().lower() for column in normalized_df.columns]
    _require_columns(normalized_df)

    optional_columns = [column for column in ["amount"] if column in normalized_df.columns]
    normalized_df = normalized_df[REQUIRED_COLUMNS + optional_columns].copy()
    normalized_df["date"] = pd.to_datetime(normalized_df["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", *optional_columns]:
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")

    normalized_df = normalized_df.dropna(subset=["date", "close"])
    normalized_df = normalized_df.sort_values("date").reset_index(drop=True)

    result = pd.DataFrame(index=normalized_df.index)
    result["symbol"] = symbol
    result["date"] = normalized_df["date"]
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = normalized_df[column]

    close = result["close"]
    open_price = result["open"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]
    amount = normalized_df["amount"] if "amount" in normalized_df.columns else None

    # Current/past-only market features. These are safe to use for training inputs.
    result["return_1d"] = close.pct_change(1)
    result["return_5d"] = close.pct_change(5)
    result["return_20d"] = close.pct_change(20)
    result["volume_change_1d"] = volume.pct_change(1)
    result["volume_change_5d"] = volume.pct_change(5)
    result["dollar_volume"] = close * volume
    result["turnover_proxy"] = amount if amount is not None else result["dollar_volume"]
    result["high_low_range_pct"] = (high - low) / close
    result["close_open_return_pct"] = (close - open_price) / open_price
    result["intraday_range_pct"] = (high - low) / open_price
    result["candle_body_pct"] = (close - open_price).abs() / open_price
    result["upper_shadow_pct"] = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / open_price
    result["lower_shadow_pct"] = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / open_price

    result["ma5"] = close.rolling(window=5).mean()
    result["ma20"] = close.rolling(window=20).mean()
    result["ma60"] = close.rolling(window=60).mean()
    result["ma5_gap_pct"] = (close - result["ma5"]) / result["ma5"]
    result["ma20_gap_pct"] = (close - result["ma20"]) / result["ma20"]
    result["ma60_gap_pct"] = (close - result["ma60"]) / result["ma60"]
    result["ma5_ma20_gap_pct"] = (result["ma5"] - result["ma20"]) / result["ma20"]
    result["ma20_ma60_gap_pct"] = (result["ma20"] - result["ma60"]) / result["ma60"]

    result["volatility_5d"] = result["return_1d"].rolling(window=5).std()
    result["volatility_20d"] = result["return_1d"].rolling(window=20).std()
    result["volatility_ratio_5d_20d"] = result["volatility_5d"] / result["volatility_20d"]
    result["rolling_high_20d"] = close.rolling(window=20).max()
    result["rolling_low_20d"] = close.rolling(window=20).min()
    rolling_high_60d = close.rolling(window=60).max()
    rolling_low_60d = close.rolling(window=60).min()
    previous_high_20d = high.shift(1).rolling(window=20).max()
    previous_low_20d = low.shift(1).rolling(window=20).min()
    result["drawdown_20d"] = close / result["rolling_high_20d"] - 1
    result["drawdown_60d"] = close / rolling_high_60d - 1
    result["price_position_20d"] = (
        close - result["rolling_low_20d"]
    ) / (result["rolling_high_20d"] - result["rolling_low_20d"])
    result["price_position_60d"] = (
        close - rolling_low_60d
    ) / (rolling_high_60d - rolling_low_60d)
    result["breakout_20d"] = (close > previous_high_20d).astype("Int64")
    result["breakout_20d"] = result["breakout_20d"].where(previous_high_20d.notna(), pd.NA)
    result["breakdown_20d"] = (close < previous_low_20d).astype("Int64")
    result["breakdown_20d"] = result["breakdown_20d"].where(previous_low_20d.notna(), pd.NA)
    result["trend_strength_20d"] = (close - close.shift(20)) / (20 * close.shift(20))

    result["volume_ma5"] = volume.rolling(window=5).mean()
    result["volume_ma20"] = volume.rolling(window=20).mean()
    result["volume_ratio_5d"] = volume / result["volume_ma5"]
    result["volume_ratio_20d"] = volume / result["volume_ma20"]

    result["RSI"] = calculate_rsi(result)
    result["CCI"] = calculate_cci(result)

    # Future labels intentionally look ahead. Keep them out of model features.
    for days in [1, 5, 10, 20]:
        future_return_column = f"future_return_{days}d"
        result[future_return_column] = close.shift(-days) / close - 1
        result[f"label_up_{days}d"] = _future_up_label(result[future_return_column])

    # Benchmark data is not available in this step, so outperform labels are placeholders.
    result["label_outperform_5d"] = pd.NA
    result["label_outperform_20d"] = pd.NA

    result = result.replace([np.inf, -np.inf], pd.NA)
    ordered_columns = []
    for columns in FEATURE_GROUPS.values():
        ordered_columns.extend(columns)

    return result[ordered_columns]
