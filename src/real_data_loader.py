from pathlib import Path
import os
import sys
import time

import akshare as ak
import pandas as pd


# Bypass the Windows system proxy for AkShare requests.
# This matches the manual AkShare test setup and avoids proxy connection errors.
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# V1 expects stock data to use these English column names.
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# AkShare returns A-share historical data with Chinese column names.
# This mapping keeps only the fields needed by the existing V1 pipeline.
# Unicode escape strings are used here to avoid editor/terminal encoding issues.
COLUMN_MAPPING = {
    "\u65e5\u671f": "date",
    "\u5f00\u76d8": "open",
    "\u6700\u9ad8": "high",
    "\u6700\u4f4e": "low",
    "\u6536\u76d8": "close",
    "\u6210\u4ea4\u91cf": "volume",
}


def fetch_a_share_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    Fetch daily historical K-line data for one A-share stock.

    Args:
        symbol: Stock code without market prefix, for example "000001".
        start_date: Start date in YYYYMMDD format, for example "20240101".
        end_date: End date in YYYYMMDD format, for example "20241231".
        adjust: Price adjustment mode used by AkShare.
            "qfq" means front-adjusted prices.

    Returns:
        A standardized DataFrame with V1-compatible columns:
        date, open, high, low, close, volume
    """
    # Network requests can occasionally fail, so retry a few times before giving up.
    for attempt in range(1, 4):
        try:
            raw_df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            break
        except Exception as exc:
            if attempt < 3:
                print(
                    f"Fetch failed for {symbol}. Retrying in 2 seconds "
                    f"({attempt}/3)..."
                )
                time.sleep(2)
            else:
                raise RuntimeError(
                    f"Failed to fetch A-share data for {symbol} after 3 attempts: {exc}"
                ) from exc

    if raw_df.empty:
        raise ValueError(
            f"AkShare returned no data for {symbol} from {start_date} to {end_date}."
        )

    missing_columns = [col for col in COLUMN_MAPPING if col not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"AkShare data is missing columns: {missing_columns}")

    # Rename Chinese AkShare columns into the English format used by V1.
    df = raw_df.rename(columns=COLUMN_MAPPING)

    # Keep only the columns used by the current backtest/data-loader pipeline.
    df = df[STANDARD_COLUMNS].copy()

    # Convert dates to pandas datetime values and sort from oldest to newest.
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def save_stock_csv(df: pd.DataFrame, symbol: str) -> Path:
    """
    Save standardized stock data to data/real/{symbol}.csv.

    The data/real folder is created automatically if it does not exist yet.
    """
    output_dir = Path("data") / "real"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{symbol}.csv"

    # Save dates as simple YYYY-MM-DD strings so the CSV stays beginner-friendly.
    output_df = df.copy()
    output_df["date"] = pd.to_datetime(output_df["date"]).dt.strftime("%Y-%m-%d")
    output_df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    test_symbol = "000001"
    test_start_date = "20240101"
    test_end_date = "20241231"

    try:
        stock_df = fetch_a_share_daily(
            symbol=test_symbol,
            start_date=test_start_date,
            end_date=test_end_date,
        )
        saved_path = save_stock_csv(stock_df, test_symbol)
        print(f"Saved {len(stock_df)} rows to {saved_path}")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
