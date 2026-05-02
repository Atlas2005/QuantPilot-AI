import argparse
from pathlib import Path
import os
import sys
import time
from datetime import datetime

import akshare as ak
import baostock as bs
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and save standardized A-share daily data."
    )
    parser.add_argument(
        "--symbol",
        default="000001",
        help="A-share stock code without market prefix, for example 600519.",
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
        help="AkShare adjustment mode, for example qfq.",
    )
    parser.add_argument(
        "--source",
        choices=["akshare", "baostock", "auto"],
        default="auto",
        help="Data source to use: akshare, baostock, or auto.",
    )
    return parser.parse_args()


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


def convert_symbol_for_baostock(symbol: str) -> str:
    """
    Convert a plain A-share symbol into Baostock's exchange-prefixed format.
    """
    if symbol.startswith(("000", "001", "002", "003", "300")):
        return f"sz.{symbol}"
    if symbol.startswith(("600", "601", "603", "605", "688")):
        return f"sh.{symbol}"

    raise ValueError(
        f"Unsupported A-share symbol prefix for Baostock: {symbol}. "
        "Expected a Shenzhen code such as 000001/300750 or "
        "a Shanghai code such as 600519/688981."
    )


def convert_date_for_baostock(date_text: str) -> str:
    """
    Convert YYYYMMDD input into Baostock's YYYY-MM-DD format.
    """
    return datetime.strptime(date_text, "%Y%m%d").strftime("%Y-%m-%d")


def convert_adjust_for_baostock(adjust: str) -> str:
    """
    Convert the command-line adjustment mode into Baostock's adjustflag.
    """
    adjust_text = adjust.lower().strip()
    if adjust_text == "qfq":
        return "2"
    if adjust_text == "hfq":
        return "1"
    if adjust_text in ("", "none"):
        return "3"

    raise ValueError(
        f"Unsupported adjust value for Baostock: {adjust}. "
        "Use qfq, hfq, none, or an empty value."
    )


def fetch_a_share_daily_baostock(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    Fetch daily historical K-line data for one A-share stock from Baostock.
    """
    bs_symbol = convert_symbol_for_baostock(symbol)
    bs_start_date = convert_date_for_baostock(start_date)
    bs_end_date = convert_date_for_baostock(end_date)
    adjustflag = convert_adjust_for_baostock(adjust)
    fields = "date,code,open,high,low,close,volume"
    login_result = None

    try:
        login_result = bs.login()
        if login_result.error_code != "0":
            raise RuntimeError(
                f"Baostock login failed: {login_result.error_msg}"
            )

        result = bs.query_history_k_data_plus(
            bs_symbol,
            fields,
            start_date=bs_start_date,
            end_date=bs_end_date,
            frequency="d",
            adjustflag=adjustflag,
        )
        if result.error_code != "0":
            raise RuntimeError(
                f"Baostock query failed for {symbol}: {result.error_msg}"
            )

        rows = []
        while result.next():
            rows.append(result.get_row_data())
    finally:
        if login_result is not None:
            bs.logout()

    if not rows:
        raise ValueError(
            f"Baostock returned no data for {symbol} from {start_date} to {end_date}."
        )

    raw_df = pd.DataFrame(rows, columns=result.fields)
    df = raw_df[STANDARD_COLUMNS].copy()

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def fetch_a_share_daily_from_source(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    source: str,
) -> pd.DataFrame:
    """
    Fetch data from the requested source, with AkShare-to-Baostock fallback in auto mode.
    """
    if source == "akshare":
        return fetch_a_share_daily(symbol, start_date, end_date, adjust)

    if source == "baostock":
        return fetch_a_share_daily_baostock(symbol, start_date, end_date, adjust)

    try:
        return fetch_a_share_daily(symbol, start_date, end_date, adjust)
    except Exception as exc:
        print(f"Warning: AkShare failed: {exc}")
        print("Trying Baostock fallback...")
        return fetch_a_share_daily_baostock(symbol, start_date, end_date, adjust)


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
    args = parse_args()

    print(
        "Fetching A-share data: "
        f"symbol={args.symbol}, start={args.start}, "
        f"end={args.end}, adjust={args.adjust}, source={args.source}"
    )

    try:
        stock_df = fetch_a_share_daily_from_source(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            adjust=args.adjust,
            source=args.source,
        )
        saved_path = save_stock_csv(stock_df, args.symbol)
        print(f"Saved {len(stock_df)} rows to {saved_path}")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
