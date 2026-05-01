import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load historical stock K-line data from a CSV file.

    Required columns:
    date, open, high, low, close, volume
    """
    df = pd.read_csv(file_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


if __name__ == "__main__":
    data = load_stock_data("data/sample/sample_stock.csv")
    print(data.head())
