from pathlib import Path

import pandas as pd
import streamlit as st

from src.backtester import run_long_only_backtest_with_trades
from src.indicators import add_all_indicators
from src.metrics import summarize_performance
from src.report_generator import generate_rule_based_report
from src.strategy import generate_ma_crossover_signals
from src.trade_metrics import summarize_trade_metrics


DEMO_DATA_PATH = Path("data/sample/demo_000001.csv")
LOCAL_REAL_DATA_DIR = Path("data/real")
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def parse_optional_float(value: str, label: str) -> float | None:
    text = value.strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        st.sidebar.error(f"{label} must be a number or blank.")
        st.stop()


def parse_optional_int(value: str, label: str) -> int | None:
    text = value.strip()
    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        st.sidebar.error(f"{label} must be a whole number or blank.")
        st.stop()


def format_trade_log(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    display_df = trades_df.copy()
    for column in ["entry_date", "exit_date"]:
        if column in display_df.columns:
            display_df[column] = pd.to_datetime(
                display_df[column],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")

    return display_df


def validate_stock_data(stock_data: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in stock_data.columns
    ]
    if missing_columns:
        st.error(
            "The selected CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()


def load_demo_data() -> tuple[pd.DataFrame, str]:
    if not DEMO_DATA_PATH.exists():
        st.error(f"Demo CSV is missing: {DEMO_DATA_PATH}")
        st.stop()

    return pd.read_csv(DEMO_DATA_PATH), f"Demo data: {DEMO_DATA_PATH}"


def load_local_real_csv() -> tuple[pd.DataFrame, str]:
    csv_files = sorted(LOCAL_REAL_DATA_DIR.glob("*.csv"))
    if not csv_files:
        st.warning(
            "No local real CSV files found in data/real/. "
            "Create one with the command-line real-data workflow first."
        )
        st.stop()

    selected_file = st.sidebar.selectbox(
        "Local real CSV",
        csv_files,
        format_func=lambda path: path.name,
    )
    return pd.read_csv(selected_file), f"Local real CSV: {selected_file}"


def load_uploaded_csv() -> tuple[pd.DataFrame, str]:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to run the dashboard with browser data.")
        st.stop()

    return pd.read_csv(uploaded_file), f"Uploaded CSV: {uploaded_file.name}"


def load_selected_data() -> tuple[pd.DataFrame, str]:
    data_source_mode = st.sidebar.selectbox(
        "Data source mode",
        ["Demo data", "Local real CSV", "Upload CSV"],
    )

    if data_source_mode == "Demo data":
        return load_demo_data()
    if data_source_mode == "Local real CSV":
        return load_local_real_csv()
    return load_uploaded_csv()


def main() -> None:
    st.set_page_config(page_title="QuantPilot-AI Dashboard", layout="wide")

    st.title("QuantPilot-AI Dashboard")
    st.caption(
        "Educational rule-based backtesting demo. This is not financial advice, "
        "and backtest results do not guarantee future returns."
    )

    st.sidebar.header("Backtest Settings")
    initial_cash = st.sidebar.number_input(
        "Initial cash",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
    )
    stop_loss_pct = parse_optional_float(
        st.sidebar.text_input("Stop loss % (blank = disabled)", value=""),
        "Stop loss %",
    )
    take_profit_pct = parse_optional_float(
        st.sidebar.text_input("Take profit % (blank = disabled)", value=""),
        "Take profit %",
    )
    max_holding_days = parse_optional_int(
        st.sidebar.text_input("Max holding days (blank = disabled)", value=""),
        "Max holding days",
    )

    stock_data, data_label = load_selected_data()
    validate_stock_data(stock_data)
    stock_data["date"] = pd.to_datetime(stock_data["date"])

    st.subheader("Loaded Data Preview")
    st.write(f"Using data source: `{data_label}`")
    st.dataframe(stock_data.head(10), width="stretch")

    stock_data = add_all_indicators(stock_data)
    stock_data = generate_ma_crossover_signals(stock_data)
    backtest_result, trades = run_long_only_backtest_with_trades(
        stock_data,
        initial_cash=initial_cash,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_days=max_holding_days,
    )
    performance_summary = summarize_performance(backtest_result)
    trade_metrics = summarize_trade_metrics(trades)
    report = generate_rule_based_report(performance_summary)

    chart_data = stock_data.set_index("date")

    st.subheader("Closing Price")
    st.line_chart(chart_data[["close"]])

    if {"MA5", "MA20"}.issubset(stock_data.columns):
        st.subheader("MA5 and MA20")
        st.line_chart(chart_data[["MA5", "MA20"]])

    st.subheader("Performance Summary")
    st.dataframe(
        pd.DataFrame(
            performance_summary.items(),
            columns=["metric", "value"],
        ),
        width="stretch",
    )

    st.subheader("Rule-Based Report")
    st.text(report)

    st.subheader("Trade Log")
    if trades.empty:
        st.write("No trades were executed.")
    else:
        st.dataframe(format_trade_log(trades), width="stretch")

    st.subheader("Trade Metrics")
    st.dataframe(
        pd.DataFrame(
            trade_metrics.items(),
            columns=["metric", "value"],
        ),
        width="stretch",
    )


if __name__ == "__main__":
    main()
