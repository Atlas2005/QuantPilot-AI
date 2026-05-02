from pathlib import Path

import matplotlib.pyplot as plt
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

    money_columns = [
        "entry_price",
        "exit_price",
        "profit",
        "unrealized_profit",
    ]
    for column in money_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_table_number)

    percent_columns = [
        "return_pct",
        "unrealized_return_pct",
    ]
    for column in percent_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_table_pct)

    if "shares" in display_df.columns:
        display_df["shares"] = display_df["shares"].apply(format_table_number)

    if "holding_days" in display_df.columns:
        display_df["holding_days"] = display_df["holding_days"].apply(
            format_table_whole_number
        )

    display_df = display_df.fillna("")
    return display_df


def is_missing(value) -> bool:
    return value is None or pd.isna(value)


def format_metric_pct(value) -> str:
    if is_missing(value):
        return "N/A"
    return f"{value:.2f}%"


def format_metric_number(value) -> str:
    if is_missing(value):
        return "N/A"
    return f"{value:.2f}"


def format_table_pct(value) -> str:
    if is_missing(value):
        return ""
    return f"{value:.2f}%"


def format_table_number(value) -> str:
    if is_missing(value):
        return ""
    return f"{value:.2f}"


def format_table_whole_number(value) -> str:
    if is_missing(value):
        return ""
    return f"{int(value)}"


def format_display_date(value) -> str:
    if is_missing(value):
        return ""

    date_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(date_value):
        return ""
    return date_value.strftime("%Y-%m-%d")


def format_data_preview(stock_data: pd.DataFrame) -> pd.DataFrame:
    display_df = stock_data.copy()
    for column in display_df.columns:
        if column == "date" or column.endswith("_date"):
            display_df[column] = display_df[column].apply(format_display_date)

    return display_df.fillna("")


def format_summary_value(metric: str, value) -> str:
    if is_missing(value):
        return ""

    percent_metrics = {
        "total_return_pct",
        "max_drawdown_pct",
    }
    whole_number_metrics = {
        "buy_signals",
        "sell_signals",
    }
    money_metrics = {
        "initial_value",
        "final_value",
    }

    if metric in percent_metrics:
        return format_table_pct(value)
    if metric == "currently_holding":
        return "True" if bool(value) else "False"
    if metric in whole_number_metrics:
        return format_table_whole_number(value)
    if metric in money_metrics:
        return format_table_number(value)
    return str(value)


def format_performance_summary(performance_summary: dict) -> pd.DataFrame:
    rows = [
        {
            "metric": metric,
            "value": format_summary_value(metric, value),
        }
        for metric, value in performance_summary.items()
    ]
    return pd.DataFrame(rows)


def format_trade_metric_value(metric: str, value) -> str:
    if is_missing(value):
        return ""

    percent_metrics = {
        "win_rate_pct",
        "average_return_pct",
        "best_trade_return_pct",
        "worst_trade_return_pct",
        "open_unrealized_return_pct",
    }
    money_metrics = {
        "total_realized_profit",
        "average_profit",
        "average_loss",
        "best_trade_profit",
        "worst_trade_profit",
        "open_unrealized_profit",
    }
    whole_number_metrics = {
        "total_trades",
        "closed_trades",
        "open_trades",
        "winning_trades",
        "losing_trades",
    }

    if metric in percent_metrics:
        return format_table_pct(value)
    if metric in money_metrics:
        return format_table_number(value)
    if metric in whole_number_metrics:
        return format_table_whole_number(value)
    if metric in {"profit_factor", "average_holding_days"}:
        return format_table_number(value)
    return str(value)


def format_trade_metrics_table(trade_metrics: dict) -> pd.DataFrame:
    rows = [
        {
            "metric": metric,
            "value": format_trade_metric_value(metric, value),
        }
        for metric, value in trade_metrics.items()
    ]
    return pd.DataFrame(rows)


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


def show_metric_cards(performance_summary: dict, trade_metrics: dict) -> None:
    st.subheader("Key Metrics")
    columns = st.columns(6)
    columns[0].metric(
        "Total return",
        format_metric_pct(performance_summary["total_return_pct"]),
    )
    columns[1].metric(
        "Maximum drawdown",
        format_metric_pct(performance_summary["max_drawdown_pct"]),
    )
    columns[2].metric("Buy signals", performance_summary["buy_signals"])
    columns[3].metric("Sell signals", performance_summary["sell_signals"])
    columns[4].metric("Win rate", format_metric_pct(trade_metrics["win_rate_pct"]))
    columns[5].metric(
        "Profit factor",
        format_metric_number(trade_metrics["profit_factor"]),
    )


def show_price_signal_chart(stock_data: pd.DataFrame) -> None:
    required_columns = {"date", "close"}
    if not required_columns.issubset(stock_data.columns):
        st.warning("Price chart cannot be drawn because date or close is missing.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(stock_data["date"], stock_data["close"], label="Close", linewidth=1.8)

    if "MA5" in stock_data.columns:
        ax.plot(stock_data["date"], stock_data["MA5"], label="MA5", linewidth=1.2)
    if "MA20" in stock_data.columns:
        ax.plot(stock_data["date"], stock_data["MA20"], label="MA20", linewidth=1.2)

    if "signal" in stock_data.columns:
        buy_points = stock_data[stock_data["signal"] == 1]
        sell_points = stock_data[stock_data["signal"] == -1]
        if not buy_points.empty:
            ax.scatter(
                buy_points["date"],
                buy_points["close"],
                marker="^",
                label="Buy signal",
                color="green",
                s=80,
                zorder=3,
            )
        if not sell_points.empty:
            ax.scatter(
                sell_points["date"],
                sell_points["close"],
                marker="v",
                label="Sell signal",
                color="red",
                s=80,
                zorder=3,
            )

    ax.set_title("Price, Moving Averages, and Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)


def show_equity_curve(backtest_result: pd.DataFrame) -> None:
    if "date" not in backtest_result.columns or "total_value" not in backtest_result.columns:
        st.warning("Equity curve cannot be drawn because date or total_value is missing.")
        return

    equity_data = backtest_result.copy()
    equity_data["date"] = pd.to_datetime(equity_data["date"])
    equity_data = equity_data.set_index("date")
    st.line_chart(equity_data[["total_value"]])


def show_drawdown_curve(backtest_result: pd.DataFrame) -> None:
    if "date" not in backtest_result.columns or "total_value" not in backtest_result.columns:
        st.warning("Drawdown curve cannot be drawn because date or total_value is missing.")
        return

    drawdown_data = backtest_result.copy()
    drawdown_data["date"] = pd.to_datetime(drawdown_data["date"])
    total_value = drawdown_data["total_value"]
    drawdown_data["Drawdown %"] = (total_value / total_value.cummax() - 1) * 100
    drawdown_data = drawdown_data.set_index("date")
    st.line_chart(drawdown_data[["Drawdown %"]])


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
    st.dataframe(format_data_preview(stock_data.head(10)), width="stretch")

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

    show_metric_cards(performance_summary, trade_metrics)

    st.subheader("Price and Strategy Signals")
    show_price_signal_chart(stock_data)

    st.subheader("Portfolio Equity Curve")
    show_equity_curve(backtest_result)

    st.subheader("Drawdown Curve")
    show_drawdown_curve(backtest_result)

    st.subheader("Performance Summary")
    st.dataframe(
        format_performance_summary(performance_summary),
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
        format_trade_metrics_table(trade_metrics),
        width="stretch",
    )


if __name__ == "__main__":
    main()
