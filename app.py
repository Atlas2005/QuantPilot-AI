from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import streamlit as st

from src.backtester import run_long_only_backtest_with_trades
from src.indicators import add_all_indicators
from src.metrics import summarize_performance
from src.real_data_loader import fetch_a_share_daily_from_source
from src.report_generator import generate_rule_based_report
from src.strategy import generate_ma_crossover_signals
from src.trade_metrics import summarize_trade_metrics


DEMO_DATA_PATH = Path("data/sample/demo_000001.csv")
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
REAL_DATA_SOURCES = {
    "Baostock real data": "baostock",
    "AkShare real data": "akshare",
    "Auto real data": "auto",
}


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


def format_chart_date_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


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


def validate_non_empty_stock_data(stock_data: pd.DataFrame, label: str) -> None:
    if stock_data.empty:
        st.error(f"{label} returned no rows. Try another symbol or date range.")
        st.stop()


def format_date_for_loader(date_value: date) -> str:
    return date_value.strftime("%Y%m%d")


@st.cache_data(show_spinner=False)
def fetch_real_stock_data(
    symbol: str,
    source: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> pd.DataFrame:
    return fetch_a_share_daily_from_source(
        symbol=symbol,
        source=source,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )


def load_demo_data() -> tuple[pd.DataFrame, str, dict]:
    if not DEMO_DATA_PATH.exists():
        st.error(f"Demo CSV is missing: {DEMO_DATA_PATH}")
        st.stop()

    return (
        pd.read_csv(DEMO_DATA_PATH),
        f"Demo data: {DEMO_DATA_PATH}",
        {"mode": "Demo data"},
    )


def load_real_data(source_mode: str) -> tuple[pd.DataFrame, str, dict]:
    source = REAL_DATA_SOURCES[source_mode]
    symbol = st.sidebar.text_input("Stock symbol", value="000001").strip()
    start_date_value = st.sidebar.date_input(
        "Start date",
        value=date(2024, 1, 1),
    )
    end_date_value = st.sidebar.date_input(
        "End date",
        value=date(2024, 12, 31),
    )
    adjust = st.sidebar.text_input("Adjust mode", value="qfq").strip()

    if not symbol:
        st.sidebar.error("Stock symbol cannot be blank.")
        st.stop()
    if start_date_value > end_date_value:
        st.sidebar.error("Start date must be before or equal to end date.")
        st.stop()
    if not adjust:
        st.sidebar.error("Adjust mode cannot be blank. Use qfq, hfq, or none.")
        st.stop()

    start_text = format_date_for_loader(start_date_value)
    end_text = format_date_for_loader(end_date_value)
    fetch_key = (source, symbol, start_text, end_text, adjust)

    fetch_clicked = st.sidebar.button("Fetch data and run backtest")
    stored_result = st.session_state.get("real_data_result")

    if fetch_clicked:
        try:
            with st.spinner(
                "Fetching real A-share data. Free data sources can take a moment..."
            ):
                stock_data = fetch_real_stock_data(
                    symbol=symbol,
                    source=source,
                    start_date=start_text,
                    end_date=end_text,
                    adjust=adjust,
                )
        except Exception as exc:
            st.error(f"Real-data fetch failed: {exc}")
            st.stop()

        validate_non_empty_stock_data(stock_data, source_mode)
        st.session_state["real_data_result"] = {
            "key": fetch_key,
            "data": stock_data.copy(),
        }
        stored_result = st.session_state["real_data_result"]

    if stored_result is None or stored_result.get("key") != fetch_key:
        st.info(
            "Choose real-data settings in the sidebar, then click "
            "'Fetch data and run backtest'."
        )
        st.stop()

    data_label = (
        f"{source_mode}: symbol={symbol}, "
        f"range={start_date_value:%Y-%m-%d} to {end_date_value:%Y-%m-%d}, "
        f"adjust={adjust}"
    )
    details = {
        "mode": source_mode,
        "source": source,
        "symbol": symbol,
        "start": f"{start_date_value:%Y-%m-%d}",
        "end": f"{end_date_value:%Y-%m-%d}",
        "adjust": adjust,
    }
    return stored_result["data"].copy(), data_label, details


def load_selected_data() -> tuple[pd.DataFrame, str, dict]:
    data_source_mode = st.sidebar.selectbox(
        "Data source mode",
        [
            "Demo data",
            "Baostock real data",
            "AkShare real data",
            "Auto real data",
        ],
    )

    if data_source_mode == "Demo data":
        return load_demo_data()
    return load_real_data(data_source_mode)


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
    format_chart_date_axis(ax)
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
    drawdown_data["drawdown_pct"] = (total_value / total_value.cummax() - 1) * 100

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        drawdown_data["date"],
        drawdown_data["drawdown_pct"],
        label="Drawdown",
        color="tab:red",
        linewidth=1.8,
    )
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, alpha=0.25)
    format_chart_date_axis(ax)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)


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

    stock_data, data_label, data_details = load_selected_data()
    validate_stock_data(stock_data)
    validate_non_empty_stock_data(stock_data, data_details["mode"])
    stock_data["date"] = pd.to_datetime(stock_data["date"])

    st.subheader("Loaded Data Preview")
    st.write(f"Using data source: `{data_label}`")
    if data_details["mode"] != "Demo data":
        st.write(f"Selected symbol: `{data_details['symbol']}`")
        st.write(f"Selected date range: `{data_details['start']}` to `{data_details['end']}`")
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
