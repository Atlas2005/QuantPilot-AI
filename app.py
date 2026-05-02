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
EXPERIMENT_SOURCE_OPTIONS = ["demo", "baostock", "akshare", "auto"]
PERIOD_SOURCE_OPTIONS = ["demo", "baostock", "akshare"]
SCENARIOS = [
    {
        "scenario": "baseline",
        "stop_loss_pct": None,
        "take_profit_pct": None,
        "max_holding_days": None,
    },
    {
        "scenario": "sl_3",
        "stop_loss_pct": 3,
        "take_profit_pct": None,
        "max_holding_days": None,
    },
    {
        "scenario": "sl_5",
        "stop_loss_pct": 5,
        "take_profit_pct": None,
        "max_holding_days": None,
    },
    {
        "scenario": "tp_10",
        "stop_loss_pct": None,
        "take_profit_pct": 10,
        "max_holding_days": None,
    },
    {
        "scenario": "max_30",
        "stop_loss_pct": None,
        "take_profit_pct": None,
        "max_holding_days": 30,
    },
    {
        "scenario": "sl_3_tp_10",
        "stop_loss_pct": 3,
        "take_profit_pct": 10,
        "max_holding_days": None,
    },
    {
        "scenario": "sl_3_max_30",
        "stop_loss_pct": 3,
        "take_profit_pct": None,
        "max_holding_days": 30,
    },
    {
        "scenario": "sl_3_tp_10_max_30",
        "stop_loss_pct": 3,
        "take_profit_pct": 10,
        "max_holding_days": 30,
    },
]
RESULT_COLUMNS = [
    "symbol",
    "scenario",
    "stop_loss_pct",
    "take_profit_pct",
    "max_holding_days",
    "final_value",
    "total_return_pct",
    "max_drawdown_pct",
    "total_trades",
    "closed_trades",
    "open_trades",
    "win_rate_pct",
    "profit_factor",
    "average_return_pct",
    "best_trade_return_pct",
    "worst_trade_return_pct",
    "average_holding_days",
    "currently_holding",
    "error",
]
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "sl_3": "Stop loss 3%",
    "sl_5": "Stop loss 5%",
    "tp_10": "Take profit 10%",
    "max_30": "Max holding 30 days",
    "sl_3_tp_10": "Stop loss 3% + take profit 10%",
    "sl_3_max_30": "Stop loss 3% + max holding 30 days",
    "sl_3_tp_10_max_30": (
        "Stop loss 3% + take profit 10% + max holding 30 days"
    ),
}
SCENARIO_SHORT_LABELS = {
    "baseline": "Baseline",
    "sl_3": "SL 3%",
    "sl_5": "SL 5%",
    "tp_10": "TP 10%",
    "max_30": "Max 30d",
    "sl_3_tp_10": "SL 3% + TP 10%",
    "sl_3_max_30": "SL 3% + Max 30d",
    "sl_3_tp_10_max_30": "SL 3% + TP 10% + Max 30d",
}
PERIOD_RESULT_COLUMNS = [
    "period",
    "symbol",
    "scenario",
    "stop_loss_pct",
    "take_profit_pct",
    "max_holding_days",
    "final_value",
    "total_return_pct",
    "max_drawdown_pct",
    "total_trades",
    "closed_trades",
    "open_trades",
    "win_rate_pct",
    "profit_factor",
    "average_return_pct",
    "best_trade_return_pct",
    "worst_trade_return_pct",
    "average_holding_days",
    "currently_holding",
    "error",
]


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


def format_table_number_or_na(value) -> str:
    if is_missing(value):
        return "N/A"
    return f"{value:.2f}"


def format_table_pct_or_na(value) -> str:
    if is_missing(value):
        return "N/A"
    return f"{value:.2f}%"


def format_table_whole_number(value) -> str:
    if is_missing(value):
        return ""
    return f"{int(value)}"


def format_table_whole_number_or_na(value) -> str:
    if is_missing(value):
        return "N/A"
    return f"{int(value)}"


def format_display_date(value) -> str:
    if is_missing(value):
        return ""

    date_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(date_value):
        return ""
    return date_value.strftime("%Y-%m-%d")


def scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


def scenario_short_label(scenario: str) -> str:
    return SCENARIO_SHORT_LABELS.get(scenario, scenario)


def add_scenario_label_column(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    if "scenario" in display_df.columns:
        scenario_index = display_df.columns.get_loc("scenario") + 1
        display_df.insert(
            scenario_index,
            "scenario_label",
            display_df["scenario"].apply(scenario_label),
        )
    return display_df


def scenario_label_column_config() -> dict:
    return {
        "scenario_label": st.column_config.TextColumn(
            "Scenario label",
            width="large",
        )
    }


def order_by_ranking(df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or ranking_df.empty or "scenario" not in df.columns:
        return df

    ranking_order = {
        scenario: index for index, scenario in enumerate(ranking_df["scenario"].tolist())
    }
    ordered_df = df.copy()
    ordered_df["_scenario_order"] = ordered_df["scenario"].map(ranking_order)
    ordered_df = ordered_df.sort_values(["_scenario_order", "scenario"]).drop(
        columns="_scenario_order"
    )
    return ordered_df


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


def format_experiment_table(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    compact_columns = [
        "symbol",
        "scenario",
        "total_return_pct",
        "max_drawdown_pct",
        "profit_factor",
        "win_rate_pct",
        "final_value",
        "currently_holding",
        "error",
    ]
    display_df = results_df[compact_columns].copy()

    for column in ["total_return_pct", "max_drawdown_pct", "win_rate_pct"]:
        display_df[column] = display_df[column].apply(format_table_pct)

    display_df["profit_factor"] = display_df["profit_factor"].apply(
        format_table_number_or_na
    )
    display_df["final_value"] = display_df["final_value"].apply(format_table_number)

    display_df["currently_holding"] = display_df["currently_holding"].apply(
        lambda value: "" if is_missing(value) else str(bool(value))
    )
    display_df["error"] = display_df["error"].fillna("")

    return display_df.fillna("")


def format_average_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    display_df = summary_df.copy()
    for column in [
        "avg_total_return_pct",
        "avg_max_drawdown_pct",
        "avg_win_rate_pct",
    ]:
        display_df[column] = display_df[column].apply(format_table_pct)

    display_df["avg_profit_factor"] = display_df["avg_profit_factor"].apply(
        format_table_number_or_na
    )
    display_df["avg_average_holding_days"] = display_df[
        "avg_average_holding_days"
    ].apply(format_table_number)

    return display_df.fillna("")


def format_ranking_table(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if ranking_df.empty:
        return ranking_df

    display_df = ranking_df.copy()
    for column in [
        "avg_total_return_pct",
        "avg_max_drawdown_pct",
        "avg_win_rate_pct",
    ]:
        display_df[column] = display_df[column].apply(format_table_pct)

    display_df["avg_profit_factor"] = display_df["avg_profit_factor"].apply(
        format_table_number_or_na
    )
    for column in ["avg_average_holding_days", "score"]:
        display_df[column] = display_df[column].apply(format_table_number)

    return display_df.fillna("")


def format_period_results_table(results_df: pd.DataFrame, compact: bool) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    if compact:
        display_columns = [
            "period",
            "symbol",
            "scenario",
            "total_return_pct",
            "max_drawdown_pct",
            "profit_factor",
            "win_rate_pct",
            "final_value",
            "currently_holding",
            "error",
        ]
    else:
        display_columns = PERIOD_RESULT_COLUMNS

    display_df = results_df[display_columns].copy()
    display_df = add_scenario_label_column(display_df)

    for column in [
        "total_return_pct",
        "max_drawdown_pct",
        "win_rate_pct",
        "average_return_pct",
        "best_trade_return_pct",
        "worst_trade_return_pct",
    ]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_table_pct_or_na)

    for column in ["profit_factor", "final_value", "average_holding_days"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_table_number_or_na)

    for column in ["total_trades", "closed_trades", "open_trades"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                format_table_whole_number_or_na
            )

    for column in ["stop_loss_pct", "take_profit_pct"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_table_pct_or_na)

    if "max_holding_days" in display_df.columns:
        display_df["max_holding_days"] = display_df["max_holding_days"].apply(
            format_table_whole_number_or_na
        )

    if "currently_holding" in display_df.columns:
        display_df["currently_holding"] = display_df["currently_holding"].apply(
            lambda value: "N/A" if is_missing(value) else str(bool(value))
        )

    display_df["error"] = display_df["error"].fillna("")
    return display_df.fillna("N/A")


def format_period_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    display_df = summary_df.copy()
    display_df = add_scenario_label_column(display_df)
    for column in [
        "avg_total_return_pct",
        "avg_max_drawdown_pct",
        "avg_win_rate_pct",
    ]:
        display_df[column] = display_df[column].apply(format_table_pct_or_na)

    for column in ["avg_profit_factor", "avg_average_holding_days"]:
        display_df[column] = display_df[column].apply(format_table_number_or_na)

    return display_df.fillna("N/A")


def format_period_ranking_table(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if ranking_df.empty:
        return ranking_df

    display_df = format_period_summary_table(ranking_df)
    if "score" in display_df.columns:
        display_df["score"] = ranking_df["score"].apply(format_table_number_or_na)
    return display_df.fillna("N/A")


def get_best_metric_row(df: pd.DataFrame, metric_column: str) -> pd.Series | None:
    if df.empty or metric_column not in df.columns:
        return None

    metric_values = pd.to_numeric(df[metric_column], errors="coerce")
    available_values = metric_values.dropna()
    if available_values.empty:
        return None

    return df.loc[available_values.idxmax()]


def format_chart_date_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


def find_missing_columns(stock_data: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in stock_data.columns]


def validate_stock_data(stock_data: pd.DataFrame) -> None:
    missing_columns = find_missing_columns(stock_data)
    if missing_columns:
        st.error(
            "The selected CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()


def ensure_stock_data_or_raise(stock_data: pd.DataFrame) -> None:
    missing_columns = find_missing_columns(stock_data)
    if missing_columns:
        raise ValueError(
            "Data is missing required columns: " + ", ".join(missing_columns)
        )
    if stock_data.empty:
        raise ValueError("Data source returned no rows.")


def validate_non_empty_stock_data(stock_data: pd.DataFrame, label: str) -> None:
    if stock_data.empty:
        st.error(
            f"Real-data fetching failed: {label} returned no rows. "
            "Try checking the network, switching source mode, or using Demo data."
        )
        st.stop()


def format_date_for_loader(date_value: date) -> str:
    return date_value.strftime("%Y%m%d")


def parse_symbols(symbols_text: str) -> list[str]:
    symbols = [symbol.strip() for symbol in symbols_text.split(",")]
    return [
        symbol.zfill(6) if symbol.isdigit() and len(symbol) < 6 else symbol
        for symbol in symbols
        if symbol
    ]


def parse_periods(periods_text: str) -> list[str]:
    return [period.strip() for period in periods_text.split(",") if period.strip()]


def period_to_dates(period: str) -> tuple[str, str]:
    if not period.isdigit() or len(period) != 4:
        raise ValueError(f"Invalid period year: {period}. Expected YYYY.")
    return f"{period}0101", f"{period}1231"


def make_error_row(symbol: str, error: str) -> dict:
    row = {column: None for column in RESULT_COLUMNS}
    row["symbol"] = symbol
    row["scenario"] = "ERROR"
    row["error"] = error
    return row


def make_period_error_row(period: str, symbol: str, error: str) -> dict:
    row = {column: None for column in PERIOD_RESULT_COLUMNS}
    row["period"] = period
    row["symbol"] = symbol
    row["scenario"] = "ERROR"
    row["error"] = error
    return row


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


@st.cache_data(show_spinner=False)
def load_cached_demo_data() -> pd.DataFrame:
    return pd.read_csv(DEMO_DATA_PATH)


def prepare_strategy_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    prepared_data = stock_data.copy()
    prepared_data["date"] = pd.to_datetime(prepared_data["date"])
    prepared_data = add_all_indicators(prepared_data)
    prepared_data = generate_ma_crossover_signals(prepared_data)
    return prepared_data


def run_experiment_scenario(
    symbol: str,
    prepared_data: pd.DataFrame,
    initial_cash: float,
    scenario: dict,
) -> dict:
    backtest_df, trades_df = run_long_only_backtest_with_trades(
        prepared_data.copy(),
        initial_cash=initial_cash,
        stop_loss_pct=scenario["stop_loss_pct"],
        take_profit_pct=scenario["take_profit_pct"],
        max_holding_days=scenario["max_holding_days"],
    )
    performance = summarize_performance(backtest_df)
    trade_metrics = summarize_trade_metrics(trades_df)

    return {
        "symbol": symbol,
        "scenario": scenario["scenario"],
        "stop_loss_pct": scenario["stop_loss_pct"],
        "take_profit_pct": scenario["take_profit_pct"],
        "max_holding_days": scenario["max_holding_days"],
        "final_value": performance["final_value"],
        "total_return_pct": performance["total_return_pct"],
        "max_drawdown_pct": performance["max_drawdown_pct"],
        "total_trades": trade_metrics["total_trades"],
        "closed_trades": trade_metrics["closed_trades"],
        "open_trades": trade_metrics["open_trades"],
        "win_rate_pct": trade_metrics["win_rate_pct"],
        "profit_factor": trade_metrics["profit_factor"],
        "average_return_pct": trade_metrics["average_return_pct"],
        "best_trade_return_pct": trade_metrics["best_trade_return_pct"],
        "worst_trade_return_pct": trade_metrics["worst_trade_return_pct"],
        "average_holding_days": trade_metrics["average_holding_days"],
        "currently_holding": performance["currently_holding"],
        "error": None,
    }


def run_period_experiment_scenario(
    period: str,
    symbol: str,
    prepared_data: pd.DataFrame,
    initial_cash: float,
    scenario: dict,
) -> dict:
    result = run_experiment_scenario(
        symbol=symbol,
        prepared_data=prepared_data,
        initial_cash=initial_cash,
        scenario=scenario,
    )
    result["period"] = period
    return {column: result.get(column) for column in PERIOD_RESULT_COLUMNS}


def build_scenario_average_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    success_df = results_df[results_df["scenario"] != "ERROR"].copy()
    if success_df.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "symbols_tested",
                "avg_total_return_pct",
                "avg_max_drawdown_pct",
                "avg_profit_factor",
                "avg_win_rate_pct",
                "avg_average_holding_days",
            ]
        )

    for column in [
        "total_return_pct",
        "max_drawdown_pct",
        "profit_factor",
        "win_rate_pct",
        "average_holding_days",
    ]:
        success_df[column] = pd.to_numeric(success_df[column], errors="coerce")

    return (
        success_df.groupby("scenario", sort=False)
        .agg(
            symbols_tested=("symbol", "nunique"),
            avg_total_return_pct=("total_return_pct", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_win_rate_pct=("win_rate_pct", "mean"),
            avg_average_holding_days=("average_holding_days", "mean"),
        )
        .reset_index()
    )


def build_scenario_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "scenario",
                "symbols_tested",
                "avg_total_return_pct",
                "avg_max_drawdown_pct",
                "avg_profit_factor",
                "avg_win_rate_pct",
                "avg_average_holding_days",
                "score",
            ]
        )

    ranking_df = summary_df.copy()
    for column in [
        "avg_total_return_pct",
        "avg_max_drawdown_pct",
        "avg_profit_factor",
    ]:
        ranking_df[column] = pd.to_numeric(ranking_df[column], errors="coerce")

    ranking_df["score"] = (
        ranking_df["avg_total_return_pct"].fillna(0)
        + ranking_df["avg_profit_factor"].fillna(0) * 2
        + ranking_df["avg_max_drawdown_pct"].fillna(0) * 0.3
    )
    ranking_df = ranking_df.sort_values("score", ascending=False).reset_index(
        drop=True
    )
    ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))

    return ranking_df[
        [
            "rank",
            "scenario",
            "symbols_tested",
            "avg_total_return_pct",
            "avg_max_drawdown_pct",
            "avg_profit_factor",
            "avg_win_rate_pct",
            "avg_average_holding_days",
            "score",
        ]
    ]


def build_period_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    success_df = results_df[results_df["scenario"] != "ERROR"].copy()
    columns = [
        "period",
        "scenario",
        "symbols_tested",
        "avg_total_return_pct",
        "avg_max_drawdown_pct",
        "avg_profit_factor",
        "avg_win_rate_pct",
        "avg_average_holding_days",
    ]
    if success_df.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "total_return_pct",
        "max_drawdown_pct",
        "profit_factor",
        "win_rate_pct",
        "average_holding_days",
    ]:
        success_df[column] = pd.to_numeric(success_df[column], errors="coerce")

    return (
        success_df.groupby(["period", "scenario"], sort=False)
        .agg(
            symbols_tested=("symbol", "nunique"),
            avg_total_return_pct=("total_return_pct", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_win_rate_pct=("win_rate_pct", "mean"),
            avg_average_holding_days=("average_holding_days", "mean"),
        )
        .reset_index()
    )


def build_period_overall_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    success_df = results_df[results_df["scenario"] != "ERROR"].copy()
    columns = [
        "scenario",
        "periods_tested",
        "symbols_tested",
        "total_cases",
        "avg_total_return_pct",
        "avg_max_drawdown_pct",
        "avg_profit_factor",
        "avg_win_rate_pct",
        "avg_average_holding_days",
    ]
    if success_df.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "total_return_pct",
        "max_drawdown_pct",
        "profit_factor",
        "win_rate_pct",
        "average_holding_days",
    ]:
        success_df[column] = pd.to_numeric(success_df[column], errors="coerce")

    return (
        success_df.groupby("scenario", sort=False)
        .agg(
            periods_tested=("period", "nunique"),
            symbols_tested=("symbol", "nunique"),
            total_cases=("symbol", "count"),
            avg_total_return_pct=("total_return_pct", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_win_rate_pct=("win_rate_pct", "mean"),
            avg_average_holding_days=("average_holding_days", "mean"),
        )
        .reset_index()
    )


def build_period_scenario_ranking(overall_df: pd.DataFrame) -> pd.DataFrame:
    if overall_df.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "scenario",
                "periods_tested",
                "symbols_tested",
                "total_cases",
                "avg_total_return_pct",
                "avg_max_drawdown_pct",
                "avg_profit_factor",
                "avg_win_rate_pct",
                "avg_average_holding_days",
                "score",
            ]
        )

    ranking_df = overall_df.copy()
    for column in [
        "avg_total_return_pct",
        "avg_max_drawdown_pct",
        "avg_profit_factor",
    ]:
        ranking_df[column] = pd.to_numeric(ranking_df[column], errors="coerce")

    ranking_df["score"] = (
        ranking_df["avg_total_return_pct"].fillna(0)
        + ranking_df["avg_profit_factor"].fillna(0) * 2
        + ranking_df["avg_max_drawdown_pct"].fillna(0) * 0.3
    )
    ranking_df = ranking_df.sort_values("score", ascending=False).reset_index(
        drop=True
    )
    ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))

    return ranking_df[
        [
            "rank",
            "scenario",
            "periods_tested",
            "symbols_tested",
            "total_cases",
            "avg_total_return_pct",
            "avg_max_drawdown_pct",
            "avg_profit_factor",
            "avg_win_rate_pct",
            "avg_average_holding_days",
            "score",
        ]
    ]


def describe_best_metric(
    ranking_df: pd.DataFrame,
    metric_column: str,
    label: str,
    value_suffix: str = "",
    value_prefix: str = "",
) -> str:
    if metric_column not in ranking_df.columns:
        return f"{label}: N/A"

    metric_values = pd.to_numeric(ranking_df[metric_column], errors="coerce")
    available_values = metric_values.dropna()
    if available_values.empty:
        return f"{label}: N/A"

    best_index = available_values.idxmax()
    best_row = ranking_df.loc[best_index]
    return (
        f"{label}: {best_row['scenario']} "
        f"({value_prefix}{metric_values.loc[best_index]:.2f}{value_suffix})"
    )


def build_quick_interpretation(ranking_df: pd.DataFrame) -> list[str]:
    if ranking_df.empty:
        return ["No successful scenario results are available for interpretation."]

    return [
        describe_best_metric(
            ranking_df,
            "score",
            "Best overall scenario by score",
            value_prefix="score ",
        ),
        describe_best_metric(
            ranking_df,
            "avg_total_return_pct",
            "Best average return scenario",
            "%",
        ),
        describe_best_metric(
            ranking_df,
            "avg_max_drawdown_pct",
            "Best drawdown control scenario",
            "%",
        ),
        describe_best_metric(
            ranking_df,
            "avg_profit_factor",
            "Best profit factor scenario",
        ),
    ]


def run_parameter_experiment(
    symbols: list[str],
    source: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for symbol in symbols:
        try:
            if source == "demo":
                stock_data = load_cached_demo_data()
            else:
                stock_data = fetch_real_stock_data(
                    symbol=symbol,
                    source=source,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                )

            ensure_stock_data_or_raise(stock_data)

            prepared_data = prepare_strategy_data(stock_data)
            rows.extend(
                run_experiment_scenario(
                    symbol=symbol,
                    prepared_data=prepared_data,
                    initial_cash=initial_cash,
                    scenario=scenario,
                )
                for scenario in SCENARIOS
            )
        except Exception as exc:
            rows.append(make_error_row(symbol, str(exc)))

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary_df = build_scenario_average_summary(results_df)
    ranking_df = build_scenario_ranking(summary_df)

    return results_df, summary_df, ranking_df


def run_period_experiment(
    symbols: list[str],
    periods: list[str],
    source: str,
    adjust: str,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for period in periods:
        try:
            start_date, end_date = period_to_dates(period)
        except ValueError as exc:
            for symbol in symbols:
                rows.append(make_period_error_row(period, symbol, str(exc)))
            continue

        for symbol in symbols:
            try:
                if source == "demo":
                    stock_data = load_cached_demo_data()
                else:
                    stock_data = fetch_real_stock_data(
                        symbol=symbol,
                        source=source,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust,
                    )

                ensure_stock_data_or_raise(stock_data)
                prepared_data = prepare_strategy_data(stock_data)
                rows.extend(
                    run_period_experiment_scenario(
                        period=period,
                        symbol=symbol,
                        prepared_data=prepared_data,
                        initial_cash=initial_cash,
                        scenario=scenario,
                    )
                    for scenario in SCENARIOS
                )
            except Exception as exc:
                rows.append(make_period_error_row(period, symbol, str(exc)))

    results_df = pd.DataFrame(rows, columns=PERIOD_RESULT_COLUMNS)
    period_summary_df = build_period_summary(results_df)
    overall_summary_df = build_period_overall_summary(results_df)
    ranking_df = build_period_scenario_ranking(overall_summary_df)

    return results_df, period_summary_df, overall_summary_df, ranking_df


def load_demo_data() -> tuple[pd.DataFrame, str, dict]:
    if not DEMO_DATA_PATH.exists():
        st.error(f"Demo CSV is missing: {DEMO_DATA_PATH}")
        st.stop()

    return (
        load_cached_demo_data(),
        f"Demo data: {DEMO_DATA_PATH}",
        {"mode": "Demo data", "ready": True},
    )


def load_real_data(source_mode: str) -> tuple[pd.DataFrame | None, str, dict]:
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
            st.error(
                "Real-data fetching failed. Try checking the network, "
                "switching source mode, or using Demo data. "
                f"Details: {exc}"
            )
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
        details = {
            "mode": source_mode,
            "source": source,
            "symbol": symbol,
            "start": f"{start_date_value:%Y-%m-%d}",
            "end": f"{end_date_value:%Y-%m-%d}",
            "adjust": adjust,
            "ready": False,
        }
        return None, "", details

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
        "ready": True,
    }
    return stored_result["data"].copy(), data_label, details


def load_selected_data() -> tuple[pd.DataFrame | None, str, dict]:
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


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_single_backtest_tab(
    initial_cash: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    max_holding_days: int | None,
) -> None:
    stock_data, data_label, data_details = load_selected_data()
    if stock_data is None:
        return

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


def display_parameter_experiment_outputs(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    error_rows = results_df[
        results_df["error"].notna() & (results_df["error"].astype(str).str.strip() != "")
    ]
    if not error_rows.empty:
        failed_symbols = ", ".join(error_rows["symbol"].dropna().astype(str).unique())
        st.warning(
            "Some symbols failed during real-data fetching or processing. "
            "They are kept as ERROR rows so the rest of the experiment can continue. "
            f"Failed symbols: {failed_symbols}"
        )

    st.subheader("Compact Batch Results")
    compact_df = format_experiment_table(results_df)
    st.dataframe(compact_df, width="stretch")
    st.download_button(
        label="Download compact batch results CSV",
        data=dataframe_to_csv_bytes(compact_df),
        file_name="compact_batch_results.csv",
        mime="text/csv",
        key="download_compact_batch_results_csv",
        type="secondary",
    )

    st.subheader("Scenario Average Summary")
    summary_display_df = format_average_summary_table(summary_df)
    st.dataframe(summary_display_df, width="stretch")
    st.download_button(
        label="Download scenario summary CSV",
        data=dataframe_to_csv_bytes(summary_df),
        file_name="scenario_average_summary.csv",
        mime="text/csv",
        key="download_scenario_summary_csv",
        type="secondary",
    )

    st.subheader("Scenario Ranking")
    ranking_display_df = format_ranking_table(ranking_df)
    st.dataframe(ranking_display_df, width="stretch")
    st.download_button(
        label="Download scenario ranking CSV",
        data=dataframe_to_csv_bytes(ranking_df),
        file_name="scenario_ranking.csv",
        mime="text/csv",
        key="download_scenario_ranking_csv",
        type="secondary",
    )

    st.subheader("Quick Interpretation")
    for line in build_quick_interpretation(ranking_df):
        st.write(line)


def display_period_experiment_outputs(
    results_df: pd.DataFrame,
    period_summary_df: pd.DataFrame,
    overall_summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    compact: bool,
) -> None:
    error_rows = results_df[
        results_df["error"].notna()
        & (results_df["error"].astype(str).str.strip() != "")
    ]
    if not error_rows.empty:
        failed_pairs = (
            error_rows[["period", "symbol"]]
            .dropna()
            .drop_duplicates()
            .astype(str)
        )
        failed_labels = ", ".join(
            f"{row.period}/{row.symbol}" for row in failed_pairs.itertuples()
        )
        st.warning(
            "Some symbol-period fetches failed during real-data fetching or "
            "processing. They are kept as ERROR rows so the rest of the "
            f"experiment can continue. Failed cases: {failed_labels}"
        )

    st.subheader("Period Experiment Summary")
    show_period_summary_cards(results_df, overall_summary_df, ranking_df)

    st.subheader("Period Experiment Charts")
    show_period_experiment_charts(overall_summary_df, ranking_df)

    result_title = "Compact Period Results" if compact else "Period Experiment Results"
    st.subheader(result_title)
    ordered_results_df = order_by_ranking(results_df, ranking_df)
    period_results_display_df = format_period_results_table(ordered_results_df, compact)
    st.dataframe(
        period_results_display_df,
        width="stretch",
        column_config=scenario_label_column_config(),
    )
    compact_period_df = format_period_results_table(results_df, True)
    st.download_button(
        label="Download compact period results CSV",
        data=dataframe_to_csv_bytes(compact_period_df),
        file_name="compact_period_results.csv",
        mime="text/csv",
        key="download_compact_period_results_csv",
        type="secondary",
    )

    st.subheader("Scenario Period Summary")
    ordered_period_summary_df = order_by_ranking(period_summary_df, ranking_df)
    period_summary_display_df = format_period_summary_table(ordered_period_summary_df)
    st.dataframe(
        period_summary_display_df,
        width="stretch",
        column_config=scenario_label_column_config(),
    )
    st.download_button(
        label="Download scenario period summary CSV",
        data=dataframe_to_csv_bytes(period_summary_df),
        file_name="scenario_period_summary.csv",
        mime="text/csv",
        key="download_scenario_period_summary_csv",
        type="secondary",
    )

    st.subheader("Overall Scenario Summary")
    ordered_overall_summary_df = order_by_ranking(overall_summary_df, ranking_df)
    overall_summary_display_df = format_period_summary_table(
        ordered_overall_summary_df
    )
    st.dataframe(
        overall_summary_display_df,
        width="stretch",
        column_config=scenario_label_column_config(),
    )
    st.download_button(
        label="Download overall scenario summary CSV",
        data=dataframe_to_csv_bytes(overall_summary_df),
        file_name="overall_scenario_summary.csv",
        mime="text/csv",
        key="download_overall_scenario_summary_csv",
        type="secondary",
    )

    st.subheader("Scenario Ranking")
    ranking_display_df = format_period_ranking_table(ranking_df)
    st.dataframe(
        ranking_display_df,
        width="stretch",
        column_config=scenario_label_column_config(),
    )
    st.info(
        "Higher score is better. The score is a simple educational ranking "
        "formula for comparing scenarios, not an investment recommendation."
    )
    st.download_button(
        label="Download scenario ranking CSV",
        data=dataframe_to_csv_bytes(ranking_df),
        file_name="period_scenario_ranking.csv",
        mime="text/csv",
        key="download_period_scenario_ranking_csv",
        type="secondary",
    )

    st.subheader("Quick Interpretation")
    for line in build_quick_interpretation(ranking_df):
        st.write(line)

    st.subheader("How to read this experiment")
    st.write(
        "Higher average return is better. Smaller drawdown loss is better, "
        "which means a drawdown closer to 0% is usually preferred. Higher "
        "profit factor is better, but it should be interpreted together with "
        "return, drawdown, and trade count. A strategy should not be judged by "
        "one stock or one year only. This is educational research, not "
        "financial advice."
    )


def show_period_summary_cards(
    results_df: pd.DataFrame,
    overall_summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    columns = st.columns(2)

    best_overall = get_best_metric_row(ranking_df, "score")
    render_period_summary_card(
        columns[0],
        "Best overall scenario",
        best_overall,
        "score",
        "Score",
    )

    best_return = get_best_metric_row(overall_summary_df, "avg_total_return_pct")
    render_period_summary_card(
        columns[1],
        "Best average return",
        best_return,
        "avg_total_return_pct",
        "Average return",
        "%",
    )

    columns = st.columns(3)
    best_drawdown = get_best_metric_row(overall_summary_df, "avg_max_drawdown_pct")
    render_period_summary_card(
        columns[0],
        "Best drawdown control",
        best_drawdown,
        "avg_max_drawdown_pct",
        "Average max drawdown",
        "%",
    )

    best_profit_factor = get_best_metric_row(overall_summary_df, "avg_profit_factor")
    render_period_summary_card(
        columns[1],
        "Best profit factor",
        best_profit_factor,
        "avg_profit_factor",
        "Profit factor",
    )

    success_df = results_df[results_df["scenario"] != "ERROR"]
    columns[2].markdown(
        f"""
        **Number of total cases tested**

        {len(success_df)}
        """
    )


def render_period_summary_card(
    column,
    title: str,
    row: pd.Series | None,
    metric_column: str,
    metric_label: str,
    suffix: str = "",
) -> None:
    if row is None or metric_column not in row or is_missing(row[metric_column]):
        column.markdown(f"**{title}**\n\nN/A")
        return

    scenario_code = row["scenario"]
    column.markdown(
        f"""
        **{title}**

        Scenario code: `{scenario_code}`  
        Label: {scenario_label(scenario_code)}  
        {metric_label}: {row[metric_column]:.2f}{suffix}
        """
    )


def show_period_bar_chart(
    df: pd.DataFrame,
    metric_column: str,
    title: str,
    ranking_df: pd.DataFrame,
    value_suffix: str = "",
) -> None:
    if df.empty or metric_column not in df.columns:
        st.info(f"{title} is unavailable because the metric is missing.")
        return

    chart_df = order_by_ranking(df, ranking_df)
    chart_df = chart_df[["scenario", metric_column]].copy()
    chart_df[metric_column] = pd.to_numeric(chart_df[metric_column], errors="coerce")
    chart_df = chart_df.dropna(subset=[metric_column])
    if chart_df.empty:
        st.info(f"{title} is unavailable because all values are N/A.")
        return

    chart_df["scenario_short_label"] = chart_df["scenario"].apply(scenario_short_label)
    st.write(title)
    st.caption(
        "Charts use short scenario labels. Full scenario labels and raw codes "
        "are shown in the tables below."
    )

    fig_height = max(3.0, 0.45 * len(chart_df))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(chart_df["scenario_short_label"], chart_df[metric_column], color="tab:blue")
    ax.invert_yaxis()
    ax.set_xlabel(title)
    ax.set_ylabel("Scenario")
    ax.grid(True, axis="x", alpha=0.25)

    for index, value in enumerate(chart_df[metric_column]):
        label = f"{value:.2f}{value_suffix}"
        ax.text(
            value,
            index,
            f" {label}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_period_experiment_charts(
    overall_summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    show_period_bar_chart(
        overall_summary_df,
        "avg_total_return_pct",
        "Average total return by scenario",
        ranking_df,
        "%",
    )
    show_period_bar_chart(
        overall_summary_df,
        "avg_max_drawdown_pct",
        "Average maximum drawdown by scenario",
        ranking_df,
        "%",
    )
    show_period_bar_chart(
        overall_summary_df,
        "avg_profit_factor",
        "Average profit factor by scenario",
        ranking_df,
    )
    show_period_bar_chart(
        ranking_df,
        "score",
        "Scenario score by scenario",
        ranking_df,
    )


def render_parameter_experiment_tab() -> None:
    st.write(
        "Compare the default risk-control scenarios across multiple symbols. "
        "Real-data modes fetch each symbol once, then run all scenarios in memory."
    )

    symbols_text = st.text_input(
        "Symbols",
        value="000001,600519",
        help="Comma-separated A-share symbols, for example 000001,600519.",
    )
    start_text = st.text_input("Start date", value="20240101")
    end_text = st.text_input("End date", value="20241231")
    source = st.selectbox("Source mode", EXPERIMENT_SOURCE_OPTIONS)
    if source == "demo":
        st.warning(
            "Demo mode reuses the local sample CSV for each entered symbol, "
            "so different symbols may show identical results. Use Baostock "
            "or Akshare real data for real multi-stock comparison."
        )
    else:
        st.info(
            "Real-data mode fetches each symbol separately, then runs all "
            "risk-control scenarios in memory."
        )

    initial_cash = st.number_input(
        "Experiment initial cash",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
    )

    symbols = parse_symbols(symbols_text)
    if not symbols:
        st.warning("Enter at least one symbol.")
        return

    if source != "demo":
        try:
            pd.to_datetime(start_text, format="%Y%m%d")
            pd.to_datetime(end_text, format="%Y%m%d")
        except ValueError:
            st.error("Start date and end date must use YYYYMMDD format.")
            return

    run_clicked = st.button(
        label="Run experiment",
        key="run_parameter_experiment_button",
        type="primary",
    )
    if run_clicked:
        with st.spinner("Running parameter experiment..."):
            results_df, summary_df, ranking_df = run_parameter_experiment(
                symbols=symbols,
                source=source,
                start_date=start_text,
                end_date=end_text,
                initial_cash=initial_cash,
            )

        st.session_state["parameter_experiment_result"] = {
            "inputs": (tuple(symbols), source, start_text, end_text, initial_cash),
            "results": results_df,
            "summary": summary_df,
            "ranking": ranking_df,
        }

    stored_result = st.session_state.get("parameter_experiment_result")
    current_inputs = (tuple(symbols), source, start_text, end_text, initial_cash)
    if stored_result is None or stored_result["inputs"] != current_inputs:
        st.info("Set experiment inputs, then click 'Run experiment'.")
        return

    display_parameter_experiment_outputs(
        stored_result["results"],
        stored_result["summary"],
        stored_result["ranking"],
    )


def render_period_experiment_tab() -> None:
    st.write(
        "Test the same risk-control scenarios across multiple symbols and "
        "multiple year periods."
    )

    symbols_text = st.text_input(
        "Period experiment symbols",
        value="000001,600519,000858,600036,601318",
        help="Comma-separated A-share symbols.",
    )
    periods_text = st.text_input(
        "Periods",
        value="2021,2022,2023,2024,2025",
        help="Comma-separated years, for example 2021,2022,2023.",
    )
    source = st.selectbox("Period source mode", PERIOD_SOURCE_OPTIONS)
    if source == "demo":
        st.warning(
            "Demo mode reuses the local sample CSV for every symbol and year. "
            "Results are useful for checking the workflow, not real multi-year "
            "market conclusions."
        )
    else:
        st.info(
            "Real-data mode fetches each symbol-year once, then runs all "
            "risk-control scenarios in memory."
        )

    initial_cash = st.number_input(
        "Period experiment initial cash",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
    )
    adjust = st.text_input("Adjust mode", value="qfq")
    compact = st.checkbox("Compact mode", value=True)

    symbols = parse_symbols(symbols_text)
    periods = parse_periods(periods_text)
    if not symbols:
        st.warning("Enter at least one symbol.")
        return
    if not periods:
        st.warning("Enter at least one period year.")
        return

    invalid_periods = [
        period for period in periods if not period.isdigit() or len(period) != 4
    ]
    if invalid_periods:
        st.error(
            "Periods must be four-digit years. Invalid values: "
            + ", ".join(invalid_periods)
        )
        return

    if not adjust.strip():
        st.error("Adjust mode cannot be blank. Use qfq, hfq, or none.")
        return

    run_clicked = st.button(
        label="Run period experiment",
        key="run_period_experiment_button",
        type="primary",
    )
    if run_clicked:
        with st.spinner("Running period experiment..."):
            results_df, period_summary_df, overall_summary_df, ranking_df = (
                run_period_experiment(
                    symbols=symbols,
                    periods=periods,
                    source=source,
                    adjust=adjust.strip(),
                    initial_cash=initial_cash,
                )
            )

        st.session_state["period_experiment_result"] = {
            "inputs": (
                tuple(symbols),
                tuple(periods),
                source,
                adjust.strip(),
                initial_cash,
                compact,
            ),
            "results": results_df,
            "period_summary": period_summary_df,
            "overall_summary": overall_summary_df,
            "ranking": ranking_df,
        }

    stored_result = st.session_state.get("period_experiment_result")
    current_inputs = (
        tuple(symbols),
        tuple(periods),
        source,
        adjust.strip(),
        initial_cash,
        compact,
    )
    if stored_result is None or stored_result["inputs"] != current_inputs:
        st.info("Set period experiment inputs, then click 'Run period experiment'.")
        return

    display_period_experiment_outputs(
        stored_result["results"],
        stored_result["period_summary"],
        stored_result["overall_summary"],
        stored_result["ranking"],
        compact=compact,
    )


def main() -> None:
    st.set_page_config(page_title="QuantPilot-AI Dashboard", layout="wide")

    st.title("QuantPilot-AI Dashboard")
    st.caption(
        "Educational rule-based backtesting demo. This is not financial advice, "
        "and backtest results do not guarantee future returns."
    )

    st.sidebar.header("Single Backtest Settings")
    st.sidebar.caption(
        "These controls apply only to the Single Backtest tab. "
        "Experiment tabs have their own settings inside each tab."
    )
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

    single_tab, experiment_tab, period_tab = st.tabs(
        ["Single Backtest", "Parameter Experiment", "Period Experiment"]
    )
    with single_tab:
        render_single_backtest_tab(
            initial_cash=initial_cash,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_holding_days=max_holding_days,
        )

    with experiment_tab:
        render_parameter_experiment_tab()

    with period_tab:
        render_period_experiment_tab()


if __name__ == "__main__":
    main()
