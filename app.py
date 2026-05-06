from datetime import date
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import streamlit as st

from src.backtester import run_long_only_backtest_with_trades
from src.batch_model_trainer import (
    parse_model_types as parse_batch_model_types,
    parse_symbols as parse_batch_symbols,
    run_batch_model_training,
)
from src.candidate_expanded_validation import (
    parse_symbols as parse_candidate_validation_symbols,
    save_candidate_expanded_validation,
)
from src.candidate_equivalence_audit import save_candidate_equivalence_audit
from src.candidate_mode_normalization import save_canonical_mode_report
from src.candidate_stress_test import save_candidate_stress_test
from src.canonical_candidate_revalidation_report import (
    save_canonical_candidate_revalidation_report,
)
from src.candidate_validation_gate import save_candidate_validation_gate
from src.validation_gate_failure_analysis import (
    save_validation_gate_failure_analysis,
)
from src.targeted_remediation_design import save_targeted_remediation_design
from src.bull_regime_threshold_remediation import (
    DEFAULT_BUY_THRESHOLDS as DEFAULT_BULL_REMEDIATION_BUYS,
    DEFAULT_SELL_THRESHOLDS as DEFAULT_BULL_REMEDIATION_SELLS,
    parse_symbols as parse_bull_remediation_symbols,
    parse_threshold_list as parse_bull_remediation_thresholds,
    save_bull_regime_threshold_remediation,
)
from src.sideways_regime_trade_sufficiency_remediation import (
    DEFAULT_BUY_THRESHOLDS as DEFAULT_SIDEWAYS_REMEDIATION_BUYS,
    DEFAULT_SELL_THRESHOLDS as DEFAULT_SIDEWAYS_REMEDIATION_SELLS,
    parse_symbols as parse_sideways_remediation_symbols,
    parse_threshold_list as parse_sideways_remediation_thresholds,
    save_sideways_regime_trade_sufficiency_remediation,
)
from src.feature_source_registry import (
    get_high_leakage_risk_features,
    get_token_required_features,
    get_training_ready_features,
    registry_to_dataframe,
    summarize_factor_families,
)
from src.feature_implementation_queue import (
    filter_feature_queue,
    queue_to_dataframe,
    summarize_feature_queue,
)
from src.factor_ablation import parse_ablation_modes as parse_factor_ablation_modes
from src.factor_ablation import parse_model_types as parse_ablation_model_types
from src.batch_model_trainer import fetch_symbol_ohlcv
from src.build_factor_dataset import save_factor_dataset
from src.factor_builder import build_factor_dataset
from src.factor_ablation import (
    build_feature_impact_ranking,
    build_group_summary,
    run_and_save_factor_ablation,
)
from src.factor_decision_report import (
    generate_factor_decision_report,
    write_factor_decision_report,
)
from src.factor_pruning_experiment import (
    parse_pruning_modes,
    run_and_save_factor_pruning_experiment,
)
from src.pruning_summary_report import (
    DEFAULT_INPUT_DIRS as DEFAULT_PRUNING_SUMMARY_DIRS,
    build_pruning_summary_report,
    parse_input_dirs as parse_pruning_summary_input_dirs,
    save_pruning_summary_report,
)
from src.reduced_feature_backtest import run_and_save_reduced_feature_backtest
from src.reduced_feature_backtest_report import (
    DEFAULT_INPUT_DIRS as DEFAULT_REDUCED_FEATURE_SUMMARY_DIRS,
    parse_input_dirs as parse_reduced_feature_summary_input_dirs,
    save_reduced_feature_backtest_report,
)
from src.reduced_feature_threshold_experiment import (
    DEFAULT_BUY_THRESHOLDS as DEFAULT_REDUCED_THRESHOLD_BUYS,
    DEFAULT_SELL_THRESHOLDS as DEFAULT_REDUCED_THRESHOLD_SELLS,
    parse_thresholds as parse_reduced_thresholds,
    run_and_save_threshold_experiment,
)
from src.generate_threshold_experiment_report import (
    parse_input_dirs as parse_threshold_report_input_dirs,
    save_threshold_experiment_report,
)
from src.threshold_decision_report import save_threshold_decision_report
from src.indicators import add_all_indicators
from src.metrics import summarize_performance
from src.ml_signal_backtester import run_ml_signal_backtest
from src.ml_threshold_experiment import (
    parse_thresholds as parse_ml_thresholds,
    rank_threshold_results,
    run_threshold_experiment,
    run_walk_forward_threshold_experiment,
)
from src.model_evaluator import evaluate_model_directory
from src.model_predictor import run_model_prediction
from src.model_report_generator import write_model_robustness_report
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
    "benchmark_final_value",
    "benchmark_return_pct",
    "benchmark_max_drawdown_pct",
    "strategy_vs_benchmark_pct",
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
    "baseline": "Base",
    "sl_3": "SL3",
    "sl_5": "SL5",
    "tp_10": "TP10",
    "max_30": "Max30",
    "sl_3_tp_10": "SL3+TP10",
    "sl_3_max_30": "SL3+Max30",
    "sl_3_tp_10_max_30": "SL3+TP10+Max30",
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
    "benchmark_final_value",
    "benchmark_return_pct",
    "benchmark_max_drawdown_pct",
    "strategy_vs_benchmark_pct",
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
        "entry_commission",
        "exit_commission",
        "stamp_tax",
        "slippage_cost",
        "total_transaction_cost",
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
        "benchmark_return_pct",
        "strategy_vs_benchmark_pct",
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
        "total_commission",
        "total_stamp_tax",
        "total_slippage_cost",
        "total_transaction_cost",
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


def calculate_max_drawdown_from_values(values: pd.Series) -> float:
    running_max = values.cummax()
    drawdown = values / running_max - 1
    return float(drawdown.min() * 100)


def build_buy_and_hold_equity_curve(
    stock_data: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    benchmark_df = stock_data[["date", "close"]].copy()
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
    benchmark_df["close"] = pd.to_numeric(benchmark_df["close"], errors="coerce")
    benchmark_df = benchmark_df.dropna(subset=["close"]).reset_index(drop=True)
    if benchmark_df.empty:
        return pd.DataFrame(columns=["date", "benchmark_total_value"])

    first_close = benchmark_df["close"].iloc[0]
    if first_close <= 0:
        return pd.DataFrame(columns=["date", "benchmark_total_value"])

    shares = initial_cash / first_close
    benchmark_df["benchmark_total_value"] = shares * benchmark_df["close"]
    return benchmark_df[["date", "benchmark_total_value"]]


def calculate_buy_and_hold_benchmark(
    stock_data: pd.DataFrame,
    initial_cash: float,
) -> dict:
    benchmark_curve = build_buy_and_hold_equity_curve(stock_data, initial_cash)
    if benchmark_curve.empty:
        return {
            "benchmark_final_value": None,
            "benchmark_return_pct": None,
            "benchmark_max_drawdown_pct": None,
        }

    final_value = float(benchmark_curve["benchmark_total_value"].iloc[-1])
    benchmark_return_pct = ((final_value - initial_cash) / initial_cash) * 100
    benchmark_max_drawdown_pct = calculate_max_drawdown_from_values(
        benchmark_curve["benchmark_total_value"]
    )
    return {
        "benchmark_final_value": final_value,
        "benchmark_return_pct": float(benchmark_return_pct),
        "benchmark_max_drawdown_pct": benchmark_max_drawdown_pct,
    }


def format_experiment_table(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    compact_columns = [
        "symbol",
        "scenario",
        "total_return_pct",
        "benchmark_return_pct",
        "strategy_vs_benchmark_pct",
        "max_drawdown_pct",
        "profit_factor",
        "win_rate_pct",
        "final_value",
        "currently_holding",
        "error",
    ]
    display_df = results_df[compact_columns].copy()

    for column in [
        "total_return_pct",
        "benchmark_return_pct",
        "strategy_vs_benchmark_pct",
        "max_drawdown_pct",
        "win_rate_pct",
    ]:
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
        "avg_strategy_vs_benchmark_pct",
        "avg_max_drawdown_pct",
        "avg_win_rate_pct",
    ]:
        if column in display_df.columns:
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
        "avg_strategy_vs_benchmark_pct",
        "avg_max_drawdown_pct",
        "avg_win_rate_pct",
    ]:
        if column in display_df.columns:
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
            "benchmark_return_pct",
            "strategy_vs_benchmark_pct",
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
    return display_df.fillna("N/A").reset_index(drop=True)


def format_period_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    display_df = summary_df.copy()
    display_df = add_scenario_label_column(display_df)
    for column in [
        "avg_total_return_pct",
        "avg_strategy_vs_benchmark_pct",
        "avg_max_drawdown_pct",
        "avg_win_rate_pct",
    ]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_table_pct_or_na)

    for column in ["avg_profit_factor", "avg_average_holding_days"]:
        display_df[column] = display_df[column].apply(format_table_number_or_na)

    return display_df.fillna("N/A").reset_index(drop=True)


def format_period_ranking_table(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if ranking_df.empty:
        return ranking_df

    display_df = format_period_summary_table(ranking_df)
    if "score" in display_df.columns:
        display_df["score"] = ranking_df["score"].apply(format_table_number_or_na)
    return display_df.fillna("N/A").reset_index(drop=True)


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
    benchmark = calculate_buy_and_hold_benchmark(prepared_data, initial_cash)
    benchmark_return_pct = benchmark["benchmark_return_pct"]
    strategy_vs_benchmark_pct = (
        None
        if benchmark_return_pct is None
        else performance["total_return_pct"] - benchmark_return_pct
    )

    return {
        "symbol": symbol,
        "scenario": scenario["scenario"],
        "stop_loss_pct": scenario["stop_loss_pct"],
        "take_profit_pct": scenario["take_profit_pct"],
        "max_holding_days": scenario["max_holding_days"],
        "final_value": performance["final_value"],
        "total_return_pct": performance["total_return_pct"],
        "max_drawdown_pct": performance["max_drawdown_pct"],
        "benchmark_final_value": benchmark["benchmark_final_value"],
        "benchmark_return_pct": benchmark_return_pct,
        "benchmark_max_drawdown_pct": benchmark["benchmark_max_drawdown_pct"],
        "strategy_vs_benchmark_pct": strategy_vs_benchmark_pct,
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
                "avg_strategy_vs_benchmark_pct",
                "avg_max_drawdown_pct",
                "avg_profit_factor",
                "avg_win_rate_pct",
                "avg_average_holding_days",
            ]
        )

    for column in [
        "total_return_pct",
        "strategy_vs_benchmark_pct",
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
            avg_strategy_vs_benchmark_pct=("strategy_vs_benchmark_pct", "mean"),
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
                "avg_strategy_vs_benchmark_pct",
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
            "avg_strategy_vs_benchmark_pct",
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
        "avg_strategy_vs_benchmark_pct",
        "avg_max_drawdown_pct",
        "avg_profit_factor",
        "avg_win_rate_pct",
        "avg_average_holding_days",
    ]
    if success_df.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "total_return_pct",
        "strategy_vs_benchmark_pct",
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
            avg_strategy_vs_benchmark_pct=("strategy_vs_benchmark_pct", "mean"),
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
        "avg_strategy_vs_benchmark_pct",
        "avg_max_drawdown_pct",
        "avg_profit_factor",
        "avg_win_rate_pct",
        "avg_average_holding_days",
    ]
    if success_df.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "total_return_pct",
        "strategy_vs_benchmark_pct",
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
            avg_strategy_vs_benchmark_pct=("strategy_vs_benchmark_pct", "mean"),
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
                "avg_strategy_vs_benchmark_pct",
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
            "avg_strategy_vs_benchmark_pct",
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


def show_benchmark_comparison_cards(
    performance_summary: dict,
    benchmark: dict,
) -> None:
    strategy_return = performance_summary["total_return_pct"]
    strategy_drawdown = performance_summary["max_drawdown_pct"]
    benchmark_return = benchmark["benchmark_return_pct"]
    benchmark_drawdown = benchmark["benchmark_max_drawdown_pct"]
    difference = (
        None if benchmark_return is None else strategy_return - benchmark_return
    )

    st.subheader("Benchmark Comparison")
    columns = st.columns(5)
    columns[0].metric("Strategy return", format_metric_pct(strategy_return))
    columns[1].metric("Buy-and-hold return", format_metric_pct(benchmark_return))
    columns[2].metric("Difference vs benchmark", format_metric_pct(difference))
    columns[3].metric("Strategy max drawdown", format_metric_pct(strategy_drawdown))
    columns[4].metric(
        "Buy-and-hold max drawdown",
        format_metric_pct(benchmark_drawdown),
    )


def show_equity_comparison_curve(
    backtest_result: pd.DataFrame,
    stock_data: pd.DataFrame,
    initial_cash: float,
) -> None:
    if "date" not in backtest_result.columns or "total_value" not in backtest_result.columns:
        st.warning("Equity comparison cannot be drawn because backtest data is missing.")
        return

    benchmark_curve = build_buy_and_hold_equity_curve(stock_data, initial_cash)
    if benchmark_curve.empty:
        st.warning("Buy-and-hold benchmark curve cannot be drawn.")
        return

    strategy_curve = backtest_result[["date", "total_value"]].copy()
    strategy_curve["date"] = pd.to_datetime(strategy_curve["date"])
    strategy_curve = strategy_curve.rename(
        columns={"total_value": "Strategy equity"}
    )
    benchmark_curve = benchmark_curve.rename(
        columns={"benchmark_total_value": "Buy-and-hold equity"}
    )
    comparison_df = pd.merge(
        strategy_curve,
        benchmark_curve,
        on="date",
        how="inner",
    ).set_index("date")
    st.line_chart(comparison_df, width="stretch")


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
    execution_mode: str,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_pct: float,
    min_commission: float,
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
        execution_mode=execution_mode,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_pct=slippage_pct,
        min_commission=min_commission,
    )
    performance_summary = summarize_performance(backtest_result)
    trade_metrics = summarize_trade_metrics(trades)
    benchmark = calculate_buy_and_hold_benchmark(stock_data, initial_cash)
    report = generate_rule_based_report(performance_summary)

    show_metric_cards(performance_summary, trade_metrics)
    show_benchmark_comparison_cards(performance_summary, benchmark)

    st.subheader("Price and Strategy Signals")
    show_price_signal_chart(stock_data)

    st.subheader("Portfolio Equity Curve")
    show_equity_curve(backtest_result)

    st.subheader("Strategy vs Buy-and-Hold Equity")
    show_equity_comparison_curve(backtest_result, stock_data, initial_cash)

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
    st.info(
        "Ranking score is unchanged for now. Also review "
        "avg_strategy_vs_benchmark_pct: positive values mean the strategy "
        "outperformed simple buy-and-hold on average."
    )
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
    st.info(
        "A positive difference vs benchmark means the strategy outperformed "
        "simple buy-and-hold for the same stock and period."
    )
    show_period_summary_cards(results_df, overall_summary_df, ranking_df)

    st.subheader("Period Experiment Charts")
    st.caption(
        "Charts use short scenario labels. Full scenario labels and raw codes "
        "are shown in the tables below."
    )
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

    fig_height = max(3.0, 0.45 * len(chart_df))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(chart_df["scenario_short_label"], chart_df[metric_column], color="tab:blue")
    ax.invert_yaxis()
    ax.set_xlabel("Value")
    ax.set_ylabel("Scenario")
    ax.grid(True, axis="x", alpha=0.25)

    min_value = float(chart_df[metric_column].min())
    max_value = float(chart_df[metric_column].max())
    value_range = max(max_value - min_value, 1.0)
    padding = value_range * 0.18
    ax.set_xlim(min_value - padding, max_value + padding)
    label_offset = value_range * 0.025

    for index, value in enumerate(chart_df[metric_column]):
        label = f"{value:.2f}{value_suffix}"
        x_position = value + label_offset if value >= 0 else value - label_offset
        ax.text(
            x_position,
            index,
            label,
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
        "Average total return (%)",
        ranking_df,
        "%",
    )
    show_period_bar_chart(
        overall_summary_df,
        "avg_max_drawdown_pct",
        "Average maximum drawdown (%)",
        ranking_df,
        "%",
    )
    show_period_bar_chart(
        overall_summary_df,
        "avg_profit_factor",
        "Average profit factor",
        ranking_df,
    )
    show_period_bar_chart(
        ranking_df,
        "score",
        "Scenario score",
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


def render_model_prediction_tab() -> None:
    st.write(
        "Load a trained baseline model and run one latest-row prediction from "
        "a factor CSV or an ML split CSV."
    )
    st.info(
        "The prediction panel uses saved feature_columns.txt to select model "
        "inputs. It does not change strategy rules or backtest behavior."
    )

    model_path = st.text_input(
        "Model path",
        value="models/demo_000001/random_forest.joblib",
        help="Path to a trained .joblib model.",
    )
    factor_csv_path = st.text_input(
        "Factor or ML CSV path",
        value="data/factors/factors_000001.csv",
        help="CSV containing the latest row to score.",
    )
    top_n = st.number_input(
        "Top factors",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
    )

    with st.expander("Optional artifact paths"):
        metrics_path = st.text_input(
            "Metrics path",
            value="",
            help="Blank uses metrics.json beside the model.",
        )
        feature_columns_path = st.text_input(
            "Feature columns path",
            value="",
            help="Blank uses feature_columns.txt beside the model.",
        )
        feature_importance_path = st.text_input(
            "Feature importance path",
            value="",
            help="Blank uses feature_importance.csv beside the model when available.",
        )

    run_clicked = st.button(
        "Run model prediction",
        key="run_model_prediction_button",
        type="primary",
    )
    if not run_clicked:
        st.caption(
            "Generated model and factor files are local ignored artifacts. "
            "Create them with the README commands before using this panel."
        )
        return

    try:
        result = run_model_prediction(
            model_path=model_path,
            input_path=factor_csv_path,
            metrics_path=metrics_path.strip() or None,
            feature_columns_path=feature_columns_path.strip() or None,
            feature_importance_path=feature_importance_path.strip() or None,
            top_n=int(top_n),
        )
    except Exception as exc:
        st.error(f"Model prediction failed: {exc}")
        return

    columns = st.columns(3)
    probability = result["predicted_probability"]
    columns[0].metric(
        "P(label_up_5d)",
        "N/A" if probability is None else f"{probability:.2%}",
    )
    columns[1].metric("Predicted class", str(result["predicted_class"]))
    columns[2].metric("Model signal", result["model_signal"].title())

    st.subheader("Scored Row")
    st.json(result["row_info"])

    st.subheader("Top Factors")
    importance_df = pd.DataFrame(result["top_feature_importance"])
    if importance_df.empty:
        st.info("Feature importance is unavailable for this model artifact.")
    else:
        st.dataframe(importance_df, width="stretch")

    st.subheader("Saved Model Metrics")
    metrics = result.get("metrics", {})
    validation_metrics = metrics.get("validation_metrics")
    test_metrics = metrics.get("test_metrics")
    if validation_metrics:
        st.write("Validation metrics")
        st.json(validation_metrics)
    if test_metrics:
        st.write("Test metrics")
        st.json(test_metrics)

    st.warning(
        "This ML output is an educational research signal, not a trading "
        "recommendation. Good prediction metrics do not guarantee profitable "
        "trading."
    )


def render_evaluation_split(name: str, report: dict) -> None:
    st.subheader(f"{name.title()} Evaluation")
    metrics = report["metrics"]
    columns = st.columns(4)
    columns[0].metric("Samples", metrics["sample_count"])
    columns[1].metric("Accuracy", format_metric_number(metrics["accuracy"]))
    columns[2].metric("F1", format_metric_number(metrics["f1"]))
    columns[3].metric("ROC AUC", format_metric_number(metrics["roc_auc"]))

    confusion_df = pd.DataFrame(
        [
            {"actual": "0", "predicted_0": metrics["tn"], "predicted_1": metrics["fp"]},
            {"actual": "1", "predicted_0": metrics["fn"], "predicted_1": metrics["tp"]},
        ]
    )
    st.write("Confusion matrix")
    st.dataframe(confusion_df, width="stretch")

    probability = report["probability_analysis"]
    if probability.get("available"):
        probability_summary = pd.DataFrame(
            [
                {
                    "metric": key,
                    "value": probability.get(key),
                }
                for key in [
                    "min_probability",
                    "max_probability",
                    "mean_probability",
                    "median_probability",
                    "avg_probability_actual_positive",
                    "avg_probability_actual_negative",
                ]
            ]
        )
        st.write("Probability summary")
        st.dataframe(probability_summary, width="stretch")
        st.write("Probability buckets")
        st.dataframe(
            pd.DataFrame(probability["bucket_distribution"]),
            width="stretch",
        )
    else:
        st.info("Probability analysis is unavailable because no probability column exists.")

    threshold_df = pd.DataFrame(report["threshold_analysis"])
    st.write("Threshold analysis")
    if threshold_df.empty:
        st.info("Threshold analysis is unavailable without probabilities.")
    else:
        st.dataframe(threshold_df, width="stretch")

    signal_backtest = report["signal_backtest"]
    st.write("Simple ML signal return check")
    if signal_backtest.get("available"):
        st.json(signal_backtest)
    else:
        st.info(signal_backtest.get("reason", "Signal check unavailable."))

    st.write("Warnings")
    for warning in report["warnings"]:
        st.warning(warning)


def render_model_evaluation_tab() -> None:
    st.write(
        "Review prediction quality for saved validation/test prediction files. "
        "This is a diagnostic view for spotting suspicious metrics and leakage risk."
    )
    model_dir = st.text_input(
        "Model directory",
        value="models/demo_000001",
        help="Directory containing metrics.json and prediction CSVs.",
    )
    target_col = st.text_input("Target column", value="label_up_5d")
    signal_threshold = st.slider(
        "Signal threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.60,
        step=0.05,
    )

    run_clicked = st.button(
        "Evaluate model outputs",
        key="run_model_evaluation_button",
        type="primary",
    )
    if not run_clicked:
        st.caption(
            "The tab automatically loads metrics.json, validation_predictions.csv, "
            "test_predictions.csv, feature_columns.txt, and feature_importance.csv "
            "from the selected model directory when available."
        )
        return

    try:
        result = evaluate_model_directory(
            model_dir=model_dir,
            target_col=target_col,
            signal_threshold=signal_threshold,
        )
    except Exception as exc:
        st.error(f"Model evaluation failed: {exc}")
        return

    st.subheader("Model Artifact Summary")
    st.write(f"Model directory: `{result['model_dir']}`")
    st.write(f"Feature count: {len(result['feature_columns'])}")
    leakage_columns = result["feature_leakage_columns"]
    if leakage_columns:
        st.error("Feature leakage columns detected: " + ", ".join(leakage_columns))
    else:
        st.success("No target/label/future columns detected in feature_columns.txt.")

    if result["feature_importance"]:
        st.subheader("Feature Importance")
        st.dataframe(pd.DataFrame(result["feature_importance"]).head(20), width="stretch")

    render_evaluation_split("validation", result["validation"])
    render_evaluation_split("test", result["test"])

    st.subheader("Interpretation")
    st.write(
        "Perfect or near-perfect validation/test metrics should be treated as a "
        "diagnostic warning, especially on small or synthetic datasets. They may "
        "indicate leakage, an overly easy label, duplicated information, or a "
        "demo dataset that is too regular. Classification quality is not the "
        "same thing as profitable trading after costs and execution assumptions."
    )


def render_ml_signal_backtest_tab() -> None:
    st.write(
        "Run a long/flat backtest by converting saved model probabilities into "
        "buy and sell signals. This uses the existing backtester execution and "
        "cost assumptions without changing rule-based strategy behavior."
    )

    model_dir = st.text_input(
        "ML model directory",
        value="models/demo_000001",
        help="Directory containing a trained .joblib model and feature_columns.txt.",
    )
    factor_csv = st.text_input(
        "Factor CSV path",
        value="data/factors/factors_000001.csv",
        help="Factor CSV or ML split CSV containing OHLCV and model feature columns.",
    )
    columns = st.columns(3)
    initial_cash = columns[0].number_input(
        "ML initial cash",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
    )
    buy_threshold = columns[1].number_input(
        "Buy threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        format="%.2f",
    )
    sell_threshold = columns[2].number_input(
        "Sell threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        format="%.2f",
    )

    execution_mode = st.selectbox(
        "ML execution mode",
        ["same_close", "next_open", "next_close"],
        key="ml_signal_execution_mode",
    )
    cost_columns = st.columns(4)
    commission_rate = cost_columns[0].number_input(
        "ML commission rate",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    stamp_tax_rate = cost_columns[1].number_input(
        "ML stamp tax rate",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    slippage_pct = cost_columns[2].number_input(
        "ML slippage %",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f",
    )
    min_commission = cost_columns[3].number_input(
        "ML min commission",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )
    compare_rule_based = st.checkbox(
        "Compare with existing MA crossover strategy",
        value=True,
    )

    run_clicked = st.button(
        "Run ML signal backtest",
        key="run_ml_signal_backtest_button",
        type="primary",
    )
    if not run_clicked:
        st.caption(
            "Create ignored local model and factor artifacts first using the "
            "factor, split, and training commands in the README."
        )
        return

    try:
        result = run_ml_signal_backtest(
            model_dir=model_dir,
            factor_csv=factor_csv,
            initial_cash=initial_cash,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            execution_mode=execution_mode,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
            compare_rule_based=compare_rule_based,
        )
    except Exception as exc:
        st.error(f"ML signal backtest failed: {exc}")
        return

    st.subheader("ML Strategy Metrics")
    st.dataframe(format_performance_summary(result["performance"]), width="stretch")

    st.subheader("Buy-and-Hold Benchmark")
    benchmark_rows = [
        {"metric": key, "value": value}
        for key, value in result["benchmark"].items()
    ]
    st.dataframe(pd.DataFrame(benchmark_rows), width="stretch")

    rule_based = result.get("rule_based_comparison")
    if rule_based and rule_based.get("available"):
        st.subheader("Existing Rule-Based Strategy Comparison")
        st.dataframe(
            format_performance_summary(rule_based["performance"]),
            width="stretch",
        )

    st.subheader("Trade Log")
    if result["trades"].empty:
        st.info("No trades were executed.")
    else:
        st.dataframe(format_trade_log(result["trades"]), width="stretch")

    st.subheader("Equity Curve")
    equity_df = result["backtest"][["date", "total_value"]].copy()
    equity_df["date"] = pd.to_datetime(equity_df["date"])
    st.line_chart(equity_df.set_index("date"))

    st.subheader("Probability / Signal Preview")
    preview_columns = ["date", "close", "prediction_probability", "signal"]
    st.dataframe(
        result["signal_data"][preview_columns].tail(30),
        width="stretch",
    )

    for warning in result["warnings"]:
        st.warning(warning)


def render_threshold_summary_card(column, title: str, row: pd.Series | None) -> None:
    if row is None:
        column.markdown(f"**{title}**\n\nN/A")
        return

    column.markdown(
        f"""
        **{title}**

        Buy threshold: `{row['buy_threshold']:.2f}`  
        Sell threshold: `{row['sell_threshold']:.2f}`  
        Total return: {row['total_return_pct']:.2f}%  
        Max drawdown: {row['max_drawdown_pct']:.2f}%  
        Score: {row['score']:.2f}
        """
    )


def render_ml_threshold_experiment_tab() -> None:
    st.write(
        "Test multiple ML probability thresholds with the existing long-only "
        "backtester. This is research-only threshold analysis."
    )
    st.warning(
        "This is threshold research only. Optimizing thresholds on past data can overfit."
    )

    model_dir = st.text_input(
        "Threshold experiment model directory",
        value="models/demo_000001",
    )
    input_path = st.text_input(
        "Threshold experiment factor CSV",
        value="data/factors/factors_000001.csv",
    )
    buy_thresholds_text = st.text_input(
        "Buy thresholds",
        value="0.50,0.55,0.60,0.65,0.70,0.75",
    )
    sell_thresholds_text = st.text_input(
        "Sell thresholds",
        value="0.40,0.45,0.50,0.55",
    )

    columns = st.columns(3)
    initial_cash = columns[0].number_input(
        "Threshold initial cash",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
    )
    execution_mode = columns[1].selectbox(
        "Threshold execution mode",
        ["same_close", "next_open", "next_close"],
        key="threshold_execution_mode",
    )
    walk_forward = columns[2].checkbox("Walk-forward mode", value=False)

    cost_columns = st.columns(4)
    commission_rate = cost_columns[0].number_input(
        "Threshold commission rate",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    stamp_tax_rate = cost_columns[1].number_input(
        "Threshold stamp tax rate",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    slippage_pct = cost_columns[2].number_input(
        "Threshold slippage %",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f",
    )
    min_commission = cost_columns[3].number_input(
        "Threshold min commission",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    with st.expander("Walk-forward settings"):
        target_col = st.text_input("Walk-forward target", value="label_up_5d")
        model_name = st.selectbox(
            "Walk-forward model",
            ["random_forest", "logistic_regression"],
        )
        wf_columns = st.columns(3)
        train_window = wf_columns[0].number_input(
            "Train window rows",
            min_value=20,
            value=120,
            step=10,
        )
        test_window = wf_columns[1].number_input(
            "Test window rows",
            min_value=10,
            value=40,
            step=10,
        )
        step_size = wf_columns[2].number_input(
            "Step size rows",
            min_value=10,
            value=40,
            step=10,
        )

    run_clicked = st.button(
        "Run ML threshold experiment",
        key="run_ml_threshold_experiment_button",
        type="primary",
    )
    if not run_clicked:
        return

    try:
        buy_thresholds = parse_ml_thresholds(buy_thresholds_text, [])
        sell_thresholds = parse_ml_thresholds(sell_thresholds_text, [])
        if walk_forward:
            results_df = run_walk_forward_threshold_experiment(
                model_dir=model_dir,
                input_path=input_path,
                target_col=target_col,
                model_name=model_name,
                buy_thresholds=buy_thresholds,
                sell_thresholds=sell_thresholds,
                train_window=int(train_window),
                test_window=int(test_window),
                step_size=int(step_size),
                initial_cash=initial_cash,
                execution_mode=execution_mode,
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax_rate,
                slippage_pct=slippage_pct,
                min_commission=min_commission,
            )
        else:
            results_df = run_threshold_experiment(
                model_dir=model_dir,
                input_path=input_path,
                buy_thresholds=buy_thresholds,
                sell_thresholds=sell_thresholds,
                initial_cash=initial_cash,
                execution_mode=execution_mode,
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax_rate,
                slippage_pct=slippage_pct,
                min_commission=min_commission,
            )
    except Exception as exc:
        st.error(f"ML threshold experiment failed: {exc}")
        return

    if results_df.empty:
        st.warning("No threshold experiment rows were produced.")
        return

    ranking_df = rank_threshold_results(results_df)
    best_score = ranking_df.iloc[0]
    best_return = results_df.sort_values(
        "total_return_pct",
        ascending=False,
    ).iloc[0]
    best_drawdown = results_df.sort_values(
        "max_drawdown_pct",
        ascending=False,
    ).iloc[0]

    summary_columns = st.columns(3)
    render_threshold_summary_card(summary_columns[0], "Best by score", best_score)
    render_threshold_summary_card(summary_columns[1], "Best total return", best_return)
    render_threshold_summary_card(
        summary_columns[2],
        "Best drawdown control",
        best_drawdown,
    )

    st.subheader("Ranking Table")
    st.dataframe(ranking_df, width="stretch")

    st.subheader("Results Table")
    st.dataframe(results_df, width="stretch")

    st.subheader("Threshold Result Chart")
    chart_df = ranking_df.head(20).copy()
    chart_df["threshold_pair"] = chart_df.apply(
        lambda row: f"B{row['buy_threshold']:.2f}/S{row['sell_threshold']:.2f}",
        axis=1,
    )
    if "window_id" in chart_df.columns:
        chart_df["threshold_pair"] = (
            "W"
            + chart_df["window_id"].astype(str)
            + " "
            + chart_df["threshold_pair"]
        )
    st.bar_chart(chart_df.set_index("threshold_pair")["score"])

    st.warning(
        "Good threshold results on past data do not imply future profitability. "
        "Review stability across windows, costs, and drawdowns."
    )


def describe_roc_auc(value) -> str:
    if is_missing(value):
        return "ROC AUC is unavailable."
    if value < 0.45:
        return f"ROC AUC {value:.2f} is below random and is a serious warning."
    if value < 0.55:
        return f"ROC AUC {value:.2f} is close to random."
    if value < 0.65:
        return f"ROC AUC {value:.2f} is modestly above random."
    if value < 0.80:
        return f"ROC AUC {value:.2f} is meaningfully above random, but still needs robustness checks."
    return f"ROC AUC {value:.2f} is high; verify leakage, sample size, and symbol stability."


def build_robustness_diagnostics(
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    results_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> dict:
    diagnostics = {
        "quick_lines": [],
        "best_model": None,
        "weakest_pair": None,
        "validation_test_gap": pd.DataFrame(),
        "warnings_summary": pd.DataFrame(),
        "consistent_model_note": "Not enough model types to compare consistency.",
    }

    if summary_df.empty or results_df.empty:
        diagnostics["quick_lines"].append("No robustness results are available yet.")
        return diagnostics

    valid_summary = summary_df.dropna(subset=["avg_test_roc_auc"]).copy()
    if not valid_summary.empty:
        best_model = valid_summary.sort_values(
            "avg_test_roc_auc",
            ascending=False,
        ).iloc[0]
        diagnostics["best_model"] = best_model
        diagnostics["quick_lines"].append(
            f"Best average test ROC AUC model: {best_model['model_type']} "
            f"({best_model['avg_test_roc_auc']:.2f}). "
            + describe_roc_auc(best_model["avg_test_roc_auc"])
        )

    successful_results = results_df[results_df["error"].isna()].copy()
    if not successful_results.empty:
        successful_results["validation_test_roc_auc_gap"] = (
            successful_results["validation_roc_auc"]
            - successful_results["test_roc_auc"]
        )
        gap_df = successful_results[
            [
                "symbol",
                "model_type",
                "validation_roc_auc",
                "test_roc_auc",
                "validation_test_roc_auc_gap",
            ]
        ].copy()
        diagnostics["validation_test_gap"] = gap_df.sort_values(
            "validation_test_roc_auc_gap",
            key=lambda values: values.abs(),
            ascending=False,
        ).reset_index(drop=True)

        weakest_pair = successful_results.sort_values(
            ["test_roc_auc", "test_f1"],
            ascending=[True, True],
        ).iloc[0]
        diagnostics["weakest_pair"] = weakest_pair
        diagnostics["quick_lines"].append(
            f"Weakest symbol/model pair: {weakest_pair['symbol']} / "
            f"{weakest_pair['model_type']} with test ROC AUC "
            f"{weakest_pair['test_roc_auc']:.2f}."
        )

        max_gap = diagnostics["validation_test_gap"][
            "validation_test_roc_auc_gap"
        ].abs().max()
        if pd.notna(max_gap) and max_gap > 0.20:
            diagnostics["quick_lines"].append(
                f"Validation/test ROC AUC diverges by up to {max_gap:.2f}; "
                "this suggests unstable generalization."
            )
        else:
            diagnostics["quick_lines"].append(
                "Validation/test ROC AUC gaps are not large in this run."
            )

        small_samples = successful_results[
            pd.to_numeric(successful_results["test_rows"], errors="coerce") < 50
        ]
        if not small_samples.empty:
            diagnostics["quick_lines"].append(
                f"{len(small_samples)} symbol/model rows have fewer than 50 test samples."
            )

        model_means = successful_results.groupby("model_type")["test_roc_auc"].mean()
        if len(model_means) >= 2:
            ordered_models = model_means.sort_values(ascending=False)
            spread = ordered_models.iloc[0] - ordered_models.iloc[-1]
            if spread >= 0.05:
                diagnostics["consistent_model_note"] = (
                    f"{ordered_models.index[0]} is ahead by {spread:.2f} average "
                    "test ROC AUC versus the weakest model."
                )
            else:
                diagnostics["consistent_model_note"] = (
                    "No model is clearly better by average test ROC AUC; "
                    "differences are small."
                )

    if not warnings_df.empty:
        diagnostics["warnings_summary"] = (
            warnings_df.groupby("warning_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        if (warnings_df["warning_type"] == "suspicious_perfect_metrics").any():
            diagnostics["quick_lines"].append(
                "Suspiciously perfect metrics are present. Treat high scores as a warning, not validation."
            )

    diagnostics["quick_lines"].append(
        "ROC AUC around 0.5 means close to random; higher test ROC AUC does not guarantee trading profit."
    )
    diagnostics["quick_lines"].append(
        "Robustness across symbols is more important than one lucky symbol or split."
    )
    return diagnostics


def render_robustness_interpretation(
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    results_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
) -> None:
    diagnostics = build_robustness_diagnostics(
        summary_df,
        ranking_df,
        results_df,
        warnings_df,
    )

    st.subheader("Quick Interpretation")
    for line in diagnostics["quick_lines"]:
        st.write(f"- {line}")
    st.info(diagnostics["consistent_model_note"])

    st.subheader("Best Model by Test ROC AUC")
    best_model = diagnostics["best_model"]
    if best_model is None:
        st.info("No model has an available average test ROC AUC.")
    else:
        st.dataframe(pd.DataFrame([best_model]), width="stretch")

    st.subheader("Weakest Symbol / Model Pair")
    weakest_pair = diagnostics["weakest_pair"]
    if weakest_pair is None:
        st.info("No successful symbol/model rows are available.")
    else:
        st.dataframe(pd.DataFrame([weakest_pair]), width="stretch")

    st.subheader("Validation-Test ROC AUC Gap")
    gap_df = diagnostics["validation_test_gap"]
    if gap_df.empty:
        st.info("Validation-test gap table is unavailable.")
    else:
        st.dataframe(gap_df, width="stretch")

    st.subheader("Warnings Summary")
    warnings_summary = diagnostics["warnings_summary"]
    if warnings_summary.empty:
        st.success("No warning rows were produced.")
    else:
        st.dataframe(warnings_summary, width="stretch")

    st.write(
        "Educational notes: ROC AUC around 0.5 is close to random. A higher "
        "test ROC AUC can still fail as a trading system after costs, slippage, "
        "execution delay, and position sizing. Prefer stable behavior across "
        "symbols and periods over one standout result."
    )


def render_model_robustness_report_export(default_input_dir: str) -> None:
    st.subheader("Markdown Research Report")
    st.write(
        "Generate a readable Markdown report from an existing robustness output "
        "directory. This uses saved CSV and JSON outputs; it does not retrain models."
    )
    report_input_dir = st.text_input(
        "Report input directory",
        value=default_input_dir,
        key="robustness_report_input_dir",
        help="Use a directory containing model_summary.csv, model_ranking.csv, training_results.csv, warnings.csv, and run_config.json.",
    )
    report_output_path = st.text_input(
        "Report output path",
        value="reports/model_robustness_report.md",
        key="robustness_report_output_path",
    )
    if st.button("Generate robustness report", key="generate_robustness_report_button"):
        try:
            saved_path, report_text = write_model_robustness_report(
                input_dir=report_input_dir,
                output_path=report_output_path,
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")
        else:
            st.success(f"Report generated: {saved_path}")
            st.markdown(report_text)
            st.download_button(
                "Download Markdown report",
                data=report_text,
                file_name=Path(saved_path).name,
                mime="text/markdown",
                key="download_robustness_report_button",
            )


def render_model_robustness_tab() -> None:
    st.write(
        "Compare baseline model quality across multiple symbols and model types. "
        "This checks whether a model looks stable beyond one symbol or one split."
    )
    st.info(
        "High ML metrics do not guarantee profitable trading. This panel checks "
        "model robustness across symbols and periods for educational research only."
    )

    symbols_text = st.text_input(
        "Robustness symbols",
        value="000001,600519",
        help="Comma-separated A-share symbols.",
    )
    source = st.selectbox(
        "Robustness source mode",
        ["demo", "baostock"],
        key="robustness_source_mode",
    )
    columns = st.columns(2)
    start = columns[0].text_input("Robustness start date", value="20240101")
    end = columns[1].text_input("Robustness end date", value="20241231")
    models_text = st.text_input(
        "Model types",
        value="logistic_regression,random_forest",
    )
    target_col = st.text_input("Robustness target column", value="label_up_5d")
    output_dir = st.text_input(
        "Robustness output directory",
        value="outputs/model_robustness_demo",
    )

    settings_columns = st.columns(4)
    purge_rows = settings_columns[0].number_input(
        "Robustness purge rows",
        min_value=0,
        value=5,
        step=1,
    )
    train_ratio = settings_columns[1].number_input(
        "Train ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
        format="%.2f",
    )
    val_ratio = settings_columns[2].number_input(
        "Validation ratio",
        min_value=0.05,
        max_value=0.8,
        value=0.2,
        step=0.05,
        format="%.2f",
    )
    test_ratio = settings_columns[3].number_input(
        "Test ratio",
        min_value=0.05,
        max_value=0.8,
        value=0.2,
        step=0.05,
        format="%.2f",
    )
    split_mode = st.selectbox(
        "Robustness split mode",
        ["global_date", "per_symbol"],
        key="robustness_split_mode",
    )

    render_model_robustness_report_export(output_dir)

    run_clicked = st.button(
        "Run robustness training",
        key="run_model_robustness_button",
        type="primary",
    )
    if not run_clicked:
        return

    try:
        symbols = parse_batch_symbols(symbols_text)
        model_types = parse_batch_model_types(models_text)
        result = run_batch_model_training(
            symbols=symbols,
            model_types=model_types,
            source=source,
            start=start,
            end=end,
            output_dir=output_dir,
            target_col=target_col,
            purge_rows=int(purge_rows),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_mode=split_mode,
        )
    except Exception as exc:
        st.error(f"Model robustness training failed: {exc}")
        return

    summary_df = result["model_summary"]
    ranking_df = result["model_ranking"]
    results_df = result["training_results"]
    warnings_df = result["warnings"]

    if not warnings_df.empty and (
        warnings_df["warning_type"] == "suspicious_perfect_metrics"
    ).any():
        st.warning(
            "Some validation or test metrics are suspiciously close to perfect. "
            "Check for leakage, tiny samples, or overly regular demo data."
        )

    render_robustness_interpretation(
        summary_df=summary_df,
        ranking_df=ranking_df,
        results_df=results_df,
        warnings_df=warnings_df,
    )

    st.subheader("Model Summary")
    st.dataframe(summary_df, width="stretch")

    st.subheader("Model Ranking")
    st.dataframe(ranking_df, width="stretch")

    st.subheader("Full Training Results")
    st.dataframe(results_df, width="stretch")

    st.subheader("Warnings")
    if warnings_df.empty:
        st.success("No warnings were produced.")
    else:
        st.dataframe(warnings_df, width="stretch")

    if not summary_df.empty:
        chart_df = summary_df.set_index("model_type")
        st.subheader("Average Test ROC AUC")
        st.bar_chart(chart_df["avg_test_roc_auc"])
        st.subheader("Average Test F1")
        st.bar_chart(chart_df["avg_test_f1"])

    if not ranking_df.empty:
        st.subheader("Robustness Score")
        st.bar_chart(ranking_df.set_index("model_type")["score"])

    st.subheader("Output Files")
    st.json(result["output_files"])


def render_feature_sources_tab() -> None:
    st.write(
        "Review the planned multi-factor feature roadmap for future model "
        "training. This registry is metadata only; it does not fetch external data."
    )
    st.info(
        "Every future feature source needs point-in-time lag control. More data "
        "does not automatically improve prediction, and no factor guarantees profit."
    )

    registry_df = registry_to_dataframe()
    family_options = ["All", *sorted(registry_df["factor_family"].unique())]
    priority_options = ["All", "P0", "P1", "P2", "P3"]

    filters = st.columns(2)
    selected_family = filters[0].selectbox(
        "Factor family filter",
        family_options,
        key="feature_source_family_filter",
    )
    selected_priority = filters[1].selectbox(
        "Priority filter",
        priority_options,
        key="feature_source_priority_filter",
    )

    filtered_df = registry_df.copy()
    if selected_family != "All":
        filtered_df = filtered_df[filtered_df["factor_family"] == selected_family]
    if selected_priority != "All":
        filtered_df = filtered_df[
            filtered_df["implementation_priority"] == selected_priority
        ]

    st.subheader("Feature Registry")
    st.dataframe(filtered_df, width="stretch")

    csv_bytes = registry_df.to_csv(index=False, encoding="utf-8-sig").encode(
        "utf-8-sig"
    )
    st.download_button(
        "Download full registry CSV",
        data=csv_bytes,
        file_name="feature_source_registry.csv",
        mime="text/csv",
        key="download_feature_source_registry_button",
    )

    st.subheader("Factor Family Summary")
    st.dataframe(summarize_factor_families(), width="stretch")

    st.subheader("Training-Ready Features")
    st.dataframe(
        registry_to_dataframe(get_training_ready_features()),
        width="stretch",
    )

    st.subheader("High Leakage Risk Features")
    st.dataframe(
        registry_to_dataframe(get_high_leakage_risk_features()),
        width="stretch",
    )

    st.subheader("Token-Required Features")
    token_df = registry_to_dataframe(get_token_required_features())
    if token_df.empty:
        st.success("No registry rows currently require tokens.")
    else:
        st.dataframe(token_df, width="stretch")


def render_feature_queue_tab() -> None:
    st.write(
        "Prioritize future factor engineering work by implementation score, "
        "leakage risk, cost, token requirements, and expected training value."
    )
    st.warning(
        "Feature expansion may improve research coverage, but more features can "
        "also increase overfitting and leakage risk. This is not financial advice."
    )

    full_queue_df = queue_to_dataframe()
    full_queue = full_queue_df.to_dict(orient="records")
    summary = summarize_feature_queue()

    metric_columns = st.columns(5)
    metric_columns[0].metric("Total items", summary["total_items"])
    metric_columns[1].metric("P0 items", summary["p0_item_count"])
    metric_columns[2].metric("Low leakage", summary["low_leakage_item_count"])
    metric_columns[3].metric("Token-free", summary["token_free_item_count"])
    metric_columns[4].metric("High training value", summary["high_training_value_item_count"])

    filter_columns = st.columns(5)
    priority_filter = filter_columns[0].selectbox(
        "Queue priority",
        ["All", "P0_now", "P1_next", "P2_later", "P3_research_only"],
        key="feature_queue_priority_filter",
    )
    category_filter = filter_columns[1].selectbox(
        "Queue category",
        ["All", *sorted(full_queue_df["category"].unique())],
        key="feature_queue_category_filter",
    )
    leakage_filter = filter_columns[2].selectbox(
        "Leakage risk",
        ["All", "low", "medium", "high"],
        key="feature_queue_leakage_filter",
    )
    token_filter = filter_columns[3].selectbox(
        "Token required",
        ["All", "false", "true"],
        key="feature_queue_token_filter",
    )
    difficulty_filter = filter_columns[4].selectbox(
        "Difficulty",
        ["All", "low", "medium", "high"],
        key="feature_queue_difficulty_filter",
    )

    filtered_queue = filter_feature_queue(
        queue=full_queue,
        priority=None if priority_filter == "All" else priority_filter,
        category=None if category_filter == "All" else category_filter,
        leakage_risk=None if leakage_filter == "All" else leakage_filter,
        token_required=None if token_filter == "All" else token_filter == "true",
        implementation_difficulty=None
        if difficulty_filter == "All"
        else difficulty_filter,
    )
    filtered_df = queue_to_dataframe(filtered_queue)

    st.subheader("Feature Implementation Queue")
    st.dataframe(filtered_df, width="stretch")

    st.subheader("Top P0 Recommendations")
    p0_df = queue_to_dataframe(
        filter_feature_queue(queue=full_queue, priority="P0_now")
    ).head(10)
    if p0_df.empty:
        st.info("No P0 queue items are available.")
    else:
        st.dataframe(
            p0_df[
                [
                    "queue_id",
                    "feature_group",
                    "feature_name",
                    "implementation_score",
                    "recommended_action",
                    "validation_checks",
                ]
            ],
            width="stretch",
        )

    csv_bytes = full_queue_df.to_csv(index=False, encoding="utf-8-sig").encode(
        "utf-8-sig"
    )
    st.download_button(
        "Download feature queue CSV",
        data=csv_bytes,
        file_name="feature_implementation_queue.csv",
        mime="text/csv",
        key="download_feature_queue_button",
    )


def load_factor_ablation_outputs(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(output_dir)
    return (
        pd.read_csv(base / "ablation_results.csv") if (base / "ablation_results.csv").exists() else pd.DataFrame(),
        pd.read_csv(base / "group_summary.csv") if (base / "group_summary.csv").exists() else pd.DataFrame(),
        pd.read_csv(base / "feature_impact_ranking.csv") if (base / "feature_impact_ranking.csv").exists() else pd.DataFrame(),
        pd.read_csv(base / "warnings.csv") if (base / "warnings.csv").exists() else pd.DataFrame(),
    )


def run_dashboard_factor_ablation(
    symbols: list[str],
    source: str,
    start: str,
    end: str,
    output_dir: str,
    model_types: list[str],
    ablation_modes: list[str],
    target_col: str,
    purge_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(output_dir)
    factor_dir = base / "factors"
    symbol_dir = base / "symbols"
    factor_dir.mkdir(parents=True, exist_ok=True)
    symbol_dir.mkdir(parents=True, exist_ok=True)

    result_frames = []
    warning_frames = []
    for symbol in symbols:
        try:
            raw_df = fetch_symbol_ohlcv(symbol, source, start, end)
            factor_df = build_factor_dataset(raw_df, symbol=symbol)
            factor_path = factor_dir / f"factors_{symbol}.csv"
            save_factor_dataset(factor_df, factor_path)
            result = run_and_save_factor_ablation(
                input_path=factor_path,
                output_dir=symbol_dir / symbol,
                target_col=target_col,
                model_types=model_types,
                ablation_modes=ablation_modes,
                purge_rows=purge_rows,
                symbol=symbol,
            )
            result_frames.append(result["ablation_results"])
            warning_frames.append(result["warnings"])
        except Exception as exc:
            warning_frames.append(
                pd.DataFrame(
                    [
                        {
                            "symbol": symbol,
                            "model_type": None,
                            "experiment_name": None,
                            "warning": str(exc),
                        }
                    ]
                )
            )

    ablation_results = (
        pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    )
    group_summary = build_group_summary(ablation_results)
    feature_ranking = build_feature_impact_ranking(ablation_results)
    warnings_df = (
        pd.concat(warning_frames, ignore_index=True) if warning_frames else pd.DataFrame()
    )

    ablation_results.to_csv(base / "ablation_results.csv", index=False)
    group_summary.to_csv(base / "group_summary.csv", index=False)
    feature_ranking.to_csv(base / "feature_impact_ranking.csv", index=False)
    warnings_df.to_csv(base / "warnings.csv", index=False)
    (base / "run_config.json").write_text(
        json.dumps(
            {
                "symbols": symbols,
                "source": source,
                "start": start,
                "end": end,
                "output_dir": output_dir,
                "models": model_types,
                "ablation_modes": ablation_modes,
                "target_col": target_col,
                "purge_rows": purge_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ablation_results, group_summary, feature_ranking, warnings_df


def render_factor_ablation_tab() -> None:
    st.write(
        "Diagnose which factor groups or individual P0 features help or hurt "
        "baseline model metrics. This is research diagnostics only."
    )
    st.warning(
        "Positive delta means the ablation experiment beat the full feature set. "
        "If dropping a group improves test ROC AUC, that group may be noisy. "
        "Good ML metrics do not guarantee profitable trading."
    )

    symbols_text = st.text_input(
        "Ablation symbols",
        value="000001,600519",
        key="factor_ablation_symbols",
    )
    source = st.selectbox(
        "Ablation source",
        ["demo", "baostock"],
        key="factor_ablation_source",
    )
    date_columns = st.columns(2)
    start = date_columns[0].text_input(
        "Ablation start date",
        value="20240101",
        key="factor_ablation_start",
    )
    end = date_columns[1].text_input(
        "Ablation end date",
        value="20241231",
        key="factor_ablation_end",
    )
    models_text = st.text_input(
        "Ablation model types",
        value="logistic_regression,random_forest",
        key="factor_ablation_models",
    )
    modes_text = st.text_input(
        "Ablation modes",
        value="drop_group,only_group",
        key="factor_ablation_modes",
    )
    output_dir = st.text_input(
        "Ablation output directory",
        value="outputs/factor_ablation_demo",
        key="factor_ablation_output_dir",
    )
    target_col = st.text_input(
        "Ablation target column",
        value="label_up_5d",
        key="factor_ablation_target",
    )
    purge_rows = st.number_input(
        "Ablation purge rows",
        min_value=0,
        value=5,
        step=1,
        key="factor_ablation_purge_rows",
    )

    button_columns = st.columns(2)
    run_clicked = button_columns[0].button(
        "Run factor ablation",
        key="run_factor_ablation_button",
        type="primary",
    )
    load_clicked = button_columns[1].button(
        "Load existing ablation outputs",
        key="load_factor_ablation_button",
    )

    if run_clicked:
        try:
            ablation_results, group_summary, feature_ranking, warnings_df = (
                run_dashboard_factor_ablation(
                    symbols=parse_batch_symbols(symbols_text),
                    source=source,
                    start=start,
                    end=end,
                    output_dir=output_dir,
                    model_types=parse_ablation_model_types(models_text),
                    ablation_modes=parse_factor_ablation_modes(modes_text),
                    target_col=target_col,
                    purge_rows=int(purge_rows),
                )
            )
        except Exception as exc:
            st.error(f"Factor ablation failed: {exc}")
            return
    elif load_clicked:
        ablation_results, group_summary, feature_ranking, warnings_df = (
            load_factor_ablation_outputs(output_dir)
        )
    else:
        return

    if ablation_results.empty:
        st.info("No ablation result rows are available.")
        return

    st.subheader("Helpful Groups")
    if group_summary.empty:
        st.info("No group summary rows are available.")
    else:
        helpful = group_summary.sort_values(
            "avg_test_roc_auc_delta_vs_full",
            ascending=False,
        ).head(10)
        st.dataframe(helpful, width="stretch")

    st.subheader("Potentially Harmful Groups")
    harmful = group_summary[
        (group_summary["ablation_type"] == "drop_group")
        & (group_summary["avg_test_roc_auc_delta_vs_full"] > 0)
    ] if not group_summary.empty else pd.DataFrame()
    if harmful.empty:
        st.info("No drop-group rows improved average test ROC AUC.")
    else:
        st.dataframe(harmful, width="stretch")

    st.subheader("Group Summary")
    st.dataframe(group_summary, width="stretch")

    st.subheader("Feature Impact Ranking")
    if feature_ranking.empty:
        st.info("Run with ablation mode drop_feature to populate this table.")
    else:
        st.dataframe(feature_ranking, width="stretch")

    st.subheader("Warnings")
    if warnings_df.empty:
        st.success("No warnings were recorded.")
    else:
        st.dataframe(warnings_df, width="stretch")

    if not group_summary.empty:
        chart_df = group_summary.copy()
        chart_df["group_model"] = (
            chart_df["factor_group"].astype(str)
            + " / "
            + chart_df["ablation_type"].astype(str)
            + " / "
            + chart_df["model_type"].astype(str)
        )
        st.subheader("Group Impact")
        st.bar_chart(
            chart_df.set_index("group_model")["avg_test_roc_auc_delta_vs_full"]
        )


def render_factor_decisions_tab() -> None:
    st.write(
        "Convert factor ablation diagnostics into transparent research decisions "
        "about which factor groups to keep, observe, reduce, or retest."
    )
    st.info(
        "This report is a feature-selection research aid. It does not change "
        "model training, strategy rules, or backtester behavior."
    )

    input_dir = st.text_input(
        "Factor ablation output directory",
        value="outputs/factor_ablation_demo",
        key="factor_decision_input_dir",
    )
    output_path = st.text_input(
        "Factor decision report path",
        value="outputs/factor_ablation_demo/factor_decision_report.md",
        key="factor_decision_output_path",
    )
    button_columns = st.columns(2)
    generate_clicked = button_columns[0].button(
        "Generate factor decision report",
        key="generate_factor_decision_report_button",
        type="primary",
    )
    load_clicked = button_columns[1].button(
        "Load factor decision report",
        key="load_factor_decision_report_button",
    )

    if generate_clicked:
        try:
            result = write_factor_decision_report(input_dir, output_path)
        except Exception as exc:
            st.error(f"Factor decision report generation failed: {exc}")
            return
    elif load_clicked:
        try:
            result = generate_factor_decision_report(input_dir)
            report_file = Path(output_path)
            if report_file.exists():
                result["markdown_report"] = report_file.read_text(encoding="utf-8")
                result["report_path"] = str(report_file)
        except Exception as exc:
            st.error(f"Factor decision report loading failed: {exc}")
            return
    else:
        return

    decision_summary = result["decision_summary"]
    strongest = result["strongest_groups"]
    weakest = result["weakest_groups"]
    pruning_recommendations = result.get(
        "feature_pruning_recommendations",
        pd.DataFrame(),
    )
    report_text = result["markdown_report"]

    st.subheader("Decision Summary")
    if decision_summary.empty:
        st.info("No decision rows are available.")
    else:
        st.dataframe(decision_summary, width="stretch")

    st.subheader("Strongest Factor Groups")
    if strongest.empty:
        st.info("No strongest groups are available.")
    else:
        st.dataframe(strongest, width="stretch")

    st.subheader("Weak or Noisy Factor Groups")
    if weakest.empty:
        st.info("No weak or noisy groups are available.")
    else:
        st.dataframe(weakest, width="stretch")

    st.subheader("Individual Feature Pruning Recommendations")
    if pruning_recommendations.empty:
        st.info(
            "No individual feature recommendations are available. Run factor "
            "ablation with drop_feature mode first."
        )
    else:
        st.dataframe(pruning_recommendations, width="stretch")

    st.subheader("Markdown Report")
    st.markdown(report_text)
    st.download_button(
        "Download factor decision report",
        data=report_text,
        file_name=Path(output_path).name,
        mime="text/markdown",
        key="download_factor_decision_report_button",
    )


def load_factor_pruning_outputs(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(output_dir)
    return (
        pd.read_csv(base / "pruning_summary.csv") if (base / "pruning_summary.csv").exists() else pd.DataFrame(),
        pd.read_csv(base / "pruning_results.csv") if (base / "pruning_results.csv").exists() else pd.DataFrame(),
        pd.read_csv(base / "feature_set_details.csv") if (base / "feature_set_details.csv").exists() else pd.DataFrame(),
        pd.read_csv(base / "warnings.csv") if (base / "warnings.csv").exists() else pd.DataFrame(),
    )


def render_factor_pruning_tab() -> None:
    st.write(
        "Use individual feature pruning recommendations to compare reduced "
        "feature sets against the full feature baseline."
    )
    st.warning(
        "Pruning can overfit one sample. Retest any reduced feature set with "
        "walk-forward validation and more symbols before trusting it."
    )

    factor_csv = st.text_input(
        "Pruning factor CSV path",
        value="data/factors/smoke_factors_000001.csv",
        key="factor_pruning_factor_csv",
    )
    recommendations_path = st.text_input(
        "Pruning recommendations CSV",
        value="outputs/factor_ablation_demo/feature_pruning_recommendations.csv",
        key="factor_pruning_recommendations",
    )
    output_dir = st.text_input(
        "Pruning output directory",
        value="outputs/factor_pruning_demo",
        key="factor_pruning_output_dir",
    )
    models_text = st.text_input(
        "Pruning model types",
        value="logistic_regression,random_forest",
        key="factor_pruning_models",
    )
    target_col = st.text_input(
        "Pruning target column",
        value="label_up_5d",
        key="factor_pruning_target",
    )
    pruning_modes_text = st.text_input(
        "Pruning modes",
        value="full,drop_reduce_weight,keep_core_only,keep_core_and_observe",
        key="factor_pruning_modes",
    )

    buttons = st.columns(2)
    run_clicked = buttons[0].button(
        "Run pruning experiment",
        key="run_factor_pruning_button",
        type="primary",
    )
    load_clicked = buttons[1].button(
        "Load pruning outputs",
        key="load_factor_pruning_button",
    )

    if run_clicked:
        try:
            result = run_and_save_factor_pruning_experiment(
                factor_csv=factor_csv,
                recommendations_path=recommendations_path,
                output_dir=output_dir,
                model_types=parse_ablation_model_types(models_text),
                pruning_modes=parse_pruning_modes(pruning_modes_text),
                target_col=target_col,
            )
            pruning_summary = result["pruning_summary"]
            pruning_results = result["pruning_results"]
            feature_set_details = result["feature_set_details"]
            warnings_df = result["warnings"]
        except Exception as exc:
            st.error(f"Factor pruning experiment failed: {exc}")
            return
    elif load_clicked:
        pruning_summary, pruning_results, feature_set_details, warnings_df = (
            load_factor_pruning_outputs(output_dir)
        )
    else:
        return

    st.subheader("Pruning Summary")
    if pruning_summary.empty:
        st.info("No pruning summary rows are available.")
    else:
        st.dataframe(pruning_summary, width="stretch")

    st.subheader("Pruning Results")
    st.dataframe(pruning_results, width="stretch")

    st.subheader("Feature Set Details")
    st.dataframe(feature_set_details, width="stretch")

    st.subheader("Warnings")
    if warnings_df.empty:
        st.success("No warnings were recorded.")
    else:
        st.dataframe(warnings_df, width="stretch")

    if not pruning_results.empty:
        chart_df = pruning_results.copy()
        chart_df["mode_model"] = (
            chart_df["pruning_mode"].astype(str)
            + " / "
            + chart_df["model_type"].astype(str)
        )
        st.subheader("Test ROC AUC by Mode")
        st.bar_chart(chart_df.set_index("mode_model")["test_roc_auc"])

        st.subheader("Delta Test ROC AUC vs Full")
        st.bar_chart(chart_df.set_index("mode_model")["delta_test_roc_auc_vs_full"])


def load_pruning_summary_outputs(output_dir: str, report_name: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / report_name
    return {
        "pruning_mode_summary": pd.read_csv(base / "pruning_mode_summary.csv")
        if (base / "pruning_mode_summary.csv").exists()
        else pd.DataFrame(),
        "per_symbol_best_modes": pd.read_csv(base / "per_symbol_best_modes.csv")
        if (base / "per_symbol_best_modes.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(base / "warnings.csv")
        if (base / "warnings.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def render_pruning_summary_tab() -> None:
    st.write(
        "Aggregate symbol-level pruning experiments to decide whether a reduced "
        "feature set is stable enough for the next research round."
    )
    st.warning(
        "This summary is educational diagnostics only. A better reduced feature "
        "set still needs walk-forward validation and out-of-symbol retesting."
    )

    input_dirs_text = st.text_area(
        "Pruning output directories",
        value=",".join(DEFAULT_PRUNING_SUMMARY_DIRS),
        key="pruning_summary_input_dirs",
    )
    output_dir = st.text_input(
        "Pruning summary output directory",
        value="outputs/pruning_summary_real_v1",
        key="pruning_summary_output_dir",
    )
    report_name = st.text_input(
        "Pruning summary report name",
        value="pruning_summary_report.md",
        key="pruning_summary_report_name",
    )

    buttons = st.columns(2)
    generate_clicked = buttons[0].button(
        "Generate pruning summary",
        key="generate_pruning_summary_button",
        type="primary",
    )
    load_clicked = buttons[1].button(
        "Load pruning summary",
        key="load_pruning_summary_button",
    )

    if generate_clicked:
        try:
            result = save_pruning_summary_report(
                input_dirs=parse_pruning_summary_input_dirs(input_dirs_text),
                output_dir=output_dir,
                report_name=report_name,
            )
        except Exception as exc:
            st.error(f"Pruning summary generation failed: {exc}")
            return
    elif load_clicked:
        result = load_pruning_summary_outputs(output_dir, report_name)
    else:
        return

    mode_summary = result["pruning_mode_summary"]
    per_symbol_best = result["per_symbol_best_modes"]
    warnings_df = result["warnings"]
    report_text = result["markdown_report"]

    st.subheader("Pruning Mode Summary")
    if mode_summary.empty:
        st.info("No pruning mode summary rows are available.")
    else:
        st.dataframe(mode_summary, width="stretch")

    st.subheader("Per-Symbol Best Modes")
    if per_symbol_best.empty:
        st.info("No per-symbol best mode rows are available.")
    else:
        st.dataframe(per_symbol_best, width="stretch")

    st.subheader("Warnings")
    if warnings_df.empty:
        st.success("No warnings were recorded.")
    else:
        st.dataframe(warnings_df, width="stretch")

    st.subheader("Markdown Report")
    if report_text:
        st.markdown(report_text)
        st.download_button(
            "Download pruning summary report",
            data=report_text,
            file_name=report_name,
            mime="text/markdown",
            key="download_pruning_summary_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def load_reduced_feature_backtest_outputs(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(output_dir)
    return (
        pd.read_csv(base / "reduced_feature_backtest_summary.csv")
        if (base / "reduced_feature_backtest_summary.csv").exists()
        else pd.DataFrame(),
        pd.read_csv(base / "reduced_feature_backtest_results.csv")
        if (base / "reduced_feature_backtest_results.csv").exists()
        else pd.DataFrame(),
        pd.read_csv(base / "warnings.csv")
        if (base / "warnings.csv").exists()
        else pd.DataFrame(),
    )


def render_reduced_feature_backtest_tab() -> None:
    st.write(
        "Compare trading backtest performance across reduced feature sets. "
        "This tests trading behavior, not just ROC/F1 metrics."
    )
    st.warning(
        "Better ROC/F1 does not necessarily mean better trading return. This "
        "panel is educational research only, not financial advice."
    )

    factor_csv = st.text_input(
        "Reduced feature factor CSV",
        value="data/factors/smoke_factors_000001.csv",
        key="reduced_feature_factor_csv",
    )
    recommendations_path = st.text_input(
        "Reduced feature recommendations CSV",
        value="outputs/factor_ablation_demo/feature_pruning_recommendations.csv",
        key="reduced_feature_recommendations",
    )
    output_dir = st.text_input(
        "Reduced feature backtest output directory",
        value="outputs/reduced_feature_backtest_demo",
        key="reduced_feature_output_dir",
    )
    models_text = st.text_input(
        "Reduced feature model types",
        value="logistic_regression,random_forest",
        key="reduced_feature_models",
    )
    modes_text = st.text_input(
        "Reduced feature modes",
        value="full,drop_reduce_weight,keep_core_only,keep_core_and_observe",
        key="reduced_feature_modes",
    )
    target_col = st.text_input(
        "Reduced feature target column",
        value="label_up_5d",
        key="reduced_feature_target",
    )

    threshold_cols = st.columns(3)
    buy_threshold = threshold_cols[0].number_input(
        "Reduced buy threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01,
    )
    sell_threshold = threshold_cols[1].number_input(
        "Reduced sell threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
    )
    initial_cash = threshold_cols[2].number_input(
        "Reduced initial cash",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
    )

    cost_cols = st.columns(4)
    commission_rate = cost_cols[0].number_input(
        "Reduced commission rate",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    stamp_tax_rate = cost_cols[1].number_input(
        "Reduced stamp tax rate",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    slippage_pct = cost_cols[2].number_input(
        "Reduced slippage %",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f",
    )
    min_commission = cost_cols[3].number_input(
        "Reduced minimum commission",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    buttons = st.columns(2)
    run_clicked = buttons[0].button(
        "Run reduced feature backtest",
        key="run_reduced_feature_backtest_button",
        type="primary",
    )
    load_clicked = buttons[1].button(
        "Load reduced feature outputs",
        key="load_reduced_feature_backtest_button",
    )

    if run_clicked:
        try:
            result = run_and_save_reduced_feature_backtest(
                factor_csv=factor_csv,
                recommendations_path=recommendations_path,
                output_dir=output_dir,
                model_types=parse_ablation_model_types(models_text),
                pruning_modes=parse_pruning_modes(modes_text),
                target_col=target_col,
                initial_cash=initial_cash,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax_rate,
                slippage_pct=slippage_pct,
                min_commission=min_commission,
            )
            summary_df = result["summary"]
            results_df = result["results"]
            warnings_df = result["warnings"]
        except Exception as exc:
            st.error(f"Reduced feature backtest failed: {exc}")
            return
    elif load_clicked:
        summary_df, results_df, warnings_df = load_reduced_feature_backtest_outputs(
            output_dir
        )
    else:
        return

    st.subheader("Reduced Feature Backtest Summary")
    st.dataframe(summary_df, width="stretch")

    st.subheader("Full Results")
    st.dataframe(results_df, width="stretch")

    st.subheader("Warnings")
    if warnings_df.empty:
        st.success("No warnings were recorded.")
    else:
        st.dataframe(warnings_df, width="stretch")

    if not results_df.empty:
        chart_df = results_df.copy()
        chart_df["mode_model"] = (
            chart_df["pruning_mode"].astype(str)
            + " / "
            + chart_df["model_type"].astype(str)
        )
        st.subheader("Total Return")
        st.bar_chart(chart_df.set_index("mode_model")["total_return_pct"])
        st.subheader("Max Drawdown")
        st.bar_chart(chart_df.set_index("mode_model")["max_drawdown_pct"])
        st.subheader("Trade Count")
        st.bar_chart(chart_df.set_index("mode_model")["trade_count"])
        st.subheader("Strategy vs Benchmark")
        st.bar_chart(chart_df.set_index("mode_model")["strategy_vs_benchmark_pct"])


def load_reduced_feature_backtest_summary_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "reduced_feature_backtest_report.md"
    return {
        "mode_summary": pd.read_csv(
            base / "reduced_feature_backtest_mode_summary.csv"
        )
        if (base / "reduced_feature_backtest_mode_summary.csv").exists()
        else pd.DataFrame(),
        "model_summary": pd.read_csv(
            base / "reduced_feature_backtest_model_summary.csv"
        )
        if (base / "reduced_feature_backtest_model_summary.csv").exists()
        else pd.DataFrame(),
        "mode_model_summary": pd.read_csv(
            base / "reduced_feature_backtest_mode_model_summary.csv"
        )
        if (base / "reduced_feature_backtest_mode_model_summary.csv").exists()
        else pd.DataFrame(),
        "per_symbol_best": pd.read_csv(base / "per_symbol_best_backtest_modes.csv")
        if (base / "per_symbol_best_backtest_modes.csv").exists()
        else pd.DataFrame(),
        "underperformance": pd.read_csv(base / "underperformance_cases.csv")
        if (base / "underperformance_cases.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(base / "warnings.csv")
        if (base / "warnings.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def render_reduced_feature_backtest_summary_tab() -> None:
    st.write(
        "Aggregate reduced feature backtest results across symbols to compare "
        "pruning modes and model types by trading performance."
    )
    st.warning(
        "Reduced feature backtest summaries are research diagnostics only. "
        "A pruning mode should not become a default unless it is stable across "
        "symbols, models, drawdown, and trade count."
    )

    input_dirs_text = st.text_area(
        "Reduced feature backtest output directories",
        value=",".join(DEFAULT_REDUCED_FEATURE_SUMMARY_DIRS),
        key="reduced_feature_summary_input_dirs",
    )
    output_dir = st.text_input(
        "Reduced feature summary output directory",
        value="outputs/reduced_feature_backtest_summary_real_v1",
        key="reduced_feature_summary_output_dir",
    )
    min_trades = st.number_input(
        "Minimum trades warning threshold",
        min_value=0,
        value=3,
        step=1,
        key="reduced_feature_summary_min_trades",
    )

    buttons = st.columns(2)
    generate_clicked = buttons[0].button(
        "Generate reduced feature summary",
        key="generate_reduced_feature_summary_button",
        type="primary",
    )
    load_clicked = buttons[1].button(
        "Load reduced feature summary",
        key="load_reduced_feature_summary_button",
    )

    if generate_clicked:
        try:
            result = save_reduced_feature_backtest_report(
                input_dirs=parse_reduced_feature_summary_input_dirs(input_dirs_text),
                output_dir=output_dir,
                min_trades=int(min_trades),
            )
            mode_summary = result["reduced_feature_backtest_mode_summary"]
            model_summary = result["reduced_feature_backtest_model_summary"]
            mode_model_summary = result[
                "reduced_feature_backtest_mode_model_summary"
            ]
            per_symbol_best = result["per_symbol_best_backtest_modes"]
            underperformance = result["underperformance_cases"]
            warnings_df = result["warnings"]
            report_text = result["markdown_report"]
        except Exception as exc:
            st.error(f"Reduced feature summary generation failed: {exc}")
            return
    elif load_clicked:
        result = load_reduced_feature_backtest_summary_outputs(output_dir)
        mode_summary = result["mode_summary"]
        model_summary = result["model_summary"]
        mode_model_summary = result["mode_model_summary"]
        per_symbol_best = result["per_symbol_best"]
        underperformance = result["underperformance"]
        warnings_df = result["warnings"]
        report_text = result["markdown_report"]
    else:
        return

    card_cols = st.columns(4)
    card_cols[0].metric("Pruning modes", len(mode_summary))
    card_cols[1].metric("Model types", len(model_summary))
    card_cols[2].metric("Per-symbol rows", len(per_symbol_best))
    card_cols[3].metric("Warnings", len(warnings_df))

    st.subheader("Pruning Mode Summary")
    if mode_summary.empty:
        st.info("No pruning mode summary rows are available.")
    else:
        st.dataframe(mode_summary, width="stretch")

    st.subheader("Model Type Summary")
    st.dataframe(model_summary, width="stretch")

    st.subheader("Pruning Mode + Model Summary")
    st.dataframe(mode_model_summary, width="stretch")

    st.subheader("Per-Symbol Best Backtest Modes")
    st.dataframe(per_symbol_best, width="stretch")

    st.subheader("Underperformance Cases")
    if underperformance.empty:
        st.success("No underperformance cases were flagged.")
    else:
        st.dataframe(underperformance, width="stretch")

    st.subheader("Warnings")
    if warnings_df.empty:
        st.success("No warnings were recorded.")
    else:
        st.dataframe(warnings_df, width="stretch")

    if not mode_summary.empty:
        st.subheader("Average Strategy vs Benchmark by Mode")
        st.bar_chart(
            mode_summary.set_index("pruning_mode")[
                "avg_strategy_vs_benchmark_pct"
            ]
        )
        st.subheader("Stability Score by Mode")
        st.bar_chart(mode_summary.set_index("pruning_mode")["stability_score"])

    st.subheader("Markdown Report")
    if report_text:
        st.markdown(report_text)
        st.download_button(
            "Download reduced feature backtest report",
            data=report_text,
            file_name="reduced_feature_backtest_report.md",
            mime="text/markdown",
            key="download_reduced_feature_summary_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def load_threshold_sensitivity_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    return {
        "threshold_summary_by_mode": pd.read_csv(
            base / "threshold_summary_by_mode.csv"
        )
        if (base / "threshold_summary_by_mode.csv").exists()
        else pd.DataFrame(),
        "best_thresholds": pd.read_csv(base / "best_thresholds.csv", dtype={"symbol": str})
        if (base / "best_thresholds.csv").exists()
        else pd.DataFrame(),
        "walk_forward_summary": pd.read_csv(base / "walk_forward_summary.csv")
        if (base / "walk_forward_summary.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(base / "warnings.csv", dtype={"symbol": str})
        if (base / "warnings.csv").exists()
        else pd.DataFrame(),
    }


def load_threshold_report_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "threshold_experiment_report.md"
    return {
        "mode_summary": pd.read_csv(base / "threshold_mode_summary.csv")
        if (base / "threshold_mode_summary.csv").exists()
        else pd.DataFrame(),
        "per_symbol_best": pd.read_csv(
            base / "per_symbol_best_thresholds.csv",
            dtype={"symbol": str},
        )
        if (base / "per_symbol_best_thresholds.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(base / "warnings.csv", dtype={"symbol": str})
        if (base / "warnings.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_threshold_decision_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "threshold_decision_report.md"
    return {
        "decision_summary": pd.read_csv(base / "threshold_decision_summary.csv")
        if (base / "threshold_decision_summary.csv").exists()
        else pd.DataFrame(),
        "rejected": pd.read_csv(
            base / "rejected_or_low_confidence_configs.csv",
            dtype={"symbol": str},
        )
        if (base / "rejected_or_low_confidence_configs.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_candidate_validation_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "candidate_validation_report.md"
    return {
        "candidate_summary": pd.read_csv(base / "candidate_validation_summary.csv")
        if (base / "candidate_validation_summary.csv").exists()
        else pd.DataFrame(),
        "per_symbol_results": pd.read_csv(
            base / "per_symbol_candidate_results.csv",
            dtype={"symbol": str},
        )
        if (base / "per_symbol_candidate_results.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(
            base / "candidate_validation_warnings.csv",
            dtype={"symbol": str},
        )
        if (base / "candidate_validation_warnings.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_candidate_stress_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "candidate_stress_report.md"
    return {
        "stress_summary": pd.read_csv(base / "candidate_stress_summary.csv")
        if (base / "candidate_stress_summary.csv").exists()
        else pd.DataFrame(),
        "per_symbol_results": pd.read_csv(
            base / "per_symbol_stress_results.csv",
            dtype={"symbol": str},
        )
        if (base / "per_symbol_stress_results.csv").exists()
        else pd.DataFrame(),
        "regime_summary": pd.read_csv(base / "regime_summary.csv")
        if (base / "regime_summary.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(
            base / "stress_warnings.csv",
            dtype={"symbol": str},
        )
        if (base / "stress_warnings.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_candidate_equivalence_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "candidate_equivalence_report.md"
    return {
        "selected_features": pd.read_csv(
            base / "selected_features_by_symbol_mode.csv",
            dtype={"symbol": str},
        )
        if (base / "selected_features_by_symbol_mode.csv").exists()
        else pd.DataFrame(),
        "overlap_matrix": pd.read_csv(
            base / "feature_set_overlap_matrix.csv",
            dtype={"symbol": str},
        )
        if (base / "feature_set_overlap_matrix.csv").exists()
        else pd.DataFrame(),
        "equivalence_summary": pd.read_csv(
            base / "feature_set_equivalence_summary.csv"
        )
        if (base / "feature_set_equivalence_summary.csv").exists()
        else pd.DataFrame(),
        "feature_frequency": pd.read_csv(base / "feature_frequency_by_mode.csv")
        if (base / "feature_frequency_by_mode.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(base / "warnings.csv", dtype={"symbol": str})
        if (base / "warnings.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_candidate_mode_normalization_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "canonical_mode_report.md"
    return {
        "canonical_summary": pd.read_csv(base / "canonical_mode_summary.csv")
        if (base / "canonical_mode_summary.csv").exists()
        else pd.DataFrame(),
        "alias_map": pd.read_csv(base / "legacy_alias_map.csv")
        if (base / "legacy_alias_map.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_canonical_revalidation_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "canonical_candidate_revalidation_report.md"
    return {
        "summary": pd.read_csv(base / "canonical_candidate_revalidation_summary.csv")
        if (base / "canonical_candidate_revalidation_summary.csv").exists()
        else pd.DataFrame(),
        "risk_flags": pd.read_csv(base / "candidate_risk_flags.csv", dtype={"symbol": str})
        if (base / "candidate_risk_flags.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_candidate_validation_gate_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "candidate_validation_gate_report.md"
    return {
        "results": pd.read_csv(base / "validation_gate_results.csv")
        if (base / "validation_gate_results.csv").exists()
        else pd.DataFrame(),
        "failures": pd.read_csv(base / "validation_gate_failures.csv")
        if (base / "validation_gate_failures.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_validation_gate_failure_analysis_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "validation_gate_failure_analysis_report.md"
    return {
        "gate_failure_summary": pd.read_csv(base / "gate_failure_summary.csv")
        if (base / "gate_failure_summary.csv").exists()
        else pd.DataFrame(),
        "failure_by_check": pd.read_csv(base / "failure_by_check.csv")
        if (base / "failure_by_check.csv").exists()
        else pd.DataFrame(),
        "failure_by_candidate": pd.read_csv(base / "failure_by_candidate.csv")
        if (base / "failure_by_candidate.csv").exists()
        else pd.DataFrame(),
        "failure_by_symbol": pd.read_csv(
            base / "failure_by_symbol.csv",
            dtype={"symbol": str},
        )
        if (base / "failure_by_symbol.csv").exists()
        else pd.DataFrame(),
        "failure_by_regime": pd.read_csv(base / "failure_by_regime.csv")
        if (base / "failure_by_regime.csv").exists()
        else pd.DataFrame(),
        "risk_flag_summary": pd.read_csv(
            base / "risk_flag_summary.csv",
            dtype={"symbol": str},
        )
        if (base / "risk_flag_summary.csv").exists()
        else pd.DataFrame(),
        "remediation_plan": pd.read_csv(base / "remediation_plan.csv")
        if (base / "remediation_plan.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_targeted_remediation_design_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "targeted_remediation_design_report.md"
    return {
        "experiments": pd.read_csv(base / "targeted_remediation_experiments.csv")
        if (base / "targeted_remediation_experiments.csv").exists()
        else pd.DataFrame(),
        "regime_plan": pd.read_csv(base / "regime_remediation_plan.csv")
        if (base / "regime_remediation_plan.csv").exists()
        else pd.DataFrame(),
        "candidate_plan": pd.read_csv(base / "candidate_remediation_plan.csv")
        if (base / "candidate_remediation_plan.csv").exists()
        else pd.DataFrame(),
        "symbol_priority": pd.read_csv(
            base / "symbol_remediation_priority.csv",
            dtype={"symbol": str},
        )
        if (base / "symbol_remediation_priority.csv").exists()
        else pd.DataFrame(),
        "success_criteria": pd.read_csv(base / "remediation_success_criteria.csv")
        if (base / "remediation_success_criteria.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(base / "warnings.csv")
        if (base / "warnings.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_bull_regime_threshold_remediation_outputs(output_dir: str) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "bull_remediation_report.md"
    return {
        "results": pd.read_csv(base / "bull_threshold_results.csv", dtype={"symbol": str})
        if (base / "bull_threshold_results.csv").exists()
        else pd.DataFrame(),
        "summary": pd.read_csv(base / "bull_threshold_summary.csv")
        if (base / "bull_threshold_summary.csv").exists()
        else pd.DataFrame(),
        "per_symbol": pd.read_csv(
            base / "per_symbol_bull_results.csv",
            dtype={"symbol": str},
        )
        if (base / "per_symbol_bull_results.csv").exists()
        else pd.DataFrame(),
        "best": pd.read_csv(base / "best_bull_thresholds.csv")
        if (base / "best_bull_thresholds.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(base / "warnings.csv", dtype={"symbol": str})
        if (base / "warnings.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def load_sideways_regime_trade_sufficiency_remediation_outputs(
    output_dir: str,
) -> dict[str, object]:
    base = Path(output_dir)
    report_path = base / "sideways_remediation_report.md"
    return {
        "results": pd.read_csv(base / "sideways_trade_results.csv", dtype={"symbol": str})
        if (base / "sideways_trade_results.csv").exists()
        else pd.DataFrame(),
        "summary": pd.read_csv(base / "sideways_trade_summary.csv")
        if (base / "sideways_trade_summary.csv").exists()
        else pd.DataFrame(),
        "per_symbol": pd.read_csv(
            base / "per_symbol_sideways_results.csv",
            dtype={"symbol": str},
        )
        if (base / "per_symbol_sideways_results.csv").exists()
        else pd.DataFrame(),
        "best": pd.read_csv(base / "best_sideways_thresholds.csv")
        if (base / "best_sideways_thresholds.csv").exists()
        else pd.DataFrame(),
        "warnings": pd.read_csv(base / "warnings.csv", dtype={"symbol": str})
        if (base / "warnings.csv").exists()
        else pd.DataFrame(),
        "markdown_report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


def render_threshold_sensitivity_tab() -> None:
    st.write(
        "Test reduced feature ML signal backtests across probability thresholds "
        "and optional chronological walk-forward windows."
    )
    st.warning(
        "Threshold tuning can overfit historical data. This panel is educational "
        "research only, not financial advice."
    )

    mode = st.radio(
        "Threshold sensitivity action",
        ["Single-symbol experiment", "Multi-symbol report"],
        horizontal=True,
        key="threshold_sensitivity_action",
    )

    if mode == "Single-symbol experiment":
        factor_csv = st.text_input(
            "Threshold factor CSV",
            value="data/factors/smoke_factors_000001.csv",
            key="threshold_factor_csv",
        )
        recommendations_path = st.text_input(
            "Threshold recommendations CSV",
            value="outputs/factor_ablation_demo/feature_pruning_recommendations.csv",
            key="threshold_recommendations",
        )
        output_dir = st.text_input(
            "Threshold experiment output directory",
            value="outputs/reduced_feature_threshold_demo",
            key="threshold_output_dir",
        )
        models_text = st.text_input(
            "Threshold model types",
            value="logistic_regression,random_forest",
            key="threshold_models",
        )
        pruning_modes_text = st.text_input(
            "Threshold pruning modes",
            value="full,drop_reduce_weight,keep_core_only,keep_core_and_observe",
            key="threshold_pruning_modes",
        )
        threshold_cols = st.columns(2)
        buy_thresholds_text = threshold_cols[0].text_input(
            "Buy thresholds",
            value=",".join(f"{value:.2f}" for value in DEFAULT_REDUCED_THRESHOLD_BUYS),
            key="threshold_buy_values",
        )
        sell_thresholds_text = threshold_cols[1].text_input(
            "Sell thresholds",
            value=",".join(f"{value:.2f}" for value in DEFAULT_REDUCED_THRESHOLD_SELLS),
            key="threshold_sell_values",
        )
        enable_walk_forward = st.checkbox(
            "Enable walk-forward validation",
            value=False,
            key="threshold_enable_walk_forward",
        )

        buttons = st.columns(2)
        run_clicked = buttons[0].button(
            "Run threshold experiment",
            key="run_threshold_sensitivity_button",
            type="primary",
        )
        load_clicked = buttons[1].button(
            "Load threshold outputs",
            key="load_threshold_sensitivity_button",
        )

        if run_clicked:
            try:
                result = run_and_save_threshold_experiment(
                    factor_csv=factor_csv,
                    recommendations_path=recommendations_path,
                    output_dir=output_dir,
                    model_types=parse_ablation_model_types(models_text),
                    pruning_modes=parse_pruning_modes(pruning_modes_text),
                    buy_thresholds=parse_reduced_thresholds(
                        buy_thresholds_text,
                        DEFAULT_REDUCED_THRESHOLD_BUYS,
                    ),
                    sell_thresholds=parse_reduced_thresholds(
                        sell_thresholds_text,
                        DEFAULT_REDUCED_THRESHOLD_SELLS,
                    ),
                    enable_walk_forward=enable_walk_forward,
                )
                output = {
                    "threshold_summary_by_mode": result["threshold_summary_by_mode"],
                    "best_thresholds": result["best_thresholds"],
                    "walk_forward_summary": result.get(
                        "walk_forward_summary",
                        pd.DataFrame(),
                    ),
                    "warnings": result["warnings"],
                }
            except Exception as exc:
                st.error(f"Threshold experiment failed: {exc}")
                return
        elif load_clicked:
            output = load_threshold_sensitivity_outputs(output_dir)
        else:
            return

        st.subheader("Threshold Summary by Mode")
        st.dataframe(output["threshold_summary_by_mode"], width="stretch")
        st.subheader("Best Thresholds")
        st.dataframe(output["best_thresholds"], width="stretch")
        st.subheader("Walk-Forward Summary")
        if output["walk_forward_summary"].empty:
            st.info("No walk-forward summary is available.")
        else:
            st.dataframe(output["walk_forward_summary"], width="stretch")
        st.subheader("Warnings")
        if output["warnings"].empty:
            st.success("No warnings were recorded.")
        else:
            st.dataframe(output["warnings"], width="stretch")
        if not output["threshold_summary_by_mode"].empty:
            st.subheader("Average Excess Return by Mode")
            st.bar_chart(
                output["threshold_summary_by_mode"].set_index("pruning_mode")[
                    "avg_strategy_vs_benchmark_pct"
                ]
            )
    else:
        input_dirs_text = st.text_area(
            "Threshold experiment output directories",
            value="outputs/reduced_feature_threshold_demo",
            key="threshold_report_input_dirs",
        )
        output_dir = st.text_input(
            "Threshold report output directory",
            value="outputs/reduced_feature_threshold_summary_demo",
            key="threshold_report_output_dir",
        )

        buttons = st.columns(2)
        generate_clicked = buttons[0].button(
            "Generate threshold report",
            key="generate_threshold_report_button",
            type="primary",
        )
        load_clicked = buttons[1].button(
            "Load threshold report",
            key="load_threshold_report_button",
        )
        if generate_clicked:
            try:
                result = save_threshold_experiment_report(
                    parse_threshold_report_input_dirs(input_dirs_text),
                    output_dir,
                )
                output = {
                    "mode_summary": result["threshold_mode_summary"],
                    "per_symbol_best": result["per_symbol_best_thresholds"],
                    "warnings": result["warnings"],
                    "markdown_report": result["markdown_report"],
                }
            except Exception as exc:
                st.error(f"Threshold report generation failed: {exc}")
                return
        elif load_clicked:
            output = load_threshold_report_outputs(output_dir)
        else:
            return

        st.subheader("Threshold Mode Summary")
        st.dataframe(output["mode_summary"], width="stretch")
        st.subheader("Per-Symbol Best Thresholds")
        st.dataframe(output["per_symbol_best"], width="stretch")
        st.subheader("Warnings")
        st.dataframe(output["warnings"], width="stretch")
        st.subheader("Markdown Report")
        if output["markdown_report"]:
            st.markdown(output["markdown_report"])
            st.download_button(
                "Download threshold experiment report",
                data=output["markdown_report"],
                file_name="threshold_experiment_report.md",
                mime="text/markdown",
                key="download_threshold_experiment_report_button",
            )


def render_threshold_decision_report_tab() -> None:
    st.write(
        "Generate a conservative research decision report from reduced feature "
        "threshold summary outputs."
    )
    st.warning(
        "This report is educational diagnostics only. It is not trading-ready "
        "and is not financial advice."
    )

    summary_dir = st.text_input(
        "Threshold summary directory",
        value="outputs/reduced_feature_threshold_summary_real_v1",
        key="threshold_decision_summary_dir",
    )
    output_dir = st.text_input(
        "Threshold decision output directory",
        value="outputs/threshold_decision_real_v1",
        key="threshold_decision_output_dir",
    )

    buttons = st.columns(2)
    generate_clicked = buttons[0].button(
        "Generate threshold decision report",
        key="generate_threshold_decision_report_button",
        type="primary",
    )
    load_clicked = buttons[1].button(
        "Load threshold decision report",
        key="load_threshold_decision_report_button",
    )

    if generate_clicked:
        try:
            result = save_threshold_decision_report(summary_dir, output_dir)
            output = {
                "decision_summary": result["decision_summary"],
                "rejected": result["rejected_or_low_confidence_configs"],
                "markdown_report": result["markdown_report"],
            }
        except Exception as exc:
            st.error(f"Threshold decision report generation failed: {exc}")
            return
    elif load_clicked:
        output = load_threshold_decision_outputs(output_dir)
    else:
        return

    st.subheader("Decision Summary")
    st.dataframe(output["decision_summary"], width="stretch")
    st.subheader("Rejected or Low-Confidence Configurations")
    st.dataframe(output["rejected"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download threshold decision report",
            data=output["markdown_report"],
            file_name="threshold_decision_report.md",
            mime="text/markdown",
            key="download_threshold_decision_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_candidate_validation_tab() -> None:
    st.write(
        "Run or load expanded validation for the recommended reduced-feature "
        "threshold candidates."
    )
    st.warning(
        "Expanded validation is still educational research only. It is not "
        "trading-ready and is not financial advice. Summaries use canonical "
        "candidate modes while preserving legacy pruning modes for traceability."
    )

    mode = st.radio(
        "Candidate validation action",
        ["Load output", "Run validation"],
        horizontal=True,
        key="candidate_validation_action",
    )
    output_dir = st.text_input(
        "Candidate validation output directory",
        value="outputs/candidate_validation_real_v1",
        key="candidate_validation_output_dir",
    )

    if mode == "Run validation":
        factor_dir = st.text_input(
            "Factor directory",
            value="outputs/model_robustness_real_v2/factors",
            key="candidate_validation_factor_dir",
        )
        symbols_text = st.text_input(
            "Symbols",
            value="000001,600519,000858,600036,601318",
            key="candidate_validation_symbols",
        )
        recommendations_path = st.text_input(
            "Recommendations CSV",
            value="outputs/feature_ablation_real_v1/feature_pruning_recommendations.csv",
            key="candidate_validation_recommendations",
        )
        enable_walk_forward = st.checkbox(
            "Enable walk-forward candidate validation",
            value=False,
            key="candidate_validation_enable_walk_forward",
        )
        if st.button(
            "Run candidate validation",
            key="run_candidate_validation_button",
            type="primary",
        ):
            try:
                result = save_candidate_expanded_validation(
                    factor_dir=factor_dir,
                    symbols=parse_candidate_validation_symbols(symbols_text),
                    recommendations_path=recommendations_path,
                    output_dir=output_dir,
                    enable_walk_forward=enable_walk_forward,
                )
                output = {
                    "candidate_summary": result["candidate_validation_summary"],
                    "per_symbol_results": result["per_symbol_candidate_results"],
                    "warnings": result["candidate_validation_warnings"],
                    "markdown_report": result["candidate_validation_report"],
                }
            except Exception as exc:
                st.error(f"Candidate validation failed: {exc}")
                return
        else:
            return
    else:
        if not st.button(
            "Load candidate validation",
            key="load_candidate_validation_button",
        ):
            return
        output = load_candidate_validation_outputs(output_dir)

    st.subheader("Candidate Summary")
    st.dataframe(output["candidate_summary"], width="stretch")
    st.subheader("Per-Symbol Results")
    st.dataframe(output["per_symbol_results"], width="stretch")
    st.subheader("Warnings")
    st.dataframe(output["warnings"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download candidate validation report",
            data=output["markdown_report"],
            file_name="candidate_validation_report.md",
            mime="text/markdown",
            key="download_candidate_validation_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_candidate_stress_test_tab() -> None:
    st.write(
        "Load or run candidate market-regime stress tests for the current "
        "research candidates."
    )
    st.warning(
        "Candidate stress tests are educational research only. They are not "
        "trading-ready and are not financial advice. Candidate summaries collapse "
        "legacy equivalent modes into canonical modes."
    )

    mode = st.radio(
        "Candidate stress action",
        ["Load output", "Run stress test"],
        horizontal=True,
        key="candidate_stress_action",
    )
    output_dir = st.text_input(
        "Candidate stress output directory",
        value="outputs/candidate_stress_real_v1",
        key="candidate_stress_output_dir",
    )
    if mode == "Run stress test":
        factor_dir = st.text_input(
            "Stress factor directory",
            value="outputs/model_robustness_real_v2/factors",
            key="candidate_stress_factor_dir",
        )
        symbols_text = st.text_input(
            "Stress symbols",
            value="000001,600519,000858,600036,601318",
            key="candidate_stress_symbols",
        )
        recommendations_path = st.text_input(
            "Stress recommendations CSV",
            value="outputs/feature_ablation_real_v1/feature_pruning_recommendations.csv",
            key="candidate_stress_recommendations",
        )
        regime_window = st.number_input(
            "Regime window",
            min_value=2,
            value=60,
            step=5,
            key="candidate_stress_regime_window",
        )
        enable_walk_forward = st.checkbox(
            "Include walk-forward candidate",
            value=True,
            key="candidate_stress_enable_walk_forward",
        )
        if st.button(
            "Run candidate stress test",
            key="run_candidate_stress_button",
            type="primary",
        ):
            try:
                result = save_candidate_stress_test(
                    factor_dir=factor_dir,
                    symbols=parse_candidate_validation_symbols(symbols_text),
                    recommendations_path=recommendations_path,
                    output_dir=output_dir,
                    regime_window=int(regime_window),
                    enable_walk_forward=enable_walk_forward,
                )
                output = {
                    "stress_summary": result["candidate_stress_summary"],
                    "per_symbol_results": result["per_symbol_stress_results"],
                    "regime_summary": result["regime_summary"],
                    "warnings": result["stress_warnings"],
                    "markdown_report": result["candidate_stress_report"],
                }
            except Exception as exc:
                st.error(f"Candidate stress test failed: {exc}")
                return
        else:
            return
    else:
        if not st.button("Load candidate stress test", key="load_candidate_stress_button"):
            return
        output = load_candidate_stress_outputs(output_dir)

    st.subheader("Candidate Stress Summary")
    st.dataframe(output["stress_summary"], width="stretch")
    st.subheader("Regime Summary")
    st.dataframe(output["regime_summary"], width="stretch")
    st.subheader("Per-Symbol Stress Results")
    st.dataframe(output["per_symbol_results"], width="stretch")
    st.subheader("Warnings")
    st.dataframe(output["warnings"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download candidate stress report",
            data=output["markdown_report"],
            file_name="candidate_stress_report.md",
            mime="text/markdown",
            key="download_candidate_stress_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_candidate_equivalence_audit_tab() -> None:
    st.write(
        "Audit actual selected feature sets by pruning mode and check whether "
        "candidate modes are equivalent or redundant."
    )
    st.warning(
        "This is an audit/reporting step only. It is not a trading recommendation."
    )

    mode = st.radio(
        "Candidate equivalence action",
        ["Load output", "Run audit"],
        horizontal=True,
        key="candidate_equivalence_action",
    )
    output_dir = st.text_input(
        "Candidate equivalence output directory",
        value="outputs/candidate_equivalence_real_v1",
        key="candidate_equivalence_output_dir",
    )
    if mode == "Run audit":
        factor_dir = st.text_input(
            "Equivalence factor directory",
            value="outputs/model_robustness_real_v2/factors",
            key="candidate_equivalence_factor_dir",
        )
        symbols_text = st.text_input(
            "Equivalence symbols",
            value="000001,600519,000858,600036,601318",
            key="candidate_equivalence_symbols",
        )
        recommendations_path = st.text_input(
            "Equivalence recommendations CSV",
            value="outputs/feature_ablation_real_v1/feature_pruning_recommendations.csv",
            key="candidate_equivalence_recommendations",
        )
        if st.button(
            "Run candidate equivalence audit",
            key="run_candidate_equivalence_button",
            type="primary",
        ):
            try:
                result = save_candidate_equivalence_audit(
                    factor_dir=factor_dir,
                    symbols=parse_candidate_validation_symbols(symbols_text),
                    recommendations_path=recommendations_path,
                    output_dir=output_dir,
                )
                output = {
                    "selected_features": result["selected_features_by_symbol_mode"],
                    "overlap_matrix": result["feature_set_overlap_matrix"],
                    "equivalence_summary": result["feature_set_equivalence_summary"],
                    "feature_frequency": result["feature_frequency_by_mode"],
                    "warnings": result["warnings"],
                    "markdown_report": result["candidate_equivalence_report"],
                }
            except Exception as exc:
                st.error(f"Candidate equivalence audit failed: {exc}")
                return
        else:
            return
    else:
        if not st.button(
            "Load candidate equivalence audit",
            key="load_candidate_equivalence_button",
        ):
            return
        output = load_candidate_equivalence_outputs(output_dir)

    st.subheader("Equivalence Summary")
    st.dataframe(output["equivalence_summary"], width="stretch")
    st.subheader("Feature Set Overlap Matrix")
    st.dataframe(output["overlap_matrix"], width="stretch")
    st.subheader("Feature Frequency by Mode")
    st.dataframe(output["feature_frequency"], width="stretch")
    st.subheader("Selected Features")
    st.dataframe(output["selected_features"], width="stretch")
    st.subheader("Warnings")
    st.dataframe(output["warnings"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download candidate equivalence report",
            data=output["markdown_report"],
            file_name="candidate_equivalence_report.md",
            mime="text/markdown",
            key="download_candidate_equivalence_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_candidate_mode_normalization_tab() -> None:
    st.write(
        "Export canonical candidate mode aliases so equivalent legacy modes are "
        "not treated as independent candidates."
    )
    st.warning(
        "`drop_reduce_weight` and `keep_core_and_observe` are legacy equivalent "
        "modes when the audit shows identical feature sets."
    )

    mode = st.radio(
        "Candidate mode normalization action",
        ["Load output", "Generate normalization report"],
        horizontal=True,
        key="candidate_mode_normalization_action",
    )
    output_dir = st.text_input(
        "Canonical mode output directory",
        value="outputs/candidate_mode_normalization_real_v1",
        key="candidate_mode_normalization_output_dir",
    )
    if mode == "Generate normalization report":
        equivalence_dir = st.text_input(
            "Candidate equivalence audit directory",
            value="outputs/candidate_equivalence_real_v1",
            key="candidate_mode_normalization_equivalence_dir",
        )
        if st.button(
            "Generate canonical mode report",
            key="generate_candidate_mode_normalization_button",
            type="primary",
        ):
            try:
                result = save_canonical_mode_report(equivalence_dir, output_dir)
                output = {
                    "canonical_summary": result["canonical_mode_summary"],
                    "alias_map": result["legacy_alias_map"],
                    "markdown_report": result["canonical_mode_report"],
                }
            except Exception as exc:
                st.error(f"Candidate mode normalization failed: {exc}")
                return
        else:
            return
    else:
        if not st.button(
            "Load canonical mode report",
            key="load_candidate_mode_normalization_button",
        ):
            return
        output = load_candidate_mode_normalization_outputs(output_dir)

    st.subheader("Canonical Candidate Modes")
    st.dataframe(output["canonical_summary"], width="stretch")
    st.subheader("Legacy Alias Map")
    st.dataframe(output["alias_map"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download canonical mode report",
            data=output["markdown_report"],
            file_name="canonical_mode_report.md",
            mime="text/markdown",
            key="download_candidate_mode_normalization_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_canonical_revalidation_tab() -> None:
    st.write(
        "Consolidate canonical candidate validation, stress, and threshold "
        "decision outputs into one final research decision report."
    )
    st.warning(
        "This is reporting-only research control. It is not trading-ready and "
        "is not financial advice."
    )

    mode = st.radio(
        "Canonical revalidation action",
        ["Load output", "Generate report"],
        horizontal=True,
        key="canonical_revalidation_action",
    )
    output_dir = st.text_input(
        "Canonical revalidation output directory",
        value="outputs/canonical_candidate_revalidation_real_v1",
        key="canonical_revalidation_output_dir",
    )
    if mode == "Generate report":
        expanded_dir = st.text_input(
            "Expanded validation directory",
            value="outputs/candidate_expanded_validation_real_v2",
            key="canonical_revalidation_expanded_dir",
        )
        stress_dir = st.text_input(
            "Stress validation directory",
            value="outputs/candidate_stress_real_v2",
            key="canonical_revalidation_stress_dir",
        )
        threshold_dir = st.text_input(
            "Threshold decision directory",
            value="outputs/threshold_decision_real_v2",
            key="canonical_revalidation_threshold_dir",
        )
        if st.button(
            "Generate canonical revalidation report",
            key="generate_canonical_revalidation_button",
            type="primary",
        ):
            try:
                result = save_canonical_candidate_revalidation_report(
                    expanded_validation_dir=expanded_dir,
                    stress_dir=stress_dir,
                    threshold_decision_dir=threshold_dir,
                    output_dir=output_dir,
                )
                output = {
                    "summary": result["canonical_candidate_revalidation_summary"],
                    "risk_flags": result["candidate_risk_flags"],
                    "markdown_report": result[
                        "canonical_candidate_revalidation_report"
                    ],
                }
            except Exception as exc:
                st.error(f"Canonical revalidation report failed: {exc}")
                return
        else:
            return
    else:
        if not st.button(
            "Load canonical revalidation report",
            key="load_canonical_revalidation_button",
        ):
            return
        output = load_canonical_revalidation_outputs(output_dir)

    st.subheader("Canonical Revalidation Summary")
    st.dataframe(output["summary"], width="stretch")
    st.subheader("Candidate Risk Flags")
    st.dataframe(output["risk_flags"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download canonical revalidation report",
            data=output["markdown_report"],
            file_name="canonical_candidate_revalidation_report.md",
            mime="text/markdown",
            key="download_canonical_revalidation_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_candidate_validation_gate_tab() -> None:
    st.write(
        "Run a strict gate over canonical revalidation outputs before any "
        "candidate can be described as trading-ready."
    )
    st.warning(
        "This is educational research diagnostics only. Passing or failing the "
        "gate is not financial advice."
    )

    mode = st.radio(
        "Candidate validation gate action",
        ["Load output", "Generate gate"],
        horizontal=True,
        key="candidate_validation_gate_action",
    )
    output_dir = st.text_input(
        "Candidate validation gate output directory",
        value="outputs/candidate_validation_gate_real_v1",
        key="candidate_validation_gate_output_dir",
    )
    if mode == "Generate gate":
        revalidation_dir = st.text_input(
            "Canonical revalidation directory",
            value="outputs/canonical_candidate_revalidation_real_v1",
            key="candidate_validation_gate_revalidation_dir",
        )
        if st.button(
            "Run candidate validation gate",
            key="run_candidate_validation_gate_button",
            type="primary",
        ):
            try:
                result = save_candidate_validation_gate(
                    revalidation_dir=revalidation_dir,
                    output_dir=output_dir,
                )
                output = {
                    "results": result["validation_gate_results"],
                    "failures": result["validation_gate_failures"],
                    "markdown_report": result["candidate_validation_gate_report"],
                }
            except Exception as exc:
                st.error(f"Candidate validation gate failed: {exc}")
                return
        else:
            return
    else:
        if not st.button(
            "Load candidate validation gate",
            key="load_candidate_validation_gate_button",
        ):
            return
        output = load_candidate_validation_gate_outputs(output_dir)

    st.subheader("Validation Gate Results")
    st.dataframe(output["results"], width="stretch")
    st.subheader("Validation Gate Failures")
    st.dataframe(output["failures"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download candidate validation gate report",
            data=output["markdown_report"],
            file_name="candidate_validation_gate_report.md",
            mime="text/markdown",
            key="download_candidate_validation_gate_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_validation_gate_failure_analysis_tab() -> None:
    st.write(
        "Analyze why canonical candidates are blocked by the validation gate "
        "and summarize remediation priorities."
    )
    st.warning(
        "This is reporting-only research diagnostics. It does not change "
        "trading, backtest, model, or strategy logic."
    )

    mode = st.radio(
        "Failure analysis action",
        ["Load output", "Generate analysis"],
        horizontal=True,
        key="validation_gate_failure_analysis_action",
    )
    output_dir = st.text_input(
        "Failure analysis output directory",
        value="outputs/validation_gate_failure_analysis_real_v1",
        key="validation_gate_failure_analysis_output_dir",
    )
    if mode == "Generate analysis":
        gate_dir = st.text_input(
            "Candidate validation gate directory",
            value="outputs/candidate_validation_gate_real_v1",
            key="validation_gate_failure_analysis_gate_dir",
        )
        revalidation_dir = st.text_input(
            "Canonical revalidation directory",
            value="outputs/canonical_candidate_revalidation_real_v1",
            key="validation_gate_failure_analysis_revalidation_dir",
        )
        stress_dir = st.text_input(
            "Candidate stress directory",
            value="outputs/candidate_stress_real_v2",
            key="validation_gate_failure_analysis_stress_dir",
        )
        if st.button(
            "Generate failure analysis",
            key="run_validation_gate_failure_analysis_button",
            type="primary",
        ):
            try:
                result = save_validation_gate_failure_analysis(
                    gate_dir=gate_dir,
                    revalidation_dir=revalidation_dir,
                    stress_dir=stress_dir,
                    output_dir=output_dir,
                )
                output = {
                    "gate_failure_summary": result["gate_failure_summary"],
                    "failure_by_check": result["failure_by_check"],
                    "failure_by_candidate": result["failure_by_candidate"],
                    "failure_by_symbol": result["failure_by_symbol"],
                    "failure_by_regime": result["failure_by_regime"],
                    "risk_flag_summary": result["risk_flag_summary"],
                    "remediation_plan": result["remediation_plan"],
                    "markdown_report": result[
                        "validation_gate_failure_analysis_report"
                    ],
                }
            except Exception as exc:
                st.error(f"Validation gate failure analysis failed: {exc}")
                return
        else:
            return
    else:
        if not st.button(
            "Load failure analysis",
            key="load_validation_gate_failure_analysis_button",
        ):
            return
        output = load_validation_gate_failure_analysis_outputs(output_dir)

    st.subheader("Gate Failure Summary")
    st.dataframe(output["gate_failure_summary"], width="stretch")
    st.subheader("Failure by Candidate")
    st.dataframe(output["failure_by_candidate"], width="stretch")
    st.subheader("Failure by Check")
    st.dataframe(output["failure_by_check"], width="stretch")
    st.subheader("Failure by Regime")
    st.dataframe(output["failure_by_regime"], width="stretch")
    st.subheader("Failure by Symbol")
    st.dataframe(output["failure_by_symbol"], width="stretch")
    st.subheader("Risk Flag Summary")
    st.dataframe(output["risk_flag_summary"], width="stretch")
    st.subheader("Remediation Plan")
    st.dataframe(output["remediation_plan"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download validation gate failure analysis report",
            data=output["markdown_report"],
            file_name="validation_gate_failure_analysis_report.md",
            mime="text/markdown",
            key="download_validation_gate_failure_analysis_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_targeted_remediation_design_tab() -> None:
    st.write(
        "Convert validation gate failure analysis into targeted next-experiment "
        "designs without adding features, agents, models, or strategy logic."
    )
    st.warning(
        "This is educational research diagnostics only. It is not trading-ready "
        "and is not financial advice."
    )

    mode = st.radio(
        "Targeted remediation design action",
        ["Load output", "Generate design"],
        horizontal=True,
        key="targeted_remediation_design_action",
    )
    output_dir = st.text_input(
        "Targeted remediation design output directory",
        value="outputs/targeted_remediation_design_real_v1",
        key="targeted_remediation_design_output_dir",
    )
    if mode == "Generate design":
        failure_analysis_dir = st.text_input(
            "Failure analysis directory",
            value="outputs/validation_gate_failure_analysis_real_v1",
            key="targeted_remediation_design_failure_analysis_dir",
        )
        gate_dir = st.text_input(
            "Candidate validation gate directory",
            value="outputs/candidate_validation_gate_real_v1",
            key="targeted_remediation_design_gate_dir",
        )
        revalidation_dir = st.text_input(
            "Canonical revalidation directory",
            value="outputs/canonical_candidate_revalidation_real_v1",
            key="targeted_remediation_design_revalidation_dir",
        )
        if st.button(
            "Generate targeted remediation design",
            key="run_targeted_remediation_design_button",
            type="primary",
        ):
            try:
                result = save_targeted_remediation_design(
                    failure_analysis_dir=failure_analysis_dir,
                    gate_dir=gate_dir,
                    revalidation_dir=revalidation_dir,
                    output_dir=output_dir,
                )
                output = {
                    "experiments": result["targeted_remediation_experiments"],
                    "regime_plan": result["regime_remediation_plan"],
                    "candidate_plan": result["candidate_remediation_plan"],
                    "symbol_priority": result["symbol_remediation_priority"],
                    "success_criteria": result["remediation_success_criteria"],
                    "warnings": result["warnings"],
                    "markdown_report": result["targeted_remediation_design_report"],
                }
            except Exception as exc:
                st.error(f"Targeted remediation design failed: {exc}")
                return
        else:
            return
    else:
        if not st.button(
            "Load targeted remediation design",
            key="load_targeted_remediation_design_button",
        ):
            return
        output = load_targeted_remediation_design_outputs(output_dir)

    st.subheader("Targeted Remediation Experiments")
    st.dataframe(output["experiments"], width="stretch")
    st.subheader("Regime Remediation Plan")
    st.dataframe(output["regime_plan"], width="stretch")
    st.subheader("Candidate Remediation Plan")
    st.dataframe(output["candidate_plan"], width="stretch")
    st.subheader("Symbol Remediation Priority")
    st.dataframe(output["symbol_priority"], width="stretch")
    st.subheader("Remediation Success Criteria")
    st.dataframe(output["success_criteria"], width="stretch")
    st.subheader("Warnings")
    st.dataframe(output["warnings"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download targeted remediation design report",
            data=output["markdown_report"],
            file_name="targeted_remediation_design_report.md",
            mime="text/markdown",
            key="download_targeted_remediation_design_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_bull_regime_threshold_remediation_tab() -> None:
    st.write(
        "Run a bull-regime-only threshold remediation diagnostic for "
        "canonical_reduced_40 + logistic_regression."
    )
    st.warning(
        "This is educational research diagnostics only. It is not trading-ready "
        "and does not add features, agents, models, or strategy logic."
    )

    mode = st.radio(
        "Bull remediation action",
        ["Load output", "Run experiment"],
        horizontal=True,
        key="bull_regime_threshold_remediation_action",
    )
    output_dir = st.text_input(
        "Bull remediation output directory",
        value="outputs/bull_regime_threshold_remediation_real_v1",
        key="bull_regime_threshold_remediation_output_dir",
    )
    if mode == "Run experiment":
        factor_dir = st.text_input(
            "Factor directory",
            value="outputs/model_robustness_real_v2/factors",
            key="bull_regime_threshold_remediation_factor_dir",
        )
        symbols_text = st.text_input(
            "Symbols",
            value="000001,600519,000858,600036,601318",
            key="bull_regime_threshold_remediation_symbols",
        )
        recommendations = st.text_input(
            "Feature pruning recommendations",
            value="outputs/feature_ablation_real_v1/feature_pruning_recommendations.csv",
            key="bull_regime_threshold_remediation_recommendations",
        )
        failure_analysis_dir = st.text_input(
            "Failure analysis directory",
            value="outputs/validation_gate_failure_analysis_real_v1",
            key="bull_regime_threshold_remediation_failure_analysis_dir",
        )
        targeted_design_dir = st.text_input(
            "Targeted design directory",
            value="outputs/targeted_remediation_design_real_v1",
            key="bull_regime_threshold_remediation_targeted_design_dir",
        )
        threshold_cols = st.columns(2)
        buy_thresholds_text = threshold_cols[0].text_input(
            "Buy thresholds",
            value=",".join(f"{value:.2f}" for value in DEFAULT_BULL_REMEDIATION_BUYS),
            key="bull_regime_threshold_remediation_buy_thresholds",
        )
        sell_thresholds_text = threshold_cols[1].text_input(
            "Sell thresholds",
            value=",".join(f"{value:.2f}" for value in DEFAULT_BULL_REMEDIATION_SELLS),
            key="bull_regime_threshold_remediation_sell_thresholds",
        )
        if st.button(
            "Run bull remediation",
            key="run_bull_regime_threshold_remediation_button",
            type="primary",
        ):
            try:
                result = save_bull_regime_threshold_remediation(
                    factor_dir=factor_dir,
                    symbols=parse_bull_remediation_symbols(symbols_text),
                    recommendations_path=recommendations,
                    output_dir=output_dir,
                    failure_analysis_dir=failure_analysis_dir,
                    targeted_design_dir=targeted_design_dir,
                    buy_thresholds=parse_bull_remediation_thresholds(
                        buy_thresholds_text,
                        DEFAULT_BULL_REMEDIATION_BUYS,
                    ),
                    sell_thresholds=parse_bull_remediation_thresholds(
                        sell_thresholds_text,
                        DEFAULT_BULL_REMEDIATION_SELLS,
                    ),
                )
                output = {
                    "results": result["bull_threshold_results"],
                    "summary": result["bull_threshold_summary"],
                    "per_symbol": result["per_symbol_bull_results"],
                    "best": result["best_bull_thresholds"],
                    "warnings": result["warnings"],
                    "markdown_report": result["bull_remediation_report"],
                }
            except Exception as exc:
                st.error(f"Bull remediation failed: {exc}")
                return
        else:
            return
    else:
        if not st.button(
            "Load bull remediation",
            key="load_bull_regime_threshold_remediation_button",
        ):
            return
        output = load_bull_regime_threshold_remediation_outputs(output_dir)

    st.subheader("Best Bull Thresholds")
    st.dataframe(output["best"], width="stretch")
    st.subheader("Bull Threshold Summary")
    st.dataframe(output["summary"], width="stretch")
    st.subheader("Per-Symbol Bull Results")
    st.dataframe(output["per_symbol"], width="stretch")
    st.subheader("Warnings")
    st.dataframe(output["warnings"], width="stretch")
    st.subheader("Bull Threshold Results")
    st.dataframe(output["results"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download bull remediation report",
            data=output["markdown_report"],
            file_name="bull_remediation_report.md",
            mime="text/markdown",
            key="download_bull_regime_threshold_remediation_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


def render_sideways_regime_trade_sufficiency_remediation_tab() -> None:
    st.write(
        "Run a sideways-regime-only trade sufficiency diagnostic for "
        "canonical_reduced_40 + logistic_regression."
    )
    st.warning(
        "This is educational research diagnostics only. It is not trading-ready "
        "and does not add data sources, agents, models, or strategy logic."
    )

    mode = st.radio(
        "Sideways remediation action",
        ["Load output", "Run experiment"],
        horizontal=True,
        key="sideways_regime_trade_sufficiency_remediation_action",
    )
    output_dir = st.text_input(
        "Sideways remediation output directory",
        value="outputs/sideways_regime_trade_sufficiency_remediation_real_v1",
        key="sideways_regime_trade_sufficiency_remediation_output_dir",
    )
    if mode == "Run experiment":
        factor_dir = st.text_input(
            "Factor directory",
            value="outputs/model_robustness_real_v2/factors",
            key="sideways_regime_trade_sufficiency_remediation_factor_dir",
        )
        symbols_text = st.text_input(
            "Symbols",
            value="000001,600519,000858,600036,601318",
            key="sideways_regime_trade_sufficiency_remediation_symbols",
        )
        recommendations = st.text_input(
            "Feature pruning recommendations",
            value="outputs/feature_ablation_real_v1/feature_pruning_recommendations.csv",
            key="sideways_regime_trade_sufficiency_remediation_recommendations",
        )
        failure_analysis_dir = st.text_input(
            "Failure analysis directory",
            value="outputs/validation_gate_failure_analysis_real_v1",
            key="sideways_regime_trade_sufficiency_remediation_failure_analysis_dir",
        )
        targeted_design_dir = st.text_input(
            "Targeted design directory",
            value="outputs/targeted_remediation_design_real_v1",
            key="sideways_regime_trade_sufficiency_remediation_targeted_design_dir",
        )
        threshold_cols = st.columns(2)
        buy_thresholds_text = threshold_cols[0].text_input(
            "Buy thresholds",
            value=",".join(
                f"{value:.2f}" for value in DEFAULT_SIDEWAYS_REMEDIATION_BUYS
            ),
            key="sideways_regime_trade_sufficiency_remediation_buy_thresholds",
        )
        sell_thresholds_text = threshold_cols[1].text_input(
            "Sell thresholds",
            value=",".join(
                f"{value:.2f}" for value in DEFAULT_SIDEWAYS_REMEDIATION_SELLS
            ),
            key="sideways_regime_trade_sufficiency_remediation_sell_thresholds",
        )
        if st.button(
            "Run sideways remediation",
            key="run_sideways_regime_trade_sufficiency_remediation_button",
            type="primary",
        ):
            try:
                result = save_sideways_regime_trade_sufficiency_remediation(
                    factor_dir=factor_dir,
                    symbols=parse_sideways_remediation_symbols(symbols_text),
                    recommendations_path=recommendations,
                    output_dir=output_dir,
                    failure_analysis_dir=failure_analysis_dir,
                    targeted_design_dir=targeted_design_dir,
                    buy_thresholds=parse_sideways_remediation_thresholds(
                        buy_thresholds_text,
                        DEFAULT_SIDEWAYS_REMEDIATION_BUYS,
                    ),
                    sell_thresholds=parse_sideways_remediation_thresholds(
                        sell_thresholds_text,
                        DEFAULT_SIDEWAYS_REMEDIATION_SELLS,
                    ),
                )
                output = {
                    "results": result["sideways_trade_results"],
                    "summary": result["sideways_trade_summary"],
                    "per_symbol": result["per_symbol_sideways_results"],
                    "best": result["best_sideways_thresholds"],
                    "warnings": result["warnings"],
                    "markdown_report": result["sideways_remediation_report"],
                }
            except Exception as exc:
                st.error(f"Sideways remediation failed: {exc}")
                return
        else:
            return
    else:
        if not st.button(
            "Load sideways remediation",
            key="load_sideways_regime_trade_sufficiency_remediation_button",
        ):
            return
        output = load_sideways_regime_trade_sufficiency_remediation_outputs(output_dir)

    st.subheader("Best Sideways Thresholds")
    st.dataframe(output["best"], width="stretch")
    st.subheader("Sideways Trade Summary")
    st.dataframe(output["summary"], width="stretch")
    st.subheader("Per-Symbol Sideways Results")
    st.dataframe(output["per_symbol"], width="stretch")
    st.subheader("Warnings")
    st.dataframe(output["warnings"], width="stretch")
    st.subheader("Sideways Trade Results")
    st.dataframe(output["results"], width="stretch")
    st.subheader("Markdown Report")
    if output["markdown_report"]:
        st.markdown(output["markdown_report"])
        st.download_button(
            "Download sideways remediation report",
            data=output["markdown_report"],
            file_name="sideways_remediation_report.md",
            mime="text/markdown",
            key="download_sideways_regime_trade_sufficiency_remediation_report_button",
        )
    else:
        st.info("No Markdown report text is available.")


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
    execution_mode = st.sidebar.selectbox(
        "Execution mode",
        ["same_close", "next_open", "next_close"],
    )
    commission_rate = st.sidebar.number_input(
        "Commission rate",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    stamp_tax_rate = st.sidebar.number_input(
        "Stamp tax rate",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    slippage_pct = st.sidebar.number_input(
        "Slippage %",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f",
    )
    min_commission = st.sidebar.number_input(
        "Minimum commission",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    (
        single_tab,
        experiment_tab,
        period_tab,
        model_tab,
        evaluation_tab,
        ml_signal_tab,
        threshold_tab,
        robustness_tab,
        feature_sources_tab,
        feature_queue_tab,
        factor_ablation_tab,
        factor_decisions_tab,
        factor_pruning_tab,
        pruning_summary_tab,
        reduced_feature_backtest_tab,
        reduced_feature_backtest_summary_tab,
        threshold_sensitivity_tab,
        threshold_decision_report_tab,
        candidate_validation_tab,
        candidate_stress_test_tab,
        candidate_equivalence_audit_tab,
        candidate_mode_normalization_tab,
        canonical_revalidation_tab,
        candidate_validation_gate_tab,
        validation_gate_failure_analysis_tab,
        targeted_remediation_design_tab,
        bull_regime_threshold_remediation_tab,
        sideways_regime_trade_sufficiency_remediation_tab,
    ) = st.tabs(
        [
            "Single Backtest",
            "Parameter Experiment",
            "Period Experiment",
            "Model Prediction",
            "Model Evaluation",
            "ML Signal Backtest",
            "ML Threshold Experiment",
            "Model Robustness",
            "Feature Sources",
            "Feature Queue",
            "Factor Ablation",
            "Factor Decisions",
            "Factor Pruning",
            "Pruning Summary",
            "Reduced Feature Backtest",
            "Reduced Feature Backtest Summary",
            "Threshold Sensitivity",
            "Threshold Decision Report",
            "Candidate Validation",
            "Candidate Stress Test",
            "Candidate Equivalence Audit",
            "Candidate Mode Normalization",
            "Canonical Revalidation",
            "Candidate Validation Gate",
            "Gate Failure Analysis",
            "Targeted Remediation Design",
            "Bull Regime Remediation",
            "Sideways Regime Remediation",
        ]
    )
    with single_tab:
        render_single_backtest_tab(
            initial_cash=initial_cash,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_holding_days=max_holding_days,
            execution_mode=execution_mode,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
        )

    with experiment_tab:
        render_parameter_experiment_tab()

    with period_tab:
        render_period_experiment_tab()

    with model_tab:
        render_model_prediction_tab()

    with evaluation_tab:
        render_model_evaluation_tab()

    with ml_signal_tab:
        render_ml_signal_backtest_tab()

    with threshold_tab:
        render_ml_threshold_experiment_tab()

    with robustness_tab:
        render_model_robustness_tab()

    with feature_sources_tab:
        render_feature_sources_tab()

    with feature_queue_tab:
        render_feature_queue_tab()

    with factor_ablation_tab:
        render_factor_ablation_tab()

    with factor_decisions_tab:
        render_factor_decisions_tab()

    with factor_pruning_tab:
        render_factor_pruning_tab()

    with pruning_summary_tab:
        render_pruning_summary_tab()

    with reduced_feature_backtest_tab:
        render_reduced_feature_backtest_tab()

    with reduced_feature_backtest_summary_tab:
        render_reduced_feature_backtest_summary_tab()

    with threshold_sensitivity_tab:
        render_threshold_sensitivity_tab()

    with threshold_decision_report_tab:
        render_threshold_decision_report_tab()

    with candidate_validation_tab:
        render_candidate_validation_tab()

    with candidate_stress_test_tab:
        render_candidate_stress_test_tab()

    with candidate_equivalence_audit_tab:
        render_candidate_equivalence_audit_tab()

    with candidate_mode_normalization_tab:
        render_candidate_mode_normalization_tab()

    with canonical_revalidation_tab:
        render_canonical_revalidation_tab()

    with candidate_validation_gate_tab:
        render_candidate_validation_gate_tab()

    with validation_gate_failure_analysis_tab:
        render_validation_gate_failure_analysis_tab()

    with targeted_remediation_design_tab:
        render_targeted_remediation_design_tab()

    with bull_regime_threshold_remediation_tab:
        render_bull_regime_threshold_remediation_tab()

    with sideways_regime_trade_sufficiency_remediation_tab:
        render_sideways_regime_trade_sufficiency_remediation_tab()


if __name__ == "__main__":
    main()
