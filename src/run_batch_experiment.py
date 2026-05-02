import argparse
import sys
import time

import pandas as pd

from backtester import run_long_only_backtest_with_trades
from indicators import add_all_indicators
from metrics import summarize_performance
from real_data_loader import fetch_a_share_daily_from_source
from strategy import generate_ma_crossover_signals
from trade_metrics import summarize_trade_metrics


DEFAULT_SYMBOLS = "000001,600519,000858,600036,601318"

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
    "total_commission",
    "total_stamp_tax",
    "total_slippage_cost",
    "total_transaction_cost",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare risk-control settings across multiple A-share stocks."
    )
    parser.add_argument(
        "--symbols",
        default=DEFAULT_SYMBOLS,
        help="Comma-separated A-share symbols, for example 000001,600519,000858.",
    )
    parser.add_argument(
        "--source",
        choices=["akshare", "baostock", "auto"],
        default="baostock",
        help="Data source to use: akshare, baostock, or auto.",
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
        help="Price adjustment mode, for example qfq.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10000.0,
        help="Starting cash for each backtest scenario.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for saving raw batch results.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print a compact result table instead of the full wide table.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["same_close", "next_open", "next_close"],
        default="same_close",
        help="Trade execution mode. Defaults to same_close for old behavior.",
    )
    parser.add_argument("--commission-rate", type=float, default=0.0)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0)
    parser.add_argument("--min-commission", type=float, default=0.0)
    return parser.parse_args()


def parse_symbols(symbols_text: str) -> list[str]:
    symbols = [symbol.strip() for symbol in symbols_text.split(",")]
    return [
        symbol.zfill(6) if symbol.isdigit() and len(symbol) < 6 else symbol
        for symbol in symbols
        if symbol
    ]


def make_error_row(symbol: str, error: str) -> dict:
    row = {column: None for column in RESULT_COLUMNS}
    row["symbol"] = symbol
    row["scenario"] = "ERROR"
    row["error"] = error
    return row


def run_scenario(
    symbol: str,
    prepared_data: pd.DataFrame,
    initial_cash: float,
    scenario: dict,
    args: argparse.Namespace,
) -> dict:
    backtest_df, trades_df = run_long_only_backtest_with_trades(
        prepared_data.copy(),
        initial_cash=initial_cash,
        stop_loss_pct=scenario["stop_loss_pct"],
        take_profit_pct=scenario["take_profit_pct"],
        max_holding_days=scenario["max_holding_days"],
        execution_mode=args.execution_mode,
        commission_rate=args.commission_rate,
        stamp_tax_rate=args.stamp_tax_rate,
        slippage_pct=args.slippage_pct,
        min_commission=args.min_commission,
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
        "total_commission": performance["total_commission"],
        "total_stamp_tax": performance["total_stamp_tax"],
        "total_slippage_cost": performance["total_slippage_cost"],
        "total_transaction_cost": performance["total_transaction_cost"],
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


def run_symbol(args: argparse.Namespace, symbol: str) -> list[dict]:
    print(f"Fetching data once for symbol {symbol}...")
    stock_data = fetch_a_share_daily_from_source(
        symbol=symbol,
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
        source=args.source,
    )

    print(f"Fetched {len(stock_data)} rows for {symbol}.")
    print(f"Preparing indicators and strategy signals once for {symbol}.")
    prepared_data = add_all_indicators(stock_data)
    prepared_data = generate_ma_crossover_signals(prepared_data)

    return [
        run_scenario(symbol, prepared_data, args.initial_cash, scenario, args)
        for scenario in SCENARIOS
    ]


def build_aggregate_summary(results_df: pd.DataFrame) -> pd.DataFrame:
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

    summary_df = (
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

    return summary_df


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


def format_pct(value) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}%"


def format_number(value) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def format_optional_pct(value) -> str:
    return "disabled" if pd.isna(value) else f"{value:.2f}%"


def format_optional_int(value) -> str:
    return "disabled" if pd.isna(value) else str(int(value))


def format_count(value) -> str:
    return "N/A" if pd.isna(value) else str(int(value))


def print_batch_results(results_df: pd.DataFrame) -> None:
    print("Batch Parameter Experiment Results")
    print("----------------------------------")

    display_df = results_df.copy()
    for column in ["stop_loss_pct", "take_profit_pct"]:
        display_df[column] = display_df[column].apply(format_optional_pct)

    display_df["max_holding_days"] = display_df["max_holding_days"].apply(
        format_optional_int
    )

    for column in [
        "total_return_pct",
        "max_drawdown_pct",
        "win_rate_pct",
        "average_return_pct",
        "best_trade_return_pct",
        "worst_trade_return_pct",
    ]:
        display_df[column] = display_df[column].apply(format_pct)

    for column in [
        "final_value",
        "profit_factor",
        "average_holding_days",
        "total_commission",
        "total_stamp_tax",
        "total_slippage_cost",
        "total_transaction_cost",
    ]:
        display_df[column] = display_df[column].apply(format_number)

    for column in ["total_trades", "closed_trades", "open_trades"]:
        display_df[column] = display_df[column].apply(format_count)

    display_df["error"] = display_df["error"].fillna("")
    display_df["currently_holding"] = display_df["currently_holding"].fillna("")

    print(display_df.to_string(index=False))


def print_compact_batch_results(results_df: pd.DataFrame) -> None:
    print("Compact Batch Results")
    print("---------------------")

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
        display_df[column] = display_df[column].apply(format_pct)

    for column in ["profit_factor", "final_value"]:
        display_df[column] = display_df[column].apply(format_number)

    display_df["currently_holding"] = display_df["currently_holding"].fillna("")
    display_df["error"] = display_df["error"].fillna("")

    print(display_df.to_string(index=False))


def print_aggregate_summary(summary_df: pd.DataFrame) -> None:
    print()
    print("Scenario Average Summary")
    print("------------------------")

    if summary_df.empty:
        print("No successful scenario results to summarize.")
        return

    display_df = summary_df.copy()
    for column in [
        "avg_total_return_pct",
        "avg_max_drawdown_pct",
        "avg_win_rate_pct",
    ]:
        display_df[column] = display_df[column].apply(format_pct)

    for column in ["avg_profit_factor", "avg_average_holding_days"]:
        display_df[column] = display_df[column].apply(format_number)

    print(display_df.to_string(index=False))


def print_scenario_ranking(ranking_df: pd.DataFrame) -> None:
    print()
    print("Scenario Ranking")
    print("----------------")

    if ranking_df.empty:
        print("No successful scenario results to rank.")
        return

    display_df = ranking_df.copy()
    for column in [
        "avg_total_return_pct",
        "avg_max_drawdown_pct",
        "avg_win_rate_pct",
    ]:
        display_df[column] = display_df[column].apply(format_pct)

    for column in ["avg_profit_factor", "avg_average_holding_days", "score"]:
        display_df[column] = display_df[column].apply(format_number)

    print(display_df.to_string(index=False))


def print_quick_interpretation(ranking_df: pd.DataFrame) -> None:
    print()
    print("Quick Interpretation")
    print("--------------------")

    if ranking_df.empty:
        print("No successful scenario results are available for interpretation.")
        return

    best_score = ranking_df.iloc[0]
    best_return = ranking_df.loc[ranking_df["avg_total_return_pct"].idxmax()]
    best_drawdown = ranking_df.loc[ranking_df["avg_max_drawdown_pct"].idxmax()]
    best_profit_factor = ranking_df.loc[ranking_df["avg_profit_factor"].idxmax()]

    print(
        "Best overall scenario by score: "
        f"{best_score['scenario']} (score {best_score['score']:.2f})"
    )
    print(
        "Best average return scenario: "
        f"{best_return['scenario']} "
        f"({best_return['avg_total_return_pct']:.2f}%)"
    )
    print(
        "Best drawdown control scenario: "
        f"{best_drawdown['scenario']} "
        f"({best_drawdown['avg_max_drawdown_pct']:.2f}%)"
    )
    print(
        "Best profit factor scenario: "
        f"{best_profit_factor['scenario']} "
        f"({best_profit_factor['avg_profit_factor']:.2f})"
    )


def main() -> None:
    args = parse_args()
    symbols = parse_symbols(args.symbols)

    if not symbols:
        print("Error: please provide at least one symbol.")
        sys.exit(1)

    print("QuantPilot-AI Batch Parameter Experiment")
    print("----------------------------------------")
    print(f"Selected symbols: {', '.join(symbols)}")
    print(f"Selected source: {args.source}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Adjust mode: {args.adjust}")
    print(f"Initial cash: {args.initial_cash:.2f}")
    print()

    rows = []
    for index, symbol in enumerate(symbols):
        try:
            rows.extend(run_symbol(args, symbol))
        except Exception as exc:
            print(f"Error: failed to process {symbol}: {exc}")
            rows.append(make_error_row(symbol, str(exc)))

        if index < len(symbols) - 1:
            print("Waiting 1 second before the next symbol...")
            print()
            time.sleep(1)

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary_df = build_aggregate_summary(results_df)
    ranking_df = build_scenario_ranking(summary_df)

    print()
    if args.compact:
        print_compact_batch_results(results_df)
    else:
        print_batch_results(results_df)
    print_aggregate_summary(summary_df)
    print_scenario_ranking(ranking_df)
    print_quick_interpretation(ranking_df)

    print()
    print("Note: This report is generated from simple rules for educational research only.")
    print("It is not financial advice.")

    if args.output:
        results_df.to_csv(args.output, index=False)
        print()
        print(f"Saved raw batch results to {args.output}")


if __name__ == "__main__":
    main()
