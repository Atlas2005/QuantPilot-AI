import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("outputs/daily_trading_plan_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/paper_trading_ledger_real_v1")
DEFAULT_STARTING_CASH = 1000.0

OUTPUT_FILENAMES = {
    "orders": "paper_orders.csv",
    "fills": "paper_fills.csv",
    "positions": "paper_positions.csv",
    "cash": "paper_cash_ledger.csv",
    "ledger": "paper_trade_ledger.csv",
    "summary": "paper_trading_summary.csv",
    "guardrails": "paper_trading_guardrails.csv",
    "report": "paper_trading_report.md",
    "run_config": "run_config.json",
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _format_symbol(value: Any) -> str:
    text = _clean_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _numeric(value: Any) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"symbol": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "symbol" in df:
        df["symbol"] = df["symbol"].map(_format_symbol)
    if "trading_ready" in df:
        df["trading_ready"] = False
    return df.reset_index(drop=True)


def _first_number(df: pd.DataFrame, column: str, fallback: float) -> float:
    if column not in df or df.empty:
        return fallback
    value = _numeric(df[column].iloc[0])
    return float(value) if pd.notna(value) else fallback


def _exit_rules_by_symbol(plan: pd.DataFrame) -> dict[str, dict[str, Any]]:
    exits = plan[plan.get("plan_section", pd.Series(dtype=str)) == "exit_plan"].copy() if not plan.empty else pd.DataFrame()
    rules: dict[str, dict[str, Any]] = {}
    for _, row in exits.iterrows():
        symbol = _format_symbol(row.get("symbol"))
        if not symbol:
            continue
        rules[symbol] = {
            "stop_loss_price": _numeric(row.get("stop_loss_price")),
            "take_profit_price": _numeric(row.get("take_profit_price")),
            "max_holding_days": _numeric(row.get("max_holding_days")),
            "benchmark_lag_exit_rule": _clean_text(row.get("benchmark_lag_exit_rule")),
        }
    return rules


def build_paper_orders(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = plan[plan.get("plan_section", pd.Series(dtype=str)).isin(["sized_position", "deferred_position"])].copy() if not plan.empty else pd.DataFrame()
    for idx, row in eligible.reset_index(drop=True).iterrows():
        plan_section = _clean_text(row.get("plan_section"))
        quantity = _numeric(row.get("quantity"))
        entry_price = _numeric(row.get("entry_price"))
        approved_notional = _numeric(row.get("approved_notional"))
        order_status = "paper_filled" if plan_section == "sized_position" else "deferred_not_filled"
        rows.append(
            {
                "paper_order_id": f"PAPER-ORDER-{idx + 1:03d}",
                "source_plan_section": plan_section,
                "symbol": _format_symbol(row.get("symbol")),
                "candidate_id": _clean_text(row.get("candidate_id")),
                "side": "BUY",
                "order_type": "paper_market_simulation",
                "entry_price": entry_price,
                "order_quantity": int(quantity) if pd.notna(quantity) else 0,
                "approved_notional": approved_notional if pd.notna(approved_notional) else 0.0,
                "order_status": order_status,
                "trading_ready": False,
                "notes": "Paper order only. No broker execution, live trading, or real order submission.",
            }
        )
    return pd.DataFrame(rows)


def build_paper_fills(orders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in orders.iterrows():
        filled = _clean_text(row.get("order_status")) == "paper_filled"
        quantity = int(_numeric(row.get("order_quantity"))) if filled and pd.notna(_numeric(row.get("order_quantity"))) else 0
        price = _numeric(row.get("entry_price"))
        fill_price = float(price) if filled and pd.notna(price) else float("nan")
        fill_notional = float(quantity) * fill_price if filled and pd.notna(fill_price) else 0.0
        rows.append(
            {
                "paper_fill_id": str(row.get("paper_order_id")).replace("ORDER", "FILL"),
                "paper_order_id": row.get("paper_order_id"),
                "symbol": _format_symbol(row.get("symbol")),
                "side": row.get("side"),
                "fill_price": fill_price,
                "fill_quantity": quantity,
                "fill_notional": fill_notional,
                "fill_status": "simulated_filled" if filled else "not_filled",
                "trading_ready": False,
                "notes": "Deterministic paper fill only. No real execution occurred.",
            }
        )
    return pd.DataFrame(rows)


def build_paper_positions(fills: pd.DataFrame, exit_rules: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    filled = fills[fills.get("fill_status", pd.Series(dtype=str)) == "simulated_filled"].copy() if not fills.empty else pd.DataFrame()
    for _, row in filled.iterrows():
        symbol = _format_symbol(row.get("symbol"))
        rules = exit_rules.get(symbol, {})
        quantity = int(_numeric(row.get("fill_quantity"))) if pd.notna(_numeric(row.get("fill_quantity"))) else 0
        entry_price = _numeric(row.get("fill_price"))
        approved_notional = _numeric(row.get("fill_notional"))
        max_holding_days = _numeric(rules.get("max_holding_days"))
        rows.append(
            {
                "symbol": symbol,
                "position_status": "open_paper_position",
                "entry_price": entry_price,
                "quantity": quantity,
                "approved_notional": approved_notional,
                "stop_loss_price": rules.get("stop_loss_price", float("nan")),
                "take_profit_price": rules.get("take_profit_price", float("nan")),
                "max_holding_days": int(max_holding_days) if pd.notna(max_holding_days) else "",
                "benchmark_lag_exit_rule": rules.get("benchmark_lag_exit_rule", ""),
                "unrealized_pnl": 0.0,
                "trading_ready": False,
                "notes": "Open paper position only. Unrealized PnL is zero because no future market data is fetched.",
            }
        )
    return pd.DataFrame(rows)


def build_paper_cash_ledger(summary: pd.DataFrame, fills: pd.DataFrame, starting_cash_override: float | None) -> pd.DataFrame:
    fallback_cash = float(starting_cash_override) if starting_cash_override is not None else DEFAULT_STARTING_CASH
    starting_cash = _first_number(summary, "available_cash", fallback_cash)
    usable_cash = _first_number(summary, "usable_cash", float("nan"))
    usable_cash_buffer = usable_cash / starting_cash if pd.notna(usable_cash) and starting_cash else float("nan")
    total_filled_notional = (
        float(pd.to_numeric(fills.get("fill_notional", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if not fills.empty
        else 0.0
    )
    ending_cash = starting_cash - total_filled_notional
    return pd.DataFrame(
        [
            {
                "ledger_item": "starting_cash",
                "cash_amount": starting_cash,
                "cash_debit": 0.0,
                "cash_credit": 0.0,
                "ending_cash": starting_cash,
                "usable_cash": usable_cash,
                "usable_cash_buffer": usable_cash_buffer,
                "trading_ready": False,
                "notes": "Starting cash sourced from Step 5 summary when available.",
            },
            {
                "ledger_item": "simulated_buy_fills",
                "cash_amount": -total_filled_notional,
                "cash_debit": total_filled_notional,
                "cash_credit": 0.0,
                "ending_cash": ending_cash,
                "usable_cash": usable_cash,
                "usable_cash_buffer": usable_cash_buffer,
                "trading_ready": False,
                "notes": "Cash debit is from deterministic paper fills only. No real cash movement occurred.",
            },
        ]
    )


def build_paper_trade_ledger(
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    positions: pd.DataFrame,
    cash_ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    fills_by_order = {row.get("paper_order_id"): row for _, row in fills.iterrows()} if not fills.empty else {}
    positions_by_symbol = {row.get("symbol"): row for _, row in positions.iterrows()} if not positions.empty else {}
    for _, order in orders.iterrows():
        order_id = order.get("paper_order_id")
        symbol = _format_symbol(order.get("symbol"))
        fill = fills_by_order.get(order_id, pd.Series(dtype=object))
        position = positions_by_symbol.get(symbol, pd.Series(dtype=object))
        rows.append(
            {
                "ledger_event": "paper_order_review",
                "paper_order_id": order_id,
                "symbol": symbol,
                "order_status": order.get("order_status"),
                "fill_status": fill.get("fill_status", ""),
                "fill_quantity": fill.get("fill_quantity", 0),
                "fill_notional": fill.get("fill_notional", 0.0),
                "cash_effect": -float(fill.get("fill_notional", 0.0) or 0.0),
                "position_status": position.get("position_status", "no_open_position"),
                "trading_ready": False,
                "notes": "Human-readable paper ledger row. No broker execution or real order submission.",
            }
        )
    for _, cash in cash_ledger.iterrows():
        rows.append(
            {
                "ledger_event": cash.get("ledger_item"),
                "paper_order_id": "",
                "symbol": "",
                "order_status": "",
                "fill_status": "",
                "fill_quantity": "",
                "fill_notional": "",
                "cash_effect": cash.get("cash_amount"),
                "position_status": "",
                "trading_ready": False,
                "notes": cash.get("notes"),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    positions: pd.DataFrame,
    cash_ledger: pd.DataFrame,
) -> pd.DataFrame:
    filled_count = int((orders["order_status"] == "paper_filled").sum()) if not orders.empty else 0
    deferred_count = int((orders["order_status"] == "deferred_not_filled").sum()) if not orders.empty else 0
    total_filled_notional = (
        float(pd.to_numeric(fills.get("fill_notional", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if not fills.empty
        else 0.0
    )
    starting_cash = (
        float(cash_ledger.loc[cash_ledger["ledger_item"] == "starting_cash", "ending_cash"].iloc[0])
        if not cash_ledger.empty and (cash_ledger["ledger_item"] == "starting_cash").any()
        else DEFAULT_STARTING_CASH
    )
    ending_cash = (
        float(cash_ledger["ending_cash"].iloc[-1])
        if not cash_ledger.empty and "ending_cash" in cash_ledger
        else starting_cash - total_filled_notional
    )
    return pd.DataFrame(
        [
            {
                "summary_item": "paper_trading_ledger_run",
                "paper_order_count": int(len(orders)),
                "paper_filled_order_count": filled_count,
                "paper_deferred_order_count": deferred_count,
                "open_paper_position_count": int(len(positions)),
                "starting_cash": starting_cash,
                "ending_cash": ending_cash,
                "total_filled_notional": total_filled_notional,
                "trading_ready": False,
                "conclusion": "research_only_paper_ledger_created",
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The ledger reads Step 5 plan CSVs and performs deterministic paper accounting only.", "No historical backtest is run."),
        ("no_threshold_change", "confirmed", "No signal threshold module or value is changed.", "Signal thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local Step 5 CSV outputs are read.", "No market data source is added and no live data is fetched."),
        ("no_broker_integration", "confirmed", "No broker API, account connection, or order route is used.", "No execution path is created."),
        ("no_live_trading", "confirmed", "Outputs are CSV/Markdown paper ledger reports only.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "Orders and fills are simulated records only.", "No real orders are generated or submitted."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False is written to Step 6 outputs.", "No deployable status is claimed."),
        ("paper_ledger_only", "confirmed", "The engine creates deterministic paper accounting rows from the daily plan.", "No strategy, threshold, model, or broker behavior is changed."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    positions: pd.DataFrame,
    cash_ledger: pd.DataFrame,
    trade_ledger: pd.DataFrame,
    summary: pd.DataFrame,
    guardrails: pd.DataFrame,
    input_dir: str | Path,
    output_dir: str | Path,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V5 Step 6 Paper Trading Ledger Report",
            "",
            "## Executive Summary",
            "V5 Step 6 creates a deterministic paper trading ledger from the V5 Step 5 daily trading plan.",
            "This is paper trading only.",
            "This is educational/research tooling only.",
            "This is not financial advice.",
            "No broker execution occurred.",
            "No real orders were submitted.",
            "No live market data was fetched.",
            "No strategy performance claim is made.",
            "All outputs remain trading_ready=False.",
            "The project remains not trading-ready.",
            "Recommended next step: V5 Step 7 Semi-Auto Order Generator.",
            "",
            "## Inputs",
            f"- Input directory: {input_dir}",
            f"- Output directory: {output_dir}",
            "",
            "## Paper Trading Summary",
            f"- Paper orders: {row.get('paper_order_count', 0)}",
            f"- Paper filled orders: {row.get('paper_filled_order_count', 0)}",
            f"- Paper deferred orders: {row.get('paper_deferred_order_count', 0)}",
            f"- Open paper positions: {row.get('open_paper_position_count', 0)}",
            f"- Starting cash: {row.get('starting_cash', '')}",
            f"- Ending cash: {row.get('ending_cash', '')}",
            f"- Total filled notional: {row.get('total_filled_notional', 0)}",
            "",
            "## Paper Orders",
            _table(orders, "No paper orders were generated."),
            "",
            "## Paper Fills",
            _table(fills, "No paper fills were generated."),
            "",
            "## Paper Positions",
            _table(positions, "No open paper positions were generated."),
            "",
            "## Paper Cash Ledger",
            _table(cash_ledger, "No paper cash ledger rows were generated."),
            "",
            "## Paper Trade Ledger",
            _table(trade_ledger, "No paper trade ledger rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Warnings",
            "- Paper trading only.",
            "- No broker execution occurred.",
            "- No real orders were submitted.",
            "- No live market data was fetched.",
            "- No strategy performance claim is made.",
            "- All outputs remain trading_ready=False.",
            "",
        ]
    )


def generate_paper_trading_ledger_outputs(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    starting_cash: float | None = None,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    plan = _read_csv(input_path / "daily_trading_plan.csv")
    daily_summary = _read_csv(input_path / "daily_trading_plan_summary.csv")
    daily_guardrails = _read_csv(input_path / "daily_trading_plan_guardrails.csv")

    exit_rules = _exit_rules_by_symbol(plan)
    orders = build_paper_orders(plan)
    fills = build_paper_fills(orders)
    positions = build_paper_positions(fills, exit_rules)
    cash_ledger = build_paper_cash_ledger(daily_summary, fills, starting_cash)
    trade_ledger = build_paper_trade_ledger(orders, fills, positions, cash_ledger)
    summary = build_summary(orders, fills, positions, cash_ledger)
    guardrails = build_guardrails()

    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    report = build_report(orders, fills, positions, cash_ledger, trade_ledger, summary, guardrails, input_path, output_path)

    orders.to_csv(paths["orders"], index=False)
    fills.to_csv(paths["fills"], index=False)
    positions.to_csv(paths["positions"], index=False)
    cash_ledger.to_csv(paths["cash"], index=False)
    trade_ledger.to_csv(paths["ledger"], index=False)
    summary.to_csv(paths["summary"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    paths["report"].write_text(report, encoding="utf-8")
    config = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "starting_cash_arg": starting_cash,
        "daily_plan_rows": int(len(plan)),
        "daily_summary_rows": int(len(daily_summary)),
        "daily_guardrail_rows": int(len(daily_guardrails)),
        "paper_order_count": int(len(orders)),
        "paper_filled_order_count": int((orders["order_status"] == "paper_filled").sum()) if not orders.empty else 0,
        "paper_deferred_order_count": int((orders["order_status"] == "deferred_not_filled").sum()) if not orders.empty else 0,
        "open_paper_position_count": int(len(positions)),
        "scope": "V5 Step 6 paper ledger only",
        "broker_execution": False,
        "live_trading": False,
        "order_execution": False,
        "live_market_data_fetched": False,
        "strategy_performance_claim": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "paper_orders": orders,
        "paper_fills": fills,
        "paper_positions": positions,
        "paper_cash_ledger": cash_ledger,
        "paper_trade_ledger": trade_ledger,
        "paper_trading_summary": summary,
        "paper_trading_guardrails": guardrails,
        "paper_trading_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
