import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DAILY_PLAN_PATH = Path("outputs/daily_trading_plan_real_v1/daily_trading_plan.csv")
DEFAULT_EXIT_PLAN_PATH = Path("outputs/exit_engine_real_v1/exit_plan.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/semi_auto_order_generator_real_v1")

OUTPUT_FILENAMES = {
    "drafts": "order_drafts.csv",
    "tickets": "broker_neutral_order_tickets.md",
    "checklist": "manual_review_checklist.csv",
    "summary": "semi_auto_order_summary.csv",
    "guardrails": "semi_auto_order_guardrails.csv",
    "run_config": "run_config.json",
}

REVIEW_CHECKS = [
    "confirm_symbol",
    "confirm_side",
    "confirm_quantity",
    "confirm_limit_price",
    "confirm_cash_available",
    "confirm_stop_loss",
    "confirm_take_profit",
    "confirm_no_broker_execution",
    "confirm_human_review_required",
]


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


def _exit_rules_by_symbol(daily_plan: pd.DataFrame, exit_plan: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    frames = []
    if not daily_plan.empty and "plan_section" in daily_plan:
        frames.append(daily_plan[daily_plan["plan_section"] == "exit_plan"].copy())
    if not exit_plan.empty:
        frames.append(exit_plan.copy())
    for frame in frames:
        for _, row in frame.iterrows():
            symbol = _format_symbol(row.get("symbol"))
            if not symbol:
                continue
            rules[symbol] = {
                "stop_loss_price": _numeric(row.get("stop_loss_price")),
                "take_profit_price": _numeric(row.get("take_profit_price")),
                "max_holding_days": _numeric(row.get("max_holding_days")),
            }
    return rules


def build_order_drafts(daily_plan: pd.DataFrame, exit_plan: pd.DataFrame) -> pd.DataFrame:
    rules_by_symbol = _exit_rules_by_symbol(daily_plan, exit_plan)
    sized = (
        daily_plan[
            (daily_plan.get("plan_section", pd.Series(dtype=str)) == "sized_position")
            & (daily_plan.get("side", pd.Series(dtype=str)).astype(str).str.upper() == "BUY")
        ].copy()
        if not daily_plan.empty
        else pd.DataFrame()
    )
    rows = []
    for idx, row in sized.reset_index(drop=True).iterrows():
        symbol = _format_symbol(row.get("symbol"))
        limit_price = _numeric(row.get("entry_price"))
        quantity = _numeric(row.get("quantity"))
        estimated_notional = (
            float(quantity) * float(limit_price)
            if pd.notna(quantity) and pd.notna(limit_price)
            else _numeric(row.get("approved_notional"))
        )
        rules = rules_by_symbol.get(symbol, {})
        max_holding_days = _numeric(rules.get("max_holding_days"))
        rows.append(
            {
                "draft_order_id": f"DRAFT-BUY-{idx + 1:03d}",
                "source_plan_section": _clean_text(row.get("plan_section")),
                "symbol": symbol,
                "side": "BUY",
                "order_type": "broker_neutral_limit_draft",
                "limit_price": limit_price,
                "quantity": int(quantity) if pd.notna(quantity) else 0,
                "estimated_notional": estimated_notional if pd.notna(estimated_notional) else 0.0,
                "stop_loss_price": rules.get("stop_loss_price", float("nan")),
                "take_profit_price": rules.get("take_profit_price", float("nan")),
                "max_holding_days": int(max_holding_days) if pd.notna(max_holding_days) else "",
                "human_review_required": True,
                "execution_allowed": False,
                "broker_connected": False,
                "trading_ready": False,
                "draft_status": "draft_only",
                "notes": "Broker-neutral draft only. Human review is required. No broker connection, live trading, or order submission.",
            }
        )
    return pd.DataFrame(rows)


def build_manual_review_checklist(order_drafts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, draft in order_drafts.iterrows():
        for check_name in REVIEW_CHECKS:
            rows.append(
                {
                    "draft_order_id": draft.get("draft_order_id"),
                    "symbol": draft.get("symbol"),
                    "check_name": check_name,
                    "required": True,
                    "review_status": "pending_human_review",
                    "execution_allowed": False,
                    "broker_connected": False,
                    "trading_ready": False,
                    "notes": "Manual confirmation required before any separate external action. This tool cannot execute orders.",
                }
            )
    return pd.DataFrame(rows)


def build_summary(order_drafts: pd.DataFrame) -> pd.DataFrame:
    if order_drafts.empty:
        execution_allowed_count = 0
        broker_connected_count = 0
        trading_ready_count = 0
        human_review_required_count = 0
        buy_draft_count = 0
        sell_draft_count = 0
    else:
        execution_allowed_count = int(order_drafts["execution_allowed"].fillna(False).astype(bool).sum())
        broker_connected_count = int(order_drafts["broker_connected"].fillna(False).astype(bool).sum())
        trading_ready_count = int(order_drafts["trading_ready"].fillna(False).astype(bool).sum())
        human_review_required_count = int(order_drafts["human_review_required"].fillna(False).astype(bool).sum())
        buy_draft_count = int((order_drafts["side"] == "BUY").sum())
        sell_draft_count = int((order_drafts["side"] == "SELL").sum())
    return pd.DataFrame(
        [
            {
                "summary_item": "semi_auto_order_generator_run",
                "draft_order_count": int(len(order_drafts)),
                "buy_draft_count": buy_draft_count,
                "sell_draft_count": sell_draft_count,
                "execution_allowed_count": execution_allowed_count,
                "broker_connected_count": broker_connected_count,
                "trading_ready_count": trading_ready_count,
                "human_review_required_count": human_review_required_count,
                "conclusion": "broker_neutral_order_drafts_created_research_only",
                "trading_ready": False,
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The generator reads existing daily plan rows and writes draft tickets only.", "No historical backtest is run."),
        ("no_threshold_change", "confirmed", "No signal threshold module or value is changed.", "Signal thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local V5 CSV outputs are read.", "No market data source is added and no live data is fetched."),
        ("no_broker_integration", "confirmed", "No broker API, account connection, or order route is used.", "No execution path is created."),
        ("no_live_trading", "confirmed", "Outputs are CSV/Markdown draft tickets only.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "Draft tickets are broker-neutral human-review records only.", "No real or broker-simulated orders are submitted."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False is written to Step 7 outputs.", "No deployable status is claimed."),
        ("semi_auto_order_draft_only", "confirmed", "The generator creates draft order tickets for already-sized BUY rows.", "No broker integration or execution action exists."),
        ("human_review_required", "confirmed", "human_review_required=True is written to every draft row.", "A person must review any draft outside this tool."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def build_ticket_markdown(order_drafts: pd.DataFrame, checklist: pd.DataFrame, guardrails: pd.DataFrame) -> str:
    lines = [
        "# V5 Step 7 Broker-Neutral Order Draft Tickets",
        "",
        "This is research/educational tooling only.",
        "No broker execution occurred.",
        "No real orders were submitted.",
        "No live trading is enabled.",
        "No broker is connected.",
        "All outputs remain trading_ready=False.",
        "",
    ]
    if order_drafts.empty:
        lines.extend(["No valid sized BUY positions were available for draft ticket generation.", ""])
    for _, row in order_drafts.iterrows():
        lines.extend(
            [
                f"## Draft Order: {row.get('draft_order_id')}",
                f"Symbol: {row.get('symbol')}",
                f"Side: {row.get('side')}",
                f"Order Type: {row.get('order_type')}",
                f"Quantity: {row.get('quantity')}",
                f"Limit Price: {row.get('limit_price')}",
                f"Estimated Notional: {row.get('estimated_notional')}",
                f"Stop Loss: {row.get('stop_loss_price')}",
                f"Take Profit: {row.get('take_profit_price')}",
                f"Max Holding Days: {row.get('max_holding_days')}",
                f"Human Review Required: {row.get('human_review_required')}",
                f"Execution Allowed: {row.get('execution_allowed')}",
                f"Broker Connected: {row.get('broker_connected')}",
                f"Trading Ready: {row.get('trading_ready')}",
                f"Draft Status: {row.get('draft_status')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Manual Review Checklist",
            checklist.to_markdown(index=False) if not checklist.empty else "No checklist rows were generated.",
            "",
            "## Guardrails",
            guardrails.to_markdown(index=False),
            "",
            "## Warning",
            "These are broker-neutral draft tickets for human review only. This tool does not connect to any broker, submit any order, or create a live execution path.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_semi_auto_order_outputs(
    daily_plan_path: str | Path = DEFAULT_DAILY_PLAN_PATH,
    exit_plan_path: str | Path = DEFAULT_EXIT_PLAN_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    daily_source = Path(daily_plan_path)
    exit_source = Path(exit_plan_path)
    output_path = Path(output_dir)

    daily_plan = _read_csv(daily_source)
    exit_plan = _read_csv(exit_source)
    order_drafts = build_order_drafts(daily_plan, exit_plan)
    checklist = build_manual_review_checklist(order_drafts)
    summary = build_summary(order_drafts)
    guardrails = build_guardrails()
    tickets = build_ticket_markdown(order_drafts, checklist, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    order_drafts.to_csv(paths["drafts"], index=False)
    paths["tickets"].write_text(tickets, encoding="utf-8")
    checklist.to_csv(paths["checklist"], index=False)
    summary.to_csv(paths["summary"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    config = {
        "daily_plan_path": str(daily_source),
        "exit_plan_path": str(exit_source),
        "output_dir": str(output_path),
        "daily_plan_rows": int(len(daily_plan)),
        "exit_plan_rows": int(len(exit_plan)),
        "draft_order_count": int(len(order_drafts)),
        "scope": "V5 Step 7 semi-auto order draft only",
        "broker_execution": False,
        "broker_connected": False,
        "live_trading": False,
        "order_execution": False,
        "order_submission": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "order_drafts": order_drafts,
        "broker_neutral_order_tickets": tickets,
        "manual_review_checklist": checklist,
        "semi_auto_order_summary": summary,
        "semi_auto_order_guardrails": guardrails,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
