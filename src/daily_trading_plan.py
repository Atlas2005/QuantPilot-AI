import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TRADABLE_PATH = Path("outputs/tradable_universe_filter_real_v1/tradable_universe.csv")
DEFAULT_SIZED_PATH = Path("outputs/position_sizing_engine_real_v1/sized_positions.csv")
DEFAULT_DEFERRED_PATH = Path("outputs/position_sizing_engine_real_v1/deferred_positions.csv")
DEFAULT_EXIT_PLAN_PATH = Path("outputs/exit_engine_real_v1/exit_plan.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/daily_trading_plan_real_v1")

OUTPUT_FILENAMES = {
    "plan": "daily_trading_plan.csv",
    "summary": "daily_trading_plan_summary.csv",
    "guardrails": "daily_trading_plan_guardrails.csv",
    "report": "daily_trading_plan.md",
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


def _bool_false_series(length: int) -> list[bool]:
    return [False for _ in range(length)]


def build_daily_plan(
    tradable: pd.DataFrame,
    sized: pd.DataFrame,
    deferred: pd.DataFrame,
    exit_plan: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in tradable.iterrows():
        rows.append(
            {
                "plan_section": "tradable_candidate",
                "symbol": _format_symbol(row.get("symbol")),
                "candidate_id": _clean_text(row.get("candidate_id")),
                "side": _clean_text(row.get("side")) or "BUY",
                "board": _clean_text(row.get("board")),
                "entry_price": _numeric(row.get("price")),
                "quantity": "",
                "approved_notional": "",
                "stop_loss_price": "",
                "take_profit_price": "",
                "max_holding_days": "",
                "benchmark_lag_exit_rule": "",
                "status": "tradable_candidate",
                "action": "review_for_sizing_context",
                "trading_ready": False,
                "notes": "Candidate passed Step 2 tradable-universe filters. Research review only; no order execution.",
            }
        )

    for _, row in sized.iterrows():
        rows.append(
            {
                "plan_section": "sized_position",
                "symbol": _format_symbol(row.get("symbol")),
                "candidate_id": _clean_text(row.get("candidate_id")),
                "side": _clean_text(row.get("side")) or "BUY",
                "board": _clean_text(row.get("board")),
                "entry_price": _numeric(row.get("price")),
                "quantity": int(_numeric(row.get("quantity"))) if pd.notna(_numeric(row.get("quantity"))) else "",
                "approved_notional": _numeric(row.get("approved_notional")),
                "stop_loss_price": "",
                "take_profit_price": "",
                "max_holding_days": "",
                "benchmark_lag_exit_rule": "",
                "status": _clean_text(row.get("sizing_status")) or "sized",
                "action": "human_review_sized_position",
                "trading_ready": False,
                "notes": "Sized by Step 3 for research planning only. No broker execution or live trading.",
            }
        )

    for _, row in deferred.iterrows():
        rows.append(
            {
                "plan_section": "deferred_position",
                "symbol": _format_symbol(row.get("symbol")),
                "candidate_id": _clean_text(row.get("candidate_id")),
                "side": _clean_text(row.get("side")) or "BUY",
                "board": _clean_text(row.get("board")),
                "entry_price": _numeric(row.get("price")),
                "quantity": int(_numeric(row.get("quantity"))) if pd.notna(_numeric(row.get("quantity"))) else "",
                "approved_notional": _numeric(row.get("approved_notional")),
                "stop_loss_price": "",
                "take_profit_price": "",
                "max_holding_days": "",
                "benchmark_lag_exit_rule": "",
                "status": _clean_text(row.get("sizing_reason")) or "deferred",
                "action": "defer_no_action",
                "trading_ready": False,
                "notes": "Deferred by Step 3. Research review only; no order execution.",
            }
        )

    for _, row in exit_plan.iterrows():
        rows.append(
            {
                "plan_section": "exit_plan",
                "symbol": _format_symbol(row.get("symbol")),
                "candidate_id": "",
                "side": "EXIT_RULE",
                "board": "",
                "entry_price": _numeric(row.get("entry_price")),
                "quantity": int(_numeric(row.get("quantity"))) if pd.notna(_numeric(row.get("quantity"))) else "",
                "approved_notional": _numeric(row.get("approved_notional")),
                "stop_loss_price": _numeric(row.get("stop_loss_price")),
                "take_profit_price": _numeric(row.get("take_profit_price")),
                "max_holding_days": int(_numeric(row.get("max_holding_days"))) if pd.notna(_numeric(row.get("max_holding_days"))) else "",
                "benchmark_lag_exit_rule": _clean_text(row.get("benchmark_lag_exit_rule")),
                "status": _clean_text(row.get("exit_plan_status")) or "planned",
                "action": "human_review_exit_rules",
                "trading_ready": False,
                "notes": "Exit rules are Step 4 planning assumptions only, not broker orders.",
            }
        )

    plan = pd.DataFrame(rows)
    if not plan.empty:
        plan["trading_ready"] = _bool_false_series(len(plan))
    return plan


def build_summary(
    tradable: pd.DataFrame,
    sized: pd.DataFrame,
    deferred: pd.DataFrame,
    exit_plan: pd.DataFrame,
    plan: pd.DataFrame,
) -> pd.DataFrame:
    available_cash = _numeric(
        sized["available_cash"].iloc[0]
        if "available_cash" in sized and not sized.empty
        else tradable["available_cash"].iloc[0]
        if "available_cash" in tradable and not tradable.empty
        else float("nan")
    )
    usable_cash = _numeric(
        sized["usable_cash"].iloc[0]
        if "usable_cash" in sized and not sized.empty
        else tradable["usable_cash_after_buffer_and_reserve"].iloc[0]
        if "usable_cash_after_buffer_and_reserve" in tradable and not tradable.empty
        else float("nan")
    )
    approved_notional = (
        float(pd.to_numeric(sized.get("approved_notional", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if not sized.empty
        else 0.0
    )
    deferred_notional = (
        float(pd.to_numeric(deferred.get("minimum_required_cash", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if not deferred.empty
        else 0.0
    )
    remaining_usable_cash = _numeric(
        sized["remaining_usable_cash_after"].iloc[-1]
        if "remaining_usable_cash_after" in sized and not sized.empty
        else usable_cash - approved_notional
        if pd.notna(usable_cash)
        else float("nan")
    )
    return pd.DataFrame(
        [
            {
                "summary_item": "daily_trading_plan_run",
                "tradable_candidate_count": int(len(tradable)),
                "sized_position_count": int(len(sized)),
                "deferred_position_count": int(len(deferred)),
                "exit_plan_count": int(len(exit_plan)),
                "daily_plan_row_count": int(len(plan)),
                "available_cash": available_cash,
                "usable_cash": usable_cash,
                "total_approved_notional": approved_notional,
                "deferred_required_notional": deferred_notional,
                "remaining_usable_cash": remaining_usable_cash,
                "trading_ready": False,
                "conclusion": "V5 Step 5 combines local V5 planning outputs for human review only. Project remains not trading-ready.",
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The plan generator combines existing V5 output CSVs only.", "No historical backtest is run."),
        ("no_threshold_change", "confirmed", "No signal threshold module or value is changed.", "Signal thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local Step 2, Step 3, and Step 4 CSV outputs are read.", "No market data source is added."),
        ("no_broker_integration", "confirmed", "No broker API, account connection, or order route is used.", "No execution path is created."),
        ("no_live_trading", "confirmed", "Outputs are CSV/Markdown daily plan reports only.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "Plan rows are human-review records only.", "No orders are generated or submitted."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False is written to Step 5 outputs.", "No deployable status is claimed."),
        ("daily_plan_only", "confirmed", "The engine composes a daily review plan from prior V5 outputs.", "No strategy, threshold, model, or broker behavior is changed."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    tradable: pd.DataFrame,
    sized: pd.DataFrame,
    deferred: pd.DataFrame,
    exit_plan: pd.DataFrame,
    daily_plan: pd.DataFrame,
    summary: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V5 Step 5 Daily Trading Plan",
            "",
            "## Executive Summary",
            "V5 Step 5 combines existing V5 Step 2, Step 3, and Step 4 local outputs into a human-reviewable daily plan.",
            "This is research only.",
            "This is not financial advice.",
            "No broker execution is performed.",
            "No live trading is performed.",
            "No order execution is performed.",
            "trading_ready=False for all rows.",
            "The project remains not trading-ready.",
            "",
            "## Capital Summary",
            f"- Available cash: {row.get('available_cash', '')}",
            f"- Usable cash: {row.get('usable_cash', '')}",
            f"- Total approved notional: {row.get('total_approved_notional', 0)}",
            f"- Deferred required notional: {row.get('deferred_required_notional', 0)}",
            f"- Remaining usable cash: {row.get('remaining_usable_cash', '')}",
            "",
            "## Tradable Candidates",
            _table(tradable, "No tradable candidates were available."),
            "",
            "## Sized Positions",
            _table(sized, "No sized positions were available."),
            "",
            "## Deferred Positions",
            _table(deferred, "No deferred positions were available."),
            "",
            "## Exit Plan",
            _table(exit_plan, "No exit plan rows were available."),
            "",
            "## Daily Plan Rows",
            _table(daily_plan, "No daily plan rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Warnings",
            "- Research only.",
            "- Not financial advice.",
            "- No broker execution.",
            "- No live trading.",
            "- No order execution.",
            "- trading_ready=False.",
            "",
        ]
    )


def generate_daily_trading_plan_outputs(
    tradable_path: str | Path = DEFAULT_TRADABLE_PATH,
    sized_path: str | Path = DEFAULT_SIZED_PATH,
    deferred_path: str | Path = DEFAULT_DEFERRED_PATH,
    exit_plan_path: str | Path = DEFAULT_EXIT_PLAN_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    tradable_source = Path(tradable_path)
    sized_source = Path(sized_path)
    deferred_source = Path(deferred_path)
    exit_source = Path(exit_plan_path)

    tradable = _read_csv(tradable_source)
    sized = _read_csv(sized_source)
    deferred = _read_csv(deferred_source)
    exit_plan = _read_csv(exit_source)
    daily_plan = build_daily_plan(tradable, sized, deferred, exit_plan)
    summary = build_summary(tradable, sized, deferred, exit_plan, daily_plan)
    guardrails = build_guardrails()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    report = build_report(tradable, sized, deferred, exit_plan, daily_plan, summary, guardrails)

    daily_plan.to_csv(paths["plan"], index=False)
    summary.to_csv(paths["summary"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    paths["report"].write_text(report, encoding="utf-8")
    config = {
        "tradable_path": str(tradable_source),
        "sized_path": str(sized_source),
        "deferred_path": str(deferred_source),
        "exit_plan_path": str(exit_source),
        "output_dir": str(output_path),
        "tradable_candidate_count": int(len(tradable)),
        "sized_position_count": int(len(sized)),
        "deferred_position_count": int(len(deferred)),
        "exit_plan_count": int(len(exit_plan)),
        "daily_plan_row_count": int(len(daily_plan)),
        "scope": "V5 Step 5 daily plan only",
        "broker_execution": False,
        "live_trading": False,
        "order_execution": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "daily_trading_plan": daily_plan,
        "daily_trading_plan_summary": summary,
        "daily_trading_plan_guardrails": guardrails,
        "daily_trading_plan_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
