import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path("outputs/position_sizing_engine_real_v1/sized_positions.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/exit_engine_real_v1")
DEFAULT_STOP_LOSS_PCT = 0.05
DEFAULT_TAKE_PROFIT_PCT = 0.10
DEFAULT_MAX_HOLDING_DAYS = 10
DEFAULT_BENCHMARK_LAG_EXIT_RULE = "exit_if_underperform_benchmark_by_3pct_after_5_days"

OUTPUT_FILENAMES = {
    "exit_plan": "exit_plan.csv",
    "guardrails": "exit_guardrails.csv",
    "summary": "exit_summary.csv",
    "report": "exit_engine_report.md",
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


def _read_sized_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"symbol": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "symbol" in df:
        df["symbol"] = df["symbol"].map(_format_symbol)
    return df.reset_index(drop=True)


def evaluate_exit_plan(
    sized_positions: pd.DataFrame,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
    benchmark_lag_exit_rule: str = DEFAULT_BENCHMARK_LAG_EXIT_RULE,
) -> pd.DataFrame:
    rows = []
    for _, row in sized_positions.reset_index(drop=True).iterrows():
        symbol = _format_symbol(row.get("symbol"))
        entry_price = _numeric(row.get("entry_price", row.get("price")))
        quantity = _numeric(row.get("quantity"))
        approved_notional = _numeric(row.get("approved_notional"))
        status = "planned"
        issues = []

        if not symbol:
            issues.append("missing_symbol")
        if pd.isna(entry_price) or float(entry_price) <= 0:
            issues.append("invalid_or_missing_entry_price")
        if pd.isna(quantity) or float(quantity) <= 0:
            issues.append("invalid_or_missing_quantity")
        if pd.isna(approved_notional) or float(approved_notional) <= 0:
            issues.append("invalid_or_missing_approved_notional")

        if issues:
            status = "not_planned_input_invalid"
            stop_loss_price = float("nan")
            take_profit_price = float("nan")
            notes = (
                "Exit planning research only. No order generation, broker execution, "
                f"or live trading. Input issues: {', '.join(issues)}."
            )
        else:
            entry_price = float(entry_price)
            stop_loss_price = round(entry_price * (1.0 - float(stop_loss_pct)), 4)
            take_profit_price = round(entry_price * (1.0 + float(take_profit_pct)), 4)
            notes = (
                "Exit planning research only. Rules are static planning assumptions, "
                "not broker orders or trading-ready instructions."
            )

        rows.append(
            {
                "symbol": symbol,
                "entry_price": entry_price,
                "quantity": int(quantity) if pd.notna(quantity) else quantity,
                "approved_notional": approved_notional,
                "stop_loss_pct": stop_loss_pct,
                "stop_loss_price": stop_loss_price,
                "take_profit_pct": take_profit_pct,
                "take_profit_price": take_profit_price,
                "max_holding_days": int(max_holding_days),
                "benchmark_lag_exit_rule": benchmark_lag_exit_rule,
                "exit_plan_status": status,
                "trading_ready": False,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    exit_plan: pd.DataFrame,
    input_path: str | Path,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_holding_days: int,
    benchmark_lag_exit_rule: str,
) -> pd.DataFrame:
    planned_count = int((exit_plan["exit_plan_status"] == "planned").sum()) if not exit_plan.empty else 0
    invalid_count = int((exit_plan["exit_plan_status"] != "planned").sum()) if not exit_plan.empty else 0
    total_notional = (
        float(pd.to_numeric(exit_plan.get("approved_notional", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if not exit_plan.empty
        else 0.0
    )
    return pd.DataFrame(
        [
            {
                "summary_item": "exit_engine_run",
                "input_path": str(input_path),
                "sized_position_count": int(len(exit_plan)),
                "planned_exit_count": planned_count,
                "invalid_exit_plan_count": invalid_count,
                "total_approved_notional": total_notional,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "max_holding_days": int(max_holding_days),
                "benchmark_lag_exit_rule": benchmark_lag_exit_rule,
                "trading_ready": False,
                "conclusion": "V5 Step 4 creates research-only exit plans for sized positions. Project remains not trading-ready.",
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The engine reads sized positions and writes planning files only.", "No historical backtest is run."),
        ("no_threshold_change", "confirmed", "No signal threshold module or value is changed.", "Signal thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only a local Step 3 sized positions CSV is read by default.", "No market data source is added."),
        ("no_broker_integration", "confirmed", "No broker API, account connection, or order route is used.", "No execution path is created."),
        ("no_live_trading", "confirmed", "Outputs are CSV/Markdown exit planning reports only.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "Stop loss and take profit values are planning fields only.", "No orders are generated or submitted."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False is written to Step 4 outputs.", "No deployable status is claimed."),
        ("exit_planning_only", "confirmed", "The engine creates explicit static exit planning rules for already-sized positions.", "No portfolio optimizer or execution engine is implemented."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def build_report(
    exit_plan: pd.DataFrame,
    summary: pd.DataFrame,
    guardrails: pd.DataFrame,
    input_path: str | Path,
    output_dir: str | Path,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V5 Step 4 Exit Engine Report",
            "",
            "## Executive Summary",
            "V5 Step 4 creates explicit research-only exit plans for V5 Step 3 sized positions.",
            "This is educational/research tooling only.",
            "This is not financial advice.",
            "No broker execution is performed.",
            "No live trading is performed.",
            "No exit plan should be treated as trading-ready.",
            "The project remains not trading-ready.",
            "",
            "## Inputs",
            f"- Input path: {input_path}",
            f"- Output directory: {output_dir}",
            "",
            "## Exit Planning Summary",
            f"- Sized position rows: {row.get('sized_position_count', 0)}",
            f"- Planned exits: {row.get('planned_exit_count', 0)}",
            f"- Invalid exit plans: {row.get('invalid_exit_plan_count', 0)}",
            f"- Total approved notional: {row.get('total_approved_notional', 0)}",
            "",
            "## Rules Applied",
            f"- Stop loss percentage: {row.get('stop_loss_pct', '')}",
            f"- Take profit percentage: {row.get('take_profit_pct', '')}",
            f"- Maximum holding days: {row.get('max_holding_days', '')}",
            f"- Benchmark lag rule: {row.get('benchmark_lag_exit_rule', '')}",
            "- Stop loss and take profit prices are deterministic planning calculations from entry price.",
            "- Rules are not broker instructions and are not order tickets.",
            "",
            "## Exit Plan Rows",
            exit_plan.to_markdown(index=False) if not exit_plan.empty else "No sized positions were available for exit planning.",
            "",
            "## Guardrails",
            guardrails.to_markdown(index=False),
            "",
            "## Educational / Research Disclaimer",
            "This report is educational/research diagnostics only. It is not financial advice.",
            "No symbol, quantity, stop loss, take profit, or exit rule in this report should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_exit_engine_outputs(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
    benchmark_lag_exit_rule: str = DEFAULT_BENCHMARK_LAG_EXIT_RULE,
) -> dict[str, Any]:
    source_path = Path(input_path)
    sized_positions = _read_sized_positions(source_path)
    exit_plan = evaluate_exit_plan(
        sized_positions=sized_positions,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_days=max_holding_days,
        benchmark_lag_exit_rule=benchmark_lag_exit_rule,
    )
    summary = build_summary(
        exit_plan=exit_plan,
        input_path=source_path,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_days=max_holding_days,
        benchmark_lag_exit_rule=benchmark_lag_exit_rule,
    )
    guardrails = build_guardrails()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    report = build_report(exit_plan, summary, guardrails, source_path, output_path)
    exit_plan.to_csv(paths["exit_plan"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["report"].write_text(report, encoding="utf-8")
    config = {
        "input_path": str(source_path),
        "output_dir": str(output_path),
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "max_holding_days": int(max_holding_days),
        "benchmark_lag_exit_rule": benchmark_lag_exit_rule,
        "sized_position_count": int(len(exit_plan)),
        "planned_exit_count": int((exit_plan["exit_plan_status"] == "planned").sum()) if not exit_plan.empty else 0,
        "scope": "V5 Step 4 exit planning only",
        "broker_execution": False,
        "live_trading": False,
        "order_execution": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "exit_plan": exit_plan,
        "exit_guardrails": guardrails,
        "exit_summary": summary,
        "exit_engine_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
