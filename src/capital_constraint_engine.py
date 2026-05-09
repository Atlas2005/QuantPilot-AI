import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILENAMES = {
    "feasibility": "capital_feasibility.csv",
    "approved": "approved_orders.csv",
    "rejected": "rejected_orders.csv",
    "summary": "capital_constraint_summary.csv",
    "report": "capital_constraint_report.md",
    "guardrails": "capital_constraint_guardrails.csv",
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


def _read_candidates(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return default_educational_candidates()
    try:
        df = pd.read_csv(path, dtype={"symbol": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "symbol" in df:
        df["symbol"] = df["symbol"].map(_format_symbol)
    return df


def default_educational_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "EDU-001",
                "symbol": "600519",
                "side": "BUY",
                "price": 1700.0,
                "board": "main",
                "notes": "High-priced main-board example expected to fail with low cash.",
            },
            {
                "candidate_id": "EDU-002",
                "symbol": "600000",
                "side": "BUY",
                "price": 8.0,
                "board": "main",
                "notes": "Low-priced main-board example expected to be affordable with 1000 cash.",
            },
            {
                "candidate_id": "EDU-003",
                "symbol": "300750",
                "side": "BUY",
                "price": 200.0,
                "board": "main",
                "notes": "Higher-priced growth-board style example using default lot rule in Step 1.",
            },
            {
                "candidate_id": "EDU-004",
                "symbol": "000001",
                "side": "BUY",
                "price": None,
                "board": "main",
                "notes": "Invalid missing price example.",
            },
            {
                "candidate_id": "EDU-005",
                "symbol": "688001",
                "side": "BUY",
                "price": 4.0,
                "board": "STAR",
                "notes": "STAR/KCB-style example requiring 200-share minimum.",
            },
        ]
    )


def infer_board(symbol: str, board: Any) -> str:
    board_text = _clean_text(board).upper()
    if board_text in {"STAR", "KCB", "科创板", "STAR_MARKET", "SCI_TECH"}:
        return "STAR"
    if symbol.startswith(("688", "689")):
        return "STAR"
    if board_text in {"ETF", "FUND", "CONVERTIBLE_BOND", "CBOND", "BOND"}:
        return "UNSUPPORTED"
    return board_text or "MAIN"


def lot_size_for_order(symbol: str, board: Any, default_lot_size: int) -> tuple[int, str]:
    inferred = infer_board(symbol, board)
    if inferred == "STAR":
        return 200, "star_or_kcb_min_lot"
    if inferred == "UNSUPPORTED":
        return int(default_lot_size), "unsupported_or_requires_rule"
    return int(default_lot_size), "default_main_board_lot"


def evaluate_capital_constraints(
    candidates: pd.DataFrame,
    cash: float,
    buffer: float = 0.97,
    default_lot_size: int = 100,
) -> pd.DataFrame:
    rows = []
    for idx, row in candidates.reset_index(drop=True).iterrows():
        symbol = _format_symbol(row.get("symbol"))
        side = _clean_text(row.get("side")).upper() or "BUY"
        price = _numeric(row.get("price"))
        min_required_cash = _numeric(row.get("min_required_cash", 0.0))
        if pd.isna(min_required_cash):
            min_required_cash = 0.0
        board = infer_board(symbol, row.get("board"))
        lot_size, lot_rule = lot_size_for_order(symbol, row.get("board"), default_lot_size)
        usable_cash = max(0.0, float(cash) * float(buffer) - float(min_required_cash)) if pd.notna(cash) else 0.0
        max_affordable_price = usable_cash / lot_size if lot_size > 0 else 0.0
        allowed = False
        rejection_reason = ""
        affordable_lots = 0
        quantity = 0
        notional = 0.0
        if not symbol:
            rejection_reason = "missing_symbol"
        elif pd.isna(cash) or float(cash) <= 0:
            rejection_reason = "invalid_cash"
        elif side != "BUY":
            rejection_reason = "unsupported_side_for_step1"
        elif board == "UNSUPPORTED":
            rejection_reason = "unsupported_or_requires_rule"
        elif pd.isna(price) or float(price) <= 0:
            rejection_reason = "invalid_or_missing_price"
        elif float(price) * lot_size > usable_cash:
            rejection_reason = "insufficient_cash_for_min_lot"
        else:
            affordable_lots = int(usable_cash // (float(price) * lot_size))
            quantity = affordable_lots * lot_size
            notional = quantity * float(price)
            allowed = quantity > 0
            if not allowed:
                rejection_reason = "insufficient_cash_for_min_lot"
        remaining_cash = float(cash) - notional if pd.notna(cash) else float("nan")
        exposure_pct = (notional / float(cash) * 100.0) if pd.notna(cash) and float(cash) > 0 else 0.0
        rows.append(
            {
                "candidate_id": _clean_text(row.get("candidate_id")) or f"candidate_{idx + 1}",
                "symbol": symbol,
                "side": side,
                "board": board,
                "price": price,
                "available_cash": cash,
                "usable_cash_buffer": buffer,
                "minimum_required_cash": min_required_cash,
                "usable_cash_after_buffer_and_reserve": usable_cash,
                "lot_size": lot_size,
                "lot_rule": lot_rule,
                "minimum_lot_notional": float(price) * lot_size if pd.notna(price) else float("nan"),
                "maximum_affordable_price": max_affordable_price,
                "affordable_lots": affordable_lots,
                "quantity": quantity,
                "order_notional": notional,
                "remaining_cash_after_order": remaining_cash,
                "exposure_pct": exposure_pct,
                "order_allowed": bool(allowed),
                "rejection_reason": rejection_reason,
                "trading_ready": False,
                "notes": "Capital feasibility only. No broker execution is performed.",
            }
        )
    return pd.DataFrame(rows)


def build_summary(feasibility: pd.DataFrame, cash: float, buffer: float, default_lot_size: int) -> pd.DataFrame:
    approved = int(feasibility["order_allowed"].fillna(False).astype(bool).sum()) if not feasibility.empty else 0
    rejected = int(len(feasibility) - approved)
    total_notional = float(pd.to_numeric(feasibility.get("order_notional", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not feasibility.empty else 0.0
    max_exposure = float(pd.to_numeric(feasibility.get("exposure_pct", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if not feasibility.empty else 0.0
    return pd.DataFrame(
        [
            {
                "summary_item": "capital_constraint_run",
                "candidate_count": int(len(feasibility)),
                "approved_order_count": approved,
                "rejected_order_count": rejected,
                "available_cash": cash,
                "usable_cash_buffer": buffer,
                "default_lot_size": default_lot_size,
                "total_approved_notional": total_notional,
                "max_single_order_exposure_pct": max_exposure,
                "trading_ready": False,
                "conclusion": "V5 Step 1 checks capital feasibility only. Project remains not trading-ready.",
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Capital engine evaluates candidate rows only.", "No historical backtest is run."),
        ("no_threshold_change", "confirmed", "No strategy threshold module or value is changed.", "Signal thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Default candidates are deterministic educational rows, or a user-provided local CSV is read.", "No market data source is added."),
        ("no_broker_integration", "confirmed", "No broker API, order route, or account connection is used.", "No execution path is created."),
        ("no_live_trading", "confirmed", "Outputs are CSV/Markdown feasibility reports only.", "No live trading is performed."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False is written to Step 1 outputs.", "No deployable status is claimed."),
        ("capital_feasibility_only", "confirmed", "The engine checks cash, buffer, reserve, lot size, and affordability.", "No sell handling or order execution."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def build_report(
    feasibility: pd.DataFrame,
    summary: pd.DataFrame,
    guardrails: pd.DataFrame,
    cash: float,
    buffer: float,
    default_lot_size: int,
    input_candidates: str | None,
    output_dir: str | Path,
) -> str:
    approved = int(summary.iloc[0].get("approved_order_count")) if not summary.empty else 0
    rejected = int(summary.iloc[0].get("rejected_order_count")) if not summary.empty else 0
    return "\n".join(
        [
            "# V5 Step 1 Capital Constraint Engine Report",
            "",
            "## Executive Summary",
            "V5 Step 1 checks candidate buy-order capital feasibility only.",
            "This is educational/research tooling only.",
            "This is not financial advice.",
            "No broker execution is performed.",
            "No order should be treated as trading-ready.",
            "The project remains not trading-ready.",
            "",
            "## Inputs",
            f"- Input candidates: {input_candidates or 'deterministic educational defaults'}",
            f"- Available cash: {cash}",
            f"- Usable cash buffer: {buffer}",
            f"- Default lot size: {default_lot_size}",
            f"- Output directory: {output_dir}",
            "",
            "## Capital Feasibility Summary",
            f"- Candidate rows: {len(feasibility)}",
            f"- Approved rows: {approved}",
            f"- Rejected rows: {rejected}",
            "",
            "## Rules Applied",
            "- Main-board default minimum lot size is 100 shares.",
            "- STAR/KCB-style symbols or board labels use a 200-share minimum lot.",
            "- Unsupported ETF/fund/convertible-bond rows are rejected unless a future rule is implemented.",
            "- Non-BUY sides are unsupported in Step 1.",
            "- Invalid cash, missing symbol, and invalid or missing prices are rejected.",
            "- Affordability uses available_cash * buffer minus minimum_required_cash.",
            "",
            "## Feasibility Rows",
            feasibility.to_markdown(index=False) if not feasibility.empty else "No candidate rows were evaluated.",
            "",
            "## Guardrails",
            guardrails.to_markdown(index=False),
            "",
            "## Educational / Research Disclaimer",
            "This report is educational/research diagnostics only. It is not financial advice.",
            "No candidate, symbol, order, quantity, or output in this report should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_capital_constraint_outputs(
    input_candidates: str | Path | None,
    cash: float,
    buffer: float,
    default_lot_size: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_candidates) if input_candidates else None
    candidates = _read_candidates(input_path)
    feasibility = evaluate_capital_constraints(
        candidates=candidates,
        cash=cash,
        buffer=buffer,
        default_lot_size=default_lot_size,
    )
    approved = feasibility[feasibility["order_allowed"].fillna(False).astype(bool)].copy()
    rejected = feasibility[~feasibility["order_allowed"].fillna(False).astype(bool)].copy()
    summary = build_summary(feasibility, cash, buffer, default_lot_size)
    guardrails = build_guardrails()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    report = build_report(feasibility, summary, guardrails, cash, buffer, default_lot_size, str(input_path) if input_path else None, output_path)
    feasibility.to_csv(paths["feasibility"], index=False)
    approved.to_csv(paths["approved"], index=False)
    rejected.to_csv(paths["rejected"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["report"].write_text(report, encoding="utf-8")
    guardrails.to_csv(paths["guardrails"], index=False)
    config = {
        "input_candidates": str(input_path) if input_path else None,
        "used_default_educational_candidates": input_path is None,
        "cash": cash,
        "buffer": buffer,
        "default_lot_size": default_lot_size,
        "output_dir": str(output_path),
        "candidate_count": int(len(feasibility)),
        "approved_order_count": int(len(approved)),
        "rejected_order_count": int(len(rejected)),
        "scope": "V5 Step 1 capital feasibility only",
        "broker_execution": False,
        "live_trading": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "capital_feasibility": feasibility,
        "approved_orders": approved,
        "rejected_orders": rejected,
        "capital_constraint_summary": summary,
        "capital_constraint_report": report,
        "capital_constraint_guardrails": guardrails,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
