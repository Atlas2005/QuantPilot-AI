import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path("outputs/tradable_universe_filter_real_v1/tradable_universe.csv")

OUTPUT_FILENAMES = {
    "summary": "position_sizing_summary.csv",
    "sized": "sized_positions.csv",
    "deferred": "deferred_positions.csv",
    "rejected": "rejected_positions.csv",
    "guardrails": "position_sizing_guardrails.csv",
    "report": "position_sizing_report.md",
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


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean_text(value).lower() in {"true", "1", "yes", "y"}


def _read_tradable_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"symbol": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "symbol" in df:
        df["symbol"] = df["symbol"].map(_format_symbol)
    if "tradable" in df:
        df = df[df["tradable"].fillna(False).map(_bool_value)].copy()
    return df.reset_index(drop=True)


def infer_lot_size(symbol: str, board: Any, row_lot_size: Any, default_lot_size: int) -> tuple[int, str]:
    explicit = _numeric(row_lot_size)
    if pd.notna(explicit) and explicit > 0:
        return int(explicit), "input_lot_size"
    board_text = _clean_text(board).upper()
    if board_text in {"STAR", "KCB", "STAR_MARKET", "SCI_TECH"} or symbol.startswith(("688", "689")):
        return 200, "star_or_kcb_min_lot"
    return int(default_lot_size), "default_main_board_lot"


def evaluate_position_sizing(
    candidates: pd.DataFrame,
    available_cash: float = 1000.0,
    usable_cash_buffer: float = 0.97,
    default_lot_size: int = 100,
) -> pd.DataFrame:
    usable_cash = float(available_cash) * float(usable_cash_buffer) if pd.notna(available_cash) else 0.0
    allocated_notional = 0.0
    rows = []
    for idx, row in candidates.reset_index(drop=True).iterrows():
        symbol = _format_symbol(row.get("symbol"))
        side = _clean_text(row.get("side")).upper() or "BUY"
        board = _clean_text(row.get("board")).upper() or "MAIN"
        price = _numeric(row.get("price"))
        lot_size, lot_rule = infer_lot_size(symbol, board, row.get("lot_size"), default_lot_size)
        min_notional = float(price) * lot_size if pd.notna(price) and price > 0 else float("nan")
        remaining_usable_cash_before = max(0.0, usable_cash - allocated_notional)
        quantity = 0
        approved_notional = 0.0
        remaining_usable_cash_after = remaining_usable_cash_before
        status = "rejected"
        reason = ""

        if not symbol or not (symbol.isdigit() and len(symbol) == 6):
            reason = "missing_or_invalid_symbol"
        elif pd.isna(available_cash) or float(available_cash) <= 0:
            reason = "invalid_cash"
        elif side != "BUY":
            reason = "unsupported_side_for_step3"
        elif pd.isna(price) or float(price) <= 0:
            reason = "invalid_or_missing_price"
        elif pd.isna(min_notional) or min_notional <= 0:
            reason = "invalid_min_lot_notional"
        elif min_notional > usable_cash:
            reason = "insufficient_account_cash_for_min_lot"
        elif min_notional > remaining_usable_cash_before:
            status = "deferred"
            reason = "account_cash_exhausted_or_allocation_limit"
        else:
            status = "sized"
            reason = "approved_minimum_lot_position"
            quantity = lot_size
            approved_notional = min_notional
            allocated_notional += approved_notional
            remaining_usable_cash_after = max(0.0, usable_cash - allocated_notional)

        exposure_pct = (approved_notional / float(available_cash) * 100.0) if pd.notna(available_cash) and float(available_cash) > 0 else 0.0
        rows.append(
            {
                "allocation_rank": idx + 1,
                "candidate_id": _clean_text(row.get("candidate_id")) or f"candidate_{idx + 1}",
                "symbol": symbol,
                "side": side,
                "board": board,
                "price": price,
                "available_cash": available_cash,
                "usable_cash_buffer": usable_cash_buffer,
                "usable_cash": usable_cash,
                "remaining_usable_cash_before": remaining_usable_cash_before,
                "lot_size": lot_size,
                "lot_rule": lot_rule,
                "minimum_required_cash": min_notional,
                "quantity": quantity,
                "approved_notional": approved_notional,
                "remaining_usable_cash_after": remaining_usable_cash_after,
                "exposure_pct": exposure_pct,
                "sizing_status": status,
                "sizing_reason": reason,
                "trading_ready": False,
                "notes": "Position sizing research only. No order generation, broker execution, or live trading.",
            }
        )
    return pd.DataFrame(rows)


def build_summary(sizing: pd.DataFrame, available_cash: float, usable_cash_buffer: float, default_lot_size: int, input_path: str | Path) -> pd.DataFrame:
    sized_count = int((sizing["sizing_status"] == "sized").sum()) if not sizing.empty else 0
    deferred_count = int((sizing["sizing_status"] == "deferred").sum()) if not sizing.empty else 0
    rejected_count = int((sizing["sizing_status"] == "rejected").sum()) if not sizing.empty else 0
    total_notional = float(pd.to_numeric(sizing.get("approved_notional", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not sizing.empty else 0.0
    usable_cash = float(available_cash) * float(usable_cash_buffer) if pd.notna(available_cash) else 0.0
    return pd.DataFrame(
        [
            {
                "summary_item": "position_sizing_run",
                "input_path": str(input_path),
                "candidate_count": int(len(sizing)),
                "sized_position_count": sized_count,
                "deferred_position_count": deferred_count,
                "rejected_position_count": rejected_count,
                "available_cash": available_cash,
                "usable_cash_buffer": usable_cash_buffer,
                "usable_cash": usable_cash,
                "default_lot_size": default_lot_size,
                "total_approved_notional": total_notional,
                "remaining_usable_cash": max(0.0, usable_cash - total_notional),
                "trading_ready": False,
                "conclusion": "V5 Step 3 sizes minimum-lot research positions only. Project remains not trading-ready.",
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Position sizing reads Step 2 tradable universe rows only.", "No historical backtest is run."),
        ("no_threshold_change", "confirmed", "No strategy threshold module or value is changed.", "Signal thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only a local Step 2 output CSV or user-provided local CSV is read.", "No market data source is added."),
        ("no_broker_integration", "confirmed", "No broker API, order route, or account connection is used.", "No execution path is created."),
        ("no_live_trading", "confirmed", "Outputs are CSV/Markdown position sizing reports only.", "No live trading is performed."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False is written to Step 3 outputs.", "No deployable status is claimed."),
        ("position_sizing_only", "confirmed", "The engine sizes bounded minimum-lot research positions within usable cash.", "No portfolio optimizer or order generator is implemented."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def build_report(
    sizing: pd.DataFrame,
    summary: pd.DataFrame,
    guardrails: pd.DataFrame,
    input_path: str | Path,
    output_dir: str | Path,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V5 Step 3 Position Sizing Engine Report",
            "",
            "## Executive Summary",
            "V5 Step 3 converts Step 2 tradable candidates into bounded minimum-lot research position allocations under account-level cash constraints.",
            "This is educational/research tooling only.",
            "This is not financial advice.",
            "No broker execution is performed.",
            "No order should be treated as trading-ready.",
            "The project remains not trading-ready.",
            "",
            "## Inputs",
            f"- Input path: {input_path}",
            f"- Output directory: {output_dir}",
            f"- Available cash: {row.get('available_cash', '')}",
            f"- Usable cash buffer: {row.get('usable_cash_buffer', '')}",
            f"- Usable cash: {row.get('usable_cash', '')}",
            "",
            "## Position Sizing Summary",
            f"- Candidate rows: {row.get('candidate_count', 0)}",
            f"- Sized positions: {row.get('sized_position_count', 0)}",
            f"- Deferred positions: {row.get('deferred_position_count', 0)}",
            f"- Rejected positions: {row.get('rejected_position_count', 0)}",
            f"- Total approved notional: {row.get('total_approved_notional', 0)}",
            f"- Remaining usable cash: {row.get('remaining_usable_cash', 0)}",
            "",
            "## Rules Applied",
            "- Only Step 2 tradable rows are considered by default.",
            "- MAIN board positions use 100-share lots unless an input lot_size is present.",
            "- STAR/KCB-style positions use 200-share lots unless an input lot_size is present.",
            "- Quantity is integer lot-based.",
            "- Approved notional cannot exceed available_cash times usable_cash_buffer.",
            "- Individually affordable candidates that do not fit after previous allocations are deferred.",
            "",
            "## Position Sizing Rows",
            sizing.to_markdown(index=False) if not sizing.empty else "No candidate rows were evaluated.",
            "",
            "## Guardrails",
            guardrails.to_markdown(index=False),
            "",
            "## Educational / Research Disclaimer",
            "This report is educational/research diagnostics only. It is not financial advice.",
            "No candidate, symbol, position, quantity, or output in this report should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_position_sizing_outputs(
    input_path: str | Path,
    available_cash: float,
    usable_cash_buffer: float,
    default_lot_size: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_path = Path(input_path)
    candidates = _read_tradable_candidates(source_path)
    sizing = evaluate_position_sizing(
        candidates=candidates,
        available_cash=available_cash,
        usable_cash_buffer=usable_cash_buffer,
        default_lot_size=default_lot_size,
    )
    sized = sizing[sizing["sizing_status"] == "sized"].copy()
    deferred = sizing[sizing["sizing_status"] == "deferred"].copy()
    rejected = sizing[sizing["sizing_status"] == "rejected"].copy()
    summary = build_summary(sizing, available_cash, usable_cash_buffer, default_lot_size, source_path)
    guardrails = build_guardrails()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    report = build_report(sizing, summary, guardrails, source_path, output_path)
    summary.to_csv(paths["summary"], index=False)
    sized.to_csv(paths["sized"], index=False)
    deferred.to_csv(paths["deferred"], index=False)
    rejected.to_csv(paths["rejected"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    paths["report"].write_text(report, encoding="utf-8")
    config = {
        "input_path": str(source_path),
        "output_dir": str(output_path),
        "available_cash": available_cash,
        "usable_cash_buffer": usable_cash_buffer,
        "default_lot_size": default_lot_size,
        "candidate_count": int(len(sizing)),
        "sized_position_count": int(len(sized)),
        "deferred_position_count": int(len(deferred)),
        "rejected_position_count": int(len(rejected)),
        "scope": "V5 Step 3 position sizing only",
        "broker_execution": False,
        "live_trading": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "position_sizing_summary": summary,
        "sized_positions": sized,
        "deferred_positions": deferred,
        "rejected_positions": rejected,
        "position_sizing_guardrails": guardrails,
        "position_sizing_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
