import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILENAMES = {
    "tradable": "tradable_universe.csv",
    "excluded": "excluded_universe.csv",
    "summary": "universe_filter_summary.csv",
    "guardrails": "universe_filter_guardrails.csv",
    "report": "universe_filter_report.md",
    "run_config": "run_config.json",
}


SUPPORTED_BOARDS = {"MAIN", "STAR"}


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
    text = _clean_text(value).lower()
    return text in {"true", "1", "yes", "y", "st", "*st", "suspended"}


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
                "candidate_id": "UF-001",
                "symbol": "600000",
                "side": "BUY",
                "price": 8.0,
                "board": "MAIN",
                "estimated_turnover": 5000000,
                "notes": "Affordable main-board example.",
            },
            {
                "candidate_id": "UF-002",
                "symbol": "600519",
                "side": "BUY",
                "price": 1700.0,
                "board": "MAIN",
                "estimated_turnover": 20000000,
                "notes": "High-priced main-board example expected to fail min-lot affordability.",
            },
            {
                "candidate_id": "UF-003",
                "symbol": "688001",
                "side": "BUY",
                "price": 4.0,
                "board": "STAR",
                "estimated_turnover": 3000000,
                "notes": "STAR/KCB-style example requiring 200-share lot.",
            },
            {
                "candidate_id": "UF-004",
                "symbol": "600001",
                "side": "BUY",
                "price": 5.0,
                "board": "MAIN",
                "is_st": True,
                "estimated_turnover": 2000000,
                "notes": "ST excluded example.",
            },
            {
                "candidate_id": "UF-005",
                "symbol": "600002",
                "side": "BUY",
                "price": 5.0,
                "board": "MAIN",
                "is_suspended": True,
                "estimated_turnover": 2000000,
                "notes": "Suspended excluded example.",
            },
            {
                "candidate_id": "UF-006",
                "symbol": "000001",
                "side": "BUY",
                "price": None,
                "board": "MAIN",
                "estimated_turnover": 2000000,
                "notes": "Invalid missing price example.",
            },
            {
                "candidate_id": "UF-007",
                "symbol": "600003",
                "side": "BUY",
                "price": 3.0,
                "board": "MAIN",
                "estimated_turnover": 1000,
                "minimum_turnover": 1000000,
                "notes": "Low-liquidity excluded example.",
            },
        ]
    )


def infer_board(symbol: str, board: Any) -> str:
    board_text = _clean_text(board).upper()
    if board_text in {"STAR", "KCB", "STAR_MARKET", "SCI_TECH"}:
        return "STAR"
    if symbol.startswith(("688", "689")):
        return "STAR"
    if board_text in {"", "MAIN", "SH", "SZ", "SSE", "SZSE"}:
        return "MAIN"
    return "UNSUPPORTED"


def lot_size_for_symbol(symbol: str, board: Any, default_lot_size: int) -> tuple[int, str]:
    inferred = infer_board(symbol, board)
    if inferred == "STAR":
        return 200, "star_or_kcb_min_lot"
    if inferred == "MAIN":
        return int(default_lot_size), "default_main_board_lot"
    return int(default_lot_size), "unsupported_or_requires_rule"


def evaluate_tradable_universe(
    candidates: pd.DataFrame,
    cash: float,
    buffer: float = 0.97,
    default_lot_size: int = 100,
    min_turnover: float | None = None,
) -> pd.DataFrame:
    rows = []
    for idx, row in candidates.reset_index(drop=True).iterrows():
        symbol = _format_symbol(row.get("symbol"))
        side = _clean_text(row.get("side")).upper() or "BUY"
        price = _numeric(row.get("price"))
        board = infer_board(symbol, row.get("board"))
        lot_size, lot_rule = lot_size_for_symbol(symbol, row.get("board"), default_lot_size)
        available_after_buffer = float(cash) * float(buffer) if pd.notna(cash) else float("nan")
        min_lot_notional = float(price) * lot_size if pd.notna(price) else float("nan")
        min_required_cash = min_lot_notional if pd.notna(min_lot_notional) else float("nan")
        usable_cash = available_after_buffer if pd.notna(available_after_buffer) else float("nan")
        row_min_turnover = _numeric(row.get("minimum_turnover"))
        turnover_threshold = float(row_min_turnover) if pd.notna(row_min_turnover) else min_turnover
        estimated_turnover = _numeric(row.get("estimated_turnover"))
        exclusion_reasons = []

        if not symbol or not (symbol.isdigit() and len(symbol) == 6):
            exclusion_reasons.append("missing_or_invalid_symbol")
        if pd.isna(price) or float(price) <= 0:
            exclusion_reasons.append("invalid_or_missing_price")
        if side != "BUY":
            exclusion_reasons.append("unsupported_side_for_step2")
        if board not in SUPPORTED_BOARDS:
            exclusion_reasons.append("unsupported_board")
        if _bool_value(row.get("is_st")) or _bool_value(row.get("st_flag")):
            exclusion_reasons.append("st_or_star_st_flag")
        if _bool_value(row.get("is_suspended")) or _bool_value(row.get("suspended")):
            exclusion_reasons.append("suspended")
        if pd.isna(cash) or float(cash) <= 0:
            exclusion_reasons.append("invalid_cash")
        elif pd.notna(min_lot_notional) and min_lot_notional > usable_cash:
            exclusion_reasons.append("insufficient_cash_for_min_lot")
        if turnover_threshold is not None and pd.notna(estimated_turnover) and estimated_turnover < float(turnover_threshold):
            exclusion_reasons.append("liquidity_below_min_turnover")

        is_tradable = len(exclusion_reasons) == 0
        max_affordable_price = usable_cash / lot_size if pd.notna(usable_cash) and lot_size > 0 else float("nan")
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
                "minimum_lot_notional": min_lot_notional,
                "maximum_affordable_price": max_affordable_price,
                "estimated_turnover": estimated_turnover,
                "minimum_turnover_threshold": turnover_threshold,
                "is_st": _bool_value(row.get("is_st")) or _bool_value(row.get("st_flag")),
                "is_suspended": _bool_value(row.get("is_suspended")) or _bool_value(row.get("suspended")),
                "tradable": bool(is_tradable),
                "exclusion_reason": "tradable" if is_tradable else ";".join(exclusion_reasons),
                "trading_ready": False,
                "notes": "Tradable universe eligibility only. No position sizing, order generation, broker execution, or live trading.",
            }
        )
    return pd.DataFrame(rows)


def build_summary(universe: pd.DataFrame, cash: float, buffer: float, default_lot_size: int, min_turnover: float | None) -> pd.DataFrame:
    tradable_count = int(universe["tradable"].fillna(False).astype(bool).sum()) if not universe.empty else 0
    excluded_count = int(len(universe) - tradable_count)
    return pd.DataFrame(
        [
            {
                "summary_item": "tradable_universe_filter_run",
                "candidate_count": int(len(universe)),
                "tradable_count": tradable_count,
                "excluded_count": excluded_count,
                "available_cash": cash,
                "usable_cash_buffer": buffer,
                "default_lot_size": default_lot_size,
                "min_turnover": min_turnover,
                "trading_ready": False,
                "conclusion": "V5 Step 2 filters practical tradable-universe eligibility only. Project remains not trading-ready.",
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Universe filter evaluates candidate rows only.", "No historical backtest is run."),
        ("no_threshold_change", "confirmed", "No strategy threshold module or value is changed.", "Signal thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Default candidates are deterministic educational rows, or a user-provided local CSV is read.", "No market data source is added."),
        ("no_broker_integration", "confirmed", "No broker API, order route, or account connection is used.", "No execution path is created."),
        ("no_live_trading", "confirmed", "Outputs are CSV/Markdown universe filter reports only.", "No live trading is performed."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False is written to Step 2 outputs.", "No deployable status is claimed."),
        ("universe_filter_only", "confirmed", "The engine filters eligibility before position sizing or order generation.", "No portfolio allocation is solved."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def build_report(
    universe: pd.DataFrame,
    summary: pd.DataFrame,
    guardrails: pd.DataFrame,
    cash: float,
    buffer: float,
    default_lot_size: int,
    min_turnover: float | None,
    input_candidates: str | None,
    output_dir: str | Path,
) -> str:
    tradable_count = int(summary.iloc[0].get("tradable_count")) if not summary.empty else 0
    excluded_count = int(summary.iloc[0].get("excluded_count")) if not summary.empty else 0
    return "\n".join(
        [
            "# V5 Step 2 Tradable Universe Filter Report",
            "",
            "## Executive Summary",
            "V5 Step 2 filters candidate universe eligibility before position sizing or order generation.",
            "This is educational/research tooling only.",
            "This is not financial advice.",
            "No broker execution is performed.",
            "No candidate should be treated as trading-ready.",
            "The project remains not trading-ready.",
            "",
            "## Inputs",
            f"- Input candidates: {input_candidates or 'deterministic educational defaults'}",
            f"- Available cash: {cash}",
            f"- Usable cash buffer: {buffer}",
            f"- Default lot size: {default_lot_size}",
            f"- Minimum turnover threshold: {min_turnover if min_turnover is not None else 'row-specific only when provided'}",
            f"- Output directory: {output_dir}",
            "",
            "## Universe Filter Summary",
            f"- Candidate rows: {len(universe)}",
            f"- Tradable rows: {tradable_count}",
            f"- Excluded rows: {excluded_count}",
            "",
            "## Rules Applied",
            "- Missing or invalid six-digit symbols are excluded.",
            "- Missing or non-positive prices are excluded.",
            "- Non-BUY sides are excluded in Step 2.",
            "- Unsupported boards are excluded.",
            "- ST and suspended rows are excluded.",
            "- MAIN board uses a 100-share default minimum lot.",
            "- STAR/KCB-style rows use a 200-share minimum lot.",
            "- Minimum required cash is price times lot size for valid price rows.",
            "- Minimum-lot affordability compares minimum_required_cash against available_cash times buffer.",
            "- Liquidity is checked when a global or row-specific turnover threshold is provided.",
            "",
            "## Universe Rows",
            universe.to_markdown(index=False) if not universe.empty else "No candidate rows were evaluated.",
            "",
            "## Guardrails",
            guardrails.to_markdown(index=False),
            "",
            "## Educational / Research Disclaimer",
            "This report is educational/research diagnostics only. It is not financial advice.",
            "No candidate, symbol, universe row, or output in this report should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_tradable_universe_outputs(
    input_candidates: str | Path | None,
    cash: float,
    buffer: float,
    default_lot_size: int,
    min_turnover: float | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_candidates) if input_candidates else None
    candidates = _read_candidates(input_path)
    universe = evaluate_tradable_universe(
        candidates=candidates,
        cash=cash,
        buffer=buffer,
        default_lot_size=default_lot_size,
        min_turnover=min_turnover,
    )
    tradable = universe[universe["tradable"].fillna(False).astype(bool)].copy()
    excluded = universe[~universe["tradable"].fillna(False).astype(bool)].copy()
    summary = build_summary(universe, cash, buffer, default_lot_size, min_turnover)
    guardrails = build_guardrails()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    report = build_report(universe, summary, guardrails, cash, buffer, default_lot_size, min_turnover, str(input_path) if input_path else None, output_path)
    tradable.to_csv(paths["tradable"], index=False)
    excluded.to_csv(paths["excluded"], index=False)
    summary.to_csv(paths["summary"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    paths["report"].write_text(report, encoding="utf-8")
    config = {
        "input_candidates": str(input_path) if input_path else None,
        "used_default_educational_candidates": input_path is None,
        "cash": cash,
        "buffer": buffer,
        "default_lot_size": default_lot_size,
        "min_turnover": min_turnover,
        "output_dir": str(output_path),
        "candidate_count": int(len(universe)),
        "tradable_count": int(len(tradable)),
        "excluded_count": int(len(excluded)),
        "scope": "V5 Step 2 tradable universe eligibility only",
        "broker_execution": False,
        "live_trading": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "tradable_universe": tradable,
        "excluded_universe": excluded,
        "universe_filter_summary": summary,
        "universe_filter_guardrails": guardrails,
        "universe_filter_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
