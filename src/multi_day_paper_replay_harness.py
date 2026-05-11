import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DAILY_PLAN_DIR = Path("outputs/daily_trading_plan_real_v1")
DEFAULT_PAPER_LEDGER_DIR = Path("outputs/paper_trading_ledger_real_v1")
DEFAULT_ORDER_GENERATOR_DIR = Path("outputs/semi_auto_order_generator_real_v1")
DEFAULT_BROKER_RESEARCH_DIR = Path("outputs/broker_integration_research_real_v1")
DEFAULT_MONITORING_DIR = Path("outputs/monitoring_reporting_layer_real_v1")
DEFAULT_V5_CLOSURE_DIR = Path("outputs/capital_aware_infrastructure_review_real_v1")
DEFAULT_COVERAGE_GAP_DIR = Path("outputs/validation_coverage_gap_review_real_v1")
DEFAULT_SIMULATION_DESIGN_DIR = Path("outputs/simulation_hardening_design_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/multi_day_paper_replay_harness_real_v1")
DEFAULT_REPLAY_START_DATE = "2026-01-02"
DEFAULT_REPLAY_HORIZON_DAYS = 5

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "manifest": "multi_day_replay_input_manifest.csv",
    "calendar": "multi_day_replay_calendar.csv",
    "snapshots": "multi_day_replay_position_snapshots.csv",
    "event_log": "multi_day_replay_event_log.csv",
    "state_transitions": "multi_day_replay_state_transitions.csv",
    "guardrails": "multi_day_replay_guardrails.csv",
    "summary": "multi_day_replay_summary.csv",
    "report": "multi_day_replay_report.md",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
]

INPUT_DEFINITIONS = [
    ("DEP-001", "daily_plan", "daily_trading_plan.csv", "local daily plan rows"),
    ("DEP-002", "paper_ledger", "paper_positions.csv", "local open/closed paper position state"),
    ("DEP-003", "paper_ledger", "paper_trading_summary.csv", "local paper ledger summary"),
    ("DEP-004", "order_generator", "order_drafts.csv", "broker-neutral draft order context"),
    ("DEP-005", "broker_research", "broker_integration_summary.csv", "broker research guardrail context"),
    ("DEP-006", "monitoring", "monitoring_summary.csv", "monitoring/reporting context"),
    ("DEP-007", "v5_closure", "v5_infrastructure_closure_summary.csv", "V5 closure context"),
    ("DEP-008", "coverage_gaps", "validation_coverage_gap_summary.csv", "V6 coverage gap context"),
    ("DEP-009", "simulation_design", "simulation_hardening_design_summary.csv", "V6 Step 8 planning context"),
]


def build_input_paths(
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    order_generator_dir: str | Path = DEFAULT_ORDER_GENERATOR_DIR,
    broker_research_dir: str | Path = DEFAULT_BROKER_RESEARCH_DIR,
    monitoring_dir: str | Path = DEFAULT_MONITORING_DIR,
    v5_closure_dir: str | Path = DEFAULT_V5_CLOSURE_DIR,
    coverage_gap_dir: str | Path = DEFAULT_COVERAGE_GAP_DIR,
    simulation_design_dir: str | Path = DEFAULT_SIMULATION_DESIGN_DIR,
) -> dict[str, Path]:
    return {
        "daily_plan": Path(daily_plan_dir),
        "paper_ledger": Path(paper_ledger_dir),
        "order_generator": Path(order_generator_dir),
        "broker_research": Path(broker_research_dir),
        "monitoring": Path(monitoring_dir),
        "v5_closure": Path(v5_closure_dir),
        "coverage_gaps": Path(coverage_gap_dir),
        "simulation_design": Path(simulation_design_dir),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"symbol": str})
    except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _bool_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df:
        return 0
    return int(df[column].map(_is_true).sum())


def _json_flag_count(value: Any, flag: str) -> int:
    if isinstance(value, dict):
        own_count = 1 if flag in value and _is_true(value[flag]) else 0
        return own_count + sum(_json_flag_count(child, flag) for child in value.values())
    if isinstance(value, list):
        return sum(_json_flag_count(child, flag) for child in value)
    return 0


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)


def _first_present(row: pd.Series, fields: list[str], default: Any = "") -> Any:
    for field in fields:
        if field in row and not pd.isna(row[field]):
            return row[field]
    return default


def build_input_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for dependency_id, key, filename, purpose in INPUT_DEFINITIONS:
        source_dir = paths[key]
        source_file = source_dir / filename
        frame = _read_csv(source_file)
        run_config = _read_json(source_dir / "run_config.json")
        true_flag_count = sum(_bool_count(frame, flag) for flag in SAFETY_FLAGS)
        true_flag_count += sum(_json_flag_count(run_config, flag) for flag in SAFETY_FLAGS)
        rows.append(
            {
                "dependency_id": dependency_id,
                "dependency_name": key,
                "source_dir": str(source_dir),
                "source_file": str(source_file),
                "expected_file": filename,
                "source_file_exists": source_file.exists(),
                "row_count": int(len(frame)),
                "purpose": purpose,
                "input_use": "local_read_only_replay_scaffold_input",
                "forbidden_true_flag_count": int(true_flag_count),
                "market_data_fetch_count": 0,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def load_position_inputs(paths: dict[str, Path]) -> pd.DataFrame:
    positions = _read_csv(paths["paper_ledger"] / "paper_positions.csv")
    daily_plan = _read_csv(paths["daily_plan"] / "daily_trading_plan.csv")
    order_drafts = _read_csv(paths["order_generator"] / "order_drafts.csv")
    if positions.empty:
        source = daily_plan[daily_plan.get("quantity", 0).fillna(0).astype(float) > 0].copy() if "quantity" in daily_plan else pd.DataFrame()
        source_name = "daily_trading_plan"
    else:
        source = positions.copy()
        source_name = "paper_positions"

    if source.empty:
        return pd.DataFrame(
            columns=[
                "position_id",
                "source_file",
                "symbol",
                "side",
                "position_status",
                "entry_price",
                "quantity",
                "approved_notional",
                "stop_loss_price",
                "take_profit_price",
                "max_holding_days",
                "benchmark_lag_exit_rule",
            ]
        )

    draft_by_symbol = {}
    if not order_drafts.empty and "symbol" in order_drafts:
        for _, draft in order_drafts.iterrows():
            draft_by_symbol.setdefault(_safe_text(draft.get("symbol")), draft)

    rows = []
    for index, row in source.iterrows():
        symbol = _safe_text(row.get("symbol"))
        draft = draft_by_symbol.get(symbol, pd.Series(dtype=object))
        quantity = _safe_int(_first_present(row, ["quantity"], 0))
        entry_price = _safe_number(_first_present(row, ["entry_price", "limit_price"], _first_present(draft, ["limit_price"], 0.0)))
        approved_notional = _safe_number(
            _first_present(row, ["approved_notional", "estimated_notional"], _first_present(draft, ["estimated_notional"], entry_price * quantity))
        )
        max_holding_days = _safe_int(_first_present(row, ["max_holding_days"], _first_present(draft, ["max_holding_days"], DEFAULT_REPLAY_HORIZON_DAYS)), DEFAULT_REPLAY_HORIZON_DAYS)
        max_holding_days = max(1, min(max_holding_days, 20))
        rows.append(
            {
                "position_id": f"POS-{index + 1:03d}",
                "source_file": source_name,
                "symbol": symbol,
                "side": _safe_text(_first_present(row, ["side"], _first_present(draft, ["side"], "BUY")), "BUY"),
                "position_status": _safe_text(row.get("position_status"), "open_paper_position"),
                "entry_price": entry_price,
                "quantity": quantity,
                "approved_notional": approved_notional,
                "stop_loss_price": _safe_number(_first_present(row, ["stop_loss_price"], _first_present(draft, ["stop_loss_price"], 0.0))),
                "take_profit_price": _safe_number(_first_present(row, ["take_profit_price"], _first_present(draft, ["take_profit_price"], 0.0))),
                "max_holding_days": max_holding_days,
                "benchmark_lag_exit_rule": _safe_text(row.get("benchmark_lag_exit_rule"), _safe_text(_first_present(draft, ["benchmark_lag_exit_rule"], ""))),
            }
        )
    return pd.DataFrame(rows)


def build_replay_calendar(positions: pd.DataFrame, replay_start_date: str = DEFAULT_REPLAY_START_DATE) -> pd.DataFrame:
    horizon = DEFAULT_REPLAY_HORIZON_DAYS
    if not positions.empty and "max_holding_days" in positions:
        horizon = max(DEFAULT_REPLAY_HORIZON_DAYS, int(positions["max_holding_days"].fillna(DEFAULT_REPLAY_HORIZON_DAYS).astype(int).max()))
    start = datetime.strptime(replay_start_date, "%Y-%m-%d").date()
    rows = []
    for day_index in range(horizon):
        replay_date = start + timedelta(days=day_index)
        rows.append(
            {
                "replay_day_index": day_index + 1,
                "replay_date": replay_date.isoformat(),
                "calendar_source": "deterministic_scaffold_not_market_calendar",
                "day_role": "initialization_day" if day_index == 0 else "scaffold_hold_day",
                "market_data_fetch_count": 0,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_position_snapshots(positions: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, position in positions.iterrows():
        max_days = _safe_int(position.get("max_holding_days"), DEFAULT_REPLAY_HORIZON_DAYS)
        for _, day in calendar.head(max_days).iterrows():
            rows.append(
                {
                    "snapshot_id": f"SNAP-{len(rows) + 1:04d}",
                    "replay_day_index": int(day["replay_day_index"]),
                    "replay_date": day["replay_date"],
                    "position_id": position["position_id"],
                    "symbol": position["symbol"],
                    "side": position["side"],
                    "position_status": position["position_status"],
                    "entry_price": position["entry_price"],
                    "reference_price": position["entry_price"],
                    "reference_price_source": "existing_entry_price_scaffold_only",
                    "quantity": position["quantity"],
                    "approved_notional": position["approved_notional"],
                    "stop_loss_price": position["stop_loss_price"],
                    "take_profit_price": position["take_profit_price"],
                    "max_holding_days": position["max_holding_days"],
                    "days_held_scaffold": int(day["replay_day_index"]),
                    "unrealized_pnl": 0.0,
                    "market_return_calculated": False,
                    "market_data_fetch_count": 0,
                    "broker_connected": False,
                    "execution_allowed": False,
                    "live_trading": False,
                    "real_order_submission": False,
                    "trading_ready": False,
                    "notes": "Reference price is the existing entry/plan price only; no live or historical market data is fetched.",
                }
            )
    return pd.DataFrame(rows)


def build_event_log(manifest: pd.DataFrame, positions: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    loaded_count = int(manifest["source_file_exists"].astype(bool).sum()) if not manifest.empty else 0
    events = [
        ("EVT-001", "replay_initialized", "Initialized deterministic local-only multi-day paper replay scaffold."),
        ("EVT-002", "input_loaded", f"Loaded {loaded_count} existing local input dependency file(s)."),
        ("EVT-003", "position_snapshot_created", f"Created {len(snapshots)} scaffold position snapshot row(s) for {len(positions)} position(s)."),
        ("EVT-004", "no_market_data_fetch", "Confirmed zero market data fetches; entry/plan prices are reference scaffold values only."),
        ("EVT-005", "no_order_execution", "Confirmed no order execution, submission, routing, broker connection, or SDK usage."),
        ("EVT-006", "replay_scaffold_completed", "Completed output scaffold without calculating market-data-derived performance."),
    ]
    return pd.DataFrame(
        [
            {
                "event_id": event_id,
                "event_sequence": index + 1,
                "event_type": event_type,
                "event_message": message,
                "market_data_fetch_count": 0,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for index, (event_id, event_type, message) in enumerate(events)
        ]
    )


def build_state_transitions(calendar: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, day in calendar.iterrows():
        day_index = int(day["replay_day_index"])
        day_snapshot_count = int((snapshots["replay_day_index"] == day_index).sum()) if not snapshots.empty else 0
        rows.append(
            {
                "transition_id": f"STATE-{day_index:03d}",
                "replay_day_index": day_index,
                "replay_date": day["replay_date"],
                "from_state": "not_started" if day_index == 1 else "prior_day_scaffold_open",
                "to_state": "scaffold_snapshots_created",
                "transition_trigger": "deterministic_calendar_row",
                "position_snapshot_count": day_snapshot_count,
                "cash_reconciliation_status": "not_applicable_no_execution",
                "market_data_fetch_count": 0,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Step 9 creates a replay scaffold only."),
        ("no_market_data_fetch", "confirmed", "No market-data loader is imported or called."),
        ("no_threshold_change", "confirmed", "No threshold value or strategy rule is modified."),
        ("no_model_retraining", "confirmed", "No training module is imported or called."),
        ("no_feature_change", "confirmed", "No feature engineering module is imported or called."),
        ("no_new_data_sources", "confirmed", "Only existing local output directories are read."),
        ("no_broker_credentials", "confirmed", "No credential argument, path, token, account id, or secret is accepted."),
        ("no_broker_sdk_import", "confirmed", "No broker SDK is imported."),
        ("no_broker_connection", "confirmed", "The harness has no broker connection path."),
        ("no_live_trading", "confirmed", "The harness has no live trading path."),
        ("no_order_execution", "confirmed", "No order execution function exists in this scaffold."),
        ("no_real_order_submission", "confirmed", "No real order submission function exists in this scaffold."),
        ("no_trading_ready_upgrade", "confirmed", "All outputs keep readiness flags false."),
        ("replay_harness_only", "confirmed", "Outputs are manifest, calendar, snapshots, events, transitions, guardrails, summary, and report."),
        ("educational_research_only", "confirmed", "Report and config state research-only educational scope."),
    ]
    return pd.DataFrame(
        [
            {
                "guardrail": guardrail,
                "status": status,
                "evidence": evidence,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for guardrail, status, evidence in rows
        ]
    )


def build_summary(
    manifest: pd.DataFrame,
    calendar: pd.DataFrame,
    positions: pd.DataFrame,
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    missing_inputs = int((~manifest["source_file_exists"].astype(bool)).sum()) if not manifest.empty else 0
    forbidden = int(manifest["forbidden_true_flag_count"].sum()) if not manifest.empty else 0
    open_position_count = int(positions["position_status"].astype(str).str.contains("open", case=False, na=False).sum()) if not positions.empty else 0
    closed_position_count = int(len(positions) - open_position_count)
    validation_status = "pass" if missing_inputs == 0 and forbidden == 0 else "warning"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step9_multi_day_paper_replay_harness",
                "input_dependency_count": int(len(manifest)),
                "missing_input_dependency_count": missing_inputs,
                "replay_calendar_day_count": int(len(calendar)),
                "replay_position_snapshot_count": int(len(snapshots)),
                "replay_event_count": int(len(events)),
                "open_position_count": open_position_count,
                "closed_position_count": closed_position_count,
                "market_data_fetch_count": 0,
                "broker_connected_count": 0,
                "execution_allowed_count": 0,
                "live_trading_count": 0,
                "real_order_submission_count": 0,
                "forbidden_true_flag_count": forbidden,
                "trading_ready": False,
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
                "validation_status": validation_status,
                "conclusion": "multi_day_paper_replay_harness_created_research_only",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
    calendar: pd.DataFrame,
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    transitions: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 9 Multi-Day Paper Replay Harness / Replay Input Scaffold",
            "",
            "## Executive Summary",
            "V6 Step 9 creates a deterministic, local-output-only replay harness scaffold from existing daily plan and paper ledger outputs.",
            "Replay days are controlled scaffold rows, not market-data-derived performance results. Existing entry and plan prices are preserved only as reference scaffold values.",
            "This step does not fetch market data, run new historical backtests, retrain models, change thresholds, change features, connect to brokers, import broker SDKs, execute orders, submit orders, perform live trading, or mark anything trading-ready.",
            "",
            "## Summary",
            f"- Input dependencies: {row.get('input_dependency_count', 0)}",
            f"- Missing input dependencies: {row.get('missing_input_dependency_count', 0)}",
            f"- Replay calendar days: {row.get('replay_calendar_day_count', 0)}",
            f"- Position snapshots: {row.get('replay_position_snapshot_count', 0)}",
            f"- Replay events: {row.get('replay_event_count', 0)}",
            f"- Open positions: {row.get('open_position_count', 0)}",
            f"- Closed positions: {row.get('closed_position_count', 0)}",
            f"- Market data fetches: {row.get('market_data_fetch_count', 0)}",
            f"- Broker connected count: {row.get('broker_connected_count', 0)}",
            f"- Execution allowed count: {row.get('execution_allowed_count', 0)}",
            f"- Live trading count: {row.get('live_trading_count', 0)}",
            f"- Real order submission count: {row.get('real_order_submission_count', 0)}",
            f"- Trading ready: {row.get('trading_ready', False)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            "",
            "## Input Manifest",
            _table(manifest, "No input manifest rows were generated."),
            "",
            "## Replay Calendar",
            _table(calendar, "No replay calendar rows were generated."),
            "",
            "## Position Snapshots",
            _table(snapshots, "No position snapshot rows were generated."),
            "",
            "## Event Log",
            _table(events, "No event rows were generated."),
            "",
            "## State Transitions",
            _table(transitions, "No state transition rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This replay harness scaffold is educational/research-only. It is not financial advice, not a backtest, not a live or broker-connected paper trading system, and not a readiness certification.",
            "",
        ]
    )


def generate_multi_day_paper_replay_harness_outputs(
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    order_generator_dir: str | Path = DEFAULT_ORDER_GENERATOR_DIR,
    broker_research_dir: str | Path = DEFAULT_BROKER_RESEARCH_DIR,
    monitoring_dir: str | Path = DEFAULT_MONITORING_DIR,
    v5_closure_dir: str | Path = DEFAULT_V5_CLOSURE_DIR,
    coverage_gap_dir: str | Path = DEFAULT_COVERAGE_GAP_DIR,
    simulation_design_dir: str | Path = DEFAULT_SIMULATION_DESIGN_DIR,
    replay_start_date: str = DEFAULT_REPLAY_START_DATE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        daily_plan_dir=daily_plan_dir,
        paper_ledger_dir=paper_ledger_dir,
        order_generator_dir=order_generator_dir,
        broker_research_dir=broker_research_dir,
        monitoring_dir=monitoring_dir,
        v5_closure_dir=v5_closure_dir,
        coverage_gap_dir=coverage_gap_dir,
        simulation_design_dir=simulation_design_dir,
    )
    output_path = Path(output_dir)
    manifest = build_input_manifest(paths)
    positions = load_position_inputs(paths)
    calendar = build_replay_calendar(positions, replay_start_date=replay_start_date)
    snapshots = build_position_snapshots(positions, calendar)
    events = build_event_log(manifest, positions, snapshots)
    transitions = build_state_transitions(calendar, snapshots)
    guardrails = build_guardrails()
    summary = build_summary(manifest, calendar, positions, snapshots, events)
    report = build_report(summary, manifest, calendar, snapshots, events, transitions, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    manifest.to_csv(output_files["manifest"], index=False)
    calendar.to_csv(output_files["calendar"], index=False)
    snapshots.to_csv(output_files["snapshots"], index=False)
    events.to_csv(output_files["event_log"], index=False)
    transitions.to_csv(output_files["state_transitions"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    summary.to_csv(output_files["summary"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "replay_start_date": replay_start_date,
        "replay_calendar_day_count": int(len(calendar)),
        "replay_position_snapshot_count": int(len(snapshots)),
        "scope": "V6 Step 9 multi-day paper replay harness scaffold only",
        "market_data_fetch_count": 0,
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    output_files["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "multi_day_replay_summary": summary,
        "multi_day_replay_input_manifest": manifest,
        "multi_day_replay_calendar": calendar,
        "multi_day_replay_position_snapshots": snapshots,
        "multi_day_replay_event_log": events,
        "multi_day_replay_state_transitions": transitions,
        "multi_day_replay_guardrails": guardrails,
        "multi_day_replay_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
