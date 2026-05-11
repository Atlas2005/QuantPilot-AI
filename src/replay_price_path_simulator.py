import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REPLAY_DIR = Path("outputs/multi_day_paper_replay_harness_real_v1")
DEFAULT_DESIGN_DIR = Path("outputs/simulation_hardening_design_real_v1")
DEFAULT_REVIEW_DIR = Path("outputs/simulation_hardening_review_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/replay_price_path_simulator_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "manifest": "replay_price_path_input_manifest.csv",
    "scenarios": "synthetic_price_scenarios.csv",
    "price_paths": "replay_price_paths.csv",
    "position_results": "replay_price_path_position_results.csv",
    "event_log": "replay_price_path_event_log.csv",
    "guardrails": "replay_price_path_guardrails.csv",
    "summary": "replay_price_path_summary.csv",
    "report": "replay_price_path_report.md",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
    "market_data_fetch",
]

INPUT_DEFINITIONS = [
    ("DEP-001", "replay", "multi_day_replay_position_snapshots.csv", "Step 9 scaffold position snapshots"),
    ("DEP-002", "replay", "multi_day_replay_summary.csv", "Step 9 scaffold summary"),
    ("DEP-003", "replay", "multi_day_replay_guardrails.csv", "Step 9 scaffold guardrails"),
    ("DEP-004", "design", "simulation_hardening_design_summary.csv", "Step 8 design summary evidence"),
    ("DEP-005", "design", "multi_day_paper_replay_plan.csv", "Step 8 planned replay phases"),
    ("DEP-006", "review", "simulation_hardening_review_summary.csv", "Step 10 review summary evidence"),
    ("DEP-007", "review", "simulation_hardening_readiness_blockers.csv", "Step 10 remaining readiness blockers"),
]


def build_input_paths(
    replay_dir: str | Path = DEFAULT_REPLAY_DIR,
    design_dir: str | Path = DEFAULT_DESIGN_DIR,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
) -> dict[str, Path]:
    return {
        "replay": Path(replay_dir),
        "design": Path(design_dir),
        "review": Path(review_dir),
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
        count = 1 if flag in value and _is_true(value[flag]) else 0
        return count + sum(_json_flag_count(child, flag) for child in value.values())
    if isinstance(value, list):
        return sum(_json_flag_count(child, flag) for child in value)
    return 0


def _safe_float(value: Any, default: float = 0.0) -> float:
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


def _first_row_value(df: pd.DataFrame, column: str, default: Any = 0) -> Any:
    if df.empty or column not in df:
        return default
    value = df.iloc[0][column]
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def build_input_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for dependency_id, key, filename, purpose in INPUT_DEFINITIONS:
        source_dir = paths[key]
        source_file = source_dir / filename
        frame = _read_csv(source_file)
        run_config = _read_json(source_dir / "run_config.json")
        forbidden_count = sum(_bool_count(frame, flag) for flag in SAFETY_FLAGS)
        forbidden_count += sum(_json_flag_count(run_config, flag) for flag in SAFETY_FLAGS)
        rows.append(
            {
                "dependency_id": dependency_id,
                "dependency_name": key,
                "source_dir": str(source_dir),
                "expected_file": filename,
                "source_file": str(source_file),
                "source_file_exists": source_file.exists(),
                "row_count": int(len(frame)),
                "purpose": purpose,
                "input_use": "local_read_only_synthetic_price_path_input",
                "forbidden_true_flag_count": int(forbidden_count),
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_synthetic_scenarios() -> pd.DataFrame:
    definitions = [
        ("SCN-001", "flat_path", "Hold reference price constant for all scaffold days."),
        ("SCN-002", "gradual_up_path", "Increase price deterministically toward the take-profit region."),
        ("SCN-003", "gradual_down_path", "Decrease price deterministically toward the stop-loss region."),
        ("SCN-004", "stop_loss_touch_path", "Touch the local stop-loss level during the scaffold horizon."),
        ("SCN-005", "take_profit_touch_path", "Touch the local take-profit level during the scaffold horizon."),
        ("SCN-006", "volatile_no_exit_path", "Oscillate locally while avoiding stop-loss and take-profit levels."),
    ]
    return pd.DataFrame(
        [
            {
                "scenario_id": scenario_id,
                "scenario_name": name,
                "scenario_description": description,
                "synthetic_only": True,
                "market_data_source": "none",
                "market_data_fetch": False,
                "research_only": True,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for scenario_id, name, description in definitions
        ]
    )


def _scenario_price(row: pd.Series, scenario_name: str) -> float:
    entry = _safe_float(row.get("entry_price"))
    reference = _safe_float(row.get("reference_price"), entry)
    stop = _safe_float(row.get("stop_loss_price"))
    take = _safe_float(row.get("take_profit_price"))
    day = max(1, _safe_int(row.get("replay_day_index"), 1))
    max_days = max(1, _safe_int(row.get("max_holding_days"), 1))
    progress = min(day / max_days, 1.0)
    if scenario_name == "flat_path":
        price = reference
    elif scenario_name == "gradual_up_path":
        target = take * 0.98 if take > 0 else reference * 1.04
        price = reference + (target - reference) * progress
    elif scenario_name == "gradual_down_path":
        target = stop * 1.02 if stop > 0 else reference * 0.96
        price = reference + (target - reference) * progress
    elif scenario_name == "stop_loss_touch_path":
        target = stop if stop > 0 else reference * 0.95
        touch_day = max(2, min(4, max_days))
        price = target if day >= touch_day else reference + (target - reference) * (day / touch_day) * 0.8
    elif scenario_name == "take_profit_touch_path":
        target = take if take > 0 else reference * 1.1
        touch_day = max(2, min(4, max_days))
        price = target if day >= touch_day else reference + (target - reference) * (day / touch_day) * 0.8
    elif scenario_name == "volatile_no_exit_path":
        upper = take * 0.96 if take > 0 else reference * 1.04
        lower = stop * 1.04 if stop > 0 else reference * 0.96
        price = upper if day % 2 == 0 else lower
    else:
        price = reference
    return round(float(price), 6)


def build_price_paths(snapshots: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    rows = []
    unique_snapshots = snapshots.sort_values(["position_id", "replay_day_index"]).copy()
    for _, scenario in scenarios.iterrows():
        for _, snapshot in unique_snapshots.iterrows():
            entry = _safe_float(snapshot.get("entry_price"))
            synthetic_price = _scenario_price(snapshot, str(scenario["scenario_name"]))
            synthetic_return_pct = ((synthetic_price / entry) - 1.0) * 100 if entry else 0.0
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_name": scenario["scenario_name"],
                    "replay_day_index": _safe_int(snapshot.get("replay_day_index")),
                    "replay_date": snapshot.get("replay_date", ""),
                    "position_id": snapshot.get("position_id", ""),
                    "symbol": snapshot.get("symbol", ""),
                    "side": snapshot.get("side", ""),
                    "entry_price": entry,
                    "reference_price": _safe_float(snapshot.get("reference_price"), entry),
                    "synthetic_price": synthetic_price,
                    "synthetic_return_pct": round(synthetic_return_pct, 6),
                    "stop_loss_price": _safe_float(snapshot.get("stop_loss_price")),
                    "take_profit_price": _safe_float(snapshot.get("take_profit_price")),
                    "max_holding_days": _safe_int(snapshot.get("max_holding_days")),
                    "price_source": "local_synthetic_scenario",
                    "synthetic_only": True,
                    "market_data_fetch": False,
                    "broker_connected": False,
                    "execution_allowed": False,
                    "live_trading": False,
                    "real_order_submission": False,
                    "trading_ready": False,
                }
            )
    return pd.DataFrame(rows)


def build_position_results(price_paths: pd.DataFrame) -> pd.DataFrame:
    if price_paths.empty:
        return pd.DataFrame()
    rows = []
    grouped = price_paths.sort_values("replay_day_index").groupby(["scenario_id", "scenario_name", "position_id"], sort=False)
    for (scenario_id, scenario_name, position_id), group in grouped:
        first = group.iloc[0]
        result_row = group.iloc[-1]
        exit_triggered = False
        exit_reason = "synthetic_max_holding_reached_or_no_exit"
        for _, row in group.iterrows():
            price = _safe_float(row["synthetic_price"])
            stop = _safe_float(row["stop_loss_price"])
            take = _safe_float(row["take_profit_price"])
            if stop > 0 and price <= stop:
                result_row = row
                exit_triggered = True
                exit_reason = "synthetic_stop_loss_touch"
                break
            if take > 0 and price >= take:
                result_row = row
                exit_triggered = True
                exit_reason = "synthetic_take_profit_touch"
                break
        entry = _safe_float(first["entry_price"])
        exit_price = _safe_float(result_row["synthetic_price"])
        quantity = 0.0
        synthetic_pnl = (exit_price - entry) * quantity
        synthetic_return_pct = ((exit_price / entry) - 1.0) * 100 if entry else 0.0
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "position_id": position_id,
                "symbol": first["symbol"],
                "side": first["side"],
                "entry_price": entry,
                "exit_triggered": exit_triggered,
                "exit_day_index": _safe_int(result_row["replay_day_index"]),
                "exit_price": exit_price,
                "scenario_exit_reason": exit_reason,
                "synthetic_pnl": round(synthetic_pnl, 6),
                "synthetic_return_pct": round(synthetic_return_pct, 6),
                "closed_position_simulated": exit_triggered,
                "scenario_only": True,
                "trading_ready": False,
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
            }
        )
    return pd.DataFrame(rows)


def build_event_log(manifest: pd.DataFrame, scenarios: pd.DataFrame, price_paths: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    loaded_count = int(manifest["source_file_exists"].astype(bool).sum()) if not manifest.empty else 0
    events = [
        ("EVT-001", "price_path_simulator_initialized", "Initialized local synthetic price path simulator."),
        ("EVT-002", "input_dependencies_loaded", f"Loaded {loaded_count} existing local input dependency file(s)."),
        ("EVT-003", "synthetic_scenarios_created", f"Created {len(scenarios)} deterministic synthetic scenario definition(s)."),
        ("EVT-004", "price_paths_generated", f"Generated {len(price_paths)} synthetic price path row(s)."),
        ("EVT-005", "synthetic_exit_rules_evaluated", f"Evaluated {len(results)} scenario-position result row(s)."),
        ("EVT-006", "no_market_data_fetch_confirmed", "Confirmed no live or historical market data fetch."),
        ("EVT-007", "no_order_execution_confirmed", "Confirmed no order execution, submission, routing, or broker connection."),
        ("EVT-008", "price_path_simulator_completed", "Completed research-only synthetic price path simulator."),
    ]
    return pd.DataFrame(
        [
            {
                "event_id": event_id,
                "event_sequence": index + 1,
                "event_type": event_type,
                "event_message": message,
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for index, (event_id, event_type, message) in enumerate(events)
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Step 11 generates local synthetic scenarios only."),
        ("no_market_data_fetch", "confirmed", "No market-data loader is imported or called."),
        ("no_live_data", "confirmed", "No live data path is accepted or used."),
        ("no_threshold_change", "confirmed", "No strategy threshold is modified."),
        ("no_model_retraining", "confirmed", "No training module is imported or called."),
        ("no_feature_engineering_change", "confirmed", "No feature engineering module is modified or called."),
        ("no_new_external_data_sources", "confirmed", "Only existing local V6 outputs are read."),
        ("no_broker_sdk_import", "confirmed", "No broker SDK is imported."),
        ("no_broker_credentials", "confirmed", "No credential argument, token, account id, or secret is accepted."),
        ("no_broker_connection", "confirmed", "No broker connection path exists."),
        ("no_order_execution", "confirmed", "No order execution function exists."),
        ("no_real_order_submission", "confirmed", "No real order submission path exists."),
        ("no_trading_ready_upgrade", "confirmed", "All outputs preserve trading_ready=False."),
        ("synthetic_price_path_only", "confirmed", "Outputs contain deterministic local scenario paths and scenario-only results."),
        ("educational_research_only", "confirmed", "The report states educational/research-only scope."),
    ]
    return pd.DataFrame(
        [
            {
                "guardrail": guardrail,
                "status": status,
                "evidence": evidence,
                "market_data_fetch": False,
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
    scenarios: pd.DataFrame,
    price_paths: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    missing = int((~manifest["source_file_exists"].astype(bool)).sum()) if not manifest.empty else 0
    forbidden = int(manifest["forbidden_true_flag_count"].sum()) if not manifest.empty else 0
    stop_count = int((results["scenario_exit_reason"] == "synthetic_stop_loss_touch").sum()) if not results.empty else 0
    take_count = int((results["scenario_exit_reason"] == "synthetic_take_profit_touch").sum()) if not results.empty else 0
    max_count = int((results["scenario_exit_reason"] == "synthetic_max_holding_reached_or_no_exit").sum()) if not results.empty else 0
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step11_replay_price_path_simulator",
                "input_dependency_count": int(len(manifest)),
                "missing_input_dependency_count": missing,
                "scenario_count": int(len(scenarios)),
                "price_path_row_count": int(len(price_paths)),
                "position_scenario_result_count": int(len(results)),
                "synthetic_exit_event_count": int(stop_count + take_count),
                "stop_loss_touch_result_count": stop_count,
                "take_profit_touch_result_count": take_count,
                "max_holding_or_no_exit_result_count": max_count,
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
                "validation_status": "pass" if missing == 0 and forbidden == 0 else "warning",
                "conclusion": "replay_price_path_simulator_completed_research_only",
                "recommended_next_step": "V6 Step 12 Synthetic Scenario Replay Result Review / Risk Interpretation",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
    scenarios: pd.DataFrame,
    results: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    blockers = [
        "Synthetic scenarios are not historical market replay prices.",
        "Scenario outcomes do not prove profitability.",
        "No broker paper account, cash ledger, or fill reconciliation is updated.",
        "No slippage, fill, tax, queue, or liquidity model is validated.",
        "No live monitoring, kill switch, compliance, or risk approval layer exists.",
        "No trading-ready candidate exists.",
    ]
    return "\n".join(
        [
            "# V6 Step 11 Multi-Day Replay Price Path Simulator / Local Synthetic Price Scenario Layer",
            "",
            "## Purpose",
            "Step 11 creates deterministic local synthetic price paths on top of the existing Step 9 multi-day paper replay scaffold.",
            "It observes how stop-loss, take-profit, benchmark-lag placeholders, and max-holding-day logic would behave under simple local synthetic paths.",
            "This is not a historical backtest, not live trading, not broker paper trading, and not trading-ready evidence.",
            "",
            "## Summary",
            f"- Input dependencies: {row.get('input_dependency_count', 0)}",
            f"- Missing input dependencies: {row.get('missing_input_dependency_count', 0)}",
            f"- Scenarios: {row.get('scenario_count', 0)}",
            f"- Price path rows: {row.get('price_path_row_count', 0)}",
            f"- Scenario-position results: {row.get('position_scenario_result_count', 0)}",
            f"- Synthetic exit events: {row.get('synthetic_exit_event_count', 0)}",
            f"- Stop-loss touches: {row.get('stop_loss_touch_result_count', 0)}",
            f"- Take-profit touches: {row.get('take_profit_touch_result_count', 0)}",
            f"- Max-holding/no-exit results: {row.get('max_holding_or_no_exit_result_count', 0)}",
            f"- Market data fetches: {row.get('market_data_fetch_count', 0)}",
            f"- Broker connected count: {row.get('broker_connected_count', 0)}",
            f"- Execution allowed count: {row.get('execution_allowed_count', 0)}",
            f"- Live trading count: {row.get('live_trading_count', 0)}",
            f"- Real order submission count: {row.get('real_order_submission_count', 0)}",
            f"- Trading ready: {row.get('trading_ready', False)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            f"- Recommended next step: {row.get('recommended_next_step', '')}",
            "",
            "## Inputs Used",
            _table(manifest, "No input manifest rows were generated."),
            "",
            "## Synthetic Scenario Definitions",
            _table(scenarios, "No synthetic scenarios were generated."),
            "",
            "## Scenario Outcomes",
            _table(results, "No scenario outcomes were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Remaining Blockers",
            "\n".join(f"- {blocker}" for blocker in blockers),
            "",
            "## Research-Only Warning",
            "This layer uses local synthetic prices only. It does not fetch market data, run backtests, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or mark anything trading-ready.",
            "",
        ]
    )


def generate_replay_price_path_simulator_outputs(
    replay_dir: str | Path = DEFAULT_REPLAY_DIR,
    design_dir: str | Path = DEFAULT_DESIGN_DIR,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(replay_dir=replay_dir, design_dir=design_dir, review_dir=review_dir)
    snapshots = _read_csv(paths["replay"] / "multi_day_replay_position_snapshots.csv")
    manifest = build_input_manifest(paths)
    scenarios = build_synthetic_scenarios()
    price_paths = build_price_paths(snapshots, scenarios)
    results = build_position_results(price_paths)
    event_log = build_event_log(manifest, scenarios, price_paths, results)
    guardrails = build_guardrails()
    summary = build_summary(manifest, scenarios, price_paths, results)
    report = build_report(summary, manifest, scenarios, results, guardrails)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    manifest.to_csv(output_files["manifest"], index=False)
    scenarios.to_csv(output_files["scenarios"], index=False)
    price_paths.to_csv(output_files["price_paths"], index=False)
    results.to_csv(output_files["position_results"], index=False)
    event_log.to_csv(output_files["event_log"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    summary.to_csv(output_files["summary"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "scenario_count": int(len(scenarios)),
        "price_path_row_count": int(len(price_paths)),
        "position_scenario_result_count": int(len(results)),
        "scope": "V6 Step 11 local synthetic price path scenario layer only",
        "market_data_fetch": False,
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
        "replay_price_path_summary": summary,
        "replay_price_path_input_manifest": manifest,
        "synthetic_price_scenarios": scenarios,
        "replay_price_paths": price_paths,
        "replay_price_path_position_results": results,
        "replay_price_path_event_log": event_log,
        "replay_price_path_guardrails": guardrails,
        "replay_price_path_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
