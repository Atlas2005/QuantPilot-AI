import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CAPITAL_DIR = Path("outputs/capital_constraint_engine_real_v1")
DEFAULT_UNIVERSE_DIR = Path("outputs/tradable_universe_filter_real_v1")
DEFAULT_POSITION_DIR = Path("outputs/position_sizing_engine_real_v1")
DEFAULT_EXIT_DIR = Path("outputs/exit_engine_real_v1")
DEFAULT_DAILY_PLAN_DIR = Path("outputs/daily_trading_plan_real_v1")
DEFAULT_PAPER_LEDGER_DIR = Path("outputs/paper_trading_ledger_real_v1")
DEFAULT_SEMI_AUTO_DIR = Path("outputs/semi_auto_order_generator_real_v1")
DEFAULT_BROKER_RESEARCH_DIR = Path("outputs/broker_integration_research_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/monitoring_reporting_layer_real_v1")

OUTPUT_FILENAMES = {
    "summary": "monitoring_summary.csv",
    "dashboard": "monitoring_status_dashboard.csv",
    "alerts": "monitoring_alerts.csv",
    "guardrails": "monitoring_guardrails.csv",
    "report": "monitoring_report.md",
    "run_config": "run_config.json",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
]


def _read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype or {"symbol": str})
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _numeric(value: Any, default: float = 0.0) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else default


def _bool_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())


def _first(df: pd.DataFrame, column: str, default: Any = "") -> Any:
    if df.empty or column not in df:
        return default
    return df[column].iloc[0]


def _status_row(step: str, output_dir: Path, status: str, metric: str, value: Any, notes: str) -> dict[str, Any]:
    return {
        "step": step,
        "output_dir": str(output_dir),
        "status": status,
        "metric": metric,
        "value": value,
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "notes": notes,
    }


def _alert(alert_id: str, severity: str, source: str, alert_type: str, message: str) -> dict[str, Any]:
    return {
        "alert_id": alert_id,
        "severity": severity,
        "source": source,
        "alert_type": alert_type,
        "message": message,
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "notes": "Monitoring/reporting alert only. No execution action is taken.",
    }


def collect_inputs(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        "capital_summary": _read_csv(paths["capital"] / "capital_constraint_summary.csv"),
        "capital_feasibility": _read_csv(paths["capital"] / "capital_feasibility.csv"),
        "tradable": _read_csv(paths["universe"] / "tradable_universe.csv"),
        "excluded": _read_csv(paths["universe"] / "excluded_universe.csv"),
        "universe_summary": _read_csv(paths["universe"] / "universe_filter_summary.csv"),
        "position_summary": _read_csv(paths["position"] / "position_sizing_summary.csv"),
        "sized": _read_csv(paths["position"] / "sized_positions.csv"),
        "deferred": _read_csv(paths["position"] / "deferred_positions.csv"),
        "rejected": _read_csv(paths["position"] / "rejected_positions.csv"),
        "exit_summary": _read_csv(paths["exit"] / "exit_summary.csv"),
        "exit_plan": _read_csv(paths["exit"] / "exit_plan.csv"),
        "daily_summary": _read_csv(paths["daily"] / "daily_trading_plan_summary.csv"),
        "daily_plan": _read_csv(paths["daily"] / "daily_trading_plan.csv"),
        "paper_summary": _read_csv(paths["paper"] / "paper_trading_summary.csv"),
        "paper_cash": _read_csv(paths["paper"] / "paper_cash_ledger.csv"),
        "semi_summary": _read_csv(paths["semi"] / "semi_auto_order_summary.csv"),
        "semi_drafts": _read_csv(paths["semi"] / "order_drafts.csv"),
        "broker_summary": _read_csv(paths["broker"] / "broker_integration_summary.csv"),
        "broker_modes": _read_csv(paths["broker"] / "broker_integration_modes.csv"),
        "broker_constraints": _read_csv(paths["broker"] / "broker_integration_constraints.csv"),
        "broker_risks": _read_csv(paths["broker"] / "broker_integration_risk_register.csv"),
    }


def build_dashboard(paths: dict[str, Path], data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected = {
        "V5 Step 1 Capital": paths["capital"],
        "V5 Step 2 Tradable Universe": paths["universe"],
        "V5 Step 3 Position Sizing": paths["position"],
        "V5 Step 4 Exit Engine": paths["exit"],
        "V5 Step 5 Daily Plan": paths["daily"],
        "V5 Step 6 Paper Ledger": paths["paper"],
        "V5 Step 7 Semi-Auto Orders": paths["semi"],
        "V5 Step 8 Broker Research": paths["broker"],
    }
    for step, output_dir in expected.items():
        rows.append(_status_row(step, output_dir, "present" if output_dir.exists() else "missing", "output_directory", output_dir.exists(), "Expected local V5 output directory."))

    cap = data["capital_summary"]
    daily = data["daily_summary"]
    approved = _numeric(_first(cap, "total_approved_notional", _first(daily, "total_approved_notional", 0.0)))
    available = _numeric(_first(cap, "available_cash", _first(daily, "available_cash", 0.0)))
    usable = _numeric(_first(cap, "usable_cash", _first(daily, "usable_cash", 0.0)))
    remaining = _numeric(_first(cap, "remaining_usable_cash", _first(daily, "remaining_usable_cash", 0.0)))
    rows.extend(
        [
            _status_row("capital_status", paths["capital"], "ok", "available_cash", available, "Local capital summary value."),
            _status_row("capital_status", paths["capital"], "ok", "usable_cash", usable, "Local usable cash summary value."),
            _status_row("capital_status", paths["capital"], "ok", "total_approved_notional", approved, "Approved notional from local V5 summaries."),
            _status_row("capital_status", paths["capital"], "ok", "remaining_usable_cash", remaining, "Remaining usable cash from local V5 summaries."),
            _status_row("capital_status", paths["capital"], "warning" if approved > available and available else "ok", "approved_exceeds_available_cash", approved > available if available else False, "Derived monitoring check."),
            _status_row("capital_status", paths["capital"], "warning" if approved > usable and usable else "ok", "approved_exceeds_usable_cash", approved > usable if usable else False, "Derived monitoring check."),
        ]
    )

    rows.extend(
        [
            _status_row("tradable_universe_status", paths["universe"], "ok", "candidate_count", int(_numeric(_first(data["universe_summary"], "candidate_count", len(data["tradable"]) + len(data["excluded"])))), "Step 2 candidate count."),
            _status_row("tradable_universe_status", paths["universe"], "ok", "tradable_count", len(data["tradable"]), "Rows in tradable_universe.csv."),
            _status_row("tradable_universe_status", paths["universe"], "ok", "excluded_count", len(data["excluded"]), "Rows in excluded_universe.csv."),
            _status_row("position_sizing_status", paths["position"], "ok", "sized_position_count", int(_numeric(_first(data["position_summary"], "sized_position_count", len(data["sized"])))), "Step 3 sized count."),
            _status_row("position_sizing_status", paths["position"], "ok", "deferred_position_count", int(_numeric(_first(data["position_summary"], "deferred_position_count", len(data["deferred"])))), "Step 3 deferred count."),
            _status_row("position_sizing_status", paths["position"], "ok", "rejected_position_count", int(_numeric(_first(data["position_summary"], "rejected_position_count", len(data["rejected"])))), "Step 3 rejected count."),
            _status_row("exit_plan_status", paths["exit"], "ok", "planned_exit_count", int(_numeric(_first(data["exit_summary"], "planned_exit_count", len(data["exit_plan"])))), "Step 4 planned exits."),
            _status_row("exit_plan_status", paths["exit"], "ok", "invalid_exit_plan_count", int(_numeric(_first(data["exit_summary"], "invalid_exit_plan_count", 0))), "Step 4 invalid exits."),
            _status_row("daily_plan_status", paths["daily"], "ok", "daily_plan_row_count", int(_numeric(_first(data["daily_summary"], "daily_plan_row_count", len(data["daily_plan"])))), "Step 5 plan rows."),
            _status_row("paper_ledger_status", paths["paper"], "ok", "paper_order_count", int(_numeric(_first(data["paper_summary"], "paper_order_count", 0))), "Step 6 paper orders."),
            _status_row("paper_ledger_status", paths["paper"], "ok", "paper_filled_order_count", int(_numeric(_first(data["paper_summary"], "paper_filled_order_count", 0))), "Step 6 filled paper orders."),
            _status_row("paper_ledger_status", paths["paper"], "ok", "open_paper_position_count", int(_numeric(_first(data["paper_summary"], "open_paper_position_count", 0))), "Step 6 open paper positions."),
            _status_row("paper_ledger_status", paths["paper"], "ok", "ending_cash", _numeric(_first(data["paper_summary"], "ending_cash", 0.0)), "Step 6 ending cash."),
            _status_row("semi_auto_order_status", paths["semi"], "ok", "draft_order_count", int(_numeric(_first(data["semi_summary"], "draft_order_count", 0))), "Step 7 draft orders."),
            _status_row("semi_auto_order_status", paths["semi"], "ok", "buy_draft_count", int(_numeric(_first(data["semi_summary"], "buy_draft_count", 0))), "Step 7 buy drafts."),
            _status_row("semi_auto_order_status", paths["semi"], "ok", "execution_allowed_count", int(_numeric(_first(data["semi_summary"], "execution_allowed_count", 0))), "Step 7 execution flags."),
            _status_row("semi_auto_order_status", paths["semi"], "ok", "broker_connected_count", int(_numeric(_first(data["semi_summary"], "broker_connected_count", 0))), "Step 7 broker flags."),
            _status_row("semi_auto_order_status", paths["semi"], "ok", "human_review_required_count", int(_numeric(_first(data["semi_summary"], "human_review_required_count", 0))), "Step 7 human review count."),
            _status_row("broker_research_status", paths["broker"], "ok", "researched_mode_count", int(_numeric(_first(data["broker_summary"], "researched_mode_count", len(data["broker_modes"])))), "Step 8 modes."),
            _status_row("broker_research_status", paths["broker"], "ok", "constraint_count", int(_numeric(_first(data["broker_summary"], "constraint_count", len(data["broker_constraints"])))), "Step 8 constraints."),
            _status_row("broker_research_status", paths["broker"], "ok", "high_risk_constraint_count", int(_numeric(_first(data["broker_summary"], "high_risk_constraint_count", 0))), "Step 8 high-risk constraints."),
            _status_row("broker_research_status", paths["broker"], "ok", "risk_register_count", int(_numeric(_first(data["broker_summary"], "risk_register_count", len(data["broker_risks"])))), "Step 8 risks."),
        ]
    )
    return pd.DataFrame(rows)


def scan_safety_flags(data: dict[str, pd.DataFrame]) -> dict[str, int]:
    counts = {flag: 0 for flag in SAFETY_FLAGS}
    for frame in data.values():
        for flag in SAFETY_FLAGS:
            counts[flag] += _bool_count(frame, flag)
    return counts


def build_alerts(paths: dict[str, Path], data: dict[str, pd.DataFrame], dashboard: pd.DataFrame) -> pd.DataFrame:
    alerts: list[dict[str, Any]] = []
    idx = 1
    if not dashboard.empty and "status" in dashboard:
        dashboard_alerts = dashboard[dashboard["status"].isin(["warning", "blocking"])]
        for _, row in dashboard_alerts.iterrows():
            severity = _clean_text(row.get("status"))
            alert_type = _clean_text(row.get("metric")) or "dashboard_status"
            value = row.get("value", "")
            message = (
                f"Dashboard row reported {severity}: {row.get('step')} "
                f"{alert_type}={value}."
            )
            alerts.append(_alert(f"MON-{idx:03d}", severity, row.get("step"), alert_type, message))
            idx += 1
    flag_counts = scan_safety_flags(data)
    for flag, count in flag_counts.items():
        if count > 0:
            alerts.append(_alert(f"MON-{idx:03d}", "blocking", "global_safety_status", f"{flag}_true_found", f"Found {count} true values for {flag} in input V5 outputs."))
            idx += 1
    ending_cash = _numeric(_first(data["paper_summary"], "ending_cash", 0.0))
    if ending_cash < 0 and not (
        not dashboard.empty
        and (dashboard["metric"] == "negative_ending_cash").any()
    ):
        alerts.append(_alert(f"MON-{idx:03d}", "warning", "paper_ledger_status", "negative_ending_cash", "Paper ending cash is below zero."))
        idx += 1
    if not alerts:
        alerts.append(_alert("MON-001", "info", "global_safety_status", "research_only_status_normal", "No blocking safety flags found in monitored local V5 outputs."))
    elif not any(alert["severity"] == "blocking" for alert in alerts):
        alerts.append(_alert(f"MON-{idx:03d}", "info", "global_safety_status", "research_only_status_normal", "No blocking safety flags found in monitored local V5 outputs."))
    return pd.DataFrame(alerts)


def build_summary(dashboard: pd.DataFrame, alerts: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    flag_counts = scan_safety_flags(data)
    return pd.DataFrame(
        [
            {
                "summary_item": "monitoring_reporting_layer_run",
                "monitored_step_count": int((dashboard["metric"] == "output_directory").sum()) if not dashboard.empty else 0,
                "dashboard_row_count": int(len(dashboard)),
                "alert_count": int(len(alerts)),
                "blocking_alert_count": int((alerts["severity"] == "blocking").sum()) if not alerts.empty else 0,
                "warning_alert_count": int((alerts["severity"] == "warning").sum()) if not alerts.empty else 0,
                "info_alert_count": int((alerts["severity"] == "info").sum()) if not alerts.empty else 0,
                "trading_ready_true_count": flag_counts["trading_ready"],
                "execution_allowed_true_count": flag_counts["execution_allowed"],
                "broker_connected_true_count": flag_counts["broker_connected"],
                "live_trading_true_count": flag_counts["live_trading"],
                "real_order_submission_true_count": flag_counts["real_order_submission"],
                "human_review_required": True,
                "educational_research_only": True,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
                "conclusion": "monitoring_reporting_only_no_execution",
            }
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "The layer reads existing V5 local outputs and writes monitoring reports only.", "No historical backtest is run."),
        ("no_threshold_change", "confirmed", "No signal threshold module or value is changed.", "Signal thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is called.", "Model artifacts are unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local V5 output files are read.", "No market data source is added and no live data is fetched."),
        ("no_broker_credentials", "confirmed", "The CLI does not accept credentials and the module does not request credentials.", "No account login or credential storage is implemented."),
        ("no_broker_sdk_import", "confirmed", "The module imports only standard library modules and pandas.", "No broker SDK is imported."),
        ("no_broker_connection", "confirmed", "broker_connected=False is written to Step 9 outputs.", "No broker API connection exists."),
        ("no_live_trading", "confirmed", "live_trading=False is written to Step 9 outputs.", "No live trading is performed."),
        ("no_order_execution", "confirmed", "execution_allowed=False and real_order_submission=False are written to Step 9 outputs.", "No orders are executed or submitted."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False is written to Step 9 outputs.", "No deployable status is claimed."),
        ("monitoring_reporting_only", "confirmed", "The output is a status dashboard, alerts table, and report.", "No trading workflow is advanced."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(summary: pd.DataFrame, dashboard: pd.DataFrame, alerts: pd.DataFrame, guardrails: pd.DataFrame) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V5 Step 9 Monitoring / Reporting Layer",
            "",
            "## Executive Summary",
            "V5 Step 9 reads existing local V5 Step 1-8 outputs and produces a unified research-only monitoring report.",
            "It does not run backtests, fetch market data, train or retrain models, change thresholds, change features, connect to brokers, generate real orders, submit orders, or upgrade trading readiness.",
            "All Step 9 outputs preserve broker_connected=False, execution_allowed=False, live_trading=False, real_order_submission=False, and trading_ready=False.",
            "",
            "## Key Metrics",
            f"- Monitored steps: {row.get('monitored_step_count', 0)}",
            f"- Dashboard rows: {row.get('dashboard_row_count', 0)}",
            f"- Alerts: {row.get('alert_count', 0)}",
            f"- Blocking alerts: {row.get('blocking_alert_count', 0)}",
            f"- Warning alerts: {row.get('warning_alert_count', 0)}",
            f"- Info alerts: {row.get('info_alert_count', 0)}",
            f"- trading_ready true count: {row.get('trading_ready_true_count', 0)}",
            f"- execution_allowed true count: {row.get('execution_allowed_true_count', 0)}",
            f"- broker_connected true count: {row.get('broker_connected_true_count', 0)}",
            f"- live_trading true count: {row.get('live_trading_true_count', 0)}",
            f"- real_order_submission true count: {row.get('real_order_submission_true_count', 0)}",
            "",
            "## Status Dashboard",
            _table(dashboard, "No dashboard rows were generated."),
            "",
            "## Alerts",
            _table(alerts, "No alerts were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This report is educational/research monitoring only. It is not financial advice and is not a trading-ready system.",
            "",
        ]
    )


def generate_monitoring_reporting_outputs(
    capital_dir: str | Path = DEFAULT_CAPITAL_DIR,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
    position_dir: str | Path = DEFAULT_POSITION_DIR,
    exit_dir: str | Path = DEFAULT_EXIT_DIR,
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    semi_auto_dir: str | Path = DEFAULT_SEMI_AUTO_DIR,
    broker_research_dir: str | Path = DEFAULT_BROKER_RESEARCH_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = {
        "capital": Path(capital_dir),
        "universe": Path(universe_dir),
        "position": Path(position_dir),
        "exit": Path(exit_dir),
        "daily": Path(daily_plan_dir),
        "paper": Path(paper_ledger_dir),
        "semi": Path(semi_auto_dir),
        "broker": Path(broker_research_dir),
    }
    output_path = Path(output_dir)
    data = collect_inputs(paths)
    dashboard = build_dashboard(paths, data)
    alerts = build_alerts(paths, data, dashboard)
    summary = build_summary(dashboard, alerts, data)
    guardrails = build_guardrails()
    report = build_report(summary, dashboard, alerts, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    out_paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(out_paths["summary"], index=False)
    dashboard.to_csv(out_paths["dashboard"], index=False)
    alerts.to_csv(out_paths["alerts"], index=False)
    guardrails.to_csv(out_paths["guardrails"], index=False)
    out_paths["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "dashboard_row_count": int(len(dashboard)),
        "alert_count": int(len(alerts)),
        "scope": "V5 Step 9 monitoring reporting only",
        "broker_connected": False,
        "execution_allowed": False,
        "live_trading": False,
        "real_order_submission": False,
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "monitoring_summary": summary,
        "monitoring_status_dashboard": dashboard,
        "monitoring_alerts": alerts,
        "monitoring_guardrails": guardrails,
        "monitoring_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in out_paths.items()},
    }
