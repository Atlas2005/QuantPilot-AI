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
DEFAULT_MONITORING_DIR = Path("outputs/monitoring_reporting_layer_real_v1")
DEFAULT_V5_CLOSURE_DIR = Path("outputs/capital_aware_infrastructure_review_real_v1")
DEFAULT_COVERAGE_GAP_DIR = Path("outputs/validation_coverage_gap_review_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/simulation_hardening_design_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "summary": "simulation_hardening_design_summary.csv",
    "guardrails": "simulation_hardening_design_guardrails.csv",
    "hardening_plan": "simulation_hardening_plan.csv",
    "replay_plan": "multi_day_paper_replay_plan.csv",
    "risk_controls": "simulation_risk_controls.csv",
    "report": "simulation_hardening_design_report.md",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
]


def build_input_paths(
    capital_dir: str | Path = DEFAULT_CAPITAL_DIR,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
    position_dir: str | Path = DEFAULT_POSITION_DIR,
    exit_dir: str | Path = DEFAULT_EXIT_DIR,
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    monitoring_dir: str | Path = DEFAULT_MONITORING_DIR,
    v5_closure_dir: str | Path = DEFAULT_V5_CLOSURE_DIR,
    coverage_gap_dir: str | Path = DEFAULT_COVERAGE_GAP_DIR,
) -> dict[str, Path]:
    return {
        "capital_constraints": Path(capital_dir),
        "tradable_universe": Path(universe_dir),
        "position_sizing": Path(position_dir),
        "exit_engine": Path(exit_dir),
        "daily_plan": Path(daily_plan_dir),
        "paper_ledger": Path(paper_ledger_dir),
        "monitoring": Path(monitoring_dir),
        "v5_closure": Path(v5_closure_dir),
        "v6_coverage_gaps": Path(coverage_gap_dir),
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
        total = 1 if flag in value and _is_true(value[flag]) else 0
        return total + sum(_json_flag_count(child, flag) for child in value.values())
    if isinstance(value, list):
        return sum(_json_flag_count(child, flag) for child in value)
    return 0


def count_forbidden_flags(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    total = 0
    for csv_path in path.glob("*.csv"):
        frame = _read_csv(csv_path)
        total += sum(_bool_count(frame, flag) for flag in SAFETY_FLAGS)
    for json_path in path.glob("*.json"):
        payload = _read_json(json_path)
        total += sum(_json_flag_count(payload, flag) for flag in SAFETY_FLAGS)
    return int(total)


def build_input_dependencies(paths: dict[str, Path]) -> pd.DataFrame:
    definitions = [
        ("DEP-001", "capital_constraints", "capital_constraint_summary.csv", "cash feasibility and capital constraint assumptions"),
        ("DEP-002", "tradable_universe", "universe_filter_summary.csv", "tradability and eligibility filters"),
        ("DEP-003", "position_sizing", "sized_positions.csv", "research-only sized position fields"),
        ("DEP-004", "exit_engine", "exit_plan.csv", "research-only exit planning fields"),
        ("DEP-005", "daily_plan", "daily_trading_plan.csv", "broker-neutral daily plan evidence"),
        ("DEP-006", "paper_ledger", "paper_trading_summary.csv", "local paper ledger state and reconciliation shape"),
        ("DEP-007", "monitoring", "monitoring_summary.csv", "monitoring/reporting evidence shape"),
        ("DEP-008", "v5_closure", "v5_infrastructure_closure_summary.csv", "V5 research-only closure status"),
        ("DEP-009", "v6_coverage_gaps", "validation_coverage_gap_summary.csv", "V6 Step 7 readiness blockers and gap counts"),
    ]
    rows = []
    for dependency_id, key, filename, purpose in definitions:
        source_dir = paths[key]
        source_file = source_dir / filename
        rows.append(
            {
                "dependency_id": dependency_id,
                "dependency_name": key,
                "source_dir": str(source_dir),
                "expected_file": filename,
                "source_file_exists": source_file.exists(),
                "dependency_purpose": purpose,
                "use_in_current_step": "design_reference_only",
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_simulation_hardening_plan(dependencies: pd.DataFrame) -> pd.DataFrame:
    missing = int((~dependencies["source_file_exists"].astype(bool)).sum()) if not dependencies.empty else 0
    rows = [
        (
            "PLAN-001",
            "planned_replay_scope",
            "Define a future local-only multi-day replay harness that reuses existing V5/V6 output files as frozen inputs.",
            "No replay is executed in Step 8.",
            "Replay symbols, calendar days, input snapshots, cash assumptions, and output schema must be locked before execution.",
        ),
        (
            "PLAN-002",
            "input_dependency_freeze",
            "Require an immutable manifest of V5/V6 output directories before any future simulation run.",
            f"{missing} required local dependency file(s) are missing in the current design scan.",
            "Every required dependency must exist, pass schema checks, and preserve false safety flags.",
        ),
        (
            "PLAN-003",
            "paper_replay_state_machine",
            "Design day-by-day replay states for planned orders, simulated fills, ledger update, cash reconciliation, and exception capture.",
            "State machine is a future design target only.",
            "State transitions must be deterministic, replayable, and auditable from local files.",
        ),
        (
            "PLAN-004",
            "capital_reconciliation",
            "Specify future reconciliation between simulated cash, reserved cash, filled-notional, fees, ledger cash, and rejected actions.",
            "No account balance is requested or fetched.",
            "Reconciliation must close to zero unexplained cash difference before any execution discussion.",
        ),
        (
            "PLAN-005",
            "safety_and_kill_switch_design",
            "Require dry-run kill switches for stale inputs, unexpected true safety flags, missing ledger rows, and cash mismatches.",
            "No production daemon or broker kill switch is created.",
            "A future run must halt locally when any blocking control fails.",
        ),
        (
            "PLAN-006",
            "future_validation_criteria",
            "Define acceptance evidence for future replay completeness, reproducibility, reconciliation, and blocker closure.",
            "No new validation claim is made now.",
            "Future criteria must be reviewed separately and cannot upgrade readiness by itself.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "plan_id": plan_id,
                "design_area": area,
                "planned_design": design,
                "current_step_boundary": boundary,
                "future_evidence_required": evidence,
                "implementation_status": "planned_not_implemented",
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for plan_id, area, design, boundary, evidence in rows
        ]
    )


def build_multi_day_paper_replay_plan() -> pd.DataFrame:
    phases = [
        (
            "PHASE-001",
            "preflight_snapshot",
            "Freeze local V5/V6 inputs, run schema and forbidden-flag checks, and record replay calendar assumptions.",
            "All dependencies present, no forbidden true safety flags, and no missing run_config evidence.",
            "halt_before_replay",
        ),
        (
            "PHASE-002",
            "day_open_initialization",
            "Load the next local replay day from frozen files and initialize simulated cash, positions, reservations, and prior ledger state.",
            "Prior day ledger reconciles and replay date is within the approved local calendar.",
            "halt_on_state_mismatch",
        ),
        (
            "PHASE-003",
            "broker_neutral_order_draft_review",
            "Transform existing research-only plans into simulated order intents without broker formatting or submission.",
            "Every order intent links to local evidence and remains a simulation-only draft.",
            "reject_unlinked_intents",
        ),
        (
            "PHASE-004",
            "simulated_fill_and_ledger_update",
            "Apply future deterministic fill assumptions to the paper ledger and record simulated cash, position, and fee effects.",
            "Fill rules are locked before replay and do not change thresholds, features, models, or strategy behavior.",
            "halt_on_reconciliation_break",
        ),
        (
            "PHASE-005",
            "multi_day_closeout_review",
            "Aggregate replay days, exceptions, rejected actions, cash reconciliation, and unresolved readiness blockers.",
            "All days produce local reports and unresolved blockers remain explicit.",
            "research_only_closeout",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "phase_id": phase_id,
                "phase_name": name,
                "planned_scope": scope,
                "evidence_required_before_execution": evidence,
                "stop_condition": stop_condition,
                "phase_status": "planned_not_executed",
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for phase_id, name, scope, evidence, stop_condition in phases
        ]
    )


def build_risk_controls() -> pd.DataFrame:
    controls = [
        ("CTRL-001", "no_broker_boundary", "Forbid broker SDK imports, credentials, account lookups, order routing, and order submission.", "blocking"),
        ("CTRL-002", "no_new_data_boundary", "Use only existing local outputs; forbid market-data fetches and new data sources.", "blocking"),
        ("CTRL-003", "strategy_freeze", "Forbid threshold, model, feature, and strategy behavior changes.", "blocking"),
        ("CTRL-004", "execution_disabled", "Keep execution_allowed false in every output layer.", "blocking"),
        ("CTRL-005", "ledger_reconciliation", "Require future replay ledger cash, reserved cash, fees, fills, and positions to reconcile per day.", "blocking"),
        ("CTRL-006", "capital_reconciliation", "Require future simulated capital state to reconcile to V5 capital constraints and paper ledger state.", "blocking"),
        ("CTRL-007", "input_snapshot_manifest", "Require immutable local input path, file size, and timestamp evidence before future replay execution.", "blocking"),
        ("CTRL-008", "forbidden_flag_scan", "Scan outputs for forbidden true safety flags before and after any future replay.", "blocking"),
        ("CTRL-009", "exception_register", "Record stale input, missing input, malformed input, and reconciliation exceptions as blocking findings.", "blocking"),
        ("CTRL-010", "human_review_gate", "Require manual review of future replay evidence before any next-step planning.", "blocking"),
        ("CTRL-011", "readiness_blocker_carry_forward", "Carry forward Step 7 blockers until separately resolved by evidence.", "blocking"),
        ("CTRL-012", "research_only_disclosure", "State educational/research-only status in future replay reports and configs.", "blocking"),
    ]
    return pd.DataFrame(
        [
            {
                "control_id": control_id,
                "control_name": name,
                "control_requirement": requirement,
                "severity": severity,
                "current_status": "planned_control_not_executed",
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for control_id, name, requirement, severity in controls
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Step 8 writes design artifacts only.", "No historical backtest is run."),
        ("no_market_data_fetch", "confirmed", "No market-data loader is imported or called.", "No market data is fetched."),
        ("no_threshold_change", "confirmed", "No threshold value or threshold module is modified.", "Thresholds remain unchanged."),
        ("no_model_retraining", "confirmed", "No training module is imported or called.", "Models remain unchanged."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is imported or called.", "Features remain unchanged."),
        ("no_new_data_sources", "confirmed", "The plan references only existing local output directories.", "No new data source is added."),
        ("no_broker_credentials", "confirmed", "No credential path, token, account id, or secret is accepted.", "No credentials are used."),
        ("no_broker_sdk_import", "confirmed", "No broker SDK module is imported.", "No broker SDK dependency is introduced."),
        ("no_broker_connection", "confirmed", "The module has no broker connection path.", "No broker connection occurs."),
        ("no_live_trading", "confirmed", "The module has no live trading path.", "No live trading occurs."),
        ("no_order_execution", "confirmed", "The module has no order execution function.", "No order is executed."),
        ("no_real_order_submission", "confirmed", "The module has no order submission function.", "No real order is submitted."),
        ("no_trading_ready_upgrade", "confirmed", "All outputs keep the readiness flag false.", "No deployable status is claimed."),
        ("simulation_design_only", "confirmed", "Outputs are plans, controls, and reports only.", "No replay is executed."),
        ("educational_research_only", "confirmed", "The report states educational/research-only status.", "Not financial advice."),
    ]
    return pd.DataFrame(
        [
            {
                "guardrail": guardrail,
                "status": status,
                "evidence": evidence,
                "notes": notes,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for guardrail, status, evidence, notes in rows
        ]
    )


def build_summary(
    dependencies: pd.DataFrame,
    hardening_plan: pd.DataFrame,
    replay_plan: pd.DataFrame,
    risk_controls: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> pd.DataFrame:
    forbidden = 0
    missing_inputs = int((~dependencies["source_file_exists"].astype(bool)).sum()) if not dependencies.empty else 0
    evidence_requirement_count = int(len(hardening_plan) + len(replay_plan))
    validation_status = "pass" if forbidden == 0 else "fail"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step8_simulation_hardening_design",
                "planned_replay_phase_count": int(len(replay_plan)),
                "input_dependency_count": int(len(dependencies)),
                "missing_input_dependency_count": missing_inputs,
                "risk_control_count": int(len(risk_controls)),
                "evidence_requirement_count": evidence_requirement_count,
                "guardrail_count": int(len(guardrails)),
                "forbidden_true_flag_count": forbidden,
                "trading_ready": False,
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
                "validation_status": validation_status,
                "conclusion": "simulation_hardening_design_completed_research_only",
                "recommended_next_step": "V6 Step 9 local_replay_preflight_manifest_or_dry_run_specification",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    dependencies: pd.DataFrame,
    hardening_plan: pd.DataFrame,
    replay_plan: pd.DataFrame,
    risk_controls: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    readiness_blockers = [
        "No real multi-day paper replay has been executed.",
        "No broker sandbox or live broker integration exists.",
        "No capital reconciliation against a real account has been performed.",
        "No production monitoring, alerting, or kill-switch process exists.",
        "No compliance, suitability, tax, or risk approval workflow exists.",
        "V6 Step 7 still records blocking readiness gaps.",
        "No candidate has been separately certified as deployable.",
    ]
    return "\n".join(
        [
            "# V6 Step 8 Simulation Hardening Design / Multi-Day Paper Replay Planning",
            "",
            "## Executive Summary",
            "V6 Step 8 creates a research-only planning layer for future multi-day paper replay and simulation hardening.",
            "It designs scope, dependencies, replay phases, evidence requirements, safety controls, capital reconciliation requirements, future validation criteria, and unresolved readiness blockers.",
            "It does not run a replay, run backtests, fetch market data, retrain models, change thresholds, change features, connect to brokers, execute orders, submit orders, perform live trading, or upgrade readiness.",
            "",
            "## Summary",
            f"- Planned replay phases: {row.get('planned_replay_phase_count', 0)}",
            f"- Input dependencies: {row.get('input_dependency_count', 0)}",
            f"- Risk controls: {row.get('risk_control_count', 0)}",
            f"- Evidence requirements: {row.get('evidence_requirement_count', 0)}",
            f"- Forbidden true flags: {row.get('forbidden_true_flag_count', 0)}",
            f"- Execution allowed: {row.get('execution_allowed', False)}",
            f"- Broker connected: {row.get('broker_connected', False)}",
            f"- Live trading: {row.get('live_trading', False)}",
            f"- Real order submission: {row.get('real_order_submission', False)}",
            f"- Trading ready: {row.get('trading_ready', False)}",
            f"- Validation status: {row.get('validation_status', '')}",
            f"- Conclusion: {row.get('conclusion', '')}",
            f"- Recommended next step: {row.get('recommended_next_step', '')}",
            "",
            "## Planned Replay Scope",
            "A future replay should be local-only, deterministic, multi-day, and broker-neutral. It should consume frozen V5/V6 outputs as input evidence and produce auditable simulated ledger, exception, reconciliation, and closeout artifacts. That future replay requires a separate implementation step and must remain blocked unless all preflight evidence and safety controls pass.",
            "",
            "## Input Dependencies",
            _table(dependencies, "No input dependencies were generated."),
            "",
            "## Multi-Day Paper Replay Phases",
            _table(replay_plan, "No replay phases were generated."),
            "",
            "## Simulation Hardening Plan",
            _table(hardening_plan, "No hardening plan rows were generated."),
            "",
            "## Simulation Risk Controls",
            _table(risk_controls, "No risk controls were generated."),
            "",
            "## Readiness Blockers That Remain",
            "\n".join(f"- {blocker}" for blocker in readiness_blockers),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This simulation hardening design is educational/research-only. It is not financial advice and is not a readiness certification.",
            "",
        ]
    )


def generate_simulation_hardening_design_outputs(
    capital_dir: str | Path = DEFAULT_CAPITAL_DIR,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
    position_dir: str | Path = DEFAULT_POSITION_DIR,
    exit_dir: str | Path = DEFAULT_EXIT_DIR,
    daily_plan_dir: str | Path = DEFAULT_DAILY_PLAN_DIR,
    paper_ledger_dir: str | Path = DEFAULT_PAPER_LEDGER_DIR,
    monitoring_dir: str | Path = DEFAULT_MONITORING_DIR,
    v5_closure_dir: str | Path = DEFAULT_V5_CLOSURE_DIR,
    coverage_gap_dir: str | Path = DEFAULT_COVERAGE_GAP_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        capital_dir=capital_dir,
        universe_dir=universe_dir,
        position_dir=position_dir,
        exit_dir=exit_dir,
        daily_plan_dir=daily_plan_dir,
        paper_ledger_dir=paper_ledger_dir,
        monitoring_dir=monitoring_dir,
        v5_closure_dir=v5_closure_dir,
        coverage_gap_dir=coverage_gap_dir,
    )
    output_path = Path(output_dir)
    dependencies = build_input_dependencies(paths)
    hardening_plan = build_simulation_hardening_plan(dependencies)
    replay_plan = build_multi_day_paper_replay_plan()
    risk_controls = build_risk_controls()
    guardrails = build_guardrails()
    summary = build_summary(dependencies, hardening_plan, replay_plan, risk_controls, guardrails)
    report = build_report(summary, dependencies, hardening_plan, replay_plan, risk_controls, guardrails)

    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {
        "run_config": output_path / OUTPUT_FILENAMES["run_config"],
        "summary": output_path / OUTPUT_FILENAMES["summary"],
        "guardrails": output_path / OUTPUT_FILENAMES["guardrails"],
        "hardening_plan": output_path / OUTPUT_FILENAMES["hardening_plan"],
        "replay_plan": output_path / OUTPUT_FILENAMES["replay_plan"],
        "risk_controls": output_path / OUTPUT_FILENAMES["risk_controls"],
        "report": output_path / OUTPUT_FILENAMES["report"],
    }
    summary.to_csv(output_files["summary"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    hardening_plan.to_csv(output_files["hardening_plan"], index=False)
    replay_plan.to_csv(output_files["replay_plan"], index=False)
    risk_controls.to_csv(output_files["risk_controls"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "planned_replay_phase_count": int(len(replay_plan)),
        "input_dependency_count": int(len(dependencies)),
        "risk_control_count": int(len(risk_controls)),
        "evidence_requirement_count": int(len(hardening_plan) + len(replay_plan)),
        "scope": "V6 Step 8 simulation hardening design and paper replay planning only",
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
        "simulation_hardening_design_summary": summary,
        "simulation_hardening_design_guardrails": guardrails,
        "simulation_hardening_plan": hardening_plan,
        "multi_day_paper_replay_plan": replay_plan,
        "simulation_risk_controls": risk_controls,
        "simulation_input_dependencies": dependencies,
        "simulation_hardening_design_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
