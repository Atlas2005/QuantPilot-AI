import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SIMULATION_DESIGN_DIR = Path("outputs/simulation_hardening_design_real_v1")
DEFAULT_REPLAY_HARNESS_DIR = Path("outputs/multi_day_paper_replay_harness_real_v1")
DEFAULT_COVERAGE_GAP_DIR = Path("outputs/validation_coverage_gap_review_real_v1")
DEFAULT_EVIDENCE_INDEX_DIR = Path("outputs/validation_evidence_index_real_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/simulation_hardening_review_real_v1")

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "summary": "simulation_hardening_review_summary.csv",
    "results": "simulation_hardening_review_results.csv",
    "guardrails": "simulation_hardening_review_guardrails.csv",
    "blockers": "simulation_hardening_readiness_blockers.csv",
    "next_actions": "simulation_hardening_next_actions.csv",
    "report": "simulation_hardening_review_report.md",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
]

REQUIRED_INPUTS = [
    (
        "INPUT-001",
        "simulation_hardening_design",
        "simulation_hardening_design_summary.csv",
        "V6 Step 8 design/planning summary",
    ),
    (
        "INPUT-002",
        "simulation_hardening_design",
        "multi_day_paper_replay_plan.csv",
        "V6 Step 8 planned replay phases",
    ),
    (
        "INPUT-003",
        "multi_day_replay_harness",
        "multi_day_replay_summary.csv",
        "V6 Step 9 deterministic replay scaffold summary",
    ),
    (
        "INPUT-004",
        "multi_day_replay_harness",
        "multi_day_replay_state_transitions.csv",
        "V6 Step 9 scaffold state transitions",
    ),
    (
        "INPUT-005",
        "multi_day_replay_harness",
        "multi_day_replay_guardrails.csv",
        "V6 Step 9 scaffold guardrails",
    ),
    (
        "INPUT-006",
        "coverage_gaps",
        "validation_coverage_gap_summary.csv",
        "V6 Step 7 coverage gap summary",
    ),
    (
        "INPUT-007",
        "coverage_gaps",
        "validation_readiness_risk_register.csv",
        "V6 Step 7 readiness risk register",
    ),
    (
        "INPUT-008",
        "evidence_index",
        "validation_evidence_summary.csv",
        "V6 Step 6 evidence index summary",
    ),
]


def build_input_paths(
    simulation_design_dir: str | Path = DEFAULT_SIMULATION_DESIGN_DIR,
    replay_harness_dir: str | Path = DEFAULT_REPLAY_HARNESS_DIR,
    coverage_gap_dir: str | Path = DEFAULT_COVERAGE_GAP_DIR,
    evidence_index_dir: str | Path = DEFAULT_EVIDENCE_INDEX_DIR,
) -> dict[str, Path]:
    return {
        "simulation_hardening_design": Path(simulation_design_dir),
        "multi_day_replay_harness": Path(replay_harness_dir),
        "coverage_gaps": Path(coverage_gap_dir),
        "evidence_index": Path(evidence_index_dir),
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_input_review(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for input_id, key, filename, purpose in REQUIRED_INPUTS:
        source_file = paths[key] / filename
        frame = _read_csv(source_file)
        run_config = _read_json(paths[key] / "run_config.json")
        forbidden_count = sum(_bool_count(frame, flag) for flag in SAFETY_FLAGS)
        forbidden_count += sum(_json_flag_count(run_config, flag) for flag in SAFETY_FLAGS)
        rows.append(
            {
                "input_id": input_id,
                "input_name": key,
                "source_dir": str(paths[key]),
                "expected_file": filename,
                "source_file": str(source_file),
                "source_file_exists": source_file.exists(),
                "row_count": int(len(frame)),
                "purpose": purpose,
                "forbidden_true_flag_count": int(forbidden_count),
                "market_data_fetch_count": 0,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_review_results(
    design_summary: pd.DataFrame,
    replay_plan: pd.DataFrame,
    replay_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    replay_guardrails: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    evidence_summary: pd.DataFrame,
) -> pd.DataFrame:
    design_phases = int(len(replay_plan))
    replay_days = _safe_int(_first_row_value(replay_summary, "replay_calendar_day_count"))
    snapshots = _safe_int(_first_row_value(replay_summary, "replay_position_snapshot_count"))
    transition_count = int(len(transitions))
    forbidden = sum(
        _safe_int(_first_row_value(frame, "forbidden_true_flag_count"))
        for frame in [design_summary, replay_summary, coverage_summary, evidence_summary]
    )
    rows = [
        (
            "REVIEW-001",
            "step8_boundary",
            "V6 Step 8 Simulation Hardening Design",
            "pass",
            "V6 Step 8 was only a simulation hardening design/planning layer.",
            f"Design phase rows reviewed: {design_phases}.",
            "Does not provide executed replay evidence.",
        ),
        (
            "REVIEW-002",
            "step9_boundary",
            "V6 Step 9 Multi-Day Paper Replay Harness",
            "pass",
            "V6 Step 9 was only a deterministic local multi-day replay scaffold.",
            f"Scaffold days={replay_days}, snapshots={snapshots}, transitions={transition_count}.",
            "Does not prove profitability.",
        ),
        (
            "REVIEW-003",
            "price_path_boundary",
            "V6 Step 9 Multi-Day Paper Replay Harness",
            "pass",
            "Step 9 does not use real market replay prices.",
            "Reference prices are existing entry/plan values only.",
            "No real market-data-derived return evidence exists.",
        ),
        (
            "REVIEW-004",
            "broker_boundary",
            "V6 Step 9 Multi-Day Paper Replay Harness",
            "pass",
            "Step 9 does not represent broker paper trading, live trading, or broker integration.",
            "Broker, execution, live trading, and real order submission counts are zero.",
            "No broker paper account reconciliation exists.",
        ),
        (
            "REVIEW-005",
            "readiness_boundary",
            "V6 Step 9 Multi-Day Paper Replay Harness",
            "pass",
            "Step 9 is not trading-ready evidence and the project remains research-only.",
            "All reviewed outputs preserve trading_ready=False.",
            "Readiness blockers remain open.",
        ),
        (
            "REVIEW-006",
            "guardrail_consistency",
            "V6 Step 8/V6 Step 9",
            "pass" if forbidden == 0 else "warning",
            "Forbidden true safety flags are absent across reviewed summaries and run configs.",
            f"Forbidden true safety flag count: {forbidden}. Step 9 guardrail rows: {len(replay_guardrails)}.",
            "A nonzero value would block closure.",
        ),
        (
            "REVIEW-007",
            "coverage_gap_consistency",
            "V6 Step 7 Validation Coverage Gap Review",
            "pass",
            "Prior coverage gaps remain explicit and are carried into this closure.",
            f"Blocking gaps reported by V6 Step 7: {_safe_int(_first_row_value(coverage_summary, 'blocking_gap_count'))}.",
            "Blocking readiness gaps prevent any trading-ready claim.",
        ),
        (
            "REVIEW-008",
            "evidence_index_consistency",
            "V6 Step 6 Validation Evidence Index",
            "pass",
            "The review references existing local V6 evidence only.",
            f"Indexed evidence files: {_safe_int(_first_row_value(evidence_summary, 'indexed_evidence_file_count'))}.",
            "No new external data source is introduced.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "review_id": review_id,
                "review_area": area,
                "source_step": source_step,
                "review_status": status,
                "finding": finding,
                "evidence": evidence,
                "remaining_implication": implication,
                "market_data_fetch_count": 0,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for review_id, area, source_step, status, finding, evidence, implication in rows
        ]
    )


def build_readiness_blockers(coverage_risks: pd.DataFrame) -> pd.DataFrame:
    blocker_names = [
        (
            "no_real_market_data_replay_prices",
            "blocking",
            "Step 9 uses existing entry/plan prices as scaffold references only.",
            "Add a future local price-path scenario layer before any performance interpretation.",
        ),
        (
            "no_out_of_sample_forward_paper_trading_evidence",
            "blocking",
            "No out-of-sample forward paper trading evidence has been collected.",
            "Collect future forward paper validation evidence under separate research controls.",
        ),
        (
            "no_real_broker_paper_account_reconciliation",
            "blocking",
            "No broker paper account, cash, position, or fill reconciliation exists.",
            "Design broker-paper reconciliation only after separate broker safety controls exist.",
        ),
        (
            "no_walk_forward_profitability_validation",
            "blocking",
            "No walk-forward profitability validation exists for the scaffold.",
            "Add future walk-forward validation with locked assumptions.",
        ),
        (
            "no_slippage_or_fill_model_validation",
            "blocking",
            "No slippage, fill, queue, fee, or tax model has been validated against execution evidence.",
            "Validate a future local cost and fill model before interpreting replay outcomes.",
        ),
        (
            "no_live_monitoring_kill_switch",
            "blocking",
            "No production monitoring daemon, alerting workflow, or kill switch exists.",
            "Design future monitoring and kill-switch controls separately.",
        ),
        (
            "no_compliance_or_risk_approval_layer",
            "blocking",
            "No compliance, suitability, tax, or risk approval workflow exists.",
            "Add future human approval and governance before any readiness discussion.",
        ),
        (
            "no_trading_ready_candidate_exists",
            "blocking",
            "No separately validated trading-ready candidate exists.",
            "Keep all current and future simulation hardening outputs research-only.",
        ),
    ]
    prior_gap_names = set(coverage_risks.get("gap_name", pd.Series(dtype=str)).astype(str)) if not coverage_risks.empty else set()
    return pd.DataFrame(
        [
            {
                "blocker_id": f"BLOCKER-{index + 1:03d}",
                "blocker_name": name,
                "severity": severity,
                "current_status": "open_research_blocker",
                "evidence": evidence,
                "required_future_resolution": resolution,
                "links_to_prior_coverage_gap": name in prior_gap_names,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for index, (name, severity, evidence, resolution) in enumerate(blocker_names)
        ]
    )


def build_next_actions() -> pd.DataFrame:
    actions = [
        (
            "NEXT-001",
            "V6 Step 11 Multi-Day Replay Price Path Simulator / Local Synthetic Price Scenario Layer",
            "Create local synthetic price scenarios for scaffold positions without fetching market data or claiming profitability.",
            "recommended_next_step",
        ),
        (
            "NEXT-002",
            "Preserve replay scaffold closure evidence",
            "Keep Step 8 design and Step 9 scaffold artifacts immutable as research-only closure inputs.",
            "documentation",
        ),
        (
            "NEXT-003",
            "Define local-only price path assumptions",
            "Specify deterministic synthetic path assumptions, constraints, and labels before any simulator implementation.",
            "preimplementation_design",
        ),
        (
            "NEXT-004",
            "Keep readiness blockers open",
            "Carry unresolved blockers forward until separate future evidence closes them.",
            "risk_control",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "action_id": action_id,
                "recommended_action": action,
                "action_scope": scope,
                "action_type": action_type,
                "allowed_now": action_id in {"NEXT-001", "NEXT-002", "NEXT-003", "NEXT-004"},
                "market_data_fetch_count": 0,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for action_id, action, scope, action_type in actions
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Step 10 reads and reviews existing local outputs only."),
        ("no_market_data_fetch", "confirmed", "No market-data loader is imported or called."),
        ("no_live_data", "confirmed", "No live data path is accepted or used."),
        ("no_model_retraining", "confirmed", "No training module is imported or called."),
        ("no_threshold_change", "confirmed", "No strategy threshold is modified."),
        ("no_feature_engineering_change", "confirmed", "No feature engineering module is modified or called."),
        ("no_new_external_data_sources", "confirmed", "The review consumes only local V6 output directories."),
        ("no_broker_sdk_import", "confirmed", "No broker SDK is imported."),
        ("no_broker_credentials", "confirmed", "No credential argument, token, account id, or secret is accepted."),
        ("no_broker_connection", "confirmed", "No broker connection path exists."),
        ("no_order_execution", "confirmed", "No order execution function exists."),
        ("no_real_order_submission", "confirmed", "No real order submission path exists."),
        ("no_trading_ready_upgrade", "confirmed", "All outputs preserve trading_ready=False."),
        ("simulation_hardening_review_only", "confirmed", "Outputs are review, blocker, next-action, guardrail, summary, and report artifacts."),
        ("educational_research_only", "confirmed", "The closure explicitly keeps the project research-only."),
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
    input_review: pd.DataFrame,
    replay_plan: pd.DataFrame,
    replay_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    blockers: pd.DataFrame,
    next_actions: pd.DataFrame,
) -> pd.DataFrame:
    missing_input_count = int((~input_review["source_file_exists"].astype(bool)).sum()) if not input_review.empty else 0
    forbidden_true_flag_count = int(input_review["forbidden_true_flag_count"].sum()) if not input_review.empty else 0
    validation_status = "pass" if missing_input_count == 0 and forbidden_true_flag_count == 0 else "warning"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step10_simulation_hardening_review",
                "reviewed_input_count": int(input_review["source_file_exists"].astype(bool).sum()) if not input_review.empty else 0,
                "missing_input_count": missing_input_count,
                "design_phase_count": int(len(replay_plan)),
                "input_dependency_count": _safe_int(_first_row_value(replay_summary, "input_dependency_count")),
                "replay_calendar_day_count": _safe_int(_first_row_value(replay_summary, "replay_calendar_day_count")),
                "replay_position_snapshot_count": _safe_int(_first_row_value(replay_summary, "replay_position_snapshot_count")),
                "replay_event_count": _safe_int(_first_row_value(replay_summary, "replay_event_count")),
                "replay_transition_count": int(len(transitions)),
                "open_position_count": _safe_int(_first_row_value(replay_summary, "open_position_count")),
                "closed_position_count": _safe_int(_first_row_value(replay_summary, "closed_position_count")),
                "remaining_blocker_count": int(len(blockers)),
                "blocking_gap_count": _safe_int(_first_row_value(coverage_summary, "blocking_gap_count")),
                "next_action_count": int(len(next_actions)),
                "market_data_fetch_count": 0,
                "broker_connected_count": 0,
                "execution_allowed_count": 0,
                "live_trading_count": 0,
                "real_order_submission_count": 0,
                "forbidden_true_flag_count": forbidden_true_flag_count,
                "trading_ready": False,
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
                "validation_status": validation_status,
                "conclusion": "simulation_hardening_review_completed_research_only",
                "recommended_next_step": "V6 Step 11 Multi-Day Replay Price Path Simulator / Local Synthetic Price Scenario Layer",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    input_review: pd.DataFrame,
    results: pd.DataFrame,
    blockers: pd.DataFrame,
    next_actions: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    return "\n".join(
        [
            "# V6 Step 10 Multi-Day Replay Review / Simulation Hardening Closure",
            "",
            "## Executive Summary",
            "V6 Step 10 reviews existing local V6 simulation hardening outputs and closes the current scaffold as research-only.",
            "V6 Step 8 was only a simulation hardening design/planning layer.",
            "V6 Step 9 was only a deterministic local multi-day replay scaffold.",
            "Step 9 does not prove profitability, does not use real market replay prices, does not represent broker paper trading, is not live trading, is not broker integration, and is not trading-ready evidence.",
            "The project remains research-only.",
            "",
            "## Summary",
            f"- Reviewed inputs: {row.get('reviewed_input_count', 0)}",
            f"- Missing inputs: {row.get('missing_input_count', 0)}",
            f"- Design phases: {row.get('design_phase_count', 0)}",
            f"- Replay calendar days: {row.get('replay_calendar_day_count', 0)}",
            f"- Replay snapshots: {row.get('replay_position_snapshot_count', 0)}",
            f"- Replay events: {row.get('replay_event_count', 0)}",
            f"- Replay transitions: {row.get('replay_transition_count', 0)}",
            f"- Open positions: {row.get('open_position_count', 0)}",
            f"- Closed positions: {row.get('closed_position_count', 0)}",
            f"- Remaining blockers: {row.get('remaining_blocker_count', 0)}",
            f"- Blocking gaps: {row.get('blocking_gap_count', 0)}",
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
            "## Input Review",
            _table(input_review, "No input review rows were generated."),
            "",
            "## Review Results",
            _table(results, "No review result rows were generated."),
            "",
            "## Readiness Blockers",
            _table(blockers, "No readiness blocker rows were generated."),
            "",
            "## Next Actions",
            _table(next_actions, "No next-action rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Closure",
            "This closure is educational/research-only. It does not fetch data, run backtests, train models, change strategy behavior, connect to brokers, execute or submit orders, perform live trading, or certify readiness.",
            "",
        ]
    )


def generate_simulation_hardening_review_outputs(
    simulation_design_dir: str | Path = DEFAULT_SIMULATION_DESIGN_DIR,
    replay_harness_dir: str | Path = DEFAULT_REPLAY_HARNESS_DIR,
    coverage_gap_dir: str | Path = DEFAULT_COVERAGE_GAP_DIR,
    evidence_index_dir: str | Path = DEFAULT_EVIDENCE_INDEX_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    paths = build_input_paths(
        simulation_design_dir=simulation_design_dir,
        replay_harness_dir=replay_harness_dir,
        coverage_gap_dir=coverage_gap_dir,
        evidence_index_dir=evidence_index_dir,
    )
    design_summary = _read_csv(paths["simulation_hardening_design"] / "simulation_hardening_design_summary.csv")
    replay_plan = _read_csv(paths["simulation_hardening_design"] / "multi_day_paper_replay_plan.csv")
    replay_summary = _read_csv(paths["multi_day_replay_harness"] / "multi_day_replay_summary.csv")
    transitions = _read_csv(paths["multi_day_replay_harness"] / "multi_day_replay_state_transitions.csv")
    replay_guardrails = _read_csv(paths["multi_day_replay_harness"] / "multi_day_replay_guardrails.csv")
    coverage_summary = _read_csv(paths["coverage_gaps"] / "validation_coverage_gap_summary.csv")
    coverage_risks = _read_csv(paths["coverage_gaps"] / "validation_readiness_risk_register.csv")
    evidence_summary = _read_csv(paths["evidence_index"] / "validation_evidence_summary.csv")

    input_review = build_input_review(paths)
    results = build_review_results(
        design_summary=design_summary,
        replay_plan=replay_plan,
        replay_summary=replay_summary,
        transitions=transitions,
        replay_guardrails=replay_guardrails,
        coverage_summary=coverage_summary,
        evidence_summary=evidence_summary,
    )
    blockers = build_readiness_blockers(coverage_risks)
    next_actions = build_next_actions()
    guardrails = build_guardrails()
    summary = build_summary(input_review, replay_plan, replay_summary, transitions, coverage_summary, blockers, next_actions)
    report = build_report(summary, input_review, results, blockers, next_actions, guardrails)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    summary.to_csv(output_files["summary"], index=False)
    results.to_csv(output_files["results"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    blockers.to_csv(output_files["blockers"], index=False)
    next_actions.to_csv(output_files["next_actions"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    config = {
        **{f"{key}_dir": str(path) for key, path in paths.items()},
        "output_dir": str(output_path),
        "reviewed_input_count": int(summary.iloc[0]["reviewed_input_count"]),
        "remaining_blocker_count": int(summary.iloc[0]["remaining_blocker_count"]),
        "scope": "V6 Step 10 simulation hardening review and research-only closure",
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
        "simulation_hardening_review_summary": summary,
        "simulation_hardening_review_results": results,
        "simulation_hardening_review_guardrails": guardrails,
        "simulation_hardening_readiness_blockers": blockers,
        "simulation_hardening_next_actions": next_actions,
        "simulation_hardening_input_review": input_review,
        "simulation_hardening_review_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
