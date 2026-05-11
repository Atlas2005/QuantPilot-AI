import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/simulation_hardening_closure_real_v1")

V6_STEPS = [
    ("V6 Step 1", "validation_baseline_manifest", Path("outputs/validation_baseline_manifest_real_v1"), "validation_baseline_summary.csv", "Validation baseline manifest"),
    ("V6 Step 2", "output_schema_validator", Path("outputs/output_schema_validator_real_v1"), "output_schema_validation_summary.csv", "Output/schema validation"),
    ("V6 Step 3", "cross_step_dependency_validator", Path("outputs/cross_step_dependency_validator_real_v1"), "cross_step_dependency_summary.csv", "Cross-step dependency validation"),
    ("V6 Step 4", "reproducibility_rerun_validator", Path("outputs/reproducibility_rerun_validator_real_v1"), "reproducibility_rerun_summary.csv", "Reproducibility rerun check"),
    ("V6 Step 5", "reproducibility_warning_triage", Path("outputs/reproducibility_warning_triage_real_v1"), "reproducibility_warning_triage_summary.csv", "Reproducibility warning triage"),
    ("V6 Step 6", "validation_evidence_index", Path("outputs/validation_evidence_index_real_v1"), "validation_evidence_summary.csv", "Validation evidence index"),
    ("V6 Step 7", "validation_coverage_gap_review", Path("outputs/validation_coverage_gap_review_real_v1"), "validation_coverage_gap_summary.csv", "Coverage gap review"),
    ("V6 Step 8", "simulation_hardening_design", Path("outputs/simulation_hardening_design_real_v1"), "simulation_hardening_design_summary.csv", "Simulation hardening design"),
    ("V6 Step 9", "multi_day_paper_replay_harness", Path("outputs/multi_day_paper_replay_harness_real_v1"), "multi_day_replay_summary.csv", "Multi-day paper replay scaffold"),
    ("V6 Step 10", "simulation_hardening_review", Path("outputs/simulation_hardening_review_real_v1"), "simulation_hardening_review_summary.csv", "Simulation hardening review closure"),
    ("V6 Step 11", "replay_price_path_simulator", Path("outputs/replay_price_path_simulator_real_v1"), "replay_price_path_summary.csv", "Synthetic price path simulator"),
    ("V6 Step 12", "synthetic_replay_result_review", Path("outputs/synthetic_replay_result_review_real_v1"), "synthetic_replay_result_summary.csv", "Synthetic replay result review"),
    ("V6 Step 13", "synthetic_replay_stress_matrix", Path("outputs/synthetic_replay_stress_matrix_real_v1"), "synthetic_replay_stress_matrix_summary.csv", "Synthetic stress matrix design"),
    ("V6 Step 14", "synthetic_stress_scenario_generator", Path("outputs/synthetic_stress_scenario_generator_real_v1"), "synthetic_stress_summary.csv", "Local synthetic scenario generator"),
]

OUTPUT_FILENAMES = {
    "run_config": "run_config.json",
    "manifest": "v6_closure_input_manifest.csv",
    "inventory": "v6_completed_step_inventory.csv",
    "capabilities": "v6_capability_summary.csv",
    "gaps": "v6_remaining_gap_register.csv",
    "transition": "v6_transition_to_v7_plan.csv",
    "reuse_policy": "v6_open_source_reuse_policy.csv",
    "guardrails": "v6_closure_guardrails.csv",
    "summary": "v6_closure_summary.csv",
    "report": "v6_closure_report.md",
}

SAFETY_FLAGS = [
    "trading_ready",
    "execution_allowed",
    "broker_connected",
    "live_trading",
    "real_order_submission",
    "market_data_fetch",
]


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


def _first_row_value(df: pd.DataFrame, column: str, default: Any = "") -> Any:
    if df.empty or column not in df:
        return default
    value = df.iloc[0][column]
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def build_input_manifest() -> pd.DataFrame:
    rows = []
    for step_label, step_key, output_dir, summary_file, description in V6_STEPS:
        summary_path = output_dir / summary_file
        summary = _read_csv(summary_path)
        run_config = _read_json(output_dir / "run_config.json")
        forbidden = sum(_bool_count(summary, flag) for flag in SAFETY_FLAGS)
        forbidden += sum(_json_flag_count(run_config, flag) for flag in SAFETY_FLAGS)
        rows.append(
            {
                "input_id": f"INPUT-{len(rows) + 1:03d}",
                "step_label": step_label,
                "step_key": step_key,
                "output_dir": str(output_dir),
                "summary_file": summary_file,
                "summary_file_exists": summary_path.exists(),
                "summary_row_count": int(len(summary)),
                "description": description,
                "conclusion": _first_row_value(summary, "conclusion", ""),
                "validation_status": _first_row_value(summary, "validation_status", ""),
                "forbidden_true_flag_count": int(forbidden),
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
        )
    return pd.DataFrame(rows)


def build_completed_step_inventory(manifest: pd.DataFrame) -> pd.DataFrame:
    capability_map = {
        "V6 Step 1": "Preserved baseline references to local validation/simulation inputs.",
        "V6 Step 2": "Validated expected local output files and schemas.",
        "V6 Step 3": "Checked cross-step dependency integrity across local outputs.",
        "V6 Step 4": "Reran selected deterministic local commands and compared outputs.",
        "V6 Step 5": "Triaged reproducibility warnings into acceptable normalized differences.",
        "V6 Step 6": "Built evidence index and traceability catalog.",
        "V6 Step 7": "Reviewed validation coverage gaps and readiness blockers.",
        "V6 Step 8": "Designed simulation hardening and multi-day replay plan.",
        "V6 Step 9": "Created deterministic local multi-day paper replay scaffold.",
        "V6 Step 10": "Reviewed and closed simulation scaffold as research-only.",
        "V6 Step 11": "Created local synthetic price path scenario layer.",
        "V6 Step 12": "Classified synthetic replay outcomes and scenario risks.",
        "V6 Step 13": "Designed synthetic stress matrix and scenario expansion plan.",
        "V6 Step 14": "Generated local synthetic stress scenario definitions and assumptions.",
    }
    rows = []
    for _, row in manifest.iterrows():
        rows.append(
            {
                "step_label": row["step_label"],
                "step_key": row["step_key"],
                "completion_status": "completed" if bool(row["summary_file_exists"]) else "missing",
                "primary_artifact": row["summary_file"],
                "capability_added": capability_map.get(row["step_label"], row["description"]),
                "research_only_boundary": "No trading readiness, broker execution, live data, or real order path added.",
                "trading_ready": False,
                "execution_allowed": False,
                "broker_connected": False,
                "live_trading": False,
                "real_order_submission": False,
            }
        )
    return pd.DataFrame(rows)


def build_capability_summary() -> pd.DataFrame:
    rows = [
        ("CAP-001", "output_schema_validation", "V6 added local output/schema validation coverage.", "research_infrastructure", "does_not_validate_alpha"),
        ("CAP-002", "cross_step_dependency_validation", "V6 added dependency checks across local V5/V6 artifacts.", "research_infrastructure", "does_not_validate_market_behavior"),
        ("CAP-003", "reproducibility_check_and_warning_triage", "V6 added deterministic rerun checks and warning triage.", "research_infrastructure", "limited_to_selected_local_commands"),
        ("CAP-004", "validation_evidence_index", "V6 added a local evidence catalog and traceability matrix.", "auditability", "does_not_create_new_evidence_quality"),
        ("CAP-005", "coverage_gap_review", "V6 identified remaining readiness blockers explicitly.", "risk_visibility", "does_not_close_blockers"),
        ("CAP-006", "multi_day_paper_replay_scaffold", "V6 created deterministic local multi-day paper replay scaffold rows.", "simulation_scaffold", "not_broker_paper_trading"),
        ("CAP-007", "synthetic_price_path_simulator", "V6 added local synthetic price path scenarios on top of scaffold positions.", "synthetic_scenario_analysis", "not_real_market_replay_prices"),
        ("CAP-008", "synthetic_replay_result_review", "V6 classified stop-loss, take-profit, and max-holding behavior under synthetic scenarios.", "risk_classification", "not_profitability_evidence"),
        ("CAP-009", "synthetic_stress_matrix_design", "V6 planned broader synthetic stress dimensions.", "scenario_planning", "not_executed_simulations"),
        ("CAP-010", "local_synthetic_scenario_generator", "V6 converted stress matrix rows into local scenario definitions and assumptions.", "scenario_metadata", "not_real_market_evidence"),
    ]
    return pd.DataFrame(
        [
            {
                "capability_id": cap_id,
                "capability_name": name,
                "v6_achievement": achievement,
                "capability_type": cap_type,
                "important_limitation": limitation,
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for cap_id, name, achievement, cap_type, limitation in rows
        ]
    )


def build_remaining_gap_register() -> pd.DataFrame:
    gaps = [
        ("GAP-001", "no_real_alpha_evidence", "blocking", "V6 did not prove a predictive or profitable edge.", "Future V7+ alpha validation must use robust data, OOS testing, and benchmarks."),
        ("GAP-002", "no_real_market_replay_prices", "blocking", "Synthetic scenarios are not historical or live replay prices.", "Build reliable market data foundation before real replay validation."),
        ("GAP-003", "no_realistic_a_share_execution_model", "blocking", "V6 did not model realistic A-share limit-up/down, lot, T+1, suspension, or auction behavior.", "Evaluate mature backtest/trading-rule engines and A-share adaptations."),
        ("GAP-004", "no_transaction_cost_slippage_engine", "blocking", "V6 did not validate cost, slippage, fill, fee, tax, queue, or liquidity models.", "Adopt or adapt a realistic cost and fill model."),
        ("GAP-005", "no_walk_forward_oos_validation", "blocking", "V6 did not establish walk-forward or out-of-sample validation.", "Create V7 validation protocol after data and framework selection."),
        ("GAP-006", "no_portfolio_engine", "blocking", "V6 did not implement portfolio construction, risk allocation, or exposure controls.", "Evaluate portfolio/risk libraries before custom implementation."),
        ("GAP-007", "no_sustained_paper_trading_feedback_loop", "blocking", "V6 did not collect sustained forward paper trading feedback.", "Design paper feedback only after data/backtest/factor foundation matures."),
        ("GAP-008", "no_broker_sandbox_or_live_validation", "blocking", "V6 did not connect broker sandbox or live validation.", "Keep broker work deferred until research foundation and compliance controls exist."),
        ("GAP-009", "no_trading_ready_candidate", "blocking", "No candidate is trading-ready.", "Keep all outputs research-only."),
    ]
    return pd.DataFrame(
        [
            {
                "gap_id": gap_id,
                "gap_name": name,
                "severity": severity,
                "v6_status": "not_proven_or_not_implemented",
                "evidence": evidence,
                "required_future_resolution": resolution,
                "trading_ready_blocker": True,
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for gap_id, name, severity, evidence, resolution in gaps
        ]
    )


def build_transition_plan() -> pd.DataFrame:
    rows = [
        ("V7-001", "Open-source Quant Stack Audit / Framework Selection", "Evaluate mature quant stacks before building major modules.", "Qlib, LEAN, vectorbt, RQAlpha, Backtrader, Alphalens/quantstats-style tools", "high"),
        ("V7-002", "A-share Data Asset Map", "Map available local/approved A-share data assets, coverage, schema, calendars, and corporate action needs.", "AkShare/Baostock outputs, exchange calendars, local CSV conventions", "high"),
        ("V7-003", "Data Quality Validator", "Validate OHLCV integrity, missing days, suspensions, splits, limits, and schema consistency.", "pandera/Great Expectations-style validation where appropriate", "high"),
        ("V7-004", "Realistic A-share Trading Rule Engine", "Model A-share market rules before trusting simulations.", "RQAlpha/LEAN/Backtrader adapters or custom thin A-share rule layer", "high"),
        ("V7-005", "Realistic Backtest Engine Evaluation", "Benchmark mature engines against project constraints before custom implementation.", "Qlib, LEAN, vectorbt, RQAlpha, Backtrader", "high"),
        ("V7-006", "Factor/Alpha Evaluation Framework", "Evaluate factor IC, turnover, decay, and grouped performance with mature tooling first.", "Qlib, Alphalens-style, quantstats-style reporting", "high"),
        ("V7-007", "Strategy Tournament Design", "Define fair candidate comparison, baselines, and promotion rules.", "Qlib workflow ideas, custom orchestration only where needed", "medium"),
        ("V7-008", "Walk-forward/OOS Validation", "Create locked walk-forward and out-of-sample validation protocol.", "Qlib/MLflow-style experiment tracking where useful", "high"),
        ("V7-009", "Portfolio/Risk Allocation Engine", "Evaluate mature portfolio/risk methods before custom code.", "PyPortfolioOpt/riskfolio-style concepts or selected quant stack portfolio modules", "medium"),
        ("V7-010", "Paper Trading Feedback Loop", "Design feedback collection after data/backtest/factor foundations are ready.", "Broker-neutral logs first; broker integrations deferred", "medium"),
        ("V7-011", "Later Multi-agent Orchestration", "Consider agent orchestration only after foundations are ready.", "RD-Agent/LangGraph/AutoGen-type orchestration later", "low"),
    ]
    return pd.DataFrame(
        [
            {
                "transition_id": transition_id,
                "v7_workstream": workstream,
                "transition_goal": goal,
                "open_source_first_evaluation": tools,
                "priority": priority,
                "custom_code_boundary": "Prefer integration/adaptation over new core infrastructure until audit justifies custom code.",
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for transition_id, workstream, goal, tools, priority in rows
        ]
    )


def build_open_source_reuse_policy() -> pd.DataFrame:
    rows = [
        ("POL-001", "open_source_first", "Use mature open-source libraries when clearly better than custom code.", "required_for_major_v7_modules"),
        ("POL-002", "qlib_evaluation", "Evaluate Qlib for data, factor, model, workflow, and benchmark research patterns.", "evaluate_before_custom_factor_stack"),
        ("POL-003", "lean_evaluation", "Evaluate LEAN for robust backtesting architecture and market rule modeling ideas.", "evaluate_before_custom_engine"),
        ("POL-004", "vectorbt_evaluation", "Evaluate vectorbt for fast vectorized research and signal exploration.", "evaluate_for_research_speed"),
        ("POL-005", "rqalpha_backtrader_evaluation", "Evaluate RQAlpha and Backtrader for A-share/backtest fit and adapter costs.", "evaluate_for_a_share_simulation"),
        ("POL-006", "alphalens_quantstats_style_tools", "Evaluate Alphalens/quantstats-style analysis for factor and performance reports.", "evaluate_before_custom_reporting"),
        ("POL-007", "agent_orchestration_later", "Evaluate RD-Agent/LangGraph/AutoGen-type orchestration only after data/backtest/factor foundations are ready.", "defer_agents"),
        ("POL-008", "custom_code_focus", "Custom code should focus on integration, A-share adaptation, proprietary alpha logic, small-capital constraints, and project-specific orchestration.", "allowed_when_project_specific"),
    ]
    return pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                "policy_name": name,
                "policy_statement": statement,
                "application_rule": rule,
                "market_data_fetch": False,
                "broker_connected": False,
                "execution_allowed": False,
                "live_trading": False,
                "real_order_submission": False,
                "trading_ready": False,
            }
            for policy_id, name, statement, rule in rows
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Step 15 is closure and transition planning only."),
        ("no_market_data_fetch", "confirmed", "No market-data loader is imported or called."),
        ("no_live_data", "confirmed", "No live data path is accepted or used."),
        ("no_model_retraining", "confirmed", "No training module is imported or called."),
        ("no_threshold_change", "confirmed", "No strategy threshold is modified."),
        ("no_feature_engineering_change", "confirmed", "No feature engineering module is modified or called."),
        ("no_new_external_data_sources", "confirmed", "Only existing local V6 outputs are read."),
        ("no_broker_sdk_import", "confirmed", "No broker SDK is imported."),
        ("no_broker_credentials", "confirmed", "No credential argument, token, account id, or secret is accepted."),
        ("no_broker_connection", "confirmed", "No broker connection path exists."),
        ("no_order_execution", "confirmed", "No order execution function exists."),
        ("no_real_order_submission", "confirmed", "No real order submission path exists."),
        ("no_trading_ready_upgrade", "confirmed", "All outputs preserve trading_ready=False."),
        ("closure_only", "confirmed", "Outputs close V6 and transition to V7 planning only."),
        ("open_source_reuse_policy_recorded", "confirmed", "Open-source-first policy is recorded for V7."),
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
    inventory: pd.DataFrame,
    gaps: pd.DataFrame,
    transition: pd.DataFrame,
    reuse_policy: pd.DataFrame,
) -> pd.DataFrame:
    missing = int((~manifest["summary_file_exists"].astype(bool)).sum()) if not manifest.empty else 0
    forbidden = int(manifest["forbidden_true_flag_count"].sum()) if not manifest.empty else 0
    validation_status = "pass" if missing == 0 and forbidden == 0 and len(inventory) == 14 else "warning"
    return pd.DataFrame(
        [
            {
                "summary_item": "v6_step15_simulation_hardening_closure",
                "reviewed_input_count": int(manifest["summary_file_exists"].astype(bool).sum()) if not manifest.empty else 0,
                "missing_input_count": missing,
                "completed_v6_step_count": int((inventory["completion_status"] == "completed").sum()) if not inventory.empty else 0,
                "remaining_gap_count": int(len(gaps)),
                "transition_plan_row_count": int(len(transition)),
                "open_source_policy_row_count": int(len(reuse_policy)),
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
                "conclusion": "v6_simulation_hardening_closed_research_only",
                "recommended_next_step": "V7 Step 1 Open-source Quant Stack Audit / Framework Selection",
            }
        ]
    )


def _table(df: pd.DataFrame, empty_message: str) -> str:
    return df.to_markdown(index=False) if not df.empty else empty_message


def build_report(
    summary: pd.DataFrame,
    inventory: pd.DataFrame,
    capabilities: pd.DataFrame,
    gaps: pd.DataFrame,
    transition: pd.DataFrame,
    reuse_policy: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    did_not_prove = [
        "No real alpha evidence.",
        "No real market replay prices.",
        "No realistic A-share execution model.",
        "No transaction cost or slippage engine.",
        "No walk-forward or out-of-sample validation.",
        "No portfolio engine.",
        "No sustained paper trading feedback loop.",
        "No broker sandbox or live validation.",
        "No trading-ready candidate.",
    ]
    return "\n".join(
        [
            "# V6 Step 15 Simulation Hardening Closure / Transition to Data & Market Reality Foundation",
            "",
            "## Executive Summary",
            "V6 validation and simulation hardening is closed as a research-only infrastructure phase.",
            "V6 improved auditability, local output validation, reproducibility review, evidence indexing, gap visibility, replay scaffolding, and synthetic scenario planning.",
            "V6 does not establish trading readiness, real market validation, broker paper trading evidence, or alpha evidence.",
            "V7 should begin with an open-source quant stack audit before major custom infrastructure work.",
            "",
            "## Summary",
            f"- Reviewed inputs: {row.get('reviewed_input_count', 0)}",
            f"- Missing inputs: {row.get('missing_input_count', 0)}",
            f"- Completed V6 steps: {row.get('completed_v6_step_count', 0)}",
            f"- Remaining gaps: {row.get('remaining_gap_count', 0)}",
            f"- Transition plan rows: {row.get('transition_plan_row_count', 0)}",
            f"- Open-source policy rows: {row.get('open_source_policy_row_count', 0)}",
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
            "## Completed V6 Step Inventory",
            _table(inventory, "No inventory rows were generated."),
            "",
            "## V6 Capability Summary",
            _table(capabilities, "No capability rows were generated."),
            "",
            "## What V6 Did Not Prove",
            "\n".join(f"- {item}" for item in did_not_prove),
            "",
            "## Remaining Gap Register",
            _table(gaps, "No gap rows were generated."),
            "",
            "## Transition To V7 Plan",
            _table(transition, "No transition rows were generated."),
            "",
            "## Open-source Reuse Policy",
            _table(reuse_policy, "No reuse policy rows were generated."),
            "",
            "## Guardrails",
            _table(guardrails, "No guardrail rows were generated."),
            "",
            "## Research-Only Warning",
            "This closure does not implement new backtesting engines, data engines, factor engines, agent systems, market data fetches, broker connections, model training, threshold changes, feature changes, order execution, order submission, or trading-ready claims.",
            "",
        ]
    )


def generate_simulation_hardening_closure_outputs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = build_input_manifest()
    inventory = build_completed_step_inventory(manifest)
    capabilities = build_capability_summary()
    gaps = build_remaining_gap_register()
    transition = build_transition_plan()
    reuse_policy = build_open_source_reuse_policy()
    guardrails = build_guardrails()
    summary = build_summary(manifest, inventory, gaps, transition, reuse_policy)
    report = build_report(summary, inventory, capabilities, gaps, transition, reuse_policy, guardrails)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_files = {label: output_path / filename for label, filename in OUTPUT_FILENAMES.items()}
    manifest.to_csv(output_files["manifest"], index=False)
    inventory.to_csv(output_files["inventory"], index=False)
    capabilities.to_csv(output_files["capabilities"], index=False)
    gaps.to_csv(output_files["gaps"], index=False)
    transition.to_csv(output_files["transition"], index=False)
    reuse_policy.to_csv(output_files["reuse_policy"], index=False)
    guardrails.to_csv(output_files["guardrails"], index=False)
    summary.to_csv(output_files["summary"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    config = {
        "output_dir": str(output_path),
        "reviewed_v6_step_count": int(len(V6_STEPS)),
        "completed_v6_step_count": int(summary.iloc[0]["completed_v6_step_count"]),
        "scope": "V6 Step 15 closure and V7 transition planning only",
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
        "v6_closure_summary": summary,
        "v6_closure_input_manifest": manifest,
        "v6_completed_step_inventory": inventory,
        "v6_capability_summary": capabilities,
        "v6_remaining_gap_register": gaps,
        "v6_transition_to_v7_plan": transition,
        "v6_open_source_reuse_policy": reuse_policy,
        "v6_closure_guardrails": guardrails,
        "v6_closure_report": report,
        "run_config": config,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }
