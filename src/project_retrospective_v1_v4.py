import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILENAMES = {
    "report": "project_retrospective_v1_v4_report.md",
    "phase_progress": "phase_progress_summary.csv",
    "architecture": "architecture_layer_summary.csv",
    "capabilities": "current_capability_inventory.csv",
    "conclusions": "reliable_conclusions.csv",
    "limitations": "unresolved_limitations.csv",
    "next_phase": "recommended_next_phase.csv",
    "guardrails": "project_retrospective_guardrails.csv",
    "run_config": "run_config.json",
}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _exists(root: Path, relative_path: str) -> bool:
    return (root / relative_path).exists()


def _module_evidence(root: Path, names: list[str]) -> str:
    present = [name for name in names if _exists(root, f"src/{name}")]
    return ", ".join(f"src/{name}" for name in present) if present else "not directly verified"


def _output_evidence(root: Path, names: list[str]) -> str:
    present = [name for name in names if _exists(root, f"outputs/{name}")]
    return ", ".join(f"outputs/{name}" for name in present) if present else "not directly verified"


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _phase_context(root: Path) -> dict[str, Any]:
    src_names = sorted(path.name for path in (root / "src").glob("*.py")) if (root / "src").exists() else []
    output_names = sorted(path.name for path in (root / "outputs").iterdir() if path.is_dir()) if (root / "outputs").exists() else []
    return {
        "src_names": src_names,
        "output_names": output_names,
        "readme": _read_text(root / "README.md"),
        "app_exists": _exists(root, "app.py"),
        "smoke_exists": _exists(root, "src/run_smoke_tests.py"),
    }


def build_phase_progress_summary(root: Path, context: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "phase": "V1",
            "phase_goal": "Baseline educational backtesting foundation.",
            "completed_capabilities": "Sample/demo data workflow, technical indicators, MA crossover signals, long-only backtest, basic metrics, trade metrics, reports, and dashboard foundation.",
            "representative_modules_or_outputs": _module_evidence(root, ["indicators.py", "strategy.py", "backtester.py", "metrics.py", "trade_metrics.py", "report_generator.py", "run_demo.py", "run_stock_backtest.py"]),
            "status": "completed_foundation",
            "conclusion": "V1 appears complete as a baseline educational project foundation, not as a production trading system.",
        },
        {
            "phase": "V2",
            "phase_goal": "Real-data and ML-oriented pipeline foundation.",
            "completed_capabilities": "Real A-share data loading, factor construction, chronological dataset splitting, baseline model training, prediction, model evaluation, ML signal backtest, threshold experiments, and robustness reports.",
            "representative_modules_or_outputs": _module_evidence(root, ["real_data_loader.py", "factor_builder.py", "build_factor_dataset.py", "dataset_splitter.py", "model_trainer.py", "model_predictor.py", "model_evaluator.py", "ml_signal_backtester.py", "ml_threshold_experiment.py", "batch_model_trainer.py"]),
            "status": "completed_or_partially_inferred_from_current_structure",
            "conclusion": "V2 is partially inferred from README and current module structure; the available files support a real-data and ML research pipeline, but not production readiness.",
        },
        {
            "phase": "V3",
            "phase_goal": "Validation, robustness, candidate comparison, and regime-aware research infrastructure.",
            "completed_capabilities": "Candidate validation, stress tests, equivalence audit, candidate normalization, canonical revalidation, validation gate, failure analysis, targeted remediation design, factor ablation, pruning, and threshold decision reporting.",
            "representative_modules_or_outputs": _module_evidence(root, ["candidate_expanded_validation.py", "candidate_stress_test.py", "candidate_equivalence_audit.py", "candidate_mode_normalization.py", "canonical_candidate_revalidation_report.py", "candidate_validation_gate.py", "validation_gate_failure_analysis.py", "targeted_remediation_design.py", "factor_ablation.py", "factor_pruning_experiment.py", "threshold_decision_report.py"]),
            "status": "completed_or_partially_inferred_from_current_structure",
            "conclusion": "V3 is partially inferred from README, modules, and output directories; it expanded research validation infrastructure but did not produce a deployable candidate.",
        },
        {
            "phase": "V4",
            "phase_goal": "Strategy remediation diagnostics and conservative closure of the bull/sideways remediation cycle.",
            "completed_capabilities": "Bull and sideways remediation diagnostics, integrated remediation review, bull failure attribution, trade/window diagnostics, error pattern classification, prototype design, prototype harness, controlled prototype simulation, and Step 43 closure review.",
            "representative_modules_or_outputs": "; ".join(
                [
                    _module_evidence(root, ["integrated_remediation_revalidation.py", "bull_regime_failure_drilldown.py", "bull_trade_window_diagnostics.py", "bull_error_pattern_remediation_design.py", "bull_remediation_prototype_design.py", "bull_prototype_experiment_harness.py", "bull_prototype_controlled_backtest.py", "bull_prototype_result_review.py"]),
                    _output_evidence(root, ["integrated_remediation_revalidation_real_v1", "bull_regime_failure_drilldown_real_v1", "bull_trade_window_diagnostics_real_v1", "bull_error_pattern_remediation_design_real_v1", "bull_remediation_prototype_design_real_v1", "bull_prototype_experiment_harness_real_v1", "bull_prototype_controlled_backtest_real_v1", "bull_prototype_result_review_real_v1"]),
                ]
            ),
            "status": "completed_as_research_diagnostic_validation_cycle",
            "conclusion": "V4 is complete as a research-diagnostic validation cycle. It did not produce a trading-ready strategy.",
        },
    ]
    return pd.DataFrame(rows)


def build_architecture_layer_summary(root: Path) -> pd.DataFrame:
    rows = [
        ("Data Layer", "available", _module_evidence(root, ["real_data_loader.py", "data_loader.py", "run_demo.py"]), "research_usable", "Tradability, data quality, corporate-action, and universe filters remain future work."),
        ("Feature / Factor Layer", "available", _module_evidence(root, ["factor_builder.py", "build_factor_dataset.py", "feature_source_registry.py", "feature_implementation_queue.py"]), "research_usable", "Future feature additions need strict leakage controls and validation."),
        ("Signal Engine", "available", _module_evidence(root, ["strategy.py", "ml_signal_backtester.py", "model_predictor.py"]), "research_usable", "Signals are not connected to capital-aware execution planning."),
        ("Backtest Engine", "available", _module_evidence(root, ["backtester.py", "run_stock_backtest.py", "run_batch_experiment.py", "run_period_experiment.py"]), "research_usable", "Portfolio-level capital allocation and live execution assumptions remain limited."),
        ("Validation / Diagnostics Layer", "strong", _module_evidence(root, ["candidate_validation_gate.py", "validation_gate_failure_analysis.py", "integrated_remediation_revalidation.py", "bull_prototype_result_review.py"]), "strong_research_diagnostics", "Continue keeping validation separate from deployment claims."),
        ("Regime Analysis Layer", "available", _module_evidence(root, ["bull_regime_threshold_remediation.py", "sideways_regime_trade_sufficiency_remediation.py", "bull_regime_failure_drilldown.py"]), "research_usable", "Regime-specific findings remain diagnostic and sample-limited."),
        ("Research Prototype Layer", "available", _module_evidence(root, ["bull_remediation_prototype_design.py", "bull_prototype_experiment_harness.py", "bull_prototype_controlled_backtest.py", "bull_prototype_result_review.py"]), "strong_research_diagnostics", "No prototype is deployable; future prototypes need broader validation before any status change."),
        ("Capital Constraint Layer", "missing", "No dedicated capital constraint module found.", "not_implemented", "Recommended V5 Step 1."),
        ("Portfolio Engine", "missing", "No capital-aware portfolio construction module found.", "not_implemented", "Build after capital constraints and tradable universe filters."),
        ("Risk Engine", "partial", "Backtester supports simple transaction costs and optional trade-level risk controls, but no live portfolio risk engine exists.", "basic_backtest_only", "Design capital-aware risk limits, exposure caps, and drawdown controls."),
        ("Execution Engine", "missing", "No broker execution or order management layer found.", "not_implemented", "Keep broker integration as later research, not autonomous trading."),
        ("Monitoring Layer", "missing", "No live or paper monitoring layer found.", "not_implemented", "Add paper-ledger and monitoring only after V5 planning layers."),
        ("Agent / Self-Research Layer", "missing", "No agent orchestration module or self-iterating research layer found.", "not_implemented", "Do not add agents until core utility infrastructure and governance exist."),
    ]
    return pd.DataFrame(
        [{"layer": layer, "current_status": status, "evidence": evidence, "maturity": maturity, "missing_or_next_work": next_work} for layer, status, evidence, maturity, next_work in rows]
    )


def build_current_capability_inventory(root: Path) -> pd.DataFrame:
    rows = [
        ("technical indicators", True, _module_evidence(root, ["indicators.py"]), False, "Available for research workflows."),
        ("real data loading", True, _module_evidence(root, ["real_data_loader.py"]), False, "Supports Baostock/AkShare/local workflows; not a production data platform."),
        ("factor construction", True, _module_evidence(root, ["factor_builder.py", "build_factor_dataset.py"]), False, "Research factor datasets only."),
        ("model training", True, _module_evidence(root, ["model_trainer.py", "train_baseline_model.py"]), False, "Baseline educational ML training, not a deployable model factory."),
        ("prediction", True, _module_evidence(root, ["model_predictor.py", "predict_with_model.py"]), False, "Research scoring only."),
        ("signal backtest", True, _module_evidence(root, ["ml_signal_backtester.py", "run_ml_signal_backtest.py", "backtester.py"]), False, "Backtest diagnostics only."),
        ("threshold experiments", True, _module_evidence(root, ["ml_threshold_experiment.py", "reduced_feature_threshold_experiment.py", "threshold_decision_report.py"]), False, "Research experiments only; no new sweeps run by this retrospective."),
        ("robustness reports", True, _module_evidence(root, ["batch_model_trainer.py", "model_report_generator.py", "generate_model_report.py"]), False, "Research reporting available."),
        ("regime analysis", True, _module_evidence(root, ["bull_regime_threshold_remediation.py", "sideways_regime_trade_sufficiency_remediation.py"]), False, "Regime diagnostics available."),
        ("integrated remediation review", True, _module_evidence(root, ["integrated_remediation_revalidation.py"]), False, "V4 integrated review available."),
        ("bull failure drilldown", True, _module_evidence(root, ["bull_regime_failure_drilldown.py"]), False, "V4 diagnostic layer available."),
        ("bull trade/window diagnostics", True, _module_evidence(root, ["bull_trade_window_diagnostics.py"]), False, "V4 diagnostic layer available."),
        ("bull error pattern classification", True, _module_evidence(root, ["bull_error_pattern_remediation_design.py"]), False, "V4 classification/design layer available."),
        ("prototype design", True, _module_evidence(root, ["bull_remediation_prototype_design.py"]), False, "Prototype specs only, not deployable logic."),
        ("prototype harness", True, _module_evidence(root, ["bull_prototype_experiment_harness.py"]), False, "Controlled research harness only."),
        ("controlled prototype simulation", True, _module_evidence(root, ["bull_prototype_controlled_backtest.py"]), False, "Diagnostic what-if simulations only."),
        ("V4 closure review", True, _module_evidence(root, ["bull_prototype_result_review.py"]), False, "Step 43 closure review available."),
        ("capital feasibility checks", False, "No dedicated capital constraint engine found.", False, "Not implemented yet; recommended V5 Step 1."),
        ("position sizing", False, "No dedicated position sizing engine found.", False, "Not implemented yet."),
        ("paper trading ledger", False, "No paper ledger module found.", False, "Not implemented yet."),
        ("semi-auto order generation", False, "No semi-auto order generator module found.", False, "Not implemented yet."),
        ("broker execution", False, "No broker execution layer found.", False, "Not implemented yet."),
    ]
    return pd.DataFrame(
        [{"capability": cap, "available": available, "evidence": evidence, "production_ready": production_ready, "notes": notes} for cap, available, evidence, production_ready, notes in rows]
    )


def build_reliable_conclusions(root: Path) -> pd.DataFrame:
    integrated = _read_csv(root / "outputs/integrated_remediation_revalidation_real_v1/integrated_remediation_summary.csv")
    candidate = _read_csv(root / "outputs/bull_prototype_result_review_real_v1/bull_candidate_selection.csv")
    closure = _read_csv(root / "outputs/bull_prototype_result_review_real_v1/bull_v4_closure_status.csv")
    bull_status = "Step 36/43 outputs not found"
    if not integrated.empty:
        row = integrated.iloc[0]
        bull_status = f"overall_decision={row.get('overall_decision')}; trading_ready={row.get('trading_ready')}; bull_final_decision={row.get('bull_final_decision')}; main_blocker={row.get('main_blocker')}"
    selection_status = "Step 43 candidate selection not found"
    if not candidate.empty:
        row = candidate.iloc[0]
        selection_status = f"selected_candidate={row.get('selected_candidate')}; selection_status={row.get('selection_status')}; trading_ready={row.get('trading_ready')}"
    closure_status = "Step 43 closure status not found"
    if not closure.empty:
        closure_status = "; ".join(f"{row.get('item')}={row.get('status')}" for _, row in closure.iterrows())
    rows = [
        ("no_candidate_trading_ready", "No candidate is trading-ready.", bull_status, "high", "Do not deploy any candidate."),
        ("canonical_reduced_40_research_only", "canonical_reduced_40 remains research-only.", bull_status, "high", "Treat canonical_reduced_40 as a research candidate only."),
        ("bull_remediation_unresolved", "Bull remediation remains unresolved.", bull_status, "high", "Bull regime remains the main blocker."),
        ("sideways_partial_progress_only", "Sideways remediation showed partial progress only.", bull_status, "medium_high", "Sideways progress is insufficient for trading-ready status."),
        ("no_step42_prototype_advanced", "No Step 42 prototype improved primary average excess enough to advance.", selection_status, "high", "No prototype should proceed from Step 42."),
        ("step43_closed_v4", "Step 43 closed V4 as research-diagnostic validation cycle.", closure_status, "high", "V4 can close without claiming trading readiness."),
        ("v5_capital_aware_infrastructure", "V5 should shift focus from pure strategy remediation to capital-aware trading utility infrastructure.", closure_status, "high", "Recommended next phase is V5, starting with capital constraints."),
    ]
    return pd.DataFrame(
        [{"conclusion_id": cid, "conclusion": conclusion, "evidence": evidence, "confidence": confidence, "implication": implication} for cid, conclusion, evidence, confidence, implication in rows]
    )


def build_unresolved_limitations() -> pd.DataFrame:
    rows = [
        ("small_symbol_count", "Current real diagnostics use a small configured symbol set.", "medium", "V4 summaries and Step 43 blockers cite five-symbol bull diagnostics.", "Generalization remains limited.", "Future validation"),
        ("no_capital_constraint_engine", "No capital constraint engine exists.", "blocking", "No dedicated capital constraint module is present.", "Research results cannot answer capital feasibility.", "V5 Step 1"),
        ("no_portfolio_engine", "No capital-aware portfolio engine exists.", "blocking", "No dedicated portfolio construction module is present.", "Symbol-level signals cannot become a deployable portfolio.", "V5"),
        ("no_live_risk_engine", "No live risk engine exists.", "blocking", "Only basic backtest risk-control assumptions are visible.", "Live risk cannot be governed.", "V5"),
        ("no_paper_trading_validation", "No paper trading validation exists.", "blocking", "No paper trading ledger module is present.", "Operational behavior is unvalidated.", "V5 Step 6"),
        ("no_broker_execution_layer", "No broker execution layer exists.", "blocking", "No broker or order execution module is present.", "No automated execution should be attempted.", "V5 Step 8 research only"),
        ("no_monitoring_layer", "No monitoring layer exists.", "high", "No monitoring module or dashboard workflow for live/paper operations is present.", "Drift and operational failures cannot be tracked.", "V5"),
        ("no_multi_agent_research_layer", "No multi-agent research layer exists.", "medium", "No agent orchestration module is present.", "Self-directed research is not implemented.", "Later phase"),
        ("no_trading_ready_candidate", "No trading-ready candidate exists.", "blocking", "Step 36 and Step 43 keep trading_ready=False.", "No deployment conclusion is justified.", "Future validation"),
        ("research_diagnostics_only", "The project remains research diagnostics only.", "blocking", "README and V4 outputs use educational/research-only framing.", "Outputs should not be treated as trading instructions.", "All phases"),
        ("not_financial_advice", "This is not financial advice.", "blocking", "Project disclaimer and V4 reports state educational/research-only use.", "Users must not treat outputs as recommendations.", "All phases"),
    ]
    return pd.DataFrame(
        [{"limitation_id": lid, "limitation": limitation, "severity": severity, "evidence": evidence, "consequence": consequence, "recommended_phase_to_address": phase} for lid, limitation, severity, evidence, consequence, phase in rows]
    )


def build_recommended_next_phase() -> pd.DataFrame:
    rows = [
        ("V5", "V5 Step 1", "Capital Constraint Engine", "Model available capital, cash reserves, exposure caps, and blocked capital.", "V4 diagnostics cannot answer capital feasibility.", "capital_constraints.csv; capital_constraint_report.md", "highest"),
        ("V5", "V5 Step 2", "Tradable Universe Filter", "Define liquidity, listing, suspension, and practical eligibility filters.", "Research outputs need tradability context before planning.", "tradable_universe.csv; universe_filter_report.md", "high"),
        ("V5", "V5 Step 3", "Position Sizing Engine", "Translate signals into bounded position sizes.", "Strategy diagnostics lack allocation discipline.", "position_sizing_plan.csv; sizing_report.md", "high"),
        ("V5", "V5 Step 4", "Exit Engine", "Design explicit exit planning utilities without claiming performance success.", "Bull prototypes exposed unresolved exit and participation questions.", "exit_plan.csv; exit_engine_report.md", "medium"),
        ("V5", "V5 Step 5", "Daily Trading Plan", "Generate human-reviewable daily research plans.", "Operational planning should remain supervised and research-only.", "daily_plan.csv; daily_plan_report.md", "medium"),
        ("V5", "V5 Step 6", "Paper Trading Ledger", "Track simulated orders, fills, cash, positions, and decisions.", "No candidate has paper-trading validation.", "paper_ledger.csv; ledger_report.md", "medium"),
        ("V5", "V5 Step 7", "Semi-Auto Order Generator", "Prepare broker-neutral order drafts for manual review.", "Execution must remain controlled and non-advisory.", "order_drafts.csv; order_generator_report.md", "low"),
        ("V5", "V5 Step 8", "Broker Integration Research", "Research broker integration constraints without enabling autonomous trading.", "Broker execution is outside V4 scope and not ready.", "broker_research_notes.md; integration_risk_register.csv", "low"),
    ]
    return pd.DataFrame(
        [{"next_phase": phase, "recommended_step": step, "step_name": name, "purpose": purpose, "reason": reason, "expected_outputs": outputs, "priority": priority} for phase, step, name, purpose, reason, outputs, priority in rows]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Retrospective reads existing files and directory names only.", "No experiment execution is performed."),
        ("no_threshold_change", "confirmed", "No threshold modules are called and no threshold values are written outside retrospective outputs.", "Audit-only step."),
        ("no_model_retraining", "confirmed", "No trainer is called.", "Model artifacts are not changed."),
        ("no_feature_change", "confirmed", "No factor builder or feature engineering module is called.", "Feature definitions are unchanged."),
        ("no_new_data_sources", "confirmed", "Only local README, app, src, and outputs metadata are inspected.", "No new source is added."),
        ("no_new_agents", "confirmed", "No agent configuration or orchestration is created.", "No agents are added."),
        ("no_previous_outputs_overwritten", "confirmed", "Retrospective writes only to its requested output directory.", "Prior output directories are untouched."),
        ("no_trading_ready_claim", "confirmed", "Outputs state the project is not trading-ready and no candidate is deployable.", "No readiness upgrade."),
        ("audit_only", "confirmed", "Outputs are retrospective, inventory, architecture, limitations, and guardrails.", "No strategy logic is changed."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def build_report(
    phase_progress: pd.DataFrame,
    architecture: pd.DataFrame,
    capabilities: pd.DataFrame,
    conclusions: pd.DataFrame,
    limitations: pd.DataFrame,
    next_phase: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# QuantPilot-AI V1-V4 Project Retrospective and Architecture Audit",
            "",
            "## Executive Summary",
            "V1-V4 created a research and diagnostics-oriented quant framework.",
            "The project is not trading-ready.",
            "No candidate should be treated as deployable.",
            "V4 is complete as a research-diagnostic validation cycle.",
            "The next phase should be V5 capital-aware trading utility infrastructure.",
            "Recommended next step: V5 Step 1 Capital Constraint Engine.",
            "",
            "## V1 Summary",
            phase_progress[phase_progress["phase"] == "V1"].to_markdown(index=False),
            "",
            "## V2 Summary",
            phase_progress[phase_progress["phase"] == "V2"].to_markdown(index=False),
            "",
            "## V3 Summary",
            phase_progress[phase_progress["phase"] == "V3"].to_markdown(index=False),
            "",
            "## V4 Summary",
            phase_progress[phase_progress["phase"] == "V4"].to_markdown(index=False),
            "",
            "## Current Architecture",
            architecture.to_markdown(index=False),
            "",
            "## What the Project Can Do Now",
            capabilities[capabilities["available"].fillna(False).astype(bool)].to_markdown(index=False),
            "",
            "## What the Project Cannot Do Yet",
            capabilities[~capabilities["available"].fillna(False).astype(bool)].to_markdown(index=False),
            "",
            "## Reliable Conclusions",
            conclusions.to_markdown(index=False),
            "",
            "## Unresolved Limitations",
            limitations.to_markdown(index=False),
            "",
            "## Why V4 Can Close",
            "V4 can close because it produced a conservative research answer: remediation diagnostics and controlled prototype review did not justify advancing a bull prototype or upgrading any candidate.",
            "Closing V4 records that diagnostic result without claiming profitability or deployment readiness.",
            "",
            "## Why V5 Should Start Now",
            "Further strategy-only remediation would not solve missing capital feasibility, position sizing, portfolio construction, paper validation, execution, or monitoring layers.",
            "V5 should therefore focus on capital-aware trading utility infrastructure before any future practical validation cycle.",
            "",
            "## Recommended V5 Roadmap",
            next_phase.to_markdown(index=False),
            "",
            "## Guardrails",
            guardrails.to_markdown(index=False),
            "",
            "## Educational / Research Disclaimer",
            "This retrospective is educational/research diagnostics only. It is not financial advice.",
            "No strategy, model, threshold, symbol, prototype, or candidate should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_project_retrospective_v1_v4(
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    context = _phase_context(root)
    phase_progress = build_phase_progress_summary(root, context)
    architecture = build_architecture_layer_summary(root)
    capabilities = build_current_capability_inventory(root)
    conclusions = build_reliable_conclusions(root)
    limitations = build_unresolved_limitations()
    next_phase = build_recommended_next_phase()
    guardrails = build_guardrails()
    report = build_report(phase_progress, architecture, capabilities, conclusions, limitations, next_phase, guardrails)

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    paths["report"].write_text(report, encoding="utf-8")
    phase_progress.to_csv(paths["phase_progress"], index=False)
    architecture.to_csv(paths["architecture"], index=False)
    capabilities.to_csv(paths["capabilities"], index=False)
    conclusions.to_csv(paths["conclusions"], index=False)
    limitations.to_csv(paths["limitations"], index=False)
    next_phase.to_csv(paths["next_phase"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    run_config = {
        "project_root": str(root),
        "output_dir": str(output_path),
        "audit_scope": "V1-V4 project retrospective",
        "audit_only": True,
        "educational_research_only": True,
        "trading_ready": False,
        "phase_summary_count": int(len(phase_progress)),
        "architecture_layer_count": int(len(architecture)),
        "capability_count": int(len(capabilities)),
        "src_file_count": int(len(context["src_names"])),
        "output_directory_count": int(len(context["output_names"])),
        "recommended_next_phase": "V5 capital-aware trading utility infrastructure",
        "recommended_next_step": "V5 Step 1 Capital Constraint Engine",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "project_retrospective_v1_v4_report": report,
        "phase_progress_summary": phase_progress,
        "architecture_layer_summary": architecture,
        "current_capability_inventory": capabilities,
        "reliable_conclusions": conclusions,
        "unresolved_limitations": limitations,
        "recommended_next_phase": next_phase,
        "project_retrospective_guardrails": guardrails,
        "run_config": run_config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
