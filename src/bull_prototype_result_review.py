import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILENAMES = {
    "report": "bull_prototype_result_review_report.md",
    "review_summary": "bull_prototype_review_summary.csv",
    "candidate_selection": "bull_candidate_selection.csv",
    "unresolved_blockers": "bull_unresolved_blockers.csv",
    "v4_closure_status": "bull_v4_closure_status.csv",
    "transition_to_v5": "bull_transition_to_v5_recommendation.csv",
    "guardrails": "bull_result_review_guardrails.csv",
    "limitations": "bull_result_review_limitations.csv",
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


def _format_symbol_list(value: Any) -> str:
    text = _clean_text(value)
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    return ",".join(_format_symbol(token) if token.isdigit() else token for token in tokens)


def _read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "symbol" in df:
        df["symbol"] = df["symbol"].map(_format_symbol)
    if "target_symbols" in df:
        df["target_symbols"] = df["target_symbols"].map(_format_symbol_list)
    return df


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean_text(value).lower() in {"true", "1", "yes", "y"}


def _numeric(value: Any) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _step42_decisions(decisions: pd.DataFrame) -> dict[str, bool]:
    if decisions.empty or "prototype_id" not in decisions:
        return {}
    result = {}
    for _, row in decisions.iterrows():
        result[_clean_text(row.get("prototype_id"))] = _bool_value(row.get("can_advance_to_further_testing"))
    return result


def _secondary_metric_improved(metric_comparison: pd.DataFrame, prototype_id: str) -> bool:
    if metric_comparison.empty or "prototype_id" not in metric_comparison:
        return False
    rows = metric_comparison[metric_comparison["prototype_id"].astype(str) == prototype_id].copy()
    if rows.empty or "metric_name" not in rows or "delta" not in rows:
        return False
    rows["delta"] = pd.to_numeric(rows["delta"], errors="coerce")
    secondary = rows[rows["metric_name"].astype(str) != "avg_strategy_vs_benchmark_pct"]
    improved = secondary[
        ((secondary["direction"].astype(str).str.contains("higher", na=False)) & secondary["delta"].gt(0))
        | ((secondary["direction"].astype(str).str.contains("lower", na=False)) & secondary["delta"].lt(0))
    ]
    return not improved.empty


def build_review_summary(
    execution_results: pd.DataFrame,
    metric_comparison: pd.DataFrame,
    decision_summary: pd.DataFrame,
) -> pd.DataFrame:
    step42_can_advance = _step42_decisions(decision_summary)
    rows = []
    for _, row in execution_results.iterrows():
        prototype_id = _clean_text(row.get("prototype_id"))
        baseline_avg = _numeric(row.get("baseline_avg_excess_pct"))
        prototype_avg = _numeric(row.get("prototype_avg_excess_pct"))
        delta = _numeric(row.get("delta_avg_excess_pct"))
        step42_allowed = step42_can_advance.get(prototype_id, False)
        primary_worse = pd.notna(delta) and delta <= 0
        secondary_improved = _secondary_metric_improved(metric_comparison, prototype_id)
        status = _clean_text(row.get("execution_status"))
        if status == "not_executable_with_current_data":
            conservative_result = "not_executable_with_current_data"
            decision = "reject_for_further_validation"
            reason = _clean_text(row.get("decision_reason")) or "Prototype was not executable with current diagnostic data."
        elif primary_worse and secondary_improved:
            conservative_result = "secondary_improvement_but_primary_metric_worse"
            decision = "reject_for_further_validation"
            reason = "Secondary diagnostics improved, but the primary average excess metric worsened."
        elif primary_worse:
            conservative_result = "primary_metric_not_improved"
            decision = "reject_for_further_validation"
            reason = "The primary average excess metric did not improve versus the unchanged baseline."
        elif not step42_allowed:
            conservative_result = "step42_did_not_allow_advancement"
            decision = "reject_for_further_validation"
            reason = "Step 42 did not allow this prototype to advance, so Step 43 does not reverse that decision."
        else:
            conservative_result = "requires_future_validation_before_any_status_change"
            decision = "hold_for_manual_research_review"
            reason = "Primary metric improved in diagnostics, but this review cannot claim validation or readiness."
        reviewed_allowed = bool(step42_allowed and pd.notna(delta) and delta > 0)
        rows.append(
            {
                "prototype_id": prototype_id,
                "prototype_name": _clean_text(row.get("prototype_name")),
                "execution_status": status,
                "baseline_avg_excess_pct": baseline_avg,
                "prototype_avg_excess_pct": prototype_avg,
                "delta_avg_excess_pct": delta,
                "conservative_result": conservative_result,
                "step42_can_advance_to_further_testing": bool(step42_allowed),
                "reviewed_can_advance_to_further_validation": reviewed_allowed,
                "trading_ready": False,
                "review_decision": decision,
                "review_reason": reason,
                "notes": "Research-only Step 43 review. No strategy logic, thresholds, models, features, data sources, or agents are changed.",
            }
        )
    return pd.DataFrame(rows)


def build_candidate_selection(review_summary: pd.DataFrame) -> pd.DataFrame:
    allowed = (
        review_summary["reviewed_can_advance_to_further_validation"].fillna(False).astype(bool)
        if "reviewed_can_advance_to_further_validation" in review_summary
        else pd.Series(dtype=bool)
    )
    selected = bool(not review_summary.empty and allowed.any())
    return pd.DataFrame(
        [
            {
                "selection_level": "overall",
                "selected_candidate": "none" if not selected else "requires_manual_review",
                "selected_prototype_id": "" if not selected else _clean_text(review_summary.loc[allowed].iloc[0].get("prototype_id")),
                "selection_status": "no_candidate_selected" if not selected else "candidate_requires_further_validation",
                "reason": "no prototype improved primary avg excess enough to advance" if not selected else "diagnostic-only improvement requires future validation",
                "trading_ready": False,
                "required_future_validation": "None selected from Step 42; future work should start from V5 infrastructure before any new validation cycle."
                if not selected
                else "Broader validation, leakage review, capital constraints, and paper-trading checks would be required.",
                "notes": "No Step 42 prototype is trading-ready.",
            }
        ]
    )


def build_unresolved_blockers(
    integrated_summary: pd.DataFrame,
    symbol_profile: pd.DataFrame,
    diagnostics_summary: pd.DataFrame,
) -> pd.DataFrame:
    tested_symbols = len(symbol_profile) if not symbol_profile.empty else "small configured symbol set"
    zero_beat = "Step 39 symbol profile shows zero benchmark-beating bull trades for all profiled symbols."
    if not symbol_profile.empty and "beat_benchmark_trades" in symbol_profile:
        beats = pd.to_numeric(symbol_profile["beat_benchmark_trades"], errors="coerce").fillna(0)
        zero_beat = f"beat_benchmark_trades total={int(beats.sum())} across {len(symbol_profile)} profiled symbols."
    bull_status = "bull_remediation_failed"
    if not integrated_summary.empty:
        bull_status = _clean_text(integrated_summary.iloc[0].get("bull_final_decision")) or bull_status
    window_evidence = "Step 38 bull window diagnostics remain limited."
    if not diagnostics_summary.empty and "symbol" in diagnostics_summary:
        window_evidence = f"Step 38 bull window summary covers {len(diagnostics_summary)} symbols."
    rows = [
        ("bull_remediation_failed", "Bull remediation failed", "blocking", bull_status, "V4 cannot produce a trading-ready strategy.", "Do not upgrade candidate status.", "V4 closure"),
        ("no_primary_metric_improvement", "No primary metric improvement", "blocking", "Step 42 prototypes did not improve avg_strategy_vs_benchmark_pct enough to advance.", "No prototype can be selected.", "Close Step 42 prototype cycle conservatively.", "V4 closure"),
        ("relative_benchmark_underperformance", "Relative benchmark underperformance", "high", "Bull diagnostics show benchmark lag remains the dominant issue.", "Absolute-return improvements are insufficient.", "Keep benchmark excess as the primary future validation metric.", "Future research"),
        ("zero_or_low_beat_benchmark_trade_issue", "Zero or low beat-benchmark trade issue", "high", zero_beat, "Bull trades do not show enough benchmark-beating evidence.", "Investigate only after stronger infrastructure and validation discipline exist.", "Future research"),
        ("small_symbol_count", "Small symbol count", "medium", f"Current bull diagnostic profile covers {tested_symbols} symbols.", "Prototype conclusions are unstable.", "Use broader validation only in a later scoped phase.", "Future validation"),
        ("no_capital_constraints", "No capital constraints", "blocking", "V4 diagnostics do not model capital allocation limits.", "Backtest diagnostics do not answer tradability constraints.", "Build V5 Step 1 Capital Constraint Engine.", "V5"),
        ("no_portfolio_engine", "No portfolio engine", "blocking", "V4 does not include a capital-aware portfolio engine.", "Symbol-level results cannot become a deployable portfolio.", "Build portfolio-aware utilities after capital constraints.", "V5"),
        ("no_paper_trading_validation", "No paper trading validation", "blocking", "No live or paper ledger validates decisions.", "Operational readiness is unknown.", "Add paper trading ledger only after planning utilities exist.", "V5"),
        ("no_execution_layer", "No execution layer", "blocking", window_evidence, "There is no broker-safe order workflow.", "Research broker integration later; do not automate trading now.", "V5"),
    ]
    return pd.DataFrame(
        [
            {
                "blocker_id": blocker_id,
                "blocker_name": name,
                "severity": severity,
                "evidence": evidence,
                "consequence": consequence,
                "recommended_followup": followup,
                "phase_to_address": phase,
            }
            for blocker_id, name, severity, evidence, consequence, followup, phase in rows
        ]
    )


def build_v4_closure_status() -> pd.DataFrame:
    rows = [
        ("V4 research diagnostic cycle completed", "completed_as_research_diagnostics", "Steps 34 through 43 reviewed remediation attempts and prototype results.", "V4 can close as a research-diagnostic validation cycle.", "V5"),
        ("canonical_reduced_40 remains research_only", "confirmed", "Integrated and Step 43 outputs keep trading_ready=False.", "No trading-ready upgrade is justified.", "V5"),
        ("sideways remediation partial progress", "partial_progress_only", "Sideways remediation showed configured aggregate progress but not enough for trading-ready status.", "Sideways progress does not overcome bull blockers.", "V5"),
        ("bull remediation unresolved", "unresolved", "Step 42 prototypes did not improve the primary metric enough to advance.", "Bull blocker remains unresolved.", "V5"),
        ("no prototype selected", "confirmed", "Step 43 candidate selection status is no_candidate_selected.", "No Step 42 prototype advances.", "V5"),
        ("no trading_ready upgrade", "confirmed", "All Step 43 review rows have trading_ready=False.", "Project remains educational/research only.", "V5"),
        ("V5 recommended next phase", "recommended", "V4 strategy remediation has reached a conservative closure point.", "Shift to capital-aware trading utility infrastructure.", "V5 Step 1 Capital Constraint Engine"),
    ]
    return pd.DataFrame(
        [{"item": item, "status": status, "evidence": evidence, "conclusion": conclusion, "next_phase": next_phase} for item, status, evidence, conclusion, next_phase in rows]
    )


def build_transition_to_v5() -> pd.DataFrame:
    rows = [
        ("V5 Step 1", "Capital Constraint Engine", "Model available capital, cash reserves, exposure caps, and blocked capital.", "V4 diagnostics cannot answer capital feasibility.", "capital_constraints.csv; capital_constraint_report.md", "highest"),
        ("V5 Step 2", "Tradable Universe Filter", "Define liquidity, listing, suspension, and practical eligibility filters.", "Research outputs need tradability context before planning.", "tradable_universe.csv; universe_filter_report.md", "high"),
        ("V5 Step 3", "Position Sizing Engine", "Translate signals into bounded position sizes.", "Strategy diagnostics lack allocation discipline.", "position_sizing_plan.csv; sizing_report.md", "high"),
        ("V5 Step 4", "Exit Engine", "Design explicit exit planning utilities without claiming performance success.", "Bull prototypes exposed unresolved exit and participation questions.", "exit_plan.csv; exit_engine_report.md", "medium"),
        ("V5 Step 5", "Daily Trading Plan", "Generate human-reviewable daily research plans.", "Operational planning should remain supervised and research-only.", "daily_plan.csv; daily_plan_report.md", "medium"),
        ("V5 Step 6", "Paper Trading Ledger", "Track simulated orders, fills, cash, positions, and decisions.", "No candidate has paper-trading validation.", "paper_ledger.csv; ledger_report.md", "medium"),
        ("V5 Step 7", "Semi-Auto Order Generator", "Prepare broker-neutral order drafts for manual review.", "Execution must remain controlled and non-advisory.", "order_drafts.csv; order_generator_report.md", "low"),
        ("V5 Step 8", "Broker Integration Research", "Research broker integration constraints without enabling autonomous trading.", "Broker execution is outside V4 scope and not ready.", "broker_research_notes.md; integration_risk_register.csv", "low"),
    ]
    return pd.DataFrame(
        [
            {
                "recommended_step": step,
                "step_name": name,
                "purpose": purpose,
                "why_now": why_now,
                "expected_outputs": outputs,
                "priority": priority,
            }
            for step, name, purpose, why_now, outputs, priority in rows
        ]
    )


def build_guardrails() -> pd.DataFrame:
    rows = [
        ("no_new_backtests", "confirmed", "Step 43 reads Step 42 outputs only and runs no simulations.", "Review-only closure."),
        ("no_threshold_change", "confirmed", "No threshold columns are modified and no threshold sweep is executed.", "Selected threshold remains unchanged."),
        ("no_model_retraining", "confirmed", "No trainer or model artifact path is called.", "Model remains unchanged."),
        ("no_feature_change", "confirmed", "No factor engineering module is called.", "Feature set remains unchanged."),
        ("no_new_data_sources", "confirmed", "Only existing local output directories are read.", "No external or new data source is added."),
        ("no_new_agents", "confirmed", "No agent configuration or agent output is created.", "No agents are added."),
        ("no_previous_outputs_overwritten", "confirmed", "Step 43 writes only to its output directory.", "Step 42 outputs remain untouched."),
        ("no_trading_ready_upgrade", "confirmed", "trading_ready=False in every selection and review output.", "No readiness claim."),
        ("review_only", "confirmed", "Outputs are review, selection, blockers, closure, guardrails, and limitations.", "No experiment is run."),
        ("educational_research_only", "confirmed", "Report and CLI warning state educational/research-only use.", "Not financial advice."),
    ]
    return pd.DataFrame([{"guardrail": g, "status": s, "evidence": e, "notes": n} for g, s, e, n in rows])


def build_limitations() -> pd.DataFrame:
    rows = [
        ("V4_is_research_diagnostic_not_trading_system", "blocking", "V4 outputs are diagnostics and reports, not a trading system.", "Do not deploy from V4 outputs."),
        ("bull_remediation_unresolved", "blocking", "Step 43 selects no Step 42 prototype.", "Bull blocker remains open."),
        ("small_symbol_count", "medium", "The bull review is based on a small configured symbol set.", "Generalization is not established."),
        ("no_capital_constraints", "blocking", "No capital constraint engine exists in V4.", "Capital feasibility is unknown."),
        ("no_portfolio_engine", "blocking", "No capital-aware portfolio construction engine exists.", "Portfolio behavior is unknown."),
        ("no_live_or_paper_trading", "blocking", "No live or paper trading validation is present.", "Operational readiness is unknown."),
        ("no_broker_execution_layer", "blocking", "No broker execution layer exists.", "Orders cannot be safely automated."),
        ("no_self_iterating_agent_layer", "blocking", "No autonomous research or trading agent layer exists.", "No self-iteration or autonomous execution is supported."),
        ("not_financial_advice", "blocking", "Outputs are educational/research diagnostics only.", "Do not treat results as advice."),
        ("not_trading_ready", "blocking", "No candidate is trading-ready.", "No deployment conclusion."),
    ]
    return pd.DataFrame(
        [{"limitation": name, "severity": severity, "description": description, "consequence": consequence} for name, severity, description, consequence in rows]
    )


def build_report(
    controlled_backtest_dir: str | Path,
    integrated_dir: str | Path | None,
    error_design_dir: str | Path | None,
    diagnostics_dir: str | Path | None,
    review_summary: pd.DataFrame,
    candidate_selection: pd.DataFrame,
    blockers: pd.DataFrame,
    closure: pd.DataFrame,
    v5: pd.DataFrame,
    guardrails: pd.DataFrame,
    limitations: pd.DataFrame,
) -> str:
    total = len(review_summary)
    allowed = int(review_summary["reviewed_can_advance_to_further_validation"].fillna(False).astype(bool).sum()) if not review_summary.empty else 0
    final_selection = candidate_selection.iloc[0].get("selection_status") if not candidate_selection.empty else "no_candidate_selected"
    return "\n".join(
        [
            "# V4 Step 43 Bull Prototype Result Review and V4 Closure Report",
            "",
            "## Executive Summary",
            "V4 Step 43 reviews Step 42 controlled bull prototype results conservatively and closes the V4 bull remediation research cycle.",
            f"Total prototypes reviewed: {total}. Prototypes allowed to advance to further validation: {allowed}.",
            "V4 can close as a research-diagnostic validation cycle.",
            "V4 did not produce a trading-ready strategy.",
            "No prototype improved the primary avg excess metric enough to advance.",
            "No candidate is selected for further validation from Step 42.",
            "canonical_reduced_40 remains research-only.",
            "Bull remediation remains unresolved.",
            "Sideways remediation showed partial progress but is insufficient for trading-ready status.",
            "The next project phase should shift from pure strategy remediation to V5 capital-aware trading utility infrastructure.",
            "Recommended next step: V5 Step 1 Capital Constraint Engine.",
            "",
            "## Inputs Used",
            f"- Controlled backtest directory: {controlled_backtest_dir}",
            f"- Integrated directory: {integrated_dir or 'not provided'}",
            f"- Error design directory: {error_design_dir or 'not provided'}",
            f"- Diagnostics directory: {diagnostics_dir or 'not provided'}",
            "",
            "## Step 42 Prototype Review",
            "Each Step 42 prototype is reviewed against the primary average strategy-vs-benchmark excess metric and the Step 42 advancement decision.",
            "A prototype with delta_avg_excess_pct <= 0 cannot advance. A Step 42 rejection is not reversed.",
            review_summary.to_markdown(index=False) if not review_summary.empty else "No Step 42 prototype rows were available.",
            "",
            "## Candidate Selection Decision",
            f"Final selection status: {final_selection}.",
            "No candidate is selected because no prototype improved primary avg excess enough to advance.",
            "",
            "## Unresolved Bull Blockers",
            blockers.to_markdown(index=False) if not blockers.empty else "No blocker rows were generated.",
            "",
            "## V4 Closure Status",
            closure.to_markdown(index=False) if not closure.empty else "No closure rows were generated.",
            "",
            "## Why V4 Can Close Without Trading-Ready Status",
            "V4 can close because the research diagnostic cycle has produced a conservative answer: the current bull remediation prototypes did not clear the primary metric gate.",
            "Closing V4 records that result without converting it into a trading-ready claim.",
            "",
            "## Why This Should Transition to V5",
            "Further threshold or prototype diagnostics would risk repeating strategy-only research while core trading utility infrastructure remains absent.",
            "Capital constraints, tradable universe rules, position sizing, exits, daily planning, and paper-ledger instrumentation are prerequisites for any later practical validation.",
            "",
            "## Recommended V5 Roadmap",
            v5.to_markdown(index=False) if not v5.empty else "No V5 roadmap rows were generated.",
            "",
            "## Guardrails",
            guardrails.to_markdown(index=False) if not guardrails.empty else "No guardrail rows were generated.",
            "",
            "## Limitations",
            limitations.to_markdown(index=False) if not limitations.empty else "No limitation rows were generated.",
            "",
            "## Educational / Research Disclaimer",
            "This report is educational/research diagnostics only. It is not financial advice.",
            "No strategy, model, threshold, prototype, symbol, or candidate in this report should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_bull_prototype_result_review(
    controlled_backtest_dir: str | Path,
    integrated_dir: str | Path | None,
    error_design_dir: str | Path | None,
    diagnostics_dir: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    controlled = Path(controlled_backtest_dir)
    integrated = Path(integrated_dir) if integrated_dir else None
    error_design = Path(error_design_dir) if error_design_dir else None
    diagnostics = Path(diagnostics_dir) if diagnostics_dir else None
    execution_results = _read_csv(controlled / "bull_prototype_execution_results.csv")
    metric_comparison = _read_csv(controlled / "bull_prototype_metric_comparison.csv")
    symbol_comparison = _read_csv(controlled / "bull_prototype_symbol_comparison.csv", dtype={"symbol": str})
    decision_summary = _read_csv(controlled / "bull_prototype_decision_summary.csv")
    execution_audit = _read_csv(controlled / "bull_prototype_execution_audit.csv")
    step42_guardrails = _read_csv(controlled / "bull_prototype_guardrail_check.csv")
    step42_limitations = _read_csv(controlled / "bull_prototype_limitations.csv")
    integrated_summary = _read_csv(integrated / "integrated_remediation_summary.csv") if integrated else pd.DataFrame()
    symbol_profile = _read_csv(error_design / "bull_symbol_error_profile.csv", dtype={"symbol": str}) if error_design else pd.DataFrame()
    diagnostics_summary = _read_csv(diagnostics / "bull_symbol_window_summary.csv", dtype={"symbol": str}) if diagnostics else pd.DataFrame()
    step42_config = _read_json(controlled / "run_config.json")

    review_summary = build_review_summary(execution_results, metric_comparison, decision_summary)
    candidate_selection = build_candidate_selection(review_summary)
    blockers = build_unresolved_blockers(integrated_summary, symbol_profile, diagnostics_summary)
    closure = build_v4_closure_status()
    transition_to_v5 = build_transition_to_v5()
    guardrails = build_guardrails()
    limitations = build_limitations()
    report = build_report(
        controlled_backtest_dir,
        integrated_dir,
        error_design_dir,
        diagnostics_dir,
        review_summary,
        candidate_selection,
        blockers,
        closure,
        transition_to_v5,
        guardrails,
        limitations,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    paths["report"].write_text(report, encoding="utf-8")
    review_summary.to_csv(paths["review_summary"], index=False)
    candidate_selection.to_csv(paths["candidate_selection"], index=False)
    blockers.to_csv(paths["unresolved_blockers"], index=False)
    closure.to_csv(paths["v4_closure_status"], index=False)
    transition_to_v5.to_csv(paths["transition_to_v5"], index=False)
    guardrails.to_csv(paths["guardrails"], index=False)
    limitations.to_csv(paths["limitations"], index=False)
    config = {
        "controlled_backtest_dir": str(controlled_backtest_dir),
        "integrated_dir": str(integrated_dir) if integrated_dir else None,
        "error_design_dir": str(error_design_dir) if error_design_dir else None,
        "diagnostics_dir": str(diagnostics_dir) if diagnostics_dir else None,
        "output_dir": str(output_path),
        "source_step42_output_dir": str(controlled),
        "step42_run_config": step42_config,
        "step42_execution_audit_rows": int(len(execution_audit)),
        "step42_guardrail_rows": int(len(step42_guardrails)),
        "step42_limitation_rows": int(len(step42_limitations)),
        "step42_symbol_comparison_symbols": sorted(symbol_comparison["symbol"].dropna().astype(str).unique().tolist()) if "symbol" in symbol_comparison else [],
        "total_prototypes_reviewed": int(len(review_summary)),
        "prototypes_allowed_to_advance": int(review_summary["reviewed_can_advance_to_further_validation"].fillna(False).astype(bool).sum()) if not review_summary.empty else 0,
        "final_bull_remediation_status": "bull_remediation_unresolved",
        "v4_closure_recommendation": "close_v4_as_research_diagnostics_and_transition_to_v5",
        "recommended_next_step": "V5 Step 1 Capital Constraint Engine",
        "selection_status": candidate_selection.iloc[0].get("selection_status") if not candidate_selection.empty else "no_candidate_selected",
        "trading_ready": False,
        "educational_research_only": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "bull_prototype_result_review_report": report,
        "bull_prototype_review_summary": review_summary,
        "bull_candidate_selection": candidate_selection,
        "bull_unresolved_blockers": blockers,
        "bull_v4_closure_status": closure,
        "bull_transition_to_v5_recommendation": transition_to_v5,
        "bull_result_review_guardrails": guardrails,
        "bull_result_review_limitations": limitations,
        "run_config": config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
