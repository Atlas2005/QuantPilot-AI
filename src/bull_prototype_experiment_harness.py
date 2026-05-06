import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILENAMES = {
    "report": "bull_prototype_experiment_harness_report.md",
    "registry": "bull_prototype_registry.csv",
    "dry_run_plan": "bull_prototype_dry_run_plan.csv",
    "config_validation": "bull_prototype_config_validation.csv",
    "baseline_requirements": "bull_prototype_baseline_requirements.csv",
    "metric_contract": "bull_prototype_metric_contract.csv",
    "execution_guardrails": "bull_prototype_execution_guardrails.csv",
    "limitations": "bull_prototype_harness_limitations.csv",
    "not_executed_log": "bull_prototype_not_executed_log.csv",
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
    if not text:
        return ""
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    formatted = [
        _format_symbol(token) if token.replace(".", "", 1).isdigit() else token
        for token in tokens
    ]
    return ",".join(formatted)


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


def _metric_baseline(metric_plan: pd.DataFrame, metric_name: str) -> str:
    if metric_plan.empty or "metric_name" not in metric_plan:
        return "unavailable"
    rows = metric_plan[metric_plan["metric_name"].astype(str) == metric_name]
    if rows.empty:
        return "unavailable"
    value = rows.iloc[0].get("baseline_value_if_available")
    return _clean_text(value) or "unavailable"


def _bool_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean_text(value).lower() in {"true", "1", "yes", "y"}


def build_prototype_registry(specs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in specs.iterrows():
        rows.append(
            {
                "prototype_id": _clean_text(row.get("prototype_id")),
                "prototype_name": _clean_text(row.get("prototype_name")),
                "source_step": "V4 Step 40 Bull Remediation Prototype Design",
                "target_symbols": _format_symbol_list(row.get("target_symbols")),
                "target_error_family": _clean_text(row.get("target_error_family")),
                "change_type": _clean_text(row.get("change_type")),
                "harness_status": "registered_for_future_controlled_test",
                "execution_status": "not_executed",
                "allowed_to_execute_in_step41": False,
                "baseline_required": True,
                "metric_contract_required": True,
                "notes": "Registered for a future controlled test only; Step 41 does not execute it.",
            }
        )
    return pd.DataFrame(rows)


def build_dry_run_plan(registry: pd.DataFrame) -> pd.DataFrame:
    phases = [
        (1, "load_prototype_config", "V4 Step 42 Bull Prototype Controlled Backtest Execution", "Prototype registry row missing or invalid."),
        (2, "validate_baseline_contract", "V4 Step 42 Bull Prototype Controlled Backtest Execution", "Required baseline metric unavailable without explicit unavailable declaration."),
        (3, "prepare_future_execution_config", "V4 Step 42 Bull Prototype Controlled Backtest Execution", "Any threshold, model, feature, data-source, or agent change is requested."),
        (4, "record_future_output_expectations", "V4 Step 42 Bull Prototype Controlled Backtest Execution", "Metric contract missing required benchmark comparison fields."),
    ]
    rows = []
    for _, proto in registry.iterrows():
        for step, action, future_step, blocking in phases:
            rows.append(
                {
                    "dry_run_step": step,
                    "prototype_id": proto.get("prototype_id"),
                    "planned_action": action,
                    "would_execute_backtest": False,
                    "would_modify_model": False,
                    "would_modify_features": False,
                    "would_modify_threshold": False,
                    "required_future_step": future_step,
                    "blocking_condition": blocking,
                    "notes": "Dry-run planning only. No prototype backtest is executed in Step 41.",
                }
            )
    return pd.DataFrame(rows)


def build_config_validation(specs: pd.DataFrame, metric_plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_available = not metric_plan.empty and "metric_name" in metric_plan
    for _, row in specs.iterrows():
        prototype_id = _clean_text(row.get("prototype_id"))
        target_symbols = _format_symbol_list(row.get("target_symbols"))
        checks = [
            (
                "prototype_exists",
                bool(prototype_id),
                f"prototype_id={prototype_id}",
                "blocking",
                "Prototype id must be present.",
            ),
            (
                "implementation_status_design_only",
                _clean_text(row.get("implementation_status")) == "prototype_design_only",
                f"implementation_status={row.get('implementation_status')}",
                "blocking",
                "Step 41 only accepts Step 40 design-only prototypes.",
            ),
            (
                "execution_status_not_executed",
                _clean_text(row.get("execution_status")) == "not_executed",
                f"execution_status={row.get('execution_status')}",
                "blocking",
                "Already executed prototypes are out of scope.",
            ),
            (
                "allowed_in_current_step_false",
                not _bool_text(row.get("allowed_in_current_step")),
                f"allowed_in_current_step={row.get('allowed_in_current_step')}",
                "blocking",
                "Step 41 cannot execute prototypes.",
            ),
            (
                "baseline_metric_available_or_declared_unavailable",
                metric_available,
                f"metric_plan_rows={len(metric_plan)}",
                "medium",
                "Metric plan must exist; individual unavailable values are allowed when declared.",
            ),
            (
                "target_symbols_preserved",
                bool(target_symbols),
                f"target_symbols={target_symbols}",
                "medium",
                "Six-digit symbols are preserved where target symbols are numeric.",
            ),
            (
                "no_trading_ready_claim",
                "trading_ready" not in " ".join(str(value).lower() for value in row.to_dict().values()),
                "prototype row contains no trading_ready claim",
                "blocking",
                "Harness rows must not claim trading readiness.",
            ),
        ]
        for check, passed, evidence, severity, notes in checks:
            rows.append(
                {
                    "prototype_id": prototype_id,
                    "validation_check": check,
                    "status": "passed" if passed else "failed",
                    "evidence": evidence,
                    "severity": severity,
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows)


def build_baseline_requirements(
    metric_plan: pd.DataFrame,
    integrated_summary: pd.DataFrame,
    trades: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    integrated = integrated_summary.iloc[0] if not integrated_summary.empty else pd.Series(dtype=object)
    threshold = "0.65 / 0.50"
    if not integrated_summary.empty:
        buy = _clean_text(integrated.get("bull_buy_threshold")) or "0.65"
        sell = _clean_text(integrated.get("bull_sell_threshold")) or "0.50"
        threshold = f"{buy} / {sell}"
    symbol_601318 = "available" if not trades.empty and "symbol" in trades and (trades["symbol"] == "601318").any() else "unavailable"
    symbol_600036 = "available" if not trades.empty and "symbol" in trades and (trades["symbol"] == "600036").any() else "unavailable"
    rows = [
        ("Step 34 selected threshold 0.65 / 0.50", "Step 34 / Step 36 context", True, True, threshold, "Must remain unchanged for future prototype comparison."),
        ("avg_strategy_vs_benchmark_pct", "Step 40 metric plan / Step 36", True, _metric_baseline(metric_plan, "avg_strategy_vs_benchmark_pct") != "unavailable", _metric_baseline(metric_plan, "avg_strategy_vs_benchmark_pct"), "Primary aggregate benchmark-excess baseline."),
        ("beat_benchmark_rate", "Step 40 metric plan / Step 36", True, _metric_baseline(metric_plan, "beat_benchmark_rate") != "unavailable", _metric_baseline(metric_plan, "beat_benchmark_rate"), "Aggregate benchmark-beating-rate baseline."),
        ("sufficient_trade_rate", "Step 40 metric plan / Step 36", True, _metric_baseline(metric_plan, "sufficient_trade_rate") != "unavailable", _metric_baseline(metric_plan, "sufficient_trade_rate"), "Trade sufficiency baseline."),
        ("trade_count", "Step 40 metric plan / Step 38", True, _metric_baseline(metric_plan, "trade_count") != "unavailable", _metric_baseline(metric_plan, "trade_count"), "Future prototypes must compare against unchanged trade count context."),
        ("beat_benchmark_trades", "Step 40 metric plan / Step 38", True, _metric_baseline(metric_plan, "beat_benchmark_trades") != "unavailable", _metric_baseline(metric_plan, "beat_benchmark_trades"), "Direct baseline for the global zero beat-benchmark-trades issue."),
        ("negative_trade_count", "Step 40 metric plan / Step 38", True, _metric_baseline(metric_plan, "negative_trade_count") != "unavailable", _metric_baseline(metric_plan, "negative_trade_count"), "Baseline for absolute-loss cluster."),
        ("positive_return_but_lagged_benchmark_count", "Step 40 metric plan / Step 38", True, _metric_baseline(metric_plan, "positive_return_but_lagged_benchmark_count") != "unavailable", _metric_baseline(metric_plan, "positive_return_but_lagged_benchmark_count"), "Baseline for profitable benchmark-lag trades."),
        ("symbol-level 601318 diagnostics", "Step 38 trade/window diagnostics", True, symbol_601318 == "available", symbol_601318, "Required for 601318 benchmark-lag prototype execution."),
        ("symbol-level 600036 diagnostics", "Step 38 trade/window diagnostics", True, symbol_600036 == "available", symbol_600036, "Required for 600036 near-neutral underperformance comparison."),
    ]
    return pd.DataFrame(
        [
            {
                "baseline_item": item,
                "source": source,
                "required_for_future_execution": required,
                "available_now": available,
                "baseline_value": value,
                "notes": notes,
            }
            for item, source, required, available, value, notes in rows
        ]
    )


def build_metric_contract(metric_plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metric_plan.iterrows():
        metric = _clean_text(row.get("metric_name"))
        direction = _clean_text(row.get("direction"))
        rows.append(
            {
                "metric_name": metric,
                "metric_level": _clean_text(row.get("metric_level")),
                "required": True,
                "direction": direction,
                "baseline_value_if_available": _clean_text(row.get("baseline_value_if_available")) or "unavailable",
                "pass_condition_for_future_step": _clean_text(row.get("target_condition")) or "Future step must define pass condition before execution.",
                "reject_condition_for_future_step": _clean_text(row.get("failure_condition")) or "Reject if metric weakens versus unchanged baseline.",
                "notes": "Contract only; no prototype passes in Step 41.",
            }
        )
    return pd.DataFrame(rows)


def build_execution_guardrails() -> pd.DataFrame:
    rows = [
        ("no_real_backtest_execution_in_step41", "confirmed", "Dry-run plan sets would_execute_backtest=False.", "Would create unvalidated prototype results.", "No real prototype backtest is run."),
        ("no_threshold_change", "confirmed", "Threshold action is reused_for_harness_baseline_only.", "Would invalidate Step 34 baseline comparison.", "Selected 0.65 / 0.50 remains unchanged."),
        ("no_model_retraining", "confirmed", "Harness reads CSV specs only and calls no trainer.", "Would create a new model experiment.", "Model remains logistic_regression context only."),
        ("no_feature_change", "confirmed", "Harness does not touch factor engineering modules.", "Would change research surface.", "Feature engineering remains unchanged."),
        ("no_new_data_sources", "confirmed", "Only Step 40, optional Step 38, and optional Step 36 outputs are read.", "Would break comparability.", "No data source is added."),
        ("no_new_agents", "confirmed", "No agent files or configurations are created.", "Would alter process scope.", "No new agents are added."),
        ("no_new_performance_claims", "confirmed", "Outputs include contracts and baselines only, not results.", "Would overclaim dry-run infrastructure.", "No new performance claim is made."),
        ("no_trading_ready_upgrade", "confirmed", "Report and run_config keep trading_ready=False.", "Would overclaim unresolved bull remediation.", "No candidate is trading-ready."),
        ("educational_research_only", "confirmed", "Report and CLI warning use educational/research-only language.", "Would risk financial-advice framing.", "Not financial advice."),
    ]
    return pd.DataFrame(
        [
            {
                "guardrail": guardrail,
                "status": status,
                "evidence": evidence,
                "consequence_if_violated": consequence,
                "notes": notes,
            }
            for guardrail, status, evidence, consequence, notes in rows
        ]
    )


def build_limitations() -> pd.DataFrame:
    rows = [
        ("harness_only_not_execution", "blocking", "Step 41 creates registry and dry-run config only.", "No prototype result exists.", "Execute only in a future explicitly scoped step."),
        ("future_results_unknown", "blocking", "No real prototype backtest is run.", "No performance conclusion can be drawn.", "Use Step 42 for controlled execution."),
        ("small_symbol_count", "medium", "Baseline context remains the configured small symbol set.", "Future results may be unstable.", "Require broader validation after any retained prototype."),
        ("overfitting_risk_remains", "high", "Prototype configs target known failure modes.", "Future experiments may overfit diagnostics.", "Require unchanged-baseline comparison and rejection criteria."),
        ("baseline_comparison_required", "blocking", "Future execution must compare against Step 34 / Step 38 baseline.", "Prototype results cannot be interpreted without baseline.", "Keep baseline requirements mandatory."),
        ("research_only_not_trading_ready", "blocking", "canonical_reduced_40 remains research-only.", "No trading-ready status changes.", "Continue gate-controlled diagnostics."),
    ]
    return pd.DataFrame(
        [
            {
                "limitation_type": kind,
                "severity": severity,
                "description": description,
                "consequence": consequence,
                "recommended_followup": followup,
            }
            for kind, severity, description, consequence, followup in rows
        ]
    )


def build_not_executed_log() -> pd.DataFrame:
    rows = [
        ("prototypes_registered_not_executed", "confirmed", "Registry execution_status=not_executed.", "Future controlled execution is required."),
        ("no_backtests_run", "confirmed", "Dry-run plan would_execute_backtest=False.", "No prototype result is generated."),
        ("no_new_results_generated", "confirmed", "Outputs are registry, validation, contracts, and guardrails only.", "No performance claim is made."),
        ("no_thresholds_changed", "confirmed", "No threshold write path exists in Step 41.", "0.65 / 0.50 remains unchanged."),
        ("no_model_retrained", "confirmed", "No model training functions are called.", "Model context remains unchanged."),
        ("no_features_changed", "confirmed", "No feature engineering code is touched.", "Feature set remains unchanged."),
        ("no_trading_ready_claim", "confirmed", "Report states no candidate is trading-ready.", "Research status remains unchanged."),
    ]
    return pd.DataFrame(
        [{"item": item, "status": status, "evidence": evidence, "notes": notes} for item, status, evidence, notes in rows]
    )


def build_report(
    prototype_design_dir: str | Path,
    diagnostics_dir: str | Path | None,
    integrated_dir: str | Path | None,
    registry: pd.DataFrame,
    validation: pd.DataFrame,
    baseline_requirements: pd.DataFrame,
    metric_contract: pd.DataFrame,
) -> str:
    failed = 0
    if not validation.empty and "status" in validation:
        failed = int((validation["status"].astype(str) == "failed").sum())
    return "\n".join(
        [
            "# V4 Step 41 Bull Prototype Experiment Harness Report",
            "",
            "## Executive Summary",
            "Step 41 creates harness infrastructure only.",
            "No prototype was executed.",
            "No real prototype backtest was run.",
            "No threshold/model/feature/data source/agent was changed.",
            "No new performance claim is made.",
            "canonical_reduced_40 remains research-only.",
            "Bull remediation remains failed.",
            "No candidate is trading-ready.",
            "",
            "## Inputs Used",
            f"- Step 40 prototype design directory: {prototype_design_dir}",
            f"- Step 38 diagnostics directory: {diagnostics_dir or 'not provided'}",
            f"- Step 36 integrated directory: {integrated_dir or 'not provided'}",
            "",
            "## Prototype Registry",
            f"- Registered prototypes: {len(registry)}",
            "- All registry rows are marked not_executed and allowed_to_execute_in_step41=False.",
            "",
            "## Dry-Run Plan",
            "The dry-run plan records future setup actions only. Every row has would_execute_backtest=False.",
            "",
            "## Config Validation",
            f"- Validation failures: {failed}",
            "Validation checks confirm design-only status, not-executed status, current-step execution block, baseline contract presence, symbol preservation, and no trading-ready claim.",
            "",
            "## Baseline Requirements",
            f"- Baseline requirement rows: {len(baseline_requirements)}",
            "Future Step 42 must compare against the unchanged selected threshold, aggregate baseline metrics, trade diagnostics, and symbol-level 601318 / 600036 diagnostics.",
            "",
            "## Metric Contract",
            f"- Metric contract rows: {len(metric_contract)}",
            "No prototype passes or fails in Step 41; the contract only defines future comparison requirements.",
            "",
            "## Execution Guardrails",
            "Guardrails block real backtest execution, threshold changes, model retraining, feature changes, new data sources, new agents, new performance claims, and trading-ready upgrades.",
            "",
            "## Limitations",
            "This is harness-only infrastructure. Future results are unknown, overfitting risk remains, and baseline comparison is mandatory.",
            "",
            "## Why This Does Not Change Trading-Ready Status",
            "Step 41 does not execute prototypes or validate remediation.",
            "It does not change Step 34, Step 36, Step 38, Step 39, or Step 40 conclusions.",
            "canonical_reduced_40 remains research_only_not_trading_ready.",
            "Bull remediation remains failed and no candidate is trading-ready.",
            "",
            "## Recommended Next Step",
            "Recommended next step: V4 Step 42 Bull Prototype Controlled Backtest Execution.",
            "",
            "## Educational / Research Disclaimer",
            "This report is educational/research diagnostics only. It is not financial advice.",
            "No strategy, model, threshold, symbol, prototype, or candidate in this report should be treated as deployable or trading-ready.",
            "",
        ]
    )


def generate_bull_prototype_experiment_harness(
    prototype_design_dir: str | Path,
    diagnostics_dir: str | Path | None,
    integrated_dir: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    design = Path(prototype_design_dir)
    diagnostics = Path(diagnostics_dir) if diagnostics_dir else None
    integrated = Path(integrated_dir) if integrated_dir else None
    specs = _read_csv(design / "bull_prototype_experiment_specs.csv")
    metric_plan = _read_csv(design / "bull_prototype_metric_plan.csv")
    design_guardrails = _read_csv(design / "bull_prototype_guardrails.csv")
    design_config = _read_json(design / "run_config.json")
    trades = _read_csv(diagnostics / "bull_trade_level_diagnostics.csv", dtype={"symbol": str}) if diagnostics else pd.DataFrame()
    windows = _read_csv(diagnostics / "bull_window_diagnostics.csv", dtype={"symbol": str}) if diagnostics else pd.DataFrame()
    integrated_summary = _read_csv(integrated / "integrated_remediation_summary.csv") if integrated else pd.DataFrame()
    registry = build_prototype_registry(specs)
    dry_run_plan = build_dry_run_plan(registry)
    config_validation = build_config_validation(specs, metric_plan)
    baseline_requirements = build_baseline_requirements(metric_plan, integrated_summary, trades, windows)
    metric_contract = build_metric_contract(metric_plan)
    execution_guardrails = build_execution_guardrails()
    limitations = build_limitations()
    not_executed_log = build_not_executed_log()
    report = build_report(
        prototype_design_dir,
        diagnostics_dir,
        integrated_dir,
        registry,
        config_validation,
        baseline_requirements,
        metric_contract,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {key: output_path / filename for key, filename in OUTPUT_FILENAMES.items()}
    paths["report"].write_text(report, encoding="utf-8")
    registry.to_csv(paths["registry"], index=False)
    dry_run_plan.to_csv(paths["dry_run_plan"], index=False)
    config_validation.to_csv(paths["config_validation"], index=False)
    baseline_requirements.to_csv(paths["baseline_requirements"], index=False)
    metric_contract.to_csv(paths["metric_contract"], index=False)
    execution_guardrails.to_csv(paths["execution_guardrails"], index=False)
    limitations.to_csv(paths["limitations"], index=False)
    not_executed_log.to_csv(paths["not_executed_log"], index=False)
    run_config = {
        "prototype_design_dir": str(prototype_design_dir),
        "diagnostics_dir": str(diagnostics_dir) if diagnostics_dir else None,
        "integrated_dir": str(integrated_dir) if integrated_dir else None,
        "output_dir": str(output_path),
        "candidate": design_config.get("candidate", "canonical_reduced_40"),
        "model": design_config.get("model", "logistic_regression"),
        "buy_threshold": design_config.get("buy_threshold", 0.65),
        "sell_threshold": design_config.get("sell_threshold", 0.50),
        "threshold_action": "reused_for_harness_baseline_only",
        "harness_status": "registered_for_future_controlled_test",
        "execution_status": "not_executed",
        "trading_ready": False,
        "prototype_count": int(len(registry)),
        "step40_guardrail_rows": design_guardrails.to_dict(orient="records") if not design_guardrails.empty else [],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    paths["run_config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "bull_prototype_experiment_harness_report": report,
        "bull_prototype_registry": registry,
        "bull_prototype_dry_run_plan": dry_run_plan,
        "bull_prototype_config_validation": config_validation,
        "bull_prototype_baseline_requirements": baseline_requirements,
        "bull_prototype_metric_contract": metric_contract,
        "bull_prototype_execution_guardrails": execution_guardrails,
        "bull_prototype_harness_limitations": limitations,
        "bull_prototype_not_executed_log": not_executed_log,
        "run_config": run_config,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
