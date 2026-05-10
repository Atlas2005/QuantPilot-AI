import subprocess
import sys
from pathlib import Path


PY_COMPILE_FILES = [
    "app.py",
    "src/check_setup.py",
    "src/run_demo.py",
    "src/real_data_loader.py",
    "src/factor_builder.py",
    "src/build_factor_dataset.py",
    "src/dataset_splitter.py",
    "src/split_factor_dataset.py",
    "src/model_trainer.py",
    "src/train_baseline_model.py",
    "src/model_predictor.py",
    "src/predict_with_model.py",
    "src/model_evaluator.py",
    "src/evaluate_model.py",
    "src/ml_signal_backtester.py",
    "src/run_ml_signal_backtest.py",
    "src/ml_threshold_experiment.py",
    "src/run_ml_threshold_experiment.py",
    "src/batch_model_trainer.py",
    "src/run_batch_model_training.py",
    "src/run_real_p0_robustness.py",
    "src/factor_ablation.py",
    "src/run_factor_ablation.py",
    "src/run_batch_factor_ablation.py",
    "src/factor_decision_report.py",
    "src/generate_factor_decision_report.py",
    "src/factor_pruning_experiment.py",
    "src/run_factor_pruning_experiment.py",
    "src/pruning_summary_report.py",
    "src/generate_pruning_summary_report.py",
    "src/reduced_feature_backtest.py",
    "src/run_reduced_feature_backtest.py",
    "src/reduced_feature_backtest_report.py",
    "src/generate_reduced_feature_backtest_report.py",
    "src/reduced_feature_threshold_experiment.py",
    "src/run_reduced_feature_threshold_experiment.py",
    "src/generate_threshold_experiment_report.py",
    "src/threshold_decision_report.py",
    "src/generate_threshold_decision_report.py",
    "src/candidate_expanded_validation.py",
    "src/run_candidate_expanded_validation.py",
    "src/candidate_stress_test.py",
    "src/run_candidate_stress_test.py",
    "src/candidate_equivalence_audit.py",
    "src/run_candidate_equivalence_audit.py",
    "src/candidate_mode_normalization.py",
    "src/run_candidate_mode_normalization.py",
    "src/canonical_candidate_revalidation_report.py",
    "src/generate_canonical_candidate_revalidation_report.py",
    "src/candidate_validation_gate.py",
    "src/run_candidate_validation_gate.py",
    "src/validation_gate_failure_analysis.py",
    "src/run_validation_gate_failure_analysis.py",
    "src/targeted_remediation_design.py",
    "src/run_targeted_remediation_design.py",
    "src/bull_regime_threshold_remediation.py",
    "src/run_bull_regime_threshold_remediation.py",
    "src/sideways_regime_trade_sufficiency_remediation.py",
    "src/run_sideways_regime_trade_sufficiency_remediation.py",
    "src/integrated_remediation_revalidation.py",
    "src/run_integrated_remediation_revalidation.py",
    "src/bull_regime_failure_drilldown.py",
    "src/run_bull_regime_failure_drilldown.py",
    "src/bull_trade_window_diagnostics.py",
    "src/run_bull_trade_window_diagnostics.py",
    "src/bull_error_pattern_remediation_design.py",
    "src/run_bull_error_pattern_remediation_design.py",
    "src/bull_remediation_prototype_design.py",
    "src/run_bull_remediation_prototype_design.py",
    "src/bull_prototype_experiment_harness.py",
    "src/run_bull_prototype_experiment_harness.py",
    "src/bull_prototype_controlled_backtest.py",
    "src/run_bull_prototype_controlled_backtest.py",
    "src/bull_prototype_result_review.py",
    "src/run_bull_prototype_result_review.py",
    "src/project_retrospective_v1_v4.py",
    "src/run_project_retrospective_v1_v4.py",
    "src/capital_constraint_engine.py",
    "src/run_capital_constraint_engine.py",
    "src/tradable_universe_filter.py",
    "src/run_tradable_universe_filter.py",
    "src/position_sizing_engine.py",
    "src/run_position_sizing_engine.py",
    "src/exit_engine.py",
    "src/run_exit_engine.py",
    "src/daily_trading_plan.py",
    "src/run_daily_trading_plan.py",
    "src/paper_trading_ledger.py",
    "src/run_paper_trading_ledger.py",
    "src/semi_auto_order_generator.py",
    "src/run_semi_auto_order_generator.py",
    "src/broker_integration_research.py",
    "src/run_broker_integration_research.py",
    "src/monitoring_reporting_layer.py",
    "src/run_monitoring_reporting_layer.py",
    "src/capital_aware_infrastructure_review.py",
    "src/run_capital_aware_infrastructure_review.py",
    "src/validation_baseline_manifest.py",
    "src/run_validation_baseline_manifest.py",
    "src/output_schema_validator.py",
    "src/run_output_schema_validator.py",
    "src/cross_step_dependency_validator.py",
    "src/run_cross_step_dependency_validator.py",
    "src/reproducibility_rerun_validator.py",
    "src/run_reproducibility_rerun_validator.py",
    "src/model_report_generator.py",
    "src/generate_model_report.py",
    "src/feature_source_registry.py",
    "src/show_feature_sources.py",
    "src/feature_implementation_queue.py",
    "src/show_feature_queue.py",
    "src/run_stock_backtest.py",
    "src/run_batch_experiment.py",
    "src/run_period_experiment.py",
    "src/analyze_period_results.py",
    "src/indicators.py",
    "src/strategy.py",
    "src/backtester.py",
    "src/metrics.py",
    "src/trade_metrics.py",
    "src/report.py",
]

COMMAND_CHECKS = [
    ("setup check", ["src/check_setup.py"]),
    ("app import check", ["-c", "import app"]),
    ("offline demo", ["src/run_demo.py"]),
    ("real_data_loader help", ["src/real_data_loader.py", "--help"]),
    ("build_factor_dataset help", ["src/build_factor_dataset.py", "--help"]),
    ("split_factor_dataset help", ["src/split_factor_dataset.py", "--help"]),
    ("train_baseline_model help", ["src/train_baseline_model.py", "--help"]),
    ("predict_with_model help", ["src/predict_with_model.py", "--help"]),
    ("evaluate_model help", ["src/evaluate_model.py", "--help"]),
    ("run_ml_signal_backtest help", ["src/run_ml_signal_backtest.py", "--help"]),
    (
        "run_ml_threshold_experiment help",
        ["src/run_ml_threshold_experiment.py", "--help"],
    ),
    ("run_batch_model_training help", ["src/run_batch_model_training.py", "--help"]),
    ("run_real_p0_robustness help", ["src/run_real_p0_robustness.py", "--help"]),
    ("run_factor_ablation help", ["src/run_factor_ablation.py", "--help"]),
    ("run_batch_factor_ablation help", ["src/run_batch_factor_ablation.py", "--help"]),
    (
        "generate_factor_decision_report help",
        ["src/generate_factor_decision_report.py", "--help"],
    ),
    (
        "run_factor_pruning_experiment help",
        ["src/run_factor_pruning_experiment.py", "--help"],
    ),
    (
        "generate_pruning_summary_report help",
        ["src/generate_pruning_summary_report.py", "--help"],
    ),
    (
        "run_reduced_feature_backtest help",
        ["src/run_reduced_feature_backtest.py", "--help"],
    ),
    (
        "generate_reduced_feature_backtest_report help",
        ["src/generate_reduced_feature_backtest_report.py", "--help"],
    ),
    (
        "run_reduced_feature_threshold_experiment help",
        ["src/run_reduced_feature_threshold_experiment.py", "--help"],
    ),
    (
        "generate_threshold_experiment_report help",
        ["src/generate_threshold_experiment_report.py", "--help"],
    ),
    (
        "generate_threshold_decision_report help",
        ["src/generate_threshold_decision_report.py", "--help"],
    ),
    (
        "run_candidate_expanded_validation help",
        ["src/run_candidate_expanded_validation.py", "--help"],
    ),
    (
        "run_candidate_stress_test help",
        ["src/run_candidate_stress_test.py", "--help"],
    ),
    (
        "run_candidate_equivalence_audit help",
        ["src/run_candidate_equivalence_audit.py", "--help"],
    ),
    (
        "run_candidate_mode_normalization help",
        ["src/run_candidate_mode_normalization.py", "--help"],
    ),
    (
        "generate_canonical_candidate_revalidation_report help",
        ["src/generate_canonical_candidate_revalidation_report.py", "--help"],
    ),
    (
        "run_candidate_validation_gate help",
        ["src/run_candidate_validation_gate.py", "--help"],
    ),
    (
        "run_validation_gate_failure_analysis help",
        ["src/run_validation_gate_failure_analysis.py", "--help"],
    ),
    (
        "run_targeted_remediation_design help",
        ["src/run_targeted_remediation_design.py", "--help"],
    ),
    (
        "run_bull_regime_threshold_remediation help",
        ["src/run_bull_regime_threshold_remediation.py", "--help"],
    ),
    (
        "run_sideways_regime_trade_sufficiency_remediation help",
        ["src/run_sideways_regime_trade_sufficiency_remediation.py", "--help"],
    ),
    (
        "run_integrated_remediation_revalidation help",
        ["src/run_integrated_remediation_revalidation.py", "--help"],
    ),
    (
        "integrated remediation revalidation import",
        ["-c", "import src.integrated_remediation_revalidation"],
    ),
    (
        "run_bull_regime_failure_drilldown help",
        ["src/run_bull_regime_failure_drilldown.py", "--help"],
    ),
    (
        "bull regime failure drilldown import",
        ["-c", "import src.bull_regime_failure_drilldown"],
    ),
    (
        "run_bull_trade_window_diagnostics help",
        ["src/run_bull_trade_window_diagnostics.py", "--help"],
    ),
    (
        "bull trade window diagnostics import",
        ["-c", "import src.bull_trade_window_diagnostics"],
    ),
    (
        "run_bull_error_pattern_remediation_design help",
        ["src/run_bull_error_pattern_remediation_design.py", "--help"],
    ),
    (
        "bull error pattern remediation design import",
        ["-c", "import src.bull_error_pattern_remediation_design"],
    ),
    (
        "offline synthetic bull error remediation design inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import json\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/bull_error_pattern_remediation_design_smoke_inputs')\\n"
                "diag=base/'diag'\\n"
                "drill=base/'drill'\\n"
                "integrated=base/'integrated'\\n"
                "for d in [diag, drill, integrated]: d.mkdir(parents=True, exist_ok=True)\\n"
                "pd.DataFrame([\\n"
                "{'symbol':'000001','candidate':'canonical_reduced_40','model':'logistic_regression','regime':'bull','buy_threshold':0.65,'sell_threshold':0.50,'entry_date':'2024-01-01','exit_date':'2024-01-03','holding_days':2,'trade_return_pct':1.0,'benchmark_return_pct':2.0,'trade_excess_pct':-1.0,'was_profitable':True,'beat_benchmark':False,'error_pattern':'positive_return_but_lagged_benchmark'},\\n"
                "{'symbol':'601318','candidate':'canonical_reduced_40','model':'logistic_regression','regime':'bull','buy_threshold':0.65,'sell_threshold':0.50,'entry_date':'2024-02-01','exit_date':'2024-04-15','holding_days':74,'trade_return_pct':6.0,'benchmark_return_pct':8.0,'trade_excess_pct':-2.0,'was_profitable':True,'beat_benchmark':False,'error_pattern':'positive_return_but_lagged_benchmark'},\\n"
                "{'symbol':'600519','candidate':'canonical_reduced_40','model':'logistic_regression','regime':'bull','buy_threshold':0.65,'sell_threshold':0.50,'entry_date':'2024-03-01','exit_date':'2024-03-20','holding_days':19,'trade_return_pct':-3.0,'benchmark_return_pct':-1.0,'trade_excess_pct':-2.0,'was_profitable':False,'beat_benchmark':False,'error_pattern':'negative_trade_return'}\\n"
                "]).to_csv(diag/'bull_trade_level_diagnostics.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'symbol':'000001','window_id':1,'start_date':'2024-01-01','end_date':'2024-01-20','rows':20,'strategy_return_pct':1.0,'benchmark_return_pct':3.0,'excess_return_pct':-2.0,'max_drawdown_pct':-1.0,'trade_count':1,'error_pattern':'positive_return_but_lagged_benchmark'},\\n"
                "{'symbol':'601318','window_id':1,'start_date':'2024-02-01','end_date':'2024-02-20','rows':20,'strategy_return_pct':0.0,'benchmark_return_pct':5.0,'excess_return_pct':-5.0,'max_drawdown_pct':-12.0,'trade_count':0,'error_pattern':'positive_return_but_lagged_benchmark'},\\n"
                "{'symbol':'600519','window_id':1,'start_date':'2024-03-01','end_date':'2024-03-20','rows':20,'strategy_return_pct':-3.0,'benchmark_return_pct':-1.0,'excess_return_pct':-2.0,'max_drawdown_pct':-4.0,'trade_count':1,'error_pattern':'negative_trade_return'}\\n"
                "]).to_csv(diag/'bull_window_diagnostics.csv', index=False)\\n"
                "pd.DataFrame([{'error_pattern':'positive_return_but_lagged_benchmark','count':2,'affected_symbols':'000001,601318'},{'error_pattern':'negative_trade_return','count':1,'affected_symbols':'600519'}]).to_csv(diag/'bull_error_pattern_summary.csv', index=False)\\n"
                "(diag/'run_config.json').write_text(json.dumps({'candidate':'canonical_reduced_40','model':'logistic_regression','buy_threshold':0.65,'sell_threshold':0.50}), encoding='utf-8')\\n"
                "pd.DataFrame([{'symbol':'601318','strategy_vs_benchmark_pct':-17.5,'contribution_to_avg_excess_pct':-3.5,'interpretation':'negative_contributor_to_bull_average'},{'symbol':'600036','strategy_vs_benchmark_pct':-0.2,'contribution_to_avg_excess_pct':-0.04,'interpretation':'negative_contributor_to_bull_average'}]).to_csv(drill/'bull_failure_contribution.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','model':'logistic_regression','overall_decision':'research_only_not_trading_ready','trading_ready':False,'bull_final_decision':'bull_remediation_failed','bull_avg_strategy_vs_benchmark_pct':-0.0837457193063983,'main_blocker':'bull_remediation_failed'}]).to_csv(integrated/'integrated_remediation_summary.csv', index=False)\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline bull error remediation design",
        [
            "src/run_bull_error_pattern_remediation_design.py",
            "--diagnostics-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/diag",
            "--drilldown-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/drill",
            "--integrated-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/integrated",
            "--output-dir",
            "outputs/bull_error_pattern_remediation_design_smoke",
        ],
    ),
    (
        "offline bull error remediation design assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/bull_error_pattern_remediation_design_smoke'); "
                "required=['bull_error_pattern_remediation_design_report.md','bull_trade_error_classification.csv','bull_window_error_classification.csv','bull_symbol_error_profile.csv','bull_aggregate_error_profile.csv','bull_remediation_design_options.csv','bull_remediation_priority_matrix.csv','bull_no_change_guardrails.csv','bull_design_limitations.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "trade=pd.read_csv(base/'bull_trade_error_classification.csv', dtype={'symbol': str}); "
                "assert {'000001','601318','600519'} <= set(trade['symbol']); "
                "assert trade['symbol'].astype(str).str.len().ge(6).all(); "
                "assert 'late_entry' not in set(trade['classified_error_pattern']); "
                "options=pd.read_csv(base/'bull_remediation_design_options.csv'); "
                "assert not options['allowed_in_current_step'].fillna(True).astype(bool).any(); "
                "assert set(options['implementation_status']) == {'design_only_not_implemented'}; "
                "guardrails=pd.read_csv(base/'bull_no_change_guardrails.csv'); "
                "required_guardrails={'no_threshold_change','no_model_retraining','no_feature_engineering_change','no_new_data_sources','no_new_agents','no_trading_ready_upgrade'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "aggregate=pd.read_csv(base/'bull_aggregate_error_profile.csv'); "
                "values=dict(zip(aggregate['metric'], aggregate['value'])); "
                "assert values['bull_final_decision']=='bull_remediation_failed'; "
                "assert values['trading_ready'] in ['False', False]; "
                "report=(base/'bull_error_pattern_remediation_design_report.md').read_text(encoding='utf-8'); "
                "phrases=['does not implement remediation','No threshold, model, factor, data source, or agent was changed','No candidate is trading-ready','V4 Step 40']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_bull_remediation_prototype_design help",
        ["src/run_bull_remediation_prototype_design.py", "--help"],
    ),
    (
        "bull remediation prototype design import",
        ["-c", "import src.bull_remediation_prototype_design"],
    ),
    (
        "offline bull remediation prototype design",
        [
            "src/run_bull_remediation_prototype_design.py",
            "--design-dir",
            "outputs/bull_error_pattern_remediation_design_smoke",
            "--diagnostics-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/diag",
            "--integrated-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/integrated",
            "--output-dir",
            "outputs/bull_remediation_prototype_design_smoke",
        ],
    ),
    (
        "offline bull remediation prototype design assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/bull_remediation_prototype_design_smoke'); "
                "required=['bull_remediation_prototype_design_report.md','bull_prototype_experiment_specs.csv','bull_prototype_metric_plan.csv','bull_prototype_guardrails.csv','bull_prototype_risk_assessment.csv','bull_prototype_execution_plan.csv','bull_prototype_not_implemented_log.csv','bull_prototype_priority_ranking.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "specs=pd.read_csv(base/'bull_prototype_experiment_specs.csv'); "
                "assert set(specs['implementation_status']) == {'prototype_design_only'}; "
                "assert set(specs['execution_status']) == {'not_executed'}; "
                "assert not specs['allowed_in_current_step'].fillna(True).astype(bool).any(); "
                "assert {'benchmark_lag_reduction_for_601318','negative_trade_cluster_review_for_000858_600519_600036'} <= set(specs['prototype_name']); "
                "targets=' '.join(specs['target_symbols'].astype(str)); "
                "assert '601318' in targets and '600519' in targets; "
                "guardrails=pd.read_csv(base/'bull_prototype_guardrails.csv'); "
                "required_guardrails={'no_execution_in_step40','no_threshold_change','no_model_retraining','no_feature_engineering_change','no_new_data_sources','no_new_agents','no_trading_ready_upgrade'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "log=pd.read_csv(base/'bull_prototype_not_implemented_log.csv'); "
                "assert {'prototypes_not_executed','no_trading_ready_claim'} <= set(log['item']); "
                "metrics=pd.read_csv(base/'bull_prototype_metric_plan.csv'); "
                "assert {'avg_strategy_vs_benchmark_pct','symbol_level_excess_for_601318','symbol_level_excess_for_600036'} <= set(metrics['metric_name']); "
                "report=(base/'bull_remediation_prototype_design_report.md').read_text(encoding='utf-8'); "
                "phrases=['No prototype was executed','No threshold was changed','No candidate is trading-ready','V4 Step 41']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_bull_prototype_experiment_harness help",
        ["src/run_bull_prototype_experiment_harness.py", "--help"],
    ),
    (
        "bull prototype experiment harness import",
        ["-c", "import src.bull_prototype_experiment_harness"],
    ),
    (
        "offline bull prototype experiment harness",
        [
            "src/run_bull_prototype_experiment_harness.py",
            "--prototype-design-dir",
            "outputs/bull_remediation_prototype_design_smoke",
            "--diagnostics-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/diag",
            "--integrated-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/integrated",
            "--output-dir",
            "outputs/bull_prototype_experiment_harness_smoke",
        ],
    ),
    (
        "offline bull prototype experiment harness assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/bull_prototype_experiment_harness_smoke'); "
                "required=['bull_prototype_experiment_harness_report.md','bull_prototype_registry.csv','bull_prototype_dry_run_plan.csv','bull_prototype_config_validation.csv','bull_prototype_baseline_requirements.csv','bull_prototype_metric_contract.csv','bull_prototype_execution_guardrails.csv','bull_prototype_harness_limitations.csv','bull_prototype_not_executed_log.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "registry=pd.read_csv(base/'bull_prototype_registry.csv'); "
                "assert set(registry['execution_status']) == {'not_executed'}; "
                "assert not registry['allowed_to_execute_in_step41'].fillna(True).astype(bool).any(); "
                "targets=' '.join(registry['target_symbols'].astype(str)); "
                "assert '601318' in targets and '600519' in targets; "
                "dry=pd.read_csv(base/'bull_prototype_dry_run_plan.csv'); "
                "assert not dry['would_execute_backtest'].fillna(True).astype(bool).any(); "
                "assert not dry['would_modify_model'].fillna(True).astype(bool).any(); "
                "assert not dry['would_modify_features'].fillna(True).astype(bool).any(); "
                "assert not dry['would_modify_threshold'].fillna(True).astype(bool).any(); "
                "guardrails=pd.read_csv(base/'bull_prototype_execution_guardrails.csv'); "
                "required_guardrails={'no_real_backtest_execution_in_step41','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_new_agents','no_trading_ready_upgrade'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "validation=pd.read_csv(base/'bull_prototype_config_validation.csv'); "
                "assert 'target_symbols_preserved' in set(validation['validation_check']); "
                "log=pd.read_csv(base/'bull_prototype_not_executed_log.csv'); "
                "assert {'prototypes_registered_not_executed','no_backtests_run','no_trading_ready_claim'} <= set(log['item']); "
                "report=(base/'bull_prototype_experiment_harness_report.md').read_text(encoding='utf-8'); "
                "phrases=['No prototype was executed','No real prototype backtest was run','No new performance claim is made','V4 Step 42']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_bull_prototype_controlled_backtest help",
        ["src/run_bull_prototype_controlled_backtest.py", "--help"],
    ),
    (
        "bull prototype controlled backtest import",
        ["-c", "import src.bull_prototype_controlled_backtest"],
    ),
    (
        "offline bull prototype controlled backtest",
        [
            "src/run_bull_prototype_controlled_backtest.py",
            "--harness-dir",
            "outputs/bull_prototype_experiment_harness_smoke",
            "--prototype-design-dir",
            "outputs/bull_remediation_prototype_design_smoke",
            "--diagnostics-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/diag",
            "--integrated-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/integrated",
            "--output-dir",
            "outputs/bull_prototype_controlled_backtest_smoke",
        ],
    ),
    (
        "offline bull prototype controlled backtest assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import json; "
                "import pandas as pd; "
                "base=Path('outputs/bull_prototype_controlled_backtest_smoke'); "
                "required=['bull_prototype_controlled_backtest_report.md','bull_prototype_execution_results.csv','bull_prototype_metric_comparison.csv','bull_prototype_symbol_comparison.csv','bull_prototype_trade_comparison.csv','bull_prototype_window_comparison.csv','bull_prototype_decision_summary.csv','bull_prototype_execution_audit.csv','bull_prototype_guardrail_check.csv','bull_prototype_limitations.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "results=pd.read_csv(base/'bull_prototype_execution_results.csv'); "
                "assert not results.empty; "
                "assert not results['trading_ready'].fillna(True).astype(bool).any(); "
                "assert results['execution_status'].isin(['executed','not_executable_with_current_data']).any(); "
                "guardrails=pd.read_csv(base/'bull_prototype_guardrail_check.csv'); "
                "assert not (guardrails['status'].astype(str)!='confirmed').any(); "
                "required_guardrails={'no_threshold_change','no_model_retraining','no_feature_engineering_change','no_new_data_sources','no_new_agents','no_trading_ready_upgrade'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "audit=pd.read_csv(base/'bull_prototype_execution_audit.csv'); "
                "assert {'no_threshold_change','no_model_retraining','no_feature_engineering_change','no_new_data_sources','no_new_agents','no_trading_ready_claim'} <= set(audit['audit_item']); "
                "config=json.loads((base/'run_config.json').read_text(encoding='utf-8')); "
                "executed=results[results['execution_status'].astype(str)=='executed'].copy(); "
                "executed['delta_avg_excess_pct']=pd.to_numeric(executed['delta_avg_excess_pct'], errors='coerce'); "
                "bad_primary=executed[executed['delta_avg_excess_pct'].le(0)]; "
                "assert not (bad_primary['conservative_result'].astype(str)=='improved_but_not_validated').any(); "
                "decisions=pd.read_csv(base/'bull_prototype_decision_summary.csv'); "
                "bad_decisions=decisions[decisions['prototype_id'].isin(set(bad_primary['prototype_id']))]; "
                "assert not bad_decisions['can_advance_to_further_testing'].fillna(True).astype(bool).any(); "
                "assert not decisions['trading_ready'].fillna(True).astype(bool).any(); "
                "best_summary=config.get('best_avg_excess_summary', {}); "
                "no_avg_improve=executed.empty or not executed['delta_avg_excess_pct'].gt(0).any(); "
                "assert (not no_avg_improve) or best_summary.get('best_diagnostic_candidate') == 'no_avg_excess_improvement', best_summary; "
                "assert no_avg_improve or best_summary.get('best_diagnostic_candidate') != 'no_avg_excess_improvement', best_summary; "
                "symbols=pd.read_csv(base/'bull_prototype_symbol_comparison.csv', dtype={'symbol': str}); "
                "assert symbols.empty or symbols['symbol'].astype(str).str.len().ge(6).all(); "
                "report=(base/'bull_prototype_controlled_backtest_report.md').read_text(encoding='utf-8'); "
                "phrases=['No candidate is trading-ready','No model was retrained','0.65 / 0.50 threshold remains unchanged','V4 Step 43','Best diagnostic candidate by average excess']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_bull_prototype_result_review help",
        ["src/run_bull_prototype_result_review.py", "--help"],
    ),
    (
        "bull prototype result review import",
        ["-c", "import src.bull_prototype_result_review"],
    ),
    (
        "offline bull prototype result review",
        [
            "src/run_bull_prototype_result_review.py",
            "--controlled-backtest-dir",
            "outputs/bull_prototype_controlled_backtest_smoke",
            "--integrated-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/integrated",
            "--error-design-dir",
            "outputs/bull_error_pattern_remediation_design_smoke",
            "--diagnostics-dir",
            "outputs/bull_error_pattern_remediation_design_smoke_inputs/diag",
            "--output-dir",
            "outputs/bull_prototype_result_review_smoke",
        ],
    ),
    (
        "offline bull prototype result review assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import json; "
                "import pandas as pd; "
                "base=Path('outputs/bull_prototype_result_review_smoke'); "
                "required=['bull_prototype_result_review_report.md','bull_prototype_review_summary.csv','bull_candidate_selection.csv','bull_unresolved_blockers.csv','bull_v4_closure_status.csv','bull_transition_to_v5_recommendation.csv','bull_result_review_guardrails.csv','bull_result_review_limitations.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'bull_prototype_review_summary.csv'); "
                "assert not summary.empty; "
                "assert not summary['trading_ready'].fillna(True).astype(bool).any(); "
                "summary['delta_avg_excess_pct']=pd.to_numeric(summary['delta_avg_excess_pct'], errors='coerce'); "
                "bad=summary[summary['delta_avg_excess_pct'].le(0)]; "
                "assert not bad['reviewed_can_advance_to_further_validation'].fillna(True).astype(bool).any(); "
                "selection=pd.read_csv(base/'bull_candidate_selection.csv'); "
                "assert selection.iloc[0]['selected_candidate']=='none'; "
                "assert selection.iloc[0]['selection_status']=='no_candidate_selected'; "
                "assert not bool(selection.iloc[0]['trading_ready']); "
                "guardrails=pd.read_csv(base/'bull_result_review_guardrails.csv'); "
                "required_guardrails={'no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_new_agents','no_trading_ready_upgrade'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "v5=pd.read_csv(base/'bull_transition_to_v5_recommendation.csv'); "
                "assert v5.iloc[0]['recommended_step']=='V5 Step 1'; "
                "assert v5.iloc[0]['step_name']=='Capital Constraint Engine'; "
                "config=json.loads((base/'run_config.json').read_text(encoding='utf-8')); "
                "symbols=config.get('step42_symbol_comparison_symbols', []); "
                "assert not symbols or all(len(str(symbol)) >= 6 for symbol in symbols); "
                "report=(base/'bull_prototype_result_review_report.md').read_text(encoding='utf-8'); "
                "phrases=['V4 can close as a research-diagnostic validation cycle','No candidate is selected for further validation from Step 42','canonical_reduced_40 remains research-only','V5 Step 1 Capital Constraint Engine']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_project_retrospective_v1_v4 help",
        ["src/run_project_retrospective_v1_v4.py", "--help"],
    ),
    (
        "project retrospective v1 v4 import",
        ["-c", "import src.project_retrospective_v1_v4"],
    ),
    (
        "offline project retrospective v1 v4",
        [
            "src/run_project_retrospective_v1_v4.py",
            "--project-root",
            ".",
            "--output-dir",
            "outputs/project_retrospective_v1_v4_smoke",
        ],
    ),
    (
        "offline project retrospective v1 v4 assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import json; "
                "import pandas as pd; "
                "base=Path('outputs/project_retrospective_v1_v4_smoke'); "
                "required=['project_retrospective_v1_v4_report.md','phase_progress_summary.csv','architecture_layer_summary.csv','current_capability_inventory.csv','reliable_conclusions.csv','unresolved_limitations.csv','recommended_next_phase.csv','project_retrospective_guardrails.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "phase=pd.read_csv(base/'phase_progress_summary.csv'); "
                "assert {'V1','V2','V3','V4'} <= set(phase['phase']); "
                "cap=pd.read_csv(base/'current_capability_inventory.csv'); "
                "capabilities=dict(zip(cap['capability'], cap['available'].astype(str))); "
                "assert capabilities['capital feasibility checks'] in {'False','false','0'}; "
                "assert capabilities['paper trading ledger'] in {'False','false','0'}; "
                "assert not cap['production_ready'].fillna(True).astype(bool).any(); "
                "next_phase=pd.read_csv(base/'recommended_next_phase.csv'); "
                "assert next_phase.iloc[0]['recommended_step']=='V5 Step 1'; "
                "assert next_phase.iloc[0]['step_name']=='Capital Constraint Engine'; "
                "guardrails=pd.read_csv(base/'project_retrospective_guardrails.csv'); "
                "required_guardrails={'no_new_backtests','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_new_agents','no_previous_outputs_overwritten','no_trading_ready_claim','audit_only','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails['status']) == {'confirmed'}; "
                "config=json.loads((base/'run_config.json').read_text(encoding='utf-8')); "
                "assert config['audit_only'] is True; "
                "assert config['educational_research_only'] is True; "
                "assert config['trading_ready'] is False; "
                "conclusions=pd.read_csv(base/'reliable_conclusions.csv'); "
                "assert 'no_candidate_trading_ready' in set(conclusions['conclusion_id']); "
                "report=(base/'project_retrospective_v1_v4_report.md').read_text(encoding='utf-8'); "
                "phrases=['The project is not trading-ready','No candidate should be treated as deployable','V5 Step 1 Capital Constraint Engine','This retrospective is educational/research diagnostics only']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_capital_constraint_engine help",
        ["src/run_capital_constraint_engine.py", "--help"],
    ),
    (
        "capital constraint engine import",
        ["-c", "import src.capital_constraint_engine"],
    ),
    (
        "offline capital constraint engine",
        [
            "src/run_capital_constraint_engine.py",
            "--cash",
            "1000",
            "--output-dir",
            "outputs/capital_constraint_engine_smoke",
        ],
    ),
    (
        "offline capital constraint engine assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/capital_constraint_engine_smoke'); "
                "required=['capital_feasibility.csv','approved_orders.csv','rejected_orders.csv','capital_constraint_summary.csv','capital_constraint_report.md','capital_constraint_guardrails.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "feas=pd.read_csv(base/'capital_feasibility.csv', dtype={'symbol': str}); "
                "assert '600519' in set(feas['symbol']); "
                "row_600519=feas[feas['symbol']=='600519'].iloc[0]; "
                "assert not bool(row_600519['order_allowed']); "
                "assert row_600519['rejection_reason']=='insufficient_cash_for_min_lot'; "
                "approved=pd.read_csv(base/'approved_orders.csv', dtype={'symbol': str}); "
                "assert '600000' in set(approved['symbol']); "
                "star=feas[feas['symbol']=='688001'].iloc[0]; "
                "assert int(star['lot_size']) == 200; "
                "assert star['lot_rule']=='star_or_kcb_min_lot'; "
                "checked=['capital_feasibility.csv','approved_orders.csv','rejected_orders.csv','capital_constraint_summary.csv']; "
                "bad=[name for name in checked if 'trading_ready' in pd.read_csv(base/name).columns and pd.read_csv(base/name)['trading_ready'].fillna(True).astype(bool).any()]; "
                "assert not bad, bad; "
                "guardrails=pd.read_csv(base/'capital_constraint_guardrails.csv'); "
                "required_guardrails={'no_model_retraining','no_threshold_change','no_feature_change','no_new_data_sources','no_broker_integration','no_live_trading','no_trading_ready_upgrade'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "report=(base/'capital_constraint_report.md').read_text(encoding='utf-8'); "
                "phrases=['This is educational/research tooling only','No broker execution is performed','The project remains not trading-ready','V5 Step 1 checks candidate buy-order capital feasibility only']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_tradable_universe_filter help",
        ["src/run_tradable_universe_filter.py", "--help"],
    ),
    (
        "tradable universe filter import",
        ["-c", "import src.tradable_universe_filter"],
    ),
    (
        "offline tradable universe filter",
        [
            "src/run_tradable_universe_filter.py",
            "--cash",
            "1000",
            "--output-dir",
            "outputs/tradable_universe_filter_smoke",
        ],
    ),
    (
        "offline tradable universe filter assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/tradable_universe_filter_smoke'); "
                "required=['tradable_universe.csv','excluded_universe.csv','universe_filter_summary.csv','universe_filter_guardrails.csv','universe_filter_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "tradable=pd.read_csv(base/'tradable_universe.csv', dtype={'symbol': str}); "
                "excluded=pd.read_csv(base/'excluded_universe.csv', dtype={'symbol': str}); "
                "summary=pd.read_csv(base/'universe_filter_summary.csv'); "
                "assert '600000' in set(tradable['symbol']); "
                "assert '688001' in set(tradable['symbol']); "
                "star=tradable[tradable['symbol']=='688001'].iloc[0]; "
                "assert int(star['lot_size']) == 200; "
                "assert star['lot_rule']=='star_or_kcb_min_lot'; "
                "assert float(star['minimum_required_cash']) == 800.0; "
                "main=tradable[tradable['symbol']=='600000'].iloc[0]; "
                "assert float(main['minimum_required_cash']) == 800.0; "
                "reasons=dict(zip(excluded['symbol'], excluded['exclusion_reason'])); "
                "row_600519=excluded[excluded['symbol']=='600519'].iloc[0]; "
                "assert float(row_600519['minimum_required_cash']) == 170000.0; "
                "assert 'insufficient_cash_for_min_lot' in reasons['600519']; "
                "assert 'st_or_star_st_flag' in reasons['600001']; "
                "assert 'suspended' in reasons['600002']; "
                "assert 'invalid_or_missing_price' in reasons['000001']; "
                "assert 'liquidity_below_min_turnover' in reasons['600003']; "
                "assert not summary['trading_ready'].fillna(True).astype(bool).any(); "
                "checked=['tradable_universe.csv','excluded_universe.csv']; "
                "bad=[name for name in checked if 'trading_ready' in pd.read_csv(base/name).columns and pd.read_csv(base/name)['trading_ready'].fillna(True).astype(bool).any()]; "
                "assert not bad, bad; "
                "guardrails=pd.read_csv(base/'universe_filter_guardrails.csv'); "
                "required_guardrails={'no_new_backtests','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_integration','no_live_trading','no_trading_ready_upgrade','universe_filter_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "report=(base/'universe_filter_report.md').read_text(encoding='utf-8'); "
                "phrases=['V5 Step 2 filters candidate universe eligibility','No broker execution is performed','The project remains not trading-ready','This is educational/research tooling only']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_position_sizing_engine help",
        ["src/run_position_sizing_engine.py", "--help"],
    ),
    (
        "position sizing engine import",
        ["-c", "import src.position_sizing_engine"],
    ),
    (
        "offline position sizing engine",
        [
            "src/run_position_sizing_engine.py",
            "--input-path",
            "outputs/tradable_universe_filter_smoke/tradable_universe.csv",
            "--cash",
            "1000",
            "--output-dir",
            "outputs/position_sizing_engine_smoke",
        ],
    ),
    (
        "offline position sizing engine assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/position_sizing_engine_smoke'); "
                "required=['position_sizing_summary.csv','sized_positions.csv','deferred_positions.csv','rejected_positions.csv','position_sizing_guardrails.csv','position_sizing_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'position_sizing_summary.csv'); "
                "sized=pd.read_csv(base/'sized_positions.csv', dtype={'symbol': str}); "
                "deferred=pd.read_csv(base/'deferred_positions.csv', dtype={'symbol': str}); "
                "rejected=pd.read_csv(base/'rejected_positions.csv', dtype={'symbol': str}); "
                "row=summary.iloc[0]; "
                "assert int(row['candidate_count']) == 2; "
                "assert int(row['sized_position_count']) == 1; "
                "assert int(row['deferred_position_count']) == 1; "
                "assert int(row['rejected_position_count']) == 0; "
                "assert float(row['total_approved_notional']) == 800.0; "
                "assert not bool(row['trading_ready']); "
                "assert len(sized) == 1 and len(deferred) == 1 and rejected.empty; "
                "assert float(sized.iloc[0]['approved_notional']) == 800.0; "
                "assert int(sized.iloc[0]['quantity']) in {100, 200}; "
                "assert deferred.iloc[0]['sizing_reason'] == 'account_cash_exhausted_or_allocation_limit'; "
                "checked=['position_sizing_summary.csv','sized_positions.csv','deferred_positions.csv','rejected_positions.csv']; "
                "bad=[name for name in checked if 'trading_ready' in pd.read_csv(base/name).columns and pd.read_csv(base/name)['trading_ready'].fillna(True).astype(bool).any()]; "
                "assert not bad, bad; "
                "guardrails=pd.read_csv(base/'position_sizing_guardrails.csv'); "
                "required_guardrails={'no_new_backtests','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_integration','no_live_trading','no_trading_ready_upgrade','position_sizing_only','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "report=(base/'position_sizing_report.md').read_text(encoding='utf-8'); "
                "phrases=['V5 Step 3 converts Step 2 tradable candidates','No broker execution is performed','The project remains not trading-ready','This is educational/research tooling only']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_exit_engine help",
        ["src/run_exit_engine.py", "--help"],
    ),
    (
        "exit engine import",
        ["-c", "import src.exit_engine"],
    ),
    (
        "offline exit engine",
        [
            "src/run_exit_engine.py",
            "--input-path",
            "outputs/position_sizing_engine_smoke/sized_positions.csv",
            "--output-dir",
            "outputs/exit_engine_smoke",
        ],
    ),
    (
        "offline exit engine assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/exit_engine_smoke'); "
                "required=['exit_plan.csv','exit_guardrails.csv','exit_summary.csv','exit_engine_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'exit_summary.csv'); "
                "plan=pd.read_csv(base/'exit_plan.csv', dtype={'symbol': str}); "
                "row=summary.iloc[0]; "
                "assert int(row['sized_position_count']) == 1; "
                "assert int(row['planned_exit_count']) == 1; "
                "assert int(row['invalid_exit_plan_count']) == 0; "
                "assert not bool(row['trading_ready']); "
                "assert len(plan) == 1; "
                "planned=plan.iloc[0]; "
                "assert planned['symbol'] == '600000'; "
                "assert float(planned['entry_price']) == 8.0; "
                "assert float(planned['stop_loss_pct']) == 0.05; "
                "assert float(planned['stop_loss_price']) == 7.6; "
                "assert float(planned['take_profit_pct']) == 0.10; "
                "assert float(planned['take_profit_price']) == 8.8; "
                "assert int(planned['max_holding_days']) == 10; "
                "assert planned['benchmark_lag_exit_rule'] == 'exit_if_underperform_benchmark_by_3pct_after_5_days'; "
                "assert planned['exit_plan_status'] == 'planned'; "
                "assert not bool(planned['trading_ready']); "
                "guardrails=pd.read_csv(base/'exit_guardrails.csv'); "
                "required_guardrails={'no_new_backtests','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_integration','no_live_trading','no_order_execution','no_trading_ready_upgrade','exit_planning_only','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "report=(base/'exit_engine_report.md').read_text(encoding='utf-8'); "
                "phrases=['V5 Step 4 creates explicit research-only exit plans','No broker execution is performed','No live trading is performed','The project remains not trading-ready']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_daily_trading_plan help",
        ["src/run_daily_trading_plan.py", "--help"],
    ),
    (
        "daily trading plan import",
        ["-c", "import src.daily_trading_plan"],
    ),
    (
        "offline daily trading plan",
        [
            "src/run_daily_trading_plan.py",
            "--tradable-path",
            "outputs/tradable_universe_filter_smoke/tradable_universe.csv",
            "--sized-path",
            "outputs/position_sizing_engine_smoke/sized_positions.csv",
            "--deferred-path",
            "outputs/position_sizing_engine_smoke/deferred_positions.csv",
            "--exit-plan-path",
            "outputs/exit_engine_smoke/exit_plan.csv",
            "--output-dir",
            "outputs/daily_trading_plan_smoke",
        ],
    ),
    (
        "offline daily trading plan assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/daily_trading_plan_smoke'); "
                "required=['daily_trading_plan.md','daily_trading_plan.csv','daily_trading_plan_summary.csv','daily_trading_plan_guardrails.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'daily_trading_plan_summary.csv'); "
                "plan=pd.read_csv(base/'daily_trading_plan.csv', dtype={'symbol': str}); "
                "row=summary.iloc[0]; "
                "assert int(row['tradable_candidate_count']) == 2; "
                "assert int(row['sized_position_count']) == 1; "
                "assert int(row['deferred_position_count']) == 1; "
                "assert int(row['exit_plan_count']) == 1; "
                "assert int(row['daily_plan_row_count']) == 5; "
                "assert not bool(row['trading_ready']); "
                "assert {'tradable_candidate','sized_position','deferred_position','exit_plan'} <= set(plan['plan_section']); "
                "assert not plan['trading_ready'].fillna(True).astype(bool).any(); "
                "exit_rows=plan[plan['plan_section']=='exit_plan']; "
                "assert len(exit_rows) == 1; "
                "assert float(exit_rows.iloc[0]['stop_loss_price']) == 7.6; "
                "assert float(exit_rows.iloc[0]['take_profit_price']) == 8.8; "
                "guardrails=pd.read_csv(base/'daily_trading_plan_guardrails.csv'); "
                "required_guardrails={'no_new_backtests','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_integration','no_live_trading','no_order_execution','no_trading_ready_upgrade','daily_plan_only','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "report=(base/'daily_trading_plan.md').read_text(encoding='utf-8'); "
                "phrases=['Capital Summary','Tradable Candidates','Sized Positions','Deferred Positions','Exit Plan','Research only','Not financial advice','No broker execution','No live trading','trading_ready=False']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_paper_trading_ledger help",
        ["src/run_paper_trading_ledger.py", "--help"],
    ),
    (
        "paper trading ledger import",
        ["-c", "import src.paper_trading_ledger"],
    ),
    (
        "offline paper trading ledger",
        [
            "src/run_paper_trading_ledger.py",
            "--input-dir",
            "outputs/daily_trading_plan_smoke",
            "--output-dir",
            "outputs/paper_trading_ledger_smoke",
        ],
    ),
    (
        "offline paper trading ledger assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/paper_trading_ledger_smoke'); "
                "required=['paper_orders.csv','paper_fills.csv','paper_positions.csv','paper_cash_ledger.csv','paper_trade_ledger.csv','paper_trading_summary.csv','paper_trading_guardrails.csv','paper_trading_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'paper_trading_summary.csv'); "
                "orders=pd.read_csv(base/'paper_orders.csv', dtype={'symbol': str}); "
                "fills=pd.read_csv(base/'paper_fills.csv', dtype={'symbol': str}); "
                "positions=pd.read_csv(base/'paper_positions.csv', dtype={'symbol': str}); "
                "cash=pd.read_csv(base/'paper_cash_ledger.csv'); "
                "ledger=pd.read_csv(base/'paper_trade_ledger.csv', dtype={'symbol': str}); "
                "row=summary.iloc[0]; "
                "assert int(row['paper_order_count']) == 2; "
                "assert int(row['paper_filled_order_count']) == 1; "
                "assert int(row['paper_deferred_order_count']) == 1; "
                "assert int(row['open_paper_position_count']) == 1; "
                "assert float(row['starting_cash']) == 1000.0; "
                "assert float(row['ending_cash']) == 200.0; "
                "assert float(row['total_filled_notional']) == 800.0; "
                "assert row['conclusion'] == 'research_only_paper_ledger_created'; "
                "assert not bool(row['trading_ready']); "
                "assert set(orders['order_status']) == {'paper_filled','deferred_not_filled'}; "
                "filled=fills[fills['fill_status']=='simulated_filled'].iloc[0]; "
                "assert filled['symbol'] == '600000'; "
                "assert int(filled['fill_quantity']) == 100; "
                "assert float(filled['fill_notional']) == 800.0; "
                "assert not fills['trading_ready'].fillna(True).astype(bool).any(); "
                "pos=positions.iloc[0]; "
                "assert pos['position_status'] == 'open_paper_position'; "
                "assert float(pos['stop_loss_price']) == 7.6; "
                "assert float(pos['take_profit_price']) == 8.8; "
                "assert float(pos['unrealized_pnl']) == 0.0; "
                "assert not positions['trading_ready'].fillna(True).astype(bool).any(); "
                "assert float(cash.iloc[-1]['ending_cash']) == 200.0; "
                "assert not ledger['trading_ready'].fillna(True).astype(bool).any(); "
                "guardrails=pd.read_csv(base/'paper_trading_guardrails.csv'); "
                "required_guardrails={'no_new_backtests','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_integration','no_live_trading','no_order_execution','no_trading_ready_upgrade','paper_ledger_only','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "report=(base/'paper_trading_report.md').read_text(encoding='utf-8'); "
                "phrases=['paper trading only','No broker execution occurred','No real orders were submitted','No live market data was fetched','No strategy performance claim is made','All outputs remain trading_ready=False','V5 Step 7 Semi-Auto Order Generator']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_semi_auto_order_generator help",
        ["src/run_semi_auto_order_generator.py", "--help"],
    ),
    (
        "semi auto order generator import",
        ["-c", "import src.semi_auto_order_generator"],
    ),
    (
        "offline semi auto order generator",
        [
            "src/run_semi_auto_order_generator.py",
            "--daily-plan-path",
            "outputs/daily_trading_plan_smoke/daily_trading_plan.csv",
            "--exit-plan-path",
            "outputs/exit_engine_smoke/exit_plan.csv",
            "--output-dir",
            "outputs/semi_auto_order_generator_smoke",
        ],
    ),
    (
        "offline semi auto order generator assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/semi_auto_order_generator_smoke'); "
                "required=['order_drafts.csv','broker_neutral_order_tickets.md','manual_review_checklist.csv','semi_auto_order_summary.csv','semi_auto_order_guardrails.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "drafts=pd.read_csv(base/'order_drafts.csv', dtype={'symbol': str}); "
                "checklist=pd.read_csv(base/'manual_review_checklist.csv', dtype={'symbol': str}); "
                "summary=pd.read_csv(base/'semi_auto_order_summary.csv'); "
                "guardrails=pd.read_csv(base/'semi_auto_order_guardrails.csv'); "
                "row=summary.iloc[0]; "
                "assert int(row['draft_order_count']) == 1; "
                "assert int(row['buy_draft_count']) == 1; "
                "assert int(row['sell_draft_count']) == 0; "
                "assert int(row['execution_allowed_count']) == 0; "
                "assert int(row['broker_connected_count']) == 0; "
                "assert int(row['trading_ready_count']) == 0; "
                "assert int(row['human_review_required_count']) == 1; "
                "assert not bool(row['trading_ready']); "
                "draft=drafts.iloc[0]; "
                "assert draft['draft_order_id'] == 'DRAFT-BUY-001'; "
                "assert draft['source_plan_section'] == 'sized_position'; "
                "assert draft['symbol'] == '600000'; "
                "assert draft['side'] == 'BUY'; "
                "assert int(draft['quantity']) == 100; "
                "assert float(draft['limit_price']) == 8.0; "
                "assert float(draft['estimated_notional']) == 800.0; "
                "assert float(draft['stop_loss_price']) == 7.6; "
                "assert float(draft['take_profit_price']) == 8.8; "
                "assert int(draft['max_holding_days']) == 10; "
                "assert bool(draft['human_review_required']); "
                "assert not bool(draft['execution_allowed']); "
                "assert not bool(draft['broker_connected']); "
                "assert not bool(draft['trading_ready']); "
                "assert draft['draft_status'] == 'draft_only'; "
                "required_checks={'confirm_symbol','confirm_side','confirm_quantity','confirm_limit_price','confirm_cash_available','confirm_stop_loss','confirm_take_profit','confirm_no_broker_execution','confirm_human_review_required'}; "
                "assert required_checks <= set(checklist['check_name']); "
                "assert not checklist['execution_allowed'].fillna(True).astype(bool).any(); "
                "assert not checklist['broker_connected'].fillna(True).astype(bool).any(); "
                "assert not checklist['trading_ready'].fillna(True).astype(bool).any(); "
                "required_guardrails={'no_new_backtests','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_integration','no_live_trading','no_order_execution','no_trading_ready_upgrade','semi_auto_order_draft_only','human_review_required','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "tickets=(base/'broker_neutral_order_tickets.md').read_text(encoding='utf-8'); "
                "phrases=['Draft Order: DRAFT-BUY-001','Symbol: 600000','Side: BUY','Quantity: 100','Limit Price: 8.0','Estimated Notional: 800.0','Stop Loss: 7.6','Take Profit: 8.8','Max Holding Days: 10','Human Review Required: True','Execution Allowed: False','Broker Connected: False','Trading Ready: False']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in tickets]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_broker_integration_research help",
        ["src/run_broker_integration_research.py", "--help"],
    ),
    (
        "broker integration research import",
        ["-c", "import src.broker_integration_research"],
    ),
    (
        "offline broker integration research",
        [
            "src/run_broker_integration_research.py",
            "--input-dir",
            "outputs/semi_auto_order_generator_smoke",
            "--output-dir",
            "outputs/broker_integration_research_smoke",
        ],
    ),
    (
        "offline broker integration research assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/broker_integration_research_smoke'); "
                "required=['broker_integration_summary.csv','broker_integration_modes.csv','broker_integration_constraints.csv','broker_integration_risk_register.csv','broker_integration_guardrails.csv','broker_integration_research_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'broker_integration_summary.csv'); "
                "modes=pd.read_csv(base/'broker_integration_modes.csv'); "
                "constraints=pd.read_csv(base/'broker_integration_constraints.csv'); "
                "risks=pd.read_csv(base/'broker_integration_risk_register.csv'); "
                "guardrails=pd.read_csv(base/'broker_integration_guardrails.csv'); "
                "row=summary.iloc[0]; "
                "assert int(row['input_draft_order_count']) == 1; "
                "assert int(row['researched_mode_count']) == 5; "
                "assert int(row['constraint_count']) >= 15; "
                "assert int(row['high_risk_constraint_count']) >= 8; "
                "assert int(row['broker_connected_count']) == 0; "
                "assert int(row['execution_allowed_count']) == 0; "
                "assert int(row['trading_ready_count']) == 0; "
                "assert int(row['live_trading_count']) == 0; "
                "assert int(row['real_order_submission_count']) == 0; "
                "assert row['conclusion'] == 'broker_integration_research_only_no_execution'; "
                "flag_cols=['broker_connected','execution_allowed','trading_ready','live_trading','real_order_submission']; "
                "frames={'summary':summary,'modes':modes,'constraints':constraints,'risks':risks}; "
                "bad_flags=[(name,col) for name,frame in frames.items() for col in flag_cols if col in frame.columns and frame[col].fillna(True).astype(bool).any()]; "
                "assert not bad_flags, bad_flags; "
                "required_modes={'manual_review_only','broker_neutral_ticket_export','paper_trading_only','broker_api_research_only','future_human_approved_broker_bridge'}; "
                "assert required_modes <= set(modes['integration_mode']); "
                "required_constraints={'account_login_credential_risk','broker_api_availability','region_market_restrictions','two_factor_authentication','trading_permissions','order_type_support','lot_size_compatibility','minimum_cash_capital_constraints','rate_limits','market_hours','quote_latency','failure_handling','audit_logging','manual_confirmation','legal_compliance_limitations'}; "
                "assert required_constraints <= set(constraints['constraint']); "
                "assert len(risks) == len(constraints); "
                "required_guardrails={'no_new_backtests','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_credentials','no_broker_sdk_import','no_broker_connection','no_live_trading','no_order_execution','no_trading_ready_upgrade','broker_research_only','human_review_required','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "report=(base/'broker_integration_research_report.md').read_text(encoding='utf-8'); "
                "phrases=['does not connect to any broker','does not request or store credentials','does not place orders','does not simulate real broker routing','All outputs preserve broker_connected=False','trading_ready=False']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_monitoring_reporting_layer help",
        ["src/run_monitoring_reporting_layer.py", "--help"],
    ),
    (
        "monitoring reporting layer import",
        ["-c", "import src.monitoring_reporting_layer"],
    ),
    (
        "offline monitoring reporting layer",
        [
            "src/run_monitoring_reporting_layer.py",
            "--capital-dir",
            "outputs/capital_constraint_engine_smoke",
            "--universe-dir",
            "outputs/tradable_universe_filter_smoke",
            "--position-dir",
            "outputs/position_sizing_engine_smoke",
            "--exit-dir",
            "outputs/exit_engine_smoke",
            "--daily-plan-dir",
            "outputs/daily_trading_plan_smoke",
            "--paper-ledger-dir",
            "outputs/paper_trading_ledger_smoke",
            "--semi-auto-dir",
            "outputs/semi_auto_order_generator_smoke",
            "--broker-research-dir",
            "outputs/broker_integration_research_smoke",
            "--output-dir",
            "outputs/monitoring_reporting_layer_smoke",
        ],
    ),
    (
        "offline monitoring reporting layer assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/monitoring_reporting_layer_smoke'); "
                "required=['monitoring_summary.csv','monitoring_status_dashboard.csv','monitoring_alerts.csv','monitoring_guardrails.csv','monitoring_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'monitoring_summary.csv'); "
                "dashboard=pd.read_csv(base/'monitoring_status_dashboard.csv'); "
                "alerts=pd.read_csv(base/'monitoring_alerts.csv'); "
                "guardrails=pd.read_csv(base/'monitoring_guardrails.csv'); "
                "row=summary.iloc[0]; "
                "assert int(row['monitored_step_count']) == 8; "
                "assert int(row['dashboard_row_count']) >= 30; "
                "assert int(row['blocking_alert_count']) == 0; "
                "assert int(row['warning_alert_count']) >= 2; "
                "assert int(row['trading_ready_true_count']) == 0; "
                "assert int(row['execution_allowed_true_count']) == 0; "
                "assert int(row['broker_connected_true_count']) == 0; "
                "assert int(row['live_trading_true_count']) == 0; "
                "assert int(row['real_order_submission_true_count']) == 0; "
                "assert row['conclusion'] == 'monitoring_reporting_only_no_execution'; "
                "warning_types=set(alerts.loc[alerts['severity']=='warning','alert_type']); "
                "assert {'approved_exceeds_available_cash','approved_exceeds_usable_cash'} <= warning_types; "
                "assert not dashboard[['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']].fillna(True).astype(bool).any().any(); "
                "assert not alerts[['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']].fillna(True).astype(bool).any().any(); "
                "required_guardrails={'no_new_backtests','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_credentials','no_broker_sdk_import','no_broker_connection','no_live_trading','no_order_execution','no_trading_ready_upgrade','monitoring_reporting_only','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "report=(base/'monitoring_report.md').read_text(encoding='utf-8'); "
                "phrases=['Monitoring / Reporting Layer','does not run backtests','fetch market data','trading_ready=False']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_capital_aware_infrastructure_review help",
        ["src/run_capital_aware_infrastructure_review.py", "--help"],
    ),
    (
        "capital aware infrastructure review import",
        ["-c", "import src.capital_aware_infrastructure_review"],
    ),
    (
        "offline capital aware infrastructure review",
        [
            "src/run_capital_aware_infrastructure_review.py",
            "--capital-dir",
            "outputs/capital_constraint_engine_smoke",
            "--universe-dir",
            "outputs/tradable_universe_filter_smoke",
            "--position-dir",
            "outputs/position_sizing_engine_smoke",
            "--exit-dir",
            "outputs/exit_engine_smoke",
            "--daily-plan-dir",
            "outputs/daily_trading_plan_smoke",
            "--paper-ledger-dir",
            "outputs/paper_trading_ledger_smoke",
            "--semi-auto-dir",
            "outputs/semi_auto_order_generator_smoke",
            "--broker-research-dir",
            "outputs/broker_integration_research_smoke",
            "--monitoring-dir",
            "outputs/monitoring_reporting_layer_smoke",
            "--output-dir",
            "outputs/capital_aware_infrastructure_review_smoke",
        ],
    ),
    (
        "offline capital aware infrastructure review assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/capital_aware_infrastructure_review_smoke'); "
                "required=['v5_infrastructure_closure_summary.csv','v5_step_capability_matrix.csv','v5_guardrail_audit.csv','v5_limitations_register.csv','v5_readiness_blockers.csv','v5_next_phase_recommendations.csv','v5_capital_aware_closure_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'v5_infrastructure_closure_summary.csv'); "
                "matrix=pd.read_csv(base/'v5_step_capability_matrix.csv'); "
                "guardrails=pd.read_csv(base/'v5_guardrail_audit.csv'); "
                "limitations=pd.read_csv(base/'v5_limitations_register.csv'); "
                "blockers=pd.read_csv(base/'v5_readiness_blockers.csv'); "
                "recs=pd.read_csv(base/'v5_next_phase_recommendations.csv'); "
                "row=summary.iloc[0]; "
                "assert int(row['reviewed_step_count']) == 9; "
                "assert int(row['completed_step_count']) == 9; "
                "assert int(row['missing_step_count']) == 0; "
                "assert int(row['trading_ready_true_count']) == 0; "
                "assert int(row['execution_allowed_true_count']) == 0; "
                "assert int(row['broker_connected_true_count']) == 0; "
                "assert int(row['live_trading_true_count']) == 0; "
                "assert int(row['real_order_submission_true_count']) == 0; "
                "assert not bool(row['trading_ready']); "
                "assert row['final_v5_status'] == 'capital_aware_infrastructure_closed_research_only'; "
                "assert str(row['recommended_next_phase']).startswith('V6'); "
                "assert len(matrix) == 9; "
                "required_capabilities={'capital feasibility','tradable universe filtering','position sizing','exit planning','daily plan generation','paper ledger','semi-auto draft order generation','broker integration research','monitoring/reporting'}; "
                "assert required_capabilities <= set(matrix['capability_added']); "
                "flag_cols=['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']; "
                "frames={'summary':summary,'matrix':matrix,'guardrails':guardrails,'blockers':blockers}; "
                "bad_flags=[(name,col) for name,frame in frames.items() for col in flag_cols if col in frame.columns and frame[col].fillna(True).astype(bool).any()]; "
                "assert not bad_flags, bad_flags; "
                "required_limitations={'no_validated_profitable_strategy','bull_remediation_unresolved_from_v4','no_live_data_pipeline','no_broker_execution','no_automated_order_routing','no_slippage_commission_tax_realistic_execution_model','no_portfolio_optimizer','no_risk_adjusted_production_validation','no_paper_trading_over_real_time','no_monitoring_daemon','no_autonomous_self_research_agent','no_trading_ready_certification'}; "
                "assert required_limitations <= set(limitations['limitation']); "
                "required_blockers={'no_profitable_validated_candidate','no_robust_out_of_sample_live_or_paper_evidence','no_broker_sandbox_or_live_integration','no_compliance_risk_approval_layer','no_production_monitoring','no_kill_switch','no_real_time_capital_account_reconciliation'}; "
                "assert required_blockers <= set(blockers['blocker']); "
                "required_recs={'V6 Step 1','V6 Step 2','V6 Step 3','V6 Step 4','V6 Step 5'}; "
                "assert required_recs <= set(recs['phase_step']); "
                "required_guardrails={'no_new_backtests','no_market_data_fetch','no_model_retraining','no_broker_connection','no_live_trading','no_real_order_submission','no_order_execution','no_trading_ready_upgrade','review_only_closure'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "confirmed_zero={'trading_ready_true_count_zero','execution_allowed_true_count_zero','broker_connected_true_count_zero','live_trading_true_count_zero','real_order_submission_true_count_zero'}; "
                "assert confirmed_zero <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(confirmed_zero),'status']) == {'confirmed'}; "
                "report=(base/'v5_capital_aware_closure_report.md').read_text(encoding='utf-8'); "
                "phrases=['Capital-Aware Infrastructure Review / Closure','review-only closure layer','does not add trading capability','capital_aware_infrastructure_closed_research_only','V6 validation_and_simulation_hardening']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_validation_baseline_manifest help",
        ["src/run_validation_baseline_manifest.py", "--help"],
    ),
    (
        "validation baseline manifest import",
        ["-c", "import src.validation_baseline_manifest"],
    ),
    (
        "offline validation baseline manifest",
        [
            "src/run_validation_baseline_manifest.py",
            "--capital-dir",
            "outputs/capital_constraint_engine_smoke",
            "--universe-dir",
            "outputs/tradable_universe_filter_smoke",
            "--position-dir",
            "outputs/position_sizing_engine_smoke",
            "--exit-dir",
            "outputs/exit_engine_smoke",
            "--daily-plan-dir",
            "outputs/daily_trading_plan_smoke",
            "--paper-ledger-dir",
            "outputs/paper_trading_ledger_smoke",
            "--semi-auto-dir",
            "outputs/semi_auto_order_generator_smoke",
            "--broker-research-dir",
            "outputs/broker_integration_research_smoke",
            "--monitoring-dir",
            "outputs/monitoring_reporting_layer_smoke",
            "--v5-closure-dir",
            "outputs/capital_aware_infrastructure_review_smoke",
            "--output-dir",
            "outputs/validation_baseline_manifest_smoke",
        ],
    ),
    (
        "offline validation baseline manifest assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/validation_baseline_manifest_smoke'); "
                "required=['validation_baseline_summary.csv','validation_baseline_manifest.csv','validation_baseline_guardrails.csv','validation_baseline_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'validation_baseline_summary.csv'); "
                "manifest=pd.read_csv(base/'validation_baseline_manifest.csv'); "
                "guardrails=pd.read_csv(base/'validation_baseline_guardrails.csv'); "
                "row=summary.iloc[0]; "
                "assert int(row['baseline_step_count']) == 10; "
                "assert int(row['present_output_dir_count']) == 10; "
                "assert int(row['missing_output_dir_count']) == 0; "
                "assert int(row['total_file_count']) >= 50; "
                "assert int(row['total_csv_file_count']) >= 35; "
                "assert int(row['total_markdown_file_count']) >= 9; "
                "assert int(row['total_json_file_count']) >= 10; "
                "assert int(row['trading_ready_true_count']) == 0; "
                "assert int(row['execution_allowed_true_count']) == 0; "
                "assert int(row['broker_connected_true_count']) == 0; "
                "assert int(row['live_trading_true_count']) == 0; "
                "assert int(row['real_order_submission_true_count']) == 0; "
                "assert row['baseline_status'] == 'v6_validation_baseline_manifest_created_research_only'; "
                "assert not bool(row['trading_ready']); "
                "assert len(manifest) == 10; "
                "assert manifest['directory_exists'].fillna(False).astype(bool).all(); "
                "required_steps={'V5 Step 1 Capital Constraint Engine','V5 Step 2 Tradable Universe Filter','V5 Step 3 Position Sizing Engine','V5 Step 4 Exit Engine','V5 Step 5 Daily Trading Plan','V5 Step 6 Paper Trading Ledger','V5 Step 7 Semi-Auto Order Generator','V5 Step 8 Broker Integration Research','V5 Step 9 Monitoring / Reporting Layer','V5 Step 10 Capital-Aware Infrastructure Review / Closure'}; "
                "assert required_steps <= set(manifest['step_name']); "
                "flag_cols=['trading_ready_true_count','execution_allowed_true_count','broker_connected_true_count','live_trading_true_count','real_order_submission_true_count']; "
                "assert not manifest[flag_cols].fillna(0).astype(int).any().any(); "
                "required_guardrails={'no_new_backtests','no_market_data_fetch','no_threshold_change','no_model_retraining','no_feature_engineering_change','no_new_data_sources','no_broker_credentials','no_broker_sdk_import','no_broker_connection','no_live_trading','no_order_execution','no_real_order_submission','no_trading_ready_upgrade','manifest_only','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "assert not guardrails[['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']].fillna(True).astype(bool).any().any(); "
                "report=(base/'validation_baseline_report.md').read_text(encoding='utf-8'); "
                "phrases=['Validation Baseline Manifest','stable research baseline','does not create any trading capability','v6_validation_baseline_manifest_created_research_only']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_output_schema_validator help",
        ["src/run_output_schema_validator.py", "--help"],
    ),
    (
        "output schema validator import",
        ["-c", "import src.output_schema_validator"],
    ),
    (
        "offline output schema validator",
        [
            "src/run_output_schema_validator.py",
            "--capital-dir",
            "outputs/capital_constraint_engine_smoke",
            "--universe-dir",
            "outputs/tradable_universe_filter_smoke",
            "--position-dir",
            "outputs/position_sizing_engine_smoke",
            "--exit-dir",
            "outputs/exit_engine_smoke",
            "--daily-plan-dir",
            "outputs/daily_trading_plan_smoke",
            "--paper-ledger-dir",
            "outputs/paper_trading_ledger_smoke",
            "--semi-auto-dir",
            "outputs/semi_auto_order_generator_smoke",
            "--broker-research-dir",
            "outputs/broker_integration_research_smoke",
            "--monitoring-dir",
            "outputs/monitoring_reporting_layer_smoke",
            "--v5-closure-dir",
            "outputs/capital_aware_infrastructure_review_smoke",
            "--baseline-dir",
            "outputs/validation_baseline_manifest_smoke",
            "--output-dir",
            "outputs/output_schema_validator_smoke",
        ],
    ),
    (
        "offline output schema validator assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/output_schema_validator_smoke'); "
                "required=['run_config.json','output_schema_validation_summary.csv','output_schema_validation_results.csv','output_schema_validation_guardrails.csv','output_schema_validation_report.md']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'output_schema_validation_summary.csv'); "
                "results=pd.read_csv(base/'output_schema_validation_results.csv'); "
                "guardrails=pd.read_csv(base/'output_schema_validation_guardrails.csv'); "
                "row=summary.iloc[0]; "
                "assert int(row['checked_directory_count']) == 11; "
                "assert int(row['checked_file_count']) >= 30; "
                "assert int(row['present_file_count']) == int(row['checked_file_count']); "
                "assert int(row['missing_file_count']) == 0; "
                "assert int(row['schema_fail_count']) == 0; "
                "assert int(row['forbidden_true_flag_count']) == 0; "
                "assert not bool(row['trading_ready']); "
                "assert row['validation_status'] in {'pass','warning'}; "
                "assert str(row['conclusion']).endswith('_research_only'); "
                "assert len(results) == int(row['checked_file_count']); "
                "assert not results[['trading_ready_true_count','execution_allowed_true_count','broker_connected_true_count','live_trading_true_count','real_order_submission_true_count']].fillna(0).astype(int).any().any(); "
                "assert not results[['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']].fillna(True).astype(bool).any().any(); "
                "required_guardrails={'no_new_backtests','no_market_data_fetch','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_credentials','no_broker_sdk_import','no_broker_connection','no_live_trading','no_order_execution','no_real_order_submission','no_trading_ready_upgrade','schema_validation_only','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "assert not guardrails[['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']].fillna(True).astype(bool).any().any(); "
                "report=(base/'output_schema_validation_report.md').read_text(encoding='utf-8'); "
                "phrases=['Output Consistency / Schema Validation Layer','required files and required columns','does not run backtests','schema_validation']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_cross_step_dependency_validator help",
        ["src/run_cross_step_dependency_validator.py", "--help"],
    ),
    (
        "cross step dependency validator import",
        ["-c", "import src.cross_step_dependency_validator"],
    ),
    (
        "offline cross step dependency validator",
        [
            "src/run_cross_step_dependency_validator.py",
            "--capital-dir",
            "outputs/capital_constraint_engine_smoke",
            "--universe-dir",
            "outputs/tradable_universe_filter_smoke",
            "--position-dir",
            "outputs/position_sizing_engine_smoke",
            "--exit-dir",
            "outputs/exit_engine_smoke",
            "--daily-plan-dir",
            "outputs/daily_trading_plan_smoke",
            "--paper-ledger-dir",
            "outputs/paper_trading_ledger_smoke",
            "--semi-auto-dir",
            "outputs/semi_auto_order_generator_smoke",
            "--broker-research-dir",
            "outputs/broker_integration_research_smoke",
            "--monitoring-dir",
            "outputs/monitoring_reporting_layer_smoke",
            "--v5-closure-dir",
            "outputs/capital_aware_infrastructure_review_smoke",
            "--baseline-dir",
            "outputs/validation_baseline_manifest_smoke",
            "--schema-validator-dir",
            "outputs/output_schema_validator_smoke",
            "--output-dir",
            "outputs/cross_step_dependency_validator_smoke",
        ],
    ),
    (
        "offline cross step dependency validator assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/cross_step_dependency_validator_smoke'); "
                "required=['cross_step_dependency_results.csv','cross_step_dependency_summary.csv','cross_step_dependency_guardrails.csv','cross_step_dependency_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'cross_step_dependency_summary.csv'); "
                "results=pd.read_csv(base/'cross_step_dependency_results.csv'); "
                "guardrails=pd.read_csv(base/'cross_step_dependency_guardrails.csv'); "
                "row=summary.iloc[0]; "
                "assert int(row['checked_dependency_count']) >= 40; "
                "assert int(row['dependency_fail_count']) == 0; "
                "assert int(row['checked_output_dir_count']) == 12; "
                "assert int(row['missing_output_dir_count']) == 0; "
                "assert int(row['forbidden_true_flag_count']) == 0; "
                "assert not bool(row['trading_ready']); "
                "assert row['validation_status'] in {'pass','warning'}; "
                "assert str(row['conclusion']).endswith('_research_only'); "
                "assert len(results) == int(row['checked_dependency_count']); "
                "assert not results[['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']].fillna(True).astype(bool).any().any(); "
                "required_ids={'DEP-001','DEP-002','DEP-006','DEP-007','DEP-010','DEP-020','DEP-029','DEP-039','DEP-050'}; "
                "assert required_ids <= set(results['dependency_id']); "
                "required_guardrails={'no_new_backtests','no_market_data_fetch','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_connection','no_live_trading','no_order_execution','no_real_order_submission','no_trading_ready_upgrade','dependency_validation_only','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "assert not guardrails[['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']].fillna(True).astype(bool).any().any(); "
                "report=(base/'cross_step_dependency_report.md').read_text(encoding='utf-8'); "
                "phrases=['Cross-Step Dependency Integrity Validator','dependency links','does not run backtests','cross_step_dependency_validation']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "run_reproducibility_rerun_validator help",
        ["src/run_reproducibility_rerun_validator.py", "--help"],
    ),
    (
        "reproducibility rerun validator import",
        ["-c", "import src.reproducibility_rerun_validator"],
    ),
    (
        "offline reproducibility rerun validator",
        [
            "src/run_reproducibility_rerun_validator.py",
            "--semi-auto-dir",
            "outputs/semi_auto_order_generator_smoke",
            "--broker-research-dir",
            "outputs/broker_integration_research_smoke",
            "--capital-dir",
            "outputs/capital_constraint_engine_smoke",
            "--universe-dir",
            "outputs/tradable_universe_filter_smoke",
            "--position-dir",
            "outputs/position_sizing_engine_smoke",
            "--exit-dir",
            "outputs/exit_engine_smoke",
            "--daily-plan-dir",
            "outputs/daily_trading_plan_smoke",
            "--paper-ledger-dir",
            "outputs/paper_trading_ledger_smoke",
            "--monitoring-dir",
            "outputs/monitoring_reporting_layer_smoke",
            "--v5-closure-dir",
            "outputs/capital_aware_infrastructure_review_smoke",
            "--baseline-dir",
            "outputs/validation_baseline_manifest_smoke",
            "--schema-validator-dir",
            "outputs/output_schema_validator_smoke",
            "--dependency-validator-dir",
            "outputs/cross_step_dependency_validator_smoke",
            "--output-dir",
            "outputs/reproducibility_rerun_validator_smoke",
        ],
    ),
    (
        "offline reproducibility rerun validator assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/reproducibility_rerun_validator_smoke'); "
                "required=['run_config.json','reproducibility_rerun_summary.csv','reproducibility_rerun_results.csv','reproducibility_rerun_guardrails.csv','reproducibility_rerun_report.md']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "assert (base/'rerun_workspace').exists(); "
                "summary=pd.read_csv(base/'reproducibility_rerun_summary.csv'); "
                "results=pd.read_csv(base/'reproducibility_rerun_results.csv'); "
                "guardrails=pd.read_csv(base/'reproducibility_rerun_guardrails.csv'); "
                "row=summary.iloc[0]; "
                "assert int(row['checked_rerun_count']) == 6; "
                "assert int(row['rerun_fail_count']) == 0; "
                "assert int(row['checked_file_count']) >= 30; "
                "assert int(row['matched_file_count']) == int(row['checked_file_count']); "
                "assert int(row['mismatched_file_count']) == 0; "
                "assert int(row['forbidden_true_flag_count']) == 0; "
                "assert not bool(row['trading_ready']); "
                "assert row['validation_status'] in {'pass','warning'}; "
                "assert str(row['conclusion']).endswith('_research_only'); "
                "assert len(results) == int(row['checked_file_count']); "
                "assert not results[['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']].fillna(True).astype(bool).any().any(); "
                "assert not results['forbidden_true_flag_count'].fillna(1).astype(int).any(); "
                "required_guardrails={'no_new_backtests','no_market_data_fetch','no_threshold_change','no_model_retraining','no_feature_change','no_new_data_sources','no_broker_credentials','no_broker_sdk_import','no_broker_connection','no_live_trading','no_order_execution','no_real_order_submission','no_trading_ready_upgrade','isolated_rerun_only','previous_outputs_not_overwritten','educational_research_only'}; "
                "assert required_guardrails <= set(guardrails['guardrail']); "
                "assert set(guardrails.loc[guardrails['guardrail'].isin(required_guardrails),'status']) == {'confirmed'}; "
                "assert not guardrails[['broker_connected','execution_allowed','live_trading','real_order_submission','trading_ready']].fillna(True).astype(bool).any().any(); "
                "report=(base/'reproducibility_rerun_report.md').read_text(encoding='utf-8'); "
                "phrases=['Historical Output Reproducibility','isolated rerun workspace','does not run backtests','reproducibility_rerun']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    ("generate_model_report help", ["src/generate_model_report.py", "--help"]),
    ("show_feature_sources help", ["src/show_feature_sources.py", "--help"]),
    ("show_feature_queue help", ["src/show_feature_queue.py", "--help"]),
    (
        "offline feature registry export",
        [
            "src/show_feature_sources.py",
            "--export",
            "outputs/feature_source_registry_smoke.csv",
        ],
    ),
    (
        "offline feature registry export file",
        [
            "-c",
            (
                "from pathlib import Path; "
                "path=Path('outputs/feature_source_registry_smoke.csv'); "
                "assert path.exists(); "
                "text=path.read_text(encoding='utf-8-sig'); "
                "assert 'factor_family' in text and 'valuation' in text"
            ),
        ],
    ),
    (
        "offline feature queue generation",
        [
            "src/show_feature_queue.py",
            "--max-rows",
            "10",
        ],
    ),
    (
        "offline feature queue export",
        [
            "src/show_feature_queue.py",
            "--output",
            "outputs/feature_implementation_queue_smoke.csv",
        ],
    ),
    (
        "offline feature queue export file",
        [
            "-c",
            (
                "from pathlib import Path; "
                "path=Path('outputs/feature_implementation_queue_smoke.csv'); "
                "assert path.exists(); "
                "text=path.read_text(encoding='utf-8-sig'); "
                "assert 'implementation_score' in text and 'price_action' in text"
            ),
        ],
    ),
    (
        "demo factor dataset build",
        [
            "src/build_factor_dataset.py",
            "--symbol",
            "000001",
            "--source",
            "demo",
            "--output",
            "data/factors/smoke_factors_000001.csv",
        ],
    ),
    (
        "demo factor dataset P0 columns",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "path=Path('data/factors/smoke_factors_000001.csv'); "
                "df=pd.read_csv(path); "
                "required=['intraday_range_pct','candle_body_pct',"
                "'upper_shadow_pct','lower_shadow_pct','volume_ma5',"
                "'volume_ma20','volume_ratio_5d','volume_ratio_20d',"
                "'turnover_proxy','price_position_20d','price_position_60d',"
                "'breakout_20d','breakdown_20d','trend_strength_20d',"
                "'volatility_ratio_5d_20d']; "
                "missing=[column for column in required if column not in df.columns]; "
                "assert not missing, missing"
            ),
        ],
    ),
    (
        "offline factor dataset split",
        [
            "src/split_factor_dataset.py",
            "--input",
            "data/factors/smoke_factors_000001.csv",
            "--output-dir",
            "data/ml/smoke_000001",
            "--target-col",
            "label_up_5d",
            "--purge-rows",
            "5",
            "--split-mode",
            "global_date",
        ],
    ),
    (
        "offline batch factor ablation",
        [
            "src/run_batch_factor_ablation.py",
            "--symbols",
            "000001,600519",
            "--source",
            "demo",
            "--start",
            "20240101",
            "--end",
            "20241231",
            "--output-dir",
            "outputs/factor_ablation_demo",
            "--models",
            "logistic_regression,random_forest",
            "--ablation-modes",
            "drop_group,only_group,drop_feature",
            "--max-drop-features",
            "6",
        ],
    ),
    (
        "offline batch factor ablation output files",
        [
            "-c",
            (
                "from pathlib import Path; "
                "base=Path('outputs/factor_ablation_demo'); "
                "required=['ablation_results.csv','feature_ablation_results.csv',"
                "'group_summary.csv','feature_impact_ranking.csv',"
                "'feature_pruning_recommendations.csv','warnings.csv',"
                "'run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing"
            ),
        ],
    ),
    (
        "offline factor decision report",
        [
            "src/generate_factor_decision_report.py",
            "--input-dir",
            "outputs/factor_ablation_demo",
            "--output",
            "outputs/factor_ablation_demo/factor_decision_report.md",
        ],
    ),
    (
        "offline factor decision report file",
        [
            "-c",
            (
                "from pathlib import Path; "
                "path=Path('outputs/factor_ablation_demo/factor_decision_report.md'); "
                "assert path.exists(); "
                "text=path.read_text(encoding='utf-8'); "
                "assert '# Factor Selection and Retention Decision Report' in text"
            ),
        ],
    ),
    (
        "offline factor pruning experiment",
        [
            "src/run_factor_pruning_experiment.py",
            "--factor-csv",
            "data/factors/smoke_factors_000001.csv",
            "--recommendations",
            "outputs/factor_ablation_demo/feature_pruning_recommendations.csv",
            "--output-dir",
            "outputs/factor_pruning_demo",
            "--models",
            "logistic_regression,random_forest",
            "--target-col",
            "label_up_5d",
        ],
    ),
    (
        "offline factor pruning output files",
        [
            "-c",
            (
                "from pathlib import Path; "
                "base=Path('outputs/factor_pruning_demo'); "
                "required=['pruning_results.csv','pruning_summary.csv',"
                "'feature_set_details.csv','warnings.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing"
            ),
        ],
    ),
    (
        "offline synthetic pruning summary inputs",
        [
            "-c",
            (
                "from pathlib import Path; import json; import pandas as pd; "
                "base=Path('outputs/pruning_summary_smoke_inputs'); "
                "rows=["
                "{'pruning_mode':'full','model_count':2,'avg_feature_count':40,'avg_test_roc_auc':0.50,'avg_test_f1':0.45,'avg_validation_roc_auc':0.51,'avg_delta_test_roc_auc_vs_full':0.0,'avg_delta_test_f1_vs_full':0.0},"
                "{'pruning_mode':'drop_reduce_weight','model_count':2,'avg_feature_count':36,'avg_test_roc_auc':0.53,'avg_test_f1':0.46,'avg_validation_roc_auc':0.52,'avg_delta_test_roc_auc_vs_full':0.03,'avg_delta_test_f1_vs_full':0.01},"
                "{'pruning_mode':'keep_core_and_observe','model_count':2,'avg_feature_count':28,'avg_test_roc_auc':0.54,'avg_test_f1':0.455,'avg_validation_roc_auc':0.53,'avg_delta_test_roc_auc_vs_full':0.04,'avg_delta_test_f1_vs_full':0.005}"
                "]; "
                "dirs=[base/f'factor_pruning_real_{symbol}' for symbol in ['000001','600519']]; "
                "[d.mkdir(parents=True, exist_ok=True) for d in dirs]; "
                "[pd.DataFrame(rows).to_csv(d/'pruning_summary.csv', index=False) for d in dirs]; "
                "[(d/'run_config.json').write_text(json.dumps({'factor_csv':f\"data/factors/factors_{d.name.replace('factor_pruning_real_','')}.csv\"}), encoding='utf-8') for d in dirs]"
            ),
        ],
    ),
    (
        "offline pruning summary report",
        [
            "src/generate_pruning_summary_report.py",
            "--input-dirs",
            "outputs/pruning_summary_smoke_inputs/factor_pruning_real_000001,outputs/pruning_summary_smoke_inputs/factor_pruning_real_600519",
            "--output-dir",
            "outputs/pruning_summary_smoke",
        ],
    ),
    (
        "offline pruning summary output files",
        [
            "-c",
            (
                "from pathlib import Path; "
                "base=Path('outputs/pruning_summary_smoke'); "
                "required=['combined_pruning_results.csv','pruning_mode_summary.csv',"
                "'per_symbol_best_modes.csv','pruning_summary_report.md',"
                "'warnings.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing"
            ),
        ],
    ),
    (
        "offline reduced feature backtest",
        [
            "src/run_reduced_feature_backtest.py",
            "--factor-csv",
            "data/factors/smoke_factors_000001.csv",
            "--recommendations",
            "outputs/factor_ablation_demo/feature_pruning_recommendations.csv",
            "--output-dir",
            "outputs/reduced_feature_backtest_demo",
            "--models",
            "logistic_regression,random_forest",
            "--target-col",
            "label_up_5d",
        ],
    ),
    (
        "offline reduced feature backtest output files",
        [
            "-c",
            (
                "from pathlib import Path; "
                "base=Path('outputs/reduced_feature_backtest_demo'); "
                "required=['reduced_feature_backtest_results.csv',"
                "'reduced_feature_backtest_summary.csv','warnings.csv',"
                "'run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing"
            ),
        ],
    ),
    (
        "offline synthetic reduced feature summary inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import json\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/reduced_feature_summary_smoke_inputs')\\n"
                "rows=[\\n"
                "{'symbol':'000001','model_type':'logistic_regression','pruning_mode':'full','feature_count':44,'buy_threshold':0.6,'sell_threshold':0.5,'total_return_pct':1.0,'benchmark_return_pct':2.0,'strategy_vs_benchmark_pct':-1.0,'max_drawdown_pct':-3.0,'trade_count':2,'win_rate_pct':50.0,'final_value':10100,'warning':'low trades'},\\n"
                "{'symbol':'000001','model_type':'random_forest','pruning_mode':'drop_reduce_weight','feature_count':40,'buy_threshold':0.6,'sell_threshold':0.5,'total_return_pct':4.0,'benchmark_return_pct':2.0,'strategy_vs_benchmark_pct':2.0,'max_drawdown_pct':-2.0,'trade_count':5,'win_rate_pct':60.0,'final_value':10400},\\n"
                "{'symbol':'600519','model_type':'logistic_regression','pruning_mode':'full','feature_count':44,'buy_threshold':0.6,'sell_threshold':0.5,'total_return_pct':-1.0,'benchmark_return_pct':1.0,'strategy_vs_benchmark_pct':-2.0,'max_drawdown_pct':-4.0,'trade_count':4,'win_rate_pct':40.0,'final_value':9900},\\n"
                "{'symbol':'600519','model_type':'random_forest','pruning_mode':'drop_reduce_weight','feature_count':40,'buy_threshold':0.6,'sell_threshold':0.5,'total_return_pct':3.0,'benchmark_return_pct':1.0,'strategy_vs_benchmark_pct':2.0,'max_drawdown_pct':-2.5,'trade_count':6,'win_rate_pct':66.7,'final_value':10300}\\n"
                "]\\n"
                "for symbol in ['000001','600519']:\\n"
                "    d=base/f'reduced_feature_backtest_real_{symbol}'\\n"
                "    d.mkdir(parents=True, exist_ok=True)\\n"
                "    df=pd.DataFrame([row for row in rows if row['symbol']==symbol])\\n"
                "    df.to_csv(d/'reduced_feature_backtest_results.csv', index=False)\\n"
                "    summary=df.groupby(['pruning_mode','model_type']).agg(symbol_count=('symbol','nunique'),avg_feature_count=('feature_count','mean'),avg_total_return_pct=('total_return_pct','mean'),avg_benchmark_return_pct=('benchmark_return_pct','mean'),avg_strategy_vs_benchmark_pct=('strategy_vs_benchmark_pct','mean'),avg_max_drawdown_pct=('max_drawdown_pct','mean'),avg_trade_count=('trade_count','mean'),avg_win_rate_pct=('win_rate_pct','mean'),avg_final_value=('final_value','mean')).reset_index()\\n"
                "    summary.to_csv(d/'reduced_feature_backtest_summary.csv', index=False)\\n"
                "    pd.DataFrame(columns=['symbol','model_type','pruning_mode','warning']).to_csv(d/'warnings.csv', index=False)\\n"
                "    (d/'run_config.json').write_text(json.dumps({'factor_csv':f'data/factors/factors_{symbol}.csv'}), encoding='utf-8')\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline reduced feature summary report",
        [
            "src/generate_reduced_feature_backtest_report.py",
            "--input-dirs",
            "outputs/reduced_feature_summary_smoke_inputs/reduced_feature_backtest_real_000001,outputs/reduced_feature_summary_smoke_inputs/reduced_feature_backtest_real_600519",
            "--output-dir",
            "outputs/reduced_feature_backtest_summary_smoke",
            "--min-trades",
            "3",
        ],
    ),
    (
        "offline reduced feature summary output files",
        [
            "-c",
            (
                "from pathlib import Path; "
                "base=Path('outputs/reduced_feature_backtest_summary_smoke'); "
                "required=['combined_reduced_feature_backtest_results.csv',"
                "'reduced_feature_backtest_mode_summary.csv',"
                "'reduced_feature_backtest_model_summary.csv',"
                "'reduced_feature_backtest_mode_model_summary.csv',"
                "'per_symbol_best_backtest_modes.csv','underperformance_cases.csv',"
                "'warnings.csv','run_config.json',"
                "'reduced_feature_backtest_report.md']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing"
            ),
        ],
    ),
    (
        "offline reduced feature threshold experiment",
        [
            "src/run_reduced_feature_threshold_experiment.py",
            "--factor-csv",
            "data/factors/smoke_factors_000001.csv",
            "--recommendations",
            "outputs/factor_ablation_demo/feature_pruning_recommendations.csv",
            "--output-dir",
            "outputs/reduced_feature_threshold_demo",
            "--models",
            "logistic_regression",
            "--pruning-modes",
            "full,keep_core_and_observe",
            "--buy-thresholds",
            "0.55,0.60",
            "--sell-thresholds",
            "0.40,0.50",
            "--minimum-commission",
            "0",
            "--commission-rate",
            "0",
            "--stamp-tax-rate",
            "0",
            "--slippage-pct",
            "0",
            "--enable-walk-forward",
            "--walk-forward-train-ratio",
            "0.45",
            "--walk-forward-validation-ratio",
            "0.15",
            "--walk-forward-test-ratio",
            "0.20",
            "--walk-forward-step-ratio",
            "0.20",
        ],
    ),
    (
        "offline reduced feature threshold output files",
        [
            "-c",
            (
                "from pathlib import Path; "
                "base=Path('outputs/reduced_feature_threshold_demo'); "
                "required=['threshold_backtest_results.csv',"
                "'threshold_summary_by_mode.csv','threshold_summary_by_model.csv',"
                "'threshold_summary_by_mode_model.csv','best_thresholds.csv',"
                "'walk_forward_results.csv','walk_forward_summary.csv',"
                "'warnings.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing"
            ),
        ],
    ),
    (
        "offline threshold experiment report",
        [
            "src/generate_threshold_experiment_report.py",
            "--input-dirs",
            "outputs/reduced_feature_threshold_demo",
            "--output-dir",
            "outputs/reduced_feature_threshold_summary_demo",
        ],
    ),
    (
        "offline threshold experiment report output files",
        [
            "-c",
            (
                "from pathlib import Path; "
                "base=Path('outputs/reduced_feature_threshold_summary_demo'); "
                "required=['combined_threshold_results.csv',"
                "'threshold_mode_summary.csv','threshold_model_summary.csv',"
                "'threshold_mode_model_summary.csv','per_symbol_best_thresholds.csv',"
                "'walk_forward_combined_results.csv','walk_forward_summary.csv',"
                "'threshold_experiment_report.md','warnings.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "report=(base/'threshold_experiment_report.md').read_text(encoding='utf-8'); "
                "assert '## Best Historical Threshold Results' in report; "
                "assert '## Recommended Walk-Forward Candidate' in report; "
                "assert '000001' in report; "
                "import pandas as pd; "
                "best=pd.read_csv(base/'per_symbol_best_thresholds.csv', dtype={'symbol': str}); "
                "assert 'selection_confidence' in best.columns; "
                "assert best['symbol'].iloc[0] == '000001'"
            ),
        ],
    ),
    (
        "offline synthetic threshold low-trade report inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import json\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/threshold_report_low_trade_smoke_inputs')\\n"
                "d=base/'reduced_feature_threshold_real_000001'\\n"
                "d.mkdir(parents=True, exist_ok=True)\\n"
                "rows=[\\n"
                "{'symbol':'000001','model_type':'logistic_regression','pruning_mode':'full','feature_count':10,'buy_threshold':0.6,'sell_threshold':0.5,'total_return_pct':2.0,'benchmark_return_pct':1.0,'strategy_vs_benchmark_pct':1.0,'max_drawdown_pct':-1.0,'trade_count':3,'win_rate_pct':50.0,'final_value':10200,'warning':'low_trade_count: 3 | underperformed_benchmark'},\\n"
                "{'symbol':'000001','model_type':'logistic_regression','pruning_mode':'full','feature_count':10,'buy_threshold':0.65,'sell_threshold':0.5,'total_return_pct':4.0,'benchmark_return_pct':1.0,'strategy_vs_benchmark_pct':3.0,'max_drawdown_pct':-1.5,'trade_count':2,'win_rate_pct':50.0,'final_value':10400,'warning':'low_trade_count: 2'},\\n"
                "{'symbol':'000001','model_type':'random_forest','pruning_mode':'drop_reduce_weight','feature_count':8,'buy_threshold':0.6,'sell_threshold':0.5,'total_return_pct':1.5,'benchmark_return_pct':1.0,'strategy_vs_benchmark_pct':0.5,'max_drawdown_pct':-1.2,'trade_count':4,'win_rate_pct':60.0,'final_value':10150,'warning':None}\\n"
                "]\\n"
                "pd.DataFrame(rows).to_csv(d/'threshold_backtest_results.csv', index=False)\\n"
                "pd.DataFrame(rows).to_csv(d/'walk_forward_results.csv', index=False)\\n"
                "warnings=[{'symbol':'000001','model_type':'logistic_regression','pruning_mode':'full','buy_threshold':0.6,'sell_threshold':0.5,'warning_type':'threshold_result_warning','message':'low_trade_count: 3'},{'symbol':'000001','model_type':'logistic_regression','pruning_mode':'full','buy_threshold':0.65,'sell_threshold':0.5,'warning_type':'threshold_result_warning','message':'low_trade_count: 2'}]\\n"
                "pd.DataFrame(warnings).to_csv(d/'warnings.csv', index=False)\\n"
                "(d/'run_config.json').write_text(json.dumps({'factor_csv':'data/factors/factors_000001.csv','min_trades':3}), encoding='utf-8')\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline synthetic threshold low-trade report",
        [
            "src/generate_threshold_experiment_report.py",
            "--input-dirs",
            "outputs/threshold_report_low_trade_smoke_inputs/reduced_feature_threshold_real_000001",
            "--output-dir",
            "outputs/threshold_report_low_trade_smoke",
        ],
    ),
    (
        "offline synthetic threshold low-trade report assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/threshold_report_low_trade_smoke'); "
                "report=(base/'threshold_experiment_report.md').read_text(encoding='utf-8'); "
                "assert '## Low-Confidence and Low-Trade Cases' in report; "
                "low_section=report.split('## Low-Confidence and Low-Trade Cases', 1)[1].split('## Research Warnings', 1)[0]; "
                "assert 'low_trade_count: 2' in low_section; "
                "assert '| 3 |' in report; "
                "assert 'low_trade_count: 3' not in report; "
                "combined=pd.read_csv(base/'combined_threshold_results.csv', dtype={'symbol': str}); "
                "assert combined['symbol'].iloc[0] == '000001'; "
                "assert 'low_trade_count: 3' not in combined.to_string(); "
                "best=pd.read_csv(base/'per_symbol_best_thresholds.csv', dtype={'symbol': str}); "
                "assert best.loc[0, 'symbol'] == '000001'; "
                "assert int(best.loc[0, 'best_trade_count']) == 3"
            ),
        ],
    ),
    (
        "offline synthetic threshold decision inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/threshold_decision_smoke_inputs')\\n"
                "base.mkdir(parents=True, exist_ok=True)\\n"
                "pd.DataFrame([\\n"
                "{'pruning_mode':'full','symbol_count':2,'threshold_count':8,'avg_feature_count':44,'avg_total_return_pct':-1.0,'avg_benchmark_return_pct':2.0,'avg_strategy_vs_benchmark_pct':-3.0,'avg_trade_count':5,'sufficient_trade_rate':1.0,'beat_benchmark_rate':0.0,'stability_score':-0.01},\\n"
                "{'pruning_mode':'keep_core_and_observe','symbol_count':2,'threshold_count':8,'avg_feature_count':18,'avg_total_return_pct':1.5,'avg_benchmark_return_pct':2.5,'avg_strategy_vs_benchmark_pct':-1.0,'avg_trade_count':4,'sufficient_trade_rate':1.0,'beat_benchmark_rate':0.25,'stability_score':0.30},\\n"
                "{'pruning_mode':'drop_reduce_weight','symbol_count':2,'threshold_count':8,'avg_feature_count':30,'avg_total_return_pct':1.2,'avg_benchmark_return_pct':2.4,'avg_strategy_vs_benchmark_pct':-1.2,'avg_trade_count':4,'sufficient_trade_rate':1.0,'beat_benchmark_rate':0.25,'stability_score':0.28}\\n"
                "]).to_csv(base/'threshold_mode_summary.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'model_type':'logistic_regression','symbol_count':2,'threshold_count':12,'avg_total_return_pct':1.4,'avg_strategy_vs_benchmark_pct':-0.8,'avg_trade_count':5,'sufficient_trade_rate':1.0,'beat_benchmark_rate':0.25,'stability_score':0.31},\\n"
                "{'model_type':'random_forest','symbol_count':2,'threshold_count':12,'avg_total_return_pct':0.5,'avg_strategy_vs_benchmark_pct':-2.0,'avg_trade_count':2,'sufficient_trade_rate':0.5,'beat_benchmark_rate':0.0,'stability_score':0.10}\\n"
                "]).to_csv(base/'threshold_model_summary.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'pruning_mode':'keep_core_and_observe','model_type':'logistic_regression','avg_total_return_pct':1.5,'avg_strategy_vs_benchmark_pct':-1.0,'avg_trade_count':5,'stability_score':0.30},\\n"
                "{'pruning_mode':'full','model_type':'random_forest','avg_total_return_pct':-1.0,'avg_strategy_vs_benchmark_pct':-3.0,'avg_trade_count':3,'stability_score':-0.01}\\n"
                "]).to_csv(base/'threshold_mode_model_summary.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'symbol':'000001','best_pruning_mode':'keep_core_and_observe','best_model_type':'logistic_regression','best_buy_threshold':0.55,'best_sell_threshold':0.40,'best_total_return_pct':2.0,'best_strategy_vs_benchmark_pct':-0.5,'best_trade_count':4,'selection_confidence':'normal'},\\n"
                "{'symbol':'000858','best_pruning_mode':'keep_core_only','best_model_type':'logistic_regression','best_buy_threshold':0.65,'best_sell_threshold':0.50,'best_total_return_pct':8.0,'best_strategy_vs_benchmark_pct':5.0,'best_trade_count':1,'selection_confidence':'low_confidence_low_trade_count'}\\n"
                "]).to_csv(base/'per_symbol_best_thresholds.csv', index=False)\\n"
                "pd.DataFrame([{'pruning_mode':'keep_core_and_observe','model_type':'logistic_regression','buy_threshold':0.55,'sell_threshold':0.40,'avg_strategy_vs_benchmark_pct':-0.5,'sufficient_trade_rate':1.0,'beat_benchmark_rate':0.25,'stability_score':0.35}]).to_csv(base/'walk_forward_summary.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000858','model_type':'logistic_regression','pruning_mode':'keep_core_only','warning_type':'threshold_result_warning','message':'low_trade_count: 1'}]).to_csv(base/'warnings.csv', index=False)\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline threshold decision report",
        [
            "src/generate_threshold_decision_report.py",
            "--summary-dir",
            "outputs/threshold_decision_smoke_inputs",
            "--output-dir",
            "outputs/threshold_decision_smoke",
        ],
    ),
    (
        "offline threshold decision report assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/threshold_decision_smoke'); "
                "required=['threshold_decision_report.md','threshold_decision_summary.csv','rejected_or_low_confidence_configs.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "report=(base/'threshold_decision_report.md').read_text(encoding='utf-8'); "
                "phrases=['not trading-ready','underperforms benchmark on average','low-confidence','recommended research candidate','Recommended Walk-Forward Candidate']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases; "
                "rejected=pd.read_csv(base/'rejected_or_low_confidence_configs.csv', dtype={'symbol': str}); "
                "assert '000001' in report; "
                "assert '000858' in set(rejected.get('symbol', pd.Series(dtype=str)).dropna())"
            ),
        ],
    ),
    (
        "offline synthetic candidate validation inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/candidate_validation_smoke_inputs')\\n"
                "factor_dir=base/'factors'\\n"
                "factor_dir.mkdir(parents=True, exist_ok=True)\\n"
                "for symbol, offset in [('000001', 0), ('000858', 1)]:\\n"
                "    rows=[]\\n"
                "    for i in range(90):\\n"
                "        close=10 + i * 0.05 + offset\\n"
                "        label=1 if i % 4 in (0, 1) else 0\\n"
                "        rows.append({'date':f'2024-01-{(i % 28) + 1:02d}','symbol':symbol,'open':close - 0.02,'high':close + 0.10,'low':close - 0.10,'close':close,'volume':1000 + i,'f_core':label + (i % 3) * 0.01,'f_observe':(i % 5) * 0.1,'f_reduce':(i % 7) * 0.1,'label_up_5d':label})\\n"
                "    pd.DataFrame(rows).to_csv(factor_dir/f'factors_{symbol}.csv', index=False)\\n"
                "pd.DataFrame([{'feature':'f_core','recommendation':'keep_core'},{'feature':'f_observe','recommendation':'keep_observe'},{'feature':'f_reduce','recommendation':'reduce_weight'}]).to_csv(base/'recommendations.csv', index=False)\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline candidate expanded validation",
        [
            "src/run_candidate_expanded_validation.py",
            "--factor-dir",
            "outputs/candidate_validation_smoke_inputs/factors",
            "--symbols",
            "000001,000858",
            "--recommendations",
            "outputs/candidate_validation_smoke_inputs/recommendations.csv",
            "--output-dir",
            "outputs/candidate_validation_smoke",
            "--candidate-pruning-mode",
            "keep_core_and_observe",
            "--candidate-model",
            "logistic_regression",
            "--candidate-buy-threshold",
            "0.50",
            "--candidate-sell-threshold",
            "0.40",
            "--minimum-commission",
            "0",
            "--commission-rate",
            "0",
            "--stamp-tax-rate",
            "0",
            "--slippage-pct",
            "0",
            "--min-trades",
            "50",
        ],
    ),
    (
        "offline candidate expanded validation assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/candidate_validation_smoke'); "
                "required=['candidate_validation_results.csv','candidate_validation_summary.csv','per_symbol_candidate_results.csv','candidate_validation_warnings.csv','candidate_validation_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "report=(base/'candidate_validation_report.md').read_text(encoding='utf-8'); "
                "assert 'not trading-ready' in report; "
                "assert 'research-only' in report; "
                "results=pd.read_csv(base/'candidate_validation_results.csv', dtype={'symbol': str}); "
                "assert set(results['symbol']) == {'000001','000858'}; "
                "assert 'canonical_mode' in results.columns; "
                "assert set(results['canonical_mode']) == {'canonical_reduced_40'}; "
                "warnings=pd.read_csv(base/'candidate_validation_warnings.csv', dtype={'symbol': str}); "
                "assert not warnings.empty; "
                "assert warnings.to_string().find('low_trade_count') >= 0"
            ),
        ],
    ),
    (
        "offline candidate stress test",
        [
            "src/run_candidate_stress_test.py",
            "--factor-dir",
            "outputs/candidate_validation_smoke_inputs/factors",
            "--symbols",
            "000001,000858",
            "--recommendations",
            "outputs/candidate_validation_smoke_inputs/recommendations.csv",
            "--output-dir",
            "outputs/candidate_stress_smoke",
            "--candidate-pruning-mode",
            "keep_core_and_observe",
            "--candidate-model",
            "logistic_regression",
            "--candidate-buy-threshold",
            "0.50",
            "--candidate-sell-threshold",
            "0.40",
            "--walk-forward-pruning-mode",
            "drop_reduce_weight",
            "--walk-forward-model",
            "logistic_regression",
            "--walk-forward-buy-threshold",
            "0.50",
            "--walk-forward-sell-threshold",
            "0.40",
            "--minimum-commission",
            "0",
            "--commission-rate",
            "0",
            "--stamp-tax-rate",
            "0",
            "--slippage-pct",
            "0",
            "--min-trades",
            "50",
            "--regime-window",
            "5",
            "--enable-walk-forward",
        ],
    ),
    (
        "offline candidate stress test assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/candidate_stress_smoke'); "
                "required=['candidate_stress_results.csv','candidate_stress_summary.csv','per_symbol_stress_results.csv','regime_summary.csv','stress_warnings.csv','candidate_stress_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "report=(base/'candidate_stress_report.md').read_text(encoding='utf-8'); "
                "assert 'not trading-ready' in report; "
                "assert 'Regime Diagnostics' in report; "
                "results=pd.read_csv(base/'candidate_stress_results.csv', dtype={'symbol': str}); "
                "assert {'000001','000858'}.issubset(set(results['symbol'])); "
                "assert {'legacy_pruning_mode','canonical_mode'}.issubset(results.columns); "
                "assert set(results['canonical_mode']) == {'canonical_reduced_40'}; "
                "assert 'bull' in set(results['regime']) or 'sideways' in set(results['regime']); "
                "assert results['feature_count'].max() <= 7; "
                "summary=pd.read_csv(base/'candidate_stress_summary.csv'); "
                "assert 'canonical_reduced_40' in set(summary['canonical_mode']); "
                "assert 'drop_reduce_weight' not in set(summary['canonical_mode']); "
                "assert 'keep_core_and_observe' not in set(summary['canonical_mode']); "
                "warnings=pd.read_csv(base/'stress_warnings.csv', dtype={'symbol': str}); "
                "assert not warnings.empty"
            ),
        ],
    ),
    (
        "offline candidate equivalence audit",
        [
            "src/run_candidate_equivalence_audit.py",
            "--factor-dir",
            "outputs/candidate_validation_smoke_inputs/factors",
            "--symbols",
            "000001,000858",
            "--recommendations",
            "outputs/candidate_validation_smoke_inputs/recommendations.csv",
            "--output-dir",
            "outputs/candidate_equivalence_smoke",
        ],
    ),
    (
        "offline candidate equivalence audit assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/candidate_equivalence_smoke'); "
                "required=['selected_features_by_symbol_mode.csv','feature_set_overlap_matrix.csv','feature_set_equivalence_summary.csv','feature_frequency_by_mode.csv','candidate_equivalence_report.md','warnings.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "selected=pd.read_csv(base/'selected_features_by_symbol_mode.csv', dtype={'symbol': str}); "
                "assert {'000001','000858'}.issubset(set(selected['symbol'])); "
                "assert {'full','drop_reduce_weight','keep_core_only','keep_core_and_observe'}.issubset(set(selected['pruning_mode'])); "
                "overlap=pd.read_csv(base/'feature_set_overlap_matrix.csv', dtype={'symbol': str}); "
                "assert 'jaccard_similarity' in overlap.columns; "
                "report=(base/'candidate_equivalence_report.md').read_text(encoding='utf-8'); "
                "assert 'drop_reduce_weight and keep_core_and_observe' in report; "
                "assert 'not a trading recommendation' in report"
            ),
        ],
    ),
    (
        "offline candidate mode normalization",
        [
            "src/run_candidate_mode_normalization.py",
            "--equivalence-dir",
            "outputs/candidate_equivalence_smoke",
            "--output-dir",
            "outputs/candidate_mode_normalization_smoke",
        ],
    ),
    (
        "offline candidate mode normalization assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/candidate_mode_normalization_smoke'); "
                "required=['canonical_mode_summary.csv','legacy_alias_map.csv','canonical_mode_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'canonical_mode_summary.csv'); "
                "aliases=pd.read_csv(base/'legacy_alias_map.csv'); "
                "assert 'canonical_reduced_40' in set(summary['canonical_mode']); "
                "alias_rows=aliases[aliases['legacy_mode'].isin(['drop_reduce_weight','keep_core_and_observe'])]; "
                "assert set(alias_rows['canonical_mode']) == {'canonical_reduced_40'}; "
                "assert set(aliases[aliases['legacy_mode'].isin(['full','keep_core_only'])]['canonical_mode']) == {'full','keep_core_only'}; "
                "report=(base/'canonical_mode_report.md').read_text(encoding='utf-8'); "
                "assert 'avoid redundant comparisons' in report; "
                "selected=pd.read_csv('outputs/candidate_equivalence_smoke/selected_features_by_symbol_mode.csv', dtype={'symbol': str}); "
                "assert {'000001','000858'}.issubset(set(selected['symbol']))"
            ),
        ],
    ),
    (
        "offline synthetic canonical revalidation inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/canonical_revalidation_smoke_inputs')\\n"
                "expanded=base/'expanded'\\n"
                "stress=base/'stress'\\n"
                "threshold=base/'threshold'\\n"
                "for d in [expanded, stress, threshold]: d.mkdir(parents=True, exist_ok=True)\\n"
                "pd.DataFrame([\\n"
                "{'canonical_mode':'canonical_reduced_40','legacy_pruning_modes':'drop_reduce_weight,keep_core_and_observe','model_type':'logistic_regression','buy_threshold':0.5,'sell_threshold':0.4,'avg_strategy_vs_benchmark_pct':1.0,'final_decision':'pass'},\\n"
                "{'canonical_mode':'full','legacy_pruning_modes':'full','model_type':'logistic_regression','buy_threshold':0.5,'sell_threshold':0.4,'avg_strategy_vs_benchmark_pct':-1.0,'final_decision':'fail'},\\n"
                "{'canonical_mode':'keep_core_only','legacy_pruning_modes':'keep_core_only','model_type':'logistic_regression','buy_threshold':0.5,'sell_threshold':0.4,'avg_strategy_vs_benchmark_pct':2.0,'final_decision':'fail'}\\n"
                "]).to_csv(expanded/'candidate_validation_summary.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000001','canonical_mode':'canonical_reduced_40','legacy_pruning_mode':'keep_core_and_observe','strategy_vs_benchmark_pct':1.0}]).to_csv(expanded/'candidate_validation_results.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000858','canonical_mode':'','legacy_pruning_mode':'keep_core_only','warning_type':'low_trade_count','message':'low_trade_count: 1'}]).to_csv(expanded/'candidate_validation_warnings.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'canonical_mode':'canonical_reduced_40','legacy_pruning_modes':'drop_reduce_weight,keep_core_and_observe','avg_strategy_vs_benchmark_pct':-0.5,'beat_benchmark_rate':0.4,'sufficient_trade_rate':0.8,'final_decision':'fail'},\\n"
                "{'canonical_mode':'full','legacy_pruning_modes':'full','avg_strategy_vs_benchmark_pct':-2.0,'beat_benchmark_rate':0.2,'sufficient_trade_rate':1.0,'final_decision':'fail'},\\n"
                "{'canonical_mode':'keep_core_only','legacy_pruning_modes':'keep_core_only','avg_strategy_vs_benchmark_pct':0.5,'beat_benchmark_rate':0.5,'sufficient_trade_rate':0.2,'final_decision':'fail'}\\n"
                "]).to_csv(stress/'candidate_stress_summary.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000001','canonical_mode':'canonical_reduced_40','legacy_pruning_mode':'drop_reduce_weight','strategy_vs_benchmark_pct':-0.5},{'symbol':'000858','canonical_mode':'keep_core_only','legacy_pruning_mode':'keep_core_only','strategy_vs_benchmark_pct':-1.0}]).to_csv(stress/'candidate_stress_results.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000001','canonical_mode':'n/a','legacy_pruning_mode':'drop_reduce_weight','warning_type':'underperformed_benchmark','message':'underperformed benchmark'}]).to_csv(stress/'stress_warnings.csv', index=False)\\n"
                "pd.DataFrame([{'decision_item':'pruning_mode','recommended_pruning_mode':'canonical_reduced_40','recommended_legacy_pruning_modes':'drop_reduce_weight,keep_core_and_observe'}]).to_csv(threshold/'threshold_decision_summary.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000858','canonical_mode':'n/a','recommended_pruning_mode':'keep_core_only','reason':'low-confidence best threshold'}]).to_csv(threshold/'rejected_or_low_confidence_configs.csv', index=False)\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline canonical candidate revalidation report",
        [
            "src/generate_canonical_candidate_revalidation_report.py",
            "--expanded-validation-dir",
            "outputs/canonical_revalidation_smoke_inputs/expanded",
            "--stress-dir",
            "outputs/canonical_revalidation_smoke_inputs/stress",
            "--threshold-decision-dir",
            "outputs/canonical_revalidation_smoke_inputs/threshold",
            "--output-dir",
            "outputs/canonical_revalidation_smoke",
        ],
    ),
    (
        "offline canonical candidate revalidation assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/canonical_revalidation_smoke'); "
                "required=['canonical_candidate_revalidation_report.md','canonical_candidate_revalidation_summary.csv','candidate_risk_flags.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'canonical_candidate_revalidation_summary.csv'); "
                "assert {'canonical_reduced_40','full','keep_core_only'} == set(summary['canonical_mode']); "
                "report=(base/'canonical_candidate_revalidation_report.md').read_text(encoding='utf-8'); "
                "phrases=['canonical_reduced_40 is the current primary research candidate','not trading-ready','full is baseline only','keep_core_only is a low-feature challenger']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases; "
                "flags=pd.read_csv(base/'candidate_risk_flags.csv', dtype={'symbol': str}); "
                "assert '000001' in set(flags.get('symbol', pd.Series(dtype=str)).dropna()); "
                "canonical_modes=flags.get('canonical_mode', pd.Series(dtype=str)).fillna('').astype(str).str.strip().str.lower(); "
                "assert len(canonical_modes) > 0; "
                "assert not canonical_modes.isin(['','n/a','na','nan']).any(); "
                "assert {'canonical_reduced_40','keep_core_only'}.issubset(set(flags['canonical_mode']))"
            ),
        ],
    ),
    (
        "offline synthetic candidate validation gate inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/candidate_validation_gate_smoke_inputs')\\n"
                "base.mkdir(parents=True, exist_ok=True)\\n"
                "pd.DataFrame([\\n"
                "{'canonical_mode':'canonical_reduced_40','role':'primary_research_candidate','validation_decision':'pass','stress_decision':'fail','avg_validation_excess_pct':1.2,'avg_stress_excess_pct':-0.4,'stress_beat_benchmark_rate':0.40,'stress_sufficient_trade_rate':0.80,'final_research_decision':'research_only_not_trading_ready','decision_reason':'stress validation still fails'},\\n"
                "{'canonical_mode':'full','role':'baseline_only','validation_decision':'pass','stress_decision':'pass','avg_validation_excess_pct':0.5,'avg_stress_excess_pct':0.3,'stress_beat_benchmark_rate':0.80,'stress_sufficient_trade_rate':1.00,'final_research_decision':'baseline_only','decision_reason':'full is retained as baseline only'},\\n"
                "{'canonical_mode':'keep_core_only','role':'low_feature_challenger','validation_decision':'not_tested','stress_decision':'not_tested','avg_validation_excess_pct':None,'avg_stress_excess_pct':None,'stress_beat_benchmark_rate':None,'stress_sufficient_trade_rate':None,'final_research_decision':'low_confidence_challenger','decision_reason':'low-trade-count risk'}\\n"
                "]).to_csv(base/'canonical_candidate_revalidation_summary.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'source':'stress_validation','risk_category':'benchmark_underperformance','canonical_mode':'canonical_reduced_40','symbol':'000001','warning_type':'underperformed_benchmark','message':'underperformed benchmark'},\\n"
                "{'source':'threshold_decision','risk_category':'low_trade_or_low_confidence','canonical_mode':'keep_core_only','symbol':'000858','reason':'low-confidence best threshold'}\\n"
                "]).to_csv(base/'candidate_risk_flags.csv', index=False)\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline candidate validation gate",
        [
            "src/run_candidate_validation_gate.py",
            "--revalidation-dir",
            "outputs/candidate_validation_gate_smoke_inputs",
            "--output-dir",
            "outputs/candidate_validation_gate_smoke",
        ],
    ),
    (
        "offline candidate validation gate assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/candidate_validation_gate_smoke'); "
                "required=['validation_gate_results.csv','validation_gate_failures.csv','candidate_validation_gate_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "results=pd.read_csv(base/'validation_gate_results.csv'); "
                "decisions=dict(zip(results['canonical_mode'], results['final_gate_decision'])); "
                "assert decisions['canonical_reduced_40'] == 'research_only_not_trading_ready'; "
                "assert decisions['full'] == 'baseline_only'; "
                "assert decisions['keep_core_only'] != 'trading_ready'; "
                "assert 'trading_ready' in results.columns; "
                "assert not results['trading_ready'].isna().any(); "
                "ready=dict(zip(results['canonical_mode'], results['trading_ready'].astype(bool))); "
                "assert not bool(ready['canonical_reduced_40']); "
                "assert not bool(ready['full']); "
                "assert not bool(ready['keep_core_only']); "
                "assert not results.loc[results['final_gate_decision'] != 'trading_ready', 'trading_ready'].astype(bool).any(); "
                "failures=pd.read_csv(base/'validation_gate_failures.csv'); "
                "assert not failures.empty; "
                "report=(base/'candidate_validation_gate_report.md').read_text(encoding='utf-8'); "
                "phrases=['not trading-ready','validation gate','canonical_reduced_40','full','keep_core_only']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "offline synthetic perfect candidate validation gate inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/candidate_validation_gate_perfect_smoke_inputs')\\n"
                "base.mkdir(parents=True, exist_ok=True)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','role':'primary_research_candidate','validation_decision':'pass','stress_decision':'pass','avg_validation_excess_pct':2.0,'avg_stress_excess_pct':1.0,'stress_beat_benchmark_rate':0.75,'stress_sufficient_trade_rate':0.90,'final_research_decision':'pass','decision_reason':'all strict checks pass'}]).to_csv(base/'canonical_candidate_revalidation_summary.csv', index=False)\\n"
                "pd.DataFrame(columns=['source','risk_category','canonical_mode','symbol','warning_type','message']).to_csv(base/'candidate_risk_flags.csv', index=False)\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline perfect candidate validation gate",
        [
            "src/run_candidate_validation_gate.py",
            "--revalidation-dir",
            "outputs/candidate_validation_gate_perfect_smoke_inputs",
            "--output-dir",
            "outputs/candidate_validation_gate_perfect_smoke",
        ],
    ),
    (
        "offline perfect candidate validation gate assertions",
        [
            "-c",
            (
                "import pandas as pd; "
                "results=pd.read_csv('outputs/candidate_validation_gate_perfect_smoke/validation_gate_results.csv'); "
                "row=results.iloc[0]; "
                "assert row['canonical_mode'] == 'canonical_reduced_40'; "
                "assert row['final_gate_decision'] == 'trading_ready'; "
                "assert bool(row['strict_gates_passed']); "
                "assert 'trading_ready' in results.columns; "
                "assert not results['trading_ready'].isna().any(); "
                "assert bool(row['trading_ready'])"
            ),
        ],
    ),
    (
        "offline synthetic validation gate failure analysis inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/validation_gate_failure_analysis_smoke_inputs')\\n"
                "gate=base/'gate'\\n"
                "reval=base/'revalidation'\\n"
                "stress=base/'stress'\\n"
                "for d in [gate, reval, stress]: d.mkdir(parents=True, exist_ok=True)\\n"
                "pd.DataFrame([\\n"
                "{'canonical_mode':'canonical_reduced_40','role':'primary_research_candidate','final_gate_decision':'research_only_not_trading_ready','trading_ready':False,'strict_gates_passed':False,'gate_reason':'blocked by stress validation failure'},\\n"
                "{'canonical_mode':'full','role':'baseline_only','final_gate_decision':'baseline_only','trading_ready':False,'strict_gates_passed':False,'gate_reason':'baseline only'},\\n"
                "{'canonical_mode':'keep_core_only','role':'low_feature_challenger','final_gate_decision':'rejected_or_not_tested','trading_ready':False,'strict_gates_passed':False,'gate_reason':'not tested'}\\n"
                "]).to_csv(gate/'validation_gate_results.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'canonical_mode':'canonical_reduced_40','check_name':'stress_validation_passed','severity':'blocking','message':'stress validation failed'},\\n"
                "{'canonical_mode':'canonical_reduced_40','check_name':'stress_beat_benchmark_rate_passed','severity':'blocking','message':'beat rate below threshold'},\\n"
                "{'canonical_mode':'canonical_reduced_40','check_name':'stress_sufficient_trade_rate_passed','severity':'blocking','message':'trade rate below threshold'},\\n"
                "{'canonical_mode':'canonical_reduced_40','check_name':'risk_flags_acceptable','severity':'blocking','message':'risk flags remain'},\\n"
                "{'canonical_mode':'full','check_name':'role_allowed','severity':'blocking','message':'baseline only'},\\n"
                "{'canonical_mode':'keep_core_only','check_name':'expanded_validation_passed','severity':'blocking','message':'not tested'}\\n"
                "]).to_csv(gate/'validation_gate_failures.csv', index=False)\\n"
                "(gate/'candidate_validation_gate_report.md').write_text('No candidate is not trading-ready under the validation gate.', encoding='utf-8')\\n"
                "pd.DataFrame([\\n"
                "{'canonical_mode':'canonical_reduced_40','role':'primary_research_candidate','validation_decision':'pass','stress_decision':'fail','avg_validation_excess_pct':1.0,'avg_stress_excess_pct':-0.5,'stress_beat_benchmark_rate':0.4,'stress_sufficient_trade_rate':0.7,'final_research_decision':'research_only_not_trading_ready','decision_reason':'stress validation fail due to regime weakness'},\\n"
                "{'canonical_mode':'full','role':'baseline_only','validation_decision':'pass','stress_decision':'pass','avg_validation_excess_pct':0.2,'avg_stress_excess_pct':0.1,'stress_beat_benchmark_rate':0.8,'stress_sufficient_trade_rate':1.0,'final_research_decision':'baseline_only','decision_reason':'baseline only'},\\n"
                "{'canonical_mode':'keep_core_only','role':'low_feature_challenger','validation_decision':'not_tested','stress_decision':'not_tested','avg_validation_excess_pct':None,'avg_stress_excess_pct':None,'stress_beat_benchmark_rate':None,'stress_sufficient_trade_rate':None,'final_research_decision':'rejected_or_not_tested','decision_reason':'low-trade-count risk'}\\n"
                "]).to_csv(reval/'canonical_candidate_revalidation_summary.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'source':'stress_validation','risk_category':'benchmark_underperformance','canonical_mode':'canonical_reduced_40','symbol':'1','warning_type':'underperformed_benchmark','message':'underperformed benchmark'},\\n"
                "{'source':'threshold_decision','risk_category':'low_trade_or_low_confidence','canonical_mode':'keep_core_only','symbol':'858','warning_type':'low_trade_count','message':'low confidence'}\\n"
                "]).to_csv(reval/'candidate_risk_flags.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','final_decision':'fail','decision_reason':'failed regimes: bear, sideways'}]).to_csv(stress/'candidate_stress_summary.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'canonical_mode':'canonical_reduced_40','regime':'bull','tested_symbol_count':2,'avg_strategy_vs_benchmark_pct':-0.5,'beat_benchmark_rate':0.75,'sufficient_trade_rate':0.80},\\n"
                "{'canonical_mode':'canonical_reduced_40','regime':'bear','tested_symbol_count':2,'avg_strategy_vs_benchmark_pct':0.5,'beat_benchmark_rate':0.75,'sufficient_trade_rate':0.80},\\n"
                "{'canonical_mode':'canonical_reduced_40','regime':'sideways','tested_symbol_count':2,'avg_strategy_vs_benchmark_pct':0.2,'beat_benchmark_rate':0.40,'sufficient_trade_rate':0.70}\\n"
                "]).to_csv(stress/'regime_summary.csv', index=False)\\n"
                "pd.DataFrame([\\n"
                "{'symbol':'1','canonical_mode':'canonical_reduced_40','regime':'bear','strategy_vs_benchmark_pct':-1.5,'trade_count':2},\\n"
                "{'symbol':'000858','canonical_mode':'canonical_reduced_40','regime':'sideways','strategy_vs_benchmark_pct':-0.5,'trade_count':1}\\n"
                "]).to_csv(stress/'per_symbol_stress_results.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'1','canonical_mode':'canonical_reduced_40','warning_type':'underperformed_benchmark','message':'stress warning'}]).to_csv(stress/'stress_warnings.csv', index=False)\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline validation gate failure analysis",
        [
            "src/run_validation_gate_failure_analysis.py",
            "--gate-dir",
            "outputs/validation_gate_failure_analysis_smoke_inputs/gate",
            "--revalidation-dir",
            "outputs/validation_gate_failure_analysis_smoke_inputs/revalidation",
            "--stress-dir",
            "outputs/validation_gate_failure_analysis_smoke_inputs/stress",
            "--output-dir",
            "outputs/validation_gate_failure_analysis_smoke",
        ],
    ),
    (
        "offline validation gate failure analysis assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/validation_gate_failure_analysis_smoke'); "
                "required=['gate_failure_summary.csv','failure_by_check.csv','failure_by_candidate.csv','failure_by_symbol.csv','failure_by_regime.csv','risk_flag_summary.csv','remediation_plan.csv','validation_gate_failure_analysis_report.md','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "report=(base/'validation_gate_failure_analysis_report.md').read_text(encoding='utf-8'); "
                "phrases=['not trading-ready','canonical_reduced_40','stress validation','remediation','full','keep_core_only']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases; "
                "remediation=pd.read_csv(base/'remediation_plan.csv'); "
                "assert not remediation.empty; "
                "summary=pd.read_csv(base/'gate_failure_summary.csv'); "
                "assert not summary['trading_ready'].fillna(False).astype(bool).any(); "
                "regimes=pd.read_csv(base/'failure_by_regime.csv'); "
                "assert {'regime_gate_failed','has_regime_warnings'}.issubset(regimes.columns); "
                "bear=regimes[(regimes['canonical_mode']=='canonical_reduced_40') & (regimes['regime']=='bear')].iloc[0]; "
                "assert str(bear['regime_gate_failed']).lower() == 'false'; "
                "assert str(bear['has_regime_warnings']).lower() == 'true'; "
                "bull=regimes[(regimes['canonical_mode']=='canonical_reduced_40') & (regimes['regime']=='bull')].iloc[0]; "
                "assert str(bull['regime_gate_failed']).lower() == 'true'; "
                "sideways=regimes[(regimes['canonical_mode']=='canonical_reduced_40') & (regimes['regime']=='sideways')].iloc[0]; "
                "assert str(sideways['regime_gate_failed']).lower() == 'true'; "
                "symbols=pd.read_csv(base/'failure_by_symbol.csv', dtype={'symbol': str}); "
                "assert {'000001','000858'}.issubset(set(symbols['symbol'].dropna().astype(str)))"
            ),
        ],
    ),
    (
        "offline targeted remediation design",
        [
            "src/run_targeted_remediation_design.py",
            "--failure-analysis-dir",
            "outputs/validation_gate_failure_analysis_smoke",
            "--gate-dir",
            "outputs/validation_gate_failure_analysis_smoke_inputs/gate",
            "--revalidation-dir",
            "outputs/validation_gate_failure_analysis_smoke_inputs/revalidation",
            "--output-dir",
            "outputs/targeted_remediation_design_smoke",
        ],
    ),
    (
        "offline targeted remediation design assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/targeted_remediation_design_smoke'); "
                "required=['targeted_remediation_experiments.csv','regime_remediation_plan.csv','candidate_remediation_plan.csv','symbol_remediation_priority.csv','remediation_success_criteria.csv','targeted_remediation_design_report.md','warnings.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "report=(base/'targeted_remediation_design_report.md').read_text(encoding='utf-8'); "
                "phrases=['not trading-ready','canonical_reduced_40','bull','sideways','full remains baseline','keep_core_only','do not add new features or agents']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases; "
                "criteria=pd.read_csv(base/'remediation_success_criteria.csv'); "
                "criteria_text=' '.join(criteria.astype(str).agg(' '.join, axis=1).tolist()); "
                "required_criteria=['stress validation must pass','beat benchmark rate','sufficient trade rate','No trading_ready=True','at least 5 symbols tested']; "
                "missing_criteria=[phrase for phrase in required_criteria if phrase.lower() not in criteria_text.lower()]; "
                "assert not missing_criteria, missing_criteria; "
                "regimes=pd.read_csv(base/'regime_remediation_plan.csv'); "
                "bear=regimes[(regimes['canonical_mode']=='canonical_reduced_40') & (regimes['regime']=='bear')].iloc[0]; "
                "assert str(bear['regime_gate_failed']).lower() == 'false'; "
                "assert str(bear['has_regime_warnings']).lower() == 'true'; "
                "assert 'monitor' in str(bear['remediation_priority']).lower(); "
                "experiments=pd.read_csv(base/'targeted_remediation_experiments.csv'); "
                "assert {'threshold_grid_refinement','regime_specific_threshold_test','trade_count_sufficiency_test','benchmark_comparison_test','risk_flag_reduction','challenger_validation','baseline_monitoring'} & set(experiments['experiment_type']); "
                "assert not experiments.empty"
            ),
        ],
    ),
    (
        "offline synthetic bull remediation inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/bull_regime_threshold_remediation_smoke_inputs')\\n"
                "factors=base/'factors'\\n"
                "failure=base/'failure_analysis'\\n"
                "design=base/'targeted_design'\\n"
                "for d in [factors, failure, design]: d.mkdir(parents=True, exist_ok=True)\\n"
                "symbols=['000001','600519','000858','600036','601318']\\n"
                "dates=pd.date_range('2020-01-01', periods=130, freq='D')\\n"
                "for idx,symbol in enumerate(symbols):\\n"
                "    rows=[]\\n"
                "    for i,date in enumerate(dates):\\n"
                "        close=10+idx+i*0.04\\n"
                "        f_core=(i % 10)/10\\n"
                "        f_observe=((i+idx) % 7)/7\\n"
                "        rows.append({'date':date.strftime('%Y-%m-%d'),'symbol':symbol,'open':close*0.99,'high':close*1.01,'low':close*0.98,'close':close,'volume':1000+i,'f_core':f_core,'f_observe':f_observe,'label_up_5d':1 if (i % 6) in [0,1,2] else 0})\\n"
                "    pd.DataFrame(rows).to_csv(factors/f'factors_{symbol}.csv', index=False)\\n"
                "pd.DataFrame([{'feature':'f_core','recommendation':'keep_core'},{'feature':'f_observe','recommendation':'keep_observe'}]).to_csv(base/'recommendations.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','regime':'bull','regime_gate_failed':True,'has_regime_warnings':False}]).to_csv(failure/'failure_by_regime.csv', index=False)\\n"
                "pd.DataFrame([{'experiment_id':'TRD-001','target_regime':'bull'}]).to_csv(design/'targeted_remediation_experiments.csv', index=False)\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline bull regime threshold remediation",
        [
            "src/run_bull_regime_threshold_remediation.py",
            "--factor-dir",
            "outputs/bull_regime_threshold_remediation_smoke_inputs/factors",
            "--symbols",
            "000001,600519,000858,600036,601318",
            "--recommendations",
            "outputs/bull_regime_threshold_remediation_smoke_inputs/recommendations.csv",
            "--failure-analysis-dir",
            "outputs/bull_regime_threshold_remediation_smoke_inputs/failure_analysis",
            "--targeted-design-dir",
            "outputs/bull_regime_threshold_remediation_smoke_inputs/targeted_design",
            "--output-dir",
            "outputs/bull_regime_threshold_remediation_smoke",
            "--buy-thresholds",
            "0.45,0.50",
            "--sell-thresholds",
            "0.30,0.40",
            "--minimum-commission",
            "0",
            "--min-trades",
            "1",
        ],
    ),
    (
        "offline bull regime threshold remediation assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/bull_regime_threshold_remediation_smoke'); "
                "required=['bull_threshold_results.csv','bull_threshold_summary.csv','per_symbol_bull_results.csv','best_bull_thresholds.csv','bull_remediation_report.md','warnings.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "results=pd.read_csv(base/'bull_threshold_results.csv', dtype={'symbol': str}); "
                "assert not results.empty; "
                "assert set(results['regime']) == {'bull'}; "
                "assert set(results['canonical_mode']) == {'canonical_reduced_40'}; "
                "summary=pd.read_csv(base/'bull_threshold_summary.csv'); "
                "assert {'avg_strategy_vs_benchmark_pct','beat_benchmark_rate','sufficient_trade_rate','tested_symbol_count','bull_gate_passed'}.issubset(summary.columns); "
                "best=pd.read_csv(base/'best_bull_thresholds.csv'); "
                "assert 'selection_decision' in best.columns; "
                "report=(base/'bull_remediation_report.md').read_text(encoding='utf-8'); "
                "phrases=['not trading-ready','not financial advice','canonical_reduced_40 remains research-only','bull regime only','sideways remediation','Do not recommend adding features or agents']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "offline synthetic sideways remediation inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/sideways_regime_trade_sufficiency_remediation_smoke_inputs')\\n"
                "factors=base/'factors'\\n"
                "failure=base/'failure_analysis'\\n"
                "design=base/'targeted_design'\\n"
                "for d in [factors, failure, design]: d.mkdir(parents=True, exist_ok=True)\\n"
                "symbols=['000001','600519','000858','600036','601318']\\n"
                "dates=pd.date_range('2020-01-01', periods=130, freq='D')\\n"
                "for idx,symbol in enumerate(symbols):\\n"
                "    rows=[]\\n"
                "    for i,date in enumerate(dates):\\n"
                "        close=10+idx+((i % 5) * 0.001)\\n"
                "        label=1 if (i + idx) % 4 in [0, 1] else 0\\n"
                "        rows.append({'date':date.strftime('%Y-%m-%d'),'symbol':symbol,'open':close*0.999,'high':close*1.001,'low':close*0.998,'close':close,'volume':1000+i,'f_core':label + (i % 3) * 0.01,'f_observe':((i+idx) % 7)/7,'label_up_5d':label})\\n"
                "    pd.DataFrame(rows).to_csv(factors/f'factors_{symbol}.csv', index=False)\\n"
                "pd.DataFrame([{'feature':'f_core','recommendation':'keep_core'},{'feature':'f_observe','recommendation':'keep_observe'}]).to_csv(base/'recommendations.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','regime':'sideways','regime_gate_failed':True,'beat_benchmark_rate':0.40,'sufficient_trade_rate':0.40}]).to_csv(failure/'failure_by_regime.csv', index=False)\\n"
                "pd.DataFrame([{'experiment_id':'TRD-002','target_regime':'sideways','experiment_type':'trade_count_sufficiency_test'}]).to_csv(design/'targeted_remediation_experiments.csv', index=False)\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline sideways regime trade sufficiency remediation",
        [
            "src/run_sideways_regime_trade_sufficiency_remediation.py",
            "--factor-dir",
            "outputs/sideways_regime_trade_sufficiency_remediation_smoke_inputs/factors",
            "--symbols",
            "000001,600519,000858,600036,601318",
            "--recommendations",
            "outputs/sideways_regime_trade_sufficiency_remediation_smoke_inputs/recommendations.csv",
            "--failure-analysis-dir",
            "outputs/sideways_regime_trade_sufficiency_remediation_smoke_inputs/failure_analysis",
            "--targeted-design-dir",
            "outputs/sideways_regime_trade_sufficiency_remediation_smoke_inputs/targeted_design",
            "--output-dir",
            "outputs/sideways_regime_trade_sufficiency_remediation_smoke",
            "--buy-thresholds",
            "0.45,0.50",
            "--sell-thresholds",
            "0.40,0.45",
            "--minimum-commission",
            "0",
            "--commission-rate",
            "0",
            "--stamp-tax-rate",
            "0",
            "--slippage-pct",
            "0",
            "--min-trades",
            "50",
        ],
    ),
    (
        "offline sideways regime trade sufficiency remediation assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/sideways_regime_trade_sufficiency_remediation_smoke'); "
                "required=['sideways_trade_results.csv','sideways_trade_summary.csv','per_symbol_sideways_results.csv','best_sideways_thresholds.csv','sideways_remediation_report.md','warnings.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "results=pd.read_csv(base/'sideways_trade_results.csv', dtype={'symbol': str}); "
                "assert not results.empty; "
                "assert set(results['regime']) == {'sideways'}; "
                "assert set(results['canonical_mode']) == {'canonical_reduced_40'}; "
                "assert {'sufficient_trade','beat_benchmark','low_trade_count','negative_total_return','underperformed_benchmark'}.issubset(results.columns); "
                "summary=pd.read_csv(base/'sideways_trade_summary.csv'); "
                "assert {'avg_strategy_vs_benchmark_pct','beat_benchmark_rate','sufficient_trade_rate','tested_symbol_count','sideways_gate_passed','final_decision'}.issubset(summary.columns); "
                "assert not summary['sideways_gate_passed'].fillna(False).astype(bool).any(); "
                "assert set(summary['final_decision']) == {'sideways_remediation_failed'}; "
                "best=pd.read_csv(base/'best_sideways_thresholds.csv'); "
                "assert 'selection_decision' in best.columns; "
                "warnings=pd.read_csv(base/'warnings.csv', dtype={'symbol': str}); "
                "assert not warnings.empty; "
                "assert 'low_trade_count' in set(warnings['warning_type']); "
                "report=(base/'sideways_remediation_report.md').read_text(encoding='utf-8'); "
                "phrases=['not trading-ready','not financial advice','canonical_reduced_40 remains research-only','sideways regime only','Sideways remediation failed','Trade sufficiency','benchmark beat']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases; "
                "banned=['trading_ready=True','Strategy is profitable','deployable','trading-ready candidate']; "
                "assert not [phrase for phrase in banned if phrase in report]"
            ),
        ],
    ),
    (
        "offline synthetic integrated remediation inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/integrated_remediation_revalidation_smoke_inputs')\\n"
                "bull=base/'bull'\\n"
                "sideways=base/'sideways'\\n"
                "gate=base/'gate'\\n"
                "failure=base/'failure'\\n"
                "for d in [bull, sideways, gate, failure]: d.mkdir(parents=True, exist_ok=True)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.65,'sell_threshold':0.50,'regime':'bull','avg_total_return_pct':6.3,'avg_benchmark_return_pct':6.4,'avg_strategy_vs_benchmark_pct':-0.1,'beat_benchmark_rate':0.60,'sufficient_trade_rate':0.80,'tested_symbol_count':5,'bull_gate_passed':False,'final_decision':'bull_remediation_failed'}]).to_csv(bull/'bull_threshold_summary.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.65,'sell_threshold':0.50,'regime':'bull','avg_total_return_pct':6.3,'avg_benchmark_return_pct':6.4,'avg_strategy_vs_benchmark_pct':-0.1,'beat_benchmark_rate':0.60,'sufficient_trade_rate':0.80,'tested_symbol_count':5,'bull_gate_passed':False,'final_decision':'bull_remediation_failed','selected':False}]).to_csv(bull/'best_bull_thresholds.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000001','canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.65,'sell_threshold':0.50,'regime':'bull','total_return_pct':1.0,'benchmark_return_pct':2.0,'strategy_vs_benchmark_pct':-1.0,'trade_count':2,'warning':'low_trade_count: 2 | underperformed_benchmark'}]).to_csv(bull/'per_symbol_bull_results.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000001','canonical_mode':'canonical_reduced_40','regime':'bull','warning_type':'underperformed_benchmark','message':'underperformed benchmark'}]).to_csv(bull/'warnings.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.55,'sell_threshold':0.50,'regime':'sideways','avg_total_return_pct':3.0,'avg_benchmark_return_pct':-1.0,'avg_strategy_vs_benchmark_pct':4.0,'beat_benchmark_rate':0.60,'sufficient_trade_rate':0.80,'tested_symbol_count':5,'sideways_gate_passed':True,'final_decision':'sideways_remediation_passed'}]).to_csv(sideways/'sideways_trade_summary.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.55,'sell_threshold':0.50,'regime':'sideways','avg_total_return_pct':3.0,'avg_benchmark_return_pct':-1.0,'avg_strategy_vs_benchmark_pct':4.0,'beat_benchmark_rate':0.60,'sufficient_trade_rate':0.80,'tested_symbol_count':5,'sideways_gate_passed':True,'final_decision':'sideways_remediation_passed','selected':True}]).to_csv(sideways/'best_sideways_thresholds.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000001','canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.55,'sell_threshold':0.50,'regime':'sideways','total_return_pct':-0.5,'benchmark_return_pct':3.0,'strategy_vs_benchmark_pct':-3.5,'trade_count':3,'beat_benchmark':False,'sufficient_trade':True,'warning':'negative_total_return | underperformed_benchmark'}]).to_csv(sideways/'per_symbol_sideways_results.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000001','canonical_mode':'canonical_reduced_40','regime':'sideways','warning_type':'negative_total_return','message':'negative total return'}]).to_csv(sideways/'warnings.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','trading_ready':False,'final_gate_decision':'research_only_not_trading_ready'}]).to_csv(gate/'validation_gate_results.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','check_name':'stress_validation_passed','severity':'blocking','message':'stress failed'},{'canonical_mode':'canonical_reduced_40','check_name':'validation_excess_positive','severity':'blocking','message':'validation excess failed'},{'canonical_mode':'full','check_name':'role_allowed','severity':'blocking','message':'baseline role blocked'}]).to_csv(gate/'validation_gate_failures.csv', index=False)\\n"
                "(gate/'candidate_validation_gate_report.md').write_text('No candidate is trading-ready.', encoding='utf-8')\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','regime':'bull','regime_gate_failed':True}]).to_csv(failure/'failure_by_regime.csv', index=False)\\n"
                "(failure/'validation_gate_failure_analysis_report.md').write_text('Bull and sideways diagnostics.', encoding='utf-8')\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline integrated remediation revalidation",
        [
            "src/run_integrated_remediation_revalidation.py",
            "--bull-dir",
            "outputs/integrated_remediation_revalidation_smoke_inputs/bull",
            "--sideways-dir",
            "outputs/integrated_remediation_revalidation_smoke_inputs/sideways",
            "--validation-gate-dir",
            "outputs/integrated_remediation_revalidation_smoke_inputs/gate",
            "--failure-analysis-dir",
            "outputs/integrated_remediation_revalidation_smoke_inputs/failure",
            "--output-dir",
            "outputs/integrated_remediation_revalidation_smoke",
        ],
    ),
    (
        "offline integrated remediation revalidation assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/integrated_remediation_revalidation_smoke'); "
                "required=['integrated_remediation_revalidation_report.md','integrated_remediation_summary.csv','regime_remediation_status.csv','per_symbol_remediation_risk.csv','integrated_gate_results.csv','integrated_risk_flags.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "summary=pd.read_csv(base/'integrated_remediation_summary.csv'); "
                "row=summary.iloc[0]; "
                "assert row['canonical_mode']=='canonical_reduced_40'; "
                "assert row['overall_decision']=='research_only_not_trading_ready'; "
                "assert not bool(row['trading_ready']); "
                "assert row['main_blocker']=='bull_remediation_failed'; "
                "regimes=pd.read_csv(base/'regime_remediation_status.csv'); "
                "decisions=dict(zip(regimes['regime'], regimes['final_decision'])); "
                "assert decisions['bull']=='bull_remediation_failed'; "
                "assert decisions['sideways']=='sideways_remediation_passed'; "
                "gates=pd.read_csv(base/'integrated_gate_results.csv'); "
                "assert not gates['trading_ready'].fillna(False).astype(bool).any(); "
                "gate_decisions=dict(zip(gates['canonical_mode'], gates['gate_decision'])); "
                "assert gate_decisions['canonical_reduced_40']=='research_only_not_trading_ready'; "
                "risk=pd.read_csv(base/'per_symbol_remediation_risk.csv', dtype={'symbol': str}); "
                "assert '000001' in set(risk['symbol']); "
                "flags=pd.read_csv(base/'integrated_risk_flags.csv', dtype={'symbol': str}); "
                "assert {'bull_remediation_failed','not_trading_ready'} <= set(flags['risk_type']); "
                "risk_types=flags['risk_type'].fillna('').astype(str); "
                "assert not risk_types.str.endswith('_passed').any(); "
                "assert not risk_types.str.endswith('_positive').any(); "
                "assert 'role_allowed' not in set(risk_types); "
                "report=(base/'integrated_remediation_revalidation_report.md').read_text(encoding='utf-8'); "
                "phrases=['No candidate is trading-ready','canonical_reduced_40 remains research-only','Bull remediation failed','Sideways aggregate remediation passed','V4 Step 37']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "offline synthetic bull failure drilldown inputs",
        [
            "-c",
            (
                "exec("
                "\"from pathlib import Path\\n"
                "import pandas as pd\\n"
                "base=Path('outputs/bull_regime_failure_drilldown_smoke_inputs')\\n"
                "bull=base/'bull'\\n"
                "integrated=base/'integrated'\\n"
                "for d in [bull, integrated]: d.mkdir(parents=True, exist_ok=True)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.65,'sell_threshold':0.50,'regime':'bull','avg_total_return_pct':5.0,'avg_benchmark_return_pct':5.2,'avg_strategy_vs_benchmark_pct':-0.2,'beat_benchmark_rate':0.5,'sufficient_trade_rate':0.5,'tested_symbol_count':2,'bull_gate_passed':False,'final_decision':'bull_remediation_failed'}]).to_csv(bull/'bull_threshold_summary.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.65,'sell_threshold':0.50,'regime':'bull','avg_total_return_pct':5.0,'avg_benchmark_return_pct':5.2,'avg_strategy_vs_benchmark_pct':-0.2,'beat_benchmark_rate':0.5,'sufficient_trade_rate':0.5,'tested_symbol_count':2,'bull_gate_passed':False,'final_decision':'bull_remediation_failed','selected':False}]).to_csv(bull/'best_bull_thresholds.csv', index=False)\\n"
                "rows=[{'symbol':'000001','canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.65,'sell_threshold':0.50,'regime':'bull','total_return_pct':2.0,'benchmark_return_pct':6.0,'strategy_vs_benchmark_pct':-4.0,'trade_count':2,'warning':'low_trade_count: 2 | underperformed_benchmark'},{'symbol':'000858','canonical_mode':'canonical_reduced_40','model_type':'logistic_regression','buy_threshold':0.65,'sell_threshold':0.50,'regime':'bull','total_return_pct':8.0,'benchmark_return_pct':4.0,'strategy_vs_benchmark_pct':4.0,'trade_count':3,'warning':''}]\\n"
                "pd.DataFrame(rows).to_csv(bull/'bull_threshold_results.csv', index=False)\\n"
                "pd.DataFrame(rows).to_csv(bull/'per_symbol_bull_results.csv', index=False)\\n"
                "pd.DataFrame([{'symbol':'000001','canonical_mode':'canonical_reduced_40','regime':'bull','warning_type':'low_trade_count','message':'low_trade_count: 2'}]).to_csv(bull/'warnings.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','overall_decision':'research_only_not_trading_ready','trading_ready':False,'main_blocker':'bull_remediation_failed'}]).to_csv(integrated/'integrated_remediation_summary.csv', index=False)\\n"
                "pd.DataFrame([{'regime':'bull','final_decision':'bull_remediation_failed'}]).to_csv(integrated/'regime_remediation_status.csv', index=False)\\n"
                "pd.DataFrame([{'canonical_mode':'canonical_reduced_40','trading_ready':False,'gate_decision':'research_only_not_trading_ready'}]).to_csv(integrated/'integrated_gate_results.csv', index=False)\\n"
                "pd.DataFrame([{'source':'smoke','canonical_mode':'canonical_reduced_40','regime':'bull','symbol':'000001','risk_type':'low_trade_count','severity':'medium','message':'low trade'}]).to_csv(integrated/'integrated_risk_flags.csv', index=False)\\n"
                "(integrated/'integrated_remediation_revalidation_report.md').write_text('No candidate is trading-ready.', encoding='utf-8')\\n"
                "\")"
            ),
        ],
    ),
    (
        "offline bull regime failure drilldown",
        [
            "src/run_bull_regime_failure_drilldown.py",
            "--bull-dir",
            "outputs/bull_regime_failure_drilldown_smoke_inputs/bull",
            "--integrated-dir",
            "outputs/bull_regime_failure_drilldown_smoke_inputs/integrated",
            "--output-dir",
            "outputs/bull_regime_failure_drilldown_smoke",
        ],
    ),
    (
        "offline bull regime failure drilldown assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import pandas as pd; "
                "base=Path('outputs/bull_regime_failure_drilldown_smoke'); "
                "required=['bull_regime_failure_drilldown_report.md','bull_symbol_failure_summary.csv','bull_failure_contribution.csv','bull_failure_reasons.csv','bull_threshold_context.csv','bull_drilldown_limitations.csv','bull_trade_level_diagnostics.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "context=pd.read_csv(base/'bull_threshold_context.csv'); "
                "assert set(context['threshold_action']) == {'reused_for_diagnosis_only'}; "
                "symbols=pd.read_csv(base/'bull_symbol_failure_summary.csv', dtype={'symbol': str}); "
                "assert '000001' in set(symbols['symbol']); "
                "assert 'trading_ready' not in symbols.columns; "
                "reasons=pd.read_csv(base/'bull_failure_reasons.csv', dtype={'symbol': str}); "
                "assert not reasons.empty; "
                "assert 'trade_level_data_unavailable' in set(reasons['reason_type']); "
                "limitations=pd.read_csv(base/'bull_drilldown_limitations.csv'); "
                "assert 'trade_level_data_may_be_unavailable' in set(limitations['limitation_type']); "
                "trade=pd.read_csv(base/'bull_trade_level_diagnostics.csv'); "
                "assert set(trade['diagnostic_status']) == {'trade_level_data_unavailable'}; "
                "report=(base/'bull_regime_failure_drilldown_report.md').read_text(encoding='utf-8'); "
                "phrases=['No candidate is trading-ready','does not tune thresholds','canonical_reduced_40 remains research-only','Bull remediation failed']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "offline bull trade window diagnostics",
        [
            "src/run_bull_trade_window_diagnostics.py",
            "--bull-dir",
            "outputs/bull_regime_threshold_remediation_smoke",
            "--output-dir",
            "outputs/bull_trade_window_diagnostics_smoke",
            "--buy-threshold",
            "0.65",
            "--sell-threshold",
            "0.50",
        ],
    ),
    (
        "offline bull trade window diagnostics assertions",
        [
            "-c",
            (
                "from pathlib import Path; "
                "import json; "
                "import pandas as pd; "
                "base=Path('outputs/bull_trade_window_diagnostics_smoke'); "
                "required=['bull_trade_window_diagnostics_report.md','bull_trade_level_diagnostics.csv','bull_signal_timeline_diagnostics.csv','bull_window_diagnostics.csv','bull_symbol_window_summary.csv','bull_error_pattern_summary.csv','bull_diagnostics_data_availability.csv','bull_diagnostics_limitations.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing; "
                "config=json.loads((base/'run_config.json').read_text(encoding='utf-8')); "
                "assert abs(config['buy_threshold'] - 0.65) < 1e-12; "
                "assert abs(config['sell_threshold'] - 0.50) < 1e-12; "
                "assert config['threshold_action']=='reused_for_diagnosis_only'; "
                "availability=pd.read_csv(base/'bull_diagnostics_data_availability.csv'); "
                "assert {'symbol_level','trade_level','date_level_timeline','window_level'} <= set(availability['diagnostic_layer']); "
                "summary=pd.read_csv(base/'bull_symbol_window_summary.csv', dtype={'symbol': str}); "
                "assert summary.empty or summary['symbol'].astype(str).str.len().ge(6).all(); "
                "checked=['bull_symbol_window_summary.csv','bull_error_pattern_summary.csv','bull_diagnostics_data_availability.csv','bull_diagnostics_limitations.csv']; "
                "bad=[name for name in checked if 'trading_ready' in pd.read_csv(base/name).columns]; "
                "assert not bad, bad; "
                "trade=pd.read_csv(base/'bull_trade_level_diagnostics.csv', dtype={'symbol': str}); "
                "assert (not trade.empty) or ('data_status' in trade.columns); "
                "report=(base/'bull_trade_window_diagnostics_report.md').read_text(encoding='utf-8'); "
                "phrases=['diagnostics-output enhancement only','0.65 / 0.50','does not tune thresholds','does not upgrade any candidate']; "
                "missing_phrases=[phrase for phrase in phrases if phrase not in report]; "
                "assert not missing_phrases, missing_phrases"
            ),
        ],
    ),
    (
        "offline baseline model train",
        [
            "src/train_baseline_model.py",
            "--dataset-dir",
            "data/ml/smoke_000001",
            "--target-col",
            "label_up_5d",
            "--model",
            "random_forest",
            "--output-dir",
            "models/smoke_000001",
        ],
    ),
    (
        "offline model prediction",
        [
            "src/predict_with_model.py",
            "--model-path",
            "models/smoke_000001/random_forest.joblib",
            "--input",
            "data/factors/smoke_factors_000001.csv",
            "--top-n",
            "5",
        ],
    ),
    (
        "offline model evaluation",
        [
            "src/evaluate_model.py",
            "--model-dir",
            "models/smoke_000001",
            "--target-col",
            "label_up_5d",
            "--signal-threshold",
            "0.6",
        ],
    ),
    (
        "offline ML signal backtest",
        [
            "src/run_ml_signal_backtest.py",
            "--model-dir",
            "models/smoke_000001",
            "--factor-csv",
            "data/factors/smoke_factors_000001.csv",
            "--initial-cash",
            "10000",
            "--buy-threshold",
            "0.60",
            "--sell-threshold",
            "0.50",
        ],
    ),
    (
        "offline ML threshold experiment",
        [
            "src/run_ml_threshold_experiment.py",
            "--model-dir",
            "models/smoke_000001",
            "--input",
            "data/factors/smoke_factors_000001.csv",
            "--buy-thresholds",
            "0.55,0.60",
            "--sell-thresholds",
            "0.40,0.50",
            "--initial-cash",
            "10000",
        ],
    ),
    (
        "offline batch model training",
        [
            "src/run_batch_model_training.py",
            "--symbols",
            "000001,600519",
            "--source",
            "demo",
            "--start",
            "20240101",
            "--end",
            "20241231",
            "--output-dir",
            "outputs/model_robustness_smoke",
            "--models",
            "logistic_regression,random_forest",
        ],
    ),
    (
        "offline batch model output files",
        [
            "-c",
            (
                "from pathlib import Path; "
                "base=Path('outputs/model_robustness_smoke'); "
                "required=['training_results.csv','model_summary.csv',"
                "'model_ranking.csv','warnings.csv','run_config.json']; "
                "missing=[name for name in required if not (base/name).exists()]; "
                "assert not missing, missing"
            ),
        ],
    ),
    (
        "offline model robustness report",
        [
            "src/generate_model_report.py",
            "--input-dir",
            "outputs/model_robustness_smoke",
            "--output",
            "reports/model_robustness_smoke.md",
        ],
    ),
    (
        "offline model robustness report file",
        [
            "-c",
            (
                "from pathlib import Path; "
                "path=Path('reports/model_robustness_smoke.md'); "
                "assert path.exists(); "
                "text=path.read_text(encoding='utf-8'); "
                "assert '# Model Robustness Research Report' in text"
            ),
        ],
    ),
    ("run_stock_backtest help", ["src/run_stock_backtest.py", "--help"]),
    ("run_batch_experiment help", ["src/run_batch_experiment.py", "--help"]),
    ("run_period_experiment help", ["src/run_period_experiment.py", "--help"]),
    ("analyze_period_results help", ["src/analyze_period_results.py", "--help"]),
    (
        "next_open cost backtest",
        [
            "-c",
            (
                "import pandas as pd; "
                "from src.backtester import run_long_only_backtest_with_trades; "
                "df=pd.DataFrame({'date':['2024-01-01','2024-01-02','2024-01-03'],"
                "'open':[10,11,12],'close':[10,11,12],'signal':[1,0,-1]}); "
                "bt,tr=run_long_only_backtest_with_trades(df, execution_mode='next_open', "
                "commission_rate=0.001, stamp_tax_rate=0.001, slippage_pct=0.1, "
                "min_commission=1.0); "
                "assert 'total_transaction_cost' in bt.columns"
            ),
        ],
    ),
]


def run_command(label: str, command: list[str]) -> bool:
    full_command = [sys.executable, *command]
    result = subprocess.run(
        full_command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        print(f"PASS: {label}")
        return True

    print(f"FAIL: {label}")
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    return False


def run_py_compile_checks() -> bool:
    all_passed = True

    for file_path in PY_COMPILE_FILES:
        path = Path(file_path)
        if not path.exists():
            continue

        passed = run_command(
            f"py_compile {file_path}",
            ["-m", "py_compile", file_path],
        )
        all_passed = all_passed and passed

    return all_passed


def run_command_checks() -> bool:
    all_passed = True

    for label, command in COMMAND_CHECKS:
        script_path = Path(command[0])
        if command[0] != "-c" and not script_path.exists():
            continue

        passed = run_command(label, command)
        all_passed = all_passed and passed

    return all_passed


def main() -> None:
    print("QuantPilot-AI Smoke Tests")
    print("-------------------------")

    compile_passed = run_py_compile_checks()
    command_passed = run_command_checks()

    print()
    if compile_passed and command_passed:
        print("All smoke tests passed.")
        sys.exit(0)

    print("Smoke tests failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
