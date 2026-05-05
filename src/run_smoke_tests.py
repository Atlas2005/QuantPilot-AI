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
