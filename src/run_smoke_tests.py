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
                "assert not missing, missing"
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
