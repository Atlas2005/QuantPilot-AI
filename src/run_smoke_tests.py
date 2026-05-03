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
