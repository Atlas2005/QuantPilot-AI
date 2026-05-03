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
    ("offline demo", ["src/run_demo.py"]),
    ("real_data_loader help", ["src/real_data_loader.py", "--help"]),
    ("build_factor_dataset help", ["src/build_factor_dataset.py", "--help"]),
    ("split_factor_dataset help", ["src/split_factor_dataset.py", "--help"]),
    ("train_baseline_model help", ["src/train_baseline_model.py", "--help"]),
    ("predict_with_model help", ["src/predict_with_model.py", "--help"]),
    ("evaluate_model help", ["src/evaluate_model.py", "--help"]),
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
