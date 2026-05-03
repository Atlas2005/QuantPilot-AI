import subprocess
import sys
from pathlib import Path


PY_COMPILE_FILES = [
    "src/check_setup.py",
    "src/run_demo.py",
    "src/real_data_loader.py",
    "src/factor_builder.py",
    "src/build_factor_dataset.py",
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
