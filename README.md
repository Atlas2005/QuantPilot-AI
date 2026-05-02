# QuantPilot-AI

QuantPilot-AI is an educational rule-based A-share backtesting project.

It is designed to help beginners understand how market data, technical
indicators, trading signals, backtesting, risk controls, performance metrics,
and experiment summaries can fit together in one explainable workflow.

The project supports real A-share daily data through AkShare and Baostock. It
can run single-stock backtests, parameter experiments, multi-stock experiments,
multi-period experiments, and exported-result analysis.

This is still a simple strategy research tool. It is not a finished trading
product and does not try to guarantee profitable results.

## Installation

Use Python 3.10 or newer from the project root.

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Command Quick Reference

Run these commands from the project root in Windows PowerShell.

| Task | Command |
| --- | --- |
| Check environment | `python src/check_setup.py` |
| Run offline smoke tests | `python src/run_smoke_tests.py` |
| Run offline demo | `python src/run_demo.py` |
| Fetch real A-share data | `python src/real_data_loader.py --symbol 000001 --source baostock --start 20240101 --end 20241231` |
| Run single-stock backtest | `python src/run_stock_backtest.py --symbol 000001 --source baostock --start 20240101 --end 20241231` |
| Run single-stock backtest with risk controls | `python src/run_stock_backtest.py --symbol 000001 --source baostock --start 20240101 --end 20241231 --stop-loss-pct 3 --take-profit-pct 10 --max-holding-days 30` |
| Run multi-stock experiment | `python src/run_batch_experiment.py --symbols 000001,600519,000858,600036,601318 --source baostock --start 20240101 --end 20241231 --compact` |
| Run multi-period experiment | `python src/run_period_experiment.py --symbols 000001,600519,000858,600036,601318 --source baostock --periods 2021,2022,2023,2024,2025 --compact` |
| Analyze exported results | `python src/analyze_period_results.py --input reports/period_2021_2025.csv --output-dir reports` |

Export multi-period results:

```powershell
New-Item -ItemType Directory -Force reports
python src/run_period_experiment.py --symbols 000001,600519,000858,600036,601318 --source baostock --periods 2021,2022,2023,2024,2025 --compact --output reports/period_2021_2025.csv
```

## Project Structure

- `README.md`: Beginner-friendly project guide and command reference.
- `requirements.txt`: Python runtime dependencies used by the project.
- `PROJECT_CONTEXT.md`: Development context and project rules for V2 work.
- `data/sample/`: Tracked sample and demo CSV data for offline testing.
- `data/real/`: Local folder for fetched real A-share data. Contents may vary by machine.
- `reports/`: Generated CSV summaries and chart outputs. This folder is ignored by Git.
- `src/check_setup.py`: Checks that required Python packages are installed.
- `src/run_smoke_tests.py`: Runs offline compile, setup, demo, and help checks.
- `src/run_demo.py`: Runs the offline demo without Baostock, AkShare, or internet access.
- `src/real_data_loader.py`: Fetches real A-share daily data from AkShare or Baostock.
- `src/run_stock_backtest.py`: Runs one real-data stock backtest with optional risk controls.
- `src/run_batch_experiment.py`: Compares risk-control scenarios across multiple stocks.
- `src/run_period_experiment.py`: Compares scenarios across multiple stocks and years.
- `src/analyze_period_results.py`: Reads exported period results and creates summaries/charts.
- `src/indicators.py`: Calculates MA5, MA20, RSI, and CCI indicators.
- `src/strategy.py`: Generates MA crossover buy/sell/no-action signals.
- `src/backtester.py`: Runs the long-only backtest and trade-log workflow.
- `src/metrics.py`: Calculates portfolio-level performance metrics.
- `src/trade_metrics.py`: Summarizes trade-log statistics.
- `src/report.py`: Reserved/simple report module name; current rule-based report code lives in `src/report_generator.py`.

## Real Data Loader

Fetch and save standardized A-share daily data:

```powershell
python src/real_data_loader.py --symbol 000001 --source baostock --start 20240101 --end 20241231
```

Saved CSV files use the V1-compatible columns:

```text
date, open, high, low, close, volume
```

## Offline Demo

Run the beginner-friendly offline demo:

```powershell
python src/run_demo.py
```

This mode uses `data/sample/demo_000001.csv` and does not use Baostock,
AkShare, or the internet. The demo data is fake sample data for checking that
the project workflow runs correctly. It should not be used for real market
conclusions.

## Smoke Tests

Run the offline smoke tests before committing changes or after pulling new code:

```powershell
python src/run_smoke_tests.py
```

Smoke tests compile the main scripts, check dependencies, run the offline demo,
and run `--help` checks for command-line tools. They do not fetch real market
data and do not call Baostock or AkShare data download commands.

## Single-Stock Backtest

Fetch real data, save it, and run the full backtest workflow:

```powershell
python src/run_stock_backtest.py --symbol 000001 --source baostock --start 20240101 --end 20241231
```

The workflow:

1. Fetches daily A-share data.
2. Adds MA5, MA20, RSI, and CCI indicators.
3. Generates MA crossover signals.
4. Runs a long-only backtest.
5. Prints performance metrics, a rule-based report, the last 10 backtest rows,
   the trade log, and trade metrics.

## Single-Stock Backtest With Risk Controls

Optional risk controls can close a trade before the MA crossover sell signal:

```powershell
python src/run_stock_backtest.py --symbol 000001 --source baostock --start 20240101 --end 20241231 --stop-loss-pct 3 --take-profit-pct 10 --max-holding-days 30
```

The risk controls use close prices:

- `--stop-loss-pct 3` exits when the current trade return is `-3%` or worse.
- `--take-profit-pct 10` exits when the current trade return is `+10%` or better.
- `--max-holding-days 30` exits after holding for 30 calendar days.

## Multi-Stock Batch Experiment

Compare the default risk-control scenarios across multiple stocks:

```powershell
python src/run_batch_experiment.py --symbols 000001,600519,000858,600036,601318 --source baostock --start 20240101 --end 20241231 --compact
```

The batch runner fetches each symbol once, prepares indicators and strategy
signals once, then runs all scenarios in memory.

## Multi-Period Experiment

Test the same scenarios across multiple symbols and multiple years:

```powershell
python src/run_period_experiment.py --symbols 000001,600519,000858,600036,601318 --source baostock --periods 2021,2022,2023,2024,2025 --compact
```

Each period is converted into a full-year range. For example, `2024` becomes
`20240101` to `20241231`.

## Exporting CSV Results

Create a reports folder and export raw period experiment results:

```powershell
New-Item -ItemType Directory -Force reports
python src/run_period_experiment.py --symbols 000001,600519,000858,600036,601318 --source baostock --periods 2021,2022,2023,2024,2025 --compact --output reports/period_2021_2025.csv
```

The exported CSV is raw experiment data. It is useful for later analysis and
should not be committed.

## Analyzing Exported Results

Analyze an exported period experiment CSV:

```powershell
python src/analyze_period_results.py --input reports/period_2021_2025.csv --output-dir reports
```

This creates:

- `scenario_summary.csv`
- `scenario_ranking.csv`
- `avg_total_return_pct.png`
- `avg_max_drawdown_pct.png`
- `avg_profit_factor.png`

The analysis script ranks scenarios with a simple educational score:

```text
score = avg_total_return_pct + avg_profit_factor * 2 + avg_max_drawdown_pct * 0.3
```

This score is only a simple comparison helper. It is not financial advice.

## Key Concepts

`total_return_pct`:
The total percentage change in portfolio value over the backtest.

`max_drawdown_pct`:
The largest portfolio drop from a previous high. Drawdown values are usually
negative, so `-10%` is better than `-20%`.

`profit_factor`:
Gross profit divided by gross loss for closed trades. Higher is usually better,
but it should be interpreted together with drawdown, trade count, and return.

`win_rate_pct`:
The percentage of closed trades that made a profit.

Risk controls:
Optional stop-loss, take-profit, and maximum-holding-day rules that can close a
trade before the normal strategy sell signal.

Compact mode:
Use `--compact` on batch and period experiments to print a narrower console
table while still keeping summaries, rankings, and quick interpretation.

## Data Source Notes

Baostock may be more stable for free historical daily data.

AkShare can be useful, but some interfaces may be unstable depending on network
or source availability.

Free data sources can fail temporarily. If a request fails, retry later instead
of repeatedly hammering the source.

## Git Hygiene

Generated files should not be committed.

The project `.gitignore` ignores generated reports and cache files, including:

- `reports/`
- Python cache files
- Matplotlib cache files
- local editor/system files

## Educational Disclaimer

This project is for programming, data analysis, and educational research.

It is not financial advice.

Backtest results do not guarantee future returns.
