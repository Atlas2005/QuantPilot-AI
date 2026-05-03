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
| Run dashboard | `streamlit run app.py` |
| Fetch real A-share data | `python src/real_data_loader.py --symbol 000001 --source baostock --start 20240101 --end 20241231` |
| Build factor dataset | `python src/build_factor_dataset.py --source csv --input data/real/000001.csv --symbol 000001 --output data/factors/factors_000001.csv` |
| Split ML dataset | `python src/split_factor_dataset.py --input data/factors/factors_000001.csv --output-dir data/ml/demo_000001 --target-col label_up_5d --purge-rows 5 --split-mode global_date` |
| Train baseline ML model | `python src/train_baseline_model.py --dataset-dir data/ml/demo_000001 --target-col label_up_5d --model random_forest --output-dir models/demo_000001` |
| Predict with baseline model | `python src/predict_with_model.py --model-path models/demo_000001/random_forest.joblib --input data/factors/factors_000001.csv` |
| Evaluate model quality | `python src/evaluate_model.py --model-dir models/demo_000001 --target-col label_up_5d` |
| Run ML signal backtest | `python src/run_ml_signal_backtest.py --model-dir models/demo_000001 --factor-csv data/factors/factors_000001.csv --initial-cash 10000 --buy-threshold 0.60 --sell-threshold 0.50` |
| Run ML threshold experiment | `python src/run_ml_threshold_experiment.py --model-dir models/demo_000001 --input data/factors/factors_000001.csv --output outputs/ml_threshold_experiment.csv` |
| Run model robustness training | `python src/run_batch_model_training.py --symbols 000001,600519 --source demo --start 20240101 --end 20241231 --output-dir outputs/model_robustness_demo --models logistic_regression,random_forest` |
| Generate robustness report | `python src/generate_model_report.py --input-dir outputs/model_robustness_demo --output reports/model_robustness_demo.md` |
| Show feature source roadmap | `python src/show_feature_sources.py --list` |
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
- `app.py`: Offline Streamlit dashboard for the demo workflow.
- `data/sample/`: Tracked sample and demo CSV data for offline testing.
- `data/real/`: Local folder for fetched real A-share data. Contents may vary by machine.
- `data/factors/`: Generated factor datasets for future ML research. This folder is ignored by Git.
- `data/ml/`: Generated chronological train/validation/test datasets. This folder is ignored by Git.
- `reports/`: Generated CSV summaries and chart outputs. This folder is ignored by Git.
- `src/check_setup.py`: Checks that required Python packages are installed.
- `src/run_smoke_tests.py`: Runs offline compile, setup, demo, and help checks.
- `src/run_demo.py`: Runs the offline demo without Baostock, AkShare, or internet access.
- `src/real_data_loader.py`: Fetches real A-share daily data from AkShare or Baostock.
- `src/factor_builder.py`: Builds feature-and-label factor datasets from OHLCV data.
- `src/build_factor_dataset.py`: Command-line tool for creating factor CSV files.
- `src/dataset_splitter.py`: Splits factor datasets into chronological ML datasets.
- `src/split_factor_dataset.py`: Command-line tool for ML dataset splitting and leakage checks.
- `src/model_trainer.py`: Trains and evaluates baseline supervised ML models.
- `src/train_baseline_model.py`: Command-line tool for baseline ML model training.
- `src/model_predictor.py`: Loads trained model artifacts and predicts the latest factor row.
- `src/predict_with_model.py`: Command-line tool for model prediction.
- `src/model_evaluator.py`: Evaluates prediction quality and leakage warnings.
- `src/evaluate_model.py`: Command-line tool for model evaluation diagnostics.
- `src/ml_signal_backtester.py`: Converts ML probabilities into long/flat backtest signals.
- `src/run_ml_signal_backtest.py`: Command-line tool for ML signal backtests.
- `src/ml_threshold_experiment.py`: Runs ML threshold and walk-forward research experiments.
- `src/run_ml_threshold_experiment.py`: Command-line tool for ML threshold experiments.
- `src/batch_model_trainer.py`: Trains and compares baseline models across symbols.
- `src/run_batch_model_training.py`: Command-line tool for model robustness training.
- `src/model_report_generator.py`: Converts robustness outputs into a Markdown research report.
- `src/generate_model_report.py`: Command-line tool for model robustness report export.
- `src/feature_source_registry.py`: Roadmap registry for future multi-factor feature sources.
- `src/show_feature_sources.py`: Command-line tool for viewing or exporting the feature registry.
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

## Dashboard

Run the first Streamlit dashboard:

```powershell
streamlit run app.py
```

The dashboard can use three offline input modes:

- Demo data from `data/sample/demo_000001.csv`
- Local real CSV files from `data/real/`
- Uploaded CSV files from your browser

The dashboard itself does not fetch real market data and does not call Baostock
or AkShare. To create a local real CSV first, run:

```powershell
python src/run_stock_backtest.py --symbol 000001 --source baostock --start 20240101 --end 20241231
```

The dashboard is for workflow demonstration and education, not trading advice.
It shows key metric cards, price and moving-average charts with buy/sell
markers, a portfolio equity curve, a drawdown curve, performance summary,
trade log, and trade metrics.

## Factor Dataset Builder

The factor dataset builder prepares clean feature-and-label CSV files for
future machine-learning research. This step does not train a model and does not
change the trading strategy or backtester.

The builder starts from standardized OHLCV data with these columns:

```text
date, open, high, low, close, volume
```

It creates identifier columns, raw market columns, price/return features, trend
features, volatility/risk features, technical indicators, and future-return
labels. Feature columns use only current and past data. Label columns such as
`future_return_5d` and `label_up_5d` intentionally use future returns and must
not be used as model input features.

The structure also reserves future feature groups for valuation, fund flow,
institutional holdings, dividend, sentiment, news, and industry data. These
external sources can be merged later without adding paid APIs in this step.

Offline CSV example:

```powershell
python src/build_factor_dataset.py --source csv --input data/real/000001.csv --symbol 000001 --output data/factors/factors_000001.csv
```

Real-data example:

```powershell
python src/build_factor_dataset.py --source baostock --symbol 000001 --start 20240101 --end 20241231 --output data/factors/factors_000001.csv
```

The generated CSV is research data for future model training. It is not a
trading recommendation.

## V4 Step 3: ML Dataset Split and Leakage Check

V4 Step 3 prepares safe train/validation/test CSV files from a factor dataset.
The split is chronological and never shuffles rows.

Create a demo factor dataset:

```powershell
python src/build_factor_dataset.py --symbol 000001 --source demo --start 20240101 --end 20241231 --output data/factors/factors_000001.csv
```

Split the factor dataset:

```powershell
python src/split_factor_dataset.py --input data/factors/factors_000001.csv --output-dir data/ml/demo_000001 --target-col label_up_5d --purge-rows 5 --split-mode global_date
```

The splitter infers feature columns by excluding `date`, `symbol`, and any
column name containing `future`, `label`, or `target`. This keeps future-return
labels out of model inputs. `purge_rows` removes rows before split boundaries
to reduce label-window leakage for targets such as `label_up_5d`.

This step still does not train a model. It only prepares educational research
datasets and is not financial advice.

## Baseline ML Model

The baseline ML model step trains a simple supervised classifier from an
already split factor dataset. It supports `random_forest` and
`logistic_regression` and uses only the columns listed in
`feature_columns.txt`, so future-return and label columns stay out of model
inputs.

Build a factor dataset:

```powershell
python src/build_factor_dataset.py --symbol 000001 --source demo --start 20240101 --end 20241231 --output data/factors/factors_000001.csv
```

Split it chronologically:

```powershell
python src/split_factor_dataset.py --input data/factors/factors_000001.csv --output-dir data/ml/demo_000001 --target-col label_up_5d --purge-rows 5 --split-mode global_date
```

Train a baseline model:

```powershell
python src/train_baseline_model.py --dataset-dir data/ml/demo_000001 --target-col label_up_5d --model random_forest --output-dir models/demo_000001
```

Outputs are saved under the selected model output directory:

- `random_forest.joblib` or `logistic_regression.joblib`
- `metrics.json`
- `feature_columns.txt`
- `validation_predictions.csv`
- `test_predictions.csv`
- `feature_importance.csv` when the selected model supports it

This is educational baseline ML, not a trading recommendation. Good validation
or test metrics do not guarantee profitable trading, because trading results
also depend on execution assumptions, transaction costs, risk controls, market
regime changes, and position sizing.

## Model Prediction

After training a baseline model, use the prediction tool to score the latest
row in a factor CSV or ML split CSV:

```powershell
python src/predict_with_model.py --model-path models/demo_000001/random_forest.joblib --input data/factors/factors_000001.csv
```

The predictor loads the `.joblib` model, `metrics.json`, and
`feature_columns.txt` from the model output directory. It reports the predicted
class, the predicted probability of `label_up_5d` when available, a simple
`bullish` / `neutral` / `bearish` model signal, and top feature importance when
the model artifact provides it.

The Streamlit dashboard also has a `Model Prediction` tab:

```powershell
streamlit run app.py
```

Enter the model path and factor CSV path in that panel to view the probability,
signal, scored row, saved metrics, and top factors. This panel does not modify
the rule-based strategy, backtester, or live trading behavior. Model prediction
is educational research output, not financial advice.

## Model Evaluation

Use the model evaluation tool to inspect prediction quality and diagnose
suspicious results:

```powershell
python src/evaluate_model.py --model-dir models/demo_000001 --target-col label_up_5d
```

The evaluator reads these files from the model output directory:

- `metrics.json`
- `feature_columns.txt`
- `validation_predictions.csv`
- `test_predictions.csv`
- `feature_importance.csv` when available

It reports classification metrics, confusion matrix values, probability
distribution buckets, threshold analysis for 0.50 through 0.70, leakage-related
feature-name checks, and warnings. If prediction CSVs contain `close`, it also
runs a simple educational long/flat signal return check based on the selected
probability threshold. This quick check is not the rule-based strategy
backtester and does not change existing trading logic.

The dashboard has a `Model Evaluation` tab:

```powershell
streamlit run app.py
```

Enter the model directory, then click `Evaluate model outputs`. Suspiciously
perfect validation or test metrics should be treated as a warning sign. They
may indicate leakage, duplicated information, an overly easy target, or a demo
dataset that is too regular. Good classification metrics do not prove a
profitable trading strategy.

## ML Signal Backtest

The ML signal backtest converts saved model prediction probabilities into
long/flat trading actions, then sends those signals through the existing
long-only backtester. It does not change the MA crossover strategy or existing
backtest behavior.

Run from PowerShell:

```powershell
python src/run_ml_signal_backtest.py --model-dir models/demo_000001 --factor-csv data/factors/factors_000001.csv --initial-cash 10000 --buy-threshold 0.60 --sell-threshold 0.50
```

Signal rules:

- Buy/open long when predicted probability is at least `buy_threshold`.
- Sell/close long when predicted probability falls below `sell_threshold`.
- Stay flat or keep holding otherwise.
- No short selling is used.

The CLI prints ML strategy metrics, the trade log, buy-and-hold comparison, and
an optional comparison against the existing MA crossover rule-based strategy.
It also warns when saved model metrics look suspiciously perfect.

The dashboard has an `ML Signal Backtest` tab with controls for model directory,
factor CSV path, thresholds, initial cash, execution mode, commission, stamp
tax, slippage, and minimum commission. This workflow is educational research,
not a trading recommendation.

## ML Threshold Experiment

The ML threshold experiment tests multiple probability threshold pairs over the
same factor data. It converts each threshold pair into long/flat ML signals and
runs the existing long-only backtester. Invalid pairs where the sell threshold
is greater than or equal to the buy threshold are skipped.

Basic threshold grid:

```powershell
python src/run_ml_threshold_experiment.py --model-dir models/demo_000001 --input data/factors/factors_000001.csv --output outputs/ml_threshold_experiment.csv
```

Custom thresholds and transaction costs:

```powershell
python src/run_ml_threshold_experiment.py --model-dir models/demo_000001 --input data/factors/factors_000001.csv --buy-thresholds 0.55,0.60,0.65,0.70 --sell-thresholds 0.40,0.45,0.50 --initial-cash 10000 --commission-rate 0 --stamp-tax-rate 0 --slippage-pct 0 --min-commission 0
```

Simple walk-forward research mode:

```powershell
python src/run_ml_threshold_experiment.py --model-dir models/demo_000001 --input data/factors/factors_000001.csv --walk-forward --train-window 120 --test-window 40 --step-size 40
```

The dashboard has an `ML Threshold Experiment` tab. It shows the best threshold
pair by score, best total return pair, best drawdown-control pair, full results,
a ranking table, and a bar chart of threshold scores.

This is threshold research only. Optimizing thresholds on past data can overfit,
and good threshold results do not imply future profitability.

## V4 Step 9: Model Robustness Training

Model robustness training repeats the factor, split, training, and evaluation
workflow across multiple symbols and model types. The goal is to check whether
a baseline model is reasonably stable across symbols and periods instead of
only looking good on one stock or one split.

Multi-symbol and multi-period testing matters because a model can look strong
on one symbol due to leakage, a tiny sample, duplicated patterns, or a market
period that happens to fit the target. Robustness checks are still not proof of
profitability, but they make weak assumptions easier to spot.

Demo command:

```powershell
python src/run_batch_model_training.py --symbols 000001,600519 --source demo --start 20240101 --end 20241231 --output-dir outputs/model_robustness_demo --models logistic_regression,random_forest
```

Real-data command:

```powershell
python src/run_batch_model_training.py --symbols 000001,600519,000858,600036,601318 --source baostock --start 20210101 --end 20241231 --output-dir outputs/model_robustness_baostock --models logistic_regression,random_forest
```

The output directory contains:

- `training_results.csv`: one row per symbol/model with validation and test metrics.
- `model_summary.csv`: average metrics by model type.
- `model_ranking.csv`: simple educational robustness ranking.
- `warnings.csv`: warnings for small samples, suspicious perfect metrics, failed symbols, and missing feature importance.
- `run_config.json`: symbols, source, date range, model types, split settings, and timestamp.

The dashboard has a `Model Robustness` tab that runs the same workflow and
shows summary tables, rankings, warnings, and bar charts for ROC AUC, F1, and
robustness score.

The tab also includes a `Quick Interpretation` section. It checks whether test
ROC AUC is close to random, whether validation and test ROC AUC diverge, whether
test samples are too small, whether one symbol/model pair is weak, whether one
model type is consistently better, and whether high metrics look suspicious.
ROC AUC around `0.5` is close to random. Stable behavior across symbols is more
important than one lucky symbol or split.

Suspiciously perfect metrics are a warning sign, not a success signal. This is
not live trading, not financial advice, and not evidence that a strategy will
work in the future.

## V4 Step 11: Model Robustness Report Export

The model robustness report generator converts an existing robustness output
directory into a readable Markdown research report. It does not retrain models,
change predictions, or change backtest behavior. It only reads saved files such
as `model_summary.csv`, `model_ranking.csv`, `training_results.csv`,
`warnings.csv`, and `run_config.json`.

CLI example:

```powershell
python src/generate_model_report.py --input-dir outputs/model_robustness_demo --output reports/model_robustness_demo.md
```

The report explains which model performed best by test ROC AUC, whether the
ROC AUC is close to random, whether validation and test results diverge, which
symbol/model pair is weakest, whether sample sizes are too small, and what to
check next. Missing optional files are handled gracefully, and an empty
`warnings.csv` is acceptable.

The `Model Robustness` dashboard tab also includes a Markdown report export
panel. Enter a robustness output directory and a report output path, then click
`Generate robustness report` to preview the report and download the Markdown
text. Generated report files under `reports/` are ignored by Git.

High ML metrics and polished reports do not guarantee profitable trading. This
report is educational research output, not financial advice.

## V4 Step 12: Feature Source Registry and Multi-Factor Roadmap

The feature source registry is a structured roadmap for improving future ML
datasets beyond basic OHLCV features. It documents planned factor families,
expected raw fields, possible engineered features, update frequency, required
lag controls, cost level, access method, token requirements, leakage risk,
expected predictive value, and P0/P1/P2/P3 implementation priority.

CLI examples:

```powershell
python src/show_feature_sources.py --list
python src/show_feature_sources.py --family valuation
python src/show_feature_sources.py --priority P0
python src/show_feature_sources.py --training-ready
python src/show_feature_sources.py --high-leakage-risk
python src/show_feature_sources.py --export outputs/feature_source_registry.csv
```

The dashboard has a `Feature Sources` tab with a full registry table, factor
family and priority filters, training-ready rows, high-leakage-risk rows,
token-required rows, a factor-family summary, and a CSV download button.

The project is moving from basic OHLCV features toward multi-factor research
because price and volume alone often miss valuation, profitability, growth,
balance-sheet quality, cash flow, dividends, institutional holdings, capital
flow, sentiment, macro context, industry strength, and market regime. These
families should be added gradually so each source can be tested and documented.

More data does not automatically improve prediction. Extra fields can add
noise, duplicate existing signals, reduce sample size, introduce vendor bias,
or create hidden future leakage. Every feature needs lag control based on when
the data was actually known, not just the fiscal period, event date, or file
label.

Valuation should include more than PE. A useful valuation roadmap should
compare PE, PB, PS, PCF, EV/EBITDA, dividend yield, market cap, own-history
percentiles, industry percentiles, and growth or profitability matching such as
PE-to-growth and PB-to-ROE.

P0/P1/P2/P3 priority keeps the roadmap practical. P0 covers features closest to
the existing pipeline. P1 covers high-value next integrations that still need
careful lag checks. P2 covers useful but slower or more complex sources. P3 is
reserved for future or difficult sources that need clearer access, timestamps,
or reproducibility before model training.

This registry does not fetch paid or private data, does not include API tokens,
and does not claim that any factor guarantees profit. It is an educational
research roadmap, not financial advice.

## Smoke Tests

Run the offline smoke tests before committing changes or after pulling new code:

```powershell
python src/run_smoke_tests.py
```

Smoke tests compile the main scripts, check dependencies, run the offline demo,
and run `--help` checks for command-line tools. They do not fetch real market
data and do not call Baostock or AkShare data download commands.

## Automated Checks

GitHub Actions runs the offline smoke test after pushes and pull requests:

```powershell
python src/run_smoke_tests.py
```

The CI workflow installs dependencies and runs the smoke tests only. It does not
fetch real market data and does not call Baostock or AkShare data download
commands.

## Example Output

These snippets are shortened examples. Demo results use fake sample data and
are only for checking the project workflow, not for real market conclusions.

Setup check:

```text
QuantPilot-AI Setup Check
-------------------------
pandas: OK
akshare: OK
baostock: OK
matplotlib: OK

Setup check passed.
```

Offline demo:

```text
QuantPilot-AI Offline Demo
--------------------------
Using demo data file: data/sample/demo_000001.csv

Performance Summary
total_return_pct: 8.51

QuantPilot-AI Rule-Based Strategy Report
Total return: 8.51%

Trade Metrics
win_rate_pct: 100.00%
```

Smoke tests:

```text
QuantPilot-AI Smoke Tests
-------------------------
PASS: py_compile src/run_demo.py
PASS: setup check
PASS: offline demo

All smoke tests passed.
```

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

## Realistic Execution Assumptions

By default, QuantPilot-AI uses `same_close` execution. This means a signal is
filled at the same day's close price. It is simple and useful for education, but
real trading usually cannot know the final close before placing the trade.

For a more realistic assumption, use `next_open`:

```powershell
python src/run_stock_backtest.py --symbol 000001 --source baostock --start 20240101 --end 20241231 --execution-mode next_open --commission-rate 0.0003 --stamp-tax-rate 0.001 --slippage-pct 0.05 --min-commission 5
```

Supported execution modes:

- `same_close`: execute at the current close. This preserves the original simple workflow.
- `next_open`: execute today's signal on the next trading day's open.
- `next_close`: execute today's signal on the next trading day's close.

Transaction costs reduce returns. The optional assumptions are:

- `--commission-rate`: commission charged on buys and sells.
- `--stamp-tax-rate`: tax charged on sell trades only.
- `--slippage-pct`: price impact added to buys and subtracted from sells.
- `--min-commission`: minimum commission per trade.

These assumptions make the backtest more realistic, but the project is still an
educational research tool and is not financial advice.

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
- `outputs/`
- `data/factors/`
- `data/ml/`
- `models/`
- Python cache files
- Matplotlib cache files
- local editor/system files

## Educational Disclaimer

This project is for programming, data analysis, and educational research.

It is not financial advice.

Backtest results do not guarantee future returns.
