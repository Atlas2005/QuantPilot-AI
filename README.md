# QuantPilot-AI

QuantPilot-AI is a beginner-friendly stock research and backtesting platform.

The goal of this project is to help users learn how stock data, technical
indicators, trading signals, backtesting, performance metrics, and simple
research reports can work together in one explainable workflow.

This project does not try to predict stock prices with certainty. It is built
for education, experimentation, and research.

## Current V1 Features

- Sample stock K-line data from CSV
- Technical indicators:
  - MA5
  - MA20
  - RSI
  - CCI
- MA crossover signal generation
- Simple long-only backtesting
- Performance metrics
- Rule-based strategy report
- One-command workflow using:

```bash
python src/main.py
```

## Project Structure

```text
QuantPilot-AI/
├── data/
│   └── sample/
│       ├── README.md
│       └── sample_stock.csv
├── docs/
│   ├── project-plan.md
│   └── stock-research-template.md
├── src/
│   ├── backtester.py
│   ├── data_loader.py
│   ├── indicators.py
│   ├── main.py
│   ├── metrics.py
│   ├── report_generator.py
│   └── strategy.py
├── .gitignore
└── README.md
```

## Installation

Clone the project and enter the project folder:

```bash
git clone <your-repository-url>
cd QuantPilot-AI
```

Install the required Python package:

```bash
pip install pandas
```

Python 3.10 or newer is recommended.

## Usage

Run the full V1 workflow from the project root:

```bash
python src/main.py
```

This command will:

1. Load `data/sample/sample_stock.csv`
2. Add MA5, MA20, RSI, and CCI indicators
3. Generate MA crossover signals
4. Run a simple long-only backtest
5. Calculate performance metrics
6. Print a rule-based strategy report
7. Print the last 10 rows of the backtest result

## Example Output

The current sample data produces output similar to:

```text
initial_value: 10000.0
final_value: 11227.91
total_return_pct: 12.28
max_drawdown_pct: -5.78
buy_signals: 1
currently_holding: True
```

The exact formatting may differ slightly, but the V1 result should show one
buy signal and a final portfolio value around `11227.91`.

## Disclaimer

This project is for educational and research purposes only. It is not financial
advice.

## V2 Roadmap

- Real stock data loader
- Streamlit dashboard
- Better risk metrics
- AI-assisted report generation
- Multi-agent research system
