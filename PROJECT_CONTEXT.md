# Project Context

## Project Goal

QuantPilot-AI is a beginner-friendly AI stock research and backtesting platform.

## V1 Completed Features

- Sample CSV data
- Technical indicators: MA5, MA20, RSI, CCI
- MA crossover signal strategy
- Simple long-only backtester
- Performance metrics
- Rule-based report generator
- Main workflow

## Current V2 Goal

- Add real A-share data support
- Use AkShare / Eastmoney as the first data source
- Standardize real data into the V1 format: date, open, high, low, close, volume

## Current Note

Eastmoney connection is currently unstable from this machine/network. The loader code exists, but live data fetching should be tested later before commit.

## Development Rules

- Always check current directory before editing.
- Work on v2-real-data branch for V2.
- Do not modify V1 modules unless explicitly requested.
- Temporary test files such as test_akshare.py should not be committed.
- Commit only after manual tests pass.
