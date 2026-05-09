import argparse
import sys

try:
    from .paper_trading_ledger import (
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        generate_paper_trading_ledger_outputs,
    )
except ImportError:
    from paper_trading_ledger import (
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        generate_paper_trading_ledger_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 6 research-only paper trading ledger generation from local Step 5 outputs.",
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--starting-cash",
        type=float,
        default=None,
        help="Optional fallback starting cash used only when daily_trading_plan_summary.csv has no available_cash.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_paper_trading_ledger_outputs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            starting_cash=args.starting_cash,
        )
    except Exception as exc:
        print(f"Error: paper trading ledger generation failed: {exc}")
        sys.exit(1)

    summary = result["paper_trading_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 6 Paper Trading Ledger")
    print("---------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Paper order count: {summary['paper_order_count']}")
    print(f"Paper filled order count: {summary['paper_filled_order_count']}")
    print(f"Paper deferred order count: {summary['paper_deferred_order_count']}")
    print(f"Open paper position count: {summary['open_paper_position_count']}")
    print(f"Starting cash: {summary['starting_cash']}")
    print(f"Ending cash: {summary['ending_cash']}")
    print("Scope: paper ledger only")
    print()
    print("Warning: This is educational/research paper trading only, not financial advice.")
    print("No broker execution, live trading, live market data fetch, or real order submission occurred.")
    print("All outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
