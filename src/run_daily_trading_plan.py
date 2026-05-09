import argparse
import sys

try:
    from .daily_trading_plan import (
        DEFAULT_DEFERRED_PATH,
        DEFAULT_EXIT_PLAN_PATH,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SIZED_PATH,
        DEFAULT_TRADABLE_PATH,
        generate_daily_trading_plan_outputs,
    )
except ImportError:
    from daily_trading_plan import (
        DEFAULT_DEFERRED_PATH,
        DEFAULT_EXIT_PLAN_PATH,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SIZED_PATH,
        DEFAULT_TRADABLE_PATH,
        generate_daily_trading_plan_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 5 research-only daily trading plan generation from local V5 outputs.",
    )
    parser.add_argument("--tradable-path", default=str(DEFAULT_TRADABLE_PATH))
    parser.add_argument("--sized-path", default=str(DEFAULT_SIZED_PATH))
    parser.add_argument("--deferred-path", default=str(DEFAULT_DEFERRED_PATH))
    parser.add_argument("--exit-plan-path", default=str(DEFAULT_EXIT_PLAN_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_daily_trading_plan_outputs(
            tradable_path=args.tradable_path,
            sized_path=args.sized_path,
            deferred_path=args.deferred_path,
            exit_plan_path=args.exit_plan_path,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: daily trading plan generation failed: {exc}")
        sys.exit(1)

    summary = result["daily_trading_plan_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 5 Daily Trading Plan")
    print("------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Tradable candidate count: {summary['tradable_candidate_count']}")
    print(f"Sized position count: {summary['sized_position_count']}")
    print(f"Deferred position count: {summary['deferred_position_count']}")
    print(f"Exit plan count: {summary['exit_plan_count']}")
    print(f"Daily plan row count: {summary['daily_plan_row_count']}")
    print("Scope: daily plan only")
    print()
    print("Warning: This is educational/research tooling only, not financial advice.")
    print("No broker execution, live trading, or order execution is performed.")
    print("All plan rows preserve trading_ready=False.")


if __name__ == "__main__":
    main()
