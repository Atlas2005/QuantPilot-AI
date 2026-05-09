import argparse
import sys

try:
    from .exit_engine import (
        DEFAULT_BENCHMARK_LAG_EXIT_RULE,
        DEFAULT_INPUT_PATH,
        DEFAULT_MAX_HOLDING_DAYS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_STOP_LOSS_PCT,
        DEFAULT_TAKE_PROFIT_PCT,
        generate_exit_engine_outputs,
    )
except ImportError:
    from exit_engine import (
        DEFAULT_BENCHMARK_LAG_EXIT_RULE,
        DEFAULT_INPUT_PATH,
        DEFAULT_MAX_HOLDING_DAYS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_STOP_LOSS_PCT,
        DEFAULT_TAKE_PROFIT_PCT,
        generate_exit_engine_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 4 research-only exit planning for Step 3 sized positions.",
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stop-loss-pct", type=float, default=DEFAULT_STOP_LOSS_PCT)
    parser.add_argument("--take-profit-pct", type=float, default=DEFAULT_TAKE_PROFIT_PCT)
    parser.add_argument("--max-holding-days", type=int, default=DEFAULT_MAX_HOLDING_DAYS)
    parser.add_argument(
        "--benchmark-lag-exit-rule",
        default=DEFAULT_BENCHMARK_LAG_EXIT_RULE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_exit_engine_outputs(
            input_path=args.input_path,
            output_dir=args.output_dir,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            max_holding_days=args.max_holding_days,
            benchmark_lag_exit_rule=args.benchmark_lag_exit_rule,
        )
    except Exception as exc:
        print(f"Error: exit engine failed: {exc}")
        sys.exit(1)

    summary = result["exit_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 4 Exit Engine")
    print("-----------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Sized position count: {summary['sized_position_count']}")
    print(f"Planned exit count: {summary['planned_exit_count']}")
    print(f"Invalid exit plan count: {summary['invalid_exit_plan_count']}")
    print("Scope: exit planning only")
    print()
    print("Warning: This is educational/research tooling only, not financial advice.")
    print("No broker execution, live trading, or order execution is performed.")
    print("No exit plan is trading-ready.")


if __name__ == "__main__":
    main()
