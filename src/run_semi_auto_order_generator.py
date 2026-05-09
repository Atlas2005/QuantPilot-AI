import argparse
import sys

try:
    from .semi_auto_order_generator import (
        DEFAULT_DAILY_PLAN_PATH,
        DEFAULT_EXIT_PLAN_PATH,
        DEFAULT_OUTPUT_DIR,
        generate_semi_auto_order_outputs,
    )
except ImportError:
    from semi_auto_order_generator import (
        DEFAULT_DAILY_PLAN_PATH,
        DEFAULT_EXIT_PLAN_PATH,
        DEFAULT_OUTPUT_DIR,
        generate_semi_auto_order_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 7 broker-neutral semi-auto order draft generation.",
    )
    parser.add_argument("--daily-plan-path", default=str(DEFAULT_DAILY_PLAN_PATH))
    parser.add_argument("--exit-plan-path", default=str(DEFAULT_EXIT_PLAN_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_semi_auto_order_outputs(
            daily_plan_path=args.daily_plan_path,
            exit_plan_path=args.exit_plan_path,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: semi-auto order generator failed: {exc}")
        sys.exit(1)

    summary = result["semi_auto_order_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 7 Semi-Auto Order Generator")
    print("--------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Draft order count: {summary['draft_order_count']}")
    print(f"Buy draft count: {summary['buy_draft_count']}")
    print(f"Execution allowed count: {summary['execution_allowed_count']}")
    print(f"Broker connected count: {summary['broker_connected_count']}")
    print(f"Trading ready count: {summary['trading_ready_count']}")
    print("Scope: broker-neutral draft tickets only")
    print()
    print("Warning: This is educational/research tooling only, not financial advice.")
    print("No broker connection, live trading, order execution, or order submission is performed.")
    print("All generated drafts preserve trading_ready=False.")


if __name__ == "__main__":
    main()
