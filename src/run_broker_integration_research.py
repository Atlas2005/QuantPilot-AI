import argparse
import sys

try:
    from .broker_integration_research import (
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        generate_broker_integration_research_outputs,
    )
except ImportError:
    from broker_integration_research import (
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        generate_broker_integration_research_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 8 research-only broker integration constraint analysis.",
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_broker_integration_research_outputs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: broker integration research failed: {exc}")
        sys.exit(1)

    summary = result["broker_integration_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 8 Broker Integration Research")
    print("----------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Input draft order count: {summary['input_draft_order_count']}")
    print(f"Researched mode count: {summary['researched_mode_count']}")
    print(f"Constraint count: {summary['constraint_count']}")
    print(f"High-risk constraint count: {summary['high_risk_constraint_count']}")
    print("Scope: broker integration research only")
    print()
    print("Warning: This is educational/research tooling only, not financial advice.")
    print("No credentials are requested. No broker SDKs are imported. No broker connection, live trading, order execution, or real order submission is performed.")
    print("All outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
