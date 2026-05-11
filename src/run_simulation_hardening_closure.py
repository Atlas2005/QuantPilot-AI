import argparse
import sys

try:
    from .simulation_hardening_closure import (
        DEFAULT_OUTPUT_DIR,
        generate_simulation_hardening_closure_outputs,
    )
except ImportError:
    from simulation_hardening_closure import (
        DEFAULT_OUTPUT_DIR,
        generate_simulation_hardening_closure_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 15 research-only simulation hardening closure and V7 transition plan.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_simulation_hardening_closure_outputs(output_dir=args.output_dir)
    except Exception as exc:
        print(f"Error: simulation hardening closure failed: {exc}")
        sys.exit(1)

    summary = result["v6_closure_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 15 Simulation Hardening Closure")
    print("------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Reviewed inputs: {summary['reviewed_input_count']}")
    print(f"Missing inputs: {summary['missing_input_count']}")
    print(f"Completed V6 steps: {summary['completed_v6_step_count']}")
    print(f"Remaining gaps: {summary['remaining_gap_count']}")
    print(f"Transition plan rows: {summary['transition_plan_row_count']}")
    print(f"Open-source policy rows: {summary['open_source_policy_row_count']}")
    print(f"Market data fetches: {summary['market_data_fetch_count']}")
    print(f"Broker connected count: {summary['broker_connected_count']}")
    print(f"Execution allowed count: {summary['execution_allowed_count']}")
    print(f"Live trading count: {summary['live_trading_count']}")
    print(f"Real order submission count: {summary['real_order_submission_count']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print(f"Recommended next step: {summary['recommended_next_step']}")
    print("Scope: V6 closure and V7 transition planning only")
    print()
    print("Warning: This is educational/research closure only, not financial advice.")
    print("No new engines, market data fetches, broker connections, model training, threshold changes, feature changes, or order actions are performed.")
    print("All V6 Step 15 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
