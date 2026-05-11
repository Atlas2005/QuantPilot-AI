import argparse
import sys

try:
    from .simulation_hardening_review import (
        DEFAULT_COVERAGE_GAP_DIR,
        DEFAULT_EVIDENCE_INDEX_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_REPLAY_HARNESS_DIR,
        DEFAULT_SIMULATION_DESIGN_DIR,
        generate_simulation_hardening_review_outputs,
    )
except ImportError:
    from simulation_hardening_review import (
        DEFAULT_COVERAGE_GAP_DIR,
        DEFAULT_EVIDENCE_INDEX_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_REPLAY_HARNESS_DIR,
        DEFAULT_SIMULATION_DESIGN_DIR,
        generate_simulation_hardening_review_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 10 research-only simulation hardening review and closure.",
    )
    parser.add_argument("--simulation-design-dir", default=str(DEFAULT_SIMULATION_DESIGN_DIR))
    parser.add_argument("--replay-harness-dir", default=str(DEFAULT_REPLAY_HARNESS_DIR))
    parser.add_argument("--coverage-gap-dir", default=str(DEFAULT_COVERAGE_GAP_DIR))
    parser.add_argument("--evidence-index-dir", default=str(DEFAULT_EVIDENCE_INDEX_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_simulation_hardening_review_outputs(
            simulation_design_dir=args.simulation_design_dir,
            replay_harness_dir=args.replay_harness_dir,
            coverage_gap_dir=args.coverage_gap_dir,
            evidence_index_dir=args.evidence_index_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: simulation hardening review failed: {exc}")
        sys.exit(1)

    summary = result["simulation_hardening_review_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 10 Multi-Day Replay Review / Simulation Hardening Closure")
    print("----------------------------------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Reviewed inputs: {summary['reviewed_input_count']}")
    print(f"Missing inputs: {summary['missing_input_count']}")
    print(f"Design phases: {summary['design_phase_count']}")
    print(f"Replay calendar days: {summary['replay_calendar_day_count']}")
    print(f"Replay position snapshots: {summary['replay_position_snapshot_count']}")
    print(f"Replay events: {summary['replay_event_count']}")
    print(f"Replay transitions: {summary['replay_transition_count']}")
    print(f"Remaining blockers: {summary['remaining_blocker_count']}")
    print(f"Blocking gaps: {summary['blocking_gap_count']}")
    print(f"Market data fetches: {summary['market_data_fetch_count']}")
    print(f"Broker connected count: {summary['broker_connected_count']}")
    print(f"Execution allowed count: {summary['execution_allowed_count']}")
    print(f"Live trading count: {summary['live_trading_count']}")
    print(f"Real order submission count: {summary['real_order_submission_count']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print(f"Recommended next step: {summary['recommended_next_step']}")
    print("Scope: simulation hardening review and research-only closure")
    print()
    print("Warning: This is educational/research review only, not financial advice.")
    print("Step 8 was design only; Step 9 was a deterministic local scaffold only.")
    print("No market data is fetched, no backtest is run, no broker connection is made, and no order is executed or submitted.")
    print("All V6 Step 10 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
