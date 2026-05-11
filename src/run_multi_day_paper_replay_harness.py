import argparse
import sys

try:
    from .multi_day_paper_replay_harness import (
        DEFAULT_BROKER_RESEARCH_DIR,
        DEFAULT_COVERAGE_GAP_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_MONITORING_DIR,
        DEFAULT_ORDER_GENERATOR_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_REPLAY_START_DATE,
        DEFAULT_SIMULATION_DESIGN_DIR,
        DEFAULT_V5_CLOSURE_DIR,
        generate_multi_day_paper_replay_harness_outputs,
    )
except ImportError:
    from multi_day_paper_replay_harness import (
        DEFAULT_BROKER_RESEARCH_DIR,
        DEFAULT_COVERAGE_GAP_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_MONITORING_DIR,
        DEFAULT_ORDER_GENERATOR_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_REPLAY_START_DATE,
        DEFAULT_SIMULATION_DESIGN_DIR,
        DEFAULT_V5_CLOSURE_DIR,
        generate_multi_day_paper_replay_harness_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 9 research-only multi-day paper replay harness scaffold.",
    )
    parser.add_argument("--daily-plan-dir", default=str(DEFAULT_DAILY_PLAN_DIR))
    parser.add_argument("--paper-ledger-dir", default=str(DEFAULT_PAPER_LEDGER_DIR))
    parser.add_argument("--order-generator-dir", default=str(DEFAULT_ORDER_GENERATOR_DIR))
    parser.add_argument("--broker-research-dir", default=str(DEFAULT_BROKER_RESEARCH_DIR))
    parser.add_argument("--monitoring-dir", default=str(DEFAULT_MONITORING_DIR))
    parser.add_argument("--v5-closure-dir", default=str(DEFAULT_V5_CLOSURE_DIR))
    parser.add_argument("--coverage-gap-dir", default=str(DEFAULT_COVERAGE_GAP_DIR))
    parser.add_argument("--simulation-design-dir", default=str(DEFAULT_SIMULATION_DESIGN_DIR))
    parser.add_argument("--replay-start-date", default=DEFAULT_REPLAY_START_DATE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_multi_day_paper_replay_harness_outputs(
            daily_plan_dir=args.daily_plan_dir,
            paper_ledger_dir=args.paper_ledger_dir,
            order_generator_dir=args.order_generator_dir,
            broker_research_dir=args.broker_research_dir,
            monitoring_dir=args.monitoring_dir,
            v5_closure_dir=args.v5_closure_dir,
            coverage_gap_dir=args.coverage_gap_dir,
            simulation_design_dir=args.simulation_design_dir,
            replay_start_date=args.replay_start_date,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: multi-day paper replay harness scaffold failed: {exc}")
        sys.exit(1)

    summary = result["multi_day_replay_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 9 Multi-Day Paper Replay Harness / Replay Input Scaffold")
    print("--------------------------------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Input dependencies: {summary['input_dependency_count']}")
    print(f"Missing input dependencies: {summary['missing_input_dependency_count']}")
    print(f"Replay calendar days: {summary['replay_calendar_day_count']}")
    print(f"Replay position snapshots: {summary['replay_position_snapshot_count']}")
    print(f"Replay events: {summary['replay_event_count']}")
    print(f"Open positions: {summary['open_position_count']}")
    print(f"Closed positions: {summary['closed_position_count']}")
    print(f"Market data fetches: {summary['market_data_fetch_count']}")
    print(f"Broker connected count: {summary['broker_connected_count']}")
    print(f"Execution allowed count: {summary['execution_allowed_count']}")
    print(f"Live trading count: {summary['live_trading_count']}")
    print(f"Real order submission count: {summary['real_order_submission_count']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print("Scope: replay harness/input scaffold only")
    print()
    print("Warning: This is educational/research scaffolding only, not financial advice.")
    print("No live or historical market data is fetched; no backtest is run; no broker SDK is imported or connected.")
    print("No order is executed, submitted, simulated-real, or routed. All V6 Step 9 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
