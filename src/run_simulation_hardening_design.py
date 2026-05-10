import argparse
import sys

try:
    from .simulation_hardening_design import (
        DEFAULT_CAPITAL_DIR,
        DEFAULT_COVERAGE_GAP_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_EXIT_DIR,
        DEFAULT_MONITORING_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_POSITION_DIR,
        DEFAULT_UNIVERSE_DIR,
        DEFAULT_V5_CLOSURE_DIR,
        generate_simulation_hardening_design_outputs,
    )
except ImportError:
    from simulation_hardening_design import (
        DEFAULT_CAPITAL_DIR,
        DEFAULT_COVERAGE_GAP_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_EXIT_DIR,
        DEFAULT_MONITORING_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_POSITION_DIR,
        DEFAULT_UNIVERSE_DIR,
        DEFAULT_V5_CLOSURE_DIR,
        generate_simulation_hardening_design_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 8 research-only simulation hardening design.",
    )
    parser.add_argument("--capital-dir", default=str(DEFAULT_CAPITAL_DIR))
    parser.add_argument("--universe-dir", default=str(DEFAULT_UNIVERSE_DIR))
    parser.add_argument("--position-dir", default=str(DEFAULT_POSITION_DIR))
    parser.add_argument("--exit-dir", default=str(DEFAULT_EXIT_DIR))
    parser.add_argument("--daily-plan-dir", default=str(DEFAULT_DAILY_PLAN_DIR))
    parser.add_argument("--paper-ledger-dir", default=str(DEFAULT_PAPER_LEDGER_DIR))
    parser.add_argument("--monitoring-dir", default=str(DEFAULT_MONITORING_DIR))
    parser.add_argument("--v5-closure-dir", default=str(DEFAULT_V5_CLOSURE_DIR))
    parser.add_argument("--coverage-gap-dir", default=str(DEFAULT_COVERAGE_GAP_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_simulation_hardening_design_outputs(
            capital_dir=args.capital_dir,
            universe_dir=args.universe_dir,
            position_dir=args.position_dir,
            exit_dir=args.exit_dir,
            daily_plan_dir=args.daily_plan_dir,
            paper_ledger_dir=args.paper_ledger_dir,
            monitoring_dir=args.monitoring_dir,
            v5_closure_dir=args.v5_closure_dir,
            coverage_gap_dir=args.coverage_gap_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: simulation hardening design failed: {exc}")
        sys.exit(1)

    summary = result["simulation_hardening_design_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 8 Simulation Hardening Design / Multi-Day Paper Replay Planning")
    print("--------------------------------------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Planned replay phases: {summary['planned_replay_phase_count']}")
    print(f"Input dependencies: {summary['input_dependency_count']}")
    print(f"Risk controls: {summary['risk_control_count']}")
    print(f"Evidence requirements: {summary['evidence_requirement_count']}")
    print(f"Forbidden true flags: {summary['forbidden_true_flag_count']}")
    print(f"Execution allowed: {summary['execution_allowed']}")
    print(f"Broker connected: {summary['broker_connected']}")
    print(f"Live trading: {summary['live_trading']}")
    print(f"Real order submission: {summary['real_order_submission']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print(f"Recommended next step: {summary['recommended_next_step']}")
    print("Scope: simulation hardening design only")
    print()
    print("Warning: This is educational/research planning only, not financial advice.")
    print("No replay, backtests, market data fetches, model training, broker connections, live trading, or order submissions are performed.")
    print("All V6 Step 8 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
