import argparse
import sys

try:
    from .capital_aware_infrastructure_review import (
        DEFAULT_BROKER_RESEARCH_DIR,
        DEFAULT_CAPITAL_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_EXIT_DIR,
        DEFAULT_MONITORING_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_POSITION_DIR,
        DEFAULT_SEMI_AUTO_DIR,
        DEFAULT_UNIVERSE_DIR,
        generate_capital_aware_infrastructure_review_outputs,
    )
except ImportError:
    from capital_aware_infrastructure_review import (
        DEFAULT_BROKER_RESEARCH_DIR,
        DEFAULT_CAPITAL_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_EXIT_DIR,
        DEFAULT_MONITORING_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_POSITION_DIR,
        DEFAULT_SEMI_AUTO_DIR,
        DEFAULT_UNIVERSE_DIR,
        generate_capital_aware_infrastructure_review_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 10 review-only capital-aware infrastructure closure.",
    )
    parser.add_argument("--capital-dir", default=str(DEFAULT_CAPITAL_DIR))
    parser.add_argument("--universe-dir", default=str(DEFAULT_UNIVERSE_DIR))
    parser.add_argument("--position-dir", default=str(DEFAULT_POSITION_DIR))
    parser.add_argument("--exit-dir", default=str(DEFAULT_EXIT_DIR))
    parser.add_argument("--daily-plan-dir", default=str(DEFAULT_DAILY_PLAN_DIR))
    parser.add_argument("--paper-ledger-dir", default=str(DEFAULT_PAPER_LEDGER_DIR))
    parser.add_argument("--semi-auto-dir", default=str(DEFAULT_SEMI_AUTO_DIR))
    parser.add_argument("--broker-research-dir", default=str(DEFAULT_BROKER_RESEARCH_DIR))
    parser.add_argument("--monitoring-dir", default=str(DEFAULT_MONITORING_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_capital_aware_infrastructure_review_outputs(
            capital_dir=args.capital_dir,
            universe_dir=args.universe_dir,
            position_dir=args.position_dir,
            exit_dir=args.exit_dir,
            daily_plan_dir=args.daily_plan_dir,
            paper_ledger_dir=args.paper_ledger_dir,
            semi_auto_dir=args.semi_auto_dir,
            broker_research_dir=args.broker_research_dir,
            monitoring_dir=args.monitoring_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: capital-aware infrastructure review failed: {exc}")
        sys.exit(1)

    summary = result["closure_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 10 Capital-Aware Infrastructure Review / Closure")
    print("-----------------------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Reviewed steps: {summary['reviewed_step_count']}")
    print(f"Completed steps: {summary['completed_step_count']}")
    print(f"Missing steps: {summary['missing_step_count']}")
    print(f"Blocking readiness issues: {summary['blocking_readiness_issue_count']}")
    print(f"Warning issues: {summary['warning_issue_count']}")
    print(f"Final V5 status: {summary['final_v5_status']}")
    print(f"Recommended next phase: {summary['recommended_next_phase']}")
    print("Scope: review-only closure")
    print()
    print("Warning: This is educational/research review only, not financial advice.")
    print("No backtests, market data fetches, model training, broker connections, live trading, or order submissions are performed.")
    print("All Step 10 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
