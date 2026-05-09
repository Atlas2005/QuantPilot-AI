import argparse
import sys

try:
    from .monitoring_reporting_layer import (
        DEFAULT_BROKER_RESEARCH_DIR,
        DEFAULT_CAPITAL_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_EXIT_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_POSITION_DIR,
        DEFAULT_SEMI_AUTO_DIR,
        DEFAULT_UNIVERSE_DIR,
        generate_monitoring_reporting_outputs,
    )
except ImportError:
    from monitoring_reporting_layer import (
        DEFAULT_BROKER_RESEARCH_DIR,
        DEFAULT_CAPITAL_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_EXIT_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_POSITION_DIR,
        DEFAULT_SEMI_AUTO_DIR,
        DEFAULT_UNIVERSE_DIR,
        generate_monitoring_reporting_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 9 research-only monitoring/reporting over local V5 outputs.",
    )
    parser.add_argument("--capital-dir", default=str(DEFAULT_CAPITAL_DIR))
    parser.add_argument("--universe-dir", default=str(DEFAULT_UNIVERSE_DIR))
    parser.add_argument("--position-dir", default=str(DEFAULT_POSITION_DIR))
    parser.add_argument("--exit-dir", default=str(DEFAULT_EXIT_DIR))
    parser.add_argument("--daily-plan-dir", default=str(DEFAULT_DAILY_PLAN_DIR))
    parser.add_argument("--paper-ledger-dir", default=str(DEFAULT_PAPER_LEDGER_DIR))
    parser.add_argument("--semi-auto-dir", default=str(DEFAULT_SEMI_AUTO_DIR))
    parser.add_argument("--broker-research-dir", default=str(DEFAULT_BROKER_RESEARCH_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_monitoring_reporting_outputs(
            capital_dir=args.capital_dir,
            universe_dir=args.universe_dir,
            position_dir=args.position_dir,
            exit_dir=args.exit_dir,
            daily_plan_dir=args.daily_plan_dir,
            paper_ledger_dir=args.paper_ledger_dir,
            semi_auto_dir=args.semi_auto_dir,
            broker_research_dir=args.broker_research_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: monitoring/reporting layer failed: {exc}")
        sys.exit(1)

    summary = result["monitoring_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 9 Monitoring / Reporting Layer")
    print("-----------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Dashboard rows: {summary['dashboard_row_count']}")
    print(f"Alerts: {summary['alert_count']}")
    print(f"Blocking alerts: {summary['blocking_alert_count']}")
    print(f"Warning alerts: {summary['warning_alert_count']}")
    print(f"Info alerts: {summary['info_alert_count']}")
    print("Scope: monitoring/reporting only")
    print()
    print("Warning: This is educational/research monitoring only, not financial advice.")
    print("No backtests, market data fetches, broker connections, live trading, or order submissions are performed.")
    print("All Step 9 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
