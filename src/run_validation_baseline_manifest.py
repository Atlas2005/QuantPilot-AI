import argparse
import sys

try:
    from .validation_baseline_manifest import (
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
        DEFAULT_V5_CLOSURE_DIR,
        generate_validation_baseline_manifest_outputs,
    )
except ImportError:
    from validation_baseline_manifest import (
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
        DEFAULT_V5_CLOSURE_DIR,
        generate_validation_baseline_manifest_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 1 research-only validation baseline manifest.",
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
    parser.add_argument("--v5-closure-dir", default=str(DEFAULT_V5_CLOSURE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_validation_baseline_manifest_outputs(
            capital_dir=args.capital_dir,
            universe_dir=args.universe_dir,
            position_dir=args.position_dir,
            exit_dir=args.exit_dir,
            daily_plan_dir=args.daily_plan_dir,
            paper_ledger_dir=args.paper_ledger_dir,
            semi_auto_dir=args.semi_auto_dir,
            broker_research_dir=args.broker_research_dir,
            monitoring_dir=args.monitoring_dir,
            v5_closure_dir=args.v5_closure_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: validation baseline manifest failed: {exc}")
        sys.exit(1)

    summary = result["validation_baseline_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 1 Validation Baseline Manifest")
    print("----------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Baseline steps: {summary['baseline_step_count']}")
    print(f"Present output directories: {summary['present_output_dir_count']}")
    print(f"Missing output directories: {summary['missing_output_dir_count']}")
    print(f"Total files: {summary['total_file_count']}")
    print(f"Baseline status: {summary['baseline_status']}")
    print("Scope: manifest/audit only")
    print()
    print("Warning: This is educational/research manifesting only, not financial advice.")
    print("No backtests, market data fetches, model training, broker connections, live trading, or order submissions are performed.")
    print("All V6 Step 1 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
