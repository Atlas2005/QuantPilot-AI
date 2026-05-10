import argparse
import sys

try:
    from .reproducibility_rerun_validator import (
        DEFAULT_BASELINE_DIR,
        DEFAULT_BROKER_RESEARCH_DIR,
        DEFAULT_CAPITAL_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_DEPENDENCY_VALIDATOR_DIR,
        DEFAULT_EXIT_DIR,
        DEFAULT_MONITORING_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_POSITION_DIR,
        DEFAULT_SCHEMA_VALIDATOR_DIR,
        DEFAULT_SEMI_AUTO_DIR,
        DEFAULT_UNIVERSE_DIR,
        DEFAULT_V5_CLOSURE_DIR,
        generate_reproducibility_rerun_validation_outputs,
    )
except ImportError:
    from reproducibility_rerun_validator import (
        DEFAULT_BASELINE_DIR,
        DEFAULT_BROKER_RESEARCH_DIR,
        DEFAULT_CAPITAL_DIR,
        DEFAULT_DAILY_PLAN_DIR,
        DEFAULT_DEPENDENCY_VALIDATOR_DIR,
        DEFAULT_EXIT_DIR,
        DEFAULT_MONITORING_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAPER_LEDGER_DIR,
        DEFAULT_POSITION_DIR,
        DEFAULT_SCHEMA_VALIDATOR_DIR,
        DEFAULT_SEMI_AUTO_DIR,
        DEFAULT_UNIVERSE_DIR,
        DEFAULT_V5_CLOSURE_DIR,
        generate_reproducibility_rerun_validation_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 4 research-only reproducibility rerun validation.",
    )
    parser.add_argument("--semi-auto-dir", default=str(DEFAULT_SEMI_AUTO_DIR))
    parser.add_argument("--broker-research-dir", default=str(DEFAULT_BROKER_RESEARCH_DIR))
    parser.add_argument("--capital-dir", default=str(DEFAULT_CAPITAL_DIR))
    parser.add_argument("--universe-dir", default=str(DEFAULT_UNIVERSE_DIR))
    parser.add_argument("--position-dir", default=str(DEFAULT_POSITION_DIR))
    parser.add_argument("--exit-dir", default=str(DEFAULT_EXIT_DIR))
    parser.add_argument("--daily-plan-dir", default=str(DEFAULT_DAILY_PLAN_DIR))
    parser.add_argument("--paper-ledger-dir", default=str(DEFAULT_PAPER_LEDGER_DIR))
    parser.add_argument("--monitoring-dir", default=str(DEFAULT_MONITORING_DIR))
    parser.add_argument("--v5-closure-dir", default=str(DEFAULT_V5_CLOSURE_DIR))
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--schema-validator-dir", default=str(DEFAULT_SCHEMA_VALIDATOR_DIR))
    parser.add_argument("--dependency-validator-dir", default=str(DEFAULT_DEPENDENCY_VALIDATOR_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_reproducibility_rerun_validation_outputs(
            semi_auto_dir=args.semi_auto_dir,
            broker_research_dir=args.broker_research_dir,
            capital_dir=args.capital_dir,
            universe_dir=args.universe_dir,
            position_dir=args.position_dir,
            exit_dir=args.exit_dir,
            daily_plan_dir=args.daily_plan_dir,
            paper_ledger_dir=args.paper_ledger_dir,
            monitoring_dir=args.monitoring_dir,
            v5_closure_dir=args.v5_closure_dir,
            baseline_dir=args.baseline_dir,
            schema_validator_dir=args.schema_validator_dir,
            dependency_validator_dir=args.dependency_validator_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: reproducibility rerun validation failed: {exc}")
        sys.exit(1)

    summary = result["reproducibility_rerun_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 4 Historical Output Reproducibility / Rerun Consistency Check")
    print("-------------------------------------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Checked reruns: {summary['checked_rerun_count']}")
    print(f"Rerun passes: {summary['rerun_pass_count']}")
    print(f"Rerun warnings: {summary['rerun_warning_count']}")
    print(f"Rerun failures: {summary['rerun_fail_count']}")
    print(f"Checked files: {summary['checked_file_count']}")
    print(f"Matched files: {summary['matched_file_count']}")
    print(f"Mismatched files: {summary['mismatched_file_count']}")
    print(f"Forbidden true flags: {summary['forbidden_true_flag_count']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print("Scope: reproducibility rerun validation only")
    print()
    print("Warning: This is educational/research validation only, not financial advice.")
    print("No backtests, market data fetches, model training, broker connections, live trading, or order submissions are performed.")
    print("All V6 Step 4 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
