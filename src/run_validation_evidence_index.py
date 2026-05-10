import argparse
import sys

try:
    from .validation_evidence_index import (
        DEFAULT_BASELINE_DIR,
        DEFAULT_DEPENDENCY_VALIDATOR_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_RERUN_VALIDATOR_DIR,
        DEFAULT_SCHEMA_VALIDATOR_DIR,
        DEFAULT_WARNING_TRIAGE_DIR,
        generate_validation_evidence_index_outputs,
    )
except ImportError:
    from validation_evidence_index import (
        DEFAULT_BASELINE_DIR,
        DEFAULT_DEPENDENCY_VALIDATOR_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_RERUN_VALIDATOR_DIR,
        DEFAULT_SCHEMA_VALIDATOR_DIR,
        DEFAULT_WARNING_TRIAGE_DIR,
        generate_validation_evidence_index_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 6 research-only validation evidence index.",
    )
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--schema-validator-dir", default=str(DEFAULT_SCHEMA_VALIDATOR_DIR))
    parser.add_argument("--dependency-validator-dir", default=str(DEFAULT_DEPENDENCY_VALIDATOR_DIR))
    parser.add_argument("--rerun-validator-dir", default=str(DEFAULT_RERUN_VALIDATOR_DIR))
    parser.add_argument("--warning-triage-dir", default=str(DEFAULT_WARNING_TRIAGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_validation_evidence_index_outputs(
            baseline_dir=args.baseline_dir,
            schema_validator_dir=args.schema_validator_dir,
            dependency_validator_dir=args.dependency_validator_dir,
            rerun_validator_dir=args.rerun_validator_dir,
            warning_triage_dir=args.warning_triage_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: validation evidence index failed: {exc}")
        sys.exit(1)

    summary = result["validation_evidence_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 6 Validation Evidence Index / Audit Trail Catalog")
    print("------------------------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Indexed steps: {summary['indexed_step_count']} / {summary['expected_step_count']}")
    print(f"Indexed evidence files: {summary['indexed_evidence_file_count']}")
    print(f"Missing required evidence: {summary['missing_required_evidence_count']}")
    print(f"Forbidden true flags: {summary['forbidden_true_flag_count']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print("Scope: evidence index only")
    print()
    print("Warning: This is educational/research indexing only, not financial advice.")
    print("No backtests, market data fetches, model training, broker connections, live trading, or order submissions are performed.")
    print("All V6 Step 6 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
