import argparse
import sys

try:
    from .reproducibility_warning_triage import (
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        generate_reproducibility_warning_triage_outputs,
    )
except ImportError:
    from reproducibility_warning_triage import (
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        generate_reproducibility_warning_triage_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 5 research-only reproducibility warning triage.",
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_reproducibility_warning_triage_outputs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: reproducibility warning triage failed: {exc}")
        sys.exit(1)

    summary = result["reproducibility_warning_triage_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 5 Validation Warning Triage / Normalization Report")
    print("-------------------------------------------------------------------------")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Warning rows: {summary['total_warning_row_count']}")
    print(f"Acceptable warnings: {summary['acceptable_warning_count']}")
    print(f"Needs investigation: {summary['needs_investigation_count']}")
    print(f"Forbidden true flags: {summary['forbidden_true_flag_count']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print("Scope: warning triage only")
    print()
    print("Warning: This is educational/research triage only, not financial advice.")
    print("No backtests, market data fetches, model training, broker connections, live trading, or order submissions are performed.")
    print("All V6 Step 5 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
