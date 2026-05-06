import argparse
import sys

try:
    from .integrated_remediation_revalidation import (
        generate_integrated_remediation_revalidation,
    )
except ImportError:
    from integrated_remediation_revalidation import (
        generate_integrated_remediation_revalidation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run integrated remediation revalidation diagnostics.",
    )
    parser.add_argument(
        "--bull-dir",
        default="outputs/bull_regime_threshold_remediation_real_v1",
    )
    parser.add_argument(
        "--sideways-dir",
        default="outputs/sideways_regime_trade_sufficiency_remediation_real_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/integrated_remediation_revalidation_real_v1",
    )
    parser.add_argument("--validation-gate-dir", default=None)
    parser.add_argument("--failure-analysis-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_integrated_remediation_revalidation(
            bull_dir=args.bull_dir,
            sideways_dir=args.sideways_dir,
            output_dir=args.output_dir,
            validation_gate_dir=args.validation_gate_dir,
            failure_analysis_dir=args.failure_analysis_dir,
        )
    except Exception as exc:
        print(f"Error: integrated remediation revalidation failed: {exc}")
        sys.exit(1)

    summary = result["integrated_remediation_summary"].iloc[0]
    print("QuantPilot-AI Integrated Remediation Revalidation")
    print("------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print(f"Final integrated decision: {summary['overall_decision']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Main blocker: {summary['main_blocker']}")
    print()
    print("Generated Files")
    print("---------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is educational research diagnostics only, not financial advice.")


if __name__ == "__main__":
    main()
