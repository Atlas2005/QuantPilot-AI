import argparse
import sys

try:
    from .targeted_remediation_design import save_targeted_remediation_design
except ImportError:
    from targeted_remediation_design import save_targeted_remediation_design


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design targeted remediation experiments from validation gate failure analysis.",
    )
    parser.add_argument(
        "--failure-analysis-dir",
        default="outputs/validation_gate_failure_analysis_real_v1",
    )
    parser.add_argument(
        "--gate-dir",
        default="outputs/candidate_validation_gate_real_v1",
    )
    parser.add_argument(
        "--revalidation-dir",
        default="outputs/canonical_candidate_revalidation_real_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/targeted_remediation_design_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = save_targeted_remediation_design(
            failure_analysis_dir=args.failure_analysis_dir,
            gate_dir=args.gate_dir,
            revalidation_dir=args.revalidation_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: targeted remediation design failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Targeted Remediation Design")
    print("-----------------------------------------")
    print(f"Failure analysis directory: {args.failure_analysis_dir}")
    print(f"Gate directory: {args.gate_dir}")
    print(f"Canonical revalidation directory: {args.revalidation_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is reporting-only research diagnostics, not financial advice.")


if __name__ == "__main__":
    main()
