import argparse
import sys

try:
    from .validation_gate_failure_analysis import (
        save_validation_gate_failure_analysis,
    )
except ImportError:
    from validation_gate_failure_analysis import (
        save_validation_gate_failure_analysis,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze candidate validation gate failures and remediation priorities.",
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
        "--stress-dir",
        default="outputs/candidate_stress_real_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/validation_gate_failure_analysis_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = save_validation_gate_failure_analysis(
            gate_dir=args.gate_dir,
            revalidation_dir=args.revalidation_dir,
            stress_dir=args.stress_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: validation gate failure analysis failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Validation Gate Failure Analysis")
    print("----------------------------------------------")
    print(f"Gate directory: {args.gate_dir}")
    print(f"Canonical revalidation directory: {args.revalidation_dir}")
    print(f"Stress directory: {args.stress_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    if result.get("input_warnings"):
        print()
        print("Input Warnings")
        print("--------------")
        for warning in result["input_warnings"]:
            print(f"- {warning}")
    print()
    print("Warning: This is reporting-only research diagnostics, not financial advice.")


if __name__ == "__main__":
    main()
