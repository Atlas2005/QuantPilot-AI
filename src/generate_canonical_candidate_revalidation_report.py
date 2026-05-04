import argparse
import sys

try:
    from .canonical_candidate_revalidation_report import (
        save_canonical_candidate_revalidation_report,
    )
except ImportError:
    from canonical_candidate_revalidation_report import (
        save_canonical_candidate_revalidation_report,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical candidate revalidation decision report.",
    )
    parser.add_argument(
        "--expanded-validation-dir",
        default="outputs/candidate_expanded_validation_real_v2",
    )
    parser.add_argument(
        "--stress-dir",
        default="outputs/candidate_stress_real_v2",
    )
    parser.add_argument(
        "--threshold-decision-dir",
        default="outputs/threshold_decision_real_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/canonical_candidate_revalidation_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = save_canonical_candidate_revalidation_report(
            expanded_validation_dir=args.expanded_validation_dir,
            stress_dir=args.stress_dir,
            threshold_decision_dir=args.threshold_decision_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: canonical candidate revalidation report failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Canonical Candidate Revalidation Report")
    print("-----------------------------------------------------")
    print(f"Expanded validation directory: {args.expanded_validation_dir}")
    print(f"Stress directory: {args.stress_dir}")
    print(f"Threshold decision directory: {args.threshold_decision_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is reporting-only research control, not financial advice.")


if __name__ == "__main__":
    main()
