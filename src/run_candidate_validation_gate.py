import argparse
import sys

try:
    from .candidate_validation_gate import save_candidate_validation_gate
except ImportError:
    from candidate_validation_gate import save_candidate_validation_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the strict canonical candidate validation gate.",
    )
    parser.add_argument(
        "--revalidation-dir",
        default="outputs/canonical_candidate_revalidation_real_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/candidate_validation_gate_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = save_candidate_validation_gate(
            revalidation_dir=args.revalidation_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: candidate validation gate failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Candidate Validation Gate")
    print("---------------------------------------")
    print(f"Canonical revalidation directory: {args.revalidation_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is educational research diagnostics only, not financial advice.")


if __name__ == "__main__":
    main()
