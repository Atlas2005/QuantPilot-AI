import argparse
import sys

try:
    from .candidate_mode_normalization import save_canonical_mode_report
except ImportError:
    from candidate_mode_normalization import save_canonical_mode_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical candidate mode aliases from equivalence audit outputs.",
    )
    parser.add_argument(
        "--equivalence-dir",
        default="outputs/candidate_equivalence_real_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/candidate_mode_normalization_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = save_canonical_mode_report(args.equivalence_dir, args.output_dir)
    except Exception as exc:
        print(f"Error: candidate mode normalization failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Candidate Mode Normalization")
    print("------------------------------------------")
    print(f"Equivalence directory: {args.equivalence_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is reporting cleanup only, not a trading recommendation.")


if __name__ == "__main__":
    main()
