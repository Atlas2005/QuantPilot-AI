import argparse
import sys

try:
    from .threshold_decision_report import save_threshold_decision_report
except ImportError:
    from threshold_decision_report import save_threshold_decision_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reduced feature threshold decision report.",
    )
    parser.add_argument(
        "--summary-dir",
        default="outputs/reduced_feature_threshold_summary_real_v1",
        help="Directory containing Step 23 threshold summary CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/threshold_decision_real_v1",
        help="Output directory for Step 24 decision report files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = save_threshold_decision_report(
            summary_dir=args.summary_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: failed to generate threshold decision report: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Threshold Decision Report")
    print("---------------------------------------")
    print(f"Summary directory: {args.summary_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: This is educational research only. "
        "It is not trading-ready and is not financial advice."
    )


if __name__ == "__main__":
    main()
