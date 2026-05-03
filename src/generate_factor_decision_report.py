import argparse
import sys

try:
    from .factor_decision_report import write_factor_decision_report
except ImportError:
    from factor_decision_report import write_factor_decision_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown factor selection decision report.",
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/factor_ablation_demo",
        help="Directory containing factor ablation output CSVs.",
    )
    parser.add_argument(
        "--output",
        default="outputs/factor_ablation_demo/factor_decision_report.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = write_factor_decision_report(args.input_dir, args.output)
    except Exception as exc:
        print(f"Error: failed to generate factor decision report: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Factor Decision Report")
    print("------------------------------------")
    print(f"Input directory: {args.input_dir}")
    print(f"Report path: {result['report_path']}")
    print(f"Decision rows: {len(result['decision_summary'])}")
    print()
    print(
        "Warning: This report supports research feature decisions only. "
        "It is not financial advice."
    )


if __name__ == "__main__":
    main()
