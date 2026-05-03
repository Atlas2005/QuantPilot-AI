import argparse
import sys

try:
    from .pruning_summary_report import (
        DEFAULT_INPUT_DIRS,
        parse_input_dirs,
        save_pruning_summary_report,
    )
except ImportError:
    from pruning_summary_report import (
        DEFAULT_INPUT_DIRS,
        parse_input_dirs,
        save_pruning_summary_report,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-symbol factor pruning summary report.",
    )
    parser.add_argument(
        "--input-dirs",
        default=",".join(DEFAULT_INPUT_DIRS),
        help="Comma-separated pruning output directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pruning_summary_real_v1",
        help="Output directory for summary CSVs and Markdown report.",
    )
    parser.add_argument(
        "--report-name",
        default="pruning_summary_report.md",
        help="Markdown report filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        input_dirs = parse_input_dirs(args.input_dirs)
        result = save_pruning_summary_report(
            input_dirs=input_dirs,
            output_dir=args.output_dir,
            report_name=args.report_name,
        )
    except Exception as exc:
        print(f"Error: failed to generate pruning summary report: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Pruning Summary Report")
    print("------------------------------------")
    print(f"Input directories: {input_dirs}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: This is educational pruning research only. "
        "It is not financial advice."
    )


if __name__ == "__main__":
    main()
