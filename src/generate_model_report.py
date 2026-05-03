import argparse
import sys

try:
    from .model_report_generator import write_model_robustness_report
except ImportError:
    from model_report_generator import write_model_robustness_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report from model robustness outputs.",
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/model_robustness_demo",
        help="Directory containing model robustness CSV/JSON outputs.",
    )
    parser.add_argument(
        "--output",
        default="reports/model_robustness_report.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report_path, _ = write_model_robustness_report(
            input_dir=args.input_dir,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"Report generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Model robustness report generated.")
    print(f"Report path: {report_path}")


if __name__ == "__main__":
    main()
