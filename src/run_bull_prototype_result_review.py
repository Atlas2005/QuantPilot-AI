import argparse
import sys

try:
    from .bull_prototype_result_review import generate_bull_prototype_result_review
except ImportError:
    from bull_prototype_result_review import generate_bull_prototype_result_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review Step 42 bull prototype results and close V4 research diagnostics conservatively.",
    )
    parser.add_argument("--controlled-backtest-dir", required=True)
    parser.add_argument("--integrated-dir", default=None)
    parser.add_argument("--error-design-dir", default=None)
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_bull_prototype_result_review(
            controlled_backtest_dir=args.controlled_backtest_dir,
            integrated_dir=args.integrated_dir,
            error_design_dir=args.error_design_dir,
            diagnostics_dir=args.diagnostics_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: bull prototype result review failed: {exc}")
        sys.exit(1)

    summary = result["bull_prototype_review_summary"]
    selection = result["bull_candidate_selection"]
    allowed = int(summary["reviewed_can_advance_to_further_validation"].fillna(False).astype(bool).sum()) if not summary.empty else 0
    selection_status = selection.iloc[0].get("selection_status") if not selection.empty else "no_candidate_selected"
    print("QuantPilot-AI Bull Prototype Result Review")
    print("------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Total prototypes reviewed: {len(summary)}")
    print(f"Prototypes allowed to advance: {allowed}")
    print("Final bull remediation status: bull_remediation_unresolved")
    print("V4 closure recommendation: close_v4_as_research_diagnostics_and_transition_to_v5")
    print(f"Candidate selection status: {selection_status}")
    print()
    print("Warning: This is educational/research diagnostics only, not financial advice.")
    print("No candidate is trading-ready.")


if __name__ == "__main__":
    main()
