import argparse
import sys

try:
    from .bull_error_pattern_remediation_design import (
        generate_bull_error_pattern_remediation_design,
    )
except ImportError:
    from bull_error_pattern_remediation_design import (
        generate_bull_error_pattern_remediation_design,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify bull error patterns and design remediation options without implementing them.",
    )
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--drilldown-dir", default=None)
    parser.add_argument("--integrated-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_bull_error_pattern_remediation_design(
            diagnostics_dir=args.diagnostics_dir,
            drilldown_dir=args.drilldown_dir,
            integrated_dir=args.integrated_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: bull error pattern remediation design failed: {exc}")
        sys.exit(1)

    trades = result["bull_trade_error_classification"]
    symbols = result["bull_symbol_error_profile"]
    options = result["bull_remediation_design_options"]
    dominant = (
        trades["classified_error_pattern"].value_counts().head(3).to_dict()
        if not trades.empty and "classified_error_pattern" in trades
        else {}
    )
    affected_symbols = symbols["symbol"].astype(str).tolist() if not symbols.empty else []
    statuses = sorted(options["implementation_status"].astype(str).unique().tolist()) if not options.empty else []
    print("QuantPilot-AI Bull Error Pattern Remediation Design")
    print("---------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Generated Files")
    print("---------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(f"Dominant bull error patterns: {dominant if dominant else 'none'}")
    print(f"Affected symbols: {', '.join(affected_symbols) if affected_symbols else 'none'}")
    print(f"Remediation design status: {', '.join(statuses) if statuses else 'design_only_not_implemented'}")
    print()
    print("Warning: This is educational/research diagnostics only, not financial advice.")
    print("No remediation is implemented and no candidate is trading-ready.")


if __name__ == "__main__":
    main()
