import argparse
import sys

try:
    from .bull_prototype_controlled_backtest import (
        generate_bull_prototype_controlled_backtest,
    )
except ImportError:
    from bull_prototype_controlled_backtest import (
        generate_bull_prototype_controlled_backtest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled research-only bull prototype simulations against unchanged diagnostics.",
    )
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--prototype-design-dir", required=True)
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--integrated-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_bull_prototype_controlled_backtest(
            harness_dir=args.harness_dir,
            prototype_design_dir=args.prototype_design_dir,
            diagnostics_dir=args.diagnostics_dir,
            integrated_dir=args.integrated_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: bull prototype controlled backtest failed: {exc}")
        sys.exit(1)
    results = result["bull_prototype_execution_results"]
    executed = int((results["execution_status"] == "executed").sum()) if not results.empty else 0
    not_exec = int((results["execution_status"] == "not_executable_with_current_data").sum()) if not results.empty else 0
    baseline = results["baseline_avg_excess_pct"].dropna().iloc[0] if not results.empty and results["baseline_avg_excess_pct"].notna().any() else "unavailable"
    best_summary = result.get("best_avg_excess_summary", {})
    best = best_summary.get("best_diagnostic_candidate", "no_avg_excess_improvement")
    summary = results["conservative_result"].value_counts().to_dict() if not results.empty else {}
    print("QuantPilot-AI Bull Prototype Controlled Backtest")
    print("------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print(f"Executed prototype count: {executed}")
    print(f"Not-executable prototype count: {not_exec}")
    print(f"Baseline avg excess: {baseline}")
    print(f"Best diagnostic candidate by avg excess: {best}")
    print(f"Best diagnostic candidate reason: {best_summary.get('reason', 'unavailable')}")
    print(f"Conservative decision summary: {summary}")
    print()
    print("Generated Files")
    print("---------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is educational/research diagnostics only, not financial advice.")
    print("No candidate is trading-ready.")


if __name__ == "__main__":
    main()
