import argparse
import sys

try:
    from .bull_regime_failure_drilldown import (
        generate_bull_regime_failure_drilldown,
    )
except ImportError:
    from bull_regime_failure_drilldown import (
        generate_bull_regime_failure_drilldown,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bull-regime failure drilldown diagnostics.",
    )
    parser.add_argument(
        "--bull-dir",
        default="outputs/bull_regime_threshold_remediation_real_v1",
    )
    parser.add_argument(
        "--integrated-dir",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/bull_regime_failure_drilldown_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_bull_regime_failure_drilldown(
            bull_dir=args.bull_dir,
            integrated_dir=args.integrated_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: bull regime failure drilldown failed: {exc}")
        sys.exit(1)

    context = result["bull_threshold_context"].iloc[0]
    contribution = result["bull_failure_contribution"]
    negative = contribution[
        contribution["strategy_vs_benchmark_pct"].fillna(0) < 0
    ].head(3)
    top_symbols = ", ".join(negative["symbol"].astype(str).tolist()) or "none"
    print("QuantPilot-AI Bull Regime Failure Drilldown")
    print("--------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print(f"Aggregate bull decision: {context['final_decision']}")
    print(f"Average excess: {context['avg_strategy_vs_benchmark_pct']}")
    print("Main failure reason: bull_average_excess_slightly_negative")
    print(f"Top negative contributing symbols: {top_symbols}")
    print(f"Trade-level diagnostics available: {result['trade_level_available']}")
    print()
    print("Generated Files")
    print("---------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is educational research diagnostics only, not financial advice.")


if __name__ == "__main__":
    main()
