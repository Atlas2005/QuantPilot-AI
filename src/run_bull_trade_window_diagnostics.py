import argparse
import sys

try:
    from .bull_trade_window_diagnostics import (
        DEFAULT_BUY_THRESHOLD,
        DEFAULT_CANDIDATE,
        DEFAULT_MODEL,
        DEFAULT_SELL_THRESHOLD,
        generate_bull_trade_window_diagnostics,
    )
except ImportError:
    from bull_trade_window_diagnostics import (
        DEFAULT_BUY_THRESHOLD,
        DEFAULT_CANDIDATE,
        DEFAULT_MODEL,
        DEFAULT_SELL_THRESHOLD,
        generate_bull_trade_window_diagnostics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bull trade/window diagnostics for the selected Step 34 threshold.",
    )
    parser.add_argument(
        "--bull-dir",
        default="outputs/bull_regime_threshold_remediation_real_v1",
    )
    parser.add_argument("--drilldown-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/bull_trade_window_diagnostics_real_v1",
    )
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--buy-threshold", type=float, default=DEFAULT_BUY_THRESHOLD)
    parser.add_argument("--sell-threshold", type=float, default=DEFAULT_SELL_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_bull_trade_window_diagnostics(
            bull_dir=args.bull_dir,
            drilldown_dir=args.drilldown_dir,
            output_dir=args.output_dir,
            candidate=args.candidate,
            model=args.model,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )
    except Exception as exc:
        print(f"Error: bull trade/window diagnostics failed: {exc}")
        sys.exit(1)

    config = result["run_config"]
    summary = result["bull_symbol_window_summary"]
    main_drags = []
    if not summary.empty and "likely_failure_mechanism" in summary:
        main_drags = summary.loc[
            summary["likely_failure_mechanism"] == "main_bull_average_drag_identified_in_step37",
            "symbol",
        ].astype(str).tolist()
    print("QuantPilot-AI Bull Trade/Window Diagnostics")
    print("--------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print(f"Real trade-level diagnostics generated: {config['trade_level_available']}")
    print(f"Date/window diagnostics generated: {config['timeline_available']} / {config['window_available']}")
    print(f"Symbols covered: {', '.join(config['symbols'])}")
    print(f"Main drag symbols: {', '.join(main_drags) if main_drags else 'none'}")
    print()
    print("Generated Files")
    print("---------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is educational research diagnostics only, not financial advice.")


if __name__ == "__main__":
    main()
