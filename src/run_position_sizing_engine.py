import argparse
import sys

try:
    from .position_sizing_engine import (
        DEFAULT_INPUT_PATH,
        generate_position_sizing_outputs,
    )
except ImportError:
    from position_sizing_engine import (
        DEFAULT_INPUT_PATH,
        generate_position_sizing_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 3 research-only position sizing for Step 2 tradable candidates.",
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--cash", type=float, default=1000.0)
    parser.add_argument("--buffer", type=float, default=0.97)
    parser.add_argument("--default-lot-size", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="outputs/position_sizing_engine_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_position_sizing_outputs(
            input_path=args.input_path,
            available_cash=args.cash,
            usable_cash_buffer=args.buffer,
            default_lot_size=args.default_lot_size,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: position sizing engine failed: {exc}")
        sys.exit(1)

    summary = result["position_sizing_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 3 Position Sizing Engine")
    print("-----------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Candidate count: {summary['candidate_count']}")
    print(f"Sized position count: {summary['sized_position_count']}")
    print(f"Deferred position count: {summary['deferred_position_count']}")
    print(f"Rejected position count: {summary['rejected_position_count']}")
    print("Scope: position sizing only")
    print()
    print("Warning: This is educational/research tooling only, not financial advice.")
    print("No broker execution is performed. No position is trading-ready.")


if __name__ == "__main__":
    main()
