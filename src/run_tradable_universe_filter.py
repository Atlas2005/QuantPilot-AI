import argparse
import sys

try:
    from .tradable_universe_filter import generate_tradable_universe_outputs
except ImportError:
    from tradable_universe_filter import generate_tradable_universe_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 2 tradable universe eligibility filters for candidate BUY rows.",
    )
    parser.add_argument("--input-candidates", default=None)
    parser.add_argument("--cash", type=float, default=1000.0)
    parser.add_argument("--buffer", type=float, default=0.97)
    parser.add_argument("--default-lot-size", type=int, default=100)
    parser.add_argument("--min-turnover", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/tradable_universe_filter_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_tradable_universe_outputs(
            input_candidates=args.input_candidates,
            cash=args.cash,
            buffer=args.buffer,
            default_lot_size=args.default_lot_size,
            min_turnover=args.min_turnover,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: tradable universe filter failed: {exc}")
        sys.exit(1)

    summary = result["universe_filter_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 2 Tradable Universe Filter")
    print("------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Candidate count: {summary['candidate_count']}")
    print(f"Tradable count: {summary['tradable_count']}")
    print(f"Excluded count: {summary['excluded_count']}")
    print("Scope: tradable universe eligibility only")
    print()
    print("Warning: This is educational/research tooling only, not financial advice.")
    print("No broker execution is performed. No candidate is trading-ready.")


if __name__ == "__main__":
    main()
