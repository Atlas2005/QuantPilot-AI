import argparse
import sys

try:
    from .capital_constraint_engine import generate_capital_constraint_outputs
except ImportError:
    from capital_constraint_engine import generate_capital_constraint_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 Step 1 capital feasibility checks for candidate BUY orders.",
    )
    parser.add_argument("--input-candidates", default=None)
    parser.add_argument("--cash", type=float, default=1000.0)
    parser.add_argument("--buffer", type=float, default=0.97)
    parser.add_argument("--default-lot-size", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="outputs/capital_constraint_engine_real_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_capital_constraint_outputs(
            input_candidates=args.input_candidates,
            cash=args.cash,
            buffer=args.buffer,
            default_lot_size=args.default_lot_size,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: capital constraint engine failed: {exc}")
        sys.exit(1)

    summary = result["capital_constraint_summary"].iloc[0]
    print("QuantPilot-AI V5 Step 1 Capital Constraint Engine")
    print("-------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Candidate count: {summary['candidate_count']}")
    print(f"Approved order count: {summary['approved_order_count']}")
    print(f"Rejected order count: {summary['rejected_order_count']}")
    print("Scope: capital feasibility only")
    print()
    print("Warning: This is educational/research tooling only, not financial advice.")
    print("No broker execution is performed. No order is trading-ready.")


if __name__ == "__main__":
    main()
