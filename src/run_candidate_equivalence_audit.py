import argparse
import sys

try:
    from .candidate_equivalence_audit import (
        parse_symbols,
        save_candidate_equivalence_audit,
    )
except ImportError:
    from candidate_equivalence_audit import (
        parse_symbols,
        save_candidate_equivalence_audit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit candidate pruning-mode feature set equivalence.",
    )
    parser.add_argument(
        "--factor-dir",
        default="outputs/model_robustness_real_v2/factors",
    )
    parser.add_argument(
        "--symbols",
        default="000001,600519,000858,600036,601318",
    )
    parser.add_argument(
        "--recommendations",
        default="outputs/feature_ablation_real_v1/feature_pruning_recommendations.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/candidate_equivalence_real_v1",
    )
    parser.add_argument("--target-col", default="label_up_5d")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        symbols = parse_symbols(args.symbols)
        result = save_candidate_equivalence_audit(
            factor_dir=args.factor_dir,
            symbols=symbols,
            recommendations_path=args.recommendations,
            output_dir=args.output_dir,
            target_col=args.target_col,
        )
    except Exception as exc:
        print(f"Error: candidate equivalence audit failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Candidate Equivalence Audit")
    print("-----------------------------------------")
    print(f"Factor directory: {args.factor_dir}")
    print(f"Symbols: {symbols}")
    print(f"Recommendations: {args.recommendations}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: This is an audit/reporting step only. "
        "It is not a trading recommendation."
    )


if __name__ == "__main__":
    main()
