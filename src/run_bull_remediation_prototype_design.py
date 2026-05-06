import argparse
import sys

try:
    from .bull_remediation_prototype_design import (
        generate_bull_remediation_prototype_design,
    )
except ImportError:
    from bull_remediation_prototype_design import (
        generate_bull_remediation_prototype_design,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design bull remediation prototype experiments without executing them.",
    )
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--integrated-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_bull_remediation_prototype_design(
            design_dir=args.design_dir,
            diagnostics_dir=args.diagnostics_dir,
            integrated_dir=args.integrated_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: bull remediation prototype design failed: {exc}")
        sys.exit(1)

    specs = result["bull_prototype_experiment_specs"]
    ranking = result["bull_prototype_priority_ranking"]
    statuses = sorted(specs["implementation_status"].astype(str).unique().tolist()) if not specs.empty else []
    top = ranking.head(3)["target"].astype(str).tolist() if not ranking.empty else []
    print("QuantPilot-AI Bull Remediation Prototype Design")
    print("------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print(f"Prototype count: {len(specs)}")
    print(f"Highest priority prototypes: {', '.join(top) if top else 'none'}")
    print(f"Implementation status: {', '.join(statuses) if statuses else 'prototype_design_only'}")
    print()
    print("Generated Files")
    print("---------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is educational/research diagnostics only, not financial advice.")
    print("No prototype is executed and no candidate is trading-ready.")


if __name__ == "__main__":
    main()
