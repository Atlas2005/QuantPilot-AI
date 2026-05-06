import argparse
import sys

try:
    from .bull_prototype_experiment_harness import (
        generate_bull_prototype_experiment_harness,
    )
except ImportError:
    from bull_prototype_experiment_harness import (
        generate_bull_prototype_experiment_harness,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a bull prototype experiment harness dry-run without executing prototypes.",
    )
    parser.add_argument("--prototype-design-dir", required=True)
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--integrated-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_bull_prototype_experiment_harness(
            prototype_design_dir=args.prototype_design_dir,
            diagnostics_dir=args.diagnostics_dir,
            integrated_dir=args.integrated_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: bull prototype experiment harness failed: {exc}")
        sys.exit(1)

    registry = result["bull_prototype_registry"]
    validation = result["bull_prototype_config_validation"]
    failed = (
        int((validation["status"].astype(str) == "failed").sum())
        if not validation.empty and "status" in validation
        else 0
    )
    print("QuantPilot-AI Bull Prototype Experiment Harness")
    print("------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print(f"Registered prototypes: {len(registry)}")
    print(f"Validation failures: {failed}")
    print("Execution status: not_executed")
    print()
    print("Generated Files")
    print("---------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is educational/research diagnostics only, not financial advice.")
    print("No prototype backtest is executed and no candidate is trading-ready.")


if __name__ == "__main__":
    main()
