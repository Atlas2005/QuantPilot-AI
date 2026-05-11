import argparse
import sys

try:
    from .synthetic_stress_scenario_generator import (
        DEFAULT_HARDENING_REVIEW_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PRICE_PATH_DIR,
        DEFAULT_RESULT_REVIEW_DIR,
        DEFAULT_STRESS_MATRIX_DIR,
        generate_synthetic_stress_scenario_generator_outputs,
    )
except ImportError:
    from synthetic_stress_scenario_generator import (
        DEFAULT_HARDENING_REVIEW_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PRICE_PATH_DIR,
        DEFAULT_RESULT_REVIEW_DIR,
        DEFAULT_STRESS_MATRIX_DIR,
        generate_synthetic_stress_scenario_generator_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 14 research-only local synthetic stress scenario generator.",
    )
    parser.add_argument("--price-path-dir", default=str(DEFAULT_PRICE_PATH_DIR))
    parser.add_argument("--result-review-dir", default=str(DEFAULT_RESULT_REVIEW_DIR))
    parser.add_argument("--stress-matrix-dir", default=str(DEFAULT_STRESS_MATRIX_DIR))
    parser.add_argument("--hardening-review-dir", default=str(DEFAULT_HARDENING_REVIEW_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_synthetic_stress_scenario_generator_outputs(
            price_path_dir=args.price_path_dir,
            result_review_dir=args.result_review_dir,
            stress_matrix_dir=args.stress_matrix_dir,
            hardening_review_dir=args.hardening_review_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: synthetic stress scenario generator failed: {exc}")
        sys.exit(1)

    summary = result["synthetic_stress_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 14 Synthetic Stress Scenario Generator")
    print("----------------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Reviewed inputs: {summary['reviewed_input_count']}")
    print(f"Missing inputs: {summary['missing_input_count']}")
    print(f"Source stress dimensions: {summary['source_stress_dimension_count']}")
    print(f"Scenario definitions: {summary['generated_scenario_definition_count']}")
    print(f"Price path assumptions: {summary['generated_price_path_assumption_count']}")
    print(f"Execution plan rows: {summary['execution_plan_row_count']}")
    print(f"High-priority scenarios: {summary['high_priority_scenario_count']}")
    print(f"Local synthetic only rows: {summary['local_synthetic_only_count']}")
    print(f"Not-real-market evidence rows: {summary['not_real_market_evidence_count']}")
    print(f"Market data fetches: {summary['market_data_fetch_count']}")
    print(f"Broker connected count: {summary['broker_connected_count']}")
    print(f"Execution allowed count: {summary['execution_allowed_count']}")
    print(f"Live trading count: {summary['live_trading_count']}")
    print(f"Real order submission count: {summary['real_order_submission_count']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print(f"Recommended next step: {summary['recommended_next_step']}")
    print("Scope: local synthetic scenario definitions only")
    print()
    print("Warning: This is educational/research infrastructure only, not financial advice.")
    print("No new backtest, market data fetch, live data, broker connection, model training, threshold change, or order action is performed.")
    print("All V6 Step 14 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
