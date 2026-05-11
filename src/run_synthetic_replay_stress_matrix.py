import argparse
import sys

try:
    from .synthetic_replay_stress_matrix import (
        DEFAULT_HARDENING_REVIEW_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PRICE_PATH_DIR,
        DEFAULT_REPLAY_DIR,
        DEFAULT_RESULT_REVIEW_DIR,
        generate_synthetic_replay_stress_matrix_outputs,
    )
except ImportError:
    from synthetic_replay_stress_matrix import (
        DEFAULT_HARDENING_REVIEW_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PRICE_PATH_DIR,
        DEFAULT_REPLAY_DIR,
        DEFAULT_RESULT_REVIEW_DIR,
        generate_synthetic_replay_stress_matrix_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 13 research-only synthetic replay stress matrix design.",
    )
    parser.add_argument("--price-path-dir", default=str(DEFAULT_PRICE_PATH_DIR))
    parser.add_argument("--result-review-dir", default=str(DEFAULT_RESULT_REVIEW_DIR))
    parser.add_argument("--hardening-review-dir", default=str(DEFAULT_HARDENING_REVIEW_DIR))
    parser.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_synthetic_replay_stress_matrix_outputs(
            price_path_dir=args.price_path_dir,
            result_review_dir=args.result_review_dir,
            hardening_review_dir=args.hardening_review_dir,
            replay_dir=args.replay_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: synthetic replay stress matrix failed: {exc}")
        sys.exit(1)

    summary = result["synthetic_replay_stress_matrix_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 13 Synthetic Replay Stress Matrix")
    print("--------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Reviewed inputs: {summary['reviewed_input_count']}")
    print(f"Missing inputs: {summary['missing_input_count']}")
    print(f"Existing scenarios: {summary['existing_scenario_count']}")
    print(f"Existing high-risk scenarios: {summary['existing_high_risk_scenario_count']}")
    print(f"Proposed stress dimensions: {summary['proposed_stress_dimension_count']}")
    print(f"Proposed stress scenarios: {summary['proposed_stress_scenario_count']}")
    print(f"Expansion plan rows: {summary['expansion_plan_row_count']}")
    print(f"High-priority expansions: {summary['high_priority_expansion_count']}")
    print(f"Market data fetches: {summary['market_data_fetch_count']}")
    print(f"Broker connected count: {summary['broker_connected_count']}")
    print(f"Execution allowed count: {summary['execution_allowed_count']}")
    print(f"Live trading count: {summary['live_trading_count']}")
    print(f"Real order submission count: {summary['real_order_submission_count']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print(f"Recommended next step: {summary['recommended_next_step']}")
    print("Scope: synthetic replay stress matrix design only")
    print()
    print("Warning: This is educational/research planning only, not financial advice.")
    print("No new price simulation, backtest, market data fetch, live data, broker connection, threshold change, model training, or order action is performed.")
    print("All V6 Step 13 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
