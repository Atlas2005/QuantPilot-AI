import argparse
import sys

try:
    from .replay_price_path_simulator import (
        DEFAULT_DESIGN_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_REPLAY_DIR,
        DEFAULT_REVIEW_DIR,
        generate_replay_price_path_simulator_outputs,
    )
except ImportError:
    from replay_price_path_simulator import (
        DEFAULT_DESIGN_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_REPLAY_DIR,
        DEFAULT_REVIEW_DIR,
        generate_replay_price_path_simulator_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 11 research-only local synthetic replay price path simulator.",
    )
    parser.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--design-dir", default=str(DEFAULT_DESIGN_DIR))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_replay_price_path_simulator_outputs(
            replay_dir=args.replay_dir,
            design_dir=args.design_dir,
            review_dir=args.review_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: replay price path simulator failed: {exc}")
        sys.exit(1)

    summary = result["replay_price_path_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 11 Multi-Day Replay Price Path Simulator")
    print("----------------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Input dependencies: {summary['input_dependency_count']}")
    print(f"Missing input dependencies: {summary['missing_input_dependency_count']}")
    print(f"Scenarios: {summary['scenario_count']}")
    print(f"Price path rows: {summary['price_path_row_count']}")
    print(f"Scenario-position results: {summary['position_scenario_result_count']}")
    print(f"Synthetic exit events: {summary['synthetic_exit_event_count']}")
    print(f"Stop-loss touches: {summary['stop_loss_touch_result_count']}")
    print(f"Take-profit touches: {summary['take_profit_touch_result_count']}")
    print(f"Max-holding/no-exit results: {summary['max_holding_or_no_exit_result_count']}")
    print(f"Market data fetches: {summary['market_data_fetch_count']}")
    print(f"Broker connected count: {summary['broker_connected_count']}")
    print(f"Execution allowed count: {summary['execution_allowed_count']}")
    print(f"Live trading count: {summary['live_trading_count']}")
    print(f"Real order submission count: {summary['real_order_submission_count']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print(f"Recommended next step: {summary['recommended_next_step']}")
    print("Scope: local synthetic price path scenario layer only")
    print()
    print("Warning: This is educational/research scenario analysis only, not financial advice.")
    print("No historical backtest, market data fetch, live data, broker connection, model training, threshold change, or order action is performed.")
    print("All V6 Step 11 outputs preserve trading_ready=False.")


if __name__ == "__main__":
    main()
