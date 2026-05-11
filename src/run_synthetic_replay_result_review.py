import argparse
import sys

try:
    from .synthetic_replay_result_review import (
        DEFAULT_DESIGN_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PRICE_PATH_DIR,
        DEFAULT_REPLAY_DIR,
        DEFAULT_REVIEW_DIR,
        generate_synthetic_replay_result_review_outputs,
    )
except ImportError:
    from synthetic_replay_result_review import (
        DEFAULT_DESIGN_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PRICE_PATH_DIR,
        DEFAULT_REPLAY_DIR,
        DEFAULT_REVIEW_DIR,
        generate_synthetic_replay_result_review_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V6 Step 12 research-only synthetic replay result review.",
    )
    parser.add_argument("--price-path-dir", default=str(DEFAULT_PRICE_PATH_DIR))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--design-dir", default=str(DEFAULT_DESIGN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_synthetic_replay_result_review_outputs(
            price_path_dir=args.price_path_dir,
            review_dir=args.review_dir,
            replay_dir=args.replay_dir,
            design_dir=args.design_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: synthetic replay result review failed: {exc}")
        sys.exit(1)

    summary = result["synthetic_replay_result_summary"].iloc[0]
    print("QuantPilot-AI V6 Step 12 Synthetic Replay Result Review")
    print("--------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Reviewed scenarios: {summary['reviewed_scenario_count']}")
    print(f"Stop-loss touches: {summary['stop_loss_touch_count']}")
    print(f"Take-profit touches: {summary['take_profit_touch_count']}")
    print(f"Max-holding/no-exit outcomes: {summary['max_holding_or_no_exit_count']}")
    print(f"Unresolved scenarios: {summary['unresolved_scenario_count']}")
    print(f"High-risk scenarios: {summary['high_risk_scenario_count']}")
    print(f"Synthetic exit events: {summary['synthetic_exit_event_count']}")
    print(f"Market data fetches: {summary['market_data_fetch_count']}")
    print(f"Broker connected count: {summary['broker_connected_count']}")
    print(f"Execution allowed count: {summary['execution_allowed_count']}")
    print(f"Live trading count: {summary['live_trading_count']}")
    print(f"Real order submission count: {summary['real_order_submission_count']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print(f"Recommended next step: {summary['recommended_next_step']}")
    print("Scope: synthetic replay result review and scenario risk classification only")
    print()
    print("Warning: This is educational/research review only, not financial advice.")
    print("Synthetic replay is not real market validation, broker paper trading, live evidence, or trading-ready evidence.")
    print("No market data fetch, backtest, model training, threshold change, broker connection, or order action is performed.")


if __name__ == "__main__":
    main()
