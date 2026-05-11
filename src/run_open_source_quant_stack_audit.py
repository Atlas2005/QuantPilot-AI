import argparse
import sys

try:
    from .open_source_quant_stack_audit import (
        DEFAULT_OUTPUT_DIR,
        generate_open_source_quant_stack_audit_outputs,
    )
except ImportError:
    from open_source_quant_stack_audit import (
        DEFAULT_OUTPUT_DIR,
        generate_open_source_quant_stack_audit_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V7 Step 1 research-only open-source quant stack audit.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_open_source_quant_stack_audit_outputs(output_dir=args.output_dir)
    except Exception as exc:
        print(f"Error: open-source quant stack audit failed: {exc}")
        sys.exit(1)

    summary = result["summary"].iloc[0]
    print("QuantPilot-AI V7 Step 1 Open-source Quant Stack Audit")
    print("-----------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Reviewed candidates: {summary['reviewed_candidate_count']}")
    print(f"Adopt directly: {summary['recommended_adopt_directly_count']}")
    print(f"Wrap and integrate: {summary['recommended_wrap_and_integrate_count']}")
    print(f"Evaluate with prototype: {summary['recommended_evaluate_with_prototype_count']}")
    print(f"Borrow architecture only: {summary['recommended_borrow_architecture_only_count']}")
    print(f"Defer until later: {summary['recommended_defer_until_later_count']}")
    print(f"Avoid for now: {summary['recommended_avoid_for_now_count']}")
    print(f"High license risk: {summary['high_license_risk_count']}")
    print(f"High integration complexity: {summary['high_integration_complexity_count']}")
    print(f"High live trading risk: {summary['high_live_trading_risk_count']}")
    print(f"V7 recommended stack count: {summary['v7_recommended_stack_count']}")
    print(f"Market data fetches: {summary['market_data_fetch_count']}")
    print(f"Broker connected count: {summary['broker_connected_count']}")
    print(f"Execution allowed count: {summary['execution_allowed_count']}")
    print(f"Live trading count: {summary['live_trading_count']}")
    print(f"Real order submission count: {summary['real_order_submission_count']}")
    print(f"Forbidden true flag count: {summary['forbidden_true_flag_count']}")
    print(f"Trading ready: {summary['trading_ready']}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Conclusion: {summary['conclusion']}")
    print(f"Recommended next step: {summary['recommended_next_step']}")
    print()
    print("Scope: framework selection and architecture decision only.")
    print("No packages are installed, no external frameworks are imported, no market data is fetched, no backtests are run, no models are trained, and no broker/order path is touched.")


if __name__ == "__main__":
    main()
