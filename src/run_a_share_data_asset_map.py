import argparse
import sys

try:
    from .a_share_data_asset_map import (
        DEFAULT_DATA_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_STEP1_DIR,
        DEFAULT_V6_CLOSURE_DIR,
        generate_a_share_data_asset_map_outputs,
    )
except ImportError:
    from a_share_data_asset_map import (
        DEFAULT_DATA_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_STEP1_DIR,
        DEFAULT_V6_CLOSURE_DIR,
        generate_a_share_data_asset_map_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V7 Step 2 research-only A-share data asset map and source selection.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--step1-dir", default=str(DEFAULT_STEP1_DIR))
    parser.add_argument("--v6-closure-dir", default=str(DEFAULT_V6_CLOSURE_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_a_share_data_asset_map_outputs(
            output_dir=args.output_dir,
            step1_dir=args.step1_dir,
            v6_closure_dir=args.v6_closure_dir,
            data_dir=args.data_dir,
        )
    except Exception as exc:
        print(f"Error: A-share data asset map failed: {exc}")
        sys.exit(1)

    summary = result["summary"].iloc[0]
    print("QuantPilot-AI V7 Step 2 A-share Data Asset Map")
    print("------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Reviewed data sources: {summary['reviewed_data_source_count']}")
    print(f"Primary candidates: {summary['primary_candidate_count']}")
    print(f"Secondary candidates: {summary['secondary_candidate_count']}")
    print(f"Prototype required: {summary['prototype_required_count']}")
    print(f"Fixture only: {summary['fixture_only_count']}")
    print(f"Defer until paid stage: {summary['defer_until_paid_stage_count']}")
    print(f"Avoid for now: {summary['avoid_for_now_count']}")
    print(f"Minimum market-reality requirements: {summary['minimum_market_reality_requirement_count']}")
    print(f"Alpha research requirements: {summary['alpha_research_requirement_count']}")
    print(f"Blocking data gaps: {summary['blocking_data_gap_count']}")
    print(f"High commercial/license risk: {summary['high_commercial_license_risk_count']}")
    print(f"High stability risk: {summary['high_stability_risk_count']}")
    print(f"Token-required sources: {summary['token_required_source_count']}")
    print(f"Local storage recommendations: {summary['local_storage_recommendation_count']}")
    print(f"Market data fetches: {summary['market_data_fetch_count']}")
    print(f"External API calls: {summary['external_api_call_count']}")
    print(f"Package installs: {summary['package_install_count']}")
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
    print("Scope: data source selection and local asset mapping only.")
    print("No packages are installed, no external data frameworks are imported, no market data is fetched, no API is called, no token is used, and no broker/order path is touched.")


if __name__ == "__main__":
    main()
