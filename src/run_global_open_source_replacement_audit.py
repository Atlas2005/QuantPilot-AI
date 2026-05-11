import argparse
import sys

try:
    from .global_open_source_replacement_audit import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PROJECT_ROOT,
        DEFAULT_STEP1_DIR,
        DEFAULT_STEP2_DIR,
        DEFAULT_V6_CLOSURE_DIR,
        generate_global_open_source_replacement_audit_outputs,
    )
except ImportError:
    from global_open_source_replacement_audit import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PROJECT_ROOT,
        DEFAULT_STEP1_DIR,
        DEFAULT_STEP2_DIR,
        DEFAULT_V6_CLOSURE_DIR,
        generate_global_open_source_replacement_audit_outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V7 Step 2.5 research-only global open-source replacement audit.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--step1-dir", default=str(DEFAULT_STEP1_DIR))
    parser.add_argument("--step2-dir", default=str(DEFAULT_STEP2_DIR))
    parser.add_argument("--v6-closure-dir", default=str(DEFAULT_V6_CLOSURE_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_global_open_source_replacement_audit_outputs(
            output_dir=args.output_dir,
            project_root=args.project_root,
            step1_dir=args.step1_dir,
            step2_dir=args.step2_dir,
            v6_closure_dir=args.v6_closure_dir,
        )
    except Exception as exc:
        print(f"Error: global open-source replacement audit failed: {exc}")
        sys.exit(1)

    summary = result["summary"].iloc[0]
    print("QuantPilot-AI V7 Step 2.5 Global Open-source Replacement Audit")
    print("----------------------------------------------------------------")
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Reviewed module areas: {summary['reviewed_module_area_count']}")
    print(f"Reviewed open-source candidates: {summary['reviewed_open_source_candidate_count']}")
    print(f"Keep as core: {summary['keep_as_core_count']}")
    print(f"Keep as guardrail: {summary['keep_as_guardrail_count']}")
    print(f"Keep as fixture/regression test: {summary['keep_as_fixture_or_regression_test_count']}")
    print(f"Wrap open source: {summary['wrap_open_source_count']}")
    print(f"Replace with open source: {summary['replace_with_open_source_count']}")
    print(f"Borrow architecture only: {summary['borrow_architecture_only_count']}")
    print(f"Defer until foundation ready: {summary['defer_until_foundation_ready_count']}")
    print(f"Deprecate later: {summary['deprecate_later_count']}")
    print(f"High replacement priority: {summary['high_replacement_priority_count']}")
    print(f"High license risk: {summary['high_license_risk_count']}")
    print(f"High integration complexity: {summary['high_integration_complexity_count']}")
    print(f"High agent overengineering risk: {summary['high_agent_overengineering_risk_count']}")
    print(f"Profitability alignment issues: {summary['profitability_alignment_issue_count']}")
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
    print("Scope: architecture reassessment only.")
    print("No packages are installed, no frameworks are imported, no market data is fetched, no API is called, no backtests or models are run, and no broker/order path is touched.")


if __name__ == "__main__":
    main()
