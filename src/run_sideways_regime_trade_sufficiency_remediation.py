import argparse
import sys

try:
    from .sideways_regime_trade_sufficiency_remediation import (
        DEFAULT_BUY_THRESHOLDS,
        DEFAULT_SELL_THRESHOLDS,
        parse_symbols,
        parse_threshold_list,
        save_sideways_regime_trade_sufficiency_remediation,
    )
except ImportError:
    from sideways_regime_trade_sufficiency_remediation import (
        DEFAULT_BUY_THRESHOLDS,
        DEFAULT_SELL_THRESHOLDS,
        parse_symbols,
        parse_threshold_list,
        save_sideways_regime_trade_sufficiency_remediation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sideways-regime trade sufficiency remediation diagnostics.",
    )
    parser.add_argument("--factor-dir", default="outputs/model_robustness_real_v2/factors")
    parser.add_argument("--symbols", default="000001,600519,000858,600036,601318")
    parser.add_argument(
        "--recommendations",
        default="outputs/feature_ablation_real_v1/feature_pruning_recommendations.csv",
    )
    parser.add_argument(
        "--failure-analysis-dir",
        default="outputs/validation_gate_failure_analysis_real_v1",
    )
    parser.add_argument(
        "--targeted-design-dir",
        default="outputs/targeted_remediation_design_real_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sideways_regime_trade_sufficiency_remediation_real_v1",
    )
    parser.add_argument("--target-col", default="label_up_5d")
    parser.add_argument("--canonical-mode", default="canonical_reduced_40")
    parser.add_argument("--model-type", default="logistic_regression")
    parser.add_argument(
        "--buy-thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_BUY_THRESHOLDS),
    )
    parser.add_argument(
        "--sell-thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_SELL_THRESHOLDS),
    )
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.001)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--min-trades", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        symbols = parse_symbols(args.symbols)
        result = save_sideways_regime_trade_sufficiency_remediation(
            factor_dir=args.factor_dir,
            symbols=symbols,
            recommendations_path=args.recommendations,
            output_dir=args.output_dir,
            failure_analysis_dir=args.failure_analysis_dir,
            targeted_design_dir=args.targeted_design_dir,
            target_col=args.target_col,
            canonical_mode=args.canonical_mode,
            model_type=args.model_type,
            buy_thresholds=parse_threshold_list(args.buy_thresholds, DEFAULT_BUY_THRESHOLDS),
            sell_thresholds=parse_threshold_list(args.sell_thresholds, DEFAULT_SELL_THRESHOLDS),
            commission_rate=args.commission_rate,
            stamp_tax_rate=args.stamp_tax_rate,
            slippage_pct=args.slippage_pct,
            min_commission=args.minimum_commission,
            min_trades=args.min_trades,
        )
    except Exception as exc:
        print(f"Error: sideways regime trade sufficiency remediation failed: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Sideways Regime Trade Sufficiency Remediation")
    print("-----------------------------------------------------------")
    print(f"Factor directory: {args.factor_dir}")
    print(f"Symbols: {symbols}")
    print(f"Recommendations: {args.recommendations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Canonical mode: {args.canonical_mode}")
    print(f"Model type: {args.model_type}")
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    print()
    print("Warning: This is educational research diagnostics only, not financial advice.")


if __name__ == "__main__":
    main()
