import argparse
from pathlib import Path
import sys

import pandas as pd

try:
    from .batch_model_trainer import (
        parse_model_types,
        parse_symbols,
        run_batch_model_training,
    )
except ImportError:
    from batch_model_trainer import (
        parse_model_types,
        parse_symbols,
        run_batch_model_training,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild real Baostock factor datasets with current P0 OHLCV "
            "factors and rerun multi-symbol robustness training."
        )
    )
    parser.add_argument(
        "--symbols",
        default="000001,600519,000858,600036,601318",
        help="Comma-separated A-share symbols.",
    )
    parser.add_argument("--start", default="20210101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument(
        "--output-dir",
        default="outputs/model_robustness_real_v2",
        help="Output directory for rebuilt factors, splits, models, and summaries.",
    )
    parser.add_argument(
        "--baseline-dir",
        default="outputs/model_robustness_real_v1",
        help="Previous robustness output directory used for optional comparison.",
    )
    parser.add_argument(
        "--models",
        default="logistic_regression,random_forest",
        help="Comma-separated model types.",
    )
    parser.add_argument("--target-col", default="label_up_5d")
    parser.add_argument("--purge-rows", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=["global_date", "per_symbol"],
        default="global_date",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip v1-vs-v2 comparison even if the baseline directory exists.",
    )
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def compare_robustness_runs(
    baseline_dir: str | Path,
    current_dir: str | Path,
) -> dict[str, str]:
    """Compare previous and current robustness CSVs when both are available."""
    baseline_path = Path(baseline_dir)
    current_path = Path(current_dir)
    output_files = {}

    baseline_summary = _read_csv_if_exists(baseline_path / "model_summary.csv")
    current_summary = _read_csv_if_exists(current_path / "model_summary.csv")
    if not baseline_summary.empty and not current_summary.empty:
        merged_summary = current_summary.merge(
            baseline_summary,
            on="model_type",
            how="outer",
            suffixes=("_v2", "_v1"),
        )
        for metric in ["avg_test_roc_auc", "avg_test_f1", "avg_test_accuracy"]:
            v2_column = f"{metric}_v2"
            v1_column = f"{metric}_v1"
            if v2_column in merged_summary.columns and v1_column in merged_summary.columns:
                merged_summary[f"{metric}_delta"] = (
                    merged_summary[v2_column] - merged_summary[v1_column]
                )
        summary_path = current_path / "model_summary_comparison_vs_v1.csv"
        merged_summary.to_csv(summary_path, index=False)
        output_files["model_summary_comparison"] = str(summary_path)

    baseline_results = _read_csv_if_exists(baseline_path / "training_results.csv")
    current_results = _read_csv_if_exists(current_path / "training_results.csv")
    if not baseline_results.empty and not current_results.empty:
        keys = ["symbol", "model_type"]
        merged_results = current_results.merge(
            baseline_results,
            on=keys,
            how="outer",
            suffixes=("_v2", "_v1"),
        )
        for metric in ["test_roc_auc", "test_f1", "test_accuracy"]:
            v2_column = f"{metric}_v2"
            v1_column = f"{metric}_v1"
            if v2_column in merged_results.columns and v1_column in merged_results.columns:
                merged_results[f"{metric}_delta"] = (
                    merged_results[v2_column] - merged_results[v1_column]
                )
        results_path = current_path / "training_results_comparison_vs_v1.csv"
        merged_results.to_csv(results_path, index=False)
        output_files["training_results_comparison"] = str(results_path)

    return output_files


def main() -> None:
    args = parse_args()

    try:
        symbols = parse_symbols(args.symbols)
        model_types = parse_model_types(args.models)
    except Exception as exc:
        print(f"Error: invalid arguments: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Real P0 Factor Robustness Workflow")
    print("------------------------------------------------")
    print(f"Symbols: {symbols}")
    print("Source: baostock")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Models: {model_types}")
    print(f"Output directory: {args.output_dir}")
    print(f"Baseline comparison directory: {args.baseline_dir}")
    print()
    print(
        "This command fetches Baostock data, rebuilds factor CSVs with the "
        "current factor_builder columns, splits chronologically, trains baseline "
        "models, and writes robustness outputs. It is research only."
    )
    print()

    try:
        result = run_batch_model_training(
            symbols=symbols,
            model_types=model_types,
            source="baostock",
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            target_col=args.target_col,
            purge_rows=args.purge_rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_mode=args.split_mode,
        )
    except Exception as exc:
        print(f"Error: real-data robustness workflow failed: {exc}")
        sys.exit(1)

    comparison_files = {}
    if not args.skip_comparison:
        comparison_files = compare_robustness_runs(
            baseline_dir=args.baseline_dir,
            current_dir=args.output_dir,
        )

    print("Model Summary")
    print("-------------")
    print(result["model_summary"].to_string(index=False))
    print()
    print("Model Ranking")
    print("-------------")
    print(result["model_ranking"].to_string(index=False))
    print()
    print("Output Files")
    print("------------")
    for label, path in result["output_files"].items():
        print(f"{label}: {path}")
    for label, path in comparison_files.items():
        print(f"{label}: {path}")
    if not comparison_files and not args.skip_comparison:
        print("Comparison files: not created because baseline v1 files were not found.")
    print()
    print(
        "Warning: Rebuilt factors and ML metrics do not guarantee profitable "
        "trading. Review leakage, samples, and walk-forward robustness."
    )


if __name__ == "__main__":
    main()
