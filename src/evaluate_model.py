import argparse
import json
import sys

from model_evaluator import DEFAULT_TARGET_COL, evaluate_model_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction quality for a trained baseline model."
    )
    parser.add_argument(
        "--model-dir",
        default="models/demo_000001",
        help="Directory containing metrics.json and prediction CSV files.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help="Target label column. Defaults to label_up_5d.",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.6,
        help="Probability threshold for the optional long/flat signal check.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full evaluation result as JSON.",
    )
    return parser.parse_args()


def print_metrics(title: str, metrics: dict) -> None:
    print(title)
    print("-" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print()


def print_probability_summary(probability: dict) -> None:
    print("Probability Analysis")
    print("--------------------")
    if not probability.get("available"):
        print("Probability column is unavailable.")
        print()
        return

    for key in [
        "min_probability",
        "max_probability",
        "mean_probability",
        "median_probability",
        "avg_probability_actual_positive",
        "avg_probability_actual_negative",
    ]:
        value = probability.get(key)
        print(f"{key}: {'N/A' if value is None else f'{value:.4f}'}")
    print()


def print_thresholds(rows: list[dict]) -> None:
    print("Threshold Analysis")
    print("------------------")
    if not rows:
        print("Threshold analysis is unavailable without probabilities.")
        print()
        return
    for row in rows:
        print(
            "threshold={threshold:.2f}, accuracy={accuracy:.4f}, "
            "precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, "
            "predicted_positive_rate={predicted_positive_rate:.4f}, "
            "signal_count={signal_count}".format(**row)
        )
    print()


def print_warnings(warnings: list[str]) -> None:
    print("Warnings")
    print("--------")
    for warning in warnings:
        print(f"- {warning}")
    print()


def print_split_report(name: str, report: dict) -> None:
    print(f"{name.title()} Evaluation")
    print("=" * (len(name) + 11))
    print_metrics("Classification Metrics", report["metrics"])
    print_probability_summary(report["probability_analysis"])
    print_thresholds(report["threshold_analysis"])
    print_metrics("Simple Signal Backtest", report["signal_backtest"])
    print_warnings(report["warnings"])


def main() -> None:
    args = parse_args()

    try:
        result = evaluate_model_directory(
            model_dir=args.model_dir,
            target_col=args.target_col,
            signal_threshold=args.signal_threshold,
        )
    except Exception as exc:
        print(f"Error: failed to evaluate model outputs: {exc}")
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("QuantPilot-AI Model Evaluation")
    print("------------------------------")
    print(f"Model directory: {result['model_dir']}")
    print(f"Target column: {result['target_col']}")
    print(f"Signal threshold: {result['signal_threshold']:.2f}")
    print(f"Feature count: {len(result['feature_columns'])}")
    if result["feature_leakage_columns"]:
        print("Feature leakage columns: " + ", ".join(result["feature_leakage_columns"]))
    else:
        print("Feature leakage columns: none detected")
    print()

    print_split_report("validation", result["validation"])
    print_split_report("test", result["test"])

    print("Interpretation")
    print("--------------")
    print(
        "Suspiciously perfect metrics are a warning sign, especially with small "
        "or synthetic datasets. Treat this as model diagnostics, not proof of "
        "tradable edge."
    )


if __name__ == "__main__":
    main()
