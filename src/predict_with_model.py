import argparse
import json
import sys

from model_predictor import run_model_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the latest factor row with a trained baseline model."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a trained .joblib model.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Factor CSV or ML split CSV used for latest-row prediction.",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Optional metrics.json path. Defaults to model directory/metrics.json.",
    )
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Optional feature_columns.txt path. Defaults to model directory.",
    )
    parser.add_argument(
        "--feature-importance",
        default=None,
        help="Optional feature_importance.csv path. Defaults to model directory.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top feature-importance rows to print.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the prediction result as JSON.",
    )
    return parser.parse_args()


def format_probability(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()

    try:
        result = run_model_prediction(
            model_path=args.model_path,
            input_path=args.input,
            metrics_path=args.metrics_path,
            feature_columns_path=args.feature_columns,
            feature_importance_path=args.feature_importance,
            top_n=args.top_n,
        )
    except Exception as exc:
        print(f"Error: failed to run model prediction: {exc}")
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("QuantPilot-AI Model Prediction")
    print("------------------------------")
    print(f"Model path: {result['model_path']}")
    print(f"Input path: {result['input_path']}")
    print(f"Latest row: {result['row_info']}")
    print(f"Feature count: {result['feature_count']}")
    print(f"Predicted class: {result['predicted_class']}")
    target_col = result.get("metrics", {}).get("target_col", "label_up_5d")
    print(
        f"Predicted probability of {target_col}: "
        f"{format_probability(result['predicted_probability'])}"
    )
    print(f"Model signal: {result['model_signal']}")
    print()
    print("Top feature importance")
    print("----------------------")
    if not result["top_feature_importance"]:
        print("N/A")
    else:
        for row in result["top_feature_importance"]:
            print(f"{row.get('feature')}: {row.get('importance')}")
    print()
    print("Warning: This prediction is educational research, not trading advice.")


if __name__ == "__main__":
    main()
