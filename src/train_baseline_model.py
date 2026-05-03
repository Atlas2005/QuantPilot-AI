import argparse
import sys

from model_trainer import DEFAULT_TARGET_COL, MODEL_CHOICES, run_training_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline ML classifier from split factor datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing train.csv, validation.csv, test.csv, and feature_columns.txt.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help="Target label column. Defaults to label_up_5d.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="random_forest",
        help="Baseline model to train.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/baseline",
        help="Directory for model outputs. Defaults to models/baseline.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by supported models.",
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


def main() -> None:
    args = parse_args()

    try:
        result = run_training_workflow(
            dataset_dir=args.dataset_dir,
            target_col=args.target_col,
            model_name=args.model,
            output_dir=args.output_dir,
            random_state=args.random_state,
        )
    except Exception as exc:
        print(f"Error: failed to train baseline model: {exc}")
        sys.exit(1)

    print("QuantPilot-AI Baseline ML Model")
    print("-------------------------------")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Selected target: {args.target_col}")
    print(f"Selected model: {args.model}")
    print(f"Actual model: {result['training_info']['actual_model']}")
    print(f"Feature count: {len(result['feature_columns'])}")
    print(f"Train rows: {result['train_rows']}")
    print(f"Validation rows: {result['validation_rows']}")
    print(f"Test rows: {result['test_rows']}")
    print()

    print_metrics("Validation Metrics", result["validation_metrics"])
    print_metrics("Test Metrics", result["test_metrics"])

    print("Output files")
    print("------------")
    for label, path in result["output_files"].items():
        if path is not None:
            print(f"{label}: {path}")
    print()
    print("Warning: This baseline model is for educational research only.")
    print("Good ML metrics do not guarantee profitable trading.")


if __name__ == "__main__":
    main()
