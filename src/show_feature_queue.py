import argparse

try:
    from .feature_implementation_queue import (
        export_feature_queue_csv,
        filter_feature_queue,
        queue_to_dataframe,
        summarize_feature_queue,
    )
except ImportError:
    from feature_implementation_queue import (
        export_feature_queue_csv,
        filter_feature_queue,
        queue_to_dataframe,
        summarize_feature_queue,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the feature engineering implementation queue.",
    )
    parser.add_argument(
        "--priority",
        choices=["P0", "P1", "P2", "P3", "P0_now", "P1_next", "P2_later", "P3_research_only"],
        help="Filter by implementation priority or stage.",
    )
    parser.add_argument("--category", help="Filter by queue category.")
    parser.add_argument(
        "--leakage-risk",
        choices=["low", "medium", "high"],
        help="Filter by leakage risk.",
    )
    parser.add_argument(
        "--token-required",
        choices=["true", "false"],
        help="Filter by whether token/vendor access is required.",
    )
    parser.add_argument(
        "--difficulty",
        choices=["low", "medium", "high"],
        help="Filter by implementation difficulty.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30,
        help="Maximum rows to print.",
    )
    parser.add_argument(
        "--output",
        help="Optional CSV output path for the full queue.",
    )
    return parser.parse_args()


def parse_token_required(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


def main() -> None:
    args = parse_args()

    if args.output:
        output_path = export_feature_queue_csv(args.output)
        print(f"Feature implementation queue exported: {output_path}")

    queue = filter_feature_queue(
        priority=args.priority,
        category=args.category,
        leakage_risk=args.leakage_risk,
        token_required=parse_token_required(args.token_required),
        implementation_difficulty=args.difficulty,
    )
    summary = summarize_feature_queue(queue)
    df = queue_to_dataframe(queue)

    print()
    print("Feature Implementation Queue Summary")
    print("------------------------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print("Feature Implementation Queue")
    print("----------------------------")
    if df.empty:
        print("No rows found.")
    else:
        print(df.head(args.max_rows).to_string(index=False))
        if len(df) > args.max_rows:
            print(f"\nShowing first {args.max_rows} of {len(df)} rows.")

    print()
    print(
        "Note: This queue is an educational engineering roadmap. More features "
        "can increase overfitting and leakage risk, and no feature guarantees profit."
    )


if __name__ == "__main__":
    main()
