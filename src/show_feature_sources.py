import argparse

try:
    from .feature_source_registry import (
        export_registry_to_csv,
        get_all_features,
        get_features_by_family,
        get_high_leakage_risk_features,
        get_high_priority_features,
        get_token_required_features,
        get_training_ready_features,
        registry_to_dataframe,
        summarize_factor_families,
    )
except ImportError:
    from feature_source_registry import (
        export_registry_to_csv,
        get_all_features,
        get_features_by_family,
        get_high_leakage_risk_features,
        get_high_priority_features,
        get_token_required_features,
        get_training_ready_features,
        registry_to_dataframe,
        summarize_factor_families,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the planned multi-factor feature source registry.",
    )
    parser.add_argument("--list", action="store_true", help="List all registry rows.")
    parser.add_argument("--family", help="Filter by factor family.")
    parser.add_argument(
        "--priority",
        choices=["P0", "P1", "P2", "P3"],
        help="Filter by implementation priority.",
    )
    parser.add_argument(
        "--training-ready",
        action="store_true",
        help="Show features marked suitable for training.",
    )
    parser.add_argument(
        "--high-leakage-risk",
        action="store_true",
        help="Show features marked high leakage risk.",
    )
    parser.add_argument(
        "--token-required",
        action="store_true",
        help="Show features that require an API token or vendor access.",
    )
    parser.add_argument(
        "--high-priority",
        action="store_true",
        help="Show P0/P1 roadmap items.",
    )
    parser.add_argument("--summary", action="store_true", help="Show family summary.")
    parser.add_argument("--export", help="Export the full registry to CSV.")
    return parser.parse_args()


def print_dataframe(title: str, df) -> None:
    print()
    print(title)
    print("-" * len(title))
    if df.empty:
        print("No rows found.")
        return
    print(df.to_string(index=False))


def filter_records(args: argparse.Namespace) -> list[dict]:
    if args.family:
        records = get_features_by_family(args.family)
    elif args.training_ready:
        records = get_training_ready_features()
    elif args.high_leakage_risk:
        records = get_high_leakage_risk_features()
    elif args.token_required:
        records = get_token_required_features()
    elif args.high_priority:
        records = get_high_priority_features()
    else:
        records = get_all_features()

    if args.priority:
        records = [
            record
            for record in records
            if record["implementation_priority"] == args.priority
        ]
    return records


def main() -> None:
    args = parse_args()

    if args.export:
        output_path = export_registry_to_csv(args.export)
        print(f"Feature source registry exported: {output_path}")

    if args.summary:
        print_dataframe("Factor Family Summary", summarize_factor_families())

    if (
        args.list
        or args.family
        or args.priority
        or args.training_ready
        or args.high_leakage_risk
        or args.token_required
        or args.high_priority
        or not args.export
    ):
        records = filter_records(args)
        print_dataframe("Feature Source Registry", registry_to_dataframe(records))

    print()
    print(
        "Note: This registry is an educational roadmap. It does not fetch paid "
        "or private data and does not guarantee trading profit."
    )


if __name__ == "__main__":
    main()
