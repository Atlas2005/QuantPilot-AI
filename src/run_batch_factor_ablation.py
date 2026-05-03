import argparse
import json
from pathlib import Path
import sys

import pandas as pd

try:
    from .batch_model_trainer import fetch_symbol_ohlcv, parse_symbols
    from .build_factor_dataset import save_factor_dataset
    from .factor_ablation import (
        build_feature_impact_ranking,
        build_feature_pruning_recommendations,
        build_group_summary,
        parse_ablation_modes,
        parse_model_types,
        run_and_save_factor_ablation,
    )
    from .factor_builder import build_factor_dataset
except ImportError:
    from batch_model_trainer import fetch_symbol_ohlcv, parse_symbols
    from build_factor_dataset import save_factor_dataset
    from factor_ablation import (
        build_feature_impact_ranking,
        build_feature_pruning_recommendations,
        build_group_summary,
        parse_ablation_modes,
        parse_model_types,
        run_and_save_factor_ablation,
    )
    from factor_builder import build_factor_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run factor ablation diagnostics across multiple symbols.",
    )
    parser.add_argument("--symbols", default="000001,600519")
    parser.add_argument("--source", choices=["demo", "baostock"], default="demo")
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--output-dir", default="outputs/factor_ablation_demo")
    parser.add_argument("--models", default="logistic_regression,random_forest")
    parser.add_argument("--target-col", default="label_up_5d")
    parser.add_argument("--ablation-modes", default="drop_group,only_group")
    parser.add_argument("--purge-rows", type=int, default=5)
    parser.add_argument("--max-drop-features", type=int)
    return parser.parse_args()


def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    try:
        symbols = parse_symbols(args.symbols)
        model_types = parse_model_types(args.models)
        ablation_modes = parse_ablation_modes(args.ablation_modes)
    except Exception as exc:
        print(f"Error: invalid arguments: {exc}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    factor_dir = output_dir / "factors"
    symbol_output_dir = output_dir / "symbols"
    factor_dir.mkdir(parents=True, exist_ok=True)
    symbol_output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_warnings = []

    print("QuantPilot-AI Batch Factor Ablation")
    print("-----------------------------------")
    print(f"Symbols: {symbols}")
    print(f"Source: {args.source}")
    print(f"Output directory: {output_dir}")
    print()

    for symbol in symbols:
        print(f"Processing symbol: {symbol}")
        try:
            raw_df = fetch_symbol_ohlcv(
                symbol=symbol,
                source=args.source,
                start=args.start,
                end=args.end,
            )
            factor_df = build_factor_dataset(raw_df, symbol=symbol)
            factor_path = factor_dir / f"factors_{symbol}.csv"
            save_factor_dataset(factor_df, factor_path)
            result = run_and_save_factor_ablation(
                input_path=factor_path,
                output_dir=symbol_output_dir / symbol,
                target_col=args.target_col,
                model_types=model_types,
                ablation_modes=ablation_modes,
                purge_rows=args.purge_rows,
                symbol=symbol,
                max_drop_features=args.max_drop_features,
            )
            all_results.append(result["ablation_results"])
            all_warnings.append(result["warnings"])
        except Exception as exc:
            all_warnings.append(
                pd.DataFrame(
                    [
                        {
                            "symbol": symbol,
                            "model_type": None,
                            "experiment_name": None,
                            "warning": str(exc),
                        }
                    ]
                )
            )

    ablation_results = _combine_frames(all_results)
    group_summary = build_group_summary(ablation_results)
    feature_impact_ranking = build_feature_impact_ranking(ablation_results)
    feature_ablation_results = (
        ablation_results[ablation_results["ablation_type"] == "drop_feature"].copy()
        if not ablation_results.empty and "ablation_type" in ablation_results
        else pd.DataFrame()
    )
    feature_pruning_recommendations = build_feature_pruning_recommendations(
        ablation_results
    )
    warnings = _combine_frames(all_warnings)

    output_files = {
        "ablation_results": output_dir / "ablation_results.csv",
        "feature_ablation_results": output_dir / "feature_ablation_results.csv",
        "group_summary": output_dir / "group_summary.csv",
        "feature_impact_ranking": output_dir / "feature_impact_ranking.csv",
        "feature_pruning_recommendations": output_dir / "feature_pruning_recommendations.csv",
        "warnings": output_dir / "warnings.csv",
        "run_config": output_dir / "run_config.json",
    }
    ablation_results.to_csv(output_files["ablation_results"], index=False)
    feature_ablation_results.to_csv(
        output_files["feature_ablation_results"],
        index=False,
    )
    group_summary.to_csv(output_files["group_summary"], index=False)
    feature_impact_ranking.to_csv(output_files["feature_impact_ranking"], index=False)
    feature_pruning_recommendations.to_csv(
        output_files["feature_pruning_recommendations"],
        index=False,
    )
    warnings.to_csv(output_files["warnings"], index=False)

    run_config = {
        "symbols": symbols,
        "source": args.source,
        "start": args.start,
        "end": args.end,
        "output_dir": str(output_dir),
        "models": model_types,
        "target_col": args.target_col,
        "ablation_modes": ablation_modes,
        "purge_rows": args.purge_rows,
        "max_drop_features": args.max_drop_features,
    }
    output_files["run_config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Output Files")
    print("------------")
    for label, path in output_files.items():
        print(f"{label}: {path}")
    print()
    print(
        "Warning: Factor ablation is research diagnostics only. It is not "
        "financial advice and does not guarantee profitable trading."
    )


if __name__ == "__main__":
    main()
