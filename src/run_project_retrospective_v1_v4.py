import argparse
import sys

try:
    from .project_retrospective_v1_v4 import generate_project_retrospective_v1_v4
except ImportError:
    from project_retrospective_v1_v4 import generate_project_retrospective_v1_v4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an audit-only QuantPilot-AI V1-V4 project retrospective.",
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_project_retrospective_v1_v4(
            project_root=args.project_root,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Error: project retrospective failed: {exc}")
        sys.exit(1)

    config = result["run_config"]
    architecture = result["architecture_layer_summary"]
    print("QuantPilot-AI V1-V4 Project Retrospective")
    print("------------------------------------------")
    print(f"Output directory: {config['output_dir']}")
    print("Generated files:")
    for label, path in result["output_files"].items():
        print(f"- {label}: {path}")
    print(f"Phase summary count: {len(result['phase_progress_summary'])}")
    print(f"Current architecture summary: {len(architecture)} layers reviewed")
    print(f"Recommended next phase: {config['recommended_next_phase']}")
    print(f"Recommended next step: {config['recommended_next_step']}")
    print()
    print("Warning: This is educational/research diagnostics only, not financial advice.")
    print("No candidate is trading-ready.")


if __name__ == "__main__":
    main()
