"""Run the full Monet -> logic circuit conversion pipeline.

Usage:
    # Single scale, using the defaults in pipeline.yaml
    python -m monet_logic_circuit.scripts.run_pipeline \
        --pipeline monet_logic_circuit/configs/pipeline.yaml

    # Dry run: print the planned execution without touching anything
    python -m monet_logic_circuit.scripts.run_pipeline \
        --pipeline monet_logic_circuit/configs/pipeline.yaml \
        --dry-run

    # Resume from a specific step (e.g. after a crash during Step 3b)
    python -m monet_logic_circuit.scripts.run_pipeline \
        --pipeline monet_logic_circuit/configs/pipeline.yaml \
        --from-step 3b

    # Run the whole pipeline across 850M -> 1.4B -> 4.1B automatically
    python -m monet_logic_circuit.scripts.run_pipeline \
        --pipeline monet_logic_circuit/configs/pipeline.yaml \
        --scale-automatically
"""

import argparse
import sys
from pathlib import Path

from monet_logic_circuit.pipeline.orchestrator import (
    Pipeline,
    load_pipeline_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Monet -> logic circuit pipeline end-to-end."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Path to pipeline.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned execution without running any steps.",
    )
    parser.add_argument(
        "--from-step",
        type=str,
        default=None,
        help="Start from this step instead of the first one (e.g. '3b').",
    )
    parser.add_argument(
        "--scale-automatically",
        action="store_true",
        help=(
            "Override pipeline.scale_automatically to true: run the full "
            "pipeline at each checkpoint in scale_progression, advancing "
            "only after the previous scale's final verdict passes."
        ),
    )
    parser.add_argument(
        "--trace-out",
        type=str,
        default=None,
        help=(
            "Where to write the pipeline trace JSON. "
            "Defaults to {results_root}/pipeline_trace.json."
        ),
    )
    args = parser.parse_args()

    spec = load_pipeline_spec(args.pipeline)
    if args.scale_automatically:
        spec.scale_automatically = True

    pipeline = Pipeline(spec, dry_run=args.dry_run, from_step=args.from_step)

    if spec.scale_automatically:
        ok = pipeline.run_with_scale_progression()
    else:
        ok = pipeline.run()

    trace_out = Path(
        args.trace_out
        or (Path(spec.results_root) / "pipeline_trace.json")
    )
    pipeline.dump_trace(trace_out)
    print(f"\nTrace written to {trace_out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
