"""Plot held-out and generated trajectories for one configured run."""

from __future__ import annotations

import argparse
from pathlib import Path

from chabuduo.config import (
    config_path,
    load_experiment_config,
    prepared_path,
    run_path,
    sample_path,
)
from chabuduo.plotting import plot_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Bundled experiment configuration name")
    parser.add_argument("--config", type=Path, help="Explicit experiment YAML")
    parser.add_argument("--run", type=Path, help="Training run directory")
    parser.add_argument("--artifact", type=Path, help="Prepared artifact directory")
    parser.add_argument("--sample", type=Path, help="Generated sample directory")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Trajectories shown in each panel (default: 100)",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config or config_path(args.experiment))
    run = (args.run or run_path(config)).resolve()
    artifact = (args.artifact or prepared_path(config)).resolve()
    sample = (args.sample or sample_path(config, run)).resolve()
    result = plot_samples(
        sample,
        artifact,
        config,
        output=None if args.output is None else args.output.resolve(),
        count=args.count,
    )
    print(f"OUTPUT={result}")


if __name__ == "__main__":
    main()
