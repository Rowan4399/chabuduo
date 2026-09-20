"""Plot the loss history stored in one training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from chabuduo.plotting import plot_loss_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing logs/")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=20,
        help="First epoch shown in the right panel (default: 20)",
    )
    args = parser.parse_args()
    result = plot_loss_curves(
        args.run.resolve(),
        output=None if args.output is None else args.output.resolve(),
        warmup_epochs=args.warmup_epochs,
    )
    print(f"OUTPUT={result}")


if __name__ == "__main__":
    main()
