from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_trajectories(input_file: Path, output_file: Path) -> None:
    data = pd.read_parquet(input_file)

    required_columns = {"flight_id", "x", "y"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"generated.parquet 缺少字段: {sorted(missing)}")

    if "timestamp" in data.columns:
        data = data.sort_values(["flight_id", "timestamp"])
    elif "timedelta" in data.columns:
        data = data.sort_values(["flight_id", "timedelta"])

    figure, ax = plt.subplots(figsize=(9, 9))

    for _, trajectory in data.groupby("flight_id", sort=False):
        ax.plot(
            trajectory["x"] / 1000,
            trajectory["y"] / 1000,
            color="#4C78A8",
            linewidth=0.8,
            alpha=0.25,
        )

    # 以机场为原点
    ax.scatter(
        0,
        0,
        color="#E45756",
        marker="*",
        s=140,
        label="Airport",
        zorder=10,
    )

    trajectory_count = data["flight_id"].nunique()

    ax.set_title(f"Generated trajectories ({trajectory_count} trajectories)")
    ax.set_xlabel("East / km")
    ax.set_ylabel("North / km")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"OUTPUT={output_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all generated trajectories."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to generated.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/generated_trajectories.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

    plot_trajectories(args.input, args.output)


if __name__ == "__main__":
    main()