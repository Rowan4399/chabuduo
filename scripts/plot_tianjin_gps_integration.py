"""Compare repaired kinematic integration directly with observed GPS positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from audit_tianjin_track import GEOD, distances, integrated, percentiles
from cartes.crs import Mercator, PlateCarree
from traffic.core import Flight
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData
from chabuduo.config import load_experiment_config


def full_resolution_positions(frame, anchor, coefficient):
    times = (frame.timestamp - frame.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    positions = np.empty((len(frame), 2))
    positions[-1] = anchor
    speed = np.clip(frame.groundspeed.to_numpy(dtype=float), 0, 600)
    track = frame.track.to_numpy(dtype=float)
    for index in range(len(frame) - 1, 0, -1):
        lon, lat, _ = GEOD.fwd(
            *positions[index],
            track[index] + 180,
            speed[index]
            * 1852
            / 3600
            * (times[index] - times[index - 1])
            * coefficient,
        )
        positions[index - 1] = lon, lat
    return positions


def draw(ax, gps, positions, title):
    ax.plot(
        gps[:, 0],
        gps[:, 1],
        transform=PlateCarree(),
        color="black",
        label="Observed GPS",
        linewidth=1.3,
    )
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        transform=PlateCarree(),
        color="#e45756",
        label="Repaired integration",
        linewidth=1,
        alpha=0.8,
    )
    ax.set_title(title, fontsize=10)
    ax.gridlines(alpha=0.25)
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selected", "artifact", "metrics", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.output_dir.exists():
        parser.error("Use a positive count and a new output directory")
    config = load_experiment_config(args.config)
    data = SplitData.load(args.artifact / "train.npz")
    values = Normalization.load(args.artifact / "normalization.json").inverse_transform(
        data.values
    )
    metrics = pd.read_parquet(args.metrics)
    metrics = metrics.loc[metrics.split.eq("train")].set_index("trajectory_id")
    if set(metrics.index) != set(data.trajectory_id):
        raise ValueError("Metrics do not cover precisely the training trajectories")
    count = min(args.count, 4 if args.smoke else len(data.values))
    sample = (
        np.random.default_rng(args.seed)
        .choice(data.trajectory_id, count, replace=False)
        .tolist()
    )
    worst = metrics.after_gps_rmse_m.nlargest(2 if args.smoke else 6).index.tolist()
    ids = list(dict.fromkeys(sample + worst))
    indices = {identifier: i for i, identifier in enumerate(data.trajectory_id)}
    selected_indices = np.asarray([indices[identifier] for identifier in ids])
    positions = integrated(
        values[selected_indices], data.elapsed_time[selected_indices], config
    )
    columns = [
        "flight_id",
        "timestamp",
        "longitude",
        "latitude",
        "track",
        "groundspeed",
    ]
    frame = pq.read_table(
        args.selected, columns=columns, filters=[("flight_id", "in", ids)]
    ).to_pandas()
    groups = frame.groupby("flight_id", sort=False)
    frames, gps_positions = {}, {}
    reconstructed = dict(zip(ids, positions, strict=True))
    for identifier in ids:
        flight_frame = groups.get_group(identifier).sort_values("timestamp")
        frames[identifier] = flight_frame
        gps_positions[identifier] = (
            Flight(flight_frame)
            .resample(200)
            .data[["longitude", "latitude"]]
            .to_numpy(dtype=float)
        )
    airport = airports["ZBTJ"]
    threshold = next(p for p in airport.runways["16L"].tuple_runway if p.name == "16L")
    anchor = (threshold.longitude, threshold.latitude)
    controls = []
    for identifier in worst:
        flight_frame = frames[identifier]
        observed = flight_frame[["longitude", "latitude"]].to_numpy(dtype=float)
        record = {
            "trajectory_id": identifier,
            "repaired_200_rmse_m": float(metrics.loc[identifier, "after_gps_rmse_m"]),
        }
        for coefficient in (0.99, 1.0):
            reconstructed_full = full_resolution_positions(
                flight_frame, observed[-1], coefficient
            )
            error = distances(reconstructed_full, observed)
            record[f"source_resolution_observed_endpoint_coef_{coefficient}_rmse_m"] = (
                float(np.sqrt(np.mean(error**2)))
            )
        controls.append(record)
    args.output_dir.mkdir(parents=True)
    figure, axes = plt.subplots(
        1, 2, figsize=(13, 6), subplot_kw={"projection": Mercator()}
    )
    all_points = np.concatenate(
        [gps_positions[i] for i in sample] + [reconstructed[i] for i in sample]
    )
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    pad = np.maximum((high - low) * 0.05, 0.02)
    extent = [low[0] - pad[0], high[0] + pad[0], low[1] - pad[1], high[1] + pad[1]]
    for ax in axes:
        ax.set_extent(extent, PlateCarree())
        ax.gridlines(alpha=0.25)
        ax.plot(*anchor, "x", transform=PlateCarree(), color="black")
    for index, identifier in enumerate(sample):
        observed = gps_positions[identifier]
        pos = reconstructed[identifier]
        axes[0].plot(
            observed[:, 0],
            observed[:, 1],
            transform=PlateCarree(),
            color="black",
            alpha=0.25,
            linewidth=0.7,
        )
        axes[1].plot(
            observed[:, 0],
            observed[:, 1],
            transform=PlateCarree(),
            color="black",
            alpha=0.25,
            linewidth=0.7,
            label="Observed GPS" if index == 0 else None,
        )
        axes[1].plot(
            pos[:, 0],
            pos[:, 1],
            transform=PlateCarree(),
            color="#e45756",
            alpha=0.3,
            linewidth=0.7,
            label="Repaired integration" if index == 0 else None,
        )
    axes[0].set_title(f"Observed GPS: {count} training flights, seed {args.seed}")
    axes[1].set_title("Same flights: observed GPS and repaired integration")
    axes[1].legend()
    figure.savefig(
        args.output_dir / "gps_vs_repaired_integrated.png", dpi=170, bbox_inches="tight"
    )
    plt.close(figure)
    figure, axes = plt.subplots(
        (len(worst) + 1) // 2,
        2,
        figsize=(12, 3.7 * ((len(worst) + 1) // 2)),
        subplot_kw={"projection": Mercator()},
        squeeze=False,
    )
    for ax, identifier in zip(axes.flat, worst, strict=True):
        rmse_km = metrics.loc[identifier, "after_gps_rmse_m"] / 1000
        draw(
            ax,
            gps_positions[identifier],
            reconstructed[identifier],
            f"{identifier}: GPS RMSE {rmse_km:.2f} km",
        )
    figure.subplots_adjust(hspace=0.25, wspace=0.15)
    figure.savefig(
        args.output_dir / "worst_gps_vs_repaired_integrated.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(7, 4))
    ordered = np.sort(metrics.after_gps_rmse_m.to_numpy())
    ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered))
    ax.set(
        xscale="log",
        xlabel="Position RMSE against observed GPS (metres)",
        ylabel="Fraction of training trajectories",
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    figure.savefig(
        args.output_dir / "gps_position_error_cdf.png", dpi=170, bbox_inches="tight"
    )
    plt.close(figure)
    summary = {
        "comparison": (
            "repaired integrated positions against observed latitude/longitude"
        ),
        "split": "train",
        "population": len(metrics),
        "seed": args.seed,
        "overview_count": count,
        "overview_trajectory_ids": sample,
        "position_rmse_m": percentiles(metrics.after_gps_rmse_m),
        "trajectories_rmse_above_100m": int(metrics.after_gps_rmse_m.gt(100).sum()),
        "trajectories_rmse_above_1km": int(metrics.after_gps_rmse_m.gt(1000).sum()),
        "trajectories_rmse_above_5km": int(metrics.after_gps_rmse_m.gt(5000).sum()),
        "worst_full_resolution_controls": controls,
        "source_resolution_control": (
            "Independent WGS84 right-endpoint backward integration "
            "at source timestamps, "
            "anchored at observed endpoint; coefficients 0.99 and 1.0; "
            "no GPS-derived replacement of reported track or groundspeed"
        ),
        "interpretation": (
            "Angle continuity is fixed; spatial agreement with GPS is not established."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Output: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
