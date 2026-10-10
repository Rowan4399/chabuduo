"""Compare old/repaired model inputs with observed GPS and runway endpoints.

No model inference is performed. Both artifacts must contain identical flights,
splits, times, speeds and altitudes. Only their scalar angle branches may differ.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Geod
from traffic.core import Flight
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData
from chabuduo.config import load_experiment_config
from chabuduo.sampling import build_generated_traffic

GEOD = Geod(ellps="WGS84")


def percentiles(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {name: None for name in ("min", "median", "p95", "max")}
    return dict(
        zip(
            ("min", "median", "p95", "max"),
            np.percentile(values, [0, 50, 95, 100]).tolist(),
            strict=True,
        )
    )


def circular_difference(left, right):
    return (np.asarray(left) - np.asarray(right) + 180) % 360 - 180


def distances(left, right):
    return GEOD.inv(left[..., 0], left[..., 1], right[..., 0], right[..., 1])[2]


def gps_motion(frame: pd.DataFrame, half_window_sec: float = 5) -> dict:
    """Compare centered 10-second GPS bearings with the reported motion."""
    elapsed = (frame.timestamp - frame.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    left = np.searchsorted(elapsed, elapsed - half_window_sec)
    right = np.minimum(
        np.searchsorted(elapsed, elapsed + half_window_sec), len(frame) - 1
    )
    dt = elapsed[right] - elapsed[left]
    lon, lat = frame.longitude.to_numpy(), frame.latitude.to_numpy()
    bearing, _, distance = GEOD.inv(lon[left], lat[left], lon[right], lat[right])
    valid = (dt >= half_window_sec) & (distance >= 10) & np.isfinite(bearing)
    speed = np.divide(distance, dt, out=np.full_like(dt, np.nan), where=dt > 0)
    speed *= 3600 / 1852
    return {
        "gps_bearing_comparison_points": int(valid.sum()),
        "source_track_vs_gps_median_deg": float(
            np.median(
                np.abs(
                    circular_difference(frame.track.to_numpy()[valid], bearing[valid])
                )
            )
        )
        if valid.any()
        else None,
        "source_track_vs_gps_p95_deg": float(
            np.percentile(
                np.abs(
                    circular_difference(frame.track.to_numpy()[valid], bearing[valid])
                ),
                95,
            )
        )
        if valid.any()
        else None,
        "source_speed_vs_gps_median_abs_kt": float(
            np.median(np.abs(frame.groundspeed.to_numpy()[valid] - speed[valid]))
        )
        if valid.any()
        else None,
    }


def integrated(values, times, config) -> np.ndarray:
    t = build_generated_traffic(values, times, config)
    return np.stack(
        [
            t.data.loc[t.data.flight_id.eq(f"TRAJ_{index}")]
            .sort_values("timestamp")[["longitude", "latitude"]]
            .to_numpy(dtype=float)
            for index in range(len(values))
        ]
    )


def plot_examples(examples, output: Path, anchor) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cartes.crs import Mercator, PlateCarree

    figure, axes = plt.subplots(len(examples), 2, figsize=(12, 3.8 * len(examples)))
    figure.subplots_adjust(hspace=0.4, wspace=0.3)
    axes = np.asarray(axes).reshape(len(examples), 2)
    for row, item in enumerate(examples):
        ax = axes[row, 0]
        elapsed = item["times"]
        ax.plot(elapsed, item["before_track"], label="Before", alpha=0.8)
        ax.plot(elapsed, item["after_track"], label="After")
        ax.set(xlabel="Elapsed seconds", ylabel="Track degrees", title=item["id"])
        ax.legend()
        old_ax = axes[row, 1]
        position = old_ax.get_position()
        old_ax.remove()
        map_ax = figure.add_axes(position, projection=Mercator())
        for name, color in (
            ("gps", "black"),
            ("before", "#e45756"),
            ("after", "#2878b5"),
            ("endpoint", "#59a14f"),
        ):
            points = item[name]
            map_ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                label=name,
                transform=PlateCarree(),
                linewidth=1,
                alpha=0.8,
            )
        map_ax.plot(*anchor, "x", transform=PlateCarree(), color="black")
        map_ax.gridlines(alpha=0.3)
        map_ax.legend(fontsize=8)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "selected",
        "before-artifact",
        "after-artifact",
        "config",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = load_experiment_config(args.config)
    features = config.prepare.representation.features
    if features != [
        "track_unwrapped",
        "groundspeed_kt",
        "altitude_ft",
        "elapsed_time_sec",
    ]:
        raise ValueError("Expected the four-channel Tianjin kinematic representation")
    if config.prepare.representation.anchor != "runway_threshold":
        raise ValueError("Runway-threshold anchoring is required for this comparison")
    airport = airports[config.preprocess.airport]
    runway = config.preprocess.runways[0]
    threshold = next(
        p for p in airport.runways[runway].tuple_runway if p.name == runway
    )
    anchor = (threshold.longitude, threshold.latitude)
    projection = f"+proj=aeqd +lat_0={anchor[1]} +lon_0={anchor[0]} +datum=WGS84"
    old_norm = Normalization.load(args.before_artifact / "normalization.json")
    new_norm = Normalization.load(args.after_artifact / "normalization.json")
    columns = [
        "flight_id",
        "timestamp",
        "longitude",
        "latitude",
        "track",
        "groundspeed",
    ]
    selected = pq.read_table(args.selected, columns=columns).to_pandas()
    groups = selected.groupby("flight_id", sort=False)
    records, examples, split_reports = [], [], {}
    for split in ("train", "validation", "test"):
        old = SplitData.load(args.before_artifact / f"{split}.npz")
        new = SplitData.load(args.after_artifact / f"{split}.npz")
        if not np.array_equal(old.trajectory_id, new.trajectory_id):
            raise ValueError(f"{split}: trajectory IDs/order changed")
        if not np.array_equal(old.split_group_id, new.split_group_id):
            raise ValueError(f"{split}: split groups changed")
        before, after = (
            old_norm.inverse_transform(old.values),
            new_norm.inverse_transform(new.values),
        )
        if not np.array_equal(old.elapsed_time, new.elapsed_time):
            raise ValueError(f"{split}: times changed")
        if not np.allclose(before[:, :, 1:], after[:, :, 1:], atol=2e-3, rtol=0):
            raise ValueError(f"{split}: non-angle model features changed")
        if args.smoke:
            before, after = before[:4], after[:4]
        times = old.elapsed_time[: len(before)]
        old_positions = integrated(before, times, config)
        new_positions = integrated(after, times, config)
        split_records = []
        for index, identifier in enumerate(old.trajectory_id[: len(before)]):
            frame = groups.get_group(identifier).sort_values("timestamp", kind="stable")
            gps = Flight(frame).resample(config.prepare.points).data
            observed = gps[["longitude", "latitude"]].to_numpy(dtype=float)
            point_errors_before = distances(old_positions[index], observed)
            point_errors_after = distances(new_positions[index], observed)
            # A controlled endpoint change uses the same public integration API.
            from traffic.algorithms.generation import Generation

            matrix = np.column_stack((after[index, :, :3], times[index])).astype(float)
            matrix[:, 0] %= 360
            adapter = Generation(
                None, ["track", "groundspeed", "altitude", "timedelta"]
            )
            endpoint_t = adapter.build_traffic(
                matrix.reshape(1, -1),
                coordinates={"longitude": observed[-1, 0], "latitude": observed[-1, 1]},
                forward=False,
            )
            endpoint = endpoint_t.data[["longitude", "latitude"]].to_numpy(dtype=float)
            endpoint_error = distances(endpoint, observed)
            bearing, _, displacement = GEOD.inv(
                observed[:-1, 0], observed[:-1, 1], observed[1:, 0], observed[1:, 1]
            )
            usable = displacement >= 10
            old_angle = np.degrees(
                np.angle(
                    np.exp(1j * np.radians(before[index, :-1, 0]))
                    + np.exp(1j * np.radians(before[index, 1:, 0]))
                )
            )
            new_angle = np.degrees(
                np.angle(
                    np.exp(1j * np.radians(after[index, :-1, 0]))
                    + np.exp(1j * np.radians(after[index, 1:, 0]))
                )
            )
            record = {
                "trajectory_id": identifier,
                "split": split,
                "before_jumps_above_180": int(
                    (np.abs(np.diff(before[index, :, 0])) > 180).sum()
                ),
                "after_jumps_above_180": int(
                    (np.abs(np.diff(after[index, :, 0])) > 180).sum()
                ),
                "angle_circular_change_max_deg": float(
                    np.abs(
                        circular_difference(before[index, :, 0], after[index, :, 0])
                    ).max()
                ),
                "before_gps_rmse_m": float(np.sqrt(np.mean(point_errors_before**2))),
                "after_gps_rmse_m": float(np.sqrt(np.mean(point_errors_after**2))),
                "observed_endpoint_gps_rmse_m": float(
                    np.sqrt(np.mean(endpoint_error**2))
                ),
                "before_after_position_max_m": float(
                    distances(old_positions[index], new_positions[index]).max()
                ),
                "threshold_endpoint_offset_m": float(
                    distances(np.asarray(anchor), observed[-1])
                ),
                "before_track_vs_gps_200_p95_deg": float(
                    np.percentile(
                        np.abs(circular_difference(old_angle[usable], bearing[usable])),
                        95,
                    )
                ),
                "after_track_vs_gps_200_p95_deg": float(
                    np.percentile(
                        np.abs(circular_difference(new_angle[usable], bearing[usable])),
                        95,
                    )
                ),
                **gps_motion(frame),
            }
            split_records.append(record)
            if (
                split == "train"
                and record["before_jumps_above_180"]
                and len(examples) < 3
            ):
                examples.append(
                    {
                        "id": identifier,
                        "times": times[index],
                        "before_track": before[index, :, 0],
                        "after_track": after[index, :, 0],
                        "gps": observed,
                        "before": old_positions[index],
                        "after": new_positions[index],
                        "endpoint": endpoint,
                    }
                )
            if index % 250 == 0:
                print(f"{split}: {index}/{len(before)} GPS comparisons", flush=True)
        metrics = pd.DataFrame(split_records)
        split_reports[split] = {
            "trajectories": len(metrics),
            "before_trajectories_with_jumps_above_180": int(
                metrics.before_jumps_above_180.gt(0).sum()
            ),
            "after_trajectories_with_jumps_above_180": int(
                metrics.after_jumps_above_180.gt(0).sum()
            ),
            "before_jumps_above_180": int(metrics.before_jumps_above_180.sum()),
            "after_jumps_above_180": int(metrics.after_jumps_above_180.sum()),
            **{
                name: percentiles(metrics[name])
                for name in (
                    "before_gps_rmse_m",
                    "after_gps_rmse_m",
                    "observed_endpoint_gps_rmse_m",
                    "threshold_endpoint_offset_m",
                    "before_after_position_max_m",
                    "source_track_vs_gps_p95_deg",
                    "source_speed_vs_gps_median_abs_kt",
                    "angle_circular_change_max_deg",
                    "before_track_vs_gps_200_p95_deg",
                    "after_track_vs_gps_200_p95_deg",
                )
            },
        }
        records.extend(split_records)
    args.output_dir.mkdir(parents=True)
    pd.DataFrame(records).to_parquet(
        args.output_dir / "trajectory_metrics.parquet", index=False
    )
    report = {
        "selected": str(args.selected.resolve()),
        "before_artifact": str(args.before_artifact.resolve()),
        "after_artifact": str(args.after_artifact.resolve()),
        "smoke": args.smoke,
        "split_reports": split_reports,
        "airport": config.preprocess.airport,
        "runway": runway,
        "threshold_lon_lat": anchor,
        "projection_for_endpoint_checks": projection,
        "gps_comparison": (
            "WGS84 geodesics; source bearings centered over approximately 10 seconds; "
            "200-point bearings use circular endpoint-average track"
        ),
        "endpoint_caveat": (
            "Selected endpoints were screened against this threshold; "
            "proximity alone is not an independent runway survey."
        ),
        "integration": (
            "Current public traffic.Generation backward integration, "
            "including its 0.99 speed coefficient; no model inference"
        ),
        "identical_ids_splits_times_and_non_angle_features": True,
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("traffic", "numpy", "pandas", "pyproj")
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8"
    )
    if examples:
        plot_examples(examples, args.output_dir / "track_and_gps_examples.png", anchor)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Output: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
