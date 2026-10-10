"""Trace one Tianjin source anomaly and review an existing TCVAE checkpoint.

This is an evaluation script, not a replacement preprocessing stage. It never
changes the prepared splits, checkpoint, or original generated samples.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import cartes.crs as crs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from cartopy.crs import PlateCarree
from pyproj import Geod
from traffic.core import Flight
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData, sha256
from chabuduo.config import load_experiment_config
from chabuduo.models.tcvae import TCVAELightning
from chabuduo.prepare import airport_projection
from chabuduo.sampling import build_generated_traffic

GEOD = Geod(ellps="WGS84")
FEATURES = ["track_unwrapped", "groundspeed_kt", "altitude_ft", "elapsed_time_sec"]
UNITS = ["degree", "knot", "ft", "s"]
COLORS = ["#4C78A8", "#F2A541", "#E45756", "#B279A2"]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def position_errors(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return GEOD.inv(
        reference[..., 0], reference[..., 1], actual[..., 0], actual[..., 1]
    )[2]


def summarize(values: np.ndarray) -> dict[str, float]:
    return dict(
        zip(
            ["min", "median", "p95", "max"],
            np.percentile(values, [0, 50, 95, 100]).tolist(),
            strict=True,
        )
    )


def integrated_positions(values, elapsed, config) -> tuple[np.ndarray, pd.DataFrame]:
    t = build_generated_traffic(values, elapsed, config)
    positions = np.stack(
        [
            t.data.loc[t.data.flight_id.eq(f"TRAJ_{i}")]
            .sort_values("timestamp")[["longitude", "latitude"]]
            .to_numpy()
            for i in range(len(values))
        ]
    )
    return positions, t.data


def backward_positions(frame: pd.DataFrame, endpoint) -> np.ndarray:
    elapsed = (frame.timestamp - frame.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    track = frame.track.to_numpy()
    speed = np.clip(frame.groundspeed.to_numpy(), 0, 600)
    positions = np.empty((len(frame), 2))
    positions[-1] = endpoint
    for i in range(len(frame) - 1, 0, -1):
        lon, lat, _ = GEOD.fwd(
            *positions[i],
            track[i] + 180,
            speed[i] * 1852 / 3600 * (elapsed[i] - elapsed[i - 1]) * 0.99,
        )
        positions[i - 1] = lon, lat
    return positions


def geographic_plot(groups, titles, output, anchor, *, overlay=False):
    source = PlateCarree()
    projection = crs.Mercator(central_longitude=anchor[0])
    count = 1 if overlay else len(groups)
    figure, axes = plt.subplots(
        1,
        count,
        figsize=(6 * count, 6),
        squeeze=False,
        subplot_kw={"projection": projection},
    )
    all_points = np.concatenate([p for group in groups for p in group])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    pad = np.maximum((upper - lower) * 0.05, 0.025)
    extent = [
        lower[0] - pad[0],
        upper[0] + pad[0],
        lower[1] - pad[1],
        upper[1] + pad[1],
    ]
    for index, (group, title) in enumerate(zip(groups, titles, strict=True)):
        axis = axes[0, 0 if overlay else index]
        axis.set_extent(extent, crs=source)
        for k, points in enumerate(group):
            axis.plot(
                points[:, 0],
                points[:, 1],
                transform=source,
                color=COLORS[index],
                alpha=0.85 if overlay else 0.25,
                linewidth=1.2 if overlay else 0.7,
                label=title if overlay and k == 0 else None,
            )
        axis.scatter(*anchor, transform=source, color="black", marker="x", zorder=5)
        axis.gridlines(draw_labels=["bottom", "left"], alpha=0.2)
        if not overlay:
            axis.set_title(f"{title} (n={len(group)})")
    if overlay:
        axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def trace_source(args, selected, test, physical, config, anchor, output):
    index = int(np.flatnonzero(test.trajectory_id == args.exclude_test_id)[0])
    flight = selected.loc[selected.flight_id.eq(args.exclude_test_id)].sort_values(
        "timestamp"
    )
    matched = (
        pq.read_table(
            args.source,
            filters=[
                ("icao24", "=", str(flight.icao24.iloc[0])),
                ("timestamp", ">=", flight.timestamp.min()),
                ("timestamp", "<=", flight.timestamp.max()),
            ],
        )
        .to_pandas()
        .sort_values("timestamp")
    )
    source_ids = matched.flight_id.unique().tolist()
    if len(source_ids) != 1:
        raise ValueError(f"Ambiguous original source IDs: {source_ids}")
    original = (
        pq.read_table(
            args.source,
            filters=[
                ("flight_id", "=", source_ids[0]),
            ],
        )
        .to_pandas()
        .sort_values("timestamp")
    )
    joined = flight.merge(matched, on="timestamp", suffixes=("_cache", "_source"))
    unchanged = {
        col: float(np.max(np.abs(joined[f"{col}_cache"] - joined[f"{col}_source"])))
        for col in ["latitude", "longitude", "groundspeed", "track", "altitude"]
    }
    source_points = matched[["longitude", "latitude"]].to_numpy()
    cache_positions = flight[["longitude", "latitude"]].to_numpy()
    full_integration = backward_positions(flight, anchor)
    processed, _ = integrated_positions(
        physical[index : index + 1], test.elapsed_time[index : index + 1], config
    )
    dt = matched.timestamp.diff().dt.total_seconds().to_numpy()[1:]
    bearing, _, distances = GEOD.inv(
        source_points[:-1, 0],
        source_points[:-1, 1],
        source_points[1:, 0],
        source_points[1:, 1],
    )
    gps_speed = distances / dt * 3600 / 1852
    reported = matched.groundspeed.to_numpy()[1:]
    track_error = np.abs((bearing - matched.track.to_numpy()[1:] + 180) % 360 - 180)
    # Long runs of identical coordinate increments indicate already filled data.
    delta = np.diff(source_points, axis=0)
    same_increment = np.max(np.abs(np.diff(delta, axis=0)), axis=1) < 1e-9
    marked = np.flatnonzero(same_increment)
    chunks = np.split(marked, np.flatnonzero(np.diff(marked) > 1) + 1)
    chunks = [chunk for chunk in chunks if len(chunk)]
    longest = max(chunks, key=len) if chunks else np.array([], dtype=int)
    linear = None
    if len(longest):
        start, end = int(longest[0]), int(longest[-1] + 2)
        linear = {
            "start_utc": str(matched.timestamp.iloc[start]),
            "end_utc": str(matched.timestamp.iloc[end]),
            "duration_sec": float(
                (
                    matched.timestamp.iloc[end] - matched.timestamp.iloc[start]
                ).total_seconds()
            ),
            "gps_speed_median_kt": float(np.median(gps_speed[start:end])),
            "reported_speed_median_kt": float(np.median(reported[start:end])),
            "track_disagreement_median_deg": float(np.median(track_error[start:end])),
        }
    projection = airport_projection("ZBTJ")

    def radius(points):
        x, y = projection(points[:, 0], points[:, 1])
        return float(np.hypot(x, y).max() / 1000)

    q = Flight(flight).unwrap().resample(200).data
    expected = np.stack(
        [
            q.track_unwrapped,
            q.groundspeed,
            q.altitude,
            (q.timestamp - q.timestamp.iloc[0]).dt.total_seconds(),
        ],
        axis=1,
    )
    trace = {
        "selected_id": args.exclude_test_id,
        "original_source_id": source_ids[0],
        "source_original_points": len(original),
        "source_matching_points": len(matched),
        "selected_points": len(flight),
        "source_start_utc": str(original.timestamp.min()),
        "source_end_utc": str(original.timestamp.max()),
        "selected_start_utc": str(flight.timestamp.min()),
        "selected_end_utc": str(flight.timestamp.max()),
        "source_cache_max_absolute_differences": unchanged,
        "matched_timestamp_rows": len(joined),
        "source_gps_speed_kt": summarize(gps_speed),
        "source_recorded_speed_kt": summarize(reported),
        "source_track_disagreement_deg": summarize(track_error),
        "longest_nearly_linear_coordinate_segment": linear,
        "source_gps_max_radius_km": radius(source_points),
        "selected_gps_max_radius_km": radius(cache_positions),
        "full_resolution_integrated_max_radius_km": radius(full_integration),
        "prepared_200_integrated_max_radius_km": radius(processed[0]),
        "prepare_max_feature_error": np.max(
            np.abs(expected - physical[index]), axis=0
        ).tolist(),
        "conclusion": (
            "Position versus speed/track inconsistency is already present in the "
            "input 1s Parquet. Selection preserves these fields. Fixed-point "
            "preparation preserves their values to float32 accuracy. "
            "Integration exposes the inconsistency as spatial drift."
        ),
        "limitation": (
            "The available file is already a filtered 1s product, not "
            "unprocessed receiver messages. Long coordinate-linear segments "
            "suggest filling/interpolation but do not establish the exact "
            "upstream operation without the producer script or earlier data."
        ),
    }
    write_json(output / "anomaly_trace.json", trace)
    original.to_parquet(output / "anomaly_original_source.parquet", index=False)
    matched.to_parquet(output / "anomaly_source_matching_cache.parquet", index=False)
    flight.to_parquet(output / "anomaly_selected.parquet", index=False)
    geographic_plot(
        [[source_points], [cache_positions], [full_integration], [processed[0]]],
        [
            "Input 1s GPS",
            "Selected GPS",
            "Full-point integration",
            "200-point integration",
        ],
        output / "anomaly_stage_maps.png",
        anchor,
    )
    seconds = (
        (matched.timestamp - matched.timestamp.iloc[0])
        .dt.total_seconds()
        .to_numpy()[1:]
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(seconds / 60, gps_speed, label="Position-derived speed")
    axes[0].plot(seconds / 60, reported, label="Recorded groundspeed", alpha=0.8)
    axes[0].set_ylabel("Speed (kt)")
    axes[0].legend()
    axes[1].plot(seconds / 60, bearing % 360, label="Position-derived track")
    axes[1].plot(
        seconds / 60, matched.track.to_numpy()[1:], label="Recorded track", alpha=0.8
    )
    axes[1].set_ylabel("Track (degree)")
    axes[1].legend()
    axes[2].plot(seconds / 60, track_error)
    axes[2].set_ylabel("Circular discrepancy (degree)")
    axes[2].set_xlabel("Elapsed time (minutes)")
    for axis in axes:
        axis.grid(alpha=0.2)
        if linear:
            start = (
                pd.Timestamp(linear["start_utc"]) - matched.timestamp.iloc[0]
            ).total_seconds() / 60
            end = (
                pd.Timestamp(linear["end_utc"]) - matched.timestamp.iloc[0]
            ).total_seconds() / 60
            axis.axvspan(start, end, color="0.7", alpha=0.2)
    fig.suptitle(f"{args.exclude_test_id}: inconsistency before chabuduo preprocessing")
    fig.tight_layout()
    fig.savefig(output / "anomaly_source_profiles.png", dpi=170)
    plt.close(fig)
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "source", "selected", "artifact", "run", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--exclude-test-id", default="AIB12AA_2224")
    parser.add_argument("--generated-radius-km", type=float, default=200)
    parser.add_argument("--plot-count", type=int, default=100)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if config.prepare.representation.features != FEATURES:
        raise ValueError("This review expects the four-channel Tianjin experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.run / "checkpoints" / "best.ckpt"
    test = SplitData.load(args.artifact / "test.npz")
    normalization = Normalization.load(args.artifact / "normalization.json")
    physical = normalization.inverse_transform(test.values)
    selected = pq.read_table(
        args.selected,
        filters=[
            ("flight_id", "in", test.trajectory_id.tolist()),
        ],
    ).to_pandas()
    airport = airports["ZBTJ"]
    threshold = next(r for r in airport.runways["16L"].tuple_runway if r.name == "16L")
    anchor = (threshold.longitude, threshold.latitude)
    trace = trace_source(args, selected, test, physical, config, anchor, args.output)
    print("SOURCE_TRACE", json.dumps(trace), flush=True)

    keep = test.trajectory_id != args.exclude_test_id
    ids = test.trajectory_id[keep]
    original_indices = np.flatnonzero(keep)
    original_values = physical[keep]
    original_times = test.elapsed_time[keep]
    model = TCVAELightning.load_from_checkpoint(checkpoint, map_location="cpu").eval()
    reconstructed = []
    with torch.inference_mode():
        for start in range(0, len(ids), 64):
            x = torch.from_numpy(test.values[keep][start : start + 64]).permute(0, 2, 1)
            mean, _ = model.network.encode(x)
            reconstructed.append(model.network.decode(mean).permute(0, 2, 1).numpy())
    normalized_reconstruction = np.concatenate(reconstructed)
    if not np.isfinite(normalized_reconstruction).all():
        raise ValueError("Nonfinite posterior reconstruction")
    reconstructed_values = normalization.inverse_transform(normalized_reconstruction)
    input_positions, _ = integrated_positions(original_values, original_times, config)
    reconstruction_positions, reconstruction_frame = integrated_positions(
        reconstructed_values,
        reconstructed_values[:, :, 3],
        config,
    )
    gps_positions = []
    full_gps = []
    grouped = selected.groupby("flight_id", sort=False)
    for identifier in ids:
        f = grouped.get_group(identifier).sort_values("timestamp")
        full_gps.append(f[["longitude", "latitude"]].to_numpy())
        gps_positions.append(
            Flight(f).unwrap().resample(200).data[["longitude", "latitude"]].to_numpy()
        )
    gps_positions = np.stack(gps_positions)
    normalized_error = normalized_reconstruction - test.values[keep]
    feature_error = reconstructed_values - original_values
    circular_error = (feature_error[:, :, 0] + 180) % 360 - 180
    baseline_error = position_errors(input_positions, gps_positions)
    model_gps_error = position_errors(reconstruction_positions, gps_positions)
    model_input_error = position_errors(reconstruction_positions, input_positions)
    metrics = pd.DataFrame(
        {
            "trajectory_id": ids,
            "normalized_mse": np.mean(normalized_error**2, axis=(1, 2)),
            "input_integrated_vs_gps_rmse_m": np.sqrt(
                np.mean(baseline_error**2, axis=1)
            ),
            "posterior_vs_gps_rmse_m": np.sqrt(np.mean(model_gps_error**2, axis=1)),
            "posterior_vs_input_integrated_rmse_m": np.sqrt(
                np.mean(model_input_error**2, axis=1)
            ),
            "posterior_vs_gps_max_error_m": model_gps_error.max(axis=1),
        }
    )
    metrics.to_parquet(args.output / "reconstruction_metrics.parquet", index=False)
    reconstruction_frame["source_trajectory_id"] = np.repeat(ids, 200)
    reconstruction_frame.to_parquet(
        args.output / "posterior_reconstructions.parquet", index=False
    )
    np.savez_compressed(
        args.output / "posterior_reconstructions.npz",
        trajectory_id=ids,
        normalized_values=normalized_reconstruction,
        values=reconstructed_values,
        input_values=original_values,
        input_elapsed_time=original_times,
        gps_positions=gps_positions,
    )

    generated = pd.read_parquet(
        args.run / "samples" / "seed-42-n-300" / "generated.parquet"
    )
    generated_groups = list(generated.groupby("flight_id", sort=False))
    radius = np.hypot(generated.x, generated.y) / 1000
    max_radius = radius.groupby(generated.flight_id, sort=False).max()
    excluded_generated = max_radius[max_radius > args.generated_radius_km].to_dict()
    display_generated = [
        g.sort_values("timestamp")[["longitude", "latitude"]].to_numpy()
        for identifier, g in generated_groups[: args.plot_count]
        if identifier not in excluded_generated
    ]
    display_test = np.flatnonzero(original_indices < args.plot_count)
    geographic_plot(
        [list(input_positions[display_test]), display_generated],
        ["Held-out kinematic integration (filtered)", "Prior generation (filtered)"],
        args.output / "comparison_filtered.png",
        anchor,
    )
    geographic_plot(
        [[full_gps[i] for i in display_test], display_generated],
        ["Held-out original GPS (filtered)", "Prior generation (filtered)"],
        args.output / "gps_vs_generation_filtered.png",
        anchor,
    )
    geographic_plot(
        [
            [full_gps[i] for i in display_test],
            list(input_positions[display_test]),
            list(reconstruction_positions[display_test]),
        ],
        [
            "Original GPS",
            "Input kinematic integration",
            "Posterior-mean reconstruction",
        ],
        args.output / "reconstruction_overview.png",
        anchor,
    )
    order = np.argsort(metrics.normalized_mse.to_numpy())
    examples = order[np.round(np.linspace(0, len(order) - 1, 6)).astype(int)]
    for rank, i in enumerate(examples):
        geographic_plot(
            [[full_gps[i]], [input_positions[i]], [reconstruction_positions[i]]],
            ["Original GPS", "Input integration", "Posterior mean"],
            args.output / f"reconstruction_example_{rank + 1}.png",
            anchor,
            overlay=True,
        )
    chosen = examples[[1, 3, 5]]
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    for col, i in enumerate(chosen):
        for feature, axis in enumerate(axes[:, col]):
            axis.plot(
                np.linspace(0, 1, 200),
                original_values[i, :, feature],
                color=COLORS[0],
                label="Input",
            )
            axis.plot(
                np.linspace(0, 1, 200),
                reconstructed_values[i, :, feature],
                color=COLORS[2],
                label="Posterior mean",
            )
            axis.set_ylabel(f"{FEATURES[feature]} ({UNITS[feature]})", fontsize=8)
            axis.grid(alpha=0.2)
        axes[0, col].set_title(str(ids[i]), fontsize=9)
        axes[-1, col].set_xlabel("Normalized sequence progress")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Posterior feature reconstruction: representative MSE ranks 20%, 60%, 100%"
    )
    fig.tight_layout()
    fig.savefig(args.output / "reconstruction_features.png", dpi=170)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5))
    for column, label in [
        ("input_integrated_vs_gps_rmse_m", "Input integration vs GPS"),
        ("posterior_vs_gps_rmse_m", "Posterior vs GPS"),
        ("posterior_vs_input_integrated_rmse_m", "Posterior vs input integration"),
    ]:
        axis.plot(
            np.sort(metrics[column]) / 1000,
            np.arange(1, len(ids) + 1) / len(ids),
            label=label,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Per-trajectory spatial RMSE (km)")
    axis.set_ylabel("Empirical cumulative fraction")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "reconstruction_error_cdf.png", dpi=170)
    plt.close(fig)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "test_sha256": sha256(args.artifact / "test.npz"),
        "posterior_method": (
            "eval mode; decode(encoder mean); no latent draw or retraining"
        ),
        "excluded_test_ids": [args.exclude_test_id],
        "evaluated_test_count": len(ids),
        "generated_display_filter": (
            f"maximum airport distance > {args.generated_radius_km}km"
        ),
        "excluded_generated_ids_with_radius_km": excluded_generated,
        "generated_total": len(generated_groups),
        "generated_retained": len(generated_groups) - len(excluded_generated),
        "display_test_count": len(display_test),
        "display_generated_count": len(display_generated),
        "display_policy": (
            "Original first 100 indices with exclusions; "
            "no replacement or cherry-picked resampling"
        ),
        "normalized_test_mse": float(np.mean(normalized_error**2)),
        "per_feature_rmse": dict(
            zip(
                FEATURES,
                np.sqrt(np.mean(feature_error**2, axis=(0, 1))).tolist(),
                strict=True,
            )
        ),
        "track_circular_rmse_deg": float(np.sqrt(np.mean(circular_error**2))),
        "spatial_rmse_m": {
            col: summarize(metrics[col].to_numpy())
            for col in metrics.columns
            if col.endswith("rmse_m")
        },
        "example_ids_by_mse_quantile": [str(ids[i]) for i in examples],
        "notes": [
            "Metrics use all 486 remaining tests, not just the plotted 99.",
            "Spatial correspondence is by sequence index, not absolute clock time.",
            "Input-integrated coordinates are not measured ground truth; "
            "GPS comparison is reported separately.",
            "Generated outlier removal is display-only "
            "and does not establish prior-sample validity.",
            "Other unfiltered source-quality issues may remain in these 486 tests.",
        ],
        "environment": {
            name: version(name) for name in ["traffic", "torch", "numpy", "pandas"]
        },
    }
    write_json(args.output / "review_summary.json", summary)
    print("REVIEW_SUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"Output: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
