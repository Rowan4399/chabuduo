"""Decode the K=256 PCA samples and compare paired reconstructions and generation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cartes.crs import Mercator, PlateCarree
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from traffic.core import Traffic
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData
from chabuduo.config import load_experiment_config
from chabuduo.models.tcvae import TCVAELightning
from chabuduo.prepare import airport_projection
from chabuduo.sampling import build_generated_traffic


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() in (
        args.run.resolve(), args.artifact.resolve(), args.latents.resolve()
    ):
        raise ValueError("Use a separate analysis output directory")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    config = load_experiment_config(args.config)
    run = json.loads((args.run / "run_summary.json").read_text())
    latent_manifest = json.loads((args.latents / "manifest.json").read_text())
    checkpoint = args.run / "checkpoints/best.ckpt"
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != run["best"]["sha256"]:
        raise ValueError("Checkpoint differs from this run's selected checkpoint")
    if checkpoint_hash != latent_manifest["checkpoint_sha256"]:
        raise ValueError("PCA latents came from another checkpoint")
    for filename in ("test.npz", "normalization.json"):
        if digest(args.artifact / filename) != run["artifact"]["files"][filename]:
            raise ValueError(f"Artifact mismatch: {filename}")
    if config.prepare.representation.features != [
        "track_unwrapped",
        "groundspeed_kt",
        "altitude_ft",
        "elapsed_time_sec",
    ]:
        raise ValueError("Unexpected feature order")
    test = SplitData.load(args.artifact / "test.npz")
    normalization = Normalization.load(args.artifact / "normalization.json")
    with np.load(args.latents / "latents.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(test.trajectory_id, data["test_trajectory_id"])
        indices = data["test_indices"][: args.count]
        latent = {
            "posterior_mean": data["test_mu"][indices],
            "posterior_sample": data["test_sample"][: args.count],
            "vamp_prior": data["vamp_sample"][: args.count],
        }
        components = data["vamp_component_id"][: args.count]
    if len(indices) != args.count:
        raise ValueError("Requested more samples than were exported")
    ids = test.trajectory_id[indices]
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = (
        TCVAELightning.load_from_checkpoint(
            checkpoint, map_location=device, weights_only=False
        )
        .to(device)
        .eval()
    )
    module.requires_grad_(False)
    if module.network.prior_type != "vamp" or module.network.prior_components != 256:
        raise ValueError("Expected VampPrior K=256")
    if (
        digest(Path(inspect.getfile(TCVAELightning)))
        != latent_manifest["model_source_sha256"]
    ):
        raise ValueError("Model source changed since the latent export")
    normalized = {"test": test.values[indices]}
    with torch.inference_mode():
        # Reuse the export batches to avoid batch-dependent CUDA rounding changes.
        means = []
        for start in range(0, len(test.values), 256):
            mean, _ = module.network.encode(
                torch.from_numpy(test.values[start : start + 256])
                .to(device)
                .permute(0, 2, 1)
            )
            means.append(mean.cpu().numpy())
        verified_mean = np.concatenate(means)[indices]
        np.testing.assert_allclose(
            verified_mean, latent["posterior_mean"], atol=2e-4, rtol=2e-4
        )
        for name, z in latent.items():
            normalized[name] = (
                module.network.decode(torch.from_numpy(z).to(device))
                .permute(0, 2, 1)
                .cpu()
                .numpy()
            )
    raw = {name: normalization.inverse_transform(v) for name, v in normalized.items()}
    times = {
        name: test.elapsed_time[indices] if name == "test" else v[:, :, 3]
        for name, v in raw.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    tables, repairs, xy = {}, {}, {}
    projection = airport_projection(config.preprocess.airport)
    for name, values in raw.items():
        if values.shape != (args.count, 200, 4) or not np.isfinite(values).all():
            raise ValueError(f"Invalid values: {name}")
        elapsed = times[name]
        repairs[name] = {
            "speed_points_clipped": int(
                ((values[:, :, 1] < 0) | (values[:, :, 1] > 600)).sum()
            ),
            "altitude_points_clipped": int(
                ((values[:, :, 2] < -1000) | (values[:, :, 2] > 60000)).sum()
            ),
            "time_steps_raised_to_1ms": int((np.diff(elapsed, axis=1) < 1e-3).sum()),
        }
        t = build_generated_traffic(values, elapsed, config).compute_xy(
            projection=projection
        )
        frame = t.data.copy()
        group_ids = frame.flight_id.unique()
        if len(group_ids) != args.count or len(frame) != args.count * 200:
            raise ValueError(f"Unexpected number of trajectories/points: {name}")
        frame["sample_index"] = frame.flight_id.map(
            dict(zip(group_ids, range(args.count), strict=True))
        )
        frame["point_index"] = frame.groupby("sample_index", sort=False).cumcount()
        frame = frame.sort_values(["sample_index", "point_index"])
        frame["source_trajectory_id"] = (
            frame.sample_index.map(dict(enumerate(ids)))
            if name != "vamp_prior"
            else "unconditional"
        )
        if not np.isfinite(frame[["longitude", "latitude", "x", "y"]]).all().all():
            raise ValueError("Nonfinite reconstructed geography")
        tables[name] = frame
        xy[name] = frame[["x", "y"]].to_numpy().reshape(args.count, 200, 2)
        frame.to_parquet(args.output / f"{name}.parquet", index=False)
        print(f"{name}: {args.count} trajectories; repairs={repairs[name]}", flush=True)

    airport = airports["LSZH"]
    runway = airport.runways["14"]
    threshold = next(p for p in runway.tuple_runway if p.name == "14")
    combined = pd.concat(tables.values())
    low = combined[["longitude", "latitude"]].min().to_numpy()
    high = combined[["longitude", "latitude"]].max().to_numpy()
    margin = np.maximum((high - low) * 0.05, 0.005)
    extent = [
        low[0] - margin[0],
        high[0] + margin[0],
        low[1] - margin[1],
        high[1] + margin[1],
    ]
    colors = {
        "test": "#2C73B9",
        "posterior_mean": "#7255A5",
        "posterior_sample": "#008F82",
        "vamp_prior": "#E69526",
    }
    titles = {
        "test": "Held-out test (kinematic reference)",
        "posterior_mean": "Posterior-mean reconstruction",
        "posterior_sample": "Posterior-sample reconstruction",
        "vamp_prior": "TCVAE generation: VampPrior K=256",
    }

    def map_style(axis):
        axis.set_extent(extent, crs=PlateCarree())
        axis.scatter(
            threshold.longitude,
            threshold.latitude,
            transform=PlateCarree(),
            marker="x",
            s=40,
            color="black",
            zorder=5,
        )
        # Rectangular Mercator ticks avoid Gridliner's failing boundary intersection.
        axis.set_xticks(np.linspace(extent[0], extent[1], 4), crs=PlateCarree())
        axis.set_yticks(np.linspace(extent[2], extent[3], 4), crs=PlateCarree())
        axis.xaxis.set_major_formatter(LongitudeFormatter(number_format=".2f"))
        axis.yaxis.set_major_formatter(LatitudeFormatter(number_format=".2f"))
        axis.tick_params(labelsize=8)
        axis.grid(linewidth=0.5, alpha=0.25)

    figure, axes = plt.subplots(
        2, 2, figsize=(13, 11), subplot_kw={"projection": Mercator()}
    )
    for axis, name in zip(axes.flat, tables, strict=True):
        t = Traffic(tables[name])
        t.plot(axis, color=colors[name], alpha=0.28, linewidth=0.8)
        map_style(axis)
        axis.set_title(f"{titles[name]} (n={args.count})", fontsize=11)
    figure.suptitle(
        f"Zurich single-flow K=256 | epoch {run['best']['epoch']}\n"
        "Same map extent and RWY14 endpoint; samples matched to the PCA analysis",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.93))
    figure.savefig(args.output / "trajectory_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        3, 3, figsize=(13, 11), subplot_kw={"projection": Mercator()}
    )
    for i, axis in enumerate(axes.flat):
        for name in ("test", "posterior_mean", "posterior_sample"):
            t = Traffic(tables[name].loc[tables[name].sample_index.eq(i)])
            t.plot(
                axis,
                color=colors[name],
                alpha=0.85,
                linewidth=1.3,
                linestyle="-" if name == "test" else "--",
                label=titles[name],
            )
        map_style(axis)
        axis.set_title(f"Selected test #{i + 1}", fontsize=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(
        "First 9 selected test trajectories: paired reconstructions\n"
        "Blue: test | Purple dashed: posterior mean | Green dashed: posterior draw",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    figure.savefig(args.output / "paired_reconstructions.png", dpi=180)
    plt.close(figure)

    errors = {}
    for name in ("posterior_mean", "posterior_sample"):
        distances = np.linalg.norm(xy[name] - xy["test"], axis=2) / 1000
        errors[name] = {
            "normalized_feature_mse": float(
                np.mean((normalized[name] - normalized["test"]) ** 2)
            ),
            "pointwise_xy_error_km_median": float(np.median(distances)),
            "pointwise_xy_error_km_p90": float(np.percentile(distances, 90)),
        }
    np.savez_compressed(
        args.output / "decoded_trajectories.npz",
        **{f"raw_{k}": v for k, v in raw.items()},
        **{f"normalized_{k}": v for k, v in normalized.items()},
        **{f"latent_{k}": v for k, v in latent.items()},
        trajectory_id=ids,
        test_indices=indices,
        vamp_component_id=components,
    )
    summary = {
        "checkpoint_sha256": checkpoint_hash,
        "cached_mean_max_abs_error": float(
            np.max(np.abs(verified_mean - latent["posterior_mean"]))
        ),
        "epoch": run["best"]["epoch"],
        "count_per_source": args.count,
        "test_trajectory_ids": ids.tolist(),
        "selection": "First 100 of the existing 500 PCA samples; no cherry-picking",
        "input_sha256": {
            "test.npz": digest(args.artifact / "test.npz"),
            "normalization.json": digest(args.artifact / "normalization.json"),
            "latent_samples.npz": digest(args.latents / "latents.npz"),
            "experiment_yaml": digest(args.config),
        },
        "latent_seeds": latent_manifest["seeds"],
        "projection": "cartes.crs.Mercator; PlateCarree source coordinates",
        "geographic_extent": extent,
        "endpoint": "RWY14 threshold, anchored by the shared reconstruction interface",
        "reference": (
            "Held-out kinematic features reconstructed with the same interface; "
            "not raw observed lat/lon"
        ),
        "postprocessing": (
            "Existing build_generated_traffic: wrap track; clip speed [0,600] kt, "
            "altitude [-1000,60000] ft; minimum time increment 1ms"
        ),
        "postprocessing_counts": repairs,
        "reconstruction_errors": errors,
        "source_script_sha256": digest(Path(__file__)),
        "sampling_source_sha256": digest(
            Path(inspect.getfile(build_generated_traffic))
        ),
        "execution": (
            f"remote frozen decoder on {device}; no training; copy outputs locally"
        ),
        "limitation": (
            "Shared endpoint and good reconstruction do not establish physical realism"
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(errors, indent=2))
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
