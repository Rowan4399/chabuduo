from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from traffic.algorithms.generation import Generation as TrafficGeneration
from traffic.core import Traffic
from traffic.data import airports

from chabuduo.artifacts import (
    Normalization,
    SplitData,
    prepared_artifact_identity,
    sha256,
    write_json,
)
from chabuduo.config import ExperimentConfig
from chabuduo.models.ddpm import sample_ddpm
from chabuduo.models.latent_ddpm import sample_latent_ddpm
from chabuduo.models.tcvae import sample_tcvae
from chabuduo.prepare import airport_projection


def _channels(values: np.ndarray, config: ExperimentConfig) -> dict[str, np.ndarray]:
    features = config.prepare.representation.features
    return {name: values[:, :, index] for index, name in enumerate(features)}


def _kinematic_matrix(
    channels: dict[str, np.ndarray],
    elapsed_time: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    if {"track_sin", "track_cos"} <= channels.keys():
        track = np.degrees(np.arctan2(channels["track_sin"], channels["track_cos"]))
    elif "track_unwrapped" in channels:
        track = channels["track_unwrapped"]
    else:
        raise ValueError("Kinematic reconstruction requires track features")

    try:
        groundspeed = channels["groundspeed_kt"]
        altitude = channels["altitude_ft"]
    except KeyError as error:
        raise ValueError(
            "Kinematic reconstruction requires groundspeed_kt and altitude_ft"
        ) from error

    result = np.stack((track, groundspeed, altitude, elapsed_time), axis=2)
    result[:, :, 0] %= 360
    if config.sampling.type in {"ddpm", "latent_ddpm"}:
        minimum_speed = config.sampling.minimum_groundspeed_kt
        maximum_speed = config.sampling.maximum_groundspeed_kt
        minimum_altitude = config.sampling.minimum_altitude_ft
        maximum_altitude = config.sampling.maximum_altitude_ft
    else:
        minimum_speed, maximum_speed = 0, 600
        minimum_altitude, maximum_altitude = -1000, 60000
    result[:, :, 1] = np.clip(result[:, :, 1], minimum_speed, maximum_speed)
    result[:, :, 2] = np.clip(result[:, :, 2], minimum_altitude, maximum_altitude)
    elapsed = result[:, :, 3] - result[:, :1, 3]
    increments = np.diff(elapsed, axis=1, prepend=elapsed[:, :1])
    increments[:, 0] = 0
    increments[:, 1:] = np.maximum(increments[:, 1:], 1e-3)
    result[:, :, 3] = np.cumsum(increments, axis=1)
    return result


def build_generated_traffic(
    values: np.ndarray,
    elapsed_time: np.ndarray,
    config: ExperimentConfig,
) -> Traffic:
    """Reconstruct samples using feature names from the experiment YAML."""
    channels = _channels(values, config)
    projection = airport_projection(config.preprocess.airport)
    if {"east_m", "north_m", "altitude_m"} <= channels.keys():
        matrix = np.stack(
            (
                channels["east_m"],
                channels["north_m"],
                channels["altitude_m"] / 0.3048,
                elapsed_time,
            ),
            axis=2,
        )
        adapter = TrafficGeneration(None, ["x", "y", "altitude", "timedelta"])
        return adapter.build_traffic(
            matrix.reshape(len(values), -1), projection=projection
        )

    airport = airports[config.preprocess.airport]
    if airport is None:
        raise ValueError(f"Unknown airport: {config.preprocess.airport}")
    latitude, longitude = airport.latitude, airport.longitude
    if config.prepare.representation.anchor == "runway_threshold":
        runways = config.preprocess.runways
        if runways is None or len(runways) != 1:
            raise ValueError("runway_threshold anchor requires exactly one runway")
        runway_name = runways[0]
        runway = airport.runways[runway_name]
        threshold = next(
            point for point in runway.tuple_runway if point.name == runway_name
        )
        latitude, longitude = threshold.latitude, threshold.longitude
    matrix = _kinematic_matrix(channels, elapsed_time, config)
    adapter = TrafficGeneration(None, ["track", "groundspeed", "altitude", "timedelta"])
    return adapter.build_traffic(
        matrix.reshape(len(values), -1),
        coordinates={
            "latitude": latitude,
            "longitude": longitude,
        },
        forward=False,
    )


def _normalized_samples(
    run: Path,
    artifact: Path,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    count = config.sampling.count
    if config.model.type == "tcvae" and config.sampling.type == "tcvae":
        checkpoint = run / "checkpoints" / "best.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        return (
            sample_tcvae(
                checkpoint,
                count,
                temperature=config.sampling.temperature,
            ),
            None,
            {},
        )

    if config.model.type == "ddpm" and config.sampling.type == "ddpm":
        checkpoint = run / "checkpoints" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if config.model.conditioning is None:
            condition_ids = np.zeros(count, dtype=np.int64)
        else:
            metadata = pd.read_parquet(artifact / "metadata.parquet")
            feature = config.model.conditioning.feature
            available = np.sort(
                metadata.loc[metadata.split.eq("train"), feature]
                .unique()
                .astype(np.int64)
            )
            if len(available) == 0:
                raise ValueError("No training condition IDs are available")
            condition_ids = np.resize(available, count)
            np.random.default_rng(config.sampling.seed).shuffle(condition_ids)
        return (
            sample_ddpm(
                checkpoint,
                condition_ids,
                config.prepare.points,
                steps=config.sampling.steps,
                eta=config.sampling.eta,
            ),
            condition_ids,
            {},
        )
    if config.model.type == "latent_ddpm" and config.sampling.type == "latent_ddpm":
        checkpoint = run / "checkpoints" / "best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["artifact_sha256"] != prepared_artifact_identity(artifact)["sha256"]:
            raise ValueError(
                "Prepared artifact does not match the latent DDPM checkpoint"
            )
        normalized, latent, standardized = sample_latent_ddpm(
            checkpoint,
            count,
            steps=config.sampling.steps,
            eta=config.sampling.eta,
        )
        return (
            normalized,
            None,
            {
                "generated_latent": latent,
                "generated_latent_standardized": standardized,
            },
        )
    raise ValueError("Model and sampling types do not match")


def sample_run(
    run: Path,
    artifact: Path,
    config: ExperimentConfig,
    *,
    output: Path | None = None,
) -> Path:
    """Dispatch sampling using the model selected in the experiment YAML."""
    count = config.sampling.count
    seed = config.sampling.seed
    suffix = f"seed-{seed}-n-{count}"
    if config.sampling.type == "tcvae" and config.sampling.temperature != 1.0:
        suffix += f"-t-{config.sampling.temperature:g}"
    if config.sampling.type in {"ddpm", "latent_ddpm"}:
        suffix += f"-steps-{config.sampling.steps}-eta-{config.sampling.eta:g}"
    output = output or run / "samples" / suffix
    if output.exists():
        raise FileExistsError(f"{output} already exists")

    torch.manual_seed(seed)
    normalized, condition_ids, extra_arrays = _normalized_samples(run, artifact, config)
    output.mkdir(parents=True)
    normalization = Normalization.load(artifact / "normalization.json")
    values = normalization.inverse_transform(normalized)
    channels = _channels(values, config)

    if "elapsed_time_sec" in channels:
        elapsed_time = channels["elapsed_time_sec"]
    else:
        timing_source = SplitData.load(artifact / "train.npz")
        rng = np.random.default_rng(seed)
        timing_indices = rng.integers(0, len(timing_source.elapsed_time), size=count)
        elapsed_time = timing_source.elapsed_time[timing_indices]

    t = build_generated_traffic(values, elapsed_time, config)
    if not {"x", "y"} <= set(t.data.columns):
        t = t.compute_xy(projection=airport_projection(config.preprocess.airport))
    frame = t.data.copy()
    frame["elapsed_time"] = frame["timedelta"].to_numpy()
    for name, channel in channels.items():
        frame[name] = channel.ravel()
    if config.preprocess.runways is not None and len(config.preprocess.runways) == 1:
        frame["runway"] = config.preprocess.runways[0]
    if condition_ids is not None:
        frame["model_condition_id"] = np.repeat(condition_ids, config.prepare.points)
    elapsed_time = frame["elapsed_time"].to_numpy(dtype=np.float32).reshape(count, -1)
    arrays = {"values": values, "elapsed_time": elapsed_time}
    arrays.update(extra_arrays)
    if condition_ids is not None:
        arrays["model_condition_id"] = condition_ids
    np.savez_compressed(output / "samples.npz", **arrays)
    frame.to_parquet(output / "generated.parquet", index=False)
    if config.model.type == "latent_ddpm":
        checkpoint = run / "checkpoints" / "best.pt"
        write_json(
            output / "sample_summary.json",
            {
                "model_type": config.model.type,
                "config_name": config.name,
                "artifact": prepared_artifact_identity(artifact),
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "sha256": sha256(checkpoint),
                },
                "sampling": config.sampling.model_dump(),
                "representation": config.prepare.representation.model_dump(),
                "sequence_length": config.prepare.points,
                "postprocessing": {
                    "track": "wrapped to [0, 360) degrees",
                    "groundspeed_altitude": "clipped to sampling limits",
                    "elapsed_time": "zero origin; increments at least 0.001 seconds",
                    "reconstruction": "backwards from configured endpoint anchor",
                },
            },
        )
    return output
