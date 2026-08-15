from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from traffic.algorithms.generation import Generation as TrafficGeneration
from traffic.core import Traffic
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData
from chabuduo.config import ExperimentConfig
from chabuduo.prepare import airport_projection
from chabuduo.training import TCVAELightning


def _channels(values: np.ndarray, config: ExperimentConfig) -> dict[str, np.ndarray]:
    features = config.prepare.representation.features
    return {name: values[:, :, index] for index, name in enumerate(features)}


def _kinematic_matrix(
    channels: dict[str, np.ndarray], elapsed_time: np.ndarray
) -> np.ndarray:
    if {"track_sin", "track_cos"} <= channels.keys():
        track = np.degrees(
            np.arctan2(channels["track_sin"], channels["track_cos"])
        )
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
    result[:, :, 1] = np.clip(result[:, :, 1], 0, 600)
    result[:, :, 2] = np.clip(result[:, :, 2], -1000, 60000)
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
        adapter = TrafficGeneration(
            None, ["x", "y", "altitude", "timedelta"]
        )
        return adapter.build_traffic(
            matrix.reshape(len(values), -1), projection=projection
        )

    airport = airports[config.preprocess.airport]
    if airport is None:
        raise ValueError(f"Unknown airport: {config.preprocess.airport}")
    matrix = _kinematic_matrix(channels, elapsed_time)
    adapter = TrafficGeneration(
        None, ["track", "groundspeed", "altitude", "timedelta"]
    )
    return adapter.build_traffic(
        matrix.reshape(len(values), -1),
        coordinates={
            "latitude": airport.latitude,
            "longitude": airport.longitude,
        },
        forward=False,
    )


def sample_run(
    run: Path,
    artifact: Path,
    config: ExperimentConfig,
    *,
    output: Path | None = None,
) -> Path:
    """Sample a trained run using the original experiment configuration."""
    if config.model.type != "tcvae" or config.sampling.type != "tcvae":
        raise NotImplementedError("DDPM sampling is not implemented yet")
    checkpoint = run / "checkpoints" / "best.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    count = config.sampling.count
    seed = config.sampling.seed
    temperature = config.sampling.temperature
    suffix = f"seed-{seed}-n-{count}"
    if temperature != 1.0:
        suffix += f"-t-{temperature:g}"
    output = output or run / "samples" / suffix
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    output.mkdir(parents=True)

    torch.manual_seed(seed)
    module = TCVAELightning.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    normalized = (
        module.generate(count, temperature=temperature)
        .permute(0, 2, 1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
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
    elapsed_time = frame["elapsed_time"].to_numpy(dtype=np.float32).reshape(
        count, -1
    )
    np.savez_compressed(
        output / "samples.npz",
        values=values,
        elapsed_time=elapsed_time,
    )
    frame.to_parquet(output / "generated.parquet", index=False)
    return output
