"""Prepare selected ``Traffic`` for a numerical trajectory model.

``preprocess`` ends with a selected and resampled ``Traffic`` collection.
This module owns the next boundary described by ``traffic.Generation``:
features are chosen, trajectories are split, a scaler is fitted on training
trajectories only, and fixed-shape matrices are written as a serializable
artifact. Models consume those arrays and never import ``Traffic``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyproj
from traffic.algorithms.generation import Generation as TrafficGeneration
from traffic.core import Traffic
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData, write_prepared_artifact
from chabuduo.config import (
    DatasetConfig,
    RepresentationSettings,
    SplitSettings,
    write_config,
)
from chabuduo.get import sha256


class _NoModel:
    def fit(self, X: np.ndarray, **kwargs: Any) -> _NoModel:
        return self

    def sample(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class _PerTimestepScaler:
    """Adapt project normalization to Traffic Generation's 2-D matrix API."""

    def __init__(self, points: int, features: int) -> None:
        self.normalization: Normalization | None = None
        self.points = points
        self.features = features

    def fit(self, values: np.ndarray) -> _PerTimestepScaler:
        reshaped = values.reshape(-1, self.points, self.features)
        self.normalization = Normalization.fit(reshaped)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            raise RuntimeError("Scaler has not been fitted")
        reshaped = values.reshape(-1, self.points, self.features)
        return self.normalization.transform(reshaped).reshape(values.shape[0], -1)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.normalization is None:
            raise RuntimeError("Scaler has not been fitted")
        reshaped = values.reshape(-1, self.points, self.features)
        return self.normalization.inverse_transform(reshaped).reshape(values.shape)


def airport_projection(airport_code: str) -> pyproj.Proj:
    airport = airports[airport_code]
    if airport is None:
        raise ValueError(f"Unknown airport: {airport_code}")
    return pyproj.Proj(
        proj="aeqd",
        lat_0=airport.latitude,
        lon_0=airport.longitude,
        datum="WGS84",
        units="m",
    )


def _elapsed(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame.timestamp, utc=True)
    return (timestamp - timestamp.iloc[0]).dt.total_seconds()


def _features(
    traffic: Traffic,
    representation: RepresentationSettings,
    airport_code: str,
) -> tuple[Traffic, list[str]]:
    """Add Traffic-compatible feature columns for one representation."""
    name = representation.name
    if name == "projected_position_200":
        traffic = traffic.compute_xy(
            projection=airport_projection(airport_code)
        )
        traffic = traffic.iterate_lazy().assign(timedelta=_elapsed).eval()
        return traffic, ["x", "y", "altitude"]

    traffic = traffic.unwrap()
    traffic = traffic.iterate_lazy().assign(timedelta=_elapsed)
    if name == "kinematic_circular_200":
        traffic = traffic.assign(
            track_sin=lambda frame: np.sin(
                np.radians(frame.track_unwrapped)
            ),
            track_cos=lambda frame: np.cos(
                np.radians(frame.track_unwrapped)
            ),
        ).eval()
        return traffic, [
            "track_sin",
            "track_cos",
            "groundspeed",
            "altitude",
            "timedelta",
        ]
    traffic = traffic.eval()
    return traffic, ["track_unwrapped", "groundspeed", "altitude", "timedelta"]


def _split_traffic(
    traffic: Traffic, split_config: SplitSettings
) -> dict[str, Traffic]:
    flights = list(traffic)
    if len(flights) < 3:
        raise ValueError("At least three trajectories are required for three splits")
    rng = np.random.default_rng(split_config.seed)
    order = rng.permutation(len(flights))
    n_train = max(1, int(len(flights) * split_config.train))
    n_validation = max(1, int(len(flights) * split_config.validation))
    if n_train + n_validation >= len(flights):
        n_train, n_validation = len(flights) - 2, 1
    indices = {
        "train": order[:n_train],
        "validation": order[n_train : n_train + n_validation],
        "test": order[n_train + n_validation :],
    }
    return {
        name: Traffic(
            pd.concat(
                [flights[index].data for index in selected], ignore_index=True
            )
        )
        for name, selected in indices.items()
    }


def _ids(traffic: Traffic) -> np.ndarray:
    return np.asarray(
        [
            str(getattr(flight, "flight_id", index))
            for index, flight in enumerate(traffic)
        ],
        dtype=str,
    )


def prepare(
    selected: Path,
    output: Path,
    config: DatasetConfig,
) -> Path:
    """Create an immutable model-ready artifact from selected Traffic."""
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    traffic = Traffic.from_file(selected)
    if traffic is None:
        raise ValueError(f"Could not load selected Traffic: {selected}")
    settings = config.prepare
    if settings is None:
        raise ValueError("Dataset configuration has no prepare section")
    points = settings.points
    traffic = traffic.resample(points).eval()
    if traffic is None:
        raise ValueError("Preparation produced no fixed-length trajectories")
    representation = settings.representation
    airport_code = config.preprocess.airport
    traffic, internal_features = _features(
        traffic, representation, airport_code
    )
    splits = _split_traffic(traffic, settings.split)
    scaler = _PerTimestepScaler(points, len(internal_features))
    generation = TrafficGeneration(_NoModel(), internal_features, scaler=scaler)
    generation.fit_preprocessing(splits["train"])

    prepared: dict[str, SplitData] = {}
    records: list[dict[str, str]] = []
    for name, split in splits.items():
        matrix = generation.transform_features(split)
        features = len(internal_features)
        values = matrix.reshape(len(split), points, features).astype(np.float32)
        identifiers = _ids(split)
        elapsed = np.asarray(
            [_elapsed(flight.data).to_numpy(dtype=np.float32) for flight in split],
            dtype=np.float32,
        )
        prepared[name] = SplitData(
            positions=values,
            elapsed_time=elapsed,
            trajectory_id=identifiers,
            split_group_id=identifiers.copy(),
        )
        records.extend(
            {"trajectory_id": identifier, "split": name}
            for identifier in identifiers
        )

    projection = pyproj.CRS(airport_projection(airport_code).srs)
    manifest = {
        "artifact_type": "prepared_dataset",
        "dataset": config.name,
        "dataset_version": config.version,
        "source": str(selected.resolve()),
        "source_sha256": sha256(selected),
        "trajectory_count": len(traffic),
        "preprocess": config.preprocess.model_dump(mode="json"),
        "representation": {
            **representation.model_dump(mode="json"),
            "points": points,
            "shape": [points, len(internal_features)],
            "internal_features": internal_features,
            "projection_wkt": projection.to_wkt(),
            "airport": airport_code,
        },
        "split": settings.split.model_dump(mode="json"),
        "normalization": {
            **settings.normalization.model_dump(mode="json"),
            "fit_on": "train",
            "method": "standard_per_timestep",
        },
    }
    if scaler.normalization is None:
        raise RuntimeError("Generation scaler was not fitted")
    write_prepared_artifact(
        output,
        prepared,
        scaler.normalization,
        pd.DataFrame.from_records(records),
        manifest,
    )
    write_config(
        {"features": internal_features, "source": str(selected.resolve())},
        output / "generation.yaml",
    )
    return output


def build_generated_traffic(
    values: np.ndarray,
    elapsed_time: np.ndarray,
    manifest: dict[str, Any],
) -> Traffic:
    """Reconstruct model samples with Traffic Generation's canonical adapter."""
    representation = manifest["representation"]
    name = representation["name"]
    projection = pyproj.Proj(pyproj.CRS.from_wkt(representation["projection_wkt"]))
    projected = name in {"projected_position_200", "enu_position_20"}
    if projected:
        altitude_ft = values[:, :, 2] / 0.3048
        model_values = np.stack(
            (values[:, :, 0], values[:, :, 1], altitude_ft, elapsed_time),
            axis=2,
        )
        features = ["x", "y", "altitude", "timedelta"]
    else:
        airport = airports[representation["airport"]]
        if name == "kinematic_circular_200":
            track = np.degrees(
                np.arctan2(values[:, :, 0], values[:, :, 1])
            ) % 360
            speed = values[:, :, 2]
            altitude = values[:, :, 3]
        else:
            track = values[:, :, 0]
            speed = values[:, :, 1]
            altitude = values[:, :, 2]
        model_values = np.stack(
            (track, speed, altitude, elapsed_time), axis=2
        )
        features = ["track", "groundspeed", "altitude", "timedelta"]
    adapter = TrafficGeneration(_NoModel(), features)
    kwargs: dict[str, Any] = {"projection": projection}
    if not projected:
        kwargs.update(
            coordinates={
                "latitude": airport.latitude,
                "longitude": airport.longitude,
            },
            forward=False,
        )
    return adapter.build_traffic(model_values.reshape(len(values), -1), **kwargs)
