"""Prepare selected ``Traffic`` for a numerical trajectory model.

``preprocess`` ends with a selected variable-length ``Traffic`` collection.
This module owns fixed-length resampling and the next boundary described by
``traffic.Generation``:
features are chosen, trajectories are split, a scaler is fitted on training
trajectories only, and fixed-shape matrices are written as a serializable
artifact. Models consume those arrays and never import ``Traffic``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from traffic.algorithms.generation import Generation as TrafficGeneration
from traffic.core import Flight, Traffic
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData, write_prepared_artifact
from chabuduo.config import ExperimentConfig, SplitSettings


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


def _elapsed_feature(flight: Flight) -> Flight:
    """Add elapsed time relative to this flight's first observation."""
    return flight.assign(elapsed_time_sec=_elapsed)

# Even if you don't use all of them at once, just define them here. It doesn't
# hurt to have them all available for any representation. We select what we
# really need later on.


_FEATURE_COLUMNS = {
    "east_m": lambda frame: frame.x,
    "north_m": lambda frame: frame.y,
    "altitude_m": lambda frame: frame.altitude * 0.3048,
    "track_sin": lambda frame: np.sin(np.radians(frame.track)),
    "track_cos": lambda frame: np.cos(np.radians(frame.track)),
    "groundspeed_kt": lambda frame: frame.groundspeed,
    "altitude_ft": lambda frame: frame.altitude,
}


def _split_traffic(t: Traffic, split_config: SplitSettings) -> dict[str, Traffic]:
    flights = list(t)
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


def _ids(t: Traffic) -> np.ndarray:
    return np.asarray(
        [
            str(getattr(flight, "flight_id", index))
            for index, flight in enumerate(t)
        ],
        dtype=str,
    )


def prepare(
    selected: Path,
    output: Path,
    config: ExperimentConfig,
) -> Path:
    """Create an immutable model-ready artifact from selected Traffic."""
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    t = Traffic.from_file(selected)
    if t is None:
        raise ValueError(f"Could not load selected Traffic: {selected}")
    settings = config.prepare
    if settings is None:
        raise ValueError("Dataset configuration has no prepare section")
    points = settings.points
    representation = settings.representation
    airport_code = config.preprocess.airport
    projection = airport_projection(airport_code)
    t = (
        t.compute_xy(projection=projection)
        .assign(**_FEATURE_COLUMNS)
        .unwrap()
        .resample(points)
        .pipe(_elapsed_feature)
        .eval()
    )
    if t is None:
        raise ValueError("Preparation produced no fixed-length trajectories")
    features = representation.features
    splits = _split_traffic(t, settings.split)
    scaler = _PerTimestepScaler(points, len(features))
    generation = TrafficGeneration(None, features, scaler=scaler)
    generation.fit_preprocessing(splits["train"])

    prepared: dict[str, SplitData] = {}
    records: list[dict[str, str]] = []
    for name, t in splits.items():
        matrix = generation.transform_features(t)
        feature_count = len(features)
        values = matrix.reshape(len(t), points, feature_count).astype(np.float32)
        identifiers = _ids(t)
        elapsed = np.asarray(
            [
                flight.data.elapsed_time_sec.to_numpy(dtype=np.float32)
                for flight in t
            ],
            dtype=np.float32,
        )
        prepared[name] = SplitData(
            values=values,
            elapsed_time=elapsed,
            trajectory_id=identifiers,
            split_group_id=identifiers.copy(),
        )
        records.extend(
            {"trajectory_id": identifier, "split": name}
            for identifier in identifiers
        )

    if scaler.normalization is None:
        raise RuntimeError("Generation scaler was not fitted")
    write_prepared_artifact(
        output,
        prepared,
        scaler.normalization,
        pd.DataFrame.from_records(records),
    )
    return output
