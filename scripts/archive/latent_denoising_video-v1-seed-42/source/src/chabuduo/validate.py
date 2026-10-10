"""Read-only checks for a prepared training artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chabuduo.artifacts import SPLITS, Normalization, SplitData
from chabuduo.config import ExperimentConfig


class ValidationError(Exception):
    """Raised when a prepared artifact is unsafe to train."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_artifact(artifact: Path, config: ExperimentConfig) -> list[str]:
    """Validate shapes, identifiers, metadata, and numerical invariants."""
    _require(artifact.is_dir(), f"Prepared artifact does not exist: {artifact}")
    required = [*(artifact / f"{name}.npz" for name in SPLITS)]
    required.extend(
        [artifact / "metadata.parquet", artifact / "normalization.json"]
    )
    missing = [path.name for path in required if not path.is_file()]
    _require(not missing, f"Prepared artifact is missing: {', '.join(missing)}")

    points = config.prepare.points
    feature_count = len(config.prepare.representation.features)
    expected_ids: dict[str, set[str]] = {}
    split_sizes: dict[str, int] = {}

    for name in SPLITS:
        data = SplitData.load(artifact / f"{name}.npz")
        _require(len(data.values) > 0, f"{name} split is empty")
        _require(
            data.values.shape[1:] == (points, feature_count),
            f"{name} values have shape {data.values.shape}; expected "
            f"[N, {points}, {feature_count}]",
        )
        _require(
            data.elapsed_time.shape == (len(data.values), points),
            f"{name} elapsed_time shape does not match values",
        )
        _require(
            len(data.trajectory_id) == len(data.values)
            and len(data.split_group_id) == len(data.values),
            f"{name} identifiers do not match values",
        )
        _require(np.isfinite(data.values).all(), f"{name} values contain NaN/Inf")
        _require(
            np.isfinite(data.elapsed_time).all(),
            f"{name} elapsed_time contains NaN/Inf",
        )
        _require(
            (np.diff(data.elapsed_time, axis=1) >= 0).all(),
            f"{name} elapsed_time is not chronological",
        )
        ids = data.trajectory_id.astype(str)
        groups = data.split_group_id.astype(str)
        _require(len(set(ids)) == len(ids), f"{name} trajectory IDs are duplicated")
        _require(
            len(set(groups)) == len(groups),
            f"{name} split group IDs are duplicated",
        )
        expected_ids[name] = set(ids)
        split_sizes[name] = len(ids)

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = expected_ids[left] & expected_ids[right]
            _require(
                not overlap,
                f"Split leakage between {left} and {right}: {len(overlap)} IDs",
            )

    normalization = Normalization.load(artifact / "normalization.json")
    expected_shape = (points, feature_count)
    _require(
        normalization.mean.shape == expected_shape,
        f"Normalization mean has shape {normalization.mean.shape}; "
        f"expected {expected_shape}",
    )
    _require(
        normalization.std.shape == expected_shape,
        f"Normalization std has shape {normalization.std.shape}; "
        f"expected {expected_shape}",
    )
    _require(
        np.isfinite(normalization.mean).all()
        and np.isfinite(normalization.std).all()
        and (normalization.std > 0).all(),
        "Normalization statistics are not finite and positive",
    )

    metadata = pd.read_parquet(artifact / "metadata.parquet")
    metadata_required = {
        "trajectory_id",
        "split",
        *config.prepare.metadata_columns,
    }
    missing_columns = metadata_required - set(metadata.columns)
    _require(
        not missing_columns,
        f"Metadata is missing columns: {sorted(missing_columns)}",
    )
    _require(
        not metadata.trajectory_id.astype(str).duplicated().any(),
        "Metadata trajectory IDs are duplicated",
    )
    for name in SPLITS:
        actual = set(
            metadata.loc[metadata.split.eq(name), "trajectory_id"].astype(str)
        )
        _require(
            actual == expected_ids[name],
            f"Metadata IDs do not match the {name} NPZ",
        )

    if config.preprocess.runways is not None:
        _require("runway" in metadata.columns, "Runway-filtered data lacks metadata")
        actual_runways = set(metadata.runway.dropna().astype(str))
        expected_runways = set(config.preprocess.runways)
        _require(
            actual_runways == expected_runways,
            f"Prepared runways are {sorted(actual_runways)}; "
            f"expected {sorted(expected_runways)}",
        )

    sizes = ", ".join(f"{name}={split_sizes[name]}" for name in SPLITS)
    return [
        f"split shapes and finite values ({sizes})",
        "trajectory IDs are disjoint across splits",
        "normalization statistics match the configured representation",
        "metadata matches every prepared trajectory",
    ]
