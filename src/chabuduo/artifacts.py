from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitData:
    values: npt.NDArray[np.float32]
    elapsed_time: npt.NDArray[np.float32]
    trajectory_id: npt.NDArray[np.str_]
    split_group_id: npt.NDArray[np.str_]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            values=self.values,
            elapsed_time=self.elapsed_time,
            trajectory_id=self.trajectory_id,
            split_group_id=self.split_group_id,
        )

    @classmethod
    def load(cls, path: Path) -> SplitData:
        with np.load(path, allow_pickle=False) as data:
            key = "values" if "values" in data else "positions"
            return cls(
                values=data[key].astype(np.float32, copy=False),
                elapsed_time=data["elapsed_time"].astype(np.float32, copy=False),
                trajectory_id=data["trajectory_id"].astype(str),
                split_group_id=data["split_group_id"].astype(str),
            )


@dataclass(frozen=True)
class Normalization:
    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]

    @classmethod
    def fit(cls, values: npt.NDArray[np.float32]) -> Normalization:
        # Match the traffic.Generation convention: each time/feature position
        # has its own statistics across training trajectories.
        mean = values.mean(axis=0, dtype=np.float64)
        std = values.std(axis=0, dtype=np.float64)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(
        self, values: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        return (values * self.std + self.mean).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}, indent=2)
            + "\n",
            encoding="utf8",
        )

    @classmethod
    def load(cls, path: Path) -> Normalization:
        value = json.loads(path.read_text(encoding="utf8"))
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            std=np.asarray(value["std"], dtype=np.float64),
        )


def write_prepared_artifact(
    output: Path,
    splits: dict[str, SplitData],
    normalization: Normalization,
    metadata: pd.DataFrame,
) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for name in SPLITS:
            splits[name].save(temporary / f"{name}.npz")
        normalization.save(temporary / "normalization.json")
        metadata.to_parquet(temporary / "metadata.parquet", index=False)
        temporary.rename(output)
    except Exception:
        for item in temporary.iterdir():
            item.unlink()
        temporary.rmdir()
        raise
