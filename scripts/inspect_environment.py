"""Record the remote input schema and reproduce the reported jump counts."""

import importlib.metadata
import inspect
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from traffic.algorithms.generation import Generation
from traffic.core import Flight, Traffic
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData

for path in Path("data/raw/tianjin").glob("*.parquet"):
    parquet = pq.ParquetFile(path)
    print("RAW", path, parquet.metadata.num_rows, parquet.schema_arrow.names)
for path in Path("data/interim").glob("*zbtj*parquet"):
    parquet = pq.ParquetFile(path)
    print("SELECTED", path, parquet.metadata.num_rows, parquet.schema_arrow.names)
for path in Path("data/processed").glob("*tianjin*"):
    normalization = Normalization.load(path / "normalization.json")
    for split in ("train", "validation", "test"):
        data = SplitData.load(path / f"{split}.npz")
        track = normalization.inverse_transform(data.values)[:, :, 0]
        jumps = np.abs(np.diff(track, axis=1))
        print(
            "SPLIT", path.name, split, len(track),
            int((jumps > 180).any(axis=1).sum()),
            int((jumps > 180).sum()), float(jumps.max()),
        )
airport = airports["ZBTJ"]
print("AIRPORT", airport.latitude, airport.longitude)
print("RUNWAYS", airport.runways["16L"].tuple_runway)
print("GENERATION", inspect.getsource(Generation.transform_features))
print("FEATURE_MATRIX", inspect.getsource(Generation._feature_matrix))
artifact = Path("data/processed/tianjin_noaib_rwy16l_tcvae_vamp_k1000-v1")
data = SplitData.load(artifact / "train.npz")
physical = Normalization.load(artifact / "normalization.json").inverse_transform(
    data.values
)
index = int(np.flatnonzero((np.abs(np.diff(physical[:, :, 0])) > 180).any(axis=1))[0])
identifier = data.trajectory_id[index]
source = Path(
    "data/interim/zbtj-rwy16l-landing-60nm-"
    "landing_tianjin_2026_08_09_60nm_filtered_1s_noaib-v1.parquet"
)
frame = pq.read_table(source, filters=[("flight_id", "=", identifier)]).to_pandas()
flight = Flight(frame.sort_values("timestamp"))
for name, f in (
    ("raw", flight),
    ("unwrap", flight.unwrap()),
    ("unwrap_resample", flight.unwrap().resample(200)),
    (
        "drop_unwrap_resample",
        flight.drop(
            columns=["track_unwrapped"], errors="ignore"
        ).unwrap().resample(200),
    ),
):
    print("STAGE", identifier, name, len(f.data))
    for column in ("track", "track_unwrapped"):
        if column in f.data:
            values = f.data[column].to_numpy()
            print(column, float(np.abs(np.diff(values)).max()), values[:30])
    if len(f.data) == 200:
        print(
            "SAVED_DIFF", np.max(np.abs(physical[index, :, 0] - f.data.track_unwrapped))
        )
        gen = Generation(None, ["track_unwrapped"])
        print("MATRIX", gen._feature_matrix(Traffic(f.data))[0, :30])
print("ENV", {
    name: importlib.metadata.version(name)
    for name in ("traffic", "numpy", "pandas", "pyarrow", "pyproj")
})
