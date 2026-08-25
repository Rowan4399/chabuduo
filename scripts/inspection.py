# %%
from pathlib import Path

import matplotlib.pyplot as plt
from cartes.crs import Mercator, PlateCarree
from httpx import HTTPError

import numpy as np
from pyproj import Geod
from traffic.core import Flight, Traffic
from traffic.data import airports
PROJECT_ROOT = Path.cwd().parent.parent
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "zbaa-landing-40nm-2025-2weeks-v1.parquet"
)

# %%
preprocessed = Traffic.from_file(DATA_FILE)

if preprocessed is None:
    raise ValueError("无法从 Parquet 读取 Traffic")

df = preprocessed.data

print(f"DataFrame shape: {df.shape}")
print(f"columns: {df.columns.tolist()}")

preprocessed
# %%
trajectory_count = df["flight_id"].nunique()
observation_count = len(df)

print(f"There are {trajectory_count:,} trajectories")
print(f"There are {observation_count:,} observation points")
print(
    "average observations per trajectory: "
    f"{observation_count / trajectory_count:.2f}"
)