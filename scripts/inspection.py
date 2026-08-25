# %%
from pathlib import Path

import matplotlib.pyplot as plt
from cartes.crs import Mercator, PlateCarree
from httpx import HTTPError

import numpy as np
from pyproj import Geod
from traffic.core import Flight, Traffic
from traffic.data import airports
PROJECT_ROOT = Path.cwd().parent
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
# %%
# 绘制所有检测到的降落轨迹，以及上面使用的 40 海里范围。
# Geod 使用米作为单位，1 海里等于 1852 米。

beijing = airports["ZBAA"]
RADIUS_NM = 40
N_SAMPLES = 200

angles = np.linspace(0, 360, 361)
longitude, latitude, _ = Geod(ellps="WGS84").fwd(
    np.full_like(angles, beijing.longitude),
    np.full_like(angles, beijing.latitude),
    angles,
    np.full_like(angles, RADIUS_NM * 1852),
)

with plt.style.context("traffic"):
    figure, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"projection": Mercator()}
    )
    preprocessed.plot(ax, color="#4c78a8", alpha=0.1, linewidth=0.5)
    beijing.plot(ax, footprint=False)
    ax.plot(
        longitude,
        latitude,
        color="#e45756",
        label=f"{RADIUS_NM} nm",
        transform=PlateCarree(),
    )
    ax.set_extent(
        (
            preprocessed.data.longitude.min(),
            preprocessed.data.longitude.max(),
            preprocessed.data.latitude.min(),
            preprocessed.data.latitude.max(),
        ),
        crs=PlateCarree(),
    )
    ax.set_title("ZBAA landing trajectories and terminal area")
    ax.legend()

FIGURE_FILE = Path.cwd() / "zbaa_radius_diagnostics.png"
figure.savefig(FIGURE_FILE, dpi=170, bbox_inches="tight")
print(f"图片文件：{FIGURE_FILE}")
figure
# %%
