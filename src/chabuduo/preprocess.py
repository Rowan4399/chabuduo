"""Select terminal trajectories with the public ``traffic`` API.

This stage keeps aviation meaning and variable-length trajectories. Here the
configured radius is a simple approximation of the ASMA (Arrival Sequencing and
Metering Area), the terminal area in which arriving flights are organised and
sequenced toward the runway. For a landing, preprocessing keeps observations
from entry into that radius up to the detected landing endpoint.

中文说明：这里用机场周围的圆形半径近似 ASMA（进场排序与流量管理区域）。
预处理保留飞机进入该区域后、直到检测到着陆终点的轨迹；这个圆形范围不是正式
公布的空域边界。

A later preparation stage handles duplicate timestamps, fixed-point resampling,
features, splits, normalization, and model tensors.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from traffic.core import Flight, Traffic
from traffic.data import airports

from chabuduo.config import ExperimentConfig


def _landing_in_asma(
    flight: Flight,
    *,
    airport_code: str,
    allowed_runways: frozenset[str] | None,
    radius_nm: float,
    max_threshold_distance_nm: float,
    min_observations: int,
    min_duration_sec: float,
) -> Flight | None:
    """Keep an ASMA arrival ending at an observed runway-threshold point."""
    airport = airports[airport_code]

    # Traffic detects runway-aligned landing segments. ILS_max is the runway
    # designator inferred from the alignment, such as "36L" or "01".
    landing = flight.landing(airport).next()
    if landing is None or landing.ILS_max is None:
        return None

    # A physical runway has two named thresholds. Select the threshold whose
    # name matches the detected landing direction.
    runway_name = landing.ILS_max
    if allowed_runways is not None and runway_name not in allowed_runways:
        return None
    runway = airport.runways[runway_name]
    threshold = next(
        point for point in runway.tuple_runway if point.name == runway_name
    )

    # Keep an observed receiver point rather than moving a point onto the
    # runway. closest_point() reports metres, while the YAML tolerance is NM.
    closest = flight.closest_point(threshold)
    if closest.distance > max_threshold_distance_nm * 1852:
        return None

    # Include the closest observation, then retain only the part inside the
    # circular ASMA approximation. Flight.distance(Airport) returns NM.
    selected = flight.before(closest.timestamp, strict=False)
    if selected is None:
        return None
    selected = selected.distance(airport).query(f"distance < {radius_nm}")
    if selected is None:
        return None
    if len(selected.data) < min_observations:
        return None
    if selected.duration.total_seconds() < min_duration_sec:
        return None
    return selected.assign(runway=runway_name)

def _takeoff_in_asma(
    flight: Flight,
    *,
    airport_code: str,
    allowed_runways: frozenset[str] | None,
    radius_nm: float,
    max_threshold_distance_nm: float,
    min_observations: int,
    min_duration_sec: float,
) -> Flight | None:
    airport = airports[airport_code]

    takeoff = flight.takeoff(airport).next()
    if takeoff is None:
        return None

    runway_name = takeoff.runway_max
    if runway_name is None:
        return None
    if allowed_runways is not None and runway_name not in allowed_runways:
        return None
    
    runway = airport.runways[runway_name]
    threshold = next(
        point for point in runway.tuple_runway if point.name == runway_name
    )

    # Keep an observed receiver point rather than moving a point onto the
    # runway. closest_point() reports metres, while the YAML tolerance is NM.
    closest = flight.closest_point(threshold)
    if closest.distance > max_threshold_distance_nm * 1852:
        return None

    # Include the closest observation, then retain only the part inside the
    # circular ASMA approximation. Flight.distance(Airport) returns NM.
    selected = flight.after(closest.timestamp, strict=False)
    if selected is None:
        return None
    
    selected = selected.distance(airport).query(f"distance < {radius_nm}")
    if selected is None:
        return None
    
    if len(selected.data) < min_observations:
        return None
    if selected.duration.total_seconds() < min_duration_sec:
        return None
    return selected.assign(runway=runway_name)

def load_source_traffic(source: Path, config: ExperimentConfig) -> Traffic:
    """Load a source Parquet as Traffic, using configured columns if present."""
    source_columns = getattr(config.source, "columns", None)
    if source_columns:
        available = set(pq.ParquetFile(source).schema.names)
        source_columns = [name for name in source_columns if name in available]
    kwargs = {"columns": source_columns} if source_columns else {}
    t = Traffic.from_file(source, **kwargs)
    if t is None:
        raise ValueError(f"Could not load Traffic from {source}")
    return t


def preprocess(
    source: Path,
    config: ExperimentConfig,
    *,
    max_workers: int = 1,
    cache: Path | None = None,
) -> Traffic:
    """Select variable-length terminal trajectories and optionally cache them."""
    if cache and cache.exists():
        t = Traffic.from_file(cache)
        if t is None:
            raise ValueError(f"Could not load preprocessed Traffic: {cache}")
        return t

    t = load_source_traffic(source, config).assign_id()
    settings = config.preprocess
    operation = settings.operation
    allowed_runways = (
        frozenset(settings.runways) if settings.runways is not None else None
    )
    allowed_flows = (
        frozenset(settings.initial_flows)
        if settings.initial_flows is not None
        else None
    )

    # Flow labels are already flight-level metadata in the Zurich source. Apply
    # this restriction before the more expensive ASMA and runway checks while
    # retaining every observation belonging to each selected flight.
    if allowed_flows is not None:
        if "initial_flow" not in t.data.columns:
            raise ValueError(
                "Source does not contain the initial_flow column required by "
                f"the configured filter {sorted(allowed_flows)}"
            )
        initial_flow = t.data["initial_flow"].astype(str)
        matching_ids = t.data.loc[
            initial_flow.isin(allowed_flows), "flight_id"
        ].unique()
        t = Traffic(t.data.loc[t.data["flight_id"].isin(matching_ids)].copy())
        if len(t.data) == 0:
            raise ValueError(
                "Source contains no trajectories for initial flows "
                f"{sorted(allowed_flows)}"
            )

    # The Zurich source already carries a flight-level runway label. Use it as
    # a cheap prefilter, then verify the detected runway inside the selector.
    # Keeping whole flight IDs avoids truncating trajectories with sparse labels.
    if allowed_runways is not None and "runway" in t.data.columns:
        runway = t.data["runway"].astype(str)
        matching_ids = t.data.loc[
            runway.isin(allowed_runways), "flight_id"
        ].unique()
        t = Traffic(t.data.loc[t.data["flight_id"].isin(matching_ids)].copy())
        if len(t.data) == 0:
            raise ValueError(
                f"Source contains no trajectories for runways {sorted(allowed_runways)}"
            )

    # if operation != "landing":
    #     raise ValueError(
    #         f"Unsupported preprocess operation: {operation!r}; "
    #         "takeoff will be added after its endpoint convention is tested"
    #     )
    if operation == "landing":
        selector = _landing_in_asma
        description = "Selecting terminal landings"
    elif operation == "takeoff":
        selector = _takeoff_in_asma
        description = "Selecting terminal takeoffs"
    else:
        raise ValueError(
            f"Unsupported preprocess operation: {operation!r}"
        )

    res = t.pipe(
        selector,
        airport_code=settings.airport,
        allowed_runways=allowed_runways,
        radius_nm=settings.radius_nm,
        max_threshold_distance_nm=settings.max_threshold_distance_nm,
        min_observations=settings.min_observations,
        min_duration_sec=settings.min_duration_sec,
    ).eval(max_workers=max_workers, desc=description)
    if res is None:
        raise ValueError("Preprocessing selected no trajectories")

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(cache)
    return res

