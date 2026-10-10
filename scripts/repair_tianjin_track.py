"""Recompute per-flight unwrapped track, preserving every other Parquet value.

Run with uv run --no-sync python scripts/repair_tianjin_track.py --help.
This repairs the variable-length input; fixed-length preparation must also
unwrap AFTER resampling, since subsampling can expose another 360-degree jump.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from traffic.core import Flight


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def repaired_track(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    required = {"flight_id", "timestamp", "track"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("flight_id, timestamp and track must be non-null")
    if not np.isfinite(frame.track.to_numpy(dtype=float)).all():
        raise ValueError("track must be finite")
    timestamps = pd.to_datetime(frame.timestamp, utc=True)
    result = np.empty(len(frame), dtype=np.float64)
    records = []
    for identifier, indices in frame.groupby("flight_id", sort=False).indices.items():
        order = indices[np.argsort(timestamps.iloc[indices], kind="stable")]
        flight = Flight(frame.iloc[order])
        # Recompute from the canonical track, independently for every flight.
        fixed = flight.drop(columns=["track_unwrapped"], errors="ignore").unwrap()
        values = fixed.data.track_unwrapped.to_numpy(dtype=float)
        result[order] = values
        record = {
            "flight_id": identifier,
            "points": len(order),
            "track_max_adjacent_jump_deg": float(
                np.max(np.abs(np.diff(flight.data.track)), initial=0)
            ),
            "repaired_max_adjacent_jump_deg": float(
                np.max(np.abs(np.diff(values)), initial=0)
            ),
        }
        if "track_unwrapped" in frame:
            old = flight.data.track_unwrapped.to_numpy(dtype=float)
            record["old_max_adjacent_jump_deg"] = float(
                np.max(np.abs(np.diff(old)), initial=0)
            )
            record["changed_points"] = int((np.abs(values - old) > 1e-8).sum())
        records.append(record)
    circular_error = (result - frame.track.to_numpy() + 180) % 360 - 180
    if np.max(np.abs(circular_error), initial=0) > 1e-8:
        raise ValueError("Unwrapping changed the physical track direction")
    metrics = pd.DataFrame(records)
    if metrics.repaired_max_adjacent_jump_deg.gt(180 + 1e-8).any():
        raise ValueError("Repaired source still has jumps above 180 degrees")
    return result, metrics


def repair(source: Path, output: Path, report_dir: Path, smoke: bool = False) -> dict:
    source, output, report_dir = (
        path.resolve() for path in (source, output, report_dir)
    )
    if output == source or report_dir == source or output.is_relative_to(report_dir):
        raise ValueError("Use separate source, output and report paths")
    for path in (output, report_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    filters = None
    if smoke:
        ids = pq.read_table(source, columns=["flight_id"])["flight_id"]
        filters = [("flight_id", "in", sorted(set(ids.to_pylist()))[:4])]
    print(f"Reading {source}", flush=True)
    table = pq.read_table(source, filters=filters)
    track, metrics = repaired_track(table.to_pandas())
    column = pa.array(track)
    if "track_unwrapped" in table.column_names:
        index = table.column_names.index("track_unwrapped")
        repaired = table.set_column(index, "track_unwrapped", column)
    else:
        repaired = table.append_column("track_unwrapped", column)
    unchanged = [name for name in table.column_names if name != "track_unwrapped"]
    if not table.select(unchanged).equals(repaired.select(unchanged)):
        raise ValueError("Repair changed columns other than track_unwrapped")
    report_dir.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(repaired, output, compression="zstd")
    # Read the output back before declaring the repair complete.
    saved = pq.read_table(output)
    if not repaired.equals(saved):
        raise ValueError("Written Parquet does not match the verified repair")
    metrics.to_parquet(report_dir / "source_trajectory_metrics.parquet", index=False)
    report = {
        "source": str(source),
        "output": str(output),
        "source_sha256": fingerprint(source),
        "output_sha256": fingerprint(output),
        "smoke": smoke,
        "rows": len(table),
        "trajectories": len(metrics),
        "existing_unwrapped_column": "track_unwrapped" in table.column_names,
        "old_trajectories_with_jumps_above_180": (
            int(metrics.old_max_adjacent_jump_deg.gt(180 + 1e-8).sum())
            if "old_max_adjacent_jump_deg" in metrics
            else None
        ),
        "repaired_trajectories_with_jumps_above_180": 0,
        "other_columns_values_order_and_metadata_preserved": True,
        "branch_convention": (
            "chronological per-flight numpy unwrap; first branch equals raw track; "
            "subsequent angle changes choose the shortest branch"
        ),
        "units": {"track": "degree", "track_unwrapped": "degree"},
        "note": "Unwrap again after fixed-point resampling; GPS is unchanged.",
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("traffic", "numpy", "pandas", "pyarrow")
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "report-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Only four whole flights")
    args = parser.parse_args()
    report = repair(args.source, args.output, args.report_dir, args.smoke)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Output: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
