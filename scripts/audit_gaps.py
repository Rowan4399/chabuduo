"""Inspect time gaps and interpolation signatures without changing trajectories."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyproj import Geod

GEOD = Geod(ellps="WGS84")
COORD_TOL = 1e-10


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)


def linear_spans(values, elapsed, tolerance):
    values = np.asarray(values, dtype=float).reshape(len(elapsed), -1)
    dt = np.diff(elapsed)
    slope = np.diff(values, axis=0) / dt[:, None]
    same = (np.abs(np.diff(slope, axis=0)) <= tolerance).all(axis=1)
    same &= (dt[:-1] == 1) & (dt[1:] == 1)
    # A run of equal adjacent slopes includes both bounding point intervals.
    return [(int(a), int(b + 1)) for a, b in spans(same)]


def longest_point_mask(mask, elapsed):
    return max((elapsed[b - 1] - elapsed[a] for a, b in spans(mask)), default=0)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def contact_rows(raw, target):
    keys = ["icao24", "callsign", "timestamp"]
    if target.duplicated(keys).any():
        raise ValueError("Target observation keys are not unique")
    collected = []
    parquet = pq.ParquetFile(raw)
    if "lastcontact" not in parquet.schema_arrow.names:
        return target.assign(lastcontact=np.nan)
    callsigns = target.callsign.unique().tolist()
    for batch in parquet.iter_batches(
        batch_size=250_000, columns=keys + ["lastcontact"]
    ):
        mask = pc.is_in(batch["callsign"], value_set=pa.array(callsigns))
        frame = batch.filter(mask).to_pandas()
        matched = target.merge(frame, on=keys, how="inner", validate="one_to_one")
        if len(matched):
            collected.append(matched)
    result = pd.concat(collected, ignore_index=True)
    if len(result) != len(target) or result.duplicated(keys).any():
        raise ValueError(f"Raw contact matching: {len(result)} != {len(target)}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    columns = [
        "flight_id",
        "timestamp",
        "icao24",
        "callsign",
        "latitude",
        "longitude",
        "track",
        "groundspeed",
    ]
    data = pq.read_table(args.selected, columns=columns).to_pandas()
    metrics = pq.read_table(args.metrics).to_pandas().set_index("trajectory_id")
    if set(data.flight_id.unique()) != set(metrics.index):
        raise ValueError("Selected file and existing audit contain different flights")
    bad_ids = metrics.index[metrics.after_gps_rmse_m.gt(5000)]
    target = data.loc[
        data.flight_id.isin(bad_ids), ["flight_id", "timestamp", "icao24", "callsign"]
    ]
    linked = contact_rows(args.raw, target)
    contacts = linked.groupby("flight_id", sort=False)
    records, segments = [], []
    for index, (identifier, frame) in enumerate(data.groupby("flight_id", sort=False)):
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        elapsed = (
            (frame.timestamp - frame.timestamp.iloc[0]).dt.total_seconds().to_numpy()
        )
        dt = np.diff(elapsed)
        if (dt <= 0).any():
            raise ValueError(f"Non-increasing timestamps: {identifier}")
        lon, lat = frame.longitude.to_numpy(), frame.latitude.to_numpy()
        track = frame.track.to_numpy()
        speed = frame.groundspeed.to_numpy()
        left = np.searchsorted(elapsed, elapsed - 5)
        right = np.minimum(np.searchsorted(elapsed, elapsed + 5), len(frame) - 1)
        window = elapsed[right] - elapsed[left]
        bearing, _, displacement = GEOD.inv(
            lon[left], lat[left], lon[right], lat[right]
        )
        valid = (window >= 5) & (displacement >= 10)
        gps_speed = (
            np.divide(
                displacement, window, out=np.full(len(frame), np.nan), where=window > 0
            )
            * 3600
            / 1852
        )
        angle_error = np.abs((track - bearing + 180) % 360 - 180)
        speed_error = np.abs(speed - gps_speed)
        mismatch = valid & ((angle_error > 30) | (speed_error > 50))
        coordinate_runs = linear_spans(np.c_[lon, lat], elapsed, COORD_TOL)
        moving_runs = []
        for a, b in coordinate_runs:
            distance = GEOD.inv(lon[a], lat[a], lon[b], lat[b])[2]
            duration = elapsed[b] - elapsed[a]
            if duration > 0 and distance / duration > 0.1:
                moving_runs.append((a, b))
                if duration >= 60 and identifier in bad_ids:
                    section = slice(a, b + 1)
                    usable = valid[section]
                    segments.append(
                        {
                            "flight_id": identifier,
                            "start_utc": str(frame.timestamp.iloc[a]),
                            "end_utc": str(frame.timestamp.iloc[b]),
                            "duration_sec": float(duration),
                            "distance_m": float(distance),
                            "angle_error_p95_deg": float(
                                np.percentile(angle_error[section][usable], 95)
                            )
                            if usable.any()
                            else None,
                            "speed_error_median_kt": float(
                                np.median(speed_error[section][usable])
                            )
                            if usable.any()
                            else None,
                            "motion_mismatch_fraction": float(mismatch[section].mean()),
                        }
                    )
        max_linear = max((elapsed[b] - elapsed[a] for a, b in moving_runs), default=0)
        frozen = np.maximum(np.abs(np.diff(lon)), np.abs(np.diff(lat))) <= COORD_TOL
        frozen_sec = max((elapsed[b] - elapsed[a] for a, b in spans(frozen)), default=0)
        record = {
            "flight_id": identifier,
            "split": metrics.loc[identifier, "split"],
            "rmse_m": float(metrics.loc[identifier, "after_gps_rmse_m"]),
            "is_bad_5km": identifier in bad_ids,
            "points": len(frame),
            "duration_sec": float(elapsed[-1]),
            "max_timestamp_gap_sec": float(dt.max()),
            "timestamp_gaps_above_1sec": int((dt > 1).sum()),
            "missing_core_rows": int(
                frame[["timestamp", "latitude", "longitude", "track", "groundspeed"]]
                .isna()
                .any(axis=1)
                .sum()
            ),
            "max_moving_linear_position_sec": float(max_linear),
            "moving_linear_position_total_sec": float(
                sum(
                    elapsed[b] - elapsed[a]
                    for a, b in moving_runs
                    if elapsed[b] - elapsed[a] >= 60
                )
            ),
            "max_position_frozen_sec": float(frozen_sec),
            "max_motion_mismatch_sec": float(longest_point_mask(mismatch, elapsed)),
            "motion_mismatch_fraction": float(mismatch.mean()),
        }
        if identifier in bad_ids:
            contact = contacts.get_group(identifier).sort_values("timestamp")
            if not np.array_equal(
                contact.timestamp.to_numpy(), frame.timestamp.to_numpy()
            ):
                raise ValueError(f"Contact times differ: {identifier}")
            contact_value = contact.lastcontact.to_numpy(dtype=float)
            timestamp_sec = elapsed + frame.timestamp.iloc[0].timestamp()
            age = timestamp_sec - contact_value
            record["lastcontact_age_max_sec"] = float(np.nanmax(age))
            record["lastcontact_age_above_60sec_points"] = int((age > 60).sum())
            contact_runs = linear_spans(contact_value, elapsed, 1e-5)
            record["max_linear_lastcontact_sec"] = float(
                max((elapsed[b] - elapsed[a] for a, b in contact_runs), default=0)
            )
            record["contact_rows_matched"] = len(contact)
        records.append(record)
        if index % 500 == 0:
            print(f"Inspected {index}/{len(metrics)}", flush=True)
    results = pd.DataFrame(records)
    summary = {
        "selected": str(args.selected.resolve()),
        "selected_sha256": digest(args.selected),
        "raw_for_lastcontact": str(args.raw.resolve()),
        "criteria": {
            "bad": "existing 200-point integrated position RMSE > 5000 m",
            "moving_linear_position": (
                "lat/lon per-second increments constant within 1e-10 degree/s; "
                "mean motion > 0.1 m/s"
            ),
            "motion_mismatch": (
                "centered approximately 10-second GPS direction error > 30 degrees "
                "OR speed error > 50 knots; window >=5s and displacement >=10m"
            ),
            "lastcontact_linearity": (
                "per-second lastcontact increments constant within 1e-5 seconds"
            ),
            "limitation": (
                "All inputs have already been filtered/resampled. Long affine runs "
                "are interpolation signatures, not proof of missing raw messages; "
                "lastcontact may itself have been interpolated."
            ),
        },
        "counts": {},
        "runtime_sec": time.perf_counter() - started,
    }
    for label, mask in (
        ("bad_128", results.is_bad_5km),
        ("remaining_3103", ~results.is_bad_5km),
    ):
        group = results[mask]
        summary["counts"][label] = {
            "trajectories": len(group),
            "points": int(group.points.sum()),
            "with_timestamp_gap_above_1sec": int(
                group.timestamp_gaps_above_1sec.gt(0).sum()
            ),
            "with_missing_core_rows": int(group.missing_core_rows.gt(0).sum()),
            **{
                f"moving_linear_position_at_least_{threshold}sec": int(
                    group.max_moving_linear_position_sec.ge(threshold).sum()
                )
                for threshold in (30, 60, 120, 300, 600, 1200)
            },
            **{
                f"motion_mismatch_at_least_{threshold}sec": int(
                    group.max_motion_mismatch_sec.ge(threshold).sum()
                )
                for threshold in (30, 60, 120, 300)
            },
            "position_frozen_at_least_60sec": int(
                group.max_position_frozen_sec.ge(60).sum()
            ),
            "max_linear_duration_sec": float(
                group.max_moving_linear_position_sec.max()
            ),
        }
    summary["counts"]["bad_128"]["lastcontact_age_above_60sec"] = int(
        results.loc[results.is_bad_5km, "lastcontact_age_max_sec"].gt(60).sum()
    )
    summary["counts"]["bad_128"]["lastcontact_linear_at_least_60sec"] = int(
        results.loc[results.is_bad_5km, "max_linear_lastcontact_sec"].ge(60).sum()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "all_trajectory_gap_metrics.csv", index=False)
    results[results.is_bad_5km].to_csv(
        args.output_dir / "bad128_gap_metrics.csv", index=False
    )
    pd.DataFrame(segments).to_csv(
        args.output_dir / "bad128_linear_segments.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Output: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
