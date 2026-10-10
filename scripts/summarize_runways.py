"""Count one runway per accepted flight and audit source/selected ground flags."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source = pq.ParquetFile(args.source)
    raw_ids, raw_ground_ids = set(), set()
    raw_ground_points = raw_unknown_ground_points = 0
    for batch in source.iter_batches(columns=["flight_id", "onground"]):
        frame = batch.to_pandas()
        raw_ids.update(frame.flight_id.dropna().unique())
        ground = frame.onground.eq(True).fillna(False)
        raw_ground_points += int(ground.sum())
        raw_unknown_ground_points += int(frame.onground.isna().sum())
        raw_ground_ids.update(frame.loc[ground, "flight_id"].dropna().unique())

    frame = pd.read_parquet(
        args.selected, columns=["flight_id", "timestamp", "runway", "onground"]
    )
    if frame.flight_id.isna().any() or frame.runway.isna().any():
        raise ValueError("Selected trajectories have missing flight/runway labels")
    frame["is_ground"] = frame.onground.eq(True).fillna(False)
    frame["unknown_ground"] = frame.onground.isna()
    groups = frame.groupby("flight_id", sort=False)
    if not groups.runway.nunique().eq(1).all():
        raise ValueError("Runway labels are not constant within every flight")
    per_flight = groups.agg(
        runway=("runway", "first"),
        points=("timestamp", "size"),
        start=("timestamp", "min"),
        stop=("timestamp", "max"),
        ground_points=("is_ground", "sum"),
        unknown_ground_points=("unknown_ground", "sum"),
    )
    per_flight["duration_sec"] = (per_flight.stop - per_flight.start).dt.total_seconds()
    per_flight["has_ground_points"] = per_flight.ground_points.gt(0)
    endpoints = frame.loc[groups.timestamp.idxmax()].set_index("flight_id")
    per_flight["endpoint_onground"] = endpoints.onground
    per_flight.to_csv(output / "trajectory_runway_statistics.csv")
    runway_counts = per_flight.groupby("runway").agg(
        trajectories=("points", "size"),
        points=("points", "sum"),
        mean_points=("points", "mean"),
        mean_duration_sec=("duration_sec", "mean"),
        ground_points=("ground_points", "sum"),
        trajectories_with_ground_points=("has_ground_points", "sum"),
    )
    runway_counts["percent_of_accepted_trajectories"] = (
        runway_counts.trajectories / len(per_flight) * 100
    )
    runway_counts["percent_of_raw_trajectories"] = (
        runway_counts.trajectories / len(raw_ids) * 100
    )
    runway_counts = runway_counts.sort_values("trajectories", ascending=False)
    assert int(runway_counts.trajectories.sum()) == len(per_flight)
    assert int(runway_counts.points.sum()) == len(frame)
    runway_counts.to_csv(output / "runway_counts.csv")
    summary = {
        "source": str(args.source.resolve()),
        "selected": str(args.selected.resolve()),
        "counting_unit": "one trajectory per flight_id; runway constant per flight",
        "raw_trajectories": len(raw_ids),
        "raw_points": source.metadata.num_rows,
        "raw_ground_points": raw_ground_points,
        "raw_trajectories_with_ground_points": len(raw_ground_ids),
        "raw_unknown_ground_points": raw_unknown_ground_points,
        "accepted_trajectories": len(per_flight),
        "accepted_points": len(frame),
        "not_retained_trajectories": len(raw_ids) - len(per_flight),
        "retained_percent": len(per_flight) / len(raw_ids) * 100,
        "selected_ground_points": int(frame.is_ground.sum()),
        "selected_trajectories_with_ground_points": int(
            per_flight.has_ground_points.sum()
        ),
        "selected_unknown_ground_points": int(frame.unknown_ground.sum()),
        "selected_endpoints_onground": int(per_flight.endpoint_onground.eq(True).sum()),
        "runway_counts": runway_counts.reset_index().to_dict(orient="records"),
        "flow_labels_computed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    main()
