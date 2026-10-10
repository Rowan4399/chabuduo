"""Calibrate, score, and display fixed samples without changing their coordinates.

Run each stage separately with uv run --no-sync scripts/trajectory_rules.py.
Physical thresholds come exclusively from real training trajectories. Quantile
levels, validation acceptance target, and score weights are design choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

KNOT = 1852 / 3600
FEATURES = ["track_unwrapped", "groundspeed_kt", "altitude_ft", "elapsed_time_sec"]
WEIGHTS = {"alignment": 0.40, "speed": 0.35, "altitude": 0.15, "turn": 0.10}
GROUPS = {
    "alignment": ["direction_error_deg", "direction_spread_deg", "aligned_length_m"],
    "speed": [
        "speed_peak_excess_kt",
        "speed_total_s",
        "speed_longest_s",
        "acceleration_peak_mps2",
        "acceleration_total_s",
        "acceleration_longest_s",
        "speed_reversals_per_min",
    ],
    "altitude": [
        "end_altitude_ft",
        "vertical_peak_mps",
        "vertical_total_s",
        "vertical_longest_s",
        "altitude_jump_m",
    ],
    "turn": [
        "turn_peak_degps",
        "turn_total_s",
        "turn_longest_s",
        "direction_jump_deg",
        "short_turn_degps",
    ],
}
LOWER = {"aligned_length_m", "path_length_m", "duration_s", "entry_radius_m"}
GATES = ["path_length_m", "duration_s", "entry_radius_m"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def angle(a, b):
    return (np.asarray(a) - b + 180) % 360 - 180


def center(a, axis=None):
    return (
        np.degrees(
            np.arctan2(
                np.nanmean(np.sin(np.radians(a)), axis=axis),
                np.nanmean(np.cos(np.radians(a)), axis=axis),
            )
        )
        % 360
    )


def remaining(xy):
    segments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.r_[np.cumsum(segments[::-1])[::-1], 0.0]


def profile(distance, values, grid):
    d, ix = np.unique(distance[::-1], return_index=True)
    return np.interp(grid, d, values[::-1][ix], left=np.nan, right=np.nan)


def durations(mask, dt):
    total = float(np.sum(dt[mask]))
    best = current = 0.0
    for flag, delta in zip(mask, dt, strict=True):
        current = current + delta if flag else 0.0
        best = max(best, current)
    return total, best


def suffix_length(error, segments, dt, soft, hard, gap_s):
    # Only short, bounded excursions can be bridged; never jump to a remote leg.
    length = gap = 0.0
    for i in range(len(segments) - 1, -1, -1):
        deviation = max(error[i], error[i + 1])
        if deviation > hard:
            break
        gap = gap + dt[i] if deviation > soft else 0.0
        if gap > gap_s:
            break
        length += segments[i]
    return float(length)


def prepare_reference(root, output):
    """Cache observed positions at the exact 200-point model time grid."""
    from traffic.data import airports

    artifact_names = [
        "zurich_tcvae-v1",
        "zurich_ddpm_unconditional-v1",
        "zurich_rwy14_flow162_216_tcvae-v1",
        "zurich_rwy14_flow162_216_ddpm_unconditional-v1",
        "zurich_rwy14_tcvae_vamp_k1000-v1",
    ]
    artifacts = [root / "data/processed" / n for n in artifact_names]
    source = (
        root / "data/interim/lszh-rwy14-landing-40nm-landing_zurich_2019-v1.parquet"
    )
    files = [source] + [
        p / n
        for p in artifacts
        for n in [
            "metadata.parquet",
            "train.npz",
            "validation.npz",
            "normalization.json",
        ]
    ]
    print("REFERENCE hashing inputs", flush=True)
    identity = {str(p.relative_to(root)): digest(p) for p in files}
    cache = output / "real-observed-200.npz"
    identity_path = output / "real-observed-200.identity.json"
    if cache.exists():
        if json.loads(identity_path.read_text())["inputs"] != identity:
            raise ValueError(
                "Reference inputs changed; choose a fresh output directory"
            )
        print("CACHE_HIT", cache, flush=True)
        return cache
    print("REFERENCE metadata", flush=True)
    metadata = [
        pd.read_parquet(p / "metadata.parquet").set_index("trajectory_id")
        for p in artifacts
    ]
    base = metadata[0]
    k1000 = metadata[-1]
    selected = {}
    for split in ["train", "validation"]:
        ids = set(base.index[base.split.eq(split)]) & set(
            k1000.index[k1000.split.eq(split)]
        )
        # A flight assigned to a different split by any target model is excluded.
        for m in metadata:
            ids -= set(m.index[~m.split.eq(split)])
        selected[split] = ids
    assert not selected["train"] & selected["validation"]
    print("REFERENCE selected", {s: len(v) for s, v in selected.items()}, flush=True)
    norm = json.loads((artifacts[0] / "normalization.json").read_text())
    values_by_id = {}
    for split in selected:
        with np.load(artifacts[0] / f"{split}.npz") as data:
            values = (
                data["values"] * np.asarray(norm["std"]) + np.asarray(norm["mean"])
            ).astype(np.float32)
            elapsed_rows = data["elapsed_time"]
            for i, ident in enumerate(data["trajectory_id"].astype(str)):
                if ident in selected[split]:
                    values_by_id[ident] = (values[i], elapsed_rows[i], split)
    print("REFERENCE observed positions", flush=True)
    frame = pd.read_parquet(
        source,
        columns=[
            "flight_id",
            "timestamp",
            "latitude",
            "longitude",
            "runway",
            "initial_flow",
        ],
    )
    frame = frame[frame.flight_id.isin(values_by_id)]
    print("REFERENCE observed rows", len(frame), flush=True)
    airport = airports["LSZH"]
    projection = pyproj.Proj(
        proj="aeqd",
        lat_0=airport.latitude,
        lon_0=airport.longitude,
        datum="WGS84",
        units="m",
    )
    rows, times, positions, latlon, identifiers, splits, flows = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for ident, f in frame.groupby("flight_id", sort=True, observed=True):
        f = f.sort_values("timestamp")
        observed_t = (f.timestamp - f.timestamp.iloc[0]).dt.total_seconds().to_numpy()
        values, elapsed, split = values_by_id[ident]
        if (
            np.any(np.diff(observed_t) <= 0)
            or not np.isfinite(f[["longitude", "latitude"]].to_numpy(float)).all()
        ):
            raise ValueError(f"Invalid observed coordinates/time for {ident}")
        if abs(observed_t[-1] - elapsed[-1]) > 0.01:
            raise ValueError(f"Cached/prepared time mismatch for {ident}")
        lon = np.interp(elapsed, observed_t, f.longitude.to_numpy(float))
        lat = np.interp(elapsed, observed_t, f.latitude.to_numpy(float))
        xy = np.column_stack(projection(lon, lat))
        # Translation changes neither heading nor path length; no rotation.
        xy -= xy[-1]
        rows.append(values)
        times.append(elapsed)
        positions.append(xy)
        latlon.append(np.column_stack([lon, lat]))
        identifiers.append(ident)
        splits.append(split)
        flows.append(str(f.initial_flow.iloc[0]))
        if len(rows) % 1000 == 0:
            print("REFERENCE flights", len(rows), flush=True)
    np.savez_compressed(
        cache,
        values=rows,
        time=times,
        xy=positions,
        lonlat=latlon,
        ids=identifiers,
        split=splits,
        flow=flows,
    )
    write_json(
        identity_path,
        {
            "inputs": identity,
            "airport": {
                "code": "LSZH",
                "latitude": airport.latitude,
                "longitude": airport.longitude,
            },
            "projection": projection.srs,
            "counts": {s: splits.count(s) for s in selected},
            "position_source": (
                "Observed longitude/latitude on existing prepared time grid; "
                "endpoint translated to airport; no rotation"
            ),
            "split_policy": (
                "Same split in every target artifact containing the flight. "
                "No generated/test trajectories used."
            ),
        },
    )
    print("CACHE_COLD", cache, flush=True)
    return cache


def base_rules(values, times, xy):
    rem = np.asarray([remaining(p) for p in xy])
    grid = np.linspace(0, np.quantile(rem[:, 0], 0.1), 81)
    headings = np.asarray(
        [
            profile(d, np.unwrap(np.radians(v[:, 0])), grid)
            for d, v in zip(rem, values, strict=True)
        ]
    )
    headings = np.degrees(headings)
    centers = center(headings, axis=0)
    spread = np.nanquantile(np.abs(angle(headings, centers)), 0.9, axis=0)
    # Calibrate stability to real endpoint variability, not the full approach
    # spread: the latter can incorrectly admit a still-turning far segment.
    endpoint_center = center(headings[:, 0])
    endpoint_band = float(
        np.quantile(np.abs(angle(headings[:, 0], endpoint_center)), 0.99)
    )
    concentrated = (spread <= endpoint_band) & (
        np.abs(angle(centers, endpoint_center)) <= endpoint_band
    )
    if not concentrated[0]:
        raise ValueError("Real endpoint direction is not in the concentrated regime")
    stop = next((i for i in range(1, len(grid)) if not concentrated[i]), len(grid) - 1)
    end = max(1, stop - 1)
    terminal = float(grid[end])
    ref = float(center(center(headings[:, : end + 1], axis=1)))
    errors = np.abs(angle(headings[:, : end + 1], ref))
    soft = float(np.quantile(np.quantile(errors, 0.95, axis=1), 0.95))
    hard = float(np.quantile(np.max(errors, axis=1), 0.99))
    dt = np.diff(times, axis=1)
    acc = np.diff(values[:, :, 1], axis=1) * KNOT / dt
    vr = np.diff(values[:, :, 2], axis=1) * 0.3048 / dt
    turn = angle(values[:, 1:, 0], values[:, :-1, 0]) / dt
    speed_profiles = np.asarray(
        [profile(d, v[:, 1], grid) for d, v in zip(rem, values, strict=True)]
    )
    rules = {
        "terminal_m": terminal,
        "terminal_detection": (
            "Contiguous endpoint prefix where cross-flight q90 angular spread "
            "and center drift stay within training endpoint q99 deviation."
        ),
        "endpoint_stability_band_deg": endpoint_band,
        "reference_deg": ref,
        "aligned_soft_deg": soft,
        "aligned_hard_deg": max(soft, hard),
        "gap_s": float(np.quantile(np.quantile(dt, 0.95, axis=1), 0.95)),
        "grid_m": grid.tolist(),
        "direction_center_deg": centers.tolist(),
        "direction_spread90_deg": spread.tolist(),
        "speed_low_kt": np.nanquantile(speed_profiles, 0.01, axis=0).tolist(),
        "speed_high_kt": np.nanquantile(speed_profiles, 0.99, axis=0).tolist(),
        "acceleration_event_mps2": float(
            np.quantile(np.quantile(np.abs(acc), 0.95, axis=1), 0.95)
        ),
        "vertical_event_mps": float(
            np.quantile(np.quantile(np.abs(vr), 0.95, axis=1), 0.95)
        ),
        "turn_event_degps": float(
            np.quantile(np.quantile(np.abs(turn), 0.95, axis=1), 0.95)
        ),
        "speed_swing_kt": float(
            np.quantile(
                np.quantile(np.abs(np.diff(values[:, :, 1], axis=1)), 0.95, axis=1),
                0.95,
            )
        ),
        "short_window_s": float(np.median(times[:, -1] - times[:, 0]) / 10),
        "weights": WEIGHTS,
    }
    return rules


def speed_reversals(speed, time, swing, window):
    # Hysteresis suppresses insignificant sign flips without smoothing values.
    low = high = speed[0]
    direction = 0
    reversals = []
    for value, stamp in zip(speed[1:], time[1:], strict=True):
        if direction >= 0:
            high = max(high, value)
            if high - value > swing:
                if direction:
                    reversals.append(float(stamp))
                direction, low = -1, value
        if direction <= 0:
            low = min(low, value)
            if value - low > swing:
                if direction:
                    reversals.append(float(stamp))
                direction, high = 1, value
    return float(np.sum(np.diff(reversals) <= window) * 60 / (time[-1] - time[0]))


def metrics(v, time, xy, rules):
    if (
        not np.isfinite(v).all()
        or not np.isfinite(xy).all()
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("nonfinite or nonchronological data")
    dt = np.diff(time)
    rem = remaining(xy)
    seg = -np.diff(rem)
    terminal = rem <= rules["terminal_m"]
    terminal_track = v[terminal, 0]
    terminal_error = np.abs(angle(terminal_track, rules["reference_deg"]))
    # Spatial interpolation gives every flight equal weight and avoids a dense
    # cluster of short time steps dominating its terminal statistics.
    grid = np.linspace(0, min(rem[0], rules["terminal_m"]), 41)
    tracks = np.degrees(profile(rem, np.unwrap(np.radians(v[:, 0])), grid))
    all_error = np.abs(angle(v[:, 0], rules["reference_deg"]))
    speed_low = np.interp(rem, rules["grid_m"], rules["speed_low_kt"])
    speed_high = np.interp(rem, rules["grid_m"], rules["speed_high_kt"])
    speed_excess = np.maximum.reduce(
        [speed_low - v[:, 1], v[:, 1] - speed_high, np.zeros(len(v))]
    )
    # Beyond the common-distance grid use the broad training entry envelope,
    # fixed during calibration, rather than extrapolating the median curve.
    far = rem > rules["grid_m"][-1]
    if far.any():
        speed_excess[far] = np.maximum.reduce(
            [
                rules["far_speed_low_kt"] - v[far, 1],
                v[far, 1] - rules["far_speed_high_kt"],
                np.zeros(far.sum()),
            ]
        )
    acc = np.abs(np.diff(v[:, 1]) * KNOT / dt)
    vr = np.abs(np.diff(v[:, 2]) * 0.3048 / dt)
    turn = np.abs(angle(v[1:, 0], v[:-1, 0])) / dt
    outside = rem[1:] > rules["terminal_m"]
    speed_total, speed_longest = durations(
        np.maximum(speed_excess[1:], speed_excess[:-1]) > 0, dt
    )
    acc_total, acc_longest = durations(acc > rules["acceleration_event_mps2"], dt)
    vr_total, vr_longest = durations(vr > rules["vertical_event_mps"], dt)
    turn_total, turn_longest = durations(
        (turn > rules["turn_event_degps"]) & outside, dt
    )
    # A time-window turning rate detects rapid reversals without penalising a
    # long, ordinary turn or self-intersection in its own right.
    short_rates = []
    for i in range(len(v) - 1):
        j = min(len(v) - 1, np.searchsorted(time, time[i] + rules["short_window_s"]))
        short_rates.append(abs(float(angle(v[j, 0], v[i, 0]))) / (time[j] - time[i]))
    return {
        "direction_error_deg": float(
            max(
                np.quantile(np.abs(angle(tracks, rules["reference_deg"])), 0.95),
                terminal_error[-1],
            )
        ),
        "direction_spread_deg": float(
            np.sqrt(np.mean(angle(tracks, center(tracks)) ** 2))
        ),
        "aligned_length_m": suffix_length(
            all_error,
            seg,
            dt,
            rules["aligned_soft_deg"],
            rules["aligned_hard_deg"],
            rules["gap_s"],
        ),
        "speed_peak_excess_kt": float(np.max(speed_excess)),
        "speed_total_s": speed_total,
        "speed_longest_s": speed_longest,
        "acceleration_peak_mps2": float(np.max(acc)),
        "acceleration_total_s": acc_total,
        "acceleration_longest_s": acc_longest,
        "speed_reversals_per_min": speed_reversals(
            v[:, 1], time, rules["speed_swing_kt"], rules["short_window_s"]
        ),
        "end_altitude_ft": float(v[-1, 2]),
        "vertical_peak_mps": float(np.max(vr)),
        "vertical_total_s": vr_total,
        "vertical_longest_s": vr_longest,
        "altitude_jump_m": float(np.max(np.abs(np.diff(v[:, 2]))) * 0.3048),
        "turn_peak_degps": float(np.max(turn[outside], initial=0)),
        "turn_total_s": turn_total,
        "turn_longest_s": turn_longest,
        "direction_jump_deg": float(np.max(np.abs(angle(v[1:, 0], v[:-1, 0])))),
        "short_turn_degps": float(max(short_rates)),
        "path_length_m": float(rem[0]),
        "duration_s": float(time[-1] - time[0]),
        "entry_radius_m": float(np.linalg.norm(xy[0] - xy[-1])),
    }


def boundaries(frame, q):
    bounds = {}
    for name in frame.columns:
        a = frame[name].to_numpy(float)
        if name == "end_altitude_ft":
            bounds[name] = {
                "kind": "interval",
                "fail_low": float(np.quantile(a, 1 - q)),
                "good_low": float(np.quantile(a, 0.1)),
                "good_high": float(np.quantile(a, 0.9)),
                "fail_high": float(np.quantile(a, q)),
            }
        elif name in LOWER:
            bounds[name] = {
                "kind": "lower",
                "fail": float(np.quantile(a, 1 - q)),
                "good": float(np.quantile(a, 0.25)),
            }
        else:
            bounds[name] = {
                "kind": "upper",
                "good": float(np.quantile(a, 0.75)),
                "fail": float(np.quantile(a, q)),
            }
    return bounds


def item_score(value, bound):
    kind = bound["kind"]
    if kind == "interval":
        if value < bound["good_low"]:
            return item_score(
                value,
                {"kind": "lower", "fail": bound["fail_low"], "good": bound["good_low"]},
            )
        return item_score(
            value,
            {"kind": "upper", "fail": bound["fail_high"], "good": bound["good_high"]},
        )
    sign = 1 if kind == "upper" else -1
    v, good, fail = value * sign, bound["good"] * sign, bound["fail"] * sign
    passed = v <= fail
    score = (
        100.0
        if v <= good
        else (
            0.0
            if fail <= good
            else float(np.clip(100 * (fail - v) / (fail - good), 0, 100))
        )
    )
    return bool(passed), score


def evaluate(measures, rules):
    items = {
        k: {
            "value": float(v),
            "passed": item_score(v, rules["bounds"][k])[0],
            "score": item_score(v, rules["bounds"][k])[1],
            "boundary": rules["bounds"][k],
        }
        for k, v in measures.items()
    }
    groups = {
        g: float(np.mean([items[n]["score"] for n in names]))
        for g, names in GROUPS.items()
    }
    failures = [n for n, r in items.items() if not r["passed"]]
    total = sum(groups[g] * w for g, w in WEIGHTS.items())
    return items, groups, failures, total


def calibrate(root, output):
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "rules.json"
    if frozen.exists():
        raise FileExistsError(
            "Rules already frozen; select a new calibration output to change them"
        )
    cache = prepare_reference(root, output)
    with np.load(cache) as data:
        arrays = {k: data[k] for k in data.files}
    train = arrays["split"] == "train"
    val = arrays["split"] == "validation"
    print("REFERENCE_COUNTS", int(train.sum()), int(val.sum()), flush=True)
    if min(train.sum(), val.sum()) < 100:
        raise ValueError(
            "Insufficient independent real calibration/validation trajectories"
        )
    rules = base_rules(
        arrays["values"][train], arrays["time"][train], arrays["xy"][train]
    )
    rules["far_speed_low_kt"] = float(
        np.quantile(np.min(arrays["values"][train, :, 1], axis=1), 0.01)
    )
    rules["far_speed_high_kt"] = float(
        np.quantile(np.max(arrays["values"][train, :, 1], axis=1), 0.99)
    )
    tables = {}
    for split, mask in [("train", train), ("validation", val)]:
        rows = [
            metrics(v, t, p, rules)
            for v, t, p in zip(
                arrays["values"][mask],
                arrays["time"][mask],
                arrays["xy"][mask],
                strict=True,
            )
        ]
        tables[split] = pd.DataFrame(rows)
        tables[split].assign(
            trajectory_id=arrays["ids"][mask], initial_flow=arrays["flow"][mask]
        ).to_csv(output / f"{split}_metrics.csv", index=False)
        print("METRICS", split, len(rows), flush=True)
    candidates = []
    chosen = None
    # Choose the tightest training quantile candidate retaining >=90% of real
    # validation flights under all rules jointly. Generated inputs are unopened.
    for q in [0.99, 0.995, 0.999, 1.0]:
        rules["bounds"] = boundaries(tables["train"], q)
        passed, failures = [], {name: 0 for name in rules["bounds"]}
        for row in tables["validation"].to_dict("records"):
            _, _, bad, _ = evaluate(row, rules)
            passed.append(not bad)
            for name in bad:
                failures[name] += 1
        result = {
            "quantile": q,
            "validation_acceptance": float(np.mean(passed)),
            "rejections_by_metric": failures,
        }
        candidates.append(result)
        if chosen is None and result["validation_acceptance"] >= 0.90:
            chosen = q
    if chosen is None:
        write_json(output / "calibration_blocked.json", {"candidates": candidates})
        raise ValueError(
            "No training-only candidate meets declared validation retention; "
            "do not score generated samples"
        )
    rules["bounds"] = boundaries(tables["train"], chosen)
    rules.update(
        {
            "schema": 1,
            "chosen_quantile": chosen,
            "design": {
                "candidate_quantiles": [0.99, 0.995, 0.999, 1.0],
                "validation_retention_target": 0.90,
                "good_quantiles": "upper: .75; lower: .25; interval: .1-.9",
                "distance_grid_points": 81,
                "terminal_grid_points": 41,
                "short_window": "training median duration / 10",
                "raw_corrections": (
                    "No nonpositive raw time increments or substantive "
                    "speed/altitude clipping. Only roundoff is tolerated."
                ),
                "numeric_precision": (
                    "32 float32 eps * per-channel max magnitude "
                    "(numerical tolerance, not physical threshold)"
                ),
            },
            "reference_identity_sha256": digest(
                output / "real-observed-200.identity.json"
            ),
            "cache_sha256": digest(cache),
            "counts": {"train": int(train.sum()), "validation": int(val.sum())},
            "validation_candidates": candidates,
            "scope": (
                "RWY14 air approach. Other runway directions fail this scenario. "
                "No clustering, similarity matching, smoothing, or sample repair."
            ),
            "reconstruction": (
                "traffic reverse geodesic: heading[i+1]-180, "
                "speed[i+1]*0.99*1852/3600*dt; measured lon/lat. Gate only."
            ),
        }
    )
    # Independent position/speed diagnostic uses observed coordinates only.
    geod = pyproj.Geod(ellps="WGS84")
    observed_ratios = []
    for ll, v, t in zip(
        arrays["lonlat"][train],
        arrays["values"][train],
        arrays["time"][train],
        strict=True,
    ):
        _, _, distance = geod.inv(ll[:-1, 0], ll[:-1, 1], ll[1:, 0], ll[1:, 1])
        expected = 0.5 * (v[:-1, 1] + v[1:, 1]) * KNOT * np.diff(t)
        observed_ratios.append(float(np.median(distance / expected)))
    rules["observed_position_speed_ratio_quantiles"] = dict(
        zip(
            ["q01", "q50", "q99"],
            np.quantile(observed_ratios, [0.01, 0.5, 0.99]).tolist(),
            strict=True,
        )
    )
    write_json(frozen, rules)
    for split in tables:
        rows = []
        for ident, m in zip(
            arrays["ids"][arrays["split"] == split],
            tables[split].to_dict("records"),
            strict=True,
        ):
            _, groups, bad, total = evaluate(m, rules)
            rows.append(
                {
                    "id": ident,
                    "passed": not bad,
                    "failures": "|".join(bad),
                    "score": total,
                    **groups,
                }
            )
        pd.DataFrame(rows).to_csv(output / f"{split}_checks.csv", index=False)
    print("FROZEN", chosen, rules["reference_deg"], rules["terminal_m"], flush=True)
    print(frozen)


def numerical_tolerance(values):
    return float(
        32 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(values))))
    )


def reconstruction_error(frame):
    # Check the exact deployed traffic formula locally per segment. Coordinates
    # are read only; no generated positions are replaced by these predictions.
    from pitot.geodesy import destination

    time = pd.to_datetime(frame.timestamp, utc=True)
    dt = np.array(
        [
            (time.iloc[i + 1] - time.iloc[i]).total_seconds()
            for i in range(len(frame) - 1)
        ]
    )
    speed = frame.groundspeed.to_numpy(float)
    heading = frame.track.to_numpy(float)
    lat = frame.latitude.to_numpy(float)
    lon = frame.longitude.to_numpy(float)
    pred_lat, pred_lon, _ = destination(
        lat[1:], lon[1:], heading[1:] - 180, 0.99 * speed[1:] * dt * KNOT
    )
    _, _, error = pyproj.Geod(ellps="WGS84").inv(lon[:-1], lat[:-1], pred_lon, pred_lat)
    return float(np.max(error))


def score(root, rules_dir, manifest, output):
    rules_path = rules_dir / "rules.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    summary = {"rules_sha256": digest(rules_path), "models": [], "manifest": spec}
    for model in spec["models"]:
        source = root / model["samples"]
        hashes = {n: digest(source / n) for n in ["generated.parquet", "samples.npz"]}
        frame = pd.read_parquet(source / "generated.parquet")
        ids = frame.flight_id.drop_duplicates().astype(str).tolist()[:100]
        if len(ids) != 100:
            raise ValueError("Exactly 100 fixed source trajectories required")
        with np.load(source / "samples.npz") as data:
            raw = data["values"][:100].copy()
        target = output / model["id"]
        target.mkdir()
        records = []
        for index, ident in enumerate(ids):
            f = frame[frame.flight_id.eq(ident)].copy()
            # Preserve acquisition/model order. Sorting would conceal bad time.
            failures, unknown = [], []
            raw_v = raw[index].astype(float)
            v = f[["track", "groundspeed", "altitude", "elapsed_time"]].to_numpy(float)
            xy = f[["x", "y"]].to_numpy(float)
            checks = {}
            m, items, groups, total = {}, {}, {}, None
            finite = (
                np.isfinite(raw_v).all()
                and np.isfinite(v).all()
                and np.isfinite(xy).all()
            )
            checks["finite"] = bool(finite)
            if len(f) != 200 or raw_v.shape != (200, 4):
                unknown.append("shape_or_feature_contract")
            elif not finite:
                failures.append("nonfinite_data")
            else:
                embedded = f[FEATURES].to_numpy(float)
                checks["raw_npz_matches_rows"] = bool(
                    np.allclose(
                        raw_v, embedded, atol=numerical_tolerance(raw_v), rtol=0
                    )
                )
                if not checks["raw_npz_matches_rows"]:
                    unknown.append("raw_sample_row_mapping")
                raw_dt = np.diff(raw_v[:, 3])
                checks["raw_min_dt_s"] = float(np.min(raw_dt))
                checks["raw_nonpositive_intervals"] = int(np.sum(raw_dt <= 0))
                if np.any(raw_dt <= 0):
                    failures.append("raw_time_nonpositive")
                for j, name in [(1, "speed"), (2, "altitude")]:
                    amount = float(np.max(np.abs(raw_v[:, j] - v[:, j])))
                    checks[f"{name}_correction_max"] = amount
                    if amount > numerical_tolerance(raw_v[:, j]):
                        failures.append(f"raw_{name}_clipping")
                timing_change = float(
                    np.max(np.abs((raw_v[:, 3] - raw_v[0, 3]) - v[:, 3]))
                )
                checks["time_correction_max_s"] = timing_change
                if timing_change > numerical_tolerance(raw_v[:, 3]):
                    failures.append("raw_time_repair")
                dt = np.diff(v[:, 3])
                if np.any(dt <= 0):
                    failures.append("reconstructed_time_nonpositive")
                else:
                    checks["reconstruction_max_error_m"] = reconstruction_error(f)
                    # Float32 integration error bound scales with travelled distance.
                    tolerance = numerical_tolerance(remaining(xy))
                    checks["reconstruction_numeric_tolerance_m"] = tolerance
                    if checks["reconstruction_max_error_m"] > tolerance:
                        failures.append("reconstruction_inconsistent")
                    m = metrics(v, v[:, 3], xy, rules)
                    items, groups, metric_failures, total = evaluate(m, rules)
                    failures += metric_failures
            status = (
                "rejected" if failures else ("undetermined" if unknown else "passed")
            )

            def safe(a):
                return [float(x) if np.isfinite(x) else None for x in a]

            records.append(
                {
                    "id": ident,
                    "source_index": index,
                    "status": status,
                    "failures": failures,
                    "unknown": unknown,
                    "checks": checks,
                    "metrics": items,
                    "groups": groups,
                    "total": total if status == "passed" else None,
                    "diagnostic_total": total,
                    "rank": None,
                    "x": safe(xy[:, 0]),
                    "y": safe(xy[:, 1]),
                    "remaining_m": safe(remaining(xy)),
                    "time_s": safe(v[:, 3]),
                    "speed_kt": safe(v[:, 1]),
                    "altitude_ft": safe(v[:, 2]),
                    "track_deg": safe(v[:, 0]),
                    "raw_time_s": safe(raw_v[:, 3]),
                    "raw_speed_kt": safe(raw_v[:, 1]),
                }
            )
        ranked = sorted(
            [r for r in records if r["status"] == "passed"],
            key=lambda r: (
                -r["total"],
                -r["groups"]["alignment"],
                -r["groups"]["speed"],
                r["id"],
            ),
        )
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        counts = {
            s: sum(r["status"] == s for r in records)
            for s in ["passed", "rejected", "undetermined"]
        }
        result = {
            "model": model,
            "rules_sha256": digest(rules_path),
            "source_hashes": hashes,
            "selection": (
                "First 100 IDs in original parquet order, fixed before scoring; "
                "no resampling."
            ),
            "source_count": int(frame.flight_id.nunique()),
            "counts": counts,
            "top10": [r["id"] for r in ranked[:10]],
            "top5": [r["id"] for r in ranked[:5]],
            "records": records,
        }
        write_json(target / "results.json", result)
        rows = []
        for r in records:
            rows.append(
                {
                    "id": r["id"],
                    "source_index": r["source_index"],
                    "status": r["status"],
                    "rank": r["rank"],
                    "total": r["total"],
                    "diagnostic_total": r["diagnostic_total"],
                    **r["groups"],
                    "failures": "|".join(r["failures"]),
                    "unknown": "|".join(r["unknown"]),
                    **r["checks"],
                    **{k: a["value"] for k, a in r["metrics"].items()},
                }
            )
        pd.DataFrame(rows).to_csv(target / "scores.csv", index=False)
        write_json(
            target / "highlights.json",
            {"top5": result["top5"], "top10": result["top10"]},
        )
        # Retain an exact subset of the original rows, including coordinates and
        # raw output features. The original sample files remain untouched.
        frame[frame.flight_id.isin(ids)].to_parquet(
            target / "fixed100.parquet", index=False
        )
        np.savez_compressed(target / "fixed100-raw.npz", values=raw, ids=np.array(ids))
        assert hashes == {n: digest(source / n) for n in hashes}
        for r in records:
            original = frame[frame.flight_id.eq(r["id"])][["x", "y"]].to_numpy(float)
            np.testing.assert_equal(
                original, np.column_stack([r["x"], r["y"]]).astype(float)
            )
        assert all(r["status"] == "passed" for r in ranked[:10])
        summary["models"].append(
            {
                "id": model["id"],
                "label": model["label"],
                **counts,
                "highlighted": len(ranked[:10]),
                "top10": result["top10"],
                "source_hashes": hashes,
            }
        )
        print("SCORED", model["id"], counts, flush=True)
    write_json(output / "summary.json", summary)
    print(output)


def training_display(rules_dir, seed=42):
    """Keep observed training examples fixed across comparable model families."""
    cache = rules_dir / "real-observed-200.npz"
    with np.load(cache) as data:
        arrays = {key: data[key] for key in ["ids", "split", "flow", "xy", "values"]}
    result = {"seed": seed, "source_sha256": digest(cache), "cohorts": {}}
    for cohort, flow in [("global", None), ("single", "162-216")]:
        mask = arrays["split"] == "train"
        if flow is not None:
            mask &= arrays["flow"] == flow
        eligible = np.flatnonzero(mask)
        eligible = eligible[np.argsort(arrays["ids"][eligible])]
        if len(eligible) < 100:
            raise ValueError(f"Only {len(eligible)} training references for {cohort}")
        indices = np.random.default_rng(seed).choice(eligible, 100, replace=False)
        rows = []
        for i in indices:
            xy, v = arrays["xy"][i], arrays["values"][i]
            rows.append(
                {
                    "id": str(arrays["ids"][i]),
                    "flow": str(arrays["flow"][i]),
                    "split": "train",
                    "x": xy[:, 0].tolist(),
                    "y": xy[:, 1].tolist(),
                    "remaining_m": remaining(xy).tolist(),
                    "speed_kt": v[:, 1].tolist(),
                    "altitude_ft": v[:, 2].tolist(),
                    "track_deg": v[:, 0].tolist(),
                    "turn_travel_deg": float(np.abs(angle(v[1:, 0], v[:-1, 0])).sum()),
                }
            )
        result["cohorts"][cohort] = {"eligible": len(eligible), "records": rows}
    return result


def turn_review(model):
    """Explain the current rule effects without inferring flight validity."""
    rows = model["records"]
    turn_names = set(GROUPS["turn"])
    return {
        "id": model["model"]["id"],
        "turn_rule_failures": sum(
            bool(turn_names.intersection(r["failures"])) for r in rows
        ),
        "entry_radius_failures": sum("entry_radius_m" in r["failures"] for r in rows),
        "entry_radius_only": [
            r["id"]
            for r in rows
            if r["failures"] == ["entry_radius_m"] and not r["unknown"]
        ],
        "alignment_failures": sum(
            bool(set(GROUPS["alignment"]).intersection(r["failures"])) for r in rows
        ),
        "passed_turn_score_full": sum(
            r["status"] == "passed" and r["groups"]["turn"] == 100 for r in rows
        ),
        "passed": model["counts"]["passed"],
    }


def render(rules_dir, results_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rules = json.loads((rules_dir / "rules.json").read_text(encoding="utf-8"))
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    identity = json.loads(
        (rules_dir / "real-observed-200.identity.json").read_text(encoding="utf-8")
    )
    data = {
        "rules": rules,
        "identity": identity,
        "models": [],
        "training": training_display(rules_dir),
    }
    for model in summary["models"]:
        data["models"].append(
            json.loads(
                (results_dir / model["id"] / "results.json").read_text(encoding="utf-8")
            )
        )
    data["turn_review"] = [turn_review(model) for model in data["models"]]
    write_json(results_dir / "training-display.json", data["training"])
    write_json(results_dir / "turn-review.json", data["turn_review"])
    template = (
        Path(__file__).with_name("trajectory_rules.html").read_text(encoding="utf-8")
    )
    html = template.replace(
        "__DATA__",
        json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"),
    )
    path = results_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    for model in data["models"]:
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        highlighted = set(model["top10"])
        # Draw every original trajectory first, then redraw only passing top K.
        for foreground in [False, True]:
            for row in model["records"]:
                if foreground and row["id"] not in highlighted:
                    continue
                color = "#ec8730" if foreground else "#b9c3d0"
                alpha = 0.9 if foreground else 0.4
                width = 1.7 if foreground else 0.6
                axes[0].plot(
                    np.array(row["x"]) / 1000,
                    np.array(row["y"]) / 1000,
                    color=color,
                    alpha=alpha,
                    lw=width,
                )
                for ax, key in zip(axes[1:], ["speed_kt", "altitude_ft"], strict=True):
                    ax.plot(
                        np.array(row["remaining_m"]) / 1000,
                        row[key],
                        color=color,
                        alpha=alpha,
                        lw=width,
                    )
        axes[0].scatter(
            [0], [0], color="#174d61", marker="+", s=90, label="LSZH airport anchor"
        )
        theta = np.radians(rules["reference_deg"])
        length = rules["terminal_m"] / 1000
        axes[0].plot(
            [-length * np.sin(theta), 0],
            [-length * np.cos(theta), 0],
            "--",
            color="#327989",
            label="Train terminal direction (not runway)",
        )
        axes[0].set(xlabel="East (km)", ylabel="North (km)", aspect="equal")
        axes[0].legend(fontsize=7)
        axes[1].set(xlabel="Remaining path (km)", ylabel="Groundspeed (kt)")
        axes[2].set(xlabel="Remaining path (km)", ylabel="MSL altitude (ft)")
        for ax in axes:
            ax.grid(alpha=0.2)
        count = model["counts"]
        fig.suptitle(
            f"{model['model']['id']} | 100 retained | "
            f"pass {count['passed']} / reject {count['rejected']} / "
            f"unknown {count['undetermined']} | top {len(highlighted)}"
        )
        fig.tight_layout()
        fig.savefig(results_dir / model["model"]["id"] / "overview.png", dpi=150)
        plt.close(fig)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["reference", "calibrate"]:
        p = commands.add_parser(name)
        p.add_argument("--root", type=Path, default=Path.cwd())
        p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("score")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("render")
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reference":
        args.output.mkdir(parents=True, exist_ok=True)
        print(prepare_reference(args.root, args.output))
    elif args.command == "calibrate":
        calibrate(args.root, args.output)
    elif args.command == "score":
        score(args.root, args.rules, args.manifest, args.output)
    else:
        render(args.rules, args.results)


if __name__ == "__main__":
    main()
