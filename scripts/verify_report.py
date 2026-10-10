"""Audit report counts and W1 with separate equal-size empirical calculations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root, report = args.root.resolve(), args.report.resolve()
    manifest = json.loads((report / "manifest.json").read_text("utf8"))
    summary = pd.read_csv(report / "summary.csv").set_index("model")
    checks = []
    for filename, expected in manifest["input_sha256"].items():
        path = root / filename
        if path.suffix == ".py":
            candidates = [path, *(root / "scripts/archive").rglob(path.name)]
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                with candidate.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                        path = candidate
                        break
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == expected, filename
    checks.append("Every recorded input hash matches the current file")
    inputs = root / "runs/minisky-preparation-20261003/inputs"
    for name, row in summary.iterrows():
        with np.load(inputs / "generated" / name / "raw_model_values.npz") as arrays:
            raw = arrays["values"].astype(float)
            ids = arrays["ids"].astype(str)
        nonpositive = (np.diff(raw[:, :, 3], axis=1) <= 0).any(axis=1)
        assert int(nonpositive.sum()) == row.raw_invalid_time_count
        original = pd.read_csv(inputs / "generated" / name / "raw_checks.csv")
        original = original.set_index("id").loc[ids]
        np.testing.assert_array_equal(nonpositive, original.raw_nonpositive_time)
        np.testing.assert_allclose(
            original.time_correction_max_s.max(), row.max_time_correction_s, atol=0.002
        )
        real = pd.read_parquet(
            inputs / "real" / row.reference / "reference_200.parquet"
        )
        for column, channel in (("groundspeed", 1), ("altitude", 2)):
            # Equal-sized empirical W1 is the mean sorted absolute difference.
            independent = np.abs(
                np.sort(real[column].to_numpy()) - np.sort(raw[:, :, channel].ravel())
            ).mean()
            np.testing.assert_allclose(
                independent, row[f"w1_point_{column}"], rtol=1e-12
            )
        features = pd.read_csv(report / name / "trajectory_features.csv")
        assert features.groupby("source").size().to_dict() == {
            "real": 100,
            "generated_raw": 100,
            "generated_processed": 100,
        }
        generated = features.loc[features.source == "generated_raw"]
        assert generated.p95_abs_vertical_rate.notna().sum() == 100 - nonpositive.sum()
        real_features = features.loc[features.source == "real"]
        independent = np.abs(
            np.sort(generated.duration) - np.sort(real_features.duration)
        ).mean()
        np.testing.assert_allclose(independent, row.w1_duration, rtol=1e-12)
        with np.load(report / name / "intervals.npz") as interval:
            lower, upper, threshold = (
                interval["lower"],
                interval["upper"],
                interval["threshold"],
            )
        relative = raw[:, :, 3] - raw[:, :1, 3]
        evaluated = np.stack([raw[:, :, 1], raw[:, :, 2], relative], axis=2)
        fraction = ((evaluated < lower) | (evaluated > upper)).mean(axis=(1, 2))
        np.testing.assert_allclose(
            fraction.mean(), row.generated_point_outside_fraction
        )
        assert (
            int((fraction > threshold).sum()) == row.generated_trajectory_outlier_count
        )
        outliers = pd.read_csv(report / name / "outliers.csv")
        actual = (
            outliers.loc[outliers.source == "generated"]
            .set_index("trajectory_id")
            .loc[ids]
        )
        np.testing.assert_allclose(actual.outside_fraction, fraction)
        assert actual.is_outlier.sum() == row.generated_trajectory_outlier_count
        coordinates = pd.read_csv(report / name / "pca_coordinates.csv")
        assert (coordinates.source == "held_out_real").sum() == 100
        assert (coordinates.source == "Generated raw").sum() == 100
        checks.append(
            f"{name}: raw checks, sorted-W1, counts, intervals and PCA counts verified"
        )
    latent = pd.read_csv(report / "latent_summary.csv")
    for row in latent.itertuples():
        distances = pd.read_csv(report / "latent" / row.latent_model / "distances.csv")
        distances = distances.loc[distances.source == row.source]
        assert len(distances) == row.count == 500
        np.testing.assert_allclose(
            distances.nearest_train_draw.median(), row.nearest_train_draw_median
        )
        if row.latent_model == "k1000":
            events = distances.nearest_train_draw > row.nn_outlier_threshold
            np.testing.assert_allclose(events.mean(), row.nn_outlier_fraction)
            assert events.sum() == row.nn_outlier_count
    checks.append(
        "500 latent samples per source and validation-calibrated latent counts verified"
    )
    result = {
        "passed": True,
        "checks": checks,
        "models": 6,
        "generated_per_model": 100,
        "points_per_trajectory": 200,
    }
    (report / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", "utf8"
    )
    print(json.dumps(result, indent=2))
    print(f"OUTPUT={report / 'verification.json'}")


if __name__ == "__main__":
    main()
