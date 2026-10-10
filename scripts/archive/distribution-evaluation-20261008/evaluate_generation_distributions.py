"""Evaluate fixed MiniSky inputs without training, sampling or repairing them."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import cdist, pdist
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

MODEL_REFERENCES = {
    "ddpm-flow700": "zurich-singleflow",
    "tcvae-flow-k256": "zurich-singleflow",
    "tcvae-k1000": "zurich-rwy14-allflow",
    "latent-k1000": "zurich-rwy14-allflow",
    "ddpm-global700": "zurich-all-airport",
    "tianjin-noaib-tcvae-k1000": "tianjin-noaib-rwy16l",
}
ARTIFACTS = {
    "zurich-singleflow": (
        "data/processed/zurich_rwy14_flow162_216_ddpm_unconditional-v1"
    ),
    "zurich-rwy14-allflow": (
        "runs/latent_denoising_video-v1-seed-42/inputs/processed/"
        "zurich_rwy14_tcvae_vamp_k1000-v1"
    ),
    "zurich-all-airport": "data/processed/zurich_ddpm_unconditional-v1",
    "tianjin-noaib-rwy16l": ("data/processed/tianjin_noaib_rwy16l_tcvae_vamp_k1000-v1"),
}
FEATURES = {
    1: ("groundspeed", "kt"),
    2: ("altitude", "ft MSL"),
    3: ("elapsed_time", "s"),
}
UNITS = {
    "duration": "s",
    "start_speed": "kt",
    "mean_speed": "kt",
    "end_speed": "kt",
    "start_altitude": "ft MSL",
    "mean_altitude": "ft MSL",
    "end_altitude": "ft MSL",
    "total_abs_turn": "degree",
    "p95_abs_turn_step": "degree/step",
    "p95_abs_speed_step": "kt/step",
    "p95_abs_altitude_step": "ft/step",
    "mean_abs_speed_second_difference": "kt/step^2",
    "mean_abs_altitude_second_difference": "ft/step^2",
    "upward_step_fraction": "fraction",
    "time_step_cv": "dimensionless",
    "p95_abs_vertical_rate": "ft/min",
    "p95_abs_turn_rate": "degree/s",
    "p95_abs_speed_rate": "kt/s",
    "path_length": "km",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf8",
    )


def circular_delta(track: np.ndarray) -> np.ndarray:
    return (np.diff(track, axis=1) + 180) % 360 - 180


def relative_time(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[:, :, 3] -= result[:, :1, 3]
    return result


def load_table(path: Path, count: int) -> dict:
    frame = pd.read_parquet(path).sort_values(
        ["flight_id", "point_index"], kind="stable"
    )
    ids = frame.flight_id.drop_duplicates().to_numpy().astype(str)[:count]
    groups = [frame.loc[frame.flight_id == identifier] for identifier in ids]
    if any(len(group) != 200 for group in groups):
        raise ValueError(f"Expected 200 points per flight in {path}")
    if any(not np.array_equal(group.point_index, np.arange(200)) for group in groups):
        raise ValueError(f"Invalid point order in {path}")
    columns = ["track", "groundspeed", "altitude", "elapsed_time_s"]
    values = np.stack([group[columns].to_numpy(float) for group in groups])
    xy = np.stack([group[["x", "y"]].to_numpy(float) for group in groups])
    if not np.isfinite(values).all() or not np.isfinite(xy).all():
        raise ValueError(f"Nonfinite input in {path}")
    return {"ids": ids, "values": relative_time(values), "xy": xy}


def load_prepared(path: Path) -> dict:
    norm = json.loads((path / "normalization.json").read_text("utf8"))
    mean, std = np.asarray(norm["mean"]), np.asarray(norm["std"])
    if mean.shape != (200, 4) or std.shape != (200, 4):
        raise ValueError(f"Expected the four kinematic channels in {path}")
    splits = {}
    for split in ("train", "validation", "test"):
        with np.load(path / f"{split}.npz", allow_pickle=False) as data:
            values = data["values"].astype(float) * std + mean
            values[:, :, 3] = data["elapsed_time"]
            splits[split] = {
                "values": relative_time(values),
                "ids": data["trajectory_id"].astype(str),
                "split_ids": data["split_group_id"].astype(str),
            }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field in ("ids", "split_ids"):
            if set(splits[a][field]) & set(splits[b][field]):
                raise ValueError(f"Split leakage in {path}: {field}")
    return splits


def validate_reference(real: dict, splits: dict) -> dict:
    test = splits["test"]
    indices = {identifier: i for i, identifier in enumerate(test["ids"])}
    if not set(real["ids"]) <= set(indices):
        raise ValueError("MiniSky reference contains IDs outside the test split")
    prepared = test["values"][[indices[i] for i in real["ids"]]]
    delta = prepared - real["values"]
    delta[:, :, 0] = (delta[:, :, 0] + 180) % 360 - 180
    max_error = np.max(np.abs(delta), axis=(0, 1))
    # Stored float32 normalization and two resampling paths can differ slightly.
    tolerance = np.array([0.02, 0.02, 0.1, 0.01])
    if np.any(max_error > tolerance):
        raise ValueError(f"Reference/prepared mismatch: {max_error.tolist()}")
    return {
        "max_reference_prepared_error": max_error.tolist(),
        "tolerance": tolerance.tolist(),
        "test_ids_verified": len(real["ids"]),
    }


def trajectory_features(values: np.ndarray, xy: np.ndarray | None = None) -> dict:
    turn = circular_delta(values[:, :, 0])
    speed, altitude, elapsed = (values[:, :, i] for i in (1, 2, 3))
    ds, dh, dt = (np.diff(a, axis=1) for a in (speed, altitude, elapsed))
    valid = np.isfinite(dt).all(axis=1) & (dt > 0).all(axis=1)
    result = {
        "duration": elapsed[:, -1] - elapsed[:, 0],
        "start_speed": speed[:, 0],
        "mean_speed": speed.mean(axis=1),
        "end_speed": speed[:, -1],
        "start_altitude": altitude[:, 0],
        "mean_altitude": altitude.mean(axis=1),
        "end_altitude": altitude[:, -1],
        "total_abs_turn": np.abs(turn).sum(axis=1),
        "p95_abs_turn_step": np.quantile(np.abs(turn), 0.95, axis=1),
        "p95_abs_speed_step": np.quantile(np.abs(ds), 0.95, axis=1),
        "p95_abs_altitude_step": np.quantile(np.abs(dh), 0.95, axis=1),
        "mean_abs_speed_second_difference": np.abs(np.diff(ds, axis=1)).mean(1),
        "mean_abs_altitude_second_difference": np.abs(np.diff(dh, axis=1)).mean(1),
        "upward_step_fraction": (dh > 0).mean(axis=1),
    }
    for key, increments, factor in (
        ("p95_abs_vertical_rate", dh, 60),
        ("p95_abs_turn_rate", turn, 1),
        ("p95_abs_speed_rate", ds, 1),
    ):
        result[key] = np.full(len(values), np.nan)
        result[key][valid] = np.quantile(
            np.abs(increments[valid] / dt[valid]) * factor, 0.95, axis=1
        )
    result["time_step_cv"] = np.full(len(values), np.nan)
    result["time_step_cv"][valid] = dt[valid].std(1) / dt[valid].mean(1)
    if xy is not None:
        result["path_length"] = (
            np.linalg.norm(np.diff(xy, axis=1), axis=2).sum(1) / 1000
        )
    return result


def compare_1d(
    real: np.ndarray,
    generated: np.ndarray,
    rng: np.random.Generator,
    bootstrap: int = 0,
) -> dict:
    real = np.asarray(real).ravel()
    generated = np.asarray(generated).ravel()
    real, generated = real[np.isfinite(real)], generated[np.isfinite(generated)]
    if not len(real) or not len(generated):
        return {"n_real": len(real), "n_generated": len(generated)}
    w1 = float(wasserstein_distance(real, generated))
    result = {
        "n_real": len(real),
        "n_generated": len(generated),
        "wasserstein": w1,
        "ks_statistic": float(ks_2samp(real, generated, method="asymp").statistic),
        "real_median": float(np.median(real)),
        "generated_median": float(np.median(generated)),
        "real_p05": float(np.quantile(real, 0.05)),
        "real_p95": float(np.quantile(real, 0.95)),
        "generated_p05": float(np.quantile(generated, 0.05)),
        "generated_p95": float(np.quantile(generated, 0.95)),
    }
    if bootstrap:
        draws = [
            wasserstein_distance(
                rng.choice(real, len(real), replace=True),
                rng.choice(generated, len(generated), replace=True),
            )
            for _ in range(bootstrap)
        ]
        low, high = np.quantile(draws, [0.025, 0.975])
        result.update(wasserstein_ci_low=float(low), wasserstein_ci_high=float(high))
    return result


def envelope_scores(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    outside = (values[:, :, 1:4] < lower) | (values[:, :, 1:4] > upper)
    return outside.mean(axis=(1, 2)), outside


def calibrated_envelope(train: np.ndarray, validation: np.ndarray) -> dict:
    lower, upper = np.quantile(train[:, :, 1:4], [0.01, 0.99], axis=0)
    scores, _ = envelope_scores(validation, lower, upper)
    threshold = float(np.quantile(scores, 0.95, method="higher"))
    return {
        "lower": lower,
        "upper": upper,
        "threshold": threshold,
        "validation_scores": scores,
    }


def mean_ci(values: np.ndarray, rng: np.random.Generator, bootstrap: int) -> tuple:
    if not bootstrap:
        return float(values.mean()), float(values.mean())
    means = rng.choice(values, (bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def wilson_interval(events: np.ndarray) -> tuple:
    n, p, z = len(events), float(np.mean(events)), 1.959963984540054
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def joint_representation(values: np.ndarray, *, fixed_scale: bool) -> np.ndarray:
    track = np.deg2rad(values[:, :, 0])
    features = np.stack(
        [
            np.sin(track),
            np.cos(track),
            values[:, :, 1],
            values[:, :, 2],
            values[:, :, 3],
        ],
        axis=2,
    )
    if fixed_scale:
        features /= np.array([1, 1, 100, 10000, 1000])
    return features.reshape(len(values), -1)


def distance_diagnostics(real: np.ndarray, generated: np.ndarray) -> tuple:
    cross = cdist(real, generated) / np.sqrt(real.shape[1])
    real_self = cdist(real, real) / np.sqrt(real.shape[1])
    gen_self = cdist(generated, generated) / np.sqrt(real.shape[1])
    np.fill_diagonal(real_self, np.inf)
    np.fill_diagonal(gen_self, np.inf)
    real_loo = real_self.min(axis=1)
    to_generated, to_real = cross.min(axis=1), cross.min(axis=0)
    real_pairs, gen_pairs = pdist(real), pdist(generated)
    baseline = float(np.median(real_loo))
    result = {
        "real_to_generated_nn_median": float(np.median(to_generated)),
        "generated_to_real_nn_median": float(np.median(to_real)),
        "real_loo_nn_median": baseline,
        "coverage_distance_ratio": float(np.median(to_generated) / baseline),
        "diversity_pair_distance_ratio": float(
            np.median(gen_pairs) / np.median(real_pairs)
        ),
        "generated_loo_nn_median": float(np.median(gen_self.min(axis=1))),
        "exact_duplicate_generated": len(generated) - len(np.unique(generated, axis=0)),
    }
    return result, to_generated, to_real, real_loo


def ecdf(axis, values: np.ndarray, label: str, color: str) -> None:
    values = np.asarray(values).ravel()
    values = np.sort(values[np.isfinite(values)])
    if len(values):
        axis.step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            label=label,
            color=color,
            linewidth=1.6,
        )
    axis.grid(alpha=0.2)
    axis.set_ylabel("Cumulative fraction")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_distributions(
    target: Path,
    name: str,
    real: dict,
    raw: np.ndarray,
    scalar_real: dict,
    scalar_gen: dict,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("Groundspeed (kt)", real["values"][:, :, 1], raw[:, :, 1]),
        ("Altitude (ft MSL)", real["values"][:, :, 2], raw[:, :, 2]),
        ("Duration (s)", scalar_real["duration"], scalar_gen["duration"]),
        (
            "Path length (km; exported coordinates)",
            scalar_real["path_length"],
            scalar_gen["path_length"],
        ),
        (
            "Total absolute turn (degree)",
            scalar_real["total_abs_turn"],
            scalar_gen["total_abs_turn"],
        ),
        (
            "Speed change p95 (kt / point interval)",
            scalar_real["p95_abs_speed_step"],
            scalar_gen["p95_abs_speed_step"],
        ),
    ]
    for axis, (label, reference, generated) in zip(axes.flat, panels, strict=True):
        ecdf(axis, reference, "Held-out real", "#2874a6")
        ecdf(axis, generated, "Generated raw", "#d35400")
        axis.set_xlabel(label)
        axis.legend(fontsize=8)
    fig.suptitle(
        f"{name}: distributions | {len(raw)} generated, {len(real['ids'])} real"
    )
    save_figure(fig, target / "distributions")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, key in zip(
        axes.flat,
        [
            "p95_abs_turn_step",
            "p95_abs_altitude_step",
            "mean_abs_speed_second_difference",
            "time_step_cv",
            "p95_abs_vertical_rate",
            "p95_abs_turn_rate",
        ],
        strict=True,
    ):
        ecdf(axis, scalar_real[key], "Real", "#2874a6")
        ecdf(axis, scalar_gen[key], "Generated", "#d35400")
        axis.set_xlabel(f"{key}\n({UNITS[key]})")
        axis.legend(fontsize=8)
    fig.suptitle(
        f"{name}: temporal diagnostics | rates use strictly increasing raw time"
    )
    save_figure(fig, target / "temporal")


def plot_envelope(
    target: Path, name: str, real: dict, raw: np.ndarray, envelope: dict, scores: dict
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    progress = np.linspace(0, 1, 200)
    for j, (feature, unit) in enumerate(FEATURES.values()):
        axis = axes[0, j]
        axis.fill_between(
            progress,
            envelope["lower"][:, j],
            envelope["upper"][:, j],
            color="0.85",
            label="Train q01-q99",
        )
        for values, label, color in (
            (real["values"], "Test real", "#2874a6"),
            (raw, "Generated raw", "#d35400"),
        ):
            axis.plot(
                progress,
                np.median(values[:, :, j + 1], axis=0),
                label=f"{label} median",
                color=color,
            )
        axis.set_ylabel(f"{feature} ({unit})")
        axis.set_xlabel("Relative time-grid progress")
        axis.legend(fontsize=8)
        for source, color in (("real", "#2874a6"), ("generated", "#d35400")):
            axes[1, j].plot(
                progress,
                scores[source][1][:, :, j].mean(axis=0),
                label=source,
                color=color,
            )
        axes[1, j].set_ylabel("Fraction outside train interval")
        axes[1, j].set_xlabel("Relative time-grid progress")
        axes[1, j].legend(fontsize=8)
    fig.suptitle(f"{name}: train-fitted intervals (statistical, not physical limits)")
    save_figure(fig, target / "envelopes")


def fit_joint_pca(train: np.ndarray, seed: int) -> tuple:
    mean, scale = train.mean(0), train.std(0)
    scale[scale < 1e-8] = 1
    normalized = (train - mean) / scale
    pca = PCA(
        n_components=min(20, len(train) - 1, train.shape[1]),
        svd_solver="randomized",
        random_state=seed,
    ).fit(normalized)
    return mean, scale, pca


def plot_pca(
    target: Path,
    name: str,
    train: np.ndarray,
    real: np.ndarray,
    sources: dict,
    fit: tuple,
    seed: int,
) -> dict:
    mean, scale, pca = fit
    transform = lambda x: pca.transform((x - mean) / scale)  # noqa: E731
    train_pc, real_pc = transform(train), transform(real)
    projected = {key: transform(values) for key, values in sources.items()}
    fig, axis = plt.subplots(figsize=(9, 7))
    indices = np.random.default_rng(seed).choice(
        len(train), min(2500, len(train)), replace=False
    )
    axis.scatter(
        *train_pc[indices, :2].T,
        s=5,
        color="0.6",
        alpha=0.2,
        label=f"Train ({len(train)}; background subsampled)",
    )
    axis.scatter(
        *real_pc[:, :2].T,
        s=18,
        color="#2874a6",
        alpha=0.65,
        label=f"Held-out real ({len(real)})",
    )
    for (key, points), color in zip(
        projected.items(), ["#d35400", "#239b56", "#884ea0"], strict=False
    ):
        axis.scatter(*points[:, :2].T, s=20, color=color, alpha=0.55, label=key)
    ratios = pca.explained_variance_ratio_
    axis.set_xlabel(f"PC1 ({ratios[0]:.1%})")
    axis.set_ylabel(f"PC2 ({ratios[1]:.1%})")
    axis.set_title(f"{name}: one training-fitted PCA | full extent, no omitted points")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    save_figure(fig, target / "pca")
    np.savez_compressed(
        target / "pca_transform.npz",
        mean=mean,
        scale=scale,
        pca_mean=pca.mean_,
        components=pca.components_,
        explained_variance_ratio=ratios,
    )
    records = [
        pd.DataFrame({"source": "train", "pc1": train_pc[:, 0], "pc2": train_pc[:, 1]})
    ]
    for label, points in {"held_out_real": real_pc, **projected}.items():
        records.append(
            pd.DataFrame({"source": label, "pc1": points[:, 0], "pc2": points[:, 1]})
        )
    pd.concat(records).to_csv(target / "pca_coordinates.csv", index=False)
    return {
        "pca_2d_explained_fraction": float(ratios[:2].sum()),
        "pca_20d_explained_fraction": float(ratios.sum()),
    }


def evaluate_model(
    name: str,
    reference_name: str,
    real: dict,
    generated: dict,
    raw: np.ndarray,
    splits: dict | None,
    target: Path,
    seed: int,
    bootstrap: int,
    joint_fit: tuple | None,
) -> dict:
    target.mkdir()
    rng = np.random.default_rng(seed)
    raw = relative_time(raw)
    features_real = trajectory_features(real["values"], real["xy"])
    features_gen = trajectory_features(raw, generated["xy"])
    features_processed = trajectory_features(generated["values"], generated["xy"])
    n = len(raw)
    summary = {
        "model": name,
        "reference": reference_name,
        "n_generated": n,
        "n_real": len(real["ids"]),
        "raw_invalid_time_count": int(
            (np.diff(raw[:, :, 3], axis=1) <= 0).any(1).sum()
        ),
        "interval_status": "train-validation-calibrated" if splits else "unavailable",
        "rate_valid_generated": int(np.isfinite(features_gen["time_step_cv"]).sum()),
    }
    scalar_records, point_records, trajectories = [], [], []
    for source, identifiers, features in (
        ("real", real["ids"], features_real),
        ("generated_raw", generated["ids"], features_gen),
        ("generated_processed", generated["ids"], features_processed),
    ):
        rows = pd.DataFrame(features)
        rows.insert(0, "trajectory_id", identifiers)
        rows.insert(0, "source", source)
        trajectories.append(rows)
    pd.concat(trajectories).to_csv(target / "trajectory_features.csv", index=False)
    for key, reference in features_real.items():
        for source, features in (
            ("generated_raw", features_gen),
            ("generated_processed", features_processed),
        ):
            record = compare_1d(
                reference,
                features[key],
                rng,
                bootstrap if source == "generated_raw" else 0,
            )
            scalar_records.append(
                {"metric": key, "unit": UNITS[key], "source": source, **record}
            )
            if source == "generated_raw" and "wasserstein" in record:
                summary[f"w1_{key}"] = record["wasserstein"]
    for i, (key, unit) in FEATURES.items():
        for source, values in (
            ("generated_raw", raw),
            ("generated_processed", generated["values"]),
        ):
            record = compare_1d(real["values"][:, :, i], values[:, :, i], rng)
            point_records.append(
                {
                    "metric": key,
                    "unit": unit,
                    "source": source,
                    "sampling": "equal weight per trajectory and point",
                    **record,
                }
            )
            if source == "generated_raw":
                summary[f"w1_point_{key}"] = record["wasserstein"]
    for key, reference, sample, unit in (
        (
            "abs_turn_step",
            np.abs(circular_delta(real["values"][:, :, 0])),
            np.abs(circular_delta(raw[:, :, 0])),
            "degree/step",
        ),
        (
            "abs_speed_step",
            np.abs(np.diff(real["values"][:, :, 1], axis=1)),
            np.abs(np.diff(raw[:, :, 1], axis=1)),
            "kt/step",
        ),
        (
            "abs_altitude_step",
            np.abs(np.diff(real["values"][:, :, 2], axis=1)),
            np.abs(np.diff(raw[:, :, 2], axis=1)),
            "ft/step",
        ),
    ):
        point_records.append(
            {
                "metric": key,
                "unit": unit,
                "source": "generated_raw",
                **compare_1d(reference, sample, rng),
            }
        )
    pd.DataFrame(scalar_records).to_csv(
        target / "scalar_distributions.csv", index=False
    )
    pd.DataFrame(point_records).to_csv(target / "point_distributions.csv", index=False)
    corrections = generated["values"] - raw
    corrections[:, :, 0] = (corrections[:, :, 0] + 180) % 360 - 180
    summary["speed_corrected_count"] = int(
        (np.abs(corrections[:, :, 1]).max(1) > 1e-3).sum()
    )
    summary["altitude_corrected_count"] = int(
        (np.abs(corrections[:, :, 2]).max(1) > 1e-3).sum()
    )
    summary["max_time_correction_s"] = float(np.abs(corrections[:, :, 3]).max())
    if splits:
        envelope = calibrated_envelope(
            splits["train"]["values"], splits["validation"]["values"]
        )
        scores = {}
        outlier_rows = []
        for label, values, identifiers in (
            ("real", real["values"], real["ids"]),
            ("generated", raw, generated["ids"]),
            ("generated_processed", generated["values"], generated["ids"]),
        ):
            score, outside = envelope_scores(
                values, envelope["lower"], envelope["upper"]
            )
            scores[label] = score, outside
            events = score > envelope["threshold"]
            summary[f"{label}_point_outside_fraction"] = float(score.mean())
            summary[f"{label}_trajectory_outlier_fraction"] = float(events.mean())
            summary[f"{label}_trajectory_outlier_count"] = int(events.sum())
            low, high = wilson_interval(events)
            summary[f"{label}_outlier_wilson_low"] = low
            summary[f"{label}_outlier_wilson_high"] = high
            low, high = mean_ci(score, rng, bootstrap)
            summary[f"{label}_point_fraction_ci_low"] = low
            summary[f"{label}_point_fraction_ci_high"] = high
            rows = pd.DataFrame(
                {
                    "source": label,
                    "trajectory_id": identifiers,
                    "outside_fraction": score,
                    "is_outlier": events,
                }
            )
            for j, (feature, _) in enumerate(FEATURES.values()):
                rows[f"{feature}_outside_fraction"] = outside[:, :, j].mean(1)
            outlier_rows.append(rows)
        summary["validation_outlier_score_threshold"] = envelope["threshold"]
        summary["validation_outlier_fraction"] = float(
            (envelope["validation_scores"] > envelope["threshold"]).mean()
        )
        summary["train_count"], summary["validation_count"] = (
            len(splits[split]["ids"]) for split in ("train", "validation")
        )
        pd.concat(outlier_rows).to_csv(target / "outliers.csv", index=False)
        np.savez_compressed(
            target / "intervals.npz",
            lower=envelope["lower"],
            upper=envelope["upper"],
            threshold=envelope["threshold"],
            validation_scores=envelope["validation_scores"],
        )
        plot_envelope(target, name, real, raw, envelope, scores)
        train = joint_representation(splits["train"]["values"], fixed_scale=False)
        summary.update(
            plot_pca(
                target,
                name,
                train,
                joint_representation(real["values"], fixed_scale=False),
                {"Generated raw": joint_representation(raw, fixed_scale=False)},
                joint_fit,
                seed,
            )
        )
    joint_real = joint_representation(real["values"], fixed_scale=True)
    joint_gen = joint_representation(raw, fixed_scale=True)
    diagnostics, to_gen, to_real, baseline = distance_diagnostics(joint_real, joint_gen)
    summary.update(diagnostics)
    pd.DataFrame(
        {
            "real_id": real["ids"],
            "real_to_generated_nn": to_gen,
            "real_leave_one_out_nn": baseline,
        }
    ).to_csv(target / "coverage.csv", index=False)
    pd.DataFrame(
        {"generated_id": generated["ids"], "generated_to_real_nn": to_real}
    ).to_csv(target / "similarity.csv", index=False)
    plot_distributions(target, name, real, raw, features_real, features_gen)
    fig, axis = plt.subplots(figsize=(8, 5))
    for values, label, color in (
        (to_gen, "Real -> generated", "#2874a6"),
        (to_real, "Generated -> real", "#d35400"),
        (baseline, "Real -> other real", "#239b56"),
    ):
        ecdf(axis, values, label, color)
    axis.set_xlabel("Nearest trajectory RMS distance (fixed physical scaling)")
    axis.set_title(f"{name}: similarity and coverage distances")
    axis.legend()
    save_figure(fig, target / "coverage")
    write_json(target / "summary.json", summary)
    print(
        f"Finished {name}: invalid raw time {summary['raw_invalid_time_count']}/{n}",
        flush=True,
    )
    return summary


def sliced_wasserstein(
    real: np.ndarray, generated: np.ndarray, directions: np.ndarray
) -> float:
    real_projected, generated_projected = real @ directions, generated @ directions
    return float(
        np.mean(
            [
                wasserstein_distance(real_projected[:, i], generated_projected[:, i])
                for i in range(directions.shape[1])
            ]
        )
    )


def evaluate_latents(root: Path, output: Path, seed: int, smoke: bool) -> list:
    records = []
    base = root / "runs/latent_denoising_video-v1-seed-42/inputs"
    k256 = root / (
        "runs/zurich_rwy14_flow162_216_tcvae_vamp_k256-v1-seed-42/"
        "latent_analysis/pca-inputs-n500-seed42/latents.npz"
    )
    k1000 = base / "ddpm/artifacts/latent_dataset.npz"
    samples = (
        base
        / "reference/three_latent_sources_n-500_steps-1000_seed-42/latent_samples.npz"
    )
    for name, source in (("k256", k256), ("k1000", k1000)):
        if not source.exists() or (name == "k1000" and not samples.exists()):
            continue
        with np.load(source, allow_pickle=False) as arrays:
            data = {key: arrays[key].copy() for key in arrays.files}
        count = 8 if smoke else 500
        rng = np.random.default_rng(seed)
        if name == "k256":
            train = data["train_mu"] + np.exp(
                0.5 * data["train_log_variance"]
            ) * rng.standard_normal(data["train_mu"].shape)
            real, sources = (
                data["test_sample"][:count],
                {"VampPrior": data["vamp_sample"][:count]},
            )
            validation = None
            metadata = {
                "target": "posterior_sample",
                "train_draw_seed": seed,
                "note": (
                    "Auxiliary seed-44 prior draws; "
                    "not the MiniSky seed-42 first-100 batch."
                ),
            }
        else:
            train, validation = data["train_latent"], data["validation_latent"]
            with np.load(samples, allow_pickle=False) as arrays:
                real = arrays["posterior"][:count].copy()
                sources = {
                    "VampPrior": arrays["vamp"][:count].copy(),
                    "Latent DDPM": arrays["ddpm"][:count].copy(),
                }
                indices = {
                    identifier: i
                    for i, identifier in enumerate(data["test_trajectory_id"])
                }
                expected = data["test_latent"][
                    [indices[i] for i in arrays["test_trajectory_id"][:count]]
                ]
                np.testing.assert_allclose(real, expected, atol=1e-5, rtol=1e-5)
            metadata = {
                "target": "posterior_sample",
                "note": "Existing 500-draw sources; no new model inference.",
            }
        target = output / "latent" / name
        target.mkdir(parents=True)
        fit = fit_joint_pca(train, seed)
        pca_info = plot_pca(target, f"{name} latent", train, real, sources, fit, seed)
        mean, scale, _ = fit
        train_std, real_std = (train - mean) / scale, (real - mean) / scale
        directions = rng.standard_normal((train.shape[1], 128))
        directions /= np.linalg.norm(directions, axis=0, keepdims=True)
        nearest_real = cdist(real_std, train_std).min(axis=1)
        real_pair_median = np.median(pdist(real_std))
        if validation is not None:
            # Same training bank for calibration and every evaluated population.
            validation_nn = cdist((validation - mean) / scale, train_std).min(axis=1)
            threshold = float(np.quantile(validation_nn, 0.95, method="higher"))
            np.savez_compressed(
                target / "calibration.npz",
                validation_nn=validation_nn,
                threshold=threshold,
            )
        else:
            threshold = None
        latent_rows = []
        for label, values in {"Test posterior": real, **sources}.items():
            standardized = (values - mean) / scale
            nearest = cdist(standardized, train_std).min(axis=1)
            record = {
                "latent_model": name,
                "source": label,
                "count": len(values),
                "nearest_train_draw_median": float(np.median(nearest)),
                "nearest_train_draw_p95": float(np.quantile(nearest, 0.95)),
                "nearest_distance_ratio_to_test": float(
                    np.median(nearest) / np.median(nearest_real)
                ),
                "diversity_ratio_to_test": float(
                    np.median(pdist(standardized)) / real_pair_median
                ),
                "sliced_wasserstein_to_test": sliced_wasserstein(
                    real_std, standardized, directions
                ),
                "nn_outlier_threshold": threshold,
                "nn_outlier_fraction": float((nearest > threshold).mean())
                if threshold is not None
                else None,
                **pca_info,
                **metadata,
            }
            if threshold is not None:
                events = nearest > threshold
                record["nn_outlier_count"] = int(events.sum())
                low, high = wilson_interval(events)
                record["nn_outlier_wilson_low"] = low
                record["nn_outlier_wilson_high"] = high
            records.append(record)
            latent_rows.append(
                pd.DataFrame(
                    {
                        "source": label,
                        "sample_index": np.arange(len(values)),
                        "nearest_train_draw": nearest,
                        "standardized_norm": np.linalg.norm(standardized, axis=1),
                    }
                )
            )
        pd.concat(latent_rows).to_csv(target / "distances.csv", index=False)
        write_json(
            target / "provenance.json",
            {
                "input": str(source),
                "sha256": digest(source),
                "samples_sha256": digest(samples) if name == "k1000" else None,
                "standardization": "training posterior draws only",
                "directions": 128,
                "seed": seed,
                **metadata,
            },
        )
        print(f"Finished auxiliary {name} latent diagnostics", flush=True)
    pd.DataFrame(records).to_csv(output / "latent_summary.csv", index=False)
    return records


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return "—"
        return f"{value:.3g}" if isinstance(value, float) else str(value)

    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_report(output: Path, summaries: list, latent: list, bootstrap: int) -> None:
    frame = pd.DataFrame(summaries)
    frame.to_csv(output / "summary.csv", index=False)
    table = frame[
        [
            "model",
            "raw_invalid_time_count",
            "generated_point_outside_fraction",
            "real_point_outside_fraction",
            "generated_trajectory_outlier_count",
            "real_trajectory_outlier_count",
            "coverage_distance_ratio",
        ]
    ].copy()
    table.columns = [
        "模型",
        "原始时间异常数",
        "生成点级越界比例",
        "真实点级越界比例",
        "生成轨迹异常数",
        "真实轨迹异常数",
        "覆盖距离比",
    ]
    report = [
        "# 生成分布评估\n",
        "本次在本机使用 MiniSky inputs 的固定样本："
        "每组100条、每条200点（smoke运行除外）。"
        "未重新训练、生成轨迹、筛选质量或修复原始输出。"
        "六个模型严格对应各自真实测试范围。\n",
        markdown_table(table),
        "\n## 指标与解释\n",
        "- 点级越界：训练集每个时间进度上的地速、高度、相对累计时间 q01–q99；"
        "分母为轨迹数 × 200 × 3，不包含航迹角。"
        "圆周角用 sine/cosine 进入联合空间，转角用最短圆周角差。",
        "- 轨迹异常：每条轨迹的点级越界比例严格超过验证集该分数的95%分位数（higher）。"
        "这是统计异常阈值，不能解释为不可飞率。真实测试组使用同一阈值。",
        "- coverage_distance_ratio：真实→生成的最近邻距离中位数 / "
        "真实→其他真实的最近邻距离中位数。"
        "数值偏大提示覆盖不足；不是标准 precision/recall 指标，也不是覆盖百分比。",
        "- 最近邻／多样性距离使用完整200点表示 "
        "[sin(track), cos(track), groundspeed/100kt, "
        "altitude/10000ft, elapsed/1000s] 的 RMS 欧氏距离。尺度是预先固定的研究选择；"
        "统计只在同一机场、跑道／flow范围内比较。时间均减去每条轨迹自己的首点。",
        "- Wasserstein 距离按原单位报告，越小表示该项经验分布越接近；"
        "不同特征和不同真实范围之间不能直接比较数值大小。",
        "- ECDF与点级KS只作描述；200个点不是200个独立样本，不报告朴素点级KS的p值。"
        f"轨迹标量的Wasserstein区间使用{bootstrap}次按轨迹重采样；"
        "点级比例的区间也按轨迹重采样。"
        "Wasserstein采用bootstrap百分位经验区间，可能受正偏差影响，"
        "不能直接据此做显著性或等价性判断。"
        "异常率另给Wilson区间，这些区间均未包含阈值校准本身的不确定性。",
        "- 相邻点变化以 degree/step、kt/step、ft/step 计；"
        "每秒／每分钟变化率只在整条原始时间"
        "严格递增时计算，表中记录可计算数量。不能把200点当作1秒采样。",
        "- 原始／处理后分布分别保存；路径长度只来自历史导出的投影x/y米坐标，"
        "包含历史积分／时间修正影响。没有用经纬度做笛卡尔距离。",
        "\n## 数据约束\n",
        "- 全机场组所需train/validation/test、normalization与metadata已从"
        "desktop-hqosbo1复制回data/processed/zurich_ddpm_unconditional-v1；"
        "本地与远端SHA256一致。六组均使用对应范围的训练／验证数据。",
        "- 单flow TCVAE K256 与 DDPM 使用相同训练／验证／测试 artifact；"
        "脚本核对K256训练数据的已保存哈希。RWY14 allflow使用已复制回项目的K1000数据；"
        "天津使用noaib版本。MiniSky参考ID被验证属于相应test，"
        "且与反归一化测试特征吻合。",
        "- 每组100条是探索性评估；正常罕见真实轨迹也可能越界。"
        "零越界或图中重叠不能证明物理真实性。",
        "- 真实200点由等时间重采样得到，时间步长CV接近零是准备方式的结果。"
        "生成时间步长CV偏大首先反映对该表示方式的偏离，不能单独当作不可飞证据。",
        "\n## 各模型图表\n",
    ]
    for name in frame.model:
        report += [
            f"### {name}\n",
            f"![分布]({name}/distributions.png)\n",
            f"![时序]({name}/temporal.png)\n",
            f"![覆盖]({name}/coverage.png)\n",
        ]
        if (output / name / "pca.png").exists():
            report += [
                f"![训练分位区间]({name}/envelopes.png)\n",
                f"![共同PCA]({name}/pca.png)\n",
            ]
    report += [
        "## 潜空间辅助诊断\n",
        "使用保存的潜变量／后验参数，比较真实后验采样、VampPrior及Latent DDPM。"
        "训练集后验采样决定标准化和PCA；距离在完整64维计算。"
        "K1000 DDPM训练目标是posterior_sample。"
        "K256的辅助先验批次seed=44，和MiniSky的seed=42前100条不同，不能逐条关联。"
        "K256未保存验证后验，因此不报告潜空间校准异常率。\n",
    ]
    if latent:
        report += [
            markdown_table(
                pd.DataFrame(latent)[
                    [
                        "latent_model",
                        "source",
                        "count",
                        "nearest_train_draw_median",
                        "sliced_wasserstein_to_test",
                        "nn_outlier_fraction",
                        "pca_2d_explained_fraction",
                    ]
                ]
            )
        ]
        for name in dict.fromkeys(row["latent_model"] for row in latent):
            report += [f"\n![潜空间 {name}](latent/{name}/pca.png)\n"]
    report += [
        "\n## 文件与复现\n",
        "summary.csv 是模型总表；各模型目录包含逐轨迹特征、原始／处理后分布、"
        "越界清单、阈值、PCA变换、最近邻距离及PNG/SVG。manifest.json保存输入哈希、环境和参数。"
        "目录内保留实际执行的脚本；重复运行须选择新的输出目录，避免覆盖结果。\n",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf8")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, key, title in zip(
        axes.flat,
        [
            "w1_point_groundspeed",
            "w1_point_altitude",
            "w1_total_abs_turn",
            "coverage_distance_ratio",
        ],
        [
            "Groundspeed W1 (kt)",
            "Altitude W1 (ft MSL)",
            "Total turn W1 (degree)",
            "Coverage distance ratio (within each scope)",
        ],
        strict=True,
    ):
        axis.barh(frame.model, frame[key], color="#2874a6")
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=0.2)
    count = int(frame.n_generated.iloc[0])
    fig.suptitle(
        f"Fixed {count} generated / {count} held-out real per scope | "
        "no cross-scope ranking"
    )
    save_figure(fig, output / "overview")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output, inputs = args.output.resolve(), args.inputs.resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing results are preserved")
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    count, bootstrap = (8, 10) if args.smoke else (100, args.bootstrap)
    paths = [inputs / "README.md", Path(__file__)]
    real_groups, prepared, joint_fits, checks = {}, {}, {}, {}
    with threadpool_limits(limits=4):
        for reference, artifact in ARTIFACTS.items():
            path = inputs / "real" / reference / "reference_200.parquet"
            paths += [path, path.parent / "selected_ids.csv"]
            real_groups[reference] = load_table(path, count)
            if artifact:
                artifact_path = root / artifact
                prepared[reference] = load_prepared(artifact_path)
                checks[reference] = validate_reference(
                    real_groups[reference], prepared[reference]
                )
                paths += [
                    artifact_path / f"{split}.npz"
                    for split in ("train", "validation", "test")
                ]
                paths.append(artifact_path / "normalization.json")
                train = joint_representation(
                    prepared[reference]["train"]["values"], fixed_scale=False
                )
                joint_fits[reference] = fit_joint_pca(train, args.seed)
        # These hashes establish that the DDPM single-flow artifact also fits K256.
        manifest_path = root / (
            "runs/zurich_rwy14_flow162_216_tcvae_vamp_k256-v1-seed-42/"
            "latent_analysis/pca-inputs-n500-seed42/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text("utf8"))
        single = root / ARTIFACTS["zurich-singleflow"]
        for filename, expected in manifest["resolved_config"]["artifact"][
            "files"
        ].items():
            if (single / filename).exists() and digest(single / filename) != expected:
                raise ValueError(f"K256/shared artifact hash mismatch: {filename}")
        copied = (
            root
            / "runs/latent_denoising_video-v1-seed-42/inputs/ddpm/artifacts"
            / "latent_manifest.json"
        )
        copied_manifest = json.loads(copied.read_text("utf8"))
        allflow = root / ARTIFACTS["zurich-rwy14-allflow"]
        for filename, expected in copied_manifest["prepared_artifact"]["files"].items():
            if digest(allflow / filename) != expected:
                raise ValueError(f"K1000 copied artifact hash mismatch: {filename}")
        paths += [manifest_path, copied]
        transfer = (
            root / "data/processed/zurich_ddpm_unconditional-v1/import_manifest.json"
        )
        transfer_data = json.loads(transfer.read_text("utf8"))
        for filename, expected in transfer_data["sha256"].items():
            if digest(transfer.parent / filename) != expected:
                raise ValueError(f"Global copied artifact hash mismatch: {filename}")
        paths.append(transfer)
        summaries = []
        for name, reference in MODEL_REFERENCES.items():
            directory = inputs / "generated" / name
            path = directory / "generated.parquet"
            generated = load_table(path, count)
            raw_path = directory / "raw_model_values.npz"
            with np.load(raw_path, allow_pickle=False) as data:
                raw_ids = data["ids"].astype(str)
                indices = {identifier: i for i, identifier in enumerate(raw_ids)}
                raw = data["values"][[indices[i] for i in generated["ids"]]].copy()
            if raw.shape != (count, 200, 4) or not np.isfinite(raw).all():
                raise ValueError(f"Invalid raw tensor: {raw_path}")
            paths += [path, raw_path, directory / "raw_checks.csv"]
            summaries.append(
                evaluate_model(
                    name,
                    reference,
                    real_groups[reference],
                    generated,
                    raw,
                    prepared.get(reference),
                    output / name,
                    args.seed,
                    bootstrap,
                    joint_fits.get(reference),
                )
            )
        latent = evaluate_latents(root, output, args.seed, args.smoke)
        write_report(output, summaries, latent, bootstrap)
    write_json(
        output / "manifest.json",
        {
            "execution": (
                "local CPU; existing project artifacts only; "
                "no training or model inference"
            ),
            "seed": args.seed,
            "count_per_scope": count,
            "bootstrap": bootstrap,
            "raw_channels": [
                "track degree",
                "groundspeed kt",
                "altitude ft MSL",
                "elapsed s",
            ],
            "interval_quantiles": [0.01, 0.99],
            "validation_quantile": 0.95,
            "rate_policy": "whole trajectory raw time must be strictly increasing",
            "references": MODEL_REFERENCES,
            "reference_checks": checks,
            "missing": {},
            "input_sha256": {
                str(path.relative_to(root)): digest(path)
                for path in dict.fromkeys(paths)
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                **{
                    package: importlib.metadata.version(package)
                    for package in (
                        "numpy",
                        "pandas",
                        "scipy",
                        "matplotlib",
                        "scikit-learn",
                        "pyarrow",
                    )
                },
            },
            "summaries": summaries,
        },
    )
    print(f"OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
