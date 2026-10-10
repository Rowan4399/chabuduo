"""Shared loss and trajectory plotting for completed runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData
from chabuduo.config import ExperimentConfig
from chabuduo.prepare import airport_projection
from chabuduo.sampling import build_generated_traffic


def _metric_series(metrics: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in metrics:
            values = metrics.dropna(subset=[name])
            if not values.empty:
                return values.groupby("epoch", sort=True)[name].last()
    return pd.Series(dtype=float)


def load_loss_series(run: Path) -> tuple[pd.Series, pd.Series]:
    """Load the common train/validation loss view for DDPM or Lightning."""
    direct = run / "logs" / "metrics.csv"
    if direct.is_file():
        metrics = pd.read_csv(direct)
    else:
        files = sorted((run / "logs" / "training").glob("version_*/metrics.csv"))
        if not files:
            raise FileNotFoundError(f"No training metrics found under {run / 'logs'}")
        metrics = pd.concat(
            (pd.read_csv(path) for path in files),
            ignore_index=True,
        )
    if "epoch" not in metrics:
        raise ValueError("Training metrics do not contain an epoch column")
    train = _metric_series(metrics, ("train_loss", "train/loss_epoch"))
    validation = _metric_series(
        metrics,
        ("validation_loss", "validation/checkpoint_loss"),
    )
    if train.empty or validation.empty:
        raise ValueError("Training or validation loss is missing")
    return train, validation


def _resume_epochs(run: Path) -> list[int]:
    summary_path = run / "run_summary.json"
    if not summary_path.is_file():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    resumes = summary.get("resumes", [])
    if not resumes and "resume" in summary:
        resumes = [summary["resume"]]
    return [int(item["source_epoch"]) for item in resumes]


def plot_loss_curves(
    run: Path,
    *,
    output: Path | None = None,
    warmup_epochs: int = 20,
) -> Path:
    """Plot full loss history beside the post-warmup interval."""
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be nonnegative")
    train, validation = load_loss_series(run)
    output = output or run / "plots" / "loss.png"
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for axis, start, title in (
        (axes[0], 0, "Complete training history"),
        (axes[1], warmup_epochs, f"After warm-up (epoch {warmup_epochs}+)"),
    ):
        visible_train = train.loc[train.index >= start]
        visible_validation = validation.loc[validation.index >= start]
        if visible_train.empty or visible_validation.empty:
            visible_train, visible_validation = train, validation
        axis.plot(
            visible_train.index,
            visible_train.values,
            linewidth=1.4,
            label="Training",
        )
        axis.plot(
            visible_validation.index,
            visible_validation.values,
            linewidth=1.4,
            label="Validation",
        )
        best_epoch = int(visible_validation.idxmin())
        axis.scatter(
            [best_epoch],
            [float(visible_validation.loc[best_epoch])],
            color="#E45756",
            s=28,
            zorder=3,
            label=f"Best validation: epoch {best_epoch}",
        )
        for resume_epoch in _resume_epochs(run):
            axis.axvline(resume_epoch, color="0.45", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    figure.suptitle(run.name)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _trajectory_groups(frame: pd.DataFrame, count: int) -> list[pd.DataFrame]:
    identifier = "flight_id" if "flight_id" in frame else "icao24"
    if identifier not in frame:
        raise ValueError("Trajectory table needs flight_id or icao24")
    order = "timestamp" if "timestamp" in frame else "timedelta"
    groups: list[pd.DataFrame] = []
    for _, group in frame.groupby(identifier, sort=False):
        groups.append(group.sort_values(order) if order in group else group)
        if len(groups) == count:
            break
    return groups


def _draw_trajectories(
    axis: plt.Axes,
    groups: list[pd.DataFrame],
    *,
    color: str,
    title: str,
) -> None:
    for group in groups:
        axis.plot(
            group.x / 1000,
            group.y / 1000,
            color=color,
            alpha=0.22,
            linewidth=0.75,
        )
    axis.set_title(f"{title} ({len(groups)} trajectories)")
    axis.set_xlabel("East of airport (km)")
    axis.grid(alpha=0.2)


def _anchor_xy(config: ExperimentConfig) -> tuple[float, float, str]:
    if config.prepare.representation.anchor != "runway_threshold":
        return 0.0, 0.0, config.preprocess.airport
    runways = config.preprocess.runways
    if runways is None or len(runways) != 1:
        raise ValueError("runway_threshold anchor requires exactly one runway")
    airport = airports[config.preprocess.airport]
    runway_name = runways[0]
    runway = airport.runways[runway_name]
    threshold = next(
        point for point in runway.tuple_runway if point.name == runway_name
    )
    projection = airport_projection(config.preprocess.airport)
    x, y = projection(threshold.longitude, threshold.latitude)
    return x / 1000, y / 1000, f"RWY {runway_name} threshold"


def plot_samples(
    sample: Path,
    artifact: Path,
    config: ExperimentConfig,
    *,
    output: Path | None = None,
    count: int = 100,
) -> Path:
    """Write a held-out/generated comparison using identical map limits."""
    if count <= 0:
        raise ValueError("count must be positive")
    generated_path = sample / "generated.parquet"
    if not generated_path.is_file():
        raise FileNotFoundError(generated_path)
    generated = pd.read_parquet(generated_path)
    if not {"x", "y"} <= set(generated.columns):
        raise ValueError("Generated Parquet must contain projected x/y columns")

    test = SplitData.load(artifact / "test.npz")
    normalization = Normalization.load(artifact / "normalization.json")
    real_values = normalization.inverse_transform(test.values)
    real = build_generated_traffic(real_values, test.elapsed_time, config)
    if not {"x", "y"} <= set(real.data.columns):
        real = real.compute_xy(projection=airport_projection(config.preprocess.airport))

    real_groups = _trajectory_groups(real.data, count)
    generated_groups = _trajectory_groups(generated, count)
    if not real_groups or not generated_groups:
        raise ValueError("Both held-out and generated trajectories are required")
    figure, axes = plt.subplots(1, 2, figsize=(15, 7.2), sharex=True, sharey=True)
    _draw_trajectories(
        axes[0], real_groups, color="#4C78A8", title="Held-out trajectories"
    )
    _draw_trajectories(
        axes[1], generated_groups, color="#E45756", title="Generated trajectories"
    )

    all_x = np.concatenate(
        [group.x.to_numpy() / 1000 for group in real_groups + generated_groups]
    )
    all_y = np.concatenate(
        [group.y.to_numpy() / 1000 for group in real_groups + generated_groups]
    )
    x_margin = max(0.5, 0.04 * float(np.ptp(all_x)))
    y_margin = max(0.5, 0.04 * float(np.ptp(all_y)))
    anchor_x, anchor_y, anchor_label = _anchor_xy(config)
    for axis in axes:
        axis.scatter(
            [anchor_x],
            [anchor_y],
            marker="x",
            color="black",
            s=45,
            label=anchor_label,
            zorder=3,
        )
        axis.set_xlim(float(all_x.min()) - x_margin, float(all_x.max()) + x_margin)
        axis.set_ylim(float(all_y.min()) - y_margin, float(all_y.max()) + y_margin)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best")
    axes[0].set_ylabel("North of airport (km)")
    figure.suptitle(config.name)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = output or sample / "trajectory_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output
