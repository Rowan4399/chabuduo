"""Replot a frozen distribution evaluation without recomputing its metrics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.ticker import MaxNLocator

COLORS = ["#2874A6", "#D35400", "#239B56"]
MODEL_LABELS = {
    "ddpm-flow700": "DDPM700 · single flow",
    "tcvae-flow-k256": "TCVAE K256 · single flow",
    "tcvae-k1000": "TCVAE K1000 · RWY14",
    "latent-k1000": "Latent DDPM · RWY14",
    "ddpm-global700": "DDPM700 · all runways",
    "tianjin-noaib-tcvae-k1000": "TCVAE K1000 · Tianjin",
}
SCOPE_LABELS = {
    "zurich-singleflow": "LSZH RWY14 / flow162-216",
    "zurich-rwy14-allflow": "LSZH RWY14 / all flows",
    "zurich-all-airport": "LSZH / all runways",
    "tianjin-noaib-rwy16l": "ZBTJ RWY16L / no AIB",
}
METRIC_LABELS = {
    "duration": "Duration [s]",
    "path_length": "Exported path length [km]",
    "total_abs_turn": "Cumulative absolute turn [deg]",
    "p95_abs_turn_step": "Turn increment p95 [deg/step]",
    "mean_abs_speed_second_difference": "Speed roughness [kt/step²]",
    "time_step_cv": "Time-step CV [dimensionless]",
    "p95_abs_vertical_rate": "Absolute vertical rate p95 [ft/min]",
    "mean_speed": "Mean groundspeed [kt]",
}
SCALAR_KEYS = [
    "duration",
    "total_abs_turn",
    "p95_abs_turn_step",
    "mean_abs_speed_second_difference",
    "time_step_cv",
    "p95_abs_vertical_rate",
]
QUANTILE_KEYS = [
    "duration",
    "path_length",
    "total_abs_turn",
    "mean_abs_speed_second_difference",
    "time_step_cv",
    "p95_abs_vertical_rate",
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "utf8"
    )


def save_figure(fig, target: Path) -> None:
    fig.savefig(target.with_suffix(".png"), dpi=170)
    fig.savefig(target.with_suffix(".svg"))
    plt.close(fig)


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def common_edges(populations: list, bins: int = 30) -> np.ndarray:
    values = np.concatenate([finite(population) for population in populations])
    low, high = values.min(), values.max()
    if high == low:
        low, high = low - 0.5, high + 0.5
    return np.linspace(low, high, bins + 1)


def boxes_with_points(axis, populations: dict, rng: np.random.Generator) -> None:
    samples = [finite(values) for values in populations.values()]
    positions = np.arange(len(samples))
    result = axis.boxplot(
        samples,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        widths=0.48,
    )
    for j, (sample, box) in enumerate(zip(samples, result["boxes"], strict=True)):
        color = COLORS[j % len(COLORS)]
        box.set(facecolor=color, alpha=0.22, edgecolor=color)
        axis.scatter(
            j + rng.uniform(-0.17, 0.17, len(sample)),
            sample,
            s=13 if len(sample) <= 100 else 7,
            alpha=0.5,
            color=color,
            edgecolors="none",
            zorder=3,
        )
    for median in result["medians"]:
        median.set(color="black", linewidth=1.5)
    axis.set_xticks(
        positions,
        [
            f"{label}\nn={len(sample)}; median={np.median(sample):.3g}"
            for label, sample in zip(populations, samples, strict=True)
        ],
    )
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="y", alpha=0.2)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.set_xlim(-0.5, len(samples) - 0.5)


def load_model(
    evaluation: Path, inputs: Path, name: str, reference: str, helpers
) -> dict:
    real = helpers.load_table(
        inputs / "real" / reference / "reference_200.parquet", 100
    )
    generated = helpers.load_table(
        inputs / "generated" / name / "generated.parquet", 100
    )
    with np.load(inputs / "generated" / name / "raw_model_values.npz") as arrays:
        index = {
            identifier: i for i, identifier in enumerate(arrays["ids"].astype(str))
        }
        values = arrays["values"][[index[i] for i in generated["ids"]]]
    generated["values"] = helpers.relative_time(values)
    scalars = pd.read_csv(evaluation / name / "trajectory_features.csv")
    outliers = pd.read_csv(evaluation / name / "outliers.csv")
    with np.load(evaluation / name / "intervals.npz") as arrays:
        intervals = {key: arrays[key].copy() for key in arrays.files}
    return {
        "real": real,
        "generated": generated,
        "scalars": scalars,
        "outliers": outliers,
        "intervals": intervals,
    }


def scalar_population(data: dict, key: str, source: str) -> np.ndarray:
    return data["scalars"].loc[data["scalars"].source.eq(source), key].to_numpy()


def plot_scope_boxes(
    output: Path, reference: str, names: list, datasets: dict, seed: int
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
    rng = np.random.default_rng(seed)
    for axis, key in zip(axes.flat, SCALAR_KEYS, strict=True):
        populations = {"Real": scalar_population(datasets[names[0]], key, "real")}
        populations.update(
            {
                MODEL_LABELS[name].split(" · ")[0]: scalar_population(
                    datasets[name], key, "generated_raw"
                )
                for name in names
            }
        )
        boxes_with_points(axis, populations, rng)
        axis.set_ylabel(METRIC_LABELS[key])
    fig.suptitle(
        f"{SCOPE_LABELS[reference]}: trajectory summaries\n"
        "Box = median / IQR / 1.5-IQR whiskers; every finite trajectory is shown",
        fontsize=13,
    )
    save_figure(fig, output / "scopes" / f"{reference}-scalar_summary")


def plot_profiles(target: Path, name: str, data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    progress = np.linspace(0, 1, 200)
    for j, (channel, label) in enumerate(
        (
            (1, "Groundspeed [kt]"),
            (2, "Altitude [ft MSL]"),
            (3, "Relative elapsed time [s]"),
        )
    ):
        axis = axes[j]
        axis.fill_between(
            progress,
            data["intervals"]["lower"][:, j],
            data["intervals"]["upper"][:, j],
            color="0.85",
            alpha=0.6,
            label="Frozen train q01-q99",
        )
        for source, color, title in (
            ("real", COLORS[0], "Real"),
            ("generated", COLORS[1], "Generated raw"),
        ):
            q05, median, q95 = np.quantile(
                data[source]["values"][:, :, channel], [0.05, 0.5, 0.95], axis=0
            )
            axis.fill_between(progress, q05, q95, color=color, alpha=0.16)
            axis.plot(
                progress,
                median,
                color=color,
                linewidth=1.8,
                label=f"{title}: median + q05-q95",
            )
        axis.set_xlabel("Prepared point progress (0=start, 1=end)")
        axis.set_xlim(0, 1)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.legend(fontsize=7.5)
    fig.suptitle(
        f"{MODEL_LABELS[name]}: phase-dependent profiles\n100 real / 100 raw generated",
        fontsize=13,
    )
    save_figure(fig, target / "profiles")


def plot_heatmap(target: Path, name: str, data: dict) -> dict:
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 8),
        layout="constrained",
        gridspec_kw={"width_ratios": [1, 1, 1, 0.12]},
    )
    palette = ListedColormap(["#2874A6", "#FFFFFF", "#D35400"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], palette.N)
    rows, invalid_counts = [], {}
    for row_index, (source, stored_source, title) in enumerate(
        (("real", "real", "Held-out real"), ("generated", "generated", "Generated raw"))
    ):
        values, identifiers = data[source]["values"], data[source]["ids"]
        scores = data["outliers"].loc[data["outliers"].source.eq(stored_source)]
        scores = scores.set_index("trajectory_id").loc[identifiers]
        signed = np.where(
            values[:, :, 1:4] < data["intervals"]["lower"],
            -1,
            np.where(values[:, :, 1:4] > data["intervals"]["upper"], 1, 0),
        )
        np.testing.assert_allclose(
            (signed != 0).mean(axis=(1, 2)), scores.outside_fraction, atol=1e-14
        )
        order = np.argsort(-scores.outside_fraction.to_numpy(), kind="stable")
        invalid_time = (np.diff(values[:, :, 3], axis=1) <= 0).any(axis=1)
        invalid_counts[source] = int(invalid_time.sum())
        for j, feature in enumerate(("Groundspeed", "Altitude", "Elapsed time")):
            artist = axes[row_index, j].imshow(
                signed[order, :, j],
                cmap=palette,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
                extent=(0, 1, len(order) + 0.5, 0.5),
            )
            axes[row_index, j].set_title(f"{title}: {feature}", fontsize=10)
            axes[row_index, j].set_xlabel("Prepared point progress")
            axes[row_index, j].xaxis.set_major_locator(MaxNLocator(nbins=4))
            if j == 0:
                axes[row_index, j].set_ylabel(
                    "Trajectory rank (largest existing score first)"
                )
            else:
                axes[row_index, j].tick_params(labelleft=False)
        axes[row_index, 3].imshow(
            invalid_time[order, None],
            aspect="auto",
            cmap=ListedColormap(["#FFFFFF", "#222222"]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=(0, 1, len(order) + 0.5, 0.5),
        )
        axes[row_index, 3].set_title("Time\ninvalid", fontsize=9)
        axes[row_index, 3].set_xticks([])
        axes[row_index, 3].tick_params(labelleft=False)
        axes[row_index, 3].set_xlabel(f"{invalid_time.sum()}/100", fontsize=9)
        ordered_rows = scores.iloc[order].reset_index()
        ordered_rows.insert(0, "rank", np.arange(1, len(order) + 1))
        ordered_rows["raw_nonpositive_time"] = invalid_time[order]
        rows.append(ordered_rows)
    colorbar = fig.colorbar(
        artist,
        ax=axes[:, :3],
        ticks=[-1, 0, 1],
        location="bottom",
        shrink=0.55,
        pad=0.02,
        fraction=0.04,
    )
    colorbar.ax.set_xticklabels(
        ["Below train interval", "Inside", "Above train interval"]
    )
    fig.suptitle(
        f"{MODEL_LABELS[name]}: where interval violations occur\n"
        "All 100 trajectories retained in each row; black strip = raw time issue",
        fontsize=13,
    )
    save_figure(fig, target / "outlier_heatmap")
    pd.concat(rows).to_csv(target / "heatmap_rows.csv", index=False)
    return invalid_counts


def histogram_values(data: dict) -> dict:
    return {
        "Groundspeed [kt; pooled points]": (
            data["real"]["values"][:, :, 1],
            data["generated"]["values"][:, :, 1],
        ),
        "Altitude [ft MSL; pooled points]": (
            data["real"]["values"][:, :, 2],
            data["generated"]["values"][:, :, 2],
        ),
        **{
            METRIC_LABELS[key]: (
                scalar_population(data, key, "real"),
                scalar_population(data, key, "generated_raw"),
            )
            for key in ("duration", "path_length", "total_abs_turn", "mean_speed")
        },
    }


def plot_histograms(
    target: Path, name: str, data: dict, edges: dict, density_limits: dict
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    records = []
    for axis, (label, populations) in zip(
        axes.flat, histogram_values(data).items(), strict=True
    ):
        for sample, title, color in zip(
            populations, ("Real", "Generated raw"), COLORS, strict=False
        ):
            values = finite(sample)
            axis.hist(
                values,
                bins=edges[label],
                density=True,
                histtype="step",
                linewidth=1.8,
                color=color,
                label=f"{title} (n={len(values):,})",
            )
            counts, _ = np.histogram(values, bins=edges[label])
            assert counts.sum() == len(values)
            records.extend(
                {
                    "metric": label,
                    "source": title,
                    "left": left,
                    "right": right,
                    "count": int(count),
                    "probability": count / len(values),
                }
                for left, right, count in zip(
                    edges[label][:-1], edges[label][1:], counts, strict=True
                )
            )
        axis.set_xlabel(label)
        axis.set_ylabel("Empirical density")
        axis.set_ylim(0, density_limits[label] * 1.08)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        f"{MODEL_LABELS[name]}: distribution shape\n"
        "30 common bins within each matched scope; full range; no KDE fitting",
        fontsize=13,
    )
    save_figure(fig, target / "histograms")
    pd.DataFrame(records).to_csv(target / "histogram_bins.csv", index=False)


def plot_quantile_difference(target: Path, name: str, data: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    probabilities = np.linspace(0.01, 0.99, 99)
    records = []
    for axis, key in zip(axes.flat, QUANTILE_KEYS, strict=True):
        real = finite(scalar_population(data, key, "real"))
        generated = finite(scalar_population(data, key, "generated_raw"))
        rq, gq = np.quantile(real, probabilities), np.quantile(generated, probabilities)
        difference = gq - rq
        axis.axhline(0, color="0.4", linewidth=1)
        axis.axvline(0.05, color="0.7", linestyle=":")
        axis.axvline(0.95, color="0.7", linestyle=":")
        axis.plot(probabilities, difference, color=COLORS[1], linewidth=1.7)
        median_delta = np.median(generated) - np.median(real)
        axis.scatter([0.5], [median_delta], color=COLORS[1], s=24, zorder=3)
        axis.set_title(
            f"Median difference: {median_delta:.3g} | n={len(real)}/{len(generated)}",
            fontsize=10,
        )
        axis.set_xlabel("Quantile probability")
        axis.set_ylabel(f"Generated − real: {METRIC_LABELS[key]}")
        axis.grid(alpha=0.2)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        records.extend(
            {
                "metric": key,
                "probability": q,
                "real_quantile": a,
                "generated_quantile": b,
                "difference": d,
            }
            for q, a, b, d in zip(probabilities, rq, gq, difference, strict=True)
        )
    fig.suptitle(
        f"{MODEL_LABELS[name]}: where the distribution differs\n"
        "Equal quantile levels, no trajectory pairing; "
        "extreme quantiles are descriptive",
        fontsize=13,
    )
    save_figure(fig, target / "quantile_difference")
    pd.DataFrame(records).to_csv(target / "quantiles.csv", index=False)


def plot_joint_density(
    target: Path,
    name: str,
    data: dict,
    xedges,
    yedges,
    maximum: float,
    delta_maximum: float,
) -> dict:
    probabilities = []
    for source in ("real", "generated"):
        values = data[source]["values"]
        count, _, _ = np.histogram2d(
            values[:, :, 1].ravel(), values[:, :, 2].ravel(), bins=(xedges, yedges)
        )
        assert count.sum() == values.shape[0] * values.shape[1]
        probabilities.append(count / count.sum())
    real, generated = probabilities
    minimum = 1 / data["real"]["values"][:, :, 1].size
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for axis, matrix, title in zip(
        axes[:2], probabilities, ("Real", "Generated raw"), strict=True
    ):
        artist = axis.imshow(
            np.ma.masked_equal(matrix.T, 0),
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            norm=LogNorm(vmin=minimum, vmax=maximum),
        )
        axis.set_title(title)
    difference = generated - real
    magnitude = delta_maximum
    delta = axes[2].imshow(
        difference.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-magnitude,
        vmax=magnitude,
    )
    axes[2].set_title("Generated − real bin probability")
    for axis in axes:
        axis.set_xlabel("Groundspeed [kt]")
        axis.set_ylabel("Altitude [ft MSL]")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.colorbar(
        artist, ax=axes[:2], label="Probability per bin (log color scale)", shrink=0.85
    )
    fig.colorbar(delta, ax=axes[2], label="Probability difference", shrink=0.85)
    fig.suptitle(
        f"{MODEL_LABELS[name]}: joint speed / altitude distribution\n"
        "Same 35×35 grid within scope; pooled prepared points, not independent flights",
        fontsize=13,
    )
    save_figure(fig, target / "joint_density")
    np.savez_compressed(
        target / "joint_density.npz",
        speed_edges=xedges,
        altitude_edges=yedges,
        real_probability=real,
        generated_probability=generated,
    )
    return {
        "real_probability_sum": float(real.sum()),
        "generated_probability_sum": float(generated.sum()),
        "shared_probability_color_max": float(maximum),
        "shared_difference_color_max": float(delta_maximum),
    }


def plot_coverage(target: Path, evaluation: Path, name: str, seed: int) -> None:
    coverage = pd.read_csv(evaluation / name / "coverage.csv")
    similarity = pd.read_csv(evaluation / name / "similarity.csv")
    fig, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")
    boxes_with_points(
        axis,
        {
            "Real → other real": coverage.real_leave_one_out_nn,
            "Real → generated": coverage.real_to_generated_nn,
            "Generated → real": similarity.generated_to_real_nn,
        },
        np.random.default_rng(seed),
    )
    axis.set_ylabel("Nearest trajectory RMS distance [fixed physical scaling]")
    axis.set_title(
        f"{MODEL_LABELS[name]}: similarity and coverage distances\n"
        "Every trajectory shown; descriptive distances, not a coverage percentage",
        fontsize=12,
    )
    save_figure(fig, target / "coverage_summary")


def plot_rates(output: Path, summary: pd.DataFrame, scope_models: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    for axis, (reference, names) in zip(axes.flat, scope_models.items(), strict=False):
        first = summary.set_index("model").loc[names[0]]
        populations = [("Real", first, "real")]
        populations.extend(
            (
                MODEL_LABELS[name].split(" · ")[0],
                summary.set_index("model").loc[name],
                "generated",
            )
            for name in names
        )
        for j, (_label, row, prefix) in enumerate(populations):
            rate = row[f"{prefix}_trajectory_outlier_fraction"] * 100
            low, high = (
                row[f"{prefix}_outlier_wilson_low"] * 100,
                row[f"{prefix}_outlier_wilson_high"] * 100,
            )
            axis.errorbar(
                rate,
                j,
                xerr=[[rate - low], [high - rate]],
                fmt="o",
                capsize=4,
                color=COLORS[j],
                markersize=6,
            )
            axis.annotate(
                f"{int(row[f'{prefix}_trajectory_outlier_count'])}/100",
                (rate, j),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        axis.set_yticks(np.arange(len(populations)), [row[0] for row in populations])
        axis.set_xlim(0, 23)
        axis.set_ylim(-0.55, len(populations) - 0.25)
        axis.invert_yaxis()
        axis.set_xlabel("Statistical trajectory outliers [%] · Wilson 95% interval")
        axis.set_title(SCOPE_LABELS[reference])
        axis.grid(axis="x", alpha=0.2)
    for axis in axes.flat[len(scope_models) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Frozen train intervals and validation thresholds | scopes kept separate",
        fontsize=13,
    )
    save_figure(fig, output / "outlier_rates")


def plot_latents(output: Path, evaluation: Path, seed: int) -> dict:
    summary = pd.read_csv(evaluation / "latent_summary.csv")
    result = {}
    for name in ("k256", "k1000"):
        target = output / "latent" / name
        coordinates = pd.read_csv(evaluation / "latent" / name / "pca_coordinates.csv")
        sources = [
            source for source in coordinates.source.unique() if source != "train"
        ]
        labels = {
            "held_out_real": "Test posterior",
            "VampPrior": "VampPrior",
            "Latent DDPM": "Latent DDPM",
        }
        points = {
            source: coordinates.loc[
                coordinates.source.eq(source), ["pc1", "pc2"]
            ].to_numpy()
            for source in sources
        }
        xedges = common_edges([value[:, 0] for value in points.values()], 40)
        yedges = common_edges([value[:, 1] for value in points.values()], 40)
        probabilities = {
            source: np.histogram2d(value[:, 0], value[:, 1], bins=(xedges, yedges))[0]
            / len(value)
            for source, value in points.items()
        }
        for matrix in probabilities.values():
            np.testing.assert_allclose(matrix.sum(), 1)
        ratio = summary.loc[
            summary.latent_model.eq(name), "pca_2d_explained_fraction"
        ].iloc[0]
        maximum = max(value.max() for value in probabilities.values())
        fig, axes = plt.subplots(
            1,
            len(sources),
            figsize=(5 * len(sources), 5.1),
            layout="constrained",
            squeeze=False,
        )
        for axis, source in zip(axes.flat, sources, strict=True):
            artist = axis.imshow(
                np.ma.masked_equal(probabilities[source].T, 0),
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                norm=LogNorm(vmin=1 / len(points[source]), vmax=maximum),
            )
            axis.scatter(*points[source].T, s=4, color="black", alpha=0.22)
            axis.set_title(f"{labels[source]} · n={len(points[source])}")
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        fig.colorbar(
            artist,
            ax=axes.ravel().tolist(),
            label="Probability per bin (shared log scale)",
            shrink=0.85,
        )
        fig.suptitle(
            f"{name.upper()}: latent populations on the existing train-fitted PCA\n"
            f"2D retained variance {ratio:.2%} | all evaluated points shown\n"
            "Same bins, axes and probability color scale",
            fontsize=13,
        )
        save_figure(fig, target / "density")
        distances = pd.read_csv(evaluation / "latent" / name / "distances.csv")
        populations = {
            source: rows.nearest_train_draw.to_numpy()
            for source, rows in distances.groupby("source", sort=False)
        }
        fig, axis = plt.subplots(figsize=(9, 5.7), layout="constrained")
        boxes_with_points(axis, populations, np.random.default_rng(seed))
        threshold = summary.loc[
            summary.latent_model.eq(name), "nn_outlier_threshold"
        ].iloc[0]
        if np.isfinite(threshold):
            axis.axhline(
                threshold,
                color="#AA2222",
                linestyle="--",
                label=f"Frozen validation q95 = {threshold:.3f}",
            )
            axis.legend(fontsize=9)
        axis.set_ylabel(
            "Nearest training posterior-draw distance [full standardized 64D]"
        )
        axis.set_title(
            f"{name.upper()}: latent tails and validation threshold\n"
            "All 500 points per source; plot-frame bounds do not define outliers",
            fontsize=12,
        )
        save_figure(fig, target / "distance_summary")
        np.savez_compressed(
            target / "density_bins.npz",
            xedges=xedges,
            yedges=yedges,
            **{source: values for source, values in probabilities.items()},
        )
        result[name] = {
            "counts": {source: len(value) for source, value in points.items()},
            "probability_sums": {
                source: float(value.sum()) for source, value in probabilities.items()
            },
        }
    return result


def write_report(output: Path, evaluation: Path, scope_models: dict, helpers) -> None:
    summary = pd.read_csv(output / "summary.csv")
    columns = [
        "model",
        "generated_point_outside_fraction",
        "real_point_outside_fraction",
        "generated_trajectory_outlier_count",
        "real_trajectory_outlier_count",
        "raw_invalid_time_count",
    ]
    sections = [
        "# 生成分布评估：按问题选择图型\n",
        "本版复用v3固定六组数据、逐轨迹统计、分位区间、异常阈值与训练拟合PCA。"
        "只改变图表和报告，未重训、采样或重新计算评估指标。\n",
        "先读 [FINDINGS.md](FINDINGS.md) 查看结果解释。数字与v3完全相同：\n",
        helpers.markdown_table(
            summary[columns]
            .assign(
                generated_point_outside_fraction=lambda frame: (
                    frame.generated_point_outside_fraction * 100
                ),
                real_point_outside_fraction=lambda frame: (
                    frame.real_point_outside_fraction * 100
                ),
            )
            .rename(
                columns={
                    "model": "模型",
                    "generated_point_outside_fraction": "生成点越界/%",
                    "real_point_outside_fraction": "真实点越界/%",
                    "generated_trajectory_outlier_count": "生成轨迹异常/100",
                    "real_trajectory_outlier_count": "真实轨迹异常/100",
                    "raw_invalid_time_count": "原始时间异常/100",
                }
            )
        ),
        "\n## 如何读新版图表\n",
        "- 箱线图显示中位数、IQR与1.5-IQR须；全部原始散点同时展示，须外点没有隐藏。"
        "每个点是一条轨迹。涉及变化率时，仅展示原始时间有效的轨迹，"
        "n在各组标签中给出。",
        "- 进度分位带显示真实／生成的中位线和q05–q95带；"
        "灰色区间仍是v3训练q01–q99。"
        "横轴是200个准备点的相对进度，不是秒。",
        "- 越界热图按既有整条轨迹分数排序，三个特征共用行顺序。"
        "蓝色低于训练区间，橙色高于，"
        "白色在区间内；每一行都保留。最右侧黑色条单独标出原始时间不递增。"
        "行与ID映射保存为heatmap_rows.csv。",
        "- 直方图在同一参考范围内共用30个分箱、覆盖全部数值。没有新增KDE拟合，避免带宽制造或抹平峰。"
        "高度／速度汇总点图混合了进近阶段，所以要结合进度分位带阅读。",
        "- 分位数差图的纵轴是生成分位数减真实分位数，零线表示该分位数吻合；"
        "比较相同概率级别，不逐条配对航班。1%和99%尾部由少量轨迹决定，只作描述。",
        "- 速度–高度联合图共用35×35网格和颜色标度，第三幅是生成减真实的格子概率。"
        "其图形数据保存为joint_density.npz，所有格子概率总和等于1。",
        "- 潜空间密度图复用原训练拟合PCA，来源共用坐标、40×40分箱和颜色标度，"
        "散点保留所有500个样本。潜空间异常仍由完整64维距离和既有验证阈值决定。",
        "- CDF继续作为补充。Wasserstein、KS统计量、置信区间、异常清单等文件保持不变；"
        "visualization_manifest.json验证原有全部CSV和NPZ文件逐字节不变。\n",
        "## 校准异常率\n",
        "![统计异常率与区间](outlier_rates.png)\n",
    ]
    for reference, names in scope_models.items():
        sections += [
            f"## {SCOPE_LABELS[reference]}\n",
            f"![逐轨迹比较](scopes/{reference}-scalar_summary.png)\n",
        ]
        for name in names:
            sections += [f"### {MODEL_LABELS[name]}\n"]
            for filename, caption in (
                ("profiles", "进度中位线与分位带"),
                ("outlier_heatmap", "越界位置与独立时间异常条"),
                ("histograms", "共同分箱直方图"),
                ("quantile_difference", "生成减真实的分位数差"),
                ("joint_density", "速度与高度联合分布"),
                ("coverage_summary", "完整轨迹距离箱线图与全部散点"),
            ):
                sections += [f"![{caption}]({name}/{filename}.png)\n"]
    sections += ["## 潜空间分布与尾部\n"]
    for name in ("k256", "k1000"):
        sections += [
            f"### {name.upper()}\n",
            f"![潜空间密度](latent/{name}/density.png)\n",
            f"![潜空间距离与阈值](latent/{name}/distance_summary.png)\n",
        ]
    sections += [
        "## 补充：原ECDF与散点图\n",
        "这些图完整保留，便于读取某个阈值以下的比例和与v3交叉检查。\n",
    ]
    for name in [item for names in scope_models.values() for item in names]:
        sections += [f"### {MODEL_LABELS[name]}\n"]
        for filename, label in (
            ("distributions", "边际ECDF"),
            ("temporal", "时序ECDF"),
            ("coverage", "距离ECDF"),
        ):
            sections += [f"![{label}]({name}/{filename}.png)\n"]
    sections += [
        "## 数据与复现\n",
        "使用 reproduce.ps1 重画到新目录。PNG与SVG均保存；bins和排序数据也保留。"
        f"原数值评估来源：{evaluation}。按机场、跑道和flow范围分别解释，统计异常不能等同于不可飞。\n",
    ]
    (output / "REPORT.md").read_text("utf8")
    (output / "REPORT.md").write_text("\n".join(sections) + "\n", "utf8")
    findings = (output / "FINDINGS.md").read_text("utf8")
    findings = findings.replace("# 本次结果的解释", "# 本次结果的解释（v4图型修订）", 1)
    findings = findings.replace(
        "![单flow TCVAE K256的分布](tcvae-flow-k256/distributions.png)",
        "![同范围逐轨迹比较](scopes/zurich-singleflow-scalar_summary.png)\n\n"
        "![单flow TCVAE K256的分位数差](tcvae-flow-k256/quantile_difference.png)",
    )
    findings = findings.replace(
        "![天津时序诊断](tianjin-noaib-tcvae-k1000/temporal.png)",
        "![天津逐轨迹比较](scopes/tianjin-noaib-rwy16l-scalar_summary.png)\n\n"
        "![天津越界位置与时间异常](tianjin-noaib-tcvae-k1000/outlier_heatmap.png)",
    )
    findings = findings.replace(
        "![K1000潜空间](latent/k1000/pca.png)",
        "![K1000潜空间密度](latent/k1000/density.png)\n\n"
        "![K1000潜空间尾部与阈值](latent/k1000/distance_summary.png)",
    )
    findings = findings.replace(
        "运行 `./runs/distribution-evaluation-20261008-v3/reproduce.ps1` "
        "可用冻结脚本重新计算，默认写入新的目录。",
        "本版只修改图表，全部原CSV/NPZ指标文件与v3逐字节相同。"
        "运行本目录的 `reproduce.ps1` 可用冻结重画脚本生成新目录。",
    )
    (output / "FINDINGS.md").write_text(findings, "utf8")
    runtime = evaluation.parents[1] / "output/distribution-evaluation-runtime"
    reproduce = output / "reproduce.ps1"
    reproduce.read_text("utf8")
    command = (
        f"& uv --cache-dir '{(runtime / 'uv-cache').as_posix()}' "
        f"run --no-project --python '{(runtime / '.venv/Scripts/python.exe').as_posix()}' "
        f"python '{(output / Path(__file__).name).as_posix()}' "
        f"--evaluation '{evaluation.as_posix()}' --inputs "
        f"'{(evaluation.parents[1] / 'runs/minisky-preparation-20261003/inputs').as_posix()}' "
        "--output $EvaluationOutput --seed 42\n"
    )
    reproduce.write_text(
        "param([string]$EvaluationOutput = '"
        + output.with_name(output.name + "-reproduced").as_posix()
        + "')\n\n"
        + command
        + "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n",
        "utf8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluation, inputs, output = (
        p.resolve() for p in (args.evaluation, args.inputs, args.output)
    )
    if output.exists() or evaluation == output or evaluation in output.parents:
        parser.error("Choose a new directory outside the source evaluation")
    source_manifest = json.loads((evaluation / "manifest.json").read_text("utf8"))
    root = inputs.parents[2]
    for filename, expected in source_manifest["input_sha256"].items():
        path = root / filename
        if path.suffix in (".parquet", ".npz") and inputs in path.parents:
            if digest(path) != expected:
                raise ValueError(f"Frozen plotting input changed: {filename}")
    spec = importlib.util.spec_from_file_location(
        "frozen_evaluation", evaluation / "evaluate_generation_distributions.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    original_files = {
        str(p.relative_to(evaluation)): digest(p)
        for p in evaluation.rglob("*")
        if p.is_file() and p.suffix in (".csv", ".npz")
    }
    shutil.copytree(evaluation, output, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    (output / "scopes").mkdir()
    datasets, scopes, checks = {}, {}, {}
    model_references = source_manifest["references"]
    if args.smoke:
        model_references = {
            name: model_references[name]
            for name in ("ddpm-flow700", "tianjin-noaib-tcvae-k1000")
        }
    for name, reference in model_references.items():
        datasets[name] = load_model(evaluation, inputs, name, reference, helpers)
        scopes.setdefault(reference, []).append(name)
    for reference, names in scopes.items():
        plot_scope_boxes(output, reference, names, datasets, args.seed)
        first = histogram_values(datasets[names[0]])
        edges = {
            label: common_edges(
                [
                    first[label][0],
                    *[histogram_values(datasets[name])[label][1] for name in names],
                ]
            )
            for label in first
        }
        speed_edges = common_edges(
            [
                datasets[names[0]]["real"]["values"][:, :, 1],
                *[datasets[name]["generated"]["values"][:, :, 1] for name in names],
            ],
            35,
        )
        altitude_edges = common_edges(
            [
                datasets[names[0]]["real"]["values"][:, :, 2],
                *[datasets[name]["generated"]["values"][:, :, 2] for name in names],
            ],
            35,
        )
        density_limits = {
            label: max(
                np.histogram(finite(values), bins=edges[label], density=True)[0].max()
                for values in [
                    first[label][0],
                    *[histogram_values(datasets[name])[label][1] for name in names],
                ]
            )
            for label in first
        }
        joint_populations = [
            datasets[names[0]]["real"]["values"],
            *[datasets[name]["generated"]["values"] for name in names],
        ]
        joint_probabilities = [
            np.histogram2d(
                values[:, :, 1].ravel(),
                values[:, :, 2].ravel(),
                bins=(speed_edges, altitude_edges),
            )[0]
            / (len(values) * 200)
            for values in joint_populations
        ]
        joint_maximum = max(matrix.max() for matrix in joint_probabilities)
        delta_maximum = max(
            1 / 20000,
            max(
                np.abs(matrix - joint_probabilities[0]).max()
                for matrix in joint_probabilities[1:]
            ),
        )
        for name in names:
            data, target = datasets[name], output / name
            plot_profiles(target, name, data)
            heatmap = plot_heatmap(target, name, data)
            plot_histograms(target, name, data, edges, density_limits)
            plot_quantile_difference(target, name, data)
            density = plot_joint_density(
                target,
                name,
                data,
                speed_edges,
                altitude_edges,
                joint_maximum,
                delta_maximum,
            )
            plot_coverage(target, evaluation, name, args.seed)
            checks[name] = {"heatmap_invalid_time_counts": heatmap, **density}
            print(f"Replotted {name}", flush=True)
    plot_rates(output, pd.read_csv(evaluation / "summary.csv"), scopes)
    latent_checks = plot_latents(output, evaluation, args.seed)
    write_report(output, evaluation, scopes, helpers)
    for filename, expected in original_files.items():
        if digest(output / filename) != expected:
            raise ValueError(f"Original numerical artifact changed: {filename}")
    write_json(
        output / "visualization_manifest.json",
        {
            "source_evaluation": str(evaluation),
            "source_manifest_sha256": digest(evaluation / "manifest.json"),
            "execution": (
                "local replot only; no training, sampling, "
                "metric or threshold refitting"
            ),
            "seed": args.seed,
            "smoke": args.smoke,
            "models_plotted": list(datasets),
            "old_numerical_files_unchanged": len(original_files),
            "original_sha256": original_files,
            "heatmap_and_density_checks": checks,
            "latent_density_checks": latent_checks,
            "histogram_bins": 30,
            "joint_bins_per_axis": 35,
            "latent_bins_per_axis": 40,
            "numeric_metrics_unchanged": True,
            "all_evaluated_points_retained": True,
            "replot_script_sha256": digest(Path(__file__)),
        },
    )
    print(f"OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
