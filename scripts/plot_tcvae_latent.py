"""Plot a VampPrior TCVAE latent space in a shared train-fitted PCA plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from chabuduo.artifacts import SplitData
from chabuduo.models.tcvae import TCVAELightning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="TCVAE run directory")
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Prepared artifact containing train.npz and test.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.ckpt"),
        help="Absolute checkpoint path, or path relative to --run",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5000,
        help="Posterior and prior samples shown in the coverage panel",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG (default: <run>/plots/latent_space.png)",
    )
    return parser.parse_args()


def resolve_checkpoint(path: Path, run: Path) -> Path:
    return path.resolve() if path.is_absolute() else (run / path).resolve()


@torch.inference_mode()
def encode_split(
    module: TCVAELightning,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[start : start + batch_size])
        batch = batch.to(device).permute(0, 2, 1)
        mean, log_variance = module.network.encode(batch)
        means.append(mean.cpu().numpy())
        log_variances.append(log_variance.cpu().numpy())
    return (
        np.concatenate(means).astype(np.float32, copy=False),
        np.concatenate(log_variances).astype(np.float32, copy=False),
    )


def component_usage(
    posterior_mean: np.ndarray,
    component_mean: np.ndarray,
    component_log_variance: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Average component responsibilities evaluated at posterior means."""
    inverse_variance = np.exp(-component_log_variance)
    weighted_mean = component_mean * inverse_variance
    constant = (
        np.sum(component_mean * weighted_mean + component_log_variance, axis=1)
        + posterior_mean.shape[1] * np.log(2 * np.pi)
    )
    total = np.zeros(len(component_mean), dtype=np.float64)

    for start in range(0, len(posterior_mean), batch_size):
        value = posterior_mean[start : start + batch_size]
        quadratic = (
            np.square(value) @ inverse_variance.T
            - 2 * value @ weighted_mean.T
            + constant[None, :]
        )
        log_probability = -0.5 * quadratic
        log_probability -= log_probability.max(axis=1, keepdims=True)
        probability = np.exp(log_probability)
        probability /= probability.sum(axis=1, keepdims=True)
        total += probability.sum(axis=0, dtype=np.float64)

    return total / len(posterior_mean)


def sample_aggregated_posterior(
    mean: np.ndarray,
    log_variance: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = rng.integers(0, len(mean), size=count)
    noise = rng.standard_normal((count, mean.shape[1])).astype(np.float32)
    return mean[indices] + np.exp(0.5 * log_variance[indices]) * noise


def sample_vamp_prior(
    component_mean: np.ndarray,
    component_log_variance: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    components = rng.integers(0, len(component_mean), size=count)
    noise = rng.standard_normal((count, component_mean.shape[1])).astype(np.float32)
    latent = component_mean[components] + (
        np.exp(0.5 * component_log_variance[components]) * noise
    )
    return latent, components


def set_pca_axes(
    axis: plt.Axes,
    explained: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    axis.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% variance)")
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, (0.002, 0.998))
    padding = max(0.05 * (upper - lower), 1e-3)
    return float(lower - padding), float(upper + padding)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be greater than zero")

    run = args.run.resolve()
    artifact = args.artifact.resolve()
    checkpoint = resolve_checkpoint(args.checkpoint, run)
    output = (
        args.output.resolve()
        if args.output is not None
        else (run / "plots" / "latent_space.png").resolve()
    )
    data_output = output.with_name("latent_space_data.npz")
    diagnostics_output = output.with_name("latent_space_diagnostics.json")
    for path in (output, data_output, diagnostics_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    for path in (
        checkpoint,
        artifact / "train.npz",
        artifact / "test.npz",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = TCVAELightning.load_from_checkpoint(checkpoint, map_location=device)
    module.to(device).eval()
    if module.network.prior_type != "vamp":
        raise ValueError("This visualization requires a VampPrior TCVAE checkpoint")

    train = SplitData.load(artifact / "train.npz")
    test = SplitData.load(artifact / "test.npz")
    expected_shape = (
        module.network.sequence_length,
        module.network.input_channels,
    )
    if train.values.shape[1:] != expected_shape:
        raise ValueError(
            "Prepared trajectories and checkpoint use different shapes: "
            f"{train.values.shape[1:]} != {expected_shape}"
        )

    train_mean, train_log_variance = encode_split(
        module,
        train.values,
        device=device,
        batch_size=args.batch_size,
    )
    test_mean, test_log_variance = encode_split(
        module,
        test.values,
        device=device,
        batch_size=args.batch_size,
    )
    with torch.inference_mode():
        component_mean_tensor, component_log_variance_tensor = (
            module.network.prior_parameters()
        )
    component_mean = component_mean_tensor.cpu().numpy().astype(np.float32)
    component_log_variance = (
        component_log_variance_tensor.cpu().numpy().astype(np.float32)
    )

    pca = PCA(n_components=2, random_state=args.seed)
    train_pca = pca.fit_transform(train_mean)
    test_pca = pca.transform(test_mean)
    component_pca = pca.transform(component_mean)

    rng = np.random.default_rng(args.seed)
    posterior_sample = sample_aggregated_posterior(
        train_mean,
        train_log_variance,
        args.sample_count,
        rng,
    )
    prior_sample, prior_component = sample_vamp_prior(
        component_mean,
        component_log_variance,
        args.sample_count,
        rng,
    )
    posterior_sample_pca = pca.transform(posterior_sample)
    prior_sample_pca = pca.transform(prior_sample)

    usage = component_usage(
        train_mean,
        component_mean,
        component_log_variance,
        batch_size=args.batch_size,
    )
    component_spread = np.sqrt(np.mean(np.exp(component_log_variance), axis=1))
    positive_usage = usage[usage > 0]
    effective_components = float(
        np.exp(-np.sum(positive_usage * np.log(positive_usage)))
    )

    all_x = np.concatenate(
        (
            train_pca[:, 0],
            test_pca[:, 0],
            component_pca[:, 0],
            posterior_sample_pca[:, 0],
            prior_sample_pca[:, 0],
        )
    )
    all_y = np.concatenate(
        (
            train_pca[:, 1],
            test_pca[:, 1],
            component_pca[:, 1],
            posterior_sample_pca[:, 1],
            prior_sample_pca[:, 1],
        )
    )
    x_limits = robust_limits(all_x)
    y_limits = robust_limits(all_y)
    explained = pca.explained_variance_ratio_

    figure, axes = plt.subplots(2, 2, figsize=(15, 12))
    axis = axes[0, 0]
    axis.scatter(
        train_pca[:, 0],
        train_pca[:, 1],
        s=6,
        alpha=0.16,
        color="#7A7A7A",
        label=f"Train posterior mean (n={len(train_mean)})",
        rasterized=True,
    )
    axis.scatter(
        test_pca[:, 0],
        test_pca[:, 1],
        s=8,
        alpha=0.28,
        color="#4C78A8",
        label=f"Test posterior mean (n={len(test_mean)})",
        rasterized=True,
    )
    axis.set_title("A. Real trajectory posterior means")
    axis.legend(markerscale=2)
    set_pca_axes(axis, explained, x_limits, y_limits)

    axis = axes[0, 1]
    axis.scatter(
        posterior_sample_pca[:, 0],
        posterior_sample_pca[:, 1],
        s=7,
        alpha=0.16,
        color="#777777",
        label="Aggregated posterior sample",
        rasterized=True,
    )
    axis.scatter(
        prior_sample_pca[:, 0],
        prior_sample_pca[:, 1],
        s=7,
        alpha=0.16,
        color="#4C78A8",
        label="VampPrior sample",
        rasterized=True,
    )
    axis.set_title("B. Prior coverage of the aggregated posterior")
    axis.legend(markerscale=2)
    set_pca_axes(axis, explained, x_limits, y_limits)

    axis = axes[1, 0]
    axis.scatter(
        train_pca[:, 0],
        train_pca[:, 1],
        s=5,
        alpha=0.08,
        color="#888888",
        rasterized=True,
    )
    relative_usage = np.clip(usage * len(usage), 0, 12)
    point_size = 8 + 18 * np.sqrt(relative_usage)
    components = axis.scatter(
        component_pca[:, 0],
        component_pca[:, 1],
        s=point_size,
        c=component_spread,
        cmap="viridis",
        alpha=0.8,
        linewidth=0.15,
        edgecolor="white",
        label=f"VampPrior means (K={len(component_mean)})",
        rasterized=True,
    )
    figure.colorbar(
        components,
        ax=axis,
        label="Component RMS standard deviation",
        fraction=0.046,
        pad=0.04,
    )
    axis.set_title("C. Pseudo-components: color=spread, size=usage")
    axis.legend(markerscale=0.8)
    set_pca_axes(axis, explained, x_limits, y_limits)

    axis = axes[1, 1]
    sorted_usage = np.sort(usage)[::-1]
    rank = np.arange(1, len(sorted_usage) + 1)
    axis.plot(rank, sorted_usage, color="#4C78A8", linewidth=1.5)
    axis.axhline(
        1 / len(usage),
        color="#E45756",
        linestyle="--",
        linewidth=1.2,
        label=f"Uniform usage = 1/{len(usage)}",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Component rank")
    axis.set_ylabel("Mean responsibility")
    axis.set_title(f"D. Component use (effective K = {effective_components:.1f})")
    axis.grid(alpha=0.2)
    axis.legend()

    figure.suptitle(
        f"{run.name}\ntrain-fitted PCA, checkpoint={checkpoint.name}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    np.savez_compressed(
        data_output,
        train_posterior_mean=train_mean,
        train_posterior_log_variance=train_log_variance,
        train_pca=train_pca,
        train_trajectory_id=np.asarray(train.trajectory_id).astype(str),
        test_posterior_mean=test_mean,
        test_posterior_log_variance=test_log_variance,
        test_pca=test_pca,
        test_trajectory_id=np.asarray(test.trajectory_id).astype(str),
        component_mean=component_mean,
        component_log_variance=component_log_variance,
        component_pca=component_pca,
        component_usage=usage,
        component_spread=component_spread,
        posterior_sample_pca=posterior_sample_pca,
        prior_sample_pca=prior_sample_pca,
        prior_sample_component=prior_component,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance_ratio=explained,
    )
    diagnostics = {
        "analysis": "vampprior_tcvae_latent_pca",
        "run": str(run),
        "artifact": str(artifact),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(checkpoint_state["epoch"]),
        "device": str(device),
        "seed": args.seed,
        "train_count": int(len(train_mean)),
        "test_count": int(len(test_mean)),
        "latent_dimension": int(train_mean.shape[1]),
        "prior_components": int(len(component_mean)),
        "pca_explained_variance_ratio": explained.tolist(),
        "pca_two_component_explained_variance": float(explained.sum()),
        "effective_components": effective_components,
        "component_usage_definition": (
            "Mean posterior responsibility evaluated at each training "
            "trajectory posterior mean."
        ),
        "component_usage_min": float(usage.min()),
        "component_usage_max": float(usage.max()),
        "component_usage_quantiles": {
            str(value): float(np.quantile(usage, value))
            for value in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "components_below_one_tenth_uniform_usage": int(
            np.count_nonzero(usage < 0.1 / len(usage))
        ),
        "component_spread_quantiles": {
            str(value): float(np.quantile(component_spread, value))
            for value in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "posterior_prior_plot_sample_count": args.sample_count,
        "notes": [
            "PCA is fitted only on training posterior means.",
            "All panels use the same PCA transform and axis limits.",
            "The two-dimensional projection does not preserve all latent distances.",
        ],
        "outputs": [str(output), str(data_output), str(diagnostics_output)],
    }
    diagnostics_output.write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf8",
    )

    print(f"OUTPUT={output}")
    print(f"DATA={data_output}")
    print(f"DIAGNOSTICS={diagnostics_output}")
    print(f"CHECKPOINT_EPOCH={diagnostics['checkpoint_epoch']}")
    print(f"EFFECTIVE_COMPONENTS={effective_components:.3f}")
    print(f"PCA_EXPLAINED={explained.sum():.6f}")


if __name__ == "__main__":
    main()
