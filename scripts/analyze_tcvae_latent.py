"""Visualize a trained TCVAE posterior-mean latent space.

The analysis is deliberately label-free: PCA shows global geometry, while a GMM
is fitted in a variance-preserving PCA space and supplies colors for t-SNE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from chabuduo.artifacts import SplitData
from chabuduo.models.tcvae import TCVAELightning

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA and GMM-colored t-SNE for a TCVAE latent space."
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "runs" / "zurich_tcvae-v1-seed-42",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "data" / "processed" / "zurich_tcvae-v1",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/last.ckpt"),
        help="Absolute path, or a path relative to --run.",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gmm-k-min", type=int, default=1)
    parser.add_argument("--gmm-k-max", type=int, default=12)
    parser.add_argument("--gmm-n-init", type=int, default=5)
    parser.add_argument("--stability-runs", type=int, default=5)
    parser.add_argument("--tsne-max-samples", type=int, default=5000)
    parser.add_argument("--tsne-perplexities", type=int, nargs="+", default=(30, 50))
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_epoch(checkpoint: Path) -> int:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(state["epoch"])


def encode_posterior_mean(
    checkpoint: Path,
    values: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = TCVAELightning.load_from_checkpoint(checkpoint, map_location=device)
    module.to(device).eval()
    means: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            mean, _ = module.network.encode(batch.permute(0, 2, 1))
            means.append(mean.cpu().numpy())
    return np.concatenate(means).astype(np.float32, copy=False)


def fit_analysis_pca(latent: np.ndarray) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=0.95, svd_solver="full")
    return pca, pca.fit_transform(latent)


def fit_gmms(
    features: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    n_init: int,
    seed: int,
) -> tuple[GaussianMixture, pd.DataFrame]:
    if not 1 <= k_min <= k_max:
        raise ValueError("Expected 1 <= gmm-k-min <= gmm-k-max")
    if k_max >= len(features):
        raise ValueError("gmm-k-max must be smaller than the sample count")

    rows: list[dict[str, float | int | bool]] = []
    models: dict[int, GaussianMixture] = {}
    for components in range(k_min, k_max + 1):
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            n_init=n_init,
            reg_covar=1e-6,
            random_state=seed,
        ).fit(features)
        probabilities = model.predict_proba(features)
        labels = probabilities.argmax(axis=1)
        counts = np.bincount(labels, minlength=components)
        rows.append(
            {
                "components": components,
                "bic": float(model.bic(features)),
                "aic": float(model.aic(features)),
                "converged": bool(model.converged_),
                "iterations": int(model.n_iter_),
                "smallest_cluster": int(counts.min()),
                "largest_cluster": int(counts.max()),
                "mean_max_probability": float(probabilities.max(axis=1).mean()),
                "low_confidence_fraction": float(
                    (probabilities.max(axis=1) < 0.6).mean()
                ),
            }
        )
        models[components] = model

    metrics = pd.DataFrame(rows)
    selected_k = int(metrics.loc[metrics.bic.idxmin(), "components"])
    return models[selected_k], metrics


def gmm_stability(
    features: np.ndarray,
    reference_labels: np.ndarray,
    *,
    components: int,
    runs: int,
    seed: int,
) -> tuple[float, list[float]]:
    scores: list[float] = []
    for offset in range(runs):
        labels = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            n_init=1,
            reg_covar=1e-6,
            random_state=seed + 100 + offset,
        ).fit_predict(features)
        scores.append(float(adjusted_rand_score(reference_labels, labels)))
    return float(np.mean(scores)), scores


def plot_pca_global(latent: np.ndarray, output: Path, seed: int) -> PCA:
    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(latent)
    rng = np.random.default_rng(seed)
    reference = rng.standard_normal(latent.shape)
    reference_projected = pca.transform(reference)
    explained = 100 * pca.explained_variance_ratio_

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    axes[0].scatter(projected[:, 0], projected[:, 1], s=7, alpha=0.22, color="#4C78A8")
    axes[0].scatter(
        [projected[:, 0].mean()],
        [projected[:, 1].mean()],
        marker="x",
        s=90,
        linewidth=2,
        color="black",
        label="Latent mean",
    )
    axes[0].set_title("Train posterior means (auto-scaled)")
    axes[0].legend()

    axes[1].scatter(
        reference_projected[:, 0],
        reference_projected[:, 1],
        s=6,
        alpha=0.08,
        color="#F58518",
        label="Reference N(0, I)",
    )
    axes[1].scatter(
        projected[:, 0],
        projected[:, 1],
        s=7,
        alpha=0.20,
        color="#4C78A8",
        label="Posterior mean",
    )
    axes[1].set_title("Posterior means versus standard-normal reference")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
        axis.grid(alpha=0.2)
    figure.suptitle("Zurich TCVAE latent-space global structure", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return pca


def plot_explained_variance(latent: np.ndarray, output: Path) -> np.ndarray:
    pca = PCA().fit(latent)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    count = min(20, len(ratios))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].bar(np.arange(1, count + 1), 100 * ratios[:count], color="#4C78A8")
    axes[0].set_title(f"First {count} principal components")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[1].plot(
        np.arange(1, len(cumulative) + 1),
        100 * cumulative,
        color="#E45756",
        linewidth=2,
    )
    for threshold in (0.90, 0.95, 0.99):
        dimension = int(np.searchsorted(cumulative, threshold) + 1)
        axes[1].axhline(100 * threshold, color="0.6", linestyle="--", linewidth=0.8)
        axes[1].scatter([dimension], [100 * cumulative[dimension - 1]], s=28)
        axes[1].annotate(
            f"{int(100 * threshold)}%: {dimension} dims",
            (dimension, 100 * cumulative[dimension - 1]),
            xytext=(5, -13),
            textcoords="offset points",
        )
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_xlabel("Number of principal components")
    axes[1].set_ylabel("Cumulative variance (%)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Zurich TCVAE latent effective dimensionality", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return ratios


def plot_gmm_selection(metrics: pd.DataFrame, output: Path, selected_k: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(metrics.components, metrics.bic, marker="o", label="BIC")
    axes[0].plot(metrics.components, metrics.aic, marker="o", label="AIC")
    axes[0].axvline(
        selected_k, color="#E45756", linestyle="--", label=f"BIC choice: K={selected_k}"
    )
    axes[0].set_xlabel("GMM components (K)")
    axes[0].set_ylabel("Information criterion (lower is better)")
    axes[0].set_title("GMM model selection")
    axes[0].legend()
    axes[1].plot(
        metrics.components,
        metrics.mean_max_probability,
        marker="o",
        color="#4C78A8",
        label="Mean max probability",
    )
    axes[1].plot(
        metrics.components,
        metrics.low_confidence_fraction,
        marker="o",
        color="#F58518",
        label="Fraction below 0.6",
    )
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("GMM components (K)")
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Assignment confidence")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Zurich TCVAE latent GMM diagnostics", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def stratified_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(labels) <= maximum:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        allocation = max(1, int(round(maximum * len(rows) / len(labels))))
        chosen.append(rng.choice(rows, size=min(allocation, len(rows)), replace=False))
    result = np.concatenate(chosen)
    if len(result) > maximum:
        result = rng.choice(result, size=maximum, replace=False)
    return np.sort(result)


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    perplexities: list[int],
    threshold: float,
    seed: int,
    output: Path,
) -> pd.DataFrame:
    figure, axes = plt.subplots(
        1, len(perplexities), figsize=(7 * len(perplexities), 6)
    )
    axes = np.atleast_1d(axes)
    embeddings: list[np.ndarray] = []
    colors = plt.get_cmap("tab20", int(labels.max()) + 1)
    for axis, requested in zip(axes, perplexities, strict=True):
        perplexity = min(float(requested), max(5.0, (len(features) - 1) / 3))
        embedded = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=seed,
            method="barnes_hut",
        ).fit_transform(features)
        embeddings.append(embedded)
        uncertain = confidence < threshold
        for label in np.unique(labels):
            rows = (labels == label) & ~uncertain
            axis.scatter(
                embedded[rows, 0],
                embedded[rows, 1],
                s=8,
                alpha=0.55,
                color=colors(int(label)),
                label=f"GMM {int(label)}",
            )
        if uncertain.any():
            axis.scatter(
                embedded[uncertain, 0],
                embedded[uncertain, 1],
                s=7,
                alpha=0.25,
                color="0.45",
                label=f"confidence < {threshold:g}",
            )
        axis.set_title(f"t-SNE perplexity {perplexity:g}")
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.grid(alpha=0.15)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, names, loc="lower center", ncol=min(6, len(names)), frameon=False
    )
    figure.suptitle("Zurich TCVAE train latent — GMM-colored t-SNE", fontsize=16)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)

    records: list[pd.DataFrame] = []
    for perplexity, embedded in zip(perplexities, embeddings, strict=True):
        records.append(
            pd.DataFrame(
                {
                    "trajectory_id": trajectory_ids,
                    "perplexity": perplexity,
                    "tsne_1": embedded[:, 0],
                    "tsne_2": embedded[:, 1],
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    artifact = args.artifact.resolve()
    checkpoint = resolve_path(args.checkpoint, run).resolve()
    split_path = artifact / f"{args.split}.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    epoch = checkpoint_epoch(checkpoint)
    checkpoint_name = checkpoint.stem.replace(".", "-")
    output = args.output or (
        run
        / "latent_analysis"
        / f"{checkpoint_name}-epoch{epoch}-{args.split}-posterior-mean-seed{args.seed}"
    )
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = SplitData.load(split_path)
    latent = encode_posterior_mean(checkpoint, data.values, batch_size=args.batch_size)
    trajectory_ids = np.asarray(data.trajectory_id).astype(str)
    if len(trajectory_ids) != len(latent):
        raise ValueError("Trajectory IDs and latent rows have different lengths")
    np.savez_compressed(
        output / "latent_mu.npz",
        latent_mu=latent,
        trajectory_id=trajectory_ids,
    )

    pca2 = plot_pca_global(latent, output / "pca_global_structure.png", args.seed)
    ratios = plot_explained_variance(latent, output / "pca_explained_variance.png")
    analysis_pca, analysis_features = fit_analysis_pca(latent)
    gmm, metrics = fit_gmms(
        analysis_features,
        k_min=args.gmm_k_min,
        k_max=args.gmm_k_max,
        n_init=args.gmm_n_init,
        seed=args.seed,
    )
    probabilities = gmm.predict_proba(analysis_features)
    labels = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    selected_k = int(gmm.n_components)
    stability_mean, stability_scores = gmm_stability(
        analysis_features,
        labels,
        components=selected_k,
        runs=args.stability_runs,
        seed=args.seed,
    )
    metrics["selected_by_bic"] = metrics.components == selected_k
    metrics.to_csv(output / "gmm_metrics.csv", index=False)
    plot_gmm_selection(metrics, output / "gmm_model_selection.png", selected_k)

    selected = stratified_indices(labels, args.tsne_max_samples, args.seed)
    assignments = pd.DataFrame(
        {
            "trajectory_id": trajectory_ids,
            "split": args.split,
            "cluster_id": labels,
            "cluster_probability": confidence,
            "tsne_included": np.isin(np.arange(len(latent)), selected),
        }
    )
    assignments.to_parquet(output / "gmm_assignments.parquet", index=False)
    tsne_records = plot_tsne(
        analysis_features[selected],
        labels[selected],
        confidence[selected],
        trajectory_ids[selected],
        perplexities=list(args.tsne_perplexities),
        threshold=args.confidence_threshold,
        seed=args.seed,
        output=output / "tsne_gmm_clusters.png",
    )
    tsne_records.to_parquet(output / "tsne_coordinates.parquet", index=False)

    cumulative = np.cumsum(ratios)
    diagnostics = {
        "sample_count": int(len(latent)),
        "latent_dimension": int(latent.shape[1]),
        "latent_norm_mean": float(np.linalg.norm(latent, axis=1).mean()),
        "latent_norm_std": float(np.linalg.norm(latent, axis=1).std()),
        "latent_dimension_mean": latent.mean(axis=0).tolist(),
        "latent_dimension_std": latent.std(axis=0).tolist(),
        "pca_two_component_explained_variance": float(
            pca2.explained_variance_ratio_.sum()
        ),
        "pca_dimensions_for_90_percent": int(np.searchsorted(cumulative, 0.90) + 1),
        "pca_dimensions_for_95_percent": int(np.searchsorted(cumulative, 0.95) + 1),
        "pca_dimensions_for_99_percent": int(np.searchsorted(cumulative, 0.99) + 1),
        "gmm_fit_dimension": int(analysis_features.shape[1]),
        "gmm_covariance_type": "diag",
        "gmm_selected_components_by_bic": selected_k,
        "gmm_stability_mean_adjusted_rand_index": stability_mean,
        "gmm_stability_adjusted_rand_indices": stability_scores,
        "gmm_mean_assignment_probability": float(confidence.mean()),
        "gmm_low_confidence_fraction": float(
            (confidence < args.confidence_threshold).mean()
        ),
        "cluster_counts": {
            str(index): int(value)
            for index, value in enumerate(np.bincount(labels, minlength=selected_k))
        },
    }
    (output / "latent_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf8"
    )
    manifest = {
        "analysis": "tcvae_train_posterior_mean_latent_visualization",
        "run": str(run),
        "artifact": str(artifact),
        "split": args.split,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_selection_note": (
            f"Explicit {checkpoint.name} checkpoint used at epoch {epoch}."
        ),
        "seed": args.seed,
        "latent_representation": "posterior_mean",
        "gmm_k_range": [args.gmm_k_min, args.gmm_k_max],
        "gmm_selected_components_by_bic": selected_k,
        "gmm_fit_space": (
            f"PCA retaining >=95% train latent variance "
            f"({analysis_pca.n_components_} dimensions), without whitening"
        ),
        "tsne_perplexities": list(args.tsne_perplexities),
        "tsne_sample_count": int(len(selected)),
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    print(f"OUTPUT={output}")
    print(f"CHECKPOINT_EPOCH={epoch}")
    print(f"SAMPLES={len(latent)} LATENT_DIM={latent.shape[1]}")
    print(f"GMM_SELECTED_K={selected_k}")
    print(f"GMM_STABILITY_ARI={stability_mean:.6f}")


if __name__ == "__main__":
    main()
