"""Compare held-out posterior, VampPrior, and DDPM latents with one decoder."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from chabuduo.artifacts import Normalization, SplitData, sha256
from chabuduo.config import load_experiment_config
from chabuduo.models.tcvae import TCVAELightning
from chabuduo.plotting import _anchor_xy, _draw_trajectories, _trajectory_groups
from chabuduo.prepare import airport_projection
from chabuduo.sampling import build_generated_traffic

root = Path.cwd()
tcvae_name = "zurich_rwy14_tcvae_vamp_k1000"
ddpm_name = "zurich_rwy14_latent_ddpm_vamp_k1000"
tcvae_run = root / "runs" / f"{tcvae_name}-v1-seed-42"
ddpm_run = root / "runs" / f"{ddpm_name}-v1-seed-42"
artifact = root / "data" / "processed" / f"{tcvae_name}-v1"
checkpoint = tcvae_run / "checkpoints" / "best.ckpt"
latent_artifact = ddpm_run / "artifacts" / "latent_dataset.npz"
latent_manifest = ddpm_run / "artifacts" / "latent_manifest.json"
ddpm_sample = ddpm_run / "samples" / "seed-42-n-500-steps-1000-eta-0" / "samples.npz"
output = tcvae_run / "diagnostics" / "three_latent_sources_n-500_steps-1000_seed-42"
if output.exists():
    raise FileExistsError(output)

manifest = json.loads(latent_manifest.read_text(encoding="utf8"))
if manifest["tcvae_checkpoint"]["sha256"] != sha256(checkpoint):
    raise ValueError("Latent artifact references a different TCVAE checkpoint")
if manifest["prepared_artifact"]["files"]["test.npz"] != sha256(artifact / "test.npz"):
    raise ValueError("Latent artifact references a different test split")

with np.load(latent_artifact, allow_pickle=False) as arrays:
    train_z = arrays["train_latent"].astype(np.float32)
    posterior_z = arrays["test_latent"][:500].astype(np.float32)
    latent_test_ids = arrays["test_trajectory_id"][:500]
with np.load(ddpm_sample, allow_pickle=False) as arrays:
    ddpm_z = arrays["generated_latent"][:500].astype(np.float32)
    ddpm_standardized = arrays["generated_latent_standardized"][:500]
if train_z.shape != (9814, 64) or posterior_z.shape != (500, 64):
    raise ValueError("Unexpected cached posterior latent shape")
if ddpm_z.shape != (500, 64):
    raise ValueError("Unexpected DDPM latent shape")

test = SplitData.load(artifact / "test.npz")
if not np.array_equal(latent_test_ids.astype(str), test.trajectory_id[:500]):
    raise ValueError("Cached posterior latents do not align with test trajectories")
latent_norm = json.loads(
    (ddpm_run / "artifacts" / "latent_normalization.json").read_text(encoding="utf8")
)
train_mean = np.asarray(latent_norm["mean"], dtype=np.float32)
train_std = np.asarray(latent_norm["std"], dtype=np.float32)
if not np.allclose((ddpm_z - train_mean) / train_std, ddpm_standardized, atol=1e-5):
    raise ValueError("DDPM latent standardization does not match its saved artifact")

module = TCVAELightning.load_from_checkpoint(checkpoint, map_location="cpu")
module.eval()
module.requires_grad_(False)
with torch.inference_mode():
    component_mean, component_log_variance = module.network.prior_parameters()
    generator = torch.Generator(device="cpu").manual_seed(42)
    component = torch.randint(len(component_mean), (500,), generator=generator)
    noise = torch.randn((500, 64), generator=generator)
    vamp_z = (
        component_mean[component]
        + torch.exp(0.5 * component_log_variance[component]) * noise
    ).numpy()

sources = {
    "Test posterior": posterior_z,
    "VampPrior": vamp_z,
    "Latent DDPM": ddpm_z,
}
colors = {
    "Test posterior": "#4C78A8",
    "VampPrior": "#54A24B",
    "Latent DDPM": "#E45756",
}

train_standardized = (train_z - train_mean) / train_std
standardized = {name: (z - train_mean) / train_std for name, z in sources.items()}
neighbors = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1)
neighbors.fit(train_standardized)
nearest = {
    name: neighbors.kneighbors(value, return_distance=True)[0][:, 0]
    for name, value in standardized.items()
}

with torch.inference_mode():

    def vamp_log_density(z: np.ndarray) -> np.ndarray:
        scores = []
        for start in range(0, len(z), 100):
            batch = torch.from_numpy(z[start : start + 100])
            component_log_probability = module.network._diagonal_gaussian_log_prob(
                batch[:, None, :],
                component_mean[None, :, :],
                component_log_variance[None, :, :],
            )
            scores.append(
                (
                    torch.logsumexp(component_log_probability, dim=1)
                    - np.log(len(component_mean))
                ).numpy()
            )
        return np.concatenate(scores)

    log_density = {name: vamp_log_density(z) for name, z in sources.items()}
    decoded = {
        name: module.network.decode(torch.from_numpy(z[:100])).permute(0, 2, 1).numpy()
        for name, z in sources.items()
    }

pca = PCA(n_components=2, random_state=42)
train_pca = pca.fit_transform(train_standardized)
projected = {name: pca.transform(value) for name, value in standardized.items()}
all_projected = np.concatenate([train_pca, *projected.values()])
lower = np.quantile(all_projected, 0.001, axis=0)
upper = np.quantile(all_projected, 0.999, axis=0)
padding = 0.05 * (upper - lower)

figure, axes = plt.subplots(2, 3, figsize=(17, 10))
for axis, (name, value) in zip(axes[0], projected.items(), strict=True):
    axis.scatter(
        train_pca[:, 0],
        train_pca[:, 1],
        s=4,
        alpha=0.055,
        color="0.45",
        rasterized=True,
    )
    axis.scatter(
        value[:, 0], value[:, 1], s=12, alpha=0.5, color=colors[name], rasterized=True
    )
    axis.set_title(f"{name} z (n=500)")
    axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axis.grid(alpha=0.15)
metric_distributions = (
    (
        "Standardized z norm",
        {name: np.linalg.norm(value, axis=1) for name, value in standardized.items()},
    ),
    ("Nearest train z distance", nearest),
    ("log VampPrior density", log_density),
)
for axis, (title, values) in zip(axes[1], metric_distributions, strict=True):
    axis.boxplot(
        list(values.values()),
        tick_labels=list(values),
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#E6E6E6"},
    )
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.2)
figure.suptitle("Same 64D latent space; train posterior in gray")
figure.tight_layout(rect=(0, 0, 1, 0.96))

config = load_experiment_config(
    root / "src" / "chabuduo" / "configs" / f"{ddpm_name}.yaml"
)
normalization = Normalization.load(artifact / "normalization.json")
real_values = normalization.inverse_transform(test.values[:100])
real = build_generated_traffic(real_values, test.elapsed_time[:100], config)
projection = airport_projection(config.preprocess.airport)
if not {"x", "y"} <= set(real.data.columns):
    real = real.compute_xy(projection=projection)
groups = {"Held-out test": _trajectory_groups(real.data, 100)}
for name, value in decoded.items():
    unnormalized = normalization.inverse_transform(value)
    t = build_generated_traffic(unnormalized, unnormalized[:, :, 3], config)
    if not {"x", "y"} <= set(t.data.columns):
        t = t.compute_xy(projection=projection)
    groups[name] = _trajectory_groups(t.data, 100)
if any(len(value) != 100 for value in groups.values()):
    raise ValueError("Expected 100 trajectories in every panel")

geo_figure, geo_axes = plt.subplots(2, 2, figsize=(13, 12), sharex=True, sharey=True)
geo_colors = {"Held-out test": "#4C78A8", **colors}
for axis, (name, value) in zip(geo_axes.flat, groups.items(), strict=True):
    _draw_trajectories(axis, value, color=geo_colors[name], title=name)
all_x = np.concatenate(
    [g.x.to_numpy() / 1000 for value in groups.values() for g in value]
)
all_y = np.concatenate(
    [g.y.to_numpy() / 1000 for value in groups.values() for g in value]
)
x_margin = max(0.5, 0.04 * float(np.ptp(all_x)))
y_margin = max(0.5, 0.04 * float(np.ptp(all_y)))
anchor_x, anchor_y, anchor_label = _anchor_xy(config)
for axis in geo_axes.flat:
    axis.scatter(
        [anchor_x],
        [anchor_y],
        marker="x",
        color="black",
        s=35,
        label=anchor_label,
        zorder=3,
    )
    axis.set_xlim(float(all_x.min()) - x_margin, float(all_x.max()) + x_margin)
    axis.set_ylim(float(all_y.min()) - y_margin, float(all_y.max()) + y_margin)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")
geo_axes[0, 0].set_ylabel("North of airport (km)")
geo_axes[1, 0].set_ylabel("North of airport (km)")
geo_figure.suptitle("100 trajectories per source, one frozen TCVAE decoder")
geo_figure.tight_layout(rect=(0, 0, 1, 0.96))

summary = {
    "tcvae_checkpoint": str(checkpoint),
    "tcvae_sha256": sha256(checkpoint),
    "ddpm_sample": str(ddpm_sample),
    "ddpm_sample_sha256": sha256(ddpm_sample),
    "latent_count_per_source": 500,
    "decoded_trajectory_count_per_source": 100,
    "posterior_source": "cached fixed-seed test posterior draws",
    "vamp_source": "uniform component; Gaussian draw; seed 42",
    "ddpm_source": "existing EMA DDIM eta=0, 1000-step sample; seed 42",
    "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "metrics": {
        name: {
            "standardized_norm_median": float(
                np.median(np.linalg.norm(standardized[name], axis=1))
            ),
            "nearest_train_z_median": float(np.median(nearest[name])),
            "nearest_train_z_p90": float(np.percentile(nearest[name], 90)),
            "vamp_log_density_median": float(np.median(log_density[name])),
            "vamp_log_density_p10": float(np.percentile(log_density[name], 10)),
            "vamp_log_density_p90": float(np.percentile(log_density[name], 90)),
        }
        for name in sources
    },
    "test_posterior_normalized_reconstruction_mse_first_100": float(
        np.mean((decoded["Test posterior"] - test.values[:100]) ** 2)
    ),
}
output.mkdir(parents=True)
figure.savefig(output / "latent_comparison.png", dpi=180, bbox_inches="tight")
geo_figure.savefig(output / "trajectory_comparison.png", dpi=180, bbox_inches="tight")
plt.close(figure)
plt.close(geo_figure)
np.savez_compressed(
    output / "latent_samples.npz",
    posterior=posterior_z,
    vamp=vamp_z,
    ddpm=ddpm_z,
    test_trajectory_id=test.trajectory_id[:500],
)
(output / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf8"
)
print(json.dumps(summary, indent=2))
print(output)
