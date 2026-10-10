"""One-off diagnostic for posterior reconstruction of held-out Zurich trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from chabuduo.artifacts import Normalization, SplitData, sha256
from chabuduo.config import load_experiment_config
from chabuduo.models.tcvae import TCVAELightning
from chabuduo.plotting import _anchor_xy, _draw_trajectories, _trajectory_groups
from chabuduo.prepare import airport_projection
from chabuduo.sampling import build_generated_traffic

root = Path.cwd()
name = "zurich_rwy14_tcvae_vamp_k1000"
config = load_experiment_config(root / "src" / "chabuduo" / "configs" / f"{name}.yaml")
artifact = root / "data" / "processed" / f"{name}-v1"
checkpoint = root / "runs" / f"{name}-v1-seed-42" / "checkpoints" / "best.ckpt"
output = (
    root
    / "runs"
    / f"{name}-v1-seed-42"
    / "diagnostics"
    / "test_posterior_reconstruction_seed-42-n-100"
)
if output.exists():
    raise FileExistsError(output)

test = SplitData.load(artifact / "test.npz")
normalization = Normalization.load(artifact / "normalization.json")
normalized = test.values[:100]
module = TCVAELightning.load_from_checkpoint(checkpoint, map_location="cpu")
module.eval()
with torch.inference_mode():
    value = torch.from_numpy(normalized).permute(0, 2, 1)
    mean, log_variance = module.network.encode(value)
    generator = torch.Generator(device="cpu").manual_seed(42)
    noise = torch.randn(mean.shape, generator=generator)
    latent = mean + torch.exp(0.5 * log_variance) * noise
    reconstructed = module.network.decode(latent).permute(0, 2, 1).numpy()

real_values = normalization.inverse_transform(normalized)
reconstructed_values = normalization.inverse_transform(reconstructed)
real = build_generated_traffic(real_values, test.elapsed_time[:100], config)
reconstruction = build_generated_traffic(
    reconstructed_values, reconstructed_values[:, :, 3], config
)
projection = airport_projection(config.preprocess.airport)
if not {"x", "y"} <= set(real.data.columns):
    real = real.compute_xy(projection=projection)
if not {"x", "y"} <= set(reconstruction.data.columns):
    reconstruction = reconstruction.compute_xy(projection=projection)
real_groups = _trajectory_groups(real.data, 100)
reconstructed_groups = _trajectory_groups(reconstruction.data, 100)
if len(real_groups) != 100 or len(reconstructed_groups) != 100:
    raise ValueError("Expected 100 real and 100 reconstructed trajectories")

figure, axes = plt.subplots(1, 2, figsize=(15, 7.2), sharex=True, sharey=True)
_draw_trajectories(axes[0], real_groups, color="#4C78A8", title="Held-out test")
_draw_trajectories(
    axes[1], reconstructed_groups, color="#E45756", title="Posterior reconstruction"
)
all_x = np.concatenate(
    [group.x.to_numpy() / 1000 for group in real_groups + reconstructed_groups]
)
all_y = np.concatenate(
    [group.y.to_numpy() / 1000 for group in real_groups + reconstructed_groups]
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
figure.suptitle("Zurich RWY14: 100 paired test trajectories, sampled posterior z")
figure.tight_layout(rect=(0, 0, 1, 0.95))

point_errors = []
for real_group, reconstructed_group in zip(
    real_groups, reconstructed_groups, strict=True
):
    real_xy = real_group[["x", "y"]].to_numpy()
    reconstructed_xy = reconstructed_group[["x", "y"]].to_numpy()
    if real_xy.shape != reconstructed_xy.shape:
        raise ValueError("Paired trajectories have different lengths")
    point_errors.append(np.linalg.norm(real_xy - reconstructed_xy, axis=1) / 1000)
point_errors = np.stack(point_errors)

output.mkdir(parents=True)
figure.savefig(output / "trajectory_comparison.png", dpi=200, bbox_inches="tight")
plt.close(figure)
np.savez_compressed(
    output / "reconstructions.npz",
    trajectory_id=test.trajectory_id[:100],
    real_values=real_values,
    reconstructed_values=reconstructed_values,
    latent=latent.numpy(),
)
summary = {
    "split": "test",
    "count": 100,
    "seed": 42,
    "latent": "one posterior sample per trajectory",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "normalized_feature_mse": float(np.mean((normalized - reconstructed) ** 2)),
    "pointwise_xy_error_km_median": float(np.median(point_errors)),
    "pointwise_xy_error_km_p90": float(np.percentile(point_errors, 90)),
    "outputs": ["trajectory_comparison.png", "reconstructions.npz"],
}
(output / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf8"
)
print(json.dumps(summary, indent=2))
print(output)
