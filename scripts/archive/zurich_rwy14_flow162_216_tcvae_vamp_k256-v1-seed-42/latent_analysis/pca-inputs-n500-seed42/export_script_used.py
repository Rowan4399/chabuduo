"""Export VampPrior latent samples, then plot a training-fitted PCA separately."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(args: argparse.Namespace) -> None:
    import inspect

    import torch

    from chabuduo.models.tcvae import TCVAELightning

    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.run / "checkpoints/best.ckpt"
    run_summary = json.loads((args.run / "run_summary.json").read_text())
    if digest(checkpoint) != run_summary["best"]["sha256"]:
        raise ValueError("Checkpoint does not match the recorded best checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["hyper_parameters"]["resolved_config"]
    prior = config["model"]["prior"]
    if prior["type"] != "vamp" or prior["components"] != 256:
        raise ValueError("Expected the requested VampPrior K=256 checkpoint")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = (
        TCVAELightning.load_from_checkpoint(
            checkpoint, map_location=device, weights_only=False
        )
        .to(device)
        .eval()
    )
    module.requires_grad_(False)
    arrays = {}
    input_hashes = {}
    for split in ("train", "test"):
        path = args.artifact / f"{split}.npz"
        if digest(path) != run_summary["artifact"]["files"][f"{split}.npz"]:
            raise ValueError(f"{split} differs from this model's training artifact")
        with np.load(path, allow_pickle=False) as data:
            values = data["values"].astype(np.float32)
            ids = data["trajectory_id"].astype(str)
        means, log_variances = [], []
        with torch.inference_mode():
            for start in range(0, len(values), 256):
                batch = torch.from_numpy(values[start : start + 256]).to(device)
                mean, log_variance = module.network.encode(batch.permute(0, 2, 1))
                means.append(mean.cpu().numpy())
                log_variances.append(log_variance.cpu().numpy())
        arrays[f"{split}_mu"] = np.concatenate(means)
        arrays[f"{split}_log_variance"] = np.concatenate(log_variances)
        arrays[f"{split}_trajectory_id"] = ids
        input_hashes[f"{split}.npz"] = digest(path)
        print(f"{split}: {arrays[f'{split}_mu'].shape}", flush=True)
    if arrays["train_mu"].shape != (3054, 64):
        raise ValueError("Expected the 3,054-trajectory single-flow training split")
    if set(arrays["train_trajectory_id"]) & set(arrays["test_trajectory_id"]):
        raise ValueError("Training and test IDs overlap")
    indices = np.random.default_rng(args.seed).choice(
        len(arrays["test_mu"]), args.count, replace=False
    )
    arrays["test_indices"] = indices
    noise = np.random.default_rng(args.seed + 1).standard_normal((args.count, 64))
    arrays["test_sample"] = (
        arrays["test_mu"][indices]
        + np.exp(0.5 * arrays["test_log_variance"][indices]) * noise
    ).astype(np.float32)
    with torch.inference_mode():
        mean, log_variance = module.network.prior_parameters()
        mean, log_variance = mean.cpu(), log_variance.cpu()
        if mean.shape != (256, 64):
            raise ValueError("Unexpected learned prior shape")
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
        component = torch.randint(256, (args.count,), generator=generator)
        noise = torch.randn((args.count, 64), generator=generator)
        sample = mean[component] + torch.exp(0.5 * log_variance[component]) * noise
    arrays.update(
        {
            "vamp_sample": sample.numpy(),
            "vamp_component_id": component.numpy(),
            "vamp_component_mu": mean.numpy(),
            "vamp_component_log_variance": log_variance.numpy(),
        }
    )
    for name, values in arrays.items():
        if values.dtype.kind in "fiu" and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {name}")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "latents.npz", **arrays)
    source = Path(inspect.getfile(TCVAELightning))
    result = {
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_path": str(checkpoint.resolve()),
        "resolved_config": config,
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": input_hashes,
        "train_count": len(arrays["train_mu"]),
        "test_count": len(arrays["test_mu"]),
        "sample_count": args.count,
        "prior_components": 256,
        "sampled_unique_components": len(np.unique(component.numpy())),
        "prior_sampling": "uniform component, diagonal Gaussian draw, temperature 1",
        "seeds": {
            "test_selection": args.seed,
            "posterior": args.seed + 1,
            "vamp_prior": args.seed + 2,
        },
        "source_script_sha256": digest(Path(__file__)),
        "model_source_sha256": digest(source),
        "execution": f"remote frozen encoder inference on {device}; no training",
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "lightning")
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8"
    )
    print(f"EPOCH={state['epoch']} K=256 COUNT={args.count}", flush=True)
    print(f"OUTPUT={args.output.resolve()}")


def plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.decomposition import PCA

    if args.output.resolve() == args.input.resolve():
        raise ValueError("Keep input and plot directories separate")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    manifest = json.loads((args.input / "manifest.json").read_text())
    with np.load(args.input / "latents.npz", allow_pickle=False) as data:
        train = data["train_mu"].copy()
        train_ids = data["train_trajectory_id"].copy()
        indices = data["test_indices"]
        test_ids = data["test_trajectory_id"][indices]
        component_ids = data["vamp_component_id"].copy()
        sources = {
            "Test posterior mean": data["test_mu"][indices],
            "Test posterior sample": data["test_sample"].copy(),
            "TCVAE VampPrior samples (K=256)": data["vamp_sample"].copy(),
        }
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-6)
    standardized = (train - mean) / scale
    pca = PCA(n_components=2, random_state=args.seed)
    background = pca.fit_transform(standardized)
    np.testing.assert_allclose(
        background, pca.transform(standardized), atol=2e-4, rtol=2e-4
    )
    projected = {
        name: pca.transform((values - mean) / scale) for name, values in sources.items()
    }
    ratios = pca.explained_variance_ratio_
    all_xy = np.concatenate([background, *projected.values()])
    if not np.isfinite(all_xy).all():
        raise ValueError("Nonfinite PCA coordinates")

    def bounds(values):
        low, high = values.min(axis=0), values.max(axis=0)
        padding = 0.06 * (high - low)
        return low - padding, high + padding

    full, zoom = bounds(all_xy), bounds(background)
    outside = {
        name: int(np.any((xy < zoom[0]) | (xy > zoom[1]), axis=1).sum())
        for name, xy in projected.items()
    }
    colors = ("#2C73B9", "#008F82", "#E69526")
    markers = ("o", "^", "s")
    count = manifest["sample_count"]

    def setup(axis, limits):
        axis.scatter(
            *background.T,
            s=7,
            alpha=0.17,
            color="0.5",
            label=f"Training posterior means ({len(train):,})",
        )
        axis.set_xlim(limits[0][0], limits[1][0])
        axis.set_ylim(limits[0][1], limits[1][1])
        axis.set_xlabel(f"PC1 ({ratios[0]:.2%})")
        axis.set_ylabel(f"PC2 ({ratios[1]:.2%})")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.13)

    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, limits, title in zip(
        axes,
        (full, zoom),
        ("Full extent: all samples", "Training extent: zoom (outside points omitted)"),
        strict=True,
    ):
        setup(axis, limits)
        for (name, xy), color, marker in zip(
            projected.items(), colors, markers, strict=True
        ):
            axis.scatter(
                *xy.T, s=19, alpha=0.50, color=color, marker=marker, label=name
            )
        axis.set_title(title, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(
        f"Zurich single-flow TCVAE | VampPrior K=256 | epoch {manifest['checkpoint_epoch']}\n"
        f"PCA fitted on this model's training means | n={count} per source | "
        f"PC1 + PC2: {sum(ratios):.2%}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.91))
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "vamp_k256_pca_overlay.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    for axis, (name, xy), color, marker in zip(
        axes, projected.items(), colors, markers, strict=True
    ):
        setup(axis, full)
        axis.scatter(*xy.T, s=20, alpha=0.55, color=color, marker=marker)
        axis.set_title(f"{name}\nn={count}, all points", fontsize=12)
    figure.suptitle(
        f"VampPrior K=256 | same training-fitted PCA and axis limits | "
        f"PC1 + PC2: {sum(ratios):.2%}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(args.output / "vamp_k256_pca_sources.png", dpi=180)
    plt.close(figure)

    records = [
        pd.DataFrame(
            {
                "source": "Training posterior mean",
                "id": train_ids,
                "pc1": background[:, 0],
                "pc2": background[:, 1],
            }
        )
    ]
    for name, xy in projected.items():
        ids = test_ids if name.startswith("Test") else np.arange(count).astype(str)
        records.append(
            pd.DataFrame(
                {
                    "source": name,
                    "id": ids,
                    "pc1": xy[:, 0],
                    "pc2": xy[:, 1],
                }
            )
        )
    pd.concat(records).to_parquet(args.output / "coordinates.parquet", index=False)
    np.savez_compressed(
        args.output / "pca_transform.npz",
        latent_mean=mean,
        latent_std=scale,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        explained_variance_ratio=ratios,
    )
    result = {
        "input_directory": str(args.input.resolve()),
        "input_sha256": {
            name: digest(args.input / name) for name in ("latents.npz", "manifest.json")
        },
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_epoch": manifest["checkpoint_epoch"],
        "prior_components": 256,
        "sampled_unique_components": len(np.unique(component_ids)),
        "train_count": len(train),
        "sample_count_per_source": count,
        "pca_fit": "K=256 training posterior means only, standardized per dimension",
        "pca_seed": args.seed,
        "whitening": False,
        "explained_variance_ratio": ratios.tolist(),
        "full_bounds": [v.tolist() for v in full],
        "zoom_bounds": [v.tolist() for v in zoom],
        "outside_zoom": outside,
        "limits": "2D overlap is not proof of 64D agreement or physical realism",
        "source_script_sha256": digest(Path(__file__)),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "matplotlib", "pandas", "pyarrow")
        },
        "execution": "Local plotting; exported remote inference inputs copied locally",
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(result, indent=2))
    print(f"OUTPUT={args.output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--run", type=Path, required=True)
    exporter.add_argument("--artifact", type=Path, required=True)
    exporter.add_argument("--count", type=int, default=500)
    plotter = commands.add_parser("plot")
    plotter.add_argument("--input", type=Path, required=True)
    plotter.add_argument("--overwrite", action="store_true")
    for command in (exporter, plotter):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "export":
        export(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
