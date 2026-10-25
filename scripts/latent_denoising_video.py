"""Record a frozen sampler's real states, then render them in a fixed PCA view."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(root: Path, *, count: int, steps: int, seed: int) -> None:
    import torch
    from sklearn.decomposition import PCA

    from chabuduo.artifacts import Normalization
    from chabuduo.models.latent_ddpm import _latent_model_from_state
    from chabuduo.models.tcvae import TCVAELightning

    destination = root / f"recording-n-{count}-steps-{steps}-seed-{seed}"
    if destination.exists():
        raise FileExistsError(destination)
    inputs = root / "inputs"
    ddpm = inputs / "ddpm" / "best.pt"
    tcvae = inputs / "tcvae" / "best.ckpt"
    artifacts = inputs / "ddpm" / "artifacts"
    prepared = inputs / "processed" / "zurich_rwy14_tcvae_vamp_k1000-v1"
    reference = inputs / "reference" / "three_latent_sources_n-500_steps-1000_seed-42"
    manifest = json.loads((artifacts / "latent_manifest.json").read_text())
    expected_hash = manifest["tcvae_checkpoint"]["sha256"]
    if digest(tcvae) != expected_hash:
        raise ValueError("TCVAE checkpoint does not match the latent dataset")
    for name, expected in manifest["prepared_artifact"]["files"].items():
        if digest(prepared / name) != expected:
            raise ValueError(f"Prepared file hash differs: {name}")

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    state = torch.load(ddpm, map_location="cpu", weights_only=False)
    if state["tcvae_checkpoint_sha256"] != expected_hash:
        raise ValueError("DDPM and TCVAE checkpoints do not match")
    model = _latent_model_from_state(state).eval()
    model.load_state_dict(
        state.get("ema_model_state_dict") or state["model_state_dict"]
    )
    snapshots: list[np.ndarray] = []
    diffusion_indices: list[int] = []

    def capture_input(module, args):
        # The hook only observes inputs. The original sampler performs all updates.
        snapshots.append(args[0].detach().cpu().numpy().copy())
        diffusion_indices.append(int(args[1][0]))
        if len(snapshots) % 100 == 0:
            print(f"Captured {len(snapshots)}/{steps} sampler inputs", flush=True)

    handle = model.denoiser.register_forward_pre_hook(capture_input)
    torch.manual_seed(seed)
    try:
        generated = model.sample(count, steps=steps, eta=0.0)
    finally:
        handle.remove()
    snapshots.append(generated.copy())
    diffusion_indices.append(-1)  # Terminal clean state, after index 0's update.
    history = np.stack(snapshots)
    if history.shape != (steps + 1, count, 64):
        raise ValueError(f"Unexpected history shape: {history.shape}")
    if not np.isfinite(history).all():
        raise ValueError("Nonfinite sampler states")

    # Confirm observation has not changed the original sampler's output.
    torch.manual_seed(seed)
    baseline = model.sample(count, steps=steps, eta=0.0)
    if not np.array_equal(generated, baseline):
        raise ValueError("Recorded output differs from the unmodified sampler")

    norm = json.loads((artifacts / "latent_normalization.json").read_text())
    mean = np.asarray(norm["mean"], dtype=np.float32)
    scale = np.asarray(norm["std"], dtype=np.float32)
    np.testing.assert_array_equal(mean, np.asarray(state["latent_mean"]))
    np.testing.assert_array_equal(scale, np.asarray(state["latent_std"]))
    with np.load(artifacts / "latent_dataset.npz") as data:
        train = (data["train_latent"].astype(np.float32) - mean) / scale
    pca = PCA(n_components=2, random_state=42)
    background = pca.fit_transform(train)
    expected = json.loads((reference / "summary.json").read_text())
    np.testing.assert_allclose(
        pca.explained_variance_ratio_,
        expected["pca_explained_variance_ratio"],
        atol=2e-6,
        rtol=2e-5,
    )
    xy = pca.transform(history.reshape(-1, 64)).reshape(steps + 1, count, 2)
    with np.load(reference / "latent_samples.npz") as data:
        previous_views = [
            pca.transform((data[key] - mean) / scale)
            for key in ("posterior", "vamp", "ddpm")
        ]
    original_view = np.concatenate([background, *previous_views])
    lower = np.quantile(original_view, 0.001, axis=0)
    upper = np.quantile(original_view, 0.999, axis=0)
    padding = 0.05 * (upper - lower)
    original_bounds = np.stack([lower - padding, upper + padding])
    # Keep one camera for the whole film, including every generated particle.
    lower = np.minimum(original_bounds[0], xy.min(axis=(0, 1)) - 0.4)
    upper = np.maximum(original_bounds[1], xy.max(axis=(0, 1)) + 0.4)

    latent = generated * scale + mean
    module = TCVAELightning.load_from_checkpoint(
        tcvae, map_location="cpu", weights_only=False
    ).eval()
    module.requires_grad_(False)
    with torch.inference_mode():
        normalized = (
            module.network.decode(torch.from_numpy(latent))
            .permute(0, 2, 1)
            .numpy()
            .astype(np.float32)
        )
    values = Normalization.load(prepared / "normalization.json").inverse_transform(
        normalized
    )
    if values.shape != (count, 200, 4) or not np.isfinite(values).all():
        raise ValueError("Invalid decoded trajectories")
    ids = np.asarray([f"sample_{i + 1:03d}" for i in range(count)])
    destination.mkdir()
    np.savez_compressed(
        destination / "denoising_states.npz",
        standardized_latent=history,
        diffusion_index=np.asarray(diffusion_indices),
        completed_updates=np.arange(steps + 1),
        sample_id=ids,
        projected_xy=xy,
        train_projected_xy=background,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        latent_mean=mean,
        latent_std=scale,
        original_axis_bounds=original_bounds,
        video_axis_bounds=np.stack([lower, upper]),
    )
    np.savez_compressed(
        destination / f"generated_{count}_trajectories.npz",
        sample_id=ids,
        generated_latent=latent,
        generated_latent_standardized=generated,
        normalized_values=normalized,
        values=values,
        feature_names=np.asarray(
            ["track_unwrapped", "groundspeed_kt", "altitude_ft", "elapsed_time_sec"]
        ),
        units=np.asarray(["degree", "knot", "ft", "s"]),
    )
    import pandas as pd

    frame = pd.DataFrame(
        values.reshape(-1, 4),
        columns=[
            "track_unwrapped_deg",
            "groundspeed_kt",
            "altitude_ft",
            "elapsed_time_sec",
        ],
    )
    frame.insert(0, "point_index", np.tile(np.arange(200), count))
    frame.insert(0, "sample_id", np.repeat(ids, 200))
    frame.to_csv(destination / f"generated_{count}_trajectories.csv", index=False)
    summary = {
        "count": count,
        "steps": steps,
        "seed": seed,
        "eta": 0.0,
        "sampler": "Original LatentDDPM.sample; EMA weights; DDIM eta=0",
        "execution": "local CPU; deterministic; no training",
        "history_shape": list(history.shape),
        "terminal_state_index": -1,
        "all_states_finite": True,
        "unmodified_sampler_bitwise_equal": True,
        "tcvae_sha256": digest(tcvae),
        "ddpm_sha256": digest(ddpm),
        "tcvae_epoch": manifest["tcvae_checkpoint"]["epoch"],
        "ddpm_epoch": state.get("epoch"),
        "latent_source": manifest["representation"],
        "projection": "Original train-standardized PCA(2, random_state=42)",
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "original_axis_bounds": original_bounds.tolist(),
        "video_axis_bounds": np.stack([lower, upper]).tolist(),
        "normalization_fit": "training split only",
        "decoded_feature_order": frame.columns.tolist()[2:],
        "decoded_shape": list(values.shape),
        "trajectory_direction": "chronological",
        "trajectory_endpoint_convention": "runway 14 threshold on reconstruction",
        "trajectory_projection": "none; raw kinematic feature sequences",
        "postprocessing": "none; no clipping or elapsed-time repair",
        "note": "New samples; not asserted equal to prior remote 500 samples",
        "limitations": "2D projection does not establish 64D or physical realism",
        "source_file_hashes": {
            str(Path(sys.modules[name].__file__).resolve()): digest(
                Path(sys.modules[name].__file__)
            )
            for name in (
                "chabuduo.models.latent_ddpm", "chabuduo.models.tcvae",
                "chabuduo.models.ddpm", "chabuduo.artifacts",
            )
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            **{
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "matplotlib",
                    "lightning",
                )
            },
        },
    }
    (destination / "recording_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(destination, flush=True)


def render(recording: Path, *, smoke: bool) -> None:
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "font.size": 12,
        }
    )
    destination = recording / "video_10s_en_green_white"
    destination.mkdir(exist_ok=True)
    with np.load(recording / "denoising_states.npz") as data:
        xy = data["projected_xy"]
        background = data["train_projected_xy"]
        bounds = data["video_axis_bounds"]
        ratios = data["explained_variance_ratio"]
    updates, count = xy.shape[0] - 1, xy.shape[1]
    green = "#2CA02C"  # Latent DDPM green sampled from the supplied reference.
    paper, ink, muted = "#FFFFFF", "#182D3A", "#61717C"
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=paper)
    ax = fig.add_axes((0.06, 0.17, 0.89, 0.73), facecolor=paper)
    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    zero_x, zero_y = -bounds[0] / (bounds[1] - bounds[0])
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_position(("data", 0))
        ax.spines[name].set_color("#8B9A9E")
        ax.spines[name].set_linewidth(0.9)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_yticks([-5, 5, 10])
    ax.set_xlabel(f"PC1 ({ratios[0]:.1%})", color=ink, weight="bold", ha="right")
    ax.set_ylabel(
        f"PC2 ({ratios[1]:.1%})", color=ink, weight="bold", rotation=0, ha="left"
    )
    ax.xaxis.set_label_coords(1, zero_y - 0.065)
    ax.yaxis.set_label_coords(zero_x, 1.025)
    ax.tick_params(colors=muted, labelsize=11)
    ax.grid(color="#E3E7E5", alpha=0.7, linewidth=0.6, zorder=0)
    ax.scatter(
        background[:, 0],
        background[:, 1],
        s=7,
        c="#808080",
        alpha=0.18,
        linewidths=0,
        zorder=1,
    )
    fig.text(0.06, 0.945, "● Training latents", color="#808080", fontsize=12)
    fig.text(0.24, 0.945, "● Latent DDPM", color=green, fontsize=12)
    particles = ax.scatter(
        [],
        [],
        s=30,
        c=green,
        alpha=0.96,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
        animated=True,
    )
    trails = LineCollection([], linewidths=1.1, zorder=3, animated=True)
    ax.add_collection(trails)
    progress_ax = fig.add_axes((0.06, 0.065, 0.89, 0.009))
    progress_ax.set_xlim(0, updates)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis("off")
    progress_ax.add_patch(Rectangle((0, 0), updates, 1, color="#DCE3E1"))
    bar = Rectangle((0, 0), 0, 1, color=green, animated=True)
    progress_ax.add_patch(bar)
    counter = fig.text(0.06, 0.095, "", color=ink, fontsize=12, animated=True)
    fig.canvas.draw()
    static = fig.canvas.copy_from_bbox(fig.bbox)
    import matplotlib.colors as mcolors

    def draw(index: int) -> np.ndarray:
        fig.canvas.restore_region(static)
        current = xy[index]
        particles.set_offsets(current)
        segments, rgba = [], []
        start = max(0, index - 90)
        for sample in range(count):
            path = xy[start : index + 1, sample]
            if len(path) > 1:
                segments.extend(np.stack([path[:-1], path[1:]], axis=1))
                rgba.extend(
                    [
                        mcolors.to_rgba(green, float(alpha))
                        for alpha in np.linspace(0.025, 0.4, len(path) - 1)
                    ]
                )
        trails.set_segments(segments)
        trails.set_color(rgba)
        counter.set_text(f"Denoising update  {index:04d} / {updates:,}")
        bar.set_width(index)
        for artist in (trails, particles):
            ax.draw_artist(artist)
        progress_ax.draw_artist(bar)
        fig.draw_artist(counter)
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    # Subsample real states; keep endpoint holds and a total duration of 10 seconds.
    fps = 30
    sampled_indices = np.linspace(0, updates, 256).round().astype(int).tolist()
    indices = [0] * 15 + sampled_indices[1:] + [updates] * 30
    if smoke:
        indices = [0, updates // 4, updates // 2, updates]
    output = destination / (
        "preview.mp4" if smoke else f"latent_denoising_{count}_10s_en.mp4"
    )
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1920, 1080),
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-preset", "medium", "-crf", "18", "-movflags", "+faststart"],
    )
    writer.send(None)
    from PIL import Image

    saved = set()
    key_indices = {sampled_indices[index] for index in (0, 51, 102, 153, 204, 255)}
    frame_directory = destination / "frames"
    frame_directory.mkdir(exist_ok=True)
    try:
        for frame_number, index in enumerate(indices):
            pixels = draw(index)
            writer.send(pixels)
            if index in key_indices and index not in saved:
                Image.fromarray(pixels).save(frame_directory / f"step_{index:04d}.png")
                saved.add(index)
            if frame_number == 0:
                Image.fromarray(pixels).save(destination / "first_frame.png")
            if index == updates and (frame_number == len(indices) - 1):
                Image.fromarray(pixels).save(destination / "final_frame.png")
            if frame_number % 100 == 0:
                print(f"Rendered {frame_number}/{len(indices)} frames", flush=True)
    finally:
        writer.close()
    plt.close(fig)
    render_summary = {
        "fps": fps,
        "frame_count": len(indices),
        "resolution": [1920, 1080],
        "duration_seconds": len(indices) / fps,
        "position_interpolation": False,
        "language": "English",
        "source_recording": str(recording / "denoising_states.npz"),
        "source_recording_sha256": digest(recording / "denoising_states.npz"),
        "recorded_state_count": len(xy),
        "displayed_unique_state_count": len(set(indices)),
        "recorded_state_indices_per_frame": indices,
        "trail_sample_ids": [f"sample_{i + 1:03d}" for i in range(count)],
        "particle_color": green,
        "axes_cross_at": [0, 0],
        "visible_text": [
            "Training latents",
            "Latent DDPM",
            "PC1",
            "PC2",
            "Denoising update",
        ],
        "background_alpha": 0.18,
        "particle_alpha": 0.96,
    }
    (
        destination / ("preview_summary.json" if smoke else "video_summary.json")
    ).write_text(json.dumps(render_summary, indent=2) + "\n", encoding="utf-8")
    print(output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recording = commands.add_parser("record")
    recording.add_argument("--root", type=Path, required=True)
    recording.add_argument("--count", type=int, default=100)
    recording.add_argument("--steps", type=int, default=1000)
    recording.add_argument("--seed", type=int, default=42)
    recording.add_argument("--smoke", action="store_true")
    rendering = commands.add_parser("render")
    rendering.add_argument("--recording", type=Path, required=True)
    rendering.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "record":
        record(
            args.root.resolve(),
            count=4 if args.smoke else args.count,
            steps=10 if args.smoke else args.steps,
            seed=args.seed,
        )
    else:
        render(args.recording.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
