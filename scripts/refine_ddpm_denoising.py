"""Continue a cached DDIM tail and render extra late denoising stages."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from plot_ddpm_diffusion_stages import map_bounds, reconstruct

from chabuduo.artifacts import Normalization, sha256
from chabuduo.config import load_experiment_config
from chabuduo.models.ddpm import UnconditionalDDPM

EXTRA_STEPS = [200, 150, 100, 50, 25]
LATE_STEPS = [250, *EXTRA_STEPS, 0]


@torch.inference_mode()
def continue_tail(model, start: np.ndarray):
    device = next(model.parameters()).device
    values = torch.from_numpy(start.copy()).to(device)
    snapshots = {}
    # These are the existing sampler's eta=0 updates, starting from cached x_250.
    for timestep in range(250, 0, -1):
        if timestep in EXTRA_STEPS:
            snapshots[timestep] = values.cpu().numpy().copy()
            print(f"Recorded t={timestep}", flush=True)
        index = timestep - 1
        step = torch.full((len(values),), index, device=device, dtype=torch.long)
        noise = model.denoiser(values, step)
        alpha_bar = model.alpha_bars[index]
        previous_alpha_bar = (
            model.alpha_bars[index - 1] if index > 0 else torch.ones((), device=device)
        )
        predicted_clean = (values - (1 - alpha_bar).sqrt() * noise) / alpha_bar.sqrt()
        values = (
            previous_alpha_bar.sqrt() * predicted_clean
            + torch.clamp(1 - previous_alpha_bar, min=0).sqrt() * noise
        )
    snapshots[0] = values.cpu().numpy().copy()
    return snapshots


def record(args):
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.cache / "summary.json").read_text())
    checkpoint = args.run / "checkpoints/best.pt"
    if sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("The checkpoint differs from the original recording")
    if sha256(args.config) != summary["config_sha256"]:
        raise ValueError("The configuration differs from the original recording")
    with np.load(args.cache / "diffusion_states.npz") as cache:
        data = {key: cache[key].copy() for key in cache.files}
    original_steps = data["reverse_timesteps"].tolist()
    if original_steps != [1000, 750, 500, 250, 0]:
        raise ValueError("Expected the original five-stage cache")
    original = data["reverse_normalized_values"]
    if original.shape != (5, 100, 200, 4):
        raise ValueError(f"Unexpected original shape: {original.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = UnconditionalDDPM(
        state["feature_dim"],
        state["hidden_dim"],
        state["time_dim"],
        kernel_size=state["kernel_size"],
        dilations=tuple(state["dilations"]),
        dropout=state["dropout"],
        use_phase=state["use_phase"],
        diffusion_steps=state["diffusion_steps"],
    )
    model.load_state_dict(
        state.get("ema_model_state_dict") or state["model_state_dict"]
    )
    model.to(device).eval()
    np.testing.assert_array_equal(model.alpha_bars.cpu().numpy(), data["alpha_bars"])
    snapshots = continue_tail(model, original[original_steps.index(250)])
    np.testing.assert_array_equal(snapshots[0], original[-1])
    print("Terminal t=0 exactly matches the original 100 trajectories", flush=True)
    new_states = np.stack([snapshots[step] for step in EXTRA_STEPS])
    if not np.isfinite(new_states).all():
        raise ValueError("Nonfinite intermediate states")
    normalization = Normalization(data["normalization_mean"], data["normalization_std"])
    config = load_experiment_config(args.config)
    new_xy = reconstruct(
        new_states, normalization, data["reverse_fixed_elapsed_time"], config
    )
    new_full_xy = reconstruct(new_states, normalization, None, config)
    refined_steps = sorted([*original_steps, *EXTRA_STEPS], reverse=True)
    for key, added in [
        ("reverse_normalized_values", new_states),
        ("reverse_fixed_time_xy_km", new_xy),
        ("reverse_all_features_xy_km", new_full_xy),
    ]:
        by_step = dict(zip(original_steps, data[key], strict=True))
        by_step.update(dict(zip(EXTRA_STEPS, added, strict=True)))
        data[key] = np.stack([by_step[step] for step in refined_steps])
    data["reverse_timesteps"] = np.array(refined_steps)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "diffusion_states.npz", **data)
    shutil.copy2(args.cache / "generated_1000steps.parquet", args.output)
    report = {
        **summary,
        "source_cache": str(args.cache.resolve()),
        "source_cache_sha256": sha256(args.cache / "diffusion_states.npz"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "additional_timesteps": EXTRA_STEPS,
        "reverse_timesteps": refined_steps,
        "late_timesteps": LATE_STEPS,
        "state_shape": list(data["reverse_normalized_values"].shape),
        "replayed_reverse_updates": 250,
        "terminal_matches_original_exactly": True,
        "original_five_states_preserved_exactly": True,
        "runtime_seconds": round(perf_counter() - started, 2),
        "environment": {**summary["environment"], "device": str(device)},
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8"
    )
    print(f"OUTPUT={args.output.resolve()}", flush=True)


def draw_maps(xy, steps, bounds, output, title):
    columns = 4 if len(steps) == 7 else 5
    figure, axes = plt.subplots(
        2, columns, figsize=(5 * columns, 10), layout="constrained"
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, xy.shape[1]))
    for stage, timestep in enumerate(steps):
        axis = axes.flat[stage]
        for trajectory, color in zip(xy[stage], colors, strict=True):
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                alpha=0.30,
                linewidth=0.7,
            )
        axis.scatter([0], [0], marker="x", color="black", s=38, zorder=5)
        axis.set(
            xlim=(bounds[0][0], bounds[1][0]),
            ylim=(bounds[0][1], bounds[1][1]),
            xlabel="East of RWY 14 threshold (km)",
            ylabel="North of RWY 14 threshold (km)",
            title=f"t = {timestep}  |  n = 100",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18)
    note = (
        "Same 100 trajectories, colors and axes.\n"
        "Black cross: LSZH RWY 14 threshold.\n\n"
        "Actual intermediate DDIM states, eta=0.\n"
        "Started from the original cached t=250.\n"
        "Final t=0 matches the original exactly.\n\n"
        "Display timing held fixed per trajectory.\n"
        "Speed uses existing 30-600 kt bounds.\n"
        "Diffusion t is not flight time."
    )
    if len(steps) < len(axes.flat):
        axes.flat[-1].axis("off")
        axes.flat[-1].text(
            0.03,
            0.95,
            note,
            va="top",
            fontsize=12,
            linespacing=1.6,
            transform=axes.flat[-1].transAxes,
        )
    else:
        figure.supxlabel(textwrap.fill(note.replace("\n", " "), width=210), fontsize=10)
    figure.suptitle(title, fontsize=17)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render(output):
    with np.load(output / "diffusion_states.npz") as cache:
        steps = cache["reverse_timesteps"].tolist()
        states = cache["reverse_normalized_values"]
        xy = cache["reverse_fixed_time_xy_km"]
        forward_xy = cache["forward_fixed_time_xy_km"]
        features = cache["feature_order"].tolist()
    bounds = map_bounds(xy, forward_xy)
    late_indices = [steps.index(step) for step in LATE_STEPS]
    draw_maps(
        xy,
        steps,
        bounds,
        output / "denoising_10stages.png",
        "RWY 14 | 100 trajectories | 10 stages of 1000-step DDIM",
    )
    draw_maps(
        xy[late_indices],
        LATE_STEPS,
        bounds,
        output / "denoising_late_7stages.png",
        "RWY 14 | 100 trajectories | Late denoising: t=250 to t=0",
    )
    late = states[late_indices]
    figure, axes = plt.subplots(4, 7, figsize=(28, 11), layout="constrained")
    for feature, name in enumerate(features):
        low, high = late[:, :, :, feature].min(), late[:, :, :, feature].max()
        for stage, timestep in enumerate(LATE_STEPS):
            axis = axes[feature, stage]
            axis.plot(
                late[stage, :, :, feature].T,
                color="#3578a3",
                alpha=0.12,
                linewidth=0.55,
            )
            axis.set_ylim(low - 0.2, high + 0.2)
            axis.grid(alpha=0.15)
            if feature == 0:
                axis.set_title(f"t = {timestep}")
            if stage == 0:
                axis.set_ylabel(f"{name}\nstandardized")
            if feature == 3:
                axis.set_xlabel("Chronological sequence index")
    figure.suptitle(
        "100 trajectories: raw model features during late denoising", fontsize=16
    )
    figure.savefig(output / "denoising_late_features_7stages.png", dpi=180)
    plt.close(figure)
    print(f"OUTPUT={output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["record", "plot"])
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    plt.switch_backend("Agg")
    if args.stage == "plot":
        render(args.output)
    else:
        if any(path is None for path in [args.cache, args.run, args.config]):
            parser.error("record requires --cache, --run and --config")
        record(args)


if __name__ == "__main__":
    main()
