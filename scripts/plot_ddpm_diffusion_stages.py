"""Observe DDPM sampler states and show forward noise on held-out trajectories."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from chabuduo.artifacts import Normalization, SplitData, sha256
from chabuduo.config import load_experiment_config
from chabuduo.models.ddpm import UnconditionalDDPM
from chabuduo.plotting import _anchor_xy
from chabuduo.prepare import airport_projection
from chabuduo.sampling import _channels, _kinematic_matrix, build_generated_traffic

TIMESTEPS = [0, 250, 500, 750, 1000]


def reverse_states(model, count: int, points: int, seed: int):
    snapshots = {}
    calls = 0

    def capture(_module, inputs):
        nonlocal calls
        calls += 1
        timestep = int(inputs[1][0]) + 1
        if timestep in TIMESTEPS:
            snapshots[timestep] = inputs[0].detach().cpu().numpy().copy()
        if calls % 250 == 0:
            print(f"Denoising updates: {calls}/1000", flush=True)

    # Observe the supported sampler; do not duplicate its reverse equations.
    handle = model.denoiser.register_forward_pre_hook(capture)
    torch.manual_seed(seed)
    try:
        generated = model.sample(count, points, steps=1000, eta=0.0)
    finally:
        handle.remove()
    if calls != 1000:
        raise ValueError(f"Expected 1000 sampler calls, got {calls}")
    snapshots[0] = generated.copy()
    print("Checking that observation leaves the original sampler unchanged", flush=True)
    torch.manual_seed(seed)
    baseline = model.sample(count, points, steps=1000, eta=0.0)
    np.testing.assert_array_equal(generated, baseline)
    return np.stack([snapshots[t] for t in reversed(TIMESTEPS)])


def forward_states(clean: np.ndarray, betas: torch.Tensor, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noisy = torch.from_numpy(clean.copy())
    snapshots = {0: clean.copy()}
    # A single Markov chain per trajectory, with fresh noise at each step.
    for index, beta in enumerate(betas.detach().cpu()):
        epsilon = torch.randn(noisy.shape, generator=generator, dtype=noisy.dtype)
        noisy = (1 - beta).sqrt() * noisy + beta.sqrt() * epsilon
        timestep = index + 1
        if timestep in TIMESTEPS:
            snapshots[timestep] = noisy.numpy().copy()
    return np.stack([snapshots[t] for t in TIMESTEPS])


def reconstruct(states, normalization, reference_time, config):
    stages, count, points, features = states.shape
    values = normalization.inverse_transform(states.reshape(-1, points, features))
    elapsed_index = config.prepare.representation.features.index("elapsed_time_sec")
    if reference_time is not None:
        values[:, :, elapsed_index] = np.tile(reference_time, (stages, 1))
    t = build_generated_traffic(values, values[:, :, elapsed_index], config)
    if not {"x", "y"} <= set(t.data.columns):
        t = t.compute_xy(projection=airport_projection(config.preprocess.airport))
    xy = np.stack(
        [
            group.sort_values("timestamp")[["x", "y"]].to_numpy()
            for _, group in t.data.groupby("flight_id", sort=False)
        ]
    )
    if xy.shape != (stages * count, points, 2):
        raise ValueError(f"Unexpected reconstructed shape: {xy.shape}")
    anchor_x, anchor_y, _ = _anchor_xy(config)
    xy = xy / 1000 - np.array([anchor_x, anchor_y])
    return xy.reshape(stages, count, points, 2)


def map_bounds(*arrays):
    xy = np.concatenate([array.reshape(-1, 2) for array in arrays])
    low, high = xy.min(axis=0), xy.max(axis=0)
    margin = np.maximum(5.0, 0.05 * (high - low))
    return low - margin, high + margin


def plot_maps(xy, timesteps, bounds, output, title, note):
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, xy.shape[1]))
    for stage, (axis, timestep) in enumerate(zip(axes.flat, timesteps, strict=False)):
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
            title=f"t = {timestep}  |  n = {xy.shape[1]}",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18)
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
    figure.suptitle(title, fontsize=17)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_features(states, timesteps, limits, output, title, features):
    figure, axes = plt.subplots(4, 5, figsize=(20, 11), layout="constrained")
    for feature, name in enumerate(features):
        for stage, timestep in enumerate(timesteps):
            axis = axes[feature, stage]
            axis.plot(
                states[stage, :, :, feature].T,
                color="#3578a3",
                alpha=0.12,
                linewidth=0.55,
            )
            axis.set_ylim(*limits[feature])
            axis.grid(alpha=0.15)
            if feature == 0:
                axis.set_title(f"t = {timestep}")
            if stage == 0:
                axis.set_ylabel(f"{name}\nstandardized")
            if feature == 3:
                axis.set_xlabel("Chronological sequence index")
    figure.suptitle(title, fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render(output: Path):
    with np.load(output / "diffusion_states.npz") as data:
        reverse = data["reverse_normalized_values"]
        forward = data["forward_normalized_values"]
        reverse_xy = data["reverse_fixed_time_xy_km"]
        forward_xy = data["forward_fixed_time_xy_km"]
        reverse_full_xy = data["reverse_all_features_xy_km"]
        forward_full_xy = data["forward_all_features_xy_km"]
        features = data["feature_order"].tolist()
    count = reverse.shape[1]
    bounds = map_bounds(reverse_xy, forward_xy)
    full_bounds = map_bounds(reverse_full_xy, forward_full_xy)
    note = (
        f"{count} identical trajectory IDs across panels\n"
        "Black cross: LSZH RWY 14 threshold\n"
        "Same projected axes in both processes\n\n"
        "Noise acts on standardized kinematics.\n"
        "Display timing is held fixed per trajectory.\n"
        "Speed uses the existing 30-600 kt bounds.\n"
        "Raw time noise is shown in feature plots.\n"
        "Diffusion t is not flight time.\n"
        "Early states are not physical flight paths."
    )
    for name, states, xy, full_xy, timesteps, label in [
        (
            "denoising",
            reverse,
            reverse_xy,
            reverse_full_xy,
            list(reversed(TIMESTEPS)),
            "Noise to generated trajectories",
        ),
        (
            "forward_noising",
            forward,
            forward_xy,
            forward_full_xy,
            TIMESTEPS,
            "Held-out trajectories to noise",
        ),
    ]:
        method = (
            "1000-step DDIM, eta=0" if name == "denoising" else "1000 forward steps"
        )
        plot_maps(
            xy,
            timesteps,
            bounds,
            output / f"{name}_5stages.png",
            f"RWY 14 | {label} | {method}",
            note,
        )
        full_note = note.replace(
            "Display timing is held fixed per trajectory.",
            "All four noised features are reconstructed.",
        ).replace(
            "Raw time noise is shown in feature plots.",
            "Nonpositive time increments are repaired.",
        )
        plot_maps(
            full_xy,
            timesteps,
            full_bounds,
            output / f"{name}_all_features_5stages.png",
            f"RWY 14 | {label} | all features including noisy time",
            full_note,
        )
        joined = np.concatenate([reverse.reshape(-1, 4), forward.reshape(-1, 4)])
        limits = [
            (joined[:, f].min() - 0.2, joined[:, f].max() + 0.2) for f in range(4)
        ]
        plot_features(
            states,
            timesteps,
            limits,
            output / f"{name}_features_5stages.png",
            f"{label}: raw standardized model states ({count} trajectories)",
            features,
        )
    print(f"OUTPUT={output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["record", "plot"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plt.switch_backend("Agg")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.stage == "plot":
        render(args.output)
        return
    if any(path is None for path in [args.run, args.artifact, args.config]):
        parser.error("record requires --run, --artifact and --config")
    if args.output.exists():
        raise FileExistsError(args.output)
    started = perf_counter()
    config = load_experiment_config(args.config)
    summary = json.loads((args.run / "run_summary.json").read_text())
    checkpoint = args.run / "checkpoints/best.pt"
    if sha256(checkpoint) != summary["best"]["sha256"]:
        raise ValueError("Best checkpoint does not match the training summary")
    for name, expected in summary["artifact"]["files"].items():
        if sha256(args.artifact / name) != expected:
            raise ValueError(f"Prepared artifact differs: {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    # Match the existing sampler's CUDA settings; strict convolution is much slower.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["architecture"] != "temporal_dilated_resnet":
        raise ValueError("This visualization requires the unconditional temporal DDPM")
    model = UnconditionalDDPM(
        feature_dim=state["feature_dim"],
        hidden_dim=state["hidden_dim"],
        time_dim=state["time_dim"],
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
    if model.diffusion_steps != 1000 or args.count < 1:
        raise ValueError("Expected 1000 diffusion steps and a positive sample count")
    test = SplitData.load(args.artifact / "test.npz")
    if args.count > len(test.values):
        raise ValueError("Not enough test trajectories")
    clean = test.values[: args.count]
    normalization = Normalization.load(args.artifact / "normalization.json")
    reverse = reverse_states(model, args.count, config.prepare.points, args.seed)
    forward = forward_states(clean, model.betas, args.seed + 1000)
    if not np.isfinite(reverse).all() or not np.isfinite(forward).all():
        raise ValueError("Nonfinite diffusion states")
    args.output.mkdir(parents=True)
    final_values = normalization.inverse_transform(reverse[-1])
    final_time = _kinematic_matrix(
        _channels(final_values, config), final_values[:, :, 3], config
    )[:, :, 3]
    real_time = test.elapsed_time[: args.count]
    print("Reconstructing the five states with fixed display timing", flush=True)
    reverse_xy = reconstruct(reverse, normalization, final_time, config)
    forward_xy = reconstruct(forward, normalization, real_time, config)
    print(
        "Reconstructing all four noised features for the supplemental view", flush=True
    )
    reverse_full_xy = reconstruct(reverse, normalization, None, config)
    forward_full_xy = reconstruct(forward, normalization, None, config)
    features = config.prepare.representation.features
    np.savez_compressed(
        args.output / "diffusion_states.npz",
        reverse_normalized_values=reverse,
        forward_normalized_values=forward,
        reverse_timesteps=np.array(list(reversed(TIMESTEPS))),
        forward_timesteps=np.array(TIMESTEPS),
        generated_id=np.array([f"sample_{i:03d}" for i in range(args.count)]),
        source_trajectory_id=test.trajectory_id[: args.count],
        forward_fixed_elapsed_time=real_time,
        reverse_fixed_elapsed_time=final_time,
        reverse_fixed_time_xy_km=reverse_xy,
        forward_fixed_time_xy_km=forward_xy,
        reverse_all_features_xy_km=reverse_full_xy,
        forward_all_features_xy_km=forward_full_xy,
        normalization_mean=normalization.mean,
        normalization_std=normalization.std,
        feature_order=np.array(features),
        betas=model.betas.cpu().numpy(),
        alpha_bars=model.alpha_bars.cpu().numpy(),
    )
    final_t = build_generated_traffic(final_values, final_values[:, :, 3], config)
    final_t.data.to_parquet(args.output / "generated_1000steps.parquet", index=False)
    report = {
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(args.config),
        "script_sha256": sha256(Path(__file__)),
        "count": args.count,
        "sampling_steps": 1000,
        "sampling_seed": args.seed,
        "eta": 0.0,
        "sampler": "unmodified UnconditionalDDPM.sample (DDIM eta=0)",
        "observer_matches_unmodified_sampler": True,
        "strict_deterministic_algorithms": False,
        "forward_seed": args.seed + 1000,
        "test_selection": f"first {args.count} test rows",
        "reverse_timesteps": list(reversed(TIMESTEPS)),
        "forward_timesteps": TIMESTEPS,
        "state_shape": list(reverse.shape),
        "feature_order": features,
        "main_maps": "fixed clean/final timing; noisy track/speed; speed clipped",
        "supplemental_maps": "all four noisy channels, usual clipping/time repair",
        "projection": "airport-centred azimuthal equidistant; km relative to threshold",
        "noise_space": "training-only per-timestep/per-feature standardization",
        "runtime_seconds": round(perf_counter() - started, 2),
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": str(device),
        },
        "method_sources": [
            "https://arxiv.org/abs/2006.11239",
            "https://arxiv.org/abs/2010.02502",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8"
    )
    print(f"OUTPUT={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
