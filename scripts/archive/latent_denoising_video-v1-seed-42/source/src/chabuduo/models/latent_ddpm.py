"""Latent diffusion over whole-trajectory TCVAE codes.

The TCVAE compresses one complete trajectory to a single latent vector. Its
coordinates do not form a temporal axis, so the denoiser uses dense residual
blocks rather than temporal convolutions.
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from chabuduo.artifacts import (
    SPLITS,
    SplitData,
    prepared_artifact_identity,
    sha256,
    write_json,
)
from chabuduo.config import PROJECT_ROOT, ExperimentConfig
from chabuduo.models.ddpm import SinusoidalEmbedding
from chabuduo.models.tcvae import TCVAELightning


class LatentResidualBlock(nn.Module):
    """Residual MLP block conditioned on the diffusion timestep."""

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input = nn.Linear(hidden_dim, hidden_dim)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features: Tensor, time: Tensor) -> Tensor:
        hidden = self.input(self.norm(features))
        hidden = hidden + self.time(time)
        hidden = self.output(self.dropout(F.silu(hidden)))
        return features + hidden


class LatentMLPDenoiser(nn.Module):
    """Predict noise for a batch of unordered latent vectors."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        time_dim: int,
        *,
        residual_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if time_dim % 2:
            raise ValueError("time_dim must be even")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")

        self.latent_dim = latent_dim
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            LatentResidualBlock(hidden_dim, time_dim, dropout=dropout)
            for _ in range(residual_blocks)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_latent: Tensor, diffusion_step: Tensor) -> Tensor:
        if noisy_latent.ndim != 2 or noisy_latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected [batch, {self.latent_dim}] noisy latent vectors"
            )
        if diffusion_step.shape != (len(noisy_latent),):
            raise ValueError("Expected one diffusion step per latent vector")

        hidden = self.input(noisy_latent)
        time = self.time_embedding(diffusion_step)
        for block in self.blocks:
            hidden = block(hidden, time)
        return self.output(F.silu(self.output_norm(hidden)))


class LatentDDPM(nn.Module):
    """Noise-prediction DDPM for fixed-width TCVAE latent vectors."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        time_dim: int,
        *,
        residual_blocks: int,
        dropout: float,
        diffusion_steps: int,
    ) -> None:
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least two")

        self.denoiser = LatentMLPDenoiser(
            latent_dim,
            hidden_dim,
            time_dim,
            residual_blocks=residual_blocks,
            dropout=dropout,
        )
        self.diffusion_steps = diffusion_steps
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    def loss(
        self,
        clean: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if clean.ndim != 2 or clean.shape[-1] != self.denoiser.latent_dim:
            raise ValueError(
                f"Expected [batch, {self.denoiser.latent_dim}] clean latent vectors"
            )

        step = torch.randint(
            0,
            self.diffusion_steps,
            (len(clean),),
            device=clean.device,
            generator=generator,
        )
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        alpha_bar = self.alpha_bars[step][:, None]
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return F.mse_loss(self.denoiser(noisy, step), noise)

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        steps: int,
        eta: float,
    ) -> np.ndarray:
        if count < 1:
            raise ValueError("count must be positive")
        if not 2 <= steps <= self.diffusion_steps:
            raise ValueError("steps must be between 2 and diffusion_steps")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0, 1]")

        device = next(self.parameters()).device
        schedule = (
            torch.linspace(self.diffusion_steps - 1, 0, steps, device=device)
            .round()
            .long()
            .unique_consecutive()
        )
        generated = torch.randn(count, self.denoiser.latent_dim, device=device)
        self.eval()

        for index, current in enumerate(schedule):
            step = torch.full(
                (count,),
                int(current),
                device=device,
                dtype=torch.long,
            )
            noise = self.denoiser(generated, step)
            alpha_bar = self.alpha_bars[current]
            previous = schedule[index + 1] if index + 1 < len(schedule) else None
            previous_alpha_bar = (
                self.alpha_bars[previous]
                if previous is not None
                else torch.ones((), device=device)
            )
            predicted_clean = (
                generated - (1 - alpha_bar).sqrt() * noise
            ) / alpha_bar.sqrt()
            sigma = eta * torch.sqrt(
                torch.clamp(
                    (1 - previous_alpha_bar)
                    / (1 - alpha_bar)
                    * (1 - alpha_bar / previous_alpha_bar),
                    min=0,
                )
            )
            generated = (
                previous_alpha_bar.sqrt() * predicted_clean
                + torch.sqrt(
                    torch.clamp(
                        1 - previous_alpha_bar - sigma.square(),
                        min=0,
                    )
                )
                * noise
            )
            if eta > 0 and previous is not None:
                generated += sigma * torch.randn_like(generated)

        return generated.cpu().numpy().astype(np.float32)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _autoencoder_checkpoint(config: ExperimentConfig) -> Path:
    reference = config.model.autoencoder
    return _project_path(reference.run) / reference.checkpoint


def _device(config: ExperimentConfig, smoke: bool) -> torch.device:
    if smoke or config.training.accelerator == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


@torch.inference_mode()
def _encode_split(
    module: TCVAELightning,
    data: SplitData,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    values = torch.from_numpy(data.values)
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size].to(device).permute(0, 2, 1)
        mean, log_variance = module.network.encode(batch)
        means.append(mean.cpu().numpy().astype(np.float32))
        log_variances.append(log_variance.cpu().numpy().astype(np.float32))
    return np.concatenate(means), np.concatenate(log_variances)


def _load_latent_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _prepare_latent_artifact(
    artifact: Path,
    output: Path,
    config: ExperimentConfig,
    *,
    smoke: bool,
    resume: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    latent_directory = output / "artifacts"
    latent_path = latent_directory / "latent_dataset.npz"
    normalization_path = latent_directory / "latent_normalization.json"
    manifest_path = latent_directory / "latent_manifest.json"
    checkpoint = _autoencoder_checkpoint(config)
    reference = config.model.autoencoder
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != reference.expected_sha256:
        raise ValueError("TCVAE checkpoint SHA256 does not match the configuration")

    current_artifact = prepared_artifact_identity(artifact)
    if resume:
        for path in (latent_path, normalization_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if manifest["tcvae_checkpoint"]["sha256"] != checkpoint_hash:
            raise ValueError("TCVAE checkpoint changed since latent encoding")
        if manifest["prepared_artifact"]["sha256"] != current_artifact["sha256"]:
            raise ValueError("Prepared artifact changed since latent encoding")
        return _load_latent_arrays(latent_path), manifest

    raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(raw_checkpoint["epoch"])
    if checkpoint_epoch != reference.expected_epoch:
        raise ValueError(
            f"Expected TCVAE epoch {reference.expected_epoch}, got {checkpoint_epoch}"
        )
    module = TCVAELightning.load_from_checkpoint(checkpoint, map_location="cpu")
    if module.network.prior_type != reference.expected_prior:
        raise ValueError(
            f"Expected {reference.expected_prior} TCVAE prior, "
            f"got {module.network.prior_type}"
        )
    if module.network.latent_dim != config.model.latent.dimension:
        raise ValueError("TCVAE and latent DDPM dimensions do not match")

    device = _device(config, smoke)
    module.to(device).eval()
    module.requires_grad_(False)
    rng = np.random.default_rng(config.model.latent.seed)
    arrays: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for split in SPLITS:
        data = SplitData.load(artifact / f"{split}.npz")
        if smoke:
            size = min(len(data.values), 8)
            data = SplitData(
                data.values[:size],
                data.elapsed_time[:size],
                data.trajectory_id[:size],
                data.split_group_id[:size],
            )
        mean, log_variance = _encode_split(
            module,
            data,
            device=device,
            batch_size=128,
        )
        noise = rng.standard_normal(mean.shape).astype(np.float32)
        latent = mean + np.exp(0.5 * log_variance).astype(np.float32) * noise
        arrays[f"{split}_mean"] = mean
        arrays[f"{split}_log_variance"] = log_variance
        arrays[f"{split}_latent"] = latent.astype(np.float32)
        arrays[f"{split}_trajectory_id"] = data.trajectory_id.astype(str)
        counts[split] = len(latent)

    train_latent = arrays["train_latent"]
    latent_mean = train_latent.mean(axis=0, dtype=np.float64)
    latent_std = train_latent.std(axis=0, dtype=np.float64)
    latent_std[latent_std < 1e-8] = 1.0
    for split in SPLITS:
        arrays[f"{split}_standardized"] = (
            (arrays[f"{split}_latent"] - latent_mean) / latent_std
        ).astype(np.float32)

    latent_directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(latent_path, **arrays)
    write_json(
        normalization_path,
        {
            "method": "per_dimension_train",
            "mean": latent_mean.tolist(),
            "std": latent_std.tolist(),
        },
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "representation": config.model.latent.representation,
        "posterior_sample_seed": config.model.latent.seed,
        "latent_dimension": config.model.latent.dimension,
        "counts": counts,
        "normalization": "per_dimension_train",
        "prepared_artifact": current_artifact,
        "tcvae_checkpoint": {
            "path": str(checkpoint.resolve()),
            "project_relative_path": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
            "epoch": checkpoint_epoch,
            "sha256": checkpoint_hash,
            "prior": module.network.prior_type,
        },
    }
    write_json(manifest_path, manifest)
    return arrays, manifest


def _latent_loader(
    values: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(values.astype(np.float32, copy=False))),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def _latent_model_from_state(state: dict[str, object]) -> LatentDDPM:
    return LatentDDPM(
        int(state["latent_dim"]),
        int(state["hidden_dim"]),
        int(state["time_dim"]),
        residual_blocks=int(state["residual_blocks"]),
        dropout=float(state["dropout"]),
        diffusion_steps=int(state["diffusion_steps"]),
    )


def train_latent_ddpm(
    artifact: Path,
    output: Path,
    config: ExperimentConfig,
    *,
    smoke: bool,
    resume: bool = False,
    additional_epochs: int | None = None,
    learning_rate: float | None = None,
) -> Path:
    """Train a residual MLP DDPM on fixed TCVAE posterior samples."""
    if config.model.type != "latent_ddpm":
        raise TypeError("train_latent_ddpm requires model.type=latent_ddpm")

    current_artifact = prepared_artifact_identity(artifact)
    arrays, latent_manifest = _prepare_latent_artifact(
        artifact,
        output,
        config,
        smoke=smoke,
        resume=resume,
    )
    latent_path = output / "artifacts" / "latent_dataset.npz"
    normalization = json.loads(
        (output / "artifacts" / "latent_normalization.json").read_text(
            encoding="utf8"
        )
    )
    latent_mean = np.asarray(normalization["mean"], dtype=np.float32)
    latent_std = np.asarray(normalization["std"], dtype=np.float32)

    hidden_dim = 64 if smoke else config.model.hidden_dim
    residual_blocks = 1 if smoke else config.model.residual_blocks
    diffusion_steps = config.model.diffusion_steps
    model = LatentDDPM(
        config.model.latent.dimension,
        hidden_dim,
        config.model.time_embedding_dim,
        residual_blocks=residual_blocks,
        dropout=0.0 if smoke else config.model.dropout,
        diffusion_steps=diffusion_steps,
    )
    device = _device(config, smoke)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = None
    if config.optimizer.lr_step_size is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.optimizer.lr_step_size,
            gamma=config.optimizer.lr_gamma,
        )
    ema_model = None
    if config.training.ema_decay is not None:
        ema_model = deepcopy(model).eval()
        ema_model.requires_grad_(False)

    batch_size = 8 if smoke else config.training.batch_size
    workers = 0 if smoke else config.training.num_workers
    target_epochs = 1 if smoke else config.training.max_epochs
    train_loader = _latent_loader(
        arrays["train_standardized"],
        batch_size=batch_size,
        workers=workers,
        shuffle=True,
    )
    validation_loader = _latent_loader(
        arrays["validation_standardized"],
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
    )
    amp_enabled = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=resume)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=resume)
    best = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    source_summary: dict[str, object] | None = None
    resume_event: dict[str, object] | None = None

    if resume:
        last_checkpoint = checkpoints / "last.pt"
        checkpoint = torch.load(
            last_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("config_name", config.name) != config.name:
            raise ValueError("Checkpoint and experiment configuration do not match")
        if checkpoint.get("artifact_sha256") != current_artifact["sha256"]:
            raise ValueError("Prepared artifact changed since the run started")
        if checkpoint.get("latent_artifact_sha256") != sha256(latent_path):
            raise ValueError("Latent artifact changed since the run started")
        source_epoch = int(checkpoint["epoch"])
        target_epochs = (
            source_epoch + additional_epochs
            if additional_epochs is not None
            else int(config.training.max_epochs)
        )
        if target_epochs <= source_epoch:
            raise ValueError(
                "Resume target must exceed the checkpoint epoch; "
                "pass --additional-epochs"
            )
        backup = output / "resume_sources" / f"epoch-{source_epoch}"
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("best.pt", "last.pt"):
            source = checkpoints / name
            target = backup / name
            if not target.exists():
                shutil.copy2(source, target)
        summary_path = output / "run_summary.json"
        if summary_path.is_file():
            source_summary = json.loads(summary_path.read_text(encoding="utf8"))
            backup_summary = backup / "run_summary.json"
            if not backup_summary.exists():
                shutil.copy2(summary_path, backup_summary)
        resume_event = {
            "started_at": datetime.now().astimezone().isoformat(),
            "source_epoch": source_epoch,
            "source_checkpoint": "checkpoints/last.pt",
            "source_checkpoint_sha256": sha256(backup / "last.pt"),
            "source_backup": str(backup.relative_to(output)),
            "target_max_epochs": target_epochs,
            "additional_epochs": target_epochs - source_epoch,
            "learning_rate_override": learning_rate,
        }
        model.load_state_dict(checkpoint["model_state_dict"])
        if ema_model is not None and checkpoint.get("ema_model_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is None:
            raise ValueError("Checkpoint does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            if checkpoint.get("scheduler_state_dict") is None:
                raise ValueError("Checkpoint does not contain scheduler state")
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if not checkpoint.get("scheduler_state_after_epoch", False):
                scheduler.step()
        if learning_rate is not None:
            if learning_rate <= 0:
                raise ValueError("learning_rate must be positive")
            previous_learning_rates = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate)
            if scheduler is not None:
                scheduler.base_lrs = [float(learning_rate)] * len(
                    optimizer.param_groups
                )
                scheduler._last_lr = [float(learning_rate)] * len(
                    optimizer.param_groups
                )
            print(
                f"resume_learning_rate_override={learning_rate:.2e} "
                f"previous={previous_learning_rates}",
                flush=True,
            )
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        metrics = logs / "metrics.csv"
        if metrics.is_file():
            previous_metrics = pd.read_csv(metrics)
            previous_metrics = previous_metrics.loc[
                previous_metrics.epoch.le(source_epoch)
            ].drop_duplicates("epoch", keep="last")
            history = previous_metrics.to_dict(orient="records")
            for previous in history:
                validation_loss = float(previous["validation_loss"])
                if best - validation_loss > config.training.early_stopping_min_delta:
                    best = validation_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
        best = float(checkpoint.get("best_validation_loss", best))
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        start_epoch = source_epoch + 1
        print(f"resumed_from_epoch={source_epoch}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"architecture=residual_mlp parameters={parameter_count:,} "
        f"device={device.type}",
        flush=True,
    )
    stopped_early = False
    training_started = perf_counter()
    for epoch in range(start_epoch, target_epochs + 1):
        epoch_started = perf_counter()
        model.train()
        train_total = 0.0
        train_count = 0
        for batch_index, (clean,) in enumerate(train_loader):
            clean = clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = model.loss(clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config.training.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                decay = config.training.ema_decay
                if decay is None:  # pragma: no cover
                    raise RuntimeError("EMA decay disappeared during training")
                with torch.no_grad():
                    for ema_parameter, parameter in zip(
                        ema_model.parameters(), model.parameters(), strict=True
                    ):
                        ema_parameter.lerp_(parameter, 1 - decay)
                    for ema_buffer, buffer in zip(
                        ema_model.buffers(), model.buffers(), strict=True
                    ):
                        ema_buffer.copy_(buffer)
            train_total += loss.item() * len(clean)
            train_count += len(clean)
            if smoke and batch_index == 0:
                break

        model.eval()
        validation_total = 0.0
        validation_count = 0
        validation_generator = torch.Generator(device=device).manual_seed(
            config.training.seed + 1
        )
        with torch.no_grad():
            for batch_index, (clean,) in enumerate(validation_loader):
                clean = clean.to(device)
                loss = model.loss(clean, generator=validation_generator)
                validation_total += loss.item() * len(clean)
                validation_count += len(clean)
                if smoke and batch_index == 0:
                    break
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_total / train_count,
            "validation_loss": validation_total / validation_count,
            "epoch_seconds": perf_counter() - epoch_started,
        }
        history.append(row)
        improvement = best - row["validation_loss"]
        improved = improvement > config.training.early_stopping_min_delta
        if improved:
            best = float(row["validation_loss"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        tcvae = latent_manifest["tcvae_checkpoint"]
        if not isinstance(tcvae, dict):
            raise TypeError("Invalid TCVAE checkpoint manifest")
        state = {
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": (
                ema_model.state_dict() if ema_model is not None else None
            ),
            "architecture": "residual_mlp",
            "latent_dim": config.model.latent.dimension,
            "hidden_dim": hidden_dim,
            "time_dim": config.model.time_embedding_dim,
            "residual_blocks": residual_blocks,
            "dropout": 0.0 if smoke else config.model.dropout,
            "diffusion_steps": diffusion_steps,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "latent_representation": config.model.latent.representation,
            "posterior_sample_seed": config.model.latent.seed,
            "tcvae_checkpoint": tcvae["project_relative_path"],
            "tcvae_checkpoint_epoch": tcvae["epoch"],
            "tcvae_checkpoint_sha256": tcvae["sha256"],
            "epoch": epoch,
            "validation_loss": row["validation_loss"],
            "best_validation_loss": best,
            "config_name": config.name,
            "artifact_sha256": current_artifact["sha256"],
            "latent_artifact_sha256": sha256(latent_path),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scheduler_state_after_epoch": True,
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
            "numpy_rng_state": np.random.get_state(),
        }
        torch.save(state, checkpoints / "last.pt")
        if improved:
            torch.save(state, checkpoints / "best.pt")
        pd.DataFrame(history).to_csv(logs / "metrics.csv", index=False)
        print(
            f"epoch={epoch}/{target_epochs} lr={row['learning_rate']:.2e} "
            f"train_loss={row['train_loss']:.6f} "
            f"validation_loss={row['validation_loss']:.6f} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
        patience = config.training.early_stopping_patience
        if patience is not None and epochs_without_improvement >= patience:
            stopped_early = True
            print(
                f"early_stopping epoch={epoch} best_validation_loss={best:.6f} "
                f"patience={patience}",
                flush=True,
            )
            break

    best_path = checkpoints / "best.pt"
    last_path = checkpoints / "last.pt"
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    last_state = torch.load(last_path, map_location="cpu", weights_only=False)
    if source_summary is None:
        previous_resumes = []
    else:
        previous_resumes = source_summary.get("resumes", [])
        if not previous_resumes and "resume" in source_summary:
            previous_resumes = [source_summary["resume"]]
    if resume_event is not None:
        resume_event["finished_at"] = datetime.now().astimezone().isoformat()
        resume_event["last_epoch"] = int(last_state["epoch"])
    summary = {
        "schema_version": 1,
        "status": "complete",
        "model_type": "latent_ddpm",
        "config_name": config.name,
        "artifact": current_artifact,
        "latent_artifact": {
            "path": str(latent_path.relative_to(output)),
            "sha256": sha256(latent_path),
        },
        "tcvae_checkpoint": latent_manifest["tcvae_checkpoint"],
        "monitor": "validation_loss",
        "best": {
            "path": str(best_path.relative_to(output)),
            "epoch": int(best_state["epoch"]),
            "score": float(best_state["validation_loss"]),
            "sha256": sha256(best_path),
        },
        "last": {
            "path": str(last_path.relative_to(output)),
            "epoch": int(last_state["epoch"]),
            "score": float(last_state["validation_loss"]),
            "sha256": sha256(last_path),
        },
        "stopped_early": stopped_early,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "runtime_seconds": round(perf_counter() - training_started, 3),
        "finished_at": datetime.now().astimezone().isoformat(),
        "resumes": [
            *previous_resumes,
            *([] if resume_event is None else [resume_event]),
        ],
    }
    write_json(output / "run_summary.json", summary)
    return output


@torch.inference_mode()
def sample_latent_ddpm(
    checkpoint: Path,
    count: int,
    *,
    steps: int,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample latent vectors and decode normalized trajectory features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = _latent_model_from_state(state)
    model.load_state_dict(
        state.get("ema_model_state_dict") or state["model_state_dict"]
    )
    model.to(device).eval()
    standardized = model.sample(count, steps=steps, eta=eta)
    latent = (
        standardized * np.asarray(state["latent_std"], dtype=np.float32)
        + np.asarray(state["latent_mean"], dtype=np.float32)
    ).astype(np.float32)

    tcvae_checkpoint = _project_path(str(state["tcvae_checkpoint"]))
    if sha256(tcvae_checkpoint) != state["tcvae_checkpoint_sha256"]:
        raise ValueError("TCVAE checkpoint changed since latent DDPM training")
    module = TCVAELightning.load_from_checkpoint(
        tcvae_checkpoint,
        map_location=device,
    )
    module.to(device).eval()
    module.requires_grad_(False)
    decoded = module.network.decode(torch.from_numpy(latent).to(device))
    normalized = decoded.permute(0, 2, 1).cpu().numpy().astype(np.float32)
    return normalized, latent, standardized
