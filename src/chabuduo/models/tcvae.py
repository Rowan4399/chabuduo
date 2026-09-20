"""Temporal convolutional variational autoencoder.

The temporal blocks and encoder/decoder layout are adapted from the MIT-licensed
Code Ocean capsule "Deep Generative Modelling of Aircraft Trajectories in
Terminal Maneuvering Areas", DOI 10.24433/CO.0357574.V1. See NOTICE.md.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, TensorDataset

from chabuduo.artifacts import (
    SplitData,
    prepared_artifact_identity,
    sha256,
    write_json,
)
from chabuduo.config import ExperimentConfig

_CHECKPOINT_MONITOR = "validation/reconstruction_mse"


class CausalTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        *,
        activate: bool,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
            )
        )
        self.activation = nn.ReLU() if activate else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            direction = self.conv.parametrizations.weight.original1
            magnitude = self.conv.parametrizations.weight.original0
            nn.init.normal_(direction, mean=0.0, std=0.01)
            magnitude.copy_(
                torch.linalg.vector_norm(direction, dim=(1, 2), keepdim=True)
            )

    def forward(self, value: Tensor) -> Tensor:
        value = F.pad(value, (self.left_padding, 0))
        return self.dropout(self.activation(self.conv(value)))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        *,
        final: bool,
    ) -> None:
        super().__init__()
        self.first = CausalTemporalBlock(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            activate=True,
        )
        self.second = CausalTemporalBlock(
            out_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            activate=not final,
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        if isinstance(self.residual, nn.Conv1d):
            nn.init.normal_(self.residual.weight, mean=0.0, std=0.01)

    def forward(self, value: Tensor) -> Tensor:
        return self.second(self.first(value)) + self.residual(value)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: Sequence[int],
        kernel_size: int,
        dilation_base: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [input_channels, *hidden_channels, output_channels]
        self.network = nn.Sequential(
            *(
                ResidualBlock(
                    dimensions[index],
                    dimensions[index + 1],
                    kernel_size,
                    dilation_base**index,
                    dropout,
                    final=index == len(dimensions) - 2,
                )
                for index in range(len(dimensions) - 1)
            )
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class StandardTCVAE(nn.Module):
    """A TCVAE accepting channel-first trajectories ``[batch, feature, time]``."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        sequence_length: int = 200,
        hidden_channels: Sequence[int] = (64, 64, 64),
        latent_dim: int = 64,
        kernel_size: int = 16,
        dilation_base: int = 2,
        sampling_factor: int = 10,
        dropout: float = 0.2,
        prior_type: str = "standard",
        prior_components: int | None = None,
    ) -> None:
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels cannot be empty")
        if sequence_length % sampling_factor != 0:
            raise ValueError("sequence_length must be divisible by sampling_factor")
        if prior_type not in {"standard", "vamp"}:
            raise ValueError(f"Unsupported TCVAE prior: {prior_type}")
        if prior_type == "vamp" and (
            prior_components is None or prior_components < 1
        ):
            raise ValueError("VampPrior requires at least one component")

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.hidden_channels = tuple(hidden_channels)
        self.latent_dim = latent_dim
        self.sampling_factor = sampling_factor
        self.pooled_length = sequence_length // sampling_factor
        self.prior_type = prior_type
        self.prior_components = prior_components if prior_type == "vamp" else None

        self.encoder_tcn = TemporalConvNet(
            input_channels,
            self.hidden_channels[-1],
            self.hidden_channels[:-1],
            kernel_size,
            dilation_base,
            dropout,
        )
        self.encoder_pool = nn.AvgPool1d(sampling_factor)
        encoded_dim = self.hidden_channels[-1] * self.pooled_length
        self.mean_head = nn.Linear(encoded_dim, latent_dim)
        self.log_variance_head = nn.Sequential(
            nn.Linear(encoded_dim, latent_dim),
            nn.Hardtanh(min_val=-6.0, max_val=2.0),
        )

        self.decoder_channels = tuple(reversed(self.hidden_channels))
        self.decoder_entry = nn.Linear(
            latent_dim, self.decoder_channels[0] * self.pooled_length
        )
        self.decoder_tcn = TemporalConvNet(
            self.decoder_channels[0],
            input_channels,
            self.decoder_channels[1:],
            kernel_size,
            dilation_base,
            dropout,
        )
        if self.prior_type == "vamp":
            self.pseudo_inputs = nn.Parameter(
                torch.empty(
                    int(self.prior_components), input_channels, sequence_length
                )
            )
            nn.init.normal_(self.pseudo_inputs, mean=0.0, std=0.01)
        else:
            self.register_parameter("pseudo_inputs", None)

    def initialize_pseudo_inputs(self, values: Tensor) -> None:
        """Initialize VampPrior pseudo-inputs from normalized training examples."""
        if self.pseudo_inputs is None:
            return
        if values.shape != self.pseudo_inputs.shape:
            raise ValueError(
                "Expected pseudo-input initialization with shape "
                f"{tuple(self.pseudo_inputs.shape)}, got {tuple(values.shape)}"
            )
        with torch.no_grad():
            self.pseudo_inputs.copy_(values.to(self.pseudo_inputs))

    def encode(self, value: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder_pool(self.encoder_tcn(value)).flatten(start_dim=1)
        return self.mean_head(hidden), self.log_variance_head(hidden)

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor) -> Tensor:
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, latent: Tensor) -> Tensor:
        hidden = self.decoder_entry(latent).view(
            latent.shape[0], self.decoder_channels[0], self.pooled_length
        )
        hidden = F.interpolate(
            hidden, scale_factor=self.sampling_factor, mode="nearest"
        )
        return self.decoder_tcn(hidden)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encode(value)
        reconstruction = self.decode(self.reparameterize(mean, log_variance))
        return reconstruction, mean, log_variance

    @staticmethod
    def _diagonal_gaussian_log_prob(
        value: Tensor, mean: Tensor, log_variance: Tensor
    ) -> Tensor:
        return -0.5 * (
            torch.log(torch.tensor(2 * torch.pi, device=value.device))
            + log_variance
            + (value - mean).square() / log_variance.exp()
        ).sum(dim=-1)

    def prior_parameters(self) -> tuple[Tensor, Tensor]:
        if self.pseudo_inputs is None:
            raise RuntimeError("The standard normal prior has no learned components")
        return self.encode(self.pseudo_inputs)

    def kl_divergence(
        self,
        mean: Tensor,
        log_variance: Tensor,
        latent: Tensor | None = None,
    ) -> Tensor:
        """Return KL to the configured standard or VampPrior distribution."""
        if self.prior_type == "standard":
            per_sample = -0.5 * torch.sum(
                1 + log_variance - mean.square() - log_variance.exp(), dim=1
            )
            return per_sample.mean()
        if latent is None:
            raise ValueError("VampPrior KL requires a posterior sample")
        component_mean, component_log_variance = self.prior_parameters()
        log_q = self._diagonal_gaussian_log_prob(latent, mean, log_variance)
        log_components = self._diagonal_gaussian_log_prob(
            latent[:, None, :],
            component_mean[None, :, :],
            component_log_variance[None, :, :],
        )
        log_prior = torch.logsumexp(log_components, dim=1) - torch.log(
            torch.tensor(log_components.shape[1], device=latent.device)
        )
        return (log_q - log_prior).mean()

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        parameter = next(self.parameters())
        target_device = device or parameter.device
        if self.prior_type == "standard":
            latent = temperature * torch.randn(
                count, self.latent_dim, device=target_device
            )
        else:
            component_mean, component_log_variance = self.prior_parameters()
            component = torch.randint(
                len(component_mean), (count,), device=target_device
            )
            mean = component_mean[component]
            standard_deviation = torch.exp(
                0.5 * component_log_variance[component]
            )
            latent = mean + temperature * standard_deviation * torch.randn_like(mean)
        return self.decode(latent)


class TCVAELightning(L.LightningModule):
    """Lightning adapter for the TCVAE loss and optimizer."""

    def __init__(self, resolved_config: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"resolved_config": resolved_config})
        self.resolved_config = resolved_config
        model = resolved_config["model"]
        prior = model.get("prior", {"type": "standard"})
        self.network = StandardTCVAE(
            input_channels=int(model["input_channels"]),
            sequence_length=int(model["sequence_length"]),
            hidden_channels=model["hidden_channels"],
            latent_dim=int(model["latent_dim"]),
            kernel_size=int(model["kernel_size"]),
            dilation_base=int(model["dilation_base"]),
            sampling_factor=int(model["sampling_factor"]),
            dropout=float(model["dropout"]),
            prior_type=prior["type"],
            prior_components=prior.get("components"),
        )
        loss_config = resolved_config["loss"]
        initial_scale = torch.tensor(float(loss_config.get("decoder_scale", 1.0)))
        self.log_decoder_scale = torch.nn.Parameter(
            initial_scale.log(),
            requires_grad=bool(loss_config.get("learn_decoder_scale", True)),
        )

    def _loss_terms(
        self,
        value: Tensor,
        reconstruction: Tensor,
        mean: Tensor,
        log_variance: Tensor,
        latent: Tensor | None = None,
    ) -> dict[str, Tensor]:
        log_scale = self.log_decoder_scale.clamp_min(-9.21034)
        scale = log_scale.exp()
        gaussian_nll = (
            0.5 * ((value - reconstruction) / scale).square()
            + log_scale
            + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=value.device))
        )
        reconstruction_loss = gaussian_nll.sum(dim=(1, 2)).mean()
        reconstruction_mse = F.mse_loss(reconstruction, value)
        kl_loss = self.network.kl_divergence(mean, log_variance, latent)
        loss_config = self.resolved_config["loss"]
        loss = (
            float(loss_config["reconstruction_weight"]) * reconstruction_loss
            + float(loss_config["kl_weight"]) * kl_loss
        )
        return {
            "loss": loss,
            "reconstruction": reconstruction_loss,
            "reconstruction_mse": reconstruction_mse,
            "kl": kl_loss,
        }

    def compute_loss(self, value: Tensor) -> dict[str, Tensor]:
        mean, log_variance = self.network.encode(value)
        latent = self.network.reparameterize(mean, log_variance)
        reconstruction = self.network.decode(latent)
        return self._loss_terms(
            value, reconstruction, mean, log_variance, latent
        )

    def compute_validation_loss(
        self, value: Tensor, *, batch_index: int = 0
    ) -> dict[str, Tensor]:
        mean, log_variance = self.network.encode(value)
        reconstruction = self.network.decode(mean)
        latent = None
        if self.network.prior_type == "vamp":
            generator = torch.Generator(device=value.device).manual_seed(
                int(self.resolved_config["seed"]) + batch_index
            )
            noise = torch.randn(
                mean.shape,
                generator=generator,
                device=mean.device,
                dtype=mean.dtype,
            )
            latent = mean + torch.exp(0.5 * log_variance) * noise
        return self._loss_terms(
            value, reconstruction, mean, log_variance, latent
        )

    def training_step(self, batch: tuple[Tensor], batch_index: int) -> Tensor:
        value = batch[0].permute(0, 2, 1)
        losses = self.compute_loss(value)
        self.log_dict(
            {f"train/{name}": item for name, item in losses.items()},
            on_epoch=True,
            batch_size=value.shape[0],
        )
        return losses["loss"]

    def validation_step(self, batch: tuple[Tensor], batch_index: int) -> Tensor:
        value = batch[0].permute(0, 2, 1)
        losses = self.compute_validation_loss(value, batch_index=batch_index)
        self.log_dict(
            {
                "validation/checkpoint_loss": losses["loss"],
                "validation/reconstruction": losses["reconstruction"],
                "validation/reconstruction_mse": losses["reconstruction_mse"],
                "validation/kl": losses["kl"],
            },
            on_epoch=True,
            prog_bar=True,
            batch_size=value.shape[0],
        )
        return losses["loss"]

    def on_train_epoch_end(self) -> None:
        scale = self.log_decoder_scale.clamp_min(-9.21034).exp()
        self.log("model/decoder_scale", scale, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optimizer_config = self.resolved_config["optimizer"]
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config.get("weight_decay", 0)),
        )
        if optimizer_config.get("lr_plateau_patience") is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(optimizer_config["lr_plateau_factor"]),
                patience=int(optimizer_config["lr_plateau_patience"]),
                min_lr=float(optimizer_config.get("min_learning_rate", 0.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": _CHECKPOINT_MONITOR,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        if optimizer_config.get("lr_step_size") is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(optimizer_config["lr_step_size"]),
            gamma=float(optimizer_config["lr_gamma"]),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @torch.inference_mode()
    def generate(self, count: int, *, temperature: float = 1.0) -> Tensor:
        return self.network.sample(count, device=self.device, temperature=temperature)


def _loader(
    data: SplitData, batch_size: int, workers: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(data.values)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


class _LearningRateOverride(Callback):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def on_fit_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        del module
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = self.learning_rate


def sample_tcvae(
    checkpoint: Path,
    count: int,
    *,
    temperature: float,
) -> np.ndarray:
    """Load a TCVAE checkpoint and return normalized ``[N, T, F]`` values."""
    module = TCVAELightning.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    return (
        module.generate(count, temperature=temperature)
        .permute(0, 2, 1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def train_tcvae(
    artifact: Path,
    output: Path,
    config: ExperimentConfig,
    *,
    smoke: bool,
    resume: bool = False,
    additional_epochs: int | None = None,
    learning_rate: float | None = None,
) -> Path:
    """Train a TCVAE from the ready arrays in a prepared artifact."""
    if config.model.type != "tcvae":
        raise TypeError("train_tcvae requires model.type=tcvae")
    train_data = SplitData.load(artifact / "train.npz")
    validation_data = SplitData.load(artifact / "validation.npz")
    sequence_length, input_channels = train_data.values.shape[1:]
    current_artifact = prepared_artifact_identity(artifact)
    configured = {
        "name": config.name,
        "artifact": current_artifact,
        "model": {
            **config.model.model_dump(exclude={"type", "loss"}),
            "input_channels": input_channels,
            "sequence_length": sequence_length,
        },
        "loss": config.model.loss.model_dump(),
        "optimizer": config.optimizer.model_dump(),
        "training": config.training.model_dump(),
        "seed": config.training.seed,
        "smoke": smoke,
    }
    if smoke:
        configured["model"] = {
            **configured["model"],
            "hidden_channels": [8, 8],
            "latent_dim": 4,
            "kernel_size": 3,
            "dropout": 0.0,
        }
        configured["training"] = {
            **configured["training"],
            "batch_size": 8,
            "max_epochs": 1,
            "num_workers": 0,
            "accelerator": "cpu",
            "devices": 1,
        }

    resume_path: Path | None = None
    source_summary: dict[str, Any] | None = None
    resume_event: dict[str, Any] | None = None
    if resume:
        resume_path = output / "checkpoints" / "last.ckpt"
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        resolved = deepcopy(state["hyper_parameters"]["resolved_config"])
        if resolved["name"] != config.name:
            raise ValueError(
                f"Checkpoint belongs to {resolved['name']!r}, not {config.name!r}"
            )
        if resolved["artifact"]["sha256"] != current_artifact["sha256"]:
            raise ValueError("Prepared artifact changed since the run started")
        source_epoch = int(state["epoch"])
        completed_epochs = source_epoch + 1
        target_epochs = (
            completed_epochs + additional_epochs
            if additional_epochs is not None
            else int(config.training.max_epochs)
        )
        if target_epochs <= completed_epochs:
            raise ValueError(
                "Resume target must exceed the completed epoch count; "
                "pass --additional-epochs"
            )
        resolved["training"]["max_epochs"] = target_epochs
        backup = output / "resume_sources" / f"epoch-{source_epoch}"
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("best.ckpt", "last.ckpt"):
            source = output / "checkpoints" / name
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
            "source_checkpoint": "checkpoints/last.ckpt",
            "source_checkpoint_sha256": sha256(backup / "last.ckpt"),
            "source_backup": str(backup.relative_to(output)),
            "target_max_epochs": target_epochs,
            "additional_epochs": target_epochs - completed_epochs,
            "learning_rate_override": learning_rate,
        }
    else:
        resolved = configured

    training = resolved["training"]
    best_checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best",
        monitor=_CHECKPOINT_MONITOR,
        mode="min",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    last_checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="last",
        monitor=None,
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
        enable_version_counter=False,
    )
    early_stopping = None
    callbacks: list[Any] = [
        best_checkpoint,
        last_checkpoint,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if training.get("early_stopping_patience") is not None:
        early_stopping = EarlyStopping(
            monitor=_CHECKPOINT_MONITOR,
            mode="min",
            patience=int(training["early_stopping_patience"]),
            min_delta=float(training.get("early_stopping_min_delta", 0.0)),
        )
        callbacks.append(early_stopping)
    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        callbacks.append(_LearningRateOverride(learning_rate))
    loggers = [
        CSVLogger(save_dir=output / "logs", name="training"),
        TensorBoardLogger(save_dir=output / "logs", name="tensorboard"),
    ]
    trainer = L.Trainer(
        max_epochs=int(training["max_epochs"]),
        accelerator=training.get("accelerator", "auto"),
        devices=training.get("devices", 1),
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=training.get("gradient_clip_norm"),
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_autolog_hparams=False,
        enable_progress_bar=False,
        limit_train_batches=1 if smoke else 1.0,
        limit_val_batches=1 if smoke else 1.0,
    )
    module = TCVAELightning(resolved)
    if not resume and module.network.prior_type == "vamp":
        prior = resolved["model"]["prior"]
        if prior.get("initialization") != "train_examples":
            raise ValueError(
                f"Unsupported VampPrior initialization: {prior.get('initialization')}"
            )
        components = int(prior["components"])
        if components > len(train_data.values):
            raise ValueError(
                "VampPrior components cannot exceed the number of training examples"
            )
        generator = np.random.default_rng(int(resolved["seed"]))
        indices = generator.choice(len(train_data.values), components, replace=False)
        initial = torch.from_numpy(train_data.values[indices]).permute(0, 2, 1)
        module.network.initialize_pseudo_inputs(initial)
    started = perf_counter()
    trainer.fit(
        module,
        train_dataloaders=_loader(
            train_data,
            int(training["batch_size"]),
            int(training["num_workers"]),
            True,
        ),
        val_dataloaders=_loader(
            validation_data,
            int(training["batch_size"]),
            int(training["num_workers"]),
            False,
        ),
        ckpt_path=resume_path,
    )
    runtime_seconds = round(perf_counter() - started, 3)
    best_path = Path(best_checkpoint.best_model_path)
    last_path = output / "checkpoints" / "last.ckpt"
    if not best_path.is_file():
        best_path = output / "checkpoints" / "best.ckpt"
    if resume and source_summary is not None and resume_event is not None:
        previous_score = float(source_summary["best"]["score"])
        current_score = float(best_checkpoint.best_model_score)
        if previous_score <= current_score:
            previous_best = output / resume_event["source_backup"] / "best.ckpt"
            shutil.copy2(previous_best, output / "checkpoints" / "best.ckpt")
            best_path = output / "checkpoints" / "best.ckpt"
            best_score = previous_score
        else:
            best_score = current_score
    else:
        best_score = float(best_checkpoint.best_model_score)
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
        "model_type": "tcvae",
        "config_name": config.name,
        "artifact": current_artifact,
        "monitor": _CHECKPOINT_MONITOR,
        "best": {
            "path": str(best_path.relative_to(output)),
            "epoch": int(best_state["epoch"]),
            "global_step": int(best_state["global_step"]),
            "score": best_score,
            "sha256": sha256(best_path),
        },
        "last": {
            "path": str(last_path.relative_to(output)),
            "epoch": int(last_state["epoch"]),
            "global_step": int(last_state["global_step"]),
            "sha256": sha256(last_path),
        },
        "stopped_early": bool(
            early_stopping is not None and early_stopping.stopped_epoch > 0
        ),
        "final_learning_rate": float(trainer.optimizers[0].param_groups[0]["lr"]),
        "runtime_seconds": runtime_seconds,
        "finished_at": datetime.now().astimezone().isoformat(),
        "resumes": [
            *previous_resumes,
            *([] if resume_event is None else [resume_event]),
        ],
    }
    write_json(output / "run_summary.json", summary)
    return output
