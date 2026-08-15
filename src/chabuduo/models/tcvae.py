"""Standard-prior temporal convolutional variational autoencoder.

The temporal blocks and encoder/decoder layout are adapted from the MIT-licensed
Code Ocean capsule "Deep Generative Modelling of Aircraft Trajectories in
Terminal Maneuvering Areas", DOI 10.24433/CO.0357574.V1. See NOTICE.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, TensorDataset

from chabuduo.artifacts import SplitData
from chabuduo.config import ExperimentConfig


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
    ) -> None:
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels cannot be empty")
        if sequence_length % sampling_factor != 0:
            raise ValueError("sequence_length must be divisible by sampling_factor")

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.hidden_channels = tuple(hidden_channels)
        self.latent_dim = latent_dim
        self.sampling_factor = sampling_factor
        self.pooled_length = sequence_length // sampling_factor

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
    def kl_divergence(mean: Tensor, log_variance: Tensor) -> Tensor:
        """Return the mean closed-form KL divergence to N(0, I)."""
        per_sample = -0.5 * torch.sum(
            1 + log_variance - mean.square() - log_variance.exp(), dim=1
        )
        return per_sample.mean()

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
        latent = temperature * torch.randn(count, self.latent_dim, device=target_device)
        return self.decode(latent)


class TCVAELightning(L.LightningModule):
    """Lightning adapter for the TCVAE loss and optimizer."""

    def __init__(self, resolved_config: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"resolved_config": resolved_config})
        self.resolved_config = resolved_config
        model = resolved_config["model"]
        self.network = StandardTCVAE(
            input_channels=int(model["input_channels"]),
            sequence_length=int(model["sequence_length"]),
            hidden_channels=model["hidden_channels"],
            latent_dim=int(model["latent_dim"]),
            kernel_size=int(model["kernel_size"]),
            dilation_base=int(model["dilation_base"]),
            sampling_factor=int(model["sampling_factor"]),
            dropout=float(model["dropout"]),
        )
        loss_config = resolved_config["loss"]
        initial_scale = torch.tensor(float(loss_config.get("decoder_scale", 1.0)))
        self.log_decoder_scale = torch.nn.Parameter(
            initial_scale.log(),
            requires_grad=bool(loss_config.get("learn_decoder_scale", True)),
        )

    def compute_loss(self, value: Tensor) -> dict[str, Tensor]:
        reconstruction, mean, log_variance = self.network(value)
        log_scale = self.log_decoder_scale.clamp_min(-9.21034)
        scale = log_scale.exp()
        gaussian_nll = (
            0.5 * ((value - reconstruction) / scale).square()
            + log_scale
            + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=value.device))
        )
        reconstruction_loss = gaussian_nll.sum(dim=(1, 2)).mean()
        kl_loss = self.network.kl_divergence(mean, log_variance)
        loss_config = self.resolved_config["loss"]
        loss = (
            float(loss_config["reconstruction_weight"]) * reconstruction_loss
            + float(loss_config["kl_weight"]) * kl_loss
        )
        return {"loss": loss, "reconstruction": reconstruction_loss, "kl": kl_loss}

    def _shared_step(self, batch: tuple[Tensor], prefix: str) -> Tensor:
        value = batch[0].permute(0, 2, 1)
        losses = self.compute_loss(value)
        self.log_dict(
            {f"{prefix}/{name}": item for name, item in losses.items()},
            on_epoch=True,
            prog_bar=prefix == "validation",
            batch_size=value.shape[0],
        )
        return losses["loss"]

    def training_step(self, batch: tuple[Tensor], batch_index: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor], batch_index: int) -> Tensor:
        return self._shared_step(batch, "validation")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.resolved_config["optimizer"]
        return torch.optim.Adam(
            self.parameters(),
            lr=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer.get("weight_decay", 0)),
        )

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


def train_tcvae(
    artifact: Path,
    output: Path,
    config: ExperimentConfig,
    *,
    smoke: bool,
) -> Path:
    """Train a TCVAE from the ready arrays in a prepared artifact."""
    if config.model.type != "tcvae":
        raise TypeError("train_tcvae requires model.type=tcvae")
    train_data = SplitData.load(artifact / "train.npz")
    validation_data = SplitData.load(artifact / "validation.npz")
    sequence_length, input_channels = train_data.values.shape[1:]
    resolved = {
        "name": config.name,
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
        resolved["model"] = {
            **resolved["model"],
            "hidden_channels": [8, 8],
            "latent_dim": 4,
            "kernel_size": 3,
            "dropout": 0.0,
        }
        resolved["training"] = {
            **resolved["training"],
            "batch_size": 8,
            "max_epochs": 1,
            "num_workers": 0,
            "accelerator": "cpu",
            "devices": 1,
        }

    training = resolved["training"]
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best",
        monitor="validation/loss",
        mode="min",
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=int(training["max_epochs"]),
        accelerator=training.get("accelerator", "auto"),
        devices=training.get("devices", 1),
        callbacks=[checkpoint],
        logger=CSVLogger(save_dir=output / "logs", name="training"),
        deterministic=True,
        enable_autolog_hparams=False,
        enable_progress_bar=True,
        limit_train_batches=1 if smoke else 1.0,
        limit_val_batches=1 if smoke else 1.0,
    )
    trainer.fit(
        TCVAELightning(resolved),
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
    )
    return output
