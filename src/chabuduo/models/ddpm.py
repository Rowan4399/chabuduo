"""Conditional DDPM from the Zurich experiment on the Windows machine.

The denoiser is the experiment's dual feature/increment one-dimensional U-Net.
This module adapts its training loop to chabuduo's prepared artifacts and run
layout; it does not perform traffic preprocessing or reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from chabuduo.artifacts import SplitData
from chabuduo.config import ExperimentConfig


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("Embedding dimension must be even")
        self.register_buffer(
            "frequencies",
            torch.exp(
                torch.linspace(np.log(1.0), np.log(10_000.0), dimension // 2)
            ),
            persistent=False,
        )

    def forward(self, values: Tensor) -> Tensor:
        angles = values.float().unsqueeze(-1) / self.frequencies
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConditionalResBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        context_dim: int,
        local_dim: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * 2
        self.norm1 = nn.GroupNorm(_group_count(input_channels), input_channels)
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            5,
            padding=padding,
            dilation=dilation,
        )
        self.context = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, output_channels * 2)
        )
        self.local = nn.Sequential(nn.SiLU(), nn.Linear(local_dim, output_channels))
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            5,
            padding=padding,
            dilation=dilation,
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, 1)
        )

    def forward(self, features: Tensor, context: Tensor, local: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(features)))
        scale, shift = self.context(context).chunk(2, dim=-1)
        hidden = hidden * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        hidden = hidden + self.local(local).transpose(1, 2)
        return self.conv2(F.silu(self.norm2(hidden))) + self.skip(features)


class FeatureUNet(nn.Module):
    """Dual feature/increment conditional U-Net used by the Windows run."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        condition_count: int,
    ) -> None:
        super().__init__()
        if feature_dim != 4 or hidden_dim < 64 or hidden_dim % 2:
            raise ValueError("Expected four features and an even hidden_dim >= 64")
        base = local_dim = hidden_dim // 2
        context_dim = hidden_dim * 2
        depth = max(2, num_layers // 2)
        self.feature_dim = feature_dim
        self.phase_embedding = SinusoidalEmbedding(local_dim)
        self.local_fusion = nn.Sequential(
            nn.Linear(local_dim, local_dim),
            nn.SiLU(),
            nn.Linear(local_dim, local_dim),
        )
        self.diffusion_embedding = SinusoidalEmbedding(context_dim)
        self.condition_embedding = nn.Embedding(condition_count, context_dim)
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )
        self.feature_input = nn.Conv1d(feature_dim, base, 5, padding=2)
        self.increment_input = nn.Conv1d(feature_dim, base, 5, padding=2)
        self.input_fusion = nn.Conv1d(base * 2, base, 1)
        self.encoder0 = nn.ModuleList(
            ConditionalResBlock(base, base, context_dim, local_dim)
            for _ in range(depth)
        )
        self.down0 = nn.Conv1d(base, base * 2, 4, stride=2, padding=1)
        self.encoder1 = nn.ModuleList(
            ConditionalResBlock(base * 2, base * 2, context_dim, local_dim)
            for _ in range(depth)
        )
        self.down1 = nn.Conv1d(base * 2, base * 4, 4, stride=2, padding=1)
        dilations = (1, 2, 4)
        self.bottleneck = nn.ModuleList(
            ConditionalResBlock(
                base * 4,
                base * 4,
                context_dim,
                local_dim,
                dilations[index % len(dilations)],
            )
            for index in range(depth + 1)
        )
        self.up1 = nn.Conv1d(base * 4, base * 2, 3, padding=1)
        self.decoder1 = nn.ModuleList(
            ConditionalResBlock(
                base * 4 if index == 0 else base * 2,
                base * 2,
                context_dim,
                local_dim,
            )
            for index in range(depth)
        )
        self.up0 = nn.Conv1d(base * 2, base, 3, padding=1)
        self.decoder0 = nn.ModuleList(
            ConditionalResBlock(
                base * 2 if index == 0 else base,
                base,
                context_dim,
                local_dim,
            )
            for index in range(depth)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(base), base),
            nn.SiLU(),
            nn.Conv1d(base, feature_dim, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _resize(local: Tensor, length: int) -> Tensor:
        return F.interpolate(
            local.transpose(1, 2),
            size=length,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

    def forward(
        self,
        noisy_features: Tensor,
        diffusion_step: Tensor,
        condition_id: Tensor,
    ) -> Tensor:
        if (
            noisy_features.ndim != 3
            or noisy_features.shape[-1] != self.feature_dim
            or noisy_features.shape[1] % 4
        ):
            raise ValueError(
                "Expected [batch, sequence length divisible by 4, four features]"
            )
        length = noisy_features.shape[1]
        phase = torch.linspace(0, 1, length, device=noisy_features.device)[
            None
        ].expand(len(noisy_features), -1)
        local0 = self.local_fusion(self.phase_embedding(phase))
        context = self.context_fusion(
            torch.cat(
                (
                    self.diffusion_embedding(diffusion_step),
                    self.condition_embedding(condition_id),
                ),
                dim=-1,
            )
        )
        direct = self.feature_input(noisy_features.transpose(1, 2))
        increments = torch.diff(
            noisy_features, dim=1, prepend=noisy_features[:, :1]
        ) / np.sqrt(2.0)
        increment = self.increment_input(increments.transpose(1, 2))
        level0 = self.input_fusion(torch.cat((direct, increment), dim=1))
        for block in self.encoder0:
            level0 = block(level0, context, local0)
        level1 = self.down0(level0)
        local1 = self._resize(local0, length // 2)
        for block in self.encoder1:
            level1 = block(level1, context, local1)
        middle = self.down1(level1)
        local2 = self._resize(local0, length // 4)
        for block in self.bottleneck:
            middle = block(middle, context, local2)
        decoded1 = torch.cat(
            (
                self.up1(
                    F.interpolate(
                        middle,
                        size=level1.shape[-1],
                        mode="linear",
                        align_corners=True,
                    )
                ),
                level1,
            ),
            dim=1,
        )
        for block in self.decoder1:
            decoded1 = block(decoded1, context, local1)
        decoded0 = torch.cat(
            (
                self.up0(
                    F.interpolate(
                        decoded1,
                        size=level0.shape[-1],
                        mode="linear",
                        align_corners=True,
                    )
                ),
                level0,
            ),
            dim=1,
        )
        for block in self.decoder0:
            decoded0 = block(decoded0, context, local0)
        return self.output(decoded0).transpose(1, 2)


class ConditionalDDPM(nn.Module):
    """Noise-prediction DDPM over four-feature trajectories."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        condition_count: int,
        diffusion_steps: int,
    ) -> None:
        super().__init__()
        self.denoiser = FeatureUNet(
            feature_dim, hidden_dim, num_layers, condition_count
        )
        self.diffusion_steps = diffusion_steps
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    def loss(self, clean: Tensor, condition: Tensor) -> Tensor:
        step = torch.randint(
            0, self.diffusion_steps, (len(clean),), device=clean.device
        )
        noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[step][:, None, None]
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return F.mse_loss(self.denoiser(noisy, step, condition), noise)

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        condition_id: int,
        sequence_length: int,
        *,
        steps: int,
        eta: float,
    ) -> np.ndarray:
        if not 2 <= steps <= self.diffusion_steps:
            raise ValueError("steps must be between 2 and diffusion_steps")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0, 1]")
        device = next(self.parameters()).device
        condition = torch.full(
            (count,), condition_id, dtype=torch.long, device=device
        )
        schedule = (
            torch.linspace(self.diffusion_steps - 1, 0, steps, device=device)
            .round()
            .long()
            .unique_consecutive()
        )
        generated = torch.randn(
            count,
            sequence_length,
            self.denoiser.feature_dim,
            device=device,
        )
        self.eval()
        for index, current in enumerate(schedule):
            step = torch.full(
                (count,), int(current), device=device, dtype=torch.long
            )
            noise = self.denoiser(generated, step, condition)
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
                    torch.clamp(1 - previous_alpha_bar - sigma.square(), min=0)
                )
                * noise
            )
            if eta > 0 and previous is not None:
                generated += sigma * torch.randn_like(generated)
        return generated.cpu().numpy()


def _conditions(
    metadata: pd.DataFrame,
    data: SplitData,
    feature: str | None,
) -> Tensor:
    if feature is None:
        return torch.zeros(len(data.values), dtype=torch.long)
    indexed = metadata.assign(
        trajectory_id=metadata.trajectory_id.astype(str)
    ).set_index("trajectory_id")
    values = indexed.loc[data.trajectory_id, feature].to_numpy()
    if not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
        raise ValueError(
            f"Conditioning feature {feature!r} must contain nonnegative IDs"
        )
    return torch.from_numpy(values.astype(np.int64, copy=False))


def _loader(
    data: SplitData,
    conditions: Tensor,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(data.values), conditions),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def _device(config: ExperimentConfig, smoke: bool) -> torch.device:
    if smoke or config.training.accelerator == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def train_ddpm(
    artifact: Path,
    output: Path,
    config: ExperimentConfig,
    *,
    smoke: bool,
) -> Path:
    """Train the Windows conditional-DDPM implementation on prepared arrays."""
    if config.model.type != "ddpm":
        raise TypeError("train_ddpm requires model.type=ddpm")
    train_data = SplitData.load(artifact / "train.npz")
    validation_data = SplitData.load(artifact / "validation.npz")
    feature_dim = train_data.values.shape[2]
    metadata = pd.read_parquet(artifact / "metadata.parquet")
    condition_feature = (
        config.model.conditioning.feature
        if config.model.conditioning is not None
        else None
    )
    train_conditions = _conditions(metadata, train_data, condition_feature)
    validation_conditions = _conditions(
        metadata, validation_data, condition_feature
    )
    condition_count = (
        int(metadata[condition_feature].max()) + 1
        if condition_feature is not None
        else 1
    )
    hidden_dim = 64 if smoke else config.model.hidden_channels
    diffusion_steps = 20 if smoke else config.model.diffusion_steps
    model = ConditionalDDPM(
        feature_dim,
        hidden_dim,
        config.model.num_layers,
        condition_count,
        diffusion_steps,
    )
    device = _device(config, smoke)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    batch_size = 8 if smoke else config.training.batch_size
    workers = 0 if smoke else config.training.num_workers
    epochs = 1 if smoke else config.training.max_epochs
    train_loader = _loader(
        train_data, train_conditions, batch_size, workers, True
    )
    validation_loader = _loader(
        validation_data, validation_conditions, batch_size, workers, False
    )
    amp_enabled = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir(parents=True)
    best = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch_index, (clean, condition) in enumerate(train_loader):
            clean, condition = clean.to(device), condition.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = model.loss(clean, condition)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config.training.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
            train_total += loss.item() * len(clean)
            train_count += len(clean)
            if smoke and batch_index == 0:
                break

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch_index, (clean, condition) in enumerate(validation_loader):
                clean, condition = clean.to(device), condition.to(device)
                loss = model.loss(clean, condition)
                validation_total += loss.item() * len(clean)
                validation_count += len(clean)
                if smoke and batch_index == 0:
                    break
        row = {
            "epoch": epoch,
            "train_loss": train_total / train_count,
            "validation_loss": validation_total / validation_count,
        }
        history.append(row)
        state = {
            "model_state_dict": model.state_dict(),
            "feature_dim": feature_dim,
            "condition_count": condition_count,
            "hidden_dim": hidden_dim,
            "num_layers": config.model.num_layers,
            "diffusion_steps": diffusion_steps,
            "epoch": epoch,
            "validation_loss": row["validation_loss"],
        }
        torch.save(state, checkpoints / "last.pt")
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            torch.save(state, checkpoints / "best.pt")

    pd.DataFrame(history).to_csv(logs / "metrics.csv", index=False)
    return output
