"""Conditional and unconditional trajectory diffusion models."""

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
    SplitData,
    prepared_artifact_identity,
    sha256,
    write_json,
)
from chabuduo.config import ExperimentConfig


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("Embedding dimension must be even")
        self.register_buffer(
            "frequencies",
            torch.exp(torch.linspace(np.log(1.0), np.log(10_000.0), dimension // 2)),
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
        phase = torch.linspace(0, 1, length, device=noisy_features.device)[None].expand(
            len(noisy_features), -1
        )
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


class TemporalResidualBlock(nn.Module):
    """Non-causal dilated residual block conditioned only on diffusion time."""

    def __init__(
        self,
        channels: int,
        time_dim: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels))
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, features: Tensor, time: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(features)))
        hidden = hidden + self.time(time).unsqueeze(-1)
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return features + hidden


class TemporalDenoiser(nn.Module):
    """Full-resolution one-dimensional denoiser for fixed-length trajectories."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        time_dim: int,
        *,
        kernel_size: int,
        dilations: tuple[int, ...],
        dropout: float,
        use_phase: bool,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TemporalDenoiser requires an odd kernel size")
        self.feature_dim = feature_dim
        self.use_phase = use_phase
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        input_dim = feature_dim + int(use_phase)
        self.input = nn.Conv1d(
            input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(
                hidden_dim,
                time_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in dilations
        )
        self.output_norm = nn.GroupNorm(_group_count(hidden_dim), hidden_dim)
        self.output = nn.Conv1d(hidden_dim, feature_dim, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_features: Tensor, diffusion_step: Tensor) -> Tensor:
        if noisy_features.ndim != 3 or noisy_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected [batch, sequence, {self.feature_dim}] noisy features"
            )
        inputs = noisy_features.transpose(1, 2)
        if self.use_phase:
            phase = torch.linspace(
                -1,
                1,
                noisy_features.shape[1],
                device=noisy_features.device,
                dtype=noisy_features.dtype,
            )
            phase = phase[None, None].expand(len(noisy_features), -1, -1)
            inputs = torch.cat((inputs, phase), dim=1)
        hidden = self.input(inputs)
        time = self.time_embedding(diffusion_step)
        for block in self.blocks:
            hidden = block(hidden, time)
        return self.output(F.silu(self.output_norm(hidden))).transpose(1, 2)


class UnconditionalDDPM(nn.Module):
    """Noise-prediction DDPM without runway or flow embeddings."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        time_dim: int,
        *,
        kernel_size: int,
        dilations: tuple[int, ...],
        dropout: float,
        use_phase: bool,
        diffusion_steps: int,
    ) -> None:
        super().__init__()
        self.denoiser = TemporalDenoiser(
            feature_dim,
            hidden_dim,
            time_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            use_phase=use_phase,
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
        alpha_bar = self.alpha_bars[step][:, None, None]
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return F.mse_loss(self.denoiser(noisy, step), noise)

    @torch.inference_mode()
    def sample(
        self,
        count: int,
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
        schedule = (
            torch.linspace(self.diffusion_steps - 1, 0, steps, device=device)
            .round()
            .long()
            .unique_consecutive()
        )
        generated = torch.randn(
            count, sequence_length, self.denoiser.feature_dim, device=device
        )
        self.eval()
        for index, current in enumerate(schedule):
            step = torch.full((count,), int(current), device=device, dtype=torch.long)
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
                    torch.clamp(1 - previous_alpha_bar - sigma.square(), min=0)
                )
                * noise
            )
            if eta > 0 and previous is not None:
                generated += sigma * torch.randn_like(generated)
        return generated.cpu().numpy()


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

    def loss(
        self,
        clean: Tensor,
        condition: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
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
        condition = torch.full((count,), condition_id, dtype=torch.long, device=device)
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
            step = torch.full((count,), int(current), device=device, dtype=torch.long)
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


def sample_ddpm(
    checkpoint: Path,
    count: int,
    sequence_length: int,
    *,
    steps: int,
    eta: float,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Load a DDPM checkpoint and return normalized ``[N, T, F]`` values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture = state.get("architecture", "dual_branch_1d_unet")
    if architecture == "temporal_dilated_resnet":
        model = UnconditionalDDPM(
            feature_dim=int(state["feature_dim"]),
            hidden_dim=int(state["hidden_dim"]),
            time_dim=int(state["time_dim"]),
            kernel_size=int(state["kernel_size"]),
            dilations=tuple(int(value) for value in state["dilations"]),
            dropout=float(state["dropout"]),
            use_phase=bool(state["use_phase"]),
            diffusion_steps=int(state["diffusion_steps"]),
        )
        model.load_state_dict(
            state.get("ema_model_state_dict") or state["model_state_dict"]
        )
        model.to(device)
        model.eval()
        return model.sample(
            count,
            sequence_length,
            steps=steps,
            eta=eta,
        )

    if condition_ids is None:
        raise ValueError("A conditional checkpoint requires condition IDs")
    if len(condition_ids) != count:
        raise ValueError("condition_ids length must match count")
    model = ConditionalDDPM(
        feature_dim=int(state["feature_dim"]),
        hidden_dim=int(state["hidden_dim"]),
        num_layers=int(state["num_layers"]),
        condition_count=int(state["condition_count"]),
        diffusion_steps=int(state["diffusion_steps"]),
    )
    model.load_state_dict(
        state.get("ema_model_state_dict") or state["model_state_dict"]
    )
    model.to(device)
    model.eval()
    result = np.empty(
        (count, sequence_length, model.denoiser.feature_dim),
        dtype=np.float32,
    )
    for condition_id in np.unique(condition_ids):
        rows = np.flatnonzero(condition_ids == condition_id)
        result[rows] = model.sample(
            len(rows),
            int(condition_id),
            sequence_length,
            steps=steps,
            eta=eta,
        )
    return result


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
    conditions: Tensor | None,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    tensors = (torch.from_numpy(data.values),)
    if conditions is not None:
        tensors += (conditions,)
    return DataLoader(
        TensorDataset(*tensors),
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
    resume: bool = False,
    additional_epochs: int | None = None,
    learning_rate: float | None = None,
) -> Path:
    """Train the configured trajectory DDPM on prepared arrays."""
    if config.model.type != "ddpm":
        raise TypeError("train_ddpm requires model.type=ddpm")
    current_artifact = prepared_artifact_identity(artifact)
    train_data = SplitData.load(artifact / "train.npz")
    validation_data = SplitData.load(artifact / "validation.npz")
    feature_dim = train_data.values.shape[2]
    hidden_dim = 64 if smoke else config.model.hidden_channels
    diffusion_steps = config.model.diffusion_steps
    metadata = pd.read_parquet(artifact / "metadata.parquet")
    condition_feature = (
        config.model.conditioning.feature
        if config.model.conditioning is not None
        else None
    )
    checkpoint_spec: dict[str, object]
    if config.model.architecture == "temporal_dilated_resnet":
        train_conditions = validation_conditions = None
        model: nn.Module = UnconditionalDDPM(
            feature_dim,
            hidden_dim,
            config.model.time_embedding_dim,
            kernel_size=config.model.kernel_size,
            dilations=tuple(config.model.dilations),
            dropout=config.model.dropout,
            use_phase=config.model.use_phase,
            diffusion_steps=diffusion_steps,
        )
        checkpoint_spec = {
            "architecture": config.model.architecture,
            "time_dim": config.model.time_embedding_dim,
            "kernel_size": config.model.kernel_size,
            "dilations": config.model.dilations,
            "dropout": config.model.dropout,
            "use_phase": config.model.use_phase,
        }
    else:
        train_conditions = _conditions(metadata, train_data, condition_feature)
        validation_conditions = _conditions(
            metadata, validation_data, condition_feature
        )
        condition_count = (
            int(metadata[condition_feature].max()) + 1
            if condition_feature is not None
            else 1
        )
        model = ConditionalDDPM(
            feature_dim,
            hidden_dim,
            config.model.num_layers,
            condition_count,
            diffusion_steps,
        )
        checkpoint_spec = {
            "architecture": config.model.architecture,
            "condition_count": condition_count,
            "num_layers": config.model.num_layers,
        }
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
    train_loader = _loader(train_data, train_conditions, batch_size, workers, True)
    validation_loader = _loader(
        validation_data, validation_conditions, batch_size, workers, False
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
            last_checkpoint, map_location=device, weights_only=False
        )
        if checkpoint.get("config_name", config.name) != config.name:
            raise ValueError("Checkpoint and experiment configuration do not match")
        if (
            checkpoint.get("artifact_sha256", current_artifact["sha256"])
            != current_artifact["sha256"]
        ):
            raise ValueError("Prepared artifact changed since the run started")
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
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        start_epoch = source_epoch + 1
        print(f"resumed_from_epoch={source_epoch}", flush=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"architecture={config.model.architecture} parameters={parameter_count:,} "
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
        for batch_index, batch in enumerate(train_loader):
            clean = batch[0].to(device)
            condition = batch[1].to(device) if len(batch) == 2 else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                if condition is None:
                    loss = model.loss(clean)
                else:
                    loss = model.loss(clean, condition)
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
                if decay is None:  # pragma: no cover - guarded at construction
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
            for batch_index, batch in enumerate(validation_loader):
                clean = batch[0].to(device)
                condition = batch[1].to(device) if len(batch) == 2 else None
                if condition is None:
                    loss = model.loss(clean, generator=validation_generator)
                else:
                    loss = model.loss(clean, condition, generator=validation_generator)
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
        state = {
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": (
                ema_model.state_dict() if ema_model is not None else None
            ),
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "diffusion_steps": diffusion_steps,
            "epoch": epoch,
            "validation_loss": row["validation_loss"],
            "best_validation_loss": best,
            "config_name": config.name,
            "artifact_sha256": current_artifact["sha256"],
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
            **checkpoint_spec,
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
        "model_type": "ddpm",
        "config_name": config.name,
        "artifact": current_artifact,
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
