"""Standard-prior temporal convolutional variational autoencoder.

The temporal blocks and encoder/decoder layout are adapted from the MIT-licensed
Code Ocean capsule "Deep Generative Modelling of Aircraft Trajectories in
Terminal Maneuvering Areas", DOI 10.24433/CO.0357574.V1. See NOTICE.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


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
