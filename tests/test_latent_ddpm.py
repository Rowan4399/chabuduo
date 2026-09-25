from __future__ import annotations

import torch

from chabuduo.config import config_path, load_experiment_config, prepared_path
from chabuduo.models.latent_ddpm import LatentDDPM, LatentMLPDenoiser


def test_latent_ddpm_config_reuses_tcvae_artifact() -> None:
    config = load_experiment_config(
        config_path("zurich_rwy14_flow162_216_latent_ddpm_mlp")
    )
    assert config.model.type == "latent_ddpm"
    assert config.model.latent.representation == "posterior_sample"
    assert config.model.residual_blocks == 4
    assert config.model.hidden_dim == 256
    assert prepared_path(config).name == "zurich_rwy14_flow162_216_tcvae-v1"


def test_latent_mlp_denoiser_preserves_vector_shape() -> None:
    denoiser = LatentMLPDenoiser(
        latent_dim=64,
        hidden_dim=256,
        time_dim=128,
        residual_blocks=4,
        dropout=0.0,
    )
    latent = torch.randn(3, 64)
    result = denoiser(latent, torch.tensor([0, 10, 999]))
    assert result.shape == latent.shape


def test_latent_ddpm_loss_and_sampling_shapes() -> None:
    model = LatentDDPM(
        latent_dim=8,
        hidden_dim=16,
        time_dim=8,
        residual_blocks=2,
        dropout=0.0,
        diffusion_steps=10,
    )
    latent = torch.randn(4, 8)
    assert model.loss(latent).ndim == 0
    assert model.sample(3, steps=2, eta=0.0).shape == (3, 8)
