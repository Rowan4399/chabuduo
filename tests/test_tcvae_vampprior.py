from __future__ import annotations

import torch

from chabuduo.models.tcvae import StandardTCVAE, TCVAELightning


def _network(*, prior_type: str = "vamp") -> StandardTCVAE:
    return StandardTCVAE(
        input_channels=2,
        sequence_length=8,
        hidden_channels=(4, 4),
        latent_dim=3,
        kernel_size=3,
        sampling_factor=2,
        dropout=0.0,
        prior_type=prior_type,
        prior_components=4 if prior_type == "vamp" else None,
    )


def test_vampprior_kl_updates_pseudo_inputs() -> None:
    network = _network()
    initial = torch.randn(4, 2, 8)
    network.initialize_pseudo_inputs(initial)
    assert torch.equal(network.pseudo_inputs, initial)

    values = torch.randn(5, 2, 8)
    mean, log_variance = network.encode(values)
    latent = network.reparameterize(mean, log_variance)
    loss = network.kl_divergence(mean, log_variance, latent)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert network.pseudo_inputs.grad is not None
    assert torch.isfinite(network.pseudo_inputs.grad).all()


def test_vampprior_sampling_uses_configured_components() -> None:
    network = _network().eval()
    network.initialize_pseudo_inputs(torch.randn(4, 2, 8))

    samples = network.sample(6)

    assert samples.shape == (6, 2, 8)
    assert torch.isfinite(samples).all()


def test_missing_prior_configuration_remains_standard_normal() -> None:
    resolved = {
        "model": {
            "input_channels": 2,
            "sequence_length": 8,
            "hidden_channels": [4, 4],
            "latent_dim": 3,
            "kernel_size": 3,
            "dilation_base": 2,
            "sampling_factor": 2,
            "dropout": 0.0,
        },
        "loss": {
            "decoder_scale": 1.0,
            "learn_decoder_scale": True,
            "reconstruction_weight": 1.0,
            "kl_weight": 1.0,
        },
        "optimizer": {"learning_rate": 0.001},
        "seed": 42,
    }

    module = TCVAELightning(resolved)

    assert module.network.prior_type == "standard"
    assert module.network.pseudo_inputs is None
