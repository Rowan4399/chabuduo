from __future__ import annotations

import json
from types import SimpleNamespace

import lightning as L
import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from typer.testing import CliRunner

from chabuduo.artifacts import (
    SPLITS,
    Normalization,
    SplitData,
    sha256,
    write_prepared_artifact,
)
from chabuduo.cli import app
from chabuduo.config import (
    ExperimentConfig,
    config_path,
    load_experiment_config,
    prepared_path,
)
from chabuduo.models import latent_ddpm
from chabuduo.models.latent_ddpm import LatentDDPM, LatentMLPDenoiser
from chabuduo.models.tcvae import TCVAELightning
from chabuduo.sampling import sample_run
from chabuduo.validate import ValidationError, validate_artifact

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """Use a tiny frozen VampPrior and synthetic splits; never fetch aviation data."""
    config = load_experiment_config(
        config_path("tianjin_noaib_rwy16l_latent_ddpm_vamp_k1000")
    ).model_dump(mode="json")
    config["name"] = "latent-test"
    config["prepare"]["points"] = 20
    config["model"].update(
        hidden_dim=16, residual_blocks=1, time_embedding_dim=8, diffusion_steps=10
    )
    config["model"]["latent"]["dimension"] = 4
    config["sampling"].update(count=3, steps=2)
    config["training"].update(accelerator="cpu", max_epochs=2)
    resolved = {
        "model": {
            "input_channels": 4,
            "sequence_length": 20,
            "hidden_channels": [4, 4],
            "latent_dim": 4,
            "kernel_size": 3,
            "dilation_base": 2,
            "sampling_factor": 2,
            "dropout": 0.0,
            "prior": {"type": "vamp", "components": 3},
        },
        "loss": {"reconstruction_weight": 1.0, "kl_weight": 1.0},
        "optimizer": config["optimizer"],
    }
    torch.manual_seed(42)
    module = TCVAELightning(resolved)
    checkpoint = tmp_path / "autoencoder" / "best.ckpt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "epoch": 3,
            "pytorch-lightning_version": L.__version__,
            "state_dict": module.state_dict(),
            "hyper_parameters": {"resolved_config": resolved},
        },
        checkpoint,
    )
    artifact = tmp_path / "prepared"
    config["model"]["autoencoder"].update(
        run=str(checkpoint.parent),
        checkpoint=checkpoint.name,
        prepared_artifact=str(artifact),
        expected_epoch=3,
        expected_sha256=sha256(checkpoint),
        expected_prior="vamp",
    )
    rng = np.random.default_rng(42)
    raw = {
        name: rng.normal(size=(size, 20, 4)).astype(np.float32)
        for name, size in zip(SPLITS, (6, 3, 3), strict=True)
    }
    normalization = Normalization.fit(raw["train"])
    splits, rows = {}, []
    for name, values in raw.items():
        ids = np.asarray([f"{name}-{i}" for i in range(len(values))])
        elapsed = np.tile(np.linspace(0, 900, 20), (len(values), 1)).astype(np.float32)
        splits[name] = SplitData(normalization.transform(values), elapsed, ids, ids)
        rows.extend(
            {"trajectory_id": ident, "split": name, "runway": "16L"} for ident in ids
        )
    write_prepared_artifact(artifact, splits, normalization, pd.DataFrame(rows))
    parsed = ExperimentConfig.model_validate(config)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf8")
    threshold = SimpleNamespace(name="16L", latitude=39.1, longitude=117.2)
    airport = SimpleNamespace(
        latitude=39.0,
        longitude=117.0,
        runways={"16L": SimpleNamespace(tuple_runway=[threshold])},
    )
    for name in ("prepare", "sampling", "plotting"):
        monkeypatch.setattr(f"chabuduo.{name}.airports", {"ZBTJ": airport})
    return parsed, path, artifact, tmp_path / "run", checkpoint


@pytest.mark.parametrize(
    "name",
    [
        "zurich_rwy14_latent_ddpm_vamp_k1000",
        "zurich_rwy14_flow162_216_latent_ddpm_mlp",
        "tianjin_noaib_rwy16l_latent_ddpm_vamp_k1000",
    ],
)
def test_pinned_configs_reuse_autoencoder_artifact(name):
    config = load_experiment_config(config_path(name))
    assert config.model.latent.representation == "posterior_sample"
    assert config.model.hidden_dim == 256
    assert config.model.residual_blocks == 4
    assert (
        prepared_path(config)
        .as_posix()
        .endswith(config.model.autoencoder.prepared_artifact)
    )


def test_latent_model_shapes_and_repeatability():
    denoiser = LatentMLPDenoiser(4, 16, 8, residual_blocks=2, dropout=0.0)
    assert denoiser(torch.randn(3, 4), torch.tensor([0, 1, 9])).shape == (3, 4)
    model = LatentDDPM(4, 16, 8, residual_blocks=2, dropout=0, diffusion_steps=10)
    clean = torch.randn(3, 4)
    loss = model.loss(clean, generator=torch.Generator().manual_seed(42))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.denoiser.output.weight.grad is not None
    torch.manual_seed(42)
    first = model.sample(3, steps=2, eta=0.0)
    torch.manual_seed(42)
    np.testing.assert_array_equal(first, model.sample(3, steps=2, eta=0.0))


def test_cli_smoke_resume_sample_and_plot(experiment, monkeypatch):
    config, path, artifact, run, checkpoint = experiment
    runner = CliRunner()
    args = [config.name, "--config", str(path)]
    before = sha256(checkpoint)
    for command in ("prepare", "validate"):
        result = runner.invoke(app, [command, *args])
        assert result.exit_code == 0, result.output
        assert str(artifact) in result.output
    result = runner.invoke(app, ["train", *args, "--smoke", "--output", str(run)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert (run / "environment.json").is_file()
    assert (run / "configs/experiment.yaml").is_file()
    manifest = json.loads((run / "artifacts/latent_manifest.json").read_text())
    assert manifest["counts"] == {"train": 6, "validation": 3, "test": 3}
    with np.load(run / "artifacts/latent_dataset.npz") as data:
        normalization = json.loads(
            (run / "artifacts/latent_normalization.json").read_text()
        )
        np.testing.assert_allclose(
            normalization["mean"], data["train_latent"].mean(axis=0, dtype=np.float64)
        )
        np.testing.assert_allclose(
            data["train_standardized"].mean(axis=0), 0, atol=1e-6
        )
    latent_hash = sha256(run / "artifacts/latent_dataset.npz")

    def no_reencoding(*args, **kwargs):
        pytest.fail("A warm resume must reuse the cached posterior samples")

    monkeypatch.setattr(latent_ddpm, "_encode_split", no_reencoding)
    changed_config = config.model_copy(deep=True)
    changed_config.model.latent.seed += 1
    with pytest.raises(ValueError, match="Latent encoding settings changed"):
        latent_ddpm._prepare_latent_artifact(
            artifact, run, changed_config, smoke=True, resume=True
        )
    result = runner.invoke(
        app,
        [
            "train",
            *args,
            "--smoke",
            "--output",
            str(run),
            "--resume",
            "--additional-epochs",
            "1",
            "--learning-rate",
            "0.00005",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert sha256(run / "artifacts/latent_dataset.npz") == latent_hash
    assert (run / "resume_sources/epoch-1/last.pt").is_file()
    state = torch.load(run / "checkpoints/last.pt", weights_only=False)
    assert state["epoch"] == 2
    assert state["optimizer_state_dict"]["param_groups"][0]["lr"] == 5e-5
    assert pd.read_csv(run / "logs/metrics.csv").epoch.tolist() == [1, 2]
    sample = run / "test-samples"
    result = runner.invoke(
        app, ["sample", *args, "--run", str(run), "--output", str(sample)]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    with np.load(sample / "samples.npz") as data:
        assert data["values"].shape == (3, 20, 4)
        assert data["generated_latent"].shape == (3, 4)
        assert data["generated_latent_standardized"].shape == (3, 4)
        assert np.isfinite(data["values"]).all()
    generated = pd.read_parquet(sample / "generated.parquet")
    endpoints = generated.groupby("flight_id", sort=False).tail(1)
    np.testing.assert_allclose(endpoints.latitude, 39.1)
    np.testing.assert_allclose(endpoints.longitude, 117.2)
    for command in (
        ["plot", *args, "--sample", str(sample)],
        ["plot-loss", str(run)],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (result.output, result.exception)
    assert (sample / "trajectory_comparison.png").stat().st_size > 1000
    assert (run / "plots/loss.png").stat().st_size > 1000
    assert sha256(checkpoint) == before

    test_path = artifact / "test.npz"
    test_bytes = test_path.read_bytes()
    data = SplitData.load(test_path)
    data.values[0, 0, 0] += 1
    data.save(test_path)
    try:
        with pytest.raises(ValueError, match="Prepared artifact does not match"):
            sample_run(run, artifact, config, output=run / "changed-artifact")
    finally:
        test_path.write_bytes(test_bytes)

    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="TCVAE checkpoint changed"):
        sample_run(run, artifact, config, output=run / "changed-checkpoint")


def test_validation_rejects_split_group_leakage(experiment):
    config, _, artifact, _, _ = experiment
    data = SplitData.load(artifact / "test.npz")
    groups = data.split_group_id.astype("U32")
    groups[0] = "train-0"
    SplitData(data.values, data.elapsed_time, data.trajectory_id, groups).save(
        artifact / "test.npz"
    )
    with pytest.raises(ValidationError, match="Split group leakage"):
        validate_artifact(artifact, config)


def test_invalid_sampling_steps_rejected(experiment):
    config, *_ = experiment
    raw = config.model_dump(mode="json")
    raw["sampling"]["steps"] = 11
    with pytest.raises(ValueError, match="sampling steps"):
        ExperimentConfig.model_validate(raw)


@pytest.mark.parametrize("prior", ["standard", "vamp"])
def test_frozen_tcvae_prior_roundtrip(experiment, prior):
    _, _, _, _, checkpoint = experiment
    state = torch.load(checkpoint, weights_only=False)
    resolved = state["hyper_parameters"]["resolved_config"]
    resolved["model"]["prior"] = {"type": prior, "components": 3}
    module = TCVAELightning(resolved).eval()
    state["state_dict"] = module.state_dict()
    torch.save(state, checkpoint)
    loaded = TCVAELightning.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    ).eval()
    inputs = torch.randn(2, 4, 20)
    with torch.inference_mode():
        mean, _ = module.network.encode(inputs)
        restored_mean, _ = loaded.network.encode(inputs)
        torch.testing.assert_close(mean, restored_mean, rtol=0, atol=0)
        torch.testing.assert_close(
            module.network.decode(mean), loaded.network.decode(mean), rtol=0, atol=0
        )
        assert loaded.network.sample(2).shape == (2, 4, 20)
        assert all(
            torch.isfinite(value) for value in loaded.compute_loss(inputs).values()
        )
