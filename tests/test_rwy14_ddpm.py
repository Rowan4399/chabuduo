from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from chabuduo.artifacts import Normalization, SplitData, write_prepared_artifact
from chabuduo.config import config_path, load_experiment_config, selected_path
from chabuduo.models.ddpm import TemporalDenoiser, train_ddpm
from chabuduo.plotting import load_loss_series, plot_loss_curves
from chabuduo.validate import validate_artifact


def test_rwy14_config_and_cache_identity() -> None:
    config = load_experiment_config(
        config_path("zurich_rwy14_ddpm_unconditional")
    )
    assert config.preprocess.runways == ["14"]
    assert config.model.conditioning is None
    assert "rwy14" in selected_path(config).name


def test_rwy14_single_flow_config_has_distinct_artifacts() -> None:
    baseline = load_experiment_config(
        config_path("zurich_rwy14_ddpm_unconditional")
    )
    config = load_experiment_config(
        config_path("zurich_rwy14_flow162_216_ddpm_unconditional")
    )
    assert config.preprocess.runways == ["14"]
    assert config.preprocess.initial_flows == ["162-216"]
    assert config.model.conditioning is None
    assert "flow162-216" in selected_path(config).name
    assert selected_path(config) != selected_path(baseline)


def test_temporal_denoiser_is_unconditional_and_shape_preserving() -> None:
    model = TemporalDenoiser(
        feature_dim=4,
        hidden_dim=32,
        time_dim=32,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.0,
        use_phase=True,
    )
    inputs = torch.randn(3, 200, 4)
    result = model(inputs, torch.tensor([0, 10, 999]))
    assert result.shape == inputs.shape
    assert not any("condition" in name for name, _ in model.named_parameters())


def test_validator_accepts_rwy14_metadata(tmp_path) -> None:
    config = load_experiment_config(
        config_path("zurich_rwy14_ddpm_unconditional")
    )
    points = config.prepare.points
    features = len(config.prepare.representation.features)
    splits: dict[str, SplitData] = {}
    records: list[dict[str, object]] = []
    for name in ("train", "validation", "test"):
        identifiers = np.asarray(
            [f"{name}-{index}" for index in range(2)], dtype=str
        )
        values = np.zeros((2, points, features), dtype=np.float32)
        elapsed = np.tile(
            np.linspace(0, 900, points, dtype=np.float32), (2, 1)
        )
        splits[name] = SplitData(values, elapsed, identifiers, identifiers.copy())
        records.extend(
            {
                "trajectory_id": identifier,
                "split": name,
                "runway": "14",
                "initial_flow": "90-132",
                "simple": True,
            }
            for identifier in identifiers
        )
    artifact = tmp_path / "prepared"
    write_prepared_artifact(
        artifact,
        splits,
        Normalization.fit(splits["train"].values),
        pd.DataFrame.from_records(records),
    )
    messages = validate_artifact(artifact, config)
    assert len(messages) == 4


def test_ddpm_smoke_run_can_resume_one_additional_epoch(tmp_path) -> None:
    config = load_experiment_config(
        config_path("zurich_rwy14_flow162_216_ddpm_unconditional")
    )
    points = config.prepare.points
    features = len(config.prepare.representation.features)
    splits: dict[str, SplitData] = {}
    records: list[dict[str, object]] = []
    for name in ("train", "validation", "test"):
        identifiers = np.asarray([f"{name}-0"], dtype=str)
        values = np.zeros((1, points, features), dtype=np.float32)
        elapsed = np.linspace(0, 900, points, dtype=np.float32)[None]
        splits[name] = SplitData(values, elapsed, identifiers, identifiers.copy())
        records.append({"trajectory_id": identifiers[0], "split": name})
    artifact = tmp_path / "prepared"
    write_prepared_artifact(
        artifact,
        splits,
        Normalization.fit(splits["train"].values),
        pd.DataFrame.from_records(records),
    )
    output = tmp_path / "run"
    output.mkdir()

    train_ddpm(artifact, output, config, smoke=True)
    train_ddpm(
        artifact,
        output,
        config,
        smoke=True,
        resume=True,
        additional_epochs=1,
    )

    metrics = pd.read_csv(output / "logs" / "metrics.csv")
    assert metrics.epoch.tolist() == [1, 2]
    assert (output / "resume_sources" / "epoch-1" / "last.pt").is_file()
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf8"))
    assert summary["last"]["epoch"] == 2
    assert summary["resumes"][0]["source_epoch"] == 1


def test_loss_plot_supports_ddpm_metrics(tmp_path) -> None:
    run = tmp_path / "run"
    logs = run / "logs"
    logs.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [1.0, 0.8, 0.7],
            "validation_loss": [1.1, 0.9, 0.75],
        }
    ).to_csv(logs / "metrics.csv", index=False)

    train, validation = load_loss_series(run)
    output = plot_loss_curves(run, warmup_epochs=2)

    assert train.index.tolist() == [1, 2, 3]
    assert validation.tolist() == [1.1, 0.9, 0.75]
    assert output.is_file()
