"""Validate a prepared artifact and dispatch to its configured model trainer."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

import lightning as L

from chabuduo.config import ExperimentConfig
from chabuduo.validate import validate_artifact


def _environment() -> dict[str, str]:
    packages = ["chabuduo", "lightning", "torch", "traffic", "numpy", "pandas"]
    values = {"python": platform.python_version(), "platform": platform.platform()}
    for package in packages:
        try:
            values[package] = version(package)
        except Exception:
            values[package] = "unknown"
    return values


def train(
    artifact: Path,
    output: Path,
    config: ExperimentConfig,
    *,
    smoke: bool = False,
    resume: bool = False,
    additional_epochs: int | None = None,
    learning_rate: float | None = None,
) -> Path:
    """Dispatch training to the model selected in the experiment YAML."""
    validate_artifact(artifact, config)
    if not resume and (additional_epochs is not None or learning_rate is not None):
        raise ValueError("--additional-epochs and --learning-rate require --resume")
    if output.exists() and not resume:
        raise FileExistsError(f"{output} already exists")
    if resume:
        suffix = "ckpt" if config.model.type == "tcvae" else "pt"
        checkpoint = output / "checkpoints" / f"last.{suffix}"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"No resumable checkpoint at {checkpoint}")
    else:
        output.mkdir(parents=True)
        (output / "environment.json").write_text(
            json.dumps(_environment(), indent=2) + "\n", encoding="utf8"
        )
    L.seed_everything(config.training.seed, workers=True)

    if config.model.type == "tcvae":
        from chabuduo.models.tcvae import train_tcvae

        return train_tcvae(
            artifact,
            output,
            config,
            smoke=smoke,
            resume=resume,
            additional_epochs=additional_epochs,
            learning_rate=learning_rate,
        )
    if config.model.type == "ddpm":
        from chabuduo.models.ddpm import train_ddpm

        return train_ddpm(
            artifact,
            output,
            config,
            smoke=smoke,
            resume=resume,
            additional_epochs=additional_epochs,
            learning_rate=learning_rate,
        )
    raise ValueError(f"Unsupported model type: {config.model.type}")
