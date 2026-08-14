"""The public command-line interface for the manual experiment workflow.

This module is the project's front door. It defines the supported stages
(`get`, `preprocess`, `prepare`, `validate`, `train`, `sample`, and `plot`) and connects
their command-line arguments to the corresponding implementation modules.

Keep this file thin: argument parsing, default paths, and user-facing output
belong here; data processing, model code, training, and artifact writing belong
in their own modules. Each command should consume an existing artifact, write
one next artifact, print its path, and stop. This deliberate separation makes
experiments understandable and lets a student run or debug one stage at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from chabuduo.config import (
    PROJECT_ROOT,
    DatasetConfig,
    config_path,
    load_config,
    load_dataset_config,
)
from chabuduo.get import resolve_traffic_dataset, stage_dataset
from chabuduo.plotting import plot_samples
from chabuduo.prepare import prepare
from chabuduo.preprocess import preprocess
from chabuduo.sampling import sample_run
from chabuduo.training import train_model
from chabuduo.validate import ValidationError, validate_artifact

app = typer.Typer(
    name="chabuduo",
    no_args_is_help=True,
    help="Run aircraft trajectory generation experiments one stage at a time.",
)


def _dataset_config(dataset: str, path: Path | None) -> DatasetConfig:
    try:
        return load_dataset_config(path or config_path(dataset))
    except ValueError as error:
        raise typer.BadParameter(f"Invalid dataset configuration: {error}") from error


@app.command("get")
def get_command(
    dataset: Annotated[str, typer.Argument(help="Dataset name: beijing or zurich")],
    source: Annotated[
        Path | None,
        typer.Option(help="Existing source file or provider directory"),
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Raw Parquet output path")] = None,
    config: Annotated[
        Path | None,
        typer.Option(help="Dataset YAML; defaults to the bundled dataset config"),
    ] = None,
    force: Annotated[bool, typer.Option(help="Replace an existing raw file")] = False,
) -> None:
    """Stage raw data; do not preprocess it."""
    configuration = _dataset_config(dataset, config)
    filename = configuration.source.filename
    destination = output or PROJECT_ROOT / "data" / "raw" / dataset / filename
    result = stage_dataset(configuration, destination, source=source, force=force)
    label = (
        "SOURCE" if configuration.source.type == "traffic_dataset" else "OUTPUT"
    )
    typer.echo(f"{label}={result}")


@app.command("preprocess")
def preprocess_command(
    dataset: Annotated[
        str,
        typer.Argument(
            help="Bundled dataset configuration, for example beijing_feiyou or zurich"
        ),
    ],
    source: Annotated[
        Path | None,
        typer.Option(help="Staged raw Parquet file; omitted for traffic datasets"),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Selected variable-length Traffic output")
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(help="Dataset YAML; defaults to the bundled dataset config"),
    ] = None,
    max_workers: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Select variable-length terminal trajectories with Traffic."""
    configuration = _dataset_config(dataset, config)
    settings = configuration.preprocess
    destination = output or (
        PROJECT_ROOT
        / "data"
        / "interim"
        / (
            f"{settings.airport.lower()}-{settings.operation}-"
            f"{settings.radius_nm:g}nm.parquet"
        )
    )
    if source is None:
        if configuration.source.type != "traffic_dataset":
            raise typer.BadParameter("--source is required for local datasets")
        source = resolve_traffic_dataset(configuration.source.dataset)
    preprocess(
        source,
        configuration,
        max_workers=max_workers,
        cache=destination,
    )
    typer.echo(f"OUTPUT={destination}")


@app.command("prepare")
def prepare_command(
    dataset: Annotated[str, typer.Argument(help="Dataset configuration name")],
    selected: Annotated[
        Path, typer.Option(help="Selected/resampled Traffic Parquet cache")
    ],
    output: Annotated[
        Path | None, typer.Option(help="Serializable generation artifact directory")
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(help="Dataset YAML; defaults to the bundled dataset config"),
    ] = None,
) -> None:
    """Convert selected Traffic into a train-only-normalized model artifact."""
    configuration = _dataset_config(dataset, config)
    destination = (
        output
        or PROJECT_ROOT
        / "data"
        / "processed"
        / f"{dataset}-{configuration.version}"
    )
    result = prepare(selected, destination, configuration)
    typer.echo(f"OUTPUT={result}")


@app.command("validate")
def validate_command(
    artifact: Annotated[Path, typer.Argument(help="Prepared artifact directory")],
) -> None:
    """Check whether a prepared dataset is safe for training."""
    try:
        messages = validate_artifact(artifact)
    except ValidationError as error:
        typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(code=1) from error
    for message in messages:
        typer.echo(f"✓ {message}")


@app.command("train")
def train_command(
    artifact: Annotated[Path, typer.Argument(help="Prepared artifact directory")],
    config: Annotated[
        Path | None,
        typer.Option(help="Model YAML; defaults to the bundled TCVAE config"),
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Run output directory")] = None,
    seed: Annotated[int, typer.Option()] = 42,
    smoke: Annotated[
        bool, typer.Option(help="Run one tiny CPU epoch to test the complete path")
    ] = False,
) -> None:
    """Train the standard-prior temporal convolutional VAE."""
    model_config = load_config(config or config_path("tcvae"))
    destination = (
        output
        or PROJECT_ROOT
        / "runs"
        / f"{artifact.name}-tcvae-seed-{seed}{'-smoke' if smoke else ''}"
    )
    result = train_model(artifact, destination, model_config, seed=seed, smoke=smoke)
    typer.echo(f"RUN={result}")


@app.command("sample")
def sample_command(
    run: Annotated[Path, typer.Argument(help="Completed training run directory")],
    count: Annotated[int, typer.Option(min=1)] = 100,
    seed: Annotated[int, typer.Option()] = 42,
    output: Annotated[
        Path | None, typer.Option(help="Generated sample artifact directory")
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(min=0.05, max=2.0, help="Latent prior temperature"),
    ] = 1.0,
) -> None:
    """Generate canonical tensors and a Traffic-compatible Parquet table."""
    result = sample_run(
        run,
        count=count,
        seed=seed,
        output=output,
        temperature=temperature,
    )
    typer.echo(f"OUTPUT={result}")


@app.command("plot")
def plot_command(
    sample: Annotated[Path, typer.Argument(help="Generated sample directory")],
    output: Annotated[Path | None, typer.Option(help="PNG output path")] = None,
) -> None:
    """Plot held-out and generated trajectories in the dataset's local frame."""
    result = plot_samples(sample, output=output)
    typer.echo(f"OUTPUT={result}")


if __name__ == "__main__":
    app()
