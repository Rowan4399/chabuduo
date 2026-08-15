"""Public CLI for running one configured experiment one stage at a time."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from chabuduo.config import (
    ExperimentConfig,
    config_path,
    load_experiment_config,
    prepared_path,
    raw_path,
    run_path,
    sample_path,
    selected_path,
)
from chabuduo.get import resolve_traffic_dataset, stage_dataset
from chabuduo.plotting import plot_samples
from chabuduo.prepare import prepare
from chabuduo.preprocess import preprocess
from chabuduo.sampling import sample_run
from chabuduo.train import train
from chabuduo.validate import ValidationError, validate_artifact

app = typer.Typer(
    name="chabuduo",
    no_args_is_help=True,
    help="Run one configured trajectory experiment one stage at a time.",
)


def _experiment_config(experiment: str, path: Path | None) -> ExperimentConfig:
    try:
        return load_experiment_config(path or config_path(experiment))
    except ValueError as error:
        raise typer.BadParameter(
            f"Invalid experiment configuration: {error}"
        ) from error


ConfigOption = Annotated[
    Path | None,
    typer.Option(help="Experiment YAML; defaults to the bundled configuration"),
]


@app.command("get")
def get_command(
    experiment: Annotated[str, typer.Argument(help="Experiment configuration name")],
    source: Annotated[
        Path | None,
        typer.Option(help="Existing source file or provider directory"),
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Raw Parquet output path")] = None,
    config: ConfigOption = None,
    force: Annotated[bool, typer.Option(help="Replace an existing raw file")] = False,
) -> None:
    """Stage raw data; do not preprocess it."""
    configuration = _experiment_config(experiment, config)
    destination = output or raw_path(configuration)
    result = stage_dataset(configuration, destination, source=source, force=force)
    label = "SOURCE" if configuration.source.type == "traffic_dataset" else "OUTPUT"
    typer.echo(f"{label}={result}")


@app.command("preprocess")
def preprocess_command(
    experiment: Annotated[str, typer.Argument(help="Experiment configuration name")],
    source: Annotated[Path | None, typer.Option(help="Raw Parquet input")] = None,
    output: Annotated[
        Path | None, typer.Option(help="Selected variable-length Traffic output")
    ] = None,
    config: ConfigOption = None,
    max_workers: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Select variable-length terminal trajectories with Traffic."""
    configuration = _experiment_config(experiment, config)
    if source is None:
        if configuration.source.type == "traffic_dataset":
            source = resolve_traffic_dataset(configuration.source.dataset)
        else:
            source = raw_path(configuration)
    destination = output or selected_path(configuration)
    preprocess(
        source,
        configuration,
        max_workers=max_workers,
        cache=destination,
    )
    typer.echo(f"OUTPUT={destination}")


@app.command("prepare")
def prepare_command(
    experiment: Annotated[str, typer.Argument(help="Experiment configuration name")],
    selected: Annotated[
        Path | None, typer.Option(help="Selected variable-length Traffic input")
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Ready-to-train tensor directory")
    ] = None,
    config: ConfigOption = None,
) -> None:
    """Create split, normalized, ready-to-train numerical tensors."""
    configuration = _experiment_config(experiment, config)
    result = prepare(
        selected or selected_path(configuration),
        output or prepared_path(configuration),
        configuration,
    )
    typer.echo(f"OUTPUT={result}")


@app.command("validate")
def validate_command(
    experiment: Annotated[str, typer.Argument(help="Experiment configuration name")],
    artifact: Annotated[
        Path | None, typer.Option(help="Prepared tensor directory")
    ] = None,
    config: ConfigOption = None,
) -> None:
    """Check whether prepared tensors are safe for training."""
    configuration = _experiment_config(experiment, config)
    try:
        messages = validate_artifact(
            artifact or prepared_path(configuration), configuration
        )
    except ValidationError as error:
        typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(code=1) from error
    for message in messages:
        typer.echo(f"✓ {message}")


@app.command("train")
def train_command(
    experiment: Annotated[str, typer.Argument(help="Experiment configuration name")],
    artifact: Annotated[
        Path | None, typer.Option(help="Prepared tensor directory")
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Run output directory")] = None,
    config: ConfigOption = None,
    smoke: Annotated[
        bool, typer.Option(help="Run one tiny CPU epoch to test the complete path")
    ] = False,
) -> None:
    """Train the model selected by the experiment configuration."""
    configuration = _experiment_config(experiment, config)
    result = train(
        artifact or prepared_path(configuration),
        output or run_path(configuration, smoke=smoke),
        configuration,
        smoke=smoke,
    )
    typer.echo(f"RUN={result}")


@app.command("sample")
def sample_command(
    experiment: Annotated[str, typer.Argument(help="Experiment configuration name")],
    run: Annotated[Path | None, typer.Option(help="Training run directory")] = None,
    artifact: Annotated[
        Path | None, typer.Option(help="Prepared tensor directory")
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Generated sample directory")
    ] = None,
    config: ConfigOption = None,
) -> None:
    """Generate tensors and a Traffic-compatible Parquet table."""
    configuration = _experiment_config(experiment, config)
    run_directory = run or run_path(configuration)
    result = sample_run(
        run_directory,
        artifact or prepared_path(configuration),
        configuration,
        output=output or sample_path(configuration, run_directory),
    )
    typer.echo(f"OUTPUT={result}")


@app.command("plot")
def plot_command(
    experiment: Annotated[str, typer.Argument(help="Experiment configuration name")],
    sample: Annotated[
        Path | None, typer.Option(help="Generated sample directory")
    ] = None,
    artifact: Annotated[
        Path | None, typer.Option(help="Prepared tensor directory")
    ] = None,
    output: Annotated[Path | None, typer.Option(help="PNG output path")] = None,
    config: ConfigOption = None,
) -> None:
    """Plot held-out and generated trajectories in the local frame."""
    configuration = _experiment_config(experiment, config)
    run_directory = run_path(configuration)
    result = plot_samples(
        sample or sample_path(configuration, run_directory),
        artifact or prepared_path(configuration),
        configuration,
        output=output,
    )
    typer.echo(f"OUTPUT={result}")


if __name__ == "__main__":
    app()
