"""Validate the one committed YAML describing each complete experiment."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

PROJECT_ROOT = Path.cwd()
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DateRange(StrictModel):
    path: Path
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def dates_are_ordered(self) -> DateRange:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self


class LocalParquetSource(StrictModel):
    type: Literal["local_parquet"]
    filename: str
    columns: list[str] | None = None


class TrafficDatasetSource(StrictModel):
    type: Literal["traffic_dataset"]
    dataset: str
    filename: str


class FeiyouSource(StrictModel):
    type: Literal["feiyou_csv"]
    filename: str
    pattern: str
    date_ranges: list[DateRange] = Field(min_length=1)
    chunk_rows: PositiveInt = 250_000


Source = Annotated[
    LocalParquetSource | TrafficDatasetSource | FeiyouSource,
    Field(discriminator="type"),
]


class PreprocessSettings(StrictModel):
    operation: Literal["landing", "takeoff"]
    airport: str = Field(pattern=r"^[A-Z0-9]{4}$")
    runways: list[str] | None = None
    initial_flows: list[str] | None = None
    radius_nm: PositiveFloat
    max_threshold_distance_nm: PositiveFloat
    min_observations: PositiveInt
    min_duration_sec: PositiveFloat

    @model_validator(mode="after")
    def threshold_is_inside_asma(self) -> PreprocessSettings:
        if self.max_threshold_distance_nm >= self.radius_nm:
            raise ValueError("max_threshold_distance_nm must be smaller than radius_nm")
        if self.runways is not None:
            if not self.runways:
                raise ValueError("runways must contain at least one runway designator")
            if len(set(self.runways)) != len(self.runways):
                raise ValueError("runways must not contain duplicates")
            if any(
                not runway or runway != runway.strip().upper()
                for runway in self.runways
            ):
                raise ValueError(
                    "runway designators must be non-empty uppercase strings"
                )
        if self.initial_flows is not None:
            if not self.initial_flows:
                raise ValueError("initial_flows must contain at least one label")
            if len(set(self.initial_flows)) != len(self.initial_flows):
                raise ValueError("initial_flows must not contain duplicates")
            if any(not flow or flow != flow.strip() for flow in self.initial_flows):
                raise ValueError("initial_flow labels must be non-empty and stripped")
        return self


class RepresentationSettings(StrictModel):
    name: str
    features: list[str] = Field(min_length=1)
    units: list[str] = Field(min_length=1)
    direction: Literal["chronological"]
    projection: str | None = None
    context: list[str] = Field(default_factory=list)
    anchor: str | None = None

    @model_validator(mode="after")
    def one_unit_per_feature(self) -> RepresentationSettings:
        if len(self.features) != len(self.units):
            raise ValueError("representation features and units must have equal length")
        return self


class SplitSettings(StrictModel):
    group_by: Literal["flight_id"]
    train: PositiveFloat
    validation: PositiveFloat
    test: PositiveFloat
    seed: int

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> SplitSettings:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self


class NormalizationSettings(StrictModel):
    method: str
    fit_on: Literal["train"]


class PrepareSettings(StrictModel):
    points: PositiveInt
    metadata_columns: list[str] = Field(default_factory=list)
    representation: RepresentationSettings
    split: SplitSettings
    normalization: NormalizationSettings


class TCVAELossSettings(StrictModel):
    reconstruction_weight: PositiveFloat
    kl_weight: PositiveFloat
    decoder_scale: PositiveFloat = 1.0
    learn_decoder_scale: bool = True


class StandardPriorSettings(StrictModel):
    type: Literal["standard"] = "standard"


class VampPriorSettings(StrictModel):
    type: Literal["vamp"]
    components: PositiveInt
    initialization: Literal["train_examples"] = "train_examples"


TCVAEPriorSettings = Annotated[
    StandardPriorSettings | VampPriorSettings,
    Field(discriminator="type"),
]


class TCVAEModelSettings(StrictModel):
    type: Literal["tcvae"]
    hidden_channels: list[PositiveInt] = Field(min_length=1)
    latent_dim: PositiveInt
    kernel_size: PositiveInt
    dilation_base: PositiveInt
    sampling_factor: PositiveInt
    dropout: float = Field(ge=0, lt=1)
    prior: TCVAEPriorSettings = Field(default_factory=StandardPriorSettings)
    loss: TCVAELossSettings


class ConditioningSettings(StrictModel):
    feature: str


class DDPMModelSettings(StrictModel):
    type: Literal["ddpm"]
    architecture: Literal["dual_branch_1d_unet", "temporal_dilated_resnet"]
    hidden_channels: PositiveInt
    num_layers: PositiveInt = 4
    residual_blocks: PositiveInt = 6
    kernel_size: PositiveInt = 3
    dilations: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    dropout: float = Field(default=0.0, ge=0, lt=1)
    use_phase: bool = True
    time_embedding_dim: PositiveInt = 128
    diffusion_steps: PositiveInt
    beta_schedule: Literal["linear"] = "linear"
    prediction: Literal["noise"]
    conditioning: ConditioningSettings | None = None

    @model_validator(mode="after")
    def temporal_architecture_is_consistent(self) -> DDPMModelSettings:
        if self.architecture == "temporal_dilated_resnet":
            if self.kernel_size % 2 == 0:
                raise ValueError("temporal_dilated_resnet requires an odd kernel_size")
            if len(self.dilations) != self.residual_blocks:
                raise ValueError("one dilation is required per residual block")
            if self.time_embedding_dim % 2:
                raise ValueError("time_embedding_dim must be even")
            if self.conditioning is not None:
                raise ValueError(
                    "temporal_dilated_resnet is unconditional in this experiment"
                )
        return self


ModelSettings = Annotated[
    TCVAEModelSettings | DDPMModelSettings,
    Field(discriminator="type"),
]


class OptimizerSettings(StrictModel):
    learning_rate: PositiveFloat
    weight_decay: float = Field(default=0, ge=0)
    lr_step_size: PositiveInt | None = None
    lr_gamma: float | None = Field(default=None, gt=0, le=1)
    lr_plateau_patience: PositiveInt | None = None
    lr_plateau_factor: float | None = Field(default=None, gt=0, lt=1)
    min_learning_rate: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def scheduler_settings_are_paired(self) -> OptimizerSettings:
        if (self.lr_step_size is None) != (self.lr_gamma is None):
            raise ValueError("lr_step_size and lr_gamma must be set together")
        if (self.lr_plateau_patience is None) != (self.lr_plateau_factor is None):
            raise ValueError(
                "lr_plateau_patience and lr_plateau_factor must be set together"
            )
        if self.lr_step_size is not None and self.lr_plateau_patience is not None:
            raise ValueError("step and plateau learning-rate schedulers are exclusive")
        return self


class TrainingSettings(StrictModel):
    seed: int
    batch_size: PositiveInt
    max_epochs: PositiveInt
    num_workers: int = Field(ge=0)
    accelerator: str = "auto"
    devices: PositiveInt = 1
    gradient_clip_norm: PositiveFloat | None = None
    mixed_precision: bool = False
    ema_decay: float | None = Field(default=None, gt=0, lt=1)
    early_stopping_patience: PositiveInt | None = None
    early_stopping_min_delta: float = Field(default=0.0, ge=0)


class TCVAESamplingSettings(StrictModel):
    type: Literal["tcvae"]
    count: PositiveInt
    seed: int
    temperature: float = Field(default=1.0, gt=0)


class DDPMSamplingSettings(StrictModel):
    type: Literal["ddpm"]
    count: PositiveInt
    seed: int
    steps: PositiveInt
    eta: float = Field(default=0, ge=0, le=1)
    minimum_groundspeed_kt: float
    maximum_groundspeed_kt: float
    minimum_altitude_ft: float
    maximum_altitude_ft: float


SamplingSettings = Annotated[
    TCVAESamplingSettings | DDPMSamplingSettings,
    Field(discriminator="type"),
]


class ExperimentConfig(StrictModel):
    """One committed configuration for every stage of one experiment."""

    name: str
    dataset: Literal["beijing", "zurich"]
    version: str
    source: Source
    preprocess: PreprocessSettings
    prepare: PrepareSettings
    model: ModelSettings
    optimizer: OptimizerSettings
    training: TrainingSettings
    sampling: SamplingSettings

    @model_validator(mode="after")
    def model_and_sampling_match(self) -> ExperimentConfig:
        if self.model.type != self.sampling.type:
            raise ValueError("model and sampling types must match")
        return self


def config_path(name: str) -> Path:
    """Return a bundled configuration path."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No configuration found for {name!r}: {path}")
    return path


def raw_path(config: ExperimentConfig) -> Path:
    return PROJECT_ROOT / "data" / "raw" / config.dataset / config.source.filename


def selected_path(config: ExperimentConfig) -> Path:
    settings = config.preprocess
    source_name = Path(config.source.filename).stem
    runway_part = ""
    if settings.runways:
        runway_part = "rwy" + "-".join(runway.lower() for runway in settings.runways)
        runway_part += "-"
    flow_part = ""
    if settings.initial_flows:
        labels = [
            re.sub(r"[^a-z0-9]+", "-", flow.lower()).strip("-")
            for flow in settings.initial_flows
        ]
        flow_part = "flow" + "-".join(labels) + "-"
    filename = (
        f"{settings.airport.lower()}-{runway_part}{flow_part}{settings.operation}-"
        f"{settings.radius_nm:g}nm-{source_name}-{config.version}.parquet"
    )
    return PROJECT_ROOT / "data" / "interim" / filename


def prepared_path(config: ExperimentConfig) -> Path:
    return PROJECT_ROOT / "data" / "processed" / f"{config.name}-{config.version}"


def run_path(config: ExperimentConfig, *, smoke: bool = False) -> Path:
    suffix = "-smoke" if smoke else ""
    return (
        PROJECT_ROOT
        / "runs"
        / f"{config.name}-{config.version}-seed-{config.training.seed}{suffix}"
    )


def sample_path(config: ExperimentConfig, run: Path) -> Path:
    sampling = config.sampling
    suffix = f"seed-{sampling.seed}-n-{sampling.count}"
    if sampling.type == "tcvae" and sampling.temperature != 1.0:
        suffix += f"-t-{sampling.temperature:g}"
    if sampling.type == "ddpm":
        suffix += f"-steps-{sampling.steps}-eta-{sampling.eta:g}"
    return run / "samples" / suffix


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load the single source of truth for one complete experiment."""
    with path.open(encoding="utf8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return ExperimentConfig.model_validate(value)
