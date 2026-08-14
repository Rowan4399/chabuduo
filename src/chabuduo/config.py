"""Load generic artifacts and validate user-editable dataset YAML files."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

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
    operation: Literal["landing"]
    airport: str = Field(pattern=r"^[A-Z0-9]{4}$")
    radius_nm: PositiveFloat
    max_threshold_distance_nm: PositiveFloat
    min_observations: PositiveInt
    min_duration_sec: PositiveFloat

    @model_validator(mode="after")
    def threshold_is_inside_asma(self) -> PreprocessSettings:
        if self.max_threshold_distance_nm >= self.radius_nm:
            raise ValueError("max_threshold_distance_nm must be smaller than radius_nm")
        return self


class RepresentationSettings(StrictModel):
    name: str
    features: list[str] = Field(min_length=1)
    units: list[str] = Field(min_length=1)
    direction: Literal["chronological"]
    projection: str | None = None
    context: list[str] = Field(default_factory=list)
    anchor: str | None = None


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
    representation: RepresentationSettings
    split: SplitSettings
    normalization: NormalizationSettings


class DatasetConfig(StrictModel):
    name: str
    version: str
    source: Source
    preprocess: PreprocessSettings
    prepare: PrepareSettings | None = None


def config_path(name: str) -> Path:
    """Return a bundled configuration path."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No configuration found for {name!r}: {path}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    """Load a generic YAML mapping used by artifacts and model configurations."""
    with path.open(encoding="utf8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return value


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset workflow configuration."""
    return DatasetConfig.model_validate(load_config(path))


def write_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
