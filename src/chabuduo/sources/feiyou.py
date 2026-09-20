"""Convert daily Feiyou CSV observations to a Traffic-compatible Parquet."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_COLUMNS = [
    "position_time",
    "lat",
    "lng",
    "altitude",
    "speed",
    "ver_speed",
    "heading",
    "callsign",
    "icao",
]

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ns", tz="UTC")),
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("altitude", pa.float64()),
        pa.field("groundspeed", pa.float64()),
        pa.field("vertical_rate", pa.float64()),
        pa.field("heading", pa.float64()),
        pa.field("callsign", pa.string()),
        pa.field("icao24", pa.string()),
    ]
)

METERS_TO_FEET = 1 / 0.3048
MPS_TO_KNOTS = 1 / 0.5144444444444445
MPS_TO_FEET_PER_MINUTE = 60 / 0.3048

CONVERSION = {
    "deduplication": "keep first observation per icao and position_time",
    "timestamp": "position_time milliseconds since epoch to UTC",
    "latitude": "lat degrees",
    "longitude": "lng degrees",
    "altitude": "altitude metres to feet",
    "groundspeed": "speed metres per second to knots",
    "vertical_rate": "ver_speed metres per second to feet per minute",
    "heading": "heading degrees",
    "callsign": "callsign",
    "icao24": "icao decimal integer to six-character hexadecimal",
}


def _date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def discover_files(
    source: Path,
    *,
    pattern: str,
    date_ranges: list[tuple[Path, tuple[date | str, date | str]]],
) -> list[Path]:
    """Return exactly one chronologically ordered CSV for each requested day."""
    if not date_ranges:
        raise ValueError("Feiyou date_ranges must not be empty")

    expected_dates: set[date] = set()
    by_date: dict[date, list[Path]] = {}
    for index, (relative_path, (start_value, end_value)) in enumerate(
        date_ranges, start=1
    ):
        start = _date(start_value)
        end = _date(end_value)
        if end < start:
            raise ValueError(
                f"Feiyou date range {index} has end_date earlier than start_date"
            )

        directory = source / relative_path
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Feiyou date range {index} source directory "
                f"does not exist: {directory}"
            )
        range_dates = {
            start + timedelta(days=offset) for offset in range((end - start).days + 1)
        }

        overlap = expected_dates & range_dates
        if overlap:
            raise ValueError(
                f"Feiyou date ranges overlap: "
                f"{', '.join(str(day) for day in sorted(overlap))}"
            )

        expected_dates.update(range_dates)

        candidates = sorted(directory.glob(pattern))
        for path in candidates:
            if not path.is_file():
                continue
            try:
                file_date = date.fromisoformat(path.stem)
            except ValueError as error:
                raise ValueError(
                    f"Feiyou CSV filename must be YYYY-MM-DD: {path}"
                ) from error
            if file_date in range_dates:
                by_date.setdefault(file_date, []).append(path)

    expected = sorted(expected_dates)
    missing = [str(day) for day in expected if day not in by_date]
    duplicates = {
        str(day): [str(path) for path in paths]
        for day, paths in by_date.items()
        if len(paths) != 1
    }
    if missing or duplicates:
        details = []
        if missing:
            details.append(f"missing dates: {', '.join(missing)}")
        if duplicates:
            details.append(f"duplicate dates: {duplicates}")
        raise ValueError("Invalid Feiyou source set; " + "; ".join(details))
    return [by_date[day][0] for day in expected]


def _drop_duplicate_observations(
    chunk: pd.DataFrame,
    seen: set[int],
    previous_seen: set[int],
) -> tuple[pd.DataFrame, int]:
    """Keep the first row for each aircraft timestamp in one daily file."""
    addresses = pd.to_numeric(chunk["icao"], errors="raise").astype("int64")
    if ((addresses < 0) | (addresses > 0xFFFFFF)).any():
        raise ValueError("Feiyou icao values must fit in a 24-bit address")
    timestamps = pd.to_numeric(chunk["position_time"], errors="raise").astype("int64")
    keep: list[bool] = []
    for address, timestamp in zip(addresses, timestamps, strict=True):
        # ICAO addresses use 24 bits, so this packs both integers exactly.
        key = (int(timestamp) << 24) | int(address)
        keep.append(key not in seen and key not in previous_seen)
        seen.add(key)
    return chunk.loc[keep], keep.count(False)


def _convert_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    icao = pd.to_numeric(chunk["icao"], errors="raise").astype("int64")
    if ((icao < 0) | (icao > 0xFFFFFF)).any():
        raise ValueError("Feiyou icao values must fit in a 24-bit address")

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                chunk["position_time"], unit="ms", utc=True, errors="raise"
            ),
            "latitude": pd.to_numeric(chunk["lat"], errors="raise"),
            "longitude": pd.to_numeric(chunk["lng"], errors="raise"),
            "altitude": pd.to_numeric(chunk["altitude"], errors="raise")
            * METERS_TO_FEET,
            "groundspeed": pd.to_numeric(chunk["speed"], errors="raise") * MPS_TO_KNOTS,
            "vertical_rate": pd.to_numeric(chunk["ver_speed"], errors="raise")
            * MPS_TO_FEET_PER_MINUTE,
            "heading": pd.to_numeric(chunk["heading"], errors="raise"),
            "callsign": chunk["callsign"].astype("string"),
            "icao24": icao.map(lambda value: f"{value:06x}").astype("string"),
        }
    )


def convert_files(
    sources: list[Path],
    destination: Path,
    *,
    chunk_rows: int = 250_000,
) -> dict[str, Any]:
    """Stream daily CSV files into one atomically replaced Parquet artifact."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    rows_by_source: list[dict[str, Any]] = []
    total_rows = 0
    previous_seen: set[int] = set()

    try:
        with pq.ParquetWriter(temporary, OUTPUT_SCHEMA, compression="snappy") as writer:
            for source in sources:
                source_rows = 0
                source_input_rows = 0
                source_duplicates = 0
                seen: set[int] = set()
                chunks = pd.read_csv(
                    source,
                    usecols=SOURCE_COLUMNS,
                    chunksize=chunk_rows,
                    low_memory=False,
                )
                for chunk in chunks:
                    source_input_rows += len(chunk)
                    chunk, duplicates = _drop_duplicate_observations(
                        chunk,
                        seen,
                        previous_seen,
                    )
                    source_duplicates += duplicates
                    converted = _convert_chunk(chunk)
                    table = pa.Table.from_pandas(
                        converted,
                        schema=OUTPUT_SCHEMA,
                        preserve_index=False,
                        safe=True,
                    )
                    writer.write_table(table)
                    source_rows += len(converted)
                    total_rows += len(converted)
                    print(f"{total_rows:,} rows written", flush=True)
                previous_seen = seen
                rows_by_source.append(
                    {
                        "path": str(source),
                        "input_rows": source_input_rows,
                        "rows": source_rows,
                        "duplicates_removed": source_duplicates,
                    }
                )
        if total_rows == 0:
            raise ValueError("The Feiyou source files contain no observations")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "rows": total_rows,
        "duplicates_removed": sum(
            item["duplicates_removed"] for item in rows_by_source
        ),
        "sources": rows_by_source,
    }
