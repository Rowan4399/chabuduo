"""Make the raw source file available to the next workflow stage.

This module is deliberately simple: it does not create trajectories or perform
scientific preprocessing. It copies an existing local Parquet, converts a
bounded set of provider CSV files into a raw Traffic-compatible Parquet, or asks
`traffic` for the path of a published dataset in traffic's cache.

The next stage turns that Parquet file into a `traffic.core.Traffic` object and
then operates on its `Flight` objects. Keeping raw files and in-memory Traffic
objects separate makes the boundary easy to inspect: `get` answers "where is
the file?"; preprocessing answers "how do we interpret its flights?".

For local copies, the SHA-256 sidecar records the identity of the exact bytes
used later. A filename or dataset name is not enough because the underlying file
may change while retaining its name.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from chabuduo.config import write_config
from chabuduo.sources.feiyou import CONVERSION, convert_files, discover_files


def sha256(path: Path) -> str:
    """Return a stable content fingerprint for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_output(output: Path, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not force:
            raise FileExistsError(
                f"{output} already exists; pass --force to replace it"
            )


def resolve_traffic_dataset(dataset_name: str) -> Path:
    """Return traffic's cached Parquet path, downloading it only if needed."""
    from traffic.data.datasets import datasets
    from traffic.data.datasets.default import Default

    try:
        entry = datasets[dataset_name]
    except KeyError as error:
        raise ValueError(f"Unknown traffic dataset: {dataset_name}") from error
    return Default().get_data(entry)


def stage_dataset(
    config: dict[str, Any],
    output: Path,
    *,
    source: Path | None = None,
    force: bool = False,
) -> Path:
    """Resolve or stage one raw dataset without scientific preprocessing."""
    source_config = config["source"]
    source_type = source_config["type"]

    if source_type == "traffic_dataset":
        # The output argument is retained for a common CLI signature, but
        # traffic owns this cache and should not be copied into data/raw.
        if source is not None:
            raise ValueError("Traffic datasets do not accept --source")
        return resolve_traffic_dataset(source_config["dataset"])

    _prepare_output(output, force)
    if source is None:
        raise ValueError("This dataset requires --source PATH")
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    if source_type == "local_parquet":
        shutil.copy2(source, output)
        provenance = {
            "mode": "local_copy",
            "source": str(source),
        }
    elif source_type == "feiyou_csv":
        date_ranges = [
            (
                Path(item["path"]),
                (item["start_date"], item["end_date"]),
            )
            for item in source_config["date_ranges"]
        ]
        sources = discover_files(
            source,
            pattern=source_config["pattern"],
            date_ranges=date_ranges,
        )
        conversion = convert_files(
            sources,
            output,
            chunk_rows=int(source_config.get("chunk_rows", 250_000)),
        )
        source_results = {
            item["path"]: item for item in conversion["sources"]
        }
        provenance = {
            "mode": "feiyou_csv_conversion",
            "source_root": str(source),
            "pattern": source_config["pattern"],
            "date_ranges": [
                {
                    "path": str(relative_path),
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                }
                for relative_path, (start_date, end_date) in date_ranges
            ],
            "conversion": CONVERSION,
            "rows": conversion["rows"],
            "duplicates_removed": conversion["duplicates_removed"],
            "sources": [
                {
                    "file": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "input_rows": source_results[str(path)]["input_rows"],
                    "rows": source_results[str(path)]["rows"],
                    "duplicates_removed": source_results[str(path)][
                        "duplicates_removed"
                    ],
                }
                for path in sources
            ],
        }
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    manifest = {
        "artifact_type": "raw_dataset",
        "dataset": config["name"],
        "version": config["version"],
        "file": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "provenance": provenance,
    }
    write_config(manifest, output.with_suffix(output.suffix + ".yaml"))
    return output
