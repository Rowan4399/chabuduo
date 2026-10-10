"""Losslessly re-encode the repaired Parquet for this slow remote transfer."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from repair_tianjin_track import fingerprint

root = Path("runs/tianjin-track-unwrap-audit-20261007")
source = Path("data/raw/tianjin/landing_tianjin_2026_08_09_60nm_filtered_1s_track_unwrapped_v2.parquet")
output = source.with_name(source.stem + "_transfer.parquet")
parquet = pq.ParquetFile(source)
float_columns = [field.name for field in parquet.schema_arrow if pa.types.is_floating(field.type)]
dictionary_columns = [field.name for field in parquet.schema_arrow if pa.types.is_string(field.type) or pa.types.is_boolean(field.type)]
kwargs = dict(compression="zstd", compression_level=12, use_byte_stream_split=float_columns, use_dictionary=dictionary_columns)
trial = root / "compression_trial.parquet"
table = parquet.read_row_group(0)
pq.write_table(table, trial, **kwargs)
old_size = parquet.metadata.row_group(0).total_byte_size
print("TRIAL", len(table), trial.stat().st_size, old_size, flush=True)
table = pq.read_table(source)
pq.write_table(table, output, **kwargs)
if not table.equals(pq.read_table(output), check_metadata=True):
    raise ValueError("Transport encoding changed Parquet values or metadata")
report = {
    "source": str(source), "output": str(output),
    "source_bytes": source.stat().st_size, "output_bytes": output.stat().st_size,
    "source_sha256": fingerprint(source), "output_sha256": fingerprint(output),
    "all_values_order_and_metadata_identical": True,
    "reason": "Slow SFTP transfer; byte-stream splitting and Zstandard level 12",
}
(root / "transport_encoding.json").write_text(json.dumps(report, indent=2), "utf8")
print(json.dumps(report, indent=2), flush=True)
